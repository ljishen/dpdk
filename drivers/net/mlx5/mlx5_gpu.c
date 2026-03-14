/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2025 - GPU-Initiated Network Stack (GINS) Project
 */

#include <stddef.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include <rte_malloc.h>
#include <rte_errno.h>
#include <rte_common.h>
#include <rte_log.h>
#include <eal_export.h>

#include <mlx5_glue.h>
#include <mlx5_common.h>
#include <mlx5_common_mr.h>
#include <mlx5_common_devx.h>
#include <mlx5_devx_cmds.h>
#include <mlx5_common_os.h>
#include <ethdev_driver.h>

#include "mlx5_gpu.h"
#include "mlx5.h"
#include "mlx5_defs.h"
#include "mlx5_rx.h"
#include "rte_pmd_mlx5.h"

/**
 * Register GPU memory with MLX5 for NIC DMA access
 *
 * nvidia-peermem must be loaded so that ibv_reg_mr() can resolve
 * GPU virtual addresses via get_user_pages() interception.
 *
 * @param pd       Protection Domain (struct ibv_pd *)
 * @param gpu_mem  GPU memory descriptor with gpu_ptr and size set
 * @return 0 on success, negative error code on failure
 */
static int
mlx5_mr_register_gpu(void *pd, struct mlx5_gpu_mem *gpu_mem)
{
	struct ibv_mr *ibv_mr = NULL;

	if (!pd || !gpu_mem) {
		DRV_LOG(ERR, "Invalid parameters for GPU MR registration");
		return -EINVAL;
	}

	if (!gpu_mem->gpu_ptr) {
		DRV_LOG(ERR, "GPU virtual address is NULL");
		return -EINVAL;
	}

	if (gpu_mem->size == 0) {
		DRV_LOG(ERR, "GPU memory size is zero");
		return -EINVAL;
	}

	ibv_mr = mlx5_glue->reg_mr(pd, gpu_mem->gpu_ptr, gpu_mem->size,
				    IBV_ACCESS_LOCAL_WRITE);
	if (!ibv_mr) {
		DRV_LOG(ERR, "ibv_reg_mr GPU ptr=%p size=%zu failed: %s",
			gpu_mem->gpu_ptr, gpu_mem->size, strerror(errno));
		return -errno;
	}

	gpu_mem->mr.lkey = ibv_mr->lkey;
	gpu_mem->mr.addr = gpu_mem->gpu_ptr;
	gpu_mem->mr.len = gpu_mem->size;
	gpu_mem->mr.obj = ibv_mr;
	gpu_mem->mr.mkey = NULL;

	DRV_LOG(DEBUG, "GPU MR registered: ptr=%p size=%zu lkey=0x%x",
		gpu_mem->gpu_ptr, gpu_mem->size, gpu_mem->mr.lkey);

	return 0;
}

/**
 * Register GPU memory as DevX UMEM
 *
 * UMEM (User Memory) registration is required for DevX queue creation.
 * Uses standard devx_umem_reg() with the GPU virtual address.
 * nvidia-peermem intercepts the kernel's get_user_pages() to resolve
 * GPU virtual addresses.
 *
 * @param ctx      Device context
 * @param gpu_mem  GPU memory region to register
 * @return 0 on success, negative error code on failure
 */
static int
mlx5_gpu_umem_register(void *ctx, struct mlx5_gpu_mem *gpu_mem)
{
	if (!ctx || !gpu_mem)
		return -EINVAL;

	gpu_mem->umem_obj = mlx5_glue->devx_umem_reg(ctx,
		gpu_mem->gpu_ptr,
		gpu_mem->size,
		IBV_ACCESS_LOCAL_WRITE);

	if (!gpu_mem->umem_obj) {
		DRV_LOG(ERR, "devx_umem_reg ptr=%p size=%zu failed: %s",
			gpu_mem->gpu_ptr, gpu_mem->size, strerror(errno));
		return -errno;
	}

	gpu_mem->umem_id = mlx5_os_get_umem_id(gpu_mem->umem_obj);

	DRV_LOG(DEBUG, "GPU UMEM registered: id=%u, size=%zu",
		gpu_mem->umem_id, gpu_mem->size);

	return 0;
}

/**
 * Deregister GPU UMEM
 */
static void
mlx5_gpu_umem_deregister(struct mlx5_gpu_mem *gpu_mem)
{
	if (gpu_mem && gpu_mem->umem_obj) {
		mlx5_glue->devx_umem_dereg(gpu_mem->umem_obj);
		gpu_mem->umem_obj = NULL;
		gpu_mem->umem_id = 0;
	}
}

/**
 * Create a DevX CQ backed by GPU memory
 *
 * Derives ctx, UAR page ID, and EQN internally from priv.
 *
 * @param priv        MLX5 device private data
 * @param cq_mem      GPU memory for CQ buffer (DBR appended at tail)
 * @param log_cq_size Log2 of CQ size
 * @return DevX CQ object, or NULL on failure
 */
static struct mlx5_devx_obj *
mlx5_gpu_cq_create(struct mlx5_priv *priv, struct mlx5_gpu_mem *cq_mem,
		    uint32_t log_cq_size, uint32_t cqe_comp_en,
		    uint32_t mini_cqe_res_format)
{
	void *ctx = priv->sh->cdev->ctx;
	uint32_t uar_page_id = mlx5_os_get_devx_uar_page_id(
				priv->sh->tx_uar.obj);
	uint32_t eqn;
	struct mlx5_devx_cq_attr cq_attr;

	if (mlx5_glue->devx_query_eqn(ctx, 0, &eqn)) {
		DRV_LOG(ERR, "Failed to query EQN for CQ creation");
		return NULL;
	}

	/*
	 * Cap log_cq_size so the CQ buffer + doorbell record both fit in the
	 * registered UMEM.  cq_mem->size covers (1 << log_cq_size) * 64-byte
	 * CQEs; the cache-line-aligned DBR (MLX5_DBR_SIZE bytes) would fall
	 * outside if we do not shrink the CQ by one power-of-two step.
	 */
	while (log_cq_size > 0 &&
	       ((1u << log_cq_size) * sizeof(struct mlx5_cqe) + MLX5_DBR_SIZE)
	       > cq_mem->size)
		log_cq_size--;

	memset(&cq_attr, 0, sizeof(cq_attr));

	cq_attr.q_umem_valid = 1;
	cq_attr.q_umem_id = cq_mem->umem_id;
	cq_attr.q_umem_offset = 0;
	cq_attr.log_cq_size = log_cq_size;
	cq_attr.log_page_size = MLX5_LOG_PAGE_SIZE - 12;  /* log2(4KB-pages) */
	cq_attr.uar_page_id = uar_page_id;
	cq_attr.eqn = eqn;

	/* Doorbell record at CQ buffer tail. */
	cq_attr.db_umem_valid = 1;
	cq_attr.db_umem_id = cq_mem->umem_id;
	cq_attr.db_umem_offset =
		(uint64_t)(1u << log_cq_size) * sizeof(struct mlx5_cqe);

	if (cqe_comp_en) {
		cq_attr.cqe_comp_en = 1;
		cq_attr.mini_cqe_res_format = mini_cqe_res_format;
	}

	return mlx5_devx_cmd_create_cq(ctx, &cq_attr);
}

/**
 * Register GPU memory for a given port
 */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_register_mem)
__rte_internal
int
mlx5_gpu_register_mem(uint16_t port_id, struct mlx5_gpu_mem *gpu_mem)
{
	struct mlx5_priv *priv;

	if (!gpu_mem)
		return -EINVAL;
	if (port_id >= RTE_MAX_ETHPORTS)
		return -EINVAL;

	priv = rte_eth_devices[port_id].data->dev_private;

	return mlx5_mr_register_gpu(priv->sh->cdev->pd, gpu_mem);
}

/**
 * Deregister GPU memory region
 *
 * Deregisters the ibverbs MR.
 *
 * @param gpu_mem  GPU memory descriptor with registered MR
 * @return 0 on success, negative error code on failure
 */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_deregister_mem)
__rte_internal
int
mlx5_gpu_deregister_mem(struct mlx5_gpu_mem *gpu_mem)
{
	if (!gpu_mem) {
		DRV_LOG(ERR, "Invalid GPU memory descriptor");
		return -EINVAL;
	}

	/* Deregister ibverbs MR */
	if (gpu_mem->mr.obj) {
		mlx5_glue->dereg_mr((struct ibv_mr *)gpu_mem->mr.obj);
		gpu_mem->mr.obj = NULL;
	}

	/* Clear MR fields */
	gpu_mem->mr.lkey = 0;
	gpu_mem->mr.addr = NULL;
	gpu_mem->mr.len = 0;

	DRV_LOG(DEBUG, "GPU memory deregistered successfully");
	return 0;
}

/**
 * Create TX queue with GPU memory via DevX
 *
 * Creates a DevX Send Queue (SQ) and associated Completion Queue (CQ)
 * backed by GPU memory. The SQ uses a cyclic work queue layout where
 * GPU kernels post WQEs directly.
 *
 * The creation flow:
 * 1. Register GPU memory regions as UMEM (via nvidia-peermem)
 * 2. Create CQ with GPU memory
 * 3. Create SQ with GPU memory, linked to the CQ
 * 4. Transition SQ from RST to RDY state
 *
 * @param port_id  DPDK port ID
 * @param idx      Queue index
 * @param params   GPU-specific parameters (sqn populated on success)
 * @return 0 on success, negative error code on failure
 */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_create_txq)
__rte_internal
int
mlx5_gpu_create_txq(uint16_t port_id, uint16_t idx,
		    struct mlx5_gpu_txq_params *params)
{
	struct mlx5_priv *priv;
	void *ctx;
	struct mlx5_devx_obj *sq_obj = NULL;
	struct mlx5_devx_obj *cq_obj = NULL;
	struct mlx5_devx_create_sq_attr sq_attr;
	struct mlx5_devx_modify_sq_attr msq_attr;
	uint32_t log_wq_size, log_cq_size;
	uint32_t uar_page_id;
	int ret;

	if (!params || !params->wq_mem || !params->cq_mem) {
		DRV_LOG(ERR, "Invalid GPU TX queue parameters");
		return -EINVAL;
	}
	if (port_id >= RTE_MAX_ETHPORTS)
		return -EINVAL;

	priv = rte_eth_devices[port_id].data->dev_private;
	ctx = priv->sh->cdev->ctx;

	/* Verify GPU memory is registered */
	if (params->wq_mem->mr.lkey == 0 || params->cq_mem->mr.lkey == 0) {
		DRV_LOG(ERR, "GPU memory not registered before queue creation");
		return -EINVAL;
	}

	/* Calculate log2 sizes */
	log_wq_size = rte_log2_u32(params->wq_size);
	log_cq_size = rte_log2_u32(params->cq_size);

	/* Get UAR page ID from the shared TX UAR */
	uar_page_id = mlx5_os_get_devx_uar_page_id(priv->sh->tx_uar.obj);

	/* Step 1: Register GPU memory regions as UMEM */
	ret = mlx5_gpu_umem_register(ctx, params->wq_mem);
	if (ret < 0) {
		DRV_LOG(ERR, "TX queue %u: WQ UMEM registration failed", idx);
		goto error;
	}

	ret = mlx5_gpu_umem_register(ctx, params->cq_mem);
	if (ret < 0) {
		DRV_LOG(ERR, "TX queue %u: CQ UMEM registration failed", idx);
		goto error;
	}

	/* Step 2: Create CQ with GPU memory */
	cq_obj = mlx5_gpu_cq_create(priv, params->cq_mem, log_cq_size, 0, 0);
	if (!cq_obj) {
		DRV_LOG(ERR, "TX queue %u: CQ creation failed", idx);
		ret = -EIO;
		goto error;
	}

	DRV_LOG(DEBUG, "TX queue %u: CQ created, cqn=%u",
		idx, cq_obj->id);

	/* Step 3: Create SQ with GPU memory */
	memset(&sq_attr, 0, sizeof(sq_attr));

	sq_attr.state = MLX5_SQC_STATE_RST;
	sq_attr.cqn = cq_obj->id;
	sq_attr.flush_in_error_en = 1;
	sq_attr.min_wqe_inline_mode =
		priv->sh->cdev->config.hca_attr.vport_inline_mode;
	sq_attr.tis_num = priv->sh->tis[0]->id;  /* Use first TIS */
	sq_attr.tis_lst_sz = 1;
	/*
	 * Query HCA capability for the SQ timestamp format, matching
	 * the RX path and the standard DPDK mlx5 SQ creation.  Hardcoding
	 * FREE_RUNNING (0x0) works on most firmware but some NIC/FW
	 * combinations (e.g. ConnectX-4 with certain firmware versions)
	 * may reject it with BAD_PARAM if the HCA only advertises DEFAULT.
	 */
	sq_attr.ts_format = mlx5_ts_format_conv(
			priv->sh->cdev->config.hca_attr.sq_ts_format);

	/* Work Queue attributes */
	sq_attr.wq_attr.wq_type = MLX5_WQ_TYPE_CYCLIC;
	sq_attr.wq_attr.pd = priv->sh->cdev->pdn;
	sq_attr.wq_attr.uar_page = uar_page_id;
	sq_attr.wq_attr.log_wq_stride = rte_log2_u32(MLX5_WQE_SIZE);
	sq_attr.wq_attr.log_wq_sz = log_wq_size;
	sq_attr.wq_attr.log_wq_pg_sz = MLX5_LOG_PAGE_SIZE - 12;

	/* Point WQ to GPU UMEM */
	sq_attr.wq_attr.wq_umem_valid = 1;
	sq_attr.wq_attr.wq_umem_id = params->wq_mem->umem_id;
	sq_attr.wq_attr.wq_umem_offset = 0;

	/*
	 * Doorbell record: place at end of WQ UMEM.
	 * The WQ buffer occupies (1 << log_wq_size) * MLX5_WQE_SIZE bytes.
	 */
	sq_attr.wq_attr.dbr_umem_valid = 1;
	sq_attr.wq_attr.dbr_umem_id = params->wq_mem->umem_id;
	sq_attr.wq_attr.dbr_addr = (uint64_t)(1u << log_wq_size) * MLX5_WQE_SIZE;

	sq_obj = mlx5_devx_cmd_create_sq(ctx, &sq_attr);
	if (!sq_obj) {
		DRV_LOG(ERR, "TX queue %u: SQ creation failed", idx);
		ret = -EIO;
		goto error;
	}

	DRV_LOG(DEBUG, "TX queue %u: SQ created, sqn=%u",
		idx, sq_obj->id);

	/* Step 4: Transition SQ from RST to RDY */
	memset(&msq_attr, 0, sizeof(msq_attr));
	msq_attr.sq_state = MLX5_SQC_STATE_RST;
	msq_attr.state = MLX5_SQC_STATE_RDY;

	ret = mlx5_devx_cmd_modify_sq(sq_obj, &msq_attr);
	if (ret) {
		DRV_LOG(ERR, "TX queue %u: SQ RST->RDY transition failed", idx);
		ret = -EIO;
		goto error;
	}

	params->sqn = sq_obj->id;

	DRV_LOG(DEBUG, "TX queue %u created: sqn=%u, cqn=%u, wq_size=%u",
		idx, sq_obj->id, cq_obj->id, params->wq_size);

	return 0;

error:
	if (sq_obj)
		mlx5_devx_cmd_destroy(sq_obj);
	if (cq_obj)
		mlx5_devx_cmd_destroy(cq_obj);

	mlx5_gpu_umem_deregister(params->wq_mem);
	mlx5_gpu_umem_deregister(params->cq_mem);

	return ret;
}

/**
 * Create RX queue with GPU memory via DevX
 *
 * Creates a DevX Receive Queue (RQ) and associated Completion Queue (CQ)
 * backed by GPU memory. The RQ uses a cyclic work queue layout where
 * GPU kernels post receive descriptors directly.
 *
 * The creation flow:
 * 1. Register GPU memory regions as UMEM (via nvidia-peermem)
 * 2. Create CQ with GPU memory
 * 3. Create RQ with GPU memory, linked to the CQ
 * 4. Transition RQ from RST to RDY state
 *
 * @param port_id  DPDK port ID
 * @param idx      Queue index
 * @param params   GPU-specific parameters (rqn populated on success)
 * @return 0 on success, negative error code on failure
 */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_create_rxq)
__rte_internal
int
mlx5_gpu_create_rxq(uint16_t port_id, uint16_t idx,
		    struct mlx5_gpu_rxq_params *params)
{
	struct mlx5_priv *priv;
	void *ctx;
	struct mlx5_devx_obj *rq_obj = NULL;
	struct mlx5_devx_obj *cq_obj = NULL;
	struct mlx5_devx_create_rq_attr rq_attr;
	struct mlx5_devx_modify_rq_attr mrq_attr;
	uint32_t log_wq_size, log_cq_size;
	int ret;

	if (!params || !params->wq_mem || !params->cq_mem) {
		DRV_LOG(ERR, "Invalid GPU RX queue parameters");
		return -EINVAL;
	}
	if (port_id >= RTE_MAX_ETHPORTS)
		return -EINVAL;

	priv = rte_eth_devices[port_id].data->dev_private;
	ctx = priv->sh->cdev->ctx;

	/* Verify GPU memory is registered */
	if (params->wq_mem->mr.lkey == 0 || params->cq_mem->mr.lkey == 0) {
		DRV_LOG(ERR, "GPU memory not registered before queue creation");
		return -EINVAL;
	}

	/* Calculate log2 sizes */
	log_wq_size = rte_log2_u32(params->wq_size);
	log_cq_size = rte_log2_u32(params->cq_size);

	/* Step 1: Register GPU memory regions as UMEM */
	ret = mlx5_gpu_umem_register(ctx, params->wq_mem);
	if (ret < 0) {
		DRV_LOG(ERR, "RX queue %u: WQ UMEM registration failed", idx);
		goto error;
	}

	ret = mlx5_gpu_umem_register(ctx, params->cq_mem);
	if (ret < 0) {
		DRV_LOG(ERR, "RX queue %u: CQ UMEM registration failed", idx);
		goto error;
	}

	/* Step 2: Create CQ with embedded DBR at buffer tail. */
	cq_obj = mlx5_gpu_cq_create(priv, params->cq_mem, log_cq_size,
				    params->cqe_comp_en,
				    params->mini_cqe_res_format);
	if (!cq_obj) {
		DRV_LOG(ERR, "RX queue %u: CQ creation failed", idx);
		ret = -EIO;
		goto error;
	}

	DRV_LOG(DEBUG, "RX queue %u: CQ created, cqn=%u",
		idx, cq_obj->id);

	/* Step 3: Create RQ with GPU memory */
	memset(&rq_attr, 0, sizeof(rq_attr));

	rq_attr.state = MLX5_RQC_STATE_RST;
	rq_attr.cqn = cq_obj->id;
	rq_attr.mem_rq_type = MLX5_RQC_MEM_RQ_TYPE_MEMORY_RQ_INLINE;
	/*
	 * Use the ts_format the HCA advertises (same as standard DPDK driver).
	 * On ConnectX-5 VF, rq_ts_format is typically FR_RT (0x2) which
	 * mlx5_ts_format_conv maps to DEFAULT (0x1 = real-time).
	 * Hardcoding FREE_RUNNING (0x0) when FW expects DEFAULT returns
	 * BAD_PARAM (syndrome 0x3b8512).
	 * Do NOT set vsd — VLAN-strip policy is PF-controlled on VFs and
	 * cannot be overridden; attempting to do so also yields 0x3b8512.
	 */
	rq_attr.ts_format = mlx5_ts_format_conv(
			priv->sh->cdev->config.hca_attr.rq_ts_format);

	/* Work Queue attributes for RQ */
	rq_attr.wq_attr.pd = priv->sh->cdev->pdn;
	/*
	 * Do NOT set uar_page for RQ: the standard DPDK driver leaves this 0.
	 * RQ doorbells are written to the DBR, not via a UAR page.
	 * Setting a non-zero UAR page in the RQ WQ context on a VF also
	 * returns BAD_PARAM (syndrome 0x3b8512).
	 */
	rq_attr.wq_attr.log_wq_sz = log_wq_size;
	rq_attr.wq_attr.log_wq_pg_sz = MLX5_LOG_PAGE_SIZE - 12;

	/* Standard cyclic RQ: one 16-byte data segment per WQE. */
	rq_attr.wq_attr.wq_type = MLX5_WQ_TYPE_CYCLIC;
	rq_attr.wq_attr.log_wq_stride =
		rte_log2_u32(sizeof(struct mlx5_wqe_data_seg));

	/* Point WQ to GPU UMEM */
	rq_attr.wq_attr.wq_umem_valid = 1;
	rq_attr.wq_attr.wq_umem_id = params->wq_mem->umem_id;
	rq_attr.wq_attr.wq_umem_offset = 0;

	/* Doorbell record at end of WQ UMEM */
	rq_attr.wq_attr.dbr_umem_valid = 1;
	rq_attr.wq_attr.dbr_umem_id = params->wq_mem->umem_id;
	rq_attr.wq_attr.dbr_addr =
		(uint64_t)(1u << log_wq_size) << rq_attr.wq_attr.log_wq_stride;

	rq_obj = mlx5_devx_cmd_create_rq(ctx, &rq_attr, priv->sh->numa_node);
	if (!rq_obj) {
		DRV_LOG(ERR, "RX queue %u: RQ creation failed", idx);
		ret = -EIO;
		goto error;
	}

	DRV_LOG(DEBUG, "RX queue %u: RQ created, rqn=%u",
		idx, rq_obj->id);

	/* Step 4: Transition RQ from RST to RDY */
	memset(&mrq_attr, 0, sizeof(mrq_attr));
	mrq_attr.rq_state = MLX5_RQC_STATE_RST;
	mrq_attr.state = MLX5_RQC_STATE_RDY;

	ret = mlx5_devx_cmd_modify_rq(rq_obj, &mrq_attr);
	if (ret) {
		DRV_LOG(ERR, "RX queue %u: RQ RST->RDY transition failed", idx);
		ret = -EIO;
		goto error;
	}

	params->rqn = rq_obj->id;

	DRV_LOG(DEBUG, "RX queue %u created: rqn=%u, cqn=%u, wq_size=%u",
		idx, rq_obj->id, cq_obj->id, params->wq_size);

	return 0;

error:
	if (rq_obj)
		mlx5_devx_cmd_destroy(rq_obj);
	if (cq_obj)
		mlx5_devx_cmd_destroy(cq_obj);

	mlx5_gpu_umem_deregister(params->wq_mem);
	mlx5_gpu_umem_deregister(params->cq_mem);

	return ret;
}

/**
 * Get UAR base addresses for GPU doorbell mapping
 *
 * Returns the CPU virtual addresses of the TX and RX UAR pages
 * that rdma-core mmaped during DevX UAR allocation. These are
 * the real NIC MMIO addresses that GINS registers with CUDA
 * (cuMemHostRegister IOMEM) for direct GPU doorbell writes.
 *
 * @param port_id  DPDK port ID
 * @param info     Output: UAR information
 * @return 0 on success, negative error code on failure
 */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_get_uar_info)
__rte_internal
int
mlx5_gpu_get_uar_info(uint16_t port_id,
		      struct mlx5_gpu_uar_info *info)
{
	struct mlx5_priv *priv;

	if (!info)
		return -EINVAL;
	if (port_id >= RTE_MAX_ETHPORTS)
		return -EINVAL;

	priv = rte_eth_devices[port_id].data->dev_private;

	if (!priv->sh->tx_uar.obj) {
		DRV_LOG(ERR, "TX DevX UAR is not allocated");
		return -ENOENT;
	}

	/*
	 * mlx5_os_get_devx_uar_base_addr() returns the CPU virtual
	 * address of the UAR page that rdma-core mmaped from the
	 * NIC's PCI BAR. This is the MMIO address where doorbell
	 * writes go directly to the NIC hardware.
	 */
	info->tx_uar_addr = mlx5_os_get_devx_uar_base_addr(
				priv->sh->tx_uar.obj);
	if (priv->sh->rx_uar.obj) {
		info->rx_uar_addr = mlx5_os_get_devx_uar_base_addr(
					priv->sh->rx_uar.obj);
	} else {
		/* RX queues may share TX UAR on some setups. */
		info->rx_uar_addr = info->tx_uar_addr;
	}
	info->page_size = getpagesize();

	if (!info->tx_uar_addr || !info->rx_uar_addr) {
		DRV_LOG(ERR, "Failed to get UAR base addresses");
		return -EFAULT;
	}

	DRV_LOG(DEBUG, "UAR info: TX=%p, RX=%p, page_size=%zu",
		info->tx_uar_addr, info->rx_uar_addr, info->page_size);

	return 0;
}


/**
 * Register GPU DevX RQ numbers as DPDK "external" RxQs
 *
 * Populates priv->ext_rxqs so that a subsequent rte_flow RSS rule
 * referencing queue IDs starting at RTE_PMD_MLX5_EXTERNAL_RX_QUEUE_ID_MIN
 * maps directly to the caller-created GPU DevX RQs.
 *
 * @param port_id   DPDK port ID
 * @param rqns      Array of RQ numbers (hardware IDs)
 * @param num_rqs   Number of RQs
 * @return 0 on success, negative errno on failure
 */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_enable_ext_rxqs)
__rte_internal
int
mlx5_gpu_enable_ext_rxqs(uint16_t port_id, const uint32_t *rqns, int num_rqs)
{
	struct mlx5_priv *priv;
	int i;

	if (port_id >= RTE_MAX_ETHPORTS)
		return -EINVAL;
	if (!rqns || num_rqs <= 0)
		return -EINVAL;
	if (num_rqs > MLX5_MAX_EXT_RX_QUEUES) {
		DRV_LOG(ERR, "Too many GPU RQs for external mapping: %d (max %u)",
			num_rqs, MLX5_MAX_EXT_RX_QUEUES);
		return -EINVAL;
	}

	priv = rte_eth_devices[port_id].data->dev_private;

	/* Allocate ext_rxqs array if DPDK did not do it automatically
	 * (only done when mlx5_imported_pd_and_ctx() && mlx5_devx_obj_ops_en()). */
	if (!priv->ext_rxqs) {
		priv->ext_rxqs = mlx5_malloc(MLX5_MEM_ZERO | MLX5_MEM_RTE,
					 sizeof(struct mlx5_external_q) *
					 MLX5_MAX_EXT_RX_QUEUES, 0,
					 SOCKET_ID_ANY);
		if (!priv->ext_rxqs) {
			DRV_LOG(ERR, "Failed to allocate ext_rxqs array");
			return -ENOMEM;
		}
	}

	for (i = 0; i < num_rqs; i++) {
		uint16_t ext_idx = (uint16_t)(RTE_PMD_MLX5_EXTERNAL_RX_QUEUE_ID_MIN + i);
		int ret = rte_pmd_mlx5_external_rx_queue_id_map(port_id, ext_idx, rqns[i]);

		/*
		 * EEXIST means the entry was already mapped (e.g. from a
		 * previous run); treat it as success.  Any other error
		 * falls back to direct ext_rxqs population.
		 */
		if (ret < 0 && rte_errno != EEXIST) {
			struct mlx5_external_q *q =
				&priv->ext_rxqs[ext_idx - RTE_PMD_MLX5_EXTERNAL_RX_QUEUE_ID_MIN];
			q->hw_id = rqns[i];
			rte_atomic_store_explicit(&q->refcnt, 1, rte_memory_order_relaxed);
			DRV_LOG(WARNING,
				"External RxQ API map failed (port=%u q=%u rqn=%u errno=%d), using direct map fallback",
				port_id, ext_idx, rqns[i], rte_errno);
		} else {
			DRV_LOG(DEBUG, "GPU RQ %u registered as external dpdk queue index %u",
				rqns[i], ext_idx);
		}
	}

	return 0;
}
