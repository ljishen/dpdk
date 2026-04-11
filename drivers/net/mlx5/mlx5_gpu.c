/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <errno.h>
#include <string.h>
#include <stdint.h>

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
#include "rte_pmd_mlx5.h"

/** Register GPU memory as DevX UMEM for queue creation. */
static int
mlx5_gpu_umem_register(void *ctx, struct mlx5_gpu_mem *gpu_mem)
{
	if (!ctx || !gpu_mem)
		return -EINVAL;

	gpu_mem->umem_obj = mlx5_glue->devx_umem_reg(ctx,
		gpu_mem->gpu_ptr, gpu_mem->size, IBV_ACCESS_LOCAL_WRITE);
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

/** Deregister GPU UMEM. */
static void
mlx5_gpu_umem_deregister(struct mlx5_gpu_mem *gpu_mem)
{
	if (gpu_mem && gpu_mem->umem_obj) {
		mlx5_glue->devx_umem_dereg(gpu_mem->umem_obj);
		gpu_mem->umem_obj = NULL;
		gpu_mem->umem_id = 0;
	}
}

/** Create a DevX CQ backed by GPU memory (DBR at buffer tail). */
static struct mlx5_devx_obj *
mlx5_gpu_cq_create(struct mlx5_priv *priv, struct mlx5_gpu_mem *cq_mem,
		    uint32_t log_cq_size)
{
	void *ctx = priv->sh->cdev->ctx;
	uint32_t eqn;
	struct mlx5_devx_cq_attr cq_attr;

	if (mlx5_glue->devx_query_eqn(ctx, 0, &eqn)) {
		DRV_LOG(ERR, "Failed to query EQN for CQ creation");
		return NULL;
	}

	/* Shrink CQ so buffer + DBR both fit in the registered UMEM. */
	while (log_cq_size > 0 &&
	       ((1u << log_cq_size) * sizeof(struct mlx5_cqe) + MLX5_DBR_SIZE)
	       > cq_mem->size)
		log_cq_size--;

	memset(&cq_attr, 0, sizeof(cq_attr));
	cq_attr.q_umem_valid  = 1;
	cq_attr.q_umem_id     = cq_mem->umem_id;
	cq_attr.log_cq_size   = log_cq_size;
	cq_attr.log_page_size = MLX5_LOG_PAGE_SIZE - 12;
	cq_attr.uar_page_id   = mlx5_os_get_devx_uar_page_id(
					priv->sh->tx_uar.obj);
	cq_attr.eqn = eqn;

	/* DBR at CQ buffer tail. */
	cq_attr.db_umem_valid  = 1;
	cq_attr.db_umem_id     = cq_mem->umem_id;
	cq_attr.db_umem_offset =
		(uint64_t)(1u << log_cq_size) * sizeof(struct mlx5_cqe);

	return mlx5_devx_cmd_create_cq(ctx, &cq_attr);
}

/** Register GPU memory with NIC for DMA (nvidia-peermem). */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_register_mem)
__rte_internal
int
mlx5_gpu_register_mem(uint16_t port_id, struct mlx5_gpu_mem *gpu_mem)
{
	struct mlx5_priv *priv;
	void *pd;
	struct ibv_mr *ibv_mr;

	if (!gpu_mem || !gpu_mem->gpu_ptr || gpu_mem->size == 0)
		return -EINVAL;
	if (port_id >= RTE_MAX_ETHPORTS)
		return -EINVAL;

	priv = rte_eth_devices[port_id].data->dev_private;
	pd = priv->sh->cdev->pd;

	ibv_mr = mlx5_glue->reg_mr(pd, gpu_mem->gpu_ptr, gpu_mem->size,
				    IBV_ACCESS_LOCAL_WRITE);
	if (!ibv_mr) {
		DRV_LOG(ERR, "ibv_reg_mr GPU ptr=%p size=%zu failed: %s",
			gpu_mem->gpu_ptr, gpu_mem->size, strerror(errno));
		return -errno;
	}

	gpu_mem->mr = (struct mlx5_pmd_mr){
		.lkey = ibv_mr->lkey,
		.addr = gpu_mem->gpu_ptr,
		.len  = gpu_mem->size,
		.obj  = ibv_mr,
	};

	DRV_LOG(DEBUG, "GPU MR registered: ptr=%p size=%zu lkey=0x%x",
		gpu_mem->gpu_ptr, gpu_mem->size, gpu_mem->mr.lkey);
	return 0;
}

/** Deregister GPU memory (ibverbs MR). */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_deregister_mem)
__rte_internal
int
mlx5_gpu_deregister_mem(struct mlx5_gpu_mem *gpu_mem)
{
	if (!gpu_mem)
		return -EINVAL;

	if (gpu_mem->mr.obj) {
		mlx5_glue->dereg_mr((struct ibv_mr *)gpu_mem->mr.obj);
		memset(&gpu_mem->mr, 0, sizeof(gpu_mem->mr));
	}
	return 0;
}

/** Common state carried between gpu_queue_setup() and gpu_queue_teardown(). */
struct gpu_qp_ctx {
	struct mlx5_priv     *priv;   /**< Device private data. */
	void                 *ctx;    /**< Device context. */
	struct mlx5_devx_obj *cq_obj; /**< Created CQ object. */
};

/**
 * Shared preamble for TX/RX queue creation.
 *
 * Validates MR registration, UMEM-registers both regions, and creates CQ.
 * On failure all partial work is undone; the caller need not clean up.
 */
static int
gpu_queue_setup(uint16_t port_id, uint16_t idx, const char *dir,
		struct mlx5_gpu_mem *wq_mem, struct mlx5_gpu_mem *cq_mem,
		uint16_t cq_size, struct gpu_qp_ctx *qc)
{
	int ret;

	qc->priv   = rte_eth_devices[port_id].data->dev_private;
	qc->ctx    = qc->priv->sh->cdev->ctx;
	qc->cq_obj = NULL;

	if (wq_mem->mr.lkey == 0 || cq_mem->mr.lkey == 0) {
		DRV_LOG(ERR, "GPU memory not registered before queue creation");
		return -EINVAL;
	}

	ret = mlx5_gpu_umem_register(qc->ctx, wq_mem);
	if (ret < 0) {
		DRV_LOG(ERR, "%s queue %u: WQ UMEM registration failed",
			dir, idx);
		return ret;
	}

	ret = mlx5_gpu_umem_register(qc->ctx, cq_mem);
	if (ret < 0) {
		DRV_LOG(ERR, "%s queue %u: CQ UMEM registration failed",
			dir, idx);
		mlx5_gpu_umem_deregister(wq_mem);
		return ret;
	}

	qc->cq_obj = mlx5_gpu_cq_create(qc->priv, cq_mem,
					 rte_log2_u32(cq_size));
	if (!qc->cq_obj) {
		DRV_LOG(ERR, "%s queue %u: CQ creation failed", dir, idx);
		mlx5_gpu_umem_deregister(cq_mem);
		mlx5_gpu_umem_deregister(wq_mem);
		return -EIO;
	}

	return 0;
}

/** Roll back a failed queue creation (destroy DevX + deregister UMEMs). */
static void
gpu_queue_teardown(struct mlx5_devx_obj *q_obj, struct gpu_qp_ctx *qc,
		   struct mlx5_gpu_mem *wq_mem, struct mlx5_gpu_mem *cq_mem)
{
	if (q_obj)
		mlx5_devx_cmd_destroy(q_obj);
	if (qc->cq_obj)
		mlx5_devx_cmd_destroy(qc->cq_obj);
	mlx5_gpu_umem_deregister(wq_mem);
	mlx5_gpu_umem_deregister(cq_mem);
}

/** Create TX queue with GPU memory via DevX. */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_create_txq)
__rte_internal
int
mlx5_gpu_create_txq(uint16_t port_id, uint16_t idx,
		    struct mlx5_gpu_txq_params *params)
{
	struct gpu_qp_ctx qc;
	struct mlx5_devx_obj *sq_obj = NULL;
	struct mlx5_devx_create_sq_attr sq_attr;
	struct mlx5_devx_modify_sq_attr msq_attr;
	uint32_t log_wq_size;
	int ret;

	if (!params || !params->wq_mem || !params->cq_mem)
		return -EINVAL;
	if (port_id >= RTE_MAX_ETHPORTS)
		return -EINVAL;

	ret = gpu_queue_setup(port_id, idx, "TX", params->wq_mem,
			      params->cq_mem, params->cq_size, &qc);
	if (ret)
		return ret;

	log_wq_size = rte_log2_u32(params->wq_size);

	memset(&sq_attr, 0, sizeof(sq_attr));
	sq_attr.state = MLX5_SQC_STATE_RST;
	sq_attr.cqn   = qc.cq_obj->id;
	sq_attr.flush_in_error_en = 1;
	sq_attr.min_wqe_inline_mode =
		qc.priv->sh->cdev->config.hca_attr.vport_inline_mode;
	sq_attr.tis_num    = qc.priv->sh->tis[0]->id;
	sq_attr.tis_lst_sz = 1;
	sq_attr.ts_format  = mlx5_ts_format_conv(
			qc.priv->sh->cdev->config.hca_attr.sq_ts_format);

	sq_attr.wq_attr.wq_type        = MLX5_WQ_TYPE_CYCLIC;
	sq_attr.wq_attr.pd             = qc.priv->sh->cdev->pdn;
	sq_attr.wq_attr.uar_page       = (params->uar && params->uar->obj)
		? params->uar->page_id
		: mlx5_os_get_devx_uar_page_id(qc.priv->sh->tx_uar.obj);
	sq_attr.wq_attr.log_wq_stride  = rte_log2_u32(MLX5_WQE_SIZE);
	sq_attr.wq_attr.log_wq_sz      = log_wq_size;
	sq_attr.wq_attr.log_wq_pg_sz   = MLX5_LOG_PAGE_SIZE - 12;
	sq_attr.wq_attr.wq_umem_valid  = 1;
	sq_attr.wq_attr.wq_umem_id     = params->wq_mem->umem_id;
	sq_attr.wq_attr.dbr_umem_valid = 1;
	sq_attr.wq_attr.dbr_umem_id    = params->wq_mem->umem_id;
	sq_attr.wq_attr.dbr_addr =
		(uint64_t)(1u << log_wq_size) * MLX5_WQE_SIZE;

	sq_obj = mlx5_devx_cmd_create_sq(qc.ctx, &sq_attr);
	if (!sq_obj) {
		DRV_LOG(ERR, "TX queue %u: SQ creation failed", idx);
		ret = -EIO;
		goto error;
	}

	/* RST → RDY */
	memset(&msq_attr, 0, sizeof(msq_attr));
	msq_attr.sq_state = MLX5_SQC_STATE_RST;
	msq_attr.state    = MLX5_SQC_STATE_RDY;
	ret = mlx5_devx_cmd_modify_sq(sq_obj, &msq_attr);
	if (ret) {
		ret = -EIO;
		goto error;
	}

	params->sqn = sq_obj->id;
	DRV_LOG(DEBUG, "TX queue %u created: sqn=%u cqn=%u wq_size=%u",
		idx, sq_obj->id, qc.cq_obj->id, params->wq_size);
	return 0;

error:
	gpu_queue_teardown(sq_obj, &qc, params->wq_mem, params->cq_mem);
	return ret;
}

/** Create RX queue with GPU memory via DevX. */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_create_rxq)
__rte_internal
int
mlx5_gpu_create_rxq(uint16_t port_id, uint16_t idx,
		    struct mlx5_gpu_rxq_params *params)
{
	struct gpu_qp_ctx qc;
	struct mlx5_devx_obj *rq_obj = NULL;
	struct mlx5_devx_create_rq_attr rq_attr;
	struct mlx5_devx_modify_rq_attr mrq_attr;
	uint32_t log_wq_size;
	int ret;

	if (!params || !params->wq_mem || !params->cq_mem)
		return -EINVAL;
	if (port_id >= RTE_MAX_ETHPORTS)
		return -EINVAL;

	ret = gpu_queue_setup(port_id, idx, "RX", params->wq_mem,
			      params->cq_mem, params->cq_size, &qc);
	if (ret)
		return ret;

	log_wq_size = rte_log2_u32(params->wq_size);

	memset(&rq_attr, 0, sizeof(rq_attr));
	rq_attr.state        = MLX5_RQC_STATE_RST;
	rq_attr.cqn          = qc.cq_obj->id;
	rq_attr.mem_rq_type  = MLX5_RQC_MEM_RQ_TYPE_MEMORY_RQ_INLINE;
	/*
	 * Use HCA-advertised ts_format; hardcoding FREE_RUNNING may fail
	 * with BAD_PARAM on certain NIC/FW combinations.
	 * Do NOT set vsd — VLAN-strip is PF-controlled on VFs.
	 */
	rq_attr.ts_format = mlx5_ts_format_conv(
			qc.priv->sh->cdev->config.hca_attr.rq_ts_format);

	rq_attr.wq_attr.wq_type        = MLX5_WQ_TYPE_CYCLIC;
	rq_attr.wq_attr.pd             = qc.priv->sh->cdev->pdn;
	/* RQ doorbells use DBR, not UAR — leave uar_page at zero. */
	rq_attr.wq_attr.log_wq_stride  =
		rte_log2_u32(sizeof(struct mlx5_wqe_data_seg));
	rq_attr.wq_attr.log_wq_sz      = log_wq_size;
	rq_attr.wq_attr.log_wq_pg_sz   = MLX5_LOG_PAGE_SIZE - 12;
	rq_attr.wq_attr.wq_umem_valid  = 1;
	rq_attr.wq_attr.wq_umem_id     = params->wq_mem->umem_id;
	rq_attr.wq_attr.dbr_umem_valid = 1;
	rq_attr.wq_attr.dbr_umem_id    = params->wq_mem->umem_id;
	rq_attr.wq_attr.dbr_addr =
		(uint64_t)(1u << log_wq_size) * sizeof(struct mlx5_wqe_data_seg);

	rq_obj = mlx5_devx_cmd_create_rq(qc.ctx, &rq_attr,
					  qc.priv->sh->numa_node);
	if (!rq_obj) {
		DRV_LOG(ERR, "RX queue %u: RQ creation failed", idx);
		ret = -EIO;
		goto error;
	}

	/* RST → RDY */
	memset(&mrq_attr, 0, sizeof(mrq_attr));
	mrq_attr.rq_state = MLX5_RQC_STATE_RST;
	mrq_attr.state    = MLX5_RQC_STATE_RDY;
	ret = mlx5_devx_cmd_modify_rq(rq_obj, &mrq_attr);
	if (ret) {
		ret = -EIO;
		goto error;
	}

	params->rqn = rq_obj->id;
	DRV_LOG(DEBUG, "RX queue %u created: rqn=%u cqn=%u wq_size=%u",
		idx, rq_obj->id, qc.cq_obj->id, params->wq_size);
	return 0;

error:
	gpu_queue_teardown(rq_obj, &qc, params->wq_mem, params->cq_mem);
	return ret;
}

/**
 * Allocate a per-queue UAR via DevX.
 *
 * Retries up to 8 times to ensure the returned UAR has a valid
 * (non-NULL) base address, matching the upstream mlx5_devx_alloc_uar()
 * retry logic.
 */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_alloc_uar)
__rte_internal
int
mlx5_gpu_alloc_uar(uint16_t port_id, struct mlx5_gpu_uar *uar)
{
	struct mlx5_priv *priv;
	void *uar_obj = NULL;
	void *base_addr = NULL;
	uint32_t retry;

	if (!uar || port_id >= RTE_MAX_ETHPORTS)
		return -EINVAL;

	memset(uar, 0, sizeof(*uar));
	priv = rte_eth_devices[port_id].data->dev_private;

	for (retry = 0; retry < 8; ++retry) {
#ifdef MLX5DV_UAR_ALLOC_TYPE_NC
		uar_obj = mlx5_glue->devx_alloc_uar(
				priv->sh->cdev->ctx,
				MLX5DV_UAR_ALLOC_TYPE_NC);
		if (!uar_obj)
			uar_obj = mlx5_glue->devx_alloc_uar(
					priv->sh->cdev->ctx,
					MLX5DV_UAR_ALLOC_TYPE_BF);
#else
		uar_obj = mlx5_glue->devx_alloc_uar(
				priv->sh->cdev->ctx, 0);
#endif
		if (!uar_obj) {
			DRV_LOG(ERR, "devx_alloc_uar failed for per-queue UAR");
			return -ENOMEM;
		}
		base_addr = mlx5_os_get_devx_uar_base_addr(uar_obj);
		if (base_addr)
			break;
		DRV_LOG(DEBUG, "Per-queue UAR retry %u (NULL base)", retry);
		mlx5_glue->devx_free_uar(uar_obj);
		uar_obj = NULL;
	}

	if (!uar_obj || !base_addr) {
		DRV_LOG(ERR, "Failed to allocate per-queue UAR with valid base");
		return -ENOMEM;
	}

	uar->obj = uar_obj;
	uar->base_addr = base_addr;
	uar->reg_addr = mlx5_os_get_devx_uar_reg_addr(uar_obj);
	uar->page_id = mlx5_os_get_devx_uar_page_id(uar_obj);

	DRV_LOG(DEBUG, "Per-queue UAR allocated: base=%p reg=%p page_id=%u",
		uar->base_addr, uar->reg_addr, uar->page_id);
	return 0;
}

/** Free a per-queue UAR. */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_free_uar)
__rte_internal
void
mlx5_gpu_free_uar(struct mlx5_gpu_uar *uar)
{
	if (uar && uar->obj) {
		mlx5_glue->devx_free_uar(uar->obj);
		memset(uar, 0, sizeof(*uar));
	}
}

/** Register GPU DevX RQ numbers as DPDK external RxQs. */
RTE_EXPORT_INTERNAL_SYMBOL(mlx5_gpu_enable_ext_rxqs)
__rte_internal
int
mlx5_gpu_enable_ext_rxqs(uint16_t port_id, const uint32_t *rqns, int num_rqs)
{
	struct mlx5_priv *priv;
	int i;

	if (port_id >= RTE_MAX_ETHPORTS || !rqns || num_rqs <= 0)
		return -EINVAL;
	if (num_rqs > MLX5_MAX_EXT_RX_QUEUES) {
		DRV_LOG(ERR, "Too many GPU RQs: %d (max %u)",
			num_rqs, MLX5_MAX_EXT_RX_QUEUES);
		return -EINVAL;
	}

	priv = rte_eth_devices[port_id].data->dev_private;

	if (!priv->ext_rxqs) {
		priv->ext_rxqs = mlx5_malloc(MLX5_MEM_ZERO | MLX5_MEM_RTE,
					     sizeof(*priv->ext_rxqs) *
					     MLX5_MAX_EXT_RX_QUEUES,
					     0, SOCKET_ID_ANY);
		if (!priv->ext_rxqs)
			return -ENOMEM;
	}

	for (i = 0; i < num_rqs; i++) {
		uint16_t ext_idx = (uint16_t)
			(RTE_PMD_MLX5_EXTERNAL_RX_QUEUE_ID_MIN + i);
		int ret = rte_pmd_mlx5_external_rx_queue_id_map(
				port_id, ext_idx, rqns[i]);

		/*
		 * EEXIST = already mapped (previous run); anything else
		 * falls back to direct ext_rxqs population.
		 */
		if (ret < 0 && rte_errno != EEXIST) {
			struct mlx5_external_q *q = &priv->ext_rxqs[
				ext_idx - RTE_PMD_MLX5_EXTERNAL_RX_QUEUE_ID_MIN];
			q->hw_id = rqns[i];
			rte_atomic_store_explicit(&q->refcnt, 1,
						  rte_memory_order_relaxed);
			DRV_LOG(WARNING,
				"ext_rxq map fallback: port=%u q=%u rqn=%u",
				port_id, ext_idx, rqns[i]);
		}
	}

	return 0;
}
