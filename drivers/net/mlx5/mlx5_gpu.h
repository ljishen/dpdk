/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2025 - GPU-Initiated Network Storage (GINS) Project
 */

#ifndef RTE_PMD_MLX5_GPU_H_
#define RTE_PMD_MLX5_GPU_H_

#include <stddef.h>
#include <stdint.h>

#include <mlx5_common_mr.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GPU memory region descriptor for queue creation.
 * GPU memory is registered with the NIC via ibv_reg_mr and
 * devx_umem_reg through nvidia-peermem.
 */
struct mlx5_gpu_mem {
	void *gpu_ptr;          /* GPU virtual address */
	size_t size;            /* Size in bytes */
	struct mlx5_pmd_mr mr;  /* MLX5 memory region */
	void *umem_obj;         /* DevX UMEM object */
	uint32_t umem_id;       /* DevX UMEM ID */
};

/**
 * Parameters for GPU TX queue creation
 */
struct mlx5_gpu_txq_params {
	struct mlx5_gpu_mem *wq_mem;     /* Work Queue memory */
	struct mlx5_gpu_mem *cq_mem;     /* Completion Queue memory */
	uint16_t wq_size;                /* Number of WQEs */
	uint16_t cq_size;                /* Number of CQEs */
	uint32_t sqn;                    /* SQ number (output) */
};

/**
 * Parameters for GPU RX queue creation
 */
struct mlx5_gpu_rxq_params {
	struct mlx5_gpu_mem *wq_mem;     /* Work Queue memory */
	struct mlx5_gpu_mem *cq_mem;     /* Completion Queue memory */
	uint16_t wq_size;                /* Number of WQEs */
	uint16_t cq_size;                /* Number of CQEs */
	uint32_t rqn;                    /* RQ number (output) */
};

/**
 * Register GPU memory for a given port
 *
 * Looks up the protection domain from the DPDK device
 * and registers the GPU memory region with ibverbs.
 *
 * @param port_id  DPDK port ID
 * @param gpu_mem  GPU memory descriptor (gpu_ptr and size must be set)
 * @return 0 on success, negative error code on failure
 */
int mlx5_gpu_register_mem(uint16_t port_id, struct mlx5_gpu_mem *gpu_mem);

/**
 * Deregister GPU memory
 *
 * Deregisters the ibverbs MR and clears the memory region fields.
 *
 * @param gpu_mem  GPU memory descriptor with registered MR
 * @return 0 on success, negative error code on failure
 */
int mlx5_gpu_deregister_mem(struct mlx5_gpu_mem *gpu_mem);

/**
 * Create TX queue with GPU memory
 *
 * @param port_id
 *   DPDK port ID
 * @param idx
 *   Queue index
 * @param params
 *   GPU-specific parameters (sqn populated on success)
 * @return
 *   0 on success, negative error code on failure
 */
int mlx5_gpu_create_txq(uint16_t port_id, uint16_t idx,
			struct mlx5_gpu_txq_params *params);

/**
 * Create RX queue with GPU memory
 *
 * @param port_id
 *   DPDK port ID
 * @param idx
 *   Queue index
 * @param params
 *   GPU-specific parameters (rqn populated on success)
 * @return
 *   0 on success, negative error code on failure
 */
int mlx5_gpu_create_rxq(uint16_t port_id, uint16_t idx,
			struct mlx5_gpu_rxq_params *params);

/**
 * UAR (User Access Region) information for GPU doorbell mapping
 *
 * Contains the CPU virtual addresses of pre-mmapped UAR pages.
 * GINS registers these with CUDA (cuMemHostRegister IOMEM) so that
 * GPU kernels can write doorbells directly to the NIC.
 */
struct mlx5_gpu_uar_info {
	void *tx_uar_addr;    /* CPU VA of TX UAR page (base_addr) */
	void *rx_uar_addr;    /* CPU VA of RX UAR page (base_addr) */
	size_t page_size;     /* UAR page size in bytes (typically 4096) */
};

/**
 * Get UAR base addresses for GPU doorbell mapping
 *
 * Returns the CPU virtual addresses of the TX and RX UAR pages,
 * which were mmaped by rdma-core during DevX UAR allocation.
 * GINS uses these to register the MMIO pages with CUDA for
 * direct GPU doorbell writes.
 *
 * @param port_id
 *   DPDK port ID
 * @param info
 *   Output: UAR information (addresses and page size)
 * @return
 *   0 on success, negative error code on failure
 */
int mlx5_gpu_get_uar_info(uint16_t port_id,
			  struct mlx5_gpu_uar_info *info);

/**
 * Register GPU DevX RQ numbers as DPDK external RxQs
 *
 * Populates priv->ext_rxqs so rte_flow RSS rules can reference the GPU RQs
 * using IDs starting at RTE_PMD_MLX5_EXTERNAL_RX_QUEUE_ID_MIN.
 *
 * @param port_id  DPDK port ID
 * @param rqns     Array of hardware RQ numbers
 * @param num_rqs  Number of RQs to register
 * @return 0 on success, negative errno on failure
 */
int mlx5_gpu_enable_ext_rxqs(uint16_t port_id,
			     const uint32_t *rqns, int num_rqs);

#ifdef __cplusplus
}
#endif

#endif /* RTE_PMD_MLX5_GPU_H_ */
