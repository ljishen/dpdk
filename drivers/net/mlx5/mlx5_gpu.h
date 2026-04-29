/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
 * GPU memory registered with the NIC via ibv_reg_mr and
 * devx_umem_reg through nvidia-peermem.
 */
struct mlx5_gpu_mem {
	void *gpu_ptr;          /**< GPU virtual address. */
	size_t size;            /**< Size in bytes. */
	struct mlx5_pmd_mr mr;  /**< MLX5 memory region. */
	void *umem_obj;         /**< DevX UMEM object. */
	uint32_t umem_id;       /**< DevX UMEM ID. */
};

/**
 * Per-queue UAR (User Access Region) for independent doorbell mapping.
 *
 * Each TX queue gets its own UAR page so the GPU can write doorbells
 * to distinct MMIO registers without serialization. UAR is allocated
 * Non-Cached (NC); the GPU only writes the 8-byte doorbell trigger to
 * @c reg_addr and the NIC DMA-fetches the WQE from the SQ ring.
 */
struct mlx5_gpu_uar {
	void *obj;          /**< DevX UAR object (opaque). */
	void *reg_addr;     /**< CPU VA of the NC doorbell register. */
	uint32_t page_id;   /**< UAR page ID for SQ creation. */
};

/** Parameters for GPU TX queue creation. */
struct mlx5_gpu_txq_params {
	struct mlx5_gpu_mem *wq_mem;     /**< Work Queue memory. */
	struct mlx5_gpu_mem *cq_mem;     /**< Completion Queue memory. */
	uint16_t wq_size;                /**< Number of WQEs. */
	uint16_t cq_size;                /**< Number of CQEs. */
	uint32_t sqn;                    /**< SQ number (output). */
	struct mlx5_gpu_uar *uar;        /**< Per-queue UAR (NULL = use shared). */
};

/** Parameters for GPU RX queue creation. */
struct mlx5_gpu_rxq_params {
	struct mlx5_gpu_mem *wq_mem;     /**< Work Queue memory. */
	struct mlx5_gpu_mem *cq_mem;     /**< Completion Queue memory. */
	uint16_t wq_size;                /**< Number of WQEs. */
	uint16_t cq_size;                /**< Number of CQEs. */
	uint32_t rqn;                    /**< RQ number (output). */
};

/**
 * Register GPU memory with the NIC for a given port.
 *
 * @param[in] port_id
 *   DPDK port ID.
 * @param[in] gpu_mem
 *   GPU memory descriptor (gpu_ptr and size must be set).
 *
 * @return
 *   0 on success, negative error code on failure.
 */
int mlx5_gpu_register_mem(uint16_t port_id, struct mlx5_gpu_mem *gpu_mem);

/**
 * Deregister GPU memory (ibverbs MR).
 *
 * @param[in] gpu_mem
 *   GPU memory descriptor with registered MR.
 *
 * @return
 *   0 on success, negative error code on failure.
 */
int mlx5_gpu_deregister_mem(struct mlx5_gpu_mem *gpu_mem);

/**
 * Create TX queue backed by GPU memory.
 *
 * @param[in] port_id
 *   DPDK port ID.
 * @param[in] idx
 *   Queue index.
 * @param[in,out] params
 *   GPU TX queue parameters; sqn is populated on success.
 *
 * @return
 *   0 on success, negative error code on failure.
 */
int mlx5_gpu_create_txq(uint16_t port_id, uint16_t idx,
			struct mlx5_gpu_txq_params *params);

/**
 * Create RX queue backed by GPU memory.
 *
 * @param[in] port_id
 *   DPDK port ID.
 * @param[in] idx
 *   Queue index.
 * @param[in,out] params
 *   GPU RX queue parameters; rqn is populated on success.
 *
 * @return
 *   0 on success, negative error code on failure.
 */
int mlx5_gpu_create_rxq(uint16_t port_id, uint16_t idx,
			struct mlx5_gpu_rxq_params *params);

/**
 * Register GPU DevX RQ numbers as DPDK external RxQs.
 *
 * Populates priv->ext_rxqs starting at
 * RTE_PMD_MLX5_EXTERNAL_RX_QUEUE_ID_MIN.
 *
 * @param[in] port_id
 *   DPDK port ID.
 * @param[in] rqns
 *   Array of hardware RQ numbers.
 * @param[in] num_rqs
 *   Number of RQs to register.
 *
 * @return
 *   0 on success, negative errno on failure.
 */
int mlx5_gpu_enable_ext_rxqs(uint16_t port_id,
			     const uint32_t *rqns, int num_rqs);

/**
 * Allocate a per-queue UAR (DevX) for independent doorbell mapping.
 *
 * The returned UAR page ID is passed to mlx5_gpu_create_txq() so the
 * SQ is bound to this UAR.  The base_addr is mapped to GPU via CUDA
 * IOMEM for direct doorbell writes.
 *
 * @param[in] port_id
 *   DPDK port ID.
 * @param[out] uar
 *   Per-queue UAR descriptor populated on success.
 *
 * @return
 *   0 on success, negative error code on failure.
 */
int mlx5_gpu_alloc_uar(uint16_t port_id, struct mlx5_gpu_uar *uar);

/**
 * Free a per-queue UAR allocated by mlx5_gpu_alloc_uar().
 *
 * @param[in] uar
 *   Per-queue UAR descriptor.
 */
void mlx5_gpu_free_uar(struct mlx5_gpu_uar *uar);

#ifdef __cplusplus
}
#endif

#endif /* RTE_PMD_MLX5_GPU_H_ */
