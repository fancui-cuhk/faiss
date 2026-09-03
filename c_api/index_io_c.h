/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-
// I/O code for indexes

#ifndef FAISS_INDEX_IO_C_H
#define FAISS_INDEX_IO_C_H

#include <stdio.h>
#include "IndexBinary_c.h"
#include "Index_c.h"
#include "VectorTransform_c.h"
#include "faiss_c.h"
#include "impl/io_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** [DIST] Write index to files -- distributed version of faiss_write_index_fname.
 */
int faiss_write_index_fname_dist(const FaissIndex* idx, const char* fname);

/** [DIST] Read index from file -- distributed version of faiss_read_index_fname.
 */
int faiss_read_index_fname_dist(const char* fname, int io_flags, FaissIndex** p_out);

/** [DIST] Get cluster-to-file mapping from IndexIVF.
 * Returns pointer to internal array (do not free) and number of lists.
 */
int faiss_get_list_to_file_mapping(
        const FaissIndex* idx,
        size_t** list_to_file,
        size_t* nlist);

/** [DIST] Write distributed IVF using explicit cluster groups (one invlist file per group). */
int faiss_write_index_fname_dist_grouped(
        const FaissIndex* idx,
        const char* fname,
        size_t n_groups,
        const size_t* group_sizes,
        const size_t* group_cluster_ids);

/** [STREAMING] Merge per-slice IVF block index files into a single IVF index
 * backed by an OnDiskInvertedLists file. The trained index file provides the
 * empty IVF shell (quantizer + params). Output: out_index_fname (header) +
 * ivfdata_fname (inverted lists data), where the header references ivfdata by
 * the path given here (use IO_FLAG_ONDISK_SAME_DIR when reading).
 */
int faiss_merge_ivf_ondisk(
        const char* trained_index_fname,
        const char* const* block_fnames,
        size_t n_blocks,
        const char* ivfdata_fname,
        const char* out_index_fname);

/** Get IVF coarse centroids (nlist * d floats). Caller does not free centroids pointer. */
int faiss_get_ivf_centroids(
        const FaissIndex* idx,
        float** centroids,
        size_t* nlist,
        size_t* d);

/** Get per-cluster inverted list sizes (number of vectors per list). */
int faiss_get_ivf_cluster_sizes(
        const FaissIndex* idx,
        size_t** sizes,
        size_t* nlist,
        size_t* code_size);

/** Write index to a file.
 * This is equivalent to `faiss::write_index` when a file descriptor is
 * provided.
 */
int faiss_write_index(const FaissIndex* idx, FILE* f);

/** Write index to a file.
 * This is equivalent to `faiss::write_index` when a file path is provided.
 */
int faiss_write_index_fname(const FaissIndex* idx, const char* fname);

/** Write index to a custom writer.
 */
int faiss_write_index_custom(
        const FaissIndex* idx,
        FaissIOWriter* io_writer,
        int io_flags);

#define FAISS_IO_FLAG_MMAP 1
#define FAISS_IO_FLAG_READ_ONLY 2
#define FAISS_IO_FLAG_ONDISK_SAME_DIR 4

/** Read index from a file.
 * This is equivalent to `faiss:read_index` when a file descriptor is given.
 */
int faiss_read_index(FILE* f, int io_flags, FaissIndex** p_out);

/** Read index from a file.
 * This is equivalent to `faiss:read_index` when a file path is given.
 */
int faiss_read_index_fname(const char* fname, int io_flags, FaissIndex** p_out);

/** Read index from a custom reader.
 */
int faiss_read_index_custom(
        FaissIOReader* io_reader,
        int io_flags,
        FaissIndex** p_out);

/** Write index to a file.
 * This is equivalent to `faiss::write_index_binary` when a file descriptor is
 * provided.
 */
int faiss_write_index_binary(const FaissIndexBinary* idx, FILE* f);

/** Write index to a file.
 * This is equivalent to `faiss::write_index_binary` when a file path is
 * provided.
 */
int faiss_write_index_binary_fname(
        const FaissIndexBinary* idx,
        const char* fname);

/** Write binary index to a custom writer.
 */
int faiss_write_index_binary_custom(
        const FaissIndexBinary* idx,
        FaissIOWriter* io_writer);

/** Read index from a file.
 * This is equivalent to `faiss:read_index_binary` when a file descriptor is
 * given.
 */
int faiss_read_index_binary(FILE* f, int io_flags, FaissIndexBinary** p_out);

/** Read index from a file.
 * This is equivalent to `faiss:read_index_binary` when a file path is given.
 */
int faiss_read_index_binary_fname(
        const char* fname,
        int io_flags,
        FaissIndexBinary** p_out);

/** Read binary index from a custom reader.
 */
int faiss_read_index_binary_custom(
        FaissIOReader* io_reader,
        int io_flags,
        FaissIndexBinary** p_out);

/** Read vector transform from a file.
 * This is equivalent to `faiss:read_VectorTransform` when a file path is given.
 */
int faiss_read_VectorTransform_fname(
        const char* fname,
        FaissVectorTransform** p_out);
#ifdef __cplusplus
}
#endif
#endif
