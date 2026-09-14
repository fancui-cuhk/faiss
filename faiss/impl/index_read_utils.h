/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Utils for index_read

#ifndef FAISS_INDEX_READ_UTILS_H
#define FAISS_INDEX_READ_UTILS_H

#include <set>
#include <faiss/IndexIVF.h>
#include <faiss/impl/io.h>

#pragma once

namespace faiss {
struct ProductQuantizer;
struct ScalarQuantizer;

void read_index_header(Index* idx, IOReader* f);
void read_direct_map(DirectMap* dm, IOReader* f);
void read_ivf_dist(IndexIVF* ivf, IOReader* f);
void read_InvertedLists_dist(IndexIVF* ivf, std::set<idx_t>& file_id_set, int io_flags = 0);
void read_InvertedLists_dist(
        IndexIVF* ivf,
        std::set<idx_t>& file_id_set,
        int io_flags,
        const char* invlist_base_path);

/** Load only the requested inverted lists from distributed shards.
 *
 * Uses FileIOReader's FILE* (fseeko/fread). Does not add seek to IOReader.
 * seek_gap_bytes: merge payload holes <= this size into one fread (0 = never).
 */
void read_InvertedLists_dist_selected(
        IndexIVF* ivf,
        const idx_t* list_ids,
        size_t n_lists,
        const idx_t* file_ids,
        size_t seek_gap_bytes,
        const char* invlist_base_path,
        IndexIVF::InvertedListsIOStats* stats);

/** Merge selected inverted lists from distributed shards into RAM.
 *
 * Keeps an existing ArrayInvertedLists (creates an empty one if missing).
 * List ids that already have entries are left unchanged. ntotal becomes
 * the sum of loaded list sizes.
 */
void absorb_InvertedLists_dist_selected(
        IndexIVF* ivf,
        const idx_t* list_ids,
        size_t n_lists,
        const idx_t* file_ids,
        size_t seek_gap_bytes,
        const char* invlist_base_path,
        IndexIVF::InvertedListsIOStats* stats);

/** Replace invlists with empty ArrayInvertedLists; ntotal=0.
 * Keeps the quantizer, list_to_file, and fname.
 */
void init_ram_invlists(IndexIVF* ivf);

/** Install already-in-memory inverted lists into resident ArrayInvertedLists.
 *
 * List ids that already have entries are left unchanged. codes[i] is
 * nvecs[i] * code_size bytes; ids[i] is nvecs[i] idx_t values.
 * ntotal becomes the sum of loaded list sizes.
 */
void install_invlists_from_memory(
        IndexIVF* ivf,
        const idx_t* list_ids,
        size_t n_lists,
        const size_t* nvecs,
        const uint8_t* const* codes,
        const idx_t* const* ids);

void read_ivf_header(
        IndexIVF* ivf,
        IOReader* f,
        std::vector<std::vector<idx_t>>* ids = nullptr);
void read_InvertedLists(IndexIVF* ivf, IOReader* f, int io_flags);
ArrayInvertedLists* set_array_invlist(
        IndexIVF* ivf,
        std::vector<std::vector<idx_t>>& ids);
void read_ProductQuantizer(ProductQuantizer* pq, IOReader* f);
void read_ScalarQuantizer(ScalarQuantizer* ivsc, IOReader* f);

} // namespace faiss

#endif
