/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "Index_c.h"
#include <faiss/Index.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/index_read_utils.h>
#include <string>
#include "macros_impl.h"

extern "C" {

DEFINE_DESTRUCTOR(SearchParameters)

int faiss_SearchParameters_new(
        FaissSearchParameters** p_sp,
        FaissIDSelector* sel) {
    try {
        faiss::SearchParameters* params = new faiss::SearchParameters;
        params->sel = reinterpret_cast<faiss::IDSelector*>(sel);
        *p_sp = reinterpret_cast<FaissSearchParameters*>(params);
        return 0;
    }
    CATCH_AND_HANDLE
}

DEFINE_DESTRUCTOR(Index)

DEFINE_GETTER(Index, int, d)

DEFINE_GETTER(Index, int, is_trained)

int faiss_Index_set_is_trained(FaissIndex* index, int is_trained) {
    try {
        reinterpret_cast<faiss::Index*>(index)->is_trained = is_trained != 0;
        return 0;
    }
    CATCH_AND_HANDLE
}

DEFINE_GETTER(Index, idx_t, ntotal)

DEFINE_GETTER(Index, FaissMetricType, metric_type)

DEFINE_GETTER(Index, int, verbose);
DEFINE_SETTER(Index, int, verbose);

int faiss_Index_train(FaissIndex* index, idx_t n, const float* x) {
    try {
        reinterpret_cast<faiss::Index*>(index)->train(n, x);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_add(FaissIndex* index, idx_t n, const float* x) {
    try {
        reinterpret_cast<faiss::Index*>(index)->add(n, x);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_add_with_ids(
        FaissIndex* index,
        idx_t n,
        const float* x,
        const idx_t* xids) {
    try {
        reinterpret_cast<faiss::Index*>(index)->add_with_ids(n, x, xids);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_search(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->search(
                n, x, k, distances, labels);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_search_with_params(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        idx_t k,
        const FaissSearchParameters* params,
        float* distances,
        idx_t* labels) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->search(
                n,
                x,
                k,
                distances,
                labels,
                reinterpret_cast<const faiss::SearchParameters*>(params));
    }
    CATCH_AND_HANDLE
}

int faiss_Index_range_search(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        float radius,
        FaissRangeSearchResult* result) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->range_search(
                n,
                x,
                radius,
                reinterpret_cast<faiss::RangeSearchResult*>(result));
    }
    CATCH_AND_HANDLE
}

int faiss_Index_assign(
        FaissIndex* index,
        idx_t n,
        const float* x,
        idx_t* labels,
        idx_t k) {
    try {
        reinterpret_cast<faiss::Index*>(index)->assign(n, x, labels, k);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_reset(FaissIndex* index) {
    try {
        reinterpret_cast<faiss::Index*>(index)->reset();
    }
    CATCH_AND_HANDLE
}

int faiss_Index_remove_ids(
        FaissIndex* index,
        const FaissIDSelector* sel,
        size_t* n_removed) {
    try {
        size_t n{reinterpret_cast<faiss::Index*>(index)->remove_ids(
                *reinterpret_cast<const faiss::IDSelector*>(sel))};
        if (n_removed) {
            *n_removed = n;
        }
    }
    CATCH_AND_HANDLE
}

int faiss_Index_reconstruct(const FaissIndex* index, idx_t key, float* recons) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->reconstruct(key, recons);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_reconstruct_n(
        const FaissIndex* index,
        idx_t i0,
        idx_t ni,
        float* recons) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->reconstruct_n(
                i0, ni, recons);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_compute_residual(
        const FaissIndex* index,
        const float* x,
        float* residual,
        idx_t key) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->compute_residual(
                x, residual, key);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_compute_residual_n(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        float* residuals,
        const idx_t* keys) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->compute_residual_n(
                n, x, residuals, keys);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_sa_code_size(const FaissIndex* index, size_t* size) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->sa_code_size();
    }
    CATCH_AND_HANDLE
}

int faiss_Index_sa_encode(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        uint8_t* bytes) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->sa_encode(n, x, bytes);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_sa_decode(
        const FaissIndex* index,
        idx_t n,
        const uint8_t* bytes,
        float* x) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->sa_decode(n, bytes, x);
    }
    CATCH_AND_HANDLE
}

int faiss_select_clusters(
        const FaissIndex* index,
        idx_t n,
        size_t nprobe,
        const float* x,
        float* distances,
        idx_t* labels,
        idx_t* file_ids) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->select_clusters(
                n,
                nprobe,
                x,
                distances,
                labels,
                file_ids,
                nullptr);
    }
    CATCH_AND_HANDLE
}

int faiss_probe_clusters(
        FaissIndex* index,
        idx_t n,
        const float* x,
        idx_t k,
        size_t nclusters,
        const idx_t* cluster_ids,
        const idx_t* file_ids,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        const char* invlist_path,
        size_t seek_gap_bytes,
        FaissInvertedListsIOStats* io_stats) {
    try {
        auto* ivf = dynamic_cast<faiss::IndexIVF*>(
                reinterpret_cast<faiss::Index*>(index));
        FAISS_THROW_IF_MSG(ivf == nullptr, "probe_clusters requires IndexIVF");
        std::string old_fname = ivf->fname;
        if (invlist_path != nullptr && invlist_path[0] != '\0') {
            ivf->fname = invlist_path;
        }
        ivf->invlist_seek_gap_bytes = seek_gap_bytes;
        ivf->probe_clusters(
                n,
                x,
                k,
                nclusters,
                cluster_ids,
                file_ids,
                centroid_dis,
                distances,
                labels);
        if (invlist_path != nullptr && invlist_path[0] != '\0') {
            ivf->fname = old_fname;
        }
        if (io_stats != nullptr) {
            const auto& st = ivf->last_invlist_io_stats;
            io_stats->table_bytes = st.table_bytes;
            io_stats->payload_bytes = st.payload_bytes;
            io_stats->skip_bytes = st.skip_bytes;
            io_stats->read_ops = st.read_ops;
            io_stats->merged_ranges = st.merged_ranges;
        }
    }
    CATCH_AND_HANDLE
}
}
