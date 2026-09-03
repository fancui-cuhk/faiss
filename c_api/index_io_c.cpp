/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-
// I/O code for indexes

#include "index_io_c.h"
#include <faiss/index_io.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/invlists/InvertedLists.h>
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexBinary;
using faiss::IOReader;
using faiss::IOWriter;
using faiss::VectorTransform;

int faiss_write_index_fname_dist(const FaissIndex* idx, const char* fname) {
    try {
        faiss::write_index_dist(reinterpret_cast<const Index*>(idx), fname);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_fname_dist(
        const char* fname,
        int io_flags,
        FaissIndex** p_out) {
    try {
        auto out = faiss::read_index_dist(fname, io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

// [DIST] Get cluster-to-file mapping from IndexIVF
int faiss_get_list_to_file_mapping(
        const FaissIndex* idx,
        size_t** list_to_file,
        size_t* nlist) {
    try {
        const faiss::IndexIVF* ivf = dynamic_cast<const faiss::IndexIVF*>(
            reinterpret_cast<const faiss::Index*>(idx));
        FAISS_THROW_IF_MSG(ivf == nullptr, "Index is not of type IndexIVF");
        
        *list_to_file = const_cast<size_t*>(ivf->list_to_file.data());
        *nlist = ivf->list_to_file.size();
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_fname_dist_grouped(
        const FaissIndex* idx,
        const char* fname,
        size_t n_groups,
        const size_t* group_sizes,
        const size_t* group_cluster_ids) {
    try {
        std::vector<std::vector<size_t>> groups(n_groups);
        size_t offset = 0;
        for (size_t g = 0; g < n_groups; g++) {
            groups[g].resize(group_sizes[g]);
            for (size_t j = 0; j < group_sizes[g]; j++) {
                groups[g][j] = group_cluster_ids[offset + j];
            }
            offset += group_sizes[g];
        }
        faiss::write_index_dist_grouped(
                reinterpret_cast<const faiss::Index*>(idx),
                fname,
                groups);
    }
    CATCH_AND_HANDLE
}

int faiss_merge_ivf_ondisk(
        const char* trained_index_fname,
        const char* const* block_fnames,
        size_t n_blocks,
        const char* ivfdata_fname,
        const char* out_index_fname) {
    try {
        std::vector<std::string> blocks(n_blocks);
        for (size_t i = 0; i < n_blocks; i++) {
            blocks[i] = block_fnames[i];
        }
        faiss::merge_ivf_ondisk(
                trained_index_fname,
                blocks,
                ivfdata_fname,
                out_index_fname);
    }
    CATCH_AND_HANDLE
}

int faiss_get_ivf_centroids(
        const FaissIndex* idx,
        float** centroids,
        size_t* nlist,
        size_t* d) {
    try {
        const faiss::IndexIVF* ivf = dynamic_cast<const faiss::IndexIVF*>(
                reinterpret_cast<const faiss::Index*>(idx));
        FAISS_THROW_IF_MSG(ivf == nullptr, "Index is not of type IndexIVF");
        const faiss::IndexFlat* flat =
                dynamic_cast<const faiss::IndexFlat*>(ivf->quantizer);
        FAISS_THROW_IF_MSG(flat == nullptr, "IVF quantizer is not IndexFlat");
        *nlist = ivf->nlist;
        *d = ivf->d;
        *centroids = const_cast<float*>(flat->get_xb());
    }
    CATCH_AND_HANDLE
}

int faiss_get_ivf_cluster_sizes(
        const FaissIndex* idx,
        size_t** sizes,
        size_t* nlist,
        size_t* code_size) {
    try {
        const faiss::IndexIVF* ivf = dynamic_cast<const faiss::IndexIVF*>(
                reinterpret_cast<const faiss::Index*>(idx));
        FAISS_THROW_IF_MSG(ivf == nullptr, "Index is not of type IndexIVF");
        const faiss::InvertedLists* ils = ivf->invlists;
        FAISS_THROW_IF_MSG(ils == nullptr, "IVF invlists is nullptr");
        *nlist = ils->nlist;
        *code_size = ils->code_size;
        static thread_local std::vector<size_t> tl_sizes;
        tl_sizes.resize(ils->nlist);
        for (size_t i = 0; i < ils->nlist; i++) {
            tl_sizes[i] = ils->list_size(i);
        }
        *sizes = tl_sizes.data();
    }
    CATCH_AND_HANDLE
}

int faiss_write_index(const FaissIndex* idx, FILE* f) {
    try {
        faiss::write_index(reinterpret_cast<const Index*>(idx), f);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_fname(const FaissIndex* idx, const char* fname) {
    try {
        faiss::write_index(reinterpret_cast<const Index*>(idx), fname);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_custom(
        const FaissIndex* idx,
        FaissIOWriter* io_writer,
        int io_flags) {
    try {
        faiss::write_index(
                reinterpret_cast<const Index*>(idx),
                reinterpret_cast<IOWriter*>(io_writer),
                io_flags);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index(FILE* f, int io_flags, FaissIndex** p_out) {
    try {
        auto out = faiss::read_index(f, io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_fname(
        const char* fname,
        int io_flags,
        FaissIndex** p_out) {
    try {
        auto out = faiss::read_index(fname, io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_custom(
        FaissIOReader* io_reader,
        int io_flags,
        FaissIndex** p_out) {
    try {
        auto out = faiss::read_index(
                reinterpret_cast<IOReader*>(io_reader), io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary(const FaissIndexBinary* idx, FILE* f) {
    try {
        faiss::write_index_binary(reinterpret_cast<const IndexBinary*>(idx), f);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary_fname(
        const FaissIndexBinary* idx,
        const char* fname) {
    try {
        faiss::write_index_binary(
                reinterpret_cast<const IndexBinary*>(idx), fname);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary_custom(
        const FaissIndexBinary* idx,
        FaissIOWriter* io_writer) {
    try {
        faiss::write_index_binary(
                reinterpret_cast<const IndexBinary*>(idx),
                reinterpret_cast<IOWriter*>(io_writer));
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_binary(FILE* f, int io_flags, FaissIndexBinary** p_out) {
    try {
        auto out = faiss::read_index_binary(f, io_flags);
        *p_out = reinterpret_cast<FaissIndexBinary*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_binary_fname(
        const char* fname,
        int io_flags,
        FaissIndexBinary** p_out) {
    try {
        auto out = faiss::read_index_binary(fname, io_flags);
        *p_out = reinterpret_cast<FaissIndexBinary*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_binary_custom(
        FaissIOReader* io_reader,
        int io_flags,
        FaissIndexBinary** p_out) {
    try {
        auto out = faiss::read_index_binary(
                reinterpret_cast<IOReader*>(io_reader), io_flags);
        *p_out = reinterpret_cast<FaissIndexBinary*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_VectorTransform_fname(
        const char* fname,
        FaissVectorTransform** p_out) {
    try {
        auto out = faiss::read_VectorTransform(fname);
        *p_out = reinterpret_cast<FaissVectorTransform*>(out);
    }
    CATCH_AND_HANDLE
}
