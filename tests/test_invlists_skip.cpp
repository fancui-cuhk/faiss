#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>

#include <gtest/gtest.h>
#include <filesystem>
#include <vector>

using namespace faiss;

namespace {

std::string tmp_prefix(const char* tag) {
    auto dir = std::filesystem::temp_directory_path() / "faiss_invlists_skip";
    std::filesystem::create_directories(dir);
    return (dir / tag).string();
}

void fill_grouped(const std::string& prefix) {
    auto* quant = new IndexFlatL2(4);
    auto* index = new IndexIVFFlat(quant, 4, 3, METRIC_L2);
    index->own_fields = true;
    std::vector<float> cents = {
            0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0,
    };
    quant->add(3, cents.data());
    index->is_trained = true;

    float x0[4] = {0, 0, 0, 0};
    idx_t id0 = 7;
    idx_t list0 = 0;
    index->add_core(1, x0, &id0, &list0);

    std::vector<float> big(4 * 4000, 0);
    std::vector<idx_t> ids1(4000), list1(4000, 1);
    for (int i = 0; i < 4000; i++) {
        big[i * 4] = 10;
        big[i * 4 + 1] = float(i) * 0.001f;
        ids1[i] = 1000 + i;
    }
    index->add_core(4000, big.data(), ids1.data(), list1.data());

    std::vector<float> big2 = big;
    std::vector<idx_t> ids2(4000), list2(4000, 2);
    for (int i = 0; i < 4000; i++) {
        big2[i * 4] = 0;
        big2[i * 4 + 1] = 10;
        ids2[i] = 9000 + i;
    }
    index->add_core(4000, big2.data(), ids2.data(), list2.data());

    std::vector<std::vector<size_t>> groups = {{0, 1, 2}};
    write_index_dist_grouped(index, prefix.c_str(), groups);
    delete index;
}

} // namespace

TEST(InvlistsSkip, ReadsOnlyRequestedList) {
    std::string prefix = tmp_prefix("one");
    fill_grouped(prefix);

    Index* raw = read_index_dist(prefix.c_str(), 0);
    auto* ivf = dynamic_cast<IndexIVF*>(raw);
    ASSERT_NE(ivf, nullptr);
    ivf->invlist_seek_gap_bytes = 0;

    float q[4] = {0, 0, 0, 0};
    float dist[1];
    idx_t lab[1];
    idx_t cluster = 0;
    idx_t file = 0;
    float cdis = 0;
    ivf->probe_clusters(1, q, 1, 1, &cluster, &file, &cdis, dist, lab);
    EXPECT_EQ(lab[0], 7);
    EXPECT_LT(dist[0], 1e-5);

    auto st = ivf->last_invlist_io_stats;
    auto sz = std::filesystem::file_size(prefix + "_invlists_0");
    EXPECT_LT(st.payload_bytes + st.table_bytes, sz / 2);
    EXPECT_EQ(st.skip_bytes, 0u);
    EXPECT_GE(st.read_ops, 2u);

    ivf->probe_clusters(1, q, 1, 1, &cluster, &file, &cdis, dist, lab);
    EXPECT_EQ(lab[0], 7);
    auto warm = ivf->last_invlist_io_stats;
    EXPECT_EQ(warm.table_bytes, 0u);
    EXPECT_EQ(warm.payload_bytes, st.payload_bytes);
    EXPECT_EQ(warm.merged_ranges, 1u);
    EXPECT_LT(warm.read_ops, st.read_ops);
    delete ivf;
}

TEST(InvlistsSkip, GapMergeCombinesNearbyLists) {
    std::string prefix = tmp_prefix("gap");
    fill_grouped(prefix);
    Index* raw = read_index_dist(prefix.c_str(), 0);
    auto* ivf = dynamic_cast<IndexIVF*>(raw);
    ASSERT_NE(ivf, nullptr);

    float q[4] = {0, 0, 0, 0};
    float dist[2];
    idx_t lab[2];
    idx_t clusters[2] = {0, 2};
    idx_t files[2] = {0, 0};
    float cdis[2] = {0, 0};

    ivf->invlist_seek_gap_bytes = size_t(1) << 30;
    ivf->probe_clusters(1, q, 2, 2, clusters, files, cdis, dist, lab);
    EXPECT_EQ(ivf->last_invlist_io_stats.merged_ranges, 1u);
    EXPECT_GT(ivf->last_invlist_io_stats.skip_bytes, 0u);

    ivf->invlist_seek_gap_bytes = 0;
    ivf->probe_clusters(1, q, 2, 2, clusters, files, cdis, dist, lab);
    EXPECT_GE(ivf->last_invlist_io_stats.merged_ranges, 2u);
    EXPECT_EQ(ivf->last_invlist_io_stats.table_bytes, 0u);
    EXPECT_EQ(ivf->last_invlist_io_stats.skip_bytes, 0u);
    delete ivf;
}

TEST(InvlistsSkip, SecondFileHasOwnTable) {
    std::string prefix = tmp_prefix("twofiles");
    auto* quant = new IndexFlatL2(4);
    auto* index = new IndexIVFFlat(quant, 4, 3, METRIC_L2);
    index->own_fields = true;
    std::vector<float> cents = {0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0};
    quant->add(3, cents.data());
    index->is_trained = true;
    float x0[4] = {0, 0, 0, 0};
    idx_t id0 = 7, list0 = 0;
    index->add_core(1, x0, &id0, &list0);
    std::vector<float> big(4 * 100, 0);
    std::vector<idx_t> ids1(100, 1000), list1(100, 1);
    for (int i = 0; i < 100; i++) {
        big[i * 4] = 10;
        ids1[i] = 1000 + i;
    }
    index->add_core(100, big.data(), ids1.data(), list1.data());
    std::vector<std::vector<size_t>> groups = {{0}, {1, 2}};
    write_index_dist_grouped(index, prefix.c_str(), groups);
    delete index;

    Index* raw = read_index_dist(prefix.c_str(), 0);
    auto* ivf = dynamic_cast<IndexIVF*>(raw);
    ASSERT_NE(ivf, nullptr);
    float q[4] = {0, 0, 0, 0};
    float dist[1];
    idx_t lab[1];
    idx_t c0 = 0, f0 = 0;
    float cd = 0;
    ivf->probe_clusters(1, q, 1, 1, &c0, &f0, &cd, dist, lab);
    ivf->probe_clusters(1, q, 1, 1, &c0, &f0, &cd, dist, lab);
    EXPECT_EQ(ivf->last_invlist_io_stats.table_bytes, 0u);

    idx_t c1 = 1, f1 = 1;
    float q1[4] = {10, 0, 0, 0};
    ivf->probe_clusters(1, q1, 1, 1, &c1, &f1, &cd, dist, lab);
    EXPECT_GT(ivf->last_invlist_io_stats.table_bytes, 0u);
    delete ivf;
}

TEST(InvlistsSkip, ResetDropsCachedTable) {
    std::string prefix = tmp_prefix("reset");
    fill_grouped(prefix);
    Index* raw = read_index_dist(prefix.c_str(), 0);
    auto* ivf = dynamic_cast<IndexIVF*>(raw);
    ASSERT_NE(ivf, nullptr);
    float q[4] = {0, 0, 0, 0};
    float dist[1];
    idx_t lab[1];
    idx_t cluster = 0, file = 0;
    float cdis = 0;
    ivf->probe_clusters(1, q, 1, 1, &cluster, &file, &cdis, dist, lab);
    ivf->probe_clusters(1, q, 1, 1, &cluster, &file, &cdis, dist, lab);
    EXPECT_EQ(ivf->last_invlist_io_stats.table_bytes, 0u);
    ivf->reset();
    ivf->probe_clusters(1, q, 1, 1, &cluster, &file, &cdis, dist, lab);
    EXPECT_EQ(lab[0], 7);
    EXPECT_GT(ivf->last_invlist_io_stats.table_bytes, 0u);
    delete ivf;
}
