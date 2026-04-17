#include "sphere_bvh.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace sdf {

SphereBVH::SphereBVH(const Eigen::MatrixXd& centers, const Eigen::VectorXd& radii,
                     int leaf_size)
{
    int n = (int)centers.rows();
    cx_.resize(n); cy_.resize(n); cz_.resize(n); ra_.resize(n);
    for (int i = 0; i < n; i++) {
        cx_[i] = (float)centers(i, 0);
        cy_[i] = (float)centers(i, 1);
        cz_[i] = (float)centers(i, 2);
        ra_[i] = std::max((float)std::abs(radii(i)), 1e-10f);
    }
    std::vector<int> idx_buf(n);
    std::iota(idx_buf.begin(), idx_buf.end(), 0);
    nodes_.reserve(2 * n / leaf_size + 16);
    leaves_.reserve(n);
    if (n > 0) build_recursive(idx_buf.data(), 0, n, leaf_size);
}

void SphereBVH::build_recursive(int* idx_buf, int start, int count, int leaf_size) {
    int node_id = (int)nodes_.size();
    nodes_.push_back({});

    float lo0 = 1e30f, lo1 = 1e30f, lo2 = 1e30f;
    float hi0 = -1e30f, hi1 = -1e30f, hi2 = -1e30f;
    for (int k = start; k < start + count; k++) {
        int i = idx_buf[k];
        lo0 = std::min(lo0, cx_[i] - ra_[i]); hi0 = std::max(hi0, cx_[i] + ra_[i]);
        lo1 = std::min(lo1, cy_[i] - ra_[i]); hi1 = std::max(hi1, cy_[i] + ra_[i]);
        lo2 = std::min(lo2, cz_[i] - ra_[i]); hi2 = std::max(hi2, cz_[i] + ra_[i]);
    }
    nodes_[node_id].lo[0] = lo0; nodes_[node_id].lo[1] = lo1; nodes_[node_id].lo[2] = lo2;
    nodes_[node_id].hi[0] = hi0; nodes_[node_id].hi[1] = hi1; nodes_[node_id].hi[2] = hi2;

    if (count <= leaf_size) {
        nodes_[node_id].left = -1;
        nodes_[node_id].right = -1;
        nodes_[node_id].leaf_start = (int)leaves_.size();
        nodes_[node_id].leaf_count = count;
        for (int k = start; k < start + count; k++) leaves_.push_back(idx_buf[k]);
        return;
    }

    float ext0 = hi0 - lo0, ext1 = hi1 - lo1, ext2 = hi2 - lo2;
    int axis = 0;
    if (ext1 > ext0 && ext1 >= ext2) axis = 1;
    else if (ext2 > ext0 && ext2 > ext1) axis = 2;

    const float* axis_ptr = (axis == 0) ? cx_.data() : (axis == 1) ? cy_.data() : cz_.data();
    std::sort(idx_buf + start, idx_buf + start + count, [axis_ptr](int a, int b) {
        return axis_ptr[a] < axis_ptr[b];
    });

    int mid = count / 2;
    nodes_[node_id].leaf_start = -1;
    nodes_[node_id].leaf_count = 0;

    int left_id = (int)nodes_.size();
    build_recursive(idx_buf, start, mid, leaf_size);
    int right_id = (int)nodes_.size();
    build_recursive(idx_buf, start + mid, count - mid, leaf_size);
    nodes_[node_id].left = left_id;
    nodes_[node_id].right = right_id;
}

bool SphereBVH::any_sphere_contains(double qx_d, double qy_d, double qz_d,
                                    double epsilon,
                                    const int* indices, int count) const
{
    float qx = (float)qx_d, qy = (float)qy_d, qz = (float)qz_d;
    float eps = (float)epsilon;
    for (int k = 0; k < count; k++) {
        int j = indices[k];
        float thr = ra_[j] - eps;
        if (thr <= 0.0f) continue;
        float dx = qx - cx_[j];
        float dy = qy - cy_[j];
        float dz = qz - cz_[j];
        if (dx*dx + dy*dy + dz*dz < thr * thr) return true;
    }
    return false;
}

bool SphereBVH::point_inside_any(double qx_d, double qy_d, double qz_d,
                                 double epsilon, int exclude_idx) const
{
    if (nodes_.empty()) return false;
    float qx = (float)qx_d, qy = (float)qy_d, qz = (float)qz_d;
    float eps = (float)epsilon;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        int nid = stack[--sp];
        const Node& nd = nodes_[nid];
        if (qx < nd.lo[0] || qx > nd.hi[0] ||
            qy < nd.lo[1] || qy > nd.hi[1] ||
            qz < nd.lo[2] || qz > nd.hi[2])
            continue;
        if (nd.left == -1) {
            for (int k = nd.leaf_start; k < nd.leaf_start + nd.leaf_count; k++) {
                int j = leaves_[k];
                if (j == exclude_idx) continue;
                float thr = ra_[j] - eps;
                if (thr <= 0.0f) continue;
                float dx = qx - cx_[j];
                float dy = qy - cy_[j];
                float dz = qz - cz_[j];
                if (dx*dx + dy*dy + dz*dz < thr * thr) return true;
            }
        } else {
            stack[sp++] = nd.left;
            stack[sp++] = nd.right;
        }
    }
    return false;
}

}  // namespace sdf
