#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <cstdint>

#include <omp.h>

namespace octtree
{

constexpr int64_t npart_task_threshold = 10000;

template <typename Real>
struct OctreeNode {
  // AABB(Axis-Aligned Bounding Box)
  Real min_x, max_x;
  Real min_y, max_y;
  Real min_z, max_z;

  std::vector<size_t> indices;

  // Oct-tree, so maximum 8 children nodes
  std::array<OctreeNode *, 8> children;
  bool is_leaf;

  OctreeNode() : is_leaf(true) { children.fill(nullptr); }
  ~OctreeNode()
  {
    for(auto *c : children) {
      if(c) delete c;
    }
  }
};

template <typename Real, typename Adaptor>
class Octree
{
public:
  using NodeType = OctreeNode<Real>;

  Octree() : m_leaf_max_size(16), m_adaptor(nullptr), m_root(nullptr) {}

  ~Octree()
  {
    if(m_root) delete m_root;
  }

  void index(Adaptor &adaptor, size_t leaf_max_size = 16)
  {
    m_adaptor = &adaptor;
    m_leaf_max_size = leaf_max_size;

    if(m_root) {
      delete m_root;
      m_root = nullptr;
    }
    size_t n = m_adaptor->octtree_get_point_count();
    if(n == 0) return;

    m_root = new NodeType();
    m_root->indices.resize(n);

#pragma omp parallel for
    for(size_t i = 0; i < n; i++) {
      m_root->indices[i] = i;
    }

    // AABB of root
    calcBoundingBox(m_root);

    // recursive partitioning
#pragma omp parallel
    {
#pragma omp single nowait
      {
        splitNode(m_root);
      }
    }
  }

  void radiusSearch(const Real query[3], Real radius, std::vector<size_t> &results) const
  {
    results.clear();
    if(!m_root) return;
    Real r2 = radius * radius;
    radiusSearchImpl(m_root, query, r2, results);
  }

  size_t knnSearch(const Real query[3], size_t k, std::vector<size_t> &out_indices, std::vector<Real> &out_dists) const
  {
    out_indices.resize(k, size_t(-1));
    out_dists.resize(k, std::numeric_limits<Real>::max());

    // Heap prioritizing (r^2, index) in increasing order
    auto cmp = [](const std::pair<Real, size_t> &a, const std::pair<Real, size_t> &b) { return a.first < b.first; };
    std::priority_queue<std::pair<Real, size_t>, std::vector<std::pair<Real, size_t>>, decltype(cmp)> best_k(cmp);

    knnSearchImpl(m_root, query, k, best_k);

    size_t found = best_k.size();
    for(size_t i = 0; i < found; i++) {
      out_indices[found - 1 - i] = best_k.top().second;
      out_dists[found - 1 - i] = best_k.top().first;
      best_k.pop();
    }
    return found;
  }

private:
  size_t m_leaf_max_size;
  Adaptor *m_adaptor;
  NodeType *m_root;

  void calcBoundingBox(NodeType *node)
  {
    Real minx = std::numeric_limits<Real>::max();
    Real miny = std::numeric_limits<Real>::max();
    Real minz = std::numeric_limits<Real>::max();
    Real maxx = -std::numeric_limits<Real>::max();
    Real maxy = -std::numeric_limits<Real>::max();
    Real maxz = -std::numeric_limits<Real>::max();

#pragma omp parallel for reduction(min : minx, miny, minz) reduction(max : maxx, maxy, maxz)
    for(size_t i = 0; i < node->indices.size(); i++) {
      auto idx = node->indices[i];
      Real x = m_adaptor->octtree_get_pt(idx, 0);
      Real y = m_adaptor->octtree_get_pt(idx, 1);
      Real z = m_adaptor->octtree_get_pt(idx, 2);
      if(x < minx) minx = x;
      if(y < miny) miny = y;
      if(z < minz) minz = z;
      if(x > maxx) maxx = x;
      if(y > maxy) maxy = y;
      if(z > maxz) maxz = z;
    }

    node->min_x = minx;
    node->max_x = maxx;
    node->min_y = miny;
    node->max_y = maxy;
    node->min_z = minz;
    node->max_z = maxz;
  }

  void splitNode(NodeType *node)
  {
    auto nidx = node->indices.size();

    if(nidx <= m_leaf_max_size) {
      node->is_leaf = true;
      return;
    }
    node->is_leaf = false;

    Real midx = 0.5 * (node->min_x + node->max_x);
    Real midy = 0.5 * (node->min_y + node->max_y);
    Real midz = 0.5 * (node->min_z + node->max_z);

    std::array<std::vector<size_t>, 8> child_indices;
    for(int i = 0; i < 8; i++) child_indices[i].reserve(nidx / 8 + 1);

    auto getOctant = [&](Real x, Real y, Real z) {
      int oct = 0;
      if(x > midx) oct |= 1;
      if(y > midy) oct |= 2;
      if(z > midz) oct |= 4;
      return oct;
    };

    for(auto idx : node->indices) {
      Real x = m_adaptor->octtree_get_pt(idx, 0);
      Real y = m_adaptor->octtree_get_pt(idx, 1);
      Real z = m_adaptor->octtree_get_pt(idx, 2);
      int oct = getOctant(x, y, z);
      child_indices[oct].push_back(idx);
    }
    node->indices.clear();

    // generate 8 children
    for(int i = 0; i < 8; i++) {

      if(child_indices[i].empty()) {
        node->children[i] = nullptr;
        continue;
      }

      node->children[i] = new NodeType();

      // AABB
      Real cminx = ((i & 1) ? midx : node->min_x);
      Real cmaxx = ((i & 1) ? node->max_x : midx);
      Real cminy = ((i & 2) ? midy : node->min_y);
      Real cmaxy = ((i & 2) ? node->max_y : midy);
      Real cminz = ((i & 4) ? midz : node->min_z);
      Real cmaxz = ((i & 4) ? node->max_z : midz);

      node->children[i]->min_x = cminx;
      node->children[i]->max_x = cmaxx;
      node->children[i]->min_y = cminy;
      node->children[i]->max_y = cmaxy;
      node->children[i]->min_z = cminz;
      node->children[i]->max_z = cmaxz;

      node->children[i]->indices.swap(child_indices[i]);
    }

    // recursive
    for(int i = 0; i < 8; i++) {
      if(node->children[i]) {

        if(node->children[i]->indices.size() < npart_task_threshold) {
          // If there are few particles, call directly without tasking
          splitNode(node->children[i]);

        } else {
#pragma omp task firstprivate(node, i)
          {
            splitNode(node->children[i]);
          } // omp task
        } // if-else
      }
    }
#pragma omp taskwait
  }

  void radiusSearchImpl(const NodeType *node, const Real q[3], Real r2, std::vector<size_t> &results) const
  {
    if(!node) return;
    if(!overlapAABBSphereSq(node, q, r2)) return;

    if(node->is_leaf) {
      for(auto idx : node->indices) {
        Real dx = q[0] - m_adaptor->octtree_get_pt(idx, 0);
        Real dy = q[1] - m_adaptor->octtree_get_pt(idx, 1);
        Real dz = q[2] - m_adaptor->octtree_get_pt(idx, 2);
        Real _r2 = dx * dx + dy * dy + dz * dz;
        if(_r2 <= r2) {
          results.push_back(idx);
        }
      }
    } else {
      for(int i = 0; i < 8; i++) {
        radiusSearchImpl(node->children[i], q, r2, results);
      }
    }
  }

  template <typename Comparator>
  void knnSearchImpl(
      const NodeType *node, const Real q[3], size_t k,
      std::priority_queue<std::pair<Real, size_t>, std::vector<std::pair<Real, size_t>>, Comparator> &best_k) const
  {
    if(!node) return;

    // After k are filled, skip if the distance between AABB and the point is already greater than the top of best_k
    if(best_k.size() == k) {
      Real bestDist = best_k.top().first;
      Real aabbDist = boxPointSq(node, q);
      if(aabbDist > bestDist) {
        return;
      }
    }

    if(node->is_leaf) {
      for(auto idx : node->indices) {
        Real dx = q[0] - m_adaptor->octtree_get_pt(idx, 0);
        Real dy = q[1] - m_adaptor->octtree_get_pt(idx, 1);
        Real dz = q[2] - m_adaptor->octtree_get_pt(idx, 2);
        Real r2 = dx * dx + dy * dy + dz * dz;
        if(best_k.size() < k) {
          best_k.push({r2, idx});
        } else {
          if(r2 < best_k.top().first) {
            best_k.pop();
            best_k.push({r2, idx});
          }
        }
      }
    } else {
      for(int i = 0; i < 8; i++) {
        knnSearchImpl(node->children[i], q, k, best_k);
      }
    }
  }

  // Collision detection between AABB and sphere based on r^2
  bool overlapAABBSphereSq(const NodeType *node, const Real q[3], Real r2) const { return (boxPointSq(node, q) <= r2); }

  // Nearest distance^2 between AABB and point
  Real boxPointSq(const NodeType *node, const Real q[3]) const
  {
    Real dx = (q[0] < node->min_x) ? (node->min_x - q[0]) : ((q[0] > node->max_x) ? (q[0] - node->max_x) : 0);
    Real dy = (q[1] < node->min_y) ? (node->min_y - q[1]) : ((q[1] > node->max_y) ? (q[1] - node->max_y) : 0);
    Real dz = (q[2] < node->min_z) ? (node->min_z - q[2]) : ((q[2] > node->max_z) ? (q[2] - node->max_z) : 0);
    Real box_r2 = dx * dx + dy * dy + dz * dz;
    return box_r2;
  }
};

}; // namespace octtree
