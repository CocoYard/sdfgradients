#pragma once

// ════════════════════════════════════════════════════════════════════
// Toggle for "full / exact computation" mode, sized for 30^3 (27k)
// spheres. Comment out the #define below to revert to the small-
// footprint truncated limits used in normal production runs.
//
// When enabled:
//   - sphere_intersect_core.cpp    : MAX_KEEP = INT_MAX
//                                    (no per-sphere neighbor truncation)
//   - existing_modules_adapter.cpp : MAX_NEIGHBORS_SCANNED = INT_MAX
//                                    per-sphere static arrays bumped to
//                                    30^3-safe sizes (MAX_CAPS=30000, …)
// Requires OMP_STACKSIZE >= 64M (stack usage ≈ 11 MB per worker thread).
// ════════════════════════════════════════════════════════════════════

// #define FULL_COMPUTE_30CUBED
