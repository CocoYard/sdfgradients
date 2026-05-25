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
//                                    per-sphere scratch buffers sized to
//                                    30^3-safe limits (MAX_CAPS=30000, …)
// Scratch buffers live in thread_local std::vector, not on the stack, so
// no OMP_STACKSIZE tuning is required (RSS is ~11 MB per worker thread).
// ════════════════════════════════════════════════════════════════════

// #define FULL_COMPUTE_30CUBED
