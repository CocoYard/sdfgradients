#pragma once

#include <cstdio>
#include <cstdlib>

// Assertion that fires in every build mode (Debug, Release, RelWithDebInfo,
// MinSizeRel) — unlike <cassert>, which compiles to a no-op when NDEBUG is
// defined. Use for invariants that should be checked even in production
// builds when the cost is negligible and a silent violation would corrupt
// downstream state.
//
// On failure: prints the condition text, the message, and source location to
// stderr, then calls std::abort() (no unwinding, so OpenMP-safe).
#define ALWAYS_ASSERT(cond, msg)                                          \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr,                                          \
                "ALWAYS_ASSERT failed: %s\n"                              \
                "  condition: %s\n"                                       \
                "  at %s:%d in %s\n",                                     \
                (msg), #cond, __FILE__, __LINE__, __func__);              \
            std::abort();                                                 \
        }                                                                 \
    } while (0)
