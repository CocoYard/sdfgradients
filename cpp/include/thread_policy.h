#pragma once
//
// Hardware-aware OpenMP thread tuning.
//
// On big multi-socket boxes:
//   • fit / main_algorithm has many small parallel sections — fork/join
//     cost dominates beyond ~12 threads.
//   • fine predict is latency-bound and keeps scaling to all physical cores.
//
// On laptops / small workstations the difference is noise, so we no-op
// (Mac runs are unaffected because they hit the small-machine branch).
//
// All overrides are scoped — applied via RAII at the call site, not as a
// process-wide side effect. Env vars (SDF_FIT_THREADS, SDF_PREDICT_THREADS,
// OMP_NUM_THREADS) take precedence over auto-tuning.
//

#include <cstdlib>
#include <string>
#include <iostream>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace sdf {

struct ThreadPolicy {
    int fit = 0;       // 0 = leave OpenMP default
    int predict = 0;   // 0 = leave OpenMP default

    ThreadPolicy() {
#ifdef USE_OPENMP
        bool verbose = (std::getenv("SDF_THREADS_VERBOSE") != nullptr);
        const char* env_fit     = std::getenv("SDF_FIT_THREADS");
        const char* env_predict = std::getenv("SDF_PREDICT_THREADS");
        const char* env_omp     = std::getenv("OMP_NUM_THREADS");

        int hw = omp_get_max_threads();

        if (hw > 16 && env_omp == nullptr && env_fit == nullptr) {
            fit = std::min(12, hw);  // sweet spot on multi-socket boxes
        }
        if (env_fit && *env_fit) {
            int n = std::atoi(env_fit);
            if (n > 0) fit = n;
        }
        if (hw > 16 && env_predict == nullptr) {
            predict = hw;
        }
        if (env_predict && *env_predict) {
            int n = std::atoi(env_predict);
            if (n > 0) predict = n;
        }

        if (verbose) {
            std::cerr << "[sdf threads] hw=" << hw
                      << "  fit=" << (fit ? std::to_string(fit) : "default")
                      << "  predict=" << (predict ? std::to_string(predict) : "default")
                      << "\n";
        }
#endif
    }
};

inline const ThreadPolicy& thread_policy() {
    static ThreadPolicy p;
    return p;
}

// Two thin wrappers around omp_set_num_threads. `n <= 0` → no-op (so on
// small machines / Mac these are zero-cost). Call `set_threads(n)` before
// the parallel section, `restore_threads(saved)` after.
inline int set_threads(int n) {
#ifdef USE_OPENMP
    if (n <= 0) return 0;
    int saved = omp_get_max_threads();
    if (n != saved) omp_set_num_threads(n);
    return saved;
#else
    (void)n;
    return 0;
#endif
}
inline void restore_threads(int saved) {
#ifdef USE_OPENMP
    if (saved > 0) omp_set_num_threads(saved);
#else
    (void)saved;
#endif
}

}  // namespace sdf
