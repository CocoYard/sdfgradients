/* scipy_openblas32 mangles all BLAS symbols with a `scipy_` prefix to
 * avoid ABI collisions with system BLAS. Eigen (when EIGEN_USE_BLAS is
 * defined) calls the plain Fortran names. This shim forwards each plain
 * name to the prefixed implementation. Add more entries if Eigen pulls
 * in additional BLAS routines (check `nm libsdf_core.a | grep ' U '`).
 */

#define FORWARD(name, ...)                                                \
    extern void scipy_##name(__VA_ARGS__);                                \
    void name(__VA_ARGS__);

#define IMPL_FORWARD(name, params, args)                                  \
    extern void scipy_##name params;                                      \
    void name params { scipy_##name args; }

/* All Eigen BLAS calls use the F77 ABI: pointer args + char* trans flags. */

extern void scipy_dgemm_(const char* transa, const char* transb,
                         const int* m, const int* n, const int* k,
                         const double* alpha,
                         const double* a, const int* lda,
                         const double* b, const int* ldb,
                         const double* beta,
                         double* c, const int* ldc);
void dgemm_(const char* transa, const char* transb,
            const int* m, const int* n, const int* k,
            const double* alpha,
            const double* a, const int* lda,
            const double* b, const int* ldb,
            const double* beta,
            double* c, const int* ldc) {
    scipy_dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

extern void scipy_dgemv_(const char* trans,
                         const int* m, const int* n,
                         const double* alpha,
                         const double* a, const int* lda,
                         const double* x, const int* incx,
                         const double* beta,
                         double* y, const int* incy);
void dgemv_(const char* trans,
            const int* m, const int* n,
            const double* alpha,
            const double* a, const int* lda,
            const double* x, const int* incx,
            const double* beta,
            double* y, const int* incy) {
    scipy_dgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

extern void scipy_dtrsm_(const char* side, const char* uplo,
                         const char* transa, const char* diag,
                         const int* m, const int* n,
                         const double* alpha,
                         const double* a, const int* lda,
                         double* b, const int* ldb);
void dtrsm_(const char* side, const char* uplo,
            const char* transa, const char* diag,
            const int* m, const int* n,
            const double* alpha,
            const double* a, const int* lda,
            double* b, const int* ldb) {
    scipy_dtrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}
