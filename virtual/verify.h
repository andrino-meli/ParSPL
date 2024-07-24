#include "runtime.h"   // for runtime barrier
#include "print_float.h"
#include "types.h"
#include "workspace.h"

#define INF (1.0/0.0)
#define NEGINF (-1.0/0.0)
#define NAN (0.0/0.0)

#define TOL (1e2) // require the error to be less than 1%

static inline int is_inf(FLOAT x) {
    return x == INF;
}

static inline int is_neginf(FLOAT x) {
    return x == NEGINF;
}

#if defined __RT_SSSR_ENABLE
    #ifdef FP32
    static inline int is_nan(float x) {
        // compiler optimizes away so we do it in a volatile
        int ret;
        asm volatile(
        //"_is_nan%=:                \n"
        "feq.s %[r], %[x], %[x]    \n"
        "xori  %[r], %[r], 1       \n"
        : [x]"+f"(x), [r]"=r"(ret) : :
        );
        return ret;
    }
    #elif defined FP64
    static inline int is_nan(float x) {
        // compiler optimizes away so we do it in a volatile
        int ret;
        asm volatile(
        //"_is_nan%=:                \n"
        "feq.s %[r], %[x], %[x]    \n"
        "xori  %[r], %[r], 1       \n"
        : [x]"+f"(x), [r]"=r"(ret) : :
        );
        return ret;
    }
    #endif
#else
static inline int is_nan(FLOAT x) {
    return x != x;
}
#endif

static inline int is_normal(FLOAT x) {
    return !(is_inf(x) || is_neginf(x) || is_nan(x));
}

static inline int is_normal_vec(FLOAT* x, int len){
    int isn = 1;
    for(int i = 0; i < len; i++){
        if(!is_normal(x[i])){
            isn = 0;
        }
    }
    return isn;
}


int verify() {
    asm volatile("_verify: \n":::);
    FLOAT maxerr = 0;
    FLOAT maxrelerr = 0;

    for(int j = 0; j < LINSYS_N; j++){
        // Access x in permutated order to be alligns with the plots.
        #ifdef PERMUTATE
        int i = Perm[j];
        #else
        int i = j;
        #endif
        // absolute error
        double err = (double)x[i] - XGOLD[i];
        double abserr = ( err >= 0 ) ? err : -err;
        if (maxerr < abserr) maxerr = abserr;
        // relative error
        double relerr = err*XGOLD_INV[i];
        double absrel = ( relerr >= 0 ) ? relerr : -relerr;
        #ifdef VERBOSE
        printf("prow %d:\t\tgold %.3e, x %.3e,\terr %.3e,\tabsrel %.3f%%\trow %d\n",
                j, XGOLD[i], x[i], err, absrel*100, i);
        #elif defined PRINTF
        printf("prow %d:\t\tgold ",j);
        printFloat(XGOLD[i]);
        printf(", x  ");
        printFloat(x[i]);
        printf(", err  ");
        printFloat(err);
        printf(", abserr  ");
        printFloat(absrel*100);
        printf("%%\trow %d\n",i);
        #endif
        if (maxrelerr < absrel) {
            maxrelerr = absrel;
            #ifdef VERBOSE
            printf("Updating maxrelerr on prow %d\n",i);
            #endif
        }
    }
    #ifdef PRINTF
        #ifdef LSOLVE
        printf("\nVERIFICATION of lsolve:\n");
        #endif
        #ifdef LTSOLVE
        printf("\nVERIFICATION of ltsolve:\n");
        #endif
        #ifdef SOLVE
        printf("\nVERIFICATION of ldlsolve:\n");
        #endif
    #endif
    #ifdef VERBOSE
    printf(" Maximum abs. rel. error %e\n", maxrelerr);
    int tol = TOL;
    printf(" Thread 0 will return max rel err. TOl = %d\n",tol);
    #endif
    return (int)(maxrelerr*TOL);
}
