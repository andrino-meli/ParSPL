#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#define N_CCS 8
#define VERBOSE
#define PRINTF

#define INF (1.0/0.0)
#define NEGINF (-1.0/0.0)
#define NAN (0.0/0.0)

extern pthread_barrier_t barr_all;

// ----- Runtime substitutes -----
inline void __rt_barrier();

inline void __rt_fpu_fence_full();

inline uint32_t __rt_get_timer();

// sum of the above three for convenience
#define __RT_SEPERATOR {\
    __rt_fpu_fence_full();\
    __rt_barrier();\
    __rt_get_timer();\
}

// ----- Runtime management -----
int smain(uint32_t coreid);

static inline int is_inf(double x) {
    return x == INF;
}

static inline int is_neginf(double x) {
    return x == NEGINF;
}

static inline int is_nan(double x) {
    return x != x;
}

static inline int is_normal(double x) {
    return !(is_inf(x) || is_neginf(x) || is_nan(x));
}

static inline int is_normal_vec(double* x, int len){
    int isn = 1;
    for(int i = 0; i < len; i++){
        if(!is_normal(x[i])){
            isn = 0;
        }
    }
    return isn;
}
