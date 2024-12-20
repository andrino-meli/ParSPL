#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#define N_CCS 8
#define VERBOSE
#define PRINTF

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
