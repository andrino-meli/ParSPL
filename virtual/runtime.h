#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#define N_CCS 8
#define VERBOSE

extern pthread_barrier_t barr_all;

// ----- Runtime substitutes -----
static inline void __rt_barrier() {
    pthread_barrier_wait(&barr_all);
}

static inline void __rt_fpu_fence_full() {
}

static inline void __rt_get_timer() {
}

// ----- Runtime management -----
extern int smain(uint32_t coreid, uint32_t num_cores);
