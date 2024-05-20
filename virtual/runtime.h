#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#define N_CCS 8
#define VERBOSE
#define PRINTF

extern pthread_barrier_t barr_all;

// ----- Runtime substitutes -----
void __rt_barrier();

void __rt_fpu_fence_full();

uint32_t __rt_get_timer();

// sum of the above three for convenience
void __rt_seperator();

// ----- Runtime management -----
int smain(uint32_t coreid, uint32_t num_cores);
