#include <pthread.h>
#include "runtime.h"
//#include "string.h"

// ----- Hardware virtualization -----
// Virtual runtime using pthread.
// Used to functionally verify code generation.
pthread_barrier_t barr_all;


//typedef struct {
//    void * dst;
//    void * src;
//    uint32_t len_bytes;
//    uint32_t id;
//} dma_job_t;
//pthread_t dma_thread;
//dma_job_t dma_queue[DMA_QUEUE_LEN];

//uint32_t dma_id_head;       // Exclusively modified by FS
//uint32_t dma_id_tail;       // Exclusively modified by DMA thread

//void* dma_job_thread(void* job) {
//    dma_job_t* dma_job = (dma_job_t*) job;
//    while (dma_id_tail < dma_job->id);                          // Wait for our turn
//    memcpy(dma_job->dst, dma_job->src, dma_job->len_bytes);
//    __atomic_fetch_add(&dma_id_tail, 1, __ATOMIC_RELAXED);      // Signal we are done atomically
//    return NULL;
//}

// ----- Runtime substitutes -----
//typedef uint32_t pulp_timer_t;


//static inline uint32_t dma_memcpy_nonblk(void* dst, void* src, uint32_t len_bytes) {
//    while (dma_id_head - dma_id_tail >= DMA_QUEUE_LEN);                         // Spin until DMA buffer has opening
//    uint32_t id = __atomic_fetch_add(&dma_id_head, 1, __ATOMIC_RELAXED);
//    dma_job_t * job_slot = dma_queue + (id % DMA_QUEUE_LEN);
//    *job_slot = (dma_job_t) {dst, src, len_bytes, id};
//    if (pthread_create(&dma_thread, NULL, dma_job_thread, job_slot)) return -1;
//    return id;
//}

//static inline void dma_wait_on_id(uint32_t id) {while (dma_id_tail <= id);}

typedef struct {
    uint32_t coreid;
    uint32_t num_cores;
    int ret;
} cc_params_t;


void *core_wrap(void *params) {
    cc_params_t* cc_params = params;
    cc_params->ret = smain(cc_params->coreid, cc_params->num_cores);
    return NULL;
}


int main() {
    // Create thread pool
    pthread_barrier_init(&barr_all, NULL, N_CCS);
    pthread_t cc[N_CCS];
    cc_params_t cc_params[N_CCS];
    // Start threads
    for (int i=0; i<N_CCS; ++i) {
        cc_params[i].coreid = i;
        cc_params[i].num_cores = N_CCS;
        if (pthread_create(&cc[i], NULL, core_wrap, &cc_params[i])) return -i;
    }
    // Join threads
    for (int i=0; i<N_CCS; ++i) {
        if (pthread_join(cc[i], NULL)) return -i;
    }
    // print return values:
    for (int i=0; i<N_CCS; ++i) {
        printf("Thread %d retunred %d\n",i,cc_params[i].ret);
    }
    return 0;
}
