#define FP32

#ifdef FP32
typedef float FLOAT;
#elif defined FP64
typedef double FLOAT;
#else
#warning no floating point format specified
#endif
