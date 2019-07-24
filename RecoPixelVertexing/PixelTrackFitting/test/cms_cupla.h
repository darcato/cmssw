#ifndef CMS_CUPLA_H
#define CMS_CUPLA_H

#ifdef FOR_CUDA
#include <cupla/standalone/GpuCudaRt.hpp>
#else
#include <cupla/standalone/CpuSerial.hpp>
#endif
//#include <cuda_to_cupla.hpp> //already included by one of the above includes

void throw_if_error(cudaError_t success) {
    if (success!=cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(success));
    }
    return;
}

#endif
