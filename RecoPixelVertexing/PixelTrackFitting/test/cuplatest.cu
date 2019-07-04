#include <Eigen/Core>
#include <cupla/standalone/GpuCudaRt.hpp>
//#include <cuda_to_cupla.hpp>

int main(int argc, char *argv[]) {
  
    float *m_gpu;
    cudaMalloc((void **)&m_gpu, sizeof(float));

    return 0;
}