#include <cuda_to_cupla.hpp>
//#include <Eigen/Core>

int main(int argc, char *argv[]) {
  
    float *m_gpu;
    cudaMalloc((void **)&m_gpu, sizeof(float));

    return 0;
}