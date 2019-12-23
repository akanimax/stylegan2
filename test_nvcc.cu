#include <cstdio>

void checkCudaError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

__global__ void cudaKernel(void)
{
    printf("GPU says hello.\n");
}

int main(void)
{
    printf("CPU says hello.\n");
    checkCudaError(cudaLaunchKernel((void*)cudaKernel, 1, 1, NULL, 0, NULL));
    checkCudaError(cudaDeviceSynchronize());
    return 0;
}
