#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__constant__ int c_N[2];

__global__ void gpu_psum_atomic(float *A, float *P){

  extern __shared__ float buffer[];
  
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  float psum = 0;
  
  buffer[threadIdx.x] = 0;
  if(idx < c_N[0]){
    buffer[threadIdx.x] = A[idx];
  }
  psum = buffer[threadIdx.x];
  __syncthreads();
  
  for(int stride=1; stride <= blockDim.x /2; stride*= 2){ //1st run over 1; second over 2; third over 4 ... ;
    
    int idxnext = threadIdx.x + stride; // problems, if you go outside of boundary
    idxnext *= (idxnext < blockDim.x); // 
        
    psum += buffer[idxnext];
    __syncthreads();
  
    buffer[threadIdx.x] = psum;
    __syncthreads();
  
  }
  
  if (threadIdx.x == 0){
   //P[blockIdx.x] = psum ;
   atomicAdd(P,psum);
   
  }
  
  
  
}


// ./a.out 16 4
int main(int argc, char ** argv){

  int n, b;
  n = atoi(argv[1]);
  b = atoi(argv[2]);
  int nblocks = (int)ceilf((float)n/b); //(16/14)->2 
  
  cudaMemcpyToSymbol(c_N, &n, sizeof(int));

  float *h_A = (float*)malloc(sizeof(float) * n);
  float *h_P = (float*)malloc(sizeof(float) * nblocks); //partial sums
  float *d_A;  cudaMalloc((void**)&d_A, sizeof(float) *n); //input vector
  float *d_P;  cudaMalloc((void**)&d_P, sizeof(float) *nblocks); // output vector
  
  for(int i=0; i<n; i++){
   h_A[i] = i+1;
  }
  //sum_{1,N} = N(N+1)/2
  
  cudaMemcpy(d_A, h_A, sizeof(float)*n, cudaMemcpyHostToDevice);
  
  dim3 grid(nblocks,1,1);
  dim3 block(b,1,1);
  
  gpu_psum_atomic<<<grid, block, sizeof(float)*b >>>(d_A,d_P); //up to 4 arguments in 
  cudaDeviceSynchronize();
  
  cudaMemcpy(h_P, d_P, sizeof(float) *nblocks, cudaMemcpyDeviceToHost);
  //for(int i=0; i<nblocks; i++){
    //int blockstart = i*b + 1;
    //int blockend   =blockstart + b -1;
    //printf("partial[%i] = %f -- %f \n",  h_P[i], (blockend*(blockend+1))/2 - blockstart*(blockstart-1)/2);
  //}
  printf("sum: %f \n",h_P[0]);

  free(h_A);
  cudaFree(d_A); cudaFree(d_P);
  return 0;
}
