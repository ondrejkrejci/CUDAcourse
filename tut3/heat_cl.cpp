#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>


#define BLOCK 32   // size of the thread block: 16x16 = 256 threads
#define NB  8      // size of matrix = BLOCK * NB


__global__ void gpu_heat_simple_all(float *A, float *B, int ii, int ij, float c, float tt) {
  //A is input; B is output;
  int n = BLOCK*NB+2;
  int i = (threadIdx.x + blockDim.x * blockIdx.x) + 1; //+1 because of the boundary
  int j = (threadIdx.y + blockDim.y * blockIdx.y) + 1;
  
   B[i*n+j] = A[i*n+j] + (A[(i-1)*n+j] + A[(i+1)*n+j] + A[i*n+(j-1)] + A[i*n+(j+1)] - 4*A[i*n+j])*c;  // a, ih, dt, everything inside the C matrix; // time development in the same step
  if (i==ii && j==ij){
    B[ii*n+ij] = tt ; // setting the heating point, to be the same;
  }
}

__global__ void gpu_heat_shared(float *A, float *B , int ii, int ij, float c, float tt) {
    // A - input only ; B - output only;
  
  int n = BLOCK*NB + 2;
  int i = (threadIdx.x + blockDim.x * blockIdx.x ) + 1;
  int j = (threadIdx.y + blockDim.y * blockIdx.y ) + 1;
  
  __shared__ float patchA[(BLOCK+2)*(BLOCK+2)]; // creating in a shared memory
  
  patchA[(threadIdx.x+1)*(BLOCK+2) + (threadIdx.y+1)] = A[i*n+j]; // here only inner block
  __syncthreads(); //wait for all threads to load
  // sync afterwards the outer block // !!! we do not need the corners !!! //
  if(threadIdx.x == 0){
    patchA[ (threadIdx.y+1)] = A[(i-1)*n+j]; // lowest line
  } 
  if(threadIdx.x == 1){
    patchA[(threadIdx.y+1)*(BLOCK+2) ] = A[(threadIdx.y + blockDim.x * blockIdx.x +1 )*n + (blockDim.y * blockIdx.y)]; // here only the left line
  }
  if(threadIdx.x == BLOCK-2){
    patchA[(threadIdx.y+2)*(BLOCK+2)-1 ] = A[(threadIdx.y + blockDim.x * blockIdx.x +1 )*n + (blockDim.y * (blockIdx.y+1)) + 1]; // here only the right line
  }
  if(threadIdx.x == BLOCK-1){
    patchA[(threadIdx.x+2)*(BLOCK+2) + (threadIdx.y+1)] = A[(i+1)*n+j]; // highest line
  }
  __syncthreads(); //wait for all threads to load

   B[i*n+j] = patchA[(threadIdx.x+1)*(BLOCK+2) + (threadIdx.y+1)] +
              (patchA[(threadIdx.x)*(BLOCK+2) + (threadIdx.y+1)] + patchA[(threadIdx.x+2)*(BLOCK+2) + (threadIdx.y+1)] + patchA[(threadIdx.x+1)*(BLOCK+2) + (threadIdx.y)] +
               patchA[(threadIdx.x+1)*(BLOCK+2) + (threadIdx.y+2)] - 4*patchA[(threadIdx.x+1)*(BLOCK+2) + (threadIdx.y+1)] )*c;  // a, ih, dt, everything inside the C matrix; //time development in the same step
  if (i==ii && j==ij){
    B[ii*n+ij] = tt ; // setting the heating point, to be the same;
  }

}

float dimax(float x[], float y[],int k)
{
  float ta,ti,tmp0;
  ta=0.;ti=0.;tmp0=0.;
  for(int i=1;i<k;i++){
    tmp0=x[i]-y[i];
    if(tmp0>ta){
     ta=tmp0;
    }else{if(tmp0<ti){
    ti=tmp0;}
    }
  }
  if (ti * -1 > ta){ta=ti;}
  
  return(ta);
}

int main(void) {
  
  const int    n = NB * BLOCK + 2; // you do not want to change the borders 
  const float T0 = 100.0;          // original & boarder temperature
  const int   ni = 33; // 15;
  const int   nj = 31; // 16;
  const float  a = 1.;             // heat spreading constant
  const float ih = 1.;             // inverted spacing
  const float dt = 0.05;           // time step
  float       T1 = 500;
  const int tmax = 1000000;              // number of time steps
  int        its = (2*(n-2*ni) + 2*(n-2*nj)-1); // time steps allong the rectangular path
  int       itmp = 0;
  
  // create matrixes A, B 
  float *hA = (float*)malloc(sizeof(float) * n * n); // Temperature
  float *hA0 = (float*)malloc(sizeof(float) * n * n); // Temperature - for storing
  float *hA1 = (float*)malloc(sizeof(float) * n * n); // Temperature - for storing
  float *hB = (float*)malloc(sizeof(float) * n * n); // dT/dt (for the CPU run), later for the Temperature storind
  //float *hC = (float*)malloc(sizeof(float) * n * n); // for the A -> B & B -> A transfer;
  int *hT = (int*)malloc(sizeof(int) * its*2 ); // for the dT movement
  
  // set B & C to 0 // C necessary only for the A -> B & B -> A transfer;
  memset(hB, 0, sizeof(float) * n * n);
  //memset(hC, 0, sizeof(float) * n * n);

  // set matrix A //  
  for(int i=0; i<n*n; i++) {
      hA[i] = T0;
  }

  // set matrix T //  give it moving path
  hT[0] = ni; hT[1]=nj;
  for(int i=1; i<(2*(n-2*ni) + 2*(n-2*nj))/2; i++) {
    if( hT[2*(i-1)]+1<=n-ni){
      hT[2*i] = hT[2*(i-1)]+1; hT[2*i+1]=nj;
    } else {//if(hT[2*(i-1)+1]+1<=n-nj) {
      hT[2*i] = n-ni; hT[2*i+1]=hT[2*(i-1)+1]+1;
    }
  }
  for(int i=(2*(n-2*ni) + 2*(n-2*nj))/2; i<its; i++) {
    if(hT[2*(i-1)]-1>=ni) {
      hT[2*i] = hT[2*(i-1)]-1; hT[2*i+1]=n-nj;
    } else {
      hT[2*i] = ni; hT[2*i+1]=hT[2*(i-1)+1]-1;
    }
  }


  
  // *** DO THE heat spreading in time ***

  hA[ni*n+nj]=T1; // setting the heating point
  
  printf("CPU heat spreading on a matrix %i x %i, where the 1st and last voxel are fixed boundaries\n",n,n);

  struct timeval t1, t2; // for accurate timing
  double elapsed;

  FILE *fptr;  
  fptr = fopen("t1_cpu.txt", "w+");/*  open for writing */  
  if (fptr == NULL){  
    printf("File does not exists \n");  
    return 0;  
  }  
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      fprintf(fptr, " %8.5lf", hA[i*n+j]);  
    }
    fprintf(fptr, "\n");  
  }
  fclose(fptr);
                                                             
  printf("t1_cpu.txt written  \n");

  for(int i=0; i<n*n; i++){ //storing the A matrix for comparison
  hA0[i]=hA[i];
  }

  gettimeofday(&t1, 0); // start time
  
  // do the CPU heat preading the simple CPU way
  for(int t=0; t<tmax; t++){ // t is a discrete time ; according to pre-tests 30 000 steps are enough

    for(int i=1; i<n-1; i++) {
      for(int j=1; j<n-1; j++) { // first we prepare the dT matrix
        hB[i*n+j] = a * (hA[(i-1)*n+j] + hA[(i+1)*n+j] + hA[i*n+(j-1)] + hA[i*n+(j+1)] - 4*hA[i*n+j])*ih*ih ; // Laplacian should be diveded by h**2 = 1 ; https://en.wikipedia.org/wiki/Discrete_Laplace_operator
      }
    }
    for(int i=1; i<n-1; i++) {
      for(int j=1; j<n-1; j++) { // we apply the time step
        hA[i*n+j] += dt * hB[i*n+j] ;
      }
    }
    itmp = (t % its)*2;
    hA[hT[itmp]*n+hT[itmp+1]]=T1; // setting the heating 
  }
  
  gettimeofday(&t2, 0); // finishing time
  // compute elapsed time in ms
  elapsed = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
  printf("CPU time: %lf ms \n", elapsed);

  FILE *fptr2;  
  fptr2 = fopen("tfinal_cpu.txt", "w+");/*  open for writing */  
  if (fptr2 == NULL){  
    printf("File does not exists \n");  
    return 0;  
  }  
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      fprintf(fptr2, " %8.5lf", hA[i*n+j]);  
    }
    fprintf(fptr2, "\n");  
  }
  fclose(fptr2);
                                                             
  printf("tfinal_cpu.txt written  \n");
  
  for(int i=0; i<n*n; i++){ //storing the A matrix for comparison & restoring the original matrix.
  hA1[i]=hA[i];hA[i]=hA0[i];
  }

  // now do the GPU one ...
  float *dA, *dB; //, *dC;
  cudaMalloc(&dA, sizeof(float) * n*n);
  cudaMalloc(&dB, sizeof(float) * n*n);
  //cudaMalloc(&dC, sizeof(float) * n*n);

  cudaMemcpy(dA, hA0, sizeof(float)*n*n, cudaMemcpyHostToDevice); // the original one
  cudaMemcpy(dB, hA0, sizeof(float)*n*n, cudaMemcpyHostToDevice); // the same as A0 (we need the same boundaries)
  //cudaMemcpy(dC, hC, sizeof(float)*n*n, cudaMemcpyHostToDevice);
  
  dim3 block(BLOCK, BLOCK, 1);
  dim3 grid(NB, NB, 1);
  
  gettimeofday(&t1, 0);

  for(int t=0; t<tmax; t++){
    itmp = (t % its)*2;  
    gpu_heat_simple_all<<<grid, block>>>(dA, dB, hT[itmp], hT[itmp+1], dt*a*ih*ih, T1);
    cudaDeviceSynchronize();
    dA = dB; // no necessity for C in this problem, since B is rewritten and the boundaries are supposed to be the same
    //dC = dA;    dA = dB;    dB = dA; 
  }
  
   gettimeofday(&t2, 0);

  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("ERROR! %i -- %s \n", error, cudaGetErrorString(error));
    return -1;
  }
  
  elapsed = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
  printf("GPU simple time: %lf ms \n", elapsed);
  
  cudaMemcpy(hA, dA, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

  printf("difference between CPU 1st and GPU 1run run: %f \n",dimax(hA,hA1,n*n));

  // now do the GPU three shared...

  cudaMemcpy(dA, hA0, sizeof(float)*n*n, cudaMemcpyHostToDevice); // the original one
  cudaMemcpy(dB, hA0, sizeof(float)*n*n, cudaMemcpyHostToDevice); // the B is output - you need to have the borders on the same temperature as anywhere else ;
  
  gettimeofday(&t1, 0);
  
  for(int t=0; t<tmax; t++){
    itmp = (t % its)*2;  
    gpu_heat_shared<<<grid, block>>>(dA, dB, hT[itmp], hT[itmp+1], dt*a*ih*ih, T1);
    cudaDeviceSynchronize();
    dA = dB ; 
    //dC = dA;    dA = dB;    dB = dA; no necessity for C since dB is anyway rewritten and boundaries should be the same
  }

  gettimeofday(&t2, 0);

  error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("ERROR! %i -- %s \n", error, cudaGetErrorString(error));
    return -1;
  }
  
  elapsed = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
  printf("GPU shared time: %lf ms \n", elapsed);

  cudaMemcpy(hA, dA, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

  printf("difference between CPU 1st and GPU 3run run: %f \n",dimax(hA,hA1,n*n));

  FILE *fptr3;  
  fptr3 = fopen("tfinal_gpu.txt", "w+");/*  open for writing */  
  if (fptr3 == NULL){  
    printf("File does not exists \n");  
    return 0;  
  }  
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      fprintf(fptr3, " %8.5lf", hA[i*n+j]);  
    }
    fprintf(fptr3, "\n");  
  }
  fclose(fptr3);
                                                             
  printf("tfinal_gpu.txt written  \n");

  printf("done!");
  return 0;
}
