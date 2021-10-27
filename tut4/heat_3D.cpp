#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>


#define BLOCK 8   // >=6; size of the thread block: 6**3 =216; 8**3 = 512; 10**3 =1000 threads
#define NB  8      // size of matrix = BLOCK * NB ** 3


__constant__ int   c_ij[3]; //storing the position of the heating element
__constant__ float c_ct[2]; //storing the heating rescale and temperature

__global__ void gpu_heat_n(float *A, float *B) {
  //A is input; B is output; // simple function without shared memory
  int n = BLOCK*NB+2;
  int i = (threadIdx.x + blockDim.x * blockIdx.x) + 1; //+1 because of the boundary
  int j = (threadIdx.y + blockDim.y * blockIdx.y) + 1;
  int k = (threadIdx.z + blockDim.z * blockIdx.z) + 1;
  
   B[i*n*n+j*n+k] = A[i*n*n+j*n+k] + ( A[(i-1)*n*n+j*n+k] + A[(i+1)*n*n+j*n+k] + A[i*n*n+(j-1)*n+k] + A[i*n*n+(j+1)*n+k] +
                                       A[i*n*n+j*n+(k-1)] + A[i*n*n+j*n+(k+1)]  - 6*A[i*n*n+j*n+k] )*c_ct[0];  // a, ih, dt, everything inside the C matrix; // time development in the same step
  if (i==c_ij[0] && j==c_ij[1] && k==c_ij[2]){
    B[c_ij[0]*n*n+c_ij[1]*n+c_ij[2]] = c_ct[1] ; // setting the heating point, to be the same;
  }
}

__global__ void gpu_heat_s(float *A, float *B ) {
  // A - input only ; B - output only; // function with shared memory, wrongly optimized - used for the outer matrixes debugging.
  int n = BLOCK*NB + 2;
  int i = (threadIdx.x + blockDim.x * blockIdx.x ) + 1;
  int j = (threadIdx.y + blockDim.y * blockIdx.y ) + 1;
  int k = (threadIdx.z + blockDim.z * blockIdx.z ) + 1;
  //int s = 5; //for quick debugging of the optimized  sh 
  
  __shared__ float patchA[(BLOCK+2)*(BLOCK+2)*(BLOCK+2)]; // creating a shared memory
  
  patchA[( threadIdx.x+1)*(BLOCK+2)*(BLOCK+2)  + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1) ] = A[ (threadIdx.x + blockDim.x * blockIdx.x +1)*n*n + (threadIdx.y + blockDim.y * blockIdx.y  + 1)*n + (threadIdx.z + blockDim.z * blockIdx.z  + 1) ]; // here only inner block
  __syncthreads(); //wait for all threads to load
  //***********************************************************************************************************************************************************************************************************************************************
  if(threadIdx.x == 0){
      patchA[                                    (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1) ] = A[ (0 + blockDim.x * blockIdx.x )*n*n+(threadIdx.y + blockDim.y * blockIdx.y + 1)*n+(threadIdx.z + blockDim.z * blockIdx.z + 1)]; // bottom table // correct
      //if(s ==0 && threadIdx.y == 5 && threadIdx.z == 5 ){
         //printf("0 i: %i; j: %i; k: %i; wi1dx %i; ridx %i \n",i,j,k, (threadIdx.y+1)*(BLOCK+2)+(threadIdx.z+1),(0 + blockDim.x * blockIdx.x )*n*n+(threadIdx.y + blockDim.y * blockIdx.y + 1)*n +(threadIdx.z + blockDim.z * blockIdx.z + 1));
      //}
  } 
  if(threadIdx.x == 1){
    patchA[(BLOCK+1)*(BLOCK+2)*(BLOCK+2)       + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1) ] = A[ (blockDim.x + blockDim.x * blockIdx.x  +1)*n*n+(threadIdx.y + blockDim.y * blockIdx.y  + 1)*n +(threadIdx.z + blockDim.z * blockIdx.z  + 1)]; // top table // correct
      //if(s ==1 && threadIdx.y == 5 && threadIdx.z == 5 ){
        //printf("0 i: %i; j: %i; k: %i; widx %i; ridx %i \n",i,j,k, (BLOCK+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1) , (blockDim.x + blockDim.x * blockIdx.x  +1)*n*n+(threadIdx.y + blockDim.y * blockIdx.y  + 1)*n +(threadIdx.z + blockDim.z * blockIdx.z  + 1) );
      //}
  }
  if(threadIdx.x == 2){
    patchA[(threadIdx.y+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.z+1)                             ] = A[ (threadIdx.y + blockDim.x * blockIdx.x +1 )*n*n + ( (blockDim.y+1) * blockIdx.y  )*n + threadIdx.z + blockDim.z * blockIdx.z +1 ]; // here only the front block // correct
      //if(s==2 && threadIdx.y == 5 && threadIdx.z == 5 ){
        //printf("0 i: %i; j: %i; k: %i; widx %i; ridx %i \n",i,j,k, (threadIdx.y+1)*(BLOCK+2)*(BLOCK+2)+(threadIdx.z+1),(threadIdx.y + blockDim.x * blockIdx.x +1 )*n*n + ( (blockDim.y+1) * blockIdx.y  )*n + threadIdx.z + blockDim.z * blockIdx.z +1 );
      //}
  }
  if(threadIdx.x == 3){
    patchA[(threadIdx.y+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.z+1) + (BLOCK+2)*(BLOCK+1)       ] = A[ (threadIdx.y + blockDim.x * blockIdx.x +1 )*n*n + ( (blockDim.y+1) * blockIdx.y + blockDim.y )*n + threadIdx.z + blockDim.z * blockIdx.z +1 ]; // here only the back block //correct
      //if(s ==3 && threadIdx.y == 5 && threadIdx.z == 5 ){
        //printf("0 i: %i; j: %i; k: %i; widx %i; ridx %i \n",i,j,k, (threadIdx.y+1)*(BLOCK+2)*(BLOCK+2)+ (threadIdx.z+1) + (BLOCK+2)*(BLOCK+1) , (threadIdx.y + blockDim.x * blockIdx.x +1 )*n*n + ( (blockDim.y+1) * blockIdx.y + blockDim.y )*n + threadIdx.z + blockDim.z * blockIdx.z +1 );
      //}
  }
  if(threadIdx.x == 4){
    patchA[(threadIdx.y+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.z+1)*(BLOCK+2)                   ] = A[ (threadIdx.y + blockDim.x * blockIdx.x +1 )*n*n + (threadIdx.z + blockDim.y * blockIdx.y + 1)*n + blockDim.z * blockIdx.z  ]; // here only the left block // correct
      //if(s == 4 && threadIdx.y == 5 && threadIdx.z == 5 ){
        //printf("0 i: %i; j: %i; k %i; widx %i; ridx %i \n",i,j,k, (threadIdx.y+1)*(BLOCK+2)*(BLOCK+2)+(threadIdx.z+1)*(BLOCK+2),(threadIdx.y + blockDim.x * blockIdx.x +1 )*n*n + (threadIdx.z + blockDim.y * blockIdx.y + 1)*n + blockDim.z * blockIdx.z  );
      //}
  }
  if(threadIdx.x == 5){
    patchA[(threadIdx.y+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.z+1)*(BLOCK+2) + (BLOCK+1)       ] = A[ (threadIdx.y + blockDim.x * blockIdx.x +1 )*n*n + (threadIdx.z + blockDim.y * blockIdx.y + 1)*n + blockDim.z * blockIdx.z + blockDim.z +1 ]; // here only the left block // correct
      //if(s == 5 && threadIdx.y == 5 && threadIdx.z == 5 ){
        //printf("0 i: %i; j: %i; k %i; widx %i; ridx %i \n",i,j,k, (threadIdx.y+1)*(BLOCK+2)*(BLOCK+2)+(threadIdx.z+1)*(BLOCK+2) + (BLOCK+1), (threadIdx.y + blockDim.x * blockIdx.x +1 )*n*n + (threadIdx.z + blockDim.y * blockIdx.y + 1)*n + blockDim.z * blockIdx.z + blockDim.z +1 );
      //}
  }
  __syncthreads(); //wait for all threads to load
   B[i*n*n+j*n+k] =  patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1)] +
                    (patchA[(threadIdx.x  )*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1)] + patchA[(threadIdx.x+2)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1)] +
                     patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y  )*(BLOCK+2) + (threadIdx.z+1)] + patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+2)*(BLOCK+2) + (threadIdx.z+1)] +
                     patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z  )] + patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+2)] -
                   6*patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1)] )*c_ct[0];  // a, ih, dt, everything inside the c_ct[0]; //time development in the same step
  if (i==c_ij[0] && j==c_ij[1] && k==c_ij[2] ){
    B[c_ij[0]*n*n+c_ij[1]*n+c_ij[2]] = c_ct[1] ; // setting the heating point, to be the same;
  }
}

__global__ void gpu_heat_sh(float *A, float *B ) {
    // A - input only ; B - output only;  the "optimized" shared memory version of the heat development function
  int n = BLOCK*NB + 2;
  int i = (threadIdx.x + blockDim.x * blockIdx.x ) + 1;
  int j = (threadIdx.y + blockDim.y * blockIdx.y ) + 1;
  int k = (threadIdx.z + blockDim.z * blockIdx.z ) + 1;

  __shared__ float patchA[(BLOCK+2)*(BLOCK+2)*(BLOCK+2)]; // creating in a shared memory

  patchA[( threadIdx.x+1)*(BLOCK+2)*(BLOCK+2)  + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1) ] = A[ (threadIdx.x + blockDim.x * blockIdx.x +1)*n*n + (threadIdx.y + blockDim.y * blockIdx.y  + 1)*n + (threadIdx.z + blockDim.z * blockIdx.z  + 1) ]; // here only inner block
  __syncthreads(); //wait for all threads to load 

  int ridx, widx; // !!! we do not need the corners !!! //
  widx = threadIdx.z+1 ;
  widx += (threadIdx.x > 3)*(BLOCK+1)*widx ; // (threadIdx.z+1)+(BLOCK+1)*(threadIdx.z+1) = (BLOCK+2)*(threadIdx.z+1)
  widx += (threadIdx.x < 2)*(BLOCK+2)*(threadIdx.y+1) + (threadIdx.x > 1)*(threadIdx.y + 1)*(BLOCK+2)*(BLOCK+2) ; //0 -done; 2 -done; 4 -done;
  widx += (BLOCK+1)*(BLOCK+2)*(BLOCK+2)*(threadIdx.x == 1) + (threadIdx.x == 3)*(BLOCK+1)*(BLOCK+2) + (threadIdx.x == 5)*(BLOCK+1) ; //1 -done; 3 -done; 5-done;
  ridx =  blockDim.x * blockIdx.x; // 0: (0 + blockDim.x * blockIdx.x )*n*n+(threadIdx.y + blockDim.y * blockIdx.y + 1)*n+(threadIdx.z + blockDim.z * blockIdx.z + 1)
  ridx += (threadIdx.x > 1)*(threadIdx.y+1) + (threadIdx.x == 1)*(blockDim.x + 1);
  ridx *= n*n; // first parenthesis done
  ridx += (threadIdx.x < 2)*(threadIdx.y + blockDim.y * blockIdx.y+1)*n + (threadIdx.x > 3)*(threadIdx.z + blockDim.y * blockIdx.y+1)*n + (threadIdx.x  == 2)*((blockDim.y+1)*blockIdx.y)*n + (threadIdx.x  == 3)*((blockDim.y+1)*blockIdx.y + blockDim.y)*n ; // seconde parentheses done
  ridx += blockDim.z*blockIdx.z ; // 4 - done;
  ridx += (threadIdx.x < 4)*(threadIdx.z+1) + (threadIdx.x == 5)*(blockDim.z + 1); //all the others are done

  if(threadIdx.x <= 5){
    //if(threadIdx.x == 5 && threadIdx.y == 5 && threadIdx.z == 5){
      //printf("1 i: %i; j: %i; k: %i; widx %i; ridx %i \n",i,j,k,widx,ridx);
    //}
    patchA[widx] = A[ridx];
  }
  __syncthreads(); //wait for all threads to load
   B[i*n*n+j*n+k] =  patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1)] +
                    (patchA[(threadIdx.x  )*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1)] + patchA[(threadIdx.x+2)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1)] +
                     patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y  )*(BLOCK+2) + (threadIdx.z+1)] + patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+2)*(BLOCK+2) + (threadIdx.z+1)] +
                     patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z  )] + patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+2)] -
                   6*patchA[(threadIdx.x+1)*(BLOCK+2)*(BLOCK+2) + (threadIdx.y+1)*(BLOCK+2) + (threadIdx.z+1)] )*c_ct[0];  // a, ih, dt, everything inside the c_ct[0]; //time development in the same step
  if (i==c_ij[0] && j==c_ij[1] && k==c_ij[2] ){
    B[c_ij[0]*n*n+c_ij[1]*n+c_ij[2]] = c_ct[1] ; // setting the heating point, to be the same;
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
  
  const int       n = NB * BLOCK + 2;    // you do not want to change the borders 
  const float     T0 = 100.0;            // original & boarder temperature
  const int   nij[3] = {10,11,12};       // where the heating element is 8*32 = 256; 
  const float fct[2] = {0.05, 500};      // heat timestep-spreading constant =   heat spreading constant(1)* inverted spacing(1/1)* inverted spacing(1/1) *time step (0.05) ; 500 K
  const int     tmax = 100000 ;              // number of time steps
  
  // create matrixes A, B 
  float *hA  = (float*)malloc(sizeof(float) * n * n * n); // Temperature
  float *hA0 = (float*)malloc(sizeof(float) * n * n * n); // Temperature - for storing
  float *hA1 = (float*)malloc(sizeof(float) * n * n * n); // Temperature - for storing
  float *hB  = (float*)malloc(sizeof(float) * n * n * n); // dT/dt (for the CPU run), later for the Temperature storind
  
  // set B to 0 
  memset(hB, 0, sizeof(float) * n * n * n);

  // set matrix A //  
  for(int i=0; i<n*n*n; i++) {
      hA[i] = T0;
  }

  cudaMemcpyToSymbol(c_ij, &nij, sizeof(int)  *3);
  cudaMemcpyToSymbol(c_ct, &fct, sizeof(float)*2);
  
  // *** DO THE heat spreading in time ***

  hA[nij[0]*n*n+nij[1]*n*nij[2]]=fct[1]; // setting the heating point
  
  printf("GPU heat spreading on a tensor %i x %i x %i, where the 1st and last voxel are fixed boundaries\n",n,n,n);

  struct timeval t1, t2; // for accurate timing
  double elapsed;

  FILE *fptr;  
  fptr = fopen("3D_i_cpu.txt", "w+");/*  open for writing */  
  if (fptr == NULL){  
    printf("File does not exists \n");  
    return 0;  
  }  
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      for(int k=0; k<n; k++){
        fprintf(fptr, " %8.5lf \n", hA[i*n*n+j*n+k]);  
      }    
    }
  }
  fclose(fptr);
                                                             
  printf("3D_i_cpu.txt written  \n");

  for(int i=0; i<n*n*n; i++){ //storing the A matrix for comparison
  hA0[i]=hA[i];
  }
  
  gettimeofday(&t1, 0); // start time
  
  // do the CPU heat preading the simple CPU way
  for(int t=0; t<tmax; t++){ // t is a discrete time ; according to pre-tests 30 000 steps are enough

    for(int i=1; i<n-1; i++) {
      for(int j=1; j<n-1; j++) { 
        for(int k=1; k<n-1; k++) { // first we prepare the dT matrix
          hB[i*n*n+j*n+k] = fct[0] * (hA[(i-1)*n*n+j*n+k] + hA[(i+1)*n*n+j*n+k] + hA[i*n*n+(j-1)*n+k] + hA[i*n*n+(j+1)*n+k] +
                                      hA[i*n*n+j*n+(k-1)] + hA[i*n*n+j*n+(k+1)]  - 6*hA[i*n*n+j*n+k]) ; // Laplacian should be diveded by h**2 = 1 ; https://en.wikipedia.org/wiki/Discrete_Laplace_operator
        }
      }
    }
    for(int i=1; i<n-1; i++) {
      for(int j=1; j<n-1; j++) { // we apply the time step
        for(int k=1; k<n-1; k++) { // we apply the time step
          hA[i*n*n+j*n+k] += hB[i*n*n+j*n+k] ;
        }
      }
    }
    hA[nij[0]*n*n+nij[1]*n+nij[2]]=fct[1]; // setting the heating 
  }
  
  gettimeofday(&t2, 0); // finishing time
  // compute elapsed time in ms
  elapsed = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
  printf("CPU time: %lf ms \n", elapsed);
  
  FILE *fptr2;  
  fptr2 = fopen("3D_f_cpu.txt", "w+");/*  open for writing */  
  if (fptr2 == NULL){  
    printf("File does not exists \n");  
    return 0;  
  }  
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      for(int k=0; k<n; k++){
        fprintf(fptr2, " %8.5lf \n", hA[i*n*n+j*n+k]);  
      }
    }
  }
  fclose(fptr2);
                                                             
  printf("3D_f_cpu.txt written  \n");
  
  for(int i=0; i<n*n*n; i++){ //storing the A matrix for comparison & restoring the original matrix.
  hA1[i]=hA[i];hA[i]=hA0[i];
  }


  // now do the GPU one ...
  float *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(float) * n*n*n);
  cudaMalloc(&dB, sizeof(float) * n*n*n);

  cudaMemcpy(dA, hA0, sizeof(float)*n*n*n, cudaMemcpyHostToDevice); // the original one
  cudaMemcpy(dB, hA0, sizeof(float)*n*n*n, cudaMemcpyHostToDevice); // the same as A0 (we need the same boundaries)
  
  dim3 block(BLOCK, BLOCK, BLOCK);
  dim3 grid(NB, NB, NB);
  
  gettimeofday(&t1, 0);

  for(int t=0; t<tmax; t++){
    gpu_heat_n<<<grid, block>>>(dA, dB);
    cudaDeviceSynchronize();
    dC = dA;    
    dA = dB;    
    dB = dC; 
  }
  
  gettimeofday(&t2, 0);

  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("ERROR! %i -- %s \n", error, cudaGetErrorString(error));
    return -1;
  }
  
  elapsed = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
  printf("GPU simple time: %lf ms \n", elapsed);
  
  cudaMemcpy(hA, dA, sizeof(float)*n*n*n, cudaMemcpyDeviceToHost);


  printf("difference between CPU 1st and GPU 1run run: %f \n",dimax(hA,hA1,n*n*n));

  FILE *fptr3;  
  fptr3 = fopen("3D_f_gpu.txt", "w+");/*  open for writing */  
  if (fptr3 == NULL){  
    printf("File does not exists \n");  
    return 0;  
  }  
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      for(int k=0; k<n; k++){
        fprintf(fptr3, " %8.5lf \n", hA[i*n*n+j*n+k]);  
      }
    }
  }
  fclose(fptr3);
                                                             
  printf("3D_f_gpu.txt written  \n");


  // now do the GPU two shared...

  cudaMemcpy(dA, hA0, sizeof(float)*n*n, cudaMemcpyHostToDevice); // the original one
  cudaMemcpy(dB, hA0, sizeof(float)*n*n, cudaMemcpyHostToDevice); // the B is output - you need to have the borders on the same temperature as anywhere else ;
  
  gettimeofday(&t1, 0);
  
  for(int t=0; t<tmax; t++){
    gpu_heat_s<<<grid, block>>>(dA, dB);
    cudaDeviceSynchronize();
    dC = dA;    
    dA = dB;    
    dB = dC; 
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

  printf("difference between GPU 1st and GPU 2run run: %f \n",dimax(hA,hA1,n*n));

  FILE *fptr4;  
  fptr4 = fopen("3D_f_gpu2.txt", "w+");/*  open for writing */  
  if (fptr4 == NULL){  
    printf("File does not exists \n");  
    return 0;  
  }  
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      for(int k=0; k<n; k++){
        fprintf(fptr4, " %8.5lf \n", hA[i*n*n+j*n+k]);  
      }
    }
  }
  fclose(fptr4);
                                                             
  printf("3D_f_gpu2.txt written  \n");


  // now do the GPU three shared...

  cudaMemcpy(dA, hA0, sizeof(float)*n*n, cudaMemcpyHostToDevice); // the original one
  cudaMemcpy(dB, hA0, sizeof(float)*n*n, cudaMemcpyHostToDevice); // the B is output - you need to have the borders on the same temperature as anywhere else ;
  
  gettimeofday(&t1, 0);
  
  for(int t=0; t<tmax; t++){
    gpu_heat_sh<<<grid, block>>>(dA, dB);
    cudaDeviceSynchronize();
    dC = dA;    
    dA = dB;    
    dB = dC; 
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

  printf("difference between GPU 1st and GPU 3run run: %f \n",dimax(hA,hA1,n*n));

  FILE *fptr5;  
  fptr5 = fopen("3D_f_gpu3.txt", "w+");/*  open for writing */  
  if (fptr5 == NULL){  
    printf("File does not exists \n");  
    return 0;  
  }  
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      for(int k=0; k<n; k++){
        fprintf(fptr5, " %8.5lf \n", hA[i*n*n+j*n+k]);  
      }
    }
  }
  fclose(fptr5);
                                                             
  printf("3D_f_gpu3.txt written  \n");



  printf("\n done! \n \n");
  return 0;
}

