#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>

int main(int argc, char *argv[]){
    
    FILE *fbin = fopen("F-HF.bin", "rb");
    // INT32 = n
    // DOUBLE[] = the matrix (linearised)
    int n; fread(&n, sizeof(int), 1, fbin);
    
    double *h_F = (double*)malloc(sizeof(double) * n * n);
    fread(h_F, sizeof(double), n*n, fbin);
    fclose(fbin);
    
    printf("F read; n= %i \n ", n);

    FILE *sbin = fopen("S.bin", "rb");
    // INT32 = n
    // DOUBLE[] = the matrix (linearised)
    int m; fread(&m, sizeof(int), 1, sbin);
    assert(n == m);
    
    double *h_S = (double*)malloc(sizeof(double) * n * n);
    fread(h_S, sizeof(double), n*n, sbin);
    fclose(sbin);
    
    
    printf("S read; n==m  \n");
    double *h_C = (double*)malloc(sizeof(double) * n * n);
    double *h_e = (double*)malloc(sizeof(double) * n);

    printf("All pointers on run allocated, Starting to allocate on the GPU \n");
    
    
    
    cudaError_t cu_error; // otherwise we cannot monitor the error.
    double *d_F; 
    cu_error = cudaMalloc ((void**)&d_F, sizeof(double) * n * n);
    double *d_S; 
    cu_error = cudaMalloc ((void**)&d_S, sizeof(double) * n * n);
    //double *d_C; // no need for the actual d_C matrix, stored in d_F automatically
    //cu_error = cudaMalloc ((void**)&d_C, sizeof(double) * n * n);
    
    double *d_e; 
    cu_error = cudaMalloc ((void**)&d_e, sizeof(double) * n );

    assert(cu_error == cudaSuccess); // allocating all the matrixes on the GPU
    
    printf("something on gpu allocated \n");
    
    
    //cudaMemcpy(void* dst, void* src, int size, flag direction); //other directions: cudaMemcpyDeviceToHost ; cudaMemcpyDeviceToDevice
    cudaMemcpy(d_F, h_F, sizeof(double) * n*n, cudaMemcpyHostToDevice );
    cudaMemcpy(d_S, h_S, sizeof(double) * n*n, cudaMemcpyHostToDevice );
    
    printf("something coppied to gpu; now preparing the amount of memory on GPU \n");
    
    cusolverDnHandle_t cus_handle = NULL;
    cusolverStatus_t cus_status = CUSOLVER_STATUS_SUCCESS;
    cus_status = cusolverDnCreate(&cus_handle);
    assert(CUSOLVER_STATUS_SUCCESS == cus_status);
    

    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
    
    int *d_info; cudaMalloc ((void**)&d_info, sizeof(int));
    
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;  
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    
    //computes optimal workspace size
    double *d_work;
    int lwork;
    //cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double *A, int lda, const double *W, int *lwork);
    //cusolverDnDsyevd_bufferSize(cus_handle, jobz, uplo, n, d_A, n, d_e, &lwork);
    //cusolverDnDsygvd_bufferSize( cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
    //                             const double *A, int lda, const double *B, int ldb, const double *W, int *lwork);
    cusolverDnDsygvd_bufferSize(cus_handle, itype, jobz, uplo, n, d_F, n, d_S, n, d_e , &lwork);
    
    cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    
    printf("buffer prepared \n");
    
    //call the solver
    
    //cusolverDnDsyevd(cus_handle, jobz, uplo, n, d_A, n, d_e, d_work, lwork, d_info);
    //check: https://docs.nvidia.com/cuda/archive/9.1/pdf/CUSOLVER_Library.pdf 
    cusolverDnDsygvd(cus_handle, itype, jobz, uplo, n, d_F, n, d_S, n,  d_e, d_work, lwork, d_info);
    assert(CUSOLVER_STATUS_SUCCESS == cus_status);
    // d_e stores eigenvalues, the d_A (d_F) should store the C matrix
    
    cudaMemcpy(h_e, d_e, sizeof(double) *n, cudaMemcpyDeviceToHost );
    cudaMemcpy(h_C, d_F, sizeof(double) *n *n, cudaMemcpyDeviceToHost );
    printf("everything copied back \n");

    /*
    // oldstyle writing into console 
    printf("eigenvalues (after an empty line) :\n");
    printf(" \n");
    for(int i=0; i<n; i++){
         printf("ev[%03i] = %8.5lf \n", i, h_e[i]);
    }
    */
    // writing eigen-values into the output.txt file
    FILE *fptr;  
    fptr = fopen("output.txt", "w+");/*  open for writing */  
    if (fptr == NULL){  
        printf("File does not exists \n");  
        return 0;  
        }  
    for(int i=0; i<n; i++){
         fprintf(fptr, "ev[%03i] = %8.5lf \n", i, h_e[i]);  
    }
    fclose(fptr);
    
    printf("output.txt written  \n");

    /*
    // oldstyle writing into console 
    printf("C-matrix(after an empty line) :\n");
    printf(" \n");
    
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            printf("%8.5lf  ", h_C[i,j]);
        }
        printf(" \n");
    }
    */
    // writing "eigen-vectors" into C-HF-out.txt
    FILE *fptr2;  
    fptr2 = fopen("C-HF-out.txt", "w+");/*  open for writing */  
    if (fptr2 == NULL){  
        printf("File does not exists \n");  
        return 0;  
        }  
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            fprintf(fptr2, "%8.5lf  ", h_C[i*n+j]);
        }
        fprintf(fptr2, " \n");
    }
    fclose(fptr2);
    
    printf("C-HF-out.txt written  \n");

    
    FILE *cbin = fopen("C-HF.bin", "rb");
    // INT32 = n
    // DOUBLE[] = the matrix (linearised)
    int o; fread(&o, sizeof(int), 1, cbin);
    assert(n == o);
    
    double *h_C1 = (double*)malloc(sizeof(double) * n * n);
    fread(h_C1, sizeof(double), n*n, cbin);
    fclose(cbin);
    
    FILE *fptr3;  
    fptr3 = fopen("C-HF-cont.txt", "w+");/*  open for writing */  
    if (fptr3 == NULL){  
        printf("File does not exists \n");  
        return 0;  
        }  
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            fprintf(fptr3, "%8.5lf  ", h_C1[i*n+j]);
        }
        fprintf(fptr3, " \n");
    }
    fclose(fptr3);
    
    printf("C-HF-cont.txt written  \n");

    printf(" \n");
    printf(" \n");
    printf(" \n");
    printf(" \n");
    printf("code ended, bye bye \n");
    
    free(h_F); free(h_S); free(h_e); free(h_C); free(h_C1);
    cudaFree(d_F); cudaFree(d_S); cudaFree(d_e); //cudaFree(d_C); 
    cudaFree(d_info);
    cudaFree(d_work);
    return 0;
}
