#ifndef CUDA_RECONS_KERNELS_CUH_
#define CUDA_RECONS_KERNELS_CUH_

#include <cuda_runtime.h>
#include "src/gpu_utils/cuda_settings.h"
#include "src/gpu_utils/cuda_device_utils.cuh"
//-----------------------------------------------------------------------------------------------
__global__ void multFTBlobKernel_noMask( 
        RFLOAT* real, 
        int XYZSIZE, int XYSIZE, 
        int XSIZE, int YSIZE,
        int padhdim, int pad_size,
        int padding_factor,
        int orixpad,
        RFLOAT  normftblob, RFLOAT sampling,
        RFLOAT* tabulatedValues , int tabftblob_xsize);

__global__ void multFTBlobKernel_withMask( 
        RFLOAT* real, 
        int XYZSIZE, int XYSIZE, 
        int XSIZE, int YSIZE,
        int padhdim, int pad_size,
        int padding_factor,
        int orixpad,
        RFLOAT  normftblob, RFLOAT sampling,
        RFLOAT* tabulatedValues , int tabftblob_xsize);
//-----------------------------------------------------------------------------------------------
__global__ void initFconvKernel(__COMPLEX_T *Fconv,double* Fnewweight,RFLOAT* Fweight,int size);
//-----------------------------------------------------------------------------------------------
__global__ void divFconvKernel(__COMPLEX_T * __restrict__ Fconv, double* Fnewweight, int max_r2, int XYZSIZE, int XYSIZE, int XSIZE, int YSIZE, int ZSIZE);

//-----------------------------------------------------------------------------------------------
__global__ void decenter_kernel(const RFLOAT* __restrict__  weight_D, RFLOAT* Fweight_D, int max_r2,
                                int xdim, int ydim, int zdim, int xdim_weight, int ydim_weight,
                                int start_x, int start_y, int start_z);

inline void decenter_gpu(RFLOAT* weight_D, RFLOAT* Fweight_D, int max_r2,
                                 int xdim, int ydim, int zdim, int xdim_weight, int ydim_weight,
                                 int start_x, int start_y, int start_z, cudaStream_t stream)
{
	int model_size = xdim * ydim * zdim;
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim((model_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
	decenter_kernel <<< gridDim, blockDim, 0, stream>>>(weight_D, Fweight_D, max_r2,
	                                         xdim, ydim, zdim, xdim_weight, ydim_weight,
	                                         start_x, start_y, start_z);

}
//-----------------------------------------------------------------------------------------------
__global__ void decenter_kernel(const __COMPLEX_T * __restrict__ data_D, 
                                __COMPLEX_T * Fconv_D, 
                                const double* __restrict__ Fnewweight_D, 
                                int max_r2,
                                int xdim, int ydim, int zdim, 
                                int xdim_weight, int ydim_weight,
                                int start_x, int start_y, int start_z);

inline void decenter_gpu(		 __COMPLEX_T * data_D,
                                 __COMPLEX_T * Fconv_D,
                                 double* Fnewweight_D,
                                 int max_r2,
                                 int xdim,
                                 int ydim,
                                 int zdim,
                                 int xdim_weight,
                                 int ydim_weight,
                                 int start_x,
                                 int start_y,
                                 int start_z,
								 cudaStream_t stream)
{
	int model_size = xdim * ydim * zdim;
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim((model_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
	decenter_kernel <<< gridDim, blockDim, 0, stream>>>(data_D, Fconv_D, Fnewweight_D, max_r2,
	                                         xdim, ydim, zdim, xdim_weight, ydim_weight,
	                                         start_x, start_y, start_z);
}
//-----------------------------------------------------------------------------------------------
#endif