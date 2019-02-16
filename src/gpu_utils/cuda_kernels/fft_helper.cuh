#ifndef FFT_HELPER_CUH_
#define FFT_HELPER_CUH_

#include <cuda_runtime.h>
#include "src/gpu_utils/cuda_settings.h"
#include "src/gpu_utils/cuda_mem_utils.h"
#define TILE_DIM 32
#define BLOCK_ROWS 8
//----------------------------------------------------------------------------
__global__ void ScaleReal_kernel(RFLOAT * a, int size, RFLOAT divScale);
__global__ void ScaleComplexPointwise_kernel(__COMPLEX_T * a, int size, int divScale);
//----------------------------------------------------------------------------
__global__ void fft_cuda_kernel_centerFFT_2D(RFLOAT *img_in,
    int image_size,
    int xdim,
    int ydim,
    int xshift,
    int yshift);

__global__ void fft_cuda_kernel_centerFFT_3D(RFLOAT *img_in,
    int image_size,
    int xdim,
    int ydim,
    int zdim,
    int xshift,
    int yshift,
    int zshift);
//----------------------------------------------------------------------------
//======================================================
void fft_runCenterFFT( CudaGlobalPtr< RFLOAT , 0 > &img_in,
				  int xSize,
				  int ySize,
				  int zSize,
				  bool forward,
				  int batchSize = 1);
//======================================================

void windowFourierTransform_gpu(
    CudaGlobalPtr<__COMPLEX_T, 0 > &d_in,
    CudaGlobalPtr<__COMPLEX_T, 0 > &d_out,
    size_t iX, size_t iY, size_t iZ, //Input dimensions
    size_t oX, size_t oY, size_t oZ  //Output dimensions
    );

void window_gpu(CudaGlobalPtr<RFLOAT,0> &self, RFLOAT normfft, long int sy0 , long int sx0, long int syF, long int sxF,
    long int y0 , long int x0, long int yF, long int xF);

void window_gpu(CudaGlobalPtr<RFLOAT,0> &self, RFLOAT normfft,long int sz0, long int sy0, long int sx0, long int szF, long int syF, long int sxF,
    long int z0, long int y0 , long int x0, long int zF, long int yF, long int xF);

void fft_runSoftMaskOutsideMap( CudaGlobalPtr<RFLOAT, 0> &vol_out, 
                                long int xdim,
                                long int ydim,
                                long int zdim,
                                long int xinit,
                                long int yinit,
                                long int zinit,
                                bool do_Mnoise = true,
                                RFLOAT radius = -1,
                                RFLOAT cosine_width = 3);

__global__ void griddingCorrect_kernel_1(RFLOAT* vol_in, int xdim, int ydim, int zdim, int xinit, int yinit, int zinit, int ori_size_padding_factor);
__global__ void griddingCorrect_kernel_2(RFLOAT* vol_in, int xdim, int ydim, int zdim, int xinit, int yinit, int zinit, int ori_size_padding_factor);
#endif