#include "src/gpu_utils/cuda_kernels/recons.cuh"
#include "src/macros.h"

__global__ void multFTBlobKernel_noMask( 
        RFLOAT* real, 
        int XYZSIZE, int XYSIZE, 
        int XSIZE, int YSIZE,
        int padhdim, int pad_size,
        int padding_factor,
        int orixpad,
        RFLOAT  normftblob, RFLOAT sampling,
        RFLOAT* tabulatedValues , int tabftblob_xsize) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int id = tid+bid*BLOCK_SIZE;

    if(id>=XYZSIZE) return ;

    const unsigned int k = id / XYSIZE;
    const unsigned int i =(id / XSIZE)%YSIZE;
    const unsigned int j = id % XSIZE;

	const int kp = (k < padhdim) ? k : k - pad_size;
	const int ip = (i < padhdim) ? i : i - pad_size;
	const int jp = (j < padhdim) ? j : j - pad_size;
    const RFLOAT rval = sqrt ( (RFLOAT)(kp * kp + ip * ip + jp * jp) ) / orixpad;

    int idx = (int)( ABS(rval) / sampling);
    if (idx >= tabftblob_xsize)
        real[id] = 0.;
    else
        real[id]*= (tabulatedValues[idx] / normftblob);

}

__global__ void multFTBlobKernel_withMask( 
        RFLOAT* real, 
        int XYZSIZE, int XYSIZE, 
        int XSIZE, int YSIZE,
        int padhdim, int pad_size,
        int padding_factor,
        int orixpad,
        RFLOAT  normftblob, RFLOAT sampling,
        RFLOAT* tabulatedValues , int tabftblob_xsize) {
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int id = tid+bid*BLOCK_SIZE;

    if(id>=XYZSIZE) return ;

    const unsigned int k = id / XYSIZE;
    const unsigned int i =(id / XSIZE)%YSIZE;
    const unsigned int j = id % XSIZE;

	const int kp = (k < padhdim) ? k : k - pad_size;
	const int ip = (i < padhdim) ? i : i - pad_size;
	const int jp = (j < padhdim) ? j : j - pad_size;
    const RFLOAT rval = sqrt ( (RFLOAT)(kp * kp + ip * ip + jp * jp) ) / orixpad;
    if (rval > 1. / (2. * padding_factor)) {
		real[id] = 0.;
	}
    else {
        int idx = (int)( ABS(rval) / sampling);
        if (idx >= tabftblob_xsize)
            real[id] = 0.;
        else
            real[id]*= (tabulatedValues[idx] / normftblob);
    }
}

__global__ void initFconvKernel(__COMPLEX_T *Fconv,double* Fnewweight,RFLOAT* Fweight,int size) {
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int id = tid+bid*BLOCK_SIZE;

	if(id>=size) return ;
	Fconv[id].x = (RFLOAT)(Fnewweight[id] * (double)Fweight[id]);
	Fconv[id].y = (RFLOAT)0.0;
}

__global__ void divFconvKernel(__COMPLEX_T * __restrict__ Fconv, double* Fnewweight, int max_r2, int XYZSIZE, int XYSIZE, int XSIZE, int YSIZE, int ZSIZE) {
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int id = tid+bid*BLOCK_SIZE;

    if(id>=XYZSIZE) return ;

    const unsigned int k = id / XYSIZE;
    const unsigned int i =(id / XSIZE)%YSIZE;
    const unsigned int j = id % XSIZE;

	const int kp = (k < XSIZE) ? k : k - ZSIZE;
	const int ip = (i < XSIZE) ? i : i - YSIZE;
    const int jp = j;
    
    if (kp * kp + ip * ip + jp * jp < max_r2) {
        const __COMPLEX_T t = Fconv[id];
        RFLOAT w = XMIPP_MAX(1e-6, sqrt(t.x*t.x+t.y*t.y));
        Fnewweight[id] /= w;
    }

}

__global__ void decenter_kernel(const RFLOAT* __restrict__  weight_D, RFLOAT* Fweight_D, int max_r2,
                                int xdim, int ydim, int zdim, int xdim_weight, int ydim_weight,
                                int start_x, int start_y, int start_z)
{
	int global_index = threadIdx.x + blockIdx.x * BLOCK_SIZE;

	int i, j, k;
	int ip, jp, kp;
	j = global_index % xdim;
	i = (global_index / xdim) % ydim;
	k =  global_index / (xdim * ydim);

	jp = j;
	ip = (i < xdim) ? i : (i - ydim);
	kp = (k < xdim) ? k : (k - zdim);
	int ires = (kp * kp + ip * ip + jp * jp);
	if (global_index >= (xdim * ydim * zdim) || ires > max_r2)
	{
		return;
	}

	Fweight_D[global_index] = weight_D[(kp - start_z) * xdim_weight * ydim_weight + (ip - start_y) * xdim_weight + jp - start_x];

}

__global__ void decenter_kernel(const __COMPLEX_T * __restrict__ data_D, __COMPLEX_T * Fconv_D, const double* __restrict__ Fnewweight_D, int max_r2,
                                int xdim, int ydim, int zdim, int xdim_weight, int ydim_weight,
                                int start_x, int start_y, int start_z)
{
	int global_index = threadIdx.x + blockIdx.x * BLOCK_SIZE;

	int i, j, k;
	int ip, jp, kp;
	j = global_index % xdim;
	i = (global_index / xdim) % ydim;
	k =  global_index / (xdim * ydim);

	jp = j;
	ip = (i < xdim) ? i : (i - ydim);
	kp = (k < xdim) ? k : (k - zdim);
	int ires = (kp * kp + ip * ip + jp * jp);
	if (global_index >= (xdim * ydim * zdim)) return;
	if (ires > max_r2)
	{
		Fconv_D[global_index].x = 0.0;
		Fconv_D[global_index].y = 0.0;
		return;
	}
	double Fnewweight = Fnewweight_D[global_index];
#ifdef  RELION_SINGLE_PRECISION
    // Prevent numerical instabilities in single-precision reconstruction with very unevenly sampled orientations
    if (Fnewweight > 1e20)
        Fnewweight = 1e20;
#endif
	Fconv_D[global_index].x = data_D[(kp - start_z) * xdim_weight * ydim_weight + (ip - start_y) * xdim_weight + jp - start_x].x * Fnewweight;
	Fconv_D[global_index].y = data_D[(kp - start_z) * xdim_weight * ydim_weight + (ip - start_y) * xdim_weight + jp - start_x].y * Fnewweight;

}