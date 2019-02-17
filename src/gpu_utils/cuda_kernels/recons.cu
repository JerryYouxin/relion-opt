#include "src/gpu_utils/cuda_kernels/recons.cuh"
#include "src/macros.h"

__global__ void symmetrise_kernel(const __COMPLEX_T * __restrict__ my_data_temp_D ,
                                  const  RFLOAT* __restrict__ my_weight_temp_D,
                                  __COMPLEX_T * my_data_D,
                                  RFLOAT* my_weight_D,
                                  int xdim,
                                  int ydim,
                                  int xydim,
                                  int zdim,
                                  int start_x,
                                  int start_y,
                                  int start_z,
                                  int my_rmax2,
                                  int nr_SymsNo)

{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	int x, y, z;
	x = id % xdim + start_x;
	y = (id / xdim) % ydim + start_y;
	z =  id / xydim + start_z;
	if ((x * x + y * y + z * z) > my_rmax2 || id >= xydim * zdim)
	{
		return;
	}

	RFLOAT  fx, fy, fz, xp, yp, zp;
	bool is_neg_x;
	int x0, x1, y0, y1, z0, z1;
	RFLOAT d000_r, d001_r, d010_r, d011_r, d100_r, d101_r, d110_r, d111_r;
	RFLOAT dx00_r, dx01_r, dx10_r, dx11_r, dxy0_r, dxy1_r;
	RFLOAT d000_i, d001_i, d010_i, d011_i, d100_i, d101_i, d110_i, d111_i;
	RFLOAT dx00_i, dx01_i, dx10_i, dx11_i, dxy0_i, dxy1_i;
	RFLOAT dd000, dd001, dd010, dd011, dd100, dd101, dd110, dd111;
	RFLOAT ddx00, ddx01, ddx10, ddx11, ddxy0, ddxy1;

	RFLOAT real, img, weight;
	weight = real = img = 0.;

	for (int i = 0; i < nr_SymsNo; i++)
	{
		// coords_output(x,y) = A * coords_input (xp,yp)

		xp = (RFLOAT)x * __R_array[i * 4 * 4] + (RFLOAT)y *  __R_array[i * 4 * 4 + 1] + (RFLOAT)z *  __R_array[i * 4 * 4 + 2];
		yp = (RFLOAT)x *  __R_array[i * 4 * 4 + 1 * 4] + (RFLOAT)y *  __R_array[i * 4 * 4 + 1 + 1 * 4] + (RFLOAT)z *  __R_array[i * 4 * 4 + 2 + 1 * 4];
		zp = (RFLOAT)x *  __R_array[i * 4 * 4 + 2 * 4] + (RFLOAT)y *  __R_array[i * 4 * 4 + 1 + 2 * 4] + (RFLOAT)z *  __R_array[i * 4 * 4 + 2 + 2 * 4];
		// Only asymmetric half is stored
		if (xp < 0)
		{
			// Get complex conjugated hermitian symmetry pair
			xp = -xp;
			yp = -yp;
			zp = -zp;
			is_neg_x = true;
		}
		else
		{
			is_neg_x = false;
		}

		// Trilinear interpolation (with physical coords)
		// Subtract STARTINGY and STARTINGZ to accelerate access to data (STARTINGX=0)
		// In that way use DIRECT_A3D_ELEM, rather than A3D_ELEM
		x0 = floor(xp);
		fx = xp - x0;
		x1 = x0 + 1;

		y0 = floor(yp);
		fy = yp - y0;
		y0 -=  start_y;
		y1 = y0 + 1;

		z0 = floor(zp);
		fz = zp - z0;
		z0 -= start_z;
		z1 = z0 + 1;

		// First interpolate (complex) data
		d000_r = my_data_temp_D[z0 * xydim + y0 * xdim + x0].x;
		d001_r = my_data_temp_D[z0 * xydim + y0 * xdim + x1].x;
		d010_r = my_data_temp_D[z0 * xydim + y1 * xdim + x0].x;
		d011_r = my_data_temp_D[z0 * xydim + y1 * xdim + x1].x;

		d000_i = my_data_temp_D[z0 * xydim + y0 * xdim + x0].y;
		d001_i = my_data_temp_D[z0 * xydim + y0 * xdim + x1].y;
		d010_i = my_data_temp_D[z0 * xydim + y1 * xdim + x0].y;
		d011_i = my_data_temp_D[z0 * xydim + y1 * xdim + x1].y;

		d100_r = my_data_temp_D[z1 * xydim + y0 * xdim + x0].x;
		d101_r = my_data_temp_D[z1 * xydim + y0 * xdim + x1].x;
		d110_r = my_data_temp_D[z1 * xydim + y1 * xdim + x0].x;
		d111_r = my_data_temp_D[z1 * xydim + y1 * xdim + x1].x;

		d100_i = my_data_temp_D[z1 * xydim + y0 * xdim + x0].y;
		d101_i = my_data_temp_D[z1 * xydim + y0 * xdim + x1].y;
		d110_i = my_data_temp_D[z1 * xydim + y1 * xdim + x0].y;
		d111_i = my_data_temp_D[z1 * xydim + y1 * xdim + x1].y;

		dx00_r = d000_r + (d001_r - d000_r) * fx;
		dx00_i = d000_i + (d001_i - d000_i) * fx;
		dx01_r = d100_r + (d101_r - d100_r) * fx;
		dx01_i = d100_i + (d101_i - d100_i) * fx;
		dx10_r = d010_r + (d011_r - d010_r) * fx;
		dx10_i = d010_i + (d011_i - d010_i) * fx;
		dx11_r = d110_r + (d111_r - d110_r) * fx;
		dx11_i = d110_i + (d111_i - d110_i) * fx;

		dxy0_r = dx00_r + (dx10_r - dx00_r) * fy;
		dxy0_i = dx00_i + (dx10_i - dx00_i) * fy;
		dxy1_r = dx01_r + (dx11_r - dx01_r) * fy;
		dxy1_i = dx01_i + (dx11_i - dx01_i) * fy;
		if (is_neg_x)
		{
			real += dxy0_r + (dxy1_r - dxy0_r) * fz;
			img -= (dxy0_i + (dxy1_i - dxy0_i) * fz);
		}
		else
		{
			real += dxy0_r + (dxy1_r - dxy0_r) * fz;
			img += (dxy0_i + (dxy1_i - dxy0_i) * fz);
		}

		// Then interpolate (real) weight
		dd000 = my_weight_temp_D[z0 * xydim + y0 * xdim + x0];
		dd001 = my_weight_temp_D[z0 * xydim + y0 * xdim + x1];
		dd010 = my_weight_temp_D[z0 * xydim + y1 * xdim + x0];
		dd011 = my_weight_temp_D[z0 * xydim + y1 * xdim + x1];
		dd100 = my_weight_temp_D[z1 * xydim + y0 * xdim + x0];
		dd101 = my_weight_temp_D[z1 * xydim + y0 * xdim + x1];
		dd110 = my_weight_temp_D[z1 * xydim + y1 * xdim + x0];
		dd111 = my_weight_temp_D[z1 * xydim + y1 * xdim + x1];

		ddx00 = dd000 + (dd001 - dd000) * fx;
		ddx01 = dd100 + (dd101 - dd100) * fx;
		ddx10 = dd010 + (dd011 - dd010) * fx;
		ddx11 = dd110 + (dd111 - dd110) * fx;
		ddxy0 = ddx00 + (ddx10 - ddx00) * fy;
		ddxy1 = ddx01 + (ddx11 - ddx01) * fy;
		weight += ddxy0 + (ddxy1 - ddxy0) * fz;

	}
	my_data_D[id].x += real;
	my_data_D[id].y += img;
	my_weight_D[id] += weight;
}

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