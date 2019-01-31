#ifndef CUDA_RECONS_KERNELS_CUH_
#define CUDA_RECONS_KERNELS_CUH_

#include <cuda_runtime.h>
#include "src/macros.h"
#include "src/gpu_utils/cuda_settings.h"
#include "src/gpu_utils/cuda_device_utils.cuh"

__device__ float atomicMin(double* address, double val)
{
    unsigned long long *address_as_ull =(unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    while (val < __longlong_as_double(old)) {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val));
    }
    return __longlong_as_double(old);
}

__device__ void rotation2DMatrix_kernel(RFLOAT ang, RFLOAT* result)
{
    RFLOAT cosine, sine;

    ang = DEG2RAD(ang);
    cosine = cos(ang);
    sine = sin(ang);

    // result is 3x3 matrix
    result[0] = cosine;
    result[1] = -sine;
    result[2] = 0;
    result[3] = sine;
    result[4] = cosine;
    result[5] = 0;
    result[6] = 0;
    result[7] = 0;
    result[8] = 1;
}

__device__ void matInv_9(RFLOAT* A) {
    RFLOAT B[9];
    B[0] = A[8]*A[4]-A[7]*A[5];
    B[1] =-A[8]*A[1]-A[7]*A[2];
    B[2] = A[5]*A[1]-A[4]*A[2];
    B[3] =-A[8]*A[3]-A[6]*A[5];
    B[4] = A[8]*A[0]-A[6]*A[2];
    B[5] =-A[5]*A[0]-A[3]*A[2];
    //B[6] = A[7]*A[3]-A[6]*A[4];
    //B[7] =-A[7]*A[0]-A[6]*A[1];
    //B[8] = A[4]*A[0]-A[3]*A[1];
    RFLOAT tmp = A[0]*B[0] + A[3]*B[1] + A[6]*B[2];
    //A[0] = B[0]/tmp;
    //A[1] = B[1]/tmp;
    //A[2] = B[2]/tmp;
    A[3] = B[3]/tmp;
    A[4] = B[4]/tmp;
    A[5] = B[5]/tmp;
    //A[6] = B[6]/tmp;
    //A[7] = B[7]/tmp;
    //A[8] = B[8]/tmp;
}

__device__ RFLOAT calculateDeltaRot_kernel(RFLOAT* my_direction, RFLOAT rot_prior)
{
	// Rotate the x,y-components of the direction, according to rot-prior
	RFLOAT my_rot_direction[3];
	RFLOAT A[9];
	rotation2DMatrix_kernel(rot_prior, A);
    matInv_9(A);
	my_rot_direction[1] = A[3]*my_direction[0] + A[4]*my_direction[1] + A[5]*my_direction[2];
	// Get component along the new Y-axis
	return fabs(ASIND(my_rot_direction[1]));
}

__device__ void Euler_angles2direction_Kernel(RFLOAT alpha, RFLOAT beta, RFLOAT* v)
{
    RFLOAT ca, sa, cb, sb;
    RFLOAT sc, ss;

    //v.resize(3);
    // v should has at least 3 elements
    alpha = DEG2RAD(alpha);
    beta  = DEG2RAD(beta);

    ca = cos(alpha);
    cb = cos(beta);
    sa = sin(alpha);
    sb = sin(beta);
    sc = sb * ca;
    ss = sb * sa;

    v[0] = sc;
    v[1] = ss;
    v[2] = cb;
}

__device__ RFLOAT dotProduct_kernel(RFLOAT* v1, RFLOAT* v2)
{
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

__device__ RFLOAT gaussian1D_kernel(RFLOAT x, RFLOAT sigma, RFLOAT mu, RFLOAT sqrt_sigma)
{
    x -= mu;
    return 1 / sqrt_sigma*exp(-0.5*((x / sigma)*(x / sigma)));
}

#ifdef USE_SHAREMEM
template<int BLOCKSIZE>
__global__ void selectRotTiltDirection(RFLOAT* directions_prior, int* findDirNum, RFLOAT* sumprior,
         RFLOAT* R_repository, RFLOAT* L_repository, int R_repository_Size,
         int rotAnglesSize,
         RFLOAT prior_rot, RFLOAT prior_tilt, RFLOAT* rot_angles, RFLOAT* tilt_angles,
         RFLOAT sigma_rot, RFLOAT sigma_tilt, RFLOAT sigma_cutoff,
         RFLOAT biggest_sigma, RFLOAT* diffang_corr, 
         RFLOAT cosSigmaBiggestXcutoff, RFLOAT sqrt_sigma,
         RFLOAT prior_direction0, RFLOAT prior_direction1, RFLOAT prior_direction2)
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int idir = tid+bid*blockDim.x;

    __shared__ RFLOAT sumprior_sh[BLOCKSIZE];
    __shared__ int    findDirnum_sh[BLOCKSIZE];

    __shared__ RFLOAT R_repository_sh[R_repository_Size*9];
    __shared__ RFLOAT L_repository_sh[R_repository_Size*9];

    sumprior_sh[tid] = 0;
    findDirnum_sh[tid] = 0;

    int i;
    for(i=tid;i<R_repository_Size*9;i+=BLOCKSIZE) {
        R_repository_sh[i] = R_repository[i];
        L_repository_sh[i] = L_repository[i];
    }

    __syncthreads();

    if(idir < rotAnglesSize) 
    {
        RFLOAT my_direction[3];
        RFLOAT sym_direction[3];
        RFLOAT best_direction[3];

        // Get the current direction in the loop
        Euler_angles2direction_Kernel(rot_angles[idir], tilt_angles[idir], my_direction);

        // Loop over all symmetry operators to find the operator that brings this direction nearest to the prior
        RFLOAT best_dotProduct = prior_direction0*my_direction[0] +
                                    prior_direction1*my_direction[1] +
                                    prior_direction2*my_direction[2];
        best_direction[0] = my_direction[0];
        best_direction[1] = my_direction[1];
        best_direction[2] = my_direction[2];
        // TODO: prefetch L_repository to share mem
        for (int j = 0; j < R_repository_Size; j++)
        {
            int s = j*9;
            // each L / R repository is 3 x 3 matrix
            const RFLOAT t1 = (my_direction[0]*R_repository[s  ]+my_direction[1]*R_repository[s+3]+my_direction[2]*R_repository[s+6]);
            const RFLOAT t2 = (my_direction[0]*R_repository[s+1]+my_direction[1]*R_repository[s+4]+my_direction[2]*R_repository[s+7]);
            const RFLOAT t3 = (my_direction[0]*R_repository[s+2]+my_direction[1]*R_repository[s+5]+my_direction[2]*R_repository[s+8]);
            sym_direction[0] = L_repository[s  ] * t1 + L_repository[s+1] * t2 + L_repository[s+2] * t3;
            sym_direction[1] = L_repository[s+3] * t1 + L_repository[s+4] * t2 + L_repository[s+5] * t3;
            sym_direction[2] = L_repository[s+6] * t1 + L_repository[s+7] * t2 + L_repository[s+8] * t3;
            RFLOAT my_dotProduct = prior_direction0*sym_direction[0] +
                                    prior_direction1*sym_direction[1] +
                                    prior_direction2*sym_direction[2];
            if (my_dotProduct > best_dotProduct)
            {
                best_direction[0] = sym_direction[0];
                best_direction[1] = sym_direction[1];
                best_direction[2] = sym_direction[2];
                best_dotProduct = my_dotProduct;
            }
        }

        // Now that we have the best direction, find the corresponding prior probability
        RFLOAT diffang = prior_direction0*best_direction[0] +
                                    prior_direction1*best_direction[1] +
                                    prior_direction2*best_direction[2];

        // Only consider differences within sigma_cutoff * sigma_ro
        diffang_corr[idir] = diffang;
        directions_prior[idir] = 0.;
        if (diffang > cosSigmaBiggestXcutoff)
        {
            diffang = ACOSD(diffang);
            RFLOAT prior = gaussian1D_kernel(diffang, biggest_sigma, 0., sqrt_sigma);
            directions_prior[idir] = prior;
        }
    }
    __syncthreads();
    // BLOCKSIZE == 128
    if(tid < 64) {
        sumprior_sh[tid]   += sumprior_sh[tid+64];
        findDirnum_sh[tid] += findDirnum_sh[tid+64];
    }
    __syncthreads();
    if(tid < 32) {
        sumprior_sh[tid]   += sumprior_sh[tid+32];
        findDirnum_sh[tid] += findDirnum_sh[tid+32];
    }
    __syncthreads();
    if(tid < 16) {
        sumprior_sh[tid]   += sumprior_sh[tid+16];
        findDirnum_sh[tid] += findDirnum_sh[tid+16];
    }
    __syncthreads();
    if(tid < 8) {
        sumprior_sh[tid]   += sumprior_sh[tid+8];
        findDirnum_sh[tid] += findDirnum_sh[tid+8];
    }
    __syncthreads();
    if(tid < 4) {
        sumprior_sh[tid]   += sumprior_sh[tid+4];
        findDirnum_sh[tid] += findDirnum_sh[tid+4];
    }
    __syncthreads();
    if(tid < 2) {
        sumprior_sh[tid]   += sumprior_sh[tid+2];
        findDirnum_sh[tid] += findDirnum_sh[tid+2];
    }
    __syncthreads();
    if(tid < 1) {
        sumprior_sh[tid]   += sumprior_sh[tid+1];
        findDirnum_sh[tid] += findDirnum_sh[tid+1];
        atomicAdd(sumprior,sumprior_sh[0]);
        atomicAdd(findDirNum,finDirNum_sh[0]);
    }
}
#else
template<int BLOCKSIZE>
__global__ void selectRotTiltDirection(RFLOAT* directions_prior, int* findDirNum, RFLOAT* sumprior,
         RFLOAT* R_repository, RFLOAT* L_repository, int R_repository_Size,
         int rotAnglesSize,
         RFLOAT prior_rot, RFLOAT prior_tilt, RFLOAT* rot_angles, RFLOAT* tilt_angles,
         RFLOAT sigma_rot, RFLOAT sigma_tilt, RFLOAT sigma_cutoff,
         RFLOAT biggest_sigma, RFLOAT* diffang_corr, 
         RFLOAT cosSigmaBiggestXcutoff, RFLOAT sqrt_sigma,
         RFLOAT prior_direction0, RFLOAT prior_direction1, RFLOAT prior_direction2)
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int idir = tid+bid*blockDim.x;
    if(idir >= rotAnglesSize) return;

    //RFLOAT prior_direction[3];
    RFLOAT my_direction[3];
    RFLOAT sym_direction[3];
    RFLOAT best_direction[3];

    // Get the direction of the prior
    //Euler_angles2direction_Kernel(prior_rot, prior_tilt, prior_direction);

    // Get the current direction in the loop
    Euler_angles2direction_Kernel(rot_angles[idir], tilt_angles[idir], my_direction);

    // Loop over all symmetry operators to find the operator that brings this direction nearest to the prior
    //RFLOAT best_dotProduct = dotProduct_kernel(prior_direction, my_direction);
    RFLOAT best_dotProduct = prior_direction0*my_direction[0] +
                                prior_direction1*my_direction[1] +
                                prior_direction2*my_direction[2];
    best_direction[0] = my_direction[0];
    best_direction[1] = my_direction[1];
    best_direction[2] = my_direction[2];
    // TODO: prefetch L_repository to share mem
    for (int j = 0; j < R_repository_Size; j++)
    {
        int s = j*9;
        // each L / R repository is 3 x 3 matrix
        const RFLOAT t1 = (my_direction[0]*R_repository[s  ]+my_direction[1]*R_repository[s+3]+my_direction[2]*R_repository[s+6]);
        const RFLOAT t2 = (my_direction[0]*R_repository[s+1]+my_direction[1]*R_repository[s+4]+my_direction[2]*R_repository[s+7]);
        const RFLOAT t3 = (my_direction[0]*R_repository[s+2]+my_direction[1]*R_repository[s+5]+my_direction[2]*R_repository[s+8]);
        //sym_direction =  tL * (my_direction.transpose() * tR).transpose();
        sym_direction[0] = L_repository[s  ] * t1 + L_repository[s+1] * t2 + L_repository[s+2] * t3;
        sym_direction[1] = L_repository[s+3] * t1 + L_repository[s+4] * t2 + L_repository[s+5] * t3;
        sym_direction[2] = L_repository[s+6] * t1 + L_repository[s+7] * t2 + L_repository[s+8] * t3;
        //RFLOAT my_dotProduct = dotProduct_kernel(prior_direction, sym_direction);
        RFLOAT my_dotProduct = prior_direction0*sym_direction[0] +
                                prior_direction1*sym_direction[1] +
                                prior_direction2*sym_direction[2];
        if (my_dotProduct > best_dotProduct)
        {
            best_direction[0] = sym_direction[0];
            best_direction[1] = sym_direction[1];
            best_direction[2] = sym_direction[2];
            best_dotProduct = my_dotProduct;
        }
    }

    // Now that we have the best direction, find the corresponding prior probability
    //RFLOAT diffang = ACOSD( dotProduct_kernel(best_direction, prior_direction) );
    //RFLOAT diffang = dotProduct_kernel(best_direction, prior_direction);
    RFLOAT diffang = prior_direction0*best_direction[0] +
                                prior_direction1*best_direction[1] +
                                prior_direction2*best_direction[2];
    //if (diffang > 180.)
    //    diffang = ABS(diffang - 360.);

    // Only consider differences within sigma_cutoff * sigma_ro
    //RFLOAT biggest_sigma = XMIPP_MAX(sigma_rot, sigma_tilt);
    diffang_corr[idir] = diffang;
    //pointer_dir_nonzeroprior[idir] = 0;
    directions_prior[idir] = 0.;
    //if (diffang < sigma_cutoff * biggest_sigma)
    if (diffang > cosSigmaBiggestXcutoff)
    {
        diffang = ACOSD(diffang);
        RFLOAT prior = gaussian1D_kernel(diffang, biggest_sigma, 0., sqrt_sigma);
        //pointer_dir_nonzeroprior[idir] = idir+1;
        directions_prior[idir] = prior;
        atomicAdd(sumprior, prior);
        atomicAdd(findDirNum, 1);
    }
}
#endif
/*
template<int BLOCKSIZE>
void selectRotDirection()
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int idir = tid+bid*blockDim.x;
    __shared__ RFLOAT diffang_sh[BLOCKSIZE];
    __shared__ int best_idir_sh[BLOCKSIZE];
    if(idir >= rotAnglesSize) return;

    RFLOAT my_direction[3];
    RFLOAT sym_direction[3];
    RFLOAT sym_rot, sym_tilt;

	// Get the current direction in the loop
	Euler_angles2direction_Kernel(rot_angles[idir], tilt_angles[idir], my_direction);
	RFLOAT diffang = calculateDeltaRot_kernel(my_direction, prior_rot);
	RFLOAT best_diffang = diffang;
	for (int j = 0; j < R_repository.size(); j++)
	{
		sym_direction =  L_repository[j] * (my_direction.transpose() * R_repository[j]).transpose();
		diffang = calculateDeltaRot_kernel(sym_direction, prior_rot);

		if (diffang < best_diffang)
			best_diffang = diffang;
	}

				// Only consider differences within sigma_cutoff * sigma_rot
				if (best_diffang < sigma_cutoff * sigma_rot)
				{
					RFLOAT prior = gaussian1D(best_diffang, sigma_rot, 0.);
					pointer_dir_nonzeroprior.push_back(idir);
					directions_prior.push_back(prior);
					sumprior += prior;
				}

				// Keep track of the nearest direction
				if (best_diffang < best_ang)
				{
					best_idir = idir;
					best_ang = diffang;
				}

}
*/
#endif