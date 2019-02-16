#ifndef CUDA_FFT_H_
#define CUDA_FFT_H_

#include "src/gpu_utils/cuda_settings.h"
#include "src/gpu_utils/cuda_mem_utils.h"
#include <cuda_runtime.h>
#include <cufft.h>

#include "src/gpu_utils/cuda_kernels/fft_helper.cuh"

#ifdef DEBUG_CUDA
#define HANDLE_CUFFT_ERROR( err ) (CufftHandleError( err, __FILE__, __LINE__ ))
#else
#define HANDLE_CUFFT_ERROR( err ) (err) //Do nothing
#endif
static void CufftHandleError( cufftResult err, const char *file, int line )
{
    if (err != CUFFT_SUCCESS)
    {
        fprintf(stderr, "Cufft error in file '%s' in line %i : %s.\n",
                __FILE__, __LINE__, "error" );
		raise(SIGSEGV);
    }
}

template<typename REAL_T, typename COMPLEX_T, bool isDouble, bool allocHost, bool isComplex=false, bool customAlloc=true>
class CudaFFT_TP
{
	bool planSet;
public:
	CudaGlobalPtr<REAL_T,customAlloc> reals;
	CudaGlobalPtr<COMPLEX_T,customAlloc> fouriers;
	cufftHandle cufftPlanForward, cufftPlanBackward;
	int direction;
	int dimension, idist, odist, istride, ostride;
	int inembed[3];
	int onembed[3];
	size_t xSize,ySize,zSize,xFSize,yFSize,zFSize;
	std::vector< int >  batchSize;
	CudaCustomAllocator *CFallocator;
	int batchSpace, batchIters, reqN;

	CudaFFT_TP(cudaStream_t stream, CudaCustomAllocator *allocator, int transformDimension = 2):
		reals(stream, allocator),
		fouriers(stream, allocator),
		cufftPlanForward(0),
		cufftPlanBackward(0),
		direction(0),
		dimension((int)transformDimension),
	    idist(0),
	    odist(0),
	    istride(1),
	    ostride(1),
		planSet(false),
		xSize(0), ySize(0), zSize(0),
		xFSize(0), yFSize(0), zFSize(0),
		batchSize(1,1),
		reqN(1),
		CFallocator(allocator)
	{};

	size_t estimate(int batch)
	{
		size_t needed(0);

	  size_t biggness;

		if(isDouble) {
			if(isComplex) {
				if(direction<=0)
				{
					HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch, &biggness));
					needed += biggness;
				}
				if(direction>=0)
				{
					HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2Z, batch, &biggness));
					needed += biggness;
				}
			}
			else {
				if(direction<=0)
				{
					HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, batch, &biggness));
					needed += biggness;
				}
				if(direction>=0)
				{
					HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, batch, &biggness));
					needed += biggness;
				}
			}
		}
		else {
			if(isComplex) {
				if(direction<=0)
				{
					HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch, &biggness));
					needed += biggness;
				}
				if(direction>=0)
				{
					HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2C, batch, &biggness));
					needed += biggness;
				}
			}
			else {
				if(direction<=0)
				{
					HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch, &biggness));
					needed += biggness;
				}
				if(direction>=0)
				{
					HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2R, batch, &biggness));
					needed += biggness;
				}
			}
		}
		size_t res;
		if(isComplex)
			res = needed + (size_t)odist*(size_t)batch*sizeof(COMPLEX_T) + (size_t)idist*(size_t)batch*sizeof(REAL_T);
		else
			res = needed + (size_t)odist*(size_t)batch*sizeof(REAL_T)*(size_t)2 + (size_t)idist*(size_t)batch*sizeof(REAL_T);

		return res;
	}

	int setSize(size_t x, size_t y, size_t z, int batch = 1, int setDirection = 0)
	{

		/* Optional direction input restricts transformer to
		 * forwards or backwards tranformation only,
		 * which reduces memory requirements, especially
		 * for large batches of simulatanous transforms.
		 *
		 * FFTW_FORWARDS  === -1
		 * FFTW_BACKWARDS === +1
		 *
		 * The default direction is 0 === forwards AND backwards
		 */

		int checkDim;
		if(z>1)
			checkDim=3;
		else if(y>1)
			checkDim=2;
		else
			checkDim=1;
		if(checkDim != dimension)
			CRITICAL(ERRCUFFTDIM);

		if( !( (setDirection==-1)||(setDirection==0)||(setDirection==1) ) )
		{
			std::cerr << "*ERROR : Setting a cuda transformer direction to non-defined value" << std::endl;
			CRITICAL(ERRCUFFTDIR);
		}

		direction = setDirection;

		if (x == xSize && y == ySize && z == zSize && batch == reqN && planSet)
			return;

		clear();

		batchSize.resize(1);
		batchSize[0] = batch;
		reqN = batch;

		xSize = x;
		ySize = y;
		zSize = z;

		xFSize = x/2 + 1;
		yFSize = y;
		zFSize = z;

	    idist = zSize*ySize*xSize;
	    odist = zSize*ySize*(xSize/2+1);
	    istride = 1;
	    ostride = 1;

	    if(dimension==3)
	    {
	    	inembed[0] =  zSize;
			inembed[1] =  ySize;
			inembed[2] =  xSize;
			onembed[0] =  zFSize;
			onembed[1] =  yFSize;
			onembed[2] =  xFSize;
	    }
	    else if(dimension==2)
	    {
			inembed[0] =  ySize;
			inembed[1] =  xSize;
			onembed[0] =  yFSize;
			onembed[1] =  xFSize;
	    }
	    else
	    {
			inembed[0] =  xSize;
			onembed[0] =  xFSize;
	    }

		size_t needed, avail, total;
		needed = estimate(batchSize[0]);
		DEBUG_HANDLE_ERROR(cudaMemGetInfo( &avail, &total ));

//		std::cout << std::endl << "needed = ";
//		printf("%15zu\n", needed);
//		std::cout << "avail  = ";
//		printf("%15zu\n", avail);

		// Check if there is enough memory
		//
		//    --- TO HOLD TEMPORARY DATA DURING TRANSFORMS ---
		//
		// If there isn't, find how many there ARE space for and loop through them in batches.

		if(needed>avail)
		{
			batchIters = 2;
			batchSpace = CEIL((double) batch / (double)batchIters);
			needed = estimate(batchSpace);

			while(needed>avail && batchSpace>1)
			{
				batchIters++;
				batchSpace = CEIL((double) batch / (double)batchIters);
				needed = estimate(batchSpace);
			}

			if(batchIters>1)
			{
				batchIters = (int)((float)batchIters*1.1 + 1);
				batchSpace = CEIL((double) batch / (double)batchIters);
				needed = estimate(batchSpace);
			}

			batchSize.assign(batchIters,batchSpace); // specify batchIters of batches, each with batchSpace orientations
			batchSize[batchIters-1] = batchSpace - (batchSpace*batchIters - batch); // set last to care for remainder.

			if(needed>avail) {
				//CRITICAL(ERRFFTMEMLIM);
				std::cout << "WARNING: GPU mem may not enough! Use CPU" << std::endl;
				return -1;
			}

//			std::cerr << std::endl << "NOTE: Having to use " << batchIters << " batches of orientations ";
//			std::cerr << "to achieve the total requested " << batch << " orientations" << std::endl;
//			std::cerr << "( this could affect performance, consider using " << std::endl;
//			std::cerr << "\t higher --ang" << std::endl;
//			std::cerr << "\t harder --shrink" << std::endl;
//			std::cerr << "\t higher --lopass with --shrink 0" << std::endl;

		}
		else
		{
			batchIters = 1;
			batchSpace = batch;
		}

		reals.setSize(idist*batchSize[0]);
		reals.device_alloc();
		if(allocHost) reals.host_alloc();

		fouriers.setSize(odist*batchSize[0]);
		fouriers.device_alloc();
		if(allocHost) fouriers.host_alloc();

//		DEBUG_HANDLE_ERROR(cudaMemGetInfo( &avail, &total ));
//		needed = estimate(batchSize[0], fudge);

//		std::cout << "after alloc: " << std::endl << std::endl << "needed = ";
//		printf("%15li\n", needed);
//		std::cout << "avail  = ";
//		printf("%15li\n", avail);
		setPlan();
		return 0;
	}

	void setPlan() {
		if(isDouble) {
			if(isComplex) {
				if(direction<=0)
				{
					HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward, fouriers.getStream()));
				}
				if(direction>=0)
				{
					HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2Z, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward, reals.getStream()));
				}
			} 
			else {
				if(direction<=0)
				{
					HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward, fouriers.getStream()));
				}
				if(direction>=0)
				{
					HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward, reals.getStream()));
				}
			}
			planSet = true;
		}
		else {
			if(isComplex) {
				if(direction<=0)
				{
					HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward, fouriers.getStream()));
				}
				if(direction>=0)
				{
					HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2C, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward, reals.getStream()));
				}
			}
			else {
				if(direction<=0)
				{
					HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward, fouriers.getStream()));
				}
				if(direction>=0)
				{
					HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2R, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward, reals.getStream()));
				}
			}
			planSet = true;
		}
	}

	void forward()
	{
		if(isComplex) {
			if(isDouble)
				HANDLE_CUFFT_ERROR( cufftExecZ2Z(cufftPlanForward, (cufftDoubleComplex*)~reals, (cufftDoubleComplex*)~fouriers, CUFFT_FORWARD )); 
			else
				HANDLE_CUFFT_ERROR( cufftExecC2C(cufftPlanForward, (cufftComplex*)~reals, (cufftComplex*)~fouriers, CUFFT_FORWARD ));
		}
		else {
			if(isDouble)
				HANDLE_CUFFT_ERROR( cufftExecD2Z(cufftPlanForward, (cufftDoubleReal*)~reals, (cufftDoubleComplex*)~fouriers) ); 
			else
				HANDLE_CUFFT_ERROR( cufftExecR2C(cufftPlanForward, (cufftReal*)~reals, (cufftComplex*)~fouriers) );
		}
	}

	void backward()
	{
		if(isComplex) {
			if(isDouble) 
				HANDLE_CUFFT_ERROR( cufftExecZ2Z(cufftPlanBackward, (cufftDoubleComplex*)~fouriers, (cufftDoubleComplex*)~reals, CUFFT_INVERSE ));
			else
				HANDLE_CUFFT_ERROR( cufftExecC2C(cufftPlanBackward, (cufftComplex*)~fouriers, (cufftComplex*)~reals, CUFFT_INVERSE ));
		}
		else {
			if(isDouble) 
				HANDLE_CUFFT_ERROR( cufftExecZ2D(cufftPlanBackward, (cufftDoubleComplex*)~fouriers, (cufftDoubleReal*)~reals) );
			else
				HANDLE_CUFFT_ERROR( cufftExecC2R(cufftPlanBackward, (cufftComplex*)~fouriers, (cufftReal*)~reals) );
		}
	}

	void backward(CudaGlobalPtr<cufftDoubleReal> &dst)
		{ HANDLE_CUFFT_ERROR( cufftExecZ2D(cufftPlanBackward, ~fouriers, ~dst) ); }

	void clearPlan() {
		if(direction<=0)
			HANDLE_CUFFT_ERROR(cufftDestroy(cufftPlanForward));
		if(direction>=0)
			HANDLE_CUFFT_ERROR(cufftDestroy(cufftPlanBackward));
		planSet = false;
	}

	void clear()
	{
		if(planSet)
		{
			if(allocHost) reals.free_host_if_set();
			reals.free_device_if_set();
			if(allocHost) fouriers.free_host_if_set();
			fouriers.free_device_if_set();
			clearPlan();
		}
	}

	~CudaFFT_TP()
	{clear();}
};

#ifdef CUDA_DOUBLE_PRECISION
	typedef CudaFFT_TP<cufftDoubleReal,cufftDoubleComplex,true,true> CudaFFT;
#else
	typedef CudaFFT_TP<cufftReal,cufftComplex,false,true> CudaFFT;
#endif

#ifdef RELION_SINGLE_PRECISION
	typedef CudaFFT_TP<cufftReal,cufftComplex,false,false,false,false> RelionCudaFFT;
#else
	typedef CudaFFT_TP<cufftDoubleReal,cufftDoubleComplex,true,false,false,false> RelionCudaFFT;
#endif

#endif
