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
		switch(err) {
			case CUFFT_INVALID_PLAN:
				fprintf(stderr, "Cufft error in file '%s' in line %i : %s.\n",
                	file, line, "invalid plan" );
				break;
			case CUFFT_ALLOC_FAILED:
				fprintf(stderr, "Cufft error in file '%s' in line %i : %s.\n",
                	file, line, "alloc failed" );
				break;
			case CUFFT_INVALID_VALUE:
				fprintf(stderr, "Cufft error in file '%s' in line %i : %s.\n",
                	file, line, "invalid value" );
				break;
			case CUFFT_INTERNAL_ERROR:
				fprintf(stderr, "Cufft error in file '%s' in line %i : %s.\n",
                	file, line, "internal error" );
				break;
			case CUFFT_SETUP_FAILED:
				fprintf(stderr, "Cufft error in file '%s' in line %i : %s.\n",
                	file, line, "setup failed" );
				break;
			case CUFFT_INVALID_SIZE:
				fprintf(stderr, "Cufft error in file '%s' in line %i : %s.\n",
                	file, line, "invalid size" );
				break;
			default:
				fprintf(stderr, "Cufft error in file '%s' in line %i : %s : %d.\n",
                	file, line, "unknown error : ", err );
		}
		fflush(stderr);
		raise(SIGSEGV);
    }
}

template<typename REAL_T, typename COMPLEX_T, bool isDouble, bool allocHost, bool isComplex=false, bool customAlloc=true>
class CudaFFT_TP
{
	bool planSet;
	CudaGlobalPtr<char,customAlloc> forward_workspace;
	CudaGlobalPtr<char,customAlloc> backward_workspace;
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
		forward_workspace(stream, allocator),
		backward_workspace(stream, allocator),
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
					//printf("Estimate D2Z %ld\n",biggness);
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
			return 0;

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
		if(customAlloc) avail = CFallocator->getTotalFreeSpace();
		else DEBUG_HANDLE_ERROR(cudaMemGetInfo( &avail, &total ));

		//printf("Need=%ld, avail=%ld\n",needed,avail);

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
			//printf("Iter=%d : Need=%ld\n",batchIters, needed);

			while(needed>avail && batchSpace>1)
			{
				batchIters++;
				batchSpace = CEIL((double) batch / (double)batchIters);
				needed = estimate(batchSpace);
				//printf("Iter=%d : Need=%ld\n",batchIters, needed);
			}

			if(batchIters>1)
			{
				batchIters = (int)((float)batchIters*1.1 + 1);
				batchSpace = CEIL((double) batch / (double)batchIters);
				needed = estimate(batchSpace);
				//printf("Iter=%d : Need=%ld\n",batchIters, needed);
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

		DEBUG_HANDLE_ERROR(cudaMemGetInfo( &avail, &total ));
//		needed = estimate(batchSize[0], fudge);

//		std::cout << "after alloc: " << std::endl << std::endl << "needed = ";
//		printf("%15li\n", needed);
//		std::cout << "avail  = ";
		//printf("%15li\n", avail);
		setPlan();
		return 0;
	}

	void setPlan() {
		if(planSet) return ;
		// create plans
		if(direction<=0) HANDLE_CUFFT_ERROR(cufftCreate(&cufftPlanForward));
		if(direction>=0) HANDLE_CUFFT_ERROR(cufftCreate(&cufftPlanBackward));
		// set workspace of plans to be self-managed
		if(direction<=0) HANDLE_CUFFT_ERROR( cufftSetAutoAllocation(cufftPlanForward, false));
		if(direction>=0) HANDLE_CUFFT_ERROR( cufftSetAutoAllocation(cufftPlanBackward, false));
		size_t workSize;
		// make plans
		if(isDouble) {
			if(isComplex) {
				if(direction<=0)
				{
					//HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batchSize[0]));
					HANDLE_CUFFT_ERROR(cufftMakePlanMany(cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batchSize[0], &workSize));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward, fouriers.getStream()));
					forward_workspace.setSize(workSize);
				}
				if(direction>=0)
				{
					//HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2Z, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftMakePlanMany(cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2Z, batchSize[0], &workSize));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward, reals.getStream()));
					backward_workspace.setSize(workSize);
				}
			} 
			else {
				if(direction<=0)
				{
					//HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, batchSize[0]));
					size_t avail, total;
					DEBUG_HANDLE_ERROR(cudaMemGetInfo( &avail, &total ));
					HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, batchSize[0], &workSize));
					//printf("%d %d %d, avail=%ld, workSize=%ld\n",dimension, batchSize[0], inembed[0], avail, workSize);
					HANDLE_CUFFT_ERROR( cufftMakePlanMany(cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, batchSize[0], &workSize));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward, fouriers.getStream()));
					forward_workspace.setSize(workSize);
					//printf("fwd workSize=%ld\n",workSize);
				}
				if(direction>=0)
				{
					//HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftMakePlanMany(cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, batchSize[0], &workSize));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward, reals.getStream()));
					backward_workspace.setSize(workSize);
					//printf("bwd workSize=%ld, allocate=%ld\n",workSize,workSize*sizeof(char));
				}
			}
		}
		else {
			if(isComplex) {
				if(direction<=0)
				{
					//HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftMakePlanMany(cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batchSize[0], &workSize));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward, fouriers.getStream()));
					forward_workspace.setSize(workSize);
				}
				if(direction>=0)
				{
					//HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2C, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftMakePlanMany(cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2C, batchSize[0], &workSize));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward, reals.getStream()));
					backward_workspace.setSize(workSize);
				}
			}
			else {
				if(direction<=0)
				{
					//HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftMakePlanMany(cufftPlanForward,  dimension, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batchSize[0], &workSize));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward, fouriers.getStream()));
					forward_workspace.setSize(workSize);
				}
				if(direction>=0)
				{
					//HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2R, batchSize[0]));
					HANDLE_CUFFT_ERROR( cufftMakePlanMany(cufftPlanBackward, dimension, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2R, batchSize[0], &workSize));
					HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward, reals.getStream()));
					backward_workspace.setSize(workSize);
				}
			}
		}
		// allocate workspace
		if(direction<=0) forward_workspace.device_alloc();
		if(direction>=0) backward_workspace.device_alloc();
		// set workspace
		if(direction<=0) HANDLE_CUFFT_ERROR( cufftSetWorkArea(cufftPlanForward, (void*)~forward_workspace) );
		if(direction>=0) HANDLE_CUFFT_ERROR( cufftSetWorkArea(cufftPlanBackward, (void*)~backward_workspace) );
		planSet = true;
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
		if(direction<=0) {
			forward_workspace.free_device_if_set();
			HANDLE_CUFFT_ERROR(cufftDestroy(cufftPlanForward));
		}
		if(direction>=0) {
			backward_workspace.free_device_if_set();
			HANDLE_CUFFT_ERROR(cufftDestroy(cufftPlanBackward));
		}
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
		free_all();
	}

	void free_all()
	{
		reals.free_if_set();
		fouriers.free_if_set();
		forward_workspace.free_if_set();
		backward_workspace.free_if_set();
	}

	~CudaFFT_TP()
	{clear();}
};

// dst = transpose(src), src[x][y] => dst[y][x], all device ptr
void run_transpose_gpu(__COMPLEX_T* dst, __COMPLEX_T* src, int x, int y, cudaStream_t stream);
void run_transpose_gpu(RFLOAT* dst, RFLOAT* src, int x, int y, cudaStream_t stream);

void run_cuda_transpose(CudaGlobalPtr<__COMPLEX_T, false> &dst, CudaGlobalPtr<__COMPLEX_T, false> &src, int x, int y, bool hostBuffer=false);

// If memory is not enough for a single FFT, it will try to split to do the FFT
class BuffCudaFFT3D
{
	bool planSet;
	bool splitted;
	bool hostBuffer;
	int inembed[3];
	int onembed[3];
	int inembed2D[2];
	int onembed2D[2];
	int inembed1D[1];
	int onembed1D[1];
	size_t idist1D, odist1D;
	size_t idist2D, odist2D;
	int dimension;
public:
#ifdef RELION_SINGLE_PRECISION
	CudaGlobalPtr<cufftReal, false> reals;
	CudaGlobalPtr<cufftComplex, false> fouriers;
	CudaGlobalPtr<cufftComplex, false> buff;
	CudaGlobalPtr<cufftComplex, false> _fft_buff;
#else
	CudaGlobalPtr<cufftDoubleReal, false> reals;
	CudaGlobalPtr<cufftDoubleComplex, false> fouriers;
	CudaGlobalPtr<cufftDoubleComplex, false> buff;
	CudaGlobalPtr<cufftDoubleComplex, false> _fft_buff;
#endif
	cufftHandle cufftPlanForward, cufftPlanBackward;
	cufftHandle cufftPlanForward1D, cufftPlanBackward1D;
	cufftHandle cufftPlanForward2D, cufftPlanBackward2D;
	int direction;
	size_t idist, odist;
	int istride, ostride;
	
	size_t xSize,ySize,zSize,xFSize,yFSize,zFSize;
	size_t batchSpace_1D, batchSpace_2D;
	size_t batchIters_1D, batchIters_2D;

	std::vector<size_t> batchSize_1D;
	std::vector<size_t> batchSize_2D;

	BuffCudaFFT3D(cudaStream_t stream):
		reals(stream),
		fouriers(stream),
		buff(stream),
		_fft_buff(stream),
		cufftPlanForward(0),
		cufftPlanBackward(0),
		cufftPlanForward1D(0),
		cufftPlanBackward1D(0),
		cufftPlanForward2D(0),
		cufftPlanBackward2D(0),
		direction(0),
		dimension(3),
	    idist(0),
	    odist(0),
	    istride(1),
	    ostride(1),
			idist1D(0),
			odist1D(0),
			idist2D(0),
			odist2D(0),
		planSet(false),
		splitted(false),
		xSize(0), ySize(0), zSize(0),
		xFSize(0), yFSize(0), zFSize(0),
		batchIters_1D(1), batchIters_2D(1),
		batchSpace_1D(0), batchSpace_2D(0),
		batchSize_1D(1,1),
		batchSize_2D(1,1),
		hostBuffer(false)
	{};

private:
#ifdef RELION_SINGLE_PRECISION
	inline void forward_1D()
	{
		cufftComplex* input = (cufftComplex*)~fouriers;
		cufftComplex* output= (cufftComplex*)~buff;
		for(int iter=0;iter<batchIters_1D;++iter) {
			HANDLE_CUFFT_ERROR( cufftExecC2C(cufftPlanForward1D, input, output, CUFFT_FORWARD )); 
			buff.streamSync();
			input += batchSpace_1D*idist1D;
			output+= batchSpace_1D*odist1D;
		}
	}

	inline void forward_2D()
	{
		cufftReal* input = (cufftReal*)~reals;
		cufftComplex* output= (cufftComplex*)~buff;
		for(int iter=0;iter<batchIters_2D;++iter) {
			HANDLE_CUFFT_ERROR( cufftExecR2C(cufftPlanForward2D, input, output ));
			buff.streamSync();
			input += batchSpace_2D*idist2D;
			output+= batchSpace_2D*odist2D;
		}
	}

	inline void forward_3D()
	{
		HANDLE_CUFFT_ERROR( cufftExecR2C(cufftPlanForward, (cufftReal*)~reals, (cufftComplex*)~fouriers) );
	}

	inline void backward_1D()
	{
		cufftComplex* input = (cufftComplex*)~buff;
		cufftComplex* output= (cufftComplex*)~fouriers;
		for(int iter=0;iter<batchIters_1D;++iter) {
			HANDLE_CUFFT_ERROR( cufftExecC2C(cufftPlanBackward1D, input, output, CUFFT_INVERSE ));
			buff.streamSync();
			input += batchSpace_1D*odist1D;
			output+= batchSpace_1D*idist1D;
		}
	}

	inline void backward_2D()
	{
		cufftComplex* input= (cufftComplex*)~buff;
		cufftReal* output = (cufftReal*)~reals;
		for(int iter=0;iter<batchIters_2D;++iter) {
			HANDLE_CUFFT_ERROR( cufftExecC2R(cufftPlanBackward2D, input, output) );
			buff.streamSync();
			input += batchSpace_2D*odist2D;
			output+= batchSpace_2D*idist2D;
		}
	}

	inline void backward_3D()
	{
		HANDLE_CUFFT_ERROR( cufftExecC2R(cufftPlanBackward, (cufftComplex*)~fouriers, (cufftReal*)~reals) );
	}
#else
	inline void forward_1D()
	{
		if(hostBuffer) {
			cufftDoubleComplex* input = (cufftDoubleComplex*)(fouriers.h_ptr);
			cufftDoubleComplex* output= (cufftDoubleComplex*)(buff.h_ptr);
			for(int iter=0;iter<batchIters_1D;++iter) {
				cudaMemcpy( ~_fft_buff, input, batchSize_1D[iter]*idist1D * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
				HANDLE_CUFFT_ERROR( cufftExecZ2Z(cufftPlanForward1D, ~_fft_buff, ~_fft_buff, CUFFT_FORWARD )); 
				_fft_buff.streamSync();
				cudaMemcpy( output, ~_fft_buff, batchSize_1D[iter]*odist1D * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
				input += batchSize_1D[iter]*idist1D;
				output+= batchSize_1D[iter]*odist1D;
			}
		} else if(batchIters_1D==1) {
			HANDLE_CUFFT_ERROR( cufftExecZ2Z(cufftPlanForward1D, ~fouriers, ~buff, CUFFT_FORWARD )); 
		} else {
			cufftDoubleComplex* input = (cufftDoubleComplex*)~fouriers;
			cufftDoubleComplex* output= (cufftDoubleComplex*)~buff;
			for(int iter=0;iter<batchIters_1D;++iter) {
				cudaMemcpyAsync( ~_fft_buff, input, batchSize_1D[iter]*idist1D * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, _fft_buff.getStream());
				HANDLE_CUFFT_ERROR( cufftExecZ2Z(cufftPlanForward1D, ~_fft_buff, ~_fft_buff, CUFFT_FORWARD )); 
				cudaMemcpyAsync( output, ~_fft_buff, batchSize_1D[iter]*odist1D * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, _fft_buff.getStream());
				input += batchSize_1D[iter]*idist1D;
				output+= batchSize_1D[iter]*odist1D;
			}
		}
	}

	inline void forward_2D()
	{
		if(hostBuffer) {
			cufftDoubleReal* input = (cufftDoubleReal*)(reals.h_ptr);
			cufftDoubleComplex* output= (cufftDoubleComplex*)(buff.h_ptr);
			for(int iter=0;iter<batchIters_2D;++iter) {
				cudaMemcpy( ~_fft_buff, input, batchSize_2D[iter]*idist2D * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
				HANDLE_CUFFT_ERROR( cufftExecD2Z(cufftPlanForward2D, (cufftDoubleReal*)~_fft_buff, ~_fft_buff ));
				_fft_buff.streamSync();
				cudaMemcpy( output, ~_fft_buff, batchSize_2D[iter]*odist2D * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
				input += batchSize_2D[iter]*idist2D;
				output+= batchSize_2D[iter]*odist2D;
			}
		} else if(batchIters_2D==1) {
			HANDLE_CUFFT_ERROR( cufftExecD2Z(cufftPlanForward2D, ~reals, ~buff ));
		} else {
			cufftDoubleReal* input = (cufftDoubleReal*)~reals;
			cufftDoubleComplex* output= (cufftDoubleComplex*)~buff;
			for(int iter=0;iter<batchIters_2D;++iter) {
				cudaMemcpyAsync( ~_fft_buff, input, batchSize_2D[iter]*idist2D * sizeof(cufftDoubleReal), cudaMemcpyDeviceToDevice, _fft_buff.getStream());
				HANDLE_CUFFT_ERROR( cufftExecD2Z(cufftPlanForward2D, (cufftDoubleReal*)~_fft_buff, ~_fft_buff ));
				cudaMemcpyAsync( output, ~_fft_buff, batchSize_2D[iter]*odist2D * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, _fft_buff.getStream());
				input += batchSize_2D[iter]*idist2D;
				output+= batchSize_2D[iter]*odist2D;
			}
		}
	}

	inline void forward_3D()
	{
		HANDLE_CUFFT_ERROR( cufftExecD2Z(cufftPlanForward, (cufftDoubleReal*)~reals, (cufftDoubleComplex*)~fouriers) );
	}

	inline void backward_1D()
	{
		if(hostBuffer) {
			cufftDoubleComplex* input = (cufftDoubleComplex*)(buff.h_ptr);
			cufftDoubleComplex* output= (cufftDoubleComplex*)(fouriers.h_ptr);
			for(int iter=0;iter<batchIters_1D;++iter) {
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				cudaMemcpy( ~_fft_buff, input, batchSize_1D[iter]*odist1D * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				HANDLE_CUFFT_ERROR( cufftExecZ2Z(cufftPlanBackward1D, ~_fft_buff, ~_fft_buff, CUFFT_INVERSE ));
				_fft_buff.streamSync();
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				cudaMemcpy( output, ~_fft_buff, batchSize_1D[iter]*idist1D * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				input += batchSize_1D[iter]*odist1D;
				output+= batchSize_1D[iter]*idist1D;
			}
			//printf("backward_1D finished\n");
		} else if(batchIters_1D==1) {
			HANDLE_CUFFT_ERROR( cufftExecZ2Z(cufftPlanBackward1D, ~buff, ~fouriers, CUFFT_INVERSE ));
		} else {
			cufftDoubleComplex* input = (cufftDoubleComplex*)~buff;
			cufftDoubleComplex* output= (cufftDoubleComplex*)~fouriers;
			for(int iter=0;iter<batchIters_1D;++iter) {
				cudaMemcpyAsync( ~_fft_buff, input, batchSize_1D[iter]*odist1D * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, _fft_buff.getStream());
				HANDLE_CUFFT_ERROR( cufftExecZ2Z(cufftPlanBackward1D, ~_fft_buff, ~_fft_buff, CUFFT_INVERSE ));
				cudaMemcpyAsync( output, ~_fft_buff, batchSize_1D[iter]*idist1D * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, _fft_buff.getStream());
				input += batchSize_1D[iter]*odist1D;
				output+= batchSize_1D[iter]*idist1D;
			}
		}
	}

	inline void backward_2D()
	{
		if(hostBuffer) {
			cufftDoubleComplex* input= (cufftDoubleComplex*)(buff.h_ptr);
			cufftDoubleReal* output = (cufftDoubleReal*)(reals.h_ptr);
			for(int iter=0;iter<batchIters_2D;++iter) {
				cudaMemcpy( ~_fft_buff, input, batchSize_2D[iter]*odist2D * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
				HANDLE_CUFFT_ERROR( cufftExecZ2D(cufftPlanBackward2D, ~_fft_buff, (cufftDoubleReal*)~_fft_buff) );
				_fft_buff.streamSync();
				cudaMemcpy( output, ~_fft_buff, batchSize_2D[iter]*idist2D * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
				input += (batchSize_2D[iter]*odist2D);
				output+= (batchSize_2D[iter]*idist2D);
			}
		} else if(batchIters_2D==1) {
			HANDLE_CUFFT_ERROR( cufftExecZ2D(cufftPlanBackward2D, ~buff, ~reals) );
		} else {
			cufftDoubleComplex* input= (cufftDoubleComplex*)~buff;
			cufftDoubleReal* output = (cufftDoubleReal*)~reals;
			for(int iter=0;iter<batchIters_2D;++iter) {
				cudaMemcpyAsync( ~_fft_buff, input, batchSize_2D[iter]*odist2D * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice, _fft_buff.getStream());
				HANDLE_CUFFT_ERROR( cufftExecZ2D(cufftPlanBackward2D, ~_fft_buff, (cufftDoubleReal*)~_fft_buff) );
				cudaMemcpyAsync( output, ~_fft_buff, batchSize_2D[iter]*idist2D * sizeof(cufftDoubleReal), cudaMemcpyDeviceToDevice, _fft_buff.getStream());
				input += (batchSize_2D[iter]*odist2D);
				output+= (batchSize_2D[iter]*idist2D);
			}
		}
	}

	inline void backward_3D()
	{
		HANDLE_CUFFT_ERROR( cufftExecZ2D(cufftPlanBackward, (cufftDoubleComplex*)~fouriers, (cufftDoubleReal*)~reals) );
	}
#endif

	size_t estimate_fft(size_t batch, size_t _idist, size_t _odist, bool isComplex)
	{
		size_t needed(0);

	  size_t biggness;

#ifdef RELION_SINGLE_PRECISION
		if(isComplex) {
			if(direction<=0)
			{
				HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, inembed, istride, _idist, onembed, ostride, _odist, CUFFT_C2C, batch, &biggness));
				needed += biggness;
			}
			if(direction>=0)
			{
				HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, onembed, ostride, _odist, inembed, istride, _idist, CUFFT_C2C, batch, &biggness));
				needed += biggness;
			}
		}
		else {
			if(direction<=0)
			{
				HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, inembed, istride, _idist, onembed, ostride, _odist, CUFFT_R2C, batch, &biggness));
				needed += biggness;
			}
			if(direction>=0)
			{
				HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, onembed, ostride, _odist, inembed, istride, _idist, CUFFT_C2R, batch, &biggness));
				needed += biggness;
			}
		}
#else
		if(isComplex) {
			if(direction<=0)
			{
				HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, inembed, istride, _idist, onembed, ostride, _odist, CUFFT_Z2Z, batch, &biggness));
				needed += biggness;
			}
			if(direction>=0)
			{
				HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, onembed, ostride, _odist, inembed, istride, _idist, CUFFT_Z2Z, batch, &biggness));
				needed += biggness;
			}
		}
		else {
			if(direction<=0)
			{
				HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, inembed, istride, _idist, onembed, ostride, _odist, CUFFT_D2Z, batch, &biggness));
				needed += biggness;
			}
			if(direction>=0)
			{
				HANDLE_CUFFT_ERROR( cufftEstimateMany(dimension, inembed, onembed, ostride, _odist, inembed, istride, _idist, CUFFT_Z2D, batch, &biggness));
				needed += biggness;
			}
		}
#endif
		return needed;
	}

	inline size_t selfAllocSize(bool split)
	{ 
		if(!split) // fourier + real
			return (size_t)odist*sizeof(RFLOAT)*(size_t)2 + (size_t)idist*sizeof(RFLOAT); 
		else // buffer + fourier + real
			return 2*(size_t)odist*sizeof(RFLOAT)*(size_t)2 + (size_t)idist*sizeof(RFLOAT);
	}

	size_t estimate_3D()
	{
		inembed[0] =  zSize;
		inembed[1] =  ySize;
		inembed[2] =  xSize;
		onembed[0] =  zFSize;
		onembed[1] =  yFSize;
		onembed[2] =  xFSize;
		dimension = 3;
		size_t needed = estimate_fft(1,idist,odist,false) + selfAllocSize(false);
		return needed;
	}

	size_t estimate_split(size_t batch1D, size_t batch2D)
	{
		//printf("In estimate_split %d %d\n",batch1D,batch2D);
		size_t needed = selfAllocSize(true);
		// estimate 1D (C2C/Z2Z 1DFFT z-axis)
		inembed1D[0] = inembed[0] = zFSize;
		onembed1D[0] = onembed[0] = zFSize;
		idist1D = odist1D = zFSize;
		dimension = 1;
		needed += estimate_fft(batch1D, idist1D, odist1D, true);
		// estimate 2D (R2C/D2Z 2DFFT xy-axis)
		inembed2D[0] = inembed[0] = ySize;
		inembed2D[1] = inembed[1] = xSize;
		onembed2D[0] = onembed[0] = yFSize;
		onembed2D[1] = onembed[1] = xFSize;
		idist2D = xSize*ySize;
		odist2D = xFSize*yFSize;
		dimension = 2;
		needed += estimate_fft(batch2D, idist2D, odist2D, false);
		// buffer
		size_t size1d = batch1D*odist1D;
		size_t size2d = batch2D*odist2D;
		needed += (size1d>size2d?size1d:size2d)*sizeof(RFLOAT)*2;
		//printf("Exit estimate_split %ld\n",needed);
		return needed;
	}

	// make sure batch size is completely divided by batchIter
	bool splitFFT(size_t avail)
	{
		//printf("--- in splitFFT %d %d %d, avail=%ld\n",xFSize, yFSize, zSize, avail);
		size_t xyFSize = xFSize * yFSize;
		bool success = false;
		size_t batchSpace_2D_old=-1;
		size_t batchSpace_1D_old=-1;
		// 2D FFT : 1-zSize, 1D FFT : 1-xyFSize
		for(batchIters_2D=1;batchIters_2D<=zSize;++batchIters_2D) {
			batchSpace_2D = CEIL((double)zSize / (double)batchIters_2D);
			if(batchSpace_2D==batchSpace_2D_old) continue;
			batchSpace_2D_old = batchSpace_2D;
			batchSpace_1D_old = -1;
			for(batchIters_1D=1;batchIters_1D<=batchIters_2D;++batchIters_1D) {
				batchSpace_1D = CEIL((double)xyFSize / (double)batchIters_1D);
				if(batchSpace_1D==batchSpace_1D_old) continue;
				batchSpace_1D_old = batchSpace_1D;
				size_t needed = estimate_split(batchSpace_1D, batchSpace_2D);
				if(needed < avail) {
					//printf("SPLITTED : 1D %d %d 2D %d %d, needed %ld (avail %ld)\n",batchIters_1D,batchSpace_1D,batchIters_2D,batchSpace_2D,needed, avail);
					success = true; break;
				}
			}
			if(success) break;
		}
		if(!success) {
			batchIters_2D = zSize;
			batchSpace_2D = 1;
			batchIters_1D = zSize + 1;
			batchSpace_1D = CEIL((double)xyFSize / (double)batchIters_1D);
			size_t needed = estimate_split(batchSpace_1D, batchSpace_2D);
			while(batchIters_1D<=xyFSize && needed >= avail) {
				++batchIters_1D;
				batchSpace_1D = CEIL((double)xyFSize / (double)batchIters_1D);
				needed = estimate_split(batchSpace_1D, batchSpace_2D);
			}
			if(batchIters_1D>xyFSize) {
				//printf("--- break splitFFT, return false\n");
				return false;
			}
		}
		batchSize_1D.resize(batchIters_1D);
		batchSize_2D.resize(batchIters_2D);
		batchSize_1D.assign(batchIters_1D,batchSpace_1D); // specify batchIters of batches, each with batchSpace orientations
		batchSize_1D[batchIters_1D-1] = batchSpace_1D - (batchSpace_1D*batchIters_1D - xyFSize); // set last to care for remainder.
		batchSize_2D.assign(batchIters_2D,batchSpace_2D); // specify batchIters of batches, each with batchSpace orientations
		batchSize_2D[batchIters_2D-1] = batchSpace_2D - (batchSpace_2D*batchIters_2D - zSize); // set last to care for remainder.
		//printf("--- exit splitFFT\n");
		return true;
	}

public:
	size_t minReq(size_t x, size_t y, size_t z, int setDirection = 0) 
	{
		int checkDim;
		if(z>1)
			checkDim=3;
		else if(y>1)
			checkDim=2;
		else
			checkDim=1;
		if(checkDim != 3)
			CRITICAL(ERRCUFFTDIM);

		if( !( (setDirection==-1)||(setDirection==0)||(setDirection==1) ) )
		{
			std::cerr << "*ERROR : Setting a cuda transformer direction to non-defined value" << std::endl;
			CRITICAL(ERRCUFFTDIR);
		}

		direction = setDirection;

		clear();

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

		size_t needed1 = estimate_3D();
		size_t needed2 = estimate_split(1, 1);

		return XMIPP_MIN(needed1, needed2);
	}
	int setSize(size_t x, size_t y, size_t z, int setDirection = 0, size_t required_free=0, bool host_buffer=false, bool forceSplit=false)
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
		if(checkDim != 3)
			CRITICAL(ERRCUFFTDIM);

		if( !( (setDirection==-1)||(setDirection==0)||(setDirection==1) ) )
		{
			std::cerr << "*ERROR : Setting a cuda transformer direction to non-defined value" << std::endl;
			CRITICAL(ERRCUFFTDIR);
		}

		direction = setDirection;

		if (x == xSize && y == ySize && z == zSize && planSet)
			return 0;

		clear();

		hostBuffer = host_buffer;

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

		size_t needed, avail, total;
		needed = estimate_3D();
		DEBUG_HANDLE_ERROR(cudaMemGetInfo( &avail, &total ));

		avail -= required_free;

		//printf("Required Free mem = %ld\n",required_free);

		splitted = (needed > avail) || forceSplit;

		if(hostBuffer) {
			std::cout << "WARNING: will use host buffer method to reduce GPU memory usage" << std::endl;
		}

		if(forceSplit) {
			std::cout << "WARNING: force split is set. may not get best performance" << std::endl;
		}

		if(splitted) {
			std::cout << "WARNING: GPU mem may not enough! Try splitting to smaller sizes" << std::endl;
			if(!splitFFT(avail)) {
				std::cout << "WARNING: GPU mem not enough! Use CPU" << std::endl;
				return -1;
			}
		} else {
			if(hostBuffer) {
				CRITICAL("BUG : should not use host bufferring without splitting!!!");
			}
			batchIters_1D = batchIters_2D = 1;
			batchSpace_1D = batchSpace_2D = 0;
		}

		//printf("1D %d %d 2D %d %d\n",batchIters_1D,batchSpace_1D,batchIters_2D,batchSpace_2D);
		reals.setSize(idist);
		fouriers.setSize(odist);
		buff.setSize(odist);
		size_t size1d = (hostBuffer||batchIters_1D>1)? batchSpace_1D*odist1D : 0;
		size_t size2d = (hostBuffer||batchIters_2D>1)? batchSpace_2D*odist2D : 0;
		if(size1d>0 || size2d>0)
			_fft_buff.setSize(size1d>size2d?size1d:size2d);
		allocate();
		//setPlan();
		return 0;
	}

	void allocate() {
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		if(!hostBuffer) {
			//printf("Alloc %ld Byte for reals, %ld Byte for fouriers and buff\n",idist*sizeof(RFLOAT),odist*2*sizeof(RFLOAT));
			reals.device_alloc();
			fouriers.device_alloc();
			//printf("reals %p fouriers %p\n",~reals,~fouriers);
		}
		if(splitted) {
			if(hostBuffer) buff.host_alloc();
			else           buff.device_alloc();
			if(_fft_buff.getSize()>0)
				_fft_buff.device_alloc();
		}
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
	}

	void setPlan() {
		if(planSet) return ;
		if(!splitted) {
#ifdef RELION_SINGLE_PRECISION
			if(direction<=0)
			{
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward,  3, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, 1));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward, fouriers.getStream()));
			}
			if(direction>=0)
			{
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward, 3, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2R, 1));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward, reals.getStream()));
			}
#else
			if(direction<=0)
			{
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward,  3, inembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, 1));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward, fouriers.getStream()));
			}
			if(direction>=0)
			{
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward, 3, inembed, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, 1));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward, reals.getStream()));
			}
#endif
		} else {
#ifdef RELION_SINGLE_PRECISION
			if(direction<=0)
			{
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward2D,  2, inembed2D, inembed2D, istride, idist2D, onembed2D, ostride, odist2D, CUFFT_R2C, batchSpace_2D));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward2D, fouriers.getStream()));
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward1D,  1, inembed1D, inembed1D, istride, idist1D, onembed1D, ostride, odist1D, CUFFT_C2C, batchSpace_1D));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward1D, fouriers.getStream()));
			}
			if(direction>=0)
			{
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward1D, 1, inembed1D, onembed1D, ostride, odist1D, inembed1D, istride, idist1D, CUFFT_C2C, batchSpace_1D));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward1D, reals.getStream()));
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward2D, 2, inembed2D, onembed2D, ostride, odist2D, inembed2D, istride, idist2D, CUFFT_C2R, batchSpace_2D));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward2D, reals.getStream()));
			}
#else
			if(direction<=0)
			{
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward2D,  2, inembed2D, inembed2D, istride, idist2D, onembed2D, ostride, odist2D, CUFFT_D2Z, batchSpace_2D));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward2D, fouriers.getStream()));
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanForward1D,  1, inembed1D, inembed1D, istride, idist1D, onembed1D, ostride, odist1D, CUFFT_Z2Z, batchSpace_1D));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanForward1D, fouriers.getStream()));
			}
			if(direction>=0)
			{
				//printf("Planning %d %d, %d %d, %d, %d, %d\n",inembed2D[0],inembed2D[1],onembed2D[0],onembed2D[1], idist2D, odist2D,batchSpace_2D);
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward1D, 1, inembed1D, onembed1D, ostride, odist1D, inembed1D, istride, idist1D, CUFFT_Z2Z, batchSpace_1D));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward1D, reals.getStream()));
				HANDLE_CUFFT_ERROR( cufftPlanMany(&cufftPlanBackward2D, 2, inembed2D, onembed2D, ostride, odist2D, inembed2D, istride, idist2D, CUFFT_Z2D, batchSpace_2D));
				HANDLE_CUFFT_ERROR( cufftSetStream(cufftPlanBackward2D, reals.getStream()));
			}
#endif
		}
		planSet = true;
	}

	void forward()
	{
		if(!splitted) {
			forward_3D();
		} else {
			// 2D FFT on xy-plane, results are in buff
			forward_2D();
			HANDLE_ERROR(cudaGetLastError());
			// transpose to (z,x,y)
			run_cuda_transpose(fouriers, buff, xFSize*yFSize, zFSize, hostBuffer);
			HANDLE_ERROR(cudaGetLastError());
			//buff.streamSync();
			// 1D FFT on z-axis, input are in fouriers
			forward_1D();
			HANDLE_ERROR(cudaGetLastError());
			// transpose back to (x,y,z)
			run_cuda_transpose(fouriers, buff, zFSize, xFSize*yFSize, hostBuffer);
			HANDLE_ERROR(cudaGetLastError());
		}
	}

	void backward()
	{
		if(!splitted) {
			backward_3D();
		} else {
			// input are in fouriers, transpose to (z,x,y)
			run_cuda_transpose(buff, fouriers, xFSize*yFSize, zFSize, hostBuffer);
			//buff.streamSync();
			// 1D invFFT on z-axis, results are in fouriers
			backward_1D();
			// transpose back to (x,y,z)
			run_cuda_transpose(buff, fouriers, zFSize, xFSize*yFSize, hostBuffer);
			// 2D invFFT on xy-plane, results are in reals
			backward_2D();
			HANDLE_ERROR(cudaGetLastError());
		}
	}

	void clearPlan() {
		if(planSet) {
			if(direction<=0 && !splitted)
				HANDLE_CUFFT_ERROR(cufftDestroy(cufftPlanForward));
			if(direction>=0 && !splitted)
				HANDLE_CUFFT_ERROR(cufftDestroy(cufftPlanBackward));
			if(direction<=0 && splitted)
				HANDLE_CUFFT_ERROR(cufftDestroy(cufftPlanForward1D));
			if(direction>=0 && splitted)
				HANDLE_CUFFT_ERROR(cufftDestroy(cufftPlanBackward1D));
			if(direction<=0 && splitted)
				HANDLE_CUFFT_ERROR(cufftDestroy(cufftPlanForward2D));
			if(direction>=0 && splitted)
				HANDLE_CUFFT_ERROR(cufftDestroy(cufftPlanBackward2D));
		}
		planSet = false;
	}

	void free() {
		reals.streamSync();
		HANDLE_ERROR(cudaGetLastError());
		//printf("before clear : reals %p fouriers %p\n",~reals,~fouriers);
		fouriers.free_device_if_set();
		HANDLE_ERROR(cudaGetLastError());
		reals.free_device_if_set();
		HANDLE_ERROR(cudaGetLastError());
		buff.free_if_set();
		HANDLE_ERROR(cudaGetLastError());
		_fft_buff.free_device_if_set();
		HANDLE_ERROR(cudaGetLastError());
	}

	void clear()
	{
		if(planSet)
		{
			HANDLE_ERROR(cudaGetLastError());
			free();
			HANDLE_ERROR(cudaGetLastError());
			clearPlan();
			HANDLE_ERROR(cudaGetLastError());
		}
	}

	~BuffCudaFFT3D()
	{clear();}
};

#ifdef PARALLEL_RECONSTRUCT_NEW
#define USE_SIMPLE_PLAN
#define DEBUG_PARALLEL_GPU

#define HOST_ALLOC(x,T,size) x=(T*)malloc(size*sizeof(T))
#define HOST_FREE(x) do { free(x); x=NULL; } while(0)

#define CUDA_ALLOC(x,T,size) cudaMalloc((void**)&(x),sizeof(T)*size)
#define CUDA_FREE(x) do { cudaFree(x); x=NULL; } while(0)

#define USE_CUDA_AWARE

// using distributed multi-GPU by MPI to accelerate FFT
template<typename REAL_T, typename COMPLEX_T, bool isDouble, bool allocHost>
class DistributedCudaFFT3D_TP
{
	private:
	bool cleared;
	bool planSet;
	bool realCopied;
	bool fourierCopied;

	bool _2DPlanSet;
	bool _1DPlanSet;

	MpiNode* node;
	MPI_Comm comm;
	int rank;
	int size;

	// location info
	int *dyLoc;
	int *dzLoc;
	int *dyFLoc;
	int *dzFLoc;
	// HOST BUFFS
	COMPLEX_T* _1DTrans_buff;

	public:
#ifdef INNER_TIMING
	// timing
	bool printTimes;
#endif
	// global vars (Host)
	REAL_T* gReals;
	COMPLEX_T* gFouriers;
	// Host Vars
	REAL_T* lReals;
	COMPLEX_T* lFouriers;
	COMPLEX_T* buff;
	COMPLEX_T* h_buff;
	COMPLEX_T* d_buff;
	// Sizes
	int xSize , ySize , zSize ;
	int xFSize, yFSize, zFSize;
	int lyStart, lzStart;
	int lyFStart, lzFStart;
	int lySize , lzSize;
	int lyFSize, lzFSize;
	// CUDA vars
	cudaStream_t stream;
	CudaFFT_TP<REAL_T, COMPLEX_T, isDouble, allocHost,false,false>* p2DFFT_trans;
	CudaFFT_TP<COMPLEX_T, COMPLEX_T, isDouble, allocHost, true,false>* p1DFFT_trans;

	void clear()
	{
		gReals = NULL;
		gFouriers = NULL;
		xSize = ySize = zSize = 0;
		xFSize= yFSize= zFSize= 0;
		if(!cleared) {
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			HOST_FREE(lReals);
			HOST_FREE(lFouriers);
			HOST_FREE(buff);
			HOST_FREE(h_buff);
			HOST_FREE(_1DTrans_buff);
			CUDA_FREE(d_buff);
			p2DFFT_trans->clear();
			delete p2DFFT_trans;
			_2DPlanSet = false;
			p1DFFT_trans->clear();
			delete p1DFFT_trans;
			_1DPlanSet = false;
			stream = 0;
			cleared = true;
		}
	}

	void Barrier() {
		MPI_Barrier(comm);
	}

	DistributedCudaFFT3D_TP(MpiNode* _node, MPI_Comm _comm, cudaStream_t _stream):
		node(_node),
		comm(_comm),
		cleared(true), 
		gReals(NULL),
		gFouriers(NULL),
		lReals(NULL),
		lFouriers(NULL),
		buff(NULL),
		p2DFFT_trans(NULL),
		p1DFFT_trans(NULL),
		_2DPlanSet(false),
		_1DPlanSet(false),
#ifdef INNER_TIMING
		printTimes(false),
#endif
		xSize(0) , ySize(0) , zSize(0),
		xFSize(0), yFSize(0), zFSize(0),
		stream(_stream)
	{
		MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
		//printf("Debug : Comm %x, Rank %d, Size %d\n",comm,rank,size);
		dyLoc = new int[size];
		dzLoc = new int[size];
		dyFLoc = new int[size];
		dzFLoc = new int[size];
	}

	~DistributedCudaFFT3D_TP()
	{
		clear();
		delete[] dyLoc;
		delete[] dzLoc;
		delete[] dyFLoc;
		delete[] dzFLoc;
	}

	int setSize(size_t x, size_t y, size_t z)
	{
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: Begin setSize %d %d %d\n",x,y,z);
#endif
		if(xSize==x && ySize==y && zSize==z)
			return 0;
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: Not skipped. Prepare new Plans.\n");		
#endif
		clear();
		cleared=false;
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: Cleared\n");
#endif
		xSize = x; xFSize = x/2+1; 
		ySize = y; yFSize = y;
		zSize = z; zFSize = z;

		//assert(zSize==zFSize);

		// local data size, only split by Z-axis on 2D, Y-axis on 1D
		int zrest = zFSize % size;
		int yrest = yFSize % size;
		bool isZRest = rank < zrest;
		bool isYRest = rank < yrest;
		lySize = ySize / size + (isYRest ? 1 : 0); 
		lyFSize = yFSize / size + (isYRest ? 1 : 0);
		lzSize = zSize / size + (isZRest ? 1 : 0);
		lzFSize = zFSize / size + (isZRest ? 1 : 0);
		// location of local data in global
		if(isZRest) {
			lzStart = rank * lzSize;
			lzFStart= rank * lzFSize;
		} else {
			lzStart = rank * lzSize  + zrest;
			lzFStart= rank * lzFSize + zrest;
		}
		if(isYRest) {
			lyStart = rank * lySize;
			lyFStart= rank * lyFSize;
		} else {
			lyStart = rank * lySize  + yrest;
			lyFStart= rank * lyFSize + yrest;
		}
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: rank %d real size lysize %d lzsize %d, lystart %d, lzstart %d\n",rank,lySize,lzSize,lyStart,lzStart);
		printf("INFO :: DistributedCudaFFT3D :: rank %d fourier size lyFSize %d lzFSize %d, lystart %d, lzstart %d\n",rank,lyFSize,lzFSize,lyFStart,lzFStart);
#endif
		int rSize = xSize * ySize  * lzSize;
		int fSize = xFSize* yFSize * lzFSize;
		int bsize = xFSize* zFSize * lyFSize;
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: rank %d alloc sizes, real %d, fourier %d x 2, buff %d x 2, total %lf MB\n",rank,rSize,fSize,bsize,(double)(rSize+fSize*2+bsize*2)/(double)1e6);
#endif
		// allocate local work space for host
		HOST_ALLOC(lReals,REAL_T,rSize);
		HOST_ALLOC(lFouriers,COMPLEX_T,fSize);
		HOST_ALLOC(_1DTrans_buff,COMPLEX_T,bsize);
#ifndef USE_CUDA_AWARE
		HOST_ALLOC(buff,COMPLEX_T,xFSize* yFSize * zFSize);
#else
		HOST_ALLOC(buff,COMPLEX_T,fSize);
#endif
		HOST_ALLOC(h_buff,COMPLEX_T,xFSize* yFSize * zFSize);
		CUDA_ALLOC(d_buff,COMPLEX_T,xFSize* yFSize * zFSize);
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: HOST ALLOCATED\n");
#endif
		// set up local transformer
		p2DFFT_trans = new CudaFFT_TP<REAL_T, COMPLEX_T, isDouble, allocHost, false, false/*don't use custom allocator*/>(stream,NULL,2);
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: local 2D transformer created, ptr=%x\n",p2DFFT_trans);
#endif
		p1DFFT_trans = new CudaFFT_TP<COMPLEX_T, COMPLEX_T, isDouble, allocHost, true, false/*don't use custom allocator*/>(stream,NULL,1);
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: local 1D transformer created, ptr=%x\n",p1DFFT_trans);
		printf("INFO :: DistributedCudaFFT3D :: using simple plan creation\n");
		printf("INFO :: DistributedCudaFFT3D :: rank %d creating 2D Plan with %d %d %d x %d\n",rank,xSize,ySize,1,lzSize);
#endif
		if(p2DFFT_trans->setSize(xSize,ySize,1,lzSize)!=0) 
			REPORT_ERROR("DistributedCudaFFT3D_TP : p2DFFT_trans->setSize failed! Memory isn't enough!");
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: 2d plan created\n");
#endif
		_2DPlanSet = true;
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: rank %d creating 1D Plan with %d %d %d x %d\n",rank,zSize,1,1,lySize*xSize);
#endif
		if(p1DFFT_trans->setSize(zFSize,1,1,lyFSize*xFSize)!=0) 
			REPORT_ERROR("DistributedCudaFFT3D_TP : p1DFFT_trans->setSize failed! Memory isn't enough!");
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: 1d plan created\n");
#endif
		_1DPlanSet = true;
#ifdef DEBUG_CUDA_FFT
		printf("INFO :: DistributedCudaFFT3D :: rank %d Plan is set. Exit\n",rank);
#endif
		return 0;
	}
	#define MPI_DISTRIBUTE_LOCTAG 111
	#define MPI_DISTRIBUTE_DATATAG 112
	#define MPI_MERGE_DATATAG 113
	// Prerequest : setSize called successfully before calling this function
	// distribute location of each rank's data
	void distribute_location()
	{
		MPI_Status status;
		int sbuff[8] = {lyStart,lySize,lyFStart,lyFSize,lzStart,lzSize,lzFStart,lzFSize};
		int rbuff[8] = {0};
		for(int n=0;n<size;++n) {
			if(n!=rank) {
				node->relion_MPI_ISend(sbuff,8,MPI_INT,n,MPI_DISTRIBUTE_LOCTAG,comm);
#ifdef DEBUG_DISTRIBUTE
				printf("Rank %d send data {%d %d %d %d %d %d %d %d} to rank %d\n",rank,
					sbuff[0],sbuff[1],sbuff[2],sbuff[3],sbuff[4],sbuff[5],sbuff[6],sbuff[7],n);
#endif
			} 
		}
		for(int n=0;n<size;++n) {
			if(n!=rank) {
				node->relion_MPI_Recv(rbuff,8,MPI_INT,n,MPI_DISTRIBUTE_LOCTAG,comm,status);
#ifdef DEBUG_DISTRIBUTE
				printf("Rank %d recv data {%d %d %d %d %d %d %d %d} from rank %d\n",rank,
					rbuff[0],rbuff[1],rbuff[2],rbuff[3],rbuff[4],rbuff[5],rbuff[6],rbuff[7],n);
#endif
				dyLoc[2*n] = rbuff[0];
				dyLoc[2*n+1] = rbuff[1];
				dyFLoc[2*n] = rbuff[2];
				dyFLoc[2*n+1] = rbuff[3];
				dzLoc[2*n] = rbuff[4];
				dzLoc[2*n+1] = rbuff[5];
				dzFLoc[2*n] = rbuff[6];
				dzFLoc[2*n+1] = rbuff[7];
			}
		}
		node->relion_MPI_WaitAll(status);
#ifdef DEBUG_DISTRIBUTE
		for(int n=0;n<size;++n) {
			MPI_Barrier(node->groupC);
			if(n==rank) {
				printf("Rank %d DISTRIBUTE_LOCATION :: \n",rank);
				for(int n=0;n<size;++n) {
					printf("\t%d locations : \n",n);
					printf("\t\tdyLoc %d %d, dyFLoc %d %d\n",dyLoc[2*n],dyLoc[2*n+1],dyFLoc[2*n],dyFLoc[2*n+1]);
					printf("\t\tdzLoc %d %d, dzFLoc %d %d\n",dzLoc[2*n],dzLoc[2*n+1],dzFLoc[2*n],dzFLoc[2*n+1]);
				}
			}
		}
#endif
	}

	// distribute data : DISTRIBUTE_REAL / DISTRIBUTE_FOURIER 
	void distribute(int mode)
	{
		// sizes already splitted by z-axis (and y-axis for 1D-FFT)
		// just distribute them
		distribute_location();
		MPI_Status status;
		switch(mode) {
			case DISTRIBUTE_REAL:
				// distribute gReals to lReals
				if(rank==0) {
					for(int n=1;n<size;++n) {
						size_t localSize = xSize * ySize * dzLoc[2*n+1];
						REAL_T* ptr = gReals + xSize * ySize * dzLoc[2*n];
						node->relion_MPI_ISend(ptr,localSize,MY_MPI_DOUBLE,n,MPI_DISTRIBUTE_DATATAG,comm);
					}
					size_t localSize = xSize * ySize * lzSize;
					memcpy(lReals,gReals,sizeof(RFLOAT)*localSize);
					node->relion_MPI_WaitAll(status);
				} else {
					size_t localSize = xSize * ySize * lzSize;
					node->relion_MPI_Recv(lReals,localSize,MY_MPI_DOUBLE,0,MPI_DISTRIBUTE_DATATAG,comm,status);
				}
				break;
			case DISTRIBUTE_FOURIER:
				// distribute gFouriers to lFouriers
				if(rank==0) {
					for(int n=1;n<size;++n) {
						size_t localSize = xFSize * yFSize * dzFLoc[2*n+1];
						COMPLEX_T* ptr = gFouriers + xFSize * yFSize * dzFLoc[2*n];
						node->relion_MPI_ISend(ptr,localSize,MY_MPI_COMPLEX,n,MPI_DISTRIBUTE_DATATAG,comm);
					}
					size_t localSize = xFSize * yFSize * lzFSize;
					memcpy(lFouriers,gFouriers,sizeof(COMPLEX_T)*localSize);
					node->relion_MPI_WaitAll(status);
				} else {
					size_t localSize = xFSize * yFSize * lzFSize;
					node->relion_MPI_Recv(lFouriers,localSize,MY_MPI_COMPLEX,0,MPI_DISTRIBUTE_DATATAG,comm,status);
				}
				break;
		}
	}

	void merge(int mode)
	{
		MPI_Status status;
		switch(mode) {
			case MERGE_REAL:
				// merge lReals to gReals
				if(rank==0) {
					for(int n=1;n<size;++n) {
						size_t localSize = xSize * ySize * dzLoc[2*n+1];
						REAL_T* ptr = gReals + xSize * ySize * dzLoc[2*n];
						node->relion_MPI_Recv(ptr,localSize,MY_MPI_DOUBLE,n,MPI_MERGE_DATATAG,comm,status);
					}
					size_t localSize = xSize * ySize * lzSize;
					memcpy(gReals,lReals,sizeof(REAL_T)*localSize);
				} else {
					size_t localSize = xSize * ySize * lzSize;
					node->relion_MPI_Send(lReals,localSize,MY_MPI_DOUBLE,0,MPI_MERGE_DATATAG,comm);
				}
				break;
			case MERGE_FOURIER:
				// merge lFouriers to gFouriers
				if(rank==0) {
					for(int n=1;n<size;++n) {
						size_t localSize = xFSize * yFSize * dzFLoc[2*n+1];
						COMPLEX_T* ptr = gFouriers + xFSize * yFSize * dzFLoc[2*n];
						node->relion_MPI_Recv(ptr,localSize,MY_MPI_COMPLEX,n,MPI_MERGE_DATATAG,comm,status);
					}
					size_t localSize = xFSize * yFSize * lzFSize;
					memcpy(gFouriers,lFouriers,sizeof(COMPLEX_T)*localSize);
				} else {
					size_t localSize = xFSize * yFSize * lzFSize;
					node->relion_MPI_Send(lFouriers,localSize,MY_MPI_COMPLEX,0,MPI_MERGE_DATATAG,comm);
				}
				break;
			default:
				printf("** Error : MERGE : not supportted mode %d\n",mode);
				break;
		}
	}
	// dptr : device pointer
	void mergeThis_ysplit(double* dptr)
	{
		MPI_Status status;
		if(rank==0) {
			for(int n=1;n<size;++n) {
				size_t localSize = xFSize * zFSize * dyFLoc[2*n+1];
				double* ptr = dptr + xFSize * zFSize * dyFLoc[2*n];
				node->relion_MPI_Recv(ptr,localSize, MPI_DOUBLE,n,MPI_MERGE_DATATAG,comm,status);
			}
		} else {
			size_t localSize = xFSize * zFSize * lyFSize;
			double* ptr = dptr + xFSize * zFSize * lyFStart;
			node->relion_MPI_Send(ptr,localSize,MPI_DOUBLE,0,MPI_MERGE_DATATAG,comm);
		}
	}

	CudaGlobalPtr<RFLOAT, false>& getDistributedRealReference() { return p2DFFT_trans->reals; }
	CudaGlobalPtr<COMPLEX_T, false>& getDistributedFourierReference() { return p1DFFT_trans->fouriers; }

	// dst = transpose(src), x by y => y by x
	void transpose(COMPLEX_T* dst, COMPLEX_T *src, int x, int y);
	void exchange(COMPLEX_T* _zsplit_buff, COMPLEX_T* _ysplit_buff);
	void exchange_back(COMPLEX_T* _ysplit_buff, COMPLEX_T* _zsplit_buff);

	void transpose_gpu(COMPLEX_T* dst, COMPLEX_T *src, int x, int y);
	void exchange_gpu(COMPLEX_T* _zsplit_buff, COMPLEX_T* _ysplit_buff);
	void exchange_back_gpu(COMPLEX_T* _ysplit_buff, COMPLEX_T* _zsplit_buff);

	void transpose_gpu(double* dst, double *src, int x, int y);

	// Precondition : 
	//     1. data and locs distributed 
	//     2. data already sent to gpu 
	//     3. plan is set and no batch
	// Results : transposed and with y splitted : (z,x,y_split)
	void forward_notrans();

	// Precondition : Already called distribute() or manually distributed data and locations
	void forward()
	{
		// first 2D-FFT across x-y
		if(!_2DPlanSet) {
			if(p2DFFT_trans->setSize(xSize,ySize,1,lzSize)!=0) 
				REPORT_ERROR("DistributedCudaFFT3D_TP : forward : p2DFFT_trans->setSize failed! Memory isn't enough!");
		}
		REAL_T* ptr = lReals;
		COMPLEX_T* fptr = lFouriers;
		for(int iter=0;iter<p2DFFT_trans->batchIters;++iter) {
			size_t iterSize = p2DFFT_trans->idist * p2DFFT_trans->batchSize[iter];
			size_t iterFSize= p2DFFT_trans->odist * p2DFFT_trans->batchSize[iter];
			memcpy(p2DFFT_trans->reals.h_ptr, ptr, iterSize*sizeof(REAL_T));
			p2DFFT_trans->reals.cp_to_device();
			p2DFFT_trans->forward();
			p2DFFT_trans->fouriers.cp_to_host();
			p2DFFT_trans->fouriers.streamSync();
			memcpy(fptr, p2DFFT_trans->fouriers.h_ptr, iterFSize*sizeof(COMPLEX_T));
			ptr += iterSize;
			fptr+= iterFSize;
		}
		if(!_2DPlanSet) {
			p2DFFT_trans->clear();
		}
		// transpose : (x,y,z) => (z,x,y)
		transpose(buff,lFouriers,xFSize*yFSize,lzFSize);
		// exchange data between all ranks : (z_split,x,y) => (z,x,y_split)
		exchange(buff, _1DTrans_buff);
		// 1D-FFT across z
		if(!_1DPlanSet) {
			if(p1DFFT_trans->setSize(zFSize,1,1,xFSize*lyFSize)!=0) 
				REPORT_ERROR("DistributedCudaFFT3D_TP : forward : p2DFFT_trans->setSize failed! Memory isn't enough!");
		}
		fptr = _1DTrans_buff;
		for(int iter=0;iter<p1DFFT_trans->batchIters;++iter) {
			size_t iterSize = p1DFFT_trans->idist * p1DFFT_trans->batchSize[iter];
			memcpy(p1DFFT_trans->reals.h_ptr, fptr, iterSize*sizeof(COMPLEX_T));
			p1DFFT_trans->reals.cp_to_device();
			p1DFFT_trans->forward();
			p1DFFT_trans->fouriers.cp_to_host();
			p1DFFT_trans->fouriers.streamSync();
			memcpy(fptr, p1DFFT_trans->fouriers.h_ptr, iterSize*sizeof(COMPLEX_T));
			fptr += iterSize;
		}
		if(!_1DPlanSet) {
			p1DFFT_trans->clear();
		}
		// exchange data back : (z,x,y_split) => (z_split,x,y)
		exchange_back(_1DTrans_buff, buff);
		// transpose back : (z,x,y) => (x,y,z)
		transpose(lFouriers,buff,lzFSize,xFSize*yFSize);
	}

	// Precondition : 
	//     1. data and locs distributed 
	//     2. data already sent to gpu and transposed with y splitted 
	//     3. plan is set and no batch
	void backward_notrans()
	{
#ifdef INNER_TIMING
		double start, end;
		start = omp_get_wtime();
#endif
		// 1D-invFFT across z
		p1DFFT_trans->backward();
		p1DFFT_trans->reals.streamSync();
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : 1DFFT time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		// exchange data back : (z,x,y_split) => (z_split,x,y)
		exchange_back_gpu(p1DFFT_trans->reals.d_ptr, d_buff);
#ifdef INNER_TIMING
		if(printTimes)
        printf("\texchange back gpu : waiting for results\n");
		p1DFFT_trans->reals.streamSync();
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : exchange back time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		// transpose back : (z,x,y) => (x,y,z)
		transpose_gpu(p2DFFT_trans->fouriers.d_ptr,d_buff,lzFSize,xFSize*yFSize);
		p2DFFT_trans->fouriers.streamSync();
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : transpose 2 time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		// then 2D-invFFT across x-y
		p2DFFT_trans->backward();
#ifdef INNER_TIMING
		p2DFFT_trans->reals.streamSync();
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : 2D FFT time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
	}

	void real_to_host()
	{
		p2DFFT_trans->reals.cp_to_host(lReals,xSize*ySize*lzSize);
	}

	void real_to_device()
	{
		cudaMemcpyAsync(p2DFFT_trans->reals.d_ptr, lReals, xSize*ySize*lzSize*sizeof(REAL_T),  cudaMemcpyHostToDevice,stream);
	}

	void fourier_to_host()
	{
		p1DFFT_trans->fouriers.cp_to_host();
	}

	// Precondition : Already called distribute() or manually distributed data and locations
	void backward()
	{
#ifdef INNER_TIMING
		double start, end;
		start = omp_get_wtime();
#endif
		// transpose : (x,y,z) => (z,x,y)
		transpose(buff,lFouriers,xFSize*yFSize,lzFSize);
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : transpose time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		// exchange data : (z_split,x,y) => (z,x,y_split)
		exchange(buff, _1DTrans_buff);
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : exchange time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		// 1D-invFFT across z
		if(!_1DPlanSet) {
			if(p1DFFT_trans->setSize(zFSize,1,1,xFSize*lyFSize)!=0) 
				REPORT_ERROR("DistributedCudaFFT3D_TP : backward : p2DFFT_trans->setSize failed! Memory isn't enough!");
		}
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : setSize time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		COMPLEX_T* fptr = _1DTrans_buff;
		for(int iter=0;iter<p1DFFT_trans->batchIters;++iter) {
			size_t iterSize = p1DFFT_trans->idist * p1DFFT_trans->batchSize[iter];
			memcpy(p1DFFT_trans->fouriers.h_ptr, fptr, iterSize*sizeof(COMPLEX_T));
			p1DFFT_trans->fouriers.cp_to_device();
			p1DFFT_trans->backward();
			p1DFFT_trans->reals.cp_to_host();
			p1DFFT_trans->reals.streamSync();
			memcpy(fptr, p1DFFT_trans->reals.h_ptr, iterSize*sizeof(COMPLEX_T));
			fptr += iterSize;
		}
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : 1DFFT time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		if(!_1DPlanSet) {
			p1DFFT_trans->clear();
		}
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : clear time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		// exchange data back : (z,x,y_split) => (z_split,x,y)
		exchange_back(_1DTrans_buff, buff);
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : exchange back time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		// transpose back : (z,x,y) => (x,y,z)
		transpose(lFouriers,buff,lzFSize,xFSize*yFSize);
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : transpose 2 time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		// then 2D-invFFT across x-y
		if(!_2DPlanSet) {
			if(p2DFFT_trans->setSize(xSize,ySize,1,lzSize)!=0) 
				REPORT_ERROR("DistributedCudaFFT3D_TP : backward : p2DFFT_trans->setSize failed! Memory isn't enough!");
		}
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : 2D plan set time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		REAL_T* ptr = lReals;
		fptr = lFouriers;
		for(int iter=0;iter<p2DFFT_trans->batchIters;++iter) {
			size_t iterSize = p2DFFT_trans->idist * p2DFFT_trans->batchSize[iter];
			size_t iterFSize= p2DFFT_trans->odist * p2DFFT_trans->batchSize[iter];
			memcpy(p2DFFT_trans->fouriers.h_ptr, fptr, iterFSize*sizeof(COMPLEX_T));
			p2DFFT_trans->fouriers.cp_to_device();
			p2DFFT_trans->backward();
			p2DFFT_trans->reals.cp_to_host();
			p2DFFT_trans->reals.streamSync();
			memcpy(ptr, p2DFFT_trans->reals.h_ptr, iterSize*sizeof(REAL_T));
			ptr += iterSize;
			fptr+= iterFSize;
		}
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : 2D FFT time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
		if(!_2DPlanSet) {
			p2DFFT_trans->clear();
		}
#ifdef INNER_TIMING
		end = omp_get_wtime();
		if(printTimes)
        printf("\tRank %d : 2D clear plan time %lf\n",rank,end-start);
		start = omp_get_wtime();
#endif
	}

	void streamSync()
	{
		cudaStreamSynchronize(stream);
	}

};
#ifdef RELION_SINGLE_PRECISION
	typedef DistributedCudaFFT3D_TP<cufftReal,cufftComplex,false,true> DistributedCudaFFT;
#else
	typedef DistributedCudaFFT3D_TP<cufftDoubleReal,cufftDoubleComplex,true,true> DistributedCudaFFT;
#endif
#endif

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
