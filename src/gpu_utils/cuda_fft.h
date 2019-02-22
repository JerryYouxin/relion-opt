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
		if(planSet) return ;
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
