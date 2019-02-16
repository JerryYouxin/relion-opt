#include "src/gpu_utils/cuda_kernels/fft_helper.cuh"
#include "src/gpu_utils/cuda_device_utils.cuh"
#include "src/gpu_utils/cuda_utils_cub.cuh"

__global__ void ScaleComplexPointwise_kernel(__COMPLEX_T * a, int size, int divScale)
{
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (global_index >= size)
	{
		return;
	}
	a[global_index].x /= (RFLOAT)divScale;
	a[global_index].y /= (RFLOAT)divScale;
}

__global__ void ScaleReal_kernel(RFLOAT * a, int size, RFLOAT divScale)
{
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (global_index >= size)
	{
		return;
	}
	a[global_index] /= divScale;
}

__global__ void fft_cuda_kernel_centerFFT_2D(RFLOAT *img_in,
    int image_size,
    int xdim,
    int ydim,
    int xshift,
    int yshift)
{

    __shared__ RFLOAT buffer[CFTT_BLOCK_SIZE];
    int tid = threadIdx.x;
    int pixel = threadIdx.x + blockIdx.x*CFTT_BLOCK_SIZE;
    long int image_offset = image_size*blockIdx.y;
    //	int pixel_pass_num = ceilfracf(image_size, CFTT_BLOCK_SIZE);

    //	for (int pass = 0; pass < pixel_pass_num; pass++, pixel+=CFTT_BLOCK_SIZE)
    //	{
    if(pixel<(image_size/2))
    {
        int y = floorf((RFLOAT)pixel/(RFLOAT)xdim);
        int x = pixel % xdim;				// also = pixel - y*xdim, but this depends on y having been calculated, i.e. serial evaluation

        int yp = y + yshift;
        if (yp < 0)
        yp += ydim;
        else if (yp >= ydim)
        yp -= ydim;

        int xp = x + xshift;
        if (xp < 0)
        xp += xdim;
        else if (xp >= xdim)
        xp -= xdim;

        int n_pixel = yp*xdim + xp;

        buffer[tid]                    = img_in[image_offset + n_pixel];
        img_in[image_offset + n_pixel] = img_in[image_offset + pixel];
        img_in[image_offset + pixel]   = buffer[tid];
    }
    //	}
}

__global__ void fft_cuda_kernel_centerFFT_3D(RFLOAT *img_in,
    int image_size,
    int xdim,
    int ydim,
    int zdim,
    int xshift,
    int yshift,
     int zshift)
{

    __shared__ RFLOAT buffer[CFTT_BLOCK_SIZE];
    int tid = threadIdx.x;
    int pixel = threadIdx.x + blockIdx.x*CFTT_BLOCK_SIZE;
    long int image_offset = image_size*blockIdx.y;

    int xydim = xdim*ydim;

    if(pixel<(image_size/2))
    {
        //int z = floorf((RFLOAT)pixel/(RFLOAT)(xydim));
        int z = pixel / xydim;
        int xy = pixel % xydim;
        //int y = floorf((RFLOAT)xy/(RFLOAT)xdim);
        int y = xy / xdim;
        int x = xy % xdim;


        int yp = y + yshift;
        if (yp < 0)
            yp += ydim;
        else if (yp >= ydim)
            yp -= ydim;

        int xp = x + xshift;
        if (xp < 0)
            xp += xdim;
        else if (xp >= xdim)
            xp -= xdim;

        int zp = z + zshift;
        if (zp < 0)
            zp += zdim;
        else if (zp >= zdim)
            zp -= zdim;

        int n_pixel = zp*xydim + yp*xdim + xp;

        buffer[tid]                    = img_in[image_offset + n_pixel];
        img_in[image_offset + n_pixel] = img_in[image_offset + pixel];
        img_in[image_offset + pixel]   = buffer[tid];
    }
}

void fft_runCenterFFT( CudaGlobalPtr< RFLOAT , 0 > &img_in,
    int xSize,
    int ySize,
    int zSize,
    bool forward,
    int batchSize)
{
    //	CudaGlobalPtr<RFLOAT >  img_aux(img_in.h_ptr, img_in.size, allocator);   // temporary holder
    //	img_aux.device_alloc();

    if(zSize>1)
    {
        int xshift = (xSize / 2);
        int yshift = (ySize / 2);
        int zshift = (ySize / 2);

        if (!forward)
        {
            xshift = -xshift;
            yshift = -yshift;
            zshift = -zshift;
        }

        dim3 blocks(ceilf((float)((xSize*ySize*zSize)/(float)(2*CFTT_BLOCK_SIZE))),batchSize);
        fft_cuda_kernel_centerFFT_3D<<<blocks,CFTT_BLOCK_SIZE, 0, img_in.getStream()>>>(
            ~img_in,
            xSize*ySize*zSize,
            xSize,
            ySize,
            zSize,
            xshift,
            yshift,
            zshift);
        LAUNCH_HANDLE_ERROR(cudaGetLastError());

        //	HANDLE_ERROR(cudaStreamSynchronize(0));
        //	img_aux.cp_on_device(img_in.d_ptr); //update input image with centered kernel-output.
    }
    else
    {
        int xshift = (xSize / 2);
        int yshift = (ySize / 2);

        if (!forward)
        {
            xshift = -xshift;
            yshift = -yshift;
        }

        dim3 blocks(ceilf((float)((xSize*ySize)/(float)(2*CFTT_BLOCK_SIZE))),batchSize);
        fft_cuda_kernel_centerFFT_2D<<<blocks,CFTT_BLOCK_SIZE, 0, img_in.getStream()>>>(
            ~img_in,
            xSize*ySize,
            xSize,
            ySize,
            xshift,
            yshift);
        LAUNCH_HANDLE_ERROR(cudaGetLastError());
    }
}
//======================================================

#ifdef WINDOW_FT_BLOCK_SIZE
#undef WINDOW_FT_BLOCK_SIZE
#endif
#define WINDOW_FT_BLOCK_SIZE 128
template<bool check_max_r2>
__global__ void fft_cuda_kernel_window_fourier_transform(
		__COMPLEX_T *g_in,
		__COMPLEX_T *g_out,
		size_t iX, size_t iY, size_t iZ, size_t iYX, //Input dimensions
		size_t oX, size_t oY, size_t oZ, size_t oYX, //Output dimensions
		size_t max_idx,
		size_t max_r2 = 0)
{
	size_t n = threadIdx.x + WINDOW_FT_BLOCK_SIZE * blockIdx.x;
	if (n >= max_idx) return;

	long int k, i, kp, ip, jp;

	if (check_max_r2)
	{
		k = n / (iX * iY);
		i = (n % (iX * iY)) / iX;

		kp = k < iX ? k : k - iZ;
		ip = i < iX ? i : i - iY;
		jp = n % iX;

		if (kp*kp + ip*ip + jp*jp > max_r2)
			return;
	}
	else
	{
		k = n / (oX * oY);
		i = (n % (oX * oY)) / oX;

		kp = k < oX ? k : k - oZ;
		ip = i < oX ? i : i - oY;
		jp = n % oX;
	}

	long int  in_idx = (kp < 0 ? kp + iZ : kp) * iYX + (ip < 0 ? ip + iY : ip)*iX + jp;
	long int out_idx = (kp < 0 ? kp + oZ : kp) * oYX + (ip < 0 ? ip + oY : ip)*oX + jp;
	g_out[out_idx] =  g_in[in_idx];
}

#define INIT_VALUE_BLOCK_SIZE 512
template< typename T>
__global__ void fft_cuda_kernel_init_complex_value(
		T *data,
		RFLOAT value,
		size_t size)
{
	size_t idx = blockIdx.x * INIT_VALUE_BLOCK_SIZE + threadIdx.x;
	if (idx < size)
	{
		data[idx].x = value;
		data[idx].y = value;
	}
}

template< typename T>
void deviceInitComplexValue(CudaGlobalPtr<T, 0> &data, RFLOAT value)
{
	int grid_size = ceil((float)(data.getSize())/(float)INIT_VALUE_BLOCK_SIZE);
	fft_cuda_kernel_init_complex_value<T><<< grid_size, INIT_VALUE_BLOCK_SIZE, 0, data.getStream() >>>(
			~data,
			value,
			data.getSize());
}


void windowFourierTransform_gpu(
    CudaGlobalPtr<__COMPLEX_T, 0 > &d_in,
    CudaGlobalPtr<__COMPLEX_T, 0 > &d_out,
    size_t iX, size_t iY, size_t iZ, //Input dimensions
    size_t oX, size_t oY, size_t oZ  //Output dimensions
    )
{
    if (iX > 1 && iY/2 + 1 != iX)
        REPORT_ERROR("windowFourierTransform ERROR: the Fourier transform should be of an image with equal sizes in all dimensions!");

    deviceInitComplexValue(d_out, 0.);
    HANDLE_ERROR(cudaStreamSynchronize(d_out.getStream()));

    if(oX==iX)
    {
        HANDLE_ERROR(cudaStreamSynchronize(d_in.getStream()));
        cudaCpyDeviceToDevice(~d_in, ~d_out, oX*oY*oZ, d_out.getStream() );
        return;
    }

    if (oX > iX)
    {
        long int max_r2 = (iX - 1) * (iX - 1);

        dim3 grid_dim(ceil((float)(iX*iY*iZ) / (float) WINDOW_FT_BLOCK_SIZE),1);
        fft_cuda_kernel_window_fourier_transform<true><<< grid_dim, WINDOW_FT_BLOCK_SIZE, 0, d_out.getStream() >>>(
                d_in.d_ptr,
                d_out.d_ptr,
                iX, iY, iZ, iX * iY, //Input dimensions
                oX, oY, oZ, oX * oY, //Output dimensions
                iX*iY*iZ,
                max_r2 );
        LAUNCH_HANDLE_ERROR(cudaGetLastError());
    }
    else
    {
        dim3 grid_dim(ceil((float)(oX*oY*oZ) / (float) WINDOW_FT_BLOCK_SIZE),1);
        fft_cuda_kernel_window_fourier_transform<false><<< grid_dim, WINDOW_FT_BLOCK_SIZE, 0, d_out.getStream() >>>(
                d_in.d_ptr,
                d_out.d_ptr,
                iX, iY, iZ, iX * iY, //Input dimensions
                oX, oY, oZ, oX * oY, //Output dimensions
                oX*oY*oZ);
        LAUNCH_HANDLE_ERROR(cudaGetLastError());
    }
}

#define WINDOW_BLOCK_SIZE 128
__global__ void fft_window_kernel(RFLOAT* in, RFLOAT* out, RFLOAT normfft,
    long int oxsize, long int oxysize, long int osize,
    long int ox0, long int oy0, long int oz0, 
    long int oxF, long int oyF, long int ozF,
    long int ixsize, long int ixysize,
    long int ix0, long int iy0, long int iz0,
    long int ixF, long int iyF, long int izF ) 
{
    long int n = threadIdx.x + WINDOW_BLOCK_SIZE * blockIdx.x;
    long int k = n / oxysize + oz0;
    long int i =(n % oxysize) / oxsize + oy0;
    long int j = n % oxsize + ox0;
    long int i_idx = (k-iz0)*ixysize+(i-iy0)*ixsize+(j-ix0);

    if(n >= osize) return;

    if ((k >= iz0 && k <= izF) &&
        (i >= iy0 && i <= iyF) &&
        (j >= ix0 && j <= ixF))
            out[n] = in[i_idx] / normfft;
    else
            out[n] = 0.0;
}

__global__ void fft_window_kernel(RFLOAT* in, RFLOAT* out, RFLOAT normfft,
    long int oxsize, long int osize,
    long int ox0, long int oy0, 
    long int oxF, long int oyF,
    long int ixsize,
    long int ix0, long int iy0,
    long int ixF, long int iyF ) 
{
    long int n = threadIdx.x + WINDOW_BLOCK_SIZE * blockIdx.x;
    long int i = n / oxsize + oy0;
    long int j = n % oxsize + ox0;
    long int i_idx = (i-iy0)*ixsize+(j-ix0);

    if(n >= osize) return;

    if ((i >= iy0 && i <= iyF) &&
        (j >= ix0 && j <= ixF))
            out[n] = in[i_idx] / normfft;
    else
            out[n] = 0.0;
}

void window_gpu(CudaGlobalPtr<RFLOAT,0> &self, RFLOAT normfft, long int sy0 , long int sx0, long int syF, long int sxF,
            long int y0 , long int x0, long int yF, long int xF) 
{
    CudaGlobalPtr<RFLOAT, 0> tmp(self.getStream());
    long int ysize = yF - y0 + 1;
    long int xsize = xF - x0 + 1;
    tmp.setSize(xsize*ysize);
    tmp.device_alloc();
    
    dim3 dimGrid((xsize*ysize+WINDOW_BLOCK_SIZE-1)/WINDOW_BLOCK_SIZE,1,1);
    dim3 dimBlock(WINDOW_BLOCK_SIZE,1,1);
    fft_window_kernel<<<dimGrid,dimBlock,0,self.getStream()>>> (~self, ~tmp, normfft,
        xsize, xsize*ysize, 
        x0, y0, xF, yF,
        sxF-sx0+1,
        sx0, sy0, sxF, syF);

    self.streamSync();

    // copy result by exchange ptr
    RFLOAT* p = self.d_ptr;
    self.d_ptr = tmp.d_ptr;
    tmp.d_ptr = p;
    tmp.setSize(self.getSize());
    self.setSize(xsize*ysize);

    tmp.free_if_set();
}

void window_gpu(CudaGlobalPtr<RFLOAT,0> &self, RFLOAT normfft, long int sz0, long int sy0, long int sx0, long int szF, long int syF, long int sxF,
            long int z0, long int y0 , long int x0, long int zF, long int yF, long int xF) 
{
    //printf("-- Now debugging... use CPU\n");
    CudaGlobalPtr<RFLOAT, 0> tmp(self.getStream());
    long int zsize = zF - z0 + 1;
    long int ysize = yF - y0 + 1;
    long int xsize = xF - x0 + 1;
    tmp.setSize(xsize*ysize*zsize);
    tmp.device_alloc();
    dim3 dimGrid((xsize*ysize*zsize+WINDOW_BLOCK_SIZE-1)/WINDOW_BLOCK_SIZE,1,1);
    dim3 dimBlock(WINDOW_BLOCK_SIZE,1,1);
    fft_window_kernel<<<dimGrid,dimBlock,0,self.getStream()>>> (~self, ~tmp, normfft,
        xsize, xsize*ysize, xsize*ysize*zsize, 
        x0, y0, z0, xF, yF, zF,
        sxF-sx0+1,(sxF-sx0+1)*(syF-sy0+1),
        sx0, sy0, sz0, sxF, syF, szF);

    self.streamSync();
    
    // copy result by exchange ptr
    RFLOAT* p = self.d_ptr;
    self.d_ptr = tmp.d_ptr;
    tmp.d_ptr = p;
    tmp.setSize(self.getSize());
    self.setSize(xsize*ysize*zsize);
    tmp.free_if_set();
}

__device__ inline double fft_cuda_atomic_add(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	}
	while (assumed != old); // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	return __longlong_as_double(old);
}

__global__ void fft_cuda_kernel_softMaskBackgroundValue(	RFLOAT *vol,
                                                            long int vol_size,
                                                            long int xdim,
                                                            long int ydim,
                                                            long int zdim,
                                                            long int xinit,
                                                            long int yinit,
                                                            long int zinit,
                                                            bool do_Mnoise,
                                                            RFLOAT radius,
                                                            RFLOAT radius_p,
                                                            RFLOAT cosine_width,
                                                            RFLOAT *g_sum,
                                                            RFLOAT *g_sum_bg)
{

		int tid = threadIdx.x;
		int bid = blockIdx.x;

//		vol.setXmippOrigin(); // sets xinit=xdim , also for y z
		RFLOAT r, raisedcos;
		int x,y,z;
		__shared__ RFLOAT     img_pixels[SOFTMASK_BLOCK_SIZE];
		__shared__ RFLOAT    partial_sum[SOFTMASK_BLOCK_SIZE];
		__shared__ RFLOAT partial_sum_bg[SOFTMASK_BLOCK_SIZE];

		long int texel_pass_num = ceilfracf(vol_size,SOFTMASK_BLOCK_SIZE*gridDim.x);
		int texel = bid*SOFTMASK_BLOCK_SIZE*texel_pass_num + tid;

		partial_sum[tid]=(RFLOAT)0.0;
		partial_sum_bg[tid]=(RFLOAT)0.0;

		for (int pass = 0; pass < texel_pass_num; pass++, texel+=SOFTMASK_BLOCK_SIZE) // loop the available warps enough to complete all translations for this orientation
		{
			if(texel<vol_size)
			{
				img_pixels[tid]=__ldg(&vol[texel]);

				z =   texel / (xdim*ydim) ;
				y = ( texel % (xdim*ydim) ) / xdim ;
				x = ( texel % (xdim*ydim) ) % xdim ;

				z-=zinit;
				y-=yinit;
				x-=xinit;

				r = sqrt(RFLOAT(x*x + y*y + z*z));

				if (r < radius)
					continue;
				else if (r > radius_p)
				{
					partial_sum[tid]    += (RFLOAT)1.0;
					partial_sum_bg[tid] += img_pixels[tid];
				}
				else
				{
#ifndef RELION_SINGLE_PRECISION
					raisedcos = 0.5 + 0.5  * cospi( (radius_p - r) / cosine_width );
#else
					raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width );
#endif
					partial_sum[tid] += raisedcos;
					partial_sum_bg[tid] += raisedcos * img_pixels[tid];
				}
			}
		}

		fft_cuda_atomic_add(&g_sum[tid]   , partial_sum[tid]);
		fft_cuda_atomic_add(&g_sum_bg[tid], partial_sum_bg[tid]);
}


__global__ void fft_cuda_kernel_cosineFilter(	RFLOAT *vol,
                                                long int vol_size,
                                                long int xdim,
                                                long int ydim,
                                                long int zdim,
                                                long int xinit,
                                                long int yinit,
                                                long int zinit,
                                                bool do_Mnoise,
                                                RFLOAT radius,
                                                RFLOAT radius_p,
                                                RFLOAT cosine_width,
                                                RFLOAT bg_value)
{

	int tid = threadIdx.x;
	int bid = blockIdx.x;

//		vol.setXmippOrigin(); // sets xinit=xdim , also for y z
	RFLOAT r, raisedcos;
	int x,y,z;
	__shared__ RFLOAT     img_pixels[SOFTMASK_BLOCK_SIZE];

	long int texel_pass_num = ceilfracf(vol_size,SOFTMASK_BLOCK_SIZE*gridDim.x);
	int texel = bid*SOFTMASK_BLOCK_SIZE*texel_pass_num + tid;

	for (int pass = 0; pass < texel_pass_num; pass++, texel+=SOFTMASK_BLOCK_SIZE) // loop the available warps enough to complete all translations for this orientation
	{
		if(texel<vol_size)
		{
			img_pixels[tid]=__ldg(&vol[texel]);

			z =   texel / (xdim*ydim) ;
			y = ( texel % (xdim*ydim) ) / xdim ;
			x = ( texel % (xdim*ydim) ) % xdim ;

			z-=zinit;
			y-=yinit;
			x-=xinit;

			r = sqrt(RFLOAT(x*x + y*y + z*z));

			if (r < radius)
				continue;
			else if (r > radius_p)
				img_pixels[tid]=bg_value;
			else
			{
#ifndef RELION_SINGLE_PRECISION
				raisedcos = 0.5  + 0.5  * cospi( (radius_p - r) / cosine_width );
#else
				raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width );
#endif
				img_pixels[tid]= img_pixels[tid]*(1-raisedcos) + bg_value*raisedcos;

			}
			vol[texel]=img_pixels[tid];
		}

	}
}

template <typename T>
static T fft_getSumOnDevice(CudaGlobalPtr<T,0> &ptr)
{
#ifdef DEBUG_CUDA
if (ptr.size == 0)
	printf("DEBUG_ERROR: getSumOnDevice called with pointer of zero size.\n");
if (ptr.d_ptr == NULL)
	printf("DEBUG_ERROR: getSumOnDevice called with null device pointer.\n");
if (ptr.getAllocator() == NULL)
	printf("DEBUG_ERROR: getSumOnDevice called with null allocator.\n");
#endif
	CudaGlobalPtr<T, 0 >  val(1, ptr.getStream());
	val.device_alloc();
	size_t temp_storage_size = 0;

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Sum( NULL, temp_storage_size, ~ptr, ~val, ptr.size));

	if(temp_storage_size==0)
		temp_storage_size=1;

    T* tmp_ptr; 
    cudaMalloc((void**)&tmp_ptr,sizeof(T)*temp_storage_size);

	DEBUG_HANDLE_ERROR(cub::DeviceReduce::Sum( tmp_ptr, temp_storage_size, ~ptr, ~val, ptr.size, ptr.getStream()));

	val.cp_to_host();
	ptr.streamSync();

	cudaFree(tmp_ptr);

	return val[0];
}

void fft_runSoftMaskOutsideMap( CudaGlobalPtr<RFLOAT, 0> &vol_out, 
                                long int xdim,
                                long int ydim,
                                long int zdim,
                                long int xinit,
                                long int yinit,
                                long int zinit,
                                bool do_Mnoise,
                                RFLOAT radius,
                                RFLOAT cosine_width) 
{
    if (radius < 0)
		radius = (RFLOAT)xdim/2.;
    RFLOAT radius_p = radius + cosine_width;
    long int vol_size = xdim*ydim*zdim;
    //dim3 dimGrid(1,1,1);
    //dim3 dimBlock(SOFTMASK_BLOCK_SIZE,1,1);
    //fft_cuda_kernel_softMaskOutsideMap<<<dimGrid,dimBlock,0,vol_out.getStream()>>> (~vol_out,vol_size,
    //    xdim,ydim,zdim,xinit,yinit,zinit,do_Mnoise,radius,radius_p,cosine_width);

    RFLOAT sum_bg(0.);
    dim3 block_dim = 128; //TODO: set balanced (hardware-dep?)
    CudaGlobalPtr<RFLOAT, 0> softMaskSum   (SOFTMASK_BLOCK_SIZE,0,0);
    CudaGlobalPtr<RFLOAT, 0> softMaskSum_bg(SOFTMASK_BLOCK_SIZE,0,0);
    softMaskSum.device_alloc();
    softMaskSum_bg.device_alloc();
    softMaskSum.device_init(0.f);
    softMaskSum_bg.device_init(0.f);
    fft_cuda_kernel_softMaskBackgroundValue<<<block_dim,SOFTMASK_BLOCK_SIZE>>>(	~vol_out,
                                                                                vol_size,
                                                                                xdim,
                                                                                ydim,
                                                                                zdim,
                                                                                xdim/2,
                                                                                ydim/2,
                                                                                zdim/2, //unused
                                                                                true,
                                                                                radius,
                                                                                radius_p,
                                                                                cosine_width,
                                                                                ~softMaskSum,
                                                                                ~softMaskSum_bg);
    LAUNCH_HANDLE_ERROR(cudaGetLastError());
    softMaskSum.streamSync();
    sum_bg = (RFLOAT) fft_getSumOnDevice(softMaskSum_bg) / (RFLOAT) fft_getSumOnDevice(softMaskSum);
    softMaskSum.streamSync();
    fft_cuda_kernel_cosineFilter<<<block_dim,SOFTMASK_BLOCK_SIZE>>>(	~vol_out,
                                                                        vol_size,
                                                                        xdim,
                                                                        ydim,
                                                                        zdim,
                                                                        xdim/2,
                                                                        ydim/2,
                                                                        zdim/2, //unused
                                                                        true,
                                                                        radius,
                                                                        radius_p,
                                                                        cosine_width,
                                                                        sum_bg);
    LAUNCH_HANDLE_ERROR(cudaGetLastError());
}

// interpolator == NEAREST_NEIGHBOUR && r_min_nn == 0
__global__ void griddingCorrect_kernel_1(RFLOAT* vol_in, int xdim, int ydim, int zdim, int xinit, int yinit, int zinit, int ori_size_padding_factor)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	int i, j, k;
	j = ((global_index % xdim) + xinit);
	i = ((global_index / xdim) % ydim + yinit);
	k = (global_index / (xdim * ydim) + zinit);
	RFLOAT r = sqrt((RFLOAT)(k * k + i * i + j * j));

	if (r == 0. || global_index >= xdim * ydim * zdim)
	{
		return;
	}

	RFLOAT rval = r / (ori_size_padding_factor);
	RFLOAT sinc = sin(PI * rval) / (PI * rval);
	// Interpolation (goes with "interpolator") to go from arbitrary to fine grid
	// NN interpolation is convolution with a rectangular pulse, which FT is a sinc function
	vol_in[global_index] /= sinc;
}
// interpolator == TRILINEAR || (interpolator == NEAREST_NEIGHBOUR && r_min_nn > 0)
__global__ void griddingCorrect_kernel_2(RFLOAT* vol_in, int xdim, int ydim, int zdim, int xinit, int yinit, int zinit, int ori_size_padding_factor)
{
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;

	int i, j, k;
	j = ((global_index % xdim) + xinit);
	i = ((global_index / xdim) % ydim + yinit);
	k = (global_index / (xdim * ydim) + zinit);
	RFLOAT r = sqrt((RFLOAT)(k * k + i * i + j * j));

	if (r == 0. || global_index >= xdim * ydim * zdim)
	{
		return;
	}

	RFLOAT rval = r / (ori_size_padding_factor);
	RFLOAT sinc = sin(PI * rval) / (PI * rval);
	// trilinear interpolation is convolution with a triangular pulse, which FT is a sinc^2 function
	vol_in[global_index] /= sinc * sinc;
}