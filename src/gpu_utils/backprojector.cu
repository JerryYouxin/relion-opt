#include "src/backprojector.h"
#include "src/gpu_utils/cuda_mem_utils.h"
#include "src/gpu_utils/cuda_settings.h"
#include <time.h>
#include "src/gpu_utils/cuda_kernels/recons.cuh"
#include "src/gpu_utils/cuda_fft.h"
#include "src/gpu_utils/cuda_helper_functions.cuh"
#include <assert.h>

#include <omp.h>

#ifdef TIMING
#define DTIC(timer,stamp) timer.tic(stamp)
#define DTOC(timer,stamp) timer.toc(stamp)
#else
#define DTIC(timer,stamp)
#define DTOC(timer,stamp)
#endif

extern __constant__ RFLOAT __R_array [100 * 4 * 4];

void BackProjector::symmetrise_gpu(int rank, int nr_helical_asu, RFLOAT helical_twist, RFLOAT helical_rise)
{
#ifdef TIMING
	Timer sym_timer;
	int TIMING_TOTAL = sym_timer.setNew("Total");
	int TIMING_ENFORCE = sym_timer.setNew("-- enforceHermitianSymmetry");
	int TIMING_HELICAL = sym_timer.setNew("-- applyHelicalSymmetry");
	int TIMING_POINTGROUP = sym_timer.setNew("-- applyPointGroupSymmetry");
	int TIMING_GPUSET  = sym_timer.setNew("-- set gpu");
	int TIMING_GPURESET= sym_timer.setNew("-- reset gpu");
	sym_timer.tic(TIMING_ENFORCE);
#endif
	// First make sure the input arrays are obeying Hermitian symmetry,
	// which is assumed in the rotation operators of both helical and point group symmetry
	enforceHermitianSymmetry();
#ifdef TIMING
	sym_timer.toc(TIMING_ENFORCE);
	sym_timer.tic(TIMING_HELICAL);
#endif
	// Then apply helical and point group symmetry (order irrelevant?)
	applyHelicalSymmetry(nr_helical_asu, helical_twist, helical_rise);
#ifdef TIMING
	sym_timer.toc(TIMING_HELICAL);
	sym_timer.tic(TIMING_GPUSET);
#endif
    int devCount;
    cudaGetDeviceCount(&devCount);
    int dev_id=rank%devCount;
	DEBUG_HANDLE_ERROR(cudaSetDevice(dev_id));
#ifdef TIMING
	sym_timer.toc(TIMING_GPUSET);
	sym_timer.tic(TIMING_POINTGROUP);
#endif
	applyPointGroupSymmetry_gpu(rank);
#ifdef TIMING
	sym_timer.toc(TIMING_POINTGROUP);
	sym_timer.tic(TIMING_GPURESET);
#endif
	HANDLE_ERROR(cudaDeviceReset());
#ifdef TIMING
	sym_timer.toc(TIMING_GPURESET);
	sym_timer.printTimes(true);
#endif
}

void BackProjector::applyPointGroupSymmetry_gpu(int rank) 
{
#ifdef TIMING
	Timer sym_timer;
	int TIMING_TOTAL = sym_timer.setNew("Total");
	int TIMING_START_1	= sym_timer.setNew("-- stream create");
	int TIMING_START_2	= sym_timer.setNew("-- start initialize");
	int TIMING_MEMALLOC = sym_timer.setNew("-- cudaMalloc");
	int TIMING_MEMFREE  = sym_timer.setNew("-- cudaFree");
	int TIMING_CPYTOSYM = sym_timer.setNew("-- cudaMemcpyToSymbol");
	int TIMING_CPYTODEV = sym_timer.setNew("-- cudaMemcpyHostToDev");
	int TIMING_DEVTODEV = sym_timer.setNew("-- cudaMemcpyDevToDev");
	int TIMING_DEVTOHST = sym_timer.setNew("-- cudaMemcpyHostToDev");
	int TIMING_KERNEL   = sym_timer.setNew("-- symmKernel");
#endif
	DTIC(sym_timer,TIMING_TOTAL);
	int rmax2 = ROUND(r_max * padding_factor) * ROUND(r_max * padding_factor);
	if (SL.SymsNo() > 0 && ref_dim == 3)
	{
		DTIC(sym_timer,TIMING_START_1);
		//cudaStream_t stream;
		//cudaStreamCreate(&stream);
		//HANDLE_ERROR(cudaGetLastError());
		DTOC(sym_timer,TIMING_START_1);
		DTIC(sym_timer,TIMING_START_2);
		int zdim = ZSIZE(weight);
		int ydim = YSIZE(weight);
		int xdim = XSIZE(weight);
		int xydim= YXSIZE(weight);
		int model_size = xydim * zdim;
		if(model_size<=0) return;
		int start_x = STARTINGX(weight);
		int start_y = STARTINGY(weight);
		int start_z = STARTINGZ(weight);
		RFLOAT* my_weight_temp_D=NULL;
		RFLOAT* my_weight_D=NULL;
		__COMPLEX_T * my_data_temp_D=NULL;
		__COMPLEX_T * my_data_D=NULL;
		DTOC(sym_timer,TIMING_START_2);
		DTIC(sym_timer,TIMING_CPYTOSYM);
		HANDLE_ERROR(cudaGetLastError());
		set_R(SL.__R.mdata, SL.SymsNo(), 0);
		//size_t avail, total_mem;
		//HANDLE_ERROR(cudaStreamSynchronize(0));
		DTOC(sym_timer,TIMING_CPYTOSYM);
		dim3 blockDim(BLOCK_SIZE, 1, 1);
		dim3 gridDim((model_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
		//printf("Model size: %d\n",model_size);
		DTIC(sym_timer,TIMING_MEMALLOC);
		// race for multi-process allocation
		#define NTRY 20
		int itry=0;
		while(1) {
			if( cudaMalloc((void**)&my_weight_temp_D, model_size * sizeof(RFLOAT      ))==cudaSuccess &&
				cudaMalloc((void**)&my_data_temp_D  , model_size * sizeof(__COMPLEX_T ))==cudaSuccess &&
				cudaMalloc((void**)&my_weight_D     , model_size * sizeof(RFLOAT      ))==cudaSuccess &&
				cudaMalloc((void**)&my_data_D       , model_size * sizeof(__COMPLEX_T ))==cudaSuccess) {
					break;
			}
			printf("######## WARNING : GPU memory not enough, wait for others ... %d #########\n",itry);
			if(my_weight_temp_D!=NULL) cudaFree(my_weight_temp_D);
			else                       cudaGetLastError();
			if(my_data_temp_D  !=NULL) cudaFree(my_data_temp_D  );
			else                       cudaGetLastError();
			if(my_weight_D     !=NULL) cudaFree(my_weight_D     );
			else                       cudaGetLastError();
			if(my_data_D       !=NULL) cudaFree(my_data_D       );
			else                       cudaGetLastError();
			++itry;
			if(itry>=NTRY) {
				printf("######## ERROR : GPU memory still not enough after trying %d attempts #########\n", itry);
				REPORT_ERROR("GPU memory not enough!");
			}
			usleep(100000);
		}
		#undef NTRY
		DTOC(sym_timer,TIMING_MEMALLOC);
		DTIC(sym_timer,TIMING_CPYTODEV);
		cudaMemcpyAsync(my_data_D, data.data, model_size * sizeof(__COMPLEX_T ), cudaMemcpyHostToDevice,0);
		cudaMemcpyAsync(my_weight_D, weight.data, model_size * sizeof(RFLOAT), cudaMemcpyHostToDevice,0);
		//HANDLE_ERROR(cudaStreamSynchronize(0));
		DTOC(sym_timer,TIMING_CPYTODEV);
		DTIC(sym_timer,TIMING_DEVTODEV);
		cudaMemcpyAsync(my_data_temp_D, my_data_D, model_size * sizeof(__COMPLEX_T ), cudaMemcpyDeviceToDevice,0);
		cudaMemcpyAsync(my_weight_temp_D, my_weight_D, model_size * sizeof(RFLOAT), cudaMemcpyDeviceToDevice,0);
		//HANDLE_ERROR(cudaGetLastError());
		HANDLE_ERROR(cudaStreamSynchronize(0));
		DTOC(sym_timer,TIMING_DEVTODEV);
		DTIC(sym_timer,TIMING_KERNEL);
		symmetrise_kernel <<< gridDim, blockDim >>>(my_data_temp_D,
													my_weight_temp_D,
													my_data_D,
													my_weight_D,
													xdim,
													ydim,
													xydim,
													zdim,
													start_x,
													start_y,
													start_z,
													rmax2,
													SL.SymsNo());
		HANDLE_ERROR(cudaGetLastError());
		HANDLE_ERROR(cudaStreamSynchronize(0));
		DTOC(sym_timer,TIMING_KERNEL);
		DTIC(sym_timer,TIMING_DEVTOHST);
		cudaMemcpyAsync(data.data, my_data_D, model_size * sizeof(__COMPLEX_T ), cudaMemcpyDeviceToHost,0);
		cudaMemcpyAsync(weight.data, my_weight_D, model_size * sizeof(RFLOAT), cudaMemcpyDeviceToHost,0);
		HANDLE_ERROR(cudaGetLastError());
		HANDLE_ERROR(cudaStreamSynchronize(0));
		DTOC(sym_timer,TIMING_DEVTOHST);
		DTIC(sym_timer,TIMING_MEMFREE);
		cudaFree(my_data_temp_D);
		cudaFree(my_weight_temp_D);
		cudaFree(my_data_D);
		cudaFree(my_weight_D);
		//cudaStreamDestroy(stream);
		HANDLE_ERROR(cudaGetLastError());
		DTOC(sym_timer,TIMING_MEMFREE);
	}
	DTOC(sym_timer,TIMING_TOTAL);
#ifdef TIMING
	if(rank==1)
		sym_timer.printTimes(1);
#endif
}

void BackProjector::reconstruct_gpu(int rank,
                                MultidimArray<RFLOAT> &vol_out,
                                int max_iter_preweight,
                                bool do_map,
                                RFLOAT tau2_fudge,
                                MultidimArray<RFLOAT> &tau2,
                                MultidimArray<RFLOAT> &sigma2,
                                MultidimArray<RFLOAT> &data_vs_prior,
                                MultidimArray<RFLOAT> &fourier_coverage,
                                MultidimArray<RFLOAT> fsc, // only input
                                RFLOAT normalise,
                                bool update_tau2_with_fsc,
                                bool is_whole_instead_of_half,
                                int nr_threads,
                                int minres_map,
                                bool printTimes)
{
	if(skip_gridding || ref_dim!=3) {
		printf("\n=== Warning : skip_gridding did not support GPU reconstructing, will use CPU only ===\n");
		reconstruct(
			vol_out,
			max_iter_preweight,
			do_map,
			tau2_fudge,
			tau2,
			sigma2,
			data_vs_prior,
			fourier_coverage,
			fsc, // only input
			normalise,
			update_tau2_with_fsc,
			is_whole_instead_of_half,
			nr_threads,
			minres_map,
			printTimes);
		return ;
	}
#ifdef TIMING
	Timer ReconTimer;
	int ReconS_0 = ReconTimer.setNew(" RcS0_SetDevice ");
	int ReconS_1 = ReconTimer.setNew(" RcS1_Init ");
	int ReconS_2 = ReconTimer.setNew(" RcS2_Shape&Noise ");
	int ReconS_3 = ReconTimer.setNew(" RcS3_skipGridding ");
	int ReconS_4 = ReconTimer.setNew(" RcS4_doGridding_norm ");
	int ReconS_5 = ReconTimer.setNew(" RcS5_doGridding_init ");
	int ReconS_6 = ReconTimer.setNew(" RcS6_doGridding_iter ");
	int ReconS_7 = ReconTimer.setNew(" RcS7_doGridding_apply ");
	int ReconS_8 = ReconTimer.setNew(" RcS8_blobConvolute ");
	int ReconS_9 = ReconTimer.setNew(" RcS9_blobResize ");
	int ReconS_10 = ReconTimer.setNew(" RcS10_blobSetReal ");
	int ReconS_11 = ReconTimer.setNew(" RcS11_blobSetTemp ");
	int ReconS_12 = ReconTimer.setNew(" RcS12_blobTransform ");
	int ReconS_13 = ReconTimer.setNew(" RcS13_blobCenterFFT ");
	int ReconS_14 = ReconTimer.setNew(" RcS14_blobNorm1 ");
	int ReconS_15 = ReconTimer.setNew(" RcS15_blobSoftMask ");
	int ReconS_16 = ReconTimer.setNew(" RcS16_blobNorm2 ");
	int ReconS_17 = ReconTimer.setNew(" RcS17_WindowReal ");
	int ReconS_18 = ReconTimer.setNew(" RcS18_GriddingCorrect ");
	int ReconS_19 = ReconTimer.setNew(" RcS19_tauInit ");
	int ReconS_20 = ReconTimer.setNew(" RcS20_tausetReal ");
	int ReconS_21 = ReconTimer.setNew(" RcS21_tauTransform ");
	int ReconS_22 = ReconTimer.setNew(" RcS22_tautauRest ");
	int ReconS_23 = ReconTimer.setNew(" RcS23_tauShrinkToFit ");
	int ReconS_24 = ReconTimer.setNew(" RcS24_extra ");
	int ReconS_25 = ReconTimer.setNew(" RcS25_iterAlgo_gpu ");
	int ReconS_26 = ReconTimer.setNew(" RcS26_resetGPU ");
#endif

	DTIC(ReconTimer,ReconS_0);
	int devCount;
	size_t Xsize, Ysize, Zsize, size, YXsize;
	cudaGetDeviceCount(&devCount);
	int dev_id=(rank+devCount-1)%devCount;
	DTOC(ReconTimer,ReconS_0);

    DTIC(ReconTimer,ReconS_1);
    FourierTransformer transformer;
	MultidimArray<RFLOAT> Fweight;
	// Fnewweight can become too large for a float: always keep this one in double-precision
	MultidimArray<double> Fnewweight;
	//MultidimArray<Complex>& Fconv = transformer.getFourierReference();
	MultidimArray<Complex> Fconv;
	int max_r2 = ROUND(r_max * padding_factor) * ROUND(r_max * padding_factor);

    cudaStream_t stream;
    DEBUG_HANDLE_ERROR(cudaSetDevice(dev_id));
	DEBUG_HANDLE_ERROR(cudaStreamCreate(&stream));
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
#ifdef FORCE_NOT_USE_BUFF_FFT
	RelionCudaFFT cutransformer(stream, NULL, ref_dim);
#else
	BuffCudaFFT3D cutransformer(stream);
#endif
	CudaGlobalPtr<RFLOAT, false> cuFweight(stream);
	CudaGlobalPtr<double, false> cuFnewweight(stream);
	CudaGlobalPtr<RFLOAT, false> tabulatedValues(stream);
	bool host_splitted = false;
	bool weight_splitted = false;
	size_t weight_buffSize = 0;

	printf("pad_size=%d, ref_dim=%d, r_max=%d, normalise=%lf\n",pad_size,ref_dim,r_max,normalise);

    // Set Fweight, Fnewweight and Fconv to the right size
    if (ref_dim == 2) {
		vol_out.setDimensions(pad_size, pad_size, 1, 1);
		Fconv.reshape(pad_size,pad_size/2+1);
#ifdef FORCE_NOT_USE_BUFF_FFT
		if(cutransformer.setSize(pad_size, pad_size, 1,1,1)<0) {
#else
		// not supported
		if(true) {
#endif
			printf("\n=== Warning : something went wrong when prepare FFT plan, use CPU instead ===\n");
			cutransformer.clear();
			HANDLE_ERROR(cudaDeviceSynchronize());
			cudaStreamDestroy(stream);
			HANDLE_ERROR(cudaDeviceReset());
			reconstruct(
				vol_out,
				max_iter_preweight,
				do_map,
				tau2_fudge,
				tau2,
				sigma2,
				data_vs_prior,
				fourier_coverage,
				fsc, // only input
				normalise,
				update_tau2_with_fsc,
				is_whole_instead_of_half,
				nr_threads,
				minres_map,
				printTimes);
			return ;
		}
	}
    else {
        // Too costly to actually allocate the space
        // Trick transformer with the right dimensions
		vol_out.setDimensions(pad_size, pad_size, pad_size, 1);
		Fconv.reshape(pad_size,pad_size,pad_size/2+1);
#ifdef FORCE_NOT_USE_BUFF_FFT
		if(cutransformer.setSize(pad_size, pad_size, pad_size,1,1)<0) {
#else
		size_t fconv_size = pad_size*pad_size*(pad_size/2+1);
		size_t needed = cutransformer.minReq(pad_size, pad_size, pad_size,1);
		size_t required_free = 70*1000*1000 + fconv_size*sizeof(double) + XSIZE(tab_ftblob.tabulatedValues)*sizeof(RFLOAT); // 70M free for accidents
		size_t weight_needed = fconv_size*(sizeof(RFLOAT));
		size_t avail, total;
		needed += weight_needed + required_free;
		DEBUG_HANDLE_ERROR(cudaMemGetInfo( &avail, &total ));
		host_splitted = (avail <= needed);
		required_free += weight_needed;
		weight_splitted = (avail <= needed) && (avail > needed-weight_needed);
		//printf("cutrans minimal need=%ld, total need=%ld, avail=%ld\n",cutransformer.minReq(pad_size, pad_size, pad_size,1), needed, avail);
		if(host_splitted) {
			printf("WARNING : will try HOST bufferring for reconstruction due to TOO large reconstruction size\n");
			REPORT_ERROR("Not Implemented!");
		}
		else if(weight_splitted) {
			printf("will try HOST bufferring only for weight in reconstruction due to large reconstruction size\n");
			weight_buffSize = CEIL(0.7 * (avail - needed)); // 0.3 for accident;
			REPORT_ERROR("Not Implemented!");
		}
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		if(cutransformer.setSize(pad_size, pad_size, pad_size,1,required_free,host_splitted)<0) {
#endif
			printf("\n=== Warning : something went wrong when prepare FFT plan, use CPU instead ===\n");
			cutransformer.clear();
			HANDLE_ERROR(cudaDeviceSynchronize());
			cudaStreamDestroy(stream);
			HANDLE_ERROR(cudaDeviceReset());
			reconstruct(
				vol_out,
				max_iter_preweight,
				do_map,
				tau2_fudge,
				tau2,
				sigma2,
				data_vs_prior,
				fourier_coverage,
				fsc, // only input
				normalise,
				update_tau2_with_fsc,
				is_whole_instead_of_half,
				nr_threads,
				minres_map,
				printTimes);
			return ;
		}
	}

	cutransformer.setPlan();
	size_t fxsize = XSIZE(Fconv);
	size_t fysize = YSIZE(Fconv);
	size_t fxysize= fxsize*fysize;
	size_t fzsize = ZSIZE(Fconv);
	size_t fsize  = fxysize*fzsize;
	if(weight_splitted) cuFweight.setSize(weight_buffSize);
	else                cuFweight.setSize(fsize);
	cuFnewweight.setSize(fsize);
	tabulatedValues.setSize(XSIZE(tab_ftblob.tabulatedValues));
	cuFweight.device_alloc();
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	cuFnewweight.device_alloc();
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	tabulatedValues.device_alloc();
	LAUNCH_HANDLE_ERROR(cudaGetLastError());

	//transformer.setReal(vol_out); // Fake set real. 1. Allocate space for Fconv 2. calculate plans.
    vol_out.clear(); // Reset dimensions to 0

    DTOC(ReconTimer,ReconS_1);
    DTIC(ReconTimer,ReconS_2);

    Fweight.reshape(Fconv);
    if (!skip_gridding)
    	Fnewweight.reshape(Fconv);
	// Go from projector-centered to FFTW-uncentered
	decenter(weight, Fweight, max_r2);

	// Take oversampling into account
	RFLOAT oversampling_correction = (ref_dim == 3) ? (padding_factor * padding_factor * padding_factor) : (padding_factor * padding_factor);
	MultidimArray<RFLOAT> counter;

	// First calculate the radial average of the (inverse of the) power of the noise in the reconstruction
	// This is the left-hand side term in the nominator of the Wiener-filter-like update formula
	// and it is stored inside the weight vector
	// Then, if (do_map) add the inverse of tau2-spectrum values to the weight
	sigma2.initZeros(ori_size/2 + 1);
	counter.initZeros(ori_size/2 + 1);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
	{
		int r2 = kp * kp + ip * ip + jp * jp;
		if (r2 < max_r2)
		{
			int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
			RFLOAT invw = oversampling_correction * DIRECT_A3D_ELEM(Fweight, k, i, j);
			DIRECT_A1D_ELEM(sigma2, ires) += invw;
			DIRECT_A1D_ELEM(counter, ires) += 1.;
		}
	}
	// Average (inverse of) sigma2 in reconstruction
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
	{
		if (DIRECT_A1D_ELEM(sigma2, i) > 1e-10)
			DIRECT_A1D_ELEM(sigma2, i) = DIRECT_A1D_ELEM(counter, i) / DIRECT_A1D_ELEM(sigma2, i);
		else if (DIRECT_A1D_ELEM(sigma2, i) == 0)
			DIRECT_A1D_ELEM(sigma2, i) = 0.;
		else
		{
			std::cerr << " DIRECT_A1D_ELEM(sigma2, i)= " << DIRECT_A1D_ELEM(sigma2, i) << std::endl;
			REPORT_ERROR("BackProjector::reconstruct: ERROR: unexpectedly small, yet non-zero sigma2 value, this should not happen...a");
		}
	}
	if (update_tau2_with_fsc)
	{
		tau2.reshape(ori_size/2 + 1);
		data_vs_prior.initZeros(ori_size/2 + 1);
		// Then calculate new tau2 values, based on the FSC
		if (!fsc.sameShape(sigma2) || !fsc.sameShape(tau2))
		{
			fsc.printShape(std::cerr);
			tau2.printShape(std::cerr);
			sigma2.printShape(std::cerr);
			REPORT_ERROR("ERROR BackProjector::reconstruct: sigma2, tau2 and fsc have different sizes");
		}
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
		{
			// FSC cannot be negative or zero for conversion into tau2
			RFLOAT myfsc = XMIPP_MAX(0.001, DIRECT_A1D_ELEM(fsc, i));
			if (is_whole_instead_of_half)
			{
				// Factor two because of twice as many particles
				// Sqrt-term to get 60-degree phase errors....
				myfsc = sqrt(2. * myfsc / (myfsc + 1.));
			}
			myfsc = XMIPP_MIN(0.999, myfsc);
			RFLOAT myssnr = myfsc / (1. - myfsc);
			// Sjors 29nov2017 try tau2_fudge for pulling harder on Refine3D runs...
			myssnr *= tau2_fudge;
			RFLOAT fsc_based_tau = myssnr * DIRECT_A1D_ELEM(sigma2, i);
			DIRECT_A1D_ELEM(tau2, i) = fsc_based_tau;
			// data_vs_prior is merely for reporting: it is not used for anything in the reconstruction
			DIRECT_A1D_ELEM(data_vs_prior, i) = myssnr;

		}
	}
    DTOC(ReconTimer,ReconS_2);
	// Apply MAP-additional term to the Fnewweight array
	// This will regularise the actual reconstruction
    if (do_map)
	{

    	// Then, add the inverse of tau2-spectrum values to the weight
		// and also calculate spherical average of data_vs_prior ratios
		if (!update_tau2_with_fsc)
			data_vs_prior.initZeros(ori_size/2 + 1);
		fourier_coverage.initZeros(ori_size/2 + 1);
		counter.initZeros(ori_size/2 + 1);
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
 		{
			int r2 = kp * kp + ip * ip + jp * jp;
			if (r2 < max_r2)
			{
				int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
				RFLOAT invw = DIRECT_A3D_ELEM(Fweight, k, i, j);

				RFLOAT invtau2;
				if (DIRECT_A1D_ELEM(tau2, ires) > 0.)
				{
					// Calculate inverse of tau2
					invtau2 = 1. / (oversampling_correction * tau2_fudge * DIRECT_A1D_ELEM(tau2, ires));
				}
				else if (DIRECT_A1D_ELEM(tau2, ires) == 0.)
				{
					// If tau2 is zero, use small value instead
					invtau2 = 1./ ( 0.001 * invw);
				}
				else
				{
					std::cerr << " sigma2= " << sigma2 << std::endl;
					std::cerr << " fsc= " << fsc << std::endl;
					std::cerr << " tau2= " << tau2 << std::endl;
					REPORT_ERROR("ERROR BackProjector::reconstruct: Negative or zero values encountered for tau2 spectrum!");
				}

				// Keep track of spectral evidence-to-prior ratio and remaining noise in the reconstruction
				if (!update_tau2_with_fsc)
					DIRECT_A1D_ELEM(data_vs_prior, ires) += invw / invtau2;

				// Keep track of the coverage in Fourier space
				if (invw / invtau2 >= 1.)
					DIRECT_A1D_ELEM(fourier_coverage, ires) += 1.;

				DIRECT_A1D_ELEM(counter, ires) += 1.;

				// Only for (ires >= minres_map) add Wiener-filter like term
				if (ires >= minres_map)
				{
					// Now add the inverse-of-tau2_class term
					invw += invtau2;
					// Store the new weight again in Fweight
					DIRECT_A3D_ELEM(Fweight, k, i, j) = invw;
				}
			}
		}

		// Average data_vs_prior
		if (!update_tau2_with_fsc)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(data_vs_prior)
			{
				if (i > r_max)
					DIRECT_A1D_ELEM(data_vs_prior, i) = 0.;
				else if (DIRECT_A1D_ELEM(counter, i) < 0.001)
					DIRECT_A1D_ELEM(data_vs_prior, i) = 999.;
				else
					DIRECT_A1D_ELEM(data_vs_prior, i) /= DIRECT_A1D_ELEM(counter, i);
			}
		}

		// Calculate Fourier coverage in each shell
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fourier_coverage)
		{
			if (DIRECT_A1D_ELEM(counter, i) > 0.)
				DIRECT_A1D_ELEM(fourier_coverage, i) /= DIRECT_A1D_ELEM(counter, i);
		}

	} //end if do_map
    DTOC(ReconTimer,ReconS_2);
	if (skip_gridding)
	{
	    DTIC(ReconTimer,ReconS_3);
		std::cerr << "Skipping gridding!" << std::endl;
		Fconv.initZeros(); // to remove any stuff from the input volume
		decenter(data, Fconv, max_r2);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			if (DIRECT_MULTIDIM_ELEM(Fweight, n) > 0.)
				DIRECT_MULTIDIM_ELEM(Fconv, n) /= DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		DTOC(ReconTimer,ReconS_3);
	}
	else
	{
		DTIC(ReconTimer,ReconS_4);
		// Divide both data and Fweight by normalisation factor to prevent FFT's with very large values....
#ifdef DEBUG_RECONSTRUCT
		std::cerr << " normalise= " << normalise << std::endl;
#endif
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fweight)
		{
			DIRECT_MULTIDIM_ELEM(Fweight, n) /= normalise;
		}
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data)
		{
			DIRECT_MULTIDIM_ELEM(data, n) /= normalise;
		}
		DTOC(ReconTimer,ReconS_4);
		DTIC(ReconTimer,ReconS_5);
		// Initialise Fnewweight with 1's and 0's. (also see comments below)
		FOR_ALL_ELEMENTS_IN_ARRAY3D(weight)
		{
			if (k * k + i * i + j * j < max_r2)
				A3D_ELEM(weight, k, i, j) = 1.;
			else
				A3D_ELEM(weight, k, i, j) = 0.;
		}
		decenter(weight, Fnewweight, max_r2);
		DTOC(ReconTimer,ReconS_5);
		// Iterative algorithm as in  Eq. [14] in Pipe & Menon (1999)
		// or Eq. (4) in Matej (2001)
		DTIC(ReconTimer,ReconS_25);

		RFLOAT normftblob = tab_ftblob(0.);
        RFLOAT orixpad = ori_size*padding_factor;
        if(ref_dim==2) {
            Xsize = pad_size;
            Ysize = pad_size;
            Zsize = 1;
            size = pad_size*pad_size;
            YXsize=size;
        }
        else {
            Xsize = pad_size;
            Ysize = pad_size;
            Zsize = pad_size;
            YXsize= pad_size*pad_size;
            size = YXsize*pad_size;
		}
		if(host_splitted) {
			printf("*=============== Enter host_splitted ====================*\n");
			REPORT_ERROR("Not implemented");
			DTOC(ReconTimer,ReconS_7);
		} else {
			size_t blknum=(size + BLOCK_SIZE - 1) / BLOCK_SIZE;
			size_t padhdim = pad_size / 2;
			cuFweight.h_ptr=Fweight.data;
			cuFnewweight.h_ptr=Fnewweight.data;
			tabulatedValues.h_ptr=tab_ftblob.tabulatedValues.data;
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			cuFweight.cp_to_device();
			cuFnewweight.cp_to_device();
			tabulatedValues.cp_to_device();
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			for (int iter = 0; iter < max_iter_preweight; iter++)
			{
				DTIC(ReconTimer,ReconS_6);
				// Set Fnewweight * Fweight in the transformer
				// In Matej et al (2001), weights w_P^i are convoluted with the kernel,
				// and the initial w_P^0 are 1 at each sampling point
				// Here the initial weights are also 1 (see initialisation Fnewweight above),
				// but each "sampling point" counts "Fweight" times!
				// That is why Fnewweight is multiplied by Fweight prior to the convolution
				initFconvKernel<<<blknum,BLOCK_SIZE,0,stream>>>(~cutransformer.fouriers,~cuFnewweight,~cuFweight,fsize);
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
				// convolute through Fourier-transform (as both grids are rectangular)
				// Note that convoluteRealSpace acts on the complex array inside the transformer
				// do_mask = false
				cutransformer.clearPlan();
				cutransformer.direction=1;
				cutransformer.setPlan();
				cutransformer.backward();
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
				multFTBlobKernel_noMask<<<blknum,BLOCK_SIZE,0,stream>>>(~(cutransformer.reals), 
							size, YXsize, Xsize, Ysize,
							padhdim,pad_size,padding_factor,
							orixpad,
							normftblob,tab_ftblob.sampling,
							~tabulatedValues,XSIZE(tab_ftblob.tabulatedValues));
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
				cutransformer.clearPlan();
				cutransformer.direction=-1;
				cutransformer.setPlan();
				cutransformer.forward();
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
				ScaleComplexPointwise_kernel <<< blknum, BLOCK_SIZE,0,stream>>>(~cutransformer.fouriers, fsize, size);
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				divFconvKernel<<<blknum,BLOCK_SIZE,0,stream>>>(~cutransformer.fouriers,~cuFnewweight,max_r2,fsize,fxysize,fxsize,fysize,fzsize);
				LAUNCH_HANDLE_ERROR(cudaGetLastError());
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
				DTOC(ReconTimer,ReconS_6);
			}
			cuFnewweight.cp_to_host();
			cutransformer.clearPlan();
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			cuFweight.free_if_set();
			tabulatedValues.free_if_set();
			DTOC(ReconTimer,ReconS_25);

			DTIC(ReconTimer,ReconS_7);
			// Clear memory
			Fweight.clear();
			cutransformer.reals.free_if_set();

			// Note that Fnewweight now holds the approximation of the inverse of the weights on a regular grid

			// Now do the actual reconstruction with the data array
			// Apply the iteratively determined weight
			//Fconv.initZeros(); // to remove any stuff from the input volume
			CudaGlobalPtr<__COMPLEX_T, 0> cudata(stream);
			cudata.setSize(NZYXSIZE(data));
			cudata.device_alloc();
			cudata.host_alloc();
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data)
			{
				cudata[n].x = data.data[n].real;
				cudata[n].y = data.data[n].imag;
			}
			cudata.cp_to_device();
			cudata.streamSync();
			decenter_gpu(~cudata,
						~cutransformer.fouriers,
						~cuFnewweight, 
						max_r2, 
						XSIZE(Fconv),
						YSIZE(Fconv),
						ZSIZE(Fconv),
						XSIZE(data),
						YSIZE(data),
						STARTINGX(data),
						STARTINGY(data),
						STARTINGZ(data),
						stream);
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			cudata.free_if_set();
			cuFnewweight.free_if_set();
			// Clear memory
			Fnewweight.clear();
			DTOC(ReconTimer,ReconS_7);
		} // end host_splitted
	} // end if skip_gridding

	// rather than doing the blob-convolution to downsample the data array, do a windowing operation:
	// This is the same as convolution with a SINC. It seems to give better maps.
	// Then just make the blob look as much as a SINC as possible....
	// The "standard" r1.9, m2 and a15 blob looks quite like a sinc until the first zero (perhaps that's why it is standard?)
	//for (RFLOAT r = 0.1; r < 10.; r+=0.01)
	//{
	//	RFLOAT sinc = sin(PI * r / padding_factor ) / ( PI * r / padding_factor);
	//	std::cout << " r= " << r << " sinc= " << sinc << " blob= " << blob_val(r, blob) << std::endl;
	//}

	// Now do inverse FFT and window to original size in real-space
	// Pass the transformer to prevent making and clearing a new one before clearing the one declared above....
	// The latter may give memory problems as detected by electric fence....
	DTIC(ReconTimer,ReconS_17);
	windowToOridimRealSpace_gpu(rank,transformer,&cutransformer, vol_out, nr_threads, printTimes, host_splitted);
	transformer.fReal=NULL;
	Fconv.clear();
	DTOC(ReconTimer,ReconS_17);

#ifdef DEBUG_RECONSTRUCT
	ttt()=vol_out;
	ttt.write("reconstruct_before_gridding_correction.spi");
#endif

	// Correct for the linear/nearest-neighbour interpolation that led to the data array
#ifdef FORCE_USE_CPU_GRIDDING
	printf("################ USE CPU FOR griddingCorrect!\n");
	DTIC(ReconTimer,ReconS_18);
	griddingCorrect(vol_out);
	DTOC(ReconTimer,ReconS_18);
#else
	DTIC(ReconTimer,ReconS_18);
	cutransformer.reals.h_ptr=vol_out.data;
	// Size of padded real-space volume
	int padoridim = ROUND(padding_factor * ori_size);
	// make sure padoridim is even
	padoridim += padoridim%2;
	int zinit = FIRST_XMIPP_INDEX(ZSIZE(vol_out));
	int yinit = FIRST_XMIPP_INDEX(YSIZE(vol_out));
	int xinit = FIRST_XMIPP_INDEX(XSIZE(vol_out));
	size_t data_size = NZYXSIZE(vol_out);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((data_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
	HANDLE_ERROR(cudaGetLastError());
	if(interpolator == NEAREST_NEIGHBOUR && r_min_nn == 0) {
		griddingCorrect_kernel_1<<<dimGrid,dimBlock,0,stream>>>(  
														 ~cutransformer.reals,
                                                         XSIZE(vol_out), 
                                                         YSIZE(vol_out), 
                                                         ZSIZE(vol_out), 
                                                         xinit,
                                                         yinit,
                                                         zinit, 
                                                         padoridim);
	}
	else if(interpolator == TRILINEAR || (interpolator == NEAREST_NEIGHBOUR && r_min_nn > 0)) {
		griddingCorrect_kernel_2<<<dimGrid,dimBlock,0,stream>>>(  
														 ~cutransformer.reals,
                                                         XSIZE(vol_out), 
                                                         YSIZE(vol_out), 
                                                         ZSIZE(vol_out), 
                                                         xinit,
                                                         yinit,
                                                         zinit,
                                                         padoridim);
	}
	HANDLE_ERROR(cudaGetLastError());
    vol_out.setXmippOrigin();
	cutransformer.reals.streamSync();
	cutransformer.reals.cp_to_host();
	cutransformer.reals.streamSync();
	// cutransformer.reals.free_if_set();
	// cutransformer.fouriers.free_if_set();
	cutransformer.free();
	cutransformer.clear();
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());
	cudaStreamDestroy(stream);
	DTOC(ReconTimer,ReconS_18);
	DTIC(ReconTimer,ReconS_26);
	HANDLE_ERROR(cudaDeviceReset());
	DTOC(ReconTimer,ReconS_26);
#endif
	// If the tau-values were calculated based on the FSC, then now re-calculate the power spectrum of the actual reconstruction
	if (update_tau2_with_fsc)
	{

		// New tau2 will be the power spectrum of the new map
		MultidimArray<RFLOAT> spectrum, count;

		// Calculate this map's power spectrum
		// Don't call getSpectrum() because we want to use the same transformer object to prevent memory trouble....
		DTIC(ReconTimer,ReconS_19);
		spectrum.initZeros(XSIZE(vol_out));
	    count.initZeros(XSIZE(vol_out));
		DTOC(ReconTimer,ReconS_19);
		DTIC(ReconTimer,ReconS_20);
	    // recycle the same transformer for all images
        transformer.setReal(vol_out);
		DTOC(ReconTimer,ReconS_20);
		DTIC(ReconTimer,ReconS_21);
        transformer.FourierTransform();
		DTOC(ReconTimer,ReconS_21);
		DTIC(ReconTimer,ReconS_22);
	    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
	    {
	    	long int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
	    	spectrum(idx) += norm(dAkij(Fconv, k, i, j));
	        count(idx) += 1.;
	    }
	    spectrum /= count;

		// Factor two because of two-dimensionality of the complex plane
		// (just like sigma2_noise estimates, the power spectra should be divided by 2)
		RFLOAT normfft = (ref_dim == 3 && data_dim == 2) ? (RFLOAT)(ori_size * ori_size) : 1.;
		spectrum *= normfft / 2.;

		// New SNR^MAP will be power spectrum divided by the noise in the reconstruction (i.e. sigma2)
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data_vs_prior)
		{
			DIRECT_MULTIDIM_ELEM(tau2, n) =  tau2_fudge * DIRECT_MULTIDIM_ELEM(spectrum, n);
		}
		DTOC(ReconTimer,ReconS_22);
	}
	DTIC(ReconTimer,ReconS_23);
	// Completely empty the transformer object
	transformer.cleanup();
    // Now can use extra mem to move data into smaller array space
    vol_out.shrinkToFit();

	DTOC(ReconTimer,ReconS_23);
#ifdef TIMING
    if(printTimes)
    	ReconTimer.printTimes(true);
#endif
#ifdef DEBUG_RECONSTRUCT
    std::cerr<<"done with reconstruct"<<std::endl;
#endif

}

void BackProjector::windowToOridimRealSpace_gpu(int rank, FourierTransformer &transformer, void* _transformer, MultidimArray<RFLOAT> &Mout, int nr_threads, bool printTimes, bool host_splitted)
{

#ifdef TIMING
	Timer OriDimTimer;
	int OriDim1  = OriDimTimer.setNew(" OrD1_getFourier ");
	int OriDim2  = OriDimTimer.setNew(" OrD2_windowFFT ");
	int OriDim3  = OriDimTimer.setNew(" OrD3_reshape ");
	int OriDim4  = OriDimTimer.setNew(" OrD4_setReal ");
	int OriDim5  = OriDimTimer.setNew(" OrD5_invFFT ");
	int OriDim6  = OriDimTimer.setNew(" OrD6_centerFFT ");
	int OriDim7  = OriDimTimer.setNew(" OrD7_window ");
	int OriDim8  = OriDimTimer.setNew(" OrD8_norm ");
	int OriDim9  = OriDimTimer.setNew(" OrD9_softMask ");
	int OriDim10 = OriDimTimer.setNew(" OrD10_copyToHost ");
#endif
	DTIC(OriDimTimer,OriDim1);
	int Xsize, Ysize, Zsize, size, YXsize, iXsize, iYsize, iZsize;
#ifdef FORCE_NOT_USE_BUFF_FFT
	RelionCudaFFT* cutransformer = (RelionCudaFFT*)_transformer;
#else
	BuffCudaFFT3D* cutransformer = (BuffCudaFFT3D*)_transformer;
#endif
	cudaStream_t stream = cutransformer->fouriers.getStream();
	CudaGlobalPtr<__COMPLEX_T, 0> Ftmp(stream);
	if(host_splitted) {
		REPORT_ERROR("Host splitted Not implemented");
	} else {
		Ftmp.setSize(cutransformer->fouriers.getSize());
		Ftmp.d_ptr=cutransformer->fouriers.d_ptr;
		Ftmp.d_do_free=true;
		cutransformer->fouriers.d_do_free=false;
		cutransformer->fouriers.d_ptr = NULL;
	}
	cutransformer->clear();
	DTOC(OriDimTimer,OriDim1);
	DTIC(OriDimTimer,OriDim2);
	// Size of padded real-space volume
	int padoridim = ROUND(padding_factor * ori_size);
	// make sure padoridim is even
	padoridim += padoridim%2;
	RFLOAT normfft;
	if (ref_dim == 2)
	{
		// only backward
#ifdef FORCE_NOT_USE_BUFF_FFT
		if(cutransformer->setSize(padoridim,padoridim,1,1,1)<0)
#else
		if(true)
#endif
		{
			CRITICAL("TOO LARGE 2D");
		}
		Xsize = padoridim;
		Ysize = padoridim;
		Zsize = 1;
		iXsize= pad_size;
		iYsize= pad_size;
		iZsize= 1;
	}
	else
	{
		// only backward
#ifdef FORCE_NOT_USE_BUFF_FFT
		if(cutransformer->setSize(padoridim,padoridim,padoridim,1,1)<0)
#else
		if(cutransformer->setSize(padoridim,padoridim,padoridim,1,-(sizeof(RFLOAT)*2*Ftmp.getSize()))<0)
#endif
		{
			CRITICAL("TOO LARGE 3D");
		}
		Xsize = padoridim;
		Ysize = padoridim;
		Zsize = padoridim;
		iXsize= pad_size;
		iYsize= pad_size;
		iZsize= pad_size;
	}
	windowFourierTransform_gpu(Ftmp,cutransformer->fouriers, iXsize/2+1, iYsize, iZsize, Xsize/2+1, Ysize, Zsize);
	cutransformer->fouriers.streamSync();
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	Ftmp.free_if_set();
	cutransformer->setPlan(); // make sure there are enough memory for plan
	DTOC(OriDimTimer,OriDim2);
	DTIC(OriDimTimer,OriDim3);
 	if (ref_dim == 2)
	{
		Mout.reshape(padoridim, padoridim);
		normfft = (RFLOAT)(padding_factor * padding_factor);
	}
	else
	{
		Mout.reshape(padoridim, padoridim, padoridim);
		if (data_dim == 3)
			normfft = (RFLOAT)(padding_factor * padding_factor * padding_factor);
		else
			normfft = (RFLOAT)(padding_factor * padding_factor * padding_factor * ori_size);
	}
	Mout.setXmippOrigin();
	DTOC(OriDimTimer,OriDim3);

#ifdef TIMING
	if(printTimes)
		std::cout << std::endl << "FFTrealDims = (" << XSIZE(Mout) << " , " << YSIZE(Mout) << " , " << ZSIZE(Mout) << " ) " << std::endl;
#endif

	// Do the inverse FFT
	DTIC(OriDimTimer,OriDim4);
	cutransformer->fouriers.streamSync();
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	DTOC(OriDimTimer,OriDim4);
	DTIC(OriDimTimer,OriDim5);
	cutransformer->backward();
	cutransformer->reals.streamSync();
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	cutransformer->clearPlan(); // clear plan
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	DTOC(OriDimTimer,OriDim5);
	Mout.setXmippOrigin();

	// Shift the map back to its origin

    DTIC(OriDimTimer,OriDim6);
	fft_runCenterFFT(cutransformer->reals,XSIZE(Mout),YSIZE(Mout),ZSIZE(Mout),true);
	cutransformer->reals.streamSync();
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	DTOC(OriDimTimer,OriDim6);

	// 1. Window in real-space
	// 2. Normalisation factor of FFTW (normfft)
	//    The Fourier Transforms are all "normalised" for 2D transforms of size = ori_size x ori_size
	DTIC(OriDimTimer,OriDim7);
	if (ref_dim==2)
	{
		window_gpu(cutransformer->reals, normfft, FIRST_XMIPP_INDEX(padoridim), FIRST_XMIPP_INDEX(padoridim), 
					LAST_XMIPP_INDEX(padoridim), LAST_XMIPP_INDEX(padoridim), 
					FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size),
					LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size));
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		int size = LAST_XMIPP_INDEX(ori_size) - FIRST_XMIPP_INDEX(ori_size) + 1;
		Mout.resize(size, size);
        Mout.yinit = FIRST_XMIPP_INDEX(ori_size);
        Mout.xinit = FIRST_XMIPP_INDEX(ori_size);
	}
	else
	{
		window_gpu(cutransformer->reals, normfft, FIRST_XMIPP_INDEX(padoridim), FIRST_XMIPP_INDEX(padoridim), FIRST_XMIPP_INDEX(padoridim),
				LAST_XMIPP_INDEX(padoridim), LAST_XMIPP_INDEX(padoridim), LAST_XMIPP_INDEX(padoridim),
				FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size),
				LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size));
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		int size = LAST_XMIPP_INDEX(ori_size) - FIRST_XMIPP_INDEX(ori_size) + 1;
		Mout.resize(size, size, size);
        Mout.zinit = FIRST_XMIPP_INDEX(ori_size);
        Mout.yinit = FIRST_XMIPP_INDEX(ori_size);
        Mout.xinit = FIRST_XMIPP_INDEX(ori_size);
	}
	Mout.setXmippOrigin();
	cutransformer->reals.streamSync();
	DTOC(OriDimTimer,OriDim7);
	// Mask out corners to prevent aliasing artefacts
	DTIC(OriDimTimer,OriDim9);
	if(ref_dim==2)
	{
		fft_runSoftMaskOutsideMap( 	cutransformer->reals, 
									LAST_XMIPP_INDEX(ori_size) - FIRST_XMIPP_INDEX(ori_size) + 1,
									LAST_XMIPP_INDEX(ori_size) - FIRST_XMIPP_INDEX(ori_size) + 1,
									1,
									FIRST_XMIPP_INDEX(ori_size),
									FIRST_XMIPP_INDEX(ori_size),
									0);
	}
	else
	{
		fft_runSoftMaskOutsideMap( 	cutransformer->reals, 
									LAST_XMIPP_INDEX(ori_size) - FIRST_XMIPP_INDEX(ori_size) + 1,
									LAST_XMIPP_INDEX(ori_size) - FIRST_XMIPP_INDEX(ori_size) + 1,
									LAST_XMIPP_INDEX(ori_size) - FIRST_XMIPP_INDEX(ori_size) + 1,
									FIRST_XMIPP_INDEX(ori_size),
									FIRST_XMIPP_INDEX(ori_size),
									FIRST_XMIPP_INDEX(ori_size));
	}
	cutransformer->reals.streamSync();
	LAUNCH_HANDLE_ERROR(cudaGetLastError());
	DTOC(OriDimTimer,OriDim9);
#ifdef FORCE_USE_CPU_GRIDDING
	std::cout << "###################### WARNING ############################" << std::endl;
	DTIC(OriDimTimer,OriDim10);
	cutransformer->reals.h_ptr = Mout.data;
	cutransformer->reals.cp_to_host();
	cutransformer->reals.streamSync();
	cutransformer->clear();
	HANDLE_ERROR(cudaDeviceSynchronize());
	cudaStreamDestroy(stream);
	HANDLE_ERROR(cudaDeviceReset());
	DTOC(OriDimTimer,OriDim10);
#endif
#ifdef TIMING
    if(printTimes)
    	OriDimTimer.printTimes(true);
#endif

}