#include "src/backprojector.h"
#include "src/gpu_utils/cuda_mem_utils.h"
#include "src/gpu_utils/cuda_settings.h"
#include <time.h>
#include "src/gpu_utils/cuda_kernels/recons.cuh"
#include "src/gpu_utils/cuda_fft.h"
#include "src/gpu_utils/cuda_helper_functions.cuh"
#include <assert.h>

#include <omp.h>

#ifndef OOMMM
void BackProjectorMpi::reconstruct_gpu(int rank,
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
	printf("\nEnter reconstruct gpu(%d), node=%p\n",rank,node);
	int devCount,devid;
	cudaGetDeviceCount(&devCount);
	devid = node->slaveRank%devCount;
	cudaSetDevice(devid);
	printf("\nRank %d Bind to device %d\n",node->rank,devid);
	if(ref_dim!=3 || node==NULL || node->grp_size<=1) {
#ifdef TIMING
		printf("Using original reconstruct GPU as parallel resource not configured node=%x\n",node);
		if(node!=NULL)
			printf("==> node->grp_size=%d\n",node->grp_size);
#endif
		BackProjector::reconstruct_gpu(
			rank,
			vol_out,
			max_iter_preweight,
			do_map,
			tau2_fudge,
			tau2,
			sigma2,
			data_vs_prior,
			fourier_coverage,
			fsc,
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
	int ReconS_1 = ReconTimer.setNew(" RcS1_Init_MPI ");
	int ReconS_2 = ReconTimer.setNew(" RcS2_Shape&Noise_MPI ");
	int ReconS_3 = ReconTimer.setNew(" RcS3_skipGridding_MPI ");
	int ReconS_4 = ReconTimer.setNew(" RcS4_doGridding_norm_MPI ");
	int ReconS_5 = ReconTimer.setNew(" RcS5_doGridding_init_MPI ");
	int ReconS_6 = ReconTimer.setNew(" RcS6_doGridding_iter_MPI ");
	int ReconS_7 = ReconTimer.setNew(" RcS7_doGridding_apply_MPI ");
	int ReconS_8 = ReconTimer.setNew(" RcS8_blobConvolute_MPI ");
	int ReconS_9 = ReconTimer.setNew(" RcS9_blobResize_MPI ");
	int ReconS_10 = ReconTimer.setNew(" RcS10_blobSetReal_MPI ");
	int ReconS_11 = ReconTimer.setNew(" RcS11_blobSetTemp_MPI ");
	int ReconS_12 = ReconTimer.setNew(" RcS12_blobTransform_MPI ");
	int ReconS_13 = ReconTimer.setNew(" RcS13_blobCenterFFT_MPI ");
	int ReconS_14 = ReconTimer.setNew(" RcS14_blobNorm1_MPI ");
	int ReconS_15 = ReconTimer.setNew(" RcS15_blobSoftMask_MPI ");
	int ReconS_16 = ReconTimer.setNew(" RcS16_blobNorm2_MPI ");
	int ReconS_17 = ReconTimer.setNew(" RcS17_WindowReal_MPI ");
	int ReconS_18 = ReconTimer.setNew(" RcS18_GriddingCorrect_MPI ");
	int ReconS_19 = ReconTimer.setNew(" RcS19_tauInit_MPI ");
	int ReconS_20 = ReconTimer.setNew(" RcS20_tausetReal_MPI ");
	int ReconS_21 = ReconTimer.setNew(" RcS21_tauTransform_MPI ");
	int ReconS_22 = ReconTimer.setNew(" RcS22_tautauRest_MPI ");
	int ReconS_23 = ReconTimer.setNew(" RcS23_tauShrinkToFit_MPI ");
	int ReconS_24 = ReconTimer.setNew(" RcS24_extra_MPI ");
#endif


	RCTIC(ReconTimer,ReconS_1);
	int sizes[7] = {XSIZE(vol_out),YSIZE(vol_out),ZSIZE(vol_out),XSIZE(tau2),XSIZE(sigma2),XSIZE(data_vs_prior),XSIZE(fourier_coverage)};
	FourierTransformer transformer;
	cudaStream_t stream;
	//DEBUG_HANDLE_ERROR(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	DEBUG_HANDLE_ERROR(cudaStreamCreate(&stream));
	DistributedCudaFFT distributedCudaFFT(node,node->groupC, stream);
	MultidimArray<RFLOAT> Fweight;
	// Fnewweight can become too large for a float: always keep this one in double-precision
	MultidimArray<double> Fnewweight;
	MultidimArray<Complex>& Fconv = transformer.getFourierReference();
	int max_r2 = ROUND(r_max * padding_factor) * ROUND(r_max * padding_factor);

	MultidimArray<RFLOAT> vol_out_ori;
	MultidimArray<RFLOAT> tau2_ori;
    MultidimArray<RFLOAT> sigma2_ori;
    MultidimArray<RFLOAT> data_vs_prior_ori;
    MultidimArray<RFLOAT> fourier_coverage_ori;

	if(node->grp_rank!=0) {
		vol_out_ori = vol_out;
		tau2_ori = tau2;
		sigma2_ori = sigma2;
		data_vs_prior_ori = data_vs_prior;
		fourier_coverage_ori = fourier_coverage;
	}

#ifdef DEBUG_RECONSTRUCT
	Image<RFLOAT> ttt;
	FileName fnttt;
	ttt()=weight;
	ttt.write("reconstruct_initial_weight.spi");
	std::cerr << " pad_size= " << pad_size << " padding_factor= " << padding_factor << " max_r2= " << max_r2 << std::endl;
#endif

    // Set Fweight, Fnewweight and Fconv to the right size
    if (ref_dim == 2) {
        vol_out.setDimensions(pad_size, pad_size, 1, 1);
		//distributed_transformer.setSize(pad_size, pad_size, 1);
		distributedCudaFFT.setSize(pad_size, pad_size, 1);
	}
    else {
        // Too costly to actually allocate the space
        // Trick transformer with the right dimensions
        vol_out.setDimensions(pad_size, pad_size, pad_size, 1);
		//distributed_transformer.setSize(pad_size, pad_size, pad_size);
		distributedCudaFFT.setSize(pad_size, pad_size, pad_size);
	}
    transformer.setReal(vol_out); // Fake set real. 1. Allocate space for Fconv 2. calculate plans.
    vol_out.clear(); // Reset dimensions to 0
    RCTOC(ReconTimer,ReconS_1);
    RCTIC(ReconTimer,ReconS_2);
    Fweight.reshape(Fconv);
    if (!skip_gridding)
    	Fnewweight.reshape(Fconv);
	if(node->grp_rank==0) {
		// Go from projector-centered to FFTW-uncentered
		// Only master of group do this work
		decenter(weight, Fweight, max_r2);	
	}
	// broadcast
	node->relion_MPI_Bcast(MULTIDIM_ARRAY(Fweight), MULTIDIM_SIZE(Fweight), MY_MPI_DOUBLE, 0, node->groupC);
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
    RCTOC(ReconTimer,ReconS_2);
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
    RCTOC(ReconTimer,ReconS_2);
	if (skip_gridding)
	{
	    RCTIC(ReconTimer,ReconS_3);
		std::cerr << "Skipping gridding!" << std::endl;
		Fconv.initZeros(); // to remove any stuff from the input volume
		decenter(data, Fconv, max_r2);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			if (DIRECT_MULTIDIM_ELEM(Fweight, n) > 0.)
				DIRECT_MULTIDIM_ELEM(Fconv, n) /= DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		RCTOC(ReconTimer,ReconS_3);
	}
	else
	{
		RCTIC(ReconTimer,ReconS_4);
		// Divide both data and Fweight by normalisation factor to prevent FFT's with very large values....
	#ifdef DEBUG_RECONSTRUCT
		std::cerr << " normalise= " << normalise << std::endl;
	#endif
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fweight)
		{
			DIRECT_MULTIDIM_ELEM(Fweight, n) /= normalise;
		}
		if(node->grp_rank==0) {
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data)
			{
				DIRECT_MULTIDIM_ELEM(data, n) /= normalise;
			}
		}
		RCTOC(ReconTimer,ReconS_4);
		RCTIC(ReconTimer,ReconS_5);
		// Initialise Fnewweight with 1's and 0's. (also see comments below)
		if(node->grp_rank==0) {
			FOR_ALL_ELEMENTS_IN_ARRAY3D(weight)
			{
				if (k * k + i * i + j * j < max_r2)
					A3D_ELEM(weight, k, i, j) = 1.;
				else
					A3D_ELEM(weight, k, i, j) = 0.;
			}
			decenter(weight, Fnewweight, max_r2);
		}
		// broadcast
		node->relion_MPI_Bcast(MULTIDIM_ARRAY(Fnewweight), MULTIDIM_SIZE(Fnewweight), MPI_DOUBLE, 0, node->groupC);
		printf("\n### Begin initialising cuda(%d) ###",node->rank);
		CudaGlobalPtr<double, false> cuFnewweight(stream);
		CudaGlobalPtr<RFLOAT, false> cuFweight(stream);
		CudaGlobalPtr<double, false> cubuff(stream);
		CudaGlobalPtr<RFLOAT, false> tabulatedValues(stream); 
		CudaGlobalPtr<__COMPLEX_T, false>& cuFconv = distributedCudaFFT.getDistributedFourierReference();
		CudaGlobalPtr<RFLOAT, false>& cuFreal = distributedCudaFFT.getDistributedRealReference();
		// set size
		printf("\n### distribute location(%d) ###",node->rank);
		distributedCudaFFT.distribute_location();
		printf("\n### set correct sizes(%d) ###",node->rank);
		int fxsize = distributedCudaFFT.xFSize;
		int fysize = distributedCudaFFT.yFSize;
		int fzsize = distributedCudaFFT.zFSize;
		int lysize = distributedCudaFFT.lyFSize;
		int lystart= distributedCudaFFT.lyFStart;
		int fsize  = fxsize*fysize*fzsize;
		int lfsize = fxsize*lysize*fzsize;
		int lfstart= fxsize*lystart*fzsize;
		int xsize  = distributedCudaFFT.xSize;
		int ysize  = distributedCudaFFT.ySize;
		int zsize  = distributedCudaFFT.zSize;
		int lzsize = distributedCudaFFT.lzSize;
		int lzstart= distributedCudaFFT.lzStart;
		int size   = xsize*ysize*zsize;
		int lsize  = xsize*ysize*lzsize;
		int lstart = xsize*ysize*lzstart;
		cuFnewweight.device_alloc(fsize);
		printf("\n### cuFnewweight allocated(%d) ###",node->rank);
		cuFweight.device_alloc(fsize);
		printf("\n### cuFweight allocated(%d) ###",node->rank);
		cubuff.device_alloc(fsize);
		printf("\n### cubuff allocated(%d) ###",node->rank);
		tabulatedValues.device_alloc(XSIZE(tab_ftblob.tabulatedValues));
		printf("\n### tabulated values allocated(%d) ###",node->rank);
		// set host ptr
		cuFnewweight.h_ptr = Fnewweight.data;
		cuFweight.h_ptr = Fweight.data;
		tabulatedValues.h_ptr=tab_ftblob.tabulatedValues.data;
		tabulatedValues.cp_to_device();
		printf("\n### tabulated Values copy to device(%d) ###",node->rank);
		// data transfer and transpose
		cubuff.h_ptr = Fnewweight.data;
		cubuff.cp_to_device();
		distributedCudaFFT.transpose_gpu(~cuFnewweight,~cubuff,fxsize*fysize,fzsize);
		cuFnewweight.streamSync();
		printf("\n### cuFnewweight transposed(%d) ###",node->rank);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());
		cubuff.h_ptr = Fweight.data;
		cubuff.cp_to_device();
		distributedCudaFFT.transpose_gpu(~cuFweight,(RFLOAT*)~cubuff,fxsize*fysize,fzsize);
		cuFnewweight.streamSync();
		printf("\n### cuFweight transposed(%d) ###",node->rank);
		LAUNCH_HANDLE_ERROR(cudaGetLastError());

		RCTOC(ReconTimer,ReconS_5);
		// Iterative algorithm as in  Eq. [14] in Pipe & Menon (1999)
		// or Eq. (4) in Matej (2001)
		int padhdim = pad_size / 2;
		RFLOAT normftblob = tab_ftblob(0.);
		RFLOAT orixpad = ori_size*padding_factor;
		printf("\n### begin iter(%d) ###",node->rank);
		for (int iter = 0; iter < max_iter_preweight; iter++)
		{
			RCTIC(ReconTimer,ReconS_6);
			// Set Fnewweight * Fweight in the transformer
			// In Matej et al (2001), weights w_P^i are convoluted with the kernel,
			// and the initial w_P^0 are 1 at each sampling point
			// Here the initial weights are also 1 (see initialisation Fnewweight above),
			// but each "sampling point" counts "Fweight" times!
			// That is why Fnewweight is multiplied by Fweight prior to the convolution
			// int fzsize = distributed_transformer.zdim;
			// int fysize = distributed_transformer.ydim;
			// int fxsize = distributed_transformer.xdim / 2 + 1;
			// int xysize = fxsize * fysize;
			// for(int k=0;k<distributed_transformer.local_n0;++k)
			// 	for(int i=0;i<fysize;++i)
			// 		for(int j=0;j<fxsize;++j) 
			// {
			// 	int kk = k+distributed_transformer.local_0_start;
			// 	distributed_transformer.fFourier_local[k*xysize+i*fxsize+j] = 
			// 		DIRECT_A3D_ELEM(Fnewweight, kk, i, j) * DIRECT_A3D_ELEM(Fweight, kk, i, j);
			// }
			printf("\n### enter iter %d (%d) ###",iter,node->rank);
			int blknum = (lfsize + BLOCK_SIZE - 1) / BLOCK_SIZE;
			initFconvKernel<<<blknum,BLOCK_SIZE,0,stream>>>(~cuFconv,&(cuFnewweight.d_ptr[lfstart]),&(cuFweight.d_ptr[lfstart]),lfsize);
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			// convolute through Fourier-transform (as both grids are rectangular)
			// Note that convoluteRealSpace acts on the complex array inside the transformer
			//convoluteBlobRealSpace(distributed_transformer);
			printf("\n### cuFconv inited(%d) ###",node->rank);
			distributedCudaFFT.backward_notrans();
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			printf("\n### backward finish(%d) ###",node->rank);
			blknum = (lsize + BLOCK_SIZE - 1) / BLOCK_SIZE;
			multFTBlobKernel_noMask_zsplit<<<blknum,BLOCK_SIZE,0,stream>>>(~cuFreal, lzstart, 
							lsize, xsize*ysize, xsize, ysize,
							padhdim,pad_size,padding_factor,
							orixpad,
							normftblob,tab_ftblob.sampling,
							~tabulatedValues,XSIZE(tab_ftblob.tabulatedValues));
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
			printf("\n### FTBlobl finished(%d) ###",node->rank);
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			distributedCudaFFT.forward_notrans();
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
			printf("\n### forward finish(%d) ###",node->rank);
			blknum = (lfsize + BLOCK_SIZE - 1) / BLOCK_SIZE;
			divFconvKernel_ysplit<<<blknum,BLOCK_SIZE,0,stream>>>(~cuFconv,~cuFnewweight,max_r2,lfsize,fxsize*fzsize,fxsize,fysize,fzsize,lystart);
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			printf("\n### div finished(%d) ###",node->rank);
			// RFLOAT w, corr_min = LARGE_NUMBER, corr_max = -LARGE_NUMBER, corr_avg=0., corr_nn=0.;
			// for (long int k = 0, kp = distributed_transformer.local_0_start; k<distributed_transformer.local_n0; k++, kp = (kp+1 < fxsize) ? kp+1 : kp+1 - fzsize)
    		// 	for (long int i = 0, ip = 0 ; i<fysize; i++, ip = (i < fxsize) ? i : i - fysize)
    		// 		for (long int j = 0, jp = 0; j<fxsize; j++, jp = j)
			// {
			// 	if (kp * kp + ip * ip + jp * jp < max_r2) {
			// 		// Make sure no division by zero can occur....
			// 		w = XMIPP_MAX(1e-6, abs(distributed_transformer.fFourier_local[k*xysize+i*fxsize+j]));
			// 		// Apply division of Eq. [14] in Pipe & Menon (1999)
			// 		DIRECT_A3D_ELEM(Fnewweight, k+distributed_transformer.local_0_start, i, j) /= w;
			// 	}
			// }
			RCTOC(ReconTimer,ReconS_6);
	#ifdef DEBUG_RECONSTRUCT
			std::cerr << " PREWEIGHTING ITERATION: "<< iter + 1 << " OF " << max_iter_preweight << std::endl;
			// report of maximum and minimum values of current conv_weight
			std::cerr << " corr_avg= " << corr_avg / corr_nn << std::endl;
			std::cerr << " corr_min= " << corr_min << std::endl;
			std::cerr << " corr_max= " << corr_max << std::endl;
	#endif
		}
		printf("\n### iterative algo finished(%d) ###",node->rank);
		distributedCudaFFT.mergeThis_ysplit(~cuFnewweight);
		printf("\n### mergeThis finished(%d) ###",node->rank);
		if(node->grp_rank==0) {
			distributedCudaFFT.transpose_gpu(~cubuff,~cuFnewweight,fzsize,fxsize*fysize);
			cubuff.h_ptr = Fnewweight.data;
			cubuff.cp_to_host();
			cubuff.streamSync();
			printf("\n### transpose back finished(%d) ###",node->rank);
		}
		MPI_Barrier(node->groupC);

		RCTIC(ReconTimer,ReconS_7);
	#ifdef DEBUG_RECONSTRUCT
		Image<double> tttt;
		tttt()=Fnewweight;
		tttt.write("reconstruct_gridding_weight.spi");
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			DIRECT_MULTIDIM_ELEM(ttt(), n) = abs(DIRECT_MULTIDIM_ELEM(Fconv, n));
		}
		ttt.write("reconstruct_gridding_correction_term.spi");
	#endif

		// Clear memory
		Fweight.clear();

		// Note that Fnewweight now holds the approximation of the inverse of the weights on a regular grid

		// Now do the actual reconstruction with the data array
		// Apply the iteratively determined weight
		Fconv.initZeros(); // to remove any stuff from the input volume
		if(node->grp_rank==0) {
			// Go from projector-centered to FFTW-uncentered
			// Only master of group do this work
			decenter(data, Fconv, max_r2);
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
			{
	#ifdef  RELION_SINGLE_PRECISION
				// Prevent numerical instabilities in single-precision reconstruction with very unevenly sampled orientations
				if (DIRECT_MULTIDIM_ELEM(Fnewweight, n) > 1e20)
					DIRECT_MULTIDIM_ELEM(Fnewweight, n) = 1e20;
	#endif
				DIRECT_MULTIDIM_ELEM(Fconv, n) *= DIRECT_MULTIDIM_ELEM(Fnewweight, n);
			}
		}
		// broadcast
		node->relion_MPI_Bcast(MULTIDIM_ARRAY(Fconv), MULTIDIM_SIZE(Fconv), MY_MPI_COMPLEX, 0, node->groupC);
		// Clear memory
		Fnewweight.clear();
		RCTOC(ReconTimer,ReconS_7);
	} // end if skip_gridding

	// Now do inverse FFT and window to original size in real-space
	// Pass the transformer to prevent making and clearing a new one before clearing the one declared above....
	// The latter may give memory problems as detected by electric fence....
	RCTIC(ReconTimer,ReconS_17);
	DistributedFourierTransformer distributed_transformer(node,node->groupC, nr_threads);
	distributed_transformer.setFourier(Fconv);
	windowToOridimRealSpace(distributed_transformer, vol_out, nr_threads, printTimes);
	transformer.fReal = NULL; // Make sure to re-calculate fftw plan
	RCTOC(ReconTimer,ReconS_17);
	if(distributed_transformer.rank==0) {
#ifdef DEBUG_RECONSTRUCT
		ttt()=vol_out;
		ttt.write("reconstruct_before_gridding_correction.spi");
#endif
		// Correct for the linear/nearest-neighbour interpolation that led to the data array
		RCTIC(ReconTimer,ReconS_18);
		griddingCorrect(vol_out);
		RCTOC(ReconTimer,ReconS_18);
		// If the tau-values were calculated based on the FSC, then now re-calculate the power spectrum of the actual reconstruction
		if (update_tau2_with_fsc)
		{

			// New tau2 will be the power spectrum of the new map
			MultidimArray<RFLOAT> spectrum, count;

			// Calculate this map's power spectrum
			// Don't call getSpectrum() because we want to use the same transformer object to prevent memory trouble....
			RCTIC(ReconTimer,ReconS_19);
			spectrum.initZeros(XSIZE(vol_out));
			count.initZeros(XSIZE(vol_out));
			RCTOC(ReconTimer,ReconS_19);
			RCTIC(ReconTimer,ReconS_20);
			// recycle the same transformer for all images
			transformer.setReal(vol_out);
			RCTOC(ReconTimer,ReconS_20);
			RCTIC(ReconTimer,ReconS_21);
			transformer.FourierTransform();
			RCTOC(ReconTimer,ReconS_21);
			RCTIC(ReconTimer,ReconS_22);
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
			RCTOC(ReconTimer,ReconS_22);
		}
		RCTIC(ReconTimer,ReconS_23);
		// Completely empty the transformer object
		transformer.cleanup();
		// Now can use extra mem to move data into smaller array space
		vol_out.shrinkToFit();
		RCTOC(ReconTimer,ReconS_23);
#ifdef TIMING
    	if(printTimes)
    		ReconTimer.printTimes(true);
#endif
	} else {
		vol_out = vol_out_ori;
		tau2 = tau2_ori;
		sigma2 = sigma2_ori;
		data_vs_prior = data_vs_prior_ori;
		fourier_coverage = fourier_coverage_ori;
		vol_out.shrinkToFit();
	}
	distributed_transformer.cleanup();
#ifdef DEBUG_RECONSTRUCT
    std::cerr<<"done with reconstruct"<<std::endl;
#endif

}
#else
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
	if(node==NULL || node->grp_size<=1) {
		printf("\n=== Warning : using single GPU reconstruction due to group size is less than or equal to 1.\n");
		Backprojector::reconstruct_gpu( rank,
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
	//int dev_id=(rank+devCount-1)%devCount;
	int dev_id=node->slaveRank%devCount;
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

	DistributedCudaFFT cutransformer(node,node->groupC, stream);
	
	CudaGlobalPtr<RFLOAT, false> cuFweight(stream);
	CudaGlobalPtr<double, false> cuFnewweight(stream);
	CudaGlobalPtr<RFLOAT, false> tabulatedValues(stream);

	printf("pad_size=%d, ref_dim=%d, r_max=%d, normalise=%lf\n",pad_size,ref_dim,r_max,normalise);

    // Set Fweight, Fnewweight and Fconv to the right size
    // Only support for ref_dim==3
    // Too costly to actually allocate the space
    // Trick transformer with the right dimensions
	vol_out.setDimensions(pad_size, pad_size, pad_size, 1);
	Fconv.reshape(pad_size,pad_size,pad_size/2+1);

	bool host_splitted = false;

	if(cutransformer.setSize(pad_size, pad_size, pad_size)<0) {
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
			cutransformer.distribute_location();
			size_t fxsize = distributedCudaFFT.xFSize;
			size_t fysize = distributedCudaFFT.yFSize;
			size_t fzsize = distributedCudaFFT.zFSize;
			size_t lysize = distributedCudaFFT.lyFSize;
			size_t lystart= distributedCudaFFT.lyFStart;
			size_t fsize  = fxsize*fysize*fzsize;
			size_t lfsize = fxsize*lysize*fzsize;
			size_t lfstart= fxsize*lystart*fzsize;
			size_t xsize  = distributedCudaFFT.xSize;
			size_t ysize  = distributedCudaFFT.ySize;
			size_t zsize  = distributedCudaFFT.zSize;
			size_t lzsize = distributedCudaFFT.lzSize;
			size_t lzstart= distributedCudaFFT.lzStart;
			size_t size   = xsize*ysize*zsize;
			size_t lsize  = xsize*ysize*lzsize;
			size_t lstart = xsize*ysize*lzstart;
			cuFweight.setSize(fsize);
			cuFnewweight.setSize(fsize);
			tabulatedValues.setSize(XSIZE(tab_ftblob.tabulatedValues));
			cuFweight.device_alloc();
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			cuFnewweight.device_alloc();
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
			tabulatedValues.device_alloc();
			LAUNCH_HANDLE_ERROR(cudaGetLastError());
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
#endif