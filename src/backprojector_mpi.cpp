/***************************************************************************
 *
 * Author: "Xin You"
 * Beihang University
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
//#define EXTRACT_RECONSTRUCT_INPUTS
#ifdef EXTRACT_RECONSTRUCT_INPUTS
#include <fstream>
#endif
#include "src/backprojector_mpi.h"
#include "src/pfftw.h"
#define TIMING
#ifdef TIMING
	#define RCTIC(timer,label) (timer.tic(label))
    #define RCTOC(timer,label) (timer.toc(label))
#else
	#define RCTIC(timer,label)
    #define RCTOC(timer,label)
#endif

void BackProjectorMpi::reconstruct(MultidimArray<RFLOAT> &vol_out,
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

#ifdef EXTRACT_RECONSTRUCT_INPUTS
{
	int sizes[7] = {XSIZE(vol_out),YSIZE(vol_out),ZSIZE(vol_out),XSIZE(tau2),XSIZE(sigma2),XSIZE(data_vs_prior),XSIZE(fourier_coverage)};
	int max_r2 = ROUND(r_max * padding_factor) * ROUND(r_max * padding_factor);
	std::cerr << "Input sizes " << sizes[0] << " " << sizes[1] << " " << sizes[2] << " " << sizes[3] << " " << sizes[4] << " " << sizes[5] << " " << sizes[6] << " " << std::endl; 
	std::cerr << "Will Extract reconstruct inputs and outputs" << std::endl;
	std::cerr << "Print initial weight" << std::endl;
	char fn[200];
	sprintf(fn, "reconstruct_initial_weight.%d.spi",node->rank);
	weight.writeBinary(fn);
	std::cerr << "Print initial data" << std::endl;
	sprintf(fn, "reconstruct_initial_data.%d.spi",node->rank);
	data.writeBinary(fn);
	std::cerr << "Print initial fsc" << std::endl;
	if(NZYXSIZE(fsc)<=0) {
		std::cerr << "Empty" << std::endl;
	}
	else {
		sprintf(fn, "reconstruct_initial_fsc.%d.spi",node->rank);
		fsc.writeBinary(fn);
	}
	std::cerr << "Print initial tau2" << std::endl;
	if(NZYXSIZE(tau2)<=0) {
		std::cerr << "Empty" << std::endl;
	}
	else {
		sprintf(fn, "reconstruct_initial_tau2.%d.spi",node->rank);
		tau2.writeBinary(fn);
	}
	std::cerr << "Print initial sigma2" << std::endl;
	if(NZYXSIZE(sigma2)<=0) {
		std::cerr << "Empty" << std::endl;
	}
	else {
		sprintf(fn, "reconstruct_initial_sigma2.%d.spi",node->rank);
		sigma2.writeBinary(fn);
	}
	std::cerr << "Print initial data_vs_prior" << std::endl;
	if(NZYXSIZE(data_vs_prior)<=0) {
		std::cerr << "Empty" << std::endl;
	}
	else {
		sprintf(fn, "reconstruct_initial_data_vs_prior.%d.spi",node->rank);
		data_vs_prior.writeBinary(fn);
	}
	std::cerr << "Print initial fourier_coverage" << std::endl;
	if(NZYXSIZE(fourier_coverage)<=0) {
		std::cerr << "Empty" << std::endl;
	}
	else {
		sprintf(fn, "reconstruct_initial_fourier_coverage.%d.spi",node->rank);
		fourier_coverage.writeBinary(fn);
	}
	sprintf(fn, "reconstruct_parameter.%d.%d.dat",node->rank,node->grp_size);
	std::ofstream fout(fn);
	fout << (NZYXSIZE(fsc)>=0) << " " << (NZYXSIZE(tau2)>=0) << " " << (NZYXSIZE(sigma2)>=0) << " " << (NZYXSIZE(data_vs_prior)>=0) << " " << (NZYXSIZE(fourier_coverage)>=0) << std::endl;
	fout << " pad_size= " << pad_size << " padding_factor= " << padding_factor << " max_r2= " << max_r2 << std::endl;
	fout << " ori_size= " << ori_size << " do_map= " << do_map << " max_iter_preweight= " << max_iter_preweight << " tau2_fudge= " << tau2_fudge << std::endl;
	fout << " normalise= " << normalise << " update_tau2_with_fsc= " << update_tau2_with_fsc << " is_whole_instead_of_half= " << is_whole_instead_of_half << " nr_threads= " << nr_threads << std::endl;
	fout << " minres_map= " << minres_map << " printTimes= " << printTimes << std::endl;
	fout.close();
	printf("node group rank %d, node cls rank %d, node rank %d\n",node->grp_rank, node->cls_rank, node->rank);
}
#endif

	if(node==NULL || node->grp_size<=1) {
#ifdef TIMING
		printf("Using original reconstruct as parallel resource not configured node=%x\n",node);
		if(node!=NULL)
			printf("==> node->grp_size=%d\n",node->grp_size);
#endif
		BackProjector::reconstruct(
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
	DistributedFourierTransformer distributed_transformer(node,node->groupC, nr_threads);
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
		distributed_transformer.setSize(pad_size, pad_size, 1);
	}
    else {
        // Too costly to actually allocate the space
        // Trick transformer with the right dimensions
        vol_out.setDimensions(pad_size, pad_size, pad_size, 1);
		distributed_transformer.setSize(pad_size, pad_size, pad_size);
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
		RCTOC(ReconTimer,ReconS_5);
		// Iterative algorithm as in  Eq. [14] in Pipe & Menon (1999)
		// or Eq. (4) in Matej (2001)
		for (int iter = 0; iter < max_iter_preweight; iter++)
		{
			RCTIC(ReconTimer,ReconS_6);
			// Set Fnewweight * Fweight in the transformer
			// In Matej et al (2001), weights w_P^i are convoluted with the kernel,
			// and the initial w_P^0 are 1 at each sampling point
			// Here the initial weights are also 1 (see initialisation Fnewweight above),
			// but each "sampling point" counts "Fweight" times!
			// That is why Fnewweight is multiplied by Fweight prior to the convolution
			int fzsize = distributed_transformer.zdim;
			int fysize = distributed_transformer.ydim;
			int fxsize = distributed_transformer.xdim / 2 + 1;
			int xysize = fxsize * fysize;
			for(int k=0;k<distributed_transformer.local_n0;++k)
				for(int i=0;i<fysize;++i)
					for(int j=0;j<fxsize;++j) 
			{
				int kk = k+distributed_transformer.local_0_start;
				distributed_transformer.fFourier_local[k*xysize+i*fxsize+j] = 
					DIRECT_A3D_ELEM(Fnewweight, kk, i, j) * DIRECT_A3D_ELEM(Fweight, kk, i, j);
			}
			// convolute through Fourier-transform (as both grids are rectangular)
			// Note that convoluteRealSpace acts on the complex array inside the transformer
			convoluteBlobRealSpace(distributed_transformer);

			RFLOAT w, corr_min = LARGE_NUMBER, corr_max = -LARGE_NUMBER, corr_avg=0., corr_nn=0.;
			for (long int k = 0, kp = (distributed_transformer.local_0_start<fxsize?distributed_transformer.local_0_start:distributed_transformer.local_0_start-fzsize); k<distributed_transformer.local_n0; k++, kp = (kp+1 < fxsize) ? kp+1 : kp+1 - fzsize)
    			for (long int i = 0, ip = 0 ; i<fysize; i++, ip = (i < fxsize) ? i : i - fysize)
    				for (long int j = 0, jp = 0; j<fxsize; j++, jp = j)
			{
				if (kp * kp + ip * ip + jp * jp < max_r2) {
					// Make sure no division by zero can occur....
					w = XMIPP_MAX(1e-6, abs(distributed_transformer.fFourier_local[k*xysize+i*fxsize+j]));
					// Apply division of Eq. [14] in Pipe & Menon (1999)
					DIRECT_A3D_ELEM(Fnewweight, k+distributed_transformer.local_0_start, i, j) /= w;
				}
			}
			RCTOC(ReconTimer,ReconS_6);
	#ifdef DEBUG_RECONSTRUCT
			std::cerr << " PREWEIGHTING ITERATION: "<< iter + 1 << " OF " << max_iter_preweight << std::endl;
			// report of maximum and minimum values of current conv_weight
			std::cerr << " corr_avg= " << corr_avg / corr_nn << std::endl;
			std::cerr << " corr_min= " << corr_min << std::endl;
			std::cerr << " corr_max= " << corr_max << std::endl;
	#endif
		}

		distributed_transformer.distribute_location();
		int sz = YXSIZE(Fnewweight);
		if(distributed_transformer.rank==0) {
			MPI_Status status;
			for(int i=1;i<distributed_transformer.size;++i) {
	#ifdef RELION_SINGLE_PRECISION
				node->relion_MPI_Recv(&(DIRECT_A3D_ELEM(Fnewweight,distributed_transformer.dLocInfo[i*2],0,0)), distributed_transformer.dLocInfo[i*2+1]*sz, MPI_FLOAT, i, 100, distributed_transformer.comm, status);
	#else
				node->relion_MPI_Recv(&(DIRECT_A3D_ELEM(Fnewweight,distributed_transformer.dLocInfo[i*2],0,0)), distributed_transformer.dLocInfo[i*2+1]*sz, MPI_DOUBLE, i, 100, distributed_transformer.comm, status);
	#endif
			}
		} else {
			MPI_Request request;
			MPI_Status status;
	#ifdef RELION_SINGLE_PRECISION
			node->relion_MPI_Send(&(DIRECT_A3D_ELEM(Fnewweight,distributed_transformer.local_0_start,0,0)), distributed_transformer.local_n0*sz, MPI_FLOAT, 0, 100, distributed_transformer.comm);
	#else
			node->relion_MPI_Send(&(DIRECT_A3D_ELEM(Fnewweight,distributed_transformer.local_0_start,0,0)), distributed_transformer.local_n0*sz, MPI_DOUBLE, 0, 100, distributed_transformer.comm);
	#endif
		}
		MPI_Barrier(distributed_transformer.comm);

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

void BackProjectorMpi::convoluteBlobRealSpace(DistributedFourierTransformer& transformer, bool do_mask)
{
	int padhdim = pad_size / 2;
	// inverse FFT
	transformer.inverseFourierTransform();

	// Blob normalisation in Fourier space
	RFLOAT normftblob = tab_ftblob(0.);

	// Multiply with FT of the blob kernel
	long int kend = transformer.local_0_start+transformer.local_n0;
	long int fxdim = 2*(transformer.xdim/2+1);
	long int xydim = fxdim * transformer.zdim;
	for (long int k=0; k<transformer.local_n0; k++)
        for (long int i=0; i<transformer.ydim; i++)
			for (long int j=0; j<transformer.xdim; j++)
    {
		int kk = k + transformer.local_0_start;
		int kp = (kk< padhdim) ? kk: kk- pad_size;
		int ip = (i < padhdim) ? i : i - pad_size;
		int jp = (j < padhdim) ? j : j - pad_size;
    	RFLOAT rval = sqrt ( (RFLOAT)(kp * kp + ip * ip + jp * jp) ) / (ori_size * padding_factor);
    	// In the final reconstruction: mask the real-space map beyond its original size to prevent aliasing ghosts
    	// Note that rval goes until 1/2 in the oversampled map
    	if (do_mask && rval > 1./(2. * padding_factor))
    		//DIRECT_A3D_ELEM(Mconv, k, i, j) = 0.;
			transformer.fReal_local[k*xydim+i*fxdim+j] = 0.;
    	else
			transformer.fReal_local[k*xydim+i*fxdim+j]*= (tab_ftblob(rval) / normftblob);
    }

    // forward FFT to go back to Fourier-space
    transformer.FourierTransform();
}

void BackProjectorMpi::windowToOridimRealSpace(DistributedFourierTransformer &transformer, MultidimArray<RFLOAT> &Mout, int nr_threads, bool printTimes)
{

#ifdef TIMING
	Timer OriDimTimer;
	int OriDim1  = OriDimTimer.setNew(" OrD1_getFourier_MPI ");
	int OriDim2  = OriDimTimer.setNew(" OrD2_windowFFT_MPI ");
	int OriDim3  = OriDimTimer.setNew(" OrD3_reshape_MPI ");
	int OriDim4  = OriDimTimer.setNew(" OrD4_setReal_MPI ");
	int OriDim5  = OriDimTimer.setNew(" OrD5_invFFT_MPI ");
	int OriDim5_1  = OriDimTimer.setNew(" OrD5_1_merge_MPI ");
	int OriDim5_2  = OriDimTimer.setNew(" OrD5_2_copy_MPI ");
	int OriDim6  = OriDimTimer.setNew(" OrD6_centerFFT_MPI ");
	int OriDim7  = OriDimTimer.setNew(" OrD7_window_MPI ");
	int OriDim8  = OriDimTimer.setNew(" OrD8_norm_MPI ");
	int OriDim9  = OriDimTimer.setNew(" OrD9_softMask_MPI ");
#endif

	RCTIC(OriDimTimer,OriDim1);
	MultidimArray<Complex>& Fin = transformer.getFourierReference();
	RCTOC(OriDimTimer,OriDim1);
	RCTIC(OriDimTimer,OriDim2);
	MultidimArray<Complex > Ftmp;
	// Size of padded real-space volume
	int padoridim = ROUND(padding_factor * ori_size);
	// make sure padoridim is even
	padoridim += padoridim%2;
	RFLOAT normfft;

#ifdef DEBUG_WINDOWORIDIMREALSPACE
	Image<RFLOAT> tt;
	tt().reshape(ZSIZE(Fin), YSIZE(Fin), XSIZE(Fin));
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fin)
	{
		DIRECT_MULTIDIM_ELEM(tt(), n) = abs(DIRECT_MULTIDIM_ELEM(Fin, n));
	}
	tt.write("windoworidim_Fin.spi");
#endif

    // Resize incoming complex array to the correct size
	//Fin.printShape();
	//printf("Node %d : padoridim=%d\n",node->rank,padoridim);
	windowFourierTransform(Fin, padoridim);
	RCTOC(OriDimTimer,OriDim2);
	RCTIC(OriDimTimer,OriDim3);
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
	RCTOC(OriDimTimer,OriDim3);

#ifdef DEBUG_WINDOWORIDIMREALSPACE
	tt().reshape(ZSIZE(Fin), YSIZE(Fin), XSIZE(Fin));
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fin)
	{
		DIRECT_MULTIDIM_ELEM(tt(), n) = abs(DIRECT_MULTIDIM_ELEM(Fin, n));
	}
	tt.write("windoworidim_Fresized.spi");
#endif

	// Do the inverse FFT
	RCTIC(OriDimTimer,OriDim4);
    //transformer.setReal(Mout);
	transformer.setSize(XSIZE(Mout),YSIZE(Mout),ZSIZE(Mout));
	transformer.distribute(DISTRIBUTE_FOURIER);
	RCTOC(OriDimTimer,OriDim4);
	RCTIC(OriDimTimer,OriDim5);
#ifdef TIMING
	if(printTimes)
		std::cout << std::endl << "FFTrealDims = (" << transformer.fReal->xdim << " , " << transformer.fReal->ydim << " , " << transformer.fReal->zdim << " ) " << std::endl;
#endif
	
	transformer.inverseFourierTransform();
	
	//std::cout << std::endl << transformer.rank << " Inverse finished" << std::endl;

	RCTOC(OriDimTimer,OriDim5);
	RCTIC(OriDimTimer,OriDim5_1);
	//std::cout << std::endl << transformer.rank << " Begin Merging..." << std::endl;
	transformer.mergeTo(Mout);
	RCTOC(OriDimTimer,OriDim5_1);
	if(transformer.rank==0) {
		RCTIC(OriDimTimer,OriDim5_2);
    	Fin.clear();
		delete transformer.fReal;
    	transformer.fReal = NULL; // Make sure to re-calculate fftw plan
		Mout.setXmippOrigin();
		RCTOC(OriDimTimer,OriDim5_2);
		// Shift the map back to its origin
		RCTIC(OriDimTimer,OriDim6);
		CenterFFT(Mout,true);
		RCTOC(OriDimTimer,OriDim6);
#ifdef DEBUG_WINDOWORIDIMREALSPACE
		tt()=Mout;
		tt.write("windoworidim_Munwindowed.spi");
#endif

		// Window in real-space
		RCTIC(OriDimTimer,OriDim7);
		if (ref_dim==2)
		{
			Mout.window(FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size),
					       LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size));
		}
		else
		{
			Mout.window(FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size),
				       	LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size));
		}
		Mout.setXmippOrigin();
		RCTOC(OriDimTimer,OriDim7);
		// Normalisation factor of FFTW
		// The Fourier Transforms are all "normalised" for 2D transforms of size = ori_size x ori_size
		RCTIC(OriDimTimer,OriDim8);
		Mout /= normfft;
		RCTOC(OriDimTimer,OriDim8);
#ifdef DEBUG_WINDOWORIDIMREALSPACE
		tt()=Mout;
		tt.write("windoworidim_Mwindowed.spi");
#endif

		// Mask out corners to prevent aliasing artefacts
		RCTIC(OriDimTimer,OriDim9);
		softMaskOutsideMap(Mout);
		RCTOC(OriDimTimer,OriDim9);

#ifdef DEBUG_WINDOWORIDIMREALSPACE
		tt()=Mout;
		tt.write("windoworidim_Mwindowed_masked.spi");
		FourierTransformer ttf;
		ttf.FourierTransform(Mout, Fin);
		tt().resize(ZSIZE(Fin), YSIZE(Fin), XSIZE(Fin));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fin)
		{
			DIRECT_MULTIDIM_ELEM(tt(), n) = abs(DIRECT_MULTIDIM_ELEM(Fin, n));
		}
		tt.write("windoworidim_Fnew.spi");
#endif

#ifdef TIMING
    	if(printTimes)
    		OriDimTimer.printTimes(true);
#endif
	}
	else {
		delete transformer.fReal;
		transformer.fReal = NULL; // Make sure to re-calculate fftw plan
	}
	MPI_Barrier(transformer.comm);
}