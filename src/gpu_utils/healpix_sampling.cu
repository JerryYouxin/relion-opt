#include "src/healpix_sampling.h"
#include "src/gpu_utils/cuda_mem_utils.h"
#include "src/gpu_utils/cuda_settings.h"
#include "src/gpu_utils/cuda_kernels/sampling.cuh"
#include "src/gpu_utils/cuda_utils_cub.cuh"
#include <time.h>
#include "src/time.h"
#include <assert.h>

//#define DEBUG_SELECT_GPU
//#define TIMING_
//#include "src/parallel.h"
//static pthread_mutex_t global_mutex = PTHREAD_MUTEX_INITIALIZER;

void HealpixSampling::printDebugInfo() {
        std::cout << "-- " << "L_repositoy = " << "(size=" << L_repository.size() << ")" << std::endl;
        for(int i=0;i<L_repository.size();++i) {
            std::cout<< " \t ";
            std::cout<< "{" << L_repository[i].mdata[0];
            for(int j=1;j<9;++j)
                std::cout << "," << L_repository[i].mdata[j];
            std::cout<< "} ," <<std::endl;
        }
        std::cout << "-- " << "R_repositoy = " << "(size=" << R_repository.size() << ")" << std::endl;
        for(int i=0;i<R_repository.size();++i) {
            std::cout<< "  \t ";
            std::cout << "{" << R_repository[i].mdata[0];
            for(int j=1;j<9;++j)
                std::cout << "," << R_repository[i].mdata[j];
            std::cout<< "} ," <<std::endl;
        }
        std::cout << "-- " << "rot_angles = " << "(size=" << rot_angles.size() << ")" << std::endl;
        for(int i=0;i<rot_angles.size();++i) {
            std::cout << " " << rot_angles[i] << ",";
        }
        std::cout<<std::endl;
        std::cout << "-- " << "tilt_angles = " << "(size=" << rot_angles.size() << ")" << std::endl;
        for(int i=0;i<rot_angles.size();++i) {
            std::cout << " " << tilt_angles[i] << ",";
        }
        std::cout<<std::endl;
        std::cout << "-- " << "psi_angles = " << "(size=" << psi_angles.size() << ")" << std::endl;
        for(int i=0;i<psi_angles.size();++i) {
            std::cout << " " << psi_angles[i] << ",";
        }
        std::cout<<std::endl;
}


void HealpixSampling::selectOrientationsWithNonZeroPriorProbability_gpu(
    RFLOAT prior_rot, RFLOAT prior_tilt, RFLOAT prior_psi,
    RFLOAT sigma_rot, RFLOAT sigma_tilt, RFLOAT sigma_psi,
    std::vector<int> &pointer_dir_nonzeroprior, std::vector<RFLOAT> &directions_prior,
    std::vector<int> &pointer_psi_nonzeroprior, std::vector<RFLOAT> &psi_prior,
    void* _allocator,
    bool couldPrint,
    bool do_bimodal_search_psi,
    RFLOAT sigma_cutoff)
{
#ifdef DEBUG_SELECT_GPU
    std::cout << "\n============== sampling Start =================\n";
#endif
    // Use custom allocator (thread-safe) to allocate device memory
    CudaCustomAllocator* allocator = (CudaCustomAllocator*)_allocator;
    pointer_dir_nonzeroprior.clear();
	directions_prior.clear();

#ifdef DEBUG_SELECT_GPU
#ifdef TIMING_
    Timer timer;
    int TT = timer.setNew(" total ");
    int T0 = timer.setNew(" setup ");
    int T00= timer.setNew(" cudaMalloc ");
    int T1 = timer.setNew(" MemcpyHostToDevice ");
    int T2 = timer.setNew(" CudaKernel ");
    int T3 = timer.setNew(" MemcpyDeviceToHost ");
    int T4 = timer.setNew(" Other ");
    int T5 = timer.setNew(" Norm ");
#endif
    if(couldPrint) {
        std::cout << "-- " << "prior_rot=" << prior_rot << ", prior_tilt=" << prior_tilt << " , prior_psi=" << prior_psi << std::endl;
        std::cout << "-- " << "sigma_rot=" << sigma_rot << ", sigma_tilt=" << sigma_tilt << " , sigma_psi=" << sigma_psi << std::endl;
        std::cout << "-- " << "do_bimodal_search_psi=" << do_bimodal_search_psi << ", sigma_cutoff=" << sigma_cutoff << std::endl;
        printDebugInfo();
    }
#endif

	if (is_3D)
	{
		// Loop over all directions
		RFLOAT sumprior = 0.;
		// Keep track of the closest distance to prevent 0 orientations
		//RFLOAT best_ang = 9999.;
        RFLOAT best_ang = -9999.;
		long int best_idir = -999;
        // Any prior involving BOTH rot and tilt.
		if ( (sigma_rot > 0.) && (sigma_tilt > 0.) ) {
#ifdef DEBUG_SELECT_GPU
#ifdef TIMING_
            timer.tic(TT);
            timer.tic(T0);
#endif
#endif
            Matrix1D<RFLOAT> prior_direction;
            // Get the direction of the prior
            Euler_angles2direction(prior_rot, prior_tilt, prior_direction);
            RFLOAT biggest_sigma = XMIPP_MAX(sigma_rot, sigma_tilt);
            RFLOAT sqrt_sigma = sqrt(2*PI*biggest_sigma*biggest_sigma);
            RFLOAT cosSigmaBiggestXcutoff= cos(sigma_cutoff * biggest_sigma * PI / 180.0);

            int rotAnglesSize = rot_angles.size();
            int R_repository_Size = R_repository.size();
            int offsetL = R_repository_Size*9;
            RFLOAT* RL_repository_tmp = (RFLOAT*)malloc(sizeof(RFLOAT)*R_repository_Size*9*2);
            //RFLOAT* L_repository_tmp = (RFLOAT*)malloc(sizeof(RFLOAT)*R_repository_Size*9);
#ifdef DEBUG_SELECT_GPU
            if(couldPrint)
                printf("-- Running select Kernel! LR size: %ld, rotAnglesSize: %ld\n",sizeof(RFLOAT)*R_repository_Size*9,rotAnglesSize);
#endif
            for (int j = 0; j < R_repository_Size; j++)
			{
                int s = 9 * j;
                // unroll copy loop
                RL_repository_tmp[s  ] = R_repository[j].mdata[0];
                RL_repository_tmp[s+1] = R_repository[j].mdata[1];
                RL_repository_tmp[s+2] = R_repository[j].mdata[2];
                RL_repository_tmp[s+3] = R_repository[j].mdata[3];
                RL_repository_tmp[s+4] = R_repository[j].mdata[4];
                RL_repository_tmp[s+5] = R_repository[j].mdata[5];
                RL_repository_tmp[s+6] = R_repository[j].mdata[6];
                RL_repository_tmp[s+7] = R_repository[j].mdata[7];
                RL_repository_tmp[s+8] = R_repository[j].mdata[8];
            }
            for (int j = 0; j < R_repository_Size; j++)
			{
                int s = offsetL + 9 * j;
                // unroll copy loop
                RL_repository_tmp[s  ] = L_repository[j].mdata[0];
                RL_repository_tmp[s+1] = L_repository[j].mdata[1];
                RL_repository_tmp[s+2] = L_repository[j].mdata[2];
                RL_repository_tmp[s+3] = L_repository[j].mdata[3];
                RL_repository_tmp[s+4] = L_repository[j].mdata[4];
                RL_repository_tmp[s+5] = L_repository[j].mdata[5];
                RL_repository_tmp[s+6] = L_repository[j].mdata[6];
                RL_repository_tmp[s+7] = L_repository[j].mdata[7];
                RL_repository_tmp[s+8] = L_repository[j].mdata[8];
            }
#ifdef DEBUG_SELECT_GPU
#ifdef TIMING_
    timer.toc(T0);
    timer.tic(T00);
#endif
#endif
#ifdef DEBUG_SELECT_GPU
            printf("Allocator %p size : Free %ld, Used %ld\n",allocator,allocator->getTotalFreeSpace(),allocator->getTotalUsedSpace());
#endif
            int size;
            cudaStream_t stream = 0;
            CudaGlobalPtr<RFLOAT > cusumprior(&sumprior,1,stream,allocator);
            CudaGlobalPtr<RFLOAT > cuR_repository(&RL_repository_tmp[0],R_repository_Size*9,stream,allocator);
            CudaGlobalPtr<RFLOAT > cuL_repository(&RL_repository_tmp[offsetL],R_repository_Size*9,stream,allocator);
            CudaGlobalPtr<RFLOAT > curot_angles(&rot_angles[0],rotAnglesSize,stream,allocator);
            CudaGlobalPtr<RFLOAT > cutilt_angles(&tilt_angles[0],rotAnglesSize,stream,allocator);
            CudaGlobalPtr<RFLOAT > cudirections_prior(rotAnglesSize,stream,allocator);
            CudaGlobalPtr<int > cufindDirNum(&size,1,stream,allocator);
            CudaGlobalPtr<RFLOAT > diffang_corr(stream,allocator);
#ifdef DEBUG_SELECT_GPU
            printf("##### rotAnglesSize = %d, R_repository_Size = %d #####\n",rotAnglesSize,R_repository_Size);
#endif
            diffang_corr.setSize(rotAnglesSize);
            
            diffang_corr.device_alloc();
            cusumprior.device_alloc();
            cufindDirNum.device_alloc();
            cudirections_prior.device_alloc();
            if(R_repository_Size>0) {   
                cuR_repository.device_alloc();
                cuL_repository.device_alloc();
            }
            curot_angles.device_alloc();
            cutilt_angles.device_alloc();
            //HANDLE_ERROR(cudaGetLastError());
            //cuR_repository.streamSync();
            HANDLE_ERROR(cudaGetLastError());
            if(R_repository_Size>0) {
                cuR_repository.cp_to_device();
                cuL_repository.cp_to_device();
            }
            curot_angles.cp_to_device();
            cutilt_angles.cp_to_device();
#ifdef DEBUG_SELECT_GPU
            cuR_repository.streamSync();
            HANDLE_ERROR(cudaGetLastError());
            printf("## After Alloc : Allocator size : Free %ld, Used %ld\n",allocator->getTotalFreeSpace(),allocator->getTotalUsedSpace());
#ifdef TIMING_
    timer.toc(T00);
    timer.tic(T1);
#endif
            HANDLE_ERROR(cudaGetLastError());
#endif
            cusumprior.device_init(0);
            cufindDirNum.device_init(0);
#ifdef DEBUG_SELECT_GPU
            HANDLE_ERROR(cudaGetLastError());
            printf("++++++++++++++++++ Device initialization launched ++++++++++++++++++++\n");
#endif
            cusumprior.streamSync();
#ifdef DEBUG_SELECT_GPU
            HANDLE_ERROR(cudaGetLastError());
            printf("++++++++++++++++++ Device initialization syncronised +++++++++++++++++\n");
#ifdef TIMING_
            timer.toc(T1);
            timer.tic(T2);
#endif
            HANDLE_ERROR(cudaGetLastError());
#endif

            int blknum = (rotAnglesSize+BLOCK_SIZE-1) / BLOCK_SIZE;
            selectRotTiltDirection<BLOCK_SIZE><<<blknum,BLOCK_SIZE,0,cufindDirNum.getStream()>>>(~cudirections_prior, 
                ~cufindDirNum, ~cusumprior,
                ~cuR_repository, ~cuL_repository, R_repository_Size,
                rotAnglesSize,
                prior_rot, prior_tilt, ~curot_angles, ~cutilt_angles,
                sigma_rot, sigma_tilt, sigma_cutoff,
                biggest_sigma, ~diffang_corr,
                cosSigmaBiggestXcutoff,sqrt_sigma,
                XX(prior_direction),YY(prior_direction),ZZ(prior_direction));
            HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUG_SELECT_GPU
            printf("++++++++++++++++++ kernel launched ++++++++++++++++++++\n");
#endif
            cufindDirNum.streamSync();
#ifdef DEBUG_SELECT_GPU
            HANDLE_ERROR(cudaGetLastError());
            printf("++++++++++++++++++ kernel syncronized +++++++++++++++++\n");
#ifdef TIMING_
            timer.toc(T2);
            timer.tic(T3);
#endif
#endif
            //int size;
            cufindDirNum.cp_to_host();
            cusumprior.cp_to_host();
            cufindDirNum.streamSync();
#ifdef DEBUG_SELECT_GPU
            HANDLE_ERROR(cudaGetLastError());
#ifdef TIMING_
            timer.toc(T3);
#endif
            if(couldPrint)
                printf("-- find direction num: %d, sum_prior: %lf\n",size,sumprior);
#endif
            if(size>0) {
#ifdef DEBUG_SELECT_GPU
#ifdef TIMING_
                timer.tic(T3);
                
#endif
#endif  
                pointer_dir_nonzeroprior.resize(size);
                directions_prior.resize(size);
                cudirections_prior.cp_to_host();
                cudirections_prior.streamSync();
                HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUG_SELECT_GPU
                printf("++++++++++++++++++ directions_prior to host +++++++++++++++++\n");
#endif
                int k = 0;
                for(int i=0; i<rotAnglesSize; ++i) {
                    if(cudirections_prior[i]!=0.0) {
                        directions_prior[k] = cudirections_prior[i];
                        pointer_dir_nonzeroprior[k] = i;
                        ++k;
                    }
                }
                assert(k==size);
#ifdef DEBUG_SELECT_GPU
#ifdef TIMING_
    timer.toc(T3);
#endif
#endif
                
#ifdef DEBUG_SELECT_GPU
                if(couldPrint) {
                    std::cout << "-- biggestSigma : " << XMIPP_MAX(sigma_rot, sigma_tilt) << std::endl;
                    //std::cout << "-- diffang : " << rotAnglesSize << std::endl;
                    //for(int i=0;i<rotAnglesSize;++i) {
                    //    std::cout << "  -- " << i << " : " << diffang_corr.h_ptr[i] << std::endl;
                    //}
                    //std::cout << std::endl;
                    std::cout << "-- directions_prior: " << std::endl;
                    for(int i=0;i<size;++i) {
                        std::cout << " " << directions_prior[i];
                    }
                    std::cout << std::endl;
                    std::cout << "-- pointer_dir_nonzeroprior : " << std::endl;
                    for(int i=0;i<size;++i) {
                        std::cout << " " << pointer_dir_nonzeroprior[i];
                    }
                    std::cout << std::endl;
                    printf("sorting...\n");
                }    
#ifdef TIMING_
                timer.tic(T4);
#endif
#endif
                for(int i=0; i<size; ++i) {
                    for(int j=0; j<i; ++j) {
                        if(pointer_dir_nonzeroprior[i]<pointer_dir_nonzeroprior[j]) {
                            int t1;
                            t1 = pointer_dir_nonzeroprior[i];
                            pointer_dir_nonzeroprior[i] = pointer_dir_nonzeroprior[j];
                            pointer_dir_nonzeroprior[j] = t1;

                            RFLOAT t2;
                            t2 = directions_prior[i];
                            directions_prior[i] = directions_prior[j];
                            directions_prior[j] = t2;
                        }
                    }
                }
#ifdef DEBUG_SELECT_GPU
#ifdef TIMING_
                timer.toc(T4);
#endif  
#endif
            }
            else {
#ifdef DEBUG_SELECT_GPU
#ifdef TIMING_
                timer.toc(T3);
                timer.tic(T4);
#endif
#endif
                // get argmin value of diffang_corr
                CudaGlobalPtr<cub::KeyValuePair<int, RFLOAT> > min_pair(1, diffang_corr.getStream(),allocator);
                min_pair.device_alloc();
                size_t temp_storage_size = 0;
                
                DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMax( NULL, temp_storage_size, ~diffang_corr, ~min_pair, rotAnglesSize));
#ifdef DEBUG_SELECT_GPU
                printf("++++++++++++++++++ ArgMax plan +++++++++++++++++\n");
#endif
                if(temp_storage_size==0)
                    temp_storage_size=1;

                CudaCustomAllocator::Alloc* alloc = allocator->alloc(temp_storage_size);

                DEBUG_HANDLE_ERROR(cub::DeviceReduce::ArgMax( alloc->getPtr(), temp_storage_size, ~diffang_corr, ~min_pair, rotAnglesSize,diffang_corr.getStream()));
#ifdef DEBUG_SELECT_GPU
                printf("+++++++++++++++++ ArgMax real work +++++++++++++++++\n");
#endif
                //cudaMemcpy(&min_pair_H,min_pair,sizeof(cub::KeyValuePair<int, RFLOAT>),cudaMemcpyDeviceToHost);
                min_pair.cp_to_host();
                diffang_corr.streamSync();

                //pthread_mutex_lock(&global_mutex);
                allocator->free(alloc);
                //pthread_mutex_unlock(&global_mutex);
                best_idir = min_pair[0].key;
                best_ang  = min_pair[0].value;
#ifdef DEBUG_SELECT_GPU
                printf("++++++++++++++++++ best_idir=%ld, best_ang=%lf +++++++++++++++++\n",best_idir,best_ang);
#ifdef TIMING_
                timer.toc(T4);
#endif
                std::cout << "-- best_ang : " << best_ang << ", best_idir : " << best_idir << std::endl;
#endif
            }
            HANDLE_ERROR(cudaGetLastError());
            //pthread_mutex_lock(&global_mutex);
#ifdef DEBUG_SELECT_GPU
            printf("## Before FREE : Allocator size : Free %ld, Used %ld\n",allocator->getTotalFreeSpace(),allocator->getTotalUsedSpace());
#endif
            diffang_corr.free_if_set();
            cusumprior.free_if_set();
            cuR_repository.free_if_set();
            cuL_repository.free_if_set();
            curot_angles.free_if_set();
            cutilt_angles.free_if_set();
            cudirections_prior.free_if_set();
            cufindDirNum.free_if_set();
#ifdef DEBUG_SELECT_GPU
            printf("## After FREE : Allocator size : Free %ld, Used %ld\n",allocator->getTotalFreeSpace(),allocator->getTotalUsedSpace());
#endif
            //pthread_mutex_unlock(&global_mutex);
            free(RL_repository_tmp);
            HANDLE_ERROR(cudaGetLastError());
        }
        else if (sigma_rot > 0.) {
            for (long int idir = 0; idir < rot_angles.size(); idir++)
            {
                Matrix1D<RFLOAT> my_direction, sym_direction;
				RFLOAT sym_rot, sym_tilt;

				// Get the current direction in the loop
				Euler_angles2direction(rot_angles[idir], tilt_angles[idir], my_direction);
				RFLOAT diffang = calculateDeltaRot(my_direction, prior_rot);
				RFLOAT best_diffang = diffang;
				for (int j = 0; j < R_repository.size(); j++)
				{
					sym_direction =  L_repository[j] * (my_direction.transpose() * R_repository[j]).transpose();
					diffang = calculateDeltaRot(sym_direction, prior_rot);

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
        }
        else if (sigma_tilt > 0.) {
            for (long int idir = 0; idir < rot_angles.size(); idir++)
            {
				Matrix1D<RFLOAT> my_direction, sym_direction;
				RFLOAT sym_rot, sym_tilt;

				// Get the current direction in the loop
				Euler_angles2direction(rot_angles[idir], tilt_angles[idir], my_direction);

				// Loop over all symmetry operators to find the operator that brings this direction nearest to the prior
				RFLOAT diffang = ABS(tilt_angles[idir] - prior_tilt);
				if (diffang > 180.)
					diffang = ABS(diffang - 360.);
				RFLOAT best_diffang = diffang;
				for (int j = 0; j < R_repository.size(); j++)
				{
					sym_direction =  L_repository[j] * (my_direction.transpose() * R_repository[j]).transpose();
					Euler_direction2angles(sym_direction, sym_rot, sym_tilt);
					diffang = ABS(sym_tilt - prior_tilt);
					if (diffang > 180.)
						diffang = ABS(diffang - 360.);
					if (diffang < best_diffang)
						best_diffang = diffang;
				}

				// Only consider differences within sigma_cutoff * sigma_tilt
				if (best_diffang < sigma_cutoff * sigma_tilt)
				{
					RFLOAT prior = gaussian1D(best_diffang, sigma_tilt, 0.);
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
        }
        else {
            for (long int idir = 0; idir < rot_angles.size(); idir++)
			{
				// If no prior on the directions: just add all of them
				pointer_dir_nonzeroprior.push_back(idir);
				directions_prior.push_back(1.);
				sumprior += 1.;
			}
        }
    #ifdef DEBUG_SELECT_GPU
    #ifdef TIMING_
        timer.tic(T5);
    #endif
    #endif
		//Normalise the prior probability distribution to have sum 1 over all psi-angles
		for (long int idir = 0; idir < directions_prior.size(); idir++)
			directions_prior[idir] /= sumprior;
    #ifdef DEBUG_SELECT_GPU
    #ifdef TIMING_
        timer.toc(T5);
    #endif
        if(couldPrint) {
            std::cout << "-- directions_prior: " << std::endl;
            for(int i=0;i<directions_prior.size();++i) {
                std::cout << " " << directions_prior[i];
            }
            std::cout << std::endl;
            std::cout << "-- pointer_dir_nonzeroprior : " << std::endl;
            for(int i=0;i<pointer_dir_nonzeroprior.size();++i) {
                std::cout << " " << pointer_dir_nonzeroprior[i];
            }
            std::cout << std::endl;
            if(directions_prior.size()==0)
                std::cout << "-- best_ang : " << best_ang << ", best_idir : " << best_idir << std::endl;
        }
    #endif
		// If there were no directions at all, just select the single nearest one:
		if (directions_prior.size() == 0)
		{
			if (best_idir < 0)
				REPORT_ERROR("HealpixSampling::selectOrientationsWithNonZeroPriorProbability BUG: best_idir < 0");
			pointer_dir_nonzeroprior.push_back(best_idir);
			directions_prior.push_back(1.);
		}
#ifdef DEBUG_SELECT_GPU
#ifdef TIMING_
        timer.toc(TT);
        if(couldPrint)
            timer.printTimes(1);
#endif
#endif
	}
	else
	{
		pointer_dir_nonzeroprior.push_back(0);
		directions_prior.push_back(1.);
	}

	// Psi-angles
	pointer_psi_nonzeroprior.clear();
	psi_prior.clear();

	RFLOAT sumprior = 0.;
	RFLOAT best_diff = 9999.;
    long int best_ipsi = -999;

    if (sigma_psi > 0.)
    {
        for (long int ipsi = 0; ipsi < psi_angles.size(); ipsi++)
	    {
            RFLOAT diffpsi = ABS(psi_angles[ipsi] - prior_psi);
            if (diffpsi > 180.)
                diffpsi = ABS(diffpsi - 360.);
            if (do_bimodal_search_psi && (diffpsi > 90.))
                diffpsi = ABS(diffpsi - 180.);

            // Only consider differences within sigma_cutoff * sigma_psi
            if (diffpsi < sigma_cutoff * sigma_psi)
            {
                RFLOAT prior = gaussian1D(diffpsi, sigma_psi, 0.);
                pointer_psi_nonzeroprior.push_back(ipsi);
                psi_prior.push_back(prior);
                sumprior += prior;

                // TMP DEBUGGING
                if (prior == 0.)
                {
                    std::cerr << " psi_angles[ipsi]= " << psi_angles[ipsi] << " prior_psi= " << prior_psi << " orientational_prior_mode= " << orientational_prior_mode << std::endl;
                    std::cerr << " diffpsi= " << diffpsi << " sigma_cutoff= " << sigma_cutoff << " sigma_psi= " << sigma_psi << std::endl;
                    REPORT_ERROR("prior on psi is zero!");
                }

            }

            // Keep track of the nearest sampling point
            if (diffpsi < best_diff)
            {
                best_ipsi = ipsi;
                best_diff = diffpsi;
            }
        }
    }
    else
    {
        for (long int ipsi = 0; ipsi < psi_angles.size(); ipsi++)
	    {
            pointer_psi_nonzeroprior.push_back(ipsi);
            psi_prior.push_back(1.);
            sumprior += 1.;
        }
    }
	// Normalise the prior probability distribution to have sum 1 over all psi-angles
	for (long int ipsi = 0; ipsi < psi_prior.size(); ipsi++)
		psi_prior[ipsi] /= sumprior;

	// If there were no directions at all, just select the single nearest one:
	if (psi_prior.size() == 0)
	{
		if (best_ipsi < 0)
			REPORT_ERROR("HealpixSampling::selectOrientationsWithNonZeroPriorProbability BUG: best_ipsi < 0");
		pointer_psi_nonzeroprior.push_back(best_ipsi);
		psi_prior.push_back(1.);
	}
#ifdef DEBUG_SELECT_GPU
    std::cout << "\n============== sampling End =================\n";
#endif
#ifdef  DEBUG_SAMPLING
	std::cerr << " psi_angles.size()= " << psi_angles.size() << " psi_step= " << psi_step << std::endl;
	std::cerr << " psi_prior.size()= " << psi_prior.size() << " pointer_psi_nonzeroprior.size()= " << pointer_psi_nonzeroprior.size() << " sumprior= " << sumprior << std::endl;
#endif
	return;
}