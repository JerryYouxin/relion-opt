#include "src/macros.h"
#include "src/pfftw.h"
#include "src/args.h"
#include <string.h>
#include <math.h>

#include "src/mpi.h"

//#undef USE_DISTRIBUTED_FFT

//#define DEBUG_DISTRIBUTED_FFT
//#define DEBUG_PLANS

static pthread_mutex_t fftw_plan_mutex = PTHREAD_MUTEX_INITIALIZER;

// Constructors and destructors --------------------------------------------
DistributedFourierTransformer::DistributedFourierTransformer(MpiNode* _node, MPI_Comm _comm, int nr_threads):
		plans_are_set(false)
{
    node = _node;
    init(_comm, nr_threads);

#ifdef DEBUG_PLANS
    std::cerr << "INIT this= "<<this<< std::endl;
#endif
}

DistributedFourierTransformer::~DistributedFourierTransformer()
{
    clear();
#ifdef DEBUG_PLANS
    std::cerr << "CLEARED this= "<<this<< std::endl;
#endif
}

void DistributedFourierTransformer::init(MPI_Comm _comm, int nr_threads) {
    fReal           = NULL;
    fReal_local     = NULL;
    fFourier_local  = NULL;
    comm = _comm;
    MPI_Comm_rank(_comm, &rank);
    MPI_Comm_size(_comm, &size);
    fftw_mpi_init();
    thread_ok = (nr_threads>1);
    if(thread_ok) {
        fftw_init_threads();
        fftw_plan_with_nthreads(nr_threads);
    }
    dLocInfo = new int[2*size];
#ifdef DEBUG_DISTRIBUTED_FFT
    std::cout << "Size : " << size << ", Rank : " << rank << std::endl;
#endif
}

void DistributedFourierTransformer::destroyPlans()
{
    // Anything to do with plans has to be protected for threads!
    pthread_mutex_lock(&fftw_plan_mutex);

    if (plans_are_set)
    {
#ifdef RELION_SINGLE_PRECISION
    	fftwf_destroy_plan(fPlanForward);
    	fftwf_destroy_plan(fPlanBackward);
#else
    	fftw_destroy_plan(fPlanForward);
    	fftw_destroy_plan(fPlanBackward);
#endif
    	plans_are_set = false;
    }

    pthread_mutex_unlock(&fftw_plan_mutex);

}

void DistributedFourierTransformer::clear()
{
    fFourier.clear();
    // Clean-up all other FFTW-allocated memory
    destroyPlans();
    // Initialise all pointers to NULL
    if(fFourier_local!=NULL) {
        fftw_free(fFourier_local);
    }
    if(fReal_local!=NULL) {
        fftw_free(fReal_local);
    }
    if(dLocInfo!=NULL) {
        delete [] dLocInfo;
    }
    fReal           = NULL;
    fReal_local     = NULL;
    fFourier_local  = NULL;
    dLocInfo        = NULL;

}

void DistributedFourierTransformer::cleanup()
{
	// First clear object and destroy plans
    clear();
    // Then clean up all the junk fftw keeps lying around
    // SOMEHOW THE FOLLOWING IS NOT ALLOWED WHEN USING MULTPLE TRANSFORMER OBJECTS....
#ifdef RELION_SINGLE_PRECISION
    fftwf_cleanup_threads();
    fftwf_mpi_cleanup();
#else
    fftw_cleanup_threads();
    fftw_mpi_cleanup();
#endif

#ifdef DEBUG_PLANS
    std::cerr << "CLEANED-UP this= "<<this<< std::endl;
#endif

}

void DistributedFourierTransformer::setSize(int nx, int ny, int nz) {
    fFourier.reshape(nz,ny,nx/2+1);
    bool recomputePlan=false;
    if (fReal==NULL) {
        recomputePlan=true;
        fReal = new MultidimArray<RFLOAT>(nz,ny,2*(nx/2+1));
    } else {
        recomputePlan=!(xdim==nx && ydim==ny && zdim==nz);
        fReal->reshape(nz,ny,2*(nx/2+1));
    }
#ifdef DEBUG_DISTRIBUTED_FFT
    std::cout << "reconmputePlan " << recomputePlan << ", " << nx << " " << ny << " " << nz << std::endl;
#endif
    if (recomputePlan)
    {
        xdim=nx;
        ydim=ny;
        zdim=nz;
        int ndim=3;
        if (nz==1)
        {
            ndim=2;
            if (ny==1)
                ndim=1;
        }
        dimension = ndim;
        std::ptrdiff_t *N  = new std::ptrdiff_t[ndim];
        std::ptrdiff_t *fN = new std::ptrdiff_t[ndim];
        switch (ndim)
        {
        case 1:
            printf("DID NOT SUPPORT FOR 1-DIMENSION MPI-FFT");
            REPORT_ERROR("DID NOT SUPPORT FOR 1-DIMENSION MPI-FFT");
            break;
        case 2:
            N[0]=ny;
            N[1]=nx;
            fN[0]=ny;
            fN[1]=nx/2+1;
            local_fsize = N[1]/2+1;
            break;
        case 3:
            N[0]=nz;
            N[1]=ny;
            N[2]=nx;
            fN[0]=nz;
            fN[1]=ny;
            fN[2]=nx/2+1;
            local_fsize = N[1]*(N[2]/2+1);
            break;
        }
#ifdef DEBUG_DISTRIBUTED_FFT
    std::cout << "C1"<< std::endl;
#endif
        // Destroy both forward and backward plans if they already exist
        destroyPlans();
#ifdef DEBUG_DISTRIBUTED_FFT
    std::cout << "C2"<< std::endl;
#endif
        alloc_local = fftw_mpi_local_size(ndim, fN, comm, &local_n0, &local_0_start);
#ifdef DEBUG_DISTRIBUTED_FFT
    std::cout << "C3"<< std::endl;
#endif
        if(fFourier_local!=NULL) 
            fftw_free(fFourier_local);
        if(fReal_local!=NULL) 
            fftw_free(fReal_local);
        fFourier_local = (Complex*)fftw_malloc(sizeof(Complex)*alloc_local);
        fReal_local = (RFLOAT*)fftw_malloc(sizeof(RFLOAT)*2*alloc_local);
        local_fsize*= local_n0;
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " REPORT ==> ndim : " << ndim << " Comm size %d : " << size << " Comm " << comm << std::endl;
        std::cout << "## Rank " << rank << " REPORT ==> (nx, ny, nz) : " << nx << " " << ny << " " << nz << std::endl;
        std::cout << "## Rank " << rank << " REPORT ==> local_n0 : " << local_n0 << ", local_0_start : " << local_0_start << std::endl;
        std::cout << "## Rank " << rank << " REPORT ==> alloc_local : " << alloc_local << ", local_fsize : " << local_fsize << std::endl;
        std::cout << "## Rank " << rank << " REPORT ==> fFourier_local : " << fFourier_local << ", fReal_local : " << fReal_local << std::endl;
        std::cout << "## Rank " << rank << " REPORT ==> FFTW_ESTIMATE : " << FFTW_ESTIMATE << std::endl;
#endif
        // Make new plans
        plans_are_set = true;

        // Anything to do with plans has to be protected for threads!
        pthread_mutex_lock(&fftw_plan_mutex);
#ifdef RELION_SINGLE_PRECISION
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " single Planning forward..." << std::endl;
        MPI_Barrier(comm);
#endif
        fPlanForward = fftwf_mpi_plan_dft_r2c(ndim, N, fReal_local,
                                         (fftwf_complex*) fFourier_local, 
                                         comm, FFTW_ESTIMATE);
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " single Planning backward..." << std::endl;
        MPI_Barrier(comm);
#endif
        fPlanBackward = fftwf_mpi_plan_dft_c2r(ndim, N,
                                          (fftwf_complex*) fFourier_local, 
                                          fReal_local,
                                          comm, FFTW_ESTIMATE);
#else
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " Planning forward..." << std::endl;
        MPI_Barrier(comm);
        std::cout << "## Rank " << rank << " begin fwd" << std::endl;
#endif
        fPlanForward = fftw_mpi_plan_dft_r2c(ndim, N, fReal_local,
                                         (fftw_complex*) fFourier_local, 
                                         comm, FFTW_ESTIMATE);
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " Planning backward..." << std::endl;
        MPI_Barrier(comm);
        std::cout << "## Rank " << rank << " begin bwd " << ndim << " " << N[0] << " " << N[1] << " " << N[2] << " " << fFourier_local << " " << fReal_local << " " << comm << " " << FFTW_ESTIMATE << std::endl;
#endif
        fPlanBackward = fftw_mpi_plan_dft_c2r(ndim, N,
                                          (fftw_complex*) fFourier_local, 
                                          fReal_local,
                                          comm, FFTW_ESTIMATE);
#endif
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " Planning finished" << std::endl;
#endif
        pthread_mutex_unlock(&fftw_plan_mutex);

        if (fPlanForward == NULL || fPlanBackward == NULL)
        {
            printf("FFTW plans cannot be created %x %x\n",fPlanForward, fPlanBackward);
            REPORT_ERROR("FFTW plans cannot be created");
        }

#ifdef DEBUG_PLANS
        std::cerr << " SETREAL fPlanForward= " << fPlanForward << " fPlanBackward= " << fPlanBackward  <<" this= "<<this<< std::endl;
#endif

        delete [] N;
        delete [] fN; 
    }
}

void DistributedFourierTransformer::setReal(MultidimArray<RFLOAT>& input) {
    if(fReal!=NULL) delete fReal;
    fReal=&input;
}

// Transform ---------------------------------------------------------------
void DistributedFourierTransformer::Transform(int sign)
{
    if (sign == FFTW_FORWARD)
    {
#ifdef RELION_SINGLE_PRECISION
        fftwf_mpi_execute_dft_r2c(fPlanForward,fReal_local,
                (fftwf_complex*) fFourier_local);
#else
        fftw_mpi_execute_dft_r2c(fPlanForward,fReal_local,
                (fftw_complex*)fFourier_local);
#endif
        // Normalisation of the transform
        unsigned long int size=xdim*ydim*zdim;
        // if(fReal!=NULL)
        //     size = MULTIDIM_SIZE(*fReal);
        // else
        //     REPORT_ERROR("No real data defined");
        for(int n=0;n<local_fsize;++n)
            fFourier_local[n] /= size;
    }
    else if (sign == FFTW_BACKWARD)
    {
#ifdef RELION_SINGLE_PRECISION
        fftwf_mpi_execute_dft_c2r(fPlanBackward,
                (fftwf_complex*)fFourier_local, fReal_local);
#else
        fftw_mpi_execute_dft_c2r(fPlanBackward,
                (fftw_complex*) fFourier_local, fReal_local);
#endif
    }
}

void DistributedFourierTransformer::FourierTransform()
{
    Transform(FFTW_FORWARD);
}

void DistributedFourierTransformer::inverseFourierTransform()
{
    Transform(FFTW_BACKWARD);
}

void DistributedFourierTransformer::distribute(bool real)
{   
    MPI_Request* request = new MPI_Request[size];
    MPI_Status status;
    int data[2] = { local_0_start, local_n0 };
    if(rank!=0) {
        MPI_Isend(data, 2, MPI_INT, 0, 101, comm, &request[0]);
    } else {
        for(int i=1;i<size;++i) {
            MPI_Irecv(&dLocInfo[2*i], 2, MPI_INT, i, 101, comm, &request[i]);
        }
    }

    if(real) {
        printf("WARNING: TRANSPOSED REAL DISTRIBUTION is not efficient!\n");
        // distribute real
        int fxdim = xdim/2+1;
        int xdim_local = 2*fxdim;
        int xydim = xdim_local*ydim;
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " REPORT : TRANSPOSED REAL" << std::endl;
        std::cout << "## Rank " << rank << " REPORT ==> distribute REAL from " << local_0_start << " to " << local_0_start+local_n0-1 << std::endl;
#endif
        for(int k=0;k<local_n0;++k) {
            for(int i=0;i<ydim;++i) {
                for(int j=0;j<xdim;++j) {
                    fReal_local[k*xydim+i*xdim_local+j] = DIRECT_A3D_ELEM(*fReal,k+local_0_start,i,j);
// #ifdef DEBUG_DISTRIBUTED_FFT
//                     if((k*xydim+i*xdim+j) % 10000 == 0)
//                         std::cout << "## Rank " << rank << " REPORT ==> copy " << k*xydim+i*xdim+j << " from " << (k+local_0_start)*YXSIZE(*fReal)+((i)*XSIZE(*fReal))+(j) << std::endl;
// #endif
                }
            }
        }
    } else {
        // distribute Fourier
        int fxdim = xdim/2+1;
        int xydim = fxdim*ydim;
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " REPORT ==> distribute FOURIER from " << local_0_start << " to " << local_0_start+local_n0-1 << std::endl;
#endif
        for(int k=0;k<local_n0;++k) {
            for(int i=0;i<ydim;++i) {
                for(int j=0;j<fxdim;++j) {
                    fFourier_local[k*xydim+i*fxdim+j] = DIRECT_A3D_ELEM(fFourier,k+local_0_start,i,j);
// #ifdef DEBUG_DISTRIBUTED_FFT
//                     if((k*xydim+i*xdim+j) % 10000 == 0)
//                         std::cout << "## Rank " << rank << " REPORT ==> copy " << k*xydim+i*xdim+j << " from " << (k+local_0_start)*YXSIZE(*fReal)+((i)*XSIZE(*fReal))+(j) << std::endl;
// #endif
                }
            }
        }
    }
    if(rank!=0) {
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " REPORT ==> Waiting for distribute mpi sending..." << std::endl;
#endif
        MPI_Wait(&request[0], &status);
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " REPORT ==> finish" << std::endl;
#endif
    } else {
        for(int i=1;i<size;++i) {
#ifdef DEBUG_DISTRIBUTED_FFT
            std::cout << "## Rank " << rank << " REPORT ==> Waiting for distribute mpi recving from " << i << std::endl;
#endif
            MPI_Wait(&request[i], &status);
#ifdef DEBUG_DISTRIBUTED_FFT
            std::cout << "## Rank " << rank << " REPORT ==> from " << i << " finish, " << dLocInfo[2*i] << ", "<< dLocInfo[2*i+1] << std::endl;
#endif
        }
    }
    MPI_Barrier(comm);
}

void DistributedFourierTransformer::distribute_location()
{   
    MPI_Request* request = new MPI_Request[size];
    MPI_Status status;
    int data[2] = { local_0_start, local_n0 };
    if(rank!=0) {
        MPI_Isend(data, 2, MPI_INT, 0, 101, comm, &request[0]);
    } else {
        for(int i=1;i<size;++i) {
            MPI_Irecv(&dLocInfo[2*i], 2, MPI_INT, i, 101, comm, &request[i]);
        }
    }
    if(rank!=0) {
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " REPORT ==> Waiting for distribute mpi sending..." << std::endl;
#endif
        MPI_Wait(&request[0], &status);
#ifdef DEBUG_DISTRIBUTED_FFT
        std::cout << "## Rank " << rank << " REPORT ==> finish" << std::endl;
#endif
    } else {
        for(int i=1;i<size;++i) {
#ifdef DEBUG_DISTRIBUTED_FFT
            std::cout << "## Rank " << rank << " REPORT ==> Waiting for distribute mpi recving from " << i << std::endl;
#endif
            MPI_Wait(&request[i], &status);
#ifdef DEBUG_DISTRIBUTED_FFT
            std::cout << "## Rank " << rank << " REPORT ==> from " << i << " finish, " << dLocInfo[2*i] << ", "<< dLocInfo[2*i+1] << std::endl;
#endif
        }
    }
    MPI_Barrier(comm);
}

void DistributedFourierTransformer::mergeTo(MultidimArray<RFLOAT>& out)
{
    //printf("%d : In merge to RFLOAT\n",rank);
    // int sz = xdim*ydim;
    // int LA = 2*(xdim/2+1)*ydim;
    // if(rank==0) {
    //     MPI_Status status;
    //     for(int i=1;i<size;++i) {
    //         if(dLocInfo[i*2+1] > local_n0) {
    //             printf("ERROR! %d > %d\n", dLocInfo[i*2+1], local_n0);
    //             exit(-1);
    //         }
    //         for(int k=0;k<dLocInfo[i*2+1];++k) {
    //             node->relion_MPI_Recv(&(out.data[(dLocInfo[i*2]+k)*sz]), sz, MY_MPI_DOUBLE, i, 100, comm, status);
    //         }
    //         printf("Recved %d\n", i);
    //     }

    //     int fxdim = xdim/2+1;
    //     int xdim_local = 2*fxdim;
    //     int xydim = xdim_local*ydim;
    //     for(int k=0;k<local_n0;++k) {
    //         for(int i=0;i<ydim;++i) {
    //             for(int j=0;j<xdim;++j) {
    //                 DIRECT_A3D_ELEM(out,k+local_0_start,i,j) = fReal_local[k*xydim+i*xdim_local+j];
    //             }
    //         }
    //     }
    //     printf("Rank %d wait Finished\n",rank);
    // } else {
    //     for(int k=0;k<local_n0;++k) {
    //         node->relion_MPI_Send(&fReal_local[k*LA], sz, MY_MPI_DOUBLE, 0, 100, comm);
    //     }
    //     printf("Rank %d wait Finished\n",rank);
    // }
    merge(true);
    if(rank==0) {
        //out.printShape();
        //fReal->printShape();
        FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(out)
        {
            DIRECT_A3D_ELEM(out,k,i,j) = DIRECT_A3D_ELEM(*fReal,k,i,j);
        }
    }
    //printf("Rank %d finish merging\n",node->rank);
    MPI_Barrier(comm);
    //printf("Rank %d finish merging after barrier\n",node->rank);
}

void DistributedFourierTransformer::merge(bool real)
{
    //printf("In merge\n");
    RFLOAT* data = NULL;
    RFLOAT* dst  = NULL;
    int sz = 2*(xdim/2+1)*ydim;
    if(real) {
        fReal->initZeros();
        dst  = (RFLOAT*)fReal->data;
        data = (RFLOAT*)fReal_local; 
    } else {
        fFourier.initZeros();
        dst  = (RFLOAT*)fFourier.data;
        data = (RFLOAT*)fFourier_local;
    }
    if(rank==0) {
        //MPI_Request* request = new MPI_Request[size*local_n0];
        MPI_Status status;
        //bool splitted = false;

        for(int i=1;i<size;++i) {
            if(((double)dLocInfo[i*2+1])*sz*sizeof(RFLOAT) >= 1024.*1024.*1024.*2) {
                //printf("Too Large data size. split by local_n0\n");
                //splitted = true;
                if(dLocInfo[i*2+1] > local_n0) {
                    //printf("ERROR! %d > %d\n", dLocInfo[i*2+1], local_n0);
                    exit(-1);
                }
                for(int k=0;k<dLocInfo[i*2+1];++k) {
#ifdef RELION_SINGLE_PRECISION
                    MPI_Recv(&dst[(dLocInfo[i*2]+k)*sz], sz, MPI_FLOAT, i, 100, comm, &status);//, &request[i*local_n0+k]);
#else
                    MPI_Recv(&dst[(dLocInfo[i*2]+k)*sz], sz, MPI_DOUBLE, i, 100, comm, &status);//, &request[i*local_n0+k]);
#endif
                }
            } else {
                // printf("Checking...%d %d %d %d\n",i,dLocInfo[i*2],dLocInfo[i*2+1],sz);
                // printf("%x... %x %x %x %x\n",dst,fReal->data,fReal_local,fFourier.data,fFourier_local);
                // fReal->printShape();
                // fFourier.printShape();
                // for(int p=dLocInfo[i*2]*sz;p<dLocInfo[i*2]*sz+dLocInfo[i*2+1]*sz-1;++p) {
                //     //printf("%d from %d to %d\n",p,dLocInfo[i*2]*sz,dLocInfo[i*2]*sz+dLocInfo[i*2+1]*sz-1);
                //     dst[p]=0.;
                // }
                //printf("Recving...\n");
#ifdef RELION_SINGLE_PRECISION
                MPI_Recv(&dst[dLocInfo[i*2]*sz], dLocInfo[i*2+1]*sz, MPI_FLOAT, i, 100, comm, &status);//, &request[i]);
#else
                MPI_Recv(&dst[dLocInfo[i*2]*sz], dLocInfo[i*2+1]*sz, MPI_DOUBLE, i, 100, comm, &status);//, &request[i]);
#endif
                //printf("Recved %d\n", i);
            }
        }

        int xydim = xdim*ydim;
        if(real) {
            int fxdim = xdim/2+1;
            int xdim_local = 2*fxdim;
            int xydim = xdim_local*ydim;
            for(int k=0;k<local_n0;++k) {
                for(int i=0;i<ydim;++i) {
                    for(int j=0;j<xdim;++j) {
                        DIRECT_A3D_ELEM(*fReal,k+local_0_start,i,j) = fReal_local[k*xydim+i*xdim_local+j];
                    }
                }
            }
        } else {
            int fxdim = xdim/2+1;
            int xydim = fxdim*ydim;
            for(int k=0;k<local_n0;++k) {
                for(int i=0;i<ydim;++i) {
                    for(int j=0;j<fxdim;++j) {
                        DIRECT_A3D_ELEM(fFourier,k+local_0_start,i,j) = fFourier_local[k*xydim+i*fxdim+j];
                    }
                }
            }
        }
        // if(splitted) {
        //     for(int n=1;n<size;++n) for(int i=0;i<dLocInfo[n*2+1];++i) {
        //         printf("Rank %d, waiting for Rank %d : package %d : request %x, status %x\n",rank, n, i, &request[n*local_n0+i], &status);
        //         MPI_Wait(&request[n], &status);
        //     }
        // } else {
        //     for(int n=1;n<size;++n) {
        //         printf("Rank %d, waiting for Rank %d request %x, status %x\n",rank, n, &request[n], &status);
        //         MPI_Wait(&request[n], &status);
        //     }
        // }
        //printf("Rank %d wait Finished\n",rank);
        //delete[] request;
    } else {
        if(((double)local_n0)*sz*sizeof(RFLOAT) >= 1024.*1024.*1024.*2.) {
            //printf("Rank %d Too Large data size. split by local_n0 %d\n",rank, local_n0);
            //MPI_Request* request = new MPI_Request[local_n0];
            //MPI_Status status;
            for(int k=0;k<local_n0;++k) {
                //printf("Sending %x %lf\n",&data[k*sz],data[k*sz+sz-1]);
#ifdef RELION_SINGLE_PRECISION
                MPI_Send(&data[k*sz], sz, MPI_FLOAT, 0, 100, comm);//, &request[k]);
#else
                MPI_Send(&data[k*sz], sz, MPI_DOUBLE, 0, 100, comm);//, &request[k]);
#endif
            }
            // for(int k=0;k<local_n0;++k) {
            //     printf("Rank %d, waiting for package %d, request %x, status %x, sending %lf MB data.\n",rank, k, &request[k], &status, sz*sizeof(RFLOAT)/1024.0/1024.0);
            //     MPI_Wait(&request[k], &status);
            // }
            //printf("Rank %d wait Finished\n",rank);
            // delete[] request;
        } else {
            //printf("Sending %x %lf\n",data,data[local_n0*sz-1]);
            for(int i=0;i<local_n0*sz;++i) {
                RFLOAT debug = data[i];
            }
            //printf("Sending...\n");
            //MPI_Request request;
            //MPI_Status status;
#ifdef RELION_SINGLE_PRECISION
            MPI_Send(data, local_n0*sz, MPI_FLOAT, 0, 100, comm);//, &request);
#else
            MPI_Send(data, local_n0*sz, MPI_DOUBLE, 0, 100, comm);//, &request);
#endif
            //printf("Rank %d, waiting for request %x, status %x, sending %lf MB data.\n",rank, &request, &status, local_n0*sz*sizeof(RFLOAT)/1024.0/1024.0);
            //MPI_Wait(&request, &status);
            //printf("Rank %d wait Finished\n",rank);
        }
    }
    MPI_Barrier(comm);
}