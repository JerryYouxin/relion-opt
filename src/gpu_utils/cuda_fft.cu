#include "src/gpu_utils/cuda_fft.h"
#include "src/gpu_utils/cuda_kernels/fft_helper.cuh"

void run_transpose_gpu(__COMPLEX_T* out, __COMPLEX_T* in, int xs, int ys, cudaStream_t stream)
{
    int nxysize = xs;
    int nzsize  = ys;
    dim3 dimGrid((nxysize+TILE_DIM-1)/TILE_DIM,(nzsize+TILE_DIM-1)/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    fft_transposeXY_kernel_complex<<<dimGrid,dimBlock,0,stream>>>(out,in,nzsize,nxysize);
}

void run_transpose_gpu(RFLOAT* out, RFLOAT* in, int xs, int ys, cudaStream_t stream)
{
    int nxysize = xs;
    int nzsize  = ys;
    dim3 dimGrid((nxysize+TILE_DIM-1)/TILE_DIM,(nzsize+TILE_DIM-1)/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    fft_transposeXY_kernel_real<<<dimGrid,dimBlock,0,stream>>>(out,in,nzsize,nxysize);
}

void run_cuda_transpose(CudaGlobalPtr<__COMPLEX_T, false> &dst, CudaGlobalPtr<__COMPLEX_T, false> &src, int xs, int ys, bool hostBuffer)
{
    if(hostBuffer) {
        __COMPLEX_T* in = src.h_ptr;
        __COMPLEX_T* out= dst.h_ptr;
        int xysize = xs*ys;
        size_t needed, avail, total;
        needed = xysize*2*sizeof(__COMPLEX_T);
        DEBUG_HANDLE_ERROR(cudaMemGetInfo( &avail, &total ));
        int maxiter = 1; // will divide to maxiter x 1 blocks to transpose
        while(avail <= needed && needed>1) {
            ++maxiter;
            if(xs%maxiter!=0) continue;
            needed = xysize*2*sizeof(__COMPLEX_T)/maxiter;
        }
        printf("-- Info :Total %ld, Avail %ld, Needed %ld, Split input to %d x %d blocks <Block size: %d x %d>\n",total,avail,needed,maxiter,1,xs/maxiter,ys/1);
        if(avail <= needed) {
            printf("GPU memory not enough for single batch GPU transpose! Use CPU\n");
            for(int i=0;i<ys;++i) {
                for(int k=0;k<xs;++k) {
                    out[k*ys+i] = in[i*xs+k];
                }
            }
        }
        else {
            cudaStream_t stream = dst.getStream();
            CudaGlobalPtr<__COMPLEX_T,0> __in(stream);
            CudaGlobalPtr<__COMPLEX_T,0> __out(stream);
            int nxsize = xs / maxiter;
            int nysize  = ys;
            __in.setSize(nxsize*nysize);
            __out.setSize(nxsize*nysize);
            __in.device_alloc();
            __out.device_alloc();
            if(maxiter==1) {
                __in.h_ptr = in;
                __out.h_ptr= out;
                __in.cp_to_device();
                run_transpose_gpu(~__out,~__in,nxsize,nysize,stream);
                __out.cp_to_host();
                __out.streamSync();
            }
            else {
                printf("WARNING: GPU memory not enough, will split transpose to %d iters\n",maxiter);__in.host_alloc();
                for(int ix=0;ix<maxiter;++ix) {
                    printf("Iteration : <%d,%d>-th block\n",ix,0);
                    __out.h_ptr= out;
                    for(int n=0;n<nysize;++n) {
                        memcpy(__in.h_ptr+n*nxsize,in+n*xs,sizeof(__COMPLEX_T)*nxsize);
                    }
                    __in.cp_to_device();
                    __in.streamSync();
                    run_transpose_gpu(~__out,~__in,nxsize,nysize,stream);
                    __out.cp_to_host();
                    __out.streamSync();
                    in += nxsize;
                    out+= nysize*nxsize;
                    // printf("Next Iter\n");
                }
                __in.free_host_if_set();
            }
            printf("-- finish transpose\n");
            __in.free_device_if_set();
            __out.free_device_if_set();
        }
    } else {
        run_transpose_gpu(~dst,~src,xs,ys,dst.getStream());
    }
}

#ifdef PARALLEL_RECONSTRUCT_NEW

#ifdef RELION_SINGLE_PRECISION
#define COMPLEX_T cufftComplex
#define REAL_T cufftReal 
#else
#define COMPLEX_T cufftDoubleComplex
#define REAL_T cufftDoubleReal 
#endif

void DistributedCudaFFT::transpose(COMPLEX_T* out, COMPLEX_T *in, int xs, int ys) {
    // printf("transpose XYZ => ZXY...\n");
    int xysize = xs*ys;
    size_t needed, avail, total;
	needed = xysize*2*sizeof(__COMPLEX_T);
	DEBUG_HANDLE_ERROR(cudaMemGetInfo( &avail, &total ));
    int maxiter = 1; // will divide to maxiter x 1 blocks to transpose
    while(avail <= needed && needed>1) {
        ++maxiter;
        if(xs%maxiter!=0) continue;
        needed = xysize*2*sizeof(__COMPLEX_T)/maxiter;
    }
    // printf("-- Info :Total %ld, Avail %ld, Needed %ld, Split input to %d x %d blocks <Block size: %d x %d>\n",total,avail,needed,maxiter,1,xs/maxiter,ys/1);
    if(avail <= needed) {
        printf("GPU memory not enough for GPU transpose! Use CPU\n");
        for(int i=0;i<ys;++i) {
            for(int k=0;k<xs;++k) {
                out[k*ys+i] = in[i*xs+k];
            }
        }
    }
    else {
        CudaGlobalPtr<__COMPLEX_T,0> __in(stream);
        CudaGlobalPtr<__COMPLEX_T,0> __out(stream);
        int nxysize = xs / maxiter;
        int nzsize  = ys;
        __in.setSize(nxysize*nzsize);
        __out.setSize(nxysize*nzsize);
        __in.device_alloc();
        __out.device_alloc();
        if(maxiter==1) {
            __in.h_ptr = in;
            __out.h_ptr= out;
            __in.cp_to_device();
            dim3 dimGrid((nxysize+TILE_DIM-1)/TILE_DIM,(nzsize+TILE_DIM-1)/TILE_DIM, 1);
            dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
            fft_transposeXY_kernel<<<dimGrid,dimBlock,0,stream>>>(~__out,~__in,nzsize,nxysize);
            __out.cp_to_host();
            __out.streamSync();
        }
        else {
            for(int ix=0;ix<maxiter;++ix) {
                // printf("Iteration : <%d,%d>-th block\n",ix,0);
                __in.h_ptr = in+ix*nxysize;
                __out.h_ptr= out+ix*zFSize*nxysize;
                for(int i=0;i<nzsize;++i) {
                    HANDLE_ERROR(cudaMemcpyAsync( __in.d_ptr+i*nxysize, __in.h_ptr+i*xysize, nxysize * sizeof(__COMPLEX_T), cudaMemcpyHostToDevice, stream));
                }
                dim3 dimGrid((nxysize+TILE_DIM-1)/TILE_DIM,(nzsize+TILE_DIM-1)/TILE_DIM, 1);
                dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
                fft_transposeXY_kernel<<<dimGrid,dimBlock,0,stream>>>(~__out,~__in,nzsize,nxysize);
                __out.cp_to_host();
                __out.streamSync();
                // printf("Next Iter\n");
            }
        }
        __in.free_if_set();
        __out.free_if_set();
    }
    //printf("end\n");
}

void DistributedCudaFFT::transpose_gpu(COMPLEX_T* out, COMPLEX_T *in, int xs, int ys) {
    // printf("transpose XYZ => ZXY...\n");
    int nxysize = xs;
    int nzsize  = ys;
    dim3 dimGrid((nxysize+TILE_DIM-1)/TILE_DIM,(nzsize+TILE_DIM-1)/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    fft_transposeXY_kernel<<<dimGrid,dimBlock,0,stream>>>(out,in,nzsize,nxysize);
}

void DistributedCudaFFT::transpose_gpu(double* out, double *in, int xs, int ys) {
    // printf("transpose XYZ => ZXY...\n");
    int nxysize = xs;
    int nzsize  = ys;
    dim3 dimGrid((nxysize+TILE_DIM-1)/TILE_DIM,(nzsize+TILE_DIM-1)/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    fft_transposeXY_kernel_double<<<dimGrid,dimBlock,0,stream>>>(out,in,nzsize,nxysize);
}

#define MPI_TAG_EXCHANGE_BASE 888
#define MPI_TAG_EXCHANGE_BACK_BASE 999
// (z_split,x,y) => (z,x,y_split)
void DistributedCudaFFT::exchange(COMPLEX_T* _zsplit_buff, COMPLEX_T* _ysplit_buff)
{
    // launch nonblocking send/recv
    for(int n=0;n<size;++n) {
        if(n!=rank) {
            // source ptr (z_split,x,dst_y_split)
            COMPLEX_T* ptr = _zsplit_buff + dyFLoc[2*n]*lzFSize*xFSize;
            node->relion_MPI_ISend(ptr, dyFLoc[2*n+1]*lzFSize*xFSize, MY_MPI_COMPLEX, n, MPI_TAG_EXCHANGE_BASE, comm);
            // recv ptr (dst_z_split,x,y_split)
            COMPLEX_T* recv= h_buff + lyFSize*dzFLoc[2*n]*xFSize;
            node->relion_MPI_IRecv(recv, lyFSize*dzFLoc[2*n+1]*xFSize, MY_MPI_COMPLEX, n, MPI_TAG_EXCHANGE_BASE, comm);
        }
    }
    // copy my own data
    for(int i=0;i<lyFSize;++i) {
        int ii=i+lyFStart;
        for(int j=0;j<xFSize;++j) {
            for(int k=0;k<lzFSize;++k) {
                int kk=k+lzFStart;
                _ysplit_buff[i*zFSize*xFSize+j*zFSize+kk] = _zsplit_buff[ii*lzFSize*xFSize+j*lzFSize+k];
            }
        }
    }
    // wait for send/recv
    MPI_Status status;
    node->relion_MPI_WaitAll(status);
    // copy recv buffer to exact place
    for(int n=0;n<size;++n) {
        if(n!=rank) {
            // recv ptr (dst_z_split,x,y_split)
            COMPLEX_T* recv= h_buff + lyFSize*dzFLoc[2*n]*xFSize;
            // dst ptr (z,x,y_split)
            COMPLEX_T* dst = _ysplit_buff + dzFLoc[2*n];
            for(int i=0;i<lyFSize;++i) {
                for(int j=0;j<xFSize;++j) {
                    for(int k=0;k<dzFLoc[2*n+1];++k) {
                        dst[i*xFSize*zFSize+j*zFSize+k] = recv[i*xFSize*dzFLoc[2*n+1]+j*dzFLoc[2*n+1]+k];
                    }
                }
            }
        }
    }
}
// (z,x,y_split) => (z_split,x,y)
void DistributedCudaFFT::exchange_back(COMPLEX_T* _ysplit_buff, COMPLEX_T* _zsplit_buff)
{
    // launch non-blocking recv first
    for(int n=0;n<size;++n) {
        if(n!=rank) {
            COMPLEX_T* recv= _zsplit_buff + dyFLoc[2*n]*(lzFSize*xFSize);
            node->relion_MPI_IRecv(recv, lzFSize*xFSize*dyFLoc[2*n+1], MY_MPI_COMPLEX, n, MPI_TAG_EXCHANGE_BACK_BASE, comm);
        }
    }
    // pack all data to send buffer and launch non-blocking send
    for(int n=0;n<size;++n) {
        if(n!=rank) {
            // source ptr (z,x,y_split)
            COMPLEX_T* ptr = _ysplit_buff + dzFLoc[2*n];
            COMPLEX_T* s_buff = h_buff + dzFLoc[2*n]*xFSize*lyFSize;
            for(int i=0;i<lyFSize;++i) {
                for(int j=0;j<xFSize;++j) {
                    for(int k=0;k<dzFLoc[2*n+1];++k) {
                        s_buff[i*xFSize*dzFLoc[2*n+1]+j*dzFLoc[2*n+1]+k] = ptr[i*xFSize*zFSize+j*zFSize+k];
                    }
                }
            }
            node->relion_MPI_ISend(s_buff, dzFLoc[2*n+1]*xFSize*lyFSize, MY_MPI_COMPLEX, n, MPI_TAG_EXCHANGE_BACK_BASE, comm);
        }
    }
    // copy my own data
    for(int i=0;i<lyFSize;++i) {
        int ii=i+lyFStart;
        for(int j=0;j<xFSize;++j) {
            for(int k=0;k<lzFSize;++k) {
                int kk=k+lzFStart;
                _zsplit_buff[ii*lzFSize*xFSize+j*lzFSize+k] = _ysplit_buff[i*zFSize*xFSize+j*zFSize+kk];
            }
        }
    }
    // wait for send/recv
    MPI_Status status;
    node->relion_MPI_WaitAll(status);
}
//-------------------------------------------------------------------------------------------------
__global__ void exchange_unpack_kernel(COMPLEX_T* _ysplit_buff, COMPLEX_T* _recv_buff, int dzFLoc_start, int dzFLoc_size, int xFSize, int lyFSize, int zFSize)
{
    int j = blockIdx.x * TILE_DIM + threadIdx.x;
    int i = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    if(i >= lyFSize || j >= xFSize) return ;
    COMPLEX_T* recv= _recv_buff + i*xFSize*dzFLoc_size + j*dzFLoc_size;
    COMPLEX_T* ptr = _ysplit_buff + i*xFSize*zFSize + j*zFSize + dzFLoc_start;
    for(int k=0;k<dzFLoc_size;++k) {
        ptr[k] = recv[k];
    }
}
// (z_split,x,y) => (z,x,y_split), gpu version
void DistributedCudaFFT::exchange_gpu(COMPLEX_T* _zsplit_buff, COMPLEX_T* _ysplit_buff)
{
    // launch nonblocking send/recv
    for(int n=0;n<size;++n) {
        if(n!=rank) {
            // source ptr (z_split,x,dst_y_split)
            COMPLEX_T* ptr = _zsplit_buff + dyFLoc[2*n]*lzFSize*xFSize;
            node->relion_MPI_ISend(ptr, dyFLoc[2*n+1]*lzFSize*xFSize, MY_MPI_COMPLEX, n, MPI_TAG_EXCHANGE_BASE, comm);
            // recv ptr (dst_z_split,x,y_split)
            COMPLEX_T* recv= d_buff + lyFSize*dzFLoc[2*n]*xFSize;
            node->relion_MPI_IRecv(recv, lyFSize*dzFLoc[2*n+1]*xFSize, MY_MPI_COMPLEX, n, MPI_TAG_EXCHANGE_BASE, comm);
        }
    }
    // copy my own data
    COMPLEX_T* ptr = _zsplit_buff + lzFSize*xFSize*lyFStart;
    dim3 dimGrid((xFSize+TILE_DIM-1)/TILE_DIM,(lyFSize+BLOCK_ROWS-1)/BLOCK_ROWS, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    exchange_unpack_kernel<<<dimGrid,dimBlock,0,stream>>>(_ysplit_buff, ptr, lzFStart, lzFSize, xFSize, lyFSize, zFSize);
    // before waiting, prepare all streams for unpack kernels
    cudaStream_t* streams = new cudaStream_t[size];
    for(int n=0;n<size;++n) {
        if(n!=rank) {
            // create a nonblocking stream for this node's recv buffer
            cudaStreamCreateWithFlags(&streams[n],cudaStreamNonBlocking);
        }
    }
    // now wait for send/recv
    MPI_Status status;
    node->relion_MPI_WaitAll(status);
    // copy recv buffer to exact place
    for(int n=0;n<size;++n) {
        if(n!=rank) {
            // recv ptr (dst_z_split,x,y_split)
            COMPLEX_T* recv = d_buff + dzFLoc[2*n]*xFSize*lyFSize;
            dim3 dimGrid((xFSize+TILE_DIM-1)/TILE_DIM,(lyFSize+BLOCK_ROWS-1)/BLOCK_ROWS, 1);
            dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
            exchange_unpack_kernel<<<dimGrid,dimBlock,0,streams[n]>>>(_ysplit_buff, recv, dzFLoc[2*n], dzFLoc[2*n+1], xFSize, lyFSize, zFSize);
        }
    }
    // sync and destroy streams
    for(int n=0;n<size;++n) {
        if(n!=rank) {
            cudaStreamSynchronize(streams[n]);
            cudaStreamDestroy(streams[n]);
        }
    }
    streamSync();
}
//-------------------------------------------------------------------------------------------------
__global__ void copy_back_kernel(COMPLEX_T* _ysplit_buff, COMPLEX_T* _zsplit_buff, int lyFStart, int lyFSize, int lzFStart, int lzFSize, int xFSize, int zFSize)
{
	//threadid
    int j = blockIdx.x * TILE_DIM + threadIdx.x;
    int i = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    int ii= i+lyFStart;
    if(i >= lyFSize || j >= xFSize) return ;
    for(int k=0;k<lzFSize;++k) {
        int kk = k+lzFStart;
        _zsplit_buff[ii*lzFSize*xFSize+j*lzFSize+k] = _ysplit_buff[i*zFSize*xFSize+j*zFSize+kk];
    }
}

// copy block of (dst_z_split,x,src_y_split) to send buffer
__global__ void back_pack_kernel(COMPLEX_T* _ysplit_buff, COMPLEX_T* _send_buff, int dzFLoc_start, int dzFLoc_size, int xFSize, int lyFSize, int zFSize)
{
    int j = blockIdx.x * TILE_DIM + threadIdx.x;
    int i = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    if(i >= lyFSize || j >= xFSize) return ;
    COMPLEX_T* send= _send_buff + i*xFSize*dzFLoc_size + j*dzFLoc_size;
    COMPLEX_T* ptr = _ysplit_buff + i*xFSize*zFSize + j*zFSize + dzFLoc_start;
    for(int k=0;k<dzFLoc_size;++k) {
        send[k] = ptr[k];
    }
}
// (z,x,y_split) => (z_split,x,y), gpu version
void DistributedCudaFFT::exchange_back_gpu(COMPLEX_T* _ysplit_buff, COMPLEX_T* _zsplit_buff)
{
    // Method : 1. pack all sending data to buffer through gpu kernel
    //          2. launch self copy kernel
    //          3. launch non-block recv (with CUDA-aware MPI)
    //          4. launch non-block send (with CUDA-aware MPI) for packed buffer
    //          5. wait for send/recv and self-copy kernel to finish
    // first launch send buffer gathering kernels
#ifdef INNER_TIMING
    double start, end;
    start = omp_get_wtime();
#endif
    cudaStream_t* streams = new cudaStream_t[size];
    for(int n=0;n<size;++n) {
        if(n!=rank) {
            // create a nonblocking stream for this node's send buffer
            cudaStreamCreateWithFlags(&streams[n],cudaStreamNonBlocking);
            // launch buffer copy kernel
            COMPLEX_T* ptr = d_buff + dzFLoc[2*n]*xFSize*lyFSize;
            dim3 dimGrid((xFSize+TILE_DIM-1)/TILE_DIM,(lyFSize+BLOCK_ROWS-1)/BLOCK_ROWS, 1);
            dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
            back_pack_kernel<<<dimGrid,dimBlock,0,streams[n]>>>(_ysplit_buff, ptr, dzFLoc[2*n], dzFLoc[2*n+1], xFSize, lyFSize, zFSize);
#ifndef USE_CUDA_AWARE
            // CUDA AWARE not enabled, copy buffer to cpu side
            COMPLEX_T* h_ptr = buff + dzFLoc[2*n]*xFSize*lyFSize;
            cudaMemcpyAsync(h_ptr, ptr, sizeof(COMPLEX_T)*dzFLoc[2*n+1]*xFSize*lyFSize, cudaMemcpyDeviceToHost, streams[n]);
#endif
        }
    }
#ifdef INNER_TIMING
    end = omp_get_wtime();
    if(printTimes)
        printf("\t\tRank %d : launch pack kernel time %lf\n",rank,end-start);
    start = omp_get_wtime();
#endif
#ifdef USE_CUDA_AWARE
    // then launch self copy kernel
    dim3 dimGrid((xFSize+TILE_DIM-1)/TILE_DIM,(lyFSize+BLOCK_ROWS-1)/BLOCK_ROWS, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    copy_back_kernel<<<dimGrid,dimBlock,0,stream>>>(_ysplit_buff,_zsplit_buff, lyFStart, lyFSize, lzFStart, lzFSize, xFSize, zFSize);
#ifdef INNER_TIMING
    end = omp_get_wtime();
    if(printTimes)
        printf("\t\tRank %d : launch copy back kernel time %lf\n",rank,end-start);
    start = omp_get_wtime();
#endif
#endif
    // then start all nonblocking recv procedure
    for(int n=0;n<size;++n) {
        if(n!=rank) {
#ifdef USE_CUDA_AWARE
            COMPLEX_T* recv= _zsplit_buff + dyFLoc[2*n]*(lzFSize*xFSize);
#else
            COMPLEX_T* recv= h_buff + dyFLoc[2*n]*(lzFSize*xFSize);
#endif
            //printf("\t\tRank %d will recv %d data from rank %d\n",rank,lzFSize*xFSize*dyFLoc[2*n+1],n);
            node->relion_MPI_IRecv(recv, lzFSize*xFSize*dyFLoc[2*n+1], MY_MPI_COMPLEX, n, MPI_TAG_EXCHANGE_BACK_BASE, comm);
        }
    }
#ifdef INNER_TIMING
    end = omp_get_wtime();
    if(printTimes)
        printf("\t\tRank %d : mpi_irecv time %lf\n",rank,end-start);
    start = omp_get_wtime();
#endif
    // then send out all ready buffers
    for(int n=0;n<size;++n) {
        if(n!=rank) {
            // make sure the buffer is ready
            cudaStreamSynchronize(streams[n]);
            // send out ready buffer
#ifdef USE_CUDA_AWARE
            COMPLEX_T* ptr = d_buff + dzFLoc[2*n]*xFSize*lyFSize;
#else
            COMPLEX_T* ptr = buff + dzFLoc[2*n]*xFSize*lyFSize;
#endif
            //printf("\t\tRank %d will send %d data to rank %d\n",rank,dzFLoc[2*n+1]*xFSize*lyFSize,n);
            node->relion_MPI_ISend(ptr, dzFLoc[2*n+1]*xFSize*lyFSize, MY_MPI_COMPLEX, n, MPI_TAG_EXCHANGE_BACK_BASE, comm);
            // release stream resources
            cudaStreamDestroy(streams[n]);
        }
    }
#ifdef INNER_TIMING
    end = omp_get_wtime();
    if(printTimes)
        printf("\t\tRank %d : wait for packing & launch mpi_isend time %lf\n",rank,end-start);
    start = omp_get_wtime();
#endif
    // release resources before waiting for communication to hidden the latency
    delete[] streams;
    // wait for send/recv
    MPI_Status status;
    node->relion_MPI_WaitAll(status);
#ifdef INNER_TIMING
    end = omp_get_wtime();
    if(printTimes)
        printf("\t\tRank %d : wait for send/recv time %lf\n",rank,end-start);
    start = omp_get_wtime();
#endif
#ifndef USE_CUDA_AWARE
    cudaMemcpy(_zsplit_buff, h_buff, sizeof(COMPLEX_T)*yFSize*lzFSize*xFSize, cudaMemcpyHostToDevice);
    // then launch self copy kernel
    dim3 dimGrid((xFSize+TILE_DIM-1)/TILE_DIM,(lyFSize+BLOCK_ROWS-1)/BLOCK_ROWS, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    copy_back_kernel<<<dimGrid,dimBlock,0,stream>>>(_ysplit_buff,_zsplit_buff, lyFStart, lyFSize, lzFStart, lzFSize, xFSize, zFSize);
#endif
    streamSync();
#ifdef INNER_TIMING
    end = omp_get_wtime();
    if(printTimes)
        printf("\t\tRank %d : wait for self copying time %lf\n",rank,end-start);
#endif
}
//--------------------------------------------------------------------------------------------------
// Precondition : 
//     1. data and locs distributed 
//     2. data already sent to gpu 
//     3. plan is set and no batch
// Results : transposed and with y splitted : (z,x,y_split)
void DistributedCudaFFT::forward_notrans()
{
	// first 2D-FFT across x-y
	p2DFFT_trans->forward();
	// transpose : (x,y,z) => (z,x,y)
    transpose_gpu(d_buff,p2DFFT_trans->fouriers.d_ptr,xFSize*yFSize,lzFSize);
    streamSync(); // ensure the input of exchange_gpu is already calculated.
	// exchange data between all ranks : (z_split,x,y) => (z,x,y_split)
	exchange_gpu(d_buff, p1DFFT_trans->reals.d_ptr);
	// 1D-FFT across z
    p1DFFT_trans->forward();
	// div with size
	int fsize = zFSize*xFSize*lyFSize;
	int size  = xSize*ySize*zSize;
    int blknum= (fsize+BLOCK_SIZE-1) / BLOCK_SIZE;
	ScaleComplexPointwise_kernel<<<blknum,BLOCK_SIZE,0,stream>>>(p1DFFT_trans->fouriers.d_ptr,fsize,size);
}

#endif