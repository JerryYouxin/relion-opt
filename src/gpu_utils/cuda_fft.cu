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