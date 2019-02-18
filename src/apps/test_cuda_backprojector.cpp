#include <stdio.h>
#include "src/time.h"

#define CUDA

#include "src/fftw.h"

#ifdef CUDA
#include "src/gpu_utils/cuda_settings.h"
#include <cuda_runtime.h>
#include "src/gpu_utils/cuda_mem_utils.h"
#include "src/gpu_utils/cuda_fft.h"
#endif

#include "src/backprojector_mpi.h"
#include <time.h>

class TestTimer
{
    private:
    static struct timeval _start,_end;
    public:
    static void start() {
        gettimeofday(&_start,NULL);
    }
    static void stop() {
        gettimeofday(&_end, NULL);
    }
    static double elaspedInSeconds() {
        return (double)(_end.tv_sec-_start.tv_sec) + (double)(_end.tv_usec-_start.tv_usec)/1e6;
    }
    static double elaspedInMiliSeconds() {
        return (double)(_end.tv_sec-_start.tv_sec) * 1e3 + (double)(_end.tv_usec-_start.tv_usec)/1e3;
    }
    static double elaspedInUSeconds() {
        return (double)(_end.tv_sec-_start.tv_sec) * 1e6 + (double)(_end.tv_usec-_start.tv_usec);
    }
    static void printTime(const char* s) {
        printf("++ %s Time : %lf ms\n",s,elaspedInMiliSeconds());
    }
};

struct timeval TestTimer::_start;
struct timeval TestTimer::_end;


RFLOAT frand(RFLOAT fMin, RFLOAT fMax)
{
    double f = (double)rand() / RAND_MAX;
    return (RFLOAT)(fMin + f * (fMax - fMin));
}

void init_random(RFLOAT* A, int size)
{
    for(int i=0;i<size;++i) {
        A[i] = frand(-1,1);
    }
}

void init_random(__COMPLEX_T* A, int size)
{
    init_random((RFLOAT*)A,2*size);
}

void init_from(RFLOAT* dst, RFLOAT* src, int size)
{
    memcpy(dst,src,size*sizeof(RFLOAT));
}

void init_from(__COMPLEX_T* dst, __COMPLEX_T* src, int size)
{
    memcpy(dst,src,size*sizeof(__COMPLEX_T));
}

bool validate(RFLOAT* opt, RFLOAT* ref, int xs, int ys, int zs, const char* prefix)
{
    double sum=0.0;
    for(int k=0;k<zs;++k) {
        for(int i=0;i<ys;++i) {
            for(int j=0;j<xs;++j) {
                int idx = k*xs*ys+i*xs+j;
                double err = fabs(opt[idx]-ref[idx]);
                if(err>1e-4) {
                    printf("%s** Error at %d %d %d : %lf(opt) vs %lf(ref), err=%lf\n",prefix,k,i,j,opt[idx],ref[idx],err);
                    return false;
                }
                sum += err;
            }
        }
    }
    printf("%s-- Check Passed, err sum = %lf\n", prefix, sum);
    return true;
}

bool validate(__COMPLEX_T* opt, __COMPLEX_T* ref, int xs, int ys, int zs, const char* prefix)
{
    double sum=0.0;
    for(int k=0;k<zs;++k) {
        for(int i=0;i<ys;++i) {
            for(int j=0;j<xs;++j) {
                int idx = k*xs*ys+i*xs+j;
                double err = fabs(opt[idx].x-ref[idx].x) + fabs(opt[idx].y-ref[idx].y);
                if(err>1e-4) {
                    printf("%s** Error at %d %d %d : %lf %lf(opt) vs %lf %lf(ref), err=%lf\n",prefix,k,i,j,opt[idx].x,opt[idx].y,ref[idx].x,ref[idx].y,err);
                    return false;
                }
                sum += err;
            }
        }
    }
    printf("%s-- Check Passed, err sum = %lf\n", prefix, sum);
    return true;
}

void printData(RFLOAT* data, int js, int is, int ks, int xi, int yi, int zi, int xs, int ys, int zs, const char* prefix)
{
    printf("%sstart from %d %d %d, will print %d %d %d size (of %d %d %d)\n",prefix,js,is,ks,xi,yi,zi,xs,ys,zs);
    for(int k=ks;k<ks+zi;++k) {
        printf("%ssection %d\t: [\n",prefix,k);
        for(int i=is;i<is+yi;++i) {
            printf("%s\t",prefix);
            for(int j=js;j<js+xi;++j) {
                int idx = k*xs*ys+i*xs+j;
                printf("\t%lf",data[idx]);
            }
            char x[3];
            if(i==yi-1) { x[0]=']'; x[1]='\n'; x[2]='\0'; }
            else        { x[0]='\n';x[1]='\0'; }
            printf("\t%s",x);
        }
        printf("\n");
    }
    return;
}

void printData(__COMPLEX_T* data, int js, int is, int ks, int xi, int yi, int zi, int xs, int ys, int zs, const char* prefix)
{
    printf("%sstart from %d %d %d, will print %d %d %d size (of %d %d %d)\n",prefix,js,is,ks,xi,yi,zi,xs,ys,zs);
    for(int k=ks;k<ks+zi;++k) {
        printf("%ssection %d\t: [\n",prefix,k);
        for(int i=is;i<is+yi;++i) {
            printf("%s\t",prefix);
            for(int j=js;j<js+xi;++j) {
                int idx = k*xs*ys+i*xs+j;
                printf("\t(%lf,%lf)",data[idx].x,data[idx].y);
            }
            char x[3];
            if(i==yi-1) { x[0]=']'; x[1]='\n'; x[2]='\0'; }
            else        { x[0]='\n';x[1]='\0'; }
            printf("\t%s",x);
        }
        printf("\n");
    }
    return;
}

void runTest(int argc, char **argv)
{
    MpiNode *node = new MpiNode(argc,argv);
    node->groupInit(2); // class number 1 for testing
    BackProjector bp(256,3,"c1");
    BackProjector bp_reference(256,3,"c1");
    bp.pad_size = 263;
    bp_reference.pad_size = 263;

    bp.r_max = 65;
    bp_reference.r_max = 65;
    if(!node->isMaster()) {
        RFLOAT tau2_fudge = 1.0;
        int max_iter_preweight = 10;
        RFLOAT normalise;
        if(node->cls_rank==1) {
            normalise = 16445.6;
        } else {
            normalise = 16454.6;
        }
        bool update_tau2_with_fsc = 1;
        bool is_whole_instead_of_half = 0;
        int nr_threads = 7;
        RFLOAT minres_map = 5;

        printf("## INFO: allocate multidim array...\n");

        MultidimArray<RFLOAT> vol_out;
        MultidimArray<RFLOAT> tau2;
        MultidimArray<RFLOAT> sigma2;
        MultidimArray<RFLOAT> evidence_vs_prior;
        MultidimArray<RFLOAT> fourier_coverage;

        MultidimArray<RFLOAT> vol_out_reference;
        MultidimArray<RFLOAT> tau2_reference;
        MultidimArray<RFLOAT> sigma2_reference;
        MultidimArray<RFLOAT> evidence_vs_prior_reference;
        MultidimArray<RFLOAT> fourier_coverage_reference;

        MultidimArray<RFLOAT> fsc;

        printf("## INFO: multidim array reshaping...\n");

        fsc.reshape(bp.ori_size/2 + 1);
        tau2.reshape(bp.ori_size/2 + 1);
        tau2_reference.reshape(bp.ori_size/2 + 1);
        sigma2.reshape(bp.ori_size/2 + 1);
        sigma2_reference.reshape(bp.ori_size/2 + 1);
        evidence_vs_prior.reshape(bp.ori_size/2 + 1);
        evidence_vs_prior_reference.reshape(bp.ori_size/2 + 1);
        fourier_coverage.reshape(bp.ori_size/2 + 1);
        fourier_coverage_reference.reshape(bp.ori_size/2 + 1);

        bp.weight.reshape(bp.pad_size,bp.pad_size,bp.pad_size/2+1);
        bp_reference.weight.reshape(bp.pad_size,bp.pad_size,bp.pad_size/2+1);
        bp.data.reshape(bp.pad_size,bp.pad_size,bp.pad_size/2+1);
        bp_reference.data.reshape(bp.pad_size,bp.pad_size,bp.pad_size/2+1);

        bp.weight.setXmippOrigin();
        bp_reference.weight.setXmippOrigin();
        bp.data.setXmippOrigin();
        bp_reference.data.setXmippOrigin();

        bp.weight.xinit = 0;
        bp_reference.weight.xinit = 0;
        bp.data.xinit = 0;
        bp_reference.data.xinit = 0;
        printf("## INFO: reading input data\n");
        char fn[200];
        sprintf(fn, "reconstruct_initial_weight.%d.spi",node->rank);
        bp.weight.readBinary(fn);
        bp_reference.weight.readBinary(fn);
        sprintf(fn, "reconstruct_initial_data.%d.spi",node->rank);
        bp.data.readBinary(fn);
        bp_reference.data.readBinary(fn);
        sprintf(fn, "reconstruct_initial_fsc.%d.spi",node->rank);
        fsc.readBinary(fn);
        sprintf(fn, "reconstruct_initial_tau2.%d.spi",node->rank);
        tau2.readBinary(fn);
        tau2_reference.readBinary(fn);
        sprintf(fn, "reconstruct_initial_sigma2.%d.spi",node->rank);
        sigma2.readBinary(fn);
        sigma2_reference.readBinary(fn);
        sprintf(fn, "reconstruct_initial_data_vs_prior.%d.spi",node->rank);
        evidence_vs_prior.readBinary(fn);
        evidence_vs_prior_reference.readBinary(fn);
        sprintf(fn, "reconstruct_initial_fourier_coverage.%d.spi",node->rank);
        fourier_coverage.readBinary(fn);
        fourier_coverage_reference.readBinary(fn);
        if(node->grp_rank==0) {
            printf("## INFO: mpi reconstruct\n");
            TestTimer::start();
            bp.reconstruct_gpu(node->rank,
                        vol_out,
                        max_iter_preweight,
                        false,
                        tau2_fudge,
                        tau2,
                        sigma2,
                        evidence_vs_prior,
                        fourier_coverage,
                        fsc,
                        normalise,
                        update_tau2_with_fsc,
                        is_whole_instead_of_half,
                        nr_threads,
                        minres_map,
                        node->grp_rank==0);
            TestTimer::stop();
            TestTimer::printTime("GPU reconstruct");
        }
        MPI_Barrier(node->groupC);
        if(node->grp_rank==0) {
            printf("## INFO: ori reconstruct\n");
            TestTimer::start();
            bp_reference.reconstruct(vol_out_reference,
                     max_iter_preweight,
                     false,
                     tau2_fudge,
                     tau2_reference,
                     sigma2_reference,
                     evidence_vs_prior_reference,
                     fourier_coverage_reference,
                     fsc,
                     normalise,
                     update_tau2_with_fsc,
                     is_whole_instead_of_half,
                     nr_threads,
                     minres_map,
                     node->rank==1);
            TestTimer::stop();
            TestTimer::printTime("ORI reconstruct");
        }
        printf("## INFO: Rank %d Waiting for barrier\n",node->grp_rank);
        MPI_Barrier(node->slaveC);
        for(int n=0;n<node->size;++n) {
            if(n==node->rank && node->grp_rank==0) {
                bool result=true;
                printf("-- Checking vol_out...\n");
                result &= validate(vol_out.data,vol_out_reference.data,XSIZE(vol_out),YSIZE(vol_out),ZSIZE(vol_out),"\t");
                printf("-- Checking weight...\n");
                result &= validate(bp.weight.data,bp_reference.weight.data,XSIZE(bp_reference.weight),YSIZE(bp_reference.weight),ZSIZE(bp_reference.weight),"\t");
                printf("-- Checking data...\n");
                result &= validate((__COMPLEX_T*)bp.data.data,(__COMPLEX_T*)bp_reference.data.data,XSIZE(bp_reference.data),YSIZE(bp_reference.data),ZSIZE(bp_reference.data),"\t");
                printf("-- Checking tau2...\n");
                result &= validate(tau2.data,tau2_reference.data,XSIZE(tau2),YSIZE(tau2),ZSIZE(tau2),"\t");
                printf("-- Checking sigma2...\n");
                result &= validate(sigma2.data,sigma2_reference.data,XSIZE(sigma2),YSIZE(sigma2),ZSIZE(sigma2),"\t");
                printf("-- Checking evidence_vs_prior...\n");
                result &= validate(evidence_vs_prior.data,evidence_vs_prior_reference.data,XSIZE(evidence_vs_prior),YSIZE(evidence_vs_prior),ZSIZE(evidence_vs_prior),"\t");
                printf("-- Checking fourier_coverage...\n");
                result &= validate(fourier_coverage.data,fourier_coverage_reference.data,XSIZE(fourier_coverage),YSIZE(fourier_coverage),ZSIZE(fourier_coverage),"\t");
                if(result)
                    printf("-- All PASS...\n");
                else
                    printf("** CHECK FAILED **\n");
            }
            MPI_Barrier(node->slaveC);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    delete node;
}

void test_symmtrise(int argc, char **argv)
{
    MpiNode *node = new MpiNode(argc,argv);
    node->groupInit(2); // class number 1 for testing
    BackProjector bp(400,3,"I3");
    BackProjector bp_reference(400,3,"I3");
    bp.pad_size = 315;
    bp_reference.pad_size = 315;
    bp.r_max = 78;
    bp_reference.r_max = 78;
    bp.padding_factor = 2;
    bp_reference.padding_factor = 2;
    if(!node->isMaster()) {
        bp.weight.reshape(bp.pad_size,bp.pad_size,bp.pad_size/2+1);
        bp_reference.weight.reshape(bp.pad_size,bp.pad_size,bp.pad_size/2+1);
        bp.data.reshape(bp.pad_size,bp.pad_size,bp.pad_size/2+1);
        bp_reference.data.reshape(bp.pad_size,bp.pad_size,bp.pad_size/2+1);

        bp.weight.setXmippOrigin();
        bp_reference.weight.setXmippOrigin();
        bp.data.setXmippOrigin();
        bp_reference.data.setXmippOrigin();

        bp.weight.xinit = 0;
        bp_reference.weight.xinit = 0;
        bp.data.xinit = 0;
        bp_reference.data.xinit = 0;

        // warm up gpu
        printf("== warm up ==\n");
        bp.symmetrise_gpu(node->rank);
        printf("==  finish ==\n");
        // initialise
        init_random(bp.weight.data, NZYXSIZE(bp.weight));
        init_random((__COMPLEX_T*)bp.data.data, NZYXSIZE(bp.data));
        init_from(bp_reference.weight.data,bp.weight.data,NZYXSIZE(bp.weight));
        init_from((__COMPLEX_T*)bp_reference.data.data,(__COMPLEX_T*)bp.data.data,NZYXSIZE(bp.data));

        TestTimer::start();
        bp.symmetrise_gpu(node->rank);
        TestTimer::stop();
        TestTimer::printTime(" OMP SYMM ");
        TestTimer::start();
        bp_reference.symmetrise();
        TestTimer::stop();
        TestTimer::printTime(" ORI SYMM ");

        MPI_Barrier(node->slaveC);
        for(int n=0;n<node->size;++n) {
            if(n==node->rank) {
                bool result=true;
                printf("-- Checking weight...\n");
                result &= validate(bp.weight.data,bp_reference.weight.data,XSIZE(bp_reference.weight),YSIZE(bp_reference.weight),ZSIZE(bp_reference.weight),"\t");
                printf("-- Checking data...\n");
                result &= validate((__COMPLEX_T*)bp.data.data,(__COMPLEX_T*)bp_reference.data.data,XSIZE(bp_reference.data),YSIZE(bp_reference.data),ZSIZE(bp_reference.data),"\t");
                printData(bp.weight.data,0,155,1,5,5,1,XSIZE(bp.weight),YSIZE(bp.weight),ZSIZE(bp.weight),"\t");
                printData(bp_reference.weight.data,0,155,1,5,5,1,XSIZE(bp.weight),YSIZE(bp.weight),ZSIZE(bp.weight),"\t");
                printData((__COMPLEX_T*)bp.data.data,0,155,1,5,5,1,XSIZE(bp.data),YSIZE(bp.data),ZSIZE(bp.data),"\t");
                printData((__COMPLEX_T*)bp_reference.data.data,0,155,1,5,5,1,XSIZE(bp.data),YSIZE(bp.data),ZSIZE(bp.data),"\t");
                if(result)
                    printf("-- All PASS...\n");
                else
                    printf("** CHECK FAILED **\n");
            }
            MPI_Barrier(node->slaveC);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    delete node;
}

int main(int argc, char **argv)
{
    //runTest(argc,argv);
    test_symmtrise(argc,argv);
    return 0;
}