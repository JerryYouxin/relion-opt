#ifndef __RELION_PFFTW_H
#define __RELION_PFFTW_H

#ifndef __RELIONFFTW_H
#define FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(V) \
    for (long int k = 0, kp = 0; k<ZSIZE(V); k++, kp = (k < XSIZE(V)) ? k : k - ZSIZE(V)) \
    	for (long int i = 0, ip = 0 ; i<YSIZE(V); i++, ip = (i < XSIZE(V)) ? i : i - YSIZE(V)) \
    		for (long int j = 0, jp = 0; j<XSIZE(V); j++, jp = j)

#define FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D(V) \
	for (long int i = 0, ip = 0 ; i<YSIZE(V); i++, ip = (i < XSIZE(V)) ? i : i - YSIZE(V)) \
		for (long int j = 0, jp = 0; j<XSIZE(V); j++, jp = j)

#define FFTW_ELEM(V, kp, ip, jp) \
    (DIRECT_A3D_ELEM((V),((kp < 0) ? (kp + ZSIZE(V)) : (kp)), ((ip < 0) ? (ip + YSIZE(V)) : (ip)), (jp)))

#define FFTW2D_ELEM(V, ip, jp) \
    (DIRECT_A2D_ELEM((V), ((ip < 0) ? (ip + YSIZE(V)) : (ip)), (jp)))
#endif

#include "src/mpi.h"
#include <fftw3-mpi.h>
#include "src/multidim_array.h"
#include "src/funcs.h"
#include "src/tabfuncs.h"
#include "src/complex.h"
#include "src/CPlot2D.h"

#define DISTRIBUTE_REAL true
#define DISTRIBUTE_FOURIER false

class DistributedFourierTransformer
{
public:
    MultidimArray<RFLOAT> *fReal;
    MultidimArray<Complex> fFourier;
    // working arrays (distributed)
    RFLOAT * fReal_local;
    Complex* fFourier_local;

    MpiNode *node;

    MPI_Comm comm;
    int rank;
    int size;

    bool thread_ok;

    int dimension;
    int xdim,ydim,zdim;
    int fxdim,fydim,fzdim;

    int local_fsize;

    std::ptrdiff_t alloc_local, local_n0, local_0_start; //, local_n1, local_1_start;
    int *dLocInfo;

#ifdef RELION_SINGLE_PRECISION
    /* fftw Forward plan */
    fftwf_plan fPlanForward;

    /* fftw Backward plan */
    fftwf_plan fPlanBackward;
#else
    /* fftw Forward plan */
    fftw_plan fPlanForward;

    /* fftw Backward plan */
    fftw_plan fPlanBackward;
#endif

    bool plans_are_set;
public:
    /** Default constructor */
    DistributedFourierTransformer(MpiNode* _node, MPI_Comm _comm = MPI_COMM_WORLD, int nr_threads=1);

    /** Destructor */
    ~DistributedFourierTransformer();

    void setFourier(MultidimArray<Complex> input) { fFourier.moveFrom(input); }
    MultidimArray<Complex>& getFourierReference() { return fFourier; }

    void FourierTransform();

    void inverseFourierTransform();

    void Transform(int sign);

    void clear();

    void cleanup();
    
    void setSize(int nx, int ny, int nz);

    void setReal(MultidimArray<RFLOAT> &img);

    void init(MPI_Comm _comm, int nr_threads=1);

    void destroyPlans();
    // distribute the input data
    void distribute(bool real=false);
    // only distribute the location info
    void distribute_location();
    // merge the result data together to rank 0
    void merge(bool real=false);
    void mergeTo(MultidimArray<RFLOAT>& out);
};

// Window an FFTW-centered Fourier-transform to a given size
template<class T>
void distributedWindowFourierTransform(MultidimArray<T > &in,
                              MultidimArray<T > &out,
                              long int newdim,
                              long int local_0_start, 
                              long int local_n0)
{
    // Check size of the input array
    if (YSIZE(in) > 1 && YSIZE(in)/2 + 1 != XSIZE(in))
        REPORT_ERROR("windowFourierTransform ERROR: the Fourier transform should be of an image with equal sizes in all dimensions!");
    long int newhdim = newdim/2 + 1;

    // If same size, just return input
    // Sjors 5dec2017: only check for xdim is not enough, even/off ydim leaves ambiguity for dim>1
    if ( newdim == YSIZE(in) && newhdim == XSIZE(in) )
    {
        out = in;
        return;
    }

    // Otherwise apply a windowing operation
    // Initialise output array
    switch (in.getDim())
    {
    case 1:
        out.initZeros(newhdim);
        break;
    case 2:
        out.initZeros(newdim, newhdim);
        break;
    case 3:
        out.initZeros(newdim, newdim, newhdim);
        break;
    default:
        REPORT_ERROR("windowFourierTransform ERROR: dimension should be 1, 2 or 3!");
    }
    if (newhdim > XSIZE(in))
    {
        long int max_r2 = (XSIZE(in) -1) * (XSIZE(in) - 1);
        FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(in)
        {
            // Make sure windowed FT has nothing in the corners, otherwise we end up with an asymmetric FT!
            if (kp*kp + ip*ip + jp*jp <= max_r2)
                FFTW_ELEM(out, kp, ip, jp) = FFTW_ELEM(in, kp, ip, jp);
        }
    }
    else
    {
        FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(out)
        {
            FFTW_ELEM(out, kp, ip, jp) = FFTW_ELEM(in, kp, ip, jp);
        }
    }
}

// Same as above, acts on the input array directly
template<class T>
void distributedWindowFourierTransform(MultidimArray<T > &V,
                              long int newdim, long int local_0_start, long int local_n0)
{
    // Check size of the input array
    if (YSIZE(V) > 1 && YSIZE(V)/2 + 1 != XSIZE(V))
        REPORT_ERROR("windowFourierTransform ERROR: the Fourier transform should be of an image with equal sizes in all dimensions!");
    long int newhdim = newdim/2 + 1;

    // If same size, just return input
    // Sjors 5dec2017: only check for xdim is not enough, even/off ydim leaves ambiguity for dim>1
    if ( newdim == YSIZE(V) && newhdim == XSIZE(V) )
    {
        return;
    }

    MultidimArray<T> tmp;
    distributedWindowFourierTransform<T>(V, tmp, newdim, local_0_start, local_n0);
    V.moveFrom(tmp);

}

#endif