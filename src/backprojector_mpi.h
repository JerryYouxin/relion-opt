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

#ifndef BACKPROJECTOR_MPI_H_
#define BACKPROJECTOR_MPI_H_
#include "src/mpi.h"
#include "src/backprojector.h"
#include "src/pfftw.h"

class BackProjectorMpi: public BackProjector
{
public:
    MpiNode *node;
    BackProjectorMpi(MpiNode* _node, int _ori_size, int _ref_dim, FileName fn_sym,
			      int _interpolator = TRILINEAR, float _padding_factor_3d = 2, int _r_min_nn = 10,
			      int _blob_order = 0, RFLOAT _blob_radius = 1.9, RFLOAT _blob_alpha = 15, int _data_dim = 2, bool _skip_gridding = false) :
                  BackProjector(_ori_size, _ref_dim, fn_sym,
			      _interpolator, _padding_factor_3d, _r_min_nn,
			      _blob_order, _blob_radius, _blob_alpha, _data_dim, _skip_gridding) 
    {
    	node = _node;
	}
    void reconstruct(MultidimArray<RFLOAT> &vol_out,
                     int max_iter_preweight,
                     bool do_map,
                     RFLOAT tau2_fudge,
                     MultidimArray<RFLOAT> &tau2,
                     MultidimArray<RFLOAT> &sigma2,
                     MultidimArray<RFLOAT> &evidence_vs_prior,
                     MultidimArray<RFLOAT> &fourier_coverage,
                     MultidimArray<RFLOAT> fsc,
                     RFLOAT normalise = 1.,
                     bool update_tau2_with_fsc = false,
                     bool is_whole_instead_of_half = false,
                     int nr_threads = 1,
                     int minres_map = -1,
                     bool printTimes= false);
    void convoluteBlobRealSpace(DistributedFourierTransformer& transformer, bool do_mask = false);
    void windowToOridimRealSpace(DistributedFourierTransformer &transformer, MultidimArray<RFLOAT> &Mout, int nr_threads = 1, bool printTimes = false);
#ifdef CUDA
    void reconstruct_gpu(int rank,
					 MultidimArray<RFLOAT> &vol_out,
                     int max_iter_preweight,
                     bool do_map,
                     RFLOAT tau2_fudge,
                     MultidimArray<RFLOAT> &tau2,
                     MultidimArray<RFLOAT> &sigma2,
                     MultidimArray<RFLOAT> &evidence_vs_prior,
                     MultidimArray<RFLOAT> &fourier_coverage,
                     MultidimArray<RFLOAT> fsc,
                     RFLOAT normalise = 1.,
                     bool update_tau2_with_fsc = false,
                     bool is_whole_instead_of_half = false,
                     int nr_threads = 1,
                     int minres_map = -1,
                     bool printTimes= false);
#endif
};

#endif