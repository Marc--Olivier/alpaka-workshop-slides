/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

//! alpaka version of explicit finite-difference 2D heat equation solver
//!
//! Solving equation u_t(x, t) = u_xx(x, t) + u_yy(y, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x and y
//!
//! \param uCurrBuf Current buffer with grid values of u for each x, y pair and the current value of t:
//!                 u(x, y, t) | t = t_current
//! \param uNextBuf resulting grid values of u for each x, y pair and the next value of t:
//!              u(x, y, t) | t = t_current + dt
//! \param chunkSize The size of the chunk or tile that the user divides the problem into. This defines the size of the
//!                  workload handled by each thread block.
//! \param haloSize Size of halo required for our stencil in {Y, X} (above and to the left)
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t
// **************************************************************
// * Pass the shared memory size at compile time to             *
// * use static share memory                                    *
// **************************************************************
template<uint32_t T_SharedMemSize1D>
struct StencilKernel
{
    template<typename TAcc, typename TMdSpan, typename TDim, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TMdSpan uCurrBuf,
        TMdSpan uNextBuf,
        alpaka::Vec<TDim, TIdx> const& chunkSize,
        alpaka::Vec<TDim, TIdx> const& haloSize,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        // **************************************************************
        // * Use shared memory for faster access to neighbour indices   *
        // * 1 - Create shared memory                                   *
        // * 2 - Fill shared memory with current values                 *
        // * 3 - Read from shared memory to update the next buffer      *
        // **************************************************************
        auto & sdata = alpaka::declareSharedVar<double[ T_SharedMemSize1D ], __COUNTER__ >(acc);
        auto sharedMemSize2D = chunkSize + haloSize + haloSize;

        // Get indexes
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const blockStartThreadIdx = gridBlockIdx * chunkSize;

        // Each kernel executes one element
        double const rX = dt / (dx * dx);
        double const rY = dt / (dy * dy);

        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

        for(auto i = blockThreadIdx[0]; i < sharedMemSize2D[0]; i += blockThreadExtent[0])
        {
            for(auto j = blockThreadIdx[1]; j < sharedMemSize2D[1]; j += blockThreadExtent[1])
            {
                auto localIdx = alpaka::Vec<TDim, TIdx>(i, j);
                auto const globalIdx = localIdx + blockStartThreadIdx; // Don't add halo in order to fill in its values.
                auto localIdx1d = alpaka::mapIdx<1>(localIdx, sharedMemSize2D)[0];
                sdata[localIdx1d] = uCurrBuf(globalIdx[0], globalIdx[1]);
            }
        }

        alpaka::syncBlockThreads(acc);

        // go over only core cells and update nextBuf
        for(auto i = blockThreadIdx[0]; i < chunkSize[0]; i += blockThreadExtent[0])
        {
            for(auto j = blockThreadIdx[1]; j < chunkSize[1]; j += blockThreadExtent[1])
            {
                // offset for halo, as we only want to go over core cells
                auto localIdx = alpaka::Vec<TDim, TIdx>(i, j) + haloSize;
                auto localIdx1d = alpaka::mapIdx<1>(localIdx, sharedMemSize2D)[0];
                auto const globalIdx = localIdx + blockStartThreadIdx;

                uNextBuf(globalIdx[0], globalIdx[1])
                    = sdata[localIdx1d] * (1.0 - 2.0 * rX - 2.0 * rY)
                      + sdata[localIdx1d + 1] * rX + sdata[localIdx1d - 1] * rX
                      + sdata[localIdx1d + sharedMemSize2D[1]] * rY + sdata[localIdx1d  - sharedMemSize2D[1]] * rY;
            }
        }
    }
};
