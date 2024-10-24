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
//! \param pitchCurr The pitch (or stride) in memory corresponding to the TDim grid in the accelerator's memory.
//!              This is used to calculate memory offsets when accessing elements in the current buffer.
//! \param pitchNext The pitch used to calculate memory offsets when accessing elements in the next buffer.
//! \param haloSize Size of halo required for our stencil in {Y, X} (above and to the left)
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t
struct StencilKernel
{
    // **************************************************************
    // * Change to use MDSpan                                       *
    // **************************************************************

    template<typename TAcc, typename TDim, typename TIdx, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TMdSpan const spanUCurrBuf,
        TMdSpan spanUNextBuf,
        alpaka::Vec<TDim, TIdx> const& haloSize,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        // Each kernel executes one element
        double const rX = dt / (dx * dx);
        double const rY = dt / (dy * dy);

        // offset for halo, as we only want to go over core cells
        auto globalIdx = gridThreadIdx + haloSize;

        spanUNextBuf(globalIdx[0], globalIdx[1]) =
                spanUCurrBuf(globalIdx[0], globalIdx[1]) * (1.0 - 2.0 * rX - 2.0 * rY)
                + spanUCurrBuf(globalIdx[0], globalIdx[1] + 1) * rX
                + spanUCurrBuf(globalIdx[0], globalIdx[1] - 1) * rX
                + spanUCurrBuf(globalIdx[0] - 1, globalIdx[1]) * rY
                + spanUCurrBuf(globalIdx[0] + 1, globalIdx[1]) * rY;
    }
};
