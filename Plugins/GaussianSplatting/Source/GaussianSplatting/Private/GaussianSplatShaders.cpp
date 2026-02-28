// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatShaders.h"
#include "ShaderParameterUtils.h"
#include "RHIStaticStates.h"

// Implement global shaders
IMPLEMENT_GLOBAL_SHADER(FGaussianSplatCalcViewDataCS, "/Plugin/GaussianSplatting/Private/CalcViewData.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FGaussianSplatCalcDistancesCS, "/Plugin/GaussianSplatting/Private/CalcDistances.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FGaussianSplatVS, "/Plugin/GaussianSplatting/Private/GaussianSplatRendering.usf", "MainVS", SF_Vertex);
IMPLEMENT_GLOBAL_SHADER(FGaussianSplatPS, "/Plugin/GaussianSplatting/Private/GaussianSplatRendering.usf", "MainPS", SF_Pixel);
IMPLEMENT_GLOBAL_SHADER(FRadixSortCountCS, "/Plugin/GaussianSplatting/Private/RadixSort.usf", "CountCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FRadixSortPrefixSumCS, "/Plugin/GaussianSplatting/Private/RadixSort.usf", "PrefixSumCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FRadixSortDigitPrefixSumCS, "/Plugin/GaussianSplatting/Private/RadixSort.usf", "DigitPrefixSumCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FRadixSortScatterCS, "/Plugin/GaussianSplatting/Private/RadixSort.usf", "ScatterCS", SF_Compute);

// Cluster culling shaders
IMPLEMENT_GLOBAL_SHADER(FClusterCullingResetCS, "/Plugin/GaussianSplatting/Private/ClusterCulling.usf", "ResetCounterCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FClusterCullingCS, "/Plugin/GaussianSplatting/Private/ClusterCulling.usf", "MainCS", SF_Compute);

// Splat compaction shaders (GPU-driven work reduction)
IMPLEMENT_GLOBAL_SHADER(FCompactSplatsCS, "/Plugin/GaussianSplatting/Private/CompactSplats.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FPrepareIndirectArgsCS, "/Plugin/GaussianSplatting/Private/PrepareIndirectArgs.usf", "MainCS", SF_Compute);

// NOTE: GPU-driven LOD shaders removed in unified approach
// LOD splats are now processed by CalcViewData.usf using the same buffers as original splats

// Global accumulator + compaction prefix-sum shaders
IMPLEMENT_GLOBAL_SHADER(FGatherVisibleCountsCS,     "/Plugin/GaussianSplatting/Private/GlobalAccumulatorPrefixSum.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FPrefixSumVisibleCountsCS,  "/Plugin/GaussianSplatting/Private/GlobalAccumulatorPrefixSum.usf", "MainCS", SF_Compute);
