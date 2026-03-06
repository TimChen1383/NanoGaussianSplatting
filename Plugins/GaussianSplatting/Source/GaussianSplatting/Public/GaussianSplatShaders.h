// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "RenderGraphResources.h"
#include "RenderGraphBuilder.h"
#include "GaussianDataTypes.h"
#include "GaussianClusterTypes.h"

/**
 * Compute shader for calculating view-dependent data for each Gaussian splat
 * This runs per-frame to transform splats to screen space and evaluate spherical harmonics
 */
class FGaussianSplatCalcViewDataCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatCalcViewDataCS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatCalcViewDataCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(ByteAddressBuffer, PackedSplatBuffer)  // 16 bytes/splat packed data
		SHADER_PARAMETER_SRV(ByteAddressBuffer, SHBuffer)           // SH data (currently unused)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<FGaussianSplatViewData>, ViewDataBuffer)
		// Cluster visibility integration (UNIFIED APPROACH)
		// SplatClusterIndexBuffer maps ALL splats to their cluster (original->leaf, LOD->parent)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SplatClusterIndexBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, ClusterVisibilityBitmap)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, LODClusterSelectedBitmap)  // For LOD splat visibility
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SelectedClusterBuffer)
		SHADER_PARAMETER(uint32, UseClusterCulling)
		SHADER_PARAMETER(uint32, UseLODRendering)  // 1 = enable LOD rendering (unified approach)
		SHADER_PARAMETER(uint32, OriginalSplatCount)  // Count of original splats (LOD splats start after)
		// Compaction mode: process only visible splats from CompactedSplatIndices
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, CompactedSplatIndices)
		SHADER_PARAMETER(uint32, UseCompaction)  // 1 = use compacted indices
		SHADER_PARAMETER(uint32, VisibleSplatCount)  // Number of visible splats (when using compaction)
		// Transform matrices
		SHADER_PARAMETER(FMatrix44f, LocalToWorld)
		SHADER_PARAMETER(FMatrix44f, WorldToPLY)  // Transforms world direction to PLY/SH space for SH evaluation
		SHADER_PARAMETER(FMatrix44f, WorldToClip)
		SHADER_PARAMETER(FMatrix44f, WorldToView)
		SHADER_PARAMETER(FVector3f, PreViewTranslation)  // For velocity calculation (TranslatedWorld = World + PreViewTranslation)
		SHADER_PARAMETER(FVector3f, CameraPosition)
		SHADER_PARAMETER(FVector2f, ScreenSize)
		SHADER_PARAMETER(FVector2f, FocalLength)
		SHADER_PARAMETER(uint32, SplatCount)
		SHADER_PARAMETER(uint32, SHOrder)
		SHADER_PARAMETER(uint32, NumSHCoeffs)      // Total SH coefficients including DC (4, 9, or 16)
		SHADER_PARAMETER(uint32, UseSHRendering)   // 1 = use view-dependent SH evaluation
		SHADER_PARAMETER(float, OpacityScale)
		SHADER_PARAMETER(float, SplatScale)
		SHADER_PARAMETER(uint32, GlobalBaseOffset)  // Offset into global ViewDataBuffer (non-compaction global path)
		// Global compaction path: GPU prefix-sum offsets
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, GlobalBaseOffsetsBuffer)  // prefix sums per proxy
		SHADER_PARAMETER(uint32, ProxyIndex)               // which proxy index this dispatch belongs to
		SHADER_PARAMETER(uint32, UseGlobalCompactionPath)  // 1 = read base from GlobalBaseOffsetsBuffer[ProxyIndex]
		SHADER_PARAMETER(uint32, MaxRenderBudget)          // Budget cap: skip writes at indices >= this value (0 = no cap)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), 256);
	}
};

// NOTE: FGaussianSplatCalcLODViewDataGPUDrivenCS and FUpdateDrawArgsCS have been removed
// in the unified approach. LOD splats are now processed by CalcViewData using the same
// buffers as original splats. See CalcViewData.usf for unified LOD handling.

//----------------------------------------------------------------------
// Splat Compaction Shaders (for GPU-driven work reduction)
//----------------------------------------------------------------------

/**
 * Compute shader that builds a compact list of visible splat indices
 * This enables subsequent passes to only process visible splats
 */
class FCompactSplatsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FCompactSplatsCS);
	SHADER_USE_PARAMETER_STRUCT(FCompactSplatsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		// Input: Cluster visibility data
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SplatClusterIndexBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, ClusterVisibilityBitmap)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, LODClusterSelectedBitmap)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SelectedClusterBuffer)
		// Output: Compact list of visible splat indices
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, CompactedSplatIndices)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, VisibleSplatCount)
		// Parameters
		SHADER_PARAMETER(uint32, TotalSplatCount)
		SHADER_PARAMETER(uint32, OriginalSplatCount)
		SHADER_PARAMETER(uint32, UseLODRendering)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), 256);
	}
};

/**
 * Compute shader that prepares indirect dispatch and draw arguments
 * from the visible splat count
 */
class FPrepareIndirectArgsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FPrepareIndirectArgsCS);
	SHADER_USE_PARAMETER_STRUCT(FPrepareIndirectArgsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		// Input: Visible splat count
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, VisibleSplatCount)
		// Output: Indirect dispatch args (uint3: numGroupsX, numGroupsY, numGroupsZ)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, IndirectDispatchArgs)
		// Output: Indirect draw args (5 uints for DrawIndexedIndirect)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, IndirectDrawArgs)
		// Output: Sort indirect dispatch args (uint3: NumTiles, 1, 1)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SortIndirectArgs)
		// Output: Sort parameters (uint2: SortCount, NumTiles)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SortParamsBuffer)
		// Thread group size for compute shaders
		SHADER_PARAMETER(uint32, ComputeThreadGroupSize)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

/**
 * Compute shader for calculating sort distances (depth) for each splat
 */
class FGaussianSplatCalcDistancesCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatCalcDistancesCS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatCalcDistancesCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(StructuredBuffer<FGaussianSplatViewData>, ViewDataBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DistanceBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, KeyBuffer)
		SHADER_PARAMETER(uint32, SplatCount)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), 256);
	}
};

/**
 * Vertex shader for rendering Gaussian splats as quads
 */
class FGaussianSplatVS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatVS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatVS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(StructuredBuffer<FGaussianSplatViewData>, ViewDataBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SortKeysBuffer)
		SHADER_PARAMETER(uint32, SplatCount)
		SHADER_PARAMETER(uint32, DebugMode)  // 0=off, 1=cluster colors (Nanite-style debug)
		SHADER_PARAMETER(uint32, EnableNanite)  // 1=Nanite enabled, 0=disabled (hidden in debug mode)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

/**
 * Pixel shader for rendering Gaussian splats with gaussian falloff
 */
class FGaussianSplatPS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatPS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatPS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		// Velocity calculation (Nanite-style): transform translated world position to previous clip space
		SHADER_PARAMETER(FMatrix44f, PrevTranslatedWorldToClip)
		// Screen size inverse for NDC conversion (avoids View uniform buffer dependency)
		SHADER_PARAMETER(FVector2f, ScreenSizeInverse)
		// TAA jitter: xy = current frame jitter, zw = previous frame jitter (in NDC space)
		SHADER_PARAMETER(FVector4f, TemporalAAJitter)
		// PreViewTranslation for converting TranslatedWorld back to World
		SHADER_PARAMETER(FVector3f, PreViewTranslation)
		// Previous frame's PreViewTranslation for correct velocity calculation
		SHADER_PARAMETER(FVector3f, PrevPreViewTranslation)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

/**
 * Radix sort - CountCS: per-tile histogram of 256 digit bins
 */
class FRadixSortCountCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FRadixSortCountCS);
	SHADER_USE_PARAMETER_STRUCT(FRadixSortCountCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, HistogramBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SrcKeys)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SortParams)
		SHADER_PARAMETER(uint32, RadixShift)
		SHADER_PARAMETER(uint32, Count)
		SHADER_PARAMETER(uint32, NumTiles)
		SHADER_PARAMETER(uint32, UseIndirectSort)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("COUNT_CS"), 1);
	}
};

/**
 * Radix sort - PrefixSumCS: exclusive prefix sum per digit across tiles
 */
class FRadixSortPrefixSumCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FRadixSortPrefixSumCS);
	SHADER_USE_PARAMETER_STRUCT(FRadixSortPrefixSumCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, HistogramBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DigitOffsetBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SortParams)
		SHADER_PARAMETER(uint32, NumTiles)
		SHADER_PARAMETER(uint32, UseIndirectSort)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("PREFIX_SUM_CS"), 1);
	}
};

/**
 * Radix sort - DigitPrefixSumCS: exclusive prefix sum across 256 digit totals
 */
class FRadixSortDigitPrefixSumCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FRadixSortDigitPrefixSumCS);
	SHADER_USE_PARAMETER_STRUCT(FRadixSortDigitPrefixSumCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DigitOffsetBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("DIGIT_PREFIX_SUM_CS"), 1);
	}
};

/**
 * Radix sort - ScatterCS: write keys+values to sorted positions
 */
class FRadixSortScatterCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FRadixSortScatterCS);
	SHADER_USE_PARAMETER_STRUCT(FRadixSortScatterCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SrcKeys)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SrcVals)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DstKeys)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DstVals)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, HistogramBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DigitOffsetBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SortParams)
		SHADER_PARAMETER(uint32, RadixShift)
		SHADER_PARAMETER(uint32, Count)
		SHADER_PARAMETER(uint32, NumTiles)
		SHADER_PARAMETER(uint32, UseIndirectSort)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("SCATTER_CS"), 1);
	}
};

//----------------------------------------------------------------------
// Cluster Culling Shaders (Nanite-style optimization)
//----------------------------------------------------------------------

/**
 * Compute shader to reset the visible cluster counter, indirect draw args, and visibility bitmap
 */
class FClusterCullingResetCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FClusterCullingResetCS);
	SHADER_USE_PARAMETER_STRUCT(FClusterCullingResetCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, VisibleClusterCountBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, IndirectDrawArgsBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, ClusterVisibilityBitmap)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SelectedClusterBuffer)
		// LOD cluster tracking buffers
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterCountBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterSelectedBitmap)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODSplatTotalBuffer)
		// GPU-driven LOD rendering
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODSplatOutputCountBuffer)
		SHADER_PARAMETER(uint32, ClusterVisibilityBitmapSize)
		SHADER_PARAMETER(uint32, LeafClusterCount)
		// Debug: must match MainCS parameters (shared .usf file)
		SHADER_PARAMETER(int32, DebugForceLODLevel)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

/**
 * Compute shader for frustum culling and LOD selection of clusters
 * Tests each cluster's bounding sphere against the view frustum
 * Calculates screen-space error for LOD selection
 * Outputs list of visible cluster indices with LOD flags
 */
class FClusterCullingCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FClusterCullingCS);
	SHADER_USE_PARAMETER_STRUCT(FClusterCullingCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(StructuredBuffer<FGaussianGPUCluster>, ClusterBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, VisibleClusterBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, VisibleClusterCountBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, IndirectDrawArgsBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, ClusterVisibilityBitmap)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SelectedClusterBuffer)
		// LOD cluster tracking buffers
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterCountBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterSelectedBitmap)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODSplatTotalBuffer)
		SHADER_PARAMETER(FMatrix44f, LocalToWorld)
		SHADER_PARAMETER(FMatrix44f, WorldToClip)
		SHADER_PARAMETER(uint32, ClusterCount)
		SHADER_PARAMETER(uint32, LeafClusterCount)
		SHADER_PARAMETER_ARRAY(FVector4f, FrustumPlanes, [6])
		// LOD selection parameters
		SHADER_PARAMETER(FVector3f, CameraPosition)
		SHADER_PARAMETER(float, ScreenHeight)
		SHADER_PARAMETER(float, ErrorThreshold)
		SHADER_PARAMETER(float, LODBias)
		SHADER_PARAMETER(uint32, UseLODRendering)
		// Debug: Force specific LOD level (-1 = auto, 0 = leaf, 1+ = specific level)
		SHADER_PARAMETER(int32, DebugForceLODLevel)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), 64);
	}
};

//----------------------------------------------------------------------
// Global Accumulator + Compaction prefix-sum shaders
//----------------------------------------------------------------------

/**
 * GatherCS: 1 thread per proxy.
 * Copies GPUResources->VisibleSplatCountBuffer[0] into GlobalVisibleCountArray[ProxyIndex].
 */
class FGatherVisibleCountsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGatherVisibleCountsCS);
	SHADER_USE_PARAMETER_STRUCT(FGatherVisibleCountsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, PerProxyVisibleCount)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalVisibleCountArray)
		SHADER_PARAMETER(uint32, ProxyIndex)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("GATHER_CS"), 1);
	}
};

/**
 * PrefixSumCS: 1 thread total.
 * Computes prefix sums over GlobalVisibleCountArray and writes all indirect
 * dispatch/draw args for the global Phase-3 passes (CalcDistances, RadixSort, DrawSplats).
 */
class FPrefixSumVisibleCountsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FPrefixSumVisibleCountsCS);
	SHADER_USE_PARAMETER_STRUCT(FPrefixSumVisibleCountsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>,   GlobalVisibleCountArray)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalBaseOffsets)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalCalcDistIndirectArgs)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalSortIndirectArgs)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalSortParams)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalDrawIndirectArgs)
		SHADER_PARAMETER(uint32, ProxyCount)
		SHADER_PARAMETER(uint32, MaxRenderBudget)  // Budget cap: clamp total visible to this value (0 = no cap)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("VISIBLE_PREFIX_SUM_CS"), 1);
	}
};
