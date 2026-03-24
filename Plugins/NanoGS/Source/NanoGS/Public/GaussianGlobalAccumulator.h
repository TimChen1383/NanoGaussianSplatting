// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "RHI.h"
#include "RHIResources.h"

/**
 * Global GPU buffer accumulator for one-draw-call Gaussian splatting.
 * Owned by FGaussianSplattingModule (render-thread lifetime).
 *
 * All proxies write their CalcViewData output into GlobalViewDataBuffer at their
 * respective GlobalBaseOffset. Then a single CalcDistances + RadixSort + DrawSplats
 * covers all splats from all proxies in one pass.
 *
 * Execution flow each frame:
 *   Phase 1 (per-proxy): ClusterCulling → CalcViewData → GlobalViewDataBuffer[BaseOffset + i]
 *   Phase 2 (once):      CalcDistances(GlobalViewDataBuffer) → RadixSort → DrawSplats
 */
struct NANOGS_API FGaussianGlobalAccumulator
{
	//----------------------------------------------------------------------
	// Global GPU buffers
	//----------------------------------------------------------------------

	/** View data for all proxies concatenated (written by CalcViewData, read by CalcDistances + VS) */
	FBufferRHIRef GlobalViewDataBuffer;
	FUnorderedAccessViewRHIRef GlobalViewDataBufferUAV;
	FShaderResourceViewRHIRef GlobalViewDataBufferSRV;

	/** Sort distance (depth) buffers — ping-pong for radix sort */
	FBufferRHIRef GlobalSortDistanceBuffer;
	FUnorderedAccessViewRHIRef GlobalSortDistanceBufferUAV;
	FBufferRHIRef GlobalSortDistanceBufferAlt;
	FUnorderedAccessViewRHIRef GlobalSortDistanceBufferAltUAV;

	/** Sort key (splat index) buffers — ping-pong for radix sort */
	FBufferRHIRef GlobalSortKeysBuffer;
	FUnorderedAccessViewRHIRef GlobalSortKeysBufferUAV;
	FShaderResourceViewRHIRef GlobalSortKeysBufferSRV;
	FBufferRHIRef GlobalSortKeysBufferAlt;
	FUnorderedAccessViewRHIRef GlobalSortKeysBufferAltUAV;

	/** Radix sort histogram: 256 * NumTiles uints */
	FBufferRHIRef GlobalRadixHistogramBuffer;
	FUnorderedAccessViewRHIRef GlobalRadixHistogramBufferUAV;

	/** Radix sort digit offset: 256 uints */
	FBufferRHIRef GlobalRadixDigitOffsetBuffer;
	FUnorderedAccessViewRHIRef GlobalRadixDigitOffsetBufferUAV;

	/** SortParams buffer (2 uints). SRV for non-compaction sort; UAV written by PrefixSumCS in compaction sort. */
	FBufferRHIRef GlobalSortParamsBuffer;
	FShaderResourceViewRHIRef GlobalSortParamsBufferSRV;
	FUnorderedAccessViewRHIRef GlobalSortParamsBufferUAV;  // Written by FPrefixSumVisibleCountsCS

	//----------------------------------------------------------------------
	// Global compaction path: fixed-size buffers (allocated once, MAX_PROXY_COUNT)
	//----------------------------------------------------------------------

	/** Maximum number of proxies supported by the compaction prefix-sum path.
	 *  Buffer cost: ~8 bytes per proxy (VisibleCount + BaseOffset), so 8192 = ~64KB. */
	static const uint32 MAX_PROXY_COUNT = 8192;

	/** [MAX_PROXY_COUNT] — one visible count per proxy, gathered from per-proxy VisibleSplatCountBuffer */
	FBufferRHIRef GlobalVisibleCountArrayBuffer;
	FUnorderedAccessViewRHIRef GlobalVisibleCountArrayBufferUAV;
	FShaderResourceViewRHIRef GlobalVisibleCountArrayBufferSRV;

	/** [MAX_PROXY_COUNT + 1] — exclusive prefix sums; last entry = TotalVisible */
	FBufferRHIRef GlobalBaseOffsetsBuffer;
	FUnorderedAccessViewRHIRef GlobalBaseOffsetsBufferUAV;
	FShaderResourceViewRHIRef GlobalBaseOffsetsBufferSRV;

	/** [3] — indirect dispatch args for CalcDistances: { ceil(TotalVisible/256), 1, 1 } */
	FBufferRHIRef GlobalCalcDistIndirectArgsBuffer;
	FUnorderedAccessViewRHIRef GlobalCalcDistIndirectArgsBufferUAV;

	/** [3] — indirect dispatch args for CountCS/ScatterCS: { ceil(TotalVisible/1024), 1, 1 } */
	FBufferRHIRef GlobalSortIndirectArgsGlobalBuffer;
	FUnorderedAccessViewRHIRef GlobalSortIndirectArgsGlobalBufferUAV;

	/** [5] — indirect draw args: { 6, TotalVisible, 0, 0, 0 } */
	FBufferRHIRef GlobalDrawIndirectArgsBuffer;
	FUnorderedAccessViewRHIRef GlobalDrawIndirectArgsBufferUAV;

	bool bCompactionBuffersAllocated = false;

	//----------------------------------------------------------------------
	// Capacity tracking
	//----------------------------------------------------------------------

	uint32 AllocatedCount = 0;    ///< Current buffer element capacity
	uint32 AllocatedNumTiles = 0; ///< NumTiles used to size the histogram buffer

	//----------------------------------------------------------------------
	// Camera-static skip cache
	//----------------------------------------------------------------------

	bool bHasCachedSortData = false;
	uint32 CachedTotalSplatCount = 0;
	FMatrix CachedViewProjectionMatrix = FMatrix::Identity;

	//----------------------------------------------------------------------
	// Previous frame data for velocity calculation (per-view)
	// (UE5's PrevViewInfo is not populated for PostOpaqueRender callbacks,
	// so we track previous frame matrices ourselves, keyed by view identity
	// to handle multiple simultaneous viewports correctly)
	// NOTE: Matrices are UN-JITTERED to ensure stable velocity when camera is static
	//----------------------------------------------------------------------

	struct FPrevFrameViewData
	{
		FMatrix TranslatedViewProjectionMatrix = FMatrix::Identity;  // Un-jittered
		FVector PreViewTranslation = FVector::ZeroVector;
	};

	/** Per-view previous frame data, keyed by FSceneViewState pointer */
	TMap<const void*, FPrevFrameViewData> PrevFrameDataPerView;

	//----------------------------------------------------------------------
	// Interface
	//----------------------------------------------------------------------

	/**
	 * Resize all global buffers if NewTotalCount exceeds current capacity.
	 * Adds 20% headroom to avoid per-frame reallocation.
	 * Must be called from the render thread.
	 */
	void ResizeIfNeeded(FRHICommandListBase& RHICmdList, uint32 NewTotalCount);

	/**
	 * Allocate the fixed-size compaction prefix-sum buffers if not already done.
	 * These are sized by MAX_PROXY_COUNT and never need to grow.
	 * Must be called from the render thread.
	 */
	void EnsureCompactionBuffersAllocated(FRHICommandListBase& RHICmdList);

	/** Release all GPU buffers. Call from the render thread before destruction. */
	void Release();

	bool IsAllocated() const { return AllocatedCount > 0; }
};
