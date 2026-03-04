// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "RenderGraphResources.h"
#include "RenderGraphBuilder.h"
#include "SceneView.h"

class FGaussianSplatSceneProxy;
class FGaussianSplatGPUResources;
struct FGaussianGlobalAccumulator;

/**
 * Handles the rendering of Gaussian Splats
 * Orchestrates compute passes for view calculation, sorting, and final rendering
 */
class GAUSSIANSPLATTING_API FGaussianSplatRenderer
{
public:
	FGaussianSplatRenderer();
	~FGaussianSplatRenderer();

	/**
	 * Dispatch the view data calculation compute shader
	 * @param bUseLODRendering If true, skip splats covered by parent LOD clusters
	 */
	static void DispatchCalcViewData(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		int32 SplatCount,
		int32 SHOrder,
		float OpacityScale,
		float SplatScale,
		bool bUseLODRendering = false
	);

	/**
	 * Dispatch the distance calculation compute shader
	 */
	static void DispatchCalcDistances(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources,
		int32 SplatCount
	);

	/**
	 * Dispatch radix sort for back-to-front ordering
	 */
	static void DispatchRadixSort(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources,
		int32 SplatCount
	);

	/**
	 * Dispatch radix sort with indirect dispatch (GPU-driven sort count)
	 * Uses SortIndirectArgsBuffer for CountCS/ScatterCS dispatch dimensions
	 * and SortParamsBuffer for Count/NumTiles read by shaders.
	 * Only sorts the visible splats after compaction.
	 */
	static void DispatchRadixSortIndirect(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources
	);

	/**
	 * Draw the Gaussian splats
	 */
	static void DrawSplats(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		int32 SplatCount
	);

	/**
	 * Dispatch cluster culling compute shader (Nanite-style optimization)
	 * Tests cluster bounding spheres against view frustum
	 * @param ErrorThreshold Screen-space error threshold in pixels for LOD selection
	 * @param bUseLODRendering If true, track unique LOD clusters for later rendering
	 * @return Number of visible clusters (for statistics)
	 */
	static int32 DispatchClusterCulling(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		float ErrorThreshold,
		bool bUseLODRendering = false
	);

	// NOTE: DispatchCalcLODViewDataGPUDriven and DispatchUpdateDrawArgs have been removed
	// in the unified approach. LOD splats are now processed by DispatchCalcViewData.

	//----------------------------------------------------------------------
	// Splat Compaction (GPU-driven work reduction)
	//----------------------------------------------------------------------

	/**
	 * Dispatch the splat compaction compute shader
	 * Builds a compact list of visible splat indices using atomics
	 */
	static void DispatchCompactSplats(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources,
		int32 TotalSplatCount,
		int32 OriginalSplatCount,
		bool bUseLODRendering
	);

	/**
	 * Dispatch the prepare indirect args compute shader
	 * Prepares indirect dispatch and draw arguments from visible splat count
	 */
	static void DispatchPrepareIndirectArgs(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources
	);

	/**
	 * Dispatch CalcViewData with compaction (indirect dispatch)
	 * Only processes visible splats from compacted list
	 */
	static void DispatchCalcViewDataCompacted(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		int32 SplatCount,
		int32 OriginalSplatCount,
		int32 SHOrder,
		float OpacityScale,
		float SplatScale
	);

	/**
	 * Dispatch CalcDistances with indirect dispatch
	 * Only processes visible splats
	 */
	static void DispatchCalcDistancesIndirect(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources
	);

	//----------------------------------------------------------------------
	// Global Accumulator dispatch (one-draw-call path)
	//----------------------------------------------------------------------

	/**
	 * Dispatch CalcViewData writing into GlobalAccumulator->GlobalViewDataBuffer
	 * at GlobalBaseOffset, instead of the per-proxy ViewDataBuffer.
	 */
	static void DispatchCalcViewDataGlobal(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		int32 SplatCount,
		int32 SHOrder,
		float OpacityScale,
		float SplatScale,
		bool bUseLODRendering,
		uint32 GlobalBaseOffset,
		FGaussianGlobalAccumulator* GlobalAccumulator
	);

	/**
	 * Dispatch CalcDistances over the full global ViewDataBuffer.
	 * Must be called after all Phase-1 CalcViewData dispatches.
	 */
	static void DispatchCalcDistancesGlobal(
		FRHICommandListImmediate& RHICmdList,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		int32 TotalSplatCount
	);

	/**
	 * Dispatch radix sort over the full global distance/key buffers.
	 */
	static void DispatchRadixSortGlobal(
		FRHICommandListImmediate& RHICmdList,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		int32 TotalSplatCount
	);

	/**
	 * Draw all splats using global sorted keys and ViewData.
	 * Borrows the IndexBuffer from the first valid proxy.
	 */
	static void DrawSplatsGlobal(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		FBufferRHIRef IndexBuffer,
		int32 TotalSplatCount,
		int32 DebugMode
	);

	//----------------------------------------------------------------------
	// Global Accumulator + Nanite Compaction dispatch (one-draw-call path)
	// Phase sequence: GatherVisibleCount × N → PrefixSumVisibleCounts ×1 →
	//   CalcViewDataCompactedGlobal × N → CalcDistancesGlobalIndirect ×1 →
	//   RadixSortGlobalIndirect ×1 → DrawSplatsGlobalIndirect ×1
	//----------------------------------------------------------------------

	/**
	 * Copy GPUResources->VisibleSplatCountBuffer[0] into
	 * GlobalAccumulator->GlobalVisibleCountArrayBuffer[ProxyIndex].
	 * Dispatch: (1,1,1) per proxy, after DispatchCompactSplats.
	 */
	static void DispatchGatherVisibleCount(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		int32 ProxyIndex
	);

	/**
	 * Compute exclusive prefix sums over GlobalVisibleCountArray and write
	 * all indirect dispatch/draw args for Phase-3 passes.
	 * Dispatch: (1,1,1) once, after all GatherVisibleCount dispatches.
	 */
	static void DispatchPrefixSumVisibleCounts(
		FRHICommandListImmediate& RHICmdList,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		int32 ProxyCount,
		uint32 MaxRenderBudget
	);

	/**
	 * CalcViewData for proxy ProxyIndex, writing into GlobalViewDataBuffer
	 * at offset GlobalBaseOffsetsBuffer[ProxyIndex].
	 * Uses IndirectDispatchArgsBuffer from GPUResources (set by PrepareIndirectArgs).
	 */
	static void DispatchCalcViewDataCompactedGlobal(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		int32 SplatCount,
		int32 OriginalSplatCount,
		int32 SHOrder,
		float OpacityScale,
		float SplatScale,
		int32 ProxyIndex,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		uint32 MaxRenderBudget
	);

	/**
	 * CalcDistances over the global ViewDataBuffer using indirect dispatch
	 * (count from GlobalCalcDistIndirectArgsBuffer written by PrefixSumCS).
	 */
	static void DispatchCalcDistancesGlobalIndirect(
		FRHICommandListImmediate& RHICmdList,
		FGaussianGlobalAccumulator* GlobalAccumulator
	);

	/**
	 * Radix sort over the global distance/key buffers using indirect dispatch
	 * (count/numTiles from GlobalSortParamsBuffer written by PrefixSumCS).
	 */
	static void DispatchRadixSortGlobalIndirect(
		FRHICommandListImmediate& RHICmdList,
		FGaussianGlobalAccumulator* GlobalAccumulator
	);

	/**
	 * Draw all visible splats using GlobalDrawIndirectArgsBuffer
	 * (instance count written by PrefixSumCS, not a CPU constant).
	 */
	static void DrawSplatsGlobalIndirect(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		FBufferRHIRef IndexBuffer,
		int32 DebugMode
	);

private:
	/** Calculate next power of 2 */
	static uint32 NextPowerOfTwo(uint32 Value);

	/**
	 * Extract frustum planes from view-projection matrix
	 * Planes are in world space, normalized with normal pointing inward
	 * Order: Left, Right, Bottom, Top, Near, Far
	 */
	static void ExtractFrustumPlanes(const FMatrix& ViewProjection, FVector4f OutPlanes[6]);
};
