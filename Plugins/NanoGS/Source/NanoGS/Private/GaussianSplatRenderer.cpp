// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatRenderer.h"
#include "GaussianSplatShaders.h"
#include "GaussianSplatSceneProxy.h"
#include "GaussianGlobalAccumulator.h"
#include "GaussianClusterTypes.h"
#include "RHICommandList.h"
#include "RHIGPUReadback.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "GlobalShader.h"
#include "ShaderParameterUtils.h"
#include "PipelineStateCache.h"
#include "RHIStaticStates.h"
#include "SceneView.h"
#include "SceneRendering.h"  // For FViewInfo::ViewRect (screen percentage support)
#include "RenderCore.h"
#include "CommonRenderResources.h"

// Console variables (declared in GaussianSplatting.cpp)
extern TAutoConsoleVariable<int32> CVarShowClusterBounds;
extern TAutoConsoleVariable<int32> CVarDebugForceLODLevel;

FGaussianSplatRenderer::FGaussianSplatRenderer()
{
}

FGaussianSplatRenderer::~FGaussianSplatRenderer()
{
}

uint32 FGaussianSplatRenderer::NextPowerOfTwo(uint32 Value)
{
	Value--;
	Value |= Value >> 1;
	Value |= Value >> 2;
	Value |= Value >> 4;
	Value |= Value >> 8;
	Value |= Value >> 16;
	Value++;
	return Value;
}

// Compute WorldToPLY matrix for SH evaluation
// SH coefficients are stored in PLY space, so we need to transform view direction from world to PLY space
// This combines: (1) WorldToLocal from actor transform, and (2) LocalToPLY coordinate system conversion
// LocalToPLY rotation: UE(X,Y,Z) -> PLY(Y,-Z,X) where PLY is X-right, Y-down, Z-forward
static FMatrix ComputeWorldToPLY(const FMatrix& LocalToWorld)
{
	static const FMatrix LocalToPLY(
		FPlane(0, 0, 1, 0),   // UE X -> PLY Z
		FPlane(1, 0, 0, 0),   // UE Y -> PLY X
		FPlane(0, -1, 0, 0),  // UE Z -> PLY -Y
		FPlane(0, 0, 0, 1)
	);
	return LocalToWorld.Inverse() * LocalToPLY;
}

void FGaussianSplatRenderer::DispatchCalcViewData(
	FRHICommandListImmediate& RHICmdList,
	const FSceneView& View,
	FGaussianSplatGPUResources* GPUResources,
	const FMatrix& LocalToWorld,
	int32 SplatCount,
	int32 SHOrder,
	float OpacityScale,
	float SplatScale,
	bool bUseLODRendering)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatCalcViewData);

	TShaderMapRef<FGaussianSplatCalcViewDataCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FGaussianSplatCalcViewDataCS shader not valid"));
		return;
	}

	// Transition buffers for compute
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->ViewDataBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	FGaussianSplatCalcViewDataCS::FParameters Parameters;
	Parameters.PackedSplatBuffer = GPUResources->PackedSplatBufferSRV;
	Parameters.SHBuffer = GPUResources->SHBufferSRV;
	Parameters.ViewDataBuffer = GPUResources->ViewDataBufferUAV;

	// Cluster visibility integration (UNIFIED APPROACH)
	Parameters.SplatClusterIndexBuffer = GPUResources->SplatClusterIndexBufferSRV;
	Parameters.ClusterVisibilityBitmap = GPUResources->ClusterVisibilityBitmapSRV;
	Parameters.LODClusterSelectedBitmap = GPUResources->LODClusterSelectedBitmapSRV;
	Parameters.SelectedClusterBuffer = GPUResources->SelectedClusterBufferSRV;
	Parameters.CompactedSplatIndices = GPUResources->CompactedSplatIndicesBufferSRV;
	Parameters.UseCompaction = 0;
	Parameters.VisibleSplatCount = 0;

	if (GPUResources->bHasClusterData)
	{
		Parameters.UseClusterCulling = 1;
		Parameters.UseLODRendering = bUseLODRendering ? 1 : 0;
		Parameters.OriginalSplatCount = SplatCount - GPUResources->LODSplatCount;
	}
	else
	{
		Parameters.UseClusterCulling = 0;
		Parameters.UseLODRendering = 0;
		Parameters.OriginalSplatCount = SplatCount;
	}

	// Cast to FViewInfo to access PrevViewInfo for velocity calculation
	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);

	// Matrices
	Parameters.LocalToWorld = FMatrix44f(LocalToWorld);
	Parameters.WorldToPLY = FMatrix44f(ComputeWorldToPLY(LocalToWorld));
	Parameters.WorldToClip = FMatrix44f(View.ViewMatrices.GetViewProjectionMatrix());
	Parameters.PreViewTranslation = FVector3f(View.ViewMatrices.GetPreViewTranslation());
	Parameters.WorldToView = FMatrix44f(View.ViewMatrices.GetViewMatrix());
	Parameters.CameraPosition = FVector3f(View.ViewMatrices.GetViewOrigin());

	FIntRect ViewRect = ViewInfo.ViewRect;
	Parameters.ScreenSize = FVector2f(ViewRect.Width(), ViewRect.Height());

	const FMatrix& ProjMatrix = View.ViewMatrices.GetProjectionMatrix();
	Parameters.FocalLength = FVector2f(
		ProjMatrix.M[0][0] * ViewRect.Width() * 0.5f,
		ProjMatrix.M[1][1] * ViewRect.Height() * 0.5f
	);

	Parameters.SplatCount = SplatCount;
	// Use effective SH order: min(requested order, available data)
	int32 EffectiveSHOrder = FMath::Min(SHOrder, GPUResources->GetSHBands());
	Parameters.SHOrder = EffectiveSHOrder;
	// SH buffer includes DC + higher-order coefficients: band1=4, band2=9, band3=16
	Parameters.NumSHCoeffs = (EffectiveSHOrder == 0) ? 0 : (EffectiveSHOrder == 1) ? 4 : (EffectiveSHOrder == 2) ? 9 : 16;
	{
		int32 StoredSHBands = GPUResources->GetSHBands();
		Parameters.SHBufferCoeffs = (StoredSHBands == 0) ? 0 : (StoredSHBands == 1) ? 4 : (StoredSHBands == 2) ? 9 : 16;
	}
	Parameters.UseSHRendering = (EffectiveSHOrder > 0) ? 1 : 0;
	Parameters.OpacityScale = OpacityScale;
	Parameters.SplatScale = SplatScale;

	// Not using global compaction path
	Parameters.GlobalBaseOffsetsBuffer = GPUResources->CompactedSplatIndicesBufferSRV;  // dummy
	Parameters.ProxyIndex = 0;
	Parameters.UseGlobalCompactionPath = 0;
	Parameters.MaxRenderBudget = 0;

	// Dispatch compute shader
	const uint32 ThreadGroupSize = 256;
	const uint32 NumGroups = FMath::DivideAndRoundUp((uint32)SplatCount, ThreadGroupSize);

	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchComputeShader(NumGroups, 1, 1);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());

	// Transition buffer for next stage
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->ViewDataBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
}

void FGaussianSplatRenderer::DispatchCalcDistances(
	FRHICommandListImmediate& RHICmdList,
	FGaussianSplatGPUResources* GPUResources,
	int32 SplatCount)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatCalcDistances);

	TShaderMapRef<FGaussianSplatCalcDistancesCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FGaussianSplatCalcDistancesCS shader not valid"));
		return;
	}

	// Transition buffers
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortDistanceBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortKeysBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	FGaussianSplatCalcDistancesCS::FParameters Parameters;
	Parameters.ViewDataBuffer = GPUResources->ViewDataBufferSRV;
	Parameters.DistanceBuffer = GPUResources->SortDistanceBufferUAV;
	Parameters.KeyBuffer = GPUResources->SortKeysBufferUAV;
	Parameters.SplatCount = SplatCount;

	// Dispatch only for actual SplatCount — no power-of-2 padding needed for radix sort
	const uint32 ThreadGroupSize = 256;
	const uint32 NumGroups = FMath::DivideAndRoundUp((uint32)SplatCount, ThreadGroupSize);

	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchComputeShader(NumGroups, 1, 1);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());
}

void FGaussianSplatRenderer::DispatchRadixSort(
	FRHICommandListImmediate& RHICmdList,
	FGaussianSplatGPUResources* GPUResources,
	int32 SplatCount)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatRadixSort);

	TShaderMapRef<FRadixSortCountCS> CountShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortPrefixSumCS> PrefixSumShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortDigitPrefixSumCS> DigitPrefixSumShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortScatterCS> ScatterShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!CountShader.IsValid() || !PrefixSumShader.IsValid() ||
		!DigitPrefixSumShader.IsValid() || !ScatterShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("Radix sort shaders not valid"));
		return;
	}

	// Radix sort works with any count — no power-of-2 padding needed
	uint32 SortCount = (uint32)SplatCount;
	uint32 NumTiles = FMath::DivideAndRoundUp(SortCount, 1024u);

	// Distance buffers: [0] = primary, [1] = alt
	FUnorderedAccessViewRHIRef DistUAVs[2] = {
		GPUResources->SortDistanceBufferUAV,
		GPUResources->SortDistanceBufferAltUAV
	};
	FBufferRHIRef DistBuffers[2] = {
		GPUResources->SortDistanceBuffer,
		GPUResources->SortDistanceBufferAlt
	};

	// Key buffers: [0] = primary, [1] = alt
	FUnorderedAccessViewRHIRef KeyUAVs[2] = {
		GPUResources->SortKeysBufferUAV,
		GPUResources->SortKeysBufferAltUAV
	};
	FBufferRHIRef KeyBuffers[2] = {
		GPUResources->SortKeysBuffer,
		GPUResources->SortKeysBufferAlt
	};

	// Ensure alt buffers are in UAVCompute state
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortDistanceBufferAlt, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortKeysBufferAlt, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixHistogramBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixDigitOffsetBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	// SortParamsBuffer is bound as SRV (unused when UseIndirectSort=0, but needs valid state)
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortParamsBuffer, ERHIAccess::Unknown, ERHIAccess::SRVCompute));

	// 4 radix passes: bits 0-7, 8-15, 16-23, 24-31
	for (uint32 Pass = 0; Pass < 4; Pass++)
	{
		uint32 RadixShift = Pass * 8;
		uint32 SrcIdx = Pass & 1;
		uint32 DstIdx = 1 - SrcIdx;

		// --- CountCS: build per-tile histograms ---
		{
			FRadixSortCountCS::FParameters Params;
			Params.HistogramBuffer = GPUResources->RadixHistogramBufferUAV;
			Params.SrcKeys = DistUAVs[SrcIdx];
			Params.SortParams = GPUResources->SortParamsBufferSRV;
			Params.RadixShift = RadixShift;
			Params.Count = SortCount;
			Params.NumTiles = NumTiles;
			Params.UseIndirectSort = 0;

			SetComputePipelineState(RHICmdList, CountShader.GetComputeShader());
			SetShaderParameters(RHICmdList, CountShader, CountShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(NumTiles, 1, 1);
			UnsetShaderUAVs(RHICmdList, CountShader, CountShader.GetComputeShader());
		}

		// UAV barrier
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixHistogramBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// --- PrefixSumCS: exclusive prefix sum per digit across tiles ---
		{
			FRadixSortPrefixSumCS::FParameters Params;
			Params.HistogramBuffer = GPUResources->RadixHistogramBufferUAV;
			Params.DigitOffsetBuffer = GPUResources->RadixDigitOffsetBufferUAV;
			Params.SortParams = GPUResources->SortParamsBufferSRV;
			Params.NumTiles = NumTiles;
			Params.UseIndirectSort = 0;

			SetComputePipelineState(RHICmdList, PrefixSumShader.GetComputeShader());
			SetShaderParameters(RHICmdList, PrefixSumShader, PrefixSumShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(256, 1, 1);
			UnsetShaderUAVs(RHICmdList, PrefixSumShader, PrefixSumShader.GetComputeShader());
		}

		// UAV barrier
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixDigitOffsetBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// --- DigitPrefixSumCS: exclusive prefix sum across digit totals ---
		{
			FRadixSortDigitPrefixSumCS::FParameters Params;
			Params.DigitOffsetBuffer = GPUResources->RadixDigitOffsetBufferUAV;

			SetComputePipelineState(RHICmdList, DigitPrefixSumShader.GetComputeShader());
			SetShaderParameters(RHICmdList, DigitPrefixSumShader, DigitPrefixSumShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(1, 1, 1);
			UnsetShaderUAVs(RHICmdList, DigitPrefixSumShader, DigitPrefixSumShader.GetComputeShader());
		}

		// UAV barrier
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixDigitOffsetBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixHistogramBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// --- ScatterCS: scatter keys+values to sorted positions ---
		{
			FRadixSortScatterCS::FParameters Params;
			Params.SrcKeys = DistUAVs[SrcIdx];
			Params.SrcVals = KeyUAVs[SrcIdx];
			Params.DstKeys = DistUAVs[DstIdx];
			Params.DstVals = KeyUAVs[DstIdx];
			Params.HistogramBuffer = GPUResources->RadixHistogramBufferUAV;
			Params.DigitOffsetBuffer = GPUResources->RadixDigitOffsetBufferUAV;
			Params.SortParams = GPUResources->SortParamsBufferSRV;
			Params.RadixShift = RadixShift;
			Params.Count = SortCount;
			Params.NumTiles = NumTiles;
			Params.UseIndirectSort = 0;

			SetComputePipelineState(RHICmdList, ScatterShader.GetComputeShader());
			SetShaderParameters(RHICmdList, ScatterShader, ScatterShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(NumTiles, 1, 1);
			UnsetShaderUAVs(RHICmdList, ScatterShader, ScatterShader.GetComputeShader());
		}

		// UAV barriers between passes
		RHICmdList.Transition(FRHITransitionInfo(DistBuffers[DstIdx], ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
		RHICmdList.Transition(FRHITransitionInfo(KeyBuffers[DstIdx], ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	}

	// After 4 passes (even count), result is back in primary buffers [0]
	// Transition SortKeysBuffer for rendering
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortKeysBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVGraphics));
}

void FGaussianSplatRenderer::DispatchRadixSortIndirect(
	FRHICommandListImmediate& RHICmdList,
	FGaussianSplatGPUResources* GPUResources)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatRadixSort);

	TShaderMapRef<FRadixSortCountCS> CountShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortPrefixSumCS> PrefixSumShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortDigitPrefixSumCS> DigitPrefixSumShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortScatterCS> ScatterShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!CountShader.IsValid() || !PrefixSumShader.IsValid() ||
		!DigitPrefixSumShader.IsValid() || !ScatterShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("Radix sort shaders not valid"));
		return;
	}

	// Distance buffers: [0] = primary, [1] = alt
	FUnorderedAccessViewRHIRef DistUAVs[2] = {
		GPUResources->SortDistanceBufferUAV,
		GPUResources->SortDistanceBufferAltUAV
	};
	FBufferRHIRef DistBuffers[2] = {
		GPUResources->SortDistanceBuffer,
		GPUResources->SortDistanceBufferAlt
	};

	// Key buffers: [0] = primary, [1] = alt
	FUnorderedAccessViewRHIRef KeyUAVs[2] = {
		GPUResources->SortKeysBufferUAV,
		GPUResources->SortKeysBufferAltUAV
	};
	FBufferRHIRef KeyBuffers[2] = {
		GPUResources->SortKeysBuffer,
		GPUResources->SortKeysBufferAlt
	};

	// Ensure alt buffers are in UAVCompute state
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortDistanceBufferAlt, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortKeysBufferAlt, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixHistogramBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixDigitOffsetBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	// SortIndirectArgsBuffer should already be in IndirectArgs state from DispatchPrepareIndirectArgs
	// SortParamsBuffer should already be in SRVCompute state from DispatchPrepareIndirectArgs

	// 4 radix passes: bits 0-7, 8-15, 16-23, 24-31
	for (uint32 Pass = 0; Pass < 4; Pass++)
	{
		uint32 RadixShift = Pass * 8;
		uint32 SrcIdx = Pass & 1;
		uint32 DstIdx = 1 - SrcIdx;

		// --- CountCS: build per-tile histograms (INDIRECT DISPATCH) ---
		{
			FRadixSortCountCS::FParameters Params;
			Params.HistogramBuffer = GPUResources->RadixHistogramBufferUAV;
			Params.SrcKeys = DistUAVs[SrcIdx];
			Params.SortParams = GPUResources->SortParamsBufferSRV;
			Params.RadixShift = RadixShift;
			Params.Count = 0;       // Unused when UseIndirectSort=1
			Params.NumTiles = 0;    // Unused when UseIndirectSort=1
			Params.UseIndirectSort = 1;

			SetComputePipelineState(RHICmdList, CountShader.GetComputeShader());
			SetShaderParameters(RHICmdList, CountShader, CountShader.GetComputeShader(), Params);
			RHICmdList.DispatchIndirectComputeShader(GPUResources->SortIndirectArgsBuffer, 0);
			UnsetShaderUAVs(RHICmdList, CountShader, CountShader.GetComputeShader());
		}

		// UAV barrier
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixHistogramBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// --- PrefixSumCS: exclusive prefix sum per digit across tiles ---
		// Fixed dispatch (256, 1, 1) - one group per digit
		// NumTiles is read from SortParams buffer
		{
			FRadixSortPrefixSumCS::FParameters Params;
			Params.HistogramBuffer = GPUResources->RadixHistogramBufferUAV;
			Params.DigitOffsetBuffer = GPUResources->RadixDigitOffsetBufferUAV;
			Params.SortParams = GPUResources->SortParamsBufferSRV;
			Params.NumTiles = 0;    // Unused when UseIndirectSort=1
			Params.UseIndirectSort = 1;

			SetComputePipelineState(RHICmdList, PrefixSumShader.GetComputeShader());
			SetShaderParameters(RHICmdList, PrefixSumShader, PrefixSumShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(256, 1, 1);
			UnsetShaderUAVs(RHICmdList, PrefixSumShader, PrefixSumShader.GetComputeShader());
		}

		// UAV barrier
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixDigitOffsetBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// --- DigitPrefixSumCS: exclusive prefix sum across digit totals ---
		// No changes needed - doesn't use Count or NumTiles
		{
			FRadixSortDigitPrefixSumCS::FParameters Params;
			Params.DigitOffsetBuffer = GPUResources->RadixDigitOffsetBufferUAV;

			SetComputePipelineState(RHICmdList, DigitPrefixSumShader.GetComputeShader());
			SetShaderParameters(RHICmdList, DigitPrefixSumShader, DigitPrefixSumShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(1, 1, 1);
			UnsetShaderUAVs(RHICmdList, DigitPrefixSumShader, DigitPrefixSumShader.GetComputeShader());
		}

		// UAV barrier
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixDigitOffsetBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->RadixHistogramBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// --- ScatterCS: scatter keys+values to sorted positions (INDIRECT DISPATCH) ---
		{
			FRadixSortScatterCS::FParameters Params;
			Params.SrcKeys = DistUAVs[SrcIdx];
			Params.SrcVals = KeyUAVs[SrcIdx];
			Params.DstKeys = DistUAVs[DstIdx];
			Params.DstVals = KeyUAVs[DstIdx];
			Params.HistogramBuffer = GPUResources->RadixHistogramBufferUAV;
			Params.DigitOffsetBuffer = GPUResources->RadixDigitOffsetBufferUAV;
			Params.SortParams = GPUResources->SortParamsBufferSRV;
			Params.RadixShift = RadixShift;
			Params.Count = 0;       // Unused when UseIndirectSort=1
			Params.NumTiles = 0;    // Unused when UseIndirectSort=1
			Params.UseIndirectSort = 1;

			SetComputePipelineState(RHICmdList, ScatterShader.GetComputeShader());
			SetShaderParameters(RHICmdList, ScatterShader, ScatterShader.GetComputeShader(), Params);
			RHICmdList.DispatchIndirectComputeShader(GPUResources->SortIndirectArgsBuffer, 0);
			UnsetShaderUAVs(RHICmdList, ScatterShader, ScatterShader.GetComputeShader());
		}

		// UAV barriers between passes
		RHICmdList.Transition(FRHITransitionInfo(DistBuffers[DstIdx], ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
		RHICmdList.Transition(FRHITransitionInfo(KeyBuffers[DstIdx], ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	}

	// After 4 passes (even count), result is back in primary buffers [0]
	// Transition SortKeysBuffer for rendering
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortKeysBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVGraphics));
}

void FGaussianSplatRenderer::DrawSplats(
	FRHICommandListImmediate& RHICmdList,
	const FSceneView& View,
	FGaussianSplatGPUResources* GPUResources,
	int32 SplatCount)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatDraw);

	if (!GPUResources || !GPUResources->IndexBuffer.IsValid())
	{
		return;
	}

	TShaderMapRef<FGaussianSplatVS> VertexShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FGaussianSplatPS> PixelShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!VertexShader.IsValid() || !PixelShader.IsValid())
	{
		return;
	}

	// Transition view data buffer for graphics reads.
	// Use Unknown source: handles both fresh compute (SRVCompute) and cached (SRVGraphics) paths.
	if (GPUResources->ViewDataBuffer.IsValid())
	{
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->ViewDataBuffer, ERHIAccess::Unknown, ERHIAccess::SRVGraphics));
	}

	// Set up graphics pipeline state
	FGraphicsPipelineStateInitializer GraphicsPSOInit;
	RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);

	GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
	// Depth test enabled (CF_DepthNearOrEqual) so splats are occluded by scene geometry
	// Depth write disabled (false) because splats are transparent and blend among themselves
	// Enable depth writes for TSR/TAA - splats write depth at their center position
	// Using DepthNearOrEqual allows splats at similar depths to all blend correctly
	GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<true, CF_DepthNearOrEqual>::GetRHI();

	// Blend mode for MRT:
	// RT0 (Color): Premultiplied alpha "over" for back-to-front compositing in sRGB space
	//   result = src + dst * (1 - srcAlpha)
	//   Using CW_RGBA: splats render to intermediate sRGB RT which needs alpha tracking.
	//   The composite pass later converts sRGB→linear and writes to SceneColor with CW_RGB.
	// RT1 (Velocity): Simple replacement - velocity values should not be blended
	GraphicsPSOInit.BlendState = TStaticBlendState<
		// RT0: ColorWriteMask, ColorBlendOp, ColorSrcBlend, ColorDestBlend, AlphaBlendOp, AlphaSrcBlend, AlphaDestBlend
		CW_RGBA, BO_Add, BF_One, BF_InverseSourceAlpha, BO_Add, BF_One, BF_InverseSourceAlpha,
		// RT1: ColorWriteMask, ColorBlendOp, ColorSrcBlend, ColorDestBlend, AlphaBlendOp, AlphaSrcBlend, AlphaDestBlend
		CW_RGBA, BO_Add, BF_One, BF_Zero, BO_Add, BF_One, BF_Zero
	>::GetRHI();

	GraphicsPSOInit.PrimitiveType = PT_TriangleList;
	GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GEmptyVertexDeclaration.VertexDeclarationRHI;
	GraphicsPSOInit.BoundShaderState.VertexShaderRHI = VertexShader.GetVertexShader();
	GraphicsPSOInit.BoundShaderState.PixelShaderRHI = PixelShader.GetPixelShader();

	SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit, 0);

	// Set viewport using FViewInfo::ViewRect which accounts for screen percentage
	// This matches the actual render target dimensions
	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);
	FIntRect ViewRect = ViewInfo.ViewRect;
	RHICmdList.SetViewport(
		ViewRect.Min.X,
		ViewRect.Min.Y,
		0.0f,
		ViewRect.Max.X,
		ViewRect.Max.Y,
		1.0f
	);

	// Set vertex shader parameters
	FGaussianSplatVS::FParameters VSParameters;
	VSParameters.ViewDataBuffer = GPUResources->ViewDataBufferSRV;
	VSParameters.SortKeysBuffer = GPUResources->SortKeysBufferSRV;
	VSParameters.SplatCount = SplatCount;
	// Debug mode from console variable (Nanite-style cluster coloring)
	VSParameters.DebugMode = static_cast<uint32>(FMath::Max(0, CVarShowClusterBounds.GetValueOnRenderThread()));
	// Pass Nanite enabled state for debug visualization (non-Nanite assets render black in debug mode)
	VSParameters.EnableNanite = GPUResources->bEnableNanite ? 1 : 0;

	SetShaderParameters(RHICmdList, VertexShader, VertexShader.GetVertexShader(), VSParameters);

	// Pixel shader parameters
	FGaussianSplatPS::FParameters PSParameters;
	// Velocity calculation (Nanite-style): use previous frame's translated view-projection matrix
	PSParameters.PrevTranslatedWorldToClip = FMatrix44f(ViewInfo.PrevViewInfo.ViewMatrices.GetTranslatedViewProjectionMatrix());
	// Pass screen size inverse for NDC conversion (avoids View uniform buffer binding)
	PSParameters.ScreenSizeInverse = FVector2f(1.0f / ViewRect.Width(), 1.0f / ViewRect.Height());
	// TAA jitter: xy = current frame, zw = previous frame (in NDC space)
	PSParameters.TemporalAAJitter = FVector4f(
		View.ViewMatrices.GetTemporalAAJitter().X,
		View.ViewMatrices.GetTemporalAAJitter().Y,
		ViewInfo.PrevViewInfo.ViewMatrices.GetTemporalAAJitter().X,
		ViewInfo.PrevViewInfo.ViewMatrices.GetTemporalAAJitter().Y
	);
	// PreViewTranslation for converting TranslatedWorld back to World
	PSParameters.PreViewTranslation = FVector3f(View.ViewMatrices.GetPreViewTranslation());
	// Previous frame's PreViewTranslation for correct velocity calculation
	PSParameters.PrevPreViewTranslation = FVector3f(ViewInfo.PrevViewInfo.ViewMatrices.GetPreViewTranslation());
	SetShaderParameters(RHICmdList, PixelShader, PixelShader.GetPixelShader(), PSParameters);

	// Draw instanced quads
	// 4 vertices per quad, N instances (either from SplatCount or indirect args)
	// Using index buffer: 6 indices per quad (2 triangles)
	RHICmdList.SetStreamSource(0, nullptr, 0);

	// Use indirect draw when Nanite is enabled and cluster data is available
	// This allows the GPU to determine how many instances to draw
	if (GPUResources->bEnableNanite && GPUResources->bSupportsIndirectDraw && GPUResources->bHasClusterData)
	{
		// GPU-driven rendering: instance count comes from cluster culling results
		RHICmdList.DrawIndexedPrimitiveIndirect(
			GPUResources->IndexBuffer,
			GPUResources->IndirectDrawArgsBuffer,
			0  // ArgumentOffset
		);
	}
	else
	{
		// Fallback: CPU-driven rendering with fixed splat count (non-Nanite path)
		RHICmdList.DrawIndexedPrimitive(
			GPUResources->IndexBuffer,
			0,  // BaseVertexIndex
			0,  // FirstInstance
			4,  // NumVertices
			0,  // StartIndex
			2,  // NumPrimitives (2 triangles per quad)
			SplatCount  // NumInstances
		);
	}
}

// NOTE: DispatchCalcLODViewDataGPUDriven and DispatchUpdateDrawArgs have been removed
// in the unified approach. LOD splats are now processed by CalcViewData shader using
// the same buffers as original splats.

//----------------------------------------------------------------------
// Splat Compaction Implementation
//----------------------------------------------------------------------

void FGaussianSplatRenderer::DispatchCompactSplats(
	FRHICommandListImmediate& RHICmdList,
	FGaussianSplatGPUResources* GPUResources,
	int32 TotalSplatCount,
	int32 OriginalSplatCount,
	bool bUseLODRendering)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatCompactSplats);

	TShaderMapRef<FCompactSplatsCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FCompactSplatsCS shader not valid"));
		return;
	}

	// Reset visible splat count to 0
	uint32 Zero = 0;
	void* Data = RHICmdList.LockBuffer(GPUResources->VisibleSplatCountBuffer, 0, sizeof(uint32), RLM_WriteOnly);
	FMemory::Memcpy(Data, &Zero, sizeof(uint32));
	RHICmdList.UnlockBuffer(GPUResources->VisibleSplatCountBuffer);

	// Transition buffers
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->CompactedSplatIndicesBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->VisibleSplatCountBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	FCompactSplatsCS::FParameters Parameters;
	Parameters.SplatClusterIndexBuffer = GPUResources->SplatClusterIndexBufferSRV;
	Parameters.ClusterVisibilityBitmap = GPUResources->ClusterVisibilityBitmapSRV;
	Parameters.LODClusterSelectedBitmap = GPUResources->LODClusterSelectedBitmapSRV;
	Parameters.SelectedClusterBuffer = GPUResources->SelectedClusterBufferSRV;
	Parameters.CompactedSplatIndices = GPUResources->CompactedSplatIndicesBufferUAV;
	Parameters.VisibleSplatCount = GPUResources->VisibleSplatCountBufferUAV;
	Parameters.TotalSplatCount = TotalSplatCount;
	Parameters.OriginalSplatCount = OriginalSplatCount;
	Parameters.UseLODRendering = bUseLODRendering ? 1 : 0;

	const uint32 ThreadGroupSize = 256;
	const uint32 NumGroups = FMath::DivideAndRoundUp((uint32)TotalSplatCount, ThreadGroupSize);

	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchComputeShader(NumGroups, 1, 1);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());

	// Transition for next stage
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->CompactedSplatIndicesBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->VisibleSplatCountBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
}

void FGaussianSplatRenderer::DispatchPrepareIndirectArgs(
	FRHICommandListImmediate& RHICmdList,
	FGaussianSplatGPUResources* GPUResources)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatPrepareIndirectArgs);

	TShaderMapRef<FPrepareIndirectArgsCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FPrepareIndirectArgsCS shader not valid"));
		return;
	}

	// Transition buffers
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->IndirectDispatchArgsBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->IndirectDrawArgsBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortIndirectArgsBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortParamsBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	FPrepareIndirectArgsCS::FParameters Parameters;
	Parameters.VisibleSplatCount = GPUResources->VisibleSplatCountBufferSRV;
	Parameters.IndirectDispatchArgs = GPUResources->IndirectDispatchArgsBufferUAV;
	Parameters.IndirectDrawArgs = GPUResources->IndirectDrawArgsBufferUAV;
	Parameters.SortIndirectArgs = GPUResources->SortIndirectArgsBufferUAV;
	Parameters.SortParamsBuffer = GPUResources->SortParamsBufferUAV;
	Parameters.ComputeThreadGroupSize = 256;

	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchComputeShader(1, 1, 1);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());

	// Transition for indirect dispatch/draw
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->IndirectDispatchArgsBuffer, ERHIAccess::UAVCompute, ERHIAccess::IndirectArgs));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->IndirectDrawArgsBuffer, ERHIAccess::UAVCompute, ERHIAccess::IndirectArgs));
	// Transition sort buffers for radix sort indirect dispatch
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortIndirectArgsBuffer, ERHIAccess::UAVCompute, ERHIAccess::IndirectArgs));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortParamsBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
}

void FGaussianSplatRenderer::DispatchCalcViewDataCompacted(
	FRHICommandListImmediate& RHICmdList,
	const FSceneView& View,
	FGaussianSplatGPUResources* GPUResources,
	const FMatrix& LocalToWorld,
	int32 SplatCount,
	int32 OriginalSplatCount,
	int32 SHOrder,
	float OpacityScale,
	float SplatScale)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatCalcViewDataCompacted);

	TShaderMapRef<FGaussianSplatCalcViewDataCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FGaussianSplatCalcViewDataCS shader not valid"));
		return;
	}

	// Transition buffers for compute
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->ViewDataBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	FGaussianSplatCalcViewDataCS::FParameters Parameters;
	Parameters.PackedSplatBuffer = GPUResources->PackedSplatBufferSRV;
	Parameters.SHBuffer = GPUResources->SHBufferSRV;
	Parameters.ViewDataBuffer = GPUResources->ViewDataBufferUAV;

	// Cluster visibility data (needed for debug visualization)
	Parameters.SplatClusterIndexBuffer = GPUResources->SplatClusterIndexBufferSRV;
	Parameters.ClusterVisibilityBitmap = GPUResources->ClusterVisibilityBitmapSRV;
	Parameters.LODClusterSelectedBitmap = GPUResources->LODClusterSelectedBitmapSRV;
	Parameters.SelectedClusterBuffer = GPUResources->SelectedClusterBufferSRV;
	Parameters.UseClusterCulling = 1;
	Parameters.UseLODRendering = 0;
	Parameters.OriginalSplatCount = OriginalSplatCount;

	// COMPACTION MODE: Use compacted indices
	Parameters.CompactedSplatIndices = GPUResources->CompactedSplatIndicesBufferSRV;
	Parameters.UseCompaction = 1;
	Parameters.VisibleSplatCount = SplatCount;

	// Cast to FViewInfo to access PrevViewInfo for velocity calculation
	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);

	// Matrices
	Parameters.LocalToWorld = FMatrix44f(LocalToWorld);
	Parameters.WorldToPLY = FMatrix44f(ComputeWorldToPLY(LocalToWorld));
	Parameters.WorldToClip = FMatrix44f(View.ViewMatrices.GetViewProjectionMatrix());
	Parameters.PreViewTranslation = FVector3f(View.ViewMatrices.GetPreViewTranslation());
	Parameters.WorldToView = FMatrix44f(View.ViewMatrices.GetViewMatrix());
	Parameters.CameraPosition = FVector3f(View.ViewMatrices.GetViewOrigin());

	FIntRect ViewRect = ViewInfo.ViewRect;
	Parameters.ScreenSize = FVector2f(ViewRect.Width(), ViewRect.Height());

	const FMatrix& ProjMatrix = View.ViewMatrices.GetProjectionMatrix();
	Parameters.FocalLength = FVector2f(
		ProjMatrix.M[0][0] * ViewRect.Width() * 0.5f,
		ProjMatrix.M[1][1] * ViewRect.Height() * 0.5f
	);

	Parameters.SplatCount = SplatCount;
	// Use effective SH order: min(requested order, available data)
	int32 EffectiveSHOrder = FMath::Min(SHOrder, GPUResources->GetSHBands());
	Parameters.SHOrder = EffectiveSHOrder;
	// SH buffer includes DC + higher-order coefficients: band1=4, band2=9, band3=16
	Parameters.NumSHCoeffs = (EffectiveSHOrder == 0) ? 0 : (EffectiveSHOrder == 1) ? 4 : (EffectiveSHOrder == 2) ? 9 : 16;
	{
		int32 StoredSHBands = GPUResources->GetSHBands();
		Parameters.SHBufferCoeffs = (StoredSHBands == 0) ? 0 : (StoredSHBands == 1) ? 4 : (StoredSHBands == 2) ? 9 : 16;
	}
	Parameters.UseSHRendering = (EffectiveSHOrder > 0) ? 1 : 0;
	Parameters.OpacityScale = OpacityScale;
	Parameters.SplatScale = SplatScale;

	// Per-proxy compaction path
	Parameters.GlobalBaseOffsetsBuffer = GPUResources->CompactedSplatIndicesBufferSRV;  // dummy
	Parameters.ProxyIndex = 0;
	Parameters.UseGlobalCompactionPath = 0;
	Parameters.MaxRenderBudget = 0;

	// INDIRECT DISPATCH using prepared args
	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchIndirectComputeShader(GPUResources->IndirectDispatchArgsBuffer, 0);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());

	// Transition buffer for next stage
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->ViewDataBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
}

void FGaussianSplatRenderer::DispatchCalcDistancesIndirect(
	FRHICommandListImmediate& RHICmdList,
	FGaussianSplatGPUResources* GPUResources)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatCalcDistancesIndirect);

	TShaderMapRef<FGaussianSplatCalcDistancesCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FGaussianSplatCalcDistancesCS shader not valid"));
		return;
	}

	// Transition buffers
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortDistanceBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortKeysBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	// No need to clear distance buffer - indirect radix sort only sorts VisibleSplatCount elements,
	// and DrawSplats only renders VisibleSplatCount instances via indirect draw.
	// Unused buffer entries beyond VisibleSplatCount are never read.

	FGaussianSplatCalcDistancesCS::FParameters Parameters;
	Parameters.ViewDataBuffer = GPUResources->ViewDataBufferSRV;
	Parameters.DistanceBuffer = GPUResources->SortDistanceBufferUAV;
	Parameters.KeyBuffer = GPUResources->SortKeysBufferUAV;
	Parameters.SplatCount = GPUResources->TotalSplatCount;  // Max count for bounds

	// INDIRECT DISPATCH
	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchIndirectComputeShader(GPUResources->IndirectDispatchArgsBuffer, 0);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());
}

//----------------------------------------------------------------------
// Global Accumulator dispatch implementations
//----------------------------------------------------------------------

void FGaussianSplatRenderer::DispatchCalcViewDataGlobal(
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
	FGaussianGlobalAccumulator* GlobalAccumulator)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatCalcViewDataGlobal);

	TShaderMapRef<FGaussianSplatCalcViewDataCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FGaussianSplatCalcViewDataCS shader not valid"));
		return;
	}

	// Transition global buffer to UAV for writing (Unknown covers both first-proxy and subsequent calls)
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalViewDataBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	FGaussianSplatCalcViewDataCS::FParameters Parameters;
	Parameters.PackedSplatBuffer = GPUResources->PackedSplatBufferSRV;
	Parameters.SHBuffer = GPUResources->SHBufferSRV;

	// Write into the GLOBAL buffer at GlobalBaseOffset
	Parameters.ViewDataBuffer = GlobalAccumulator->GlobalViewDataBufferUAV;

	// Cluster visibility
	Parameters.SplatClusterIndexBuffer = GPUResources->SplatClusterIndexBufferSRV;
	Parameters.ClusterVisibilityBitmap = GPUResources->ClusterVisibilityBitmapSRV;
	Parameters.LODClusterSelectedBitmap = GPUResources->LODClusterSelectedBitmapSRV;
	Parameters.SelectedClusterBuffer = GPUResources->SelectedClusterBufferSRV;

	if (GPUResources->bHasClusterData)
	{
		Parameters.UseClusterCulling = 1;
		Parameters.UseLODRendering = bUseLODRendering ? 1 : 0;
		Parameters.OriginalSplatCount = SplatCount - GPUResources->LODSplatCount;
	}
	else
	{
		Parameters.UseClusterCulling = 0;
		Parameters.UseLODRendering = 0;
		Parameters.OriginalSplatCount = SplatCount;
	}

	Parameters.CompactedSplatIndices = GPUResources->CompactedSplatIndicesBufferSRV;
	Parameters.UseCompaction = 0;
	Parameters.VisibleSplatCount = 0;

	// Cast to FViewInfo to access PrevViewInfo for velocity calculation
	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);

	// Transform matrices
	Parameters.LocalToWorld = FMatrix44f(LocalToWorld);
	Parameters.WorldToPLY = FMatrix44f(ComputeWorldToPLY(LocalToWorld));
	Parameters.WorldToClip = FMatrix44f(View.ViewMatrices.GetViewProjectionMatrix());
	Parameters.PreViewTranslation = FVector3f(View.ViewMatrices.GetPreViewTranslation());
	Parameters.WorldToView = FMatrix44f(View.ViewMatrices.GetViewMatrix());
	Parameters.CameraPosition = FVector3f(View.ViewMatrices.GetViewOrigin());

	FIntRect ViewRect = ViewInfo.ViewRect;
	Parameters.ScreenSize = FVector2f(ViewRect.Width(), ViewRect.Height());

	const FMatrix& ProjMatrix = View.ViewMatrices.GetProjectionMatrix();
	Parameters.FocalLength = FVector2f(
		ProjMatrix.M[0][0] * ViewRect.Width() * 0.5f,
		ProjMatrix.M[1][1] * ViewRect.Height() * 0.5f
	);

	Parameters.SplatCount = SplatCount;
	// Use effective SH order: min(requested order, available data)
	int32 EffectiveSHOrder = FMath::Min(SHOrder, GPUResources->GetSHBands());
	Parameters.SHOrder = EffectiveSHOrder;
	// SH buffer includes DC + higher-order coefficients: band1=4, band2=9, band3=16
	Parameters.NumSHCoeffs = (EffectiveSHOrder == 0) ? 0 : (EffectiveSHOrder == 1) ? 4 : (EffectiveSHOrder == 2) ? 9 : 16;
	{
		int32 StoredSHBands = GPUResources->GetSHBands();
		Parameters.SHBufferCoeffs = (StoredSHBands == 0) ? 0 : (StoredSHBands == 1) ? 4 : (StoredSHBands == 2) ? 9 : 16;
	}
	Parameters.UseSHRendering = (EffectiveSHOrder > 0) ? 1 : 0;
	Parameters.OpacityScale = OpacityScale;
	Parameters.SplatScale = SplatScale;

	// KEY: tell the shader where to write in the global buffer
	Parameters.GlobalBaseOffset = GlobalBaseOffset;

	// Non-compaction global path
	Parameters.GlobalBaseOffsetsBuffer = GPUResources->CompactedSplatIndicesBufferSRV;  // dummy
	Parameters.ProxyIndex = 0;
	Parameters.UseGlobalCompactionPath = 0;
	Parameters.MaxRenderBudget = 0;

	const uint32 ThreadGroupSize = 256;
	const uint32 NumGroups = FMath::DivideAndRoundUp((uint32)SplatCount, ThreadGroupSize);

	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchComputeShader(NumGroups, 1, 1);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());
	// Note: caller is responsible for the final SRVCompute transition after all proxies have written
}

void FGaussianSplatRenderer::DispatchCalcDistancesGlobal(
	FRHICommandListImmediate& RHICmdList,
	FGaussianGlobalAccumulator* GlobalAccumulator,
	int32 TotalSplatCount)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatCalcDistancesGlobal);

	TShaderMapRef<FGaussianSplatCalcDistancesCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FGaussianSplatCalcDistancesCS shader not valid"));
		return;
	}

	// Transition global ViewData to SRV (all proxies have finished writing)
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalViewDataBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortDistanceBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortKeysBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	FGaussianSplatCalcDistancesCS::FParameters Parameters;
	Parameters.ViewDataBuffer = GlobalAccumulator->GlobalViewDataBufferSRV;
	Parameters.DistanceBuffer = GlobalAccumulator->GlobalSortDistanceBufferUAV;
	Parameters.KeyBuffer = GlobalAccumulator->GlobalSortKeysBufferUAV;
	Parameters.SplatCount = TotalSplatCount;

	const uint32 ThreadGroupSize = 256;
	const uint32 NumGroups = FMath::DivideAndRoundUp((uint32)TotalSplatCount, ThreadGroupSize);

	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchComputeShader(NumGroups, 1, 1);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());
}

void FGaussianSplatRenderer::DispatchRadixSortGlobal(
	FRHICommandListImmediate& RHICmdList,
	FGaussianGlobalAccumulator* GlobalAccumulator,
	int32 TotalSplatCount)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatRadixSortGlobal);

	TShaderMapRef<FRadixSortCountCS> CountShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortPrefixSumCS> PrefixSumShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortDigitPrefixSumCS> DigitPrefixSumShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortScatterCS> ScatterShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!CountShader.IsValid() || !PrefixSumShader.IsValid() ||
		!DigitPrefixSumShader.IsValid() || !ScatterShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("Radix sort global shaders not valid"));
		return;
	}

	uint32 SortCount = (uint32)TotalSplatCount;
	uint32 NumTiles = FMath::DivideAndRoundUp(SortCount, 1024u);

	FUnorderedAccessViewRHIRef DistUAVs[2] = {
		GlobalAccumulator->GlobalSortDistanceBufferUAV,
		GlobalAccumulator->GlobalSortDistanceBufferAltUAV
	};
	FBufferRHIRef DistBuffers[2] = {
		GlobalAccumulator->GlobalSortDistanceBuffer,
		GlobalAccumulator->GlobalSortDistanceBufferAlt
	};
	FUnorderedAccessViewRHIRef KeyUAVs[2] = {
		GlobalAccumulator->GlobalSortKeysBufferUAV,
		GlobalAccumulator->GlobalSortKeysBufferAltUAV
	};
	FBufferRHIRef KeyBuffers[2] = {
		GlobalAccumulator->GlobalSortKeysBuffer,
		GlobalAccumulator->GlobalSortKeysBufferAlt
	};

	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortDistanceBufferAlt, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortKeysBufferAlt, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixHistogramBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixDigitOffsetBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortParamsBuffer, ERHIAccess::Unknown, ERHIAccess::SRVCompute));

	for (uint32 Pass = 0; Pass < 4; Pass++)
	{
		uint32 RadixShift = Pass * 8;
		uint32 SrcIdx = Pass & 1;
		uint32 DstIdx = 1 - SrcIdx;

		// CountCS
		{
			FRadixSortCountCS::FParameters Params;
			Params.HistogramBuffer = GlobalAccumulator->GlobalRadixHistogramBufferUAV;
			Params.SrcKeys = DistUAVs[SrcIdx];
			Params.SortParams = GlobalAccumulator->GlobalSortParamsBufferSRV;
			Params.RadixShift = RadixShift;
			Params.Count = SortCount;
			Params.NumTiles = NumTiles;
			Params.UseIndirectSort = 0;

			SetComputePipelineState(RHICmdList, CountShader.GetComputeShader());
			SetShaderParameters(RHICmdList, CountShader, CountShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(NumTiles, 1, 1);
			UnsetShaderUAVs(RHICmdList, CountShader, CountShader.GetComputeShader());
		}

		RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixHistogramBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// PrefixSumCS
		{
			FRadixSortPrefixSumCS::FParameters Params;
			Params.HistogramBuffer = GlobalAccumulator->GlobalRadixHistogramBufferUAV;
			Params.DigitOffsetBuffer = GlobalAccumulator->GlobalRadixDigitOffsetBufferUAV;
			Params.SortParams = GlobalAccumulator->GlobalSortParamsBufferSRV;
			Params.NumTiles = NumTiles;
			Params.UseIndirectSort = 0;

			SetComputePipelineState(RHICmdList, PrefixSumShader.GetComputeShader());
			SetShaderParameters(RHICmdList, PrefixSumShader, PrefixSumShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(256, 1, 1);
			UnsetShaderUAVs(RHICmdList, PrefixSumShader, PrefixSumShader.GetComputeShader());
		}

		RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixDigitOffsetBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// DigitPrefixSumCS
		{
			FRadixSortDigitPrefixSumCS::FParameters Params;
			Params.DigitOffsetBuffer = GlobalAccumulator->GlobalRadixDigitOffsetBufferUAV;

			SetComputePipelineState(RHICmdList, DigitPrefixSumShader.GetComputeShader());
			SetShaderParameters(RHICmdList, DigitPrefixSumShader, DigitPrefixSumShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(1, 1, 1);
			UnsetShaderUAVs(RHICmdList, DigitPrefixSumShader, DigitPrefixSumShader.GetComputeShader());
		}

		RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixDigitOffsetBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
		RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixHistogramBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// ScatterCS
		{
			FRadixSortScatterCS::FParameters Params;
			Params.SrcKeys = DistUAVs[SrcIdx];
			Params.SrcVals = KeyUAVs[SrcIdx];
			Params.DstKeys = DistUAVs[DstIdx];
			Params.DstVals = KeyUAVs[DstIdx];
			Params.HistogramBuffer = GlobalAccumulator->GlobalRadixHistogramBufferUAV;
			Params.DigitOffsetBuffer = GlobalAccumulator->GlobalRadixDigitOffsetBufferUAV;
			Params.SortParams = GlobalAccumulator->GlobalSortParamsBufferSRV;
			Params.RadixShift = RadixShift;
			Params.Count = SortCount;
			Params.NumTiles = NumTiles;
			Params.UseIndirectSort = 0;

			SetComputePipelineState(RHICmdList, ScatterShader.GetComputeShader());
			SetShaderParameters(RHICmdList, ScatterShader, ScatterShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(NumTiles, 1, 1);
			UnsetShaderUAVs(RHICmdList, ScatterShader, ScatterShader.GetComputeShader());
		}

		RHICmdList.Transition(FRHITransitionInfo(DistBuffers[DstIdx], ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
		RHICmdList.Transition(FRHITransitionInfo(KeyBuffers[DstIdx], ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	}

	// After 4 passes (even), result is in primary buffers [0]
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortKeysBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVGraphics));
}

void FGaussianSplatRenderer::DrawSplatsGlobal(
	FRHICommandListImmediate& RHICmdList,
	const FSceneView& View,
	FGaussianGlobalAccumulator* GlobalAccumulator,
	FBufferRHIRef IndexBuffer,
	int32 TotalSplatCount,
	int32 DebugMode)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatDrawGlobal);

	if (!IndexBuffer.IsValid())
	{
		return;
	}

	TShaderMapRef<FGaussianSplatVS> VertexShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FGaussianSplatPS> PixelShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!VertexShader.IsValid() || !PixelShader.IsValid())
	{
		return;
	}

	// Transition global ViewData for graphics reads
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalViewDataBuffer, ERHIAccess::Unknown, ERHIAccess::SRVGraphics));

	FGraphicsPipelineStateInitializer GraphicsPSOInit;
	RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);

	GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
	// Enable depth writes for TSR/TAA - splats write depth at their center position
	// Using DepthNearOrEqual allows splats at similar depths to all blend correctly
	GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<true, CF_DepthNearOrEqual>::GetRHI();
	// Blend mode for MRT: RT0 (sRGB intermediate) with premultiplied alpha, RT1 (Velocity) with replacement
	// CW_RGBA on RT0 so accumulated alpha is tracked for the composite pass
	GraphicsPSOInit.BlendState = TStaticBlendState<
		// RT0: ColorWriteMask, ColorBlendOp, ColorSrcBlend, ColorDestBlend, AlphaBlendOp, AlphaSrcBlend, AlphaDestBlend
		CW_RGBA, BO_Add, BF_One, BF_InverseSourceAlpha, BO_Add, BF_One, BF_InverseSourceAlpha,
		// RT1: ColorWriteMask, ColorBlendOp, ColorSrcBlend, ColorDestBlend, AlphaBlendOp, AlphaSrcBlend, AlphaDestBlend
		CW_RGBA, BO_Add, BF_One, BF_Zero, BO_Add, BF_One, BF_Zero
	>::GetRHI();
	GraphicsPSOInit.PrimitiveType = PT_TriangleList;
	GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GEmptyVertexDeclaration.VertexDeclarationRHI;
	GraphicsPSOInit.BoundShaderState.VertexShaderRHI = VertexShader.GetVertexShader();
	GraphicsPSOInit.BoundShaderState.PixelShaderRHI = PixelShader.GetPixelShader();

	SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit, 0);

	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);
	FIntRect ViewRect = ViewInfo.ViewRect;
	RHICmdList.SetViewport(
		ViewRect.Min.X, ViewRect.Min.Y, 0.0f,
		ViewRect.Max.X, ViewRect.Max.Y, 1.0f
	);

	FGaussianSplatVS::FParameters VSParameters;
	VSParameters.ViewDataBuffer = GlobalAccumulator->GlobalViewDataBufferSRV;
	VSParameters.SortKeysBuffer = GlobalAccumulator->GlobalSortKeysBufferSRV;
	VSParameters.SplatCount = TotalSplatCount;
	VSParameters.DebugMode = static_cast<uint32>(FMath::Max(0, DebugMode));
	VSParameters.EnableNanite = 1;

	SetShaderParameters(RHICmdList, VertexShader, VertexShader.GetVertexShader(), VSParameters);

	FGaussianSplatPS::FParameters PSParameters;
	// Velocity calculation (Nanite-style): use previous frame's translated view-projection matrix
	PSParameters.PrevTranslatedWorldToClip = FMatrix44f(ViewInfo.PrevViewInfo.ViewMatrices.GetTranslatedViewProjectionMatrix());
	// Pass screen size inverse for NDC conversion (avoids View uniform buffer binding)
	PSParameters.ScreenSizeInverse = FVector2f(1.0f / ViewRect.Width(), 1.0f / ViewRect.Height());
	// TAA jitter: xy = current frame, zw = previous frame (in NDC space)
	PSParameters.TemporalAAJitter = FVector4f(
		View.ViewMatrices.GetTemporalAAJitter().X,
		View.ViewMatrices.GetTemporalAAJitter().Y,
		ViewInfo.PrevViewInfo.ViewMatrices.GetTemporalAAJitter().X,
		ViewInfo.PrevViewInfo.ViewMatrices.GetTemporalAAJitter().Y
	);
	// PreViewTranslation for converting TranslatedWorld back to World
	PSParameters.PreViewTranslation = FVector3f(View.ViewMatrices.GetPreViewTranslation());
	// Previous frame's PreViewTranslation for correct velocity calculation
	PSParameters.PrevPreViewTranslation = FVector3f(ViewInfo.PrevViewInfo.ViewMatrices.GetPreViewTranslation());
	SetShaderParameters(RHICmdList, PixelShader, PixelShader.GetPixelShader(), PSParameters);

	RHICmdList.SetStreamSource(0, nullptr, 0);
	RHICmdList.DrawIndexedPrimitive(
		IndexBuffer,
		0,              // BaseVertexIndex
		0,              // FirstInstance
		4,              // NumVertices
		0,              // StartIndex
		2,              // NumPrimitives (2 triangles per quad)
		TotalSplatCount // NumInstances
	);
}

//----------------------------------------------------------------------
// Global Accumulator + Nanite Compaction dispatch implementations
//----------------------------------------------------------------------

void FGaussianSplatRenderer::DispatchGatherVisibleCount(
	FRHICommandListImmediate& RHICmdList,
	FGaussianSplatGPUResources* GPUResources,
	FGaussianGlobalAccumulator* GlobalAccumulator,
	int32 ProxyIndex)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatGatherVisibleCount);

	TShaderMapRef<FGatherVisibleCountsCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FGatherVisibleCountsCS shader not valid"));
		return;
	}

	// VisibleSplatCountBuffer is already in SRVCompute after DispatchCompactSplats
	// GlobalVisibleCountArrayBuffer: transition to UAV for writing
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalVisibleCountArrayBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	FGatherVisibleCountsCS::FParameters Parameters;
	Parameters.PerProxyVisibleCount  = GPUResources->VisibleSplatCountBufferSRV;
	Parameters.GlobalVisibleCountArray = GlobalAccumulator->GlobalVisibleCountArrayBufferUAV;
	Parameters.ProxyIndex            = (uint32)ProxyIndex;

	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchComputeShader(1, 1, 1);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());

	// UAV barrier so next proxy's write (or the PrefixSum read) sees this result
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalVisibleCountArrayBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
}

void FGaussianSplatRenderer::DispatchPrefixSumVisibleCounts(
	FRHICommandListImmediate& RHICmdList,
	FGaussianGlobalAccumulator* GlobalAccumulator,
	int32 ProxyCount,
	uint32 MaxRenderBudget)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatPrefixSumVisibleCounts);

	TShaderMapRef<FPrefixSumVisibleCountsCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FPrefixSumVisibleCountsCS shader not valid"));
		return;
	}

	// GlobalVisibleCountArray: all GatherCS writes done → transition to SRV for reading
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalVisibleCountArrayBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));

	// Output buffers → UAV for writing
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalBaseOffsetsBuffer,            ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalCalcDistIndirectArgsBuffer,    ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortIndirectArgsGlobalBuffer,  ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortParamsBuffer,              ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalDrawIndirectArgsBuffer,        ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	FPrefixSumVisibleCountsCS::FParameters Parameters;
	Parameters.GlobalVisibleCountArray   = GlobalAccumulator->GlobalVisibleCountArrayBufferSRV;
	Parameters.GlobalBaseOffsets         = GlobalAccumulator->GlobalBaseOffsetsBufferUAV;
	Parameters.GlobalCalcDistIndirectArgs = GlobalAccumulator->GlobalCalcDistIndirectArgsBufferUAV;
	Parameters.GlobalSortIndirectArgs    = GlobalAccumulator->GlobalSortIndirectArgsGlobalBufferUAV;
	Parameters.GlobalSortParams          = GlobalAccumulator->GlobalSortParamsBufferUAV;
	Parameters.GlobalDrawIndirectArgs    = GlobalAccumulator->GlobalDrawIndirectArgsBufferUAV;
	Parameters.ProxyCount                = (uint32)ProxyCount;
	Parameters.MaxRenderBudget           = MaxRenderBudget;

	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchComputeShader(1, 1, 1);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());

	// GlobalBaseOffsetsBuffer → SRV so CalcViewDataCompactedGlobal can read it
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalBaseOffsetsBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));

	// Indirect arg buffers → IndirectArgs state for the GPU-driven dispatches/draw
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalCalcDistIndirectArgsBuffer,   ERHIAccess::UAVCompute, ERHIAccess::IndirectArgs));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortIndirectArgsGlobalBuffer, ERHIAccess::UAVCompute, ERHIAccess::IndirectArgs));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalDrawIndirectArgsBuffer,       ERHIAccess::UAVCompute, ERHIAccess::IndirectArgs));

	// GlobalSortParamsBuffer → SRV for radix sort shaders (UseIndirectSort=1)
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortParamsBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
}

void FGaussianSplatRenderer::DispatchCalcViewDataCompactedGlobal(
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
	uint32 MaxRenderBudget)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatCalcViewDataCompactedGlobal);

	TShaderMapRef<FGaussianSplatCalcViewDataCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FGaussianSplatCalcViewDataCS shader not valid"));
		return;
	}

	// Transition global ViewData buffer to UAV (Unknown handles first proxy and subsequent proxies)
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalViewDataBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	FGaussianSplatCalcViewDataCS::FParameters Parameters;
	Parameters.PackedSplatBuffer = GPUResources->PackedSplatBufferSRV;
	Parameters.SHBuffer = GPUResources->SHBufferSRV;

	// Write into GLOBAL buffer
	Parameters.ViewDataBuffer = GlobalAccumulator->GlobalViewDataBufferUAV;

	// Cluster visibility data
	Parameters.SplatClusterIndexBuffer   = GPUResources->SplatClusterIndexBufferSRV;
	Parameters.ClusterVisibilityBitmap   = GPUResources->ClusterVisibilityBitmapSRV;
	Parameters.LODClusterSelectedBitmap  = GPUResources->LODClusterSelectedBitmapSRV;
	Parameters.SelectedClusterBuffer     = GPUResources->SelectedClusterBufferSRV;
	Parameters.UseClusterCulling         = 1;
	Parameters.UseLODRendering           = 0;
	Parameters.OriginalSplatCount        = OriginalSplatCount;

	// COMPACTION MODE
	Parameters.CompactedSplatIndices = GPUResources->CompactedSplatIndicesBufferSRV;
	Parameters.UseCompaction         = 1;
	Parameters.VisibleSplatCount     = SplatCount;

	// GLOBAL COMPACTION PATH
	Parameters.GlobalBaseOffsetsBuffer   = GlobalAccumulator->GlobalBaseOffsetsBufferSRV;
	Parameters.ProxyIndex                = (uint32)ProxyIndex;
	Parameters.UseGlobalCompactionPath   = 1;
	Parameters.GlobalBaseOffset          = 0;
	Parameters.MaxRenderBudget           = MaxRenderBudget;

	// Cast to FViewInfo to access PrevViewInfo for velocity calculation
	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);

	// Transform matrices
	Parameters.LocalToWorld    = FMatrix44f(LocalToWorld);
	Parameters.WorldToPLY      = FMatrix44f(ComputeWorldToPLY(LocalToWorld));
	Parameters.WorldToClip     = FMatrix44f(View.ViewMatrices.GetViewProjectionMatrix());
	Parameters.PreViewTranslation = FVector3f(View.ViewMatrices.GetPreViewTranslation());
	Parameters.WorldToView     = FMatrix44f(View.ViewMatrices.GetViewMatrix());
	Parameters.CameraPosition  = FVector3f(View.ViewMatrices.GetViewOrigin());

	FIntRect ViewRect = ViewInfo.ViewRect;
	Parameters.ScreenSize = FVector2f(ViewRect.Width(), ViewRect.Height());

	const FMatrix& ProjMatrix = View.ViewMatrices.GetProjectionMatrix();
	Parameters.FocalLength = FVector2f(
		ProjMatrix.M[0][0] * ViewRect.Width() * 0.5f,
		ProjMatrix.M[1][1] * ViewRect.Height() * 0.5f
	);

	Parameters.SplatCount    = SplatCount;
	// Use actual available SH data, not requested SH order
	int32 EffectiveSHOrder = FMath::Min(SHOrder, GPUResources->GetSHBands());
	Parameters.SHOrder       = EffectiveSHOrder;
	// SH buffer includes DC + higher-order coefficients: band1=4, band2=9, band3=16
	Parameters.NumSHCoeffs   = (EffectiveSHOrder == 0) ? 0 : (EffectiveSHOrder == 1) ? 4 : (EffectiveSHOrder == 2) ? 9 : 16;
	{
		int32 StoredSHBands = GPUResources->GetSHBands();
		Parameters.SHBufferCoeffs = (StoredSHBands == 0) ? 0 : (StoredSHBands == 1) ? 4 : (StoredSHBands == 2) ? 9 : 16;
	}
	Parameters.UseSHRendering = (EffectiveSHOrder > 0) ? 1 : 0;
	Parameters.OpacityScale  = OpacityScale;
	Parameters.SplatScale    = SplatScale;

	// Use the per-proxy indirect dispatch args (filled by DispatchPrepareIndirectArgs)
	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchIndirectComputeShader(GPUResources->IndirectDispatchArgsBuffer, 0);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());

	// Note: caller transitions GlobalViewDataBuffer to SRVCompute after all proxies are done
}

void FGaussianSplatRenderer::DispatchCalcDistancesGlobalIndirect(
	FRHICommandListImmediate& RHICmdList,
	FGaussianGlobalAccumulator* GlobalAccumulator)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatCalcDistancesGlobalIndirect);

	TShaderMapRef<FGaussianSplatCalcDistancesCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	if (!ComputeShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("FGaussianSplatCalcDistancesCS shader not valid"));
		return;
	}

	// All proxies have finished writing — transition global ViewData to SRV
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalViewDataBuffer,    ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortDistanceBuffer, ERHIAccess::Unknown,   ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortKeysBuffer,     ERHIAccess::Unknown,   ERHIAccess::UAVCompute));

	FGaussianSplatCalcDistancesCS::FParameters Parameters;
	Parameters.ViewDataBuffer  = GlobalAccumulator->GlobalViewDataBufferSRV;
	Parameters.DistanceBuffer  = GlobalAccumulator->GlobalSortDistanceBufferUAV;
	Parameters.KeyBuffer       = GlobalAccumulator->GlobalSortKeysBufferUAV;
	// AllocatedCount is always >= TotalVisible — safe upper bound for the shader guard
	Parameters.SplatCount      = GlobalAccumulator->AllocatedCount;

	// GlobalCalcDistIndirectArgsBuffer is in IndirectArgs state from DispatchPrefixSumVisibleCounts
	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchIndirectComputeShader(GlobalAccumulator->GlobalCalcDistIndirectArgsBuffer, 0);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());
}

void FGaussianSplatRenderer::DispatchRadixSortGlobalIndirect(
	FRHICommandListImmediate& RHICmdList,
	FGaussianGlobalAccumulator* GlobalAccumulator)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatRadixSortGlobalIndirect);

	TShaderMapRef<FRadixSortCountCS> CountShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortPrefixSumCS> PrefixSumShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortDigitPrefixSumCS> DigitPrefixSumShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FRadixSortScatterCS> ScatterShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!CountShader.IsValid() || !PrefixSumShader.IsValid() ||
		!DigitPrefixSumShader.IsValid() || !ScatterShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("Radix sort global indirect shaders not valid"));
		return;
	}

	FUnorderedAccessViewRHIRef DistUAVs[2] = {
		GlobalAccumulator->GlobalSortDistanceBufferUAV,
		GlobalAccumulator->GlobalSortDistanceBufferAltUAV
	};
	FBufferRHIRef DistBuffers[2] = {
		GlobalAccumulator->GlobalSortDistanceBuffer,
		GlobalAccumulator->GlobalSortDistanceBufferAlt
	};
	FUnorderedAccessViewRHIRef KeyUAVs[2] = {
		GlobalAccumulator->GlobalSortKeysBufferUAV,
		GlobalAccumulator->GlobalSortKeysBufferAltUAV
	};
	FBufferRHIRef KeyBuffers[2] = {
		GlobalAccumulator->GlobalSortKeysBuffer,
		GlobalAccumulator->GlobalSortKeysBufferAlt
	};

	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortDistanceBufferAlt,  ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortKeysBufferAlt,      ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixHistogramBuffer,   ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixDigitOffsetBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	// GlobalSortParamsBuffer and GlobalSortIndirectArgsGlobalBuffer are already
	// in SRVCompute / IndirectArgs state from DispatchPrefixSumVisibleCounts

	// 4 radix passes: bits 0-7, 8-15, 16-23, 24-31
	for (uint32 Pass = 0; Pass < 4; Pass++)
	{
		uint32 RadixShift = Pass * 8;
		uint32 SrcIdx = Pass & 1;
		uint32 DstIdx = 1 - SrcIdx;

		// --- CountCS: indirect dispatch (GlobalSortIndirectArgsGlobalBuffer) ---
		{
			FRadixSortCountCS::FParameters Params;
			Params.HistogramBuffer = GlobalAccumulator->GlobalRadixHistogramBufferUAV;
			Params.SrcKeys         = DistUAVs[SrcIdx];
			Params.SortParams      = GlobalAccumulator->GlobalSortParamsBufferSRV;
			Params.RadixShift      = RadixShift;
			Params.Count           = 0;   // Unused when UseIndirectSort=1
			Params.NumTiles        = 0;   // Unused when UseIndirectSort=1
			Params.UseIndirectSort = 1;

			SetComputePipelineState(RHICmdList, CountShader.GetComputeShader());
			SetShaderParameters(RHICmdList, CountShader, CountShader.GetComputeShader(), Params);
			RHICmdList.DispatchIndirectComputeShader(GlobalAccumulator->GlobalSortIndirectArgsGlobalBuffer, 0);
			UnsetShaderUAVs(RHICmdList, CountShader, CountShader.GetComputeShader());
		}

		RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixHistogramBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// --- PrefixSumCS: fixed dispatch (256, 1, 1) — reads NumTiles from GlobalSortParamsBuffer ---
		{
			FRadixSortPrefixSumCS::FParameters Params;
			Params.HistogramBuffer    = GlobalAccumulator->GlobalRadixHistogramBufferUAV;
			Params.DigitOffsetBuffer  = GlobalAccumulator->GlobalRadixDigitOffsetBufferUAV;
			Params.SortParams         = GlobalAccumulator->GlobalSortParamsBufferSRV;
			Params.NumTiles           = 0;   // Unused when UseIndirectSort=1
			Params.UseIndirectSort    = 1;

			SetComputePipelineState(RHICmdList, PrefixSumShader.GetComputeShader());
			SetShaderParameters(RHICmdList, PrefixSumShader, PrefixSumShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(256, 1, 1);
			UnsetShaderUAVs(RHICmdList, PrefixSumShader, PrefixSumShader.GetComputeShader());
		}

		RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixDigitOffsetBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// --- DigitPrefixSumCS: single dispatch ---
		{
			FRadixSortDigitPrefixSumCS::FParameters Params;
			Params.DigitOffsetBuffer = GlobalAccumulator->GlobalRadixDigitOffsetBufferUAV;

			SetComputePipelineState(RHICmdList, DigitPrefixSumShader.GetComputeShader());
			SetShaderParameters(RHICmdList, DigitPrefixSumShader, DigitPrefixSumShader.GetComputeShader(), Params);
			RHICmdList.DispatchComputeShader(1, 1, 1);
			UnsetShaderUAVs(RHICmdList, DigitPrefixSumShader, DigitPrefixSumShader.GetComputeShader());
		}

		RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixDigitOffsetBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
		RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalRadixHistogramBuffer,   ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

		// --- ScatterCS: indirect dispatch ---
		{
			FRadixSortScatterCS::FParameters Params;
			Params.SrcKeys         = DistUAVs[SrcIdx];
			Params.SrcVals         = KeyUAVs[SrcIdx];
			Params.DstKeys         = DistUAVs[DstIdx];
			Params.DstVals         = KeyUAVs[DstIdx];
			Params.HistogramBuffer = GlobalAccumulator->GlobalRadixHistogramBufferUAV;
			Params.DigitOffsetBuffer = GlobalAccumulator->GlobalRadixDigitOffsetBufferUAV;
			Params.SortParams      = GlobalAccumulator->GlobalSortParamsBufferSRV;
			Params.RadixShift      = RadixShift;
			Params.Count           = 0;   // Unused when UseIndirectSort=1
			Params.NumTiles        = 0;   // Unused when UseIndirectSort=1
			Params.UseIndirectSort = 1;

			SetComputePipelineState(RHICmdList, ScatterShader.GetComputeShader());
			SetShaderParameters(RHICmdList, ScatterShader, ScatterShader.GetComputeShader(), Params);
			RHICmdList.DispatchIndirectComputeShader(GlobalAccumulator->GlobalSortIndirectArgsGlobalBuffer, 0);
			UnsetShaderUAVs(RHICmdList, ScatterShader, ScatterShader.GetComputeShader());
		}

		RHICmdList.Transition(FRHITransitionInfo(DistBuffers[DstIdx], ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
		RHICmdList.Transition(FRHITransitionInfo(KeyBuffers[DstIdx],  ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	}

	// After 4 passes (even), result is in primary buffers [0]
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalSortKeysBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVGraphics));
}

void FGaussianSplatRenderer::DrawSplatsGlobalIndirect(
	FRHICommandListImmediate& RHICmdList,
	const FSceneView& View,
	FGaussianGlobalAccumulator* GlobalAccumulator,
	FBufferRHIRef IndexBuffer,
	int32 DebugMode)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatDrawGlobalIndirect);

	if (!IndexBuffer.IsValid())
	{
		return;
	}

	TShaderMapRef<FGaussianSplatVS> VertexShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FGaussianSplatPS> PixelShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!VertexShader.IsValid() || !PixelShader.IsValid())
	{
		return;
	}

	// Transition global ViewData for graphics reads (SortKeysBuffer already in SRVGraphics)
	RHICmdList.Transition(FRHITransitionInfo(GlobalAccumulator->GlobalViewDataBuffer, ERHIAccess::Unknown, ERHIAccess::SRVGraphics));

	FGraphicsPipelineStateInitializer GraphicsPSOInit;
	RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);

	GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
	// Enable depth writes for TSR/TAA - splats write depth at their center position
	// Using DepthNearOrEqual allows splats at similar depths to all blend correctly
	GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<true, CF_DepthNearOrEqual>::GetRHI();
	// Blend mode for MRT: RT0 (sRGB intermediate) with premultiplied alpha, RT1 (Velocity) with replacement
	// CW_RGBA on RT0 so accumulated alpha is tracked for the composite pass
	GraphicsPSOInit.BlendState = TStaticBlendState<
		// RT0: ColorWriteMask, ColorBlendOp, ColorSrcBlend, ColorDestBlend, AlphaBlendOp, AlphaSrcBlend, AlphaDestBlend
		CW_RGBA, BO_Add, BF_One, BF_InverseSourceAlpha, BO_Add, BF_One, BF_InverseSourceAlpha,
		// RT1: ColorWriteMask, ColorBlendOp, ColorSrcBlend, ColorDestBlend, AlphaBlendOp, AlphaSrcBlend, AlphaDestBlend
		CW_RGBA, BO_Add, BF_One, BF_Zero, BO_Add, BF_One, BF_Zero
	>::GetRHI();
	GraphicsPSOInit.PrimitiveType = PT_TriangleList;
	GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GEmptyVertexDeclaration.VertexDeclarationRHI;
	GraphicsPSOInit.BoundShaderState.VertexShaderRHI = VertexShader.GetVertexShader();
	GraphicsPSOInit.BoundShaderState.PixelShaderRHI  = PixelShader.GetPixelShader();

	SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit, 0);

	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);
	FIntRect ViewRect = ViewInfo.ViewRect;
	RHICmdList.SetViewport(
		ViewRect.Min.X, ViewRect.Min.Y, 0.0f,
		ViewRect.Max.X, ViewRect.Max.Y, 1.0f
	);

	// GlobalDrawIndirectArgsBuffer is in IndirectArgs state from DispatchPrefixSumVisibleCounts
	// (or retained from previous frame when bCanSkip is true)
	FGaussianSplatVS::FParameters VSParameters;
	VSParameters.ViewDataBuffer  = GlobalAccumulator->GlobalViewDataBufferSRV;
	VSParameters.SortKeysBuffer  = GlobalAccumulator->GlobalSortKeysBufferSRV;
	VSParameters.SplatCount      = GlobalAccumulator->AllocatedCount;  // Upper bound for VS guard
	VSParameters.DebugMode       = static_cast<uint32>(FMath::Max(0, DebugMode));
	VSParameters.EnableNanite    = 1;

	SetShaderParameters(RHICmdList, VertexShader, VertexShader.GetVertexShader(), VSParameters);

	FGaussianSplatPS::FParameters PSParameters;
	// Velocity calculation (Nanite-style): use previous frame's translated view-projection matrix
	PSParameters.PrevTranslatedWorldToClip = FMatrix44f(ViewInfo.PrevViewInfo.ViewMatrices.GetTranslatedViewProjectionMatrix());
	// Pass screen size inverse for NDC conversion (avoids View uniform buffer binding)
	PSParameters.ScreenSizeInverse = FVector2f(1.0f / ViewRect.Width(), 1.0f / ViewRect.Height());
	// TAA jitter: xy = current frame, zw = previous frame (in NDC space)
	PSParameters.TemporalAAJitter = FVector4f(
		View.ViewMatrices.GetTemporalAAJitter().X,
		View.ViewMatrices.GetTemporalAAJitter().Y,
		ViewInfo.PrevViewInfo.ViewMatrices.GetTemporalAAJitter().X,
		ViewInfo.PrevViewInfo.ViewMatrices.GetTemporalAAJitter().Y
	);
	// PreViewTranslation for converting TranslatedWorld back to World
	PSParameters.PreViewTranslation = FVector3f(View.ViewMatrices.GetPreViewTranslation());
	// Previous frame's PreViewTranslation for correct velocity calculation
	PSParameters.PrevPreViewTranslation = FVector3f(ViewInfo.PrevViewInfo.ViewMatrices.GetPreViewTranslation());
	SetShaderParameters(RHICmdList, PixelShader, PixelShader.GetPixelShader(), PSParameters);

	RHICmdList.SetStreamSource(0, nullptr, 0);
	RHICmdList.DrawIndexedPrimitiveIndirect(
		IndexBuffer,
		GlobalAccumulator->GlobalDrawIndirectArgsBuffer,
		0  // ArgumentOffset
	);
}

void FGaussianSplatRenderer::ExtractFrustumPlanes(const FMatrix& ViewProjection, FVector4f OutPlanes[6])
{
	// Extract frustum planes from view-projection matrix
	// Using the Gribb/Hartmann method
	// Each row of VP matrix gives us coefficients for a plane equation
	// Plane equation: dot(Normal, P) + D = 0

	const FMatrix& M = ViewProjection;

	// Left plane: Row3 + Row0
	OutPlanes[0] = FVector4f(
		M.M[0][3] + M.M[0][0],
		M.M[1][3] + M.M[1][0],
		M.M[2][3] + M.M[2][0],
		M.M[3][3] + M.M[3][0]
	);

	// Right plane: Row3 - Row0
	OutPlanes[1] = FVector4f(
		M.M[0][3] - M.M[0][0],
		M.M[1][3] - M.M[1][0],
		M.M[2][3] - M.M[2][0],
		M.M[3][3] - M.M[3][0]
	);

	// Bottom plane: Row3 + Row1
	OutPlanes[2] = FVector4f(
		M.M[0][3] + M.M[0][1],
		M.M[1][3] + M.M[1][1],
		M.M[2][3] + M.M[2][1],
		M.M[3][3] + M.M[3][1]
	);

	// Top plane: Row3 - Row1
	OutPlanes[3] = FVector4f(
		M.M[0][3] - M.M[0][1],
		M.M[1][3] - M.M[1][1],
		M.M[2][3] - M.M[2][1],
		M.M[3][3] - M.M[3][1]
	);

	// Near plane: Row3 + Row2 (for reversed-Z: Row2)
	OutPlanes[4] = FVector4f(
		M.M[0][2],
		M.M[1][2],
		M.M[2][2],
		M.M[3][2]
	);

	// Far plane: Row3 - Row2
	OutPlanes[5] = FVector4f(
		M.M[0][3] - M.M[0][2],
		M.M[1][3] - M.M[1][2],
		M.M[2][3] - M.M[2][2],
		M.M[3][3] - M.M[3][2]
	);

	// Normalize planes (important for distance calculations)
	for (int32 i = 0; i < 6; i++)
	{
		float Length = FMath::Sqrt(
			OutPlanes[i].X * OutPlanes[i].X +
			OutPlanes[i].Y * OutPlanes[i].Y +
			OutPlanes[i].Z * OutPlanes[i].Z
		);
		if (Length > SMALL_NUMBER)
		{
			OutPlanes[i] = OutPlanes[i] / Length;
		}
	}
}

int32 FGaussianSplatRenderer::DispatchClusterCulling(
	FRHICommandListImmediate& RHICmdList,
	const FSceneView& View,
	FGaussianSplatGPUResources* GPUResources,
	const FMatrix& LocalToWorld,
	float ErrorThreshold,
	bool bUseLODRendering)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatClusterCulling);

	if (!GPUResources->bHasClusterData || GPUResources->ClusterCount <= 0)
	{
		return 0;
	}

	TShaderMapRef<FClusterCullingResetCS> ResetShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FClusterCullingCS> CullingShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!ResetShader.IsValid() || !CullingShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("Cluster culling shaders not valid"));
		return 0;
	}

	// Calculate visibility bitmap size
	uint32 VisibilityBitmapSize = FMath::DivideAndRoundUp(GPUResources->ClusterCount, 32);
	VisibilityBitmapSize = FMath::Max(VisibilityBitmapSize, 1u);

	// Transition buffers for compute
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->VisibleClusterBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->VisibleClusterCountBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->ClusterVisibilityBitmap, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SelectedClusterBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	if (GPUResources->bSupportsIndirectDraw)
	{
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->IndirectDrawArgsBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	}

	// Transition LOD cluster tracking buffers (always needed for reset)
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODClusterBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODClusterCountBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODClusterSelectedBitmap, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODSplatTotalBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODSplatOutputCountBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	// Step 1: Reset the visible cluster counter, indirect draw args, visibility bitmap, selected cluster buffer, and LOD tracking buffers
	{
		FClusterCullingResetCS::FParameters ResetParams;
		ResetParams.VisibleClusterCountBuffer = GPUResources->VisibleClusterCountBufferUAV;
		ResetParams.IndirectDrawArgsBuffer = GPUResources->IndirectDrawArgsBufferUAV;
		ResetParams.ClusterVisibilityBitmap = GPUResources->ClusterVisibilityBitmapUAV;
		ResetParams.SelectedClusterBuffer = GPUResources->SelectedClusterBufferUAV;
		ResetParams.LODClusterBuffer = GPUResources->LODClusterBufferUAV;
		ResetParams.LODClusterCountBuffer = GPUResources->LODClusterCountBufferUAV;
		ResetParams.LODClusterSelectedBitmap = GPUResources->LODClusterSelectedBitmapUAV;
		ResetParams.LODSplatTotalBuffer = GPUResources->LODSplatTotalBufferUAV;
		ResetParams.LODSplatOutputCountBuffer = GPUResources->LODSplatOutputCountBufferUAV;
		ResetParams.ClusterVisibilityBitmapSize = VisibilityBitmapSize;
		ResetParams.LeafClusterCount = GPUResources->LeafClusterCount;
		ResetParams.DebugForceLODLevel = CVarDebugForceLODLevel.GetValueOnRenderThread();  // Must bind (shared .usf)

		// Dispatch enough threads to clear the bitmap and initialize selected cluster buffer
		uint32 NumGroups = FMath::DivideAndRoundUp(FMath::Max(VisibilityBitmapSize, (uint32)GPUResources->LeafClusterCount), 64u);

		SetComputePipelineState(RHICmdList, ResetShader.GetComputeShader());
		SetShaderParameters(RHICmdList, ResetShader, ResetShader.GetComputeShader(), ResetParams);
		RHICmdList.DispatchComputeShader(NumGroups, 1, 1);
		UnsetShaderUAVs(RHICmdList, ResetShader, ResetShader.GetComputeShader());
	}

	// UAV barrier
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->VisibleClusterCountBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->ClusterVisibilityBitmap, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SelectedClusterBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODClusterCountBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODClusterSelectedBitmap, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODSplatTotalBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

	// Step 2: Run cluster culling with LOD selection
	{
		// Extract frustum planes from world-space ViewProjection matrix
		// The shader transforms cluster bounds to world space, so frustum planes must also be in world space
		FVector4f FrustumPlanes[6];
		ExtractFrustumPlanes(View.ViewMatrices.GetViewProjectionMatrix(), FrustumPlanes);

		FClusterCullingCS::FParameters CullingParams;
		CullingParams.ClusterBuffer = GPUResources->ClusterBufferSRV;
		CullingParams.VisibleClusterBuffer = GPUResources->VisibleClusterBufferUAV;
		CullingParams.VisibleClusterCountBuffer = GPUResources->VisibleClusterCountBufferUAV;
		CullingParams.IndirectDrawArgsBuffer = GPUResources->IndirectDrawArgsBufferUAV;
		CullingParams.ClusterVisibilityBitmap = GPUResources->ClusterVisibilityBitmapUAV;
		CullingParams.SelectedClusterBuffer = GPUResources->SelectedClusterBufferUAV;
		CullingParams.LODClusterBuffer = GPUResources->LODClusterBufferUAV;
		CullingParams.LODClusterCountBuffer = GPUResources->LODClusterCountBufferUAV;
		CullingParams.LODClusterSelectedBitmap = GPUResources->LODClusterSelectedBitmapUAV;
		CullingParams.LODSplatTotalBuffer = GPUResources->LODSplatTotalBufferUAV;
		CullingParams.LocalToWorld = FMatrix44f(LocalToWorld);
		CullingParams.WorldToClip = FMatrix44f(View.ViewMatrices.GetViewProjectionMatrix());
		CullingParams.ClusterCount = GPUResources->ClusterCount;
		CullingParams.LeafClusterCount = GPUResources->LeafClusterCount;

		for (int32 i = 0; i < 6; i++)
		{
			CullingParams.FrustumPlanes[i] = FrustumPlanes[i];
		}

		// LOD selection parameters
		CullingParams.CameraPosition = FVector3f(View.ViewMatrices.GetViewOrigin());
		// Use projection matrix scaling factor for FOV-based LOD (resolution-independent, like Nanite)
		// ProjMatrix[1][1] = 1/tan(HalfFOV_Y), depends only on FOV, not viewport pixel size
		const FMatrix& ProjMatrix = View.ViewMatrices.GetProjectionMatrix();
		CullingParams.ScreenHeight = FMath::Max(ProjMatrix.M[0][0], ProjMatrix.M[1][1]);
		CullingParams.ErrorThreshold = FMath::Max(0.001f, ErrorThreshold);
		CullingParams.LODBias = 0.0f;         // No bias (can be made configurable)
		CullingParams.UseLODRendering = bUseLODRendering ? 1 : 0;
		// Debug: Force specific LOD level (-1 = auto, 0 = leaf, 1+ = specific level)
		CullingParams.DebugForceLODLevel = CVarDebugForceLODLevel.GetValueOnRenderThread();

		const uint32 ThreadGroupSize = 64;
		const uint32 NumGroups = FMath::DivideAndRoundUp((uint32)GPUResources->LeafClusterCount, ThreadGroupSize);

		SetComputePipelineState(RHICmdList, CullingShader.GetComputeShader());
		SetShaderParameters(RHICmdList, CullingShader, CullingShader.GetComputeShader(), CullingParams);
		RHICmdList.DispatchComputeShader(NumGroups, 1, 1);
		UnsetShaderUAVs(RHICmdList, CullingShader, CullingShader.GetComputeShader());
	}

	// Transition for potential readback (for debugging/statistics)
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->VisibleClusterCountBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->VisibleClusterBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));

	// Transition visibility bitmap and selected cluster buffer for CalcViewData to read
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->ClusterVisibilityBitmap, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SelectedClusterBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));

	// Transition LOD cluster buffers for LOD rendering pass
	if (bUseLODRendering)
	{
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODClusterBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODClusterCountBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODSplatTotalBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
		// Transition LODClusterSelectedBitmap to SRV for GPU-driven LOD shader to read
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->LODClusterSelectedBitmap, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
	}

	// Transition indirect draw args buffer for draw call
	if (GPUResources->bSupportsIndirectDraw)
	{
		RHICmdList.Transition(FRHITransitionInfo(GPUResources->IndirectDrawArgsBuffer, ERHIAccess::UAVCompute, ERHIAccess::IndirectArgs));
	}

	// Return leaf cluster count as placeholder (actual count would require GPU readback)
	// In a real implementation, you'd use the VisibleClusterBuffer in subsequent passes
	return GPUResources->LeafClusterCount;
}

void FGaussianSplatRenderer::CompositeToSceneColor(
	FRHICommandListImmediate& RHICmdList,
	const FSceneView& View,
	FTextureRHIRef IntermediateTexture)
{
	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatComposite);

	TShaderMapRef<FGaussianSplatCompositeVS> VertexShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FGaussianSplatCompositePS> PixelShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!VertexShader.IsValid() || !PixelShader.IsValid())
	{
		return;
	}

	FGraphicsPipelineStateInitializer GraphicsPSOInit;
	RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);

	GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
	// No depth testing for full-screen composite
	GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<false, CF_Always>::GetRHI();
	// Premultiplied alpha blend onto SceneColor
	// CW_RGB preserves SceneColor alpha for Movie Render Queue export
	GraphicsPSOInit.BlendState = TStaticBlendState<
		CW_RGB, BO_Add, BF_One, BF_InverseSourceAlpha, BO_Add, BF_One, BF_InverseSourceAlpha
	>::GetRHI();
	GraphicsPSOInit.PrimitiveType = PT_TriangleList;
	GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GEmptyVertexDeclaration.VertexDeclarationRHI;
	GraphicsPSOInit.BoundShaderState.VertexShaderRHI = VertexShader.GetVertexShader();
	GraphicsPSOInit.BoundShaderState.PixelShaderRHI = PixelShader.GetPixelShader();

	SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit, 0);

	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);
	FIntRect ViewRect = ViewInfo.ViewRect;
	RHICmdList.SetViewport(
		ViewRect.Min.X, ViewRect.Min.Y, 0.0f,
		ViewRect.Max.X, ViewRect.Max.Y, 1.0f
	);

	// Set pixel shader parameters
	FGaussianSplatCompositePS::FParameters PSParameters;
	PSParameters.IntermediateTexture = IntermediateTexture;
	PSParameters.IntermediateSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	SetShaderParameters(RHICmdList, PixelShader, PixelShader.GetPixelShader(), PSParameters);

	// Draw full-screen triangle (3 vertices, no index buffer)
	RHICmdList.SetStreamSource(0, nullptr, 0);
	RHICmdList.DrawPrimitive(0, 1, 1);
}
