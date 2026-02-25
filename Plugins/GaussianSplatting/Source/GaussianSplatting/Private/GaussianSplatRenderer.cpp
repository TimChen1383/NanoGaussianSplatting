// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatRenderer.h"
#include "GaussianSplatShaders.h"
#include "GaussianSplatSceneProxy.h"
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
extern TAutoConsoleVariable<float> CVarLODErrorThreshold;
extern TAutoConsoleVariable<int32> CVarUseLODRendering;
extern TAutoConsoleVariable<int32> CVarDebugForceLODLevel;
extern TAutoConsoleVariable<int32> CVarUseCompaction;

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

//Trigger all dispatch functions
void FGaussianSplatRenderer::Render(
	FRHICommandListImmediate& RHICmdList,
	const FSceneView& View,
	FGaussianSplatGPUResources* GPUResources,
	const FMatrix& LocalToWorld,
	int32 SplatCount,
	int32 SHOrder,
	float OpacityScale,
	float SplatScale)
{
	if (!GPUResources || !GPUResources->IsValid() || SplatCount <= 0)
	{
		return;
	}

	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatRendering);

	// Check if we have a valid ColorTexture for CalcViewData
	bool bHasColorTexture = GPUResources->ColorTextureSRV.IsValid();

	// Camera-static sort skipping: skip entire compute pipeline when nothing has changed
	FMatrix CurrentVP = View.ViewMatrices.GetViewProjectionMatrix();
	float CurrentErrorThreshold = FMath::Max(0.1f, CVarLODErrorThreshold.GetValueOnRenderThread());
	int32 CurrentDebugMode = CVarShowClusterBounds.GetValueOnRenderThread();
	int32 CurrentDebugForceLODLevel = CVarDebugForceLODLevel.GetValueOnRenderThread();
	bool bCanSkipCompute = GPUResources->bHasCachedSortData &&
		GPUResources->CachedViewProjectionMatrix.Equals(CurrentVP, 0.0f) &&
		GPUResources->CachedLocalToWorld.Equals(LocalToWorld, 0.0f) &&
		GPUResources->CachedOpacityScale == OpacityScale &&
		GPUResources->CachedSplatScale == SplatScale &&
		GPUResources->CachedHasColorTexture == bHasColorTexture &&
		GPUResources->CachedErrorThreshold == CurrentErrorThreshold &&
		GPUResources->CachedDebugMode == CurrentDebugMode &&
		GPUResources->CachedDebugForceLODLevel == CurrentDebugForceLODLevel;

	// Check if LOD rendering is enabled
	// GPU-driven LOD rendering: no CPU readback, all processing done on GPU
	bool bUseLODRendering = CVarUseLODRendering.GetValueOnRenderThread() != 0 && GPUResources->bHasLODSplats;

	// Check if splat compaction is enabled and supported
	bool bUseCompaction = CVarUseCompaction.GetValueOnRenderThread() != 0 &&
		GPUResources->bSupportsCompaction &&
		GPUResources->bHasClusterData;

	if (!bCanSkipCompute)
	{
		// Step 0: Cluster culling (Nanite-style optimization)
		// Also performs LOD selection and tracks unique LOD clusters
		if (GPUResources->bHasClusterData)
		{
			int32 VisibleClusters = DispatchClusterCulling(RHICmdList, View, GPUResources, LocalToWorld, bUseLODRendering);
			// Uncomment for debug logging:
			// UE_LOG(LogTemp, Log, TEXT("Cluster culling: %d / %d clusters visible"), VisibleClusters, GPUResources->LeafClusterCount);
		}

		if (bUseCompaction)
		{
			// COMPACTION PIPELINE: Only process visible splats
			// This provides significant performance gains when many splats are culled

			int32 OriginalSplatCount = SplatCount - GPUResources->LODSplatCount;

			// Step 1: Compact visible splats
			DispatchCompactSplats(RHICmdList, GPUResources, SplatCount, OriginalSplatCount, bUseLODRendering);

			// Step 2: Prepare indirect args from visible count
			DispatchPrepareIndirectArgs(RHICmdList, GPUResources);

			// Step 3: Calculate view data (indirect dispatch - only visible splats)
			DispatchCalcViewDataCompacted(RHICmdList, View, GPUResources, LocalToWorld, SplatCount, OriginalSplatCount, SHOrder, OpacityScale, SplatScale, bHasColorTexture);

			// Step 4: Calculate sort distances (indirect dispatch - only visible splats)
			DispatchCalcDistancesIndirect(RHICmdList, GPUResources);

			// Step 5: Sort splats back-to-front
			// Note: Sort still operates on max buffer size, but invalid entries sort to end
			DispatchRadixSort(RHICmdList, GPUResources, SplatCount);
		}
		else
		{
			// ORIGINAL PIPELINE: Process all splats

			// Step 1: Calculate view data for each splat (UNIFIED APPROACH)
			// Processes both original and LOD splats in the same buffer
			// Original splats: check ClusterVisibilityBitmap + LOD skip logic
			// LOD splats: check LODClusterSelectedBitmap
			DispatchCalcViewData(RHICmdList, View, GPUResources, LocalToWorld, SplatCount, SHOrder, OpacityScale, SplatScale, bHasColorTexture, bUseLODRendering);

			// NOTE: Separate LOD pass removed - unified approach handles LOD splats in CalcViewData

			// Step 2: Calculate sort distances
			// SplatCount now includes both original and LOD splats (unified buffer)
			int32 SortCount = SplatCount;
			DispatchCalcDistances(RHICmdList, GPUResources, SortCount);

			// Step 3: Sort splats back-to-front
			DispatchRadixSort(RHICmdList, GPUResources, SortCount);
		}

		// Update cache
		GPUResources->CachedViewProjectionMatrix = CurrentVP;
		GPUResources->CachedLocalToWorld = LocalToWorld;
		GPUResources->CachedOpacityScale = OpacityScale;
		GPUResources->CachedSplatScale = SplatScale;
		GPUResources->CachedHasColorTexture = bHasColorTexture;
		GPUResources->CachedErrorThreshold = CurrentErrorThreshold;
		GPUResources->CachedDebugMode = CurrentDebugMode;
		GPUResources->CachedDebugForceLODLevel = CurrentDebugForceLODLevel;
		GPUResources->bHasCachedSortData = true;
	}

	// Step 4: Draw the splats (always — uses cached buffers when compute is skipped)
	DrawSplats(RHICmdList, View, GPUResources, SplatCount);
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
	bool bHasColorTexture,
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
	Parameters.PositionBuffer = GPUResources->PositionBufferSRV;
	Parameters.OtherDataBuffer = GPUResources->OtherDataBufferSRV;
	Parameters.SHBuffer = GPUResources->SHBufferSRV;
	Parameters.ChunkBuffer = GPUResources->ChunkBufferSRV;
	Parameters.ColorTexture = GPUResources->GetColorTextureSRVOrDummy();  // Uses dummy texture if real one not available
	Parameters.ColorSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	Parameters.ViewDataBuffer = GPUResources->ViewDataBufferUAV;

	// Cluster visibility integration (UNIFIED APPROACH)
	// SplatClusterIndexBuffer maps ALL splats (original + LOD) to their clusters
	if (GPUResources->bHasClusterData && GPUResources->SplatClusterIndexBufferSRV.IsValid())
	{
		Parameters.SplatClusterIndexBuffer = GPUResources->SplatClusterIndexBufferSRV;
		Parameters.ClusterVisibilityBitmap = GPUResources->ClusterVisibilityBitmapSRV;
		Parameters.LODClusterSelectedBitmap = GPUResources->LODClusterSelectedBitmapSRV;
		Parameters.SelectedClusterBuffer = GPUResources->SelectedClusterBufferSRV;
		Parameters.UseClusterCulling = 1;
		Parameters.UseLODRendering = bUseLODRendering ? 1 : 0;
		// OriginalSplatCount = TotalSplatCount - LODSplatCount
		Parameters.OriginalSplatCount = SplatCount - GPUResources->LODSplatCount;
	}
	else
	{
		// No cluster data - process all splats
		Parameters.SplatClusterIndexBuffer = nullptr;
		Parameters.ClusterVisibilityBitmap = nullptr;
		Parameters.LODClusterSelectedBitmap = nullptr;
		Parameters.SelectedClusterBuffer = nullptr;
		Parameters.UseClusterCulling = 0;
		Parameters.UseLODRendering = 0;
		Parameters.OriginalSplatCount = SplatCount;
	}

	// Matrices
	Parameters.LocalToWorld = FMatrix44f(LocalToWorld);
	Parameters.WorldToClip = FMatrix44f(View.ViewMatrices.GetViewProjectionMatrix());
	Parameters.WorldToView = FMatrix44f(View.ViewMatrices.GetViewMatrix());
	Parameters.CameraPosition = FVector3f(View.ViewMatrices.GetViewOrigin());

	// Screen info - use FViewInfo::ViewRect which accounts for screen percentage
	// This is the actual render target viewport, scaled when screen percentage < 100%
	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);
	FIntRect ViewRect = ViewInfo.ViewRect;
	Parameters.ScreenSize = FVector2f(ViewRect.Width(), ViewRect.Height());

	// Focal length approximation from projection matrix
	const FMatrix& ProjMatrix = View.ViewMatrices.GetProjectionMatrix();
	Parameters.FocalLength = FVector2f(
		ProjMatrix.M[0][0] * ViewRect.Width() * 0.5f,
		ProjMatrix.M[1][1] * ViewRect.Height() * 0.5f
	);

	Parameters.SplatCount = SplatCount;
	Parameters.SHOrder = SHOrder;
	Parameters.OpacityScale = OpacityScale;
	Parameters.SplatScale = SplatScale;
	Parameters.ColorTextureSize = FIntPoint(GaussianSplattingConstants::ColorTextureWidth,
		FMath::DivideAndRoundUp(SplatCount, GaussianSplattingConstants::ColorTextureWidth));
	Parameters.PositionFormat = 0;  // Always Float32 (simplified format)
	Parameters.UseDefaultColor = bHasColorTexture ? 0 : 1;  // Use default color if no texture

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
			Params.RadixShift = RadixShift;
			Params.Count = SortCount;
			Params.NumTiles = NumTiles;

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
			Params.NumTiles = NumTiles;

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
			Params.RadixShift = RadixShift;
			Params.Count = SortCount;
			Params.NumTiles = NumTiles;

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
	GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<false, CF_DepthNearOrEqual>::GetRHI();

	// Blend mode: Standard premultiplied alpha "over" for back-to-front compositing
	// result.rgb = src.rgb + dst.rgb * (1 - srcAlpha)
	// This properly attenuates the background behind splats
	// NOTE: Using CW_RGB (not CW_RGBA) to preserve destination alpha.
	// This fixes Movie Render Queue PNG export showing transparent splats.
	// The scene buffer starts with alpha=1, and we don't want splats to reduce it.
	GraphicsPSOInit.BlendState = TStaticBlendState<
		CW_RGB,
		BO_Add, BF_One, BF_InverseSourceAlpha   // Color: Src + Dst * (1 - SrcAlpha)
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

	SetShaderParameters(RHICmdList, VertexShader, VertexShader.GetVertexShader(), VSParameters);

	// Pixel shader parameters
	FGaussianSplatPS::FParameters PSParameters;
	SetShaderParameters(RHICmdList, PixelShader, PixelShader.GetPixelShader(), PSParameters);

	// Draw instanced quads
	// 4 vertices per quad, N instances (either from SplatCount or indirect args)
	// Using index buffer: 6 indices per quad (2 triangles)
	RHICmdList.SetStreamSource(0, nullptr, 0);

	// Use indirect draw when cluster culling is available
	// This allows the GPU to determine how many instances to draw
	if (GPUResources->bSupportsIndirectDraw && GPUResources->bHasClusterData)
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
		// Fallback: CPU-driven rendering with fixed splat count
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

	FPrepareIndirectArgsCS::FParameters Parameters;
	Parameters.VisibleSplatCount = GPUResources->VisibleSplatCountBufferSRV;
	Parameters.IndirectDispatchArgs = GPUResources->IndirectDispatchArgsBufferUAV;
	Parameters.IndirectDrawArgs = GPUResources->IndirectDrawArgsBufferUAV;
	Parameters.ComputeThreadGroupSize = 256;

	SetComputePipelineState(RHICmdList, ComputeShader.GetComputeShader());
	SetShaderParameters(RHICmdList, ComputeShader, ComputeShader.GetComputeShader(), Parameters);
	RHICmdList.DispatchComputeShader(1, 1, 1);
	UnsetShaderUAVs(RHICmdList, ComputeShader, ComputeShader.GetComputeShader());

	// Transition for indirect dispatch/draw
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->IndirectDispatchArgsBuffer, ERHIAccess::UAVCompute, ERHIAccess::IndirectArgs));
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->IndirectDrawArgsBuffer, ERHIAccess::UAVCompute, ERHIAccess::IndirectArgs));
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
	float SplatScale,
	bool bHasColorTexture)
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
	Parameters.PositionBuffer = GPUResources->PositionBufferSRV;
	Parameters.OtherDataBuffer = GPUResources->OtherDataBufferSRV;
	Parameters.SHBuffer = GPUResources->SHBufferSRV;
	Parameters.ChunkBuffer = GPUResources->ChunkBufferSRV;
	Parameters.ColorTexture = GPUResources->GetColorTextureSRVOrDummy();
	Parameters.ColorSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	Parameters.ViewDataBuffer = GPUResources->ViewDataBufferUAV;

	// Cluster visibility data (needed for debug visualization)
	Parameters.SplatClusterIndexBuffer = GPUResources->SplatClusterIndexBufferSRV;
	Parameters.ClusterVisibilityBitmap = GPUResources->ClusterVisibilityBitmapSRV;
	Parameters.LODClusterSelectedBitmap = GPUResources->LODClusterSelectedBitmapSRV;
	Parameters.SelectedClusterBuffer = GPUResources->SelectedClusterBufferSRV;
	Parameters.UseClusterCulling = 1;
	Parameters.UseLODRendering = 0;  // Not needed in compaction mode
	Parameters.OriginalSplatCount = OriginalSplatCount;

	// COMPACTION MODE: Use compacted indices
	Parameters.CompactedSplatIndices = GPUResources->CompactedSplatIndicesBufferSRV;
	Parameters.UseCompaction = 1;
	Parameters.VisibleSplatCount = SplatCount;  // Will be read from buffer by shader

	// Matrices
	Parameters.LocalToWorld = FMatrix44f(LocalToWorld);
	Parameters.WorldToClip = FMatrix44f(View.ViewMatrices.GetViewProjectionMatrix());
	Parameters.WorldToView = FMatrix44f(View.ViewMatrices.GetViewMatrix());
	Parameters.CameraPosition = FVector3f(View.ViewMatrices.GetViewOrigin());

	// Screen info - use FViewInfo::ViewRect which accounts for screen percentage
	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);
	FIntRect ViewRect = ViewInfo.ViewRect;
	Parameters.ScreenSize = FVector2f(ViewRect.Width(), ViewRect.Height());

	// Focal length
	const FMatrix& ProjMatrix = View.ViewMatrices.GetProjectionMatrix();
	Parameters.FocalLength = FVector2f(
		ProjMatrix.M[0][0] * ViewRect.Width() * 0.5f,
		ProjMatrix.M[1][1] * ViewRect.Height() * 0.5f
	);

	Parameters.SplatCount = SplatCount;  // Total splat count for bounds checking
	Parameters.SHOrder = SHOrder;
	Parameters.OpacityScale = OpacityScale;
	Parameters.SplatScale = SplatScale;
	Parameters.ColorTextureSize = FIntPoint(GaussianSplattingConstants::ColorTextureWidth,
		FMath::DivideAndRoundUp(SplatCount, GaussianSplattingConstants::ColorTextureWidth));
	Parameters.PositionFormat = 0;
	Parameters.UseDefaultColor = bHasColorTexture ? 0 : 1;

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

	// CRITICAL FIX: Clear distance buffer to 0xFFFFFFFF before indirect dispatch.
	// In compaction mode, CalcDistancesIndirect only writes to elements 0..VisibleSplatCount-1,
	// but RadixSort sorts ALL TotalSplatCount elements. Without this clear, uninitialized
	// elements (VisibleSplatCount..TotalSplatCount-1) contain garbage distances that can
	// sort into the first VisibleSplatCount positions, causing ghost noise and flickering.
	// Setting unused elements to 0xFFFFFFFF (max uint) ensures they sort to the END
	// (back-to-front uses ascending sort), where DrawSplats never reads them.
	RHICmdList.ClearUAVUint(GPUResources->SortDistanceBufferUAV, FUintVector4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));

	// UAV barrier after clear to ensure clear completes before compute shader writes
	RHICmdList.Transition(FRHITransitionInfo(GPUResources->SortDistanceBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

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
		// Use FViewInfo::ViewRect for screen-based LOD calculations (accounts for screen percentage)
		const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);
		FIntRect ViewRect = ViewInfo.ViewRect;
		CullingParams.ScreenHeight = static_cast<float>(ViewRect.Height());
		CullingParams.ErrorThreshold = FMath::Max(0.1f, CVarLODErrorThreshold.GetValueOnRenderThread());
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
