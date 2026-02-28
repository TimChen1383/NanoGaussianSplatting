// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatting.h"
#include "GaussianSplatViewExtension.h"
#include "GaussianSplatRenderer.h"
#include "GaussianSplatSceneProxy.h"
#include "GaussianGlobalAccumulator.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "ShaderCore.h"
#include "SceneViewExtension.h"
#include "Misc/CoreDelegates.h"
#include "Engine/Engine.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "SceneView.h"
#include "ScreenPass.h"

#define LOCTEXT_NAMESPACE "FGaussianSplattingModule"

//----------------------------------------------------------------------
// Console Variables for Gaussian Splatting
//----------------------------------------------------------------------

/** Show cluster debug visualization (Nanite-style coloring) */
TAutoConsoleVariable<int32> CVarShowClusterBounds(
	TEXT("gs.ShowClusterBounds"),
	0,
	TEXT("Debug visualization for Gaussian Splat clusters (Nanite-style).\n")
	TEXT("When enabled, shows cluster colors on black background (like Nanite debug view).\n")
	TEXT(" 0: Off (default)\n")
	TEXT(" 1: Show cluster colors (each cluster gets a unique random color)"),
	ECVF_RenderThreadSafe);

/** LOD error threshold in pixels - controls when LOD kicks in */
TAutoConsoleVariable<float> CVarLODErrorThreshold(
	TEXT("gs.LODErrorThreshold"),
	32.0f,
	TEXT("Screen-space error threshold in pixels for LOD selection.\n")
	TEXT("Lower values = more conservative (keep detail longer, less LOD savings)\n")
	TEXT("Higher values = more aggressive (switch to LOD sooner, more savings)\n")
	TEXT("Default: 32.0"),
	ECVF_RenderThreadSafe);

/** Enable/disable global accumulator (one draw call for all proxies). Default ON. */
TAutoConsoleVariable<int32> CVarUseGlobalAccumulator(
	TEXT("gs.UseGlobalAccumulator"),
	1,
	TEXT("Enable the global accumulator for one-draw-call rendering of all Gaussian Splat proxies.\n")
	TEXT(" 1: Global accumulator (single sort + single draw for all proxies, correct cross-tile blending)\n")
	TEXT(" 0: Legacy per-proxy rendering (N sorts + N draws, kept for debugging/regression)"),
	ECVF_RenderThreadSafe);

/** Debug: Force a specific LOD level for debugging LOD hierarchy */
TAutoConsoleVariable<int32> CVarDebugForceLODLevel(
	TEXT("gs.DebugForceLODLevel"),
	-1,
	TEXT("Force rendering of a specific LOD level for debugging (only affects Nanite-enabled assets).\n")
	TEXT("This ignores normal LOD selection and forces all clusters to use specified level.\n")
	TEXT(" -1: Auto - normal LOD selection based on distance/error (default)\n")
	TEXT("  0: Force leaf clusters only - render original splats (finest detail)\n")
	TEXT("  1+: Force specific LOD level (1 = first parent level, 2 = second, etc.)\n")
	TEXT("Note: Higher levels have fewer, coarser splats. Max level depends on asset size.\n")
	TEXT("Use with gs.ShowClusterBounds 2 to visualize which LOD level is being rendered."),
	ECVF_RenderThreadSafe);

// Export for other modules
int32 GGaussianSplatShowClusterBounds = 0;

// Helper to get the renderer module
static IRendererModule& GetRendererModuleRef()
{
	return FModuleManager::GetModuleChecked<IRendererModule>("Renderer");
}

void FGaussianSplattingModule::StartupModule()
{
	// Register the shader directory so we can use our custom shaders
	FString PluginShaderDir = FPaths::Combine(IPluginManager::Get().FindPlugin(TEXT("GaussianSplatting"))->GetBaseDir(), TEXT("Shaders"));
	AddShaderSourceDirectoryMapping(TEXT("/Plugin/GaussianSplatting"), PluginShaderDir);

	// Create and register the view extension for Gaussian Splat rendering
	ViewExtension = FSceneViewExtensions::NewExtension<FGaussianSplatViewExtension>();

	if (!ViewExtension.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("GaussianSplatting: Failed to create ViewExtension!"));
	}

	// Allocate the global accumulator (buffers are created lazily on first render)
	GlobalAccumulator = MakeUnique<FGaussianGlobalAccumulator>();

	// Register post-opaque render delegate for rendering
	PostOpaqueRenderDelegateHandle = GetRendererModuleRef().RegisterPostOpaqueRenderDelegate(
		FPostOpaqueRenderDelegate::CreateRaw(this, &FGaussianSplattingModule::OnPostOpaqueRender_RenderThread));

	UE_LOG(LogTemp, Log, TEXT("GaussianSplatting module started. Shader directory: %s"), *PluginShaderDir);
}

void FGaussianSplattingModule::OnPostOpaqueRender_RenderThread(FPostOpaqueRenderParameters& Parameters)
{
	FGaussianSplatViewExtension* Ext = FGaussianSplatViewExtension::Get();
	if (!Ext || !Parameters.GraphBuilder || !Parameters.View)
	{
		return;
	}

	FRDGBuilder& GraphBuilder = *Parameters.GraphBuilder;
	const FSceneView* SceneView = reinterpret_cast<const FSceneView*>(Parameters.View);

	TArray<FGaussianSplatSceneProxy*> Proxies;
	Ext->GetRegisteredProxies(Proxies);

	if (Proxies.Num() == 0)
	{
		return;
	}

	// Sort proxies back-to-front for correct depth ordering
	if (Proxies.Num() > 1)
	{
		FVector CameraPosition = SceneView->ViewMatrices.GetViewOrigin();
		Proxies.Sort([CameraPosition](const FGaussianSplatSceneProxy& A, const FGaussianSplatSceneProxy& B)
		{
			float DistA = FVector::DistSquared(CameraPosition, A.GetBounds().Origin);
			float DistB = FVector::DistSquared(CameraPosition, B.GetBounds().Origin);
			return DistA > DistB;
		});
	}

	FRDGTexture* ColorTexture = Parameters.ColorTexture;
	FRDGTexture* DepthTexture = Parameters.DepthTexture;
	if (!ColorTexture)
	{
		return;
	}

	int32 DebugMode = CVarShowClusterBounds.GetValueOnRenderThread();
	ERenderTargetLoadAction ColorLoadAction = (DebugMode > 0) ? ERenderTargetLoadAction::EClear : ERenderTargetLoadAction::ELoad;

	FRenderTargetParameters* PassParameters = GraphBuilder.AllocParameters<FRenderTargetParameters>();
	PassParameters->RenderTargets[0] = FRenderTargetBinding(ColorTexture, ColorLoadAction);
	if (DepthTexture)
	{
		PassParameters->RenderTargets.DepthStencil = FDepthStencilBinding(
			DepthTexture,
			ERenderTargetLoadAction::ELoad,
			ERenderTargetLoadAction::ELoad,
			FExclusiveDepthStencil::DepthRead_StencilRead
		);
	}

	const bool bUseGlobalAccumulator = (CVarUseGlobalAccumulator.GetValueOnRenderThread() != 0);

	if (bUseGlobalAccumulator && GlobalAccumulator.IsValid())
	{
		//------------------------------------------------------------------
		// GLOBAL ACCUMULATOR PATH: Phase 1 (per-proxy CalcViewData) +
		// Phase 2 (single CalcDistances + RadixSort) + single DrawSplats
		//------------------------------------------------------------------

		// Build the list of visible proxies and compute total splat count (CPU-side)
		struct FProxyRenderInfo
		{
			FGaussianSplatSceneProxy* Proxy;
			FMatrix LocalToWorld;
			uint32 GlobalBaseOffset;
			bool bUseLODRendering;
		};
		TArray<FProxyRenderInfo> VisibleProxies;
		uint32 TotalSplatCount = 0;
		bool bAllNanite = true;  // True if every visible proxy supports compaction

		for (FGaussianSplatSceneProxy* Proxy : Proxies)
		{
			if (!Proxy) continue;
			if (&Proxy->GetScene() != SceneView->Family->Scene) continue;
			if (!Proxy->IsShown(SceneView)) continue;

			const FBoxSphereBounds& Bounds = Proxy->GetBounds();
			if (!SceneView->ViewFrustum.IntersectBox(Bounds.Origin, Bounds.BoxExtent)) continue;

			FGaussianSplatGPUResources* GPUResources = Proxy->GetGPUResources();
			if (!GPUResources || !GPUResources->IsValid()) continue;

			FProxyRenderInfo Info;
			Info.Proxy = Proxy;
			Info.LocalToWorld = Proxy->GetLocalToWorld();
			Info.GlobalBaseOffset = TotalSplatCount;
			Info.bUseLODRendering = GPUResources->bEnableNanite && GPUResources->bHasLODSplats;
			VisibleProxies.Add(Info);
			TotalSplatCount += (uint32)Proxy->GetSplatCount();

			// All proxies must support Nanite compaction for the fast global path
			if (!GPUResources->bEnableNanite || !GPUResources->bHasClusterData || !GPUResources->bSupportsCompaction)
			{
				bAllNanite = false;
			}
		}

		// Safety: global accumulator only supports up to MAX_PROXY_COUNT proxies
		if ((uint32)VisibleProxies.Num() > FGaussianGlobalAccumulator::MAX_PROXY_COUNT)
		{
			bAllNanite = false;
		}

		if (TotalSplatCount == 0)
		{
			return;
		}

		// Check camera-static skip: if nothing has changed, skip Phase 1+2 and reuse cached sort
		FMatrix CurrentVP = SceneView->ViewMatrices.GetViewProjectionMatrix();
		float CurrentErrorThreshold = FMath::Max(0.1f, CVarLODErrorThreshold.GetValueOnRenderThread());
		int32 CurrentDebugMode = DebugMode;
		int32 CurrentDebugForceLODLevel = CVarDebugForceLODLevel.GetValueOnRenderThread();

		bool bCanSkip = GlobalAccumulator->bHasCachedSortData &&
			GlobalAccumulator->CachedTotalSplatCount == TotalSplatCount &&
			GlobalAccumulator->CachedViewProjectionMatrix.Equals(CurrentVP, 0.0f);

		if (bCanSkip)
		{
			for (const FProxyRenderInfo& Info : VisibleProxies)
			{
				FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
				if (!GPUResources->bHasCachedSortData ||
					!GPUResources->CachedViewProjectionMatrix.Equals(CurrentVP, 0.0f) ||
					!GPUResources->CachedLocalToWorld.Equals(Info.LocalToWorld, 0.0f) ||
					GPUResources->CachedOpacityScale != Info.Proxy->GetOpacityScale() ||
					GPUResources->CachedSplatScale != Info.Proxy->GetSplatScale() ||
					GPUResources->CachedErrorThreshold != CurrentErrorThreshold ||
					GPUResources->CachedDebugMode != CurrentDebugMode ||
					GPUResources->CachedDebugForceLODLevel != CurrentDebugForceLODLevel)
				{
					bCanSkip = false;
					break;
				}
			}
		}

		// Grab index buffer from the first proxy (all proxies use identical quad geometry)
		FBufferRHIRef SharedIndexBuffer;
		if (VisibleProxies.Num() > 0)
		{
			FGaussianSplatGPUResources* FirstRes = VisibleProxies[0].Proxy->GetGPUResources();
			SharedIndexBuffer = FirstRes ? FirstRes->IndexBuffer : FBufferRHIRef();
		}

		FGaussianGlobalAccumulator* RawAccumulator = GlobalAccumulator.Get();

		GraphBuilder.AddPass(
			RDG_EVENT_NAME("GaussianSplatRendering_Global"),
			PassParameters,
			ERDGPassFlags::Raster,
			[SceneView, VisibleProxies, TotalSplatCount, bCanSkip, bAllNanite, RawAccumulator,
			 SharedIndexBuffer, CurrentVP, CurrentErrorThreshold, CurrentDebugMode,
			 CurrentDebugForceLODLevel, DebugMode](FRHICommandListImmediate& RHICmdList)
			{
				if (!SceneView) return;
				SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatRendering_Global);

				// Initialize color textures (deferred init)
				for (const auto& Info : VisibleProxies)
				{
					Info.Proxy->TryInitializeColorTexture(RHICmdList);
				}

				// Ensure global buffers are large enough for all splats
				RawAccumulator->ResizeIfNeeded(RHICmdList, TotalSplatCount);

				if (bAllNanite)
				{
					//==================================================
					// GLOBAL + COMPACTION PATH
					// All proxies are Nanite-enabled: GPU compaction
					// reduces working set from TotalSplatCount → TotalVisible
					// (~140x reduction at LOD5 for a 719K-splat tile).
					//==================================================

					// Ensure fixed-size prefix-sum buffers exist (allocated once)
					RawAccumulator->EnsureCompactionBuffersAllocated(RHICmdList);

					if (!bCanSkip)
					{
						// --------------------------------------------------
						// Phase 0: Per-proxy culling + compaction + indirect args
						// --------------------------------------------------
						for (const auto& Info : VisibleProxies)
						{
							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							int32 SplatCount = Info.Proxy->GetSplatCount();
							int32 OriginalSplatCount = SplatCount - GPUResources->LODSplatCount;

							// Cluster culling → fills ClusterVisibilityBitmap
							FGaussianSplatRenderer::DispatchClusterCulling(
								RHICmdList, *SceneView, GPUResources,
								Info.LocalToWorld, Info.bUseLODRendering);

							// Compact → fills CompactedSplatIndices + VisibleSplatCountBuffer
							FGaussianSplatRenderer::DispatchCompactSplats(
								RHICmdList, GPUResources,
								SplatCount, OriginalSplatCount, Info.bUseLODRendering);

							// PrepareIndirectArgs → fills IndirectDispatchArgsBuffer for CalcViewData
							FGaussianSplatRenderer::DispatchPrepareIndirectArgs(RHICmdList, GPUResources);
						}

						// --------------------------------------------------
						// Phase 1: Gather visible counts + GPU prefix sum
						// --------------------------------------------------
						int32 NumProxies = VisibleProxies.Num();
						for (int32 i = 0; i < NumProxies; i++)
						{
							FGaussianSplatGPUResources* GPUResources = VisibleProxies[i].Proxy->GetGPUResources();
							FGaussianSplatRenderer::DispatchGatherVisibleCount(
								RHICmdList, GPUResources, RawAccumulator, i);
						}

						// Single 1-thread dispatch: computes prefix sums + writes all indirect args
						FGaussianSplatRenderer::DispatchPrefixSumVisibleCounts(
							RHICmdList, RawAccumulator, NumProxies);

						// --------------------------------------------------
						// Phase 2: Per-proxy CalcViewData → global buffer
						// (indirect dispatch, only visible splats per proxy)
						// --------------------------------------------------
						for (int32 i = 0; i < NumProxies; i++)
						{
							const auto& Info = VisibleProxies[i];
							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							int32 SplatCount = Info.Proxy->GetSplatCount();
							int32 OriginalSplatCount = SplatCount - GPUResources->LODSplatCount;
							bool bHasColorTexture = GPUResources->ColorTextureSRV.IsValid();

							FGaussianSplatRenderer::DispatchCalcViewDataCompactedGlobal(
								RHICmdList, *SceneView, GPUResources,
								Info.LocalToWorld,
								SplatCount,
								OriginalSplatCount,
								Info.Proxy->GetSHOrder(),
								Info.Proxy->GetOpacityScale(),
								Info.Proxy->GetSplatScale(),
								bHasColorTexture,
								i,
								RawAccumulator);
						}

						// --------------------------------------------------
						// Phase 3: Single global CalcDistances + RadixSort
						// (all indirect — count driven by GPU prefix sum)
						// --------------------------------------------------
						FGaussianSplatRenderer::DispatchCalcDistancesGlobalIndirect(RHICmdList, RawAccumulator);
						FGaussianSplatRenderer::DispatchRadixSortGlobalIndirect(RHICmdList, RawAccumulator);

						// Update caches
						RawAccumulator->bHasCachedSortData = true;
						RawAccumulator->CachedTotalSplatCount = TotalSplatCount;
						RawAccumulator->CachedViewProjectionMatrix = CurrentVP;

						for (const auto& Info : VisibleProxies)
						{
							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							GPUResources->CachedViewProjectionMatrix = CurrentVP;
							GPUResources->CachedLocalToWorld = Info.LocalToWorld;
							GPUResources->CachedOpacityScale = Info.Proxy->GetOpacityScale();
							GPUResources->CachedSplatScale = Info.Proxy->GetSplatScale();
							GPUResources->CachedHasColorTexture = GPUResources->ColorTextureSRV.IsValid();
							GPUResources->CachedErrorThreshold = CurrentErrorThreshold;
							GPUResources->CachedDebugMode = CurrentDebugMode;
							GPUResources->CachedDebugForceLODLevel = CurrentDebugForceLODLevel;
							GPUResources->bHasCachedSortData = true;
						}
					}

					// Single draw call — instance count from GlobalDrawIndirectArgsBuffer
					FGaussianSplatRenderer::DrawSplatsGlobalIndirect(
						RHICmdList, *SceneView, RawAccumulator, SharedIndexBuffer, DebugMode);
				}
				else
				{
					//==================================================
					// NON-COMPACTION GLOBAL PATH (fallback)
					// Not all proxies are Nanite-enabled.
					// Sorts all TotalSplatCount splats (no compaction benefit).
					// Still provides correct cross-tile alpha blending.
					//==================================================

					if (!bCanSkip)
					{
						// --------------------------------------------------
						// Phase 1: Per-proxy ClusterCulling + CalcViewData
						// --------------------------------------------------
						for (const auto& Info : VisibleProxies)
						{
							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							bool bHasColorTexture = GPUResources->ColorTextureSRV.IsValid();

							// Cluster culling for Nanite-enabled proxies
							if (GPUResources->bEnableNanite && GPUResources->bHasClusterData)
							{
								FGaussianSplatRenderer::DispatchClusterCulling(
									RHICmdList, *SceneView, GPUResources,
									Info.LocalToWorld, Info.bUseLODRendering);
							}

							// CalcViewData → writes to GlobalViewDataBuffer at GlobalBaseOffset
							FGaussianSplatRenderer::DispatchCalcViewDataGlobal(
								RHICmdList, *SceneView, GPUResources,
								Info.LocalToWorld,
								Info.Proxy->GetSplatCount(),
								Info.Proxy->GetSHOrder(),
								Info.Proxy->GetOpacityScale(),
								Info.Proxy->GetSplatScale(),
								bHasColorTexture,
								Info.bUseLODRendering,
								Info.GlobalBaseOffset,
								RawAccumulator);
						}

						// --------------------------------------------------
						// Phase 2: Single global CalcDistances + RadixSort
						// --------------------------------------------------
						FGaussianSplatRenderer::DispatchCalcDistancesGlobal(RHICmdList, RawAccumulator, (int32)TotalSplatCount);
						FGaussianSplatRenderer::DispatchRadixSortGlobal(RHICmdList, RawAccumulator, (int32)TotalSplatCount);

						// Update caches
						RawAccumulator->bHasCachedSortData = true;
						RawAccumulator->CachedTotalSplatCount = TotalSplatCount;
						RawAccumulator->CachedViewProjectionMatrix = CurrentVP;

						for (const auto& Info : VisibleProxies)
						{
							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							GPUResources->CachedViewProjectionMatrix = CurrentVP;
							GPUResources->CachedLocalToWorld = Info.LocalToWorld;
							GPUResources->CachedOpacityScale = Info.Proxy->GetOpacityScale();
							GPUResources->CachedSplatScale = Info.Proxy->GetSplatScale();
							GPUResources->CachedHasColorTexture = GPUResources->ColorTextureSRV.IsValid();
							GPUResources->CachedErrorThreshold = CurrentErrorThreshold;
							GPUResources->CachedDebugMode = CurrentDebugMode;
							GPUResources->CachedDebugForceLODLevel = CurrentDebugForceLODLevel;
							GPUResources->bHasCachedSortData = true;
						}
					}

					// Single draw call for ALL proxies
					FGaussianSplatRenderer::DrawSplatsGlobal(
						RHICmdList, *SceneView, RawAccumulator,
						SharedIndexBuffer, (int32)TotalSplatCount, DebugMode);
				}
			}
		);
	}
	else
	{
		//------------------------------------------------------------------
		// LEGACY PER-PROXY PATH (fallback, kept for debugging/regression)
		//------------------------------------------------------------------
		GraphBuilder.AddPass(
			RDG_EVENT_NAME("GaussianSplatRendering"),
			PassParameters,
			ERDGPassFlags::Raster,
			[SceneView, Proxies](FRHICommandListImmediate& RHICmdList)
			{
				if (!SceneView) return;
				SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatRendering);

				for (FGaussianSplatSceneProxy* Proxy : Proxies)
				{
					if (!Proxy) continue;
					if (&Proxy->GetScene() != SceneView->Family->Scene) continue;
					if (!Proxy->IsShown(SceneView)) continue;

					Proxy->TryInitializeColorTexture(RHICmdList);

					FGaussianSplatGPUResources* GPUResources = Proxy->GetGPUResources();
					if (!GPUResources || !GPUResources->IsValid()) continue;

					const FBoxSphereBounds& Bounds = Proxy->GetBounds();
					if (!SceneView->ViewFrustum.IntersectBox(Bounds.Origin, Bounds.BoxExtent)) continue;

					FGaussianSplatRenderer::Render(
						RHICmdList, *SceneView, GPUResources,
						Proxy->GetLocalToWorld(),
						Proxy->GetSplatCount(),
						Proxy->GetSHOrder(),
						Proxy->GetOpacityScale(),
						Proxy->GetSplatScale()
					);
				}
			}
		);
	}
}

void FGaussianSplattingModule::ShutdownModule()
{
	// Unregister post-opaque render delegate
	if (PostOpaqueRenderDelegateHandle.IsValid())
	{
		GetRendererModuleRef().RemovePostOpaqueRenderDelegate(PostOpaqueRenderDelegateHandle);
		PostOpaqueRenderDelegateHandle.Reset();
	}

	// Release global accumulator GPU buffers from the render thread
	if (GlobalAccumulator.IsValid())
	{
		FGaussianGlobalAccumulator* RawAccumulator = GlobalAccumulator.Release();
		ENQUEUE_RENDER_COMMAND(ReleaseGlobalAccumulator)(
			[RawAccumulator](FRHICommandListImmediate& RHICmdList)
			{
				RawAccumulator->Release();
				delete RawAccumulator;
			});
	}

	// Clear the view extension
	ViewExtension.Reset();
}

FGaussianSplattingModule& FGaussianSplattingModule::Get()
{
	return FModuleManager::LoadModuleChecked<FGaussianSplattingModule>("GaussianSplatting");
}

bool FGaussianSplattingModule::IsAvailable()
{
	return FModuleManager::Get().IsModuleLoaded("GaussianSplatting");
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FGaussianSplattingModule, GaussianSplatting)
