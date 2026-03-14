// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatting.h"
#include "GaussianSplatViewExtension.h"
#include "GaussianSplatRenderer.h"
#include "GaussianSplatSceneProxy.h"
#include "GaussianGlobalAccumulator.h"
#include "GlobalSplatBufferManager.h"
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

// Empty parameter struct for Copy passes (no render target binding slots)
BEGIN_SHADER_PARAMETER_STRUCT(FCopyPassParameters, )
END_SHADER_PARAMETER_STRUCT()

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

/** Maximum number of splats the global accumulator will allocate working buffers for.
 *  Caps VRAM usage for ViewData/sort/histogram buffers. If total visible splats exceed
 *  this budget (after Nanite LOD compaction), excess splats are simply not rendered.
 *  When budget is active, closer assets get priority (farther assets culled first).
 *  Default: 0 (unlimited). Example: 3M budget uses ~195 MB working buffers. */
TAutoConsoleVariable<int32> CVarMaxRenderBudget(
	TEXT("gs.MaxRenderBudget"),
	0,
	TEXT("Maximum number of splats to render per frame (render budget).\n")
	TEXT("Caps global accumulator buffer allocation and GPU-side visible count.\n")
	TEXT("When budget is exceeded, farther assets are culled first (closer assets have priority).\n")
	TEXT("Default: 0 (unlimited). Set to a positive value (e.g. 3000000) to limit splat count."),
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

	// Allocate the global splat buffer manager (Stage 1: buffer concat + metadata)
	GlobalSplatBufferManager = MakeUnique<FGlobalSplatBufferManager>();

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
	FRDGTexture* VelocityTexture = Parameters.VelocityTexture;
	if (!ColorTexture)
	{
		return;
	}

	int32 DebugMode = CVarShowClusterBounds.GetValueOnRenderThread();
	ERenderTargetLoadAction ColorLoadAction = (DebugMode > 0) ? ERenderTargetLoadAction::EClear : ERenderTargetLoadAction::ELoad;

	FRenderTargetParameters* PassParameters = GraphBuilder.AllocParameters<FRenderTargetParameters>();
	PassParameters->RenderTargets[0] = FRenderTargetBinding(ColorTexture, ColorLoadAction);
	// Bind velocity texture for TAA/TSR motion vector output
	if (VelocityTexture)
	{
		PassParameters->RenderTargets[1] = FRenderTargetBinding(VelocityTexture, ERenderTargetLoadAction::ELoad);
	}
	if (DepthTexture)
	{
		// Enable depth writes so TSR/TAA can properly detect disocclusion
		// Splats will write their center depth for each rendered pixel
		PassParameters->RenderTargets.DepthStencil = FDepthStencilBinding(
			DepthTexture,
			ERenderTargetLoadAction::ELoad,
			ERenderTargetLoadAction::ELoad,
			FExclusiveDepthStencil::DepthWrite_StencilRead
		);
	}

	if (!GlobalAccumulator.IsValid())
	{
		return;
	}

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
			float DistanceToCamera;  // For budget priority sorting (closer = higher priority)
		};
		TArray<FProxyRenderInfo> VisibleProxies;
		uint32 TotalSplatCount = 0;
		bool bAllNanite = true;  // True if every visible proxy supports compaction

		FVector CameraLocation = SceneView->ViewLocation;

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
			Info.GlobalBaseOffset = 0;  // Will be computed after sorting
			Info.bUseLODRendering = GPUResources->bEnableNanite && GPUResources->bHasLODSplats;
			Info.DistanceToCamera = FVector::Dist(Bounds.Origin, CameraLocation);
			VisibleProxies.Add(Info);

			// All proxies must support Nanite compaction for the fast global path
			if (!GPUResources->bEnableNanite || !GPUResources->bHasClusterData || !GPUResources->bSupportsCompaction)
			{
				bAllNanite = false;
			}
		}

		// Sort by distance: closer proxies first (get budget priority when MaxRenderBudget is active)
		VisibleProxies.Sort([](const FProxyRenderInfo& A, const FProxyRenderInfo& B)
		{
			return A.DistanceToCamera < B.DistanceToCamera;
		});

		// Compute GlobalBaseOffset and TotalSplatCount after sorting
		for (FProxyRenderInfo& Info : VisibleProxies)
		{
			Info.GlobalBaseOffset = TotalSplatCount;
			TotalSplatCount += (uint32)Info.Proxy->GetSplatCount();
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
		// Use ProjectionNoAAMatrix to ignore TSR/TAA per-frame jitter that changes every frame
		FMatrix CurrentVP = SceneView->ViewMatrices.GetViewMatrix() * SceneView->ViewMatrices.GetProjectionNoAAMatrix();
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
				float ProxyErrorThreshold = FMath::Max(0.1f, Info.Proxy->GetLODErrorThreshold());
				if (!GPUResources->bHasCachedSortData ||
					!GPUResources->CachedViewProjectionMatrix.Equals(CurrentVP, 0.0f) ||
					!GPUResources->CachedLocalToWorld.Equals(Info.LocalToWorld, 0.0f) ||
					GPUResources->CachedOpacityScale != Info.Proxy->GetOpacityScale() ||
					GPUResources->CachedSplatScale != Info.Proxy->GetSplatScale() ||
					GPUResources->CachedErrorThreshold != ProxyErrorThreshold ||
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
		FGlobalSplatBufferManager* RawSplatBufferManager = GlobalSplatBufferManager.Get();

		// Read render budget for global accumulator buffer cap.
		// Disable budget when forcing LOD level — the debug command needs to show
		// all assets regardless of splat count (user expects to see quality vs performance).
		int32 BudgetVal = CVarMaxRenderBudget.GetValueOnRenderThread();
		uint32 MaxRenderBudget = (BudgetVal > 0) ? (uint32)BudgetVal : 0;
		if (CurrentDebugForceLODLevel >= 0)
		{
			MaxRenderBudget = 0;  // Unlimited — debug mode overrides budget
		}

	// Stage 1 pre-pass: CopyBufferRegion requires IsOutsideRenderPass() == true.
	// SkipRenderPass runs on the graphics queue without starting a render pass.
	{
		FCopyPassParameters* RebuildParams = GraphBuilder.AllocParameters<FCopyPassParameters>();
		GraphBuilder.AddPass(
			RDG_EVENT_NAME("GaussianSplatGlobalBufferRebuild"),
			RebuildParams,
			ERDGPassFlags::Copy | ERDGPassFlags::NeverCull,
			[Proxies, RawSplatBufferManager](FRHICommandListImmediate& RHICmdList)
			{
				if (RawSplatBufferManager)
				{
					RawSplatBufferManager->UpdateIfNeeded(RHICmdList, Proxies);
				}
			});
	}

		GraphBuilder.AddPass(
			RDG_EVENT_NAME("GaussianSplatRendering_Global"),
			PassParameters,
			ERDGPassFlags::Raster,
			[SceneView, VisibleProxies, TotalSplatCount, bCanSkip, bAllNanite, RawAccumulator,
			 RawSplatBufferManager, SharedIndexBuffer, CurrentVP, CurrentDebugMode,
			 CurrentDebugForceLODLevel, DebugMode, MaxRenderBudget](FRHICommandListImmediate& RHICmdList)
			{
				if (!SceneView) return;
				SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatRendering_Global);

				// SAFETY CHECK: Re-validate all proxies before rendering.
				// Proxies may have been destroyed between when we built VisibleProxies
				// and when this lambda executes (RDG deferred execution).
				// We need to rebuild the list with only valid proxies.
				TArray<FProxyRenderInfo> ValidProxies;
				ValidProxies.Reserve(VisibleProxies.Num());
				uint32 NewTotalSplatCount = 0;

				for (const auto& Info : VisibleProxies)
				{
					// Check if proxy is still valid (not destroyed or pending destruction)
					if (Info.Proxy && Info.Proxy->IsValidForRendering())
					{
						FProxyRenderInfo ValidInfo = Info;
						ValidInfo.GlobalBaseOffset = NewTotalSplatCount;
						NewTotalSplatCount += (uint32)Info.Proxy->GetSplatCount();
						ValidProxies.Add(ValidInfo);
					}
				}

				// If no valid proxies remain, skip rendering entirely
				if (ValidProxies.Num() == 0 || NewTotalSplatCount == 0)
				{
					return;
				}

				// Invalidate cache skip if the proxy list changed (some proxies were destroyed)
				// This ensures we don't use stale cached data when the scene has changed
				bool bCanSkipAdjusted = bCanSkip;
				if (ValidProxies.Num() != VisibleProxies.Num() || NewTotalSplatCount != TotalSplatCount)
				{
					bCanSkipAdjusted = false;
					// Also invalidate the global accumulator cache since proxy set changed
					RawAccumulator->bHasCachedSortData = false;
				}

				// Initialize color textures (deferred init) - only for valid proxies
				for (const auto& Info : ValidProxies)
				{
					Info.Proxy->TryInitializeColorTexture(RHICmdList);
				}

				// Ensure global buffers are large enough for all splats
				RawAccumulator->ResizeIfNeeded(RHICmdList, NewTotalSplatCount);

				// Re-check if all valid proxies support Nanite compaction
				bool bAllValidNanite = true;
				for (const auto& Info : ValidProxies)
				{
					FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
					if (!GPUResources || !GPUResources->bEnableNanite || !GPUResources->bHasClusterData || !GPUResources->bSupportsCompaction)
					{
						bAllValidNanite = false;
						break;
					}
				}

				// Cap to MAX_PROXY_COUNT
				if ((uint32)ValidProxies.Num() > FGaussianGlobalAccumulator::MAX_PROXY_COUNT)
				{
					bAllValidNanite = false;
				}

				if (bAllValidNanite)
				{
					//==================================================
					// GLOBAL + COMPACTION PATH
					// All proxies are Nanite-enabled: GPU compaction
					// reduces working set from TotalSplatCount → TotalVisible
					// (~140x reduction at LOD5 for a 719K-splat tile).
					//==================================================

					// Ensure fixed-size prefix-sum buffers exist (allocated once)
					RawAccumulator->EnsureCompactionBuffersAllocated(RHICmdList);

					if (!bCanSkipAdjusted)
					{
						// --------------------------------------------------
						// Phase 0: Per-proxy culling + compaction + indirect args
						// Early-out: skip proxies once cumulative splat count
						// exceeds MaxRenderBudget (CPU-side estimate using total
						// splat count as conservative upper bound for visible count).
						// Proxies are sorted by distance, so closer ones get priority.
						// --------------------------------------------------
						int32 NumProcessedProxies = 0;
						uint32 CumulativeSplatCount = 0;

						for (const auto& Info : ValidProxies)
						{
							// Budget early-out: if cumulative total already exceeds budget,
							// skip culling/compaction for remaining (farther) proxies
							if (MaxRenderBudget > 0 && CumulativeSplatCount >= MaxRenderBudget)
							{
								break;
							}

							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							if (!GPUResources) continue;  // Extra safety check
							int32 SplatCount = Info.Proxy->GetSplatCount();
							int32 OriginalSplatCount = SplatCount - GPUResources->LODSplatCount;

							// Cluster culling → fills ClusterVisibilityBitmap
							FGaussianSplatRenderer::DispatchClusterCulling(
								RHICmdList, *SceneView, GPUResources,
								Info.LocalToWorld, Info.Proxy->GetLODErrorThreshold(), Info.bUseLODRendering);

							// Compact → fills CompactedSplatIndices + VisibleSplatCountBuffer
							FGaussianSplatRenderer::DispatchCompactSplats(
								RHICmdList, GPUResources,
								SplatCount, OriginalSplatCount, Info.bUseLODRendering);

							// PrepareIndirectArgs → fills IndirectDispatchArgsBuffer for CalcViewData
							FGaussianSplatRenderer::DispatchPrepareIndirectArgs(RHICmdList, GPUResources);

							CumulativeSplatCount += (uint32)SplatCount;
							NumProcessedProxies++;
						}

						// --------------------------------------------------
						// Stage 2: Global Cluster Culling (shadow mode)
						// Runs AFTER per-proxy culling to validate results
						// --------------------------------------------------
						if (RawSplatBufferManager && RawSplatBufferManager->IsReady())
						{
							// Build proxy array for validation readback
							TArray<FGaussianSplatSceneProxy*> ProcessedProxies;
							ProcessedProxies.Reserve(NumProcessedProxies);
							for (int32 pi = 0; pi < NumProcessedProxies; pi++)
							{
								ProcessedProxies.Add(ValidProxies[pi].Proxy);
							}
							RawSplatBufferManager->DispatchGlobalClusterCulling(
								RHICmdList, *SceneView, ProcessedProxies);

							// --------------------------------------------------
							// Stage 3: Global Compact Splats (shadow mode)
							// Runs AFTER per-proxy compaction AND Stage 2 global culling
							// Uses Stage 2 shadow bitmaps to validate splat counts
							// --------------------------------------------------
							RawSplatBufferManager->DispatchGlobalCompactSplats(
								RHICmdList, ProcessedProxies);
						}

						// --------------------------------------------------
						// Phase 1: Gather visible counts + GPU prefix sum
						// Only gather from proxies that were actually processed
						// --------------------------------------------------
						for (int32 i = 0; i < NumProcessedProxies; i++)
						{
							FGaussianSplatGPUResources* GPUResources = ValidProxies[i].Proxy->GetGPUResources();
							if (!GPUResources) continue;  // Extra safety check
							FGaussianSplatRenderer::DispatchGatherVisibleCount(
								RHICmdList, GPUResources, RawAccumulator, i);
						}

						// Single 1-thread dispatch: computes prefix sums + writes all indirect args
						FGaussianSplatRenderer::DispatchPrefixSumVisibleCounts(
							RHICmdList, RawAccumulator, NumProcessedProxies, MaxRenderBudget);

						// --------------------------------------------------
						// Phase 2: Per-proxy CalcViewData → global buffer
						// (indirect dispatch, only visible splats per proxy)
						// Only process proxies that went through culling/compaction
						// --------------------------------------------------
						for (int32 i = 0; i < NumProcessedProxies; i++)
						{
							const auto& Info = ValidProxies[i];
							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							if (!GPUResources) continue;  // Extra safety check
							int32 SplatCount = Info.Proxy->GetSplatCount();
							int32 OriginalSplatCount = SplatCount - GPUResources->LODSplatCount;

							FGaussianSplatRenderer::DispatchCalcViewDataCompactedGlobal(
								RHICmdList, *SceneView, GPUResources,
								Info.LocalToWorld,
								SplatCount,
								OriginalSplatCount,
								Info.Proxy->GetSHOrder(),
								Info.Proxy->GetOpacityScale(),
								Info.Proxy->GetSplatScale(),
								i,
								RawAccumulator,
								MaxRenderBudget);
						}

						// --------------------------------------------------
						// Phase 3: Single global CalcDistances + RadixSort
						// (all indirect — count driven by GPU prefix sum)
						// --------------------------------------------------
						FGaussianSplatRenderer::DispatchCalcDistancesGlobalIndirect(RHICmdList, RawAccumulator);
						FGaussianSplatRenderer::DispatchRadixSortGlobalIndirect(RHICmdList, RawAccumulator);

						// Update caches — only for processed proxies
						RawAccumulator->bHasCachedSortData = true;
						RawAccumulator->CachedTotalSplatCount = NewTotalSplatCount;
						RawAccumulator->CachedViewProjectionMatrix = CurrentVP;

						for (int32 i = 0; i < ValidProxies.Num(); i++)
						{
							FGaussianSplatGPUResources* GPUResources = ValidProxies[i].Proxy->GetGPUResources();
							if (!GPUResources) continue;

							if (i < NumProcessedProxies)
							{
								const auto& Info = ValidProxies[i];
								GPUResources->CachedViewProjectionMatrix = CurrentVP;
								GPUResources->CachedLocalToWorld = Info.LocalToWorld;
								GPUResources->CachedOpacityScale = Info.Proxy->GetOpacityScale();
								GPUResources->CachedSplatScale = Info.Proxy->GetSplatScale();
								GPUResources->CachedErrorThreshold = FMath::Max(0.1f, Info.Proxy->GetLODErrorThreshold());
								GPUResources->CachedDebugMode = CurrentDebugMode;
								GPUResources->CachedDebugForceLODLevel = CurrentDebugForceLODLevel;
								GPUResources->bHasCachedSortData = true;
							}
							else
							{
								// Invalidate cache for budget-skipped proxies so they
								// don't block the camera-static skip check
								GPUResources->bHasCachedSortData = false;
							}
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
					// Sorts all NewTotalSplatCount splats (no compaction benefit).
					// Still provides correct cross-tile alpha blending.
					//==================================================

					// Cap splat count to render budget (CPU-side enforcement)
					uint32 CappedTotalSplatCount = NewTotalSplatCount;
					if (MaxRenderBudget > 0 && CappedTotalSplatCount > MaxRenderBudget)
					{
						CappedTotalSplatCount = MaxRenderBudget;
					}

					if (!bCanSkipAdjusted)
					{
						// --------------------------------------------------
						// Phase 1: Per-proxy ClusterCulling + CalcViewData
						// --------------------------------------------------
						for (const auto& Info : ValidProxies)
						{
							// Skip proxies that would write beyond the render budget
							if (MaxRenderBudget > 0 && Info.GlobalBaseOffset >= MaxRenderBudget)
							{
								break;  // All subsequent proxies also exceed the budget
							}

							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							if (!GPUResources) continue;  // Extra safety check

							// Cluster culling for Nanite-enabled proxies
							if (GPUResources->bEnableNanite && GPUResources->bHasClusterData)
							{
								FGaussianSplatRenderer::DispatchClusterCulling(
									RHICmdList, *SceneView, GPUResources,
									Info.LocalToWorld, Info.Proxy->GetLODErrorThreshold(), Info.bUseLODRendering);
							}

							// CalcViewData → writes to GlobalViewDataBuffer at GlobalBaseOffset
							FGaussianSplatRenderer::DispatchCalcViewDataGlobal(
								RHICmdList, *SceneView, GPUResources,
								Info.LocalToWorld,
								Info.Proxy->GetSplatCount(),
								Info.Proxy->GetSHOrder(),
								Info.Proxy->GetOpacityScale(),
								Info.Proxy->GetSplatScale(),
								Info.bUseLODRendering,
								Info.GlobalBaseOffset,
								RawAccumulator);
						}

						// --------------------------------------------------
						// Phase 2: Single global CalcDistances + RadixSort
						// --------------------------------------------------
						FGaussianSplatRenderer::DispatchCalcDistancesGlobal(RHICmdList, RawAccumulator, (int32)CappedTotalSplatCount);
						FGaussianSplatRenderer::DispatchRadixSortGlobal(RHICmdList, RawAccumulator, (int32)CappedTotalSplatCount);

						// Update caches
						RawAccumulator->bHasCachedSortData = true;
						RawAccumulator->CachedTotalSplatCount = NewTotalSplatCount;
						RawAccumulator->CachedViewProjectionMatrix = CurrentVP;

						for (const auto& Info : ValidProxies)
						{
							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							if (!GPUResources) continue;  // Extra safety check
							GPUResources->CachedViewProjectionMatrix = CurrentVP;
							GPUResources->CachedLocalToWorld = Info.LocalToWorld;
							GPUResources->CachedOpacityScale = Info.Proxy->GetOpacityScale();
							GPUResources->CachedSplatScale = Info.Proxy->GetSplatScale();

							GPUResources->CachedErrorThreshold = FMath::Max(0.1f, Info.Proxy->GetLODErrorThreshold());
							GPUResources->CachedDebugMode = CurrentDebugMode;
							GPUResources->CachedDebugForceLODLevel = CurrentDebugForceLODLevel;
							GPUResources->bHasCachedSortData = true;
						}
					}

					// Single draw call for ALL proxies (capped to render budget)
					FGaussianSplatRenderer::DrawSplatsGlobal(
						RHICmdList, *SceneView, RawAccumulator,
						SharedIndexBuffer, (int32)CappedTotalSplatCount, DebugMode);
				}
			}
		);
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

	// Release global splat buffer manager GPU buffers from the render thread
	if (GlobalSplatBufferManager.IsValid())
	{
		FGlobalSplatBufferManager* RawManager = GlobalSplatBufferManager.Release();
		ENQUEUE_RENDER_COMMAND(ReleaseGlobalSplatBufferManager)(
			[RawManager](FRHICommandListImmediate& RHICmdList)
			{
				RawManager->Release();
				delete RawManager;
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
