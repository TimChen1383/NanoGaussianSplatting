// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatViewExtension.h"
#include "GaussianSplatSceneProxy.h"
#include "GaussianSplatRenderer.h"
#include "GaussianGlobalAccumulator.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "ScreenPass.h"
#include "SceneRendering.h"
#include "PostProcess/PostProcessMaterialInputs.h"

FGaussianSplatViewExtension* FGaussianSplatViewExtension::Instance = nullptr;

FGaussianSplatViewExtension::FGaussianSplatViewExtension(const FAutoRegister& AutoRegister)
	: FSceneViewExtensionBase(AutoRegister)
{
	Instance = this;
}

FGaussianSplatViewExtension::~FGaussianSplatViewExtension()
{
	if (Instance == this)
	{
		Instance = nullptr;
	}
}

FGaussianSplatViewExtension* FGaussianSplatViewExtension::Get()
{
	return Instance;
}

void FGaussianSplatViewExtension::GetRegisteredProxies(TArray<FGaussianSplatSceneProxy*>& OutProxies) const
{
	FScopeLock Lock(&ProxyLock);
	OutProxies = RegisteredProxies;
}

bool FGaussianSplatViewExtension::IsActiveThisFrame_Internal(const FSceneViewExtensionContext& Context) const
{
	FScopeLock Lock(&ProxyLock);
	return RegisteredProxies.Num() > 0;
}

void FGaussianSplatViewExtension::SetupViewFamily(FSceneViewFamily& InViewFamily)
{
}

void FGaussianSplatViewExtension::BeginRenderViewFamily(FSceneViewFamily& InViewFamily)
{
}

void FGaussianSplatViewExtension::RegisterProxy(FGaussianSplatSceneProxy* Proxy)
{
	if (Proxy)
	{
		FScopeLock Lock(&ProxyLock);
		RegisteredProxies.AddUnique(Proxy);
	}
}

void FGaussianSplatViewExtension::UnregisterProxy(FGaussianSplatSceneProxy* Proxy)
{
	if (Proxy)
	{
		FScopeLock Lock(&ProxyLock);
		RegisteredProxies.Remove(Proxy);
	}
}

void FGaussianSplatViewExtension::PreRenderView_RenderThread(FRDGBuilder& GraphBuilder, FSceneView& InView)
{
}

void FGaussianSplatViewExtension::PreRenderViewFamily_RenderThread(FRDGBuilder& GraphBuilder, FSceneViewFamily& InViewFamily)
{
}

void FGaussianSplatViewExtension::PostRenderViewFamily_RenderThread(FRDGBuilder& GraphBuilder, FSceneViewFamily& InViewFamily)
{
}

void FGaussianSplatViewExtension::PostRenderBasePassDeferred_RenderThread(FRDGBuilder& GraphBuilder, FSceneView& InView, const FRenderTargetBindingSlots& RenderTargets, TRDGUniformBufferRef<FSceneTextureUniformParameters> SceneTextures)
{
	// Too early in the pipeline (before lighting) for Gaussian splats
}

void FGaussianSplatViewExtension::SubscribeToPostProcessingPass(EPostProcessingPass Pass, const FSceneView& InView, FPostProcessingPassDelegateArray& InOutPassCallbacks, bool bIsPassEnabled)
{
	if (Pass == EPostProcessingPass::MotionBlur)
	{
		InOutPassCallbacks.Add(FAfterPassCallbackDelegate::CreateRaw(
			this, &FGaussianSplatViewExtension::PostProcessPass_RenderThread));
	}
}

// Console variables (declared in GaussianSplatting.cpp)
extern TAutoConsoleVariable<int32> CVarShowClusterBounds;
extern TAutoConsoleVariable<int32> CVarMaxRenderBudget;
extern TAutoConsoleVariable<int32> CVarDebugForceLODLevel;
extern int32 GGaussianSplatShowClusterBounds;

FScreenPassTexture FGaussianSplatViewExtension::PostProcessPass_RenderThread(
	FRDGBuilder& GraphBuilder,
	const FSceneView& View,
	const FPostProcessMaterialInputs& Inputs)
{
	// Get SceneColor from post-process inputs
	FScreenPassTextureSlice SceneColorSlice = Inputs.GetInput(EPostProcessMaterialInput::SceneColor);
	check(SceneColorSlice.IsValid());
	FScreenPassTexture SceneColor(SceneColorSlice);

	// Early out if no proxies or no accumulator
	TArray<FGaussianSplatSceneProxy*> Proxies;
	GetRegisteredProxies(Proxies);

	if (Proxies.Num() == 0 || !GlobalAccumulator)
	{
		return Inputs.ReturnUntouchedSceneColorForPostProcessing(GraphBuilder);
	}

	// Handle OverrideOutput (when this is the last pass in the chain)
	FScreenPassRenderTarget Output = Inputs.OverrideOutput;
	if (Output.IsValid())
	{
		// Copy SceneColor to OverrideOutput first, then we render splats on top
		AddDrawTexturePass(GraphBuilder, FScreenPassViewInfo(View), SceneColor, Output);
		Output.LoadAction = ERenderTargetLoadAction::ELoad;
	}
	else
	{
		Output = FScreenPassRenderTarget(SceneColor.Texture, SceneColor.ViewRect, ERenderTargetLoadAction::ELoad);
	}

	// Get scene depth for depth testing (read-only)
	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);
	FRDGTextureRef DepthTexture = nullptr;
	if (const FSceneTextures* SceneTex = ViewInfo.GetSceneTexturesChecked())
	{
		FRDGTextureRef Depth = SceneTex->Depth.Target;
		// Only bind depth when extent matches color (TSR upscaling guard)
		if (Depth && Depth->Desc.Extent == Output.Texture->Desc.Extent)
		{
			DepthTexture = Depth;
		}
	}

	// Sort proxies back-to-front for correct depth ordering
	if (Proxies.Num() > 1)
	{
		FVector CameraPosition = View.ViewMatrices.GetViewOrigin();
		Proxies.Sort([CameraPosition](const FGaussianSplatSceneProxy& A, const FGaussianSplatSceneProxy& B)
		{
			float DistA = FVector::DistSquared(CameraPosition, A.GetBounds().Origin);
			float DistB = FVector::DistSquared(CameraPosition, B.GetBounds().Origin);
			return DistA > DistB;
		});
	}

	int32 DebugMode = CVarShowClusterBounds.GetValueOnRenderThread();
	ERenderTargetLoadAction ColorLoadAction = (DebugMode > 0) ? ERenderTargetLoadAction::EClear : ERenderTargetLoadAction::ELoad;

	FRenderTargetParameters* PassParameters = GraphBuilder.AllocParameters<FRenderTargetParameters>();
	PassParameters->RenderTargets[0] = FRenderTargetBinding(Output.Texture, ColorLoadAction);
	// Bind scene depth as read-only for depth testing against static mesh geometry
	if (DepthTexture)
	{
		PassParameters->RenderTargets.DepthStencil = FDepthStencilBinding(
			DepthTexture,
			ERenderTargetLoadAction::ELoad,
			ERenderTargetLoadAction::ELoad,
			FExclusiveDepthStencil::DepthRead_StencilNop
		);
	}

	// Build visible proxy list and compute total splat count
	struct FProxyRenderInfo
	{
		FGaussianSplatSceneProxy* Proxy;
		FMatrix LocalToWorld;
		uint32 GlobalBaseOffset;
		bool bUseLODRendering;
		float DistanceToCamera;
	};
	TArray<FProxyRenderInfo> VisibleProxies;
	uint32 TotalSplatCount = 0;
	bool bAllNanite = true;

	FVector CameraLocation = View.ViewLocation;

	for (FGaussianSplatSceneProxy* Proxy : Proxies)
	{
		if (!Proxy) continue;
		if (&Proxy->GetScene() != View.Family->Scene) continue;
		if (!Proxy->IsShown(&View)) continue;

		const FBoxSphereBounds& Bounds = Proxy->GetBounds();
		if (!View.ViewFrustum.IntersectBox(Bounds.Origin, Bounds.BoxExtent)) continue;

		FGaussianSplatGPUResources* GPUResources = Proxy->GetGPUResources();
		if (!GPUResources || !GPUResources->IsValid()) continue;

		FProxyRenderInfo Info;
		Info.Proxy = Proxy;
		Info.LocalToWorld = Proxy->GetLocalToWorld();
		Info.GlobalBaseOffset = 0;
		Info.bUseLODRendering = GPUResources->bEnableNanite && GPUResources->bHasLODSplats;
		Info.DistanceToCamera = FVector::Dist(Bounds.Origin, CameraLocation);
		VisibleProxies.Add(Info);

		if (!GPUResources->bEnableNanite || !GPUResources->bHasClusterData || !GPUResources->bSupportsCompaction)
		{
			bAllNanite = false;
		}
	}

	if (VisibleProxies.Num() == 0)
	{
		return MoveTemp(Output);
	}

	// Sort by distance: closer proxies first (budget priority)
	VisibleProxies.Sort([](const FProxyRenderInfo& A, const FProxyRenderInfo& B)
	{
		return A.DistanceToCamera < B.DistanceToCamera;
	});

	// Compute GlobalBaseOffset and TotalSplatCount
	for (FProxyRenderInfo& Info : VisibleProxies)
	{
		Info.GlobalBaseOffset = TotalSplatCount;
		TotalSplatCount += (uint32)Info.Proxy->GetSplatCount();
	}

	if ((uint32)VisibleProxies.Num() > FGaussianGlobalAccumulator::MAX_PROXY_COUNT)
	{
		bAllNanite = false;
	}

	if (TotalSplatCount == 0)
	{
		return MoveTemp(Output);
	}

	// Check camera-static skip
	// Use GetProjectionMatrix() with jitter removed (not ComputeProjectionNoAAMatrix which strips screen percentage)
	FMatrix ProjNoJitter = View.ViewMatrices.GetProjectionMatrix();
	ProjNoJitter.M[2][0] = 0.0f;
	ProjNoJitter.M[2][1] = 0.0f;
	FMatrix CurrentVP = View.ViewMatrices.GetViewMatrix() * ProjNoJitter;
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

	// Grab index buffer from the first proxy
	FBufferRHIRef SharedIndexBuffer;
	if (VisibleProxies.Num() > 0)
	{
		FGaussianSplatGPUResources* FirstRes = VisibleProxies[0].Proxy->GetGPUResources();
		SharedIndexBuffer = FirstRes ? FirstRes->IndexBuffer : FBufferRHIRef();
	}

	FGaussianGlobalAccumulator* RawAccumulator = GlobalAccumulator;

	// Read render budget
	int32 BudgetVal = CVarMaxRenderBudget.GetValueOnRenderThread();
	uint32 MaxRenderBudget = (BudgetVal > 0) ? (uint32)BudgetVal : 0;
	if (CurrentDebugForceLODLevel >= 0)
	{
		MaxRenderBudget = 0;
	}

	const FSceneView* SceneView = &View;

	GraphBuilder.AddPass(
		RDG_EVENT_NAME("GaussianSplatRendering_Global"),
		PassParameters,
		ERDGPassFlags::Raster,
		[SceneView, VisibleProxies, TotalSplatCount, bCanSkip, bAllNanite, RawAccumulator,
		 SharedIndexBuffer, CurrentVP, CurrentDebugMode,
		 CurrentDebugForceLODLevel, DebugMode, MaxRenderBudget](FRHICommandListImmediate& RHICmdList)
		{
			if (!SceneView) return;
			SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatRendering_Global);

			// Re-validate all proxies before rendering
			TArray<FProxyRenderInfo> ValidProxies;
			ValidProxies.Reserve(VisibleProxies.Num());
			uint32 NewTotalSplatCount = 0;

			for (const auto& Info : VisibleProxies)
			{
				if (Info.Proxy && Info.Proxy->IsValidForRendering())
				{
					FProxyRenderInfo ValidInfo = Info;
					ValidInfo.GlobalBaseOffset = NewTotalSplatCount;
					NewTotalSplatCount += (uint32)Info.Proxy->GetSplatCount();
					ValidProxies.Add(ValidInfo);
				}
			}

			if (ValidProxies.Num() == 0 || NewTotalSplatCount == 0)
			{
				return;
			}

			bool bCanSkipAdjusted = bCanSkip;
			if (ValidProxies.Num() != VisibleProxies.Num() || NewTotalSplatCount != TotalSplatCount)
			{
				bCanSkipAdjusted = false;
				RawAccumulator->bHasCachedSortData = false;
			}

			// Initialize color textures (deferred init)
			for (const auto& Info : ValidProxies)
			{
				Info.Proxy->TryInitializeColorTexture(RHICmdList);
			}

			// Ensure global buffers are large enough
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

			if ((uint32)ValidProxies.Num() > FGaussianGlobalAccumulator::MAX_PROXY_COUNT)
			{
				bAllValidNanite = false;
			}

			if (bAllValidNanite)
			{
				//==================================================
				// GLOBAL + COMPACTION PATH
				//==================================================

				RawAccumulator->EnsureCompactionBuffersAllocated(RHICmdList);

				if (!bCanSkipAdjusted)
				{
					// Phase 0: Per-proxy culling + compaction + indirect args
					for (const auto& Info : ValidProxies)
					{
						FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
						if (!GPUResources) continue;
						int32 SplatCount = Info.Proxy->GetSplatCount();
						int32 OriginalSplatCount = SplatCount - GPUResources->LODSplatCount;

						FGaussianSplatRenderer::DispatchClusterCulling(
							RHICmdList, *SceneView, GPUResources,
							Info.LocalToWorld, Info.Proxy->GetLODErrorThreshold(), Info.bUseLODRendering);

						FGaussianSplatRenderer::DispatchCompactSplats(
							RHICmdList, GPUResources,
							SplatCount, OriginalSplatCount, Info.bUseLODRendering);

						FGaussianSplatRenderer::DispatchPrepareIndirectArgs(RHICmdList, GPUResources);
					}

					// Phase 1: Gather visible counts + GPU prefix sum
					int32 NumProxies = ValidProxies.Num();
					for (int32 i = 0; i < NumProxies; i++)
					{
						FGaussianSplatGPUResources* GPUResources = ValidProxies[i].Proxy->GetGPUResources();
						if (!GPUResources) continue;
						FGaussianSplatRenderer::DispatchGatherVisibleCount(
							RHICmdList, GPUResources, RawAccumulator, i);
					}

					FGaussianSplatRenderer::DispatchPrefixSumVisibleCounts(
						RHICmdList, RawAccumulator, NumProxies, MaxRenderBudget);

					// Phase 2: Per-proxy CalcViewData -> global buffer
					for (int32 i = 0; i < NumProxies; i++)
					{
						const auto& Info = ValidProxies[i];
						FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
						if (!GPUResources) continue;
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

					// Phase 3: Single global CalcDistances + RadixSort
					FGaussianSplatRenderer::DispatchCalcDistancesGlobalIndirect(RHICmdList, RawAccumulator);
					FGaussianSplatRenderer::DispatchRadixSortGlobalIndirect(RHICmdList, RawAccumulator);

					// Update caches
					RawAccumulator->bHasCachedSortData = true;
					RawAccumulator->CachedTotalSplatCount = NewTotalSplatCount;
					RawAccumulator->CachedViewProjectionMatrix = CurrentVP;

					for (const auto& Info : ValidProxies)
					{
						FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
						if (!GPUResources) continue;
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

				FGaussianSplatRenderer::DrawSplatsGlobalIndirect(
					RHICmdList, *SceneView, RawAccumulator, SharedIndexBuffer, DebugMode);
			}
			else
			{
				//==================================================
				// NON-COMPACTION GLOBAL PATH (fallback)
				//==================================================

				uint32 CappedTotalSplatCount = NewTotalSplatCount;
				if (MaxRenderBudget > 0 && CappedTotalSplatCount > MaxRenderBudget)
				{
					CappedTotalSplatCount = MaxRenderBudget;
				}

				if (!bCanSkipAdjusted)
				{
					// Phase 1: Per-proxy ClusterCulling + CalcViewData
					for (const auto& Info : ValidProxies)
					{
						if (MaxRenderBudget > 0 && Info.GlobalBaseOffset >= MaxRenderBudget)
						{
							break;
						}

						FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
						if (!GPUResources) continue;

						if (GPUResources->bEnableNanite && GPUResources->bHasClusterData)
						{
							FGaussianSplatRenderer::DispatchClusterCulling(
								RHICmdList, *SceneView, GPUResources,
								Info.LocalToWorld, Info.Proxy->GetLODErrorThreshold(), Info.bUseLODRendering);
						}

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

					// Phase 2: Single global CalcDistances + RadixSort
					FGaussianSplatRenderer::DispatchCalcDistancesGlobal(RHICmdList, RawAccumulator, (int32)CappedTotalSplatCount);
					FGaussianSplatRenderer::DispatchRadixSortGlobal(RHICmdList, RawAccumulator, (int32)CappedTotalSplatCount);

					// Update caches
					RawAccumulator->bHasCachedSortData = true;
					RawAccumulator->CachedTotalSplatCount = NewTotalSplatCount;
					RawAccumulator->CachedViewProjectionMatrix = CurrentVP;

					for (const auto& Info : ValidProxies)
					{
						FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
						if (!GPUResources) continue;
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

				FGaussianSplatRenderer::DrawSplatsGlobal(
					RHICmdList, *SceneView, RawAccumulator,
					SharedIndexBuffer, (int32)CappedTotalSplatCount, DebugMode);
			}
		}
	);

	return MoveTemp(Output);
}

