// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatting.h"
#include "GaussianSplatViewExtension.h"
#include "GaussianSplatRenderer.h"
#include "GaussianSplatSceneProxy.h"
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
	TEXT("Colors each splat based on its cluster ID for debugging.\n")
	TEXT(" 0: Off (default)\n")
	TEXT(" 1: Show cluster colors (each cluster gets a unique random color)\n")
	TEXT(" 2: Show LOD level brightness (dark blue=leaf, bright yellow=higher LOD)"),
	ECVF_RenderThreadSafe);

/** Show cluster visibility statistics */
TAutoConsoleVariable<int32> CVarShowClusterStats(
	TEXT("gs.ShowClusterStats"),
	0,
	TEXT("Show cluster culling statistics on screen.\n")
	TEXT(" 0: Off (default)\n")
	TEXT(" 1: On"),
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

/** Enable LOD rendering for Gaussian Splats */
TAutoConsoleVariable<int32> CVarUseLODRendering(
	TEXT("gs.UseLODRendering"),
	0,
	TEXT("[WIP - Not yet functional] Enable LOD rendering for Gaussian Splats.\n")
	TEXT("When enabled, distant clusters render simplified LOD splats instead of all original splats.\n")
	TEXT(" 0: Off - always render original splats (default)\n")
	TEXT(" 1: On - use LOD splats for distant clusters (requires GPU-driven implementation)"),
	ECVF_RenderThreadSafe);

/** Debug: Force a specific LOD level for debugging LOD hierarchy */
TAutoConsoleVariable<int32> CVarDebugForceLODLevel(
	TEXT("gs.DebugForceLODLevel"),
	-1,
	TEXT("Force rendering of a specific LOD level for debugging.\n")
	TEXT("This ignores normal LOD selection and forces all clusters to use specified level.\n")
	TEXT(" -1: Auto - normal LOD selection based on distance/error (default)\n")
	TEXT("  0: Force leaf clusters only - render original splats (finest detail)\n")
	TEXT("  1+: Force specific LOD level (1 = first parent level, 2 = second, etc.)\n")
	TEXT("Note: Higher levels have fewer, coarser splats. Max level depends on asset size.\n")
	TEXT("Use with gs.ShowClusterBounds 2 to visualize which LOD level is being rendered."),
	ECVF_RenderThreadSafe);

/** Enable splat compaction for GPU-driven work reduction */
TAutoConsoleVariable<int32> CVarUseCompaction(
	TEXT("gs.UseCompaction"),
	1,
	TEXT("Enable splat compaction for GPU-driven work reduction.\n")
	TEXT("When enabled, only visible splats are processed in CalcViewData, CalcDistances, and Sort.\n")
	TEXT("This provides significant performance improvement especially when camera is far from assets.\n")
	TEXT(" 0: Off - process all splats (original behavior)\n")
	TEXT(" 1: On - compact visible splats first, then process only those (default)"),
	ECVF_RenderThreadSafe);

// Export for other modules
int32 GGaussianSplatShowClusterBounds = 0;
int32 GGaussianSplatShowClusterStats = 0;

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

	// Register post-opaque render delegate for rendering
	PostOpaqueRenderDelegateHandle = GetRendererModuleRef().RegisterPostOpaqueRenderDelegate(
		FPostOpaqueRenderDelegate::CreateRaw(this, &FGaussianSplattingModule::OnPostOpaqueRender_RenderThread));

	UE_LOG(LogTemp, Log, TEXT("GaussianSplatting module started. Shader directory: %s"), *PluginShaderDir);
}

void FGaussianSplattingModule::OnPostOpaqueRender_RenderThread(FPostOpaqueRenderParameters& Parameters)
{
	// Get registered proxies from the view extension
	FGaussianSplatViewExtension* Ext = FGaussianSplatViewExtension::Get();
	if (!Ext)
	{
		return;
	}

	// Access the parameters - UE5.6 uses FRDGBuilder
	if (!Parameters.GraphBuilder || !Parameters.View)
	{
		return;
	}

	FRDGBuilder& GraphBuilder = *Parameters.GraphBuilder;
	const FViewInfo& View = *Parameters.View;

	// Get registered proxies
	TArray<FGaussianSplatSceneProxy*> Proxies;
	Ext->GetRegisteredProxies(Proxies);

	if (Proxies.Num() == 0)
	{
		return;
	}

	// Store view pointer for lambda capture (FViewInfo inherits from FSceneView)
	const FSceneView* SceneView = reinterpret_cast<const FSceneView*>(Parameters.View);

	// Get the color texture from parameters - we need to render to this
	FRDGTexture* ColorTexture = Parameters.ColorTexture;
	FRDGTexture* DepthTexture = Parameters.DepthTexture;

	if (!ColorTexture)
	{
		return;
	}

	// Create render pass parameters with the scene color as render target
	FRenderTargetParameters* PassParameters = GraphBuilder.AllocParameters<FRenderTargetParameters>();
	PassParameters->RenderTargets[0] = FRenderTargetBinding(ColorTexture, ERenderTargetLoadAction::ELoad);
	if (DepthTexture)
	{
		PassParameters->RenderTargets.DepthStencil = FDepthStencilBinding(
			DepthTexture,
			ERenderTargetLoadAction::ELoad,
			ERenderTargetLoadAction::ELoad,
			FExclusiveDepthStencil::DepthRead_StencilRead
		);
	}

	// Add a render pass for Gaussian splats with proper render target binding
	GraphBuilder.AddPass(
		RDG_EVENT_NAME("GaussianSplatRendering"),
		PassParameters,
		ERDGPassFlags::Raster,
		[SceneView, Proxies](FRHICommandListImmediate& RHICmdList)
		{
			if (!SceneView)
			{
				return;
			}

			SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatRendering);

			for (FGaussianSplatSceneProxy* Proxy : Proxies)
			{
				if (!Proxy)
				{
					continue;
				}

				// Check if proxy is shown in this view (respects visibility flags)
				if (!Proxy->IsShown(SceneView))
				{
					continue;
				}

				// Try to initialize color texture SRV if not already done (deferred init)
				Proxy->TryInitializeColorTexture(RHICmdList);

				FGaussianSplatGPUResources* GPUResources = Proxy->GetGPUResources();

				// Require full validation
				if (!GPUResources || !GPUResources->IsValid())
				{
					continue;
				}

				// Check visibility - frustum culling
				const FBoxSphereBounds& Bounds = Proxy->GetBounds();
				if (!SceneView->ViewFrustum.IntersectBox(Bounds.Origin, Bounds.BoxExtent))
				{
					continue;
				}

				// Get transform
				FMatrix LocalToWorld = Proxy->GetLocalToWorld();

				// Render this proxy
				FGaussianSplatRenderer::Render(
					RHICmdList,
					*SceneView,
					GPUResources,
					LocalToWorld,
					Proxy->GetSplatCount(),
					Proxy->GetSHOrder(),
					Proxy->GetOpacityScale(),
					Proxy->GetSplatScale()
				);
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
