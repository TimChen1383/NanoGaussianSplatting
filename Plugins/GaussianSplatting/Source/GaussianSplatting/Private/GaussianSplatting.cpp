// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatting.h"
#include "GaussianSplatViewExtension.h"
#include "GaussianGlobalAccumulator.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "ShaderCore.h"
#include "SceneViewExtension.h"
#include "Misc/CoreDelegates.h"

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

void FGaussianSplattingModule::StartupModule()
{
	// Register the shader directory so we can use our custom shaders
	FString PluginShaderDir = FPaths::Combine(IPluginManager::Get().FindPlugin(TEXT("GaussianSplatting"))->GetBaseDir(), TEXT("Shaders"));
	AddShaderSourceDirectoryMapping(TEXT("/Plugin/GaussianSplatting"), PluginShaderDir);

	// Allocate the global accumulator (buffers are created lazily on first render)
	GlobalAccumulator = MakeUnique<FGaussianGlobalAccumulator>();

	// Defer ViewExtension creation until GEngine is available
	// (StartupModule runs before GEngine is initialized, and NewExtension requires it)
	FCoreDelegates::OnPostEngineInit.AddLambda([this]()
	{
		ViewExtension = FSceneViewExtensions::NewExtension<FGaussianSplatViewExtension>();
		if (ViewExtension.IsValid())
		{
			ViewExtension->SetGlobalAccumulator(GlobalAccumulator.Get());
			UE_LOG(LogTemp, Log, TEXT("GaussianSplatting: ViewExtension created successfully (deferred)"));
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("GaussianSplatting: Failed to create ViewExtension!"));
		}
	});

	UE_LOG(LogTemp, Log, TEXT("GaussianSplatting module started. Shader directory: %s"), *PluginShaderDir);
}

void FGaussianSplattingModule::ShutdownModule()
{
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
