// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "Containers/Ticker.h"
#include "RendererInterface.h"
#include "GaussianGlobalAccumulator.h"

class FGaussianSplatViewExtension;

class FGaussianSplattingModule : public IModuleInterface
{
public:
	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

	/**
	 * Singleton-like access to this module's interface.
	 * @return Returns singleton instance, loading the module on demand if needed
	 */
	static FGaussianSplattingModule& Get();

	/**
	 * Checks to see if this module is loaded and ready.
	 * @return True if the module is loaded and ready to use
	 */
	static bool IsAvailable();

private:
	/** View extension for Gaussian Splat rendering */
	TSharedPtr<FGaussianSplatViewExtension, ESPMode::ThreadSafe> ViewExtension;

	/** Ticker delegate handle for periodic status checks */
	FTSTicker::FDelegateHandle TickDelegateHandle;

	/** Post opaque render delegate handle */
	FDelegateHandle PostOpaqueRenderDelegateHandle;

	/** Global GPU accumulator for one-draw-call path (render-thread owned) */
	TUniquePtr<FGaussianGlobalAccumulator> GlobalAccumulator;

	/** Render callback */
	void OnPostOpaqueRender_RenderThread(FPostOpaqueRenderParameters& Parameters);
};
