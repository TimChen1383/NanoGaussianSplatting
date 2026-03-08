// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "SceneViewExtension.h"

class FGaussianSplatSceneProxy;
struct FGaussianGlobalAccumulator;
struct FPostProcessMaterialInputs;
struct FScreenPassTexture;

/**
 * Scene View Extension for Gaussian Splatting
 * Manages registration of Gaussian Splat scene proxies.
 * Rendering hooks into the post-processing chain after TSR via SubscribeToPostProcessingPass.
 */
class GAUSSIANSPLATTING_API FGaussianSplatViewExtension : public FSceneViewExtensionBase
{
public:
	FGaussianSplatViewExtension(const FAutoRegister& AutoRegister);
	virtual ~FGaussianSplatViewExtension();

	//~ Begin ISceneViewExtension Interface
	//Different render pipeline timing for hooking
	virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override;
	virtual void SetupView(FSceneViewFamily& InViewFamily, FSceneView& InView) override {}
	virtual void BeginRenderViewFamily(FSceneViewFamily& InViewFamily) override;

	virtual void PreRenderView_RenderThread(FRDGBuilder& GraphBuilder, FSceneView& InView) override;
	virtual void PreRenderViewFamily_RenderThread(FRDGBuilder& GraphBuilder, FSceneViewFamily& InViewFamily) override;
	virtual void PostRenderViewFamily_RenderThread(FRDGBuilder& GraphBuilder, FSceneViewFamily& InViewFamily) override;

	virtual void PostRenderBasePassDeferred_RenderThread(FRDGBuilder& GraphBuilder, FSceneView& InView, const FRenderTargetBindingSlots& RenderTargets, TRDGUniformBufferRef<FSceneTextureUniformParameters> SceneTextures) override;

	virtual void SubscribeToPostProcessingPass(EPostProcessingPass Pass, const FSceneView& InView, FPostProcessingPassDelegateArray& InOutPassCallbacks, bool bIsPassEnabled) override;

	virtual bool IsActiveThisFrame_Internal(const FSceneViewExtensionContext& Context) const override;
	//~ End ISceneViewExtension Interface

	/** Register a scene proxy for rendering */
	void RegisterProxy(FGaussianSplatSceneProxy* Proxy);

	/** Unregister a scene proxy */
	void UnregisterProxy(FGaussianSplatSceneProxy* Proxy);

	/** Get the singleton instance */
	static FGaussianSplatViewExtension* Get();

	/** Get registered proxies for external rendering */
	void GetRegisteredProxies(TArray<FGaussianSplatSceneProxy*>& OutProxies) const;

	/** Get proxy lock for thread-safe access */
	FCriticalSection& GetProxyLock() const { return ProxyLock; }

	/** Set the global accumulator pointer (called by module during startup) */
	void SetGlobalAccumulator(FGaussianGlobalAccumulator* InGlobalAccumulator) { GlobalAccumulator = InGlobalAccumulator; }

private:
	/** Post-processing pass callback for rendering gaussian splats after TSR */
	FScreenPassTexture PostProcessPass_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View, const FPostProcessMaterialInputs& Inputs);
	/** Registered scene proxies */
	TArray<FGaussianSplatSceneProxy*> RegisteredProxies;

	/** Critical section for thread-safe proxy access */
	mutable FCriticalSection ProxyLock;

	/** Singleton instance */
	static FGaussianSplatViewExtension* Instance;

	/** Global accumulator for one-draw-call path (owned by Module, not this class) */
	FGaussianGlobalAccumulator* GlobalAccumulator = nullptr;
};
