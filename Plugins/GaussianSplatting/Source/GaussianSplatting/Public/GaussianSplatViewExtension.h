// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "SceneViewExtension.h"

class FGaussianSplatSceneProxy;

/**
 * Scene View Extension for Gaussian Splatting
 * Manages registration of Gaussian Splat scene proxies.
 * Actual rendering is handled by PostOpaqueRenderDelegate in FGaussianSplattingModule.
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

	virtual void PrePostProcessPass_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View, const FPostProcessingInputs& Inputs) override;

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

private:
	/** Registered scene proxies */
	TArray<FGaussianSplatSceneProxy*> RegisteredProxies;

	/** Critical section for thread-safe proxy access */
	mutable FCriticalSection ProxyLock;

	/** Singleton instance */
	static FGaussianSplatViewExtension* Instance;
};
