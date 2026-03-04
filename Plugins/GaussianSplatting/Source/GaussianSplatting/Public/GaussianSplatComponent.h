// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/PrimitiveComponent.h"
#include "GaussianDataTypes.h"
#include "GaussianSplatComponent.generated.h"

class UGaussianSplatAsset;
class FGaussianSplatSceneProxy;

/**
 * Component for rendering Gaussian Splatting assets in the scene
 */
UCLASS(ClassGroup = (Rendering), meta = (BlueprintSpawnableComponent), hidecategories = (Collision, Physics, Navigation))
class GAUSSIANSPLATTING_API UGaussianSplatComponent : public UPrimitiveComponent
{
	GENERATED_BODY()

public:
	UGaussianSplatComponent(const FObjectInitializer& ObjectInitializer);

	//~ Begin UObject Interface
	virtual void PostLoad() override;
#if WITH_EDITOR
	virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif
	//~ End UObject Interface

	//~ Begin UActorComponent Interface
	virtual void OnRegister() override;
	virtual void OnUnregister() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
	//~ End UActorComponent Interface

	//~ Begin UPrimitiveComponent Interface
	virtual FPrimitiveSceneProxy* CreateSceneProxy() override;
	virtual FBoxSphereBounds CalcBounds(const FTransform& LocalToWorld) const override;
	virtual void GetUsedMaterials(TArray<UMaterialInterface*>& OutMaterials, bool bGetDebugMaterials = false) const override;
	//~ End UPrimitiveComponent Interface

	/** Set the Gaussian Splat asset to render */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting")
	void SetSplatAsset(UGaussianSplatAsset* NewAsset);

	/** Get the currently assigned asset */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting")
	UGaussianSplatAsset* GetSplatAsset() const { return SplatAsset; }

	/** Get the number of splats being rendered */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting")
	int32 GetSplatCount() const;

public:
	/** The Gaussian Splat asset to render */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Gaussian Splatting")
	TObjectPtr<UGaussianSplatAsset> SplatAsset;

	/** Spherical Harmonic order to use for rendering (0-3). Higher = more color detail but slower. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Gaussian Splatting|Quality", meta = (ClampMin = "0", ClampMax = "3"))
	int32 SHOrder = 3;

	/** Sort splats every N frames. 1 = every frame. Higher values reduce GPU cost but may cause artifacts during fast camera movement. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Gaussian Splatting|Performance", meta = (ClampMin = "1", ClampMax = "10"))
	int32 SortEveryNthFrame = 1;

	/** Global opacity multiplier */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Gaussian Splatting|Rendering", meta = (ClampMin = "0.0", ClampMax = "2.0"))
	float OpacityScale = 1.0f;

	/** Scale multiplier for splat sizes */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Gaussian Splatting|Rendering", meta = (ClampMin = "0.1", ClampMax = "10.0"))
	float SplatScale = 1.0f;

	/** Enable frustum culling for better performance */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Gaussian Splatting|Performance")
	bool bEnableFrustumCulling = true;

	/** Screen-space error threshold in pixels for LOD selection.
	 *  Lower values = more conservative (keep detail longer, less LOD savings)
	 *  Higher values = more aggressive (switch to LOD sooner, better performance)
	 *  Adjust per-asset based on point density - low density assets need lower values. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Gaussian Splatting|Performance", meta = (ClampMin = "0.1", ClampMax = "256.0"))
	float LODErrorThreshold = 32.0f;

protected:
	/** Called when the asset changes */
	void OnAssetChanged();

	/** Mark the render state as dirty */
	void MarkRenderStateDirty();

	/** Called when the asset's data changes (e.g., Nanite enabled/disabled) */
	void OnAssetDataChanged(class UGaussianSplatAsset* ChangedAsset);

	/** Subscribe to asset change notifications */
	void SubscribeToAssetChanges();

	/** Unsubscribe from asset change notifications */
	void UnsubscribeFromAssetChanges();

private:
	/** Cached bounds */
	mutable FBoxSphereBounds CachedBounds;
	mutable bool bBoundsCached = false;

	/** Delegate handle for asset change subscription */
	FDelegateHandle AssetChangedDelegateHandle;
};
