// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "ThumbnailRendering/DefaultSizedThumbnailRenderer.h"
#include "GaussianSplatThumbnailRenderer.generated.h"

/**
 * Draws the pre-baked thumbnail texture stored inside a UGaussianSplatAsset.
 */
UCLASS()
class UGaussianSplatThumbnailRenderer : public UDefaultSizedThumbnailRenderer
{
	GENERATED_BODY()

public:
	// UThumbnailRenderer interface
	virtual void Draw(UObject* Object, int32 X, int32 Y, uint32 Width, uint32 Height,
		FRenderTarget* RenderTarget, FCanvas* Canvas, bool bAdditionalViewFamily) override;

	virtual bool CanVisualizeAsset(UObject* Object) override;
};
