// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatThumbnailRenderer.h"
#include "GaussianSplatAsset.h"
#include "CanvasItem.h"
#include "Engine/Canvas.h"
#include "Engine/Texture2D.h"

void UGaussianSplatThumbnailRenderer::Draw(
	UObject* Object,
	int32 X, int32 Y,
	uint32 Width, uint32 Height,
	FRenderTarget* /*RenderTarget*/,
	FCanvas* Canvas,
	bool /*bAdditionalViewFamily*/)
{
	UGaussianSplatAsset* Asset = Cast<UGaussianSplatAsset>(Object);
	if (!IsValid(Asset)) return;

#if WITH_EDITORONLY_DATA
	UTexture2D* Thumb = Asset->ThumbnailTexture;
	if (!IsValid(Thumb)) return;

	// Make sure the GPU resource exists
	if (!Thumb->GetResource())
	{
		Thumb->UpdateResource();
	}

	FTexture* TextureResource = Thumb->GetResource();
	if (!TextureResource) return;

	FCanvasTileItem TileItem(
		FVector2D(X, Y),
		TextureResource,
		FVector2D(Width, Height),
		FLinearColor::White);

	TileItem.BlendMode = SE_BLEND_Opaque;
	Canvas->DrawItem(TileItem);
#endif
}

bool UGaussianSplatThumbnailRenderer::CanVisualizeAsset(UObject* Object)
{
#if WITH_EDITORONLY_DATA
	UGaussianSplatAsset* Asset = Cast<UGaussianSplatAsset>(Object);
	return IsValid(Asset) && IsValid(Asset->ThumbnailTexture);
#else
	return false;
#endif
}
