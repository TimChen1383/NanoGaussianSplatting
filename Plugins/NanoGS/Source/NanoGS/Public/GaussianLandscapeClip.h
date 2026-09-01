// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "RHI.h"
#include "RHIResources.h"

/**
 * Render-thread landscape height used to hide Gaussian splats below the terrain.
 * Game thread uploads a normalized [0,1] heightmap; shaders reconstruct world Z
 * with the same origin / scale convention as AProceduralLandscapeActor.
 */
struct NANOGS_API FGaussianLandscapeClipShaderParams
{
	FRHITexture* HeightTexture = nullptr;
	FRHISamplerState* Sampler = nullptr;
	uint32 bEnabled = 0;
	FVector3f Origin = FVector3f::ZeroVector;
	FVector2f HalfExtent = FVector2f::ZeroVector;
	float ScaleXY = 1.0f;
	float ScaleZ = 1.0f;
	float ZOffset = 0.0f;
	float CenterZ = 0.0f;
	FVector2f Dimensions = FVector2f::ZeroVector;
	uint32 Version = 0;
};

class NANOGS_API FGaussianLandscapeClip
{
public:
	static void UpdateFromGameThread(
		const TArray<float>& Heights,
		int32 Width,
		int32 Height,
		const FVector& Origin,
		const FVector2D& HalfExtent,
		double ScaleXY,
		double ScaleZ,
		double ZOffset,
		double CenterZ);

	static void ClearFromGameThread();

	/** Must be called on the render thread. */
	static FGaussianLandscapeClipShaderParams Get_RenderThread();

	static uint32 GetVersion_RenderThread();

	/** Release GPU resources. Call from the render thread during module shutdown. */
	static void Release_RenderThread();
};
