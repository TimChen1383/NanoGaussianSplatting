// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianLandscapeClip.h"
#include "RHICommandList.h"
#include "RHIStaticStates.h"
#include "RenderingThread.h"
#include "RenderUtils.h"

namespace
{
	struct FGaussianLandscapeClipState
	{
		FTextureRHIRef HeightTexture;
		FTextureRHIRef DummyTexture;
		FVector3f Origin = FVector3f::ZeroVector;
		FVector2f HalfExtent = FVector2f::ZeroVector;
		float ScaleXY = 1.0f;
		float ScaleZ = 1.0f;
		float ZOffset = 0.0f;
		float CenterZ = 0.0f;
		int32 Width = 0;
		int32 Height = 0;
		bool bEnabled = false;
		uint32 Version = 0;
	};

	FGaussianLandscapeClipState GClipState;

	void EnsureDummyTexture()
	{
		if (GClipState.DummyTexture.IsValid())
		{
			return;
		}

		const FRHITextureCreateDesc Desc =
			FRHITextureCreateDesc::Create2D(TEXT("GaussianLandscapeClipDummy"), 1, 1, PF_R32_FLOAT)
			.SetFlags(ETextureCreateFlags::ShaderResource)
			.SetInitialState(ERHIAccess::SRVMask);

		GClipState.DummyTexture = RHICreateTexture(Desc);

		const float Zero = 0.0f;
		const FUpdateTextureRegion2D Region(0, 0, 0, 0, 1, 1);
		RHIUpdateTexture2D(GClipState.DummyTexture, 0, Region, sizeof(float), reinterpret_cast<const uint8*>(&Zero));
	}

	void UploadHeightTexture(const TArray<float>& Heights, int32 Width, int32 Height)
	{
		const bool bNeedNewTexture =
			!GClipState.HeightTexture.IsValid() ||
			GClipState.Width != Width ||
			GClipState.Height != Height;

		if (bNeedNewTexture)
		{
			GClipState.HeightTexture.SafeRelease();

			const FRHITextureCreateDesc Desc =
				FRHITextureCreateDesc::Create2D(TEXT("GaussianLandscapeHeight"), Width, Height, PF_R32_FLOAT)
				.SetNumMips(1)
				.SetFlags(ETextureCreateFlags::ShaderResource)
				.SetInitialState(ERHIAccess::SRVMask);

			GClipState.HeightTexture = RHICreateTexture(Desc);
			GClipState.Width = Width;
			GClipState.Height = Height;
		}

		const FUpdateTextureRegion2D Region(0, 0, 0, 0, Width, Height);
		const uint32 SourcePitch = static_cast<uint32>(Width) * sizeof(float);
		RHIUpdateTexture2D(
			GClipState.HeightTexture,
			0,
			Region,
			SourcePitch,
			reinterpret_cast<const uint8*>(Heights.GetData()));
	}
}

void FGaussianLandscapeClip::UpdateFromGameThread(
	const TArray<float>& Heights,
	int32 Width,
	int32 Height,
	const FVector& Origin,
	const FVector2D& HalfExtent,
	double ScaleXY,
	double ScaleZ,
	double ZOffset,
	double CenterZ)
{
	if (Width < 2 || Height < 2 || Heights.Num() != Width * Height)
	{
		ClearFromGameThread();
		return;
	}

	TArray<float> HeightsCopy = Heights;
	const FVector3f OriginF(Origin);
	const FVector2f HalfExtentF(static_cast<float>(HalfExtent.X), static_cast<float>(HalfExtent.Y));
	const float ScaleXYF = static_cast<float>(ScaleXY);
	const float ScaleZF = static_cast<float>(ScaleZ);
	const float ZOffsetF = static_cast<float>(ZOffset);
	const float CenterZF = static_cast<float>(CenterZ);

	ENQUEUE_RENDER_COMMAND(UpdateGaussianLandscapeClip)(
		[HeightsCopy = MoveTemp(HeightsCopy), Width, Height, OriginF, HalfExtentF, ScaleXYF, ScaleZF, ZOffsetF, CenterZF]
		(FRHICommandListImmediate& /*RHICmdList*/) mutable
		{
			UploadHeightTexture(HeightsCopy, Width, Height);
			GClipState.Origin = OriginF;
			GClipState.HalfExtent = HalfExtentF;
			GClipState.ScaleXY = ScaleXYF;
			GClipState.ScaleZ = ScaleZF;
			GClipState.ZOffset = ZOffsetF;
			GClipState.CenterZ = CenterZF;
			GClipState.bEnabled = GClipState.HeightTexture.IsValid();
			++GClipState.Version;
		});
}

void FGaussianLandscapeClip::ClearFromGameThread()
{
	ENQUEUE_RENDER_COMMAND(ClearGaussianLandscapeClip)(
		[](FRHICommandListImmediate& /*RHICmdList*/)
		{
			GClipState.HeightTexture.SafeRelease();
			GClipState.bEnabled = false;
			GClipState.Width = 0;
			GClipState.Height = 0;
			++GClipState.Version;
		});
}

FGaussianLandscapeClipShaderParams FGaussianLandscapeClip::Get_RenderThread()
{
	EnsureDummyTexture();

	FGaussianLandscapeClipShaderParams Params;
	Params.HeightTexture = GClipState.bEnabled && GClipState.HeightTexture.IsValid()
		? GClipState.HeightTexture.GetReference()
		: GClipState.DummyTexture.GetReference();
	Params.Sampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	Params.bEnabled = GClipState.bEnabled && GClipState.HeightTexture.IsValid() ? 1u : 0u;
	Params.Origin = GClipState.Origin;
	Params.HalfExtent = GClipState.HalfExtent;
	Params.ScaleXY = GClipState.ScaleXY;
	Params.ScaleZ = GClipState.ScaleZ;
	Params.ZOffset = GClipState.ZOffset;
	Params.CenterZ = GClipState.CenterZ;
	Params.Dimensions = FVector2f(
		static_cast<float>(FMath::Max(GClipState.Width, 1)),
		static_cast<float>(FMath::Max(GClipState.Height, 1)));
	Params.Version = GClipState.Version;
	return Params;
}

uint32 FGaussianLandscapeClip::GetVersion_RenderThread()
{
	return GClipState.Version;
}

void FGaussianLandscapeClip::Release_RenderThread()
{
	GClipState.HeightTexture.SafeRelease();
	GClipState.DummyTexture.SafeRelease();
	GClipState.bEnabled = false;
	GClipState.Width = 0;
	GClipState.Height = 0;
	++GClipState.Version;
}
