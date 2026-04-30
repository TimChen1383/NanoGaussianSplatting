#pragma once

#include "RHICommandList.h"
#include "RHIResources.h"
#include "Runtime/Launch/Resources/Version.h"

#if ENGINE_MAJOR_VERSION == 5 && ENGINE_MINOR_VERSION <= 3
struct FRHIBufferCreateDesc
{
	static FRHIBufferCreateDesc Create(const TCHAR* InDebugName, uint32 InSize, uint32 InStride, EBufferUsageFlags InUsage)
	{
		return FRHIBufferCreateDesc(InDebugName, InSize, InStride, InUsage);
	}

	FRHIBufferCreateDesc& SetInitialState(ERHIAccess InInitialState)
	{
		InitialState = InInitialState;
		return *this;
	}

	uint32 Size = 0;
	uint32 Stride = 0;
	EBufferUsageFlags Usage = BUF_None;
	ERHIAccess InitialState = ERHIAccess::Unknown;
	FRHIResourceCreateInfo CreateInfo;

private:
	FRHIBufferCreateDesc(const TCHAR* InDebugName, uint32 InSize, uint32 InStride, EBufferUsageFlags InUsage)
		: Size(InSize)
		, Stride(InStride)
		, Usage(InUsage)
		, CreateInfo(InDebugName)
	{
	}
};
#endif

static FORCEINLINE FBufferRHIRef NanoGSCreateBuffer(FRHICommandListBase& RHICmdList, FRHIBufferCreateDesc& Desc)
{
#if ENGINE_MAJOR_VERSION == 5 && ENGINE_MINOR_VERSION <= 3
	return RHICmdList.CreateBuffer(Desc.Size, Desc.Usage, Desc.Stride, Desc.InitialState, Desc.CreateInfo);
#else
	return RHICmdList.CreateBuffer(Desc);
#endif
}
