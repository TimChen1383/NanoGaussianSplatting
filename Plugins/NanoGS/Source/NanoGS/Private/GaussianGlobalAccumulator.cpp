// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianGlobalAccumulator.h"
#include "GaussianDataTypes.h"
#include "NanoGSRHICompat.h"
#include "RHICommandList.h"
#include "RHIResources.h"

void FGaussianGlobalAccumulator::ResizeIfNeeded(FRHICommandListBase& RHICmdList, uint32 NewTotalCount)
{
	// Apply render budget cap: working buffers only need to hold up to MaxRenderBudget splats,
	// not ALL splats. With Nanite LOD compaction, visible count is typically much smaller.
	static IConsoleVariable* CVarBudget = IConsoleManager::Get().FindConsoleVariable(TEXT("gs.MaxRenderBudget"));
	uint32 MaxBudget = (CVarBudget && CVarBudget->GetInt() > 0) ? (uint32)CVarBudget->GetInt() : 0;
	if (MaxBudget > 0 && NewTotalCount > MaxBudget)
	{
		NewTotalCount = MaxBudget;
	}

	if (NewTotalCount <= AllocatedCount)
	{
		return;
	}

	// Add 20% headroom to avoid per-frame reallocation when streaming adds tiles,
	// but never exceed the budget cap.
	uint32 NewAllocatedCount = FMath::Max(NewTotalCount + NewTotalCount / 5, 4096u);
	if (MaxBudget > 0)
	{
		NewAllocatedCount = FMath::Min(NewAllocatedCount, MaxBudget);
	}
	uint32 NewNumTiles = FMath::DivideAndRoundUp(NewAllocatedCount, 1024u);

	UE_LOG(LogTemp, Verbose, TEXT("GaussianGlobalAccumulator: Resizing from %u to %u splats (%u tiles)"),
		AllocatedCount, NewAllocatedCount, NewNumTiles);

	// Release old buffers before reallocation
	Release();

	const uint32 UintStride = sizeof(uint32);
	const uint32 ViewDataStride = sizeof(FGaussianSplatViewData);

	// --- ViewData buffer ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalViewDataBuffer"),
			NewAllocatedCount * ViewDataStride,
			ViewDataStride,
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalViewDataBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalViewDataBufferUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalViewDataBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(ViewDataStride));
		GlobalViewDataBufferSRV = RHICmdList.CreateShaderResourceView(
			GlobalViewDataBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(ViewDataStride));
	}

	// --- Sort distance buffer (primary) ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalSortDistanceBuffer"),
			NewAllocatedCount * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalSortDistanceBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalSortDistanceBufferUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalSortDistanceBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// --- Sort distance buffer (alt) ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalSortDistanceBufferAlt"),
			NewAllocatedCount * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalSortDistanceBufferAlt = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalSortDistanceBufferAltUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalSortDistanceBufferAlt, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// --- Sort keys buffer (primary) ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalSortKeysBuffer"),
			NewAllocatedCount * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalSortKeysBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalSortKeysBufferUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalSortKeysBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
		GlobalSortKeysBufferSRV = RHICmdList.CreateShaderResourceView(
			GlobalSortKeysBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// --- Sort keys buffer (alt) ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalSortKeysBufferAlt"),
			NewAllocatedCount * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalSortKeysBufferAlt = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalSortKeysBufferAltUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalSortKeysBufferAlt, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// --- Radix histogram: 256 * NumTiles ---
	{
		uint32 HistogramCount = 256 * NewNumTiles;
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalRadixHistogramBuffer"),
			HistogramCount * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalRadixHistogramBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalRadixHistogramBufferUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalRadixHistogramBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// --- Radix digit offset: 256 entries ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalRadixDigitOffsetBuffer"),
			256 * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalRadixDigitOffsetBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalRadixDigitOffsetBufferUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalRadixDigitOffsetBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// --- SortParams: 2 uints — SRV for non-compaction sort, UAV written by PrefixSumCS ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalSortParamsBuffer"),
			2 * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalSortParamsBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalSortParamsBufferSRV = RHICmdList.CreateShaderResourceView(
			GlobalSortParamsBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
		GlobalSortParamsBufferUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalSortParamsBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	AllocatedCount = NewAllocatedCount;
	AllocatedNumTiles = NewNumTiles;

	// Invalidate cache since buffers changed
	bHasCachedSortData = false;
}

void FGaussianGlobalAccumulator::EnsureCompactionBuffersAllocated(FRHICommandListBase& RHICmdList)
{
	if (bCompactionBuffersAllocated)
	{
		return;
	}

	UE_LOG(LogTemp, Verbose, TEXT("GaussianGlobalAccumulator: Allocating compaction prefix-sum buffers (MaxProxies=%u)"), MAX_PROXY_COUNT);

	const uint32 UintStride = sizeof(uint32);

	// --- GlobalVisibleCountArray: [MAX_PROXY_COUNT] uints ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalVisibleCountArrayBuffer"),
			MAX_PROXY_COUNT * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalVisibleCountArrayBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalVisibleCountArrayBufferUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalVisibleCountArrayBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
		GlobalVisibleCountArrayBufferSRV = RHICmdList.CreateShaderResourceView(
			GlobalVisibleCountArrayBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// --- GlobalBaseOffsets: [MAX_PROXY_COUNT + 1] uints ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalBaseOffsetsBuffer"),
			(MAX_PROXY_COUNT + 1) * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalBaseOffsetsBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalBaseOffsetsBufferUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalBaseOffsetsBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
		GlobalBaseOffsetsBufferSRV = RHICmdList.CreateShaderResourceView(
			GlobalBaseOffsetsBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// --- GlobalCalcDistIndirectArgs: [3] uints ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalCalcDistIndirectArgsBuffer"),
			3 * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_DrawIndirect | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalCalcDistIndirectArgsBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalCalcDistIndirectArgsBufferUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalCalcDistIndirectArgsBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// --- GlobalSortIndirectArgsGlobal: [3] uints ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalSortIndirectArgsGlobalBuffer"),
			3 * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_DrawIndirect | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalSortIndirectArgsGlobalBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalSortIndirectArgsGlobalBufferUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalSortIndirectArgsGlobalBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// --- GlobalDrawIndirectArgs: [5] uints ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalDrawIndirectArgsBuffer"),
			5 * UintStride,
			UintStride,
			BUF_UnorderedAccess | BUF_DrawIndirect | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		GlobalDrawIndirectArgsBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		GlobalDrawIndirectArgsBufferUAV = RHICmdList.CreateUnorderedAccessView(
			GlobalDrawIndirectArgsBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	bCompactionBuffersAllocated = true;
}

void FGaussianGlobalAccumulator::Release()
{
	GlobalViewDataBuffer.SafeRelease();
	GlobalViewDataBufferUAV.SafeRelease();
	GlobalViewDataBufferSRV.SafeRelease();

	GlobalSortDistanceBuffer.SafeRelease();
	GlobalSortDistanceBufferUAV.SafeRelease();
	GlobalSortDistanceBufferAlt.SafeRelease();
	GlobalSortDistanceBufferAltUAV.SafeRelease();

	GlobalSortKeysBuffer.SafeRelease();
	GlobalSortKeysBufferUAV.SafeRelease();
	GlobalSortKeysBufferSRV.SafeRelease();
	GlobalSortKeysBufferAlt.SafeRelease();
	GlobalSortKeysBufferAltUAV.SafeRelease();

	GlobalRadixHistogramBuffer.SafeRelease();
	GlobalRadixHistogramBufferUAV.SafeRelease();

	GlobalRadixDigitOffsetBuffer.SafeRelease();
	GlobalRadixDigitOffsetBufferUAV.SafeRelease();

	GlobalSortParamsBuffer.SafeRelease();
	GlobalSortParamsBufferSRV.SafeRelease();
	GlobalSortParamsBufferUAV.SafeRelease();

	// Compaction prefix-sum buffers
	GlobalVisibleCountArrayBuffer.SafeRelease();
	GlobalVisibleCountArrayBufferUAV.SafeRelease();
	GlobalVisibleCountArrayBufferSRV.SafeRelease();

	GlobalBaseOffsetsBuffer.SafeRelease();
	GlobalBaseOffsetsBufferUAV.SafeRelease();
	GlobalBaseOffsetsBufferSRV.SafeRelease();

	GlobalCalcDistIndirectArgsBuffer.SafeRelease();
	GlobalCalcDistIndirectArgsBufferUAV.SafeRelease();

	GlobalSortIndirectArgsGlobalBuffer.SafeRelease();
	GlobalSortIndirectArgsGlobalBufferUAV.SafeRelease();

	GlobalDrawIndirectArgsBuffer.SafeRelease();
	GlobalDrawIndirectArgsBufferUAV.SafeRelease();

	AllocatedCount = 0;
	AllocatedNumTiles = 0;
	bHasCachedSortData = false;
	bCompactionBuffersAllocated = false;
}
