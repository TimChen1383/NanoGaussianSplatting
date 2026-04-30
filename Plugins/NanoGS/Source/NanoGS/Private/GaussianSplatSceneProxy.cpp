// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatSceneProxy.h"
#include "GaussianSplatComponent.h"
#include "GaussianSplatAsset.h"
#include "GaussianSplatRenderData.h"
#include "GaussianSplatViewExtension.h"
#include "NanoGSRHICompat.h"
#include "Engine/Texture2D.h"
#include "RHICommandList.h"
#include "RenderGraphBuilder.h"
#include "SceneView.h"
#include "SceneManagement.h"
#include "DynamicMeshBuilder.h"
#include "Materials/Material.h"
#include "EngineUtils.h"

//////////////////////////////////////////////////////////////////////////
// FGaussianSplatGPUResources

FGaussianSplatGPUResources::FGaussianSplatGPUResources()
{
}

FGaussianSplatGPUResources::~FGaussianSplatGPUResources()
{
}

void FGaussianSplatGPUResources::Initialize(UGaussianSplatAsset* Asset)
{
	if (!Asset || !Asset->IsValid())
	{
		return;
	}

	// Get or create shared render data (CPU-side cached data, loaded once per asset)
	SharedData = Asset->GetOrCreateRenderData();
	if (!SharedData.IsValid() || !SharedData->IsInitialized())
	{
		return;
	}

	// Copy metadata from shared render data
	SplatCount = SharedData->SplatCount;
	PositionFormat = SharedData->PositionFormat;
	SHBands = SharedData->SHBands;
	bEnableNanite = SharedData->bEnableNanite;
	bHasClusterData = SharedData->bHasClusterData;
	ClusterCount = SharedData->ClusterCount;
	LeafClusterCount = SharedData->LeafClusterCount;
	LODSplatCount = SharedData->LODSplatCount;
	bHasLODSplats = SharedData->bHasLODSplats;

	// Initialize render resource
	if (!bInitialized)
	{
		InitResource(FRHICommandListImmediate::Get());
		bInitialized = true;
	}
}

void FGaussianSplatGPUResources::InitRHI(FRHICommandListBase& RHICmdList)
{
	if (SplatCount <= 0 || !SharedData.IsValid())
	{
		return;
	}

	// Create shared GPU buffers (only runs once per asset, thread-safe)
	SharedData->CreateGPUBuffers(RHICmdList);

	// Copy shared GPU buffer refs (just ref-count increments, no GPU allocation)
	PackedSplatBuffer = SharedData->PackedSplatBuffer;
	PackedSplatBufferSRV = SharedData->PackedSplatBufferSRV;
	SHBuffer = SharedData->SHBuffer;
	SHBufferSRV = SharedData->SHBufferSRV;
	ChunkBuffer = SharedData->ChunkBuffer;
	ChunkBufferSRV = SharedData->ChunkBufferSRV;
	IndexBuffer = SharedData->IndexBuffer;
	ClusterBuffer = SharedData->ClusterBuffer;
	ClusterBufferSRV = SharedData->ClusterBufferSRV;
	SplatClusterIndexBuffer = SharedData->SplatClusterIndexBuffer;
	SplatClusterIndexBufferSRV = SharedData->SplatClusterIndexBufferSRV;

	// TotalSplatCount is needed by per-instance buffer sizing
	TotalSplatCount = SplatCount;

	// Create dummy white texture (tiny, per-proxy is fine)
	CreateDummyWhiteTexture(RHICmdList);

	// Create per-instance buffers (cluster visibility, compaction, sort args, etc.)
	CreatePerInstanceBuffers(RHICmdList);

	int32 PerInstanceBufferCount = 0;
	if (bSupportsCompaction) PerInstanceBufferCount += 15; // cluster/compaction/sort buffers
	if (bSupportsIndirectDraw) PerInstanceBufferCount += 1;
	UE_LOG(LogTemp, Verbose, TEXT("GaussianSplatGPUResources: Created %d per-instance buffers (shared data: %s)"),
		PerInstanceBufferCount, *SharedData->GetAssetName());
}

void FGaussianSplatGPUResources::ReleaseRHI()
{
	// Release shared buffer refs (just ref-count decrements, actual GPU memory
	// is freed when SharedData releases its refs)
	PackedSplatBuffer.SafeRelease();
	PackedSplatBufferSRV.SafeRelease();
	SHBuffer.SafeRelease();
	SHBufferSRV.SafeRelease();
	ChunkBuffer.SafeRelease();
	ChunkBufferSRV.SafeRelease();
	IndexBuffer.SafeRelease();
	ClusterBuffer.SafeRelease();
	ClusterBufferSRV.SafeRelease();
	SplatClusterIndexBuffer.SafeRelease();
	SplatClusterIndexBufferSRV.SafeRelease();

	// Release legacy buffers
	PositionBuffer.SafeRelease();
	PositionBufferSRV.SafeRelease();
	OtherDataBuffer.SafeRelease();
	OtherDataBufferSRV.SafeRelease();

	// Release per-instance buffers
	ViewDataBuffer.SafeRelease();
	ViewDataBufferUAV.SafeRelease();
	ViewDataBufferSRV.SafeRelease();
	SortDistanceBuffer.SafeRelease();
	SortDistanceBufferUAV.SafeRelease();
	SortDistanceBufferSRV.SafeRelease();
	SortKeysBuffer.SafeRelease();
	SortKeysBufferUAV.SafeRelease();
	SortKeysBufferSRV.SafeRelease();
	SortKeysBufferAlt.SafeRelease();
	SortKeysBufferAltUAV.SafeRelease();
	SortKeysBufferAltSRV.SafeRelease();
	SortDistanceBufferAlt.SafeRelease();
	SortDistanceBufferAltUAV.SafeRelease();
	RadixHistogramBuffer.SafeRelease();
	RadixHistogramBufferUAV.SafeRelease();
	RadixDigitOffsetBuffer.SafeRelease();
	RadixDigitOffsetBufferUAV.SafeRelease();
	SortIndirectArgsBuffer.SafeRelease();
	SortIndirectArgsBufferUAV.SafeRelease();
	SortParamsBuffer.SafeRelease();
	SortParamsBufferUAV.SafeRelease();
	SortParamsBufferSRV.SafeRelease();
	ColorTexture.SafeRelease();
	ColorTextureSRV.SafeRelease();
	DummyWhiteTexture.SafeRelease();
	DummyWhiteTextureSRV.SafeRelease();
	VisibleClusterBuffer.SafeRelease();
	VisibleClusterBufferUAV.SafeRelease();
	VisibleClusterBufferSRV.SafeRelease();
	VisibleClusterCountBuffer.SafeRelease();
	VisibleClusterCountBufferUAV.SafeRelease();
	VisibleClusterCountBufferSRV.SafeRelease();

	// Release LOD splat buffers
	LODSplatBuffer.SafeRelease();
	LODSplatBufferSRV.SafeRelease();

	// Release indirect draw buffers
	IndirectDrawArgsBuffer.SafeRelease();
	IndirectDrawArgsBufferUAV.SafeRelease();

	// Release cluster visibility integration buffers
	ClusterVisibilityBitmap.SafeRelease();
	ClusterVisibilityBitmapUAV.SafeRelease();
	ClusterVisibilityBitmapSRV.SafeRelease();
	SelectedClusterBuffer.SafeRelease();
	SelectedClusterBufferUAV.SafeRelease();
	SelectedClusterBufferSRV.SafeRelease();

	// Release LOD cluster tracking buffers
	LODClusterBuffer.SafeRelease();
	LODClusterBufferUAV.SafeRelease();
	LODClusterBufferSRV.SafeRelease();
	LODClusterCountBuffer.SafeRelease();
	LODClusterCountBufferUAV.SafeRelease();
	LODClusterCountBufferSRV.SafeRelease();
	LODClusterSelectedBitmap.SafeRelease();
	LODClusterSelectedBitmapUAV.SafeRelease();
	LODClusterSelectedBitmapSRV.SafeRelease();
	LODSplatTotalBuffer.SafeRelease();
	LODSplatTotalBufferUAV.SafeRelease();
	LODSplatTotalBufferSRV.SafeRelease();

	// Release GPU-driven LOD rendering buffers
	LODSplatClusterIndexBuffer.SafeRelease();
	LODSplatClusterIndexBufferSRV.SafeRelease();
	LODSplatOutputCountBuffer.SafeRelease();
	LODSplatOutputCountBufferUAV.SafeRelease();
	LODSplatOutputCountBufferSRV.SafeRelease();

	// Release splat compaction buffers
	CompactedSplatIndicesBuffer.SafeRelease();
	CompactedSplatIndicesBufferUAV.SafeRelease();
	CompactedSplatIndicesBufferSRV.SafeRelease();
	VisibleSplatCountBuffer.SafeRelease();
	VisibleSplatCountBufferUAV.SafeRelease();
	VisibleSplatCountBufferSRV.SafeRelease();
	IndirectDispatchArgsBuffer.SafeRelease();
	IndirectDispatchArgsBufferUAV.SafeRelease();

	// Release shared data reference
	SharedData.Reset();

	bInitialized = false;
}

void FGaussianSplatGPUResources::CreateDummyWhiteTexture(FRHICommandListBase& RHICmdList)
{
	// Create a 1x1 white texture
	const FRHITextureCreateDesc TextureDesc =
		FRHITextureCreateDesc::Create2D(TEXT("GaussianSplatDummyWhiteTexture"), 1, 1, PF_R8G8B8A8)
		.SetFlags(ETextureCreateFlags::ShaderResource)
		.SetInitialState(ERHIAccess::SRVMask);

	DummyWhiteTexture = RHICreateTexture(TextureDesc);

	// Fill with white color
	uint32 WhitePixel = 0xFFFFFFFF;  // RGBA = (255, 255, 255, 255)
	FUpdateTextureRegion2D Region(0, 0, 0, 0, 1, 1);
	RHIUpdateTexture2D(DummyWhiteTexture, 0, Region, sizeof(uint32), (const uint8*)&WhitePixel);

	// Create SRV
	DummyWhiteTextureSRV = RHICmdList.CreateShaderResourceView(
		DummyWhiteTexture,
		FRHIViewDesc::CreateTextureSRV()
			.SetDimension(ETextureDimension::Texture2D));
}

void FGaussianSplatGPUResources::CreatePerInstanceBuffers(FRHICommandListBase& RHICmdList)
{
	// When Nanite is disabled (no cluster data), create dummy buffers for shader binding
	if (!bHasClusterData)
	{
		// Dummy ClusterVisibilityBitmap (1 uint)
		{
			const uint32 BufferSize = sizeof(uint32);
			FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
				TEXT("GaussianClusterVisibilityBitmapDummy"),
				BufferSize,
				sizeof(uint32),
				BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
				.SetInitialState(ERHIAccess::SRVMask);
			ClusterVisibilityBitmap = NanoGSCreateBuffer(RHICmdList, Desc);

			uint32 Zero = 0;
			void* Data = RHICmdList.LockBuffer(ClusterVisibilityBitmap, 0, BufferSize, RLM_WriteOnly);
			FMemory::Memcpy(Data, &Zero, BufferSize);
			RHICmdList.UnlockBuffer(ClusterVisibilityBitmap);

			ClusterVisibilityBitmapSRV = RHICmdList.CreateShaderResourceView(
				ClusterVisibilityBitmap, FRHIViewDesc::CreateBufferSRV()
					.SetType(FRHIViewDesc::EBufferType::Structured)
					.SetStride(sizeof(uint32)));
		}

		// Dummy LODClusterSelectedBitmap (1 uint)
		{
			const uint32 BufferSize = sizeof(uint32);
			FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
				TEXT("GaussianLODClusterSelectedBitmapDummy"),
				BufferSize,
				sizeof(uint32),
				BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
				.SetInitialState(ERHIAccess::SRVMask);
			LODClusterSelectedBitmap = NanoGSCreateBuffer(RHICmdList, Desc);

			uint32 Zero = 0;
			void* Data = RHICmdList.LockBuffer(LODClusterSelectedBitmap, 0, BufferSize, RLM_WriteOnly);
			FMemory::Memcpy(Data, &Zero, BufferSize);
			RHICmdList.UnlockBuffer(LODClusterSelectedBitmap);

			LODClusterSelectedBitmapSRV = RHICmdList.CreateShaderResourceView(
				LODClusterSelectedBitmap, FRHIViewDesc::CreateBufferSRV()
					.SetType(FRHIViewDesc::EBufferType::Structured)
					.SetStride(sizeof(uint32)));
		}

		// Dummy SelectedClusterBuffer (1 uint)
		{
			const uint32 BufferSize = sizeof(uint32);
			FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
				TEXT("GaussianSelectedClusterBufferDummy"),
				BufferSize,
				sizeof(uint32),
				BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
				.SetInitialState(ERHIAccess::SRVMask);
			SelectedClusterBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

			uint32 Zero = 0;
			void* Data = RHICmdList.LockBuffer(SelectedClusterBuffer, 0, BufferSize, RLM_WriteOnly);
			FMemory::Memcpy(Data, &Zero, BufferSize);
			RHICmdList.UnlockBuffer(SelectedClusterBuffer);

			SelectedClusterBufferSRV = RHICmdList.CreateShaderResourceView(
				SelectedClusterBuffer, FRHIViewDesc::CreateBufferSRV()
					.SetType(FRHIViewDesc::EBufferType::Structured)
					.SetStride(sizeof(uint32)));
		}

		// Dummy CompactedSplatIndicesBuffer (1 uint) - needed even when UseCompaction=0
		{
			const uint32 BufferSize = sizeof(uint32);
			FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
				TEXT("GaussianCompactedSplatIndicesBufferDummy"),
				BufferSize,
				sizeof(uint32),
				BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
				.SetInitialState(ERHIAccess::SRVMask);
			CompactedSplatIndicesBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

			uint32 Zero = 0;
			void* Data = RHICmdList.LockBuffer(CompactedSplatIndicesBuffer, 0, BufferSize, RLM_WriteOnly);
			FMemory::Memcpy(Data, &Zero, BufferSize);
			RHICmdList.UnlockBuffer(CompactedSplatIndicesBuffer);

			CompactedSplatIndicesBufferSRV = RHICmdList.CreateShaderResourceView(
				CompactedSplatIndicesBuffer, FRHIViewDesc::CreateBufferSRV()
					.SetType(FRHIViewDesc::EBufferType::Structured)
					.SetStride(sizeof(uint32)));
		}

		return;
	}

	// Create visible cluster buffer (dynamic, written by culling shader)
	// Size = max possible visible clusters (all clusters could be visible)
	{
		const uint32 BufferSize = ClusterCount * sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianVisibleClusterBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		VisibleClusterBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

		VisibleClusterBufferUAV = RHICmdList.CreateUnorderedAccessView(
			VisibleClusterBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		VisibleClusterBufferSRV = RHICmdList.CreateShaderResourceView(
			VisibleClusterBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// Create visible cluster count buffer (single uint, atomic counter)
	{
		const uint32 BufferSize = sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianVisibleClusterCountBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		VisibleClusterCountBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

		VisibleClusterCountBufferUAV = RHICmdList.CreateUnorderedAccessView(
			VisibleClusterCountBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		VisibleClusterCountBufferSRV = RHICmdList.CreateShaderResourceView(
			VisibleClusterCountBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// Create indirect draw argument buffer for GPU-driven rendering
	// Structure: IndexCountPerInstance(4), InstanceCount(4), StartIndexLocation(4), BaseVertexLocation(4), StartInstanceLocation(4)
	// Total: 20 bytes, but we use 32 bytes for alignment
	{
		const uint32 BufferSize = 32;  // 5 uints + padding for alignment
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianIndirectDrawArgsBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_DrawIndirect | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::IndirectArgs);
		IndirectDrawArgsBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

		// Initialize with default values
		// IndexCountPerInstance = 6 (2 triangles per quad)
		// InstanceCount = SplatCount (will be updated by culling shader)
		// StartIndexLocation = 0
		// BaseVertexLocation = 0
		// StartInstanceLocation = 0
		uint32 InitData[8] = { 6, (uint32)SplatCount, 0, 0, 0, 0, 0, 0 };
		void* Data = RHICmdList.LockBuffer(IndirectDrawArgsBuffer, 0, BufferSize, RLM_WriteOnly);
		FMemory::Memcpy(Data, InitData, BufferSize);
		RHICmdList.UnlockBuffer(IndirectDrawArgsBuffer);

		IndirectDrawArgsBufferUAV = RHICmdList.CreateUnorderedAccessView(
			IndirectDrawArgsBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));

		bSupportsIndirectDraw = true;
		UE_LOG(LogTemp, Verbose, TEXT("GaussianSplatGPUResources: Created indirect draw buffer"));
	}

	// Create cluster visibility bitmap buffer
	// One bit per cluster, rounded up to uint32 boundary
	{
		uint32 BitmapSize = FMath::DivideAndRoundUp(ClusterCount, 32) * sizeof(uint32);
		BitmapSize = FMath::Max(BitmapSize, (uint32)sizeof(uint32));  // At least one uint32

		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianClusterVisibilityBitmap"),
			BitmapSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		ClusterVisibilityBitmap = NanoGSCreateBuffer(RHICmdList, Desc);

		ClusterVisibilityBitmapUAV = RHICmdList.CreateUnorderedAccessView(
			ClusterVisibilityBitmap, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		ClusterVisibilityBitmapSRV = RHICmdList.CreateShaderResourceView(
			ClusterVisibilityBitmap, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));

		UE_LOG(LogTemp, Verbose, TEXT("GaussianSplatGPUResources: Created cluster visibility bitmap (%d bytes for %d clusters)"), BitmapSize, ClusterCount);
	}

	// Create selected cluster buffer for Nanite-style debug visualization
	// One entry per leaf cluster, stores which cluster ID is selected based on LOD
	{
		uint32 BufferSize = LeafClusterCount * sizeof(uint32);
		BufferSize = FMath::Max(BufferSize, (uint32)sizeof(uint32));  // At least one uint32

		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSelectedClusterBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		SelectedClusterBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

		SelectedClusterBufferUAV = RHICmdList.CreateUnorderedAccessView(
			SelectedClusterBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		SelectedClusterBufferSRV = RHICmdList.CreateShaderResourceView(
			SelectedClusterBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));

		UE_LOG(LogTemp, Verbose, TEXT("GaussianSplatGPUResources: Created selected cluster buffer (%d bytes for %d leaf clusters)"), BufferSize, LeafClusterCount);
	}

	// Create LOD cluster tracking buffers for LOD rendering
	// LOD cluster buffer - stores unique parent cluster indices
	{
		uint32 BufferSize = ClusterCount * sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianLODClusterBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		LODClusterBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

		LODClusterBufferUAV = RHICmdList.CreateUnorderedAccessView(
			LODClusterBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		LODClusterBufferSRV = RHICmdList.CreateShaderResourceView(
			LODClusterBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// LOD cluster count buffer (atomic counter)
	{
		uint32 BufferSize = sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianLODClusterCountBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		LODClusterCountBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

		LODClusterCountBufferUAV = RHICmdList.CreateUnorderedAccessView(
			LODClusterCountBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		LODClusterCountBufferSRV = RHICmdList.CreateShaderResourceView(
			LODClusterCountBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// LOD cluster selected bitmap (same size as cluster visibility bitmap)
	// Also needs SRV for GPU-driven LOD shader to read
	{
		uint32 BitmapSize = FMath::DivideAndRoundUp(ClusterCount, 32) * sizeof(uint32);
		BitmapSize = FMath::Max(BitmapSize, (uint32)sizeof(uint32));

		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianLODClusterSelectedBitmap"),
			BitmapSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		LODClusterSelectedBitmap = NanoGSCreateBuffer(RHICmdList, Desc);

		LODClusterSelectedBitmapUAV = RHICmdList.CreateUnorderedAccessView(
			LODClusterSelectedBitmap, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		LODClusterSelectedBitmapSRV = RHICmdList.CreateShaderResourceView(
			LODClusterSelectedBitmap, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// LOD splat total buffer (atomic counter for total LOD splats)
	{
		uint32 BufferSize = sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianLODSplatTotalBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		LODSplatTotalBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

		LODSplatTotalBufferUAV = RHICmdList.CreateUnorderedAccessView(
			LODSplatTotalBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		LODSplatTotalBufferSRV = RHICmdList.CreateShaderResourceView(
			LODSplatTotalBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// LOD splat output count buffer (atomic counter for valid LOD splat output)
	{
		uint32 BufferSize = sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianLODSplatOutputCountBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		LODSplatOutputCountBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

		LODSplatOutputCountBufferUAV = RHICmdList.CreateUnorderedAccessView(
			LODSplatOutputCountBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		LODSplatOutputCountBufferSRV = RHICmdList.CreateShaderResourceView(
			LODSplatOutputCountBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	UE_LOG(LogTemp, Verbose, TEXT("GaussianSplatGPUResources: Created LOD cluster tracking buffers"));

	// Create splat compaction buffers (GPU-driven work reduction)
	// CompactedSplatIndicesBuffer - stores visible splat indices
	{
		const uint32 BufferSize = TotalSplatCount * sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianCompactedSplatIndicesBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		CompactedSplatIndicesBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

		CompactedSplatIndicesBufferUAV = RHICmdList.CreateUnorderedAccessView(
			CompactedSplatIndicesBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		CompactedSplatIndicesBufferSRV = RHICmdList.CreateShaderResourceView(
			CompactedSplatIndicesBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// VisibleSplatCountBuffer - atomic counter for visible splat count
	{
		const uint32 BufferSize = sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianVisibleSplatCountBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		VisibleSplatCountBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

		VisibleSplatCountBufferUAV = RHICmdList.CreateUnorderedAccessView(
			VisibleSplatCountBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		VisibleSplatCountBufferSRV = RHICmdList.CreateShaderResourceView(
			VisibleSplatCountBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// IndirectDispatchArgsBuffer - for indirect compute dispatch
	// Format: uint3 (numGroupsX, numGroupsY, numGroupsZ)
	{
		const uint32 BufferSize = 3 * sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianIndirectDispatchArgsBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_DrawIndirect | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::IndirectArgs);
		IndirectDispatchArgsBuffer = NanoGSCreateBuffer(RHICmdList, Desc);

		// Initialize with default values (1, 1, 1)
		uint32 InitData[3] = { 1, 1, 1 };
		void* Data = RHICmdList.LockBuffer(IndirectDispatchArgsBuffer, 0, BufferSize, RLM_WriteOnly);
		FMemory::Memcpy(Data, InitData, BufferSize);
		RHICmdList.UnlockBuffer(IndirectDispatchArgsBuffer);

		IndirectDispatchArgsBufferUAV = RHICmdList.CreateUnorderedAccessView(
			IndirectDispatchArgsBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// Sort indirect args + sort params: needed by PrepareIndirectArgs in the
	// Nanite compaction path. Created here (not in CreateDynamicBuffers) because
	// per-proxy dynamic buffers are skipped under the global accumulator, but
	// PrepareIndirectArgs still runs per-proxy to set up indirect dispatch for
	// CalcViewDataCompactedGlobal.
	{
		const uint32 BufferSize = 3 * sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSortIndirectArgsBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_DrawIndirect | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		SortIndirectArgsBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		SortIndirectArgsBufferUAV = RHICmdList.CreateUnorderedAccessView(
			SortIndirectArgsBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}
	{
		const uint32 BufferSize = 2 * sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSortParamsBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		SortParamsBuffer = NanoGSCreateBuffer(RHICmdList, Desc);
		SortParamsBufferUAV = RHICmdList.CreateUnorderedAccessView(
			SortParamsBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		SortParamsBufferSRV = RHICmdList.CreateShaderResourceView(
			SortParamsBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	bSupportsCompaction = true;
	UE_LOG(LogTemp, Verbose, TEXT("GaussianSplatGPUResources: Created splat compaction buffers for %d total splats"), TotalSplatCount);
	UE_LOG(LogTemp, Verbose, TEXT("GaussianSplatGPUResources: Created cluster buffers for %d clusters"), ClusterCount);
}

//////////////////////////////////////////////////////////////////////////
// FGaussianSplatSceneProxy

//Constructor. Copies rendering parameters from the component
FGaussianSplatSceneProxy::FGaussianSplatSceneProxy(const UGaussianSplatComponent* InComponent)
	: FPrimitiveSceneProxy(InComponent)
	, CachedAsset(InComponent->SplatAsset)
	, SplatCount(InComponent->SplatAsset ? InComponent->SplatAsset->GetSplatCount() : 0)
	, SHOrder(InComponent->SHOrder)
	, OpacityScale(InComponent->OpacityScale)
	, SplatScale(InComponent->SplatScale)
	, LODErrorThreshold(InComponent->LODErrorThreshold)
	, bEnableFrustumCulling(InComponent->bEnableFrustumCulling)
{
	bWillEverBeLit = false;
}

FGaussianSplatSceneProxy::~FGaussianSplatSceneProxy()
{
}

SIZE_T FGaussianSplatSceneProxy::GetTypeHash() const
{
	static size_t UniquePointer;
	return reinterpret_cast<size_t>(&UniquePointer);
}

uint32 FGaussianSplatSceneProxy::GetMemoryFootprint() const
{
	return sizeof(*this) + GetAllocatedSize();
}

FPrimitiveViewRelevance FGaussianSplatSceneProxy::GetViewRelevance(const FSceneView* View) const
{
	FPrimitiveViewRelevance Result;
	Result.bDrawRelevance = IsShown(View);
	Result.bShadowRelevance = false; // Gaussian splats don't cast shadows (yet)
	Result.bDynamicRelevance = true;
	Result.bStaticRelevance = false;
	Result.bRenderInMainPass = true;
	Result.bUsesLightingChannels = false;
	Result.bRenderCustomDepth = ShouldRenderCustomDepth();

	return Result;
}

#if WITH_EDITOR
HHitProxy* FGaussianSplatSceneProxy::CreateHitProxies(UPrimitiveComponent* Component, TArray<TRefCountPtr<HHitProxy>>& OutHitProxies)
{
	// Let the base class create the default HActor hit proxy for the owning actor.
	HHitProxy* DefaultProxy = FPrimitiveSceneProxy::CreateHitProxies(Component, OutHitProxies);
	// Cache it so GetDynamicMeshElements can use it without touching game-thread UObjects.
	SelectionHitProxy = DefaultProxy;
	return DefaultProxy;
}
#endif

void FGaussianSplatSceneProxy::GetDynamicMeshElements(
	const TArray<const FSceneView*>& Views,
	const FSceneViewFamily& ViewFamily,
	uint32 VisibilityMap,
	FMeshElementCollector& Collector) const
{
	for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ViewIndex++)
	{
		if (VisibilityMap & (1 << ViewIndex))
		{
			FPrimitiveDrawInterface* PDI = Collector.GetPDI(ViewIndex);

			// Draw bounds when selected
			if (IsSelected())
			{
				RenderBounds(PDI, ViewFamily.EngineShowFlags, GetBounds(), true);
			}

#if WITH_EDITOR
			// During the hit proxy pass, render a solid box covering the local bounds.
			// This makes the splat selectable by clicking anywhere within its bounds in
			// the editor viewport. The box is invisible in normal rendering.
			if (ViewFamily.EngineShowFlags.HitProxies)
			{
				const FBox LocalBox = GetLocalBounds().GetBox();
				const FVector3f Min(LocalBox.Min);
				const FVector3f Max(LocalBox.Max);

				FDynamicMeshBuilder MeshBuilder(Views[ViewIndex]->GetFeatureLevel());

				// Tangent basis (arbitrary – not shaded, just needs to be valid)
				const FVector2f UV(0.f, 0.f);
				const FVector3f TX(1.f, 0.f, 0.f);
				const FVector3f TY(0.f, 1.f, 0.f);
				const FVector3f TZ(0.f, 0.f, 1.f);
				const FColor White = FColor::White;

				// 8 corners  (named by which axes are at Max: 0=Min, 1=Max)
				const int32 V000 = MeshBuilder.AddVertex(FVector3f(Min.X, Min.Y, Min.Z), UV, TX, TY, TZ, White);
				const int32 V100 = MeshBuilder.AddVertex(FVector3f(Max.X, Min.Y, Min.Z), UV, TX, TY, TZ, White);
				const int32 V010 = MeshBuilder.AddVertex(FVector3f(Min.X, Max.Y, Min.Z), UV, TX, TY, TZ, White);
				const int32 V110 = MeshBuilder.AddVertex(FVector3f(Max.X, Max.Y, Min.Z), UV, TX, TY, TZ, White);
				const int32 V001 = MeshBuilder.AddVertex(FVector3f(Min.X, Min.Y, Max.Z), UV, TX, TY, TZ, White);
				const int32 V101 = MeshBuilder.AddVertex(FVector3f(Max.X, Min.Y, Max.Z), UV, TX, TY, TZ, White);
				const int32 V011 = MeshBuilder.AddVertex(FVector3f(Min.X, Max.Y, Max.Z), UV, TX, TY, TZ, White);
				const int32 V111 = MeshBuilder.AddVertex(FVector3f(Max.X, Max.Y, Max.Z), UV, TX, TY, TZ, White);

				// -Z face
				MeshBuilder.AddTriangle(V000, V010, V100);
				MeshBuilder.AddTriangle(V010, V110, V100);
				// +Z face
				MeshBuilder.AddTriangle(V001, V101, V011);
				MeshBuilder.AddTriangle(V101, V111, V011);
				// -Y face
				MeshBuilder.AddTriangle(V000, V100, V001);
				MeshBuilder.AddTriangle(V100, V101, V001);
				// +Y face
				MeshBuilder.AddTriangle(V010, V011, V110);
				MeshBuilder.AddTriangle(V011, V111, V110);
				// -X face
				MeshBuilder.AddTriangle(V000, V001, V010);
				MeshBuilder.AddTriangle(V001, V011, V010);
				// +X face
				MeshBuilder.AddTriangle(V100, V110, V101);
				MeshBuilder.AddTriangle(V110, V111, V101);

				// Use default opaque surface material – only its depth output matters here.
				// bDisableBackfaceCulling=true so all faces are drawn regardless of winding.
				UMaterialInterface* HitMaterial = UMaterial::GetDefaultMaterial(MD_Surface);
				MeshBuilder.GetMesh(
					GetLocalToWorld(),
					HitMaterial->GetRenderProxy(),
					SDPG_World,
					/*bDisableBackfaceCulling=*/true,
					/*bReceivesDecals=*/false,
					/*bUseSelectionOutline=*/false,
					ViewIndex,
					Collector,
					SelectionHitProxy);
			}
#endif // WITH_EDITOR
		}
	}
}

//critical setup. creates FGaussianSplatGPUResources which uploads all the GPU buffers 
// registers itself with the ViewExtension
#if ENGINE_MAJOR_VERSION == 5 && ENGINE_MINOR_VERSION <= 3
void FGaussianSplatSceneProxy::CreateRenderThreadResources()
#else
void FGaussianSplatSceneProxy::CreateRenderThreadResources(FRHICommandListBase& RHICmdList)
#endif
{
#if ENGINE_MAJOR_VERSION == 5 && ENGINE_MINOR_VERSION <= 3
	FRHICommandListBase& RHICmdList = FRHICommandListImmediate::Get();
#endif

	if (CachedAsset && CachedAsset->IsValid())
	{
		GPUResources = new FGaussianSplatGPUResources();
		GPUResources->Initialize(CachedAsset);

		// Get color texture reference
		if (CachedAsset->ColorTexture)
		{
			// If platform data exists but no resource, try to create it
			FTextureResource* TextureResource = CachedAsset->ColorTexture->GetResource();
			FTexturePlatformData* PlatformData = CachedAsset->ColorTexture->GetPlatformData();
			int64 BulkDataSize = 0;
			if (PlatformData && PlatformData->Mips.Num() > 0)
			{
				BulkDataSize = PlatformData->Mips[0].BulkData.GetBulkDataSize();
			}

			if (!TextureResource && PlatformData && BulkDataSize > 0)
			{
				CachedAsset->ColorTexture->UpdateResource();
				TextureResource = CachedAsset->ColorTexture->GetResource();
			}

			if (TextureResource && TextureResource->TextureRHI)
			{
				GPUResources->ColorTexture = TextureResource->TextureRHI;
				GPUResources->ColorTextureSRV = RHICmdList.CreateShaderResourceView(
					GPUResources->ColorTexture,
					FRHIViewDesc::CreateTextureSRV()
						.SetDimension(ETextureDimension::Texture2D));
			}
		}

		// Register with view extension for rendering
		FGaussianSplatViewExtension* ViewExtension = FGaussianSplatViewExtension::Get();
		if (ViewExtension)
		{
			ViewExtension->RegisterProxy(const_cast<FGaussianSplatSceneProxy*>(this));
		}
	}
}

void FGaussianSplatSceneProxy::DestroyRenderThreadResources()
{
	// CRITICAL: Mark as pending destruction FIRST, before any other operations.
	// This prevents render commands that have already captured a pointer to this proxy
	// from accessing our resources. The atomic flag is checked by IsValidForRendering().
	MarkPendingDestruction();

	// Unregister from view extension (under lock, so no new render passes will see us)
	FGaussianSplatViewExtension* ViewExtension = FGaussianSplatViewExtension::Get();
	if (ViewExtension)
	{
		ViewExtension->UnregisterProxy(const_cast<FGaussianSplatSceneProxy*>(this));
	}

	// Flush any pending render commands that might be referencing our resources.
	// This ensures that any RDG passes that captured our proxy pointer have completed
	// before we release the GPU resources.
	// Note: We're already on the render thread, so this flushes GPU work.
	FRHICommandListImmediate& RHICmdList = FRHICommandListImmediate::Get();
	RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);

	if (GPUResources)
	{
		GPUResources->ReleaseResource();
		delete GPUResources;
		GPUResources = nullptr;
	}
}

void FGaussianSplatSceneProxy::TryInitializeColorTexture(FRHICommandListBase& RHICmdList)
{
	// Safety check: don't access resources if proxy is being destroyed
	if (bPendingDestruction.load(std::memory_order_acquire))
	{
		return;
	}

	if (!GPUResources || GPUResources->ColorTextureSRV.IsValid())
	{
		// Already initialized or no resources
		return;
	}

	if (!CachedAsset || !CachedAsset->ColorTexture)
	{
		return;
	}

	FTextureResource* TextureResource = CachedAsset->ColorTexture->GetResource();

	// If resource doesn't exist but platform data does, try to create it
	if (!TextureResource)
	{
		FTexturePlatformData* PlatformData = CachedAsset->ColorTexture->GetPlatformData();
		int64 BulkDataSize = 0;
		if (PlatformData && PlatformData->Mips.Num() > 0)
		{
			BulkDataSize = PlatformData->Mips[0].BulkData.GetBulkDataSize();
		}

		if (PlatformData && BulkDataSize > 0)
		{
			CachedAsset->ColorTexture->UpdateResource();
			TextureResource = CachedAsset->ColorTexture->GetResource();
		}
	}

	if (TextureResource && TextureResource->TextureRHI)
	{
		GPUResources->ColorTexture = TextureResource->TextureRHI;
		GPUResources->ColorTextureSRV = RHICmdList.CreateShaderResourceView(
			GPUResources->ColorTexture,
			FRHIViewDesc::CreateTextureSRV()
				.SetDimension(ETextureDimension::Texture2D));
	}
}

