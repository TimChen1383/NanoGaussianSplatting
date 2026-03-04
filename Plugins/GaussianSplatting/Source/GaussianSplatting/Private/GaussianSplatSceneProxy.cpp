// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatSceneProxy.h"
#include "GaussianSplatComponent.h"
#include "GaussianSplatAsset.h"
#include "GaussianSplatViewExtension.h"
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

	SplatCount = Asset->GetSplatCount();

	// Store the position format from the asset (critical for shader to read correctly)
	PositionFormat = Asset->PositionFormat;

	// --- Packed Splat Data (16 bytes/splat) ---
	// Pack position (float16×3), rotation (octahedral 3 bytes), scale (log uint8×3),
	// and color+opacity (uint8×4) into a single buffer, replacing separate
	// PositionBuffer (12 B), OtherDataBuffer (28 B), and ColorTexture (~8 B).
	{
		// Read source data from asset bulk data
		TArray<uint8> RawPositionData;
		TArray<uint8> RawOtherData;
		TArray<uint8> RawColorTextureData;
		Asset->GetPositionData(RawPositionData);
		Asset->GetOtherData(RawOtherData);
		Asset->GetColorTextureData(RawColorTextureData);

		const int32 ColorTexWidth = Asset->ColorTextureWidth;
		const int32 ColorTexHeight = Asset->ColorTextureHeight;
		const FFloat16Color* ColorPixels = nullptr;
		bool bHasColor = (RawColorTextureData.Num() > 0 && ColorTexWidth > 0 && ColorTexHeight > 0);
		if (bHasColor)
		{
			ColorPixels = reinterpret_cast<const FFloat16Color*>(RawColorTextureData.GetData());
		}

		const int32 PosBytesPerSplat = 12;   // 3 × float32
		const int32 OtherBytesPerSplat = 28; // 4 × float32 (quat) + 3 × float32 (scale)

		CachedPackedSplatData.SetNumUninitialized(SplatCount * GaussianSplattingConstants::PackedSplatStride);
		uint32* PackedPtr = reinterpret_cast<uint32*>(CachedPackedSplatData.GetData());
		const float* PosFloats = reinterpret_cast<const float*>(RawPositionData.GetData());
		const float* OtherFloats = reinterpret_cast<const float*>(RawOtherData.GetData());

		for (int32 i = 0; i < SplatCount; i++)
		{
			// Read position (3 × float32)
			FVector3f Position(
				PosFloats[i * 3 + 0],
				PosFloats[i * 3 + 1],
				PosFloats[i * 3 + 2]);

			// Read rotation (quaternion XYZW) and scale (3 × float32)
			const float* OtherBase = OtherFloats + i * 7; // 28 bytes / 4 = 7 floats
			FQuat4f Rotation(OtherBase[0], OtherBase[1], OtherBase[2], OtherBase[3]);
			FVector3f Scale(OtherBase[4], OtherBase[5], OtherBase[6]);

			// Read color from Morton-swizzled texture
			float ColorR = 1.0f, ColorG = 1.0f, ColorB = 1.0f, Opacity = 1.0f;
			if (bHasColor)
			{
				int32 TexX, TexY;
				GaussianSplattingUtils::SplatIndexToTextureCoord(i, ColorTexWidth, TexX, TexY);
				if (TexY < ColorTexHeight)
				{
					const FFloat16Color& Pixel = ColorPixels[TexY * ColorTexWidth + TexX];
					ColorR = FMath::Clamp(Pixel.R.GetFloat(), 0.0f, 1.0f);
					ColorG = FMath::Clamp(Pixel.G.GetFloat(), 0.0f, 1.0f);
					ColorB = FMath::Clamp(Pixel.B.GetFloat(), 0.0f, 1.0f);
					Opacity = FMath::Clamp(Pixel.A.GetFloat(), 0.0f, 1.0f);
				}
			}

			// Pack into 4 × uint32 (16 bytes)
			GaussianSplattingUtils::PackSplatToUint4(
				Position, Rotation, Scale,
				ColorR, ColorG, ColorB, Opacity,
				&PackedPtr[i * 4]);
		}

		UE_LOG(LogTemp, Log, TEXT("GaussianSplatGPUResources: Packed %d splats into %d bytes (16 B/splat, was %d B/splat)"),
			SplatCount, CachedPackedSplatData.Num(), PosBytesPerSplat + OtherBytesPerSplat);

		// Source data no longer needed
		RawPositionData.Empty();
		RawOtherData.Empty();
		RawColorTextureData.Empty();
	}

	// Legacy: chunk data still needed for shader binding (dummy)
	CachedChunkData = Asset->ChunkData;

	// Load SH data for view-dependent rendering (f_rest coefficients)
	if (Asset->SHBands > 0)
	{
		Asset->GetSHData(CachedSHData);
		SHBands = Asset->SHBands;
		UE_LOG(LogTemp, Log, TEXT("GaussianSplatGPUResources: Loaded %lld bytes of SH data (bands=%d)"),
			CachedSHData.Num(), SHBands);
	}

	// Set Nanite enabled flag from asset
	bEnableNanite = Asset->IsNaniteEnabled();

	// Cache cluster hierarchy data if Nanite is enabled and available
	if (bEnableNanite && Asset->HasClusterHierarchy())
	{
		const FGaussianClusterHierarchy& Hierarchy = Asset->GetClusterHierarchy();
		Hierarchy.ToGPUClusters(CachedClusterData);
		ClusterCount = CachedClusterData.Num();
		LeafClusterCount = Hierarchy.NumLeafClusters;
		bHasClusterData = true;

		// UNIFIED APPROACH: LOD splats are now appended to main buffer
		// TotalSplatCount = original splats, TotalLODSplatCount = LOD splats
		// SplatCount (from asset) = TotalSplatCount + TotalLODSplatCount
		const uint32 OriginalSplatCount = Hierarchy.TotalSplatCount;
		LODSplatCount = Hierarchy.TotalLODSplatCount;
		bHasLODSplats = (LODSplatCount > 0);

		// Build unified splat-to-cluster index mapping
		// Each splat needs to know which cluster it belongs to for visibility checks
		CachedSplatClusterIndices.SetNumZeroed(SplatCount);

		// Map original splats to their leaf clusters
		for (int32 ClusterIdx = 0; ClusterIdx < Hierarchy.Clusters.Num(); ++ClusterIdx)
		{
			const FGaussianCluster& Cluster = Hierarchy.Clusters[ClusterIdx];
			// Only leaf clusters contain original splats
			if (Cluster.IsLeaf())
			{
				for (uint32 i = 0; i < Cluster.SplatCount; ++i)
				{
					uint32 SplatIdx = Cluster.SplatStartIndex + i;
					if (SplatIdx < OriginalSplatCount)
					{
						CachedSplatClusterIndices[SplatIdx] = ClusterIdx;
					}
				}
			}
		}

		// Map LOD splats to their parent clusters (unified buffer approach)
		// LOD splats are at indices [OriginalSplatCount, SplatCount)
		// Cluster.LODSplatStartIndex now points into the unified buffer
		for (int32 ClusterIdx = 0; ClusterIdx < Hierarchy.Clusters.Num(); ++ClusterIdx)
		{
			const FGaussianCluster& Cluster = Hierarchy.Clusters[ClusterIdx];
			// Only non-leaf clusters have LOD splats
			if (!Cluster.IsLeaf() && Cluster.LODSplatCount > 0)
			{
				for (uint32 i = 0; i < Cluster.LODSplatCount; ++i)
				{
					uint32 SplatIdx = Cluster.LODSplatStartIndex + i;
					if (SplatIdx < static_cast<uint32>(SplatCount))
					{
						CachedSplatClusterIndices[SplatIdx] = ClusterIdx;
					}
				}
			}
		}

		UE_LOG(LogTemp, Log, TEXT("GaussianSplatGPUResources: Loaded %d clusters (%d leaf), %d original + %d LOD = %d total splats (unified buffer)"),
			ClusterCount, LeafClusterCount, OriginalSplatCount, LODSplatCount, SplatCount);
	}
	else
	{
		bHasClusterData = false;
		ClusterCount = 0;
		LeafClusterCount = 0;
		bHasLODSplats = false;
		LODSplatCount = 0;
	}

	// Initialize render resource
	if (!bInitialized)
	{
		InitResource(FRHICommandListImmediate::Get());
		bInitialized = true;
	}
}

void FGaussianSplatGPUResources::InitRHI(FRHICommandListBase& RHICmdList)
{
	if (SplatCount <= 0)
	{
		return;
	}

	CreateStaticBuffers(RHICmdList);

	// TotalSplatCount is needed by CreateClusterBuffers (compaction buffer sizing).
	TotalSplatCount = SplatCount;

	// NOTE: Per-proxy dynamic buffers (ViewData, sort, histogram) are NOT created here.
	// The global accumulator owns all working buffers, saving ~49 MB per proxy.

	CreateIndexBuffer(RHICmdList);
	CreateClusterBuffers(RHICmdList);
}

void FGaussianSplatGPUResources::ReleaseRHI()
{
	PackedSplatBuffer.SafeRelease();
	PackedSplatBufferSRV.SafeRelease();
	PositionBuffer.SafeRelease();
	PositionBufferSRV.SafeRelease();
	OtherDataBuffer.SafeRelease();
	OtherDataBufferSRV.SafeRelease();
	SHBuffer.SafeRelease();
	SHBufferSRV.SafeRelease();
	ChunkBuffer.SafeRelease();
	ChunkBufferSRV.SafeRelease();
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
	IndexBuffer.SafeRelease();
	ColorTexture.SafeRelease();
	ColorTextureSRV.SafeRelease();
	DummyWhiteTexture.SafeRelease();
	DummyWhiteTextureSRV.SafeRelease();

	// Release cluster buffers
	ClusterBuffer.SafeRelease();
	ClusterBufferSRV.SafeRelease();
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
	SplatClusterIndexBuffer.SafeRelease();
	SplatClusterIndexBufferSRV.SafeRelease();
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

	bInitialized = false;
}

void FGaussianSplatGPUResources::CreateStaticBuffers(FRHICommandListBase& RHICmdList)
{
	// Packed splat buffer (16 bytes/splat): replaces PositionBuffer + OtherDataBuffer + ColorTexture
	if (CachedPackedSplatData.Num() > 0)
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianPackedSplatBuffer"),
			CachedPackedSplatData.Num(),
			0,
			BUF_Static | BUF_ShaderResource | BUF_ByteAddressBuffer)
			.SetInitialState(ERHIAccess::SRVMask);
		PackedSplatBuffer = RHICmdList.CreateBuffer(Desc);

		void* Data = RHICmdList.LockBuffer(PackedSplatBuffer, 0, CachedPackedSplatData.Num(), RLM_WriteOnly);
		FMemory::Memcpy(Data, CachedPackedSplatData.GetData(), CachedPackedSplatData.Num());
		RHICmdList.UnlockBuffer(PackedSplatBuffer);

		PackedSplatBufferSRV = RHICmdList.CreateShaderResourceView(
			PackedSplatBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Raw));
	}

	// SH buffer for view-dependent rendering (f_rest coefficients)
	// Always create at least a dummy buffer for shader binding
	{
		uint32 SHDataSize = CachedSHData.Num();
		if (SHDataSize == 0)
		{
			SHDataSize = 16;  // Create small dummy buffer
		}

		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSHBuffer"),
			SHDataSize,
			0,
			BUF_Static | BUF_ShaderResource | BUF_ByteAddressBuffer)
			.SetInitialState(ERHIAccess::SRVMask);
		SHBuffer = RHICmdList.CreateBuffer(Desc);

		void* Data = RHICmdList.LockBuffer(SHBuffer, 0, SHDataSize, RLM_WriteOnly);
		if (CachedSHData.Num() > 0)
		{
			FMemory::Memcpy(Data, CachedSHData.GetData(), CachedSHData.Num());
		}
		else
		{
			FMemory::Memzero(Data, SHDataSize);  // Zero-initialize dummy
		}
		RHICmdList.UnlockBuffer(SHBuffer);

		SHBufferSRV = RHICmdList.CreateShaderResourceView(
			SHBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Raw));

		UE_LOG(LogTemp, Log, TEXT("GaussianSplatGPUResources: Created SH buffer (%u bytes, bands=%d)"),
			SHDataSize, SHBands);
	}

	// Chunk buffer - Always create at least a dummy buffer for shader binding
	{
		uint32 ChunkCount = CachedChunkData.Num();
		if (ChunkCount == 0)
		{
			ChunkCount = 1;  // Create dummy entry
		}

		const uint32 ChunkSize = ChunkCount * sizeof(FGaussianChunkInfo);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianChunkBuffer"),
			ChunkSize,
			sizeof(FGaussianChunkInfo),
			BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::SRVMask);
		ChunkBuffer = RHICmdList.CreateBuffer(Desc);

		void* Data = RHICmdList.LockBuffer(ChunkBuffer, 0, ChunkSize, RLM_WriteOnly);
		if (CachedChunkData.Num() > 0)
		{
			FMemory::Memcpy(Data, CachedChunkData.GetData(), ChunkSize);
		}
		else
		{
			FMemory::Memzero(Data, ChunkSize);  // Zero-initialize dummy entry
		}
		RHICmdList.UnlockBuffer(ChunkBuffer);

		ChunkBufferSRV = RHICmdList.CreateShaderResourceView(
			ChunkBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(FGaussianChunkInfo)));
	}

	// Clear cached data
	CachedPackedSplatData.Empty();
	CachedPositionData.Empty();
	CachedOtherData.Empty();
	CachedSHData.Empty();
	CachedChunkData.Empty();
}

void FGaussianSplatGPUResources::CreateDynamicBuffers(FRHICommandListBase& RHICmdList)
{
	// Pad to power of 2 for bitonic sort
	uint32 PaddedCount = FMath::RoundUpToPowerOfTwo(TotalSplatCount);

	// View data buffer (per-frame computed data)
	// Sized to hold both original splats and LOD splats
	{
		const uint32 BufferSize = TotalSplatCount * sizeof(FGaussianSplatViewData);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianViewDataBuffer"),
			BufferSize,
			sizeof(FGaussianSplatViewData),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		ViewDataBuffer = RHICmdList.CreateBuffer(Desc);

		ViewDataBufferUAV = RHICmdList.CreateUnorderedAccessView(
			ViewDataBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(FGaussianSplatViewData)));
		ViewDataBufferSRV = RHICmdList.CreateShaderResourceView(
			ViewDataBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(FGaussianSplatViewData)));
	}

	// Sort distance buffer - sized to PaddedCount for bitonic sort
	{
		const uint32 BufferSize = PaddedCount * sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSortDistanceBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		SortDistanceBuffer = RHICmdList.CreateBuffer(Desc);

		SortDistanceBufferUAV = RHICmdList.CreateUnorderedAccessView(
			SortDistanceBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		SortDistanceBufferSRV = RHICmdList.CreateShaderResourceView(
			SortDistanceBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// Sort keys buffer - sized to PaddedCount for bitonic sort
	{
		const uint32 BufferSize = PaddedCount * sizeof(uint32);

		FRHIBufferCreateDesc Desc1 = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSortKeysBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		SortKeysBuffer = RHICmdList.CreateBuffer(Desc1);
		SortKeysBufferUAV = RHICmdList.CreateUnorderedAccessView(
			SortKeysBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		SortKeysBufferSRV = RHICmdList.CreateShaderResourceView(
			SortKeysBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));

		FRHIBufferCreateDesc Desc2 = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSortKeysBufferAlt"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		SortKeysBufferAlt = RHICmdList.CreateBuffer(Desc2);
		SortKeysBufferAltUAV = RHICmdList.CreateUnorderedAccessView(
			SortKeysBufferAlt, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		SortKeysBufferAltSRV = RHICmdList.CreateShaderResourceView(
			SortKeysBufferAlt, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// Sort distance buffer alt (for radix sort ping-pong)
	{
		const uint32 BufferSize = PaddedCount * sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSortDistanceBufferAlt"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		SortDistanceBufferAlt = RHICmdList.CreateBuffer(Desc);
		SortDistanceBufferAltUAV = RHICmdList.CreateUnorderedAccessView(
			SortDistanceBufferAlt, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// Radix sort histogram buffer: NumTiles * 256 entries
	{
		uint32 NumTiles = FMath::DivideAndRoundUp(PaddedCount, 1024u);
		const uint32 BufferSize = NumTiles * 256 * sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianRadixHistogramBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		RadixHistogramBuffer = RHICmdList.CreateBuffer(Desc);
		RadixHistogramBufferUAV = RHICmdList.CreateUnorderedAccessView(
			RadixHistogramBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// Radix sort digit offset buffer: 256 entries
	{
		const uint32 BufferSize = 256 * sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianRadixDigitOffsetBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_UnorderedAccess | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::UAVCompute);
		RadixDigitOffsetBuffer = RHICmdList.CreateBuffer(Desc);
		RadixDigitOffsetBufferUAV = RHICmdList.CreateUnorderedAccessView(
			RadixDigitOffsetBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// NOTE: SortIndirectArgsBuffer and SortParamsBuffer are created in
	// CreateClusterBuffers() because they're needed by PrepareIndirectArgs
	// in the Nanite compaction path, even when per-proxy dynamic buffers
	// are skipped under the global accumulator.
}

void FGaussianSplatGPUResources::CreateIndexBuffer(FRHICommandListBase& RHICmdList)
{
	// 6 indices per quad (2 triangles): 0,1,2, 1,3,2
	TArray<uint16> Indices = { 0, 1, 2, 1, 3, 2 };

	FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
		TEXT("GaussianSplatIndexBuffer"),
		Indices.Num() * sizeof(uint16),
		sizeof(uint16),
		BUF_Static | BUF_IndexBuffer)
		.SetInitialState(ERHIAccess::VertexOrIndexBuffer);
	IndexBuffer = RHICmdList.CreateBuffer(Desc);

	void* Data = RHICmdList.LockBuffer(IndexBuffer, 0, Indices.Num() * sizeof(uint16), RLM_WriteOnly);
	FMemory::Memcpy(Data, Indices.GetData(), Indices.Num() * sizeof(uint16));
	RHICmdList.UnlockBuffer(IndexBuffer);

	// Create dummy white texture for fallback when ColorTexture isn't available
	CreateDummyWhiteTexture(RHICmdList);
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

void FGaussianSplatGPUResources::CreateClusterBuffers(FRHICommandListBase& RHICmdList)
{
	// When Nanite is disabled (no cluster data), we still need to create dummy buffers
	// because UE5's shader parameter system requires all SHADER_PARAMETER_SRV to be valid
	if (!bHasClusterData || CachedClusterData.Num() == 0)
	{
		// Create minimal dummy buffers for non-Nanite assets
		// These are 1-element buffers that satisfy the shader binding requirements
		// The shader checks UseClusterCulling == 0 and won't actually read from them

		// Dummy SplatClusterIndexBuffer (1 uint)
		{
			const uint32 BufferSize = sizeof(uint32);
			FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
				TEXT("GaussianSplatClusterIndexBufferDummy"),
				BufferSize,
				sizeof(uint32),
				BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
				.SetInitialState(ERHIAccess::SRVMask);
			SplatClusterIndexBuffer = RHICmdList.CreateBuffer(Desc);

			uint32 Zero = 0;
			void* Data = RHICmdList.LockBuffer(SplatClusterIndexBuffer, 0, BufferSize, RLM_WriteOnly);
			FMemory::Memcpy(Data, &Zero, BufferSize);
			RHICmdList.UnlockBuffer(SplatClusterIndexBuffer);

			SplatClusterIndexBufferSRV = RHICmdList.CreateShaderResourceView(
				SplatClusterIndexBuffer, FRHIViewDesc::CreateBufferSRV()
					.SetType(FRHIViewDesc::EBufferType::Structured)
					.SetStride(sizeof(uint32)));
		}

		// Dummy ClusterVisibilityBitmap (1 uint)
		{
			const uint32 BufferSize = sizeof(uint32);
			FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
				TEXT("GaussianClusterVisibilityBitmapDummy"),
				BufferSize,
				sizeof(uint32),
				BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
				.SetInitialState(ERHIAccess::SRVMask);
			ClusterVisibilityBitmap = RHICmdList.CreateBuffer(Desc);

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
			LODClusterSelectedBitmap = RHICmdList.CreateBuffer(Desc);

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
			SelectedClusterBuffer = RHICmdList.CreateBuffer(Desc);

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
			CompactedSplatIndicesBuffer = RHICmdList.CreateBuffer(Desc);

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

	// Create cluster buffer (static, read-only)
	{
		const uint32 BufferSize = CachedClusterData.Num() * sizeof(FGaussianGPUCluster);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianClusterBuffer"),
			BufferSize,
			sizeof(FGaussianGPUCluster),
			BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::SRVMask);
		ClusterBuffer = RHICmdList.CreateBuffer(Desc);

		void* Data = RHICmdList.LockBuffer(ClusterBuffer, 0, BufferSize, RLM_WriteOnly);
		FMemory::Memcpy(Data, CachedClusterData.GetData(), BufferSize);
		RHICmdList.UnlockBuffer(ClusterBuffer);

		ClusterBufferSRV = RHICmdList.CreateShaderResourceView(
			ClusterBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(FGaussianGPUCluster)));
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
		VisibleClusterBuffer = RHICmdList.CreateBuffer(Desc);

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
		VisibleClusterCountBuffer = RHICmdList.CreateBuffer(Desc);

		VisibleClusterCountBufferUAV = RHICmdList.CreateUnorderedAccessView(
			VisibleClusterCountBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		VisibleClusterCountBufferSRV = RHICmdList.CreateShaderResourceView(
			VisibleClusterCountBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	// Clear cached cluster data
	CachedClusterData.Empty();

	// UNIFIED APPROACH: No separate LOD splat buffer needed
	// LOD splats are stored in the same buffers as original splats (Position, Other, Color)
	// The SplatClusterIndexBuffer maps all splats to their clusters

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
		IndirectDrawArgsBuffer = RHICmdList.CreateBuffer(Desc);

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
		UE_LOG(LogTemp, Log, TEXT("GaussianSplatGPUResources: Created indirect draw buffer"));
	}

	// Create splat-to-cluster index buffer
	if (CachedSplatClusterIndices.Num() > 0)
	{
		const uint32 BufferSize = CachedSplatClusterIndices.Num() * sizeof(uint32);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSplatClusterIndexBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::SRVMask);
		SplatClusterIndexBuffer = RHICmdList.CreateBuffer(Desc);

		void* Data = RHICmdList.LockBuffer(SplatClusterIndexBuffer, 0, BufferSize, RLM_WriteOnly);
		FMemory::Memcpy(Data, CachedSplatClusterIndices.GetData(), BufferSize);
		RHICmdList.UnlockBuffer(SplatClusterIndexBuffer);

		SplatClusterIndexBufferSRV = RHICmdList.CreateShaderResourceView(
			SplatClusterIndexBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));

		UE_LOG(LogTemp, Log, TEXT("GaussianSplatGPUResources: Created splat-to-cluster index buffer for %d splats"), CachedSplatClusterIndices.Num());
	}
	CachedSplatClusterIndices.Empty();

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
		ClusterVisibilityBitmap = RHICmdList.CreateBuffer(Desc);

		ClusterVisibilityBitmapUAV = RHICmdList.CreateUnorderedAccessView(
			ClusterVisibilityBitmap, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		ClusterVisibilityBitmapSRV = RHICmdList.CreateShaderResourceView(
			ClusterVisibilityBitmap, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));

		UE_LOG(LogTemp, Log, TEXT("GaussianSplatGPUResources: Created cluster visibility bitmap (%d bytes for %d clusters)"), BitmapSize, ClusterCount);
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
		SelectedClusterBuffer = RHICmdList.CreateBuffer(Desc);

		SelectedClusterBufferUAV = RHICmdList.CreateUnorderedAccessView(
			SelectedClusterBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		SelectedClusterBufferSRV = RHICmdList.CreateShaderResourceView(
			SelectedClusterBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));

		UE_LOG(LogTemp, Log, TEXT("GaussianSplatGPUResources: Created selected cluster buffer (%d bytes for %d leaf clusters)"), BufferSize, LeafClusterCount);
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
		LODClusterBuffer = RHICmdList.CreateBuffer(Desc);

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
		LODClusterCountBuffer = RHICmdList.CreateBuffer(Desc);

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
		LODClusterSelectedBitmap = RHICmdList.CreateBuffer(Desc);

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
		LODSplatTotalBuffer = RHICmdList.CreateBuffer(Desc);

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
		LODSplatOutputCountBuffer = RHICmdList.CreateBuffer(Desc);

		LODSplatOutputCountBufferUAV = RHICmdList.CreateUnorderedAccessView(
			LODSplatOutputCountBuffer, FRHIViewDesc::CreateBufferUAV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		LODSplatOutputCountBufferSRV = RHICmdList.CreateShaderResourceView(
			LODSplatOutputCountBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
	}

	UE_LOG(LogTemp, Log, TEXT("GaussianSplatGPUResources: Created LOD cluster tracking buffers"));

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
		CompactedSplatIndicesBuffer = RHICmdList.CreateBuffer(Desc);

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
		VisibleSplatCountBuffer = RHICmdList.CreateBuffer(Desc);

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
		IndirectDispatchArgsBuffer = RHICmdList.CreateBuffer(Desc);

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
		SortIndirectArgsBuffer = RHICmdList.CreateBuffer(Desc);
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
		SortParamsBuffer = RHICmdList.CreateBuffer(Desc);
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
	UE_LOG(LogTemp, Log, TEXT("GaussianSplatGPUResources: Created splat compaction buffers for %d total splats"), TotalSplatCount);

	UE_LOG(LogTemp, Log, TEXT("GaussianSplatGPUResources: Created cluster buffers for %d clusters"), ClusterCount);
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
void FGaussianSplatSceneProxy::CreateRenderThreadResources(FRHICommandListBase& RHICmdList)
{
	UE_LOG(LogTemp, Warning, TEXT("GaussianSplat: CreateRenderThreadResources called! SplatCount=%d"), SplatCount);

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
	UE_LOG(LogTemp, Warning, TEXT("GaussianSplat: DestroyRenderThreadResources called!"));

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

