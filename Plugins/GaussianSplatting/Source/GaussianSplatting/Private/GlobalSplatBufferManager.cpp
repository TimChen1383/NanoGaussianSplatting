// Copyright Epic Games, Inc. All Rights Reserved.

#include "GlobalSplatBufferManager.h"
#include "GaussianSplatSceneProxy.h"
#include "GaussianDataTypes.h"
#include "GaussianClusterTypes.h"
#include "RHICommandList.h"
#include "RHIResources.h"

// ============================================================================
// UpdateIfNeeded
// ============================================================================

bool FGlobalSplatBufferManager::UpdateIfNeeded(
	FRHICommandListImmediate& RHICmdList,
	const TArray<FGaussianSplatSceneProxy*>& AllRegisteredProxies)
{
	// Filter: only proxies that are valid and have cluster (Nanite) data
	TArray<FGaussianSplatSceneProxy*> ValidProxies;
	ValidProxies.Reserve(AllRegisteredProxies.Num());

	for (FGaussianSplatSceneProxy* Proxy : AllRegisteredProxies)
	{
		if (!Proxy || !Proxy->IsValidForRendering())
		{
			continue;
		}
		FGaussianSplatGPUResources* Res = Proxy->GetGPUResources();
		if (Res && Res->bHasClusterData && Res->TotalSplatCount > 0)
		{
			ValidProxies.Add(Proxy);
		}
	}

	// Detect proxy set change (count or identity)
	bool bChanged = (ValidProxies.Num() != LastProxySet.Num());
	if (!bChanged)
	{
		for (int32 i = 0; i < ValidProxies.Num(); i++)
		{
			if (ValidProxies[i] != LastProxySet[i])
			{
				bChanged = true;
				break;
			}
		}
	}

	if (bChanged || !bStaticBuffersBuilt)
	{
		RebuildStaticBuffers(RHICmdList, ValidProxies);
		LastProxySet = ValidProxies;
		bStaticBuffersBuilt = true;
	}

	// Always upload metadata — transforms update every frame for moving objects
	UploadProxyMetadata(RHICmdList, ValidProxies, bChanged);

	return bChanged;
}

// ============================================================================
// RebuildStaticBuffers
// ============================================================================

void FGlobalSplatBufferManager::RebuildStaticBuffers(
	FRHICommandListImmediate& RHICmdList,
	const TArray<FGaussianSplatSceneProxy*>& ValidProxies)
{
	// Release old static buffers
	GlobalPackedSplatBuffer.SafeRelease();
	GlobalPackedSplatBufferSRV.SafeRelease();
	GlobalClusterBuffer.SafeRelease();
	GlobalClusterBufferSRV.SafeRelease();
	GlobalSplatClusterIndexBuffer.SafeRelease();
	GlobalSplatClusterIndexBufferSRV.SafeRelease();

	TotalSplatCount          = 0;
	TotalClusterCount        = 0;
	TotalLeafClusterCount    = 0;
	TotalVisibilityBitmapUints = 0;

	if (ValidProxies.Num() == 0)
	{
		UE_LOG(LogTemp, Log, TEXT("[GS Stage1] RebuildStaticBuffers: No valid proxies — global buffers cleared."));
		return;
	}

	// Compute totals
	for (FGaussianSplatSceneProxy* Proxy : ValidProxies)
	{
		FGaussianSplatGPUResources* Res = Proxy->GetGPUResources();
		TotalSplatCount          += (uint32)Res->TotalSplatCount;
		TotalClusterCount        += (uint32)Res->ClusterCount;
		TotalLeafClusterCount    += (uint32)Res->LeafClusterCount;
		TotalVisibilityBitmapUints += FMath::DivideAndRoundUp((uint32)Res->ClusterCount, 32u);
	}

	UE_LOG(LogTemp, Log,
		TEXT("[GS Stage1] RebuildStaticBuffers: ProxyCount=%d TotalSplats=%u TotalClusters=%u TotalLeafClusters=%u TotalBitmapUints=%u"),
		ValidProxies.Num(), TotalSplatCount, TotalClusterCount, TotalLeafClusterCount, TotalVisibilityBitmapUints);

	const uint32 PackedSplatStride = (uint32)GaussianSplattingConstants::PackedSplatStride; // 16 bytes
	const uint32 ClusterStride     = (uint32)sizeof(FGaussianGPUCluster);                   // 80 bytes
	const uint32 UintStride        = (uint32)sizeof(uint32);                                // 4 bytes

	// --- Create GlobalPackedSplatBuffer (ByteAddressBuffer) ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalPackedSplatBuffer"),
			TotalSplatCount * PackedSplatStride,
			0, // stride = 0 for raw/byte-address buffers
			BUF_Static | BUF_ShaderResource | BUF_ByteAddressBuffer)
			.SetInitialState(ERHIAccess::CopyDest);
		GlobalPackedSplatBuffer = RHICmdList.CreateBuffer(Desc);
		GlobalPackedSplatBufferSRV = RHICmdList.CreateShaderResourceView(
			GlobalPackedSplatBuffer,
			FRHIViewDesc::CreateBufferSRV().SetType(FRHIViewDesc::EBufferType::Raw));
	}

	// --- Create GlobalClusterBuffer (StructuredBuffer<FGaussianGPUCluster>) ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalClusterBuffer"),
			TotalClusterCount * ClusterStride,
			ClusterStride,
			BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::CopyDest);
		GlobalClusterBuffer = RHICmdList.CreateBuffer(Desc);
		GlobalClusterBufferSRV = RHICmdList.CreateShaderResourceView(
			GlobalClusterBuffer,
			FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(ClusterStride));
	}

	// --- Create GlobalSplatClusterIndexBuffer (StructuredBuffer<uint>) ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalSplatClusterIndexBuffer"),
			TotalSplatCount * UintStride,
			UintStride,
			BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::CopyDest);
		GlobalSplatClusterIndexBuffer = RHICmdList.CreateBuffer(Desc);
		GlobalSplatClusterIndexBufferSRV = RHICmdList.CreateShaderResourceView(
			GlobalSplatClusterIndexBuffer,
			FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// TODO Stage 4: Build GlobalSHBuffer here (ByteAddressBuffer, variable stride per proxy)

	// --- GPU-to-GPU copy: per-proxy buffers → global buffers ---
	uint32 SplatOffset   = 0;
	uint32 ClusterOffset = 0;

	for (int32 i = 0; i < ValidProxies.Num(); i++)
	{
		FGaussianSplatGPUResources* Res = ValidProxies[i]->GetGPUResources();

		const uint32 ProxySplatBytes        = (uint32)Res->TotalSplatCount * PackedSplatStride;
		const uint32 ProxyClusterBytes      = (uint32)Res->ClusterCount   * ClusterStride;
		const uint32 ProxySplatIndexBytes   = (uint32)Res->TotalSplatCount * UintStride;

		// Transition source buffers to CopySrc
		// Using ERHIAccess::Unknown as "before" — safe when prior state is unknown
		RHICmdList.Transition(FRHITransitionInfo(Res->PackedSplatBuffer,        ERHIAccess::Unknown, ERHIAccess::CopySrc));
		RHICmdList.Transition(FRHITransitionInfo(Res->ClusterBuffer,            ERHIAccess::Unknown, ERHIAccess::CopySrc));
		RHICmdList.Transition(FRHITransitionInfo(Res->SplatClusterIndexBuffer,  ERHIAccess::Unknown, ERHIAccess::CopySrc));

		// Copy into global buffers at proxy's offset
		RHICmdList.CopyBufferRegion(
			GlobalPackedSplatBuffer,       SplatOffset   * PackedSplatStride,
			Res->PackedSplatBuffer,        0,
			ProxySplatBytes);

		RHICmdList.CopyBufferRegion(
			GlobalClusterBuffer,           ClusterOffset * ClusterStride,
			Res->ClusterBuffer,            0,
			ProxyClusterBytes);

		RHICmdList.CopyBufferRegion(
			GlobalSplatClusterIndexBuffer, SplatOffset   * UintStride,
			Res->SplatClusterIndexBuffer,  0,
			ProxySplatIndexBytes);

		// Transition source buffers back to SRVCompute for normal rendering
		RHICmdList.Transition(FRHITransitionInfo(Res->PackedSplatBuffer,        ERHIAccess::CopySrc, ERHIAccess::SRVCompute));
		RHICmdList.Transition(FRHITransitionInfo(Res->ClusterBuffer,            ERHIAccess::CopySrc, ERHIAccess::SRVCompute));
		RHICmdList.Transition(FRHITransitionInfo(Res->SplatClusterIndexBuffer,  ERHIAccess::CopySrc, ERHIAccess::SRVCompute));

		SplatOffset   += (uint32)Res->TotalSplatCount;
		ClusterOffset += (uint32)Res->ClusterCount;
	}

	// Transition global buffers to SRVCompute — ready for shader reads in Stages 2+
	RHICmdList.Transition(FRHITransitionInfo(GlobalPackedSplatBuffer,       ERHIAccess::CopyDest, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalClusterBuffer,           ERHIAccess::CopyDest, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(GlobalSplatClusterIndexBuffer, ERHIAccess::CopyDest, ERHIAccess::SRVCompute));

	UE_LOG(LogTemp, Log, TEXT("[GS Stage1] RebuildStaticBuffers: Complete. GlobalPackedSplatBuffer=%uB GlobalClusterBuffer=%uB GlobalSplatClusterIndexBuffer=%uB"),
		TotalSplatCount * PackedSplatStride,
		TotalClusterCount * ClusterStride,
		TotalSplatCount * UintStride);
}

// ============================================================================
// UploadProxyMetadata
// ============================================================================

void FGlobalSplatBufferManager::UploadProxyMetadata(
	FRHICommandListImmediate& RHICmdList,
	const TArray<FGaussianSplatSceneProxy*>& ValidProxies,
	bool bLogDetails)
{
	if (ValidProxies.Num() == 0)
	{
		return;
	}

	const uint32 ProxyCount = (uint32)ValidProxies.Num();

	// Recreate ProxyMetadataBuffer if size grew
	if (ProxyCount > AllocatedMetadataCount)
	{
		ProxyMetadataBuffer.SafeRelease();
		ProxyMetadataBufferSRV.SafeRelease();

		// 20% headroom to avoid recreation when a couple of proxies are added
		const uint32 NewCapacity = FMath::Max(ProxyCount + ProxyCount / 5, 16u);

		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("ProxyMetadataBuffer"),
			NewCapacity * (uint32)sizeof(FProxyGPUMetadata),
			(uint32)sizeof(FProxyGPUMetadata),
			BUF_Dynamic | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::SRVCompute);
		ProxyMetadataBuffer = RHICmdList.CreateBuffer(Desc);
		ProxyMetadataBufferSRV = RHICmdList.CreateShaderResourceView(
			ProxyMetadataBuffer,
			FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride((uint32)sizeof(FProxyGPUMetadata)));

		AllocatedMetadataCount = NewCapacity;

		UE_LOG(LogTemp, Log, TEXT("[GS Stage1] ProxyMetadataBuffer allocated: capacity=%u (%u bytes)"),
			NewCapacity, NewCapacity * (uint32)sizeof(FProxyGPUMetadata));
	}

	// Build the metadata array on CPU
	TArray<FProxyGPUMetadata> MetadataArray;
	MetadataArray.SetNumUninitialized(ProxyCount);

	uint32 CumulativeSplats      = 0;
	uint32 CumulativeClusters    = 0;
	uint32 CumulativeLeafClusters = 0;
	uint32 CumulativeBitmapUints = 0;

	for (uint32 i = 0; i < ProxyCount; i++)
	{
		FGaussianSplatSceneProxy*  Proxy = ValidProxies[i];
		FGaussianSplatGPUResources* Res  = Proxy->GetGPUResources();
		FProxyGPUMetadata& M = MetadataArray[i];

		// Transform: convert from FMatrix (double) to float4x4 row-major
		const FMatrix LTW = Proxy->GetLocalToWorld();
		for (int32 Row = 0; Row < 4; Row++)
		{
			for (int32 Col = 0; Col < 4; Col++)
			{
				M.LocalToWorld[Row * 4 + Col] = (float)LTW.M[Row][Col];
			}
		}

		// Splat offsets
		M.GlobalSplatStartIndex      = CumulativeSplats;
		M.TotalSplatCount            = (uint32)Res->TotalSplatCount;
		M.OriginalSplatCount         = (uint32)(Res->TotalSplatCount - Res->LODSplatCount);

		// Cluster offsets
		M.GlobalClusterStartIndex    = CumulativeClusters;
		M.ClusterCount               = (uint32)Res->ClusterCount;
		M.LeafClusterCount           = (uint32)Res->LeafClusterCount;

		// Bitmap offsets (filled in Stage 2 with actual global bitmap layout)
		const uint32 ProxyBitmapUints = FMath::DivideAndRoundUp((uint32)Res->ClusterCount, 32u);
		M.GlobalVisibilityBitmapStart = CumulativeBitmapUints;
		M.GlobalSelectedClusterStart  = CumulativeLeafClusters;
		M.GlobalLODBitmapStart        = CumulativeBitmapUints;

		// Compaction
		M.GlobalSplatClusterIndexStart = CumulativeSplats;
		M.CompactionCounterIndex       = i;
		M.CompactionOutputStart        = CumulativeSplats;

		// SH — TODO Stage 4: set GlobalSHStartIndex once GlobalSHBuffer is built
		M.GlobalSHStartIndex = 0;

		// Cumulative totals used by GPU binary search
		CumulativeSplats       += (uint32)Res->TotalSplatCount;
		CumulativeClusters     += (uint32)Res->ClusterCount;
		CumulativeLeafClusters += (uint32)Res->LeafClusterCount;
		CumulativeBitmapUints  += ProxyBitmapUints;

		M.GlobalLeafClusterEnd = CumulativeLeafClusters;
		M.GlobalSplatEnd       = CumulativeSplats;
		M.Pad[0]               = 0;
	}

	// Stage 1 verification log (only on rebuild, to avoid per-frame spam)
	if (bLogDetails)
	{
		UE_LOG(LogTemp, Log, TEXT("[GS Stage1] UploadProxyMetadata: ProxyCount=%u"), ProxyCount);

		for (uint32 i = 0; i < ProxyCount; i++)
		{
			const FProxyGPUMetadata& M = MetadataArray[i];
			UE_LOG(LogTemp, Log,
				TEXT("[GS Stage1]   Proxy[%u]: SplatStart=%u SplatCount=%u ClusterStart=%u ClusterCount=%u LeafClusters=%u SplatEnd=%u LeafClusterEnd=%u"),
				i,
				M.GlobalSplatStartIndex, M.TotalSplatCount,
				M.GlobalClusterStartIndex, M.ClusterCount, M.LeafClusterCount,
				M.GlobalSplatEnd, M.GlobalLeafClusterEnd);
		}

		// Consistency checks
		const FProxyGPUMetadata& Last = MetadataArray[ProxyCount - 1];
		bool bSplatsConsistent   = (Last.GlobalSplatEnd       == TotalSplatCount);
		bool bClustersConsistent = (Last.GlobalLeafClusterEnd == TotalLeafClusterCount);

		UE_LOG(LogTemp, Log,
			TEXT("[GS Stage1]   SplatEnd=%u == TotalSplatCount=%u : %s"),
			Last.GlobalSplatEnd, TotalSplatCount,
			bSplatsConsistent ? TEXT("OK") : TEXT("MISMATCH - BUG!"));

		UE_LOG(LogTemp, Log,
			TEXT("[GS Stage1]   LeafClusterEnd=%u == TotalLeafClusterCount=%u : %s"),
			Last.GlobalLeafClusterEnd, TotalLeafClusterCount,
			bClustersConsistent ? TEXT("OK") : TEXT("MISMATCH - BUG!"));

		// Check monotonically increasing SplatStart and ClusterStart
		bool bMonotonic = true;
		for (uint32 i = 1; i < ProxyCount; i++)
		{
			if (MetadataArray[i].GlobalSplatStartIndex   <= MetadataArray[i-1].GlobalSplatStartIndex ||
				MetadataArray[i].GlobalClusterStartIndex <= MetadataArray[i-1].GlobalClusterStartIndex)
			{
				bMonotonic = false;
				break;
			}
		}
		UE_LOG(LogTemp, Log, TEXT("[GS Stage1]   Offsets monotonically increasing: %s"),
			bMonotonic ? TEXT("OK") : TEXT("FAIL - BUG!"));
	}

	// Upload to GPU
	void* Data = RHICmdList.LockBuffer(
		ProxyMetadataBuffer, 0,
		ProxyCount * (uint32)sizeof(FProxyGPUMetadata),
		RLM_WriteOnly);

	FMemory::Memcpy(Data, MetadataArray.GetData(), ProxyCount * sizeof(FProxyGPUMetadata));

	RHICmdList.UnlockBuffer(ProxyMetadataBuffer);
}

// ============================================================================
// Release
// ============================================================================

void FGlobalSplatBufferManager::Release()
{
	GlobalPackedSplatBuffer.SafeRelease();
	GlobalPackedSplatBufferSRV.SafeRelease();

	GlobalClusterBuffer.SafeRelease();
	GlobalClusterBufferSRV.SafeRelease();

	GlobalSplatClusterIndexBuffer.SafeRelease();
	GlobalSplatClusterIndexBufferSRV.SafeRelease();

	ProxyMetadataBuffer.SafeRelease();
	ProxyMetadataBufferSRV.SafeRelease();

	LastProxySet.Empty();
	TotalSplatCount           = 0;
	TotalClusterCount         = 0;
	TotalLeafClusterCount     = 0;
	TotalVisibilityBitmapUints = 0;
	AllocatedMetadataCount    = 0;
	bStaticBuffersBuilt       = false;
}
