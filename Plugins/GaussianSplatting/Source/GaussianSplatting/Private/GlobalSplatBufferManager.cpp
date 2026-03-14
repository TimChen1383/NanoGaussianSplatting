// Copyright Epic Games, Inc. All Rights Reserved.

#include "GlobalSplatBufferManager.h"
#include "GaussianSplatSceneProxy.h"
#include "GaussianSplatShaders.h"
#include "GaussianSplatRenderer.h"
#include "GaussianDataTypes.h"
#include "GaussianClusterTypes.h"
#include "RHICommandList.h"
#include "RHIResources.h"
#include "SceneView.h"
#include "GlobalShader.h"
#include "ShaderParameterUtils.h"

extern TAutoConsoleVariable<int32> CVarDebugForceLODLevel;

static TAutoConsoleVariable<int32> CVarValidateGlobalCulling(
	TEXT("gs.ValidateGlobalCulling"),
	1,
	TEXT("Stage 2: Compare per-proxy cluster culling results against global culling.\n")
	TEXT("0 = disabled, 1 = enabled (GPU readback + CPU compare, stalls GPU)"),
	ECVF_RenderThreadSafe);

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

	// Invalidate shadow buffers since sizes may change
	bShadowBuffersAllocated = false;
	bShadowCompactBuffersAllocated = false;

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

	const uint32 TotalPackedBytes  = TotalSplatCount   * PackedSplatStride;
	const uint32 TotalClusterBytes = TotalClusterCount * ClusterStride;
	const uint32 TotalIndexBytes   = TotalSplatCount   * UintStride;

	// ---------------------------------------------------------------
	// CPU-side staging: read per-proxy GPU buffers → build combined
	// arrays on CPU → upload to global buffers via LockBuffer.
	//
	// This avoids CopyBufferRegion entirely, which crashes on D3D12
	// when source and dest are suballocated from the same placed resource.
	// Only runs when proxy set changes (not per-frame), so CPU cost is fine.
	// ---------------------------------------------------------------

	TArray<uint8> CombinedPackedSplats;
	TArray<uint8> CombinedClusters;
	TArray<uint8> CombinedSplatClusterIndices;
	CombinedPackedSplats.SetNumUninitialized(TotalPackedBytes);
	CombinedClusters.SetNumUninitialized(TotalClusterBytes);
	CombinedSplatClusterIndices.SetNumUninitialized(TotalIndexBytes);

	uint32 SplatByteOffset   = 0;
	uint32 ClusterByteOffset = 0;
	uint32 IndexByteOffset   = 0;

	for (int32 i = 0; i < ValidProxies.Num(); i++)
	{
		FGaussianSplatGPUResources* Res = ValidProxies[i]->GetGPUResources();

		const uint32 ProxySplatBytes      = (uint32)Res->TotalSplatCount * PackedSplatStride;
		const uint32 ProxyClusterBytes    = (uint32)Res->ClusterCount   * ClusterStride;
		const uint32 ProxySplatIndexBytes = (uint32)Res->TotalSplatCount * UintStride;

		// Read per-proxy packed splats to CPU
		{
			void* Src = RHICmdList.LockBuffer(Res->PackedSplatBuffer, 0, ProxySplatBytes, RLM_ReadOnly);
			FMemory::Memcpy(CombinedPackedSplats.GetData() + SplatByteOffset, Src, ProxySplatBytes);
			RHICmdList.UnlockBuffer(Res->PackedSplatBuffer);
		}

		// Read per-proxy clusters to CPU
		{
			void* Src = RHICmdList.LockBuffer(Res->ClusterBuffer, 0, ProxyClusterBytes, RLM_ReadOnly);
			FMemory::Memcpy(CombinedClusters.GetData() + ClusterByteOffset, Src, ProxyClusterBytes);
			RHICmdList.UnlockBuffer(Res->ClusterBuffer);
		}

		// Read per-proxy splat cluster indices to CPU
		{
			void* Src = RHICmdList.LockBuffer(Res->SplatClusterIndexBuffer, 0, ProxySplatIndexBytes, RLM_ReadOnly);
			FMemory::Memcpy(CombinedSplatClusterIndices.GetData() + IndexByteOffset, Src, ProxySplatIndexBytes);
			RHICmdList.UnlockBuffer(Res->SplatClusterIndexBuffer);
		}

		SplatByteOffset   += ProxySplatBytes;
		ClusterByteOffset += ProxyClusterBytes;
		IndexByteOffset   += ProxySplatIndexBytes;
	}

	// --- Create and upload GlobalPackedSplatBuffer (ByteAddressBuffer) ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalPackedSplatBuffer"),
			TotalPackedBytes,
			0, // stride = 0 for raw/byte-address buffers
			BUF_Static | BUF_ShaderResource | BUF_ByteAddressBuffer)
			.SetInitialState(ERHIAccess::SRVCompute);
		GlobalPackedSplatBuffer = RHICmdList.CreateBuffer(Desc);

		void* Dst = RHICmdList.LockBuffer(GlobalPackedSplatBuffer, 0, TotalPackedBytes, RLM_WriteOnly);
		FMemory::Memcpy(Dst, CombinedPackedSplats.GetData(), TotalPackedBytes);
		RHICmdList.UnlockBuffer(GlobalPackedSplatBuffer);

		GlobalPackedSplatBufferSRV = RHICmdList.CreateShaderResourceView(
			GlobalPackedSplatBuffer,
			FRHIViewDesc::CreateBufferSRV().SetType(FRHIViewDesc::EBufferType::Raw));
	}

	// --- Create and upload GlobalClusterBuffer (StructuredBuffer<FGaussianGPUCluster>) ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalClusterBuffer"),
			TotalClusterBytes,
			ClusterStride,
			BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::SRVCompute);
		GlobalClusterBuffer = RHICmdList.CreateBuffer(Desc);

		void* Dst = RHICmdList.LockBuffer(GlobalClusterBuffer, 0, TotalClusterBytes, RLM_WriteOnly);
		FMemory::Memcpy(Dst, CombinedClusters.GetData(), TotalClusterBytes);
		RHICmdList.UnlockBuffer(GlobalClusterBuffer);

		GlobalClusterBufferSRV = RHICmdList.CreateShaderResourceView(
			GlobalClusterBuffer,
			FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(ClusterStride));
	}

	// --- Create and upload GlobalSplatClusterIndexBuffer (StructuredBuffer<uint>) ---
	{
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GlobalSplatClusterIndexBuffer"),
			TotalIndexBytes,
			UintStride,
			BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::SRVCompute);
		GlobalSplatClusterIndexBuffer = RHICmdList.CreateBuffer(Desc);

		void* Dst = RHICmdList.LockBuffer(GlobalSplatClusterIndexBuffer, 0, TotalIndexBytes, RLM_WriteOnly);
		FMemory::Memcpy(Dst, CombinedSplatClusterIndices.GetData(), TotalIndexBytes);
		RHICmdList.UnlockBuffer(GlobalSplatClusterIndexBuffer);

		GlobalSplatClusterIndexBufferSRV = RHICmdList.CreateShaderResourceView(
			GlobalSplatClusterIndexBuffer,
			FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(UintStride));
	}

	// TODO Stage 4: Build GlobalSHBuffer here (ByteAddressBuffer, variable stride per proxy)

	UE_LOG(LogTemp, Log, TEXT("[GS Stage1] RebuildStaticBuffers: Complete. GlobalPackedSplatBuffer=%uB GlobalClusterBuffer=%uB GlobalSplatClusterIndexBuffer=%uB"),
		TotalPackedBytes, TotalClusterBytes, TotalIndexBytes);
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
		M.ErrorThreshold       = FMath::Max(0.001f, Proxy->GetLODErrorThreshold());
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

	// Stage 2: Shadow buffers
	ShadowClusterVisibilityBitmap.SafeRelease();
	ShadowClusterVisibilityBitmapUAV.SafeRelease();
	ShadowSelectedClusterBuffer.SafeRelease();
	ShadowSelectedClusterBufferUAV.SafeRelease();
	ShadowLODClusterSelectedBitmap.SafeRelease();
	ShadowLODClusterSelectedBitmapUAV.SafeRelease();
	ShadowVisibleClusterCountBuffer.SafeRelease();
	ShadowVisibleClusterCountBufferUAV.SafeRelease();
	ShadowLODClusterBuffer.SafeRelease();
	ShadowLODClusterBufferUAV.SafeRelease();
	ShadowLODClusterCountBuffer.SafeRelease();
	ShadowLODClusterCountBufferUAV.SafeRelease();
	ShadowLODSplatTotalBuffer.SafeRelease();
	ShadowLODSplatTotalBufferUAV.SafeRelease();
	ShadowValidationStagingBuffer.SafeRelease();
	bShadowBuffersAllocated = false;

	// Stage 3: Shadow compaction buffers
	ShadowCompactedSplatIndices.SafeRelease();
	ShadowCompactedSplatIndicesUAV.SafeRelease();
	ShadowVisibleCountArray.SafeRelease();
	ShadowVisibleCountArrayUAV.SafeRelease();
	ShadowVisibleCountArraySRV.SafeRelease();
	bShadowCompactBuffersAllocated = false;

	LastProxySet.Empty();
	TotalSplatCount           = 0;
	TotalClusterCount         = 0;
	TotalLeafClusterCount     = 0;
	TotalVisibilityBitmapUints = 0;
	AllocatedMetadataCount    = 0;
	bStaticBuffersBuilt       = false;
}

// ============================================================================
// Stage 2: EnsureShadowBuffers
// ============================================================================

static FBufferRHIRef CreateStructuredBuffer(FRHICommandListImmediate& RHICmdList, const TCHAR* Name, uint32 Stride, uint32 Count, EBufferUsageFlags ExtraFlags = BUF_None)
{
	FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
		Name,
		Count * Stride,
		Stride,
		BUF_Static | BUF_ShaderResource | BUF_UnorderedAccess | BUF_StructuredBuffer | ExtraFlags)
		.SetInitialState(ERHIAccess::UAVCompute);
	return RHICmdList.CreateBuffer(Desc);
}

void FGlobalSplatBufferManager::EnsureShadowBuffers(FRHICommandListImmediate& RHICmdList)
{
	if (bShadowBuffersAllocated)
	{
		return;
	}

	const uint32 UintStride = sizeof(uint32);

	// Visibility bitmap: TotalVisibilityBitmapUints uint32s
	uint32 BitmapCount = FMath::Max(TotalVisibilityBitmapUints, 1u);
	ShadowClusterVisibilityBitmap = CreateStructuredBuffer(RHICmdList, TEXT("ShadowClusterVisibilityBitmap"), UintStride, BitmapCount);
	ShadowClusterVisibilityBitmapUAV = RHICmdList.CreateUnorderedAccessView(ShadowClusterVisibilityBitmap, FRHIViewDesc::CreateBufferUAV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(sizeof(uint32)));

	// Selected cluster buffer: one entry per leaf cluster
	uint32 LeafCount = FMath::Max(TotalLeafClusterCount, 1u);
	ShadowSelectedClusterBuffer = CreateStructuredBuffer(RHICmdList, TEXT("ShadowSelectedClusterBuffer"), UintStride, LeafCount);
	ShadowSelectedClusterBufferUAV = RHICmdList.CreateUnorderedAccessView(ShadowSelectedClusterBuffer, FRHIViewDesc::CreateBufferUAV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(sizeof(uint32)));

	// LOD cluster selected bitmap: same size as visibility bitmap
	ShadowLODClusterSelectedBitmap = CreateStructuredBuffer(RHICmdList, TEXT("ShadowLODClusterSelectedBitmap"), UintStride, BitmapCount);
	ShadowLODClusterSelectedBitmapUAV = RHICmdList.CreateUnorderedAccessView(ShadowLODClusterSelectedBitmap, FRHIViewDesc::CreateBufferUAV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(sizeof(uint32)));

	// Visible cluster count: single uint32
	ShadowVisibleClusterCountBuffer = CreateStructuredBuffer(RHICmdList, TEXT("ShadowVisibleClusterCountBuffer"), UintStride, 1);
	ShadowVisibleClusterCountBufferUAV = RHICmdList.CreateUnorderedAccessView(ShadowVisibleClusterCountBuffer, FRHIViewDesc::CreateBufferUAV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(sizeof(uint32)));

	// LOD cluster buffer: up to TotalClusterCount (worst case)
	uint32 MaxLODClusters = FMath::Max(TotalClusterCount, 1u);
	ShadowLODClusterBuffer = CreateStructuredBuffer(RHICmdList, TEXT("ShadowLODClusterBuffer"), UintStride, MaxLODClusters);
	ShadowLODClusterBufferUAV = RHICmdList.CreateUnorderedAccessView(ShadowLODClusterBuffer, FRHIViewDesc::CreateBufferUAV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(sizeof(uint32)));

	// LOD cluster count: single uint32
	ShadowLODClusterCountBuffer = CreateStructuredBuffer(RHICmdList, TEXT("ShadowLODClusterCountBuffer"), UintStride, 1);
	ShadowLODClusterCountBufferUAV = RHICmdList.CreateUnorderedAccessView(ShadowLODClusterCountBuffer, FRHIViewDesc::CreateBufferUAV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(sizeof(uint32)));

	// LOD splat total: single uint32
	ShadowLODSplatTotalBuffer = CreateStructuredBuffer(RHICmdList, TEXT("ShadowLODSplatTotalBuffer"), UintStride, 1);
	ShadowLODSplatTotalBufferUAV = RHICmdList.CreateUnorderedAccessView(ShadowLODSplatTotalBuffer, FRHIViewDesc::CreateBufferUAV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(sizeof(uint32)));

	bShadowBuffersAllocated = true;

	UE_LOG(LogTemp, Log, TEXT("[GS Stage2] Shadow buffers allocated: BitmapUints=%u LeafClusters=%u TotalClusters=%u"),
		BitmapCount, LeafCount, MaxLODClusters);
}

// ============================================================================
// Stage 2: DispatchGlobalClusterCulling
// ============================================================================

void FGlobalSplatBufferManager::DispatchGlobalClusterCulling(
	FRHICommandListImmediate& RHICmdList,
	const FSceneView& View,
	const TArray<FGaussianSplatSceneProxy*>& ValidProxies)
{
	if (!IsReady() || TotalLeafClusterCount == 0 || GetProxyCount() == 0)
	{
		return;
	}

	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatGlobalClusterCulling_Shadow);

	// Ensure shadow buffers exist
	EnsureShadowBuffers(RHICmdList);

	TShaderMapRef<FGlobalClusterCullingResetCS> ResetShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FGlobalClusterCullingCS> CullingShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!ResetShader.IsValid() || !CullingShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("[GS Stage2] Global cluster culling shaders not valid"));
		return;
	}

	// Transition shadow buffers to UAVCompute
	RHICmdList.Transition(FRHITransitionInfo(ShadowClusterVisibilityBitmap, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowSelectedClusterBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowLODClusterSelectedBitmap, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowVisibleClusterCountBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowLODClusterBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowLODClusterCountBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowLODSplatTotalBuffer, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	// Transition global input buffers to SRVCompute
	RHICmdList.Transition(FRHITransitionInfo(GlobalClusterBuffer, ERHIAccess::Unknown, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ProxyMetadataBuffer, ERHIAccess::Unknown, ERHIAccess::SRVCompute));

	// Step 1: Reset shadow bitmaps and counters
	{
		FGlobalClusterCullingResetCS::FParameters ResetParams;
		ResetParams.ShadowClusterVisibilityBitmap = ShadowClusterVisibilityBitmapUAV;
		ResetParams.ShadowSelectedClusterBuffer = ShadowSelectedClusterBufferUAV;
		ResetParams.ShadowLODClusterSelectedBitmap = ShadowLODClusterSelectedBitmapUAV;
		ResetParams.ShadowVisibleClusterCountBuffer = ShadowVisibleClusterCountBufferUAV;
		ResetParams.ShadowLODClusterBuffer = ShadowLODClusterBufferUAV;
		ResetParams.ShadowLODClusterCountBuffer = ShadowLODClusterCountBufferUAV;
		ResetParams.ShadowLODSplatTotalBuffer = ShadowLODSplatTotalBufferUAV;
		ResetParams.ProxyMetadataBuffer = ProxyMetadataBufferSRV;
		ResetParams.ProxyCount = GetProxyCount();
		ResetParams.TotalLeafClusterCount = TotalLeafClusterCount;
		ResetParams.TotalClusterCount = TotalClusterCount;
		ResetParams.TotalVisibilityBitmapUints = TotalVisibilityBitmapUints;

		uint32 ResetThreads = FMath::Max3(TotalVisibilityBitmapUints, TotalLeafClusterCount, 1u);
		uint32 NumGroups = FMath::DivideAndRoundUp(ResetThreads, 64u);

		SetComputePipelineState(RHICmdList, ResetShader.GetComputeShader());
		SetShaderParameters(RHICmdList, ResetShader, ResetShader.GetComputeShader(), ResetParams);
		RHICmdList.DispatchComputeShader(NumGroups, 1, 1);
		UnsetShaderUAVs(RHICmdList, ResetShader, ResetShader.GetComputeShader());
	}

	// UAV barrier between reset and culling
	RHICmdList.Transition(FRHITransitionInfo(ShadowClusterVisibilityBitmap, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowSelectedClusterBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowLODClusterSelectedBitmap, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowVisibleClusterCountBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowLODClusterCountBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowLODSplatTotalBuffer, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

	// Step 2: Global cluster culling dispatch
	{
		FVector4f FrustumPlanes[6];
		FGaussianSplatRenderer::ExtractFrustumPlanes(View.ViewMatrices.GetViewProjectionMatrix(), FrustumPlanes);

		FGlobalClusterCullingCS::FParameters CullingParams;
		CullingParams.GlobalClusterBuffer = GlobalClusterBufferSRV;
		CullingParams.ProxyMetadataBuffer = ProxyMetadataBufferSRV;
		CullingParams.ShadowClusterVisibilityBitmap = ShadowClusterVisibilityBitmapUAV;
		CullingParams.ShadowSelectedClusterBuffer = ShadowSelectedClusterBufferUAV;
		CullingParams.ShadowLODClusterSelectedBitmap = ShadowLODClusterSelectedBitmapUAV;
		CullingParams.ShadowVisibleClusterCountBuffer = ShadowVisibleClusterCountBufferUAV;
		CullingParams.ShadowLODClusterBuffer = ShadowLODClusterBufferUAV;
		CullingParams.ShadowLODClusterCountBuffer = ShadowLODClusterCountBufferUAV;
		CullingParams.ShadowLODSplatTotalBuffer = ShadowLODSplatTotalBufferUAV;
		CullingParams.ProxyCount = GetProxyCount();
		CullingParams.TotalLeafClusterCount = TotalLeafClusterCount;
		CullingParams.TotalClusterCount = TotalClusterCount;
		CullingParams.TotalVisibilityBitmapUints = TotalVisibilityBitmapUints;
		CullingParams.WorldToClip = FMatrix44f(View.ViewMatrices.GetViewProjectionMatrix());
		for (int32 i = 0; i < 6; i++)
		{
			CullingParams.FrustumPlanes[i] = FrustumPlanes[i];
		}
		CullingParams.CameraPosition = FVector3f(View.ViewMatrices.GetViewOrigin());
		const FMatrix& ProjMatrix = View.ViewMatrices.GetProjectionMatrix();
		CullingParams.ScreenHeight = FMath::Max(ProjMatrix.M[0][0], ProjMatrix.M[1][1]);
		CullingParams.ErrorThreshold = 0.1f;  // Fallback only — shader reads per-proxy threshold from metadata.ProxyErrorThreshold
		CullingParams.LODBias = 0.0f;
		CullingParams.UseLODRendering = 1;
		CullingParams.DebugForceLODLevel = CVarDebugForceLODLevel.GetValueOnRenderThread();

		const uint32 NumGroups = FMath::DivideAndRoundUp(TotalLeafClusterCount, 64u);

		SetComputePipelineState(RHICmdList, CullingShader.GetComputeShader());
		SetShaderParameters(RHICmdList, CullingShader, CullingShader.GetComputeShader(), CullingParams);
		RHICmdList.DispatchComputeShader(NumGroups, 1, 1);
		UnsetShaderUAVs(RHICmdList, CullingShader, CullingShader.GetComputeShader());
	}

	// Transition shadow buffers to SRVCompute — ready for validation reads
	RHICmdList.Transition(FRHITransitionInfo(ShadowClusterVisibilityBitmap, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowSelectedClusterBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowVisibleClusterCountBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowLODClusterCountBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowLODSplatTotalBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));

	// ---------------------------------------------------------------
	// Validation: compare per-proxy visible cluster counts (old path)
	// against shadow bitmap popcount per proxy (new path)
	// ---------------------------------------------------------------
	if (CVarValidateGlobalCulling.GetValueOnRenderThread() > 0 && ValidProxies.Num() > 0)
	{
		// Read back the shadow visibility bitmap to CPU
		const uint32 BitmapBytes = TotalVisibilityBitmapUints * sizeof(uint32);
		TArray<uint32> ShadowBitmap;
		ShadowBitmap.SetNumZeroed(TotalVisibilityBitmapUints);
		{
			void* Src = RHICmdList.LockBuffer(ShadowClusterVisibilityBitmap, 0, BitmapBytes, RLM_ReadOnly);
			FMemory::Memcpy(ShadowBitmap.GetData(), Src, BitmapBytes);
			RHICmdList.UnlockBuffer(ShadowClusterVisibilityBitmap);
		}

		// Read back total global visible count
		uint32 GlobalTotalVisible = 0;
		{
			void* Src = RHICmdList.LockBuffer(ShadowVisibleClusterCountBuffer, 0, sizeof(uint32), RLM_ReadOnly);
			FMemory::Memcpy(&GlobalTotalVisible, Src, sizeof(uint32));
			RHICmdList.UnlockBuffer(ShadowVisibleClusterCountBuffer);
		}

		// Build per-metadata-index bitmap offset table
		// LastProxySet is in metadata order (same order as ProxyMetadataBuffer)
		TArray<uint32> MetadataBitmapStarts;
		MetadataBitmapStarts.SetNumZeroed(LastProxySet.Num());
		{
			uint32 CumBitmapUints = 0;
			for (int32 m = 0; m < LastProxySet.Num(); m++)
			{
				MetadataBitmapStarts[m] = CumBitmapUints;
				FGaussianSplatGPUResources* MRes = LastProxySet[m]->GetGPUResources();
				CumBitmapUints += FMath::DivideAndRoundUp((uint32)MRes->ClusterCount, 32u);
			}
		}

		// Per-proxy comparison
		// ValidProxies may be in a different order than LastProxySet (metadata order)
		// so we match each proxy by pointer identity to find the correct bitmap section
		uint32 OldTotalVisible = 0;
		uint32 NewTotalVisible = 0;
		bool bAllMatch = true;

		for (int32 i = 0; i < ValidProxies.Num(); i++)
		{
			FGaussianSplatGPUResources* Res = ValidProxies[i]->GetGPUResources();
			if (!Res) continue;

			// Find this proxy's metadata index
			int32 MetadataIdx = LastProxySet.Find(ValidProxies[i]);
			if (MetadataIdx == INDEX_NONE)
			{
				UE_LOG(LogTemp, Warning, TEXT("[GS Stage2] Proxy[%d]: not found in metadata — skipping"), i);
				continue;
			}

			// Old path: read per-proxy VisibleClusterCountBuffer[0]
			uint32 OldCount = 0;
			{
				RHICmdList.Transition(FRHITransitionInfo(Res->VisibleClusterCountBuffer, ERHIAccess::Unknown, ERHIAccess::CopySrc));
				void* Src = RHICmdList.LockBuffer(Res->VisibleClusterCountBuffer, 0, sizeof(uint32), RLM_ReadOnly);
				FMemory::Memcpy(&OldCount, Src, sizeof(uint32));
				RHICmdList.UnlockBuffer(Res->VisibleClusterCountBuffer);
			}

			// New path: popcount this proxy's section of the shadow bitmap
			uint32 ProxyBitmapUints = FMath::DivideAndRoundUp((uint32)Res->ClusterCount, 32u);
			uint32 BitmapStart = MetadataBitmapStarts[MetadataIdx];
			uint32 NewCount = 0;
			for (uint32 j = 0; j < ProxyBitmapUints && (BitmapStart + j) < (uint32)ShadowBitmap.Num(); j++)
			{
				NewCount += FMath::CountBits(ShadowBitmap[BitmapStart + j]);
			}

			bool bMatch = (OldCount == NewCount);
			if (!bMatch) bAllMatch = false;

			UE_LOG(LogTemp, Log,
				TEXT("[GS Stage2] Proxy[%d] (meta=%d): old=%u visible clusters  new=%u %s"),
				i, MetadataIdx, OldCount, NewCount, bMatch ? TEXT("MATCH") : TEXT("MISMATCH"));

			OldTotalVisible += OldCount;
			NewTotalVisible += NewCount;
		}

		UE_LOG(LogTemp, Log,
			TEXT("[GS Stage2] Total: old=%u  new(bitmap)=%u  new(counter)=%u  %s"),
			OldTotalVisible, NewTotalVisible, GlobalTotalVisible,
			bAllMatch ? TEXT("All proxies MATCH") : TEXT("MISMATCH DETECTED"));
	}
}

// ============================================================================
// Stage 3: EnsureShadowCompactBuffers
// ============================================================================

static TAutoConsoleVariable<int32> CVarValidateGlobalCompact(
	TEXT("gs.ValidateGlobalCompact"),
	1,
	TEXT("Stage 3: Compare per-proxy visible splat counts against global compact.\n")
	TEXT("0 = disabled, 1 = enabled (GPU readback + CPU compare, stalls GPU)"),
	ECVF_RenderThreadSafe);

void FGlobalSplatBufferManager::EnsureShadowCompactBuffers(FRHICommandListImmediate& RHICmdList)
{
	if (bShadowCompactBuffersAllocated)
	{
		return;
	}

	const uint32 UintStride = sizeof(uint32);

	// Compacted splat indices: worst case = TotalSplatCount
	uint32 MaxSplats = FMath::Max(TotalSplatCount, 1u);
	ShadowCompactedSplatIndices = CreateStructuredBuffer(RHICmdList, TEXT("ShadowCompactedSplatIndices"), UintStride, MaxSplats);
	ShadowCompactedSplatIndicesUAV = RHICmdList.CreateUnorderedAccessView(
		ShadowCompactedSplatIndices,
		FRHIViewDesc::CreateBufferUAV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(UintStride));

	// Visible count array: one uint32 per proxy
	uint32 ProxyCount = FMath::Max(GetProxyCount(), 1u);
	ShadowVisibleCountArray = CreateStructuredBuffer(RHICmdList, TEXT("ShadowVisibleCountArray"), UintStride, ProxyCount);
	ShadowVisibleCountArrayUAV = RHICmdList.CreateUnorderedAccessView(
		ShadowVisibleCountArray,
		FRHIViewDesc::CreateBufferUAV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(UintStride));
	ShadowVisibleCountArraySRV = RHICmdList.CreateShaderResourceView(
		ShadowVisibleCountArray,
		FRHIViewDesc::CreateBufferSRV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(UintStride));

	bShadowCompactBuffersAllocated = true;

	UE_LOG(LogTemp, Log, TEXT("[GS Stage3] Shadow compact buffers allocated: MaxSplats=%u ProxyCount=%u"),
		MaxSplats, ProxyCount);
}

// ============================================================================
// Stage 3: DispatchGlobalCompactSplats
// ============================================================================

void FGlobalSplatBufferManager::DispatchGlobalCompactSplats(
	FRHICommandListImmediate& RHICmdList,
	const TArray<FGaussianSplatSceneProxy*>& ValidProxies)
{
	if (!IsReady() || TotalSplatCount == 0 || GetProxyCount() == 0)
	{
		return;
	}

	// Stage 2 shadow bitmaps must exist (DispatchGlobalClusterCulling must run first)
	if (!bShadowBuffersAllocated)
	{
		return;
	}

	SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatGlobalCompactSplats_Shadow);

	// Ensure shadow compact buffers exist
	EnsureShadowCompactBuffers(RHICmdList);

	TShaderMapRef<FGlobalCompactSplatsResetCS> ResetShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	TShaderMapRef<FGlobalCompactSplatsCS> CompactShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	if (!ResetShader.IsValid() || !CompactShader.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("[GS Stage3] Global compact splats shaders not valid"));
		return;
	}

	// Transition shadow compact buffers to UAVCompute
	RHICmdList.Transition(FRHITransitionInfo(ShadowCompactedSplatIndices, ERHIAccess::Unknown, ERHIAccess::UAVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowVisibleCountArray, ERHIAccess::Unknown, ERHIAccess::UAVCompute));

	// Transition Stage 2 shadow bitmaps to SRVCompute (they should already be there after Stage 2)
	RHICmdList.Transition(FRHITransitionInfo(ShadowClusterVisibilityBitmap, ERHIAccess::Unknown, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowSelectedClusterBuffer, ERHIAccess::Unknown, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowLODClusterSelectedBitmap, ERHIAccess::Unknown, ERHIAccess::SRVCompute));

	// Transition global input buffers to SRVCompute
	RHICmdList.Transition(FRHITransitionInfo(GlobalSplatClusterIndexBuffer, ERHIAccess::Unknown, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ProxyMetadataBuffer, ERHIAccess::Unknown, ERHIAccess::SRVCompute));

	// Create SRVs for shadow bitmaps (read-only in this stage)
	FShaderResourceViewRHIRef ShadowClusterVisibilityBitmapSRV = RHICmdList.CreateShaderResourceView(
		ShadowClusterVisibilityBitmap,
		FRHIViewDesc::CreateBufferSRV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(sizeof(uint32)));
	FShaderResourceViewRHIRef ShadowSelectedClusterBufferSRV = RHICmdList.CreateShaderResourceView(
		ShadowSelectedClusterBuffer,
		FRHIViewDesc::CreateBufferSRV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(sizeof(uint32)));
	FShaderResourceViewRHIRef ShadowLODClusterSelectedBitmapSRV = RHICmdList.CreateShaderResourceView(
		ShadowLODClusterSelectedBitmap,
		FRHIViewDesc::CreateBufferSRV().SetType(FRHIViewDesc::EBufferType::Structured).SetStride(sizeof(uint32)));

	// Step 1: Reset visible count array
	{
		FGlobalCompactSplatsResetCS::FParameters ResetParams;
		ResetParams.ShadowVisibleCountArray = ShadowVisibleCountArrayUAV;
		ResetParams.ProxyCount = GetProxyCount();
		ResetParams.TotalSplatCount = TotalSplatCount;
		ResetParams.UseLODRendering = 1;

		uint32 NumGroups = FMath::DivideAndRoundUp(GetProxyCount(), 256u);
		NumGroups = FMath::Max(NumGroups, 1u);

		SetComputePipelineState(RHICmdList, ResetShader.GetComputeShader());
		SetShaderParameters(RHICmdList, ResetShader, ResetShader.GetComputeShader(), ResetParams);
		RHICmdList.DispatchComputeShader(NumGroups, 1, 1);
		UnsetShaderUAVs(RHICmdList, ResetShader, ResetShader.GetComputeShader());
	}

	// UAV barrier between reset and compaction
	RHICmdList.Transition(FRHITransitionInfo(ShadowVisibleCountArray, ERHIAccess::UAVCompute, ERHIAccess::UAVCompute));

	// Step 2: Global compact splats dispatch
	{
		FGlobalCompactSplatsCS::FParameters CompactParams;
		CompactParams.GlobalSplatClusterIndexBuffer = GlobalSplatClusterIndexBufferSRV;
		CompactParams.ProxyMetadataBuffer = ProxyMetadataBufferSRV;
		CompactParams.ShadowClusterVisibilityBitmap = ShadowClusterVisibilityBitmapSRV;
		CompactParams.ShadowSelectedClusterBuffer = ShadowSelectedClusterBufferSRV;
		CompactParams.ShadowLODClusterSelectedBitmap = ShadowLODClusterSelectedBitmapSRV;
		CompactParams.ShadowCompactedSplatIndices = ShadowCompactedSplatIndicesUAV;
		CompactParams.ShadowVisibleCountArray = ShadowVisibleCountArrayUAV;
		CompactParams.ProxyCount = GetProxyCount();
		CompactParams.TotalSplatCount = TotalSplatCount;
		CompactParams.UseLODRendering = 1;

		const uint32 NumGroups = FMath::DivideAndRoundUp(TotalSplatCount, 256u);

		SetComputePipelineState(RHICmdList, CompactShader.GetComputeShader());
		SetShaderParameters(RHICmdList, CompactShader, CompactShader.GetComputeShader(), CompactParams);
		RHICmdList.DispatchComputeShader(NumGroups, 1, 1);
		UnsetShaderUAVs(RHICmdList, CompactShader, CompactShader.GetComputeShader());
	}

	// Transition shadow compact buffers to readable state for validation
	RHICmdList.Transition(FRHITransitionInfo(ShadowVisibleCountArray, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));
	RHICmdList.Transition(FRHITransitionInfo(ShadowCompactedSplatIndices, ERHIAccess::UAVCompute, ERHIAccess::SRVCompute));

	// ---------------------------------------------------------------
	// Validation: compare per-proxy visible splat counts (old path)
	// against ShadowVisibleCountArray per proxy (new path)
	// ---------------------------------------------------------------
	if (CVarValidateGlobalCompact.GetValueOnRenderThread() > 0 && ValidProxies.Num() > 0)
	{
		// Read back the shadow visible count array to CPU
		const uint32 CountBytes = GetProxyCount() * sizeof(uint32);
		TArray<uint32> ShadowCounts;
		ShadowCounts.SetNumZeroed(GetProxyCount());
		{
			void* Src = RHICmdList.LockBuffer(ShadowVisibleCountArray, 0, CountBytes, RLM_ReadOnly);
			FMemory::Memcpy(ShadowCounts.GetData(), Src, CountBytes);
			RHICmdList.UnlockBuffer(ShadowVisibleCountArray);
		}

		// Per-proxy comparison
		bool bAllMatch = true;
		uint32 OldTotal = 0;
		uint32 NewTotal = 0;

		for (int32 i = 0; i < ValidProxies.Num(); i++)
		{
			FGaussianSplatGPUResources* Res = ValidProxies[i]->GetGPUResources();
			if (!Res) continue;

			// Find this proxy's metadata index
			int32 MetadataIdx = LastProxySet.Find(ValidProxies[i]);
			if (MetadataIdx == INDEX_NONE)
			{
				UE_LOG(LogTemp, Warning, TEXT("[GS Stage3] Proxy[%d]: not found in metadata -- skipping"), i);
				continue;
			}

			// Old path: read per-proxy VisibleSplatCountBuffer[0]
			uint32 OldCount = 0;
			{
				RHICmdList.Transition(FRHITransitionInfo(Res->VisibleSplatCountBuffer, ERHIAccess::Unknown, ERHIAccess::CopySrc));
				void* Src = RHICmdList.LockBuffer(Res->VisibleSplatCountBuffer, 0, sizeof(uint32), RLM_ReadOnly);
				FMemory::Memcpy(&OldCount, Src, sizeof(uint32));
				RHICmdList.UnlockBuffer(Res->VisibleSplatCountBuffer);
			}

			// New path: read from ShadowVisibleCountArray
			uint32 NewCount = ((uint32)MetadataIdx < GetProxyCount()) ? ShadowCounts[MetadataIdx] : 0;

			bool bMatch = (OldCount == NewCount);
			if (!bMatch) bAllMatch = false;

			UE_LOG(LogTemp, Log,
				TEXT("[GS Stage3] Proxy[%d] (meta=%d): old=%u visible splats  new=%u %s"),
				i, MetadataIdx, OldCount, NewCount, bMatch ? TEXT("MATCH") : TEXT("MISMATCH"));

			OldTotal += OldCount;
			NewTotal += NewCount;
		}

		UE_LOG(LogTemp, Log,
			TEXT("[GS Stage3] Total: old=%u  new=%u  %s"),
			OldTotal, NewTotal,
			bAllMatch ? TEXT("All proxies MATCH") : TEXT("MISMATCH DETECTED"));
	}
}
