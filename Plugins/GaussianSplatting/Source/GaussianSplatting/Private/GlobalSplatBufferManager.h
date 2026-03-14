// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "RHI.h"
#include "RHIResources.h"

class FGaussianSplatSceneProxy;

/**
 * Per-proxy GPU metadata uploaded to ProxyMetadataBuffer each frame.
 * Must be exactly 128 bytes to match the HLSL struct in global shaders (Stages 2-5).
 *
 * Layout (row-major, float32):
 *   [0..63]   float LocalToWorld[16]           4x4 transform matrix
 *   [64..127] uint32 fields x 16               offsets, counts, cumulative totals
 */
struct FProxyGPUMetadata
{
	// Transform: row-major float4x4 (64 bytes)
	float LocalToWorld[16];

	// Splat buffer offsets
	uint32 GlobalSplatStartIndex;        // Byte offset / PackedSplatStride into GlobalPackedSplatBuffer
	uint32 TotalSplatCount;              // Total splats for this proxy (original + LOD)
	uint32 OriginalSplatCount;           // Original splats only (excludes LOD splats)

	// Cluster buffer offsets
	uint32 GlobalClusterStartIndex;      // Element offset into GlobalClusterBuffer
	uint32 ClusterCount;                 // Total clusters for this proxy
	uint32 LeafClusterCount;             // Leaf clusters only (used by culling dispatch size)

	// Dynamic bitmap offsets (uint32 units) — populated in Stage 2
	uint32 GlobalVisibilityBitmapStart;  // uint32 offset into GlobalClusterVisibilityBitmap
	uint32 GlobalSelectedClusterStart;   // Element offset into GlobalSelectedClusterBuffer
	uint32 GlobalLODBitmapStart;         // uint32 offset into GlobalLODClusterSelectedBitmap

	// Compaction
	uint32 GlobalSplatClusterIndexStart; // Element offset into GlobalSplatClusterIndexBuffer
	uint32 CompactionCounterIndex;       // Index into GlobalVisibleCountArray (equals proxy index)
	uint32 CompactionOutputStart;        // Element offset into GlobalCompactedSplatIndices (worst-case slot)

	// SH
	uint32 GlobalSHStartIndex;           // Element offset into GlobalSHBuffer — TODO Stage 4

	// Cumulative totals for binary search in global shaders
	uint32 GlobalLeafClusterEnd;         // Exclusive end of this proxy's leaf clusters in global space
	uint32 GlobalSplatEnd;               // Exclusive end of this proxy's splats in global space

	// Padding to reach 128 bytes
	uint32 Pad[1];
};

static_assert(sizeof(FProxyGPUMetadata) == 128, "FProxyGPUMetadata must be 128 bytes — HLSL struct must match");

/**
 * Manages global GPU buffers that concatenate data across all registered proxies.
 *
 * Stage 1 (this file): CPU offset table, ProxyMetadataBuffer, and static buffer
 *   concatenation (GlobalPackedSplatBuffer, GlobalClusterBuffer,
 *   GlobalSplatClusterIndexBuffer). No rendering changes.
 *
 * Stages 2-5 will add shadow/global bitmaps, compaction buffers, and view data.
 *
 * Owned by FGaussianSplattingModule (render-thread lifetime), like FGaussianGlobalAccumulator.
 */
struct FGlobalSplatBufferManager
{
	//------------------------------------------------------------------------
	// Global static GPU buffers (rebuilt when the registered proxy set changes)
	//------------------------------------------------------------------------

	/** All proxy packed splat data concatenated. ByteAddressBuffer, 16 bytes/splat. */
	FBufferRHIRef GlobalPackedSplatBuffer;
	FShaderResourceViewRHIRef GlobalPackedSplatBufferSRV;

	/** All proxy cluster data concatenated. StructuredBuffer<FGaussianGPUCluster>, 80 bytes/cluster. */
	FBufferRHIRef GlobalClusterBuffer;
	FShaderResourceViewRHIRef GlobalClusterBufferSRV;

	/** All proxy splat-to-cluster index data concatenated. StructuredBuffer<uint>, 4 bytes/entry. */
	FBufferRHIRef GlobalSplatClusterIndexBuffer;
	FShaderResourceViewRHIRef GlobalSplatClusterIndexBufferSRV;

	//------------------------------------------------------------------------
	// Per-frame dynamic metadata (rebuilt every frame for updated transforms)
	//------------------------------------------------------------------------

	/** One FProxyGPUMetadata per proxy. StructuredBuffer, 128 bytes/entry. */
	FBufferRHIRef ProxyMetadataBuffer;
	FShaderResourceViewRHIRef ProxyMetadataBufferSRV;

	//------------------------------------------------------------------------
	// CPU-side state
	//------------------------------------------------------------------------

	/** Last known valid proxy set — used to detect changes requiring static buffer rebuild. */
	TArray<FGaussianSplatSceneProxy*> LastProxySet;

	uint32 TotalSplatCount = 0;
	uint32 TotalClusterCount = 0;
	uint32 TotalLeafClusterCount = 0;

	/** Total uint32s needed for all per-proxy visibility bitmaps combined (for Stage 2). */
	uint32 TotalVisibilityBitmapUints = 0;

	/** Allocated capacity of ProxyMetadataBuffer (in proxy count). Recreated when proxy count grows. */
	uint32 AllocatedMetadataCount = 0;

	bool bStaticBuffersBuilt = false;

	//------------------------------------------------------------------------
	// Interface
	//------------------------------------------------------------------------

	/**
	 * Call once per frame from the render loop.
	 * - Filters AllRegisteredProxies to valid Nanite proxies.
	 * - Rebuilds static GPU buffers if the proxy set changed.
	 * - Uploads ProxyMetadataBuffer (transforms + offsets) every frame.
	 *
	 * Returns true if static buffers were rebuilt this frame.
	 */
	bool UpdateIfNeeded(
		FRHICommandListImmediate& RHICmdList,
		const TArray<FGaussianSplatSceneProxy*>& AllRegisteredProxies);

	/** Release all GPU buffers. Must be called from the render thread before destruction. */
	void Release();

	bool IsReady() const { return bStaticBuffersBuilt && ProxyMetadataBuffer.IsValid(); }
	uint32 GetProxyCount() const { return (uint32)LastProxySet.Num(); }

private:
	/** GPU-to-GPU copy of per-proxy static buffers into global concatenated buffers. */
	void RebuildStaticBuffers(
		FRHICommandListImmediate& RHICmdList,
		const TArray<FGaussianSplatSceneProxy*>& ValidProxies);

	/** Build FProxyGPUMetadata array and upload to ProxyMetadataBuffer. */
	void UploadProxyMetadata(
		FRHICommandListImmediate& RHICmdList,
		const TArray<FGaussianSplatSceneProxy*>& ValidProxies,
		bool bLogDetails);
};
