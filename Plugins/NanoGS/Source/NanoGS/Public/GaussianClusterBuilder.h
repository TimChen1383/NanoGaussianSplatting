// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GaussianDataTypes.h"
#include "GaussianClusterTypes.h"

/**
 * Builder class for constructing hierarchical cluster structures from Gaussian splat data.
 * Implements Nanite-style spatial clustering for efficient LOD and culling.
 *
 * Algorithm overview:
 * 1. Sort splats by Morton code (Z-order curve) for spatial locality
 * 2. Group into leaf clusters of N splats each
 * 3. Recursively merge clusters into parent nodes
 * 4. Calculate bounding volumes and error metrics
 */
class NANOGS_API FGaussianClusterBuilder
{
public:
	/**
	 * Configuration for cluster building
	 */
	struct FBuildSettings
	{
		/** Number of splats per leaf cluster (default: 128) */
		int32 SplatsPerCluster;

		/** Maximum children per parent cluster (default: 8) */
		int32 MaxChildrenPerCluster;

		/** Whether to reorder splat data for cluster locality */
		bool bReorderSplats;

		/** Whether to generate LOD splats for parent clusters */
		bool bGenerateLOD;

		/** LOD reduction ratio - how many original splats per LOD splat (default: divide by 4) */
		int32 LODReductionRatio;

		// Explicit ctor: Clang rejects nested `= default` + DMIs referencing
		// enclosing-scope constants when used as a default function argument.
		FBuildSettings()
			: SplatsPerCluster(GaussianClusterConstants::DefaultSplatsPerCluster)
			, MaxChildrenPerCluster(GaussianClusterConstants::MaxChildrenPerCluster)
			, bReorderSplats(true)
			, bGenerateLOD(true)
			, LODReductionRatio(4)
		{
		}
	};

	/**
	 * Build cluster hierarchy from splat data
	 *
	 * @param InOutSplats Splat data array - will be reordered if bReorderSplats is true
	 * @param OutHierarchy Output cluster hierarchy
	 * @param Settings Build configuration
	 * @return true if successful, false on error
	 */
	static bool BuildClusterHierarchy(
		TArray<FGaussianSplatData>& InOutSplats,
		FGaussianClusterHierarchy& OutHierarchy,
		const FBuildSettings& Settings = FBuildSettings());

private:
	/**
	 * Internal structure for sorting splats by Morton code
	 */
	struct FSplatSortEntry
	{
		uint64 MortonCode;
		int32 OriginalIndex;

		bool operator<(const FSplatSortEntry& Other) const
		{
			return MortonCode < Other.MortonCode;
		}
	};

	/**
	 * Calculate global bounding box of all splats
	 */
	static FBox CalculateGlobalBounds(const TArray<FGaussianSplatData>& Splats);

	/**
	 * Sort splats by Morton code and optionally reorder the array
	 *
	 * @param InOutSplats Splat array to sort
	 * @param GlobalBounds Bounding box for Morton code normalization
	 * @param bReorder If true, physically reorder the splat array
	 * @param OutSortedIndices Output array of sorted indices (original index order)
	 */
	static void SortSplatsByMortonCode(
		TArray<FGaussianSplatData>& InOutSplats,
		const FBox& GlobalBounds,
		bool bReorder,
		TArray<int32>& OutSortedIndices);

	/**
	 * Create leaf clusters from sorted splats
	 *
	 * @param Splats Sorted splat array
	 * @param SplatsPerCluster Number of splats per leaf cluster
	 * @param OutClusters Output array of leaf clusters
	 */
	static void CreateLeafClusters(
		const TArray<FGaussianSplatData>& Splats,
		int32 SplatsPerCluster,
		TArray<FGaussianCluster>& OutClusters);

	/**
	 * Build parent level by merging child clusters
	 *
	 * @param ChildClusters Clusters from previous level
	 * @param MaxChildrenPerCluster Maximum children per parent
	 * @param ParentLODLevel LOD level for parent clusters
	 * @param NextClusterID Starting cluster ID for new clusters
	 * @param OutParentClusters Output array of parent clusters
	 */
	static void BuildParentLevel(
		TArray<FGaussianCluster>& ChildClusters,
		int32 MaxChildrenPerCluster,
		uint32 ParentLODLevel,
		uint32& NextClusterID,
		TArray<FGaussianCluster>& OutParentClusters);

	/**
	 * Calculate bounding volume for a cluster from its splats
	 *
	 * @param Cluster Cluster to update
	 * @param Splats All splat data
	 */
	static void CalculateClusterBounds(
		FGaussianCluster& Cluster,
		const TArray<FGaussianSplatData>& Splats);

	/**
	 * Calculate bounding volume for a parent cluster from its children
	 *
	 * @param ParentCluster Parent cluster to update
	 * @param ChildClusters All clusters (to look up children)
	 */
	static void CalculateParentClusterBounds(
		FGaussianCluster& ParentCluster,
		const TArray<FGaussianCluster>& AllClusters);

	/**
	 * Calculate max error for a parent cluster
	 * Error = maximum distance from any child cluster center to parent center
	 *
	 * @param ParentCluster Parent cluster
	 * @param AllClusters All clusters (to look up children)
	 */
	static void CalculateClusterError(
		FGaussianCluster& ParentCluster,
		const TArray<FGaussianCluster>& AllClusters);

	//----------------------------------------------------------------------
	// LOD Generation (Unified Approach - LOD splats use same format as original splats)
	//----------------------------------------------------------------------

	/**
	 * Merge multiple Gaussians into a single representative Gaussian
	 * Used to create simplified LOD representations
	 * Returns FGaussianSplatData for unified buffer approach
	 *
	 * @param Splats Array of splats to merge
	 * @param StartIndex First splat index
	 * @param Count Number of splats to merge
	 * @return Merged splat in unified format
	 */
	static FGaussianSplatData MergeGaussians(
		const TArray<FGaussianSplatData>& Splats,
		int32 StartIndex,
		int32 Count);

	/**
	 * Merge multiple LOD splats into a single representative
	 * Used for higher LOD levels
	 * Works with FGaussianSplatData for unified buffer approach
	 *
	 * @param LODSplats Array of LOD splats to merge (unified format)
	 * @param StartIndex First splat index
	 * @param Count Number of splats to merge
	 * @return Merged splat in unified format
	 */
	static FGaussianSplatData MergeLODSplats(
		const TArray<FGaussianSplatData>& LODSplats,
		int32 StartIndex,
		int32 Count);

	/**
	 * Generate LOD splats for all non-leaf clusters in the hierarchy
	 *
	 * @param Splats Original splat data (for leaf cluster LOD generation)
	 * @param Hierarchy Cluster hierarchy to populate with LOD data
	 * @param ReductionRatio How many source splats per LOD splat
	 */
	static void GenerateLODSplats(
		const TArray<FGaussianSplatData>& Splats,
		FGaussianClusterHierarchy& Hierarchy,
		int32 ReductionRatio);
};
