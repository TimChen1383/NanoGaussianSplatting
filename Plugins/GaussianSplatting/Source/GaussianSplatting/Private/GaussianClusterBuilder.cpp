// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianClusterBuilder.h"

bool FGaussianClusterBuilder::BuildClusterHierarchy(
	TArray<FGaussianSplatData>& InOutSplats,
	FGaussianClusterHierarchy& OutHierarchy,
	const FBuildSettings& Settings)
{
	// Reset output
	OutHierarchy.Reset();

	const int32 NumSplats = InOutSplats.Num();
	if (NumSplats == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("FGaussianClusterBuilder: No splats to cluster"));
		return false;
	}

	UE_LOG(LogTemp, Log, TEXT("FGaussianClusterBuilder: Building cluster hierarchy for %d splats"), NumSplats);

	// Store settings in hierarchy
	OutHierarchy.SplatsPerCluster = Settings.SplatsPerCluster;
	OutHierarchy.TotalSplatCount = NumSplats;

	// Step 1: Calculate global bounds
	const FBox GlobalBounds = CalculateGlobalBounds(InOutSplats);
	UE_LOG(LogTemp, Log, TEXT("  Global bounds: Min(%s) Max(%s)"),
		*GlobalBounds.Min.ToString(), *GlobalBounds.Max.ToString());

	// Step 2: Sort splats by Morton code for spatial locality
	TArray<int32> SortedIndices;
	SortSplatsByMortonCode(InOutSplats, GlobalBounds, Settings.bReorderSplats, SortedIndices);
	UE_LOG(LogTemp, Log, TEXT("  Sorted splats by Morton code"));

	// Step 3: Create leaf clusters
	TArray<FGaussianCluster> LeafClusters;
	CreateLeafClusters(InOutSplats, Settings.SplatsPerCluster, LeafClusters);
	OutHierarchy.NumLeafClusters = LeafClusters.Num();
	UE_LOG(LogTemp, Log, TEXT("  Created %d leaf clusters"), LeafClusters.Num());

	// Step 4: Build hierarchy bottom-up
	TArray<FGaussianCluster> CurrentLevel = MoveTemp(LeafClusters);
	TArray<FGaussianCluster> AllClusters;
	uint32 CurrentLODLevel = 0;
	uint32 NextClusterID = CurrentLevel.Num();

	// Add leaf clusters to all clusters
	AllClusters.Append(CurrentLevel);

	// Build parent levels until we have a single root
	while (CurrentLevel.Num() > 1)
	{
		CurrentLODLevel++;
		TArray<FGaussianCluster> ParentLevel;

		BuildParentLevel(
			CurrentLevel,
			Settings.MaxChildrenPerCluster,
			CurrentLODLevel,
			NextClusterID,
			ParentLevel);

		UE_LOG(LogTemp, Log, TEXT("  LOD Level %d: %d clusters"), CurrentLODLevel, ParentLevel.Num());

		// Update child clusters in AllClusters with their new ParentClusterID
		// (BuildParentLevel updated CurrentLevel, but AllClusters has old copies)
		for (const FGaussianCluster& UpdatedChild : CurrentLevel)
		{
			// Find and update the corresponding cluster in AllClusters
			for (FGaussianCluster& StoredCluster : AllClusters)
			{
				if (StoredCluster.ClusterID == UpdatedChild.ClusterID)
				{
					StoredCluster.ParentClusterID = UpdatedChild.ParentClusterID;
					break;
				}
			}
		}

		// Add parent clusters to all clusters
		AllClusters.Append(ParentLevel);

		// Move to next level
		CurrentLevel = MoveTemp(ParentLevel);
	}

	// Step 5: Error metrics are computed during LOD generation (Step 6)
	// MaxError is set based on actual merge displacement, not cluster radius

	// Store results
	OutHierarchy.Clusters = MoveTemp(AllClusters);
	OutHierarchy.NumLODLevels = CurrentLODLevel + 1;
	OutHierarchy.RootClusterIndex = OutHierarchy.Clusters.Num() - 1; // Root is last cluster added

	UE_LOG(LogTemp, Log, TEXT("FGaussianClusterBuilder: Hierarchy complete"));
	UE_LOG(LogTemp, Log, TEXT("  Total clusters: %d"), OutHierarchy.Clusters.Num());
	UE_LOG(LogTemp, Log, TEXT("  LOD levels: %d"), OutHierarchy.NumLODLevels);
	UE_LOG(LogTemp, Log, TEXT("  Root cluster index: %d"), OutHierarchy.RootClusterIndex);

	// Step 6: Generate LOD splats for non-leaf clusters
	// This also computes MaxError based on actual merge displacement
	if (Settings.bGenerateLOD && OutHierarchy.NumLODLevels > 1)
	{
		GenerateLODSplats(InOutSplats, OutHierarchy, Settings.LODReductionRatio);
		UE_LOG(LogTemp, Log, TEXT("  Generated %d LOD splats"), OutHierarchy.TotalLODSplatCount);

		// Log error metrics per LOD level for debugging
		for (uint32 Level = 1; Level <= OutHierarchy.NumLODLevels; Level++)
		{
			float MinError = MAX_FLT, MaxError = 0.0f, SumError = 0.0f;
			float MinRadius = MAX_FLT, MaxRadius = 0.0f, SumRadius = 0.0f;
			int32 Count = 0;
			for (const FGaussianCluster& C : OutHierarchy.Clusters)
			{
				if (C.LODLevel == Level)
				{
					MinError = FMath::Min(MinError, C.MaxError);
					MaxError = FMath::Max(MaxError, C.MaxError);
					SumError += C.MaxError;
					MinRadius = FMath::Min(MinRadius, C.BoundingSphereRadius);
					MaxRadius = FMath::Max(MaxRadius, C.BoundingSphereRadius);
					SumRadius += C.BoundingSphereRadius;
					Count++;
				}
			}
			if (Count > 0)
			{
				UE_LOG(LogTemp, Log, TEXT("  LOD Level %d: MaxError min=%.2f avg=%.2f max=%.2f, Radius min=%.2f avg=%.2f max=%.2f (%d clusters)"),
					Level, MinError, SumError / Count, MaxError,
					MinRadius, SumRadius / Count, MaxRadius, Count);
			}
		}
	}

	return true;
}

FBox FGaussianClusterBuilder::CalculateGlobalBounds(const TArray<FGaussianSplatData>& Splats)
{
	FBox Bounds(ForceInit);

	for (const FGaussianSplatData& Splat : Splats)
	{
		Bounds += FVector(Splat.Position);
	}

	// Ensure non-zero size bounds
	if (Bounds.GetSize().IsNearlyZero())
	{
		Bounds = Bounds.ExpandBy(1.0f);
	}

	return Bounds;
}

void FGaussianClusterBuilder::SortSplatsByMortonCode(
	TArray<FGaussianSplatData>& InOutSplats,
	const FBox& GlobalBounds,
	bool bReorder,
	TArray<int32>& OutSortedIndices)
{
	const int32 NumSplats = InOutSplats.Num();

	// Create sort entries with Morton codes
	TArray<FSplatSortEntry> SortEntries;
	SortEntries.SetNum(NumSplats);

	const FVector3f BoundsMin(GlobalBounds.Min);
	const FVector3f BoundsMax(GlobalBounds.Max);

	for (int32 i = 0; i < NumSplats; i++)
	{
		SortEntries[i].OriginalIndex = i;
		SortEntries[i].MortonCode = GaussianClusterUtils::EncodeMorton3D(
			InOutSplats[i].Position, BoundsMin, BoundsMax);
	}

	// Sort by Morton code
	SortEntries.Sort();

	// Extract sorted indices
	OutSortedIndices.SetNum(NumSplats);
	for (int32 i = 0; i < NumSplats; i++)
	{
		OutSortedIndices[i] = SortEntries[i].OriginalIndex;
	}

	// Reorder splat array if requested
	if (bReorder)
	{
		TArray<FGaussianSplatData> ReorderedSplats;
		ReorderedSplats.SetNum(NumSplats);

		for (int32 i = 0; i < NumSplats; i++)
		{
			ReorderedSplats[i] = InOutSplats[OutSortedIndices[i]];
		}

		InOutSplats = MoveTemp(ReorderedSplats);

		// Update sorted indices to reflect new order (now identity mapping)
		for (int32 i = 0; i < NumSplats; i++)
		{
			OutSortedIndices[i] = i;
		}
	}
}

void FGaussianClusterBuilder::CreateLeafClusters(
	const TArray<FGaussianSplatData>& Splats,
	int32 SplatsPerCluster,
	TArray<FGaussianCluster>& OutClusters)
{
	const int32 NumSplats = Splats.Num();
	const int32 NumClusters = FMath::DivideAndRoundUp(NumSplats, SplatsPerCluster);

	OutClusters.Reset(NumClusters);

	for (int32 ClusterIdx = 0; ClusterIdx < NumClusters; ClusterIdx++)
	{
		FGaussianCluster Cluster;
		Cluster.ClusterID = static_cast<uint32>(ClusterIdx);
		Cluster.ParentClusterID = GaussianClusterConstants::RootParentID; // Will be set when parent is created
		Cluster.LODLevel = 0; // Leaf level
		Cluster.SplatStartIndex = static_cast<uint32>(ClusterIdx * SplatsPerCluster);
		Cluster.SplatCount = static_cast<uint32>(FMath::Min(SplatsPerCluster, NumSplats - static_cast<int32>(Cluster.SplatStartIndex)));
		Cluster.MaxError = 0.0f; // Leaf clusters have no simplification error

		// Calculate bounds from splats
		CalculateClusterBounds(Cluster, Splats);

		OutClusters.Add(MoveTemp(Cluster));
	}
}

void FGaussianClusterBuilder::BuildParentLevel(
	TArray<FGaussianCluster>& ChildClusters,
	int32 MaxChildrenPerCluster,
	uint32 ParentLODLevel,
	uint32& NextClusterID,
	TArray<FGaussianCluster>& OutParentClusters)
{
	const int32 NumChildren = ChildClusters.Num();
	const int32 NumParents = FMath::DivideAndRoundUp(NumChildren, MaxChildrenPerCluster);

	OutParentClusters.Reset(NumParents);

	// Greedy nearest-neighbor clustering based on spatial proximity
	TArray<bool> Claimed;
	Claimed.SetNumZeroed(NumChildren);

	while (true)
	{
		// Find first unclaimed child as seed
		int32 SeedIdx = INDEX_NONE;
		for (int32 i = 0; i < NumChildren; i++)
		{
			if (!Claimed[i])
			{
				SeedIdx = i;
				break;
			}
		}
		if (SeedIdx == INDEX_NONE)
		{
			break; // All children claimed
		}

		// Gather unclaimed indices and their distances to the seed
		struct FDistEntry
		{
			int32 ChildIdx;
			float DistSq;
		};
		TArray<FDistEntry> Distances;
		const FVector3f SeedCenter = ChildClusters[SeedIdx].BoundingSphereCenter;

		for (int32 i = 0; i < NumChildren; i++)
		{
			if (!Claimed[i] && i != SeedIdx)
			{
				float DistSq = FVector3f::DistSquared(SeedCenter, ChildClusters[i].BoundingSphereCenter);
				Distances.Add({i, DistSq});
			}
		}

		// Sort by distance ascending
		Distances.Sort([](const FDistEntry& A, const FDistEntry& B) { return A.DistSq < B.DistSq; });

		// Build parent from seed + nearest neighbors
		FGaussianCluster Parent;
		Parent.ClusterID = NextClusterID++;
		Parent.ParentClusterID = GaussianClusterConstants::RootParentID;
		Parent.LODLevel = ParentLODLevel;
		Parent.ResetBounds();

		// Claim seed
		Claimed[SeedIdx] = true;
		{
			FGaussianCluster& Child = ChildClusters[SeedIdx];
			Child.ParentClusterID = Parent.ClusterID;
			Parent.ChildClusterIDs.Add(Child.ClusterID);
			Parent.ExpandBounds(Child);
		}

		uint32 TotalSplatCount = ChildClusters[SeedIdx].SplatCount;

		// Claim nearest neighbors up to MaxChildrenPerCluster - 1
		int32 NeighborsToTake = FMath::Min(Distances.Num(), MaxChildrenPerCluster - 1);
		for (int32 i = 0; i < NeighborsToTake; i++)
		{
			int32 ChildIdx = Distances[i].ChildIdx;
			Claimed[ChildIdx] = true;

			FGaussianCluster& Child = ChildClusters[ChildIdx];
			Child.ParentClusterID = Parent.ClusterID;
			Parent.ChildClusterIDs.Add(Child.ClusterID);
			Parent.ExpandBounds(Child);

			TotalSplatCount += Child.SplatCount;
		}

		Parent.SplatCount = TotalSplatCount;
		Parent.SplatStartIndex = 0; // Not meaningful for non-leaf clusters with spatial grouping

		Parent.ComputeBoundingSphereFromAABB();

		OutParentClusters.Add(MoveTemp(Parent));
	}
}

void FGaussianClusterBuilder::CalculateClusterBounds(
	FGaussianCluster& Cluster,
	const TArray<FGaussianSplatData>& Splats)
{
	Cluster.ResetBounds();

	const uint32 EndIndex = Cluster.SplatStartIndex + Cluster.SplatCount;
	for (uint32 i = Cluster.SplatStartIndex; i < EndIndex && i < (uint32)Splats.Num(); i++)
	{
		Cluster.ExpandBounds(Splats[i].Position);
	}

	Cluster.ComputeBoundingSphereFromAABB();
}

void FGaussianClusterBuilder::CalculateParentClusterBounds(
	FGaussianCluster& ParentCluster,
	const TArray<FGaussianCluster>& AllClusters)
{
	ParentCluster.ResetBounds();

	for (uint32 ChildID : ParentCluster.ChildClusterIDs)
	{
		// Find child cluster (assuming ClusterID matches array index for simplicity)
		// In a more robust implementation, use a map
		for (const FGaussianCluster& Cluster : AllClusters)
		{
			if (Cluster.ClusterID == ChildID)
			{
				ParentCluster.ExpandBounds(Cluster);
				break;
			}
		}
	}

	ParentCluster.ComputeBoundingSphereFromAABB();
}

void FGaussianClusterBuilder::CalculateClusterError(
	FGaussianCluster& ParentCluster,
	const TArray<FGaussianCluster>& AllClusters)
{
	// Error metric: use parent's bounding sphere radius, ensuring monotonic increase up the tree.
	//
	// The bounding sphere radius represents the spatial extent being simplified by LOD splats.
	// Larger clusters cover more space, so they have more potential visual error.
	// This naturally increases up the tree (parent radius >= any child radius).
	//
	// The projected error formula: projectedError = (MaxError / distance) * ScreenHeight * 0.5
	// - Near the camera: projectedError is large → use leaf clusters (finest detail)
	// - Far from camera: projectedError is small → accept parent LOD (coarser but cheaper)
	//
	// Ensuring monotonic increase (parent error > max child error) guarantees that the LOD
	// walk-up in ClusterCulling never skips a level — if a parent's error is acceptable,
	// all its children's errors are also acceptable.

	float MaxChildError = 0.0f;
	for (uint32 ChildID : ParentCluster.ChildClusterIDs)
	{
		for (const FGaussianCluster& Child : AllClusters)
		{
			if (Child.ClusterID == ChildID)
			{
				MaxChildError = FMath::Max(MaxChildError, Child.MaxError);
				break;
			}
		}
	}

	// Use parent's bounding sphere radius as error, but ensure strictly greater than any child's error
	ParentCluster.MaxError = FMath::Max(ParentCluster.BoundingSphereRadius, MaxChildError + SMALL_NUMBER);
}

//----------------------------------------------------------------------
// LOD Generation (Unified Approach - LOD splats use same format as original splats)
//----------------------------------------------------------------------

FGaussianSplatData FGaussianClusterBuilder::MergeGaussians(
	const TArray<FGaussianSplatData>& Splats,
	int32 StartIndex,
	int32 Count)
{
	FGaussianSplatData Result;

	if (Count <= 0 || StartIndex < 0 || StartIndex >= Splats.Num())
	{
		return Result;
	}

	// Clamp count to available splats
	Count = FMath::Min(Count, Splats.Num() - StartIndex);

	// Calculate total opacity weight for weighted averaging
	float TotalWeight = 0.0f;
	for (int32 i = 0; i < Count; i++)
	{
		TotalWeight += Splats[StartIndex + i].Opacity;
	}

	if (TotalWeight < SMALL_NUMBER)
	{
		TotalWeight = 1.0f; // Avoid division by zero
	}

	// Weighted centroid position
	FVector3f WeightedPosition = FVector3f::ZeroVector;
	FVector3f WeightedSHDC = FVector3f::ZeroVector;

	for (int32 i = 0; i < Count; i++)
	{
		const FGaussianSplatData& Splat = Splats[StartIndex + i];
		float Weight = Splat.Opacity / TotalWeight;

		WeightedPosition += Splat.Position * Weight;

		// Average SH_DC directly (keeps same format as original splats)
		WeightedSHDC += Splat.SH_DC * Weight;
	}

	Result.Position = WeightedPosition;
	Result.SH_DC = WeightedSHDC;

	// Combined scale: weighted average of original scales
	// Don't add position spread — it dominates over individual scales in dense scenes
	// and produces massively oversized LOD splats. The combined opacity handles
	// the increased visual impact of merging N splats into 1.
	FVector3f AvgScale = FVector3f::ZeroVector;

	for (int32 i = 0; i < Count; i++)
	{
		const FGaussianSplatData& Splat = Splats[StartIndex + i];
		float Weight = Splat.Opacity / TotalWeight;
		AvgScale += Splat.Scale * Weight;
	}

	Result.Scale = AvgScale;

	// Use identity rotation for merged splat (could compute average quaternion)
	Result.Rotation = FQuat4f::Identity;

	// Combined opacity: probabilistic - 1 - product(1 - opacity_i)
	// For small opacities, this approximates the sum
	float CombinedTransparency = 1.0f;
	for (int32 i = 0; i < Count; i++)
	{
		CombinedTransparency *= (1.0f - FMath::Clamp(Splats[StartIndex + i].Opacity, 0.0f, 1.0f));
	}
	Result.Opacity = FMath::Clamp(1.0f - CombinedTransparency, 0.0f, 1.0f);

	// Zero out SH coefficients (LOD splats don't need view-dependent effects)
	for (int32 i = 0; i < 15; i++)
	{
		Result.SH[i] = FVector3f::ZeroVector;
	}

	return Result;
}

FGaussianSplatData FGaussianClusterBuilder::MergeLODSplats(
	const TArray<FGaussianSplatData>& LODSplats,
	int32 StartIndex,
	int32 Count)
{
	FGaussianSplatData Result;

	if (Count <= 0 || StartIndex < 0 || StartIndex >= LODSplats.Num())
	{
		return Result;
	}

	Count = FMath::Min(Count, LODSplats.Num() - StartIndex);

	// Calculate total opacity weight
	float TotalWeight = 0.0f;
	for (int32 i = 0; i < Count; i++)
	{
		TotalWeight += LODSplats[StartIndex + i].Opacity;
	}

	if (TotalWeight < SMALL_NUMBER)
	{
		TotalWeight = 1.0f;
	}

	// Weighted averages
	FVector3f WeightedPosition = FVector3f::ZeroVector;
	FVector3f WeightedSHDC = FVector3f::ZeroVector;
	FVector3f AvgScale = FVector3f::ZeroVector;

	for (int32 i = 0; i < Count; i++)
	{
		const FGaussianSplatData& Splat = LODSplats[StartIndex + i];
		float Weight = Splat.Opacity / TotalWeight;

		WeightedPosition += Splat.Position * Weight;
		WeightedSHDC += Splat.SH_DC * Weight;
		AvgScale += Splat.Scale * Weight;
	}

	Result.Position = WeightedPosition;
	Result.SH_DC = WeightedSHDC;

	Result.Scale = AvgScale;
	Result.Rotation = FQuat4f::Identity;

	// Combined opacity
	float CombinedTransparency = 1.0f;
	for (int32 i = 0; i < Count; i++)
	{
		CombinedTransparency *= (1.0f - FMath::Clamp(LODSplats[StartIndex + i].Opacity, 0.0f, 1.0f));
	}
	Result.Opacity = FMath::Clamp(1.0f - CombinedTransparency, 0.0f, 1.0f);

	// Zero out SH coefficients (LOD splats don't need view-dependent effects)
	for (int32 i = 0; i < 15; i++)
	{
		Result.SH[i] = FVector3f::ZeroVector;
	}

	return Result;
}

void FGaussianClusterBuilder::GenerateLODSplats(
	const TArray<FGaussianSplatData>& Splats,
	FGaussianClusterHierarchy& Hierarchy,
	int32 ReductionRatio)
{
	if (ReductionRatio < 1)
	{
		ReductionRatio = 4; // Default
	}

	Hierarchy.LODSplats.Empty();
	uint32 NextLODSplatIndex = 0;

	UE_LOG(LogTemp, Log, TEXT("GenerateLODSplats: Generating LOD splats with reduction ratio %d"), ReductionRatio);

	// Process clusters by LOD level (bottom-up)
	// Also compute MaxError based on actual merge displacement
	for (uint32 LODLevel = 1; LODLevel <= Hierarchy.NumLODLevels; LODLevel++)
	{
		for (FGaussianCluster& Cluster : Hierarchy.Clusters)
		{
			if (Cluster.LODLevel != LODLevel)
			{
				continue;
			}

			// This cluster needs LOD splats
			Cluster.LODSplatStartIndex = NextLODSplatIndex;

			// Track max displacement across all merge groups for error metric
			float ClusterMaxDisplacement = 0.0f;

			if (LODLevel == 1)
			{
				// First LOD level - collect original splats from all child leaf clusters
				// (children may not be contiguous due to spatial nearest-neighbor grouping)
				TArray<FGaussianSplatData> ChildSplats;

				for (uint32 ChildID : Cluster.ChildClusterIDs)
				{
					for (const FGaussianCluster& Child : Hierarchy.Clusters)
					{
						if (Child.ClusterID == ChildID)
						{
							for (uint32 j = 0; j < Child.SplatCount; j++)
							{
								uint32 SplatIdx = Child.SplatStartIndex + j;
								if (SplatIdx < (uint32)Splats.Num())
								{
									ChildSplats.Add(Splats[SplatIdx]);
								}
							}
							break;
						}
					}
				}

				if (ChildSplats.Num() > 0)
				{
					int32 NumLODSplats = FMath::DivideAndRoundUp(ChildSplats.Num(), ReductionRatio);
					NumLODSplats = FMath::Max(1, NumLODSplats);

					for (int32 i = 0; i < NumLODSplats; i++)
					{
						int32 SrcStart = i * ReductionRatio;
						int32 SrcCount = FMath::Min(ReductionRatio, ChildSplats.Num() - SrcStart);

						if (SrcCount > 0)
						{
							FGaussianSplatData MergedSplat = MergeGaussians(ChildSplats, SrcStart, SrcCount);
							Hierarchy.LODSplats.Add(MergedSplat);
							NextLODSplatIndex++;

							// Compute max displacement: how far each source splat is from the merged centroid
							for (int32 j = 0; j < SrcCount; j++)
							{
								float Displacement = FVector3f::Dist(ChildSplats[SrcStart + j].Position, MergedSplat.Position);
								ClusterMaxDisplacement = FMath::Max(ClusterMaxDisplacement, Displacement);
							}
						}
					}

					Cluster.LODSplatCount = NextLODSplatIndex - Cluster.LODSplatStartIndex;
				}

				// MaxError = actual merge displacement, floored at half the bounding sphere radius
				// The floor prevents dense clusters (tiny displacement) from selecting LOD too early
				Cluster.MaxError = FMath::Max(ClusterMaxDisplacement, Cluster.BoundingSphereRadius * 0.5f);
			}
			else
			{
				// Higher LOD levels - merge from children's LOD splats (unified format)
				TArray<FGaussianSplatData> ChildLODSplats;

				// Also find max child error for compound error propagation
				float MaxChildError = 0.0f;
				for (uint32 ChildID : Cluster.ChildClusterIDs)
				{
					for (const FGaussianCluster& Child : Hierarchy.Clusters)
					{
						if (Child.ClusterID == ChildID)
						{
							MaxChildError = FMath::Max(MaxChildError, Child.MaxError);
							if (Child.LODSplatCount > 0)
							{
								for (uint32 j = 0; j < Child.LODSplatCount; j++)
								{
									uint32 LODIndex = Child.LODSplatStartIndex + j;
									if (LODIndex < (uint32)Hierarchy.LODSplats.Num())
									{
										ChildLODSplats.Add(Hierarchy.LODSplats[LODIndex]);
									}
								}
							}
							break;
						}
					}
				}

				if (ChildLODSplats.Num() > 0)
				{
					int32 NumLODSplats = FMath::DivideAndRoundUp(ChildLODSplats.Num(), ReductionRatio);
					NumLODSplats = FMath::Max(1, NumLODSplats);

					for (int32 i = 0; i < NumLODSplats; i++)
					{
						int32 SrcStart = i * ReductionRatio;
						int32 SrcCount = FMath::Min(ReductionRatio, ChildLODSplats.Num() - SrcStart);

						if (SrcCount > 0)
						{
							FGaussianSplatData MergedSplat = MergeLODSplats(ChildLODSplats, SrcStart, SrcCount);
							Hierarchy.LODSplats.Add(MergedSplat);
							NextLODSplatIndex++;

							// Compute max displacement from source LOD splats to merged centroid
							for (int32 j = 0; j < SrcCount; j++)
							{
								float Displacement = FVector3f::Dist(ChildLODSplats[SrcStart + j].Position, MergedSplat.Position);
								ClusterMaxDisplacement = FMath::Max(ClusterMaxDisplacement, Displacement);
							}
						}
					}

					Cluster.LODSplatCount = NextLODSplatIndex - Cluster.LODSplatStartIndex;
				}

				// Compound error: this level's merge displacement + worst child error
				// Floored at half the bounding sphere radius (same as level 1)
				Cluster.MaxError = FMath::Max(ClusterMaxDisplacement + MaxChildError, Cluster.BoundingSphereRadius * 0.5f);
			}
		}
	}

	Hierarchy.TotalLODSplatCount = Hierarchy.LODSplats.Num();

	UE_LOG(LogTemp, Log, TEXT("GenerateLODSplats: Generated %d LOD splats"), Hierarchy.TotalLODSplatCount);
}
