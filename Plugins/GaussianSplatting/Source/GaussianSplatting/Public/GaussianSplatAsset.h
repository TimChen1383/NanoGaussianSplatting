// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Serialization/BulkData.h"
#include "GaussianDataTypes.h"
#include "GaussianClusterTypes.h"
#include "GaussianSplatAsset.generated.h"

// Forward declaration
class UGaussianSplatAsset;

// Delegate fired when asset data changes (e.g., Nanite enabled/disabled)
DECLARE_MULTICAST_DELEGATE_OneParam(FOnGaussianSplatAssetChanged, UGaussianSplatAsset*);

// Serialization magic/version for format identification
#define GAUSSIAN_SPLAT_ASSET_MAGIC   0x47535056  // "GSPV"
#define GAUSSIAN_SPLAT_ASSET_VERSION 4  // Added bEnableNanite, OriginalSplatCount

/**
 * Asset containing Gaussian Splatting data loaded from PLY files
 * Stores compressed splat data optimized for GPU rendering
 */
UCLASS(BlueprintType, hidecategories = Object)
class GAUSSIANSPLATTING_API UGaussianSplatAsset : public UObject
{
	GENERATED_BODY()

public:
	UGaussianSplatAsset();

	//~ Begin UObject Interface
	virtual void Serialize(FArchive& Ar) override;
	virtual void PostLoad() override;
#if WITH_EDITOR
	virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif
	//~ End UObject Interface

	/** Get the number of splats in this asset */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting")
	int32 GetSplatCount() const { return SplatCount; }

	/** Get the bounding box of all splats */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting")
	FBox GetBounds() const { return BoundingBox; }

	/** Get estimated memory usage in bytes */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting")
	int64 GetMemoryUsage() const;

	/** Check if asset has valid data */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting")
	bool IsValid() const { return SplatCount > 0 && PositionBulkData.GetBulkDataSize() > 0; }

	/** Check if cluster hierarchy is available */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting|Clustering")
	bool HasClusterHierarchy() const { return ClusterHierarchy.IsValid(); }

	/** Check if Nanite is enabled for this asset */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting|Nanite")
	bool IsNaniteEnabled() const { return bEnableNanite; }

#if WITH_EDITOR
	/**
	 * Generate a thumbnail image from raw splat data and store it in ThumbnailTexture.
	 * Called automatically at the end of InitializeFromSplatData().
	 */
	void GenerateThumbnail(const TArray<struct FGaussianSplatData>& InSplats);

	/**
	 * Build Nanite cluster hierarchy from source PLY file
	 * This re-reads the source file and builds cluster data
	 * @return True if successful, false if source file not found or build failed
	 */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting|Nanite")
	bool BuildNaniteClusterHierarchy();

	/**
	 * Clear Nanite cluster hierarchy to reduce asset size
	 * Removes cluster data and LOD splats
	 */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting|Nanite")
	void ClearNaniteClusterHierarchy();

	/**
	 * Set Nanite enabled state (internal use)
	 */
	void SetNaniteEnabled(bool bEnable) { bEnableNanite = bEnable; }
#endif

	/** Get number of clusters in hierarchy */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting|Clustering")
	int32 GetClusterCount() const { return ClusterHierarchy.Clusters.Num(); }

	/** Get number of LOD levels */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting|Clustering")
	int32 GetNumLODLevels() const { return ClusterHierarchy.NumLODLevels; }

	/** Get original splat count (excluding LOD splats) */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting|Nanite")
	int32 GetOriginalSplatCount() const { return OriginalSplatCount > 0 ? OriginalSplatCount : SplatCount; }

	/** Get the cluster hierarchy (const reference) */
	const FGaussianClusterHierarchy& GetClusterHierarchy() const { return ClusterHierarchy; }

	/** Delegate fired when asset data changes (Nanite enabled/disabled, reimport, etc.) */
	FOnGaussianSplatAssetChanged OnAssetChanged;

public:
	/** Total number of splats */
	UPROPERTY(VisibleAnywhere, Category = "Info")
	int32 SplatCount = 0;

	/** World-space bounding box of all splats */
	UPROPERTY(VisibleAnywhere, Category = "Info")
	FBox BoundingBox;

	/** Position compression format */
	UPROPERTY(VisibleAnywhere, Category = "Format")
	EGaussianPositionFormat PositionFormat = EGaussianPositionFormat::Float32;

	/** Color compression format */
	UPROPERTY(VisibleAnywhere, Category = "Format")
	EGaussianColorFormat ColorFormat = EGaussianColorFormat::Float16x4;

	/** Spherical harmonics compression format */
	UPROPERTY(VisibleAnywhere, Category = "Format")
	EGaussianSHFormat SHFormat = EGaussianSHFormat::Float16;

	/** Number of SH bands stored (0-3) */
	UPROPERTY(VisibleAnywhere, Category = "Format")
	int32 SHBands = 3;

	/** Compressed position data (stored as bulk data for fast loading) */
	FByteBulkData PositionBulkData;

	/** Compressed rotation + scale data (stored as bulk data for fast loading) */
	FByteBulkData OtherBulkData;

	/** Compressed spherical harmonics data (stored as bulk data for fast loading) */
	FByteBulkData SHBulkData;

	/** Chunk quantization info (one per 256 splats) - kept as TArray since it's small */
	UPROPERTY()
	TArray<FGaussianChunkInfo> ChunkData;

	/** Color texture (Morton-swizzled, 2048 x N) - created at runtime from ColorTextureBulkData */
	UPROPERTY(Transient)
	TObjectPtr<UTexture2D> ColorTexture;

	/** Raw color texture pixel data (stored as bulk data for fast loading) */
	FByteBulkData ColorTextureBulkData;

	/** Color texture width */
	UPROPERTY()
	int32 ColorTextureWidth = 0;

	/** Color texture height */
	UPROPERTY()
	int32 ColorTextureHeight = 0;

	/** Source file path (for reimport) */
	UPROPERTY(VisibleAnywhere, Category = "Import")
	FString SourceFilePath;

	/** Quality level used during import */
	UPROPERTY(VisibleAnywhere, Category = "Import")
	EGaussianQualityLevel ImportQuality = EGaussianQualityLevel::Medium;

	/**
	 * Whether Nanite-style LOD and culling is enabled for this asset
	 * Enable via Asset Actions > Nanite in Content Browser
	 */
	UPROPERTY(VisibleAnywhere, Category = "Nanite")
	bool bEnableNanite = false;

	/**
	 * Hierarchical cluster structure for Nanite-style LOD and culling
	 * Only populated when bEnableNanite is true
	 */
	UPROPERTY()
	FGaussianClusterHierarchy ClusterHierarchy;

	/**
	 * Number of original splats (before LOD splats are appended)
	 * When Nanite is enabled, SplatCount includes both original + LOD splats
	 */
	UPROPERTY()
	int32 OriginalSplatCount = 0;

#if WITH_EDITORONLY_DATA
	/**
	 * Thumbnail image generated at import time.
	 * Stored persistently in the asset package so the Content Browser can display it.
	 * NOTE: This is Transient - recreated from ThumbnailData in PostLoad()
	 */
	UPROPERTY(Transient)
	TObjectPtr<UTexture2D> ThumbnailTexture;

	/**
	 * Raw thumbnail pixel data (BGRA, 256x256).
	 * Stored persistently and used to recreate ThumbnailTexture on load.
	 */
	UPROPERTY()
	TArray<uint8> ThumbnailData;

	/** Size of the thumbnail (width = height) */
	UPROPERTY()
	int32 ThumbnailSize = 0;

private:
	/** Recreate ThumbnailTexture from stored ThumbnailData (called in PostLoad) */
	void CreateThumbnailTextureFromData();
#endif

public:
	/**
	 * Initialize asset from raw splat data
	 * @param InSplats Raw splat data from PLY file
	 * @param InQuality Compression quality level
	 */
	void InitializeFromSplatData(const TArray<FGaussianSplatData>& InSplats, EGaussianQualityLevel InQuality);

	/**
	 * Decompress and return all splat positions (for debugging)
	 * @return Array of world-space positions
	 */
	UFUNCTION(BlueprintCallable, Category = "Gaussian Splatting|Debug")
	TArray<FVector> GetDecompressedPositions() const;

	/** Get bytes per splat for position data based on format */
	static int32 GetPositionBytesPerSplat(EGaussianPositionFormat Format);

	/** Get bytes per splat for color data based on format */
	static int32 GetColorBytesPerSplat(EGaussianColorFormat Format);

	/** Get bytes per splat for SH data based on format */
	static int32 GetSHBytesPerSplat(EGaussianSHFormat Format, int32 Bands);

	/**
	 * Copy position bulk data to a TArray (for GPU upload)
	 * @param OutData Array to copy data into
	 */
	void GetPositionData(TArray<uint8>& OutData) const;

	/**
	 * Copy rotation/scale bulk data to a TArray (for GPU upload)
	 * @param OutData Array to copy data into
	 */
	void GetOtherData(TArray<uint8>& OutData) const;

	/**
	 * Copy SH bulk data to a TArray (for GPU upload)
	 * @param OutData Array to copy data into
	 */
	void GetSHData(TArray<uint8>& OutData) const;

	/**
	 * Copy color texture bulk data to a TArray
	 * @param OutData Array to copy data into
	 */
	void GetColorTextureData(TArray<uint8>& OutData) const;

	/** Get size of position bulk data in bytes */
	int64 GetPositionDataSize() const { return PositionBulkData.GetBulkDataSize(); }

	/** Get size of other bulk data in bytes */
	int64 GetOtherDataSize() const { return OtherBulkData.GetBulkDataSize(); }

	/** Get size of SH bulk data in bytes */
	int64 GetSHDataSize() const { return SHBulkData.GetBulkDataSize(); }

	/** Get size of color texture bulk data in bytes */
	int64 GetColorTextureDataSize() const { return ColorTextureBulkData.GetBulkDataSize(); }

private:
	/** Compress and store position data */
	void CompressPositions(const TArray<FGaussianSplatData>& InSplats);

	/** Compress and store rotation/scale data */
	void CompressRotationScale(const TArray<FGaussianSplatData>& InSplats);

	/** Create color texture data with Morton swizzling (stores raw data for serialization) */
	void CreateColorTextureData(const TArray<FGaussianSplatData>& InSplats);

	/** Create UTexture2D from stored ColorTextureData (called after load or import) */
	void CreateColorTextureFromData();

	/** Compress and store SH data */
	void CompressSH(const TArray<FGaussianSplatData>& InSplats);

	/** Calculate bounding box from splat data */
	void CalculateBounds(const TArray<FGaussianSplatData>& InSplats);

	/** Calculate chunk quantization bounds */
	void CalculateChunkBounds(const TArray<FGaussianSplatData>& InSplats);
};
