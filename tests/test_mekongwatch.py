"""
Automated pytest verification suite for MekongWatch (Kratie Province).
Tests both pure mathematical/geospatial logic and real generated outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

MEKONGG_DIR = Path(__file__).resolve().parent.parent
if str(MEKONGG_DIR) not in sys.path:
    sys.path.insert(0, str(MEKONGG_DIR))

from phase1_optical import (
    build_cloud_mask,
    compute_ndvi,
    compute_ndwi,
    pixels_to_km2,
    scale_reflectance,
)
from phase2_sar_flood import (
    denoise_sar_db,
    detect_sar_flood,
    linear_to_db,
    train_geoai_random_forest,
    vectorize_flood_mask,
)

DATA_DIR = MEKONGG_DIR / "data"
OUTPUTS_DIR = MEKONGG_DIR / "outputs"


# ---------------------------------------------------------------------------
# Unit Tests: Phase 1 (Optical Spectral Indices & Cloud Masking)
# ---------------------------------------------------------------------------

def test_ndvi_bounds_and_zero_division():
    nir = np.array([0.8, 0.2, 0.0, np.nan], dtype=np.float32)
    red = np.array([0.1, 0.2, 0.0, 0.3], dtype=np.float32)
    ndvi = compute_ndvi(nir, red)

    assert np.isclose(ndvi[0], 0.7777778, atol=1e-4)  # Healthy vegetation
    assert np.isclose(ndvi[1], 0.0, atol=1e-5)        # Concrete / bare
    assert np.isnan(ndvi[2])                          # Zero-division safe
    assert np.isnan(ndvi[3])                          # NaN propagation
    valid = ndvi[np.isfinite(ndvi)]
    assert np.all((valid >= -1.0) & (valid <= 1.0))


def test_ndwi_bounds_and_water_vs_forest():
    green = np.array([0.15, 0.10, 0.0], dtype=np.float32)
    nir = np.array([0.03, 0.60, 0.0], dtype=np.float32)
    ndwi = compute_ndwi(green, nir)

    assert ndwi[0] > 0.5   # Water
    assert ndwi[1] < -0.5  # Forest / vegetation
    assert np.isnan(ndwi[2])
    valid = ndwi[np.isfinite(ndwi)]
    assert np.all((valid >= -1.0) & (valid <= 1.0))


def test_scl_cloud_mask_excludes_clouds_and_shadows():
    scl = np.array([0, 1, 3, 4, 5, 6, 8, 9, 10], dtype=np.uint16)
    is_cloud, is_clear = build_cloud_mask(scl)

    # Cloud/shadow classes: 3, 8, 9, 10
    assert is_cloud.tolist() == [False, False, True, False, False, False, True, True, True]
    # Clear valid classes: 4 (veg), 5 (not veg), 6 (water)
    assert is_clear.tolist() == [False, False, False, True, True, True, False, False, False]


def test_scale_reflectance_and_area_math():
    dn = np.array([0, 1000, 3500, 15000], dtype=np.uint16)
    refl = scale_reflectance(dn, offset=-1000.0, scale=0.0001)
    assert np.isnan(refl[0])
    assert np.isclose(refl[1], 0.0)
    assert np.isclose(refl[2], 0.25)
    assert np.isclose(refl[3], 1.0)  # Clipped to 1.0
    assert np.isclose(pixels_to_km2(2500, pixel_size_m=20.0), 1.0)  # 2500 * 400 m² = 1 km²


# ---------------------------------------------------------------------------
# Unit Tests: Phase 2 (SAR dB, Change Detection, Vectorization, GeoAI)
# ---------------------------------------------------------------------------

def test_sar_db_and_change_detection_logic():
    # 10x10 synthetic patch: left = dry land, middle = permanent river, right = newly flooded
    dry_lin = np.full((10, 10), 0.15, dtype=np.float32)   # ~ -8.2 dB (land)
    wet_lin = np.full((10, 10), 0.14, dtype=np.float32)   # ~ -8.5 dB (land)

    # Permanent river block (rows 0..9, cols 3..5): dark in both seasons (~ -20 dB)
    dry_lin[:, 3:6] = 0.01
    wet_lin[:, 3:6] = 0.01

    # Newly flooded block (rows 1..8, cols 6..9): bright in dry, dark in rainy (~ -20 dB)
    wet_lin[1:9, 6:10] = 0.01

    dry_db = denoise_sar_db(linear_to_db(dry_lin), size=3)
    wet_db = denoise_sar_db(linear_to_db(wet_lin), size=3)

    assert np.isclose(linear_to_db(np.array([0.01]))[0], -20.0, atol=1e-4)

    change_map, _ = detect_sar_flood(dry_db, wet_db, water_thresh_db=-15.5, drop_thresh_db=-2.5, min_cluster_pixels=5)
    assert change_map[5, 1] == 0  # Dry land
    assert change_map[5, 4] == 1  # Permanent river
    assert change_map[5, 8] == 2  # Newly flooded land

    # Vectorize newly flooded pixels into GeoJSON
    transform = from_origin(600000.0, 1380000.0, 20.0, 20.0)
    fc = vectorize_flood_mask(change_map == 2, transform=transform, src_crs="EPSG:32648", min_pixels=5)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 1
    assert fc["features"][0]["properties"]["area_km2"] > 0


def test_geoai_random_forest_synthetic():
    rng = np.random.default_rng(7)
    h, w = 30, 30
    feats = rng.normal(loc=0.2, scale=0.05, size=(h, w, 8)).astype(np.float32)
    labels = np.zeros((h, w), dtype=np.uint8)

    # Permanent river signature
    labels[0:10, :] = 1
    feats[0:10, :, 4] = -20.0  # SAR Dry VV
    feats[0:10, :, 6] = -20.0  # SAR Wet VV

    # Newly flooded signature
    labels[10:20, :] = 2
    feats[10:20, :, 4] = -9.0   # SAR Dry VV
    feats[10:20, :, 6] = -20.0  # SAR Wet VV

    res = train_geoai_random_forest(
        feats,
        ["B", "G", "R", "NIR", "DryVV", "DryVH", "WetVV", "WetVH"],
        labels,
        max_samples_per_class=200,
        random_state=42,
    )
    assert res["overall_accuracy"] >= 0.90
    assert res["macro_f1"] >= 0.90
    assert res["pred_map"].shape == (h, w)
    assert len(res["confusion_matrix"]) == 3


# ---------------------------------------------------------------------------
# Integration Tests: Real 2026 Kratie GeoTIFFs & Pipeline Outputs
# ---------------------------------------------------------------------------

def test_real_aligned_geotiffs_and_outputs_exist():
    expected_tifs = ["s2_dry.tif", "s2_wet.tif", "s1_dry.tif", "s1_wet.tif"]
    shapes_seen = set()
    crs_seen = set()
    for fname in expected_tifs:
        path = DATA_DIR / fname
        assert path.exists(), f"Missing GeoTIFF: {path}"
        with rasterio.open(path) as src:
            shapes_seen.add((src.height, src.width))
            crs_seen.add(str(src.crs))

    assert len(shapes_seen) == 1, f"GeoTIFF dimensions do not match: {shapes_seen}"
    assert crs_seen == {"EPSG:32648"}, f"Expected EPSG:32648, got {crs_seen}"

    p1 = json.loads((OUTPUTS_DIR / "phase1_summary.json").read_text(encoding="utf-8"))
    p2 = json.loads((OUTPUTS_DIR / "phase2_summary.json").read_text(encoding="utf-8"))

    assert p1["dry_season"]["normal_river_area_km2"] > 0
    assert p1["dry_season"]["healthy_farmland_area_km2"] > 0
    assert p1["rainy_season"]["aoi_cloud_cover_pct"] > p1["dry_season"]["aoi_cloud_cover_pct"]

    assert p2["areas_km2"]["permanent_river_km2"] > 0
    assert p2["areas_km2"]["newly_flooded_land_km2"] > 0
    assert 0.5 <= p2["geoai"]["overall_accuracy"] <= 1.0

    # Validate generated flood_zones.geojson
    fc = json.loads((OUTPUTS_DIR / "flood_zones.geojson").read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) > 0
    for feat in fc["features"][:20]:
        assert feat["type"] == "Feature"
        assert feat["properties"]["area_km2"] > 0
        assert feat["geometry"]["type"] in ("Polygon", "MultiPolygon")

    # Validate all 3 PNG figures exist and are non-empty
    for png_name in (
        "phase1_optical_baseline.png",
        "phase2_sar_flood_map.png",
        "phase2_geoai_landcover.png",
    ):
        png_path = OUTPUTS_DIR / png_name
        assert png_path.exists() and png_path.stat().st_size > 20_000
