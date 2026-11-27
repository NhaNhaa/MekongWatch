"""
Layer 2: Phase 1 — Optical Baseline & Cloud Limitation (MekongWatch — Kratie Province)

Strictly demonstrates Lesson 1 concepts (1–10):
  - Optical Satellite (Passive Sensor — Sentinel-2 L2A)
  - Spectral Bands (RGB + Near-Infrared NIR)
  - Raster & GeoTIFF (with exact Top-Left / Bottom-Right GPS coordinates)
  - Spatial Resolution (20m x 20m = 400 m²/pixel) & Meter-based CRS (EPSG:32648)
  - Cloud Cover vs. Cloud Masking (SCL band filtering)
  - NDWI = (Green - NIR) / (Green + NIR) -> Normal Dry-Season Mekong River (km²)
  - NDVI = (NIR - Red) / (NIR + Red)     -> Healthy Green Farmland (km²)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pyproj import Transformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Sentinel-2 Scene Classification Layer (SCL) cloud & shadow classes
CLOUD_SCL_CLASSES = (3, 8, 9, 10)   # 3=Cloud Shadow, 8=Cloud Med, 9=Cloud High, 10=Thin Cirrus
INVALID_SCL_CLASSES = (0, 1)        # 0=No Data, 1=Saturated/Defective

NDWI_WATER_THRESHOLD = 0.05
NDVI_FARMLAND_THRESHOLD = 0.35


def scale_reflectance(
    dn: np.ndarray,
    offset: float = -1000.0,
    scale: float = 0.0001,
) -> np.ndarray:
    """Convert raw Sentinel-2 L2A uint16 digital numbers to surface reflectance [0, 1]."""
    arr = dn.astype(np.float32)
    valid = dn > 0
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    out[valid] = np.clip((arr[valid] + offset) * scale, 0.0, 1.0)
    return out


def build_cloud_mask(scl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build cloud mask and clear-pixel mask from Sentinel-2 SCL band.
    Returns (is_cloud, is_clear) boolean arrays.
    """
    is_cloud = np.isin(scl, CLOUD_SCL_CLASSES)
    is_invalid = np.isin(scl, INVALID_SCL_CLASSES)
    is_clear = (~is_cloud) & (~is_invalid)
    return is_cloud, is_clear


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR), safely clipped to [-1, 1]."""
    green_f = np.asarray(green, dtype=np.float32)
    nir_f = np.asarray(nir, dtype=np.float32)
    denom = green_f + nir_f
    valid = np.isfinite(green_f) & np.isfinite(nir_f) & (np.abs(denom) > 1e-6)
    ndwi = np.full(green_f.shape, np.nan, dtype=np.float32)
    ndwi[valid] = np.clip((green_f[valid] - nir_f[valid]) / denom[valid], -1.0, 1.0)
    return ndwi


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red), safely clipped to [-1, 1]."""
    nir_f = np.asarray(nir, dtype=np.float32)
    red_f = np.asarray(red, dtype=np.float32)
    denom = nir_f + red_f
    valid = np.isfinite(nir_f) & np.isfinite(red_f) & (np.abs(denom) > 1e-6)
    ndvi = np.full(nir_f.shape, np.nan, dtype=np.float32)
    ndvi[valid] = np.clip((nir_f[valid] - red_f[valid]) / denom[valid], -1.0, 1.0)
    return ndvi


def pixels_to_km2(pixel_count: int, pixel_size_m: float = 20.0) -> float:
    """Convert pixel count on a meter-based CRS grid to square kilometers (km²)."""
    return float(pixel_count) * (pixel_size_m ** 2) / 1_000_000.0


def make_rgb_display(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    gain: float = 3.2,
    gamma: float = 0.85,
) -> np.ndarray:
    """Create a bright, natural-looking RGB image array in [0, 1] for matplotlib."""
    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = np.clip(rgb * gain, 0.0, 1.0)
    return np.power(rgb, gamma)


def load_s2_geotiff(
    tif_path: Path,
    offset: float,
    scale: float,
) -> dict[str, Any]:
    """Load a 5-band aligned Sentinel-2 GeoTIFF (B02, B03, B04, B08, SCL)."""
    with rasterio.open(tif_path) as src:
        blue = scale_reflectance(src.read(1), offset=offset, scale=scale)
        green = scale_reflectance(src.read(2), offset=offset, scale=scale)
        red = scale_reflectance(src.read(3), offset=offset, scale=scale)
        nir = scale_reflectance(src.read(4), offset=offset, scale=scale)
        scl = src.read(5)
        bounds = src.bounds
        crs_str = str(src.crs)
        pixel_size_m = float(abs(src.transform.a))
        width, height = src.width, src.height

    transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
    tl_lon, tl_lat = transformer.transform(bounds.left, bounds.top)
    br_lon, br_lat = transformer.transform(bounds.right, bounds.bottom)

    return {
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
        "scl": scl,
        "crs": crs_str,
        "pixel_size_m": pixel_size_m,
        "width": width,
        "height": height,
        "bounds_utm": {
            "left": round(bounds.left, 1),
            "bottom": round(bounds.bottom, 1),
            "right": round(bounds.right, 1),
            "top": round(bounds.top, 1),
        },
        "corners_wgs84": {
            "top_left": {"lat": round(tl_lat, 5), "lon": round(tl_lon, 5)},
            "bottom_right": {"lat": round(br_lat, 5), "lon": round(br_lon, 5)},
        },
    }


def main() -> None:
    print("=== Layer 2: Phase 1 — Optical Baseline & Cloud Masking (Sentinel-2) ===")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))

    dry_meta = meta["sentinel2_dry"]
    wet_meta = meta["sentinel2_wet"]

    s2_dry = load_s2_geotiff(
        DATA_DIR / "s2_dry.tif",
        offset=dry_meta.get("reflectance_offset", -1000.0),
        scale=dry_meta.get("reflectance_scale", 0.0001),
    )
    s2_wet = load_s2_geotiff(
        DATA_DIR / "s2_wet.tif",
        offset=wet_meta.get("reflectance_offset", -1000.0),
        scale=wet_meta.get("reflectance_scale", 0.0001),
    )

    px_m = s2_dry["pixel_size_m"]
    total_pixels = s2_dry["width"] * s2_dry["height"]
    total_area_km2 = pixels_to_km2(total_pixels, px_m)

    # 1. Cloud Masking on Dry vs. Rainy Season
    dry_cloud, dry_clear = build_cloud_mask(s2_dry["scl"])
    wet_cloud, wet_clear = build_cloud_mask(s2_wet["scl"])

    dry_cloud_pct = round(float(dry_cloud.mean()) * 100.0, 2)
    dry_clear_pct = round(float(dry_clear.mean()) * 100.0, 2)
    wet_cloud_pct = round(float(wet_cloud.mean()) * 100.0, 2)
    wet_clear_pct = round(float(wet_clear.mean()) * 100.0, 2)

    # 2. Spectral Indices on Cloud-Masked Dry Season Baseline
    dry_ndwi = compute_ndwi(s2_dry["green"], s2_dry["nir"])
    dry_ndvi = compute_ndvi(s2_dry["nir"], s2_dry["red"])

    # Apply cloud masking (set cloudy pixels to NaN so they don't corrupt calculations)
    dry_ndwi_masked = np.where(dry_clear, dry_ndwi, np.nan)
    dry_ndvi_masked = np.where(dry_clear, dry_ndvi, np.nan)

    dry_water_mask = np.isfinite(dry_ndwi_masked) & (dry_ndwi_masked > NDWI_WATER_THRESHOLD)
    dry_farmland_mask = (
        np.isfinite(dry_ndvi_masked)
        & (dry_ndvi_masked > NDVI_FARMLAND_THRESHOLD)
        & (~dry_water_mask)
    )

    dry_river_km2 = round(pixels_to_km2(int(dry_water_mask.sum()), px_m), 2)
    dry_farmland_km2 = round(pixels_to_km2(int(dry_farmland_mask.sum()), px_m), 2)
    median_river_ndwi = round(float(np.nanmedian(dry_ndwi_masked[dry_water_mask])), 3) if dry_water_mask.any() else 0.0
    median_farm_ndvi = round(float(np.nanmedian(dry_ndvi_masked[dry_farmland_mask])), 3) if dry_farmland_mask.any() else 0.0

    # Save masks for Phase 2 cross-layer integration (flooded farmland calculation)
    np.save(OUTPUTS_DIR / "dry_farmland_mask.npy", dry_farmland_mask)
    np.save(OUTPUTS_DIR / "dry_water_ndwi_mask.npy", dry_water_mask)

    # Save Phase 1 summary JSON
    summary = {
        "province": meta["province"],
        "sensor": "Sentinel-2 L2A (Optical / Passive Sensor)",
        "crs": s2_dry["crs"],
        "pixel_size_m": px_m,
        "pixel_area_m2": px_m ** 2,
        "raster_shape": [s2_dry["height"], s2_dry["width"]],
        "corners_wgs84": s2_dry["corners_wgs84"],
        "total_aoi_area_km2": round(total_area_km2, 2),
        "dry_season": {
            "date": dry_meta["date"],
            "scene_id": dry_meta["id"],
            "aoi_cloud_cover_pct": dry_cloud_pct,
            "aoi_clear_pct": dry_clear_pct,
            "ndwi_threshold": NDWI_WATER_THRESHOLD,
            "ndvi_threshold": NDVI_FARMLAND_THRESHOLD,
            "normal_river_area_km2": dry_river_km2,
            "healthy_farmland_area_km2": dry_farmland_km2,
            "median_river_ndwi": median_river_ndwi,
            "median_farmland_ndvi": median_farm_ndvi,
        },
        "rainy_season": {
            "date": wet_meta["date"],
            "scene_id": wet_meta["id"],
            "aoi_cloud_cover_pct": wet_cloud_pct,
            "aoi_clear_pct": wet_clear_pct,
            "optical_blocked_note": (
                f"{wet_cloud_pct:.1f}% of the Kratie AOI is blocked by monsoon clouds/shadows "
                f"in Sentinel-2 on {wet_meta['date']}, motivating Phase 2 SAR radar."
            ),
        },
    }
    (OUTPUTS_DIR / "phase1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 3. Generate crisp 4-panel Phase 1 PNG figure
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5), dpi=160)
    fig.patch.set_facecolor("#F8FAFC")

    dry_rgb = make_rgb_display(s2_dry["red"], s2_dry["green"], s2_dry["blue"])
    wet_rgb = make_rgb_display(s2_wet["red"], s2_wet["green"], s2_wet["blue"])

    # Panel 1: Dry Season True Color RGB
    ax = axes[0, 0]
    ax.imshow(dry_rgb)
    ax.set_title(
        f"1. Dry Season Optical RGB ({dry_meta['date']})\n"
        f"Cloud Cover: {dry_cloud_pct:.1f}% — Clear Baseline View",
        fontsize=11,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 2: Rainy Season RGB + Red Cloud Mask
    ax = axes[0, 1]
    ax.imshow(wet_rgb)
    cloud_rgba = np.zeros((*wet_cloud.shape, 4), dtype=np.float32)
    cloud_rgba[wet_cloud] = [0.95, 0.18, 0.18, 0.42]
    ax.imshow(cloud_rgba)
    ax.set_title(
        f"2. Rainy Season Optical RGB + Cloud Mask ({wet_meta['date']})\n"
        f"Cloud Blocked: {wet_cloud_pct:.1f}% — Why Optical Fails in Monsoon",
        fontsize=11,
        fontweight="bold",
        color="#991B1B",
        pad=8,
    )
    ax.legend(
        handles=[Patch(facecolor=(0.95, 0.18, 0.18, 0.55), edgecolor="#991B1B", label=f"SCL Cloud/Shadow Masked ({wet_cloud_pct:.1f}%)")],
        loc="lower right",
        fontsize=8.5,
        framealpha=0.92,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 3: Dry Season NDWI (Surface Water)
    ax = axes[1, 0]
    im_ndwi = ax.imshow(dry_ndwi_masked, cmap="RdYlBu", vmin=-0.5, vmax=0.5)
    # Outline water mask subtly
    water_overlay = np.zeros((*dry_water_mask.shape, 4), dtype=np.float32)
    water_overlay[dry_water_mask] = [0.05, 0.35, 0.85, 0.35]
    ax.imshow(water_overlay)
    ax.set_title(
        f"3. Dry Season NDWI — Normal Mekong River\n"
        f"Water (NDWI > {NDWI_WATER_THRESHOLD}): {dry_river_km2:.2f} km² (median={median_river_ndwi:+.2f})",
        fontsize=11,
        fontweight="bold",
        color="#0C4A6E",
        pad=8,
    )
    cbar1 = fig.colorbar(im_ndwi, ax=ax, fraction=0.042, pad=0.02)
    cbar1.set_label("NDWI  (Green - NIR) / (Green + NIR)", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 4: Dry Season NDVI (Healthy Green Farmland / Vegetation)
    ax = axes[1, 1]
    im_ndvi = ax.imshow(dry_ndvi_masked, cmap="YlGn", vmin=0.0, vmax=0.85)
    ax.set_title(
        f"4. Dry Season NDVI — Healthy Green Vegetation & Farmland\n"
        f"Green Cover (NDVI > {NDVI_FARMLAND_THRESHOLD}): {dry_farmland_km2:.2f} km² (median={median_farm_ndvi:+.2f})",
        fontsize=11,
        fontweight="bold",
        color="#14532D",
        pad=8,
    )
    cbar2 = fig.colorbar(im_ndvi, ax=ax, fraction=0.042, pad=0.02)
    cbar2.set_label("NDVI  (NIR - Red) / (NIR + Red)", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])

    tl = s2_dry["corners_wgs84"]["top_left"]
    br = s2_dry["corners_wgs84"]["bottom_right"]
    fig.suptitle(
        f"MekongWatch Phase 1 — Sentinel-2 Optical Baseline (Kratie Province, Cambodia)\n"
        f"CRS: {s2_dry['crs']} · 20m Resolution (400 m²/px) · "
        f"Top-Left ({tl['lat']:.3f}°N, {tl['lon']:.3f}°E) → Bottom-Right ({br['lat']:.3f}°N, {br['lon']:.3f}°E) · AOI {total_area_km2:.1f} km²",
        fontsize=12.5,
        fontweight="bold",
        color="#0F172A",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.945])
    out_png = OUTPUTS_DIR / "phase1_optical_baseline.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"  Dry Date ({dry_meta['date']}): Cloud={dry_cloud_pct:.1f}% | Normal River={dry_river_km2:.2f} km² | Healthy Green/Farmland={dry_farmland_km2:.2f} km²")
    print(f"  Rainy Date ({wet_meta['date']}): Cloud={wet_cloud_pct:.1f}% (Optical blocked!)")
    print(f"  Saved {out_png.name} and phase1_summary.json")


if __name__ == "__main__":
    main()
