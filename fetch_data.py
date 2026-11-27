"""
Layer 1: Data Acquisition & Spatial Alignment (MekongWatch — Kratie Province)

Fetches real 2026 satellite imagery from a single free source:
  Microsoft Planetary Computer STAC API (no API key required)
    - Sentinel-2 L2A (Optical: Dry Season baseline + Rainy Season cloudy scene)
    - Sentinel-1 RTC (SAR Radar VV + VH: Dry Season + Peak Rainy Season flood)

Crops every scene to the exact same Kratie Province bounding box and reprojects
onto a shared meter-based CRS (EPSG:32648, UTM Zone 48N, 20m x 20m pixels) so
every pixel across all 4 GeoTIFFs aligns 1-to-1 (1 pixel = 400 m²).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape

# Single unified free STAC source
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Kratie Province AOI (100% inside Sentinel-2 MGRS tile T48PXU)
# Covers Kratie town (12.49°N, 106.02°E), Koh Trong island, Prek Prasab & Chhloung floodplains
AOI_BBOX_WGS84 = [105.95, 12.26, 106.15, 12.52]  # [lon_min, lat_min, lon_max, lat_max]
TARGET_MGRS_TILE = "48PXU"

TARGET_CRS = "EPSG:32648"  # UTM Zone 48N (meters — standard projected CRS for Cambodia)
PIXEL_SIZE_M = 20.0        # 20m x 20m = 400 m² per pixel

# 2026 seasonal search windows
DRY_WINDOW = "2026-02-01/2026-04-15"
WET_WINDOW = "2026-08-25/2026-10-31"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "NO",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_CHUNK_SIZE": "1048576",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
}


def build_target_grid(
    bbox_wgs84: list[float],
    target_crs: str = TARGET_CRS,
    pixel_size_m: float = PIXEL_SIZE_M,
) -> dict[str, Any]:
    """Compute a snapped meter-based raster grid covering the WGS84 bounding box."""
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84

    xs, ys = [], []
    for lon in (lon_min, lon_max):
        for lat in (lat_min, lat_max):
            x, y = transformer.transform(lon, lat)
            xs.append(x)
            ys.append(y)

    west = math.floor(min(xs) / pixel_size_m) * pixel_size_m
    east = math.ceil(max(xs) / pixel_size_m) * pixel_size_m
    south = math.floor(min(ys) / pixel_size_m) * pixel_size_m
    north = math.ceil(max(ys) / pixel_size_m) * pixel_size_m

    width = int(round((east - west) / pixel_size_m))
    height = int(round((north - south) / pixel_size_m))
    transform = from_origin(west, north, pixel_size_m, pixel_size_m)

    return {
        "crs": target_crs,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "width": width,
        "height": height,
        "pixel_size_m": pixel_size_m,
        "transform": transform,
    }


def aoi_coverage_fraction(item: Any, aoi_geom: Any) -> float:
    """Return fraction (0..1) of the AOI covered by a STAC item footprint."""
    if not item.geometry:
        return 0.0
    item_geom = shape(item.geometry)
    inter = item_geom.intersection(aoi_geom)
    return float(inter.area / aoi_geom.area) if aoi_geom.area > 0 else 0.0


def reproject_asset_to_grid(
    href: str,
    grid: dict[str, Any],
    dst_dtype: np.dtype,
    resampling: Resampling,
    dst_nodata: float | int,
    overview_downsample: int = 1,
) -> np.ndarray:
    """Read ONLY the Kratie COG window over HTTPS (HTTP range requests) and reproject onto target grid."""
    dst = np.full((grid["height"], grid["width"]), dst_nodata, dtype=dst_dtype)
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            src_bounds = transform_bounds(
                grid["crs"],
                src.crs,
                grid["west"],
                grid["south"],
                grid["east"],
                grid["north"],
                densify_pts=8,
            )
            win = from_bounds(*src_bounds, transform=src.transform)
            col_off = max(0, int(math.floor(win.col_off)) - 4)
            row_off = max(0, int(math.floor(win.row_off)) - 4)
            col_end = min(src.width, int(math.ceil(win.col_off + win.width)) + 4)
            row_end = min(src.height, int(math.ceil(win.row_off + win.height)) + 4)
            win_w = max(1, col_end - col_off)
            win_h = max(1, row_end - row_off)
            clipped_win = Window(col_off, row_off, win_w, win_h)

            if overview_downsample > 1:
                out_h = max(1, win_h // overview_downsample)
                out_w = max(1, win_w // overview_downsample)
                src_data = src.read(1, window=clipped_win, out_shape=(out_h, out_w), resampling=resampling)
                win_transform = src.window_transform(clipped_win)
                src_win_transform = win_transform * win_transform.scale(
                    win_w / float(out_w), win_h / float(out_h)
                )
            else:
                src_data = src.read(1, window=clipped_win)
                src_win_transform = src.window_transform(clipped_win)

            reproject(
                source=src_data,
                destination=dst,
                src_transform=src_win_transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=grid["transform"],
                dst_crs=grid["crs"],
                dst_nodata=dst_nodata,
                resampling=resampling,
            )
    return dst


def write_geotiff(
    out_path: Path,
    bands: list[np.ndarray],
    band_names: list[str],
    grid: dict[str, Any],
    dtype: str,
    nodata: float | int,
) -> None:
    """Write a multi-band compressed GeoTIFF with band descriptions."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": len(bands),
        "dtype": dtype,
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        for idx, (arr, name) in enumerate(zip(bands, band_names), start=1):
            dst.write(arr.astype(dtype), idx)
            dst.set_band_description(idx, name)


def select_sentinel2_scenes(
    catalog: pystac_client.Client,
    aoi_geom: Any,
) -> tuple[Any, Any]:
    """Select 1 clear dry-season S2 scene and 1 cloudy rainy-season S2 scene on tile 48PXU."""
    def _filter_tile(items: list[Any]) -> list[Any]:
        matched = []
        for it in items:
            tile = str(it.properties.get("s2:mgrs_tile", ""))
            if (TARGET_MGRS_TILE in tile or TARGET_MGRS_TILE in it.id) and aoi_coverage_fraction(it, aoi_geom) >= 0.99:
                matched.append(it)
        return matched

    dry_items = list(
        catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=AOI_BBOX_WGS84,
            datetime=DRY_WINDOW,
            limit=50,
        ).items()
    )
    dry_tile_items = _filter_tile(dry_items)
    if not dry_tile_items:
        raise RuntimeError("No full-coverage Dry Season Sentinel-2 scenes found for Kratie in 2026.")
    dry_tile_items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100.0))
    s2_dry = dry_tile_items[0]

    wet_items = list(
        catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=AOI_BBOX_WGS84,
            datetime=WET_WINDOW,
            limit=60,
        ).items()
    )
    wet_tile_items = _filter_tile(wet_items)
    if not wet_tile_items:
        raise RuntimeError("No full-coverage Rainy Season Sentinel-2 scenes found for Kratie in 2026.")

    # Prefer a visibly cloudy monsoon scene (55%–92% scene cloud cover) in Sep–Oct 2026
    cloudy_mid = [
        it for it in wet_tile_items
        if 55.0 <= float(it.properties.get("eo:cloud_cover", 0.0)) <= 92.0
    ]
    if cloudy_mid:
        cloudy_mid.sort(key=lambda x: abs(float(x.properties.get("eo:cloud_cover", 75.0)) - 75.0))
        s2_wet = cloudy_mid[0]
    else:
        wet_tile_items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 0.0), reverse=True)
        s2_wet = wet_tile_items[0]

    return s2_dry, s2_wet


def select_sentinel1_scenes(
    catalog: pystac_client.Client,
    aoi_geom: Any,
    grid: dict[str, Any],
) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Select matching-orbit Dry & Peak-Flood Rainy Sentinel-1 RTC scenes over Kratie.
    Samples candidate Sep–Oct 2026 scenes in the same orbit direction to pick the
    date with the strongest real Mekong flood signal (highest dark-water pixel fraction).
    """
    dry_all = list(
        catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=AOI_BBOX_WGS84,
            datetime=DRY_WINDOW,
            limit=50,
        ).items()
    )
    wet_all = list(
        catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=AOI_BBOX_WGS84,
            datetime=WET_WINDOW,
            limit=60,
        ).items()
    )

    dry_full = [
        it for it in dry_all
        if aoi_coverage_fraction(it, aoi_geom) >= 0.98 and "vv" in it.assets and "vh" in it.assets
    ]
    wet_full = [
        it for it in wet_all
        if aoi_coverage_fraction(it, aoi_geom) >= 0.98 and "vv" in it.assets and "vh" in it.assets
    ]

    if not dry_full or not wet_full:
        raise RuntimeError(
            f"Insufficient full-coverage Sentinel-1 RTC scenes (dry={len(dry_full)}, wet={len(wet_full)})."
        )

    orbit_pairs = {}
    for orbit in ("ascending", "descending"):
        d_orb = [it for it in dry_full if it.properties.get("sat:orbit_state") == orbit]
        w_orb = [it for it in wet_full if it.properties.get("sat:orbit_state") == orbit]
        if d_orb and w_orb:
            orbit_pairs[orbit] = (d_orb, w_orb)

    if not orbit_pairs:
        raise RuntimeError("No matching Sentinel-1 orbit direction found across Dry and Rainy 2026.")

    chosen_orbit = max(orbit_pairs.keys(), key=lambda o: len(orbit_pairs[o][1]))
    dry_candidates, wet_candidates = orbit_pairs[chosen_orbit]

    dry_candidates.sort(key=lambda it: it.datetime)
    s1_dry = dry_candidates[-1]

    print(f"  Selected S1 Dry ({chosen_orbit}): {s1_dry.id} ({s1_dry.datetime.date()})", flush=True)
    dry_vv = reproject_asset_to_grid(s1_dry.assets["vv"].href, grid, np.float32, Resampling.bilinear, np.nan, overview_downsample=2)
    dry_vh = reproject_asset_to_grid(s1_dry.assets["vh"].href, grid, np.float32, Resampling.bilinear, np.nan, overview_downsample=2)

    coarse_grid = build_target_grid(AOI_BBOX_WGS84, TARGET_CRS, pixel_size_m=100.0)
    best_wet = None
    best_water_frac = -1.0
    # Focus candidate scan on late Sep – Oct 2026 (peak Mekong flood season), max 5 scenes
    wet_sorted = sorted(wet_candidates, key=lambda it: it.datetime)
    sep_oct = [it for it in wet_sorted if it.datetime.month >= 9]
    scan_list = sep_oct[-5:] if len(sep_oct) > 5 else (sep_oct or wet_sorted[-5:])
    print(f"  Scanning {len(scan_list)} Rainy Season S1 candidates ({chosen_orbit}) for peak Mekong water...", flush=True)
    for cand in scan_list:
        preview_vv = reproject_asset_to_grid(
            cand.assets["vv"].href,
            coarse_grid,
            np.float32,
            Resampling.bilinear,
            np.nan,
            overview_downsample=8,
        )
        valid = np.isfinite(preview_vv) & (preview_vv > 0)
        if valid.mean() < 0.90:
            continue
        vv_db = 10.0 * np.log10(np.clip(preview_vv[valid], 1e-6, None))
        water_frac = float((vv_db < -15.5).mean())
        print(
            f"    - {cand.datetime.date()} | {cand.id[:28]}... | "
            f"valid={valid.mean()*100:.1f}% | dark-water(<-15.5 dB)={water_frac*100:.1f}%",
            flush=True,
        )
        if water_frac > best_water_frac:
            best_water_frac = water_frac
            best_wet = cand

    if best_wet is None:
        raise RuntimeError("No valid Rainy Season Sentinel-1 RTC scene with >=90% valid pixels found.")

    print(f"  Selected S1 Peak-Flood Rainy ({chosen_orbit}): {best_wet.id} ({best_wet.datetime.date()})", flush=True)
    wet_vv = reproject_asset_to_grid(best_wet.assets["vv"].href, grid, np.float32, Resampling.bilinear, np.nan, overview_downsample=2)
    wet_vh = reproject_asset_to_grid(best_wet.assets["vh"].href, grid, np.float32, Resampling.bilinear, np.nan, overview_downsample=2)

    return s1_dry, best_wet, dry_vv, dry_vh, wet_vv, wet_vh


def fetch_s2_stack(item: Any, grid: dict[str, Any]) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Fetch B02, B03, B04, B08, SCL from a signed Sentinel-2 L2A STAC item."""
    band_keys = ["B02", "B03", "B04", "B08", "SCL"]
    arrays: list[np.ndarray] = []
    for key in band_keys:
        resampling = Resampling.nearest if key == "SCL" else Resampling.bilinear
        # 10m optical bands (B02/B03/B04/B08) read directly from 20m COG overview (4x faster)
        downsample = 1 if key == "SCL" else 2
        print(f"    -> reading {key}...", flush=True)
        arr = reproject_asset_to_grid(
            item.assets[key].href,
            grid=grid,
            dst_dtype=np.uint16,
            resampling=resampling,
            dst_nodata=0,
            overview_downsample=downsample,
        )
        arrays.append(arr)

    b04_extra = item.assets["B04"].extra_fields or {}
    raster_bands = b04_extra.get("raster:bands", [{}])
    pb = str(item.properties.get("s2:processing_baseline", ""))
    default_offset = -1000.0 if (pb and float(pb) >= 4.0) else 0.0
    offset = raster_bands[0].get("offset", default_offset) if raster_bands else default_offset
    scale = raster_bands[0].get("scale", 0.0001) if raster_bands else 0.0001

    meta = {
        "id": item.id,
        "date": str(item.datetime.date()),
        "datetime": item.datetime.isoformat(),
        "mgrs_tile": item.properties.get("s2:mgrs_tile"),
        "scene_cloud_cover_pct": round(float(item.properties.get("eo:cloud_cover", 0.0)), 2),
        "processing_baseline": pb,
        "reflectance_scale": float(scale),
        "reflectance_offset": float(offset),
        "bands": ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR", "SCL"],
    }
    return arrays, meta


def main() -> None:
    print("=== Layer 1: Fetching 2026 Kratie Satellite Imagery (Microsoft Planetary Computer) ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    aoi_geom = box(*AOI_BBOX_WGS84)
    grid = build_target_grid(AOI_BBOX_WGS84, TARGET_CRS, PIXEL_SIZE_M)
    print(
        f"Target Grid: {grid['crs']} | {grid['width']}x{grid['height']} px "
        f"({grid['pixel_size_m']:.0f}m = {int(grid['pixel_size_m']**2)} m²/px) | "
        f"Area: {grid['width'] * grid['height'] * (grid['pixel_size_m']**2) / 1e6:.2f} km²"
    )

    catalog = pystac_client.Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)

    print("\n[1/2] Searching Sentinel-2 L2A (Optical) scenes...")
    s2_dry_item, s2_wet_item = select_sentinel2_scenes(catalog, aoi_geom)
    print(
        f"  Selected S2 Dry: {s2_dry_item.id} ({s2_dry_item.datetime.date()}, "
        f"scene cloud={s2_dry_item.properties.get('eo:cloud_cover'):.1f}%)"
    )
    print(
        f"  Selected S2 Wet: {s2_wet_item.id} ({s2_wet_item.datetime.date()}, "
        f"scene cloud={s2_wet_item.properties.get('eo:cloud_cover'):.1f}%)"
    )

    print("  Downloading & aligning Sentinel-2 Dry bands (B02, B03, B04, B08, SCL)...")
    s2_dry_bands, s2_dry_meta = fetch_s2_stack(s2_dry_item, grid)
    write_geotiff(
        DATA_DIR / "s2_dry.tif",
        s2_dry_bands,
        s2_dry_meta["bands"],
        grid,
        dtype="uint16",
        nodata=0,
    )

    print("  Downloading & aligning Sentinel-2 Rainy bands (B02, B03, B04, B08, SCL)...")
    s2_wet_bands, s2_wet_meta = fetch_s2_stack(s2_wet_item, grid)
    write_geotiff(
        DATA_DIR / "s2_wet.tif",
        s2_wet_bands,
        s2_wet_meta["bands"],
        grid,
        dtype="uint16",
        nodata=0,
    )

    print("\n[2/2] Searching Sentinel-1 RTC (SAR Radar) scenes...")
    s1_dry_item, s1_wet_item, dry_vv, dry_vh, wet_vv, wet_vh = select_sentinel1_scenes(
        catalog, aoi_geom, grid
    )
    write_geotiff(
        DATA_DIR / "s1_dry.tif",
        [dry_vv, dry_vh],
        ["VV", "VH"],
        grid,
        dtype="float32",
        nodata=float("nan"),
    )
    write_geotiff(
        DATA_DIR / "s1_wet.tif",
        [wet_vv, wet_vh],
        ["VV", "VH"],
        grid,
        dtype="float32",
        nodata=float("nan"),
    )

    metadata = {
        "project": "MekongWatch",
        "province": "Kratie Province, Cambodia",
        "data_source": "Microsoft Planetary Computer STAC (free, keyless)",
        "stac_api": STAC_API_URL,
        "aoi_bbox_wgs84": AOI_BBOX_WGS84,
        "grid": {
            "crs": grid["crs"],
            "west": grid["west"],
            "south": grid["south"],
            "east": grid["east"],
            "north": grid["north"],
            "width": grid["width"],
            "height": grid["height"],
            "pixel_size_m": grid["pixel_size_m"],
            "pixel_area_m2": grid["pixel_size_m"] ** 2,
            "total_aoi_area_km2": round(
                grid["width"] * grid["height"] * (grid["pixel_size_m"] ** 2) / 1e6, 2
            ),
        },
        "sentinel2_dry": s2_dry_meta,
        "sentinel2_wet": s2_wet_meta,
        "sentinel1_dry": {
            "id": s1_dry_item.id,
            "date": str(s1_dry_item.datetime.date()),
            "datetime": s1_dry_item.datetime.isoformat(),
            "platform": s1_dry_item.properties.get("platform"),
            "orbit_state": s1_dry_item.properties.get("sat:orbit_state"),
            "bands": ["VV", "VH"],
        },
        "sentinel1_wet": {
            "id": s1_wet_item.id,
            "date": str(s1_wet_item.datetime.date()),
            "datetime": s1_wet_item.datetime.isoformat(),
            "platform": s1_wet_item.properties.get("platform"),
            "orbit_state": s1_wet_item.properties.get("sat:orbit_state"),
            "bands": ["VV", "VH"],
        },
    }
    meta_path = DATA_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved all 4 aligned GeoTIFFs + {meta_path.name} to {DATA_DIR}")


if __name__ == "__main__":
    main()
