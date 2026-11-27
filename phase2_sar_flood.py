"""
Layer 3: Phase 2 — SAR Flood Change Detection, Vector GIS & GeoAI (MekongWatch — Kratie Province)

Strictly demonstrates Lesson 2 concepts (11–16):
  - SAR Active Radar Satellite (Sentinel-1 RTC, VV + VH — sees through monsoon clouds)
  - Backscatter (dB): smooth calm water = dark mirror reflection, buildings/trees = bright
  - Dry-vs-Rainy Change Detection: Permanent Mekong River vs. Newly Flooded Land (+ flooded farmland km²)
  - Raster -> Vector Conversion: outputs/flood_zones.geojson (EPSG:4326 polygons with area_km2)
  - GeoAI Machine Learning: RandomForestClassifier on raw multi-sensor pixel bands
  - Ground-Truthing (Human-in-the-Loop QA): explicit false-positive review notes
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
from rasterio.features import shapes
from scipy.ndimage import label as connected_components
from scipy.ndimage import median_filter
from shapely import transform as shapely_transform_coords
from shapely.geometry import mapping, shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from phase1_optical import load_s2_geotiff, pixels_to_km2

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# SAR physical thresholds (Sentinel-1 VV gamma0 in dB)
SAR_WATER_THRESHOLD_DB = -15.5     # Calm open water reflects radar away -> dark (< -15.5 dB)
SAR_FLOOD_DROP_DB = -2.5           # Significant backscatter drop from Dry to Rainy season
MIN_FLOOD_CLUSTER_PIXELS = 5       # 5 px = 2,000 m² minimum polygon to drop isolated speckle

CLASS_NAMES = {
    0: "Dry Land",
    1: "Permanent Mekong River",
    2: "Newly Flooded Land",
}


def linear_to_db(gamma0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert linear SAR gamma0 backscatter to decibels (dB), preserving NaNs."""
    arr = np.asarray(gamma0, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    out[valid] = 10.0 * np.log10(np.clip(arr[valid], eps, None))
    return out


def denoise_sar_db(db_arr: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply a 3x3 median filter to reduce radar speckle while preserving river edges."""
    valid = np.isfinite(db_arr)
    if not valid.any():
        return db_arr.copy()
    fill_val = float(np.nanmedian(db_arr))
    filled = np.where(valid, db_arr, fill_val)
    filtered = median_filter(filled, size=size).astype(np.float32)
    filtered[~valid] = np.nan
    return filtered


def remove_small_clusters(mask: np.ndarray, min_pixels: int = MIN_FLOOD_CLUSTER_PIXELS) -> np.ndarray:
    """Remove isolated connected components smaller than min_pixels to eliminate speckle noise."""
    labeled, num_features = connected_components(mask.astype(np.uint8))
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    keep = counts >= min_pixels
    keep[0] = False  # background
    return keep[labeled]


def detect_sar_flood(
    dry_vv_db: np.ndarray,
    wet_vv_db: np.ndarray,
    water_thresh_db: float = SAR_WATER_THRESHOLD_DB,
    drop_thresh_db: float = SAR_FLOOD_DROP_DB,
    min_cluster_pixels: int = MIN_FLOOD_CLUSTER_PIXELS,
    dry_optical_water_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare Dry vs. Rainy SAR backscatter (VV dB) to classify:
      0 = Dry Land
      1 = Permanent Water (dark in both Dry & Rainy, or confirmed water in Dry Optical baseline)
      2 = Newly Flooded Land (dry in Dry -> dark water in Rainy with drop < drop_thresh_db)
    Returns (change_class_map, diff_db).
    """
    valid = np.isfinite(dry_vv_db) & np.isfinite(wet_vv_db)
    diff_db = np.full(dry_vv_db.shape, np.nan, dtype=np.float32)
    diff_db[valid] = wet_vv_db[valid] - dry_vv_db[valid]

    dry_water = valid & (dry_vv_db < water_thresh_db)
    if dry_optical_water_mask is not None:
        # Fusing Phase 1 clear dry-season river prevents dry wind-ripple pixels from looking like "new flood"
        dry_water = dry_water | (valid & dry_optical_water_mask.astype(bool))
    wet_water = valid & (wet_vv_db < water_thresh_db)

    permanent_water = dry_water & wet_water
    raw_new_flood = valid & (~dry_water) & wet_water & (diff_db < drop_thresh_db)
    clean_new_flood = remove_small_clusters(raw_new_flood, min_pixels=min_cluster_pixels)

    change_map = np.zeros(dry_vv_db.shape, dtype=np.uint8)
    change_map[permanent_water] = 1
    change_map[clean_new_flood] = 2
    return change_map, diff_db


def vectorize_flood_mask(
    flood_mask: np.ndarray,
    transform: Any,
    src_crs: str,
    min_pixels: int = MIN_FLOOD_CLUSTER_PIXELS,
) -> dict[str, Any]:
    """Convert newly flooded raster pixels into a WGS 84 (EPSG:4326) GeoJSON FeatureCollection."""
    cleaned = remove_small_clusters(flood_mask, min_pixels=min_pixels)
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    def _to_wgs84(coords: np.ndarray) -> np.ndarray:
        lons, lats = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([lons, lats])

    features: list[dict[str, Any]] = []
    for geom_dict, val in shapes(cleaned.astype(np.uint8), mask=cleaned, transform=transform):
        if int(val) != 1:
            continue
        geom_utm = shape(geom_dict)
        area_km2 = float(geom_utm.area) / 1_000_000.0
        if area_km2 <= 0:
            continue
        geom_wgs = shapely_transform_coords(geom_utm, _to_wgs84)
        centroid = geom_wgs.centroid
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "class": "Newly Flooded Land",
                    "area_km2": round(area_km2, 4),
                    "centroid_lat": round(float(centroid.y), 5),
                    "centroid_lon": round(float(centroid.x), 5),
                },
                "geometry": mapping(geom_wgs),
            }
        )

    # Sort polygons from largest flooded zone to smallest and assign clean IDs
    features.sort(key=lambda f: f["properties"]["area_km2"], reverse=True)
    for idx, feat in enumerate(features, start=1):
        feat["properties"]["zone_id"] = f"KRATIE_FLOOD_{idx:04d}"

    return {
        "type": "FeatureCollection",
        "name": "MekongWatch_Kratie_Flood_Zones_2026",
        "features": features,
    }


def train_geoai_random_forest(
    feature_stack: np.ndarray,
    feature_names: list[str],
    seed_labels: np.ndarray,
    max_samples_per_class: int = 4000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train a scikit-learn RandomForestClassifier on raw multi-sensor pixel features
    [Blue, Green, Red, NIR, SAR_Dry_VV_dB, SAR_Dry_VH_dB, SAR_Wet_VV_dB, SAR_Wet_VH_dB].
    """
    h, w, n_feat = feature_stack.shape
    flat_x = feature_stack.reshape(-1, n_feat)
    flat_y = seed_labels.reshape(-1)
    valid_rows = np.all(np.isfinite(flat_x), axis=1)

    rng = np.random.default_rng(random_state)
    sampled_idx: list[np.ndarray] = []
    class_counts: dict[str, int] = {}

    for cls_id in (0, 1, 2):
        cls_indices = np.where(valid_rows & (flat_y == cls_id))[0]
        if len(cls_indices) == 0:
            raise RuntimeError(f"GeoAI training failed: zero valid pixels for class {cls_id} ({CLASS_NAMES[cls_id]}).")
        take = min(len(cls_indices), max_samples_per_class)
        chosen = rng.choice(cls_indices, size=take, replace=False)
        sampled_idx.append(chosen)
        class_counts[CLASS_NAMES[cls_id]] = int(take)

    all_idx = np.concatenate(sampled_idx)
    X = flat_x[all_idx]
    y = flat_y[all_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()

    importances = {
        name: round(float(imp), 4)
        for name, imp in zip(feature_names, clf.feature_importances_)
    }

    # Predict full Kratie raster map in chunks to keep memory low
    pred_flat = np.zeros(flat_x.shape[0], dtype=np.uint8)
    valid_indices = np.where(valid_rows)[0]
    chunk_size = 100_000
    for start in range(0, len(valid_indices), chunk_size):
        idx_chunk = valid_indices[start : start + chunk_size]
        pred_flat[idx_chunk] = clf.predict(flat_x[idx_chunk])
    pred_map = pred_flat.reshape(h, w)

    return {
        "model": "RandomForestClassifier (n_estimators=100, max_depth=12)",
        "features": feature_names,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "sampled_per_class": class_counts,
        "overall_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
        "class_labels": [CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2]],
        "feature_importances": importances,
        "pred_map": pred_map,
    }


def main() -> None:
    print("=== Layer 3: Phase 2 — SAR Flood Change Detection, GeoJSON & GeoAI (Sentinel-1) ===")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))

    # 1. Load aligned Sentinel-1 Dry & Rainy GeoTIFFs
    with rasterio.open(DATA_DIR / "s1_dry.tif") as src_dry:
        dry_vv_lin = src_dry.read(1)
        dry_vh_lin = src_dry.read(2)
        transform = src_dry.transform
        crs_str = str(src_dry.crs)
        px_m = float(abs(transform.a))

    with rasterio.open(DATA_DIR / "s1_wet.tif") as src_wet:
        wet_vv_lin = src_wet.read(1)
        wet_vh_lin = src_wet.read(2)

    # 2. Convert linear gamma0 -> dB & apply 3x3 median speckle filter
    dry_vv_db = denoise_sar_db(linear_to_db(dry_vv_lin), size=3)
    dry_vh_db = denoise_sar_db(linear_to_db(dry_vh_lin), size=3)
    wet_vv_db = denoise_sar_db(linear_to_db(wet_vv_lin), size=3)
    wet_vh_db = denoise_sar_db(linear_to_db(wet_vh_lin), size=3)

    # 3. Dry-vs-Rainy Change Detection (fusing Phase 1 cloud-free dry river mask to ignore dry wind ripples)
    dry_water_opt_path = OUTPUTS_DIR / "dry_water_ndwi_mask.npy"
    dry_optical_water = np.load(dry_water_opt_path) if dry_water_opt_path.exists() else None

    change_map, diff_db = detect_sar_flood(
        dry_vv_db,
        wet_vv_db,
        water_thresh_db=SAR_WATER_THRESHOLD_DB,
        drop_thresh_db=SAR_FLOOD_DROP_DB,
        min_cluster_pixels=MIN_FLOOD_CLUSTER_PIXELS,
        dry_optical_water_mask=dry_optical_water,
    )

    perm_water_mask = change_map == 1
    new_flood_mask = change_map == 2
    dry_land_mask = change_map == 0

    perm_river_km2 = round(pixels_to_km2(int(perm_water_mask.sum()), px_m), 2)
    newly_flooded_km2 = round(pixels_to_km2(int(new_flood_mask.sum()), px_m), 2)
    dry_land_km2 = round(pixels_to_km2(int(dry_land_mask.sum()), px_m), 2)

    # Intersect with Phase 1 dry-season farmland mask
    farmland_path = OUTPUTS_DIR / "dry_farmland_mask.npy"
    if farmland_path.exists():
        dry_farmland_mask = np.load(farmland_path)
        flooded_farmland_mask = new_flood_mask & dry_farmland_mask
        flooded_farmland_km2 = round(pixels_to_km2(int(flooded_farmland_mask.sum()), px_m), 2)
    else:
        flooded_farmland_mask = np.zeros_like(new_flood_mask)
        flooded_farmland_km2 = 0.0

    # 4. Raster -> Vector GeoJSON Export (EPSG:4326)
    geojson_fc = vectorize_flood_mask(
        new_flood_mask,
        transform=transform,
        src_crs=crs_str,
        min_pixels=MIN_FLOOD_CLUSTER_PIXELS,
    )
    geojson_path = OUTPUTS_DIR / "flood_zones.geojson"
    geojson_path.write_text(json.dumps(geojson_fc, indent=2), encoding="utf-8")

    num_polygons = len(geojson_fc["features"])
    largest_poly_km2 = (
        round(geojson_fc["features"][0]["properties"]["area_km2"], 2) if num_polygons > 0 else 0.0
    )

    # 5. GeoAI Random Forest on raw multi-sensor pixel features
    dry_s2_meta = meta["sentinel2_dry"]
    s2_dry = load_s2_geotiff(
        DATA_DIR / "s2_dry.tif",
        offset=dry_s2_meta.get("reflectance_offset", -1000.0),
        scale=dry_s2_meta.get("reflectance_scale", 0.0001),
    )

    feature_names = [
        "Optical_Blue",
        "Optical_Green",
        "Optical_Red",
        "Optical_NIR",
        "SAR_Dry_VV_dB",
        "SAR_Dry_VH_dB",
        "SAR_Wet_VV_dB",
        "SAR_Wet_VH_dB",
    ]
    feature_stack = np.stack(
        [
            s2_dry["blue"],
            s2_dry["green"],
            s2_dry["red"],
            s2_dry["nir"],
            dry_vv_db,
            dry_vh_db,
            wet_vv_db,
            wet_vh_db,
        ],
        axis=-1,
    )

    geoai_res = train_geoai_random_forest(
        feature_stack=feature_stack,
        feature_names=feature_names,
        seed_labels=change_map,
    )
    pred_map = geoai_res.pop("pred_map")
    geoai_flood_km2 = round(pixels_to_km2(int((pred_map == 2).sum()), px_m), 2)

    # Save Phase 2 summary JSON
    s1_dry_date = meta["sentinel1_dry"]["date"]
    s1_wet_date = meta["sentinel1_wet"]["date"]
    s1_orbit = meta["sentinel1_dry"]["orbit_state"]

    summary = {
        "province": meta["province"],
        "sensor": f"Sentinel-1 RTC SAR (Active Radar, VV + VH, {s1_orbit} orbit)",
        "crs": crs_str,
        "pixel_size_m": px_m,
        "dry_date": s1_dry_date,
        "rainy_date": s1_wet_date,
        "thresholds": {
            "water_vv_db": SAR_WATER_THRESHOLD_DB,
            "flood_drop_db": SAR_FLOOD_DROP_DB,
            "min_cluster_pixels": MIN_FLOOD_CLUSTER_PIXELS,
        },
        "areas_km2": {
            "permanent_river_km2": perm_river_km2,
            "newly_flooded_land_km2": newly_flooded_km2,
            "flooded_farmland_km2": flooded_farmland_km2,
            "dry_land_km2": dry_land_km2,
            "geoai_predicted_flood_km2": geoai_flood_km2,
        },
        "vector_geojson": {
            "file": "outputs/flood_zones.geojson",
            "crs": "EPSG:4326",
            "polygon_count": num_polygons,
            "largest_flood_zone_km2": largest_poly_km2,
        },
        "geoai": geoai_res,
        "ground_truthing_note": (
            "Seed training pixels come from multi-sensor SAR physics (dry vs. rainy backscatter). "
            "In real disaster deployment, Human-in-the-Loop Ground-Truthing is essential to verify "
            "dark radar false positives (such as smooth asphalt roads, wet bare soil, or flat roofs) "
            "against field photos, river-gauge readings, or local NCDM reports."
        ),
    }
    (OUTPUTS_DIR / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 6. Figure A: 3-Panel SAR Backscatter & Flood Change Detection Map
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.2), dpi=160)
    fig.patch.set_facecolor("#F8FAFC")

    ax = axes[0]
    im0 = ax.imshow(dry_vv_db, cmap="gray", vmin=-22, vmax=-4)
    ax.set_title(
        f"1. Dry Season SAR Backscatter ({s1_dry_date})\n"
        f"Smooth River = Dark · Land/Trees = Bright",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    cbar0 = fig.colorbar(im0, ax=ax, fraction=0.044, pad=0.02)
    cbar0.set_label("VV Backscatter (dB)", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    im1 = ax.imshow(wet_vv_db, cmap="gray", vmin=-22, vmax=-4)
    ax.set_title(
        f"2. Rainy Season SAR Backscatter ({s1_wet_date})\n"
        f"Active Radar Sees Straight Through Clouds",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.044, pad=0.02)
    cbar1.set_label("VV Backscatter (dB)", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[2]
    # Base grayscale SAR backdrop + clear color classes
    ax.imshow(wet_vv_db, cmap="gray", vmin=-22, vmax=-4, alpha=0.55)
    cmap_flood = ListedColormap([(0, 0, 0, 0), (0.08, 0.45, 0.90, 0.85), (0.92, 0.14, 0.14, 0.90)])
    ax.imshow(change_map, cmap=cmap_flood, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(
        f"3. Dry-vs-Rainy Change Detection\n"
        f"Newly Flooded: {newly_flooded_km2:.2f} km² (Farmland: {flooded_farmland_km2:.2f} km²)",
        fontsize=10.5,
        fontweight="bold",
        color="#991B1B",
        pad=8,
    )
    ax.legend(
        handles=[
            Patch(facecolor=(0.08, 0.45, 0.90, 0.9), label=f"Permanent River ({perm_river_km2:.2f} km²)"),
            Patch(facecolor=(0.92, 0.14, 0.14, 0.95), label=f"Newly Flooded Land ({newly_flooded_km2:.2f} km²)"),
        ],
        loc="lower right",
        fontsize=8.5,
        framealpha=0.92,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(
        f"MekongWatch Phase 2 — Sentinel-1 SAR Flood Mapping (Kratie Province, 2026)\n"
        f"Same {s1_orbit.title()} Orbit · Vectorized to {num_polygons:,} GeoJSON Polygons (Largest Zone: {largest_poly_km2:.2f} km²)",
        fontsize=12.5,
        fontweight="bold",
        color="#0F172A",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_sar_png = OUTPUTS_DIR / "phase2_sar_flood_map.png"
    fig.savefig(out_sar_png, dpi=160)
    plt.close(fig)

    # 7. Figure B: 2-Panel GeoAI Random Forest Map + Feature Importances
    fig, (ax_map, ax_bar) = plt.subplots(
        1, 2, figsize=(13.5, 6.2), dpi=160, gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    fig.patch.set_facecolor("#F8FAFC")

    cmap_rf = ListedColormap(["#D6D3C4", "#1D4ED8", "#DC2626"])
    ax_map.imshow(pred_map, cmap=cmap_rf, vmin=0, vmax=2, interpolation="nearest")
    ax_map.set_title(
        f"GeoAI Random Forest Map (8 Raw Optical + SAR Bands)\n"
        f"Test Accuracy: {geoai_res['overall_accuracy']*100:.1f}% · Macro F1: {geoai_res['macro_f1']*100:.1f}%",
        fontsize=11,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax_map.legend(
        handles=[
            Patch(facecolor="#D6D3C4", edgecolor="#64748B", label="Dry Land"),
            Patch(facecolor="#1D4ED8", label="Permanent Mekong River"),
            Patch(facecolor="#DC2626", label=f"Predicted Flood ({geoai_flood_km2:.2f} km²)"),
        ],
        loc="lower right",
        fontsize=8.5,
        framealpha=0.92,
    )
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    sorted_imp = sorted(geoai_res["feature_importances"].items(), key=lambda kv: kv[1])
    labels_bar = [k.replace("_", " ") for k, _ in sorted_imp]
    vals_bar = [v * 100.0 for _, v in sorted_imp]
    colors_bar = ["#DC2626" if "SAR Wet" in lbl else ("#2563EB" if "SAR Dry" in lbl else "#16A34A") for lbl in labels_bar]

    bars = ax_bar.barh(labels_bar, vals_bar, color=colors_bar, edgecolor="#1E293B", linewidth=0.6)
    ax_bar.set_xlabel("Feature Importance (%)", fontsize=10, fontweight="bold")
    ax_bar.set_title(
        "Which Satellite Bands Matter Most to the Model?\n"
        "(Red = Rainy SAR · Blue = Dry SAR · Green = Dry Optical)",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        pad=8,
    )
    ax_bar.set_xlim(0, max(vals_bar) * 1.22)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, vals_bar):
        ax_bar.text(val + 0.6, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

    # Add Ground-Truthing callout box cleanly below the x-axis label
    fig.text(
        0.50,
        0.02,
        "Ground-Truthing (Human-in-the-Loop QA): Model trained on multi-sensor physics seeds (70/30 split). "
        "Real disaster deployment requires human verification to rule out flat smooth false positives (roads, rooftops, runways).",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FEF9C3", edgecolor="#CA8A04", alpha=0.92),
    )

    fig.suptitle(
        "MekongWatch Phase 2 — GeoAI Multi-Sensor Classification (Kratie Province, 2026)",
        fontsize=12.5,
        fontweight="bold",
        color="#0F172A",
        y=0.975,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.925])
    out_rf_png = OUTPUTS_DIR / "phase2_geoai_landcover.png"
    fig.savefig(out_rf_png, dpi=160)
    plt.close(fig)

    print(
        f"  Permanent River: {perm_river_km2:.2f} km² | Newly Flooded: {newly_flooded_km2:.2f} km² "
        f"| Flooded Farmland: {flooded_farmland_km2:.2f} km²"
    )
    print(f"  Vectorized {num_polygons:,} polygons -> {geojson_path.name} (largest={largest_poly_km2:.2f} km²)")
    print(
        f"  GeoAI Random Forest: Accuracy={geoai_res['overall_accuracy']*100:.2f}% | "
        f"Macro F1={geoai_res['macro_f1']*100:.2f}%"
    )
    print(f"  Saved {out_sar_png.name}, {out_rf_png.name}, and phase2_summary.json")


if __name__ == "__main__":
    main()
