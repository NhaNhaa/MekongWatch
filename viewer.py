"""
Layer 4: Clean 2-Tab Streamlit Visual Viewer (MekongWatch — Kratie Province)

Run locally:
    streamlit run mekongwatch/viewer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="MekongWatch — Kratie Flood Monitoring",
    page_icon="🛰️",
    layout="wide",
)


def load_json(path: Path) -> dict:
    if not path.exists():
        st.error(f"Missing `{path.name}`. Run `fetch_data.py`, `phase1_optical.py`, and `phase2_sar_flood.py` first.")
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


meta = load_json(DATA_DIR / "metadata.json")
p1 = load_json(OUTPUTS_DIR / "phase1_summary.json")
p2 = load_json(OUTPUTS_DIR / "phase2_summary.json")

st.title("🛰️ MekongWatch — Kratie Province Mekong Flood Monitoring (2026)")
st.caption(
    f"Data Source: {meta['data_source']} · "
    f"Aligned Grid: {p1['crs']} ({int(p1['pixel_size_m'])}m resolution = {int(p1['pixel_area_m2'])} m²/pixel) · "
    f"AOI: {p1['total_aoi_area_km2']} km²"
)

tab1, tab2 = st.tabs([
    "☀️ Phase 1: Optical Baseline (Sentinel-2)",
    "📡 Phase 2: SAR Radar Flood, GeoJSON & GeoAI (Sentinel-1)",
])

with tab1:
    st.subheader("Phase 1 — Optical Satellite (Passive Sensor), Cloud Masking, NDWI & NDVI")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dry Season Cloud Cover", f"{p1['dry_season']['aoi_cloud_cover_pct']:.1f}%", p1["dry_season"]["date"])
    c2.metric("Rainy Season Cloud Cover", f"{p1['rainy_season']['aoi_cloud_cover_pct']:.1f}%", p1["rainy_season"]["date"])
    c3.metric("Normal Mekong River (NDWI)", f"{p1['dry_season']['normal_river_area_km2']:.2f} km²")
    c4.metric("Healthy Green Farmland (NDVI)", f"{p1['dry_season']['healthy_farmland_area_km2']:.2f} km²")

    st.info(p1["rainy_season"]["optical_blocked_note"])

    p1_img = OUTPUTS_DIR / "phase1_optical_baseline.png"
    if p1_img.exists():
        st.image(str(p1_img), width="stretch")

    tl = p1["corners_wgs84"]["top_left"]
    br = p1["corners_wgs84"]["bottom_right"]
    st.markdown(
        f"**GeoTIFF Spatial Metadata:** `{p1['crs']}` | Raster Size: `{p1['raster_shape'][1]} × {p1['raster_shape'][0]}` px | "
        f"Top-Left: `({tl['lat']}°N, {tl['lon']}°E)` → Bottom-Right: `({br['lat']}°N, {br['lon']}°E)`"
    )

with tab2:
    st.subheader("Phase 2 — SAR Radar Backscatter, Change Detection, Vector GeoJSON & GeoAI")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Permanent Mekong River", f"{p2['areas_km2']['permanent_river_km2']:.2f} km²", p2["dry_date"])
    c2.metric("Newly Flooded Land", f"{p2['areas_km2']['newly_flooded_land_km2']:.2f} km²", p2["rainy_date"])
    c3.metric("Flooded Farmland", f"{p2['areas_km2']['flooded_farmland_km2']:.2f} km²")
    c4.metric(
        "GeoAI Random Forest Accuracy",
        f"{p2['geoai']['overall_accuracy']*100:.1f}%",
        f"Macro F1: {p2['geoai']['macro_f1']*100:.1f}%",
    )

    p2_sar_img = OUTPUTS_DIR / "phase2_sar_flood_map.png"
    if p2_sar_img.exists():
        st.image(str(p2_sar_img), width="stretch")

    st.divider()
    st.subheader("GeoAI Multi-Sensor Classification & Vector GeoJSON Summary")
    p2_rf_img = OUTPUTS_DIR / "phase2_geoai_landcover.png"
    if p2_rf_img.exists():
        st.image(str(p2_rf_img), width="stretch")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Confusion Matrix (30% Holdout Test Pixels)**")
        cm_df = pd.DataFrame(
            p2["geoai"]["confusion_matrix"],
            index=[f"True: {lbl}" for lbl in p2["geoai"]["class_labels"]],
            columns=[f"Pred: {lbl}" for lbl in p2["geoai"]["class_labels"]],
        )
        st.dataframe(cm_df, width="stretch")

    with col_right:
        st.markdown("**Vector Export (`outputs/flood_zones.geojson`)**")
        st.write(
            f"- **Format:** GeoJSON (`{p2['vector_geojson']['crs']}`)\n"
            f"- **Flooded Polygons:** `{p2['vector_geojson']['polygon_count']:,}`\n"
            f"- **Largest Flood Zone:** `{p2['vector_geojson']['largest_flood_zone_km2']:.2f} km²`"
        )
        st.warning(f"**Ground-Truthing (Human-in-the-Loop QA):** {p2['ground_truthing_note']}")
