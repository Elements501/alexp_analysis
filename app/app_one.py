from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, cast

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
from sklearn.linear_model import LinearRegression
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
RECYCLE_CSV = BASE_DIR / "recycle_points" / "wasteless250918.csv"
PRH_JSON = BASE_DIR / "housing" / "prh-estates.json"
HOS_JSON = BASE_DIR / "housing" / "hos-courts.json"
PRIVATE_GEOJSON = BASE_DIR / "housing" / "private.geojson"
SHOPPING_JSON = BASE_DIR / "housing" / "shopping-centres.json"
FLATTED_FACTORY_JSON = BASE_DIR / "housing" / "flatted-factory.json"


DISTRICT_DISPLAY = {
    "CENTRALANDWESTERN": "Central and Western",
    "CENTRALWESTERN": "Central and Western",
    "EASTERN": "Eastern",
    "SOUTHERN": "Southern",
    "WANCHAI": "Wan Chai",
    "KOWLOONCITY": "Kowloon City",
    "KWUNTONG": "Kwun Tong",
    "SHAMSHUIPO": "Sham Shui Po",
    "WONGTAISIN": "Wong Tai Sin",
    "YAUTSIMMONG": "Yau Tsim Mong",
    "ISLANDS": "Islands",
    "KWAITSING": "Kwai Tsing",
    "NORTH": "North",
    "SAIKUNG": "Sai Kung",
    "SHATIN": "Sha Tin",
    "TAIPO": "Tai Po",
    "TSUENWAN": "Tsuen Wan",
    "TUENMUN": "Tuen Mun",
    "YUENLONG": "Yuen Long",
}

LAND_USE_COLORS = {
    "PRH": [52, 152, 219, 180],
    "HOS": [46, 204, 113, 180],
    "Private": [231, 76, 60, 180],
    "Shopping": [241, 196, 15, 180],
    "Factory": [155, 89, 182, 180],
}


def normalize_district(value: Any) -> str:
    text = str(value or "").upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def district_display(norm: str) -> str:
    if norm in DISTRICT_DISPLAY:
        return DISTRICT_DISPLAY[norm]
    if not norm:
        return "Unknown"
    return norm.title()


def extract_en(value: Any) -> str:
    if isinstance(value, dict):
        if value.get("en") is not None:
            return str(value.get("en")).strip()
        for _, v in value.items():
            if v is not None and str(v).strip() != "":
                return str(v).strip()
        return ""
    if value is None:
        return ""
    return str(value).strip()


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_number_from_text(value: Any) -> float:
    text = extract_en(value)
    if text == "":
        return float("nan")
    text = text.replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in {"", "-", "."}:
        return float("nan")
    return parse_float(text)


def shannon_norm(counts: pd.Series) -> float:
    vals = pd.to_numeric(counts, errors="coerce").dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return 0.0
    p = vals / vals.sum()
    h = float(-(p * np.log(p)).sum())
    h_max = math.log(len(vals)) if len(vals) > 1 else 1.0
    return float(h / max(h_max, 1e-9))


def parse_materials(waste_type: Any) -> list[str]:
    text = str(waste_type or "").strip()
    if text == "":
        return []
    items = [part.strip().lower() for part in text.split(",")]
    return [x for x in items if x]


def has_material(materials: list[str], keywords: tuple[str, ...]) -> int:
    joined = ",".join(materials)
    return int(any(k in joined for k in keywords))


def estimate_point_volume_m3_per_day(legend: Any, material_count: int) -> float:
    key = str(legend or "").strip().lower()

    if "smart bin" in key:
        base = 2.0
    elif "station" in key or "store" in key:
        base = 6.0
    elif "street corner" in key:
        base = 4.0
    elif "spot" in key:
        base = 1.5
    elif "private collection" in key:
        base = 1.2
    else:
        base = 1.0

    # More accepted material streams generally means larger or more modular bins.
    material_factor = 0.75 + 0.25 * max(1, material_count)
    return float(base * material_factor)


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)

    dlat = lat2r[None, :] - lat1r[:, None]
    dlon = lon2r[None, :] - lon1r[:, None]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r[:, None]) * np.cos(lat2r[None, :]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def district_area_proxy_km2(group: pd.DataFrame) -> float:
    g = group[["lat", "lon"]].dropna()
    if len(g) < 2:
        return 1.0

    lat_min = g["lat"].min()
    lat_max = g["lat"].max()
    lon_min = g["lon"].min()
    lon_max = g["lon"].max()
    lat_mid = (lat_min + lat_max) / 2.0

    lat_km = max(0.5, (lat_max - lat_min) * 111.0)
    lon_km = max(0.5, (lon_max - lon_min) * 111.0 * math.cos(math.radians(lat_mid)))
    return float(lat_km * lon_km)


@st.cache_data(show_spinner=False)
def load_recycle_points() -> pd.DataFrame:
    df = pd.read_csv(RECYCLE_CSV, encoding="utf-8-sig")
    df["district_norm"] = df["district_id"].map(normalize_district)
    df["district"] = df["district_norm"].map(district_display)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lgt"], errors="coerce")
    df = df[df["lat"].between(22.05, 22.65) & df["lon"].between(113.75, 114.55)].copy()
    df["materials"] = df["waste_type"].apply(parse_materials)
    df["material_count"] = df["materials"].apply(len)
    df["estimated_volume_m3_day"] = [
        estimate_point_volume_m3_per_day(legend, int(cnt))
        for legend, cnt in zip(df["legend"], df["material_count"])
    ]
    df["has_paper"] = df["materials"].apply(lambda x: has_material(x, ("paper",)))
    df["has_plastic"] = df["materials"].apply(lambda x: has_material(x, ("plastics",)))
    df["has_metal"] = df["materials"].apply(lambda x: has_material(x, ("metal", "metals")))
    df["has_plastic_bottles"] = df["materials"].apply(lambda x: has_material(x, ("plastic bottle",)))
    df["has_glass_bottles"] = df["materials"].apply(lambda x: has_material(x, ("glass bottle", "glass bottles")))
    df["has_food_waste"] = df["materials"].apply(lambda x: has_material(x, ("food waste",)))
    return df


@st.cache_data(show_spinner=False)
def load_housing_points() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    with PRH_JSON.open("r", encoding="utf-8") as f:
        prh = json.load(f)
    for item in prh:
        rows.append(
            {
                "land_use": "PRH",
                "district_norm": normalize_district(extract_en(item.get("District Name"))),
                "lat": parse_float(item.get("Estate Map Latitude")),
                "lon": parse_float(item.get("Estate Map Longitude")),
                "units_proxy": parse_number_from_text(item.get("No. of Rental Flats")),
            }
        )

    with HOS_JSON.open("r", encoding="utf-8") as f:
        hos = json.load(f)
    for item in hos:
        rows.append(
            {
                "land_use": "HOS",
                "district_norm": normalize_district(extract_en(item.get("District Name"))),
                "lat": parse_float(item.get("Estate Map Latitude")),
                "lon": parse_float(item.get("Estate Map Longitude")),
                "units_proxy": parse_number_from_text(item.get("No. of Flats")),
            }
        )

    with SHOPPING_JSON.open("r", encoding="utf-8") as f:
        shopping = json.load(f)
    for item in shopping:
        rows.append(
            {
                "land_use": "Shopping",
                "district_norm": normalize_district(extract_en(item.get("District Name"))),
                "lat": parse_float(item.get("Estate Map Latitude")),
                "lon": parse_float(item.get("Estate Map Longitude")),
                "units_proxy": 250.0,
            }
        )

    with FLATTED_FACTORY_JSON.open("r", encoding="utf-8") as f:
        factory = json.load(f)
    for item in factory:
        rows.append(
            {
                "land_use": "Factory",
                "district_norm": normalize_district(extract_en(item.get("District Name"))),
                "lat": parse_float(item.get("Estate Map Latitude")),
                "lon": parse_float(item.get("Estate Map Longitude")),
                "units_proxy": parse_number_from_text(item.get("No. of Units")),
            }
        )

    with PRIVATE_GEOJSON.open("r", encoding="utf-8") as f:
        private_geo = json.load(f)
    for feat in private_geo.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        lon = coords[0] if isinstance(coords, list) and len(coords) >= 2 else None
        lat = coords[1] if isinstance(coords, list) and len(coords) >= 2 else None
        address = extract_en(props.get("ADDR_OF_BUILDING_IN_ENG"))
        district_guess = ""
        for name in DISTRICT_DISPLAY.values():
            if name.upper() in address.upper():
                district_guess = normalize_district(name)
                break

        rows.append(
            {
                "land_use": "Private",
                "district_norm": district_guess,
                "lat": parse_float(lat),
                "lon": parse_float(lon),
                "units_proxy": 400.0,
            }
        )

    df = pd.DataFrame(rows)
    df = df[df["lat"].between(22.05, 22.65) & df["lon"].between(113.75, 114.55)].copy()
    df["district_norm"] = df["district_norm"].fillna("")
    df["district"] = df["district_norm"].map(district_display)
    df["units_proxy"] = pd.to_numeric(df["units_proxy"], errors="coerce").fillna(0.0)
    return df


def build_nearest_distance_table(housing_df: pd.DataFrame, recycle_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for district, h_sub in housing_df.groupby("district_norm"):
        if district == "":
            continue

        r_sub = recycle_df[recycle_df["district_norm"] == district]
        if r_sub.empty:
            out = h_sub[["district_norm", "land_use", "units_proxy", "lat", "lon"]].copy()
            out["nearest_bin_km"] = np.nan
            rows.append(out)
            continue

        h_lat = h_sub["lat"].to_numpy(dtype=float)
        h_lon = h_sub["lon"].to_numpy(dtype=float)
        r_lat = r_sub["lat"].to_numpy(dtype=float)
        r_lon = r_sub["lon"].to_numpy(dtype=float)

        dist_matrix = haversine_km(h_lat, h_lon, r_lat, r_lon)
        nearest = dist_matrix.min(axis=1)

        out = h_sub[["district_norm", "land_use", "units_proxy", "lat", "lon"]].copy()
        out["nearest_bin_km"] = nearest
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["district_norm", "land_use", "units_proxy", "lat", "lon", "nearest_bin_km"])
    return pd.concat(rows, ignore_index=True)


@st.cache_data(show_spinner=False)
def build_district_table(
    people_per_unit: float = 2.8,
    private_people_per_site: float = 420.0,
    per_person_recyclables_liters_day: float = 1.1,
    target_distance_km: float = 0.8,
    bin_volume_multiplier: float = 0.75,
    smart_bin_multiplier: float = 1.00,
    station_multiplier: float = 0.90,
    street_bin_multiplier: float = 0.75,
    private_bin_multiplier: float = 0.70,
    material_stream_capacity_bonus: float = 0.02,
    distribution_efficiency: float = 0.85,
    clustering_penalty_strength: float = 0.25,
    demand_uplift_factor: float = 1.30,
    capacity_utilization_factor: float = 0.55,
    normalize_baseline_pressure: bool = True,
    baseline_pressure_target: float = 1.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recycle_df = load_recycle_points().copy()
    housing_df = load_housing_points()

    legend_key = recycle_df["legend"].fillna("").astype(str).str.lower()
    recycle_df["legend_capacity_multiplier"] = np.select(
        [
            legend_key.str.contains("smart bin", regex=False),
            legend_key.str.contains("station", regex=False) | legend_key.str.contains("store", regex=False),
            legend_key.str.contains("street corner", regex=False) | legend_key.str.contains("spot", regex=False),
            legend_key.str.contains("private collection", regex=False),
        ],
        [
            float(smart_bin_multiplier),
            float(station_multiplier),
            float(street_bin_multiplier),
            float(private_bin_multiplier),
        ],
        default=1.0,
    )
    recycle_df["material_bonus_multiplier"] = (
        1.0
        + (recycle_df["material_count"].clip(lower=1) - 1.0)
        * float(material_stream_capacity_bonus)
    )
    recycle_df["adjusted_volume_m3_day"] = (
        recycle_df["estimated_volume_m3_day"]
        * float(bin_volume_multiplier)
        * recycle_df["legend_capacity_multiplier"]
        * recycle_df["material_bonus_multiplier"]
    )
    nearest_df = build_nearest_distance_table(housing_df, recycle_df)

    housing_agg = (
        housing_df.groupby("district_norm", dropna=False)
        .agg(
            housing_sites=("district_norm", "count"),
            units_proxy_total=("units_proxy", "sum"),
            land_use_types=("land_use", "nunique"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
        .reset_index()
    )

    land_mix = (
        housing_df.groupby(["district_norm", "land_use"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    land_div = (
        land_mix.groupby("district_norm", dropna=False)["count"]
        .apply(shannon_norm)
        .reset_index(name="land_use_diversity")
    )

    private_sites = (
        housing_df[housing_df["land_use"] == "Private"]
        .groupby("district_norm", dropna=False)
        .size()
        .reset_index(name="private_sites")
    )

    recycle_agg = (
        recycle_df.groupby("district_norm", dropna=False)
        .agg(
            recycle_bins=("cp_id", "count"),
            capacity_m3_day_raw_base=("estimated_volume_m3_day", "sum"),
            capacity_m3_day_adjusted=("adjusted_volume_m3_day", "sum"),
            bin_material_diversity=("material_count", "mean"),
            paper_bins=("has_paper", "sum"),
            plastic_bins=("has_plastic", "sum"),
            metal_bins=("has_metal", "sum"),
            plastic_bottle_bins=("has_plastic_bottles", "sum"),
            glass_bottle_bins=("has_glass_bottles", "sum"),
            food_waste_bins=("has_food_waste", "sum"),
            lat_recycle=("lat", "mean"),
            lon_recycle=("lon", "mean"),
        )
        .reset_index()
    )

    material_mix = recycle_df.explode("materials").copy()
    material_mix = material_mix[material_mix["materials"].notna() & material_mix["materials"].ne("")]
    material_count = (
        material_mix.groupby(["district_norm", "materials"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    material_div = (
        material_count.groupby("district_norm", dropna=False)["count"]
        .apply(shannon_norm)
        .reset_index(name="recycle_material_diversity")
    )
    material_types = (
        material_count.groupby("district_norm", dropna=False)
        .agg(material_types=("materials", "nunique"))
        .reset_index()
    )

    nearest_agg = (
        nearest_df.groupby("district_norm", dropna=False)
        .agg(
            mean_nearest_bin_km=("nearest_bin_km", "mean"),
            median_nearest_bin_km=("nearest_bin_km", "median"),
            housing_records=("district_norm", "count"),
        )
        .reset_index()
    )

    coverage_df = nearest_df.copy()
    coverage_df["covered"] = coverage_df["nearest_bin_km"].fillna(1e9) <= target_distance_km
    coverage = (
        coverage_df.groupby("district_norm", dropna=False)
        .agg(coverage_rate=("covered", "mean"))
        .reset_index()
    )

    spatial_input = pd.concat(
        [
            housing_df[["district_norm", "lat", "lon"]],
            recycle_df[["district_norm", "lat", "lon"]],
        ],
        ignore_index=True,
    )
    area_proxy = (
        spatial_input.groupby("district_norm", dropna=False)[["lat", "lon"]]
        .apply(district_area_proxy_km2)
        .reset_index(name="area_proxy_km2")
    )

    district_df = housing_agg.copy()
    for extra in [
        private_sites,
        recycle_agg,
        land_div,
        material_div,
        material_types,
        nearest_agg,
        coverage,
        area_proxy,
    ]:
        district_df = district_df.merge(extra, on="district_norm", how="left")

    district_df = district_df.fillna(0.0)

    district_df["estimated_population"] = (
        district_df["units_proxy_total"] * people_per_unit
        + district_df["private_sites"] * private_people_per_site
    )
    district_df["population_density_proxy"] = (
        district_df["estimated_population"] / district_df["area_proxy_km2"].replace(0, np.nan)
    ).fillna(0.0)

    district_df["required_volume_m3_day"] = (
        district_df["estimated_population"] * per_person_recyclables_liters_day / 1000.0
    ) * float(demand_uplift_factor)

    district_df["distribution_density"] = (
        district_df["recycle_bins"] / district_df["area_proxy_km2"].replace(0, np.nan)
    ).fillna(0.0)
    district_df["distribution_factor"] = (
        float(distribution_efficiency)
        / (1.0 + float(clustering_penalty_strength) * district_df["distribution_density"])
    ).clip(lower=0.20, upper=1.80)

    district_df["capacity_m3_day_pre_utilization"] = (
        district_df["capacity_m3_day_adjusted"] * district_df["distribution_factor"]
    )
    district_df["capacity_m3_day"] = (
        district_df["capacity_m3_day_pre_utilization"] * float(capacity_utilization_factor)
    )

    territory_required = float(district_df["required_volume_m3_day"].sum())
    territory_capacity = float(district_df["capacity_m3_day"].sum())
    territory_pressure_pre = territory_required / territory_capacity if territory_capacity > 0 else 0.0

    pressure_normalization_factor = 1.0
    if bool(normalize_baseline_pressure) and territory_pressure_pre > 0:
        pressure_normalization_factor = float(baseline_pressure_target) / territory_pressure_pre
        pressure_normalization_factor = float(np.clip(pressure_normalization_factor, 0.4, 4.0))
        district_df["required_volume_m3_day"] = (
            district_df["required_volume_m3_day"] * pressure_normalization_factor
        )

    territory_pressure_post = (
        float(district_df["required_volume_m3_day"].sum()) / max(float(district_df["capacity_m3_day"].sum()), 1e-9)
    )
    district_df["territory_pressure_pre_normalization"] = territory_pressure_pre
    district_df["territory_pressure_post_normalization"] = territory_pressure_post
    district_df["pressure_normalization_factor"] = pressure_normalization_factor

    pressure_raw = district_df["required_volume_m3_day"] / district_df["capacity_m3_day"].replace(0, np.nan)
    district_df["pressure_ratio"] = pressure_raw.fillna(0.0)
    district_df.loc[
        (district_df["capacity_m3_day"] <= 0) & (district_df["required_volume_m3_day"] > 0),
        "pressure_ratio",
    ] = 3.0

    district_df["bins_per_10k_pop"] = (
        district_df["recycle_bins"] / district_df["estimated_population"].replace(0, np.nan) * 10000.0
    ).fillna(0.0)

    district_df["diversity_gap"] = district_df["recycle_material_diversity"] - district_df["land_use_diversity"]

    access_score = (1.0 - district_df["mean_nearest_bin_km"] / max(target_distance_km, 0.05)).clip(0.0, 1.0)
    capacity_score = (1.0 - (district_df["pressure_ratio"] - 1.0).abs() / 1.5).clip(0.0, 1.0)
    diversity_score = (1.0 - district_df["diversity_gap"].abs()).clip(0.0, 1.0)

    district_df["effectiveness_score"] = (100.0 * (0.45 * access_score + 0.35 * capacity_score + 0.20 * diversity_score)).round(1)

    density_norm = (district_df["population_density_proxy"] / district_df["population_density_proxy"].max()).fillna(0.0).clip(0.0, 1.0)
    district_df["recycling_possibility_score"] = (
        100.0 * (0.40 * density_norm + 0.35 * (1.0 - access_score) + 0.25 * (1.0 - diversity_score))
    ).round(1)

    bins_75 = district_df["bins_per_10k_pop"].quantile(0.75)
    district_df["bin_balance_status"] = "Balanced"
    district_df.loc[
        (district_df["pressure_ratio"] > 1.10) | (district_df["coverage_rate"] < 0.60),
        "bin_balance_status",
    ] = "Too Few"
    district_df.loc[
        (district_df["pressure_ratio"] < 0.55)
        & (district_df["coverage_rate"] > 0.90)
        & (district_df["bins_per_10k_pop"] >= bins_75),
        "bin_balance_status",
    ] = "Too Many"

    district_df["district"] = district_df["district_norm"].map(district_display)

    # Blend centroids from housing and recycle points for map display.
    district_df["lat"] = np.where(
        district_df["lat"].eq(0),
        district_df["lat_recycle"],
        district_df["lat"],
    )
    district_df["lon"] = np.where(
        district_df["lon"].eq(0),
        district_df["lon_recycle"],
        district_df["lon"],
    )

    district_df = district_df.sort_values("effectiveness_score", ascending=False).reset_index(drop=True)

    return district_df, recycle_df, housing_df, nearest_df


def add_sklearn_point_estimates(
    district_df: pd.DataFrame,
    added_point_capacity_m3_day: float,
    target_pressure: float = 1.0,
    demand_multiplier: float = 1.0,
    safety_buffer_ratio: float = 0.0,
    pressure_boost_factor: float = 0.0,
    coverage_penalty_factor: float = 0.0,
    ml_weight: float = 0.5,
    min_high_pressure_points: int = 0,
    max_points_per_district: int = 5000,
) -> pd.DataFrame:
    out = district_df.copy()

    point_capacity = max(0.1, float(added_point_capacity_m3_day))
    target = max(0.4, float(target_pressure))
    demand_mult = max(0.5, float(demand_multiplier))
    safety = max(0.0, float(safety_buffer_ratio))
    pressure_boost = max(0.0, float(pressure_boost_factor))
    coverage_penalty = max(0.0, float(coverage_penalty_factor))
    ml_blend = float(np.clip(ml_weight, 0.0, 1.0))
    min_hp = max(0, int(min_high_pressure_points))
    max_cap = max(0, int(max_points_per_district))

    out["required_volume_for_plan_m3_day"] = (
        out["required_volume_m3_day"] * demand_mult * (1.0 + safety)
    )
    out["target_capacity_m3_day"] = out["required_volume_for_plan_m3_day"] / target
    out["capacity_shortage_m3_day"] = (
        out["target_capacity_m3_day"] - out["capacity_m3_day"]
    ).clip(lower=0.0)
    out["new_points_needed"] = np.ceil(
        out["capacity_shortage_m3_day"] / point_capacity
    ).astype(int)

    pressure_gap = ((out["pressure_ratio"] / target) - 1.0).clip(lower=0.0)
    coverage_gap = (1.0 - out["coverage_rate"]).clip(lower=0.0)
    out["pressure_boost_points"] = np.ceil(
        pressure_gap * pressure_boost * 10.0
    ).astype(int)
    out["coverage_penalty_points"] = np.ceil(
        coverage_gap * coverage_penalty * 8.0
    ).astype(int)
    out["new_points_needed_rule"] = (
        out["new_points_needed"] + out["pressure_boost_points"] + out["coverage_penalty_points"]
    ).astype(int)

    # Train on district features to provide a smoothed, ML-assisted estimate.
    feature_cols = [
        "estimated_population",
        "population_density_proxy",
        "recycle_bins",
        "capacity_m3_day",
        "required_volume_m3_day",
        "required_volume_for_plan_m3_day",
        "target_capacity_m3_day",
        "capacity_shortage_m3_day",
        "pressure_ratio",
        "coverage_rate",
        "mean_nearest_bin_km",
        "land_use_diversity",
        "recycle_material_diversity",
        "pressure_boost_points",
        "coverage_penalty_points",
    ]

    X = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = out["new_points_needed_rule"].astype(float)

    model = LinearRegression()
    model.fit(X, y)
    ml_pred = np.clip(model.predict(X), a_min=0.0, a_max=None)

    out["new_points_needed_ml"] = np.ceil(ml_pred).astype(int)
    blended = np.ceil(
        (1.0 - ml_blend) * out["new_points_needed_rule"] + ml_blend * out["new_points_needed_ml"]
    ).astype(int)
    out["new_points_needed_final"] = blended

    hp_mask = out["pressure_ratio"] > target
    out.loc[~hp_mask, "new_points_needed_final"] = 0
    out.loc[hp_mask, "new_points_needed_final"] = np.maximum(
        out.loc[hp_mask, "new_points_needed_final"],
        min_hp,
    )
    out["new_points_needed_final"] = out["new_points_needed_final"].clip(lower=0, upper=max_cap)

    out["projected_capacity_after_add_m3_day"] = (
        out["capacity_m3_day"] + out["new_points_needed_final"] * point_capacity
    )
    out["projected_pressure_after_add"] = (
        out["required_volume_for_plan_m3_day"] / out["projected_capacity_after_add_m3_day"].replace(0, np.nan)
    ).fillna(0.0)

    return out


def render_pressure_map(district_df: pd.DataFrame, color_mode: str) -> None:
    map_df = district_df[
        (district_df["lat"] > 0)
        & (district_df["lon"] > 0)
        & (district_df["recycle_bins"] > 0)
    ].copy()
    if map_df.empty:
        st.info("No district points available for map.")
        return

    if color_mode == "Pressure":
        vals = map_df["pressure_ratio"]
    else:
        vals = 1.0 - (map_df["effectiveness_score"] / 100.0)

    # Use absolute 0..1 scale so colors do not depend on other districts.
    scale = vals.clip(lower=0.0, upper=1.0)

    map_df["r"] = (255 * scale).astype(int)
    map_df["g"] = (200 * (1 - scale)).astype(int)
    map_df["b"] = (70 * (1 - scale)).astype(int)
    map_df["radius"] = (900 + 3600 * scale).astype(float)
    map_df["is_high_pressure"] = map_df["pressure_ratio"] > 1.0
    map_df["line_r"] = np.where(map_df["is_high_pressure"], 220, 40)
    map_df["line_g"] = np.where(map_df["is_high_pressure"], 40, 40)
    map_df["line_b"] = np.where(map_df["is_high_pressure"], 40, 40)
    map_df["line_w"] = np.where(map_df["is_high_pressure"], 2.5, 1.0)

    deck = pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(latitude=22.33, longitude=114.15, zoom=9.5, pitch=0),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_fill_color="[r, g, b, 190]",
                get_radius="radius",
                pickable=True,
                stroked=True,
                get_line_color="[line_r, line_g, line_b, 240]",
                line_width_min_pixels=2,
                get_line_width="line_w",
            ),
            pdk.Layer(
                "TextLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_text="district",
                get_size=12,
                get_color=[30, 30, 30, 220],
                get_alignment_baseline="bottom",
            ),
        ],
        tooltip=cast(Any, {
            "html": "<b>{district}</b><br/>Pressure: {pressure_ratio}<br/>Effectiveness: {effectiveness_score}<br/>Coverage: {coverage_rate}<br/>Balance: {bin_balance_status}",
            "style": {"color": "white", "backgroundColor": "#1e1e1e"},
        }),
    )
    st.pydeck_chart(deck, width="stretch")


def render_material_district_map(
    district_df: pd.DataFrame,
    value_col: str,
    map_title: str,
    fill_color: list[int],
) -> None:
    st.markdown(f"**{map_title}**")

    map_df = district_df[(district_df["lat"] > 0) & (district_df["lon"] > 0)].copy()
    map_df = map_df[map_df[value_col] > 0].copy()
    if map_df.empty:
        st.info("No matching recycle points for this material type.")
        return

    vmax = float(map_df[value_col].max()) if float(map_df[value_col].max()) > 0 else 1.0
    scale = (map_df[value_col] / vmax).clip(0.0, 1.0)
    map_df["radius"] = (700 + 2800 * scale).astype(float)

    map_df["material_share"] = (
        map_df[value_col] / map_df["recycle_bins"].replace(0, np.nan)
    ).fillna(0.0)
    map_df["material_pressure_ratio"] = (
        map_df["pressure_ratio"] / map_df["material_share"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    p95 = float(map_df["pressure_ratio"].quantile(0.95)) if not map_df.empty else 1.0
    p95 = p95 if p95 > 0 else 1.0
    pnorm = (map_df["pressure_ratio"].clip(lower=0, upper=p95) / p95).clip(0.0, 1.0)

    map_df["fill_r"] = int(fill_color[0])
    map_df["fill_g"] = int(fill_color[1])
    map_df["fill_b"] = int(fill_color[2])
    map_df["fill_a"] = (100 + 120 * pnorm).astype(int)
    map_df["line_r"] = (130 + 120 * pnorm).astype(int)
    map_df["line_g"] = (170 * (1 - pnorm)).astype(int)
    map_df["line_b"] = (40 * (1 - pnorm)).astype(int)

    deck = pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(latitude=22.33, longitude=114.15, zoom=9.6, pitch=0),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_fill_color="[fill_r, fill_g, fill_b, fill_a]",
                get_radius="radius",
                pickable=True,
                stroked=True,
                get_line_color="[line_r, line_g, line_b, 230]",
                line_width_min_pixels=1,
            )
        ],
        tooltip=cast(Any, {
            "html": "<b>{district}</b><br/>Type bins: {" + value_col + "}<br/>Total bins: {recycle_bins}<br/>Pressure ratio: {pressure_ratio}<br/>Material share: {material_share}<br/>Material pressure: {material_pressure_ratio}",
            "style": {"color": "white", "backgroundColor": "#1e1e1e"},
        }),
    )
    st.pydeck_chart(deck, width="stretch")


def render_housing_distance_map(housing_view_df: pd.DataFrame, recycle_df: pd.DataFrame) -> None:
    house_map = housing_view_df[(housing_view_df["lat"] > 0) & (housing_view_df["lon"] > 0)].copy()
    bins_map = recycle_df[(recycle_df["lat"] > 0) & (recycle_df["lon"] > 0)].copy()

    if house_map.empty:
        st.info("No housing points available for threshold map.")
        return

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=house_map,
            get_position="[lon, lat]",
            get_fill_color="[r, g, b, 200]",
            get_radius=120,
            pickable=True,
            stroked=False,
        )
    ]

    if not bins_map.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=bins_map,
                get_position="[lon, lat]",
                get_fill_color=[33, 150, 243, 110],
                get_radius=65,
                pickable=False,
                stroked=False,
            )
        )

    deck = pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(latitude=22.33, longitude=114.15, zoom=10, pitch=0),
        layers=layers,
        tooltip=cast(Any, {
            "html": "<b>{district}</b><br/>Land use: {land_use}<br/>Nearest bin: {nearest_bin_km} km<br/>Threshold status: {distance_status}",
            "style": {"color": "white", "backgroundColor": "#1e1e1e"},
        }),
    )
    st.pydeck_chart(deck, width="stretch")


def scalar_haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return float(haversine_km(np.array([lat1]), np.array([lon1]), np.array([lat2]), np.array([lon2]))[0, 0])


def build_recycle_centres(recycle_df: pd.DataFrame, district_df: pd.DataFrame) -> pd.DataFrame:
    centre_mask = recycle_df["legend"].fillna("").astype(str).str.lower().str.contains("station|store|smart", regex=True)
    centres = recycle_df[centre_mask].copy()

    if not centres.empty:
        out = centres[["district", "lat", "lon", "legend", "cp_id"]].copy()
        out["centre_name"] = "Centre " + out["cp_id"].astype(str)
        out["centre_type"] = out["legend"].fillna("Recycling Centre")
        return out[["centre_name", "centre_type", "district", "lat", "lon"]]

    # Fallback when no explicit centres are tagged in source data.
    fallback = district_df[(district_df["lat"] > 0) & (district_df["lon"] > 0)].copy()
    fallback = fallback.sort_values("recycle_bins", ascending=False).head(6)
    fallback["centre_name"] = "District Hub " + fallback["district"].astype(str)
    fallback["centre_type"] = "Synthetic centre (district centroid)"
    return fallback[["centre_name", "centre_type", "district", "lat", "lon"]]


def nearest_neighbour_bin_route(bin_df: pd.DataFrame, start_lat: float, start_lon: float) -> list[int]:
    if bin_df.empty:
        return []

    remaining = set(bin_df.index.tolist())
    route: list[int] = []
    current_lat, current_lon = float(start_lat), float(start_lon)

    while remaining:
        best_idx = min(
            remaining,
            key=lambda idx: scalar_haversine_km(
                current_lat,
                current_lon,
                parse_float(bin_df.at[idx, "lat"]),
                parse_float(bin_df.at[idx, "lon"]),
            ),
        )
        route.append(best_idx)
        current_lat = parse_float(bin_df.at[best_idx, "lat"])
        current_lon = parse_float(bin_df.at[best_idx, "lon"])
        remaining.remove(best_idx)

    return route


def weighted_priority_bin_route(
    bin_df: pd.DataFrame,
    start_lat: float,
    start_lon: float,
    distance_weight: float,
    volume_weight: float,
    pressure_weight: float,
) -> list[int]:
    if bin_df.empty:
        return []

    remaining = set(bin_df.index.tolist())
    route: list[int] = []
    current_lat, current_lon = float(start_lat), float(start_lon)

    vmin = float(bin_df["estimated_volume_m3_day"].min())
    vmax = float(bin_df["estimated_volume_m3_day"].max())
    pmin = float(bin_df["district_pressure_ratio"].min())
    pmax = float(bin_df["district_pressure_ratio"].max())

    while remaining:
        dists = {
            idx: scalar_haversine_km(
                current_lat,
                current_lon,
                parse_float(bin_df.at[idx, "lat"]),
                parse_float(bin_df.at[idx, "lon"]),
            )
            for idx in remaining
        }
        dmin = min(dists.values()) if dists else 0.0
        dmax = max(dists.values()) if dists else 1.0

        def norm(val: float, lo: float, hi: float) -> float:
            if hi <= lo + 1e-9:
                return 0.0
            return (val - lo) / (hi - lo)

        best_idx = None
        best_score = None
        for idx in remaining:
            d_n = norm(dists[idx], dmin, dmax)
            v_n = norm(parse_float(bin_df.at[idx, "estimated_volume_m3_day"]), vmin, vmax)
            p_n = norm(parse_float(bin_df.at[idx, "district_pressure_ratio"]), pmin, pmax)

            # Lower score is better: penalize distance, reward volume and pressure need.
            score = float(distance_weight) * d_n - float(volume_weight) * v_n - float(pressure_weight) * p_n
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        route.append(best_idx)
        current_lat = parse_float(bin_df.at[best_idx, "lat"])
        current_lon = parse_float(bin_df.at[best_idx, "lon"])
        remaining.remove(best_idx)

    return route


def render_truck_path_page(recycle_df: pd.DataFrame, district_df: pd.DataFrame) -> None:
    st.subheader("Garbage Truck Path Prediction")
    st.caption(
        "Simulates a truck route that visits recycle bins in sequence and returns to the nearest recycle centre after collection."
    )
    st.info(
        "Routing algorithm here is a greedy heuristic (nearest-neighbour or weighted-priority), not an exact Travelling Salesman optimum."
    )

    centres_df = build_recycle_centres(recycle_df, district_df)
    if centres_df.empty:
        st.warning("No recycle centres could be derived from data.")
        return

    district_options = sorted([d for d in recycle_df["district"].dropna().unique().tolist() if str(d).strip() != ""])
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_district = st.selectbox(
            "District for trip",
            options=district_options,
            index=0 if district_options else None,
            key="truck_route_district",
        )
    with c2:
        max_stops = st.slider(
            "Max bins visited in one trip",
            min_value=5,
            max_value=200,
            value=40,
            step=5,
            key="truck_route_max_stops",
        )
    with c3:
        demand_first = st.selectbox(
            "Bin priority",
            options=["Nearest-neighbour only", "Volume/pressure prioritization"],
            index=0,
            key="truck_route_priority",
        )

    st.markdown("Focus trash types (tickboxes)")
    t1, t2, t3 = st.columns(3)
    with t1:
        focus_paper = st.checkbox("Paper", value=True, key="truck_focus_paper")
        focus_metal = st.checkbox("Metal", value=True, key="truck_focus_metal")
    with t2:
        focus_plastic = st.checkbox("Plastic", value=True, key="truck_focus_plastic")
        focus_plastic_bottle = st.checkbox("Plastic Bottles", value=True, key="truck_focus_plastic_bottle")
    with t3:
        focus_glass_bottle = st.checkbox("Glass Bottles", value=True, key="truck_focus_glass_bottle")
        focus_food_waste = st.checkbox("Food Waste", value=True, key="truck_focus_food_waste")

    w1, w2, w3 = st.columns(3)
    with w1:
        distance_weight = st.slider(
            "Distance weight",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key="truck_distance_weight",
        )
    with w2:
        volume_weight = st.slider(
            "Volume priority weight",
            min_value=0.0,
            max_value=3.0,
            value=1.2,
            step=0.1,
            key="truck_volume_weight",
        )
    with w3:
        pressure_weight = st.slider(
            "Neighbourhood pressure weight",
            min_value=0.0,
            max_value=3.0,
            value=1.2,
            step=0.1,
            key="truck_pressure_weight",
        )

    district_bins = recycle_df[(recycle_df["district"] == selected_district) & (recycle_df["lat"] > 0) & (recycle_df["lon"] > 0)].copy()
    if district_bins.empty:
        st.info("No recycle bins found in the selected district.")
        return

    material_cols: list[str] = []
    if focus_paper:
        material_cols.append("has_paper")
    if focus_plastic:
        material_cols.append("has_plastic")
    if focus_metal:
        material_cols.append("has_metal")
    if focus_plastic_bottle:
        material_cols.append("has_plastic_bottles")
    if focus_glass_bottle:
        material_cols.append("has_glass_bottles")
    if focus_food_waste:
        material_cols.append("has_food_waste")

    if material_cols:
        mask = district_bins[material_cols].sum(axis=1) > 0
        district_bins = district_bins[mask].copy()
    if district_bins.empty:
        st.warning("No bins match the selected trash-type focus in this district.")
        return

    pressure_lookup = district_df.set_index("district_norm")["pressure_ratio"].to_dict()
    district_bins["district_pressure_ratio"] = district_bins["district_norm"].map(pressure_lookup).fillna(1.0)
    district_bins = district_bins.sort_values(["estimated_volume_m3_day", "material_count"], ascending=[False, False]).head(int(max_stops)).copy()

    district_lat = float(district_bins["lat"].mean())
    district_lon = float(district_bins["lon"].mean())

    centres_df = centres_df.copy()
    centres_df["dist_to_district_km"] = centres_df.apply(
        lambda r: scalar_haversine_km(district_lat, district_lon, float(r["lat"]), float(r["lon"])),
        axis=1,
    )
    start_centre = centres_df.sort_values("dist_to_district_km", ascending=True).iloc[0]

    if demand_first == "Volume/pressure prioritization":
        route_order = weighted_priority_bin_route(
            district_bins,
            float(start_centre["lat"]),
            float(start_centre["lon"]),
            float(distance_weight),
            float(volume_weight),
            float(pressure_weight),
        )
    else:
        route_order = nearest_neighbour_bin_route(
            district_bins,
            float(start_centre["lat"]),
            float(start_centre["lon"]),
        )
    route_bins = district_bins.loc[route_order].copy()
    if route_bins.empty:
        st.info("No valid route could be generated.")
        return

    last_lat = parse_float(route_bins.iloc[-1]["lat"])
    last_lon = parse_float(route_bins.iloc[-1]["lon"])
    centres_df["dist_from_last_km"] = centres_df.apply(
        lambda r: scalar_haversine_km(last_lat, last_lon, float(r["lat"]), float(r["lon"])),
        axis=1,
    )
    return_centre = centres_df.sort_values("dist_from_last_km", ascending=True).iloc[0]

    segments: list[dict[str, Any]] = []
    seq_rows: list[dict[str, Any]] = []

    prev_name = str(start_centre["centre_name"])
    prev_lat = parse_float(start_centre["lat"])
    prev_lon = parse_float(start_centre["lon"])
    cumulative = 0.0

    for i, row in enumerate(route_bins.itertuples(index=False), start=1):
        curr_lat = parse_float(row.lat)
        curr_lon = parse_float(row.lon)
        leg = scalar_haversine_km(prev_lat, prev_lon, curr_lat, curr_lon)
        cumulative += leg
        curr_name = f"Bin {getattr(row, 'cp_id')}"

        segments.append(
            {
                "path": [[prev_lon, prev_lat], [curr_lon, curr_lat]],
                "name": f"{prev_name} -> {curr_name}",
                "leg_km": leg,
            }
        )
        seq_rows.append(
            {
                "stop_no": i,
                "from": prev_name,
                "to": curr_name,
                "leg_km": leg,
                "cumulative_km": cumulative,
                "district": getattr(row, "district"),
                "materials": getattr(row, "waste_type"),
                "est_volume_m3_day": getattr(row, "estimated_volume_m3_day"),
                "district_pressure": getattr(row, "district_pressure_ratio"),
            }
        )

        prev_name = curr_name
        prev_lat = curr_lat
        prev_lon = curr_lon

    return_leg = scalar_haversine_km(prev_lat, prev_lon, float(return_centre["lat"]), float(return_centre["lon"]))
    cumulative += return_leg
    segments.append(
        {
            "path": [[prev_lon, prev_lat], [float(return_centre["lon"]), float(return_centre["lat"])]],
            "name": f"{prev_name} -> {return_centre['centre_name']}",
            "leg_km": return_leg,
        }
    )
    seq_rows.append(
        {
            "stop_no": len(route_bins) + 1,
            "from": prev_name,
            "to": str(return_centre["centre_name"]),
            "leg_km": return_leg,
            "cumulative_km": cumulative,
            "district": selected_district,
            "materials": "Return to centre",
            "est_volume_m3_day": 0.0,
            "district_pressure": np.nan,
        }
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Bins visited", f"{len(route_bins):,}")
    k2.metric("Total path distance", f"{cumulative:.2f} km")
    k3.metric("Start centre", str(start_centre["centre_name"]))
    k4.metric("Return centre", str(return_centre["centre_name"]))

    route_map_bins = route_bins[["lat", "lon", "district", "cp_id"]].copy()
    route_map_bins["point_type"] = "Bin"
    route_map_bins["r"] = 230
    route_map_bins["g"] = 126
    route_map_bins["b"] = 34

    route_map_centres = pd.DataFrame(
        [
            {
                "lat": float(start_centre["lat"]),
                "lon": float(start_centre["lon"]),
                "district": str(start_centre["district"]),
                "cp_id": str(start_centre["centre_name"]),
                "point_type": "Start centre",
                "r": 52,
                "g": 152,
                "b": 219,
            },
            {
                "lat": float(return_centre["lat"]),
                "lon": float(return_centre["lon"]),
                "district": str(return_centre["district"]),
                "cp_id": str(return_centre["centre_name"]),
                "point_type": "Return centre",
                "r": 46,
                "g": 204,
                "b": 113,
            },
        ]
    )

    deck = pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(latitude=district_lat, longitude=district_lon, zoom=11.5, pitch=35),
        layers=[
            pdk.Layer(
                "PathLayer",
                data=pd.DataFrame(segments),
                get_path="path",
                get_width=6,
                get_color=[231, 76, 60, 220],
                pickable=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=route_map_bins,
                get_position="[lon, lat]",
                get_fill_color="[r, g, b, 220]",
                get_radius=70,
                pickable=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=route_map_centres,
                get_position="[lon, lat]",
                get_fill_color="[r, g, b, 230]",
                get_radius=130,
                pickable=True,
            ),
        ],
        tooltip=cast(Any, {
            "html": "<b>{point_type}</b><br/>{cp_id}<br/>District: {district}",
            "style": {"color": "white", "backgroundColor": "#1e1e1e"},
        }),
    )
    st.pydeck_chart(deck, width="stretch")

    route_table = pd.DataFrame(seq_rows)
    st.dataframe(route_table, width="stretch", hide_index=True)


def render_sklearn_planner_page(base_district_df: pd.DataFrame, nearest_df: pd.DataFrame) -> None:
    st.subheader("SKLearn Point-Addition Planner")
    st.caption(
        "This page simulates how adding recycle points changes district pressure. "
        "Use the sliders, then click the button to run the scenario."
    )
    st.info(
        "Material sliders in this tab adjust district-level material bin counts in the model. "
        "They do not place bins at random map coordinates."
    )

    st.markdown("Housing access tracker (distance to nearest existing recycling facility)")
    access_m1, access_m2, access_m3 = st.columns(3)
    with access_m1:
        distance_threshold_m = st.slider(
            "Housing-distance threshold (meters)",
            min_value=50,
            max_value=3000,
            value=800,
            step=50,
            key="ml_distance_threshold_m",
        )

    nearest_view = nearest_df.copy()
    nearest_view = nearest_view[np.isfinite(nearest_view["nearest_bin_km"])].copy()
    nearest_view["nearest_bin_m"] = nearest_view["nearest_bin_km"] * 1000.0

    if nearest_view.empty:
        with access_m2:
            st.metric("Housing points > threshold", "0 / 0", delta="0.0%")
        with access_m3:
            st.metric("Average nearest distance", "0 m")
    else:
        far_mask = nearest_view["nearest_bin_m"] > float(distance_threshold_m)
        far_count = int(far_mask.sum())
        total_count = int(len(nearest_view))
        far_share = 100.0 * far_count / max(1, total_count)
        with access_m2:
            st.metric("Housing points > threshold", f"{far_count:,} / {total_count:,}", delta=f"{far_share:.1f}%")
        with access_m3:
            st.metric("Average nearest distance", f"{nearest_view['nearest_bin_m'].mean():.0f} m")

        far_by_district = (
            nearest_view.assign(is_far=far_mask)
            .groupby("district_norm", dropna=False)
            .agg(
                housing_points=("district_norm", "count"),
                far_points=("is_far", "sum"),
                avg_nearest_m=("nearest_bin_m", "mean"),
            )
            .reset_index()
        )
        far_by_district["district"] = far_by_district["district_norm"].map(district_display)
        far_by_district["far_share"] = (
            far_by_district["far_points"] / far_by_district["housing_points"].replace(0, np.nan)
        ).fillna(0.0)
        st.dataframe(
            far_by_district[["district", "housing_points", "far_points", "far_share", "avg_nearest_m"]]
            .sort_values(["far_share", "far_points"], ascending=[False, False])
            .head(10),
            width="stretch",
            hide_index=True,
        )

    def find_best_planner_params() -> tuple[float, int, float, float, float, float]:
        capacities = [round(x, 1) for x in np.arange(1.0, 6.01, 0.8)]
        extras = list(range(0, 701, 50))
        targets = [0.85, 0.95, 1.00, 1.10]
        demand_factors = [1.00, 1.20, 1.40, 1.60]
        impact_factors = [0.8, 1.2, 1.6, 2.0]
        ml_weights = [0.30, 0.55, 0.80]

        best_obj = None
        best_tuple = (2.0, 0, 1.0, 1.2, 1.2, 0.55)

        for cap in capacities:
            for tgt in targets:
                for demand_factor in demand_factors:
                    for impact in impact_factors:
                        for mlw in ml_weights:
                            base = add_sklearn_point_estimates(
                                base_district_df,
                                float(cap),
                                target_pressure=float(tgt),
                                demand_multiplier=float(demand_factor),
                                pressure_boost_factor=float(impact),
                                ml_weight=float(mlw),
                            ).copy()

                            hp_mask = base["pressure_ratio"] > float(tgt)
                            for extra in extras:
                                planned = base["new_points_needed_final"].copy()
                                planned.loc[hp_mask] = planned.loc[hp_mask] + int(extra)

                                projected_capacity = base["capacity_m3_day"] + planned * float(cap)
                                projected_pressure = (
                                    base["required_volume_for_plan_m3_day"] / projected_capacity.replace(0, np.nan)
                                ).fillna(0.0)

                                after_count = int((projected_pressure > float(tgt)).sum())
                                total_add = int(planned.sum())
                                avg_after = float(projected_pressure.mean())
                                obj = (after_count, total_add, avg_after)
                                if best_obj is None or obj < best_obj:
                                    best_obj = obj
                                    best_tuple = (
                                        float(cap),
                                        int(extra),
                                        float(tgt),
                                        float(demand_factor),
                                        float(impact),
                                        float(mlw),
                                    )

        return best_tuple

    auto_tune_clicked = st.button("Auto-tune Best Parameters", key="ml_auto_tune_button")
    if auto_tune_clicked:
        with st.spinner("Searching best SKLearn planner settings..."):
            best_cap, best_extra, best_target, best_demand, best_impact, best_mlw = find_best_planner_params()
        st.session_state["ml_capacity"] = best_cap
        st.session_state["ml_extra_points"] = best_extra
        st.session_state["ml_target_pressure"] = best_target
        st.session_state["ml_demand_multiplier"] = best_demand
        st.session_state["ml_pressure_boost"] = best_impact
        st.session_state["ml_ml_weight"] = best_mlw
        st.session_state["ml_run_done"] = True
        st.rerun()

    p1, p2, p3 = st.columns(3)
    with p1:
        added_point_capacity_m3_day = st.slider(
            "Capacity per new recycle point (m3/day)",
            min_value=0.5,
            max_value=12.0,
            value=2.0,
            step=0.1,
            key="ml_capacity",
        )
    with p2:
        extra_points_per_high_pressure = st.slider(
            "Extra points per high-pressure district",
            min_value=0,
            max_value=1500,
            value=0,
            step=10,
            key="ml_extra_points",
        )
    with p3:
        target_pressure = st.slider(
            "Target pressure threshold",
            min_value=0.60,
            max_value=1.20,
            value=1.00,
            step=0.05,
            key="ml_target_pressure",
        )

    q1, q2, q3 = st.columns(3)
    with q1:
        demand_multiplier = st.slider(
            "Demand multiplier (overflow / surge)",
            min_value=0.80,
            max_value=2.50,
            value=1.20,
            step=0.05,
            key="ml_demand_multiplier",
        )
    with q2:
        safety_buffer_ratio = st.slider(
            "Safety buffer on demand",
            min_value=0.00,
            max_value=0.80,
            value=0.15,
            step=0.01,
            key="ml_safety_buffer",
        )
    with q3:
        pressure_boost_factor = st.slider(
            "Pressure impact factor",
            min_value=0.00,
            max_value=4.00,
            value=1.20,
            step=0.05,
            key="ml_pressure_boost",
        )

    r1, r2, r3 = st.columns(3)
    with r1:
        coverage_penalty_factor = st.slider(
            "Coverage penalty factor",
            min_value=0.00,
            max_value=4.00,
            value=0.60,
            step=0.05,
            key="ml_coverage_penalty",
        )
    with r2:
        ml_weight = st.slider(
            "ML prediction weight",
            min_value=0.00,
            max_value=1.00,
            value=0.55,
            step=0.05,
            key="ml_ml_weight",
        )
    with r3:
        min_high_pressure_points = st.slider(
            "Minimum points for high-pressure districts",
            min_value=0,
            max_value=300,
            value=0,
            step=5,
            key="ml_min_hp_points",
        )

    s1, s2 = st.columns(2)
    with s1:
        global_added_points = st.slider(
            "Global added points budget",
            min_value=0,
            max_value=5000,
            value=0,
            step=50,
            key="ml_global_points",
        )
    with s2:
        max_points_per_district = st.slider(
            "Max points per district",
            min_value=50,
            max_value=5000,
            value=1200,
            step=50,
            key="ml_max_points_district",
        )

    run_clicked = st.button("Run SKLearn Simulation", type="primary", key="ml_run_button")
    if run_clicked:
        st.session_state["ml_run_done"] = True

    if not st.session_state.get("ml_run_done", False):
        st.info("Set parameters and click Run SKLearn Simulation to generate recommendations.")
        return

    scenario_df = add_sklearn_point_estimates(
        base_district_df,
        float(added_point_capacity_m3_day),
        target_pressure=float(target_pressure),
        demand_multiplier=float(demand_multiplier),
        safety_buffer_ratio=float(safety_buffer_ratio),
        pressure_boost_factor=float(pressure_boost_factor),
        coverage_penalty_factor=float(coverage_penalty_factor),
        ml_weight=float(ml_weight),
        min_high_pressure_points=int(min_high_pressure_points),
        max_points_per_district=int(max_points_per_district),
    ).copy()
    hp_mask = scenario_df["pressure_ratio"] > float(target_pressure)
    scenario_df["planned_added_points"] = scenario_df["new_points_needed_final"]
    scenario_df.loc[hp_mask, "planned_added_points"] = (
        scenario_df.loc[hp_mask, "planned_added_points"] + int(extra_points_per_high_pressure)
    )

    if int(global_added_points) > 0:
        alloc_df = scenario_df[hp_mask].copy()
        if not alloc_df.empty:
            weights = (
                (alloc_df["pressure_ratio"] - float(target_pressure)).clip(lower=0.0)
                + (1.0 - alloc_df["coverage_rate"]).clip(lower=0.0) * 0.5
                + 0.05
            )
            raw_alloc = (weights / weights.sum() * int(global_added_points)).round().astype(int)
            scenario_df.loc[alloc_df.index, "planned_added_points"] += raw_alloc.values

    scenario_df["planned_added_points"] = scenario_df["planned_added_points"].clip(
        lower=0,
        upper=int(max_points_per_district),
    )

    scenario_df["projected_capacity_after_plan_m3_day"] = (
        scenario_df["capacity_m3_day"] + scenario_df["planned_added_points"] * float(added_point_capacity_m3_day)
    )
    scenario_df["projected_pressure_after_plan"] = (
        scenario_df["required_volume_m3_day"] / scenario_df["projected_capacity_after_plan_m3_day"].replace(0, np.nan)
    ).fillna(0.0)

    before_count = int((scenario_df["pressure_ratio"] > float(target_pressure)).sum())
    after_count = int((scenario_df["projected_pressure_after_plan"] > float(target_pressure)).sum())
    total_add = int(scenario_df["planned_added_points"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("High-pressure districts (before)", f"{before_count:,}")
    k2.metric("High-pressure districts (after)", f"{after_count:,}", delta=f"{after_count - before_count:+d}")
    k3.metric("Total points to add", f"{total_add:,}")
    k4.metric(
        "Average pressure reduction",
        f"{(scenario_df['pressure_ratio'] - scenario_df['projected_pressure_after_plan']).mean():.3f}",
    )

    st.markdown("Scenario map (projected pressure after planned additions)")
    map_df = scenario_df.copy()
    map_df["pressure_ratio"] = map_df["projected_pressure_after_plan"]
    render_pressure_map(map_df, color_mode="Pressure")

    st.markdown("Scenario material maps (projected pressure + material-specific added bins)")
    st.caption("Each slider adds that material bin count to every district in the corresponding map.")

    def estimate_uniform_material_addition(
        scenario_map_df: pd.DataFrame,
        value_col: str,
        min_share_threshold: float,
    ) -> int:
        issue_df = scenario_map_df[(scenario_map_df["lat"] > 0) & (scenario_map_df["lon"] > 0)].copy()
        if issue_df.empty:
            return 0

        total_bins = issue_df["recycle_bins"].astype(float).clip(lower=0.0)
        current_bins = issue_df[value_col].astype(float).clip(lower=0.0)
        current_share = (current_bins / total_bins.replace(0, np.nan)).fillna(0.0)
        target_share = issue_df["projected_pressure_after_plan"].astype(float).clip(lower=float(min_share_threshold), upper=0.92)
        target_share = pd.Series(
            np.maximum(target_share.to_numpy(dtype=float), current_share.to_numpy(dtype=float)),
            index=issue_df.index,
        )

        numerator = target_share * total_bins - current_bins
        denominator = pd.Series(1.0 - target_share.to_numpy(dtype=float), index=issue_df.index)
        needed = (numerator / denominator.replace(0, np.nan)).clip(lower=0.0)
        needed = needed.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        needed = needed.apply(np.ceil)

        high_need = issue_df[
            (issue_df["projected_pressure_after_plan"] > float(target_pressure))
            | (current_share < float(min_share_threshold))
        ].copy()
        if high_need.empty:
            return 0

        needed_high = needed.loc[high_need.index]
        recommendation = int(np.ceil(float(needed_high.quantile(0.75))))
        return int(np.clip(recommendation, 0, 1000))

    perfect_material_clicked = st.button(
        "Find Perfect Material Parameters",
        key="ml_find_perfect_material_params",
    )
    if perfect_material_clicked:
        material_thresholds = {
            "paper_bins": 0.18,
            "plastic_bins": 0.18,
            "metal_bins": 0.12,
            "plastic_bottle_bins": 0.10,
            "glass_bottle_bins": 0.08,
            "food_waste_bins": 0.08,
        }
        slider_keys = {
            "paper_bins": "ml_add_paper_per_district",
            "plastic_bins": "ml_add_plastic_per_district",
            "metal_bins": "ml_add_metal_per_district",
            "plastic_bottle_bins": "ml_add_plastic_bottle_per_district",
            "glass_bottle_bins": "ml_add_glass_bottle_per_district",
            "food_waste_bins": "ml_add_food_waste_per_district",
        }
        for material_col, threshold in material_thresholds.items():
            st.session_state[slider_keys[material_col]] = estimate_uniform_material_addition(
                map_df,
                material_col,
                threshold,
            )
        st.rerun()

    mctrl1, mctrl2, mctrl3 = st.columns(3)
    with mctrl1:
        add_paper_per_district = st.slider(
            "Add paper bins / district",
            min_value=0,
            max_value=1000,
            value=0,
            step=5,
            key="ml_add_paper_per_district",
        )
        add_metal_per_district = st.slider(
            "Add metal bins / district",
            min_value=0,
            max_value=1000,
            value=0,
            step=5,
            key="ml_add_metal_per_district",
        )
    with mctrl2:
        add_plastic_per_district = st.slider(
            "Add plastic bins / district",
            min_value=0,
            max_value=1000,
            value=0,
            step=5,
            key="ml_add_plastic_per_district",
        )
        add_plastic_bottle_per_district = st.slider(
            "Add plastic-bottle bins / district",
            min_value=0,
            max_value=1000,
            value=0,
            step=5,
            key="ml_add_plastic_bottle_per_district",
        )
    with mctrl3:
        add_glass_bottle_per_district = st.slider(
            "Add glass-bottle bins / district",
            min_value=0,
            max_value=1000,
            value=0,
            step=5,
            key="ml_add_glass_bottle_per_district",
        )
        add_food_waste_per_district = st.slider(
            "Add food-waste bins / district",
            min_value=0,
            max_value=1000,
            value=0,
            step=5,
            key="ml_add_food_waste_per_district",
        )

    def render_material_panel(
        scenario_map_df: pd.DataFrame,
        value_col: str,
        map_title: str,
        fill_color: list[int],
        add_per_district: int,
        min_share_threshold: float,
    ) -> None:
        mat_df = scenario_map_df.copy()
        add_n = int(add_per_district)
        mat_df[value_col] = mat_df[value_col] + add_n
        mat_df["recycle_bins"] = mat_df["recycle_bins"] + add_n
        render_material_district_map(mat_df, value_col, map_title, fill_color)

        issue_df = mat_df[(mat_df["lat"] > 0) & (mat_df["lon"] > 0)].copy()
        if issue_df.empty:
            st.info("No districts available for issue listing.")
            return

        issue_df["material_share"] = (
            issue_df[value_col] / issue_df["recycle_bins"].replace(0, np.nan)
        ).fillna(0.0)
        issue_df["material_pressure_ratio"] = (
            issue_df["pressure_ratio"] / issue_df["material_share"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(999.0)
        issue_df["material_bins_added_per_district"] = add_n
        issue_df["material_bins_after"] = issue_df[value_col]
        issue_df["material_bins_before"] = (issue_df[value_col] - add_n).clip(lower=0)
        issue_df["material_pressure_score"] = issue_df["material_pressure_ratio"]

        problematic = issue_df[
            (issue_df["material_pressure_ratio"] > 1.0)
            | (issue_df["material_share"] < float(min_share_threshold))
            | (issue_df["pressure_ratio"] > float(target_pressure))
        ].copy()

        st.caption(
            f"Problematic bubbles: material pressure > 1.0, material share < {min_share_threshold:.2f}, or pressure > target."
        )
        if problematic.empty:
            st.success("No problematic bubbles under current settings.")
            return

        problematic = problematic.sort_values(
            ["material_pressure_ratio", "pressure_ratio"],
            ascending=[False, False],
        )
        st.dataframe(
            problematic[
                [
                    "district",
                    "material_bins_before",
                    "material_bins_added_per_district",
                    "material_bins_after",
                    "recycle_bins",
                    "material_share",
                    "material_pressure_score",
                    "material_pressure_ratio",
                    "pressure_ratio",
                    "projected_pressure_after_plan",
                    "required_volume_for_plan_m3_day",
                    "projected_capacity_after_plan_m3_day",
                    "planned_added_points",
                ]
            ].head(12),
            width="stretch",
            hide_index=True,
        )

    s1, s2 = st.columns(2)
    with s1:
        render_material_panel(
            map_df,
            "paper_bins",
            "Paper (Scenario)",
            [52, 152, 219, 185],
            add_paper_per_district,
            0.18,
        )
    with s2:
        render_material_panel(
            map_df,
            "plastic_bins",
            "Plastic (Scenario)",
            [46, 204, 113, 185],
            add_plastic_per_district,
            0.18,
        )

    s3, s4 = st.columns(2)
    with s3:
        render_material_panel(
            map_df,
            "metal_bins",
            "Metal (Scenario)",
            [241, 196, 15, 190],
            add_metal_per_district,
            0.12,
        )
    with s4:
        render_material_panel(
            map_df,
            "plastic_bottle_bins",
            "Plastic Bottles (Scenario)",
            [230, 126, 34, 190],
            add_plastic_bottle_per_district,
            0.10,
        )

    s5, s6 = st.columns(2)
    with s5:
        render_material_panel(
            map_df,
            "glass_bottle_bins",
            "Glass Bottles (Scenario)",
            [155, 89, 182, 190],
            add_glass_bottle_per_district,
            0.08,
        )
    with s6:
        render_material_panel(
            map_df,
            "food_waste_bins",
            "Food Waste (Scenario)",
            [231, 76, 60, 190],
            add_food_waste_per_district,
            0.08,
        )

    chart_df = scenario_df[["district", "pressure_ratio", "projected_pressure_after_plan"]].copy()
    chart_df = chart_df.melt(id_vars="district", var_name="stage", value_name="pressure")
    chart_df["stage"] = chart_df["stage"].map(
        {
            "pressure_ratio": "Before",
            "projected_pressure_after_plan": "After",
        }
    )
    pressure_chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("pressure:Q", title="Pressure ratio"),
        y=alt.Y("district:N", sort="-x", title="District"),
        color=alt.Color("stage:N", scale=alt.Scale(domain=["Before", "After"], range=["#e74c3c", "#2ecc71"])),
        tooltip=[
            alt.Tooltip("district:N", title="District"),
            alt.Tooltip("stage:N", title="Stage"),
            alt.Tooltip("pressure:Q", title="Pressure", format=".3f"),
        ],
    ).properties(height=420)
    st.altair_chart(pressure_chart, width="stretch")

    recommendation_cols = [
        "district",
        "pressure_ratio",
        "required_volume_for_plan_m3_day",
        "capacity_shortage_m3_day",
        "new_points_needed",
        "new_points_needed_rule",
        "new_points_needed_ml",
        "new_points_needed_final",
        "planned_added_points",
        "projected_pressure_after_plan",
    ]
    recommendation_df = scenario_df[recommendation_cols].sort_values(
        ["pressure_ratio", "planned_added_points"],
        ascending=[False, False],
    )
    st.dataframe(recommendation_df, width="stretch", hide_index=True)

    csv_bytes = recommendation_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download SKLearn recommendations CSV",
        data=csv_bytes,
        file_name="sklearn_recycle_point_recommendations.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Recycling Effectiveness and Pressure", layout="wide")

    st.title("Recycling Effectiveness vs Population Density and Housing Access")
    st.caption(
        "Rebuilt analysis using population-density proxy, housing locations, and recycle-point locations. "
        "Bin volume and pressure are estimated from available fields in this repository."
    )

    with st.sidebar:
        page_mode = st.radio(
            "Page",
            options=["Main Analysis", "SKLearn Planner", "Truck Path Prediction"],
            index=0,
        )
        st.header("Model Assumptions")
        people_per_unit = st.slider("People per housing unit", min_value=1.8, max_value=4.5, value=2.8, step=0.1)
        private_people_per_site = st.slider("Private-site population proxy", min_value=120, max_value=1200, value=420, step=20)
        per_person_recyclables_liters_day = st.slider("Recyclables generated per person (liters/day)", min_value=0.3, max_value=3.0, value=1.1, step=0.1)
        target_distance_km = st.slider("Target max walking distance to nearest bin (km)", min_value=0.05, max_value=2.5, value=0.8, step=0.05)
        st.subheader("Bin Volume and Distribution")
        bin_volume_multiplier = st.slider("Global bin volume multiplier", min_value=0.3, max_value=2.5, value=0.75, step=0.05)
        material_stream_capacity_bonus = st.slider("Extra capacity per additional material stream", min_value=0.00, max_value=0.40, value=0.02, step=0.01)
        smart_bin_multiplier = st.slider("Smart-bin capacity multiplier", min_value=0.4, max_value=3.0, value=1.00, step=0.05)
        station_multiplier = st.slider("Station/store capacity multiplier", min_value=0.4, max_value=3.0, value=0.90, step=0.05)
        street_bin_multiplier = st.slider("Street-corner/spot capacity multiplier", min_value=0.4, max_value=3.0, value=0.75, step=0.05)
        private_bin_multiplier = st.slider("Private-collection capacity multiplier", min_value=0.4, max_value=3.0, value=0.70, step=0.05)
        distribution_efficiency = st.slider("Distribution efficiency", min_value=0.3, max_value=1.8, value=0.85, step=0.05)
        clustering_penalty_strength = st.slider("Distribution clustering penalty", min_value=0.0, max_value=2.0, value=0.25, step=0.05)
        st.subheader("Pressure Calibration")
        demand_uplift_factor = st.slider("Demand uplift factor", min_value=0.6, max_value=3.0, value=1.30, step=0.05)
        capacity_utilization_factor = st.slider("Effective capacity utilization", min_value=0.20, max_value=1.20, value=0.55, step=0.05)
        normalize_baseline_pressure = st.checkbox("Normalize baseline pressure", value=True)
        baseline_pressure_target = st.slider("Baseline pressure target", min_value=0.60, max_value=2.50, value=1.20, step=0.05)

    district_df, recycle_df, housing_df, nearest_df = build_district_table(
        people_per_unit=float(people_per_unit),
        private_people_per_site=float(private_people_per_site),
        per_person_recyclables_liters_day=float(per_person_recyclables_liters_day),
        target_distance_km=float(target_distance_km),
        bin_volume_multiplier=float(bin_volume_multiplier),
        smart_bin_multiplier=float(smart_bin_multiplier),
        station_multiplier=float(station_multiplier),
        street_bin_multiplier=float(street_bin_multiplier),
        private_bin_multiplier=float(private_bin_multiplier),
        material_stream_capacity_bonus=float(material_stream_capacity_bonus),
        distribution_efficiency=float(distribution_efficiency),
        clustering_penalty_strength=float(clustering_penalty_strength),
        demand_uplift_factor=float(demand_uplift_factor),
        capacity_utilization_factor=float(capacity_utilization_factor),
        normalize_baseline_pressure=bool(normalize_baseline_pressure),
        baseline_pressure_target=float(baseline_pressure_target),
    )

    if page_mode == "SKLearn Planner":
        render_sklearn_planner_page(district_df, nearest_df)
        return

    if page_mode == "Truck Path Prediction":
        render_truck_path_page(recycle_df, district_df)
        return

    district_df = add_sklearn_point_estimates(district_df, 2.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts analysed", f"{len(district_df):,}")
    c2.metric("Total recycle points", f"{int(district_df['recycle_bins'].sum()):,}")
    c3.metric("Estimated total capacity (m3/day)", f"{district_df['capacity_m3_day'].sum():,.0f}")
    c4.metric("Estimated population covered", f"{district_df['estimated_population'].sum():,.0f}")
    st.caption(
        f"Capacity model: raw {district_df['capacity_m3_day_raw_base'].sum():,.0f} m3/day -> "
        f"volume-adjusted {district_df['capacity_m3_day_adjusted'].sum():,.0f} m3/day -> "
        f"distribution-adjusted {district_df['capacity_m3_day'].sum():,.0f} m3/day."
    )
    st.caption(
        f"Territory pressure: pre-normalization {district_df['territory_pressure_pre_normalization'].iloc[0]:.2f}, "
        f"normalization factor {district_df['pressure_normalization_factor'].iloc[0]:.2f}, "
        f"post-normalization {district_df['territory_pressure_post_normalization'].iloc[0]:.2f}."
    )

    st.subheader("1) District Pressure and Effectiveness")
    map_mode = st.radio("Map color mode", options=["Pressure", "Effectiveness Gap"], horizontal=True)
    with st.expander("How to read these values"):
        st.markdown(
            "- **Pressure ratio** = required recyclable volume / estimated recycle-bin capacity.  \n"
            "  - > 1.0: likely capacity shortage.  \n"
            "  - < 1.0: estimated spare capacity.  \n"
            "  - **Map color is absolute**: green = 0 pressure, red = 1 pressure (and above).  \n"
            "- **Effectiveness score (0-100)** combines: housing access distance, pressure balance, and diversity alignment. Higher is better.  \n"
            "- **Effectiveness gap** color mode highlights districts with lower effectiveness (larger gap from ideal score 100)."
        )
    render_pressure_map(district_df, color_mode=map_mode)

    st.subheader("2) Too Few / Too Many Bin Diagnosis (distance + pressure)")
    status_counts = district_df["bin_balance_status"].value_counts().rename_axis("status").reset_index(name="districts")
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(status_counts, width="stretch", hide_index=True)
    with right:
        diag = alt.Chart(district_df).mark_bar().encode(
            y=alt.Y("district:N", sort="-x", title="District"),
            x=alt.X("pressure_ratio:Q", title="Pressure ratio (required / capacity)"),
            color=alt.Color("bin_balance_status:N", scale=alt.Scale(domain=["Too Few", "Balanced", "Too Many"], range=["#e74c3c", "#f1c40f", "#2ecc71"])),
            tooltip=[
                alt.Tooltip("district:N", title="District"),
                alt.Tooltip("pressure_ratio:Q", title="Pressure", format=".2f"),
                alt.Tooltip("coverage_rate:Q", title="Coverage", format=".2f"),
                alt.Tooltip("mean_nearest_bin_km:Q", title="Mean nearest bin km", format=".2f"),
                alt.Tooltip("bins_per_10k_pop:Q", title="Bins per 10k", format=".1f"),
                alt.Tooltip("bin_balance_status:N", title="Status"),
            ],
        ).properties(height=420)
        st.altair_chart(diag, width="stretch")

    st.subheader("2.1) High-pressure Regions (Pressure > 1)")
    high_pressure_df = district_df[district_df["pressure_ratio"] > 1.0].copy()
    if high_pressure_df.empty:
        st.success("No districts currently exceed pressure ratio 1.0 under the selected assumptions.")
    else:
        st.warning(
            "Highlighted on map with red outline: these districts have estimated recyclable demand above capacity."
        )
        hp1, hp2, hp3 = st.columns(3)
        hp1.metric("High-pressure districts", f"{len(high_pressure_df):,}")
        hp2.metric("Total capacity shortfall (m3/day)", f"{high_pressure_df['capacity_shortage_m3_day'].sum():,.1f}")
        hp3.metric("Estimated new points needed", f"{int(high_pressure_df['new_points_needed_final'].sum()):,}")

        st.caption(
            "Baseline estimate uses 2.0 m3/day capacity per added point and the sidebar overflow target. "
            "Use the SKLearn Planner page for adjustable sliders and simulation button."
        )

        pressure_list_cols = [
            "district",
            "pressure_ratio",
            "capacity_shortage_m3_day",
            "new_points_needed",
            "new_points_needed_ml",
            "new_points_needed_final",
            "projected_pressure_after_add",
            "recycle_bins",
            "coverage_rate",
            "mean_nearest_bin_km",
            "effectiveness_score",
        ]
        st.dataframe(
            high_pressure_df[pressure_list_cols].sort_values(
                ["pressure_ratio", "capacity_shortage_m3_day"],
                ascending=[False, False],
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("3) Housing Access to Recycle Bins")
    access_df = nearest_df.copy()
    access_df = access_df[np.isfinite(access_df["nearest_bin_km"])]
    if access_df.empty:
        st.info("No nearest-distance records could be computed.")
    else:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            view_percentile = st.slider(
                "Chart y-axis cap percentile (ignore extreme outliers in chart view)",
                min_value=80,
                max_value=100,
                value=95,
                step=1,
            )
        with col_b:
            distance_alert_km = st.slider(
                "Distance threshold to highlight under-served housing (km)",
                min_value=0.05,
                max_value=3.0,
                value=max(0.2, float(target_distance_km)),
                step=0.05,
            )

        y_cap = float(np.nanpercentile(access_df["nearest_bin_km"], view_percentile))
        clipped_df = access_df[access_df["nearest_bin_km"] <= y_cap].copy()

        access_chart = alt.Chart(clipped_df).mark_boxplot().encode(
            x=alt.X("land_use:N", title="Land-use type"),
            y=alt.Y(
                "nearest_bin_km:Q",
                title=f"Nearest recycle point distance (km, capped at p{view_percentile})",
                scale=alt.Scale(domain=[0, y_cap]),
            ),
            color=alt.Color(
                "land_use:N",
                scale=alt.Scale(domain=list(LAND_USE_COLORS.keys()), range=["#3498db", "#2ecc71", "#e74c3c", "#f1c40f", "#9b59b6"]),
            ),
            tooltip=[alt.Tooltip("land_use:N", title="Land use")],
        ).properties(height=360)
        st.altair_chart(access_chart, width="stretch")

        outlier_count = int((access_df["nearest_bin_km"] > y_cap).sum())
        st.caption(
            f"Outlier points above cap: {outlier_count:,}. Use percentile 100 to include all points in the chart."
        )

        housing_threshold_df = access_df.copy()
        housing_threshold_df["district"] = housing_threshold_df["district_norm"].map(district_display)
        housing_threshold_df["distance_status"] = np.where(
            housing_threshold_df["nearest_bin_km"] > float(distance_alert_km),
            "Beyond threshold",
            "Within threshold",
        )
        housing_threshold_df["r"] = np.where(housing_threshold_df["distance_status"].eq("Beyond threshold"), 231, 46)
        housing_threshold_df["g"] = np.where(housing_threshold_df["distance_status"].eq("Beyond threshold"), 76, 204)
        housing_threshold_df["b"] = np.where(housing_threshold_df["distance_status"].eq("Beyond threshold"), 60, 113)

        far_count = int((housing_threshold_df["distance_status"] == "Beyond threshold").sum())
        total_count = len(housing_threshold_df)
        far_share = 100.0 * far_count / max(1, total_count)
        st.metric(
            f"Housing points farther than {distance_alert_km:.1f} km",
            f"{far_count:,} / {total_count:,}",
            delta=f"{far_share:.1f}%",
        )

        st.markdown("Housing-distance threshold map (red = farther than threshold, green = within; blue = recycle points)")
        render_housing_distance_map(housing_threshold_df, recycle_df)

        far_table = housing_threshold_df[housing_threshold_df["distance_status"] == "Beyond threshold"].copy()
        far_table = far_table.sort_values("nearest_bin_km", ascending=False)
        st.dataframe(
            far_table[["district", "land_use", "nearest_bin_km", "units_proxy"]].head(200),
            width="stretch",
            hide_index=True,
        )

    st.subheader("4) Diversity: Recycle Materials vs Land-use Diversity")
    div_chart = alt.Chart(district_df).mark_circle(size=120).encode(
        x=alt.X("land_use_diversity:Q", title="Land-use diversity (normalized Shannon)"),
        y=alt.Y("recycle_material_diversity:Q", title="Recycle-material diversity (normalized Shannon)"),
        color=alt.Color("diversity_gap:Q", title="Diversity gap", scale=alt.Scale(scheme="redyellowgreen")),
        tooltip=[
            alt.Tooltip("district:N", title="District"),
            alt.Tooltip("land_use_types:Q", title="Land-use types"),
            alt.Tooltip("material_types:Q", title="Material types"),
            alt.Tooltip("land_use_diversity:Q", format=".2f", title="Land-use div"),
            alt.Tooltip("recycle_material_diversity:Q", format=".2f", title="Recycle div"),
            alt.Tooltip("diversity_gap:Q", format=".2f", title="Gap (recycle - land)"),
        ],
    ).properties(height=360)
    st.altair_chart(div_chart, width="stretch")

    st.subheader("5) District Ranking Table")
    out_cols = [
        "district",
        "estimated_population",
        "population_density_proxy",
        "recycle_bins",
        "capacity_m3_day",
        "required_volume_m3_day",
        "capacity_shortage_m3_day",
        "new_points_needed",
        "new_points_needed_ml",
        "new_points_needed_final",
        "projected_pressure_after_add",
        "pressure_ratio",
        "mean_nearest_bin_km",
        "coverage_rate",
        "land_use_types",
        "material_types",
        "recycle_material_diversity",
        "land_use_diversity",
        "diversity_gap",
        "effectiveness_score",
        "recycling_possibility_score",
        "bin_balance_status",
    ]
    st.dataframe(
        district_df[out_cols].sort_values(["effectiveness_score", "pressure_ratio"], ascending=[False, False]),
        width="stretch",
        hide_index=True,
    )

    top_need = district_df.sort_values(["bin_balance_status", "pressure_ratio", "mean_nearest_bin_km"], ascending=[True, False, False]).head(8)
    st.subheader("6) Priority Districts for Bin Intervention")
    st.dataframe(
        top_need[["district", "bin_balance_status", "pressure_ratio", "mean_nearest_bin_km", "coverage_rate", "effectiveness_score", "recycling_possibility_score"]],
        width="stretch",
        hide_index=True,
    )

    st.subheader("7) Separate Maps by Main Recycle Types")
    st.caption(
        "Bubble size = number of recycle points for that material. "
        "Darker outline/opacity = higher district pressure ratio. Tooltip includes overall and material-adjusted pressure."
    )
    m1, m2 = st.columns(2)
    with m1:
        render_material_district_map(district_df, "paper_bins", "Paper", [52, 152, 219, 185])
    with m2:
        render_material_district_map(district_df, "plastic_bins", "Plastic", [46, 204, 113, 185])

    m3, m4 = st.columns(2)
    with m3:
        render_material_district_map(district_df, "metal_bins", "Metal", [241, 196, 15, 190])
    with m4:
        render_material_district_map(district_df, "plastic_bottle_bins", "Plastic Bottles", [230, 126, 34, 190])

    m5, m6 = st.columns(2)
    with m5:
        render_material_district_map(district_df, "glass_bottle_bins", "Glass Bottles", [155, 89, 182, 190])
    with m6:
        render_material_district_map(district_df, "food_waste_bins", "Food Waste", [231, 76, 60, 190])

    st.markdown(
        "**Interpretation guide**  \n"
        "- Pressure ratio > 1 means estimated recyclable demand exceeds estimated recycle-point capacity.  \n"
        "- Too Few / Too Many classification combines pressure with housing-distance coverage and bins-per-population.  \n"
        "- Diversity gap near 0 means recycle-point material diversity roughly matches local land-use complexity."
    )


if __name__ == "__main__":
    main()
