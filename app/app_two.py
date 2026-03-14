from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors


st.set_page_config(page_title="Smart Bin Planning Lab", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]

RECYCLE_POINTS_CSV = BASE_DIR / "recycle_points" / "wasteless250918.csv"
DISTRICT_WASTE_CSV = BASE_DIR / "waste_map" / "hk_district_waste_2022.csv"
MSW_COMPOSITION_CSV = BASE_DIR / "waste_map" / "hk_msw_composition_2022.csv"
RECYCLABLES_CSV = BASE_DIR / "waste_map" / "hk_recyclables_2022.csv"
PRH_JSON = BASE_DIR / "housing" / "prh-estates.json"
HOS_JSON = BASE_DIR / "housing" / "hos-courts.json"
PRIVATE_GEOJSON = BASE_DIR / "housing" / "private.geojson"
OPEN_RECYCLE_GEOJSON = BASE_DIR / "open_recycle" / "OSDRS_converted.geojson"
WASTE_FACILITY_GEOJSON = BASE_DIR / "waste_management" / "EPDWMF_20260228.gdb_converted.geojson"


DISTRICT_CENTROIDS = {
    "Central & Western": (22.287, 114.144),
    "Eastern": (22.284, 114.224),
    "Southern": (22.247, 114.158),
    "Wan Chai": (22.278, 114.173),
    "Kowloon City": (22.328, 114.190),
    "Kwun Tong": (22.313, 114.225),
    "Sham Shui Po": (22.331, 114.161),
    "Wong Tai Sin": (22.346, 114.194),
    "Yau Tsim Mong": (22.321, 114.169),
    "Kwai Tsing": (22.360, 114.129),
    "North": (22.495, 114.128),
    "Sai Kung": (22.383, 114.271),
    "Sha Tin": (22.387, 114.195),
    "Tai Po": (22.450, 114.168),
    "Tsuen Wan": (22.375, 114.114),
    "Tuen Mun": (22.392, 113.972),
    "Yuen Long": (22.446, 114.034),
    "Cheung Chau": (22.209, 114.028),
    "Hei Ling Chau": (22.244, 114.032),
    "Lamma Island": (22.226, 114.108),
    "Ma Wan": (22.355, 114.060),
    "Mui Wo": (22.265, 113.997),
    "Lantau": (22.282, 113.944),
    "Peng Chau": (22.285, 114.039),
}

DISTRICT_DISPLAY = {
    "CENTRALWESTERN": "Central & Western",
    "EASTERN": "Eastern",
    "SOUTHERN": "Southern",
    "WANCHAI": "Wan Chai",
    "KOWLOONCITY": "Kowloon City",
    "KWUNTONG": "Kwun Tong",
    "SHAMSHUIPO": "Sham Shui Po",
    "WONGTAISIN": "Wong Tai Sin",
    "YAUTSIMMONG": "Yau Tsim Mong",
    "ISLANDS": "Lantau",
    "KWAITSING": "Kwai Tsing",
    "NORTH": "North",
    "SAIKUNG": "Sai Kung",
    "SHATIN": "Sha Tin",
    "TAIPO": "Tai Po",
    "TSUENWAN": "Tsuen Wan",
    "TUENMUN": "Tuen Mun",
    "YUENLONG": "Yuen Long",
    "CHEUNGCHAU": "Cheung Chau",
    "HEILINGCHAU": "Hei Ling Chau",
    "LAMMAISLAND": "Lamma Island",
    "MAWAN": "Ma Wan",
    "MUIWO": "Mui Wo",
    "LANTAU": "Lantau",
    "PENGCHAU": "Peng Chau",
}


def normalize_district(value: object) -> str:
    text = str(value or "").upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def district_display(norm: str) -> str:
    if norm in DISTRICT_DISPLAY:
        return DISTRICT_DISPLAY[norm]
    if not norm:
        return "Unknown"
    return norm.title()


def extract_en(value: object) -> str:
    if isinstance(value, dict):
        if value.get("en") is not None:
            return str(value.get("en", "")).strip()
        for _, v in value.items():
            if v is not None and str(v).strip() != "":
                return str(v).strip()
        return ""
    return str(value or "").strip()


def parse_number(value: object) -> float:
    text = str(value or "").replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in {"", "-", "."}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def split_waste_types(value: object) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def normalize_material_name(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("plastic bottle", "plastics")
    text = text.replace("metal", "metals")
    text = text.replace("ferrous metals", "metals")
    text = text.replace("non-ferrous metals", "metals")
    text = text.replace("food waste", "putrescibles")
    return text


def parse_district_from_text(value: object) -> str:
    text = str(value or "").upper()
    for district_name in DISTRICT_CENTROIDS.keys():
        if re.search(rf"\b{re.escape(district_name.upper())}\b", text):
            return normalize_district(district_name)
    return ""


def classify_area_type(row: pd.Series) -> str:
    density = float(row.get("density_proxy", 0.0) or 0.0)
    facilities = float(row.get("waste_facilities", 0.0) or 0.0)
    if facilities >= 3:
        return "Industrial"
    if density >= 1200:
        return "Residential"
    return "Commercial"


def coverage_area_km2(bin_count: float, radius_m: float = 150.0) -> float:
    return float(bin_count) * math.pi * (radius_m**2) / 1_000_000.0


def color_scale(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    if pd.isna(value) or vmax <= vmin:
        return (145, 145, 145)
    ratio = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    r = int(80 + ratio * 170)
    g = int(180 - ratio * 130)
    b = int(120 - ratio * 80)
    return (r, g, b)


def meters_to_degree_offsets(lat: float, meters: float) -> tuple[float, float]:
    lat_deg = meters / 111_320.0
    lon_deg = meters / max(1.0, 111_320.0 * math.cos(math.radians(lat)))
    return lat_deg, lon_deg


def generate_candidate_points(lat: float, lon: float, count: int, min_distance_m: float) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    pts: list[tuple[float, float]] = []
    ring = 0
    remaining = count
    while remaining > 0:
        ring += 1
        slots = max(4, ring * 6)
        radial = min_distance_m * ring
        lat_off, lon_off = meters_to_degree_offsets(lat, radial)
        use_slots = min(slots, remaining)
        for i in range(use_slots):
            angle = 2 * math.pi * i / slots
            pts.append((lat + lat_off * math.sin(angle), lon + lon_off * math.cos(angle)))
            remaining -= 1
            if remaining <= 0:
                break
    return pts


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_population_proxy_by_district() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for item in load_json(PRH_JSON):
        rows.append(
            {
                "district_norm": normalize_district(extract_en(item.get("District Name"))),
                "units": parse_number(extract_en(item.get("No. of Rental Flats"))),
                "housing_site": 1,
            }
        )

    for item in load_json(HOS_JSON):
        rows.append(
            {
                "district_norm": normalize_district(extract_en(item.get("District Name"))),
                "units": parse_number(extract_en(item.get("No. of Flats"))),
                "housing_site": 1,
            }
        )

    housing = pd.DataFrame(rows)
    grouped_housing = (
        housing.groupby("district_norm", as_index=False)
        .agg(
            housing_units_proxy=("units", "sum"),
            housing_sites=("housing_site", "sum"),
        )
    )

    private_rows: list[dict[str, object]] = []
    private_geo = load_json(PRIVATE_GEOJSON)
    for feat in private_geo.get("features", []):
        props = feat.get("properties", {})
        private_rows.append(
            {
                "district_norm": parse_district_from_text(props.get("ADDR_OF_BUILDING_IN_ENG", "")),
                "private_site": 1,
            }
        )

    private_df = pd.DataFrame(private_rows)
    private_grouped = (
        private_df[private_df["district_norm"] != ""]
        .groupby("district_norm", as_index=False)
        .agg(private_sites=("private_site", "sum"))
    )

    out = grouped_housing.merge(private_grouped, how="outer", on="district_norm")
    out["housing_units_proxy"] = out["housing_units_proxy"].fillna(0.0)
    out["housing_sites"] = out["housing_sites"].fillna(0.0)
    out["private_sites"] = out["private_sites"].fillna(0.0)
    out["population_proxy"] = out["housing_units_proxy"] + out["private_sites"] * 400.0
    out["density_proxy"] = out["population_proxy"] / (out["housing_sites"] + out["private_sites"]).replace(0, np.nan)
    out["district"] = out["district_norm"].map(district_display)
    return out


@st.cache_data(show_spinner=False)
def load_green_hubs_by_district() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    geo = load_json(OPEN_RECYCLE_GEOJSON)
    for feat in geo.get("features", []):
        props = feat.get("properties", {})
        name = str(props.get("BLDG_ENGNM", ""))
        if "GREEN@" not in name.upper():
            continue
        district = normalize_district(name.upper().split("GREEN@", maxsplit=1)[-1].strip())
        rows.append({"district_norm": district, "green_hub": 1})

    if not rows:
        return pd.DataFrame(columns=["district_norm", "green_hubs"])
    return pd.DataFrame(rows).groupby("district_norm", as_index=False).agg(green_hubs=("green_hub", "sum"))


@st.cache_data(show_spinner=False)
def load_waste_facilities_by_district() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    geo = load_json(WASTE_FACILITY_GEOJSON)
    for feat in geo.get("features", []):
        props = feat.get("properties", {})
        district = parse_district_from_text(props.get("ADDRESS_EN", ""))
        if district:
            rows.append({"district_norm": district, "facility": 1})

    if not rows:
        return pd.DataFrame(columns=["district_norm", "waste_facilities"])
    return pd.DataFrame(rows).groupby("district_norm", as_index=False).agg(waste_facilities=("facility", "sum"))


@st.cache_data(show_spinner=False)
def load_recycle_points() -> pd.DataFrame:
    df = load_csv(RECYCLE_POINTS_CSV)
    df["district_norm"] = df["district_id"].map(normalize_district)
    df["district"] = df["district_norm"].map(district_display)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lgt"] = pd.to_numeric(df["lgt"], errors="coerce")
    df["coord_valid"] = df["lat"].between(22.0, 23.0) & df["lgt"].between(113.7, 114.5)
    df["waste_list"] = df["waste_type"].apply(split_waste_types)
    df["material_tokens"] = df["waste_list"].apply(lambda xs: [normalize_material_name(x) for x in xs])
    return df


@st.cache_data(show_spinner=False)
def build_base_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points = load_recycle_points()

    district_waste = load_csv(DISTRICT_WASTE_CSV)
    district_waste["municipal_solid_waste_tpd"] = pd.to_numeric(district_waste["municipal_solid_waste_tpd"], errors="coerce")
    district_waste["district_norm"] = district_waste["district"].map(normalize_district)
    district_waste["district"] = district_waste["district_norm"].map(district_display)

    valid_districts = {normalize_district(k) for k in DISTRICT_CENTROIDS.keys()}
    district_waste = district_waste[district_waste["district_norm"].isin(valid_districts)].copy()
    district_waste = district_waste[["district_norm", "district", "municipal_solid_waste_tpd"]]

    pop = load_population_proxy_by_district()
    hubs = load_green_hubs_by_district()
    facilities = load_waste_facilities_by_district()

    district_points = (
        points.groupby(["district_norm", "district"], as_index=False)
        .agg(
            bins_total=("cp_id", "count"),
            bins_accepted=("cp_state", lambda s: int((s == "Accepted").sum())),
            station_type_diversity=("legend", lambda s: int(pd.Series(s.fillna("Unknown")).nunique())),
            avg_material_types=("material_tokens", lambda s: float(np.mean([len(x) for x in s]))),
        )
    )

    composition = load_csv(MSW_COMPOSITION_CSV)
    composition["total_msw_tpd"] = pd.to_numeric(composition["total_msw_tpd"], errors="coerce")
    composition["material"] = composition["waste_type"].map(normalize_material_name)
    composition = composition.groupby("material", as_index=False).agg(total_msw_tpd=("total_msw_tpd", "sum"))
    composition_total = float(composition["total_msw_tpd"].sum())
    composition["expected_share"] = composition["total_msw_tpd"] / composition_total if composition_total > 0 else 0.0

    accepted = points[points["cp_state"] == "Accepted"].copy()
    rows: list[dict[str, object]] = []
    for _, row in accepted.iterrows():
        for material in row["material_tokens"]:
            rows.append({"district_norm": row["district_norm"], "material": material, "cp_id": row["cp_id"]})
    material_df = pd.DataFrame(rows)

    material_cov = pd.DataFrame(columns=["district_norm", "material", "points_accepting_material"])
    if not material_df.empty:
        material_cov = material_df.groupby(["district_norm", "material"], as_index=False).agg(points_accepting_material=("cp_id", "nunique"))

    district = (
        district_waste.merge(district_points, how="left", on=["district_norm", "district"])
        .merge(pop[["district_norm", "population_proxy", "density_proxy"]], how="left", on="district_norm")
        .merge(hubs, how="left", on="district_norm")
        .merge(facilities, how="left", on="district_norm")
    )

    for col in [
        "bins_total",
        "bins_accepted",
        "station_type_diversity",
        "avg_material_types",
        "population_proxy",
        "density_proxy",
        "green_hubs",
        "waste_facilities",
    ]:
        district[col] = district[col].fillna(0.0)

    mismatch_rows: list[dict[str, object]] = []
    for district_norm, dsub in district.groupby("district_norm"):
        bins = float(dsub["bins_accepted"].iloc[0])
        dmat = material_cov[material_cov["district_norm"] == district_norm]
        mismatch = 0.0
        for _, crow in composition.iterrows():
            material = str(crow["material"])
            expected = float(crow["expected_share"])
            found = dmat[dmat["material"] == material]
            points_accepting = float(found["points_accepting_material"].iloc[0]) if not found.empty else 0.0
            availability = min(1.0, points_accepting / bins) if bins > 0 else 0.0
            mismatch += expected * (1.0 - availability)
        mismatch_rows.append({"district_norm": district_norm, "type_fit_mismatch": mismatch})

    mismatch_df = pd.DataFrame(mismatch_rows)
    district = district.merge(mismatch_df, how="left", on="district_norm")
    district["type_fit_mismatch"] = district["type_fit_mismatch"].fillna(1.0)

    district["waste_per_proxy_person_kg_day"] = district["municipal_solid_waste_tpd"] * 1000.0 / district["population_proxy"].replace(0, np.nan)
    district["bin_pressure_kg_day"] = district["municipal_solid_waste_tpd"] * 1000.0 / district["bins_accepted"].replace(0, np.nan)
    district["bins_per_10k_people"] = district["bins_accepted"] / district["population_proxy"].replace(0, np.nan) * 10000.0

    district["amenity_score"] = district["green_hubs"] / district["population_proxy"].replace(0, np.nan) * 10000.0
    district["amenity_score"] = district["amenity_score"].fillna(0.0)
    district["area_type"] = district.apply(classify_area_type, axis=1)

    for dname, (lat, lon) in DISTRICT_CENTROIDS.items():
        norm = normalize_district(dname)
        district.loc[district["district_norm"] == norm, "lat"] = lat
        district.loc[district["district_norm"] == norm, "lon"] = lon

    recyclables = load_csv(RECYCLABLES_CSV)
    recyclables["total_recovered_thousand_tonnes"] = pd.to_numeric(recyclables["total_recovered_thousand_tonnes"], errors="coerce")

    return district, points, composition, recyclables


def find_duplicate_bins(points_df: pd.DataFrame, min_distance_m: float) -> pd.DataFrame:
    valid = points_df[(points_df["coord_valid"]) & (points_df["cp_state"] == "Accepted")].copy()
    if len(valid) < 3:
        return valid.iloc[0:0].copy()

    coords = valid[["lat", "lgt"]].to_numpy()
    model = NearestNeighbors(n_neighbors=2)
    model.fit(coords)
    dists, _ = model.kneighbors(coords)

    # Approximate meters from degrees near HK latitude.
    nearest_m = dists[:, 1] * 111_320.0
    valid["nearest_neighbor_m"] = nearest_m
    return valid[valid["nearest_neighbor_m"] < min_distance_m].copy()


def train_bin_need_model(district_df: pd.DataFrame, n_estimators: int, max_depth: int, target_pressure: float) -> tuple[RandomForestRegressor, pd.DataFrame, pd.Series]:
    work = district_df.copy()
    work["bins_needed_target"] = np.maximum(
        0,
        np.ceil(work["municipal_solid_waste_tpd"] * 1000.0 / max(1.0, target_pressure)) - work["bins_accepted"],
    )

    features = [
        "population_proxy",
        "density_proxy",
        "municipal_solid_waste_tpd",
        "green_hubs",
        "waste_facilities",
        "type_fit_mismatch",
        "bins_per_10k_people",
    ]
    X = work[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = work["bins_needed_target"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    preds = pd.Series(model.predict(X), index=work.index)
    return model, importance, preds


def train_utilization_model(district_df: pd.DataFrame) -> LinearRegression:
    work = district_df.copy()
    encoded = pd.get_dummies(work[["area_type"]], prefix="area")
    X = pd.concat([encoded, work[["density_proxy", "amenity_score"]].fillna(0.0)], axis=1)
    y = work["bin_pressure_kg_day"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_fill_rate(model: LinearRegression, area_type: str, density: float, amenity_score: float) -> tuple[float, int, pd.DataFrame]:
    row = {
        "area_Residential": 1.0 if area_type == "Residential" else 0.0,
        "area_Commercial": 1.0 if area_type == "Commercial" else 0.0,
        "area_Industrial": 1.0 if area_type == "Industrial" else 0.0,
        "density_proxy": density,
        "amenity_score": amenity_score,
    }

    X = pd.DataFrame([row])
    # Align columns to model expectation.
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0.0
    X = X[list(model.feature_names_in_)]

    pressure = float(model.predict(X)[0])
    pressure = max(0.0, pressure)

    bin_capacity_kg = 120.0
    fill_rate_pct_per_day = min(100.0, pressure / bin_capacity_kg * 100.0)

    if fill_rate_pct_per_day <= 0:
        collection_every_days = 7
    else:
        collection_every_days = int(max(1, min(7, round(80.0 / fill_rate_pct_per_day))))

    daily_curve = []
    cumulative = 0.0
    for day in range(1, 8):
        cumulative = min(100.0, cumulative + fill_rate_pct_per_day)
        daily_curve.append({"day": day, "predicted_fill_pct": cumulative})
        if day % collection_every_days == 0:
            cumulative = max(0.0, cumulative - 80.0)

    curve_df = pd.DataFrame(daily_curve)
    return fill_rate_pct_per_day, collection_every_days, curve_df


def run_optimizer(
    district_df: pd.DataFrame,
    points_df: pd.DataFrame,
    predicted_bins_needed: pd.Series,
    density_threshold: float,
    min_distance_m: float,
    removal_utilization_threshold: float,
) -> dict[str, pd.DataFrame]:
    work = district_df.copy()
    pressure = work["bin_pressure_kg_day"].replace([np.inf, -np.inf], np.nan)
    pmin = float(pressure.min(skipna=True))
    pmax = float(pressure.max(skipna=True))
    if np.isnan(pmin) or np.isnan(pmax) or pmax <= pmin:
        work["utilization_rate"] = 0.5
    else:
        work["utilization_rate"] = (pressure - pmin) / (pmax - pmin)

    work["pred_bins_needed"] = np.maximum(0, np.round(predicted_bins_needed)).astype(int)

    lacking = work[
        (work["density_proxy"] >= density_threshold)
        & (work["pred_bins_needed"] > 0)
    ].copy()

    suggestion_rows: list[dict[str, object]] = []
    for _, row in lacking.iterrows():
        district = str(row["district"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        needed = int(row["pred_bins_needed"])
        pts = generate_candidate_points(lat, lon, needed, min_distance_m)
        for i, (plat, plon) in enumerate(pts, start=1):
            suggestion_rows.append(
                {
                    "district": district,
                    "lat": plat,
                    "lgt": plon,
                    "suggestion": f"NEW_BIN_{district}_{i}",
                }
            )

    suggestions = pd.DataFrame(suggestion_rows)

    duplicates = find_duplicate_bins(points_df, min_distance_m)
    low_util_districts = work[work["utilization_rate"] < removal_utilization_threshold]["district_norm"].tolist()
    removals = duplicates[duplicates["district_norm"].isin(low_util_districts)].copy()

    return {
        "lacking": lacking,
        "suggestions": suggestions,
        "removals": removals,
        "work": work,
    }


def render_current_distribution_map(points_df: pd.DataFrame) -> None:
    map_df = points_df[points_df["coord_valid"]].copy()
    if map_df.empty:
        st.warning("No valid coordinates in current dataset.")
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        map_df,
        get_position="[lgt, lat]",
        get_radius=48,
        get_fill_color=[52, 122, 226, 140],
        pickable=True,
    )

    st.pydeck_chart(
        pdk.Deck(
            map_provider="carto",
            map_style="light",
            initial_view_state=pdk.ViewState(latitude=22.33, longitude=114.15, zoom=10.1, pitch=0),
            layers=[layer],
            tooltip={  # type: ignore[arg-type]
                "html": "<b>{district}</b><br/>Point ID: {cp_id}<br/>Type: {legend}<br/>Accepted: {cp_state}",
                "style": {"backgroundColor": "#132231", "color": "white"},
            },
        ),
        width="stretch",
    )


def render_before_after_map(points_df: pd.DataFrame, suggestions: pd.DataFrame, removals: pd.DataFrame) -> None:
    base = points_df[points_df["coord_valid"]].copy()
    layers: list[pdk.Layer] = []

    if not base.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                base,
                get_position="[lgt, lat]",
                get_radius=46,
                get_fill_color=[120, 120, 120, 90],
                pickable=True,
            )
        )

    if not suggestions.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                suggestions,
                get_position="[lgt, lat]",
                get_radius=70,
                get_fill_color=[39, 175, 95, 210],
                pickable=True,
            )
        )

    if not removals.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                removals,
                get_position="[lgt, lat]",
                get_radius=70,
                get_fill_color=[224, 72, 72, 210],
                pickable=True,
            )
        )

    if not layers:
        st.info("No map layers to render yet.")
        return

    st.pydeck_chart(
        pdk.Deck(
            map_provider="carto",
            map_style="light",
            initial_view_state=pdk.ViewState(latitude=22.33, longitude=114.15, zoom=10.1, pitch=0),
            layers=layers,
            tooltip={"text": "{district}"},  # type: ignore[arg-type]
        ),
        width="stretch",
    )


def main() -> None:
    st.title("Smart Bin Planning and Prediction App")
    st.caption("Interactive optimization, prediction, conversion advice, and what-if simulation using repo datasets.")

    required_files = [
        RECYCLE_POINTS_CSV,
        DISTRICT_WASTE_CSV,
        MSW_COMPOSITION_CSV,
        RECYCLABLES_CSV,
        PRH_JSON,
        HOS_JSON,
        PRIVATE_GEOJSON,
        OPEN_RECYCLE_GEOJSON,
        WASTE_FACILITY_GEOJSON,
    ]
    missing = [p.name for p in required_files if not p.exists()]
    if missing:
        st.error(f"Missing required files: {', '.join(missing)}")
        return

    district_df, points_df, composition_df, recyclables_df = build_base_tables()

    with st.sidebar:
        st.header("Model Parameters")
        n_estimators = st.slider("Random forest trees", min_value=50, max_value=400, value=180, step=10)
        max_depth = st.slider("Random forest max depth (0 = unlimited)", min_value=0, max_value=20, value=8)
        target_pressure = st.slider("Target bin pressure (kg/day/bin)", min_value=20.0, max_value=250.0, value=90.0, step=5.0)

    model, feature_importance, predicted_bins_needed = train_bin_need_model(
        district_df=district_df,
        n_estimators=n_estimators,
        max_depth=max_depth,
        target_pressure=target_pressure,
    )

    utilization_model = train_utilization_model(district_df)

    st.subheader("1) Data Loading and Display")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recycle points", f"{len(points_df):,}")
    c2.metric("Accepted points", f"{int((points_df['cp_state'] == 'Accepted').sum()):,}")
    c3.metric("Districts in model", f"{district_df['district_norm'].nunique()}")
    c4.metric("Material categories", f"{composition_df['material'].nunique()}")

    with st.expander("Raw recycle point dataset preview"):
        show_cols = ["cp_id", "district", "cp_state", "legend", "waste_type", "lat", "lgt"]
        st.dataframe(points_df[show_cols], width="stretch", hide_index=True)

    st.caption("Current bin distribution map")
    render_current_distribution_map(points_df)

    st.subheader("2a) Bin Placement Optimizer")
    o1, o2, o3 = st.columns(3)
    with o1:
        density_threshold = st.slider("Population density threshold for new bins", min_value=100.0, max_value=3000.0, value=900.0, step=50.0)
    with o2:
        min_distance_m = st.slider("Minimum distance between bins (meters)", min_value=30, max_value=400, value=120, step=10)
    with o3:
        removal_threshold = st.slider("Utilization rate threshold for bin removal", min_value=0.05, max_value=0.95, value=0.35, step=0.05)

    if "optimizer_result" not in st.session_state:
        st.session_state["optimizer_result"] = run_optimizer(
            district_df,
            points_df,
            predicted_bins_needed,
            density_threshold,
            float(min_distance_m),
            removal_threshold,
        )

    if st.button("Predict Optimal Locations", use_container_width=True):
        st.session_state["optimizer_result"] = run_optimizer(
            district_df,
            points_df,
            predicted_bins_needed,
            density_threshold,
            float(min_distance_m),
            removal_threshold,
        )

    optimizer_result = st.session_state["optimizer_result"]
    lacking_df = optimizer_result["lacking"]
    suggestions_df = optimizer_result["suggestions"]
    removals_df = optimizer_result["removals"]

    m1, m2, m3 = st.columns(3)
    m1.metric("High-density lacking districts", f"{len(lacking_df):,}")
    m2.metric("Suggested new bins", f"{len(suggestions_df):,}")
    m3.metric("Possible duplicate/low-utilization bins", f"{len(removals_df):,}")

    with st.expander("Suggested new bin locations"):
        if suggestions_df.empty:
            st.info("No new bin suggestions under current settings.")
        else:
            st.dataframe(suggestions_df, width="stretch", hide_index=True)

    with st.expander("Potential bin removal/merge candidates"):
        if removals_df.empty:
            st.info("No removal candidates under current settings.")
        else:
            show_cols = ["cp_id", "district", "legend", "nearest_neighbor_m", "lat", "lgt"]
            st.dataframe(removals_df[show_cols], width="stretch", hide_index=True)

    st.subheader("3) Before/After Visualization")
    left_map, right_map = st.columns(2)
    with left_map:
        st.caption("Before: current bin map")
        render_current_distribution_map(points_df)
    with right_map:
        st.caption("After: optimized adds/removals")
        render_before_after_map(points_df, suggestions_df, removals_df)

    current_bins = int((points_df["cp_state"] == "Accepted").sum())
    optimized_bins = max(1, current_bins + len(suggestions_df) - len(removals_df))
    current_coverage = coverage_area_km2(current_bins)
    optimized_coverage = coverage_area_km2(optimized_bins)

    current_pressure_avg = float(district_df["bin_pressure_kg_day"].replace([np.inf, -np.inf], np.nan).dropna().mean())
    predicted_pressure_avg = current_pressure_avg * (current_bins / optimized_bins)

    collection_eff_current = 1.0 / max(1e-6, current_pressure_avg)
    collection_eff_after = 1.0 / max(1e-6, predicted_pressure_avg)

    annual_cost_current = current_bins * 18_000
    annual_cost_after = optimized_bins * 18_000
    efficiency_gain = max(0.0, (collection_eff_after - collection_eff_current) / max(1e-6, collection_eff_current))
    overflow_savings = efficiency_gain * 220_000
    cost_savings = (annual_cost_current - annual_cost_after) + overflow_savings

    st.caption("Metrics comparison (current vs optimized)")
    a1, a2, a3 = st.columns(3)
    a1.metric("Coverage area (km2)", f"{current_coverage:.2f}", f"{optimized_coverage - current_coverage:+.2f}")
    a2.metric("Collection efficiency index", f"{collection_eff_current:.5f}", f"{collection_eff_after - collection_eff_current:+.5f}")
    a3.metric("Estimated annual cost savings (HKD)", f"{cost_savings:,.0f}")

    st.subheader("2b) Bin Utilization Predictor")
    p1, p2, p3 = st.columns(3)
    with p1:
        input_area_type = st.selectbox("Area type", ["Residential", "Commercial", "Industrial"])
    with p2:
        input_density = st.number_input("Population density", min_value=0.0, value=1200.0, step=50.0)
    with p3:
        input_amenity = st.number_input("Proximity to amenities score", min_value=0.0, value=1.5, step=0.1)

    fill_rate, freq_days, curve_df = predict_fill_rate(
        model=utilization_model,
        area_type=input_area_type,
        density=float(input_density),
        amenity_score=float(input_amenity),
    )

    u1, u2 = st.columns(2)
    u1.metric("Predicted fill rate (% per day)", f"{fill_rate:.1f}%")
    u2.metric("Recommended collection frequency", f"Every {freq_days} day(s)")

    st.caption("Predicted weekly fill trajectory")
    st.line_chart(curve_df.set_index("day")["predicted_fill_pct"])

    st.subheader("2c) Recycling Bin Conversion Advisor")
    t1, t2 = st.columns(2)
    with t1:
        convert_underutilized = st.toggle("Convert underutilized trash bins to recycling", value=True)
    with t2:
        adjust_density_by_area = st.toggle("Adjust recycling bin density based on area type", value=True)

    base_recovery = float(recyclables_df["total_recovered_thousand_tonnes"].sum())
    conversion_candidates = points_df[
        (points_df["cp_state"] == "Accepted") & points_df["legend"].str.contains("Recycling Bins at Public Place", na=False)
    ]

    underutilized_districts = district_df[
        district_df["bin_pressure_kg_day"]
        < district_df["bin_pressure_kg_day"].replace([np.inf, -np.inf], np.nan).median()
    ]["district_norm"]

    underutilized_bins = conversion_candidates[conversion_candidates["district_norm"].isin(underutilized_districts)]
    convertible_count = int(len(underutilized_bins)) if convert_underutilized else 0

    area_adjust_factor = 1.0
    if adjust_density_by_area:
        residential_share = float((district_df["area_type"] == "Residential").mean())
        area_adjust_factor = 1.0 + 0.15 * residential_share

    recovery_gain_factor = 1.0 + (convertible_count / max(1, current_bins)) * 0.30
    predicted_recovery = base_recovery * recovery_gain_factor * area_adjust_factor

    r1, r2, r3 = st.columns(3)
    r1.metric("Convertible bins", f"{convertible_count:,}")
    r2.metric("Current recovered (thousand tonnes)", f"{base_recovery:,.1f}")
    r3.metric("Predicted recovered (thousand tonnes)", f"{predicted_recovery:,.1f}", f"{predicted_recovery - base_recovery:+.1f}")

    st.subheader("2d) What-If Analysis")
    w1, w2, w3 = st.columns(3)
    with w1:
        zone = st.selectbox("Selected zone", sorted(district_df["district"].dropna().unique().tolist()))
    with w2:
        add_x = st.number_input("Add X bins", min_value=0, max_value=400, value=20, step=1)
    with w3:
        remove_y = st.number_input("Remove Y underperforming bins", min_value=0, max_value=400, value=5, step=1)

    zone_row = district_df[district_df["district"] == zone].head(1)
    if zone_row.empty:
        st.warning("Selected zone not found in district table.")
    else:
        zone_bins = int(zone_row["bins_accepted"].iloc[0])
        zone_pressure = float(zone_row["bin_pressure_kg_day"].iloc[0])

        zone_new_bins = max(1, zone_bins + int(add_x) - int(remove_y))
        zone_current_cov = coverage_area_km2(zone_bins)
        zone_new_cov = coverage_area_km2(zone_new_bins)

        zone_new_pressure = zone_pressure * zone_bins / zone_new_bins
        zone_eff_current = 1.0 / max(1e-6, zone_pressure)
        zone_eff_new = 1.0 / max(1e-6, zone_new_pressure)

        zone_cost_delta = (zone_bins - zone_new_bins) * 18_000
        zone_eff_savings = max(0.0, (zone_eff_new - zone_eff_current) / max(1e-6, zone_eff_current)) * 45_000
        zone_total_savings = zone_cost_delta + zone_eff_savings

        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Coverage area (km2)", f"{zone_current_cov:.2f}", f"{zone_new_cov - zone_current_cov:+.2f}")
        wc2.metric("Collection efficiency index", f"{zone_eff_current:.5f}", f"{zone_eff_new - zone_eff_current:+.5f}")
        wc3.metric("Estimated annual cost savings (HKD)", f"{zone_total_savings:,.0f}")

    st.subheader("4) Machine Learning Integration")
    i1, i2 = st.columns(2)
    with i1:
        st.caption("Feature importance for optimal bin placement model")
        st.bar_chart(feature_importance.set_index("feature")["importance"])
    with i2:
        st.caption("Predicted bins needed by district")
        pred_table = district_df[["district", "bins_accepted", "municipal_solid_waste_tpd"]].copy()
        pred_table["predicted_bins_needed"] = np.maximum(0, np.round(predicted_bins_needed)).astype(int)
        st.bar_chart(pred_table.set_index("district")["predicted_bins_needed"])

    st.dataframe(
        pred_table.sort_values("predicted_bins_needed", ascending=False),
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()
