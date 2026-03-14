from pathlib import Path
import json
import re

import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(page_title="HK Waste per Person and Recycle Pressure", layout="wide")

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent

DISTRICT_WASTE_CSV = BASE_DIR / "hk_district_waste_2022.csv"
MSW_COMPOSITION_CSV = BASE_DIR / "hk_msw_composition_2022.csv"
MSW_COMPOSITION_JSON = BASE_DIR / "hk_msw_composition_2022.json"
RECYCLABLES_CSV = BASE_DIR / "hk_recyclables_2022.csv"

RECYCLE_POINTS_CSV = ROOT_DIR / "recycle_points" / "wasteless250918.csv"
PRH_JSON = ROOT_DIR / "housing" / "prh-estates.json"
HOS_JSON = ROOT_DIR / "housing" / "hos-courts.json"
PRIVATE_GEOJSON = ROOT_DIR / "housing" / "private.geojson"


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


def parse_district_from_address(text: object) -> str:
	upper_text = str(text or "").upper()
	for display in DISTRICT_CENTROIDS.keys():
		pattern = rf"\b{re.escape(display.upper())}\b"
		if re.search(pattern, upper_text):
			return normalize_district(display)
	return ""


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
	return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> pd.DataFrame:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	return pd.DataFrame(data)


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
	out = df.copy()
	for col in columns:
		out[col] = pd.to_numeric(out[col], errors="coerce")
	return out


@st.cache_data(show_spinner=False)
def load_population_proxy_by_district() -> pd.DataFrame:
	housing_rows: list[dict[str, object]] = []

	with PRH_JSON.open("r", encoding="utf-8") as f:
		prh = json.load(f)
	for item in prh:
		district_name = extract_en(item.get("District Name"))
		housing_rows.append(
			{
				"district_norm": normalize_district(district_name),
				"units": parse_number(extract_en(item.get("No. of Rental Flats"))),
				"housing_site": 1,
			}
		)

	with HOS_JSON.open("r", encoding="utf-8") as f:
		hos = json.load(f)
	for item in hos:
		district_name = extract_en(item.get("District Name"))
		housing_rows.append(
			{
				"district_norm": normalize_district(district_name),
				"units": parse_number(extract_en(item.get("No. of Flats"))),
				"housing_site": 1,
			}
		)

	housing_df = pd.DataFrame(housing_rows)
	if housing_df.empty:
		housing_grouped = pd.DataFrame(columns=["district_norm", "housing_units_proxy", "housing_sites"])
	else:
		housing_grouped = (
			housing_df.groupby("district_norm", as_index=False)
			.agg(
				housing_units_proxy=("units", "sum"),
				housing_sites=("housing_site", "sum"),
			)
		)

	with PRIVATE_GEOJSON.open("r", encoding="utf-8") as f:
		private_geo = json.load(f)
	private_rows: list[dict[str, object]] = []
	for feat in private_geo.get("features", []):
		props = feat.get("properties", {})
		address_en = props.get("ADDR_OF_BUILDING_IN_ENG", "")
		private_rows.append({"district_norm": parse_district_from_address(address_en), "private_site": 1})

	private_df = pd.DataFrame(private_rows)
	if private_df.empty:
		private_grouped = pd.DataFrame(columns=["district_norm", "private_sites"])
	else:
		private_grouped = (
			private_df[private_df["district_norm"] != ""]
			.groupby("district_norm", as_index=False)
			.agg(private_sites=("private_site", "sum"))
		)

	out = housing_grouped.merge(private_grouped, how="outer", on="district_norm")
	out["housing_units_proxy"] = out["housing_units_proxy"].fillna(0.0)
	out["housing_sites"] = out["housing_sites"].fillna(0.0)
	out["private_sites"] = out["private_sites"].fillna(0.0)
	out["population_proxy"] = out["housing_units_proxy"] + out["private_sites"] * 400.0
	out["density_proxy"] = out["population_proxy"] / (out["housing_sites"] + out["private_sites"]).replace(0, pd.NA)
	out["district"] = out["district_norm"].map(district_display)
	return out


@st.cache_data(show_spinner=False)
def load_recycle_points_by_district() -> tuple[pd.DataFrame, pd.DataFrame]:
	df = load_csv(RECYCLE_POINTS_CSV)
	df["district_norm"] = df["district_id"].map(normalize_district)
	df["district"] = df["district_norm"].map(district_display)
	df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
	df["lgt"] = pd.to_numeric(df["lgt"], errors="coerce")
	df["coord_valid"] = df["lat"].between(22.0, 23.0) & df["lgt"].between(113.7, 114.5)

	grouped = (
		df.groupby(["district_norm", "district"], as_index=False)
		.agg(
			recycle_points_total=("cp_id", "count"),
			recycle_points_accepted=("cp_state", lambda s: int((s == "Accepted").sum())),
		)
	)
	return grouped, df


@st.cache_data(show_spinner=False)
def build_district_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
	waste_df = to_numeric(load_csv(DISTRICT_WASTE_CSV), ["municipal_solid_waste_tpd"])
	waste_df["district_norm"] = waste_df["district"].map(normalize_district)
	waste_df["district"] = waste_df["district_norm"].map(district_display)

	allowed_norms = {normalize_district(k) for k in DISTRICT_CENTROIDS.keys()}
	waste_df = waste_df[waste_df["district_norm"].isin(allowed_norms)].copy()
	waste_df = waste_df[["district_norm", "district", "municipal_solid_waste_tpd"]]

	population_df = load_population_proxy_by_district()
	recycle_grouped, recycle_points_raw = load_recycle_points_by_district()

	merged = (
		waste_df.merge(
			population_df[["district_norm", "population_proxy", "density_proxy"]],
			how="left",
			on="district_norm",
		)
		.merge(
			recycle_grouped[["district_norm", "recycle_points_total", "recycle_points_accepted"]],
			how="left",
			on="district_norm",
		)
	)

	merged["population_proxy"] = merged["population_proxy"].fillna(0.0)
	merged["density_proxy"] = merged["density_proxy"].fillna(0.0)
	merged["recycle_points_total"] = merged["recycle_points_total"].fillna(0).astype(int)
	merged["recycle_points_accepted"] = merged["recycle_points_accepted"].fillna(0).astype(int)

	merged["waste_per_person_kg_day_proxy"] = (
		merged["municipal_solid_waste_tpd"] * 1000.0 / merged["population_proxy"].replace(0, pd.NA)
	)
	merged["recycle_point_pressure_kg_day"] = (
		merged["municipal_solid_waste_tpd"] * 1000.0 / merged["recycle_points_accepted"].replace(0, pd.NA)
	)
	merged["points_per_10k_proxy_people"] = (
		merged["recycle_points_accepted"] / merged["population_proxy"].replace(0, pd.NA) * 10000
	)

	for district_name, (lat, lon) in DISTRICT_CENTROIDS.items():
		norm = normalize_district(district_name)
		merged.loc[merged["district_norm"] == norm, "lat"] = lat
		merged.loc[merged["district_norm"] == norm, "lon"] = lon

	return merged.sort_values("waste_per_person_kg_day_proxy", ascending=False), recycle_points_raw


def color_scale(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
	if pd.isna(value) or vmax <= vmin:
		return (140, 140, 140)
	ratio = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
	r = int(70 + ratio * 185)
	g = int(180 - ratio * 130)
	b = int(120 - ratio * 80)
	return (r, g, b)


def render_waste_per_person_map(df: pd.DataFrame) -> None:
	map_df = df.dropna(subset=["lat", "lon"]).copy()
	if map_df.empty:
		st.warning("No district map rows available.")
		return

	vmin = float(map_df["waste_per_person_kg_day_proxy"].min(skipna=True))
	vmax = float(map_df["waste_per_person_kg_day_proxy"].max(skipna=True))
	map_df[["r", "g", "b"]] = map_df["waste_per_person_kg_day_proxy"].apply(
		lambda v: pd.Series(color_scale(v, vmin, vmax))
	)
	map_df["radius"] = (map_df["municipal_solid_waste_tpd"].fillna(0) * 7.5).clip(lower=750)

	layer = pdk.Layer(
		"ScatterplotLayer",
		map_df,
		get_position="[lon, lat]",
		get_radius="radius",
		get_fill_color="[r, g, b, 195]",
		get_line_color=[30, 30, 30, 120],
		line_width_min_pixels=1,
		stroked=True,
		pickable=True,
	)
	text_layer = pdk.Layer(
		"TextLayer",
		map_df,
		get_position="[lon, lat]",
		get_text="district",
		get_size=12,
		get_color=[35, 35, 35, 200],
		get_alignment_baseline="bottom",
		get_pixel_offset=[0, -8],
	)

	st.pydeck_chart(
		pdk.Deck(
			map_provider="carto",
			map_style="light",
			initial_view_state=pdk.ViewState(latitude=22.33, longitude=114.15, zoom=9.5, pitch=0),
			layers=[layer, text_layer],
			tooltip={  # type: ignore[arg-type]
				"html": "<b>{district}</b><br/>MSW: {municipal_solid_waste_tpd} tpd<br/>Waste per person (proxy): {waste_per_person_kg_day_proxy} kg/day",
				"style": {"backgroundColor": "#10222f", "color": "white"},
			},
		),
		width="stretch",
	)


def render_recycle_pressure_map(points_df: pd.DataFrame, district_df: pd.DataFrame) -> None:
	merge_cols = district_df[["district_norm", "recycle_point_pressure_kg_day"]]
	map_df = points_df.merge(merge_cols, how="left", on="district_norm")
	map_df = map_df[map_df["coord_valid"]].copy()
	if map_df.empty:
		st.warning("No valid recycle-point coordinates for pressure mapping.")
		return

	vmin = float(map_df["recycle_point_pressure_kg_day"].min(skipna=True))
	vmax = float(map_df["recycle_point_pressure_kg_day"].max(skipna=True))
	map_df[["r", "g", "b"]] = map_df["recycle_point_pressure_kg_day"].apply(
		lambda v: pd.Series(color_scale(v, vmin, vmax))
	)

	layer = pdk.Layer(
		"ScatterplotLayer",
		map_df,
		get_position="[lgt, lat]",
		get_radius=55,
		get_fill_color="[r, g, b, 170]",
		pickable=True,
	)

	st.pydeck_chart(
		pdk.Deck(
			map_provider="carto",
			map_style="light",
			initial_view_state=pdk.ViewState(latitude=22.33, longitude=114.15, zoom=10.2, pitch=0),
			layers=[layer],
			tooltip={  # type: ignore[arg-type]
				"html": "<b>{district}</b><br/>Point ID: {cp_id}<br/>State: {cp_state}<br/>Estimated pressure: {recycle_point_pressure_kg_day} kg/day",
				"style": {"backgroundColor": "#10222f", "color": "white"},
			},
		),
		width="stretch",
	)


def build_waste_type_estimates(district_df: pd.DataFrame, composition_df: pd.DataFrame) -> pd.DataFrame:
	comp = composition_df[["waste_type", "total_msw_tpd", "domestic_tpd", "commercial_industrial_tpd"]].copy()
	total = float(comp["total_msw_tpd"].sum())
	comp["share"] = comp["total_msw_tpd"] / total if total > 0 else 0.0

	rows: list[dict[str, object]] = []
	for _, drow in district_df.iterrows():
		for _, crow in comp.iterrows():
			rows.append(
				{
					"district": drow["district"],
					"district_norm": drow["district_norm"],
					"lat": drow["lat"],
					"lon": drow["lon"],
					"waste_type": crow["waste_type"],
					"type_share": crow["share"],
					"estimated_tpd": float(drow["municipal_solid_waste_tpd"]) * float(crow["share"]),
				}
			)
	return pd.DataFrame(rows)


def render_waste_type_analysis(composition_df: pd.DataFrame, district_df: pd.DataFrame) -> None:
	st.subheader("Type of trash disposed")
	c1, c2 = st.columns(2)
	with c1:
		st.caption("Total disposed by waste type (MSW, tpd)")
		bar = composition_df.set_index("waste_type")["total_msw_tpd"].sort_values(ascending=False)
		st.bar_chart(bar)
	with c2:
		st.caption("Domestic vs C&I by waste type")
		stacked = composition_df.set_index("waste_type")[["domestic_tpd", "commercial_industrial_tpd"]]
		st.bar_chart(stacked)

	est = build_waste_type_estimates(district_df, composition_df)
	if est.empty:
		st.info("No district-level rows available for type-estimate mapping.")
		return

	waste_options = composition_df["waste_type"].tolist()
	selected_waste = st.selectbox("Estimated district map for waste type", waste_options)
	map_df = est[est["waste_type"] == selected_waste].copy()
	vmin = float(map_df["estimated_tpd"].min(skipna=True))
	vmax = float(map_df["estimated_tpd"].max(skipna=True))
	map_df[["r", "g", "b"]] = map_df["estimated_tpd"].apply(lambda v: pd.Series(color_scale(v, vmin, vmax)))
	map_df["radius"] = (map_df["estimated_tpd"].fillna(0) * 14.0).clip(lower=600)

	layer = pdk.Layer(
		"ScatterplotLayer",
		map_df,
		get_position="[lon, lat]",
		get_radius="radius",
		get_fill_color="[r, g, b, 190]",
		pickable=True,
	)

	st.pydeck_chart(
		pdk.Deck(
			map_provider="carto",
			map_style="light",
			initial_view_state=pdk.ViewState(latitude=22.33, longitude=114.15, zoom=9.5, pitch=0),
			layers=[layer],
			tooltip={  # type: ignore[arg-type]
				"html": "<b>{district}</b><br/>Waste type: {waste_type}<br/>Estimated disposed: {estimated_tpd} tpd",
				"style": {"backgroundColor": "#10222f", "color": "white"},
			},
		),
		width="stretch",
	)


def render_consistency_check(msw_csv: pd.DataFrame, msw_json: pd.DataFrame) -> None:
	numeric_cols = ["domestic_tpd", "commercial_industrial_tpd", "total_msw_tpd"]
	merged = msw_csv.merge(msw_json, on="waste_type", suffixes=("_csv", "_json"), how="outer")
	for col in numeric_cols:
		merged[f"abs_diff_{col}"] = (merged[f"{col}_csv"] - merged[f"{col}_json"]).abs()

	st.caption("CSV vs JSON consistency (MSW composition)")
	st.bar_chart(merged.set_index("waste_type")[[f"abs_diff_{c}" for c in numeric_cols]])


def main() -> None:
	st.title("Hong Kong Waste Pressure Analyzer")
	st.caption("Construction waste is excluded. Analysis focuses on municipal solid waste, per-person proxy, and recycle-point pressure.")

	required_files = [
		DISTRICT_WASTE_CSV,
		MSW_COMPOSITION_CSV,
		MSW_COMPOSITION_JSON,
		RECYCLABLES_CSV,
		RECYCLE_POINTS_CSV,
		PRH_JSON,
		HOS_JSON,
		PRIVATE_GEOJSON,
	]
	missing = [p.name for p in required_files if not p.exists()]
	if missing:
		st.error(f"Missing files: {', '.join(missing)}")
		return

	district_df, recycle_points_raw = build_district_analysis()
	msw_comp_df = to_numeric(load_csv(MSW_COMPOSITION_CSV), ["domestic_tpd", "commercial_industrial_tpd", "total_msw_tpd"])
	msw_comp_json = to_numeric(load_json(MSW_COMPOSITION_JSON), ["domestic_tpd", "commercial_industrial_tpd", "total_msw_tpd"])
	recyclables_df = to_numeric(
		load_csv(RECYCLABLES_CSV),
		[
			"delivered_outside_hk_thousand_tonnes",
			"recycled_locally_thousand_tonnes",
			"total_recovered_thousand_tonnes",
		],
	)

	valid_rows = district_df.dropna(subset=["waste_per_person_kg_day_proxy"]).copy()
	peak_per_person_district = "N/A"
	if not valid_rows.empty:
		peak_per_person_district = str(valid_rows.sort_values("waste_per_person_kg_day_proxy", ascending=False).iloc[0]["district"])

	k1, k2, k3, k4 = st.columns(4)
	k1.metric("Total MSW (tpd)", f"{district_df['municipal_solid_waste_tpd'].sum():,.0f}")
	k2.metric("Districts merged", f"{district_df['district_norm'].nunique()}")
	k3.metric("Highest waste/person district", peak_per_person_district)
	k4.metric("Accepted recycle points", f"{int(district_df['recycle_points_accepted'].sum()):,}")

	st.subheader("Maps")
	left, right = st.columns(2)
	with left:
		st.caption("Waste produced per person (proxy) by district")
		render_waste_per_person_map(district_df)
	with right:
		st.caption("Pressure on each recycle point (from district load)")
		render_recycle_pressure_map(recycle_points_raw, district_df)

	st.subheader("Per-person and recycle-point pressure rankings")
	chart_cols = [
		"district",
		"municipal_solid_waste_tpd",
		"population_proxy",
		"waste_per_person_kg_day_proxy",
		"recycle_points_accepted",
		"recycle_point_pressure_kg_day",
		"points_per_10k_proxy_people",
	]
	ranking = district_df[chart_cols].sort_values("waste_per_person_kg_day_proxy", ascending=False)

	c1, c2 = st.columns(2)
	with c1:
		st.caption("Waste produced per person (proxy, kg/day)")
		st.bar_chart(ranking.set_index("district")["waste_per_person_kg_day_proxy"])
	with c2:
		st.caption("Recycle point pressure (kg/day per accepted point)")
		st.bar_chart(ranking.set_index("district")["recycle_point_pressure_kg_day"])

	st.dataframe(ranking, width="stretch", hide_index=True)

	render_waste_type_analysis(msw_comp_df, district_df)

	st.subheader("Recovered recyclables")
	recycle_chart = recyclables_df.set_index("recyclable_type")[["delivered_outside_hk_thousand_tonnes", "recycled_locally_thousand_tonnes"]]
	st.bar_chart(recycle_chart)

	render_consistency_check(msw_comp_df, msw_comp_json)

	with st.expander("Assumptions used in this analysis"):
		st.markdown(
			"- Population and density are proxy metrics from housing units and private-site counts.\n"
			"- Per-person figures are estimated using that proxy rather than census district population.\n"
			"- District-level waste type maps are estimated by applying citywide composition shares to each district's MSW total.\n"
			"- Construction waste is intentionally excluded as requested."
		)


if __name__ == "__main__":
	main()
