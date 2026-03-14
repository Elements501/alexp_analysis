from __future__ import annotations

import json
import math
import re
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
RECYCLE_CSV = BASE_DIR / "recycle_points" / "wasteless250918.csv"
PRH_JSON = BASE_DIR / "housing" / "prh-estates.json"
HOS_JSON = BASE_DIR / "housing" / "hos-courts.json"
PRIVATE_GEOJSON = BASE_DIR / "housing" / "private.geojson"
WASTE_FACILITY_GEOJSON = BASE_DIR / "waste_management" / "EPDWMF_20260228.gdb_converted.geojson"
OPEN_RECYCLE_GEOJSON = BASE_DIR / "open_recycle" / "OSDRS_converted.geojson"
SHOPPING_JSON = BASE_DIR / "housing" / "shopping-centres.json"
FLATTED_FACTORY_JSON = BASE_DIR / "housing" / "flatted-factory.json"


DISTRICT_DISPLAY = {
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
	"SHA TIN": "Sha Tin",
	"SHATIN": "Sha Tin",
	"TAIPO": "Tai Po",
	"TSUENWAN": "Tsuen Wan",
	"TUENMUN": "Tuen Mun",
	"YUENLONG": "Yuen Long",
}


def normalize_district(value: object) -> str:
	text = str(value or "").upper()
	text = re.sub(r"[^A-Z0-9]", "", text)
	return text


def district_display(norm: str) -> str:
	if norm in DISTRICT_DISPLAY:
		return DISTRICT_DISPLAY[norm]
	if not norm:
		return "Unknown"
	return norm.title()


def parse_number(value: object) -> float:
	text = str(value or "")
	text = text.replace(",", "")
	text = re.sub(r"[^0-9.\\-]", "", text)
	if text in {"", "-", "."}:
		return math.nan
	try:
		return float(text)
	except ValueError:
		return math.nan


@st.cache_data(show_spinner=False)
def load_recycle_points() -> pd.DataFrame:
	df = pd.read_csv(RECYCLE_CSV, encoding="utf-8-sig")
	df["district_norm"] = df["district_id"].map(normalize_district)
	df["district"] = df["district_norm"].map(district_display)
	df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
	df["lgt"] = pd.to_numeric(df["lgt"], errors="coerce")
	df["is_valid_coord"] = (
		df["lat"].between(22.1, 22.6)
		& df["lgt"].between(113.8, 114.5)
	)
	return df


def _extract_nested_text(record: dict, field: str) -> str:
	value = record.get(field)
	if isinstance(value, dict):
		return str(value.get("en", ""))
	return str(value or "")


@st.cache_data(show_spinner=False)
def load_housing_units() -> pd.DataFrame:
	records: list[dict[str, object]] = []

	with PRH_JSON.open("r", encoding="utf-8") as f:
		prh = json.load(f)
	for item in prh:
		district_name = _extract_nested_text(item, "District Name")
		units = parse_number(_extract_nested_text(item, "No. of Rental Flats"))
		records.append(
			{
				"dataset": "PRH",
				"district_norm": normalize_district(district_name),
				"district": district_name,
				"units": units,
				"lat": pd.to_numeric(item.get("Estate Map Latitude"), errors="coerce"),
				"lon": pd.to_numeric(item.get("Estate Map Longitude"), errors="coerce"),
			}
		)

	with HOS_JSON.open("r", encoding="utf-8") as f:
		hos = json.load(f)
	for item in hos:
		district_name = _extract_nested_text(item, "District Name")
		units = parse_number(_extract_nested_text(item, "No. of Flats"))
		records.append(
			{
				"dataset": "HOS",
				"district_norm": normalize_district(district_name),
				"district": district_name,
				"units": units,
				"lat": pd.to_numeric(item.get("Estate Map Latitude"), errors="coerce"),
				"lon": pd.to_numeric(item.get("Estate Map Longitude"), errors="coerce"),
			}
		)

	housing_df = pd.DataFrame(records)
	housing_df["district"] = housing_df["district_norm"].map(district_display)
	return housing_df


def _build_private_district_patterns() -> list[tuple[re.Pattern[str], str]]:
	names = sorted(
		{v for v in DISTRICT_DISPLAY.values()},
		key=len,
		reverse=True,
	)
	patterns: list[tuple[re.Pattern[str], str]] = []
	for name in names:
		patt = re.compile(rf"\\b{re.escape(name.upper())}\\b")
		patterns.append((patt, normalize_district(name)))
	return patterns


PRIVATE_DISTRICT_PATTERNS = _build_private_district_patterns()


def parse_district_from_text(text: str) -> str:
	upper_text = str(text or "").upper()
	for pattern, district_norm in PRIVATE_DISTRICT_PATTERNS:
		if pattern.search(upper_text):
			return district_norm
	return ""


def parse_private_district(address_en: str) -> str:
	return parse_district_from_text(address_en)


@st.cache_data(show_spinner=False)
def load_waste_facilities_by_district() -> tuple[pd.DataFrame, float]:
	with WASTE_FACILITY_GEOJSON.open("r", encoding="utf-8") as f:
		geo = json.load(f)

	rows: list[dict[str, object]] = []
	parsed_count = 0
	for feat in geo.get("features", []):
		props = feat.get("properties", {})
		address = str(props.get("ADDRESS_EN", ""))
		facility_type = str(props.get("SEARCH01_EN", "Unknown"))
		district_norm = parse_district_from_text(address)
		if district_norm:
			parsed_count += 1
		rows.append(
			{
				"district_norm": district_norm,
				"waste_facility_type": facility_type,
			}
		)

	df = pd.DataFrame(rows)
	df = df[df["district_norm"] != ""].copy()
	if df.empty:
		return pd.DataFrame(columns=["district_norm", "waste_facilities", "waste_facility_types"]), 0.0

	out = (
		df.groupby("district_norm", dropna=False)
		.agg(
			waste_facilities=("district_norm", "count"),
			waste_facility_types=("waste_facility_type", "nunique"),
		)
		.reset_index()
	)
	parse_rate = parsed_count / max(1, len(rows))
	return out, float(parse_rate)


@st.cache_data(show_spinner=False)
def load_green_hubs_by_district() -> pd.DataFrame:
	with OPEN_RECYCLE_GEOJSON.open("r", encoding="utf-8") as f:
		geo = json.load(f)

	rows: list[dict[str, object]] = []
	for feat in geo.get("features", []):
		props = feat.get("properties", {})
		name = str(props.get("BLDG_ENGNM", ""))
		if "GREEN@" not in name.upper():
			continue
		name_part = name.upper().split("GREEN@", maxsplit=1)[-1].strip()
		district_norm = normalize_district(name_part)
		rows.append({"district_norm": district_norm})

	df = pd.DataFrame(rows)
	if df.empty:
		return pd.DataFrame(columns=["district_norm", "green_hubs"])
	return (
		df.groupby("district_norm", dropna=False)
		.agg(green_hubs=("district_norm", "count"))
		.reset_index()
	)


@st.cache_data(show_spinner=False)
def load_shopping_centres_by_district() -> pd.DataFrame:
	with SHOPPING_JSON.open("r", encoding="utf-8") as f:
		rows = json.load(f)

	data: list[dict[str, object]] = []
	for item in rows:
		district = _extract_nested_text(item, "District Name")
		data.append({"district_norm": normalize_district(district)})

	df = pd.DataFrame(data)
	if df.empty:
		return pd.DataFrame(columns=["district_norm", "shopping_centres"])
	return (
		df.groupby("district_norm", dropna=False)
		.agg(shopping_centres=("district_norm", "count"))
		.reset_index()
	)


@st.cache_data(show_spinner=False)
def load_flatted_factories_by_district() -> pd.DataFrame:
	with FLATTED_FACTORY_JSON.open("r", encoding="utf-8") as f:
		rows = json.load(f)

	data: list[dict[str, object]] = []
	for item in rows:
		district = _extract_nested_text(item, "District Name")
		data.append({"district_norm": normalize_district(district)})

	df = pd.DataFrame(data)
	if df.empty:
		return pd.DataFrame(columns=["district_norm", "flatted_factories"])
	return (
		df.groupby("district_norm", dropna=False)
		.agg(flatted_factories=("district_norm", "count"))
		.reset_index()
	)


@st.cache_data(show_spinner=False)
def load_private_sites() -> pd.DataFrame:
	with PRIVATE_GEOJSON.open("r", encoding="utf-8") as f:
		geo = json.load(f)

	rows: list[dict[str, object]] = []
	for feat in geo.get("features", []):
		props = feat.get("properties", {})
		geom = feat.get("geometry", {})
		coords = geom.get("coordinates", [None, None])
		lon = parse_number(coords[0] if len(coords) > 0 else None)
		lat = parse_number(coords[1] if len(coords) > 1 else None)
		addr_en = str(props.get("ADDR_OF_BUILDING_IN_ENG", ""))
		district_norm = parse_private_district(addr_en)
		rows.append(
			{
				"district_norm": district_norm,
				"district": district_display(district_norm),
				"lat": lat,
				"lon": lon,
				"address_en": addr_en,
			}
		)

	return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_district_model() -> tuple[pd.DataFrame, dict[str, float]]:
	recycle = load_recycle_points()
	housing = load_housing_units()
	private = load_private_sites()
	waste_facilities, waste_parse_rate = load_waste_facilities_by_district()
	green_hubs = load_green_hubs_by_district()
	shopping = load_shopping_centres_by_district()
	flatted = load_flatted_factories_by_district()

	recycle_by_district = (
		recycle.groupby(["district_norm", "district"], dropna=False)
		.agg(
			recycle_bins=("cp_id", "count"),
			recycle_bins_accepted=("cp_state", lambda s: int((s == "Accepted").sum())),
			centroid_lat=("lat", "mean"),
			centroid_lon=("lgt", "mean"),
			valid_coord_ratio=("is_valid_coord", "mean"),
		)
		.reset_index()
	)

	housing_by_district = (
		housing.groupby(["district_norm", "district"], dropna=False)
		.agg(
			housing_sites=("district_norm", "count"),
			housing_units_proxy=("units", "sum"),
		)
		.reset_index()
	)

	private_known = private[private["district_norm"] != ""].copy()
	private_by_district = (
		private_known.groupby(["district_norm", "district"], dropna=False)
		.agg(private_sites=("district_norm", "count"))
		.reset_index()
	)

	combined = recycle_by_district.merge(
		housing_by_district[["district_norm", "housing_sites", "housing_units_proxy"]],
		how="left",
		on="district_norm",
	).merge(
		private_by_district[["district_norm", "private_sites"]],
		how="left",
		on="district_norm",
	).merge(
		waste_facilities,
		how="left",
		on="district_norm",
	).merge(
		green_hubs,
		how="left",
		on="district_norm",
	).merge(
		shopping,
		how="left",
		on="district_norm",
	).merge(
		flatted,
		how="left",
		on="district_norm",
	)

	combined["housing_sites"] = combined["housing_sites"].fillna(0).astype(int)
	combined["housing_units_proxy"] = combined["housing_units_proxy"].fillna(0.0)
	combined["private_sites"] = combined["private_sites"].fillna(0).astype(int)
	combined["waste_facilities"] = combined["waste_facilities"].fillna(0).astype(int)
	combined["waste_facility_types"] = combined["waste_facility_types"].fillna(0).astype(int)
	combined["green_hubs"] = combined["green_hubs"].fillna(0).astype(int)
	combined["shopping_centres"] = combined["shopping_centres"].fillna(0).astype(int)
	combined["flatted_factories"] = combined["flatted_factories"].fillna(0).astype(int)

	combined["population_pressure_proxy"] = (
		combined["housing_units_proxy"]
		+ combined["private_sites"] * 400
	)
	combined["bins_per_10k_pressure"] = (
		combined["recycle_bins"] / combined["population_pressure_proxy"].replace(0, math.nan) * 10000
	)
	combined["need_score_pressure_per_bin"] = (
		combined["population_pressure_proxy"] / combined["recycle_bins"].replace(0, math.nan)
	)
	combined = combined.sort_values("recycle_bins", ascending=False).reset_index(drop=True)

	qc = {
		"recycle_missing_district": float((recycle["district_norm"] == "").sum()),
		"recycle_invalid_coords": float((~recycle["is_valid_coord"]).sum()),
		"private_total": float(len(private)),
		"private_district_parsed": float((private["district_norm"] != "").sum()),
		"district_rows": float(len(combined)),
		"rows_with_zero_pressure": float((combined["population_pressure_proxy"] <= 0).sum()),
		"waste_facility_parse_rate": float(waste_parse_rate),
	}

	return combined, qc

def render_scatter(df: pd.DataFrame) -> None:
	chart_df = df[
		[
			"district",
			"recycle_bins",
			"population_pressure_proxy",
			"bins_per_10k_pressure",
		]
	].copy()
	chart_df = chart_df[chart_df["population_pressure_proxy"] > 0]

	st.subheader("Relationship: recycle bins vs population-pressure proxy")
	if chart_df.empty:
		st.warning("Not enough merged district data to render relationship chart.")
		return

	base = alt.Chart(chart_df).encode(
		x=alt.X("population_pressure_proxy:Q", title="Population-pressure proxy"),
		y=alt.Y("recycle_bins:Q", title="Recycle bins"),
		tooltip=[
			alt.Tooltip("district:N", title="District"),
			alt.Tooltip("recycle_bins:Q", title="Bins", format=",.0f"),
			alt.Tooltip("population_pressure_proxy:Q", title="Pressure proxy", format=",.0f"),
			alt.Tooltip("bins_per_10k_pressure:Q", title="Bins per 10k pressure", format=".2f"),
		],
	)

	points = base.mark_circle(color="#0D6EFD", opacity=0.75).encode(
		size=alt.Size("bins_per_10k_pressure:Q", title="Bins per 10k pressure", scale=alt.Scale(range=[40, 400]))
	)
	labels = base.mark_text(align="left", baseline="middle", dx=7, fontSize=11).encode(text="district:N")

	st.altair_chart((points + labels).interactive(), use_container_width=True)


def render_district_map(df: pd.DataFrame) -> None:
	map_df = df.dropna(subset=["centroid_lat", "centroid_lon"]).copy()
	if map_df.empty:
		st.warning("No district centroid coordinates available for map.")
		return

	map_df["radius"] = map_df["recycle_bins"].clip(lower=1).pow(0.5) * 160
	map_df["radius"] = map_df["radius"].clip(40, 240)
	map_df["tooltip_bins_per_10k"] = map_df["bins_per_10k_pressure"].round(2)

	# Slight longitude offset prevents text from sitting directly on top of bubbles.
	map_df["label_lon"] = map_df["centroid_lon"] + 0.018
	map_df["label_lat"] = map_df["centroid_lat"]

	layer = pdk.Layer(
		"ScatterplotLayer",
		data=map_df,
		get_position="[centroid_lon, centroid_lat]",
		get_radius="radius",
		get_fill_color="[13, 110, 253, 150]",
		pickable=True,
		stroked=True,
		get_line_color=[20, 20, 20, 180],
		line_width_min_pixels=1,
	)
	text_layer = pdk.Layer(
		"TextLayer",
		data=map_df,
		get_position="[label_lon, label_lat]",
		get_text="district",
		get_size=13,
		get_color=[20, 20, 20, 230],
		get_angle=0,
		get_text_anchor="start",
		get_alignment_baseline="center",
		pickable=False,
	)

	view_state = pdk.ViewState(
		latitude=float(map_df["centroid_lat"].mean()),
		longitude=float(map_df["centroid_lon"].mean()),
		zoom=9,
		pitch=0,
	)

	st.subheader("Map: district-level recycle-bin concentration")
	st.pydeck_chart(
		pdk.Deck(
			layers=[layer, text_layer],
			initial_view_state=view_state,
			tooltip={
				"html": "<b>{district}</b><br/>Bins: {recycle_bins}<br/>Pressure proxy: {population_pressure_proxy}<br/>Bins / 10k pressure: {tooltip_bins_per_10k}",
				"style": {"backgroundColor": "#222", "color": "#fff"},
			},  # type: ignore[arg-type]
		),
		use_container_width=True,
	)


def render_repo_correlations(district_df: pd.DataFrame) -> None:
	st.subheader("Useful repo-dataset correlations vs bin-need score")
	st.caption(
		"Need score = population-pressure proxy / recycle bins. Higher value means a district likely needs more bins."
	)

	metrics = [
		"waste_facilities",
		"waste_facility_types",
		"green_hubs",
		"shopping_centres",
		"flatted_factories",
		"private_sites",
		"housing_sites",
	]

	rows: list[dict[str, object]] = []
	for metric in metrics:
		tmp = district_df[["need_score_pressure_per_bin", metric]].dropna().copy()
		if len(tmp) < 3:
			rows.append(
				{
					"metric": metric,
					"n": len(tmp),
					"pearson_r": math.nan,
					"spearman_rho": math.nan,
				}
			)
			continue

		pearson = tmp["need_score_pressure_per_bin"].corr(tmp[metric], method="pearson")
		x_rank = tmp["need_score_pressure_per_bin"].rank(method="average")
		y_rank = tmp[metric].rank(method="average")
		spearman = x_rank.corr(y_rank, method="pearson")
		rows.append(
			{
				"metric": metric,
				"n": len(tmp),
				"pearson_r": pearson,
				"spearman_rho": spearman,
			}
		)

	corr_table = pd.DataFrame(rows)
	corr_table["abs_spearman"] = corr_table["spearman_rho"].abs()
	corr_table = corr_table.sort_values("abs_spearman", ascending=False)
	st.dataframe(
		corr_table[["metric", "n", "pearson_r", "spearman_rho"]],
		use_container_width=True,
		hide_index=True,
	)

	selected_metric = st.selectbox(
		"Choose repo metric for labeled scatter",
		options=metrics,
		index=0,
	)
	plot_df = district_df[["district", "need_score_pressure_per_bin", selected_metric]].dropna().copy()
	if plot_df.empty:
		st.warning("No data available for the selected metric.")
		return

	base = alt.Chart(plot_df).encode(
		x=alt.X(f"{selected_metric}:Q", title=selected_metric.replace("_", " ").title()),
		y=alt.Y("need_score_pressure_per_bin:Q", title="Need score (pressure per bin)"),
		tooltip=[
			alt.Tooltip("district:N", title="District"),
			alt.Tooltip("need_score_pressure_per_bin:Q", title="Need score", format=",.0f"),
			alt.Tooltip(f"{selected_metric}:Q", title=selected_metric.replace("_", " ").title(), format=",.2f"),
		],
	)
	points = base.mark_circle(color="#8C2F39", opacity=0.8, size=100)
	labels = base.mark_text(dx=7, dy=-2, fontSize=11, color="#222").encode(text="district:N")
	st.altair_chart((points + labels).interactive(), use_container_width=True)


def render_critical_errors(qc: dict[str, float], district_df: pd.DataFrame) -> None:
	st.subheader("Critical errors and reliability warnings")
	warnings: list[str] = []

	if qc["recycle_missing_district"] > 0:
		warnings.append(
			f"Recycle dataset has {int(qc['recycle_missing_district'])} rows with missing district IDs."
		)
	if qc["recycle_invalid_coords"] > 0:
		warnings.append(
			f"Recycle dataset has {int(qc['recycle_invalid_coords'])} rows with out-of-range or missing coordinates."
		)

	private_total = int(qc["private_total"])
	private_parsed = int(qc["private_district_parsed"])
	if private_total > 0:
		parse_rate = private_parsed / private_total
		if parse_rate < 0.8:
			warnings.append(
				f"Only {private_parsed}/{private_total} private-housing points were assigned to districts from address text."
			)

	if int(qc["rows_with_zero_pressure"]) > 0:
		warnings.append(
			"Some districts have zero population-pressure proxy, so bins-per-pressure cannot be computed there."
		)

	if qc["waste_facility_parse_rate"] < 0.7:
		warnings.append(
			f"Only {qc['waste_facility_parse_rate']:.0%} of waste-facility addresses were mapped to districts."
		)

	analysis_rows = district_df[district_df["population_pressure_proxy"] > 0]
	if len(analysis_rows) < 10:
		warnings.append(
			f"Only {len(analysis_rows)} districts are usable for ratio/correlation analysis; results may be unstable."
		)

	warnings.append(
		"Population density is proxied using housing units and private-site counts, not official census density."
	)

	for item in warnings:
		st.error(item)


def render_correlation(district_df: pd.DataFrame) -> None:
	st.subheader("Correlation outputs")
	corr_df = district_df[district_df["population_pressure_proxy"] > 0].copy()

	if len(corr_df) < 3:
		st.warning("Not enough districts for meaningful correlation statistics.")
		return

	pearson = corr_df["recycle_bins"].corr(corr_df["population_pressure_proxy"], method="pearson")
	# Spearman rho without SciPy: Pearson correlation on rank-transformed values.
	x_rank = corr_df["recycle_bins"].rank(method="average")
	y_rank = corr_df["population_pressure_proxy"].rank(method="average")
	spearman = x_rank.corr(y_rank, method="pearson")

	k1, k2, k3 = st.columns(3)
	k1.metric("Districts in model", f"{len(corr_df)}")
	k2.metric("Pearson r", f"{pearson:.3f}" if pd.notna(pearson) else "N/A")
	k3.metric("Spearman rho", f"{spearman:.3f}" if pd.notna(spearman) else "N/A")

	st.caption(
		"Interpretation: positive values suggest districts with higher population-pressure proxy tend to have more recycle bins."
	)


def main() -> None:
	st.set_page_config(page_title="Recycle Bins vs Population Density", layout="wide")
	st.title("Recycle-bin effectiveness vs population density (proxy)")
	st.write(
		"This app estimates whether recycle-bin distribution aligns with population pressure across Hong Kong districts."
	)

	district_df, qc = build_district_model()

	m1, m2, m3 = st.columns(3)
	m1.metric("Total recycle bins", f"{int(district_df['recycle_bins'].sum()):,}")
	m2.metric("Districts represented", f"{int(qc['district_rows'])}")
	m3.metric(
		"Avg bins per 10k pressure",
		f"{district_df['bins_per_10k_pressure'].mean(skipna=True):.2f}",
	)

	render_correlation(district_df)
	render_scatter(district_df)
	render_district_map(district_df)
	render_repo_correlations(district_df)
	render_critical_errors(qc, district_df)

	st.subheader("District-level merged table")
	show_cols = [
		"district",
		"recycle_bins",
		"recycle_bins_accepted",
		"housing_sites",
		"housing_units_proxy",
		"private_sites",
		"population_pressure_proxy",
		"bins_per_10k_pressure",
		"need_score_pressure_per_bin",
		"waste_facilities",
		"waste_facility_types",
		"green_hubs",
		"shopping_centres",
		"flatted_factories",
		"valid_coord_ratio",
	]
	st.dataframe(
		district_df[show_cols].sort_values("bins_per_10k_pressure", ascending=False),
		use_container_width=True,
		hide_index=True,
	)


if __name__ == "__main__":
	main()
