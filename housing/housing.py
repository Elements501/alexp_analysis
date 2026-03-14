import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(page_title="Housing Dataset Analyzer", layout="wide")

DATA_DIR = Path(__file__).parent
HK_LAT_RANGE = (22.0, 23.0)
HK_LON_RANGE = (113.7, 114.5)

DATASET_ORDER = [
	("prh-estates.json", "Public Rental Housing Estates"),
	("hos-courts.json", "Home Ownership Scheme Courts"),
	("shopping-centres.json", "Shopping Centres"),
	("flatted-factory.json", "Flatted Factory Estates"),
	("private.geojson", "Private Housing (Owners' Committees)"),
]

DATASET_COLORS = {
	"Public Rental Housing Estates": [52, 152, 219, 190],
	"Home Ownership Scheme Courts": [46, 204, 113, 190],
	"Shopping Centres": [241, 196, 15, 190],
	"Flatted Factory Estates": [155, 89, 182, 190],
	"Private Housing (Owners' Committees)": [231, 76, 60, 190],
}


def extract_en(value: Any) -> str:
	if isinstance(value, dict):
		if "en" in value and value["en"] is not None:
			return str(value["en"]).strip()
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
	match = re.search(r"(\d{1,3}(?:[ ,]\d{3})*(?:\.\d+)?)", text)
	if not match:
		return float("nan")
	num_text = match.group(1).replace(" ", "").replace(",", "")
	return parse_float(num_text)


def parse_year(value: Any) -> float:
	text = extract_en(value)
	match = re.search(r"(19\d{2}|20\d{2})", text)
	if not match:
		return float("nan")
	return parse_float(match.group(1))


@st.cache_data(show_spinner=False)
def load_dataset(file_path: Path, dataset_title: str) -> pd.DataFrame:
	if file_path.suffix.lower() == ".geojson":
		return load_private_geojson(file_path, dataset_title)

	with file_path.open("r", encoding="utf-8") as f:
		items = json.load(f)

	rows: list[dict[str, Any]] = []
	for item in items:
		name = extract_en(item.get("Estate Name"))
		district = extract_en(item.get("District Name"))
		region = extract_en(item.get("Region Name"))
		lat = parse_float(item.get("Estate Map Latitude"))
		lon = parse_float(item.get("Estate Map Longitude"))

		year = parse_year(item.get("Year of Intake"))
		if pd.isna(year):
			year = parse_year(item.get("Year of Completion"))

		type_label = extract_en(item.get("Type of Estate"))
		if type_label == "":
			type_label = extract_en(item.get("Type(s) of Block(s)"))

		blocks = parse_number_from_text(item.get("No. of Blocks"))
		units = parse_number_from_text(item.get("No. of Units"))
		if pd.isna(units):
			units = parse_number_from_text(item.get("No. of Flats"))
		if pd.isna(units):
			units = parse_number_from_text(item.get("No. of Rental Flats"))

		rows.append(
			{
				"dataset": dataset_title,
				"name": name,
				"district": district,
				"region": region,
				"lat": lat,
				"lon": lon,
				"year": year,
				"type_label": type_label,
				"blocks": blocks,
				"units": units,
			}
		)

	df = pd.DataFrame(rows)
	df["coord_valid"] = (
		df["lat"].between(*HK_LAT_RANGE) & df["lon"].between(*HK_LON_RANGE)
	).fillna(False)
	df["year"] = pd.to_numeric(df["year"], errors="coerce")
	return df


def infer_region_from_address(address_en: str) -> str:
	text = str(address_en or "").upper()
	if "HONG KONG" in text:
		return "Hong Kong"
	if "KOWLOON" in text:
		return "Kowloon"
	if "NEW TERRITORIES" in text:
		return "New Territories"
	return "Unknown"


def infer_district_from_address(address_en: str) -> str:
	text = str(address_en or "").upper()
	known = [
		"CENTRAL AND WESTERN", "WAN CHAI", "EASTERN", "SOUTHERN",
		"YAU TSIM MONG", "SHAM SHUI PO", "KOWLOON CITY", "WONG TAI SIN", "KWUN TONG",
		"KWAI TSING", "TSUEN WAN", "TUEN MUN", "YUEN LONG", "NORTH", "TAI PO",
		"SHA TIN", "SAI KUNG", "ISLANDS",
	]
	for d in known:
		if d in text:
			return d.title()
	return "Unknown"


def load_private_geojson(file_path: Path, dataset_title: str) -> pd.DataFrame:
	with file_path.open("r", encoding="utf-8") as f:
		geo = json.load(f)

	rows: list[dict[str, Any]] = []
	for feat in geo.get("features", []):
		props = feat.get("properties", {})
		geom = feat.get("geometry", {})
		coords = geom.get("coordinates", [None, None])
		lon = coords[0] if isinstance(coords, list) and len(coords) >= 2 else None
		lat = coords[1] if isinstance(coords, list) and len(coords) >= 2 else None

		name = extract_en(props.get("NAME_OF_OCOMM_IN_ENG"))
		if name == "":
			name = extract_en(props.get("NAME_OF_OCOMM_IN_CHN"))
		address_en = extract_en(props.get("ADDR_OF_BUILDING_IN_ENG"))

		rows.append(
			{
				"dataset": dataset_title,
				"name": name,
				"district": infer_district_from_address(address_en),
				"region": infer_region_from_address(address_en),
				"lat": parse_float(lat),
				"lon": parse_float(lon),
				"year": float("nan"),
				"type_label": "Owners' Committee",
				"blocks": float("nan"),
				"units": float("nan"),
			}
		)

	df = pd.DataFrame(rows)
	df["coord_valid"] = (
		df["lat"].between(*HK_LAT_RANGE) & df["lon"].between(*HK_LON_RANGE)
	).fillna(False)
	df["year"] = pd.to_numeric(df["year"], errors="coerce")
	return df


def detect_issues(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame(columns=["issue", "rule", "affected_count", "severity", "status"])

	issues = []

	def add(issue: str, rule: str, mask: pd.Series, severity: str) -> None:
		count = int(mask.fillna(False).sum())
		issues.append(
			{
				"issue": issue,
				"rule": rule,
				"affected_count": count,
				"severity": severity,
				"status": "detected" if count > 0 else "clear",
			}
		)

	add("Missing name", "name is blank", df["name"].astype(str).str.strip() == "", "high")
	add("Missing district", "district is blank", df["district"].astype(str).str.strip() == "", "high")
	add("Invalid coordinates", "lat/lon missing or outside HK bounds", ~df["coord_valid"], "high")
	add("Missing year", "year is null", df["year"].isna(), "medium")
	add("Missing type", "type_label is blank", df["type_label"].astype(str).str.strip() == "", "low")
	dup = df.duplicated(subset=["name", "district"], keep=False)
	add("Potential duplicate estates", "duplicated by name + district", dup, "medium")

	return pd.DataFrame(issues)


def render_section(section_idx: int, dataset_title: str, df: pd.DataFrame) -> None:
	st.subheader(f"{section_idx}. {dataset_title}")

	if df.empty:
		st.warning("No records found in this dataset.")
		st.markdown("---")
		return

	issues_df = detect_issues(df)

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Records", f"{len(df):,}")
	c2.metric("Districts", f"{df['district'].replace('', pd.NA).nunique(dropna=True):,}")
	c3.metric("Regions", f"{df['region'].replace('', pd.NA).nunique(dropna=True):,}")
	c4.metric("Valid Coordinates", f"{int(df['coord_valid'].sum()):,}")

	st.write("Possible Issues and Detection")
	st.dataframe(issues_df, width="stretch", height=220)

	left, right = st.columns(2)
	with left:
		st.write("Distribution by District")
		district_counts = df["district"].replace("", "Unknown").value_counts()
		if district_counts.empty:
			st.info("No district data available.")
		else:
			st.bar_chart(district_counts)

	with right:
		st.write("Distribution by Year")
		year_counts = df["year"].dropna().astype(int).value_counts().sort_index()
		if year_counts.empty:
			st.info("No year data available.")
		else:
			st.bar_chart(year_counts)

	if df["type_label"].astype(str).str.strip().any():
		st.write("Type Breakdown")
		type_counts = df["type_label"].replace("", "Unknown").value_counts()
		st.bar_chart(type_counts.head(20))

	map_df = df[df["coord_valid"]].copy()
	st.write("Map")
	if map_df.empty:
		st.info("No valid coordinates available for map in this dataset.")
	else:
		center_lat = float(map_df["lat"].mean())
		center_lon = float(map_df["lon"].mean())
		layer = pdk.Layer(
			"ScatterplotLayer",
			map_df,
			get_position="[lon, lat]",
			get_radius=130,
			pickable=True,
			filled=True,
			get_fill_color=[45, 125, 210, 190],
			get_line_color=[15, 30, 50, 220],
			line_width_min_pixels=1,
		)
		tooltip: Any = {
			"html": "<b>{name}</b><br/>District: {district}<br/>Region: {region}<br/>Year: {year}<br/>Type: {type_label}",
			"style": {"backgroundColor": "#111827", "color": "white"},
		}
		st.pydeck_chart(
			pdk.Deck(
				map_provider="carto",
				map_style="light",
				initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10.3, pitch=0),
				layers=[layer],
				tooltip=tooltip,
			),
			width="stretch",
		)

	st.write("Data Table")
	st.dataframe(
		df[["name", "district", "region", "year", "type_label", "blocks", "units", "lat", "lon", "coord_valid"]],
		width="stretch",
		height=300,
	)
	st.markdown("---")


def render_summary_map(all_df: pd.DataFrame) -> None:
	st.subheader("Summary Map: Housing Distribution Across All Datasets")

	map_df = all_df[all_df["coord_valid"]].copy()
	if map_df.empty:
		st.info("No valid coordinates found across the selected housing datasets.")
		return

	map_df["dataset_color"] = map_df["dataset"].map(DATASET_COLORS).apply(
		lambda c: c if isinstance(c, list) else [127, 140, 141, 180]
	)

	center_lat = float(map_df["lat"].mean())
	center_lon = float(map_df["lon"].mean())

	layer = pdk.Layer(
		"ScatterplotLayer",
		map_df,
		get_position="[lon, lat]",
		get_radius=145,
		pickable=True,
		filled=True,
		get_fill_color="dataset_color",
		get_line_color=[20, 20, 20, 210],
		line_width_min_pixels=1,
	)

	tooltip: Any = {
		"html": "<b>{name}</b><br/>Dataset: {dataset}<br/>District: {district}<br/>Region: {region}<br/>Year: {year}",
		"style": {"backgroundColor": "#111827", "color": "white"},
	}

	st.pydeck_chart(
		pdk.Deck(
			map_provider="carto",
			map_style="light",
			initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10.2, pitch=0),
			layers=[layer],
			tooltip=tooltip,
		),
		width="stretch",
	)

	legend_rows = []
	for dataset_name, _ in DATASET_ORDER:
		title = dict(DATASET_ORDER)[dataset_name]
		if title not in map_df["dataset"].unique():
			continue
		color = DATASET_COLORS.get(title, [127, 140, 141, 180])
		swatch = f"rgba({color[0]}, {color[1]}, {color[2]}, 0.95)"
		legend_rows.append(
			f"<div style='display:flex;align-items:center;margin:4px 0;'>"
			f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;border:1px solid #111;background:{swatch};margin-right:8px;'></span>"
			f"<span style='font-size:12px;color:#0f172a;'>{title}</span>"
			f"</div>"
		)

	if legend_rows:
		st.markdown(
			f"""
			<div style="max-width:420px;background:#ffffffd9;border:1px solid #d0d7de;border-radius:8px;padding:10px 12px;margin-top:-6px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.12);">
				<div style="font-weight:700;font-size:13px;color:#0f172a;margin-bottom:6px;">Legend</div>
				{''.join(legend_rows)}
			</div>
			""",
			unsafe_allow_html=True,
		)

	st.markdown("---")


def main() -> None:
	st.title("Housing Dataset Analyzer")
	st.caption("Separate analyses for all housing datasets, including newly added private housing GeoJSON")

	loaded_sections: list[tuple[str, pd.DataFrame]] = []
	section_idx = 1
	for file_name, title in DATASET_ORDER:
		path = DATA_DIR / file_name
		if not path.exists():
			st.warning(f"Missing dataset: {file_name}")
			continue
		df = load_dataset(path, title)
		loaded_sections.append((title, df))

	if loaded_sections:
		all_df = pd.concat([df for _, df in loaded_sections], ignore_index=True)
		render_summary_map(all_df)

	for title, df in loaded_sections:
		render_section(section_idx, title, df)
		section_idx += 1


if __name__ == "__main__":
	main()
