import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(page_title="Waste Management Demographics", layout="wide")

DATA_FILE = Path(__file__).with_name("EPDWMF_20260228.gdb_converted.geojson")
HK_LAT_RANGE = (22.0, 23.0)
HK_LON_RANGE = (113.7, 114.5)

DISTRICT_KEYWORDS = {
	"Tuen Mun": "Tuen Mun",
	"Yuen Long": "Yuen Long",
	"North Lantau": "Islands",
	"Lantau": "Islands",
	"Tseung Kwan O": "Sai Kung",
	"Shatin": "Sha Tin",
	"Sha Tin": "Sha Tin",
	"Tsing Yi": "Kwai Tsing",
	"Kwai Chung": "Kwai Tsing",
	"Kowloon Bay": "Kwun Tong",
	"Nam Sang Wai": "Yuen Long",
	"Nim Wan": "Tuen Mun",
}

TYPE_COLORS = {
	"Landfill": [220, 53, 69, 200],
	"Transfer Station": [32, 156, 238, 200],
	"Waste-to-Energy": [255, 159, 67, 200],
	"Chemical Waste Facility": [153, 102, 255, 200],
	"Recovery / Recycling Facility": [46, 204, 113, 200],
	"Treatment Facility": [0, 184, 148, 200],
	"Integrated Waste Management Facilities": [255, 99, 132, 200],
	"Yard Waste Recycling Centre": [75, 192, 192, 200],
}


def normalize_type(type_name: str) -> str:
	label = str(type_name or "Unknown").strip().upper()
	if "LANDFILL" in label:
		return "Landfill"
	if "TRANSFER STATION" in label:
		return "Transfer Station"
	if "WASTE-TO-ENERGY" in label or "INCINERATION" in label:
		return "Waste-to-Energy"
	if "CHEMICAL" in label:
		return "Chemical Waste Facility"
	if "RECOVERY" in label or "PARK" in label:
		return "Recovery / Recycling Facility"
	if "TREATMENT" in label:
		return "Treatment Facility"
	return label.title()


def infer_district(address_en: str) -> str:
	text = str(address_en or "")
	for key, district in DISTRICT_KEYWORDS.items():
		if key.lower() in text.lower():
			return district
	return "Unknown"


def parse_first_number(value: str) -> float:
	if not isinstance(value, str):
		return float("nan")
	match = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", value)
	if not match:
		return float("nan")
	return float(match.group(1).replace(",", ""))


def parse_tonnes_per_day(text: str) -> float:
	if not isinstance(text, str):
		return float("nan")
	lower = text.lower()
	if "tonne" not in lower:
		return float("nan")
	value = parse_first_number(text)
	if pd.isna(value):
		return float("nan")
	if "per year" in lower or "a year" in lower:
		return value / 365.0
	if "per day" in lower or "daily" in lower:
		return value
	return float("nan")


def to_float(value: Any) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return float("nan")


def color_for_type(type_name: str) -> list[int]:
	return TYPE_COLORS.get(type_name, [148, 163, 184, 180])


@st.cache_data(show_spinner=False)
def load_data(file_path: Path) -> pd.DataFrame:
	with file_path.open("r", encoding="utf-8") as f:
		geo = json.load(f)

	rows: list[dict[str, Any]] = []
	for feat in geo.get("features", []):
		props = feat.get("properties", {})
		geom = feat.get("geometry", {})
		coords = geom.get("coordinates", [None, None])
		lon = coords[0] if isinstance(coords, list) and len(coords) >= 2 else None
		lat = coords[1] if isinstance(coords, list) and len(coords) >= 2 else None

		type_raw = str(props.get("SEARCH01_EN") or "Unknown")
		cap_text = str(props.get("NSEARCH01_EN") or "")
		throughput_text = str(props.get("NSEARCH02_EN") or "")
		address = str(props.get("ADDRESS_EN") or "")

		rows.append(
			{
				"OBJECTID": props.get("OBJECTID"),
				"NAME_EN": str(props.get("NAME_EN") or ""),
				"ADDRESS_EN": address,
				"facility_type_raw": type_raw,
				"facility_type": normalize_type(type_raw),
				"capacity_text": cap_text,
				"throughput_text": throughput_text,
				"hours_text": str(props.get("NSEARCH04_EN") or ""),
				"reference_url": str(props.get("NSEARCH08_EN") or ""),
				"district": infer_district(address),
				"lat": to_float(lat),
				"lon": to_float(lon),
				"latitude_ref": pd.to_numeric(props.get("LATITUDE"), errors="coerce"),
				"longitude_ref": pd.to_numeric(props.get("LONGITUDE"), errors="coerce"),
				"last_update": str(props.get("LASTUPDATE") or ""),
			}
		)

	df = pd.DataFrame(rows)
	df["coord_valid"] = (
		df["lat"].between(*HK_LAT_RANGE) & df["lon"].between(*HK_LON_RANGE)
	).fillna(False)
	df["capacity_value"] = df["capacity_text"].apply(parse_first_number)
	df["throughput_tpd"] = df["throughput_text"].apply(parse_tonnes_per_day)
	df["has_capacity_data"] = df["capacity_value"].notna()
	df["has_throughput_data"] = df["throughput_tpd"].notna()
	df["last_update_dt"] = pd.to_datetime(df["last_update"], format="%Y%m%d%H%M%S", errors="coerce")
	df["type_color"] = df["facility_type"].apply(color_for_type)

	return df


def build_facility_demographics(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame(columns=["facility_type", "count", "district_count", "with_capacity", "with_throughput"])

	return (
		df.groupby("facility_type", as_index=False)
		.agg(
			count=("OBJECTID", "count"),
			district_count=("district", "nunique"),
			with_capacity=("has_capacity_data", "sum"),
			with_throughput=("has_throughput_data", "sum"),
		)
		.sort_values("count", ascending=False)
	)


def build_district_demographics(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame(columns=["district", "facility_count", "type_count", "avg_throughput_tpd"])

	return (
		df.groupby("district", as_index=False)
		.agg(
			facility_count=("OBJECTID", "count"),
			type_count=("facility_type", "nunique"),
			avg_throughput_tpd=("throughput_tpd", "mean"),
		)
		.sort_values("facility_count", ascending=False)
	)


def color_by_type() -> list[Any]:
	return [
		"case",
		["==", ["get", "facility_type"], "Landfill"], [220, 53, 69, 200],
		["==", ["get", "facility_type"], "Transfer Station"], [32, 156, 238, 200],
		["==", ["get", "facility_type"], "Waste-to-Energy"], [255, 159, 67, 200],
		["==", ["get", "facility_type"], "Chemical Waste Facility"], [153, 102, 255, 200],
		["==", ["get", "facility_type"], "Recovery / Recycling Facility"], [46, 204, 113, 200],
		[148, 163, 184, 180],
	]


def build_issue_checklist(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame(columns=["issue", "rule", "affected_count", "severity", "status"])

	now = pd.Timestamp.now()
	district_stats = df.groupby("district", as_index=False).agg(
		facility_count=("OBJECTID", "count"),
		type_count=("facility_type", "nunique"),
	)
	district_p25 = float(district_stats["facility_count"].quantile(0.25)) if not district_stats.empty else 0.0
	throughput_q3 = float(df["throughput_tpd"].dropna().quantile(0.75)) if df["throughput_tpd"].notna().any() else float("nan")

	checks = [
		{
			"issue": "Low district coverage",
			"rule": f"district facility_count <= P25 ({district_p25:,.1f})",
			"affected_count": int((district_stats["facility_count"] <= district_p25).sum()) if not district_stats.empty else 0,
			"severity": "high",
			"status": "detected" if (district_stats["facility_count"] <= district_p25).any() else "clear",
		},
		{
			"issue": "Low type diversity by district",
			"rule": "district type_count <= 1",
			"affected_count": int((district_stats["type_count"] <= 1).sum()) if not district_stats.empty else 0,
			"severity": "medium",
			"status": "detected" if (district_stats["type_count"] <= 1).any() else "clear",
		},
		{
			"issue": "Missing throughput data",
			"rule": "throughput_tpd is null",
			"affected_count": int(df["throughput_tpd"].isna().sum()),
			"severity": "medium",
			"status": "detected" if df["throughput_tpd"].isna().any() else "clear",
		},
		{
			"issue": "Missing capacity data",
			"rule": "capacity_value is null",
			"affected_count": int(df["capacity_value"].isna().sum()),
			"severity": "medium",
			"status": "detected" if df["capacity_value"].isna().any() else "clear",
		},
		{
			"issue": "Potential stale updates",
			"rule": "last_update_dt older than 3 years",
			"affected_count": int((df["last_update_dt"] < (now - pd.DateOffset(years=3))).fillna(False).sum()),
			"severity": "low",
			"status": "detected" if (df["last_update_dt"] < (now - pd.DateOffset(years=3))).fillna(False).any() else "clear",
		},
		{
			"issue": "Unknown district mapping",
			"rule": "district == 'Unknown'",
			"affected_count": int((df["district"] == "Unknown").sum()),
			"severity": "low",
			"status": "detected" if (df["district"] == "Unknown").any() else "clear",
		},
		{
			"issue": "High throughput concentration",
			"rule": f"throughput_tpd >= Q3 ({throughput_q3:,.1f})",
			"affected_count": int((df["throughput_tpd"] >= throughput_q3).fillna(False).sum()) if pd.notna(throughput_q3) else 0,
			"severity": "low",
			"status": "detected" if pd.notna(throughput_q3) and (df["throughput_tpd"] >= throughput_q3).any() else "clear",
		},
	]

	return pd.DataFrame(checks)


def render_type_legend_bottom_left(map_df: pd.DataFrame) -> None:
	types_in_view = sorted(map_df["facility_type"].dropna().unique().tolist())
	if not types_in_view:
		return

	legend_rows = []
	for t in types_in_view:
		c = TYPE_COLORS.get(t, [148, 163, 184, 180])
		swatch = f"rgba({c[0]}, {c[1]}, {c[2]}, 0.95)"
		legend_rows.append(
			f"<div style='display:flex;align-items:center;margin:4px 0;'>"
			f"<span style='width:14px;height:14px;background:{swatch};display:inline-block;border:1px solid #111;border-radius:3px;margin-right:8px;'></span>"
			f"<span style='font-size:12px;color:#101828;'>{t}</span>"
			f"</div>"
		)

	legend_html = "".join(legend_rows)
	st.markdown(
		f"""
		<div style="max-width:340px;background:#ffffffd9;border:1px solid #d0d7de;border-radius:8px;padding:10px 12px;margin-top:-8px;margin-bottom:8px;box-shadow:0 1px 4px rgba(0,0,0,0.12);">
			<div style="font-weight:700;font-size:13px;color:#0f172a;margin-bottom:6px;">Map 1B Color Legend</div>
			{legend_html}
		</div>
		""",
		unsafe_allow_html=True,
	)


def main() -> None:
	st.title("Waste Management Facility Demographics")
	st.caption("Demographic profile of facility types and locations, with capacity/throughput fields where available")

	if not DATA_FILE.exists():
		st.error(f"Data file not found: {DATA_FILE}")
		return

	df = load_data(DATA_FILE)

	with st.sidebar:
		st.header("Filters")
		type_options = sorted(df["facility_type"].dropna().unique().tolist())
		selected_types = st.multiselect("Facility Type", type_options, default=type_options)

		district_options = sorted(df["district"].dropna().unique().tolist())
		selected_districts = st.multiselect("District", district_options, default=district_options)

		show_only_valid = st.checkbox("Only valid HK coordinates", value=True)

	filtered = df.copy()
	if selected_types:
		filtered = filtered[filtered["facility_type"].isin(selected_types)]
	if selected_districts:
		filtered = filtered[filtered["district"].isin(selected_districts)]
	if show_only_valid:
		filtered = filtered[filtered["coord_valid"]]

	facility_demo = build_facility_demographics(filtered)
	district_demo = build_district_demographics(filtered)
	issues_df = build_issue_checklist(filtered)

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Facilities", f"{len(filtered):,}")
	c2.metric("Facility Types", f"{filtered['facility_type'].nunique():,}")
	c3.metric("Districts", f"{filtered['district'].nunique():,}")
	c4.metric("With Throughput Data", f"{int(filtered['has_throughput_data'].sum()):,}")

	st.subheader("Possible Issues and Detection")
	st.caption("Rules below flag potential distribution and data quality issues from available fields.")
	st.dataframe(issues_df, width="stretch", height=280)

	st.subheader("Map 1: All Facilities by Type")
	map_df = filtered[filtered["coord_valid"]].copy()
	if map_df.empty:
		st.warning("No mappable facilities under the current filters.")
	else:
		center_lat = float(map_df["lat"].mean())
		center_lon = float(map_df["lon"].mean())
		point_layer = pdk.Layer(
			"ScatterplotLayer",
			map_df,
			get_position="[lon, lat]",
			get_radius=120,
			filled=True,
			pickable=True,
			get_fill_color=color_by_type(),
			get_line_color=[15, 23, 42, 220],
			line_width_min_pixels=1,
		)
		tooltip: Any = {
			"html": "<b>{NAME_EN}</b><br/>Type: {facility_type}<br/>District: {district}<br/>Address: {ADDRESS_EN}<br/>Throughput(tpd): {throughput_tpd}",
			"style": {"backgroundColor": "#18212e", "color": "white"},
		}
		st.pydeck_chart(
			pdk.Deck(
				map_provider="carto",
				map_style="light",
				initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10.3, pitch=0),
				layers=[point_layer],
				tooltip=tooltip,
			),
			width="stretch",
		)

		st.subheader("Map 1B: Large Facility Markers by Type (Visibility Mode)")
		large_point_layer = pdk.Layer(
			"ScatterplotLayer",
			map_df,
			get_position="[lon, lat]",
			get_radius=520,
			filled=True,
			pickable=True,
			opacity=0.8,
			get_fill_color="type_color",
			get_line_color=[0, 0, 0, 220],
			line_width_min_pixels=2,
		)
		large_tooltip: Any = {
			"html": "<b>{NAME_EN}</b><br/>Type: {facility_type}<br/>District: {district}",
			"style": {"backgroundColor": "#131a22", "color": "white"},
		}
		st.pydeck_chart(
			pdk.Deck(
				map_provider="carto",
				map_style="light",
				initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10.1, pitch=0),
				layers=[large_point_layer],
				tooltip=large_tooltip,
			),
			width="stretch",
		)
		render_type_legend_bottom_left(map_df)

		st.subheader("Map 2: Spatial Density of Facilities")
		hex_layer = pdk.Layer(
			"HexagonLayer",
			map_df,
			get_position="[lon, lat]",
			radius=2800,
			elevation_scale=30,
			extruded=True,
			coverage=0.85,
			pickable=True,
		)
		st.pydeck_chart(
			pdk.Deck(
				map_provider="carto",
				map_style="light",
				initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10.1, pitch=42),
				layers=[hex_layer],
			),
			width="stretch",
		)

	left, right = st.columns(2)
	with left:
		st.subheader("Graph: Facilities by Type")
		if facility_demo.empty:
			st.info("No type demographic data available for current filters.")
		else:
			chart_data = facility_demo.set_index("facility_type")["count"]
			st.bar_chart(chart_data)

	with right:
		st.subheader("Graph: Facilities by District")
		if district_demo.empty:
			st.info("No district demographic data available for current filters.")
		else:
			chart_data = district_demo.set_index("district")["facility_count"]
			st.bar_chart(chart_data)

	st.subheader("Table: Facility Type Demographics")
	st.dataframe(facility_demo, width="stretch", height=260)

	st.subheader("Table: District Demographics")
	st.dataframe(district_demo, width="stretch", height=280)

	st.subheader("Table: Facility Detail")
	detail_cols = [
		"OBJECTID",
		"NAME_EN",
		"facility_type",
		"district",
		"ADDRESS_EN",
		"capacity_text",
		"throughput_text",
		"throughput_tpd",
		"hours_text",
		"reference_url",
		"lat",
		"lon",
		"coord_valid",
	]
	st.dataframe(filtered[detail_cols], width="stretch", height=420)


if __name__ == "__main__":
	main()
