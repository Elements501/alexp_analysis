from pathlib import Path
import math
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(page_title="Recycling Point Analyzer", layout="wide")

DATA_FILE = Path(__file__).with_name("wasteless250918.csv")
HK_LAT_RANGE = (22.0, 23.0)
HK_LON_RANGE = (113.7, 114.5)


STATION_TYPES_ORDER = [
	"Recycling Bins at Public Place",
	"Smart Bin",
	"Recycling Stations/Recycling Stores",
	"Street Corner Recycling Shops",
	"NGO Collection Points",
	"Private Collection Points",
	"Recycling Spots",
	"Unknown",
]

TYPE_COLORS_RGB: dict[str, list[int]] = {
	"Recycling Bins at Public Place":      [13,  110, 253],
	"Smart Bin":                           [0,   188, 188],
	"Recycling Stations/Recycling Stores": [40,  167,  69],
	"Street Corner Recycling Shops":       [230, 100,   0],
	"NGO Collection Points":               [111,  66, 193],
	"Private Collection Points":           [220,  53, 130],
	"Recycling Spots":                     [200, 176,   0],
	"Unknown":                             [150, 150, 150],
}

TYPE_COLORS_HEX: dict[str, str] = {
	"Recycling Bins at Public Place":      "#0D6EFD",
	"Smart Bin":                           "#00BCBC",
	"Recycling Stations/Recycling Stores": "#28A745",
	"Street Corner Recycling Shops":       "#E66400",
	"NGO Collection Points":               "#6F42C1",
	"Private Collection Points":           "#DC3582",
	"Recycling Spots":                     "#C8B000",
	"Unknown":                             "#969696",
}


def normalize_station_type(legend: str) -> str:
	v = str(legend or "").strip()
	if not v:
		return "Unknown"
	if v.startswith("Private Collection"):
		return "Private Collection Points"
	return v


def split_waste_types(value: Any) -> list[str]:
	if value is None:
		return []
	parts = [part.strip() for part in str(value).split(",")]
	return [part for part in parts if part]


@st.cache_data(show_spinner=False)
def load_data(file_path: Path) -> pd.DataFrame:
	df = pd.read_csv(file_path, dtype=str)

	for col in ["lat", "lgt"]:
		df[col] = pd.to_numeric(df[col], errors="coerce")

	df["district"] = df["district_id"].fillna("Unknown").str.replace("_", " ", regex=False)
	df["waste_type"] = df["waste_type"].fillna("Unknown")
	df["cp_state"] = df["cp_state"].fillna("Unknown")

	# Prefer English labels for this dashboard, fallback to Chinese if English is empty.
	df["display_address"] = (
		df["address_en"].fillna("").str.strip().where(df["address_en"].fillna("").str.strip() != "", df["address_tc"].fillna(""))
	)
	df["display_note"] = df["accessibilty_notes"].fillna("").str.strip()
	df["waste_list"] = df["waste_type"].apply(split_waste_types)
	df["waste_type_count"] = df["waste_list"].apply(len)

	note_lower = df["display_note"].str.lower()
	df["restricted_access_flag"] = note_lower.str.contains(
		"member|residen|appointment|staff|private|closed|not open",
		regex=True,
		na=False,
	)

	valid_coords = (
		df["lat"].between(*HK_LAT_RANGE)
		& df["lgt"].between(*HK_LON_RANGE)
	)
	df["coord_valid"] = valid_coords.fillna(False)

	df["station_type"] = df["legend"].fillna("").map(normalize_station_type)
	df["r"] = df["station_type"].map(lambda t: TYPE_COLORS_RGB.get(t, [150, 150, 150])[0])
	df["g"] = df["station_type"].map(lambda t: TYPE_COLORS_RGB.get(t, [150, 150, 150])[1])
	df["b"] = df["station_type"].map(lambda t: TYPE_COLORS_RGB.get(t, [150, 150, 150])[2])

	return df


def count_waste_types(series: pd.Series) -> pd.Series:
	exploded = series.apply(split_waste_types).explode()
	if exploded.empty:
		return pd.Series(dtype="int64")
	exploded = exploded.fillna("").astype(str).str.strip()
	exploded = exploded[exploded != ""]
	if exploded.empty:
		return pd.Series(dtype="int64")
	return exploded.value_counts()


def compute_waste_diversity(filtered: pd.DataFrame) -> dict[str, float]:
	counts = count_waste_types(filtered["waste_type"])
	if counts.empty:
		return {
			"unique_materials": 0.0,
			"shannon": 0.0,
			"evenness": 0.0,
		}

	total = float(counts.sum())
	probs = counts / total
	shannon = float(-(probs * probs.apply(math.log)).sum())
	unique_materials = float(counts.shape[0])
	evenness = float(shannon / math.log(unique_materials)) if unique_materials > 1 else 1.0

	return {
		"unique_materials": unique_materials,
		"shannon": shannon,
		"evenness": evenness,
	}


def filter_points_accepting_waste(df: pd.DataFrame, waste_name: str) -> pd.DataFrame:
	target = waste_name.strip().lower()
	if not target:
		return df.iloc[0:0].copy()
	return df[df["waste_list"].apply(lambda xs: any(item.lower() == target for item in xs))]


def render_waste_type_maps(stype: str, subset: pd.DataFrame) -> None:
	st.caption("Map by accepted waste type")
	waste_counts = count_waste_types(subset["waste_type"])
	if waste_counts.empty:
		st.info("No waste-type data for map breakdown.")
		return

	rgb = TYPE_COLORS_RGB.get(stype, [150, 150, 150])
	waste_tabs = st.tabs(waste_counts.index.tolist())
	for tab, waste_name in zip(waste_tabs, waste_counts.index.tolist()):
		with tab:
			waste_df = filter_points_accepting_waste(subset, waste_name)
			map_df = waste_df[waste_df["coord_valid"]].copy()
			st.caption(f"{len(waste_df):,} points accept {waste_name}")
			if map_df.empty:
				st.info("No valid coordinates for this waste type.")
				continue

			layer = pdk.Layer(
				"ScatterplotLayer",
				map_df,
				get_position="[lgt, lat]",
				get_fill_color=[*rgb, 210],
				get_radius=62,
				pickable=True,
			)
			center_lat = float(map_df["lat"].mean())
			center_lon = float(map_df["lgt"].mean())
			st.pydeck_chart(
				pdk.Deck(
					map_provider="carto",
					map_style="light",
					initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10.5, pitch=0),
					layers=[layer],
					tooltip={  # type: ignore[arg-type]
						"html": "<b>{display_address}</b><br/>District: {district}<br/>Waste: {waste_type}<br/>State: {cp_state}",
						"style": {"backgroundColor": "#162029", "color": "white"},
					},
				),
				width="stretch",
			)


def issue_hypothesis_table(df: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
	district_stats = filtered.groupby("district", as_index=False).agg(
		points=("cp_id", "count"),
		avg_types=("waste_type_count", "mean"),
	)

	low_coverage_threshold = float(district_stats["points"].quantile(0.25)) if not district_stats.empty else 0.0
	low_diversity_threshold = float(filtered["waste_type_count"].quantile(0.25)) if not filtered.empty else 0.0

	total_waste_mentions = int(count_waste_types(filtered["waste_type"]).sum())
	top_three_mentions = int(count_waste_types(filtered["waste_type"]).head(3).sum())
	concentration_ratio = (top_three_mentions / total_waste_mentions) if total_waste_mentions else 0.0

	rows = [
		{
			"possible_issue": "Uneven district coverage",
			"indicator": f"Districts with points <= P25 ({low_coverage_threshold:,.1f})",
			"affected_records": int((district_stats["points"] <= low_coverage_threshold).sum()) if not district_stats.empty else 0,
			"severity": "high",
			"data_status": "supported",
		},
		{
			"possible_issue": "Low material diversity per point",
			"indicator": f"Points with waste_type_count <= P25 ({low_diversity_threshold:,.1f})",
			"affected_records": int((filtered["waste_type_count"] <= low_diversity_threshold).sum()) if not filtered.empty else 0,
			"severity": "medium",
			"data_status": "supported",
		},
		{
			"possible_issue": "Waste-type concentration",
			"indicator": f"Top 3 waste types share ({concentration_ratio:.1%})",
			"affected_records": top_three_mentions,
			"severity": "medium" if concentration_ratio < 0.75 else "high",
			"data_status": "supported",
		},
		{
			"possible_issue": "Potential access restriction",
			"indicator": "Records with restrictive wording in notes",
			"affected_records": int(filtered["restricted_access_flag"].sum()) if not filtered.empty else 0,
			"severity": "low",
			"data_status": "supported",
		},
		{
			"possible_issue": "Unknown number of bins",
			"indicator": "No numeric bin-count field in source",
			"affected_records": len(df),
			"severity": "info",
			"data_status": "not available",
		},
		{
			"possible_issue": "Unknown trash volume / fill-rate",
			"indicator": "No tonnage/volume/timestamped pickup volume field",
			"affected_records": len(df),
			"severity": "info",
			"data_status": "not available",
		},
	]

	return pd.DataFrame(rows)


def district_summary(filtered: pd.DataFrame) -> pd.DataFrame:
	if filtered.empty:
		return pd.DataFrame(columns=["district", "lat", "lgt", "points", "avg_types"])

	summary = filtered.groupby("district", as_index=False).agg(
		lat=("lat", "mean"),
		lgt=("lgt", "mean"),
		points=("cp_id", "count"),
		avg_types=("waste_type_count", "mean"),
	)
	point_q1 = float(summary["points"].quantile(0.25)) if not summary.empty else 0.0
	summary["low_coverage"] = summary["points"] <= point_q1
	return summary


def render_summary_map(filtered: pd.DataFrame) -> None:
	st.subheader("Summary map: all station types")
	map_df = filtered[filtered["coord_valid"]].copy()
	if map_df.empty:
		st.warning("No mappable points for current filters.")
		return

	layer = pdk.Layer(
		"ScatterplotLayer",
		map_df,
		get_position="[lgt, lat]",
		get_fill_color="[r, g, b, 200]",
		get_radius=50,
		pickable=True,
		stroked=True,
		get_line_color=[30, 30, 30, 100],
		line_width_min_pixels=1,
	)
	center_lat = float(map_df["lat"].mean())
	center_lon = float(map_df["lgt"].mean())
	st.pydeck_chart(
		pdk.Deck(
			map_provider="carto",
			map_style="light",
			initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10.5, pitch=0),
			layers=[layer],
			tooltip={  # type: ignore[arg-type]
				"html": "<b>{station_type}</b><br/>{display_address}<br/>District: {district}<br/>State: {cp_state}",
				"style": {"backgroundColor": "#162029", "color": "white"},
			},
		),
		width="stretch",
	)

	present_types = [t for t in STATION_TYPES_ORDER if t in map_df["station_type"].unique()]
	items = [
		f'<span style="display:inline-flex;align-items:center;gap:5px;">'
		f'<span style="display:inline-block;width:13px;height:13px;border-radius:50%;background:{TYPE_COLORS_HEX.get(t, "#ccc")}"></span>'
		f'{t}</span>'
		for t in present_types
	]
	st.markdown(
		'<div style="display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:6px;font-size:13px;">'
		+ "".join(items)
		+ "</div>",
		unsafe_allow_html=True,
	)


def render_type_section(stype: str, subset: pd.DataFrame, total: int) -> None:
	count = len(subset)
	pct = count / max(1, total) * 100
	diversity = compute_waste_diversity(subset)
	avg_materials = float(subset["waste_type_count"].mean()) if not subset.empty else 0.0

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Points", f"{count:,}")
	c2.metric("Share of filtered set", f"{pct:.1f}%")
	c3.metric("Districts covered", f"{subset['district'].nunique()}")
	c4.metric("Diversity (Shannon H')", f"{diversity['shannon']:.2f}")

	d1, d2, d3 = st.columns(3)
	d1.metric("Unique waste types", f"{int(diversity['unique_materials'])}")
	d2.metric("Evenness (Pielou J)", f"{diversity['evenness']:.2f}")
	d3.metric("Avg waste types per point", f"{avg_materials:.2f}")

	charts_left, charts_right = st.columns(2)
	with charts_left:
		st.caption("Points per district")
		dc = subset["district"].value_counts().head(15)
		if not dc.empty:
			st.bar_chart(dc)
		else:
			st.info("No district data.")
	with charts_right:
		st.caption("Accepted waste types")
		wc = count_waste_types(subset["waste_type"])
		if not wc.empty:
			st.bar_chart(wc.head(15))
		else:
			st.info("No waste-type data.")

	issue_table = issue_hypothesis_table(subset, subset)
	with st.expander("Possible issues for this station type"):
		st.dataframe(issue_table, width="stretch", hide_index=True)

	map_df = subset[subset["coord_valid"]].copy()
	if map_df.empty:
		st.warning("No valid coordinates for this station type.")
	else:
		rgb = TYPE_COLORS_RGB.get(stype, [150, 150, 150])
		layer = pdk.Layer(
			"ScatterplotLayer",
			map_df,
			get_position="[lgt, lat]",
			get_fill_color=[*rgb, 200],
			get_radius=60,
			pickable=True,
		)
		center_lat = float(map_df["lat"].mean())
		center_lon = float(map_df["lgt"].mean())
		st.pydeck_chart(
			pdk.Deck(
				map_provider="carto",
				map_style="light",
				initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10.5, pitch=0),
				layers=[layer],
				tooltip={  # type: ignore[arg-type]
					"html": "<b>{display_address}</b><br/>District: {district}<br/>Waste: {waste_type}<br/>State: {cp_state}",
					"style": {"backgroundColor": "#162029", "color": "white"},
				},
			),
			width="stretch",
		)

	render_waste_type_maps(stype, subset)

	relevant_cols = ["cp_id", "cp_state", "district", "display_address", "waste_type", "display_note", "lat", "lgt"]
	visible_cols = [c for c in relevant_cols if c in subset.columns]
	st.dataframe(subset[visible_cols], width="stretch", hide_index=True)


def main() -> None:
	st.title("Recycling Point Dataset Analyzer")
	st.caption("Analysis split by station type — summary map of all locations at top")

	if not DATA_FILE.exists():
		st.error(f"Data file not found: {DATA_FILE}")
		return

	df = load_data(DATA_FILE)

	with st.sidebar:
		st.header("Filters")

		state_options = sorted(df["cp_state"].dropna().unique().tolist())
		selected_states = st.multiselect("Collection point state", state_options, default=["Accepted"] if "Accepted" in state_options else state_options)

		type_options = [t for t in STATION_TYPES_ORDER if t in df["station_type"].unique()]
		selected_types = st.multiselect("Station type", type_options, default=type_options)

		district_options = sorted(df["district"].dropna().unique().tolist())
		selected_districts = st.multiselect("District", district_options, default=district_options)

		waste_options = sorted(count_waste_types(df["waste_type"]).index.tolist())
		selected_waste = st.multiselect("Accepts waste type", waste_options, default=waste_options)

		show_invalid_coords = st.checkbox("Include invalid coordinates", value=False)

	filtered = df.copy()
	if selected_states:
		filtered = filtered[filtered["cp_state"].isin(selected_states)]
	if selected_types:
		filtered = filtered[filtered["station_type"].isin(selected_types)]
	if selected_districts:
		filtered = filtered[filtered["district"].isin(selected_districts)]
	if selected_waste:
		selected_waste_lower = {w.strip().lower() for w in selected_waste if w.strip()}
		filtered = filtered[
			filtered["waste_list"].apply(
				lambda xs: any(item.lower() in selected_waste_lower for item in xs)
			)
		]
	if not show_invalid_coords:
		filtered = filtered[filtered["coord_valid"]]

	coord_valid_count = int(df["coord_valid"].sum())
	invalid_coord_count = int((~df["coord_valid"]).sum())

	m1, m2, m3, m4 = st.columns(4)
	m1.metric("Filtered points", f"{len(filtered):,}")
	m2.metric("Station types", f"{filtered['station_type'].nunique()}")
	m3.metric("Districts", f"{filtered['district'].nunique()}")
	m4.metric("Accepted", f"{int(filtered['cp_state'].eq('Accepted').sum()):,}")
	st.caption(f"Coordinate quality (full dataset): {coord_valid_count:,} valid, {invalid_coord_count:,} invalid")

	st.subheader("Station type breakdown")
	type_counts = filtered["station_type"].value_counts()
	if not type_counts.empty:
		st.bar_chart(type_counts)

	render_summary_map(filtered)

	st.subheader("Analysis by station type")
	tab_labels = [t for t in STATION_TYPES_ORDER if t in filtered["station_type"].unique()]
	if not tab_labels:
		st.info("No station types match the current filters.")
		return

	tabs = st.tabs(tab_labels)
	for tab, stype in zip(tabs, tab_labels):
		with tab:
			render_type_section(stype, filtered[filtered["station_type"] == stype].copy(), len(filtered))


if __name__ == "__main__":
	main()
