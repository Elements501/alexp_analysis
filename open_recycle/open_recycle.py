import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(page_title="Open Space Data Analyzer", layout="wide")

DATA_FILE = Path(__file__).with_name("OSDRS_converted.geojson")
HK_LAT_RANGE = (22.0, 23.0)
HK_LON_RANGE = (113.7, 114.5)
SHAPE_COLUMNS = {"Shape_Area", "Shape_Length"}
HELPER_COLUMNS = {"centroid_lat", "centroid_lon", "_feature_idx", "last_update_dt", "nearest_km"}


def iter_positions(coords):
	"""Yield [lon, lat] pairs from arbitrarily nested GeoJSON coordinate lists."""
	if isinstance(coords, list):
		if len(coords) >= 2 and all(isinstance(v, (int, float)) for v in coords[:2]):
			yield coords[0], coords[1]
		else:
			for item in coords:
				yield from iter_positions(item)


def approx_centroid(geometry):
	if not geometry or "coordinates" not in geometry:
		return None, None, 0
	points = list(iter_positions(geometry["coordinates"]))
	if not points:
		return None, None, 0
	lon = sum(p[0] for p in points) / len(points)
	lat = sum(p[1] for p in points) / len(points)
	return lat, lon, len(points)


def haversine_km(lat1, lon1, lat2, lon2):
	if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
		return float("nan")
	R = 6371.0
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlambda = math.radians(lon2 - lon1)
	a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
	return 2 * R * math.asin(math.sqrt(a))


def compute_nearest_km(df: pd.DataFrame):
	coords = df[["centroid_lat", "centroid_lon"]].copy()
	nearest = []
	for idx, row in coords.iterrows():
		if row.isna().any():
			nearest.append(float("nan"))
			continue
		best = float("inf")
		for other_idx, other in coords.iterrows():
			if idx == other_idx or other.isna().any():
				continue
			d = haversine_km(row["centroid_lat"], row["centroid_lon"], other["centroid_lat"], other["centroid_lon"])
			if d < best:
				best = d
		nearest.append(best if best != float("inf") else float("nan"))
	return pd.Series(nearest, index=df.index)


@st.cache_data(show_spinner=False)
def load_data(file_path: Path):
	with file_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	features = data.get("features", [])
	rows = []

	for feat in features:
		props = feat.get("properties", {}).copy()
		geom = feat.get("geometry", {})
		lat, lon, vertex_count = approx_centroid(geom)

		props["centroid_lat"] = lat
		props["centroid_lon"] = lon
		props["_feature_idx"] = len(rows)
		rows.append(props)

	df = pd.DataFrame(rows)
	source_columns = [c for c in df.columns if c not in HELPER_COLUMNS]

	for col in ["Shape_Area", "Shape_Length", "OBJECTID", "COMPLETION_YEAR"]:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	if "LAST_UPDATE" in df.columns:
		df["last_update_dt"] = pd.to_datetime(df["LAST_UPDATE"], format="%Y%m%d%H%M%S", errors="coerce")
	else:
		df["last_update_dt"] = pd.NaT

	if {"centroid_lat", "centroid_lon"}.issubset(df.columns):
		df["nearest_km"] = compute_nearest_km(df)
	else:
		df["nearest_km"] = float("nan")

	return df, data, source_columns


def detect_critical_issues(df: pd.DataFrame):
	issues = []
	now = pd.Timestamp(datetime.now())

	if "OBJECTID" in df.columns:
		dup_count = int(df["OBJECTID"].duplicated().sum())
		if dup_count > 0:
			issues.append(("Duplicate OBJECTID", dup_count, "high"))

	for col in ["OBJECTID", "ADDRESS", "OS_TYPE", "OS_STATUS", "COMPLETION_YEAR", "LAST_UPDATE"]:
		if col in df.columns:
			missing = int(df[col].isna().sum())
			if missing > 0:
				severity = "high" if col in {"OBJECTID", "ADDRESS", "OS_TYPE", "OS_STATUS"} else "medium"
				issues.append((f"Missing {col}", missing, severity))
		else:
			issues.append((f"Missing column: {col}", len(df), "high"))

	if "Shape_Area" in df.columns:
		non_positive_area = int((df["Shape_Area"] <= 0).fillna(False).sum())
		if non_positive_area > 0:
			issues.append(("Non-positive Shape_Area", non_positive_area, "high"))

	if "COMPLETION_YEAR" in df.columns:
		invalid_year = int(
			(
				(df["COMPLETION_YEAR"].notna())
				& ((df["COMPLETION_YEAR"] < 1900) | (df["COMPLETION_YEAR"] > now.year + 1))
			).sum()
		)
		if invalid_year > 0:
			issues.append(("Invalid COMPLETION_YEAR", invalid_year, "medium"))

	if "last_update_dt" in df.columns:
		stale_threshold = now - pd.DateOffset(years=2)
		stale_updates = int((df["last_update_dt"] < stale_threshold).fillna(False).sum())
		invalid_last_update = int(df["last_update_dt"].isna().sum())
		if stale_updates > 0:
			issues.append(("Stale LAST_UPDATE (>2 years)", stale_updates, "medium"))
		if invalid_last_update > 0:
			issues.append(("Invalid LAST_UPDATE timestamp", invalid_last_update, "medium"))

	if {"centroid_lat", "centroid_lon"}.issubset(df.columns):
		out_of_hk = int(
			(
				(df["centroid_lat"].notna())
				& (~df["centroid_lat"].between(*HK_LAT_RANGE) | ~df["centroid_lon"].between(*HK_LON_RANGE))
			).sum()
		)
		if out_of_hk > 0:
			issues.append(("Centroid out of HK bounds", out_of_hk, "high"))

	if "nearest_km" in df.columns:
		isolated = int((df["nearest_km"] > 5).fillna(False).sum())
		if isolated > 0:
			issues.append(("Potentially isolated open spaces (>5km nearest)", isolated, "medium"))

	return pd.DataFrame(issues, columns=["issue", "count", "severity"])


def build_column_profile(df: pd.DataFrame, source_columns):
	rows = []
	for col in source_columns:
		series = df[col] if col in df.columns else pd.Series(dtype="object")
		missing = int(series.isna().sum())
		non_null = int(series.notna().sum())
		unique = int(series.nunique(dropna=True))
		top_value = ""
		top_freq = 0
		if non_null > 0:
			counts = series.astype("string").value_counts(dropna=True)
			if not counts.empty:
				top_value = str(counts.index[0])
				top_freq = int(counts.iloc[0])
		rows.append(
			{
				"column": col,
				"dtype": str(series.dtype),
				"missing": missing,
				"missing_pct": round((missing / len(df) * 100) if len(df) else 0.0, 2),
				"unique_values": unique,
				"top_value": top_value,
				"top_freq": top_freq,
			}
		)
	return pd.DataFrame(rows).sort_values(["missing", "unique_values"], ascending=[False, False])


def issue_checklist(df: pd.DataFrame):
	now = pd.Timestamp(datetime.now())
	checks = []

	def add_check(issue, mask, severity, rule):
		count = int(mask.fillna(False).sum())
		checks.append(
			{
				"issue": issue,
				"rule": rule,
				"exists": "Yes" if count > 0 else "No",
				"affected_count": count,
				"severity": severity,
			}
		)

	if "nearest_km" in df.columns:
		add_check("Too far away", df["nearest_km"] > 2, "medium", "nearest_km > 2")
		add_check("Severely isolated", df["nearest_km"] > 5, "high", "nearest_km > 5")

	if "Shape_Area" in df.columns:
		positive = df.loc[df["Shape_Area"] > 0, "Shape_Area"].dropna()
		if not positive.empty:
			small_threshold = float(positive.quantile(0.1))
			add_check("Too small", (df["Shape_Area"] > 0) & (df["Shape_Area"] < small_threshold), "medium", f"Shape_Area < P10 ({small_threshold:,.2f})")
		add_check("Invalid area", df["Shape_Area"] <= 0, "high", "Shape_Area <= 0")

	if "OS_STATUS" in df.columns:
		not_open = ~df["OS_STATUS"].astype(str).str.contains("open to public", case=False, na=False)
		add_check("Not open to public", not_open, "medium", "OS_STATUS not like 'open to public'")

	if "last_update_dt" in df.columns:
		add_check("Stale updates", df["last_update_dt"] < (now - pd.DateOffset(years=2)), "medium", "LAST_UPDATE older than 2 years")

	if "COMPLETION_YEAR" in df.columns:
		add_check("Invalid completion year", (df["COMPLETION_YEAR"] < 1900) | (df["COMPLETION_YEAR"] > now.year + 1), "medium", "COMPLETION_YEAR outside [1900, current+1]")

	for col, sev in [("ADDRESS", "high"), ("OS_TYPE", "high"), ("AMENITY_AREA", "medium"), ("BEACH", "low")]:
		if col in df.columns:
			add_check(f"Missing {col}", df[col].isna(), sev, f"{col} is null")

	return pd.DataFrame(checks)


def strip_z(coords):
	if isinstance(coords, list):
		if len(coords) >= 2 and all(isinstance(v, (int, float)) for v in coords[:2]):
			return [coords[0], coords[1]]
		return [strip_z(x) for x in coords]
	return coords


def format_year(value: Any) -> str:
	if pd.isna(value):
		return ""
	try:
		return str(int(float(value)))
	except (TypeError, ValueError):
		return ""


def build_polygon_rows(filtered: pd.DataFrame, geojson_data: dict):
	rows = []
	for _, row in filtered.iterrows():
		idx = row.get("_feature_idx")
		if pd.isna(idx):
			continue
		idx = int(idx)
		if idx < 0 or idx >= len(geojson_data.get("features", [])):
			continue

		feat = geojson_data["features"][idx]
		geometry = feat.get("geometry") or {}
		g_type = geometry.get("type")
		coords = strip_z(geometry.get("coordinates", []))
		if not coords:
			continue

		nearest = row.get("nearest_km")
		if pd.notna(nearest):
			if nearest <= 0.5:
				fill_color = [34, 139, 34, 180]
			elif nearest <= 2:
				fill_color = [255, 165, 0, 180]
			else:
				fill_color = [220, 53, 69, 180]
		else:
			fill_color = [128, 128, 128, 150]

		base_props = {
			"ADDRESS": "" if pd.isna(row.get("ADDRESS")) else str(row.get("ADDRESS")),
			"OS_TYPE": "" if pd.isna(row.get("OS_TYPE")) else str(row.get("OS_TYPE")),
			"OS_STATUS": "" if pd.isna(row.get("OS_STATUS")) else str(row.get("OS_STATUS")),
			"COMPLETION_YEAR": format_year(row.get("COMPLETION_YEAR")),
			"LAST_UPDATE": "" if pd.isna(row.get("LAST_UPDATE")) else str(row.get("LAST_UPDATE")),
			"nearest_km": None if pd.isna(nearest) else round(float(nearest), 3),
			"fill_color": fill_color,
		}

		if g_type == "Polygon":
			rows.append({**base_props, "polygon": coords})
		elif g_type == "MultiPolygon":
			for poly in coords:
				rows.append({**base_props, "polygon": poly})

	return rows


def build_polygon_view_data(filtered: pd.DataFrame, geojson_data: dict):
	feature_list = []
	for _, row in filtered.iterrows():
		idx = row.get("_feature_idx")
		if pd.isna(idx):
			continue
		idx = int(idx)
		if idx < 0 or idx >= len(geojson_data.get("features", [])):
			continue
		feat = geojson_data["features"][idx]
		geometry = feat.get("geometry")
		if not geometry:
			continue
		geometry = {
			"type": geometry.get("type"),
			"coordinates": strip_z(geometry.get("coordinates", [])),
		}
		nearest = row.get("nearest_km")
		if pd.notna(nearest):
			if nearest <= 0.5:
				fill_color = [34, 139, 34, 140]
			elif nearest <= 2:
				fill_color = [255, 165, 0, 140]
			else:
				fill_color = [220, 53, 69, 140]
		else:
			fill_color = [128, 128, 128, 120]

		feature_list.append(
			{
				"type": "Feature",
				"geometry": geometry,
				"properties": {
					"ADDRESS": "" if pd.isna(row.get("ADDRESS")) else str(row.get("ADDRESS")),
					"OS_TYPE": "" if pd.isna(row.get("OS_TYPE")) else str(row.get("OS_TYPE")),
					"OS_STATUS": "" if pd.isna(row.get("OS_STATUS")) else str(row.get("OS_STATUS")),
					"COMPLETION_YEAR": format_year(row.get("COMPLETION_YEAR")),
					"LAST_UPDATE": "" if pd.isna(row.get("LAST_UPDATE")) else str(row.get("LAST_UPDATE")),
					"nearest_km": None if pd.isna(nearest) else round(float(nearest), 3),
					"fill_color": fill_color,
				},
			}
		)

	return {"type": "FeatureCollection", "features": feature_list}


def main():
	st.title("Open Space Dataset Analyzer")
	st.caption("Streamlit + pandas dashboard for pattern discovery and critical issue detection")

	if not DATA_FILE.exists():
		st.error(f"Data file not found: {DATA_FILE}")
		return

	df, geojson_data, source_columns = load_data(DATA_FILE)
	issues_df = detect_critical_issues(df)
	column_profile = build_column_profile(df, source_columns)
	checklist_df = issue_checklist(df)

	with st.sidebar:
		st.header("Filters")
		if "OS_STATUS" in df.columns:
			status_choices = sorted(df["OS_STATUS"].dropna().unique().tolist())
			selected_status = st.multiselect("OS_STATUS", status_choices, default=status_choices)
		else:
			selected_status = []

		if "OS_TYPE" in df.columns:
			type_choices = sorted(df["OS_TYPE"].dropna().unique().tolist())
			selected_type = st.multiselect("OS_TYPE", type_choices, default=type_choices)
		else:
			selected_type = []

		if "BEACH" in df.columns:
			beach_choices = sorted(df["BEACH"].dropna().astype(str).unique().tolist())
			selected_beach = st.multiselect("BEACH", beach_choices, default=beach_choices)
		else:
			selected_beach = []

		if "AMENITY_AREA" in df.columns:
			amenity_choices = sorted(df["AMENITY_AREA"].dropna().astype(str).unique().tolist())
			selected_amenity = st.multiselect("AMENITY_AREA", amenity_choices, default=amenity_choices)
		else:
			selected_amenity = []

	filtered = df.copy()
	if selected_status and "OS_STATUS" in filtered.columns:
		filtered = filtered[filtered["OS_STATUS"].isin(selected_status)]
	if selected_type and "OS_TYPE" in filtered.columns:
		filtered = filtered[filtered["OS_TYPE"].isin(selected_type)]
	if selected_beach and "BEACH" in filtered.columns:
		filtered = filtered[filtered["BEACH"].astype(str).isin(selected_beach)]
	if selected_amenity and "AMENITY_AREA" in filtered.columns:
		filtered = filtered[filtered["AMENITY_AREA"].astype(str).isin(selected_amenity)]

	total_records = len(filtered)
	public_open = (
		int(filtered["OS_STATUS"].astype(str).str.contains("open", case=False, na=False).sum())
		if "OS_STATUS" in filtered.columns
		else 0
	)
	amenity_yes = (
		int(filtered["AMENITY_AREA"].astype(str).str.contains("yes", case=False, na=False).sum())
		if "AMENITY_AREA" in filtered.columns
		else 0
	)
	median_nearest = filtered["nearest_km"].median(skipna=True) if "nearest_km" in filtered.columns else float("nan")
	missing_completion_year = (
		int(filtered["COMPLETION_YEAR"].isna().sum()) if "COMPLETION_YEAR" in filtered.columns else total_records
	)

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Records", f"{total_records:,}")
	c2.metric("Open / Accessible Records", f"{public_open:,}")
	c3.metric("Amenity Area = Yes", f"{amenity_yes:,}")
	c4.metric("Missing COMPLETION_YEAR", f"{missing_completion_year:,}")
	st.caption(f"Median nearest open-space distance: {median_nearest:,.2f} km" if pd.notna(median_nearest) else "Median nearest open-space distance: N/A")

	st.subheader("Critical Issues")
	if issues_df.empty:
		st.success("No critical issues detected with current rules.")
	else:
		high_count = int((issues_df["severity"] == "high").sum())
		medium_count = int((issues_df["severity"] == "medium").sum())
		st.warning(f"Detected {len(issues_df)} issue categories ({high_count} high, {medium_count} medium).")
		st.dataframe(issues_df.sort_values(["severity", "count"], ascending=[True, False]), use_container_width=True)

	st.subheader("Issue Checklist (Tested One by One)")
	st.dataframe(checklist_df.sort_values(["exists", "affected_count"], ascending=[False, False]), use_container_width=True)
	if not checklist_df.empty:
		issue_choice = st.selectbox("Inspect issue rows", checklist_df["issue"].tolist())
		selected = checklist_df[checklist_df["issue"] == issue_choice].iloc[0]
		st.caption(f"Rule: {selected['rule']}")
		if issue_choice == "Too far away":
			sample = filtered[filtered["nearest_km"] > 2]
		elif issue_choice == "Severely isolated":
			sample = filtered[filtered["nearest_km"] > 5]
		elif issue_choice == "Too small":
			positive = filtered.loc[filtered["Shape_Area"] > 0, "Shape_Area"].dropna() if "Shape_Area" in filtered.columns else pd.Series(dtype=float)
			threshold = float(positive.quantile(0.1)) if not positive.empty else float("nan")
			sample = filtered[(filtered["Shape_Area"] > 0) & (filtered["Shape_Area"] < threshold)] if "Shape_Area" in filtered.columns else filtered.iloc[0:0]
		elif issue_choice == "Invalid area":
			sample = filtered[filtered["Shape_Area"] <= 0] if "Shape_Area" in filtered.columns else filtered.iloc[0:0]
		elif issue_choice == "Not open to public":
			sample = filtered[~filtered["OS_STATUS"].astype(str).str.contains("open to public", case=False, na=False)] if "OS_STATUS" in filtered.columns else filtered.iloc[0:0]
		elif issue_choice == "Stale updates":
			sample = filtered[filtered["last_update_dt"] < (pd.Timestamp(datetime.now()) - pd.DateOffset(years=2))] if "last_update_dt" in filtered.columns else filtered.iloc[0:0]
		elif issue_choice == "Invalid completion year":
			sample = filtered[(filtered["COMPLETION_YEAR"] < 1900) | (filtered["COMPLETION_YEAR"] > datetime.now().year + 1)] if "COMPLETION_YEAR" in filtered.columns else filtered.iloc[0:0]
		elif isinstance(issue_choice, str) and issue_choice.startswith("Missing "):
			col = issue_choice.replace("Missing ", "")
			sample = filtered[filtered[col].isna()] if col in filtered.columns else filtered.iloc[0:0]
		else:
			sample = filtered.iloc[0:0]

		st.dataframe(sample.head(20), use_container_width=True, height=220)

	st.subheader("Field Coverage (All Dataset Columns)")
	st.dataframe(column_profile, use_container_width=True, height=320)

	st.subheader("Socially Relevant Trends")
	t1, t2 = st.columns(2)

	if "OS_TYPE" in filtered.columns:
		t1.write("Open Space Types")
		t1.bar_chart(filtered["OS_TYPE"].fillna("Unknown").value_counts())

	if "OS_STATUS" in filtered.columns:
		t2.write("Status / Access")
		t2.bar_chart(filtered["OS_STATUS"].fillna("Unknown").value_counts())

	t3, t4 = st.columns(2)
	if "COMPLETION_YEAR" in filtered.columns:
		year = filtered["COMPLETION_YEAR"].dropna()
		if not year.empty:
			decades = ((year // 10) * 10).astype(int).astype(str) + "s"
			t3.write("Development Trend by Decade")
			t3.bar_chart(decades.value_counts().sort_index())

	if "LAST_UPDATE" in filtered.columns:
		update_years = filtered["last_update_dt"].dropna().dt.year
		if not update_years.empty:
			t4.write("Data Maintenance Trend (LAST_UPDATE Year)")
			t4.bar_chart(update_years.value_counts().sort_index())

	t5, t6 = st.columns(2)
	if {"AMENITY_AREA", "OS_TYPE"}.issubset(filtered.columns):
		cross = pd.crosstab(filtered["OS_TYPE"].fillna("Unknown"), filtered["AMENITY_AREA"].fillna("Unknown"))
		if not cross.empty:
			t5.write("Amenity Availability by OS_TYPE")
			t5.bar_chart(cross)

	if {"BEACH", "OS_TYPE"}.issubset(filtered.columns):
		cross_beach = pd.crosstab(filtered["OS_TYPE"].fillna("Unknown"), filtered["BEACH"].fillna("Unknown"))
		if not cross_beach.empty:
			t6.write("Beach Flag by OS_TYPE")
			t6.bar_chart(cross_beach)

	if "nearest_km" in filtered.columns:
		nn = filtered["nearest_km"].dropna().sort_values().reset_index(drop=True)
		if not nn.empty:
			st.write("Proximity Trend: Nearest Open-Space Distance (km)")
			st.line_chart(nn)

	st.subheader("Spatial Proximity Map (Polygons)")
	polygon_geojson = build_polygon_view_data(filtered, geojson_data)
	if polygon_geojson["features"]:
		center_lat = float(filtered["centroid_lat"].dropna().mean()) if filtered["centroid_lat"].notna().any() else 22.32
		center_lon = float(filtered["centroid_lon"].dropna().mean()) if filtered["centroid_lon"].notna().any() else 114.17
		layer = pdk.Layer(
			"GeoJsonLayer",
			polygon_geojson,
			pickable=True,
			filled=True,
			stroked=True,
			get_line_color=[25, 25, 25, 220],
			get_fill_color="properties.fill_color",
			line_width_min_pixels=2,
		)
		tooltip: Any = {
			"html": "<b>{OS_TYPE}</b><br/>{OS_STATUS}<br/>{ADDRESS}<br/>Nearest space: {nearest_km} km<br/>Completion: {COMPLETION_YEAR}<br/>Last update: {LAST_UPDATE}",
			"style": {"backgroundColor": "#1b1f24", "color": "white"},
		}
		st.pydeck_chart(
			pdk.Deck(
				map_provider="carto",
				map_style="light",
				initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=25),
				layers=[layer],
				tooltip=tooltip,
			),
			use_container_width=True,
		)
		st.caption("Color encodes nearest neighboring open space: green <= 0.5 km, orange <= 2 km, red > 2 km.")
	else:
		st.info("No polygon features available for the selected filters.")

	st.subheader("Raw Data")
	visible_cols = [c for c in filtered.columns if c not in HELPER_COLUMNS and c != "nearest_km"] + ["nearest_km"]
	st.dataframe(filtered[visible_cols], use_container_width=True, height=360)


if __name__ == "__main__":
	main()
