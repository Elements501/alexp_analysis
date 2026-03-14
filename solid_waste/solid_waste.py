from pathlib import Path
from typing import Any
import csv

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Solid Waste Analyzer", layout="wide")

DATA_DIR = Path(__file__).parent

# Explicit catalog keeps meaning clear and scales when new CSVs are added later.
DATASET_CATALOG = {
	"solid-waste-disposal-rate-en-2024.csv": {
		"title": "Per Capita Waste Disposal Rates (Selected Categories)",
		"description": "This dataset is for per capita waste disposal rates of selected waste categories.",
		"kind": "per_capita_rates",
		"column_map": {
			"year": "year",
			"waste_cat_en": "category",
			"pc_disposal_rate": "value",
		},
		"source_value_col": "pc_disposal_rate",
		"value_label": "Per-capita disposal rate",
		"value_unit": "kg/person/day",
	},
	"solid-waste-generation-quantity-en-2024.csv": {
		"title": "Waste Disposed / Generated Quantity",
		"description": "Time series of waste quantity (interpreted as disposed/generated quantity in the source file).",
		"kind": "quantity",
		"column_map": {
			"year": "year",
			"waste_cat_en": "category",
			"generation_q": "value",
		},
		"source_value_col": "generation_q",
		"value_label": "Disposed/generated quantity",
		"value_unit": "tonnes/year",
	},
	"solid-waste-recovery-quantity-en-2024.csv": {
		"title": "Waste Recovered Quantity",
		"description": "Time series of recovered waste quantity.",
		"kind": "quantity",
		"column_map": {
			"year": "year",
			"waste_cat_en": "category",
			"recovery_q": "value",
		},
		"source_value_col": "recovery_q",
		"value_label": "Recovered quantity",
		"value_unit": "tonnes/year",
	},
	"recyclables-from-muncipial.csv": {
		"title": "Composition of Recyclables Recovered from Municipal Solid Waste",
		"description": "Breakdown by recyclable type (paper, plastics, metals, etc.) with yearly totals in thousand tonnes.",
		"kind": "municipal_composition",
		"column_map": {
			"Year": "year",
			"Composition": "category",
			"value": "value",
		},
		"source_value_col": "value",
		"value_label": "Recovered quantity by composition",
		"value_unit": "thousand tonnes/year",
	},
}

DATASET_ORDER = [
	"solid-waste-disposal-rate-en-2024.csv",
	"solid-waste-generation-quantity-en-2024.csv",
	"solid-waste-recovery-quantity-en-2024.csv",
	"recyclables-from-muncipial.csv",
]


@st.cache_data(show_spinner=False)
def load_dataset(csv_path: Path, catalog_entry: dict[str, Any]) -> pd.DataFrame:
	if catalog_entry.get("kind") == "municipal_composition":
		df = load_municipal_composition(csv_path)
	else:
		df = pd.read_csv(csv_path)
	col_map = catalog_entry.get("column_map", {})

	for source_col, target_col in col_map.items():
		if source_col in df.columns:
			df[target_col] = df[source_col]

	if "year" in df.columns:
		df["year"] = pd.to_numeric(df["year"], errors="coerce")
	if "value" in df.columns:
		df["value"] = pd.to_numeric(df["value"], errors="coerce")
	if "category" in df.columns:
		df["category"] = df["category"].astype(str)

	return df


def load_municipal_composition(csv_path: Path) -> pd.DataFrame:
	with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
		rows = list(csv.reader(f))

	composition_header_idx = None
	year_header_idx = None
	for idx, row in enumerate(rows):
		first = row[0].strip() if row else ""
		if first == "Composition":
			composition_header_idx = idx
		if first == "Year":
			year_header_idx = idx
			break

	if composition_header_idx is None or year_header_idx is None:
		return pd.DataFrame(columns=["year", "category", "value", "is_total"])

	columns = [c.strip() for c in rows[composition_header_idx]]
	data_rows = rows[year_header_idx + 1 :]

	records: list[dict[str, Any]] = []
	for row in data_rows:
		if not row:
			continue
		year_raw = row[0].strip() if len(row) > 0 else ""
		if not year_raw.isdigit():
			continue
		year = int(year_raw)
		for col_idx in range(1, min(len(columns), len(row))):
			cat = columns[col_idx].strip()
			if not cat:
				continue
			val_raw = row[col_idx].strip()
			if val_raw in {"", "-", "N.A.", "N.A"}:
				continue
			try:
				val = float(val_raw.replace(",", ""))
			except ValueError:
				continue
			records.append(
				{
					"year": year,
					"category": cat,
					"value": val,
					"is_total": cat.strip().lower() == "total",
				}
			)

	return pd.DataFrame(records)


def detect_time_series_issues(df: pd.DataFrame, value_label: str) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame(columns=["issue", "rule", "affected_count", "severity", "status"])

	checks: list[dict[str, Any]] = []

	def add_check(issue: str, rule: str, mask: pd.Series, severity: str) -> None:
		count = int(mask.fillna(False).sum())
		checks.append(
			{
				"issue": issue,
				"rule": rule,
				"affected_count": count,
				"severity": severity,
				"status": "detected" if count > 0 else "clear",
			}
		)

	add_check("Missing year", "year is null", df["year"].isna(), "high")
	add_check("Missing category", "category is null/blank", df["category"].isna() | (df["category"].astype(str).str.strip() == ""), "high")
	add_check(f"Missing {value_label.lower()}", "value is null", df["value"].isna(), "high")
	add_check(f"Non-positive {value_label.lower()}", "value <= 0", df["value"] <= 0, "medium")

	dup_mask = df.duplicated(subset=["year", "category"], keep=False)
	add_check("Duplicate year-category rows", "duplicated on (year, category)", dup_mask, "medium")

	# Category-level temporal checks.
	spike_flags = []
	missing_year_flags = []
	for _, g in df.dropna(subset=["year", "value", "category"]).groupby("category"):
		g = g.sort_values("year")
		if len(g) > 1:
			yoy = g["value"].pct_change().abs()
			spike_flags.extend((yoy > 0.15).fillna(False).tolist())
			year_gap = g["year"].diff()
			missing_year_flags.extend((year_gap > 1).fillna(False).tolist())

	checks.append(
		{
			"issue": "Large year-over-year jump",
			"rule": "abs(pct_change) > 15% within same category",
			"affected_count": int(sum(spike_flags)),
			"severity": "medium",
			"status": "detected" if any(spike_flags) else "clear",
		}
	)
	checks.append(
		{
			"issue": "Missing years within category timeline",
			"rule": "year gap > 1 within same category",
			"affected_count": int(sum(missing_year_flags)),
			"severity": "low",
			"status": "detected" if any(missing_year_flags) else "clear",
		}
	)

	return pd.DataFrame(checks)


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame(columns=["category", "records", "avg_rate", "min_rate", "max_rate", "latest_year", "latest_rate", "trend_change_pct"])

	rows = []
	for cat, g in df.dropna(subset=["year", "value"]).groupby("category"):
		g = g.sort_values("year")
		first = float(g.iloc[0]["value"])
		last = float(g.iloc[-1]["value"])
		change_pct = ((last - first) / first * 100.0) if first != 0 else float("nan")
		rows.append(
			{
				"category": cat,
				"records": int(len(g)),
				"avg_rate": round(float(g["value"].mean()), 3),
				"min_rate": round(float(g["value"].min()), 3),
				"max_rate": round(float(g["value"].max()), 3),
				"latest_year": int(g.iloc[-1]["year"]),
				"latest_rate": round(last, 3),
				"trend_change_pct": round(change_pct, 2),
			}
		)

	return pd.DataFrame(rows).sort_values("latest_rate", ascending=False)


def render_dataset_section(section_idx: int, dataset_name: str, meta: dict[str, Any]) -> pd.DataFrame:
	df = load_dataset(DATA_DIR / dataset_name, meta)

	st.subheader(f"{section_idx}. {meta['title']}")
	st.caption(meta["description"])
	if dataset_name == "solid-waste-disposal-rate-en-2024.csv":
		st.success("Confirmed: solid-waste-disposal-rate-en-2024.csv is for Per capita waste disposal rates of selected waste categories.")

	required_cols = ["year", "category", "value"]
	missing_required = [c for c in required_cols if c not in df.columns]
	if missing_required:
		st.error(f"Missing required mapped columns for {dataset_name}: {missing_required}")
		return df.iloc[0:0]

	filter_box = st.expander(f"Filters for {meta['title']}", expanded=False)
	with filter_box:
		categories = sorted(df["category"].dropna().unique().tolist())
		selected_categories = st.multiselect(
			"Category",
			categories,
			default=categories,
			key=f"categories_{dataset_name}",
		)
		min_year = int(df["year"].min()) if df["year"].notna().any() else 0
		max_year = int(df["year"].max()) if df["year"].notna().any() else 0
		selected_years = (
			st.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year), key=f"years_{dataset_name}")
			if min_year <= max_year
			else (0, 0)
		)

	filtered = df.copy()
	if selected_categories:
		filtered = filtered[filtered["category"].isin(selected_categories)]
	filtered = filtered[(filtered["year"] >= selected_years[0]) & (filtered["year"] <= selected_years[1])]

	issues_df = detect_time_series_issues(filtered, meta["value_label"])
	category_summary = build_category_summary(filtered)

	latest_year = int(filtered["year"].max()) if filtered["year"].notna().any() else 0
	latest_slice = filtered[filtered["year"] == latest_year] if latest_year else filtered.iloc[0:0]
	latest_avg = float(latest_slice["value"].mean()) if not latest_slice.empty else float("nan")

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Rows in View", f"{len(filtered):,}")
	c2.metric("Categories", f"{filtered['category'].nunique():,}")
	c3.metric("Latest Year", f"{latest_year:,}" if latest_year else "N/A")
	c4.metric(f"Avg {meta['value_label']} ({meta['value_unit']})", f"{latest_avg:.3f}" if pd.notna(latest_avg) else "N/A")

	st.write("Possible Issues and Detection")
	st.dataframe(issues_df, width="stretch", height=240)

	left, right = st.columns(2)
	with left:
		st.write("Trend by Category")
		if filtered.dropna(subset=["year", "value"]).empty:
			st.info("No plottable year/value records under current filters.")
		else:
			pivot = filtered.pivot_table(index="year", columns="category", values="value", aggfunc="mean").sort_index()
			st.line_chart(pivot)

	with right:
		st.write("Latest-Year Category Comparison")
		if latest_slice.empty:
			st.info("No data available for latest year under current filters.")
		else:
			bar = (
				latest_slice.groupby("category", as_index=False)
				.agg(value=("value", "mean"))
				.sort_values(by="value", ascending=False)
			)
			st.bar_chart(bar.set_index("category")["value"])

	st.write("Category Summary")
	st.dataframe(category_summary, width="stretch", height=240)

	st.write("Data Table")
	source_value_col = meta.get("source_value_col", "value")
	candidate_cols = ["year", "category", "value", "waste_cat_en", source_value_col]
	visible: list[str] = []
	for c in candidate_cols:
		if c in filtered.columns and c not in visible:
			visible.append(c)
	st.dataframe(filtered[visible].sort_values(["year", "category"]), width="stretch", height=260)

	st.markdown("---")
	return df


def render_cross_dataset_comparison(loaded_data: dict[str, pd.DataFrame]) -> None:
	gen_name = "solid-waste-generation-quantity-en-2024.csv"
	rec_name = "solid-waste-recovery-quantity-en-2024.csv"
	mun_name = "recyclables-from-muncipial.csv"

	if gen_name not in loaded_data or rec_name not in loaded_data:
		return

	gen = loaded_data[gen_name].dropna(subset=["year", "value"])
	rec = loaded_data[rec_name].dropna(subset=["year", "value"])
	if gen.empty or rec.empty:
		return

	gen_by_year = gen.groupby("year", as_index=False).agg(disposed_or_generated_q=("value", "sum"))
	rec_by_year = rec.groupby("year", as_index=False).agg(recovered_q=("value", "sum"))
	combo = gen_by_year.merge(rec_by_year, on="year", how="inner")
	if combo.empty:
		return

	combo["recovery_rate_pct"] = (
		combo["recovered_q"].div(combo["disposed_or_generated_q"]).mul(100.0).round(2)
	)

	st.subheader("4. Cross-Dataset Comparison: Recovered vs Disposed/Generated")
	st.caption("This comparison uses the recovery and generation quantity datasets aligned by year.")

	left, right = st.columns(2)
	with left:
		st.write("Quantity Trend")
		st.line_chart(combo.set_index("year")[["disposed_or_generated_q", "recovered_q"]])
	with right:
		st.write("Recovery Rate (%)")
		st.line_chart(combo.set_index("year")["recovery_rate_pct"])

	st.dataframe(combo, width="stretch", height=250)

	if mun_name not in loaded_data:
		return

	mun = loaded_data[mun_name].dropna(subset=["year", "value"])
	if mun.empty:
		return

	mun_total = mun[mun["category"].str.lower() == "total"].copy()
	if mun_total.empty:
		return

	mun_total["municipal_total_tonnes"] = mun_total["value"] * 1000.0
	rel = rec_by_year.merge(mun_total[["year", "municipal_total_tonnes"]], on="year", how="inner")
	if rel.empty:
		st.warning("No overlapping years between recovery quantity and municipal composition total.")
		return

	rel["gap_tonnes"] = rel["recovered_q"] - rel["municipal_total_tonnes"]
	rel["gap_pct_vs_recovery"] = rel["gap_tonnes"].div(rel["recovered_q"]).mul(100.0).round(2)

	issues: list[dict[str, Any]] = []
	high_gap = rel["gap_pct_vs_recovery"].abs() > 5
	issues.append(
		{
			"issue": "Recovery total mismatch between datasets",
			"rule": "abs(gap_pct_vs_recovery) > 5%",
			"affected_years": int(high_gap.sum()),
			"status": "detected" if high_gap.any() else "clear",
		}
	)

	common_years = len(rel)
	rec_years = rec_by_year["year"].nunique()
	mun_years = mun_total["year"].nunique()
	issues.append(
		{
			"issue": "Year coverage misalignment",
			"rule": "common_years < min(recovery_years, municipal_years)",
			"affected_years": int(min(rec_years, mun_years) - common_years),
			"status": "detected" if common_years < min(rec_years, mun_years) else "clear",
		}
	)

	mun_non_total = mun[mun["category"].str.lower() != "total"].copy()
	latest_year = int(mun_non_total["year"].max()) if not mun_non_total.empty else 0
	latest_non_total = mun_non_total[mun_non_total["year"] == latest_year]
	if not latest_non_total.empty:
		top_share = float(latest_non_total["value"].max() / latest_non_total["value"].sum() * 100.0)
		issues.append(
			{
				"issue": "Single-category concentration in municipal recyclables",
				"rule": "max category share in latest year > 60%",
				"affected_years": 1 if top_share > 60 else 0,
				"status": "detected" if top_share > 60 else "clear",
			}
		)

	st.subheader("5. Municipal Composition Relation Analysis")
	st.caption("Compares municipal recyclable composition total (converted from thousand tonnes) with recovery quantity dataset.")

	left2, right2 = st.columns(2)
	with left2:
		st.write("Recovery vs Municipal Total (tonnes)")
		st.line_chart(rel.set_index("year")[["recovered_q", "municipal_total_tonnes"]])
	with right2:
		st.write("Gap Percentage vs Recovery")
		st.line_chart(rel.set_index("year")["gap_pct_vs_recovery"])

	st.write("Relation Issues Found")
	st.dataframe(pd.DataFrame(issues), width="stretch", height=200)
	st.write("Year-level Comparison")
	st.dataframe(rel, width="stretch", height=240)


def main() -> None:
	st.title("Solid Waste Dataset Analyzer")
	st.caption("Multi-dataset solid-waste analysis with explicit dataset meaning and sequential sections")

	available_csv = sorted(p.name for p in DATA_DIR.glob("*.csv"))
	if not available_csv:
		st.error("No CSV datasets found in this folder.")
		return

	missing_catalog = [n for n in DATASET_ORDER if n not in DATASET_CATALOG]
	if missing_catalog:
		st.error(f"Missing dataset catalog entries: {missing_catalog}")
		return

	st.subheader("Dataset Sequence")
	st.caption("The sections below are shown one by one: disposal-rate, disposed/generated quantity, then recovered quantity.")

	loaded_data: dict[str, pd.DataFrame] = {}
	section_idx = 1
	for dataset_name in DATASET_ORDER:
		if dataset_name not in available_csv:
			st.warning(f"Dataset not found in folder and skipped: {dataset_name}")
			continue
		meta = DATASET_CATALOG[dataset_name]
		loaded_data[dataset_name] = render_dataset_section(section_idx, dataset_name, meta)
		section_idx += 1

	render_cross_dataset_comparison(loaded_data)

	st.subheader("Map Availability")
	st.info(
		"No map is shown for current solid-waste CSVs because they are time-series tables without coordinates. "
		"When future solid-waste datasets include spatial columns, this app can add maps with the same pattern used in other dashboards."
	)


if __name__ == "__main__":
	main()
