import csv
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Population Income Analyzer", layout="wide")

DATA_FILE = Path(__file__).with_name("size_and_wage.csv")


def quarter_sort_key(value: str) -> int:
	order = {"Annual": 0, "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
	return order.get(str(value).strip(), 99)


def income_sort_key(value: str) -> float:
	text = str(value).strip().replace(",", "")
	if text == "Total":
		return float("inf")
	if text.startswith("<"):
		match = pd.Series([text]).str.extract(r"(\d+(?:\.\d+)?)", expand=False).iloc[0]
		return float(match) - 0.5 if pd.notna(match) else -1.0
	if text.startswith("≥") or text.startswith(">="):
		match = pd.Series([text]).str.extract(r"(\d+(?:\.\d+)?)", expand=False).iloc[0]
		return float(match) if pd.notna(match) else float("inf")
	match = pd.Series([text]).str.extract(r"(\d+(?:\.\d+)?)", expand=False).iloc[0]
	return float(match) if pd.notna(match) else float("inf")


@st.cache_data(show_spinner=False)
def load_data(file_path: Path) -> pd.DataFrame:
	with file_path.open("r", encoding="utf-8-sig", newline="") as f:
		rows = list(csv.reader(f))

	header_idx = None
	for idx, row in enumerate(rows):
		if len(row) >= 3 and row[0].strip() == "Year" and row[1].strip() == "Quarter":
			header_idx = idx
			break

	if header_idx is None:
		return pd.DataFrame()

	data_rows = rows[header_idx + 1 :]
	records = []
	for row in data_rows:
		if len(row) < 7:
			continue
		year_raw = row[0].strip()
		if not year_raw.isdigit():
			continue
		records.append(
			{
				"year": pd.to_numeric(year_raw, errors="coerce"),
				"quarter": row[1].strip() if row[1].strip() != "" else "Annual",
				"income_range": row[2].strip(),
				"econ_active_no_000": pd.to_numeric(row[3], errors="coerce"),
				"econ_active_share_pct": pd.to_numeric(row[4], errors="coerce"),
				"domestic_no_000": pd.to_numeric(row[5], errors="coerce"),
				"domestic_share_pct": pd.to_numeric(row[6], errors="coerce"),
			}
		)

	df = pd.DataFrame(records)
	df["is_total"] = df["income_range"].eq("Total")
	df["is_group_heading"] = df["income_range"].str.startswith("< ") | df["income_range"].str.startswith("≥ ")
	df["is_indented_band"] = df["income_range"].str.startswith("     ")
	df["income_range_clean"] = df["income_range"].str.strip()
	df["income_sort_key"] = df["income_range_clean"].apply(income_sort_key)
	df["quarter_sort_key"] = df["quarter"].apply(quarter_sort_key)
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

	add("Missing year", "year is null", df["year"].isna(), "high")
	add("Missing income range", "income_range is blank", df["income_range_clean"].eq(""), "high")
	add("Missing economically active count", "econ_active_no_000 is null", df["econ_active_no_000"].isna(), "medium")
	add("Missing domestic count", "domestic_no_000 is null", df["domestic_no_000"].isna(), "medium")
	dup = df.duplicated(subset=["year", "quarter", "income_range_clean"], keep=False)
	add("Duplicate year-quarter-income rows", "duplicated on year + quarter + income_range", dup, "medium")

	annual_totals = df[df["quarter"].eq("Annual") & df["is_total"]]
	add(
		"Annual total domestic share not 100%",
		"Annual Total domestic_share_pct != 100",
		annual_totals["domestic_share_pct"].round(1) != 100.0,
		"low",
	)
	add(
		"Annual total econ-active share not 100%",
		"Annual Total econ_active_share_pct != 100",
		annual_totals["econ_active_share_pct"].round(1) != 100.0,
		"low",
	)

	return pd.DataFrame(issues)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
	base = df[~df["is_total"] & ~df["is_group_heading"]].copy()
	if base.empty:
		return pd.DataFrame(columns=["income_range", "records", "avg_domestic_no_000", "avg_domestic_share_pct", "avg_econ_active_no_000"])

	return (
		base.groupby("income_range_clean", as_index=False)
		.agg(
			records=("income_range_clean", "count"),
			income_sort_key=("income_sort_key", "min"),
			avg_domestic_no_000=("domestic_no_000", "mean"),
			avg_domestic_share_pct=("domestic_share_pct", "mean"),
			avg_econ_active_no_000=("econ_active_no_000", "mean"),
		)
		.sort_values("income_sort_key")
	)


def main() -> None:
	st.title("Population and Household Income Analyzer")
	st.caption("Analysis of domestic households by monthly household income range, including economically active vs all domestic households")

	if not DATA_FILE.exists():
		st.error(f"Data file not found: {DATA_FILE}")
		return

	df = load_data(DATA_FILE)
	if df.empty:
		st.error("Unable to parse the CSV structure.")
		return

	with st.sidebar:
		st.header("Filters")
		quarter_options = df["quarter"].dropna().unique().tolist()
		selected_quarters = st.multiselect("Quarter", quarter_options, default=quarter_options)
		min_year = int(df["year"].min())
		max_year = int(df["year"].max())
		year_range = st.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

	filtered = df.copy()
	if selected_quarters:
		filtered = filtered[filtered["quarter"].isin(selected_quarters)]
	filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

	issues_df = detect_issues(filtered)
	summary_df = build_summary(filtered)

	annual_totals = filtered[filtered["quarter"].eq("Annual") & filtered["is_total"]]
	latest_annual = annual_totals.sort_values("year").tail(1)
	latest_year = int(latest_annual["year"].iloc[0]) if not latest_annual.empty else int(filtered["year"].max())

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Rows in View", f"{len(filtered):,}")
	c2.metric("Years", f"{filtered['year'].nunique():,}")
	c3.metric("Quarters", f"{filtered['quarter'].nunique():,}")
	c4.metric("Latest Annual Year", f"{latest_year:,}")

	st.subheader("Possible Issues and Detection")
	st.dataframe(issues_df, width="stretch", height=230)

	base = filtered[~filtered["is_total"] & ~filtered["is_group_heading"]].copy()

	left, right = st.columns(2)
	with left:
		st.subheader("Domestic Household Count by Income Band")
		if base.empty:
			st.info("No detailed income-band records under current filters.")
		else:
			chart = (
				base.groupby("income_range_clean", as_index=False)
				.agg(income_sort_key=("income_sort_key", "min"), domestic_no_000=("domestic_no_000", "mean"))
				.sort_values("income_sort_key")
			)
			st.bar_chart(chart.set_index("income_range_clean")["domestic_no_000"])

	with right:
		st.subheader("Economically Active Share by Income Band")
		if base.empty:
			st.info("No detailed income-band records under current filters.")
		else:
			chart = (
				base.groupby("income_range_clean", as_index=False)
				.agg(income_sort_key=("income_sort_key", "min"), econ_active_share_pct=("econ_active_share_pct", "mean"))
				.sort_values("income_sort_key")
			)
			st.bar_chart(chart.set_index("income_range_clean")["econ_active_share_pct"])

	st.subheader("Trend of Annual Totals")
	if annual_totals.empty:
		st.info("No annual total rows under current filters.")
	else:
		trend = annual_totals.set_index("year")[["econ_active_no_000", "domestic_no_000"]].sort_index()
		st.line_chart(trend)

	st.subheader("Summary Table by Income Band")
	st.dataframe(summary_df.drop(columns=["income_sort_key"], errors="ignore"), width="stretch", height=320)

	st.subheader("Raw / Parsed Data")
	visible_cols = [
		"year",
		"quarter",
		"income_range",
		"econ_active_no_000", # Number of active households in that band
		"econ_active_share_pct", # Percentage of active households
		"domestic_no_000", # Number of dosmetic household in that band
		"domestic_share_pct", # Percentage of total of household in that band
		"is_total",
		"is_group_heading",
	]
	sorted_filtered = filtered.sort_values(["year", "quarter_sort_key", "income_sort_key", "income_range_clean"])
	st.dataframe(sorted_filtered[visible_cols], width="stretch", height=420)


if __name__ == "__main__":
	main()
