from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="HK Food & Restaurant Analysis", layout="wide")

DATA_DIR = Path(__file__).parent
CSV_PATH = DATA_DIR / "restauraunt.csv"

# ── colour palette per category ─────────────────────────────────────────────
CATEGORY_COLORS = {
    "Markets":        "#3498DB",
    "Licences":       "#2ECC71",
    "Hawker":         "#E67E22",
    "Livestock":      "#9B59B6",
    "Water_Sampling": "#1ABC9C",
}

METRIC_LABELS = {
    "No_of_Markets":              "Number of Markets",
    "Market_Stalls":              "Market Stalls",
    "Food_Business_Licences_Full": "Food Business Licences (Full)",
    "Liquor_Licences":            "Liquor Licences",
    "Fixed_Pitch":                "Fixed-Pitch Hawkers",
    "Itinerant":                  "Itinerant Hawkers",
    "Total_Slaughtered":          "Livestock Slaughtered",
    "Fish_Tank":                  "Fish-Tank Water Samples",
}

CATEGORY_DESCRIPTIONS = {
    "Markets":        "Government-run wet markets and market stalls under the Food and Environmental Hygiene Department (FEHD).",
    "Licences":       "Licences issued for food businesses (full licences) and liquor sale, reflecting formal F&B activity.",
    "Hawker":         "Licensed street hawkers split between permanent fixed-pitch stalls and mobile itinerant traders.",
    "Livestock":      "Total livestock slaughtered at public abattoirs, a proxy for Hong Kong's meat-supply demand.",
    "Water_Sampling": "Fish-tank water samples tested for compliance, indicating the scale of live seafood retail premises.",
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    df["Metric_Label"] = df["Metric"].map(METRIC_LABELS).fillna(df["Metric"])
    return df


def yoy_delta(series: pd.Series) -> float:
    """Return absolute year-on-year change between the two most recent values."""
    s = series.dropna().sort_index()
    if len(s) < 2:
        return 0.0
    return float(s.iloc[-1] - s.iloc[-2])


def yoy_pct(series: pd.Series) -> float:
    s = series.dropna().sort_index()
    if len(s) < 2 or s.iloc[-2] == 0:
        return 0.0
    return float((s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2]) * 100)


def trend_arrow(delta: float) -> str:
    if delta > 0:
        return "▲"
    if delta < 0:
        return "▼"
    return "–"


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("🍜 HK Food & Restaurant Sector Analysis")
    st.caption(
        "Source: Food and Environmental Hygiene Department (FEHD) — annual statistics 2022–2025. "
        "Data file: `restauraunt.csv`"
    )

    df = load_data()
    years = sorted(df["Year"].unique())
    latest_year = max(years)
    prev_year = latest_year - 1

    # ── 0. sidebar filters ───────────────────────────────────────────────────
    st.sidebar.header("Filters")
    selected_categories = st.sidebar.multiselect(
        "Categories",
        options=sorted(df["Category"].unique()),
        default=sorted(df["Category"].unique()),
    )
    year_range = st.sidebar.slider(
        "Year range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years))),
    )

    filtered = df[
        df["Category"].isin(selected_categories)
        & df["Year"].between(year_range[0], year_range[1])
    ]

    # ── 1. KPI snapshot ──────────────────────────────────────────────────────
    st.subheader("1 · Key Metrics — Latest Year Snapshot")
    st.caption(
        f"Values for **{latest_year}** with year-on-year change vs {prev_year}."
    )

    latest_df = df[df["Year"] == latest_year]
    cols = st.columns(len(latest_df))
    for col, (_, row) in zip(cols, latest_df.iterrows()):
        series = (
            df[df["Metric"] == row["Metric"]]
            .set_index("Year")["Value"]
            .sort_index()
        )
        delta = yoy_delta(series)
        pct = yoy_pct(series)
        arrow = trend_arrow(delta)
        col.metric(
            label=row["Metric_Label"],
            value=f"{row['Value']:,}",
            delta=f"{arrow} {abs(int(delta)):,} ({pct:+.1f}%)",
        )

    st.divider()

    # ── 2. category descriptions ─────────────────────────────────────────────
    st.subheader("2 · Category Overview")
    for cat in sorted(df["Category"].unique()):
        if cat not in selected_categories:
            continue
        color = CATEGORY_COLORS.get(cat, "#888888")
        st.markdown(
            f"<span style='color:{color}; font-weight:700'>{cat}</span> — "
            f"{CATEGORY_DESCRIPTIONS.get(cat, '')}",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 3. trend lines ───────────────────────────────────────────────────────
    st.subheader("3 · Trend Lines (filtered year range)")

    for cat in sorted(filtered["Category"].unique()):
        cat_df = filtered[filtered["Category"] == cat]
        metrics = cat_df["Metric"].unique()

        color = CATEGORY_COLORS.get(cat, "#888888")
        st.markdown(
            f"<span style='color:{color}; font-weight:700; font-size:1.05rem'>"
            f"► {cat}</span>",
            unsafe_allow_html=True,
        )

        pivot = (
            cat_df.pivot_table(index="Year", columns="Metric_Label", values="Value")
            .reset_index()
        )

        # Split into columns if multiple metrics exist
        metric_cols = [c for c in pivot.columns if c != "Year"]
        chart_cols = st.columns(len(metric_cols))
        for i, metric in enumerate(metric_cols):
            sub = pivot[["Year", metric]].dropna()
            chart_cols[i].line_chart(
                sub.set_index("Year"),
                height=220,
                color=color,
            )
            chart_cols[i].caption(metric)

    st.divider()

    # ── 4. year-on-year change ───────────────────────────────────────────────
    st.subheader("4 · Year-on-Year Absolute Change")

    yoy_rows = []
    for metric in df["Metric"].unique():
        m_df = df[df["Metric"] == metric].set_index("Year")["Value"].sort_index()
        for i in range(1, len(m_df)):
            yr = int(m_df.index[i])
            if not (year_range[0] <= yr <= year_range[1]):
                continue
            cat = df.loc[df["Metric"] == metric, "Category"].iloc[0]
            if cat not in selected_categories:
                continue
            yoy_rows.append(
                {
                    "Category": cat,
                    "Metric": METRIC_LABELS.get(metric, metric),
                    "Year": yr,
                    "YoY Change": int(m_df.iloc[i] - m_df.iloc[i - 1]),
                    "YoY %": round((m_df.iloc[i] - m_df.iloc[i - 1]) / abs(m_df.iloc[i - 1]) * 100, 2)
                    if m_df.iloc[i - 1] != 0 else 0.0,
                }
            )

    if yoy_rows:
        yoy_df = pd.DataFrame(yoy_rows)

        # bar chart of YoY % change
        pivot_yoy = yoy_df.pivot_table(
            index="Metric", columns="Year", values="YoY %"
        ).fillna(0)
        st.bar_chart(pivot_yoy, height=320)

        st.caption("Table: absolute and percentage year-on-year changes")
        st.dataframe(
            yoy_df.sort_values(["Category", "Metric", "Year"]).reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No year-on-year data available for the selected filters.")

    st.divider()

    # ── 5. growth index (2022 = 100) ─────────────────────────────────────────
    st.subheader("5 · Indexed Growth (Base Year = 100)")
    st.caption(
        "All series re-based to 100 at the earliest available year so metrics "
        "on very different scales can be compared directly."
    )

    base_year = year_range[0]
    index_rows = []
    for metric in df["Metric"].unique():
        cat = df.loc[df["Metric"] == metric, "Category"].iloc[0]
        if cat not in selected_categories:
            continue
        m_df = (
            df[df["Metric"] == metric]
            .set_index("Year")["Value"]
            .sort_index()
        )
        base_val = m_df.get(base_year)
        if base_val is None or base_val == 0:
            continue
        for yr, val in m_df.items():
            if year_range[0] <= yr <= year_range[1]:
                index_rows.append(
                    {
                        "Year": yr,
                        METRIC_LABELS.get(metric, metric): round(val / base_val * 100, 1),
                    }
                )

    if index_rows:
        idx_df = (
            pd.DataFrame(index_rows)
            .groupby("Year")
            .first()
            .reset_index()
            .set_index("Year")
        )
        st.line_chart(idx_df, height=340)

    st.divider()

    # ── 6. full data table ───────────────────────────────────────────────────
    st.subheader("6 · Full Dataset")
    display_df = filtered[["Category", "Metric_Label", "Year", "Value"]].rename(
        columns={"Metric_Label": "Metric"}
    )
    st.dataframe(
        display_df.sort_values(["Category", "Metric", "Year"]).reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )

    # ── 7. observations ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("7 · Key Observations")

    observations = []

    # Decline in hawkers
    hawker_df = df[df["Category"] == "Hawker"].pivot_table(index="Year", columns="Metric", values="Value")
    if "Fixed_Pitch" in hawker_df.columns:
        chg = int(hawker_df["Fixed_Pitch"].iloc[-1] - hawker_df["Fixed_Pitch"].iloc[0])
        pct = chg / hawker_df["Fixed_Pitch"].iloc[0] * 100
        observations.append(
            f"**Fixed-pitch hawker stalls** fell by {abs(chg):,} ({pct:.1f}%) over "
            f"{int(hawker_df.index[0])}–{int(hawker_df.index[-1])}, reflecting the long-run "
            "decline of street hawking in Hong Kong."
        )

    # Livestock trend
    live_df = df[df["Category"] == "Livestock"].pivot_table(index="Year", columns="Metric", values="Value")
    if "Total_Slaughtered" in live_df.columns:
        chg = int(live_df["Total_Slaughtered"].iloc[-1] - live_df["Total_Slaughtered"].iloc[0])
        pct = chg / live_df["Total_Slaughtered"].iloc[0] * 100
        observations.append(
            f"**Livestock slaughtered** changed by {chg:+,} ({pct:+.1f}%) over the period, "
            "indicating shifting meat-supply patterns at public abattoirs."
        )

    # Water sampling rise
    ws_df = df[df["Category"] == "Water_Sampling"].pivot_table(index="Year", columns="Metric", values="Value")
    if "Fish_Tank" in ws_df.columns:
        chg = int(ws_df["Fish_Tank"].iloc[-1] - ws_df["Fish_Tank"].iloc[0])
        pct = chg / ws_df["Fish_Tank"].iloc[0] * 100
        observations.append(
            f"**Fish-tank water samples** increased by {chg:+,} ({pct:+.1f}%) — suggesting "
            "either expanded live-seafood retail or intensified hygiene inspection coverage."
        )

    # Licence shift
    lic_df = df[df["Category"] == "Licences"].pivot_table(index="Year", columns="Metric", values="Value")
    if "Food_Business_Licences_Full" in lic_df.columns and "Liquor_Licences" in lic_df.columns:
        food_chg_pct = (lic_df["Food_Business_Licences_Full"].iloc[-1] - lic_df["Food_Business_Licences_Full"].iloc[0]) / lic_df["Food_Business_Licences_Full"].iloc[0] * 100
        liq_chg_pct  = (lic_df["Liquor_Licences"].iloc[-1] - lic_df["Liquor_Licences"].iloc[0]) / lic_df["Liquor_Licences"].iloc[0] * 100
        observations.append(
            f"**Food business licences** changed {food_chg_pct:+.1f}% while **liquor licences** "
            f"changed {liq_chg_pct:+.1f}% over the same window — the divergence hints at "
            "post-pandemic shifts in the dining and nightlife segment."
        )

    for obs in observations:
        st.markdown(f"- {obs}")

    st.caption("Analysis generated automatically from `restauraunt.csv` data.")


if __name__ == "__main__":
    main()
