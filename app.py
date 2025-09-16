import streamlit as st
import pandas as pd
from io import StringIO
from plotly.subplots import make_subplots
import plotly.graph_objects as go

DATA = """Country,GDP_per_Capita_PPP,GINI_Index,AQI,Clean_Water_Access_Pct,Life_Expectancy,Healthcare_Index,Safety_Index,Internet_Speed_Mbps,Cost_of_Living_Index
Argentina,26500,42.3,45,99,75.4,65.2,40.4,93.04,29.4
Australia,64500,34.4,28,100,83.2,74.1,58.1,79.18,70.2
Austria,67900,30.8,39,100,81.3,80.0,76.3,100.6,65.1
Belgium,65000,27.2,51,100,81.6,74.6,56.8,113.21,61.1
Brazil,16600,48.9,41,100,72.8,55.5,32.6,186.31,30.2
Canada,58400,33.3,36,99,82.7,71.3,61.9,231.74,64.8
Chile,29900,45.8,65,100,79.0,66.3,55.3,276.77,39.1
China,23300,38.2,79,98,78.2,67.8,75.4,238.04,31.7
Colombia,16500,51.5,48,98,72.8,68.8,36.9,166.9,28.8
Costa Rica,26300,47.2,35,100,77.3,64.5,60.1,108.73,52.3
Denmark,74900,28.3,31,100,81.4,81.8,73.4,247.62,72.3
Egypt,17100,31.5,88,99,70.2,50.1,50.8,80.32,21.0
Finland,60700,27.3,19,100,82.0,75.9,76.5,144.79,63.2
France,55500,32.4,49,100,82.4,79.8,54.7,287.44,63.7
Germany,66000,31.9,43,100,80.9,75.3,65.6,94.78,62.2
Ghana,6400,43.5,61,88,64.9,45.2,45.9,51.08,30.9
Greece,36600,32.9,53,100,80.3,59.2,63.8,62.17,52.0
Hungary,42700,29.8,58,100,74.5,61.3,68.2,212.14,41.7
India,9100,35.7,117,93,67.2,58.5,58.3,62.07,21.2
Indonesia,14600,37.9,86,94,67.6,51.8,54.4,32.13,26.7
Ireland,133600,32.8,33,96,82.4,68.0,60.8,146.47,64.4
Israel,54800,39.0,55,100,82.6,61.7,70.7,226.64,62.7
Italy,51800,35.9,59,100,82.8,67.7,59.3,91.83,56.2
Japan,45500,32.9,40,99,84.5,81.1,80.7,212.06,46.1
Kenya,5600,40.8,72,67,61.4,44.8,36.5,14.65,30.2
Malaysia,33400,41.2,68,97,74.9,68.1,43.7,129.45,30.0
Mexico,23000,45.4,71,100,70.2,60.1,46.1,85.45,40.2
Netherlands,71400,28.2,41,100,81.5,76.5,72.7,201.73,63.1
New Zealand,54000,36.2,25,100,82.0,73.0,59.4,174.64,64.6
Nigeria,5800,35.1,95,72,52.7,40.1,30.7,25.39,30.5
Norway,82500,27.0,29,100,83.2,74.5,66.5,150.87,76.0
Philippines,9000,42.3,75,95,69.3,61.9,42.1,94.1,31.0
Poland,43300,30.2,62,90,75.6,62.0,70.8,181.57,40.8
Portugal,41600,33.5,38,100,81.1,70.8,69.8,205.63,45.1
Qatar,114200,41.1,53,100,79.3,73.3,85.2,189.93,51.3
Saudi Arabia,59000,45.9,91,99,77.9,59.5,78.1,120.25,45.3
Singapore,133100,39.0,44,100,83.1,70.9,71.0,336.45,76.7
South Africa,14300,63.0,66,94,59.3,53.2,19.2,48.4,34.5
South Korea,50200,35.4,60,100,83.5,77.7,75.1,193.49,60.1
Spain,47300,34.3,47,100,82.4,78.8,68.9,245.58,47.3
Sweden,64600,30.0,34,100,82.8,70.7,52.1,174.52,59.3
Switzerland,84300,33.1,30,100,83.8,79.2,78.6,242.32,101.1
Thailand,21100,35.0,81,100,79.3,70.4,67.5,237.05,34.1
Turkey,37300,41.9,73,97,76.0,65.0,59.9,49.11,37.4
United Arab Emirates,87700,26.0,70,100,78.7,67.3,84.9,310.05,55.8
United Kingdom,54600,37.4,46,100,80.7,74.8,53.8,131.77,62.0
United States,80400,41.5,42,100,77.5,75.9,50.3,274.16,70.4
"""

METRIC_CONFIG = {
    "GDP_per_Capita_PPP": {
        "label": "GDP per Capita (PPP, USD)",
        "description": "Purchasing power parity adjusted GDP per person.",
        "higher_is_better": True,
    },
    "GINI_Index": {
        "label": "GINI (Income Inequality)",
        "description": "Lower scores indicate a more equal income distribution.",
        "higher_is_better": False,
    },
    "AQI": {
        "label": "Air Quality Index (AQI)",
        "description": "Lower scores represent cleaner air.",
        "higher_is_better": False,
    },
    "Clean_Water_Access_Pct": {
        "label": "Clean Water Access (%)",
        "description": "Share of the population with access to safe drinking water.",
        "higher_is_better": True,
    },
    "Life_Expectancy": {
        "label": "Life Expectancy (Years)",
        "description": "Average lifespan at birth.",
        "higher_is_better": True,
    },
    "Healthcare_Index": {
        "label": "Healthcare Quality Index",
        "description": "Composite score of healthcare system quality.",
        "higher_is_better": True,
    },
    "Safety_Index": {
        "label": "Safety Index",
        "description": "Perceived safety and crime levels (higher is safer).",
        "higher_is_better": True,
    },
    "Internet_Speed_Mbps": {
        "label": "Internet Speed (Mbps)",
        "description": "Average fixed broadband download speed.",
        "higher_is_better": True,
    },
    "Cost_of_Living_Index": {
        "label": "Cost of Living Index",
        "description": "Lower scores indicate a more affordable cost of living.",
        "higher_is_better": False,
    },
}


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the country-level data as a DataFrame."""
    df = pd.read_csv(StringIO(DATA))
    return df.sort_values("Country").reset_index(drop=True)


def normalize_metrics(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Normalize selected metrics to a 0-100 range where higher is better."""
    normalized = pd.DataFrame(index=df.index)
    for metric in metrics:
        values = df[metric].astype(float)
        min_val = values.min()
        max_val = values.max()
        if max_val == min_val:
            scaled = pd.Series(100.0, index=df.index)
        else:
            scaled = (values - min_val) / (max_val - min_val)
            if not METRIC_CONFIG[metric]["higher_is_better"]:
                scaled = 1 - scaled
            scaled = scaled * 100
        normalized[metric] = scaled
    return normalized


def build_radar_chart(
    df: pd.DataFrame,
    normalized: pd.DataFrame,
    selected_countries: list[str],
    selected_metrics: list[str],
) -> go.Figure:
    """Create a side-by-side radar chart for the selected countries."""
    metric_labels = [METRIC_CONFIG[m]["label"] for m in selected_metrics]
    fig = make_subplots(
        rows=1,
        cols=len(selected_countries),
        specs=[[{"type": "polar"} for _ in selected_countries]],
        subplot_titles=selected_countries,
    )

    for idx, country in enumerate(selected_countries):
        mask = df["Country"] == country
        scores = normalized.loc[mask, selected_metrics].iloc[0].tolist()
        hover_text = [
            f"{METRIC_CONFIG[col]['label']}: {normalized.loc[mask, col].iloc[0]:.1f}" for col in selected_metrics
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=scores,
                theta=metric_labels,
                fill="toself",
                name=country,
                hovertemplate="<b>%{customdata}</b><br>Score: %{r:.1f}<extra></extra>",
                customdata=hover_text,
            ),
            row=1,
            col=idx + 1,
        )

    fig.update_polars(
        radialaxis=dict(
            range=[0, 100],
            showline=False,
            tickvals=[0, 25, 50, 75, 100],
            gridcolor="rgba(128, 128, 128, 0.2)",
        ),
        angularaxis=dict(direction="clockwise"),
    )
    fig.update_layout(
        showlegend=False,
        height=500,
        margin=dict(l=40, r=40, t=60, b=20),
        template="plotly_white",
    )
    return fig


def build_ranking_table(
    df: pd.DataFrame, normalized: pd.DataFrame, selected_metrics: list[str]
) -> pd.DataFrame:
    """Compute an overall livability score and return a ranking table."""
    ranking = normalized[selected_metrics].mean(axis=1).round(1)
    table = pd.DataFrame({
        "Country": df["Country"],
        "Livability Score (0-100)": ranking,
    })
    for metric in selected_metrics:
        table[METRIC_CONFIG[metric]["label"]] = normalized[metric].round(1)
    return table.sort_values("Livability Score (0-100)", ascending=False).reset_index(drop=True)


def render_metric_tooltips(selected_metrics: list[str]) -> None:
    """Show descriptions for the selected metrics."""
    with st.expander("Metric details"):
        for metric in selected_metrics:
            config = METRIC_CONFIG[metric]
            orientation = "Higher values are better" if config["higher_is_better"] else "Lower values are better"
            st.markdown(
                f"**{config['label']}** — {config['description']} ({orientation}.)"
            )


def main() -> None:
    st.set_page_config(page_title="Global Livability Radar", layout="wide")
    st.title("Global Livability Radar")
    st.caption(
        "Compare countries across multiple quality-of-life indicators using normalized radar charts."
    )

    df = load_data()

    metric_options = {config["label"]: column for column, config in METRIC_CONFIG.items()}
    default_metrics = list(metric_options.keys())
    metric_labels_selected = st.multiselect(
        "Metrics to include",
        options=list(metric_options.keys()),
        default=default_metrics,
        help="Pick the indicators that should appear on the radar charts.",
    )

    selected_metrics = [metric_options[label] for label in metric_labels_selected]

    country_options = df["Country"].tolist()
    default_countries = ["Canada", "Germany", "Japan"]
    default_countries = [c for c in default_countries if c in country_options]
    selected_countries = st.multiselect(
        "Countries to compare",
        options=country_options,
        default=default_countries or country_options[:3],
        help="Select one or more countries to visualize side-by-side.",
    )

    if not selected_metrics:
        st.warning("Select at least one metric to build the comparison chart.")
        return
    if not selected_countries:
        st.info("Use the country selector above to add countries to the comparison.")
        return

    normalized = normalize_metrics(df, selected_metrics)

    render_metric_tooltips(selected_metrics)

    st.subheader("Radar comparison")
    st.markdown(
        "Scores are normalized between 0 and 100 relative to the countries in this dataset, "
        "and all metrics are oriented so that higher scores represent more livable conditions."
    )

    radar_fig = build_radar_chart(df, normalized, selected_countries, selected_metrics)
    st.plotly_chart(radar_fig, use_container_width=True)

    st.subheader("Selected country details")
    cols = st.columns(2)
    raw_selected = df[df["Country"].isin(selected_countries)][["Country"] + selected_metrics]
    raw_selected = raw_selected.set_index("Country")
    normalized_selected = normalized[df["Country"].isin(selected_countries)][selected_metrics]
    normalized_selected.index = raw_selected.index

    rename_map = {metric: METRIC_CONFIG[metric]["label"] for metric in selected_metrics}
    with cols[0]:
        st.markdown("**Raw indicator values**")
        st.dataframe(raw_selected.rename(columns=rename_map))
    with cols[1]:
        st.markdown("**Normalized scores (0-100)**")
        st.dataframe(normalized_selected.rename(columns=rename_map).round(1))

    st.subheader("Global ranking across selected metrics")
    ranking_table = build_ranking_table(df, normalized, selected_metrics)
    st.dataframe(ranking_table.style.format({
        "Livability Score (0-100)": "{:.1f}",
        **{METRIC_CONFIG[m]["label"]: "{:.1f}" for m in selected_metrics},
    }), height=600)

    st.markdown("---")
    st.markdown(
        "Download the underlying data to continue your own analysis."
    )
    st.download_button(
        label="Download data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="global_livability_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
