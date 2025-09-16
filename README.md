# Global Livability Radar

An interactive Streamlit web application for exploring the livability of countries around the world. The app visualizes each
country's performance across multiple quality-of-life indicators using side-by-side radar charts, making it simple to compare
strengths and weaknesses at a glance.

## Features

- **Interactive radar charts:** Compare any set of countries with normalized spider charts, ensuring that higher scores always
  represent more livable conditions.
- **Metric selection:** Choose which indicators to include in the comparison and review detailed descriptions for each metric.
- **Detailed breakdowns:** Inspect both the raw indicator values and normalized scores for selected countries.
- **Livability rankings:** Rank all countries according to the metrics you selected and download the underlying dataset for
  further analysis.

## Data

The application uses a curated dataset that combines economic, environmental, health, infrastructure, and safety indicators for
countries across the globe. Every metric is normalized on a 0–100 scale within the app, with directionality adjusted so that
higher scores are always better.

## Getting started

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Open the provided local URL in your browser to begin exploring the comparisons.

## Project structure

```
├── app.py             # Streamlit application
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## License

This project is provided for demonstration purposes. Feel free to adapt the code to suit your own analyses or visualizations.
