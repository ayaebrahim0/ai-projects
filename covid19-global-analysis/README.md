# 🦠 COVID-19 Global Dataset Analysis

Exploratory data analysis of global COVID-19 case, death, recovery, and testing data (Worldometer data, daily time series + per-country summary).

**Notebook:** [`COVID19_Global_Analysis.ipynb`](./COVID19_Global_Analysis.ipynb)

## Datasets

- `worldometer_coronavirus_daily_data.csv` — 184,787 rows: daily cumulative/new cases and deaths per country.
- `worldometer_coronavirus_summary_data.csv` — 226 rows: per-country totals (confirmed, deaths, recovered, active, tests, population).

## What's covered

- Preliminary exploration: shape, columns, duplicates, summary statistics for both datasets.
- Derived metrics: **mortality rate** (deaths / population) and **fatality rate** (deaths / confirmed cases) per country.
- Cases and deaths aggregated by country and by continent.
- Countries with the maximum / minimum cases, and the top 5 most-affected countries.
- Visualizations: line charts by country/continent, choropleth world maps (confirmed cases, deaths, recovered, active cases), pie charts per continent, top-10 bar charts across multiple metrics, and per-country time series (case/death curves with 7-day moving averages) for selected countries (USA, Nigeria).

## Running the notebook

```bash
pip install pandas plotly seaborn matplotlib
```

Place both CSV files in the same folder as the notebook (or update the paths in the second cell) and run all cells. Interactive Plotly charts render inline when viewed in Jupyter/Colab; on GitHub they display as the static preview captured when the notebook was last run.
