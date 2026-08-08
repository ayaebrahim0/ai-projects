# 🏏 IPL 2023 Auction — Exploratory Data Analysis

Exploratory data analysis of the IPL (Indian Premier League) 2023 player auction: cleaning the raw auction data and answering questions about team composition, pricing, and player roles.

**Notebook:** [`IPL_2023_Auction_EDA.ipynb`](./IPL_2023_Auction_EDA.ipynb)

## Dataset

`IPL_Squad_2023_Auction_Dataset.csv` — 568 players with base price, role (batsman / bowler / all-rounder / wicketkeeper), and the team that acquired them (or "Unsold").

## What's covered

- Data cleaning: dropping columns that were empty for unsold players (final sale price, 2022 squad), fixing the `Base Price` dtype, checking for duplicates/nulls.
- Univariate analysis of numeric and categorical columns (distributions, boxplots, countplots).
- Answering specific questions with visualizations:
  - How many players did each team acquire?
  - What does the spread of base prices look like?
  - Do batsmen, bowlers, all-rounders, or wicketkeepers have higher base prices on average?
  - How are player roles distributed across teams?
  - Who are the players with the highest base prices?

## Running the notebook

```bash
pip install pandas numpy seaborn matplotlib
```

Place `IPL_Squad_2023_Auction_Dataset.csv` in the same folder as the notebook and run all cells.
