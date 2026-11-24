# Smart City Traffic Intelligence

An AI, ML and Data Science capstone using hourly westbound I-94 traffic, weather and holiday observations near Minneapolis–St Paul.

The project will connect historical analysis, a reproducible Python workflow and machine learning to support evidence-based travel-timing recommendations.

## Repository

The public project repository is [FSLiao375426/smart-city-traffic-capstone-003](https://github.com/FSLiao375426/smart-city-traffic-capstone-003).

## Current status

The first data-cleaning pipeline, five time features and SQL analysis draft have been implemented and verified locally. Cleaning preserves the supplied dataset, builds one traffic row per observed hour, retains multiple weather conditions, records data-quality decisions and reports coverage gaps. The separate feature step adds calendar and cyclic-hour fields. SQLite retains raw/hourly tables and provides annual totals, coverage-aware changes and holiday-temperature comparisons. Further features, descriptive statistics/probabilities, Power BI, models and deployment components are still pending.

Run cleaning from the project root with `python -m part2_python.pipeline`, then calendar features with `python -m part2_python.features`. See [pipeline instructions](part2_python/README.md) for dependencies, outputs, cleaning rules and verification.

After cleaning, run SQL analysis with `python -m part1_data_analytics.sqlite_analysis`. See [Part 1 instructions](part1_data_analytics/README.md) for the database, queries, results and interpretations.

## Project scope

1. **Data analytics:** SQLite queries, descriptive statistics, probability analysis, a Power BI dashboard and an insights report.
2. **Python:** Data validation and cleaning, feature engineering, Matplotlib figures, a command-line analytics application, logging and reproducibility documentation.
3. **Machine learning and AI:** Classification and regression comparisons, clustering, association rules, a neural network with explainability, MLflow experiment tracking, travel-timing recommendations and a deployment/monitoring simulation.

## Files currently included

| File | Purpose |
|---|---|
| `Metro_Interstate_Traffic_Volume.csv` | Unmodified input dataset supplied for the capstone. |
| `README.md` | Project overview and documentation entry point. |
| `.gitignore` | Excludes local environment files, temporary files, supplied course documents and personal learning notes from version tracking. |
| `part1_data_analytics/sqlite_analysis.py` | Imports and verifies raw/hourly tables, then runs three historical-analysis SQL files. |
| `part1_data_analytics/sql/` | Annual summaries, annual changes and holiday temperatures with explicit coverage and date definitions. |
| `part1_data_analytics/traffic.sqlite` | Generated database with raw and hourly tables. |
| `part1_data_analytics/results/` | Three query-result CSVs and source/query validation metadata. |
| `part2_python/pipeline.py` | Validates, cleans and constructs hourly traffic/weather tables; logs decisions and failures. |
| `part2_python/features.py` | Adds three calendar features and two cyclic-hour coordinates to a separate output, preserving all cleaned observations. |
| `part2_python/tests/test_pipeline.py` | Tests preservation, conflicts, holiday scope, imputation reference data and raw-input protection. |
| `part2_python/requirements.txt` | Verified runtime dependencies for the first pipeline. |
| `data/processed/` | Generated hourly data, weather detail, coverage, gaps and processing metadata. |
| `part2_python/pipeline.log` | Sample INFO/WARNING log from the successful real-data run. |
| `part2_python/features.log` | Sample log for the separate calendar-feature step. |

## Planned organisation

- `part1_data_analytics/`: SQL, statistics, Power BI and the insights report.
- `part2_python/`: Data-processing modules, the CLI application, figures, a sample log and usage instructions.
- `part3_machine_learning/`: Models, experiment tracking, recommendations, deployment, monitoring and responsible-AI analysis.
- A final report integrating the three parts.

These components will be added incrementally. Reproduction commands and dependency specifications will be documented alongside the implemented components.

## Data considerations

- The input contains 48,204 rows and nine columns, covering October 2012 to September 2018.
- Multiple weather records can share the same timestamp. Traffic volume must not be counted repeatedly when constructing hourly observations.
- The dataset has substantial gaps in time coverage; yearly totals need coverage context.
- Saved input hashes describe the bytes used in each recorded run. Git can normalise CSV line endings, so hashes can differ across checkouts even when data values match. Rerunning the workflow records the input hash for that checkout; the supplied source file remains preserved.
- Implausible sensor values require documented validation and handling.
- The dataset covers a single corridor. Recommendations will focus on departure timing.
- No real accident dataset is supplied. The classification task will use a documented proxy risk label; its results must not be interpreted as verified accident probabilities.

## Dataset attribution

Hogue, J. (2019). *Metro Interstate Traffic Volume* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5X60B](https://doi.org/10.24432/C5X60B).

The [UCI dataset page](https://archive.ics.uci.edu/dataset/492/metro%2Binterstate%2Btraffic%2Bvolume) identifies the dataset licence as [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). This attribution refers to the dataset.

## Part 2 reproduction

Run python -m part2_python.pipeline --all from this repository root for raw-to-figures processing. The three user queries are python -m part2_python.cli at, summary and compare; arguments and evidence are documented in [Part 2 README](part2_python/README.md). The Part 2 report and final submission packaging remain pending.
