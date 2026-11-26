# Smart City Traffic Intelligence

An AI, ML and Data Science capstone using hourly westbound I-94 traffic, weather and holiday observations near Minneapolis–St Paul.

The project connects historical analysis, a reproducible Python workflow and machine learning to compare departure times under specified weather conditions.

## Repository

The public project repository is [FSLiao375426/smart-city-traffic-capstone-003](https://github.com/FSLiao375426/smart-city-traffic-capstone-003).

## Current status

| Course deliverable | Where to start | Supporting files |
|---|---|---|
| Part 1: Data analytics | [Insights report](output/pdf/part1_data_analytics_insights_report.pdf) | [SQL, statistics and probability](part1_data_analytics/README.md) and [Power BI dashboard](part1_data_analytics/traffic_dashboard.pbix) |
| Part 2: Reproducible Python pipeline | [Methods and findings](output/pdf/part2_traffic_pipeline_report.pdf) | [Course folder in ZIP format](output/submission/capstone_part2.zip), including the CLI, figures, sample log and README |
| Part 3: Machine learning and AI | [Final capstone report](part3_machine_learning/final_capstone_report.md) and [bias and fairness report](part3_machine_learning/bias_and_fairness_report.md) | [Course folder in ZIP format](output/submission/capstone_part3.zip) and [extracted-package checks](output/submission/capstone_part3_acceptance.json) |

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

## Project folders

- `part1_data_analytics/`: SQL, statistics, Power BI and the insights report.
- `part2_python/`: Data-processing modules, the CLI application, figures, a sample log and usage instructions.
- `part3_machine_learning/`: Models, experiment tracking, recommendations, deployment, monitoring and responsible-AI analysis.
- A final report integrating the three parts.

Each part includes instructions and evidence alongside its implementation.

## Data considerations

- The input contains 48,204 rows and nine columns, covering October 2012 to September 2018.
- Multiple weather records can share the same timestamp. Traffic volume must not be counted repeatedly when constructing hourly observations.
- The dataset has substantial gaps in time coverage; yearly totals need coverage context.
- Saved input hashes describe the bytes used in each recorded run. Git can normalise CSV line endings, so hashes can differ across checkouts even when data values match. Rerunning the workflow records the input hash for that checkout; the supplied source file remains preserved.
- Implausible sensor values require documented validation and handling.
- The dataset covers a single corridor. Recommendations focus on departure timing.
- No real accident dataset was supplied. The classification task uses a documented proxy risk label; its results cannot establish accident probabilities.

## Dataset attribution

Hogue, J. (2019). *Metro Interstate Traffic Volume* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5X60B](https://doi.org/10.24432/C5X60B).

The [UCI dataset page](https://archive.ics.uci.edu/dataset/492/metro%2Binterstate%2Btraffic%2Bvolume) identifies the dataset licence as [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). This attribution refers to the dataset.

## Part 2 reproduction

Run python -m part2_python.pipeline --all from this repository root for raw-to-figures processing. The three user queries are python -m part2_python.cli at, summary and compare; arguments and evidence are documented in [Part 2 README](part2_python/README.md). The Part 2 report and portable submission package are available below. See the submission acceptance records for the scope and limitations of prior checks.

Part 2 report: [two-page methods and results](output/pdf/part2_traffic_pipeline_report.pdf). Portable course package: output/submission/capstone_part2.zip; build with python -m part2_python.build_submission. See Part 2 README for reproduction and verification details.

Part 1 statistical/probability reproduction and deliverable mapping: [analysis README](part1_data_analytics/README.md). [Part 1 insights report](output/pdf/part1_data_analytics_insights_report.pdf) consolidates the verified findings and operational implications.

## Part 3 machine learning and submission

The [final capstone report](part3_machine_learning/final_capstone_report.md) covers all seven tasks, including the frozen models' evaluation on 6,533 observed hours in 2018. The [bias and fairness report](part3_machine_learning/bias_and_fairness_report.md) explains the proxy label, subgroup errors and limits of the recommendations. For code and reviewer instructions, open the [Part 3 technical README](part3_machine_learning/README.md) or extract the [portable Part 3 submission](output/submission/capstone_part3.zip) and start with its top-level README.
