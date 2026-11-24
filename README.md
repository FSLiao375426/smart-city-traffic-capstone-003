# Smart City Traffic Intelligence

An AI, ML and Data Science capstone using hourly westbound I-94 traffic, weather and holiday observations near Minneapolis–St Paul.

The project will connect historical analysis, a reproducible Python workflow and machine learning to support evidence-based travel-timing recommendations.

## Repository

The public project repository is [FSLiao375426/smart-city-traffic-capstone-003](https://github.com/FSLiao375426/smart-city-traffic-capstone-003).

## Current status

Initial project setup. The supplied dataset is preserved unchanged. Analysis outputs, executable pipelines, trained models and deployment components have not yet been implemented.

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
- Implausible sensor values require documented validation and handling.
- The dataset covers a single corridor. Recommendations will focus on departure timing.
- No real accident dataset is supplied. The classification task will use a documented proxy risk label; its results must not be interpreted as verified accident probabilities.

## Dataset attribution

Hogue, J. (2019). *Metro Interstate Traffic Volume* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5X60B](https://doi.org/10.24432/C5X60B).

The [UCI dataset page](https://archive.ics.uci.edu/dataset/492/metro%2Binterstate%2Btraffic%2Bvolume) identifies the dataset licence as [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). This attribution refers to the dataset.
