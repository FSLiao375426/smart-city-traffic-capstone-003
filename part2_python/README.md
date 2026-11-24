# Traffic Data Pipeline

This first implementation validates the supplied observations, builds one traffic row per observed hour and preserves multiple weather labels in a separate table. A separate feature step adds three calendar fields and two cyclic-hour coordinates. Further features, scaling, plots and the analytics CLI remain pending.

## Run

From the project root, use the existing `capstone` Python environment:

```text
python -m part2_python.pipeline
python -m part2_python.features
```

Verified with Python 3.11.17, NumPy 2.4.6 and pandas 3.1.1. For a separate environment, install the two runtime dependencies from `part2_python/requirements.txt` before running the module. No package installation is required in the verified existing environment.

Defaults: the original CSV in the project root, outputs in `data/processed/`, a log in `part2_python/pipeline.log`, and imputation parameters fitted only on data through 2016-12-31. The explicit reference period leaves 2017/2018 available for later chronological validation/testing. No model has been trained yet.

The second command reads `data/processed/traffic_hourly.csv`, adds `hour` (0–23), `day_of_week` (Monday=0 to Sunday=6), `is_weekend` (Saturday/Sunday=1), `hour_sin` and `hour_cos`, and writes `traffic_time_features.csv`. The two cyclic fields are sin/cos of `2*pi*hour/24`. Its log is `part2_python/features.log`. It does not rerun cleaning or replace the cleaned input. These deterministic time features require no fitted parameters and use no traffic targets.

```text
python -m part2_python.pipeline --log-level DEBUG
python -m unittest discover -s part2_python/tests -v
```

Optional arguments: `--input`, `--output-dir`, `--log-file`, `--fit-end`, `--log-level`. Keep the imputation reference cutoff aligned with the eventual modelling split. Defaults resolve relative to the code location, so the same layout works in the project work folder and the separate Git repository.

## Outputs

| File | Contents |
|---|---|
| `traffic_hourly.csv` | One row per observed hour, numeric weather summaries, preserved traffic, holiday flags, multiple weather labels and weather presence indicators. |
| `traffic_time_features.csv` | Current feature-step output: all 28 cleaned columns plus five time features, with 40,575 rows and 33 columns. |
| `hourly_weather.csv` | One row per hour/weather category, retaining detailed descriptions. A traffic hour can occur under multiple categories. |
| `coverage_by_year.csv` | Observed hours, full-calendar-year coverage, observed traffic sums and mean traffic per observed hour. |
| `time_gaps.csv` | Unobserved hourly intervals; these hours are not filled or treated as zero traffic. |
| `quality_report.json` | Input hash, row counts, data issues, imputation counts and coverage diagnostics. |
| `imputation_values.json` | Reference cutoff and fitted monthly/global fallback medians. |

## Cleaning decisions

- Check all expected columns before processing. Preserve the literal holiday value `None`.
- Standardise category text, parse hourly timestamps and reject invalid traffic targets or conflicting same-hour traffic values.
- Remove only exact duplicate observations. Preserve different weather labels.
- Mark temperature at/below 0 K, negative precipitation, rainfall above the assignment's 9,000 mm guard, invalid cloud percentages and non-finite readings as missing.
- Within an hour, deduplicate numeric weather vectors before taking medians. This avoids giving repeated weather descriptions extra numeric weight. Retain a flag when numerical readings differ.
- Keep the common hourly traffic value once. Preserve zero traffic; possible closures or sensor issues are hypotheses, not confirmed events in this dataset.
- Extend a known holiday label only within its own calendar date. Do not invent a holiday calendar for absent observations.
- Fit monthly weather medians on the stated reference period; use explicit month-by-month loops to fill invalid weather readings. Preserve imputation flags. No traffic target is imputed.
- Do not interpolate across missing hours or use later held-out years to fit preprocessing values.
- Preserve all weather categories. In particular, Clear and Clouds can coexist in a few hours; a later Clear-versus-Clouds odds ratio must explicitly define disjoint comparison groups rather than silently double-counting those hours.

## Logging and failures

Every module uses `logging.getLogger(__name__)`. Only the CLI entry point configures handlers, sending logs to both console and a file. Each line includes UTC timestamp, level, module name and message.

- INFO: schema/load checks, hourly construction and each saved output.
- WARNING: changed/dropped/imputed records, extended holiday labels and missing coverage.
- DEBUG: fitted imputation values; hidden during ordinary INFO runs.
- ERROR: a failed pipeline, including exception details, followed by exit code 1.

The pipeline contains no internal progress `print()` calls. It preserves the input and checks that its SHA-256 is unchanged after processing. A normal run replaces generated outputs and the sample pipeline log; it does not modify the raw input.

## First verified run: 2026-11-24

- 48,204 raw rows; 17 exact duplicate rows removed; 40,575 distinct traffic hours.
- All hourly traffic values agree with the original, including the two zero-traffic hours.
- 47,893 hour/weather-category pairs retain all distinct normalised descriptions.
- Ten invalid temperatures filled: four January hours with 264.25 K and six February hours with 264.15 K. One invalid July rainfall value filled with 0 mm. Every filled hourly value retains its imputation flag; these are estimates, not recovered measurements.
- Nine full months absent; 11,976 unobserved hours inside the supplied time range; longest missing interval 7,386 hours. These missing hours remain absent.
- Seven unit tests passed, plus independent real-data checks for target/weather preservation, coverage, raw-input integrity and logging/failure behaviour.

The reference cutoff and monthly-median method are a documented first demonstration. Modelling choices, final time splits and further feature engineering remain to be discussed. Generated quality flags and the traffic target are not automatically model inputs.

## First calendar-feature step: 2026-11-24

The feature module completed successfully with warnings treated as errors. An independent check using Python's standard-library datetime verified the three values for every one of the 40,575 rows. Every original column matched the cleaned input after saving and reloading; raw-input integrity was also checked. There are 28,979 observed weekday hours and 11,596 observed weekend hours. Both zero-traffic hours remain present. For example, Saturday 2017-01-07 08:00:00 appears in the real output as 8, 5, 1.

## Cyclic-hour extension: 2026-11-24

The current output extends the earlier 31-column table to 33 columns without changing any prior value or row. All 40,575 sine/cosine pairs were checked against standard-library math and the unit-circle identity within CSV precision. The 24 integer hours map to 24 distinct coordinate pairs. The coordinate distance from 23 to 0 equals the distance from 0 to 1 (about 0.2611), whereas 0 to 12 has distance 2. These are dimensionless coordinate distances, not elapsed hours. The fixed denominator is 24 hours per cycle, rather than the largest observed hour value. Raw and cleaned input hashes are unchanged.
