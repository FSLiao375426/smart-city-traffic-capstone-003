# Traffic Data Pipeline

This first implementation validates the supplied observations, builds one traffic row per observed hour and preserves multiple weather labels in a separate table. A separate feature step adds three calendar fields and two cyclic-hour coordinates. Reference-period scaling, quartile labels and three saved Matplotlib figures are implemented. The analytics CLI and end-to-end entry point are implemented; the short report and final submission packaging remain pending.

## Run

From the project root, use the existing `capstone` Python environment:

```text
python -m part2_python.pipeline
python -m part2_python.features
```

Verified with Python 3.11.17, NumPy 2.4.6 and pandas 3.1.1. For a separate environment, install the runtime dependencies from `part2_python/requirements.txt` before running the module. No package installation is required in the verified existing environment.

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

## Reference-period scaling: 2026-11-24

Run from the repository root with the capstone Python environment:

```powershell
python -m part2_python.features
python -m part2_python.features --scale
python -m unittest discover -s part2_python/tests -v
```

The default command preserves the 40,575-row / 33-column time-feature output. `--scale` builds the same time features from `traffic_hourly.csv`, appends `temp_z` and `clouds_all_z`, and saves `data/processed/traffic_scaled_features.csv` (40,575 rows / 35 columns). Every existing column is preserved. Both commands log to `part2_python/features.log`; each run replaces that log.

Scaling uses `(value - reference_mean) / reference_std`, with population standard deviation (`ddof=0`). The cutoff comes from `imputation_values.json`: 2016-12-31 23:59:59, inclusive (25,329 reference hours). The remaining 15,246 hours use the same parameters. `scaling_values.json` records the fitted values: temp mean 280.9408813020648 K / std 13.154219264568958 K; clouds_all mean 45.43313987918986% / std 38.4797818724198 percentage points.

Saved-CSV acceptance checks verify unchanged original columns, reference means within 1e-8 of zero, population standard deviations within 1e-8 of one, and the transformation of every row. Holdout perturbation and empty/constant/non-finite input checks also pass. Nine tests passed in total. This remains a demonstration reference cutoff; refit preprocessing on the final model training period once that split is decided. Quartile traffic-volume labels are implemented below.

## Reference-period traffic quartiles: 2026-11-24

```powershell
python -m part2_python.features --quartiles
```

`--quartiles` includes scaling automatically; `--scale --quartiles` has the same result. It saves `traffic_model_features.csv` (40,575 rows / 36 columns), preserving all 35 scaled-table columns and adding `traffic_volume_quartile`. Default and scale-only output filenames and shapes remain unchanged. `quartile_values.json` records the reference cutoff, interpolation method, thresholds and boundary convention.

Only the 25,329 hours through the inclusive cleaning `fit_end` (2016-12-31 23:59:59) fit the thresholds, using linear quantile interpolation. Q1=1204, Q2=3339 and Q3=4932 vehicles per observed hour. Apply these same thresholds to all hours:

| Level | Traffic volume | Reference hours | Held-out hours |
|---|---|---:|---:|
| 1 | x <= 1204 | 6337 | 3625 |
| 2 | 1204 < x <= 3339 | 6329 | 3567 |
| 3 | 3339 < x <= 4932 | 6332 | 4062 |
| 4 | x > 4932 | 6331 | 3992 |

Ties stay in the lower interval, so groups need not contain exactly 25% of reference hours. Held-out proportions may also differ because the distribution changes. Empty reference periods, invalid/non-finite/negative traffic values and coincident quartile thresholds are rejected rather than silently creating fewer than four levels.

These are traffic-volume proxy labels, not observed accident risks, and differ from the probability task's >5500 event and the Power BI fixed bins. The label and its source traffic_volume must be excluded from classification inputs; weather and timing availability still require a prediction-time design. The final training split remains undecided: refit imputation, scaling and quartile thresholds on its training period before final modelling.

Eleven tests passed, covering saved artifacts, every row's interval assignment, exact boundaries, original-column preservation, holdout isolation and existing cleaning checks. Default, scale-only, quartile-only and combined flag CLI runs succeeded.
## Descriptive Matplotlib figures: 2026-11-24

Run `python -W error -m part2_python.visualize` from the repository root. Requires the pinned Matplotlib dependency in requirements.txt. The noninteractive Agg backend saves figures without opening windows; failures return exit code 1 and are logged to visualize.log.

Outputs in `part2_python/figures/`:

| Figure | Supporting data | Interpretation |
|---|---|---|
| 01_hourly_profile.png | hourly_profile.csv (48 groups with counts) | Weekday peaks at 16:00 (6240.88 vehicles/hour), with another peak at 07:00 (6061.16); weekend peak at 13:00 (4407.61). These are descriptive patterns, not confirmed commuting causes. |
| 02_traffic_distribution.png | traffic_histogram.csv | 250-vehicle bins show low and high traffic regimes; mean 3290.65 and median 3427. The full distribution contains more information than either statistic alone. |
| 03_temperature_traffic.png | temperature_traffic.csv (all 40,575 points) | Weak positive overall linear association, Pearson r=0.13949. This is not a causal effect; time of day and seasonal composition may confound it. |

All three use one row per observed hour over 2012-10-02 to 2018-09-30, preserve both zero-traffic hours and leave missing hours absent. Mean hourly profiles avoid comparing unequal observation totals, but do not remove year/season composition differences. Scatter points use transparency without sampling. Histogram intervals are left-inclusive/right-exclusive except the final right edge. The source table is unchanged, verified by SHA-256; figure_metadata.json records scope, source hash, version and correlation. Full-period descriptive plots are not held-out model performance evidence.

The command succeeded with warnings treated as errors. Saved-data checks independently verified all 48 group means/counts, every histogram bin and all scatter values against the source. All three 1600x960 PNGs were visually inspected for readable labels and unclipped layout. These are three saved Matplotlib deliverables; the broader Part 2 integration and short report remain pending.
## Submission requirements audit (2026-11-24)

This checklist explicitly tracks the logging requirements and submission checklist quoted from the course instructions. Documenting a requirement does not mean the final submission has passed it.

| Logging requirement | Current evidence / remaining verification |
|---|---|
| Module loggers via logging.getLogger(__name__); no bare root logging in library operations | pipeline.py, features.py and visualize.py each declare a named module logger; current source search found no root-level logging.info/warning/error/debug calls. Root configuration via basicConfig is setup, not an operational root-log message. |
| Configure logging only at the main/entry point; console and file output | Each main() calls the shared configure_logging helper; imports do not configure logging. Current setup attaches one StreamHandler and one FileHandler to the root logger. This is one configuration with TWO destination handlers, not literally one handler object. The original DOCX says 'Configure a handler only in your main/entry-point script' and cites a multiple-handler cookbook. We interpret this as an entry-point-only configuration requirement, with one handler per output destination. pipeline.py --all now provides the integrated entry point and unified event trail. |
| Timestamp, level, module and message | Formatter: %(asctime)s | %(levelname)s | %(name)s | %(message)s; UTC timestamp. Direct -m execution reports the module name as __main__, while imported library execution uses its qualified module name. |
| Appropriate DEBUG / INFO / WARNING / ERROR | Cleaning has DEBUG fitted parameters, INFO milestones, WARNING recoverable drops/imputation/coverage, ERROR failed execution with traceback and return code 1. Features/visualization use INFO milestones and ERROR failures. Final CLI and integrated run still require review. |
| No internal progress print(); CLI user-facing result is the stated exception | Current Part 2 source contains no print() calls. Continue using logger for status/progress. Future CLI output must follow the course's stated distinction; re-audit all final source files before submission. |

| Submission item | Status and evidence |
|---|---|
| End-to-end run from raw CSV without errors | PARTIAL: cleaning and downstream stages have individual successful run evidence. A fresh complete raw-to-final run including the future analytics CLI has NOT yet been accepted. |
| Generated, clear and ordered event trail | PARTIAL: pipeline.log, features.log and visualize.log exist with formatted event records. Each command overwrites its own log; these are not yet a unified complete-pipeline trace. |
| No print() | Current Part 2 source audit passes; final repository-wide submission audit remains pending. Ad hoc verification output in chat is not submitted program source. |
| At least three visualizations saved to disk | DONE: figures/01_hourly_profile.png, 02_traffic_distribution.png and 03_temperature_traffic.png, plus supporting CSVs and metadata; inspected 2026-11-24. |
| CLI responds to at least three distinct commands | PENDING: feature flags and three SQL files are not counted as the three analytics commands. Implement and verify three user-facing queries with input validation and useful results. |
| Multiple incremental GitHub commits | Local history contains 5854a56, 3b5f3e3, c5cb5b6 and bfa2339. Current new work is uncommitted; user handles stage/commit/push. Current remote publication was not reverified during this audit. |
| README and report | PARTIAL: repository and Part 2 README exist; Part 2 1–2 page methods/results report remains pending, along with final README reconciliation. |

Final acceptance must include command lines, return codes, output checks and a clear complete event trail. Do not mark the whole Part 2 complete on the strength of individual module runs or the three saved figures alone.
## End-to-end entry point and three query commands (2026-11-24)

```powershell
python -W error -m part2_python.pipeline --all
python -m part2_python.cli at --time "2017-01-01 00:00:00"
python -m part2_python.cli summary --start 2017-01-01 --end 2017-01-31
python -m part2_python.cli compare --start 2017-01-01 --end 2017-01-31
```

`pipeline.py --all` loads the raw CSV and regenerates cleaning outputs, all three feature tables and three figures, using one logging configuration throughout. It writes a complete event trail to pipeline.log. Standalone modules retain their own logs. `--all` requires the default processed-output directory; a different --fit-end flows through cleaning metadata to both fitted transformations. DEBUG mode is available via `--log-level DEBUG`; quartile thresholds appear only in DEBUG. Matplotlib's font diagnostics are suppressed to keep application DEBUG logs readable. Figure-save INFO messages include full file paths.

The CLI reads traffic_model_features.csv. `at` returns one observed hour's traffic, temperature, cloud cover and quartile level; use exactly YYYY-MM-DD HH:00:00. `summary` returns counts, mean/min/max and fixed quartile frequencies. `compare` returns weekday/weekend means and observation counts. Range dates must be YYYY-MM-DD; both calendar endpoints are inclusive. Timestamps retain the source's clock convention; no timezone conversion is applied. Missing hours and absent comparison groups produce null means, never fabricated zero traffic. An empty result is a valid query with status no_data and exit code 0; malformed dates, reversed ranges, unknown commands and unreadable inputs return 2 with a clear ERROR and no traceback. CLI answers are JSON on stdout; operational messages are on stderr and in cli.log (overwritten each run).

The original course DOCX was checked directly: its prohibition applies to print() for internal progress, with an explicit exception for end-user CLI answers. This implementation uses sys.stdout.write for answers and named loggers for internal status; it has no print() calls. Module main functions configure logging only when standalone; the integrated entry point disables child reconfiguration. Named library loggers propagate to the entry point's console and file handlers.

### Updated acceptance status

- Full raw-to-figures run passed at INFO with warnings treated as errors; the complete sample is pipeline.log. A DEBUG run also confirmed quartile diagnostics.
- Sixteen unit tests passed, including inclusive date boundaries, true zero versus no observations, single-group comparison, CLI JSON-only stdout and clear invalid-input errors.
- Three CLI commands and empty/invalid scenarios are recorded in validation/cli_acceptance.json with arguments, return codes, responses, logs and processed-source hash.
- Requirements still pending: the 1–2 page Part 2 report, final packaging/name alignment (the course calls its deliverable capstone_part2 and examples feature_engineering.py / visualizations.py; this repository currently uses part2_python, features.py / visualize.py), final GitHub publication and final cross-part acceptance. Stage/commit/push remain the user's responsibility. The earlier audit table is the pre-integration snapshot; this section supersedes its end-to-end and CLI pending statuses.
## Incremental commit plan (2026-11-24)

Apply these commits in order using explicit file selection. The user performs staging, commits and pushes; this plan does not record commits as completed. Each batch includes its implementation and generated evidence. Do not use a blanket stage-all operation for these batches.

| Order | Suggested commit subject | Files to select |
|---|---|---|
| 1 | Add reference-fitted scaling and traffic quartile labels | part2_python/features.py; part2_python/features.log; part2_python/tests/test_features.py; data/processed/scaling_values.json; data/processed/quartile_values.json; data/processed/traffic_scaled_features.csv; data/processed/traffic_model_features.csv |
| 2 | Add three reproducible traffic visualizations | part2_python/visualize.py; part2_python/visualize.log; part2_python/requirements.txt; all eight current files in part2_python/figures/ |
| 3 | Add three traffic query commands with validation | part2_python/cli.py; part2_python/cli.log; part2_python/tests/test_cli.py; part2_python/validation/cli_acceptance.json |
| 4 | Integrate end-to-end pipeline and document acceptance | part2_python/pipeline.py; part2_python/pipeline.log; part2_python/README.md; README.md |

Dependency order: labels before CLI; figures before integrated --all. Each earlier feature/figure/CLI module works standalone with the pre-integration pipeline logging helper. The final commit documents the combined state and adds the integrated entry point. Review git diff --cached --stat before each commit. Latest complete-state verification: 16 tests, seven CLI acceptance cases and successful raw-to-figures runs. This is verification of the full working tree, not a claim that each proposed historical commit was separately checked out and tested.

Leave the existing untracked part1_data_analytics/PowerBI/ and traffic_dashboard.pbix for a separate Part 1 deliverable commit after its artifacts are assembled and reviewed. The pending Part 2 short report and final packaging should be another descriptive commit when actually finished. Personal learning notes, supplied course files, tmp verification scripts and unrelated files are not part of these four batches. No new commits or pushes have been made by the assistant.
