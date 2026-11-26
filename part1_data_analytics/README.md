# Historical Traffic Analysis

This section contains SQL analysis, descriptive statistics, probability analysis, the Power BI dashboard and the two-page insights report. Read the linked results and validation records for their scope. Later dated sections preserve development history; historical statements about pending publication do not describe the current repository status.

## Run

**Preserve existing evidence:** these commands regenerate processed data, database tables, results and logs. To reproduce without changing the submitted artifacts, work in a separate copy and compare outputs with the recorded evidence.

From the project root in the verified `capstone` Python environment:

```text
python -m part2_python.pipeline
python -m part1_data_analytics.sqlite_analysis
```

If the verified cleaning output already exists, only the second command is needed. Dependencies are the existing pandas/NumPy environment and Python's built-in sqlite3. No additional installation is needed.

## Files and data grain

- `traffic.sqlite`: SQLite database containing `traffic_raw` (the supplied CSV, all nine columns) and `traffic_hourly` (the cleaned hourly table). The latter has a unique timestamp index.
- `sql/01_yearly_traffic.sql`: First query, covering 2012–2017 as required. It groups by source-calendar year and returns observed hours, observed traffic totals and mean traffic per observed hour.
- `sql/02_yearly_changes.sql`: Observed year-to-year changes with hour/month coverage and full-calendar coverage percentages. Monthly counts do not imply complete observations within those months.
- `sql/03_holiday_temperature.sql`: Six calendar holiday dates, plus any differently dated source-label alternatives. Missing dates remain visible through a LEFT JOIN.
- `results/yearly_traffic_2012_2017.csv`: Query results.
- `results/yearly_changes_2012_2017.csv`: Annual changes and coverage context.
- `results/holiday_temperature_2015_2017.csv`: Holiday temperatures, traffic context, observation coverage and date basis.
- `results/load_validation.json`: Source hashes, row/hour counts, SQLite integrity and query scope.
- `sqlite_analysis.log`: Execution and validation evidence.

The raw table retains repeated weather observations and invalid readings for provenance. Annual traffic analysis uses the hourly table so that multiple weather descriptions do not repeatedly count the same hour's traffic. Zero traffic is retained; unobserved hours and months remain absent. Full-calendar coverage is available in `data/processed/coverage_by_year.csv`. The source describes its timestamps as local CST but does not clarify in its field description whether clock time switches with daylight saving. Distinct timestamp labels are therefore not proof of elapsed-hour coverage around clock changes; the current analysis does not infer a DST effect.

SUM measures observed traffic, COUNT measures observed hours in the hourly table, and AVG measures traffic per observed hour. Incomplete calendar coverage and differing seasonal/time-of-day coverage limit comparisons; neither totals nor averages alone establish a change in underlying demand.

The loader replaces only its two generated database tables and regenerates the result files. Source CSV files remain unchanged. SQL files can also be opened in DB Browser for SQLite against the generated database.

Result CSVs use `NA` for SQL NULL, including a missing previous-year baseline or an unobserved holiday's temperature. The source holiday label `None` is a literal string. For pandas, preserve this distinction with `keep_default_na=False, na_values=['NA']` when reading these result files.

## First verified run: 2026-11-24

Python's SQLite runtime is 3.53.4 in the existing environment. The loader completed with warnings treated as errors. The raw table has 48,204 rows and nine columns; the hourly table has 40,575 rows and 28 columns. An independent read-back checked every source cell (1,569,936 values), the hourly unique index and database integrity. All six annual results match an independent calculation directly from the original CSV's unique traffic hours. Source hashes remain unchanged.

For example, 2013 has 7,294 observed hours, traffic sum 24,139,878 and mean hourly traffic 3,309.55; 2014 has 4,501 hours, sum 14,718,915 and mean 3,270.14. Their differing coverage remains essential context. The extended queries below were subsequently implemented and independently checked.

## Annual findings and limits

Change percentage is `(current - previous) / previous * 100`. SQL LAG aligns the prior year, and NULLIF prevents division by a zero baseline. Changes in mean hourly traffic use the unrounded yearly means. The first available year has no previous-year change; it is reported as NA.

1. From 2013 to 2014, the observed traffic total fell 39.03%, while observed hours fell 38.29% and mean traffic per observed hour fell 1.19%. Full-year coverage fell from 83.26% to 51.38%, and the observed months fell from 12 to eight. Arithmetically, most of the recorded-total reduction accompanies fewer observed hours. These samples do not establish a comparable 39.03% drop in underlying demand.
2. From 2016 to 2017, the observed total rose 17.53%, observed hours rose 11.16%, and mean hourly traffic rose 5.73%. Full-year coverage improved from 89.23% to 99.46%. The recorded average also increased, but a demand-growth claim still requires comparable month/hour coverage and other relevant controls.

The 2012 series begins in October; its large apparent increase into 2013 compares a partial year with much broader coverage. Do not treat the percentage changes as estimates for complete, comparable years.

## Holiday findings and date definitions

The calendar-date series uses January 1 for New Year's Day and the Labor Day dates recorded in the dataset. Where a source label occurs on another date, it is retained separately as `dataset_label_date`; it is not merged into the calendar-date series.

| Calendar date | Holiday | Observed hours | Mean temperature (C) | Mean hourly traffic |
|---|---|---:|---:|---:|
| 2015-01-01 | New Year's Day | 0 | NA | NA |
| 2015-09-07 | Labor Day | 24 | 22.17 | 2,430.29 |
| 2016-01-01 | New Year's Day | 18 | -6.06 | 1,832.00 |
| 2016-09-05 | Labor Day | 24 | 21.86 | 2,151.54 |
| 2017-01-01 | New Year's Day | 24 | -3.06 | 2,127.63 |
| 2017-09-04 | Labor Day | 24 | 18.14 | 2,395.17 |

There are no observations for New Year's Day 2015, so no temperature or traffic average is invented. The 2016 date has 18 observed hours (missing hours 02, 11, 15, 17, 19 and 22); its recorded mean is not a complete 24-hour mean. The calendar-date mean for 2017 is about 3.00 C warmer than the observed 2016 mean, with this coverage qualification.

The raw `New Years Day` label for 2017 occurs on January 2. That separate date has 24 observations, mean temperature -1.30 C and mean hourly traffic 2,091.08. January 1 has no such source label. Both definitions remain explicit; the supplied data does not establish why the label date differs.

Labor Day mean temperatures decline from 22.17 C (2015) to 21.86 C (2016) and 18.14 C (2017), with approximately -0.32 C and -3.72 C changes calculated from unrounded means. All three dates have 24 observed hours. Mean traffic falls and then rises, so these few holiday summaries do not demonstrate a simple temperature-to-traffic causal relationship. No temperatures in the requested holiday rows were imputed.

## Extended verification

All three SQL files completed with warnings treated as errors: six annual rows, six change rows and seven holiday/date-basis rows. Independent calculations checked every change percentage, calendar coverage, holiday aggregate and missing-value case. The raw/hourly database read-back and integrity checks were repeated after the extension. Input and query hashes match the saved metadata.

## Reproducible statistics and probabilities (2026-11-24)

```powershell
python -W error -m part1_data_analytics.statistical_analysis
python -W error -m unittest discover -s part1_data_analytics/tests -v
```

Input: data/processed/traffic_hourly.csv (40,575 unique hours). The analysis leaves the source unchanged, checks event counts with independent SQLite queries and population standard deviation with Python statistics. Its named logger writes normal events and failures to statistical_analysis.log and the console. Three independent checks passed, including exact four-cell counts and rational odds ratio, standard-library descriptive statistics, correlation and standardization weights/regression values.

| Result in results/ | Purpose |
|---|---|
| descriptive_statistics.csv | Mean 3290.65047443, median 3427, population SD 1984.74845105, variance 3939226.41392553, range 7280. Sample SD/variance are separately labelled. |
| temperature_correlation.csv | Main: 40,565 directly observed temperature pairs, Pearson r=0.1392263864. Sensitivity including ten imputed hours: 40,575 pairs, r=0.1394926890. |
| probabilities.csv | Event numerator/denominator counts and six probabilities, including all required basic and conditional probabilities. |
| weather_contingency.csv | Two mutually exclusive Clear/Clouds comparison groups and their congested/noncongested counts. |
| weather_standardized_rates.csv | Common-weight rates adjusted for hour or hour x weekday/weekend. |
| weather_strata.csv | All component counts and pooled weights; NA weekend values identify hour-only strata. |
| hourly_profile_2017.csv | 24 mean traffic values and observation counts for the dashboard's required year. |
| weather_average_traffic.csv | Mean traffic and support count for each nonexclusive weather category. |
| congested_temperature_range.csv | Observed temperature minimum, 10th/90th percentiles and maximum among congested hours. |
| statistical_validation.json | Source SHA-256, definitions, overlap exclusions, independence comparison and crude odds ratio. |

### Interpretation and definitions

Congestion is strictly traffic_volume >5500; high temperature is strictly >292 K. Clear means weather_Clear=1, including hours with additional weather labels. Counts give P(congestion)=6107/40575=15.05%, P(clear)=13371/40575=32.95%, P(both)=1763/40575=4.35%, P(clear given congestion)=1763/6107=28.87% and P(high temperature given congestion)=1699/6107=27.82%. All ten imputed-temperature hours are noncongested, so excluding them does not change this last conditional probability.

The observed joint probability 0.04345040049 differs from the independence product 0.04959920915. This is an empirical equality comparison, not a formal hypothesis test or proof of a causal effect.

For the odds ratio only, compare Clear without Clouds (1762 congested, 11602 not) with Clouds without Clear (2586 congested, 12534 not), excluding seven overlapping hours and 12,084 hours with neither category. Other labels may coexist. OR=(1762/11602)/(2586/12534)=0.7360955848. It describes lower crude odds in the first group; odds and probability are different quantities.

Common weights come from the pooled 28,484 hours in these two groups. Every hour stratum and hour/weekend stratum has both groups. Hour-standardized congestion rates are 15.2799% (clear group) and 15.2304% (cloudy group); hour/weekend rates are 15.6451% and 14.9888%. Compare these descriptive rates, not the adjusted rate ratio with the crude odds ratio. Weather group composition and time of day limit a simple weather ranking.

Clouds has the highest nonexclusive weather-category mean (3616.95 vehicles/hour; 15,127 hours), Squall the lowest (2061.75; only four hours), difference 1555.20. The tiny Squall group does not support a reliable operational recommendation. Congested hours span -26.00 to 36.14 C; the middle 80% span -6.93 to 25.97 C. The maximum observed traffic (7280 at 2017-03-09 16:00) occurred at -2.40 C. These are descriptive ranges, not learned decision thresholds or causal effects.

## Power BI and report deliverables

- traffic_dashboard.pbix: saved report, previously interactively verified by the user. Current archive integrity and report layout show one page, four charts, three KPI cards, three slicers and one title. No live Power BI interactions were repeated in this round.
- PowerBI/: three existing PNGs documenting earlier tooltip/visual checks; these are not represented as a fresh screenshot of the entire final dashboard.
- [Two-page Data Analytics Insights Report](../output/pdf/part1_data_analytics_insights_report.pdf): methodology, statistical/probability results, annual coverage, holiday-date limits, dashboard findings and recommendations for the mobility team. Both final pages were rendered and visually checked. This repo-local report consolidates the prior discussion; it does not claim to be the inaccessible earlier cloud PDF's exact bytes.
- build_report.py: optional ReportLab authoring tool. It runs with the bundled PDF Python/ReportLab environment; ReportLab is not needed for the statistical analysis or dashboard.

The course's four Part 1 deliverables map to sql/ and traffic.sqlite, results/ plus statistical_analysis.py, traffic_dashboard.pbix, and the linked report. The dashboard's fixed Low/Medium/High bins are distinct from Part 2 quartile labels. Final source-to-report verification, user review and publication remain part of final acceptance.

## Suggested incremental commits

1. Add reproducible Part 1 statistical and probability analysis: statistical_analysis.py, statistical_analysis.log, tests/test_statistics.py and the ten new results files (nine CSVs plus statistical_validation.json).
2. Add Power BI traffic dashboard and supporting screenshots: traffic_dashboard.pbix and the three files under PowerBI/.
3. Add Part 1 insights report and delivery documentation: build_report.py, build_report.log, output/pdf/part1_data_analytics_insights_report.pdf and this README. The root README now links both reports; commit it with the documentation batch after reviewing its shared Part 2 changes.

The Part 2 report/package changes from the preceding round remain a separate pending commit. Do not stage all of output/ when preparing only the Part 1 report commit. The user handles stage, commit and push; no new Git write was performed by the assistant.

Current file/hash/layout checks are recorded in results/deliverable_validation.json. Include this additional validation file in the analysis evidence commit; it records the latest PDF hash and prior-interaction scope without claiming a fresh Power BI UI test.
