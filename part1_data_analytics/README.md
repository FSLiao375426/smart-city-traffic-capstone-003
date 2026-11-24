# Historical Traffic Analysis

This SQL draft implements raw/hourly loading and verification, annual totals and changes for 2012–2017, and holiday-temperature comparisons for 2015–2017. Interpretations and qualifications appear below. Formal descriptive statistics/probabilities, Power BI and the integrated Part 1 report remain pending.

## Run

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

The raw table retains repeated weather observations and invalid readings for provenance. Annual traffic analysis uses the hourly table so that multiple weather descriptions do not repeatedly count the same hour's traffic. Zero traffic is retained; unobserved hours and months remain absent. Full-calendar coverage is available in `data/processed/coverage_by_year.csv`.

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
