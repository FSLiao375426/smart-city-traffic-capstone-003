-- Changes describe observed records; differing coverage limits demand claims.
WITH yearly AS (
    SELECT
        CAST(strftime('%Y', date_time) AS INTEGER) AS year,
        COUNT(DISTINCT substr(date_time, 1, 7)) AS months_observed,
        COUNT(*) AS observed_hours,
        SUM(traffic_volume) AS traffic_sum_observed,
        AVG(traffic_volume) AS mean_hourly_traffic
    FROM traffic_hourly
    WHERE date_time >= '2012-01-01 00:00:00'
      AND date_time <  '2018-01-01 00:00:00'
    GROUP BY strftime('%Y', date_time)
), aligned AS (
    SELECT *,
        LAG(year) OVER (ORDER BY year) AS previous_year,
        LAG(observed_hours) OVER (ORDER BY year) AS previous_hours,
        LAG(traffic_sum_observed) OVER (ORDER BY year) AS previous_total,
        LAG(mean_hourly_traffic) OVER (ORDER BY year) AS previous_mean
    FROM yearly
)
SELECT
    year,
    months_observed,
    observed_hours,
    ROUND(100.0 * observed_hours / (
        24 * (julianday(printf('%04d-01-01', year + 1))
            - julianday(printf('%04d-01-01', year)))
    ), 2) AS calendar_coverage_pct,
    traffic_sum_observed,
    ROUND(mean_hourly_traffic, 2) AS avg_hourly_traffic,
    previous_year,
    traffic_sum_observed - previous_total AS traffic_change,
    ROUND(100.0 * (traffic_sum_observed - previous_total)
          / NULLIF(previous_total, 0), 2) AS traffic_change_pct,
    ROUND(100.0 * (observed_hours - previous_hours)
          / NULLIF(previous_hours, 0), 2) AS hours_change_pct,
    ROUND(100.0 * (mean_hourly_traffic - previous_mean)
          / NULLIF(previous_mean, 0), 2) AS avg_hourly_change_pct
FROM aligned
ORDER BY year;
