-- Each traffic_hourly row is one observed hour, including observed zero traffic.
-- Totals describe observed coverage; absent hours are not imputed as zero.
SELECT
    CAST(strftime('%Y', date_time) AS INTEGER) AS year,
    COUNT(*) AS observed_hours,
    SUM(traffic_volume) AS traffic_sum_observed,
    ROUND(AVG(traffic_volume), 2) AS avg_hourly_traffic
FROM traffic_hourly
WHERE date_time >= '2012-01-01 00:00:00'
  AND date_time <  '2018-01-01 00:00:00'
GROUP BY strftime('%Y', date_time)
ORDER BY year;
