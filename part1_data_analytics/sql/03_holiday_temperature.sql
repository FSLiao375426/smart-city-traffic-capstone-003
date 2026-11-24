-- Calendar New Year's Day is January 1. Labor Day dates below match the dataset.
-- Also retain a separately labelled alternative if a source holiday date differs.
WITH calendar_holidays(holiday_date, holiday_name) AS (
    VALUES
        ('2015-01-01', 'New Years Day'),
        ('2015-09-07', 'Labor Day'),
        ('2016-01-01', 'New Years Day'),
        ('2016-09-05', 'Labor Day'),
        ('2017-01-01', 'New Years Day'),
        ('2017-09-04', 'Labor Day')
), source_holidays AS (
    SELECT DISTINCT substr(date_time, 1, 10) AS holiday_date, holiday AS holiday_name
    FROM traffic_raw
    WHERE holiday IN ('New Years Day', 'Labor Day')
      AND date_time >= '2015-01-01 00:00:00'
      AND date_time <  '2018-01-01 00:00:00'
), requested_dates AS (
    SELECT holiday_date, holiday_name, 'calendar_date' AS date_basis
    FROM calendar_holidays
    UNION ALL
    SELECT s.holiday_date, s.holiday_name, 'dataset_label_date' AS date_basis
    FROM source_holidays AS s
    WHERE NOT EXISTS (
        SELECT 1 FROM calendar_holidays AS c
        WHERE c.holiday_date = s.holiday_date AND c.holiday_name = s.holiday_name
    )
)
SELECT
    CAST(substr(d.holiday_date, 1, 4) AS INTEGER) AS year,
    d.holiday_name,
    d.date_basis,
    d.holiday_date,
    COALESCE(GROUP_CONCAT(DISTINCT NULLIF(h.holiday, 'None')), 'None') AS source_holiday_labels,
    -- COUNT(date_time) excludes the unmatched placeholder kept by LEFT JOIN.
    COUNT(h.date_time) AS observed_hours,
    ROUND(100.0 * COUNT(h.date_time) / 24, 2) AS daily_coverage_pct,
    COALESCE(SUM(h.temp_imputed), 0) AS temp_imputed_hours,
    ROUND(AVG(h.temp), 2) AS avg_temp_k,
    ROUND(AVG(h.temp - 273.15), 2) AS avg_temp_c,
    ROUND(MIN(h.temp - 273.15), 2) AS min_temp_c,
    ROUND(MAX(h.temp - 273.15), 2) AS max_temp_c,
    ROUND(AVG(h.traffic_volume), 2) AS avg_hourly_traffic,
    CASE
        WHEN COUNT(h.date_time) = 0 THEN 'NO_OBSERVATIONS'
        WHEN COUNT(h.date_time) < 24 THEN 'PARTIAL_DAY'
        ELSE 'COMPLETE_DAY'
    END AS coverage_status
FROM requested_dates AS d
LEFT JOIN traffic_hourly AS h
    ON substr(h.date_time, 1, 10) = d.holiday_date
GROUP BY d.holiday_date, d.holiday_name, d.date_basis
ORDER BY year,
         CASE WHEN d.holiday_name = 'New Years Day' THEN 0 ELSE 1 END,
         d.holiday_date;
