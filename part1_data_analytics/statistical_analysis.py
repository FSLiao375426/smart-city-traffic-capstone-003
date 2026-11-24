"""Reproduce descriptive, probability and weather-stratified analyses."""
import hashlib
import json
import logging
import sqlite3
import statistics
import sys

import numpy as np
import pandas as pd

from part2_python.pipeline import PROJECT_ROOT, configure_logging

logger = logging.getLogger(__name__)


def analyse(frame):
    times = pd.to_datetime(frame.date_time, errors="raise")
    if times.isna().any() or times.duplicated().any():
        raise ValueError("Expected unique observed hourly timestamps")
    volume = frame.traffic_volume
    if not np.isfinite(frame[["traffic_volume", "temp"]].to_numpy()).all():
        raise ValueError("Non-finite traffic or temperature")
    n = len(frame)
    congested = volume.gt(5500)
    clear, clouds = frame.weather_Clear.eq(1), frame.weather_Clouds.eq(1)
    observed_temp = frame.temp_imputed.eq(0)
    high_temp = frame.temp.gt(292)
    congested_temperatures = frame.loc[congested & observed_temp, "temp"] - 273.15
    temperature_range = congested_temperatures.quantile([0, .1, .9, 1]).rename_axis("quantile").reset_index(name="temperature_c")
    descriptive = pd.DataFrame([
        ("observed_hours", n), ("mean", volume.mean()), ("median", volume.median()),
        ("population_std_ddof0", volume.std(ddof=0)), ("population_variance_ddof0", volume.var(ddof=0)),
        ("sample_std_ddof1", volume.std(ddof=1)), ("sample_variance_ddof1", volume.var(ddof=1)),
        ("minimum", volume.min()), ("maximum", volume.max()), ("range", volume.max() - volume.min()),
    ], columns=["metric", "value"])
    correlation = pd.DataFrame([
        {"scope": name, "paired_hours": int(mask.sum()),
         "pearson_r": frame.loc[mask, "temp"].corr(volume.loc[mask])}
        for name, mask in (("observed_temperature_only", observed_temp), ("including_imputed_sensitivity", pd.Series(True, index=frame.index)))])
    probability = pd.DataFrame([
        ("P(congestion)", int(congested.sum()), n),
        ("P(clear)", int(clear.sum()), n),
        ("P(congestion_and_clear)", int((congested & clear).sum()), n),
        ("P(clear_given_congestion)", int((congested & clear).sum()), int(congested.sum())),
        ("P(congestion_given_clear)", int((congested & clear).sum()), int(clear.sum())),
        ("P(high_temperature_given_congestion)", int((high_temp & congested & observed_temp).sum()), int((congested & observed_temp).sum())),
    ], columns=["event", "numerator_hours", "denominator_hours"])
    if probability.denominator_hours.eq(0).any():
        raise ValueError("A probability denominator is zero")
    probability["probability"] = probability.numerator_hours / probability.denominator_hours
    groups = [("clear_without_clouds", clear & ~clouds), ("clouds_without_clear", clouds & ~clear)]
    contingency = pd.DataFrame([
        {"group": name, "congested_hours": int((mask & congested).sum()),
         "noncongested_hours": int((mask & ~congested).sum()), "observed_hours": int(mask.sum())}
        for name, mask in groups])
    if contingency[["congested_hours", "noncongested_hours"]].le(0).any().any():
        raise ValueError("Odds ratio requires positive contingency cells")
    contingency["congestion_probability"] = contingency.congested_hours / contingency.observed_hours
    contingency["congestion_odds"] = contingency.congested_hours / contingency.noncongested_hours
    odds_ratio = contingency.congestion_odds.iloc[0] / contingency.congestion_odds.iloc[1]
    weather = np.select([groups[0][1], groups[1][1]], ["clear_without_clouds", "clouds_without_clear"], default="excluded")
    strata_frame = pd.DataFrame({"group": weather, "hour": times.dt.hour,
                                 "is_weekend": times.dt.day_of_week.ge(5).astype(int),
                                 "congested": congested.astype(int)})
    strata_frame = strata_frame.loc[strata_frame.group.ne("excluded")]
    standardized, strata_tables = [], []
    for method, keys in (("hour", ["hour"]), ("hour_weekend", ["hour", "is_weekend"])):
        table = strata_frame.groupby(keys + ["group"]).congested.agg(
            congested_hours="sum", observed_hours="size").reset_index()
        rates = table.pivot(index=keys, columns="group", values="congested_hours") / table.pivot(index=keys, columns="group", values="observed_hours")
        if rates.isna().any().any():
            raise ValueError("Every stratum must contain both comparison groups")
        weights = strata_frame.groupby(keys).size() / len(strata_frame)
        adjusted = rates.mul(weights, axis=0).sum()
        for name, _ in groups:
            standardized.append({"method": method, "group": name, "standardized_congestion_probability": adjusted[name],
                                 "weight_population_hours": len(strata_frame)})
        table["method"] = method
        table = table.merge(weights.rename("common_weight").reset_index(), on=keys)
        strata_tables.append(table)
    hourly_2017 = frame.loc[times.dt.year.eq(2017)].groupby(times.loc[times.dt.year.eq(2017)].dt.hour).traffic_volume.agg(
        mean_traffic="mean", observed_hours="size").reset_index()
    hourly_2017 = hourly_2017.rename(columns={"date_time": "hour"})
    weather_means = pd.DataFrame([
        {"weather": name.removeprefix("weather_"), "observed_hours": int(frame[name].eq(1).sum()),
         "mean_traffic": frame.loc[frame[name].eq(1), "traffic_volume"].mean()}
        for name in frame.columns if name.startswith("weather_") and name not in ("weather_main_labels", "weather_description_labels")])
    joint = float((congested & clear).mean())
    product = float(congested.mean() * clear.mean())
    summary = {"observed_hours": n, "congested_hours": int(congested.sum()),
               "congestion_definition": "traffic_volume > 5500", "high_temperature_definition": "temp > 292 K; observed temperature only",
               "joint_clear_congestion": joint, "independence_product": product,
               "joint_minus_product": joint - product, "independence_equality_holds": bool(np.isclose(joint, product, atol=1e-12, rtol=0)),
               "clear_vs_clouds_crude_odds_ratio": float(odds_ratio),
               "excluded_both_weather_hours": int((clear & clouds).sum()),
               "excluded_neither_weather_hours": int((~clear & ~clouds).sum()),
               "temperature_imputed_hours": int((~observed_temp).sum()),
               "interpretation": "Descriptive association only; no causal or predictive performance claim"}
    tables = {"descriptive_statistics": descriptive, "temperature_correlation": correlation,
              "probabilities": probability, "weather_contingency": contingency,
              "weather_standardized_rates": pd.DataFrame(standardized),
              "weather_strata": pd.concat(strata_tables, ignore_index=True),
              "hourly_profile_2017": hourly_2017, "weather_average_traffic": weather_means,
              "congested_temperature_range": temperature_range}
    return tables, summary


def main():
    configure_logging(PROJECT_ROOT / "part1_data_analytics/statistical_analysis.log")
    source = PROJECT_ROOT / "data/processed/traffic_hourly.csv"
    folder = PROJECT_ROOT / "part1_data_analytics/results"
    try:
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        frame = pd.read_csv(source, keep_default_na=False)
        logger.info("Loaded %d rows and %d columns", *frame.shape)
        tables, summary = analyse(frame)
        # Independent SQL counts and standard-library population statistics.
        with sqlite3.connect(":memory:") as connection:
            frame[["traffic_volume", "weather_Clear", "weather_Clouds", "temp", "temp_imputed"]].to_sql("hourly", connection, index=False)
            counts = connection.execute("SELECT COUNT(*), SUM(traffic_volume > 5500), SUM(weather_Clear=1), SUM(traffic_volume>5500 AND weather_Clear=1) FROM hourly").fetchone()
        assert counts == (len(frame), summary["congested_hours"], int(frame.weather_Clear.eq(1).sum()), int(tables["probabilities"].iloc[2].numerator_hours))
        std = statistics.pstdev(frame.traffic_volume.tolist())
        np.testing.assert_allclose(std, frame.traffic_volume.std(ddof=0), atol=1e-9, rtol=0)
        folder.mkdir(exist_ok=True)
        for name, table in tables.items():
            table.to_csv(folder / (name + ".csv"), index=False, float_format="%.12f", na_rep="NA")
            logger.info("Saved %s: %d rows", name + ".csv", len(table))
        if hashlib.sha256(source.read_bytes()).hexdigest() != before:
            raise RuntimeError("Source changed during analysis")
        summary.update({"source_sha256": before, "cross_checks": "SQLite event counts and statistics.pstdev passed", "source": "data/processed/traffic_hourly.csv"})
        (folder / "statistical_validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logger.info("Analysis verified; source unchanged; crude odds ratio %.10f", summary["clear_vs_clouds_crude_odds_ratio"])
    except Exception:
        logger.error("Statistical analysis failed", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
