"""Save three descriptive figures: python -m part2_python.visualize."""
from __future__ import annotations

import hashlib
import json
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part2_python.pipeline import PROJECT_ROOT, configure_logging

logger = logging.getLogger(__name__)


def main(*, configure: bool = True) -> int:
    if configure:
        configure_logging(PROJECT_ROOT / "part2_python" / "visualize.log")
    source = PROJECT_ROOT / "data/processed/traffic_time_features.csv"
    folder = PROJECT_ROOT / "part2_python/figures"
    try:
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        frame = pd.read_csv(source, keep_default_na=False)
        times = pd.to_datetime(frame["date_time"], errors="raise")
        values = frame[["traffic_volume", "temp"]].apply(pd.to_numeric, errors="raise")
        if times.isna().any() or times.duplicated().any():
            raise ValueError("Expected one row per observed hour")
        if not np.isfinite(values.to_numpy()).all() or values.traffic_volume.lt(0).any():
            raise ValueError("Expected finite weather and nonnegative traffic")
        frame["hour"] = times.dt.hour
        frame["is_weekend"] = times.dt.day_of_week.ge(5).astype(int)
        frame["temp_c"] = values.temp - 273.15
        folder.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                             "axes.spines.right": False, "figure.facecolor": "white"})
        note = (f"All {len(frame):,} observed hours | {times.min():%Y-%m-%d} to {times.max():%Y-%m-%d}\n"
                "Missing hours remain absent; descriptive analysis, not a model evaluation.")

        def save(fig, name):
            fig.text(.08, .025, note, fontsize=8, color="#555555")
            fig.tight_layout(rect=(0, .11, 1, 1))
            fig.savefig(folder / name, dpi=160)
            plt.close(fig)
            logger.info("Saved figure %s", folder / name)

        profile = frame.groupby(["is_weekend", "hour"]).traffic_volume.agg(
            mean_traffic="mean", observed_hours="size").reset_index()
        profile.to_csv(folder / "hourly_profile.csv", index=False, float_format="%.8f")
        fig, ax = plt.subplots(figsize=(10, 6))
        for weekend, label, color in [(0, "Weekday (Mon-Fri)", "#176B9A"),
                                      (1, "Weekend (Sat-Sun)", "#D06A25")]:
            part = profile.loc[profile.is_weekend.eq(weekend)]
            ax.plot(part.hour, part.mean_traffic, marker="o", ms=4, label=label, color=color)
        ax.set(title="Weekday and weekend traffic follow different daily patterns",
               xlabel="Hour of day (source timestamps)", ylabel="Mean traffic (vehicles per observed hour)",
               xlim=(0, 23), ylim=(0, None), xticks=range(0, 24, 2))
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
        save(fig, "01_hourly_profile.png")

        edges = np.arange(0, np.ceil(values.traffic_volume.max() / 250) * 250 + 1, 250)
        counts, edges = np.histogram(values.traffic_volume, bins=edges)
        pd.DataFrame({"left_inclusive": edges[:-1], "right_edge": edges[1:],
                      "observed_hours": counts}).to_csv(folder / "traffic_histogram.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stairs(counts, edges, fill=True, color="#176B9A", alpha=.8)
        ax.axvline(values.traffic_volume.mean(), color="#A64122", label="Mean: %.2f" % values.traffic_volume.mean())
        ax.axvline(values.traffic_volume.median(), color="#313131", ls="--", label="Median: %.0f" % values.traffic_volume.median())
        ax.set(title="Observed hourly traffic spans low and high volume regimes",
               xlabel="Traffic volume (vehicles per hour); bin width = 250", ylabel="Observed hours",
               xlim=(0, edges[-1]))
        ax.legend(frameon=False)
        save(fig, "02_traffic_distribution.png")

        frame[["date_time", "temp_c", "traffic_volume"]].to_csv(
            folder / "temperature_traffic.csv", index=False, float_format="%.8f")
        correlation = frame.temp_c.corr(values.traffic_volume)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(frame.temp_c, values.traffic_volume, s=3, alpha=.07, color="#176B9A", linewidths=0)
        ax.set(title="Temperature has a weak overall linear association with traffic",
               xlabel="Hourly temperature (°C)", ylabel="Traffic volume (vehicles per hour)")
        ax.text(.03, .97, f"Pearson r = {correlation:.3f}\nAll observations; no sampling\nTime of day may confound the association",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
        save(fig, "03_temperature_traffic.png")
        if int(counts.sum()) != len(frame) or int(profile.observed_hours.sum()) != len(frame):
            raise RuntimeError("Chart counts do not preserve observed hours")
        if hashlib.sha256(source.read_bytes()).hexdigest() != before:
            raise RuntimeError("Source changed while plotting")
        metadata = {"source": str(source.relative_to(PROJECT_ROOT)), "source_sha256": before,
                    "observed_hours": len(frame), "zero_traffic_hours": int(values.traffic_volume.eq(0).sum()),
                    "pearson_temp_traffic": float(correlation), "histogram_bin_width": 250,
                    "histogram_intervals": "[left, right), except last includes right edge",
                    "matplotlib_version": matplotlib.__version__,
                    "scope": "Full-period descriptive analysis; not held-out model performance"}
        (folder / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        logger.info("Verified %d observed hours; source hash unchanged; r=%.8f", len(frame), correlation)
    except Exception:
        logger.error("Visualization failed", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
