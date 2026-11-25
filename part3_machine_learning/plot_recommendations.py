"""Plot saved conditional recommendation examples without fitting or querying new data."""
import json
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from part2_python.pipeline import PROJECT_ROOT,configure_logging

logger=logging.getLogger(__name__)
BASE=PROJECT_ROOT / "part3_machine_learning"


def main():
    configure_logging(BASE / "plot_recommendations.log")
    try:
        fig,ax=plt.subplots(figsize=(10,5.5))
        cases=(("weekday_clear","Weekday / Clear"),("weekend_clear","Weekend / Clear"),("weekday_rain","Weekday / Rain"))
        for name,label in cases:
            result=json.loads((BASE / "results/recommendation" / (name+".json")).read_text(encoding="utf-8"))
            best=result["best_window"]
            if best:
                label+=f" | window {best['start_hour']:02}:00-{best['end_hour']:02}:00"
            rows=result["hourly_estimates"]
            ax.plot([row["hour"] for row in rows],[row["estimated_traffic"] for row in rows],marker="o",markersize=4,label=label)
        ax.set(title="Departure-time scenarios | permitted interval 08:00-19:00",
               xlabel="Source-calendar departure hour",ylabel="Estimated traffic (vehicles/hour)",xticks=range(8,19),ylim=(0,None))
        ax.legend(frameon=False,fontsize=9)
        ax.grid(axis="y",alpha=.2)
        fig.text(.06,.025,"Fixed supplied weather per scenario; 2-hour departure windows, not journey duration.\nModel estimates, not observed future traffic; support cohorts use training hours only.",fontsize=8)
        fig.tight_layout(rect=(0,.1,1,1))
        output=BASE / "figures/recommendation_scenarios.png"
        fig.savefig(output,dpi=160)
        plt.close(fig)
        logger.info("Saved %s",output)
    except Exception:
        logger.error("Recommendation plot failed",exc_info=True)
        return 1
    return 0


if __name__=="__main__":
    sys.exit(main())
