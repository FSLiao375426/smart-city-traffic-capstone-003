"""Visualize the saved feature/architecture comparison without retraining."""
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from part2_python.pipeline import PROJECT_ROOT,configure_logging

logger=logging.getLogger(__name__)
BASE=PROJECT_ROOT / "part3_machine_learning"


def save_plot():
    folder=BASE / "results/refinement"
    features=pd.read_csv(folder / "feature_ablation.csv").sort_values("mean_fold_mae",ascending=False)
    architectures=pd.read_csv(folder / "architecture_comparison.csv").sort_values("mean_fold_mae",ascending=False)
    labels={"full":"Full: 21 inputs","without_weekend":"Without weekend: 20",
            "time_only":"Time only: 6 (diagnostic)","time_numeric":"Time + numeric: 10 (diagnostic)",
            "time_weather_flags":"Time + weather flags: 17"}
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for axis,table,names,title in ((axes[0],features,[labels[name] for name in features.feature_group],"Feature groups | fixed 64/32 MLP"),
                                   (axes[1],architectures,architectures.architecture.astype(str).str.replace("_"," / ").tolist(),"Architecture | selected feature group")):
        colors=["#176B9A" if value else "#9B9B9B" for value in table.eligible_final_candidate] if "eligible_final_candidate" in table else "#176B9A"
        axis.barh(names,table.mean_fold_mae,color=colors)
        for y,value in enumerate(table.mean_fold_mae):
            axis.text(value+3,y,f"{value:.1f}",va="center",fontsize=9)
        axis.set(title=title,xlabel="Equal mean of 2015/2016 fold MAEs (vehicles/hour)",xlim=(0,table.mean_fold_mae.max()*1.16))
    fig.suptitle("Train-period bounded search | lower MAE is better",fontsize=13)
    fig.text(.02,.02,"Fold-specific imputation/scaling. Grey groups are diagnostic-only. No 2017/2018 data used to rank configurations.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94))
    output=BASE / "figures/refinement_comparison.png"
    fig.savefig(output,dpi=160)
    plt.close(fig)
    logger.info("Saved %s",output)


if __name__=="__main__":
    configure_logging(BASE / "plot_refinement.log")
    try:
        save_plot()
    except Exception:
        logger.error("Refinement plotting failed",exc_info=True)
        sys.exit(1)
