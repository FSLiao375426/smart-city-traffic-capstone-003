# Bias and fairness in the I-94 traffic prototype

The recommendation engine ranks departure windows by estimated hourly vehicle volume on one westbound I-94 corridor. For a weekend user, the most relevant observed volume error was **287 vehicles/hour**, compared with 223 on weekdays. The difference warrants separate reporting to users and follow-up evaluation before operational use. These averages describe groups of observed hours. They do not give an error bound for an individual trip.

## Who and what the observations cover

The source contains 48,204 rows for 40,575 distinct hours. Several weather descriptions can share a single hourly count. Counting each such row as a separate traffic observation would give those hours disproportionate weight. The pipeline keeps one volume per observed hour and preserves the weather labels. Nine complete months from September 2014 through May 2015 are missing. The final evaluation covers 6,533 observed hours from January through September 2018, leaving the last quarter of that year outside the test. Performance on other roads, directions, years and unobserved hours requires new evidence.

The data describe traffic passing a sensor. They contain no individual traveller demographics, trip durations or verified crash outcomes. The audit can compare service quality across hours and weather conditions. Assessing equity across neighbourhoods or demographic groups would require appropriately governed data and broader geographic coverage.

## Where the errors fall

The following groups were specified before the final scores were calculated. Each pair partitions the same 6,533 hours. MAE is the mean absolute difference between estimated and observed vehicles per hour. A positive label requires both volume in the upper two training-defined quartiles and a designated adverse-weather or low-visibility category.

| Group | Hours | Volume MAE | Proxy positives | False alarms | Misses | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Weekday | 4,671 | 223.34 | 604 | 51 | 1 | 92.20% | 99.83% |
| Weekend | 1,862 | 287.19 | 123 | 67 | 5 | 63.78% | 95.93% |
| Outside the designated weather categories | 4,904 | 225.97 | 0 | 0 | 0 | Undefined | Undefined |
| In a designated weather category | 1,629 | 288.42 | 727 | 118 | 6 | 85.94% | 99.17% |

The classifier flagged 185 weekend hours, of which 118 met the proxy rule. Weekend users would see more false alarms per flag than weekday users in this sample. The weekend group also had a lower proportion of actual proxy positives (123/1,862 versus 604/4,671). These counts describe unequal performance; they do not establish its cause or discrimination against a population. Weather is both a model input and part of the proxy definition. Outside the designated categories the rule makes every actual label negative, and the classifier issued no flags. Both precision and recall therefore have zero denominators. Reporting 100% for either would misrepresent the evidence.

A flag indicates agreement with a constructed volume-and-weather rule. Describing it as an accident probability could create unnecessary alarm or false reassurance. Similarly, lower estimated traffic supplies no measured travel-time or safety benefit. The holdout used historical same-hour weather. A day-ahead service would also need evaluation against weather forecasts archived at the time each prediction was issued.

## Oversight and resource use

Before operational use, a transport analyst should review the intended corridor and decision, compare errors and sample counts across seasons and day types, and test on a later period with inputs available at decision time. A service owner should retain the model version, issued estimate, weather source and imputed values so that later traffic observations can be matched to the original estimate. Show weather assumptions before the result and withhold a recommended window when support is insufficient. Reserve safety-critical decisions for a human with verified incident information.

The current offline monitor checks the distribution of hours. A proposed operational response is to inspect coverage, timestamp alignment and weather feeds after an alert, then review errors once measured traffic arrives. Changes to a model or its thresholds should receive a new version and independent evaluation. The 2018 test has already served its final-evaluation purpose.

There is a measurable compute trade-off in the saved baseline runs: linear regression took about 0.0022 seconds to fit with validation MAE 825.75, while the forest took 2.90 seconds with MAE 246.26. These timings describe individual local fits. They exclude data preparation, serving and hardware energy consumption. The project bounded neural architecture trials, used early stopping for the original MLP and reused saved models for recommendations and the final evaluation. For a deployment decision, benchmark latency, throughput and energy per batch on the target hardware before claiming energy or emissions savings.

Evidence: [frozen evaluation plan](results/final_evaluation_plan.json), [holdout metrics and group counts](results/final_2018/metrics.json), [hourly predictions](results/final_2018/predictions.csv), [baseline timings and validation metrics](results/validation_metrics.csv), [monitoring simulation](results/monitoring/simulation.json).
