# Smart City Traffic Intelligence: choosing a departure window on I-94

For the westbound I-94 corridor near Minneapolis–St Paul, when might a traveller encounter lower vehicle volume? A model can compare departure times given a date and weather assumptions. Its typical error matters: on the 2018 holdout the forest missed observed hourly volume by 242 vehicles on average, rising to 287 on weekends. A small difference between two candidate windows therefore deserves caution. The dataset measures traffic counts; evaluating travel-time savings or collision risk would require additional outcomes.

## Data and evaluation design

Part 1's SQL, probability analysis and Power BI dashboard established the unit of analysis: several weather records may share one hourly traffic count. Part 2 produced a reproducible pipeline that preserved the original 48,204 rows and aggregated them into 40,575 observed hours. Nine consecutive months are missing. Missing hours were left unobserved, while recorded zero volumes were retained. The observed pattern of weekday peaks and weekend midday traffic motivated comparing departure windows within a day type. [Part 1 findings](../output/pdf/part1_data_analytics_insights_report.pdf) and [Part 2 methods](../output/pdf/part2_traffic_pipeline_report.pdf).

Training used 25,329 observations through 2016. The 8,713 hours of 2017 informed model selection. The frozen classifiers and regressors were represented in the final evaluation by one selected forest per task, evaluated on 6,533 observed hours in January–September 2018. Monthly imputation values, scaling and volume quartiles were fitted on training data. Both tasks used the same 21 inputs: cyclical hour and weekday encodings, weekend and holiday flags, four numeric weather variables and eleven weather indicators. Actual volume and derived target flags were excluded from model inputs. [Feature and target specification](results/dataset_metadata.json).

## 1. Supervised models

Regression estimates vehicles per hour. Classification predicts a constructed label: volume above the training median (3,339 vehicles/hour) together with a designated severe-weather or low-visibility category. No accident dataset was supplied or used.

| Classification model and evaluation period | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Majority baseline, 2017 | .8849 | .0000 | .0000 | .0000 | .5000 |
| Logistic regression, 2017 | .9412 | .6687 | .9701 | .7917 | .9899 |
| Random forest, 2017 | .9882 | .9198 | .9831 | .9504 | .9986 |
| Frozen random forest, 2018 | .9810 | .8594 | .9917 | .9208 | .9964 |

| Regression model and evaluation period | MAE (vehicles/hour) | R² |
|---|---:|---:|
| Training-mean baseline, 2017 | 1,744.46 | −.0039 |
| Linear regression, 2017 | 825.75 | .7106 |
| Random forest, 2017 | 246.26 | .9583 |
| Frozen random forest, 2018 | 241.54 | .9600 |

The selection criteria were highest validation F1 and lowest validation MAE. Despite its high accuracy, the majority baseline detected zero positive hours. In 2018 the chosen classifier found 721 of 727 actual proxy positives, missed six and flagged another 118 hours incorrectly. Weather contributes to the label definition, which helps explain why reproducing that rule is feasible. [Validation comparison](results/validation_metrics.csv), [frozen protocol](results/final_evaluation_plan.json) and [holdout results](results/final_2018/metrics.json).

## 2. Unsupervised patterns

K-means used standardized traffic volume, cyclic hour coordinates and a binary adverse-weather indicator. Across k=2–6, k=6 had the highest sampled training silhouette (.4613), close to k=5 (.4597). Four clusters without a designated adverse-weather flag described overnight low volume, morning-to-noon traffic, afternoon traffic and evening traffic. The two flagged clusters separated mostly night/early morning from mostly daytime conditions. The cluster numbers are descriptive identifiers. Because volume was used to form the clusters, the assignments were excluded from the same-hour supervised inputs. [Cluster selection and evidence](results/clustering/clustering_manifest.json).

Association mining on the training period yielded 83 retained rules. The highest-lift examples for Low, Medium, High and Severe volume respectively were weekend 00–05 (lift 3.59), weekend 21–23 (3.20), cloudy weekend 13–16 (3.43) and clear weekday 06–09 (3.40). For the last rule, 689 of 810 matching training hours had Severe volume (85.06%), approximately 3.4 times the overall training prevalence of Severe. Its 2017 confidence was 87.08%. These are observed associations, so changing the weather in a scenario cannot establish a causal traffic effect. [Rules and support counts](results/association_rules/top_rules_per_level.csv).

## 3. Neural demand model and explanation

The initial 21-input MLP used 64 and 32 hidden units and early stopping on 2017 validation loss. Its validation MAE was 268.78 vehicles/hour (R² .9548). Kernel SHAP explained the actual MLP using 64 training background hours and 24 sampled validation cases. For one observed 04:00 hour, the background prediction was about 3,168 vehicles/hour and the feature contributions together lowered the estimate to about 894; observed traffic was 843. The background and case selection matter when interpreting those contributions. The explanation applies to this MLP. [Neural experiment](results/neural/neural_manifest.json).

A bounded comparison of feature sets and architectures on two earlier chronological folds selected a 17-input, 128/64-unit model. Its frozen configuration scored MAE 288.45 on 2017, giving no improvement over the initial MLP or the forest. The original MLP used 2017 for early stopping, so its validation protocol differs from the refined model's. [Refinement evidence](results/refinement/refinement_manifest.json).

## 4. Advanced technique: MLflow

Model comparison required tracing each reported score to a particular feature contract, split and saved estimator. MLflow tracked run IDs, parameters, validation metrics and artifacts. Portable exports allow review without the original local tracking server; model hashes tie the selected runs to the saved files. This proved useful when comparing the original and refined neural configurations and when serving the selected forest. Tracking records make choices auditable, while split integrity and the validity of the proxy still require separate checks. [Baseline run export](results/mlflow_evidence.json) and [refinement run export](results/refinement/mlflow_evidence.json).

## 5. Departure-time recommendation

For a user-specified interval, the service derives calendar features, holds supplied weather constant and estimates each candidate hour. It ranks contiguous windows by mean predicted volume and checks that each included hour has enough matching training observations. The default floor is 30 matching hours, an analyst setting. With a weekday clear-sky assumption of 20°C, 1% cloud and zero precipitation, the saved 08:00–19:00 scenario ranked 10:00–12:00 first at 4,425 vehicles/hour. The comparable weekend scenario ranked 08:00–10:00 first at 2,842. The rare Squall scenario yielded no supported window. A two-hour window gives the traveller flexibility to depart within it. [Scenario results](results/recommendation/recommendation_manifest.json).

## 6. Deployment and monitoring

The Flask mock exposes the saved forest at POST /predict. It uses a frozen bundle and reports the model run ID. Its recommendation text states the supplied weather and source, including any median-imputed inputs, before giving an estimate. Eight loopback HTTP checks covered readiness, four scenarios and three invalid requests. The offline monitoring exercise compares the distribution of source-calendar hours with the training reference. Replay gave PASS at total variation distance zero; 1,000 synthetic 08:00 observations triggered ALERT at .9583 against a prototype threshold of .20. A small batch also raised a sample-size warning. This tests an input-monitoring procedure. Operational error monitoring would require matching issued estimates to subsequent measured volumes. [HTTP evidence](results/serving/http_acceptance.json) and [monitoring evidence](results/monitoring/simulation.json).

## 7. Responsible use and decision

Weekday precision was **92.20%** (603 of 654 flags). Weekend precision was **63.78%** (118 of 185 flags), while weekend recall was **95.93%** (118 of 123 actual proxy positives). Thus weekend flags were less often confirmed by the proxy, even though the model captured most positive weekend hours. Neither percentage measures accidents. Weekend volume MAE was also higher: 287.19 versus 223.34 vehicles/hour on weekdays. These differences warrant showing condition-specific error to travellers. Coverage of one corridor and the absence of traveller demographics limit broader fairness claims.

Use the prototype for conditional timing comparisons. Before giving operational advice, evaluate later seasons, archived forecast inputs and feasible user windows; assign a reviewer to investigate alerts. Reusing frozen models and bounded trials controls compute expenditure. Recorded fit durations support a local cost comparison, but hardware energy and emissions were not measured. The [standalone bias and fairness report](bias_and_fairness_report.md) provides the subgroup counts, governance recommendations and resource trade-offs.
