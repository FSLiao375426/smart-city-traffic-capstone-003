# Part 3: Machine learning for I-94 traffic

Read the [final capstone report](final_capstone_report.md) for the seven tasks and frozen 2018 results, and the [bias and fairness report](bias_and_fairness_report.md) for subgroup performance. The sections below document successive development stages and their status at the time.

The first Part 3 implementation estimates traffic volume and a constructed binary risk proxy, conditional on the same hour's observed weather and calendar information. It is not a one-hour-ahead forecast or a real accident prediction system. The user approved this initial task and the operational weather definitions on 2026-11-25.

## Review and reproduction

For current results, start with the final report and `results/final_2018/metrics.json`. The submission acceptance record is available at `output/submission/capstone_part3_acceptance.json` in the project repository; it is not included in the portable ZIP. Later sections document successive development stages; historical statements that the 2018 test was unscored or that tasks remained pending do not describe the current status.

**Protect the frozen evidence:** training and verification commands write outputs and logs. Retraining replaces saved models and result files and creates MLflow runs; `verify_baselines` also writes an export and needs the original local tracking store. Use a separate copy for reproduction and preserve the frozen evaluation protocol.

The documented environment used Python 3.11.17 and the dependencies in `requirements.txt`. The acceptance record describes its checks and notes that installation in a fresh environment was not tested. Reproduction also requires Part 2 outputs; the Part 2 pipeline regenerates processed files and logs.

Original development commands, for an isolated reproduction:

```powershell
python -m part3_machine_learning.dataset
python -m part3_machine_learning.train_baselines
python -W error -m unittest discover -s part3_machine_learning/tests -v
python -m part3_machine_learning.verify_baselines
```

The original local MLflow store is required for `verify_baselines`; portable exports permit review without it. Model scores use same-hour observed weather and do not measure end-to-end performance using future weather forecasts.

## Targets and feature availability

- Regression: traffic_volume, in vehicles per hour. Travel time is not supplied, so no travel-time target is invented.
- Congestion category: fixed training-reference quartiles, levels 1/2/3/4 corresponding to Low/Medium/High/Severe. Cutoffs are 1204/3339/4932, with equality in the lower interval.
- Severe weather: any of Snow, Thunderstorm or Squall.
- Assumed low visibility: any of Fog, Mist, Haze or Smoke. These categorical proxies are not measured visibility or an official severity classification.
- Binary classification target: high_risk = (quartile >= 3) AND (severe weather OR assumed low visibility). Multi-hot weather is preserved. A Clear flag can coexist with Fog: such a record still satisfies risky weather. The teaching example of Clear alone does not.

Both tasks use the same explicit whitelist of 21 input columns: hour and weekday sine/cosine pairs, weekend and holiday flags, scaled temperature/cloud cover, rain/snow amounts and 11 weather flags. Weekday encoding uses period 7, preserving the Sunday/Monday boundary. The feature matrix excludes traffic_volume, quartile level, high_risk, the three derived weather-rule flags, split labels, timestamps, source counts and quality/imputation flags. This prevents direct target-derived input leakage. Weather flags still overlap with the proxy's definition, so classification partly reproduces the constructed rule; high validation scores do not demonstrate accident prediction.

The same-hour observed weather assumption is essential. There are no lags or shifted next-row targets. Turning this into a future forecast would require prediction-time weather availability and true timestamp alignment across missing hours. Dates and holiday flags follow the source convention; timezone conversion and a corrected external holiday calendar have not been introduced.

## Chronological split and fitted transformations

| Split | Period | Hours | Proxy positive hours |
|---|---|---:|---:|
| Training | Through 2016-12-31 23:59:59 inclusive | 25329 | 2257 |
| Validation | Calendar 2017 | 8713 | 1003 |
| Final test | Available 2018 observations | 6533 | Not used for evaluation or selection |

The demonstration reference cutoff from Part 2 is adopted as this first training split. Dataset construction verifies that imputation, scaling and quartile metadata all use the training cutoff, recalculates training-only means/SDs and thresholds, and verifies every existing column is preserved. Linear/logistic models also fit their StandardScaler on training rows only. No randomized train/test split is used.

`traffic_supervised.csv` has 40,575 rows and 43 columns: the 36 Part 2 columns plus two weekday coordinates, three derived weather-rule flags, high_risk and split. Preparation assigns deterministic labels to all partitions and checks schema/integrity. Training, predictions, metrics and candidate selection use only training/2017 validation. No 2018 metrics or predictions were produced. The final test is reserved until the modelling decisions are frozen; these are validation results, not final test claims.

## First validation results

All classifiers use the default 0.5 decision threshold. Logistic regression and the classifier forest use balanced class weights. Random seed is 42. The logistic solver has max_iter=2000. The forests have 100 trees and min_samples_leaf=5, with classifier/regressor max_depth 14/16 respectively and two CPU workers. Full estimator settings are logged in MLflow.

| Classification model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Dummy prior | 0.8849 | 0 | 0 | 0 | 0.5000 |
| Logistic regression | 0.9412 | 0.6687 | 0.9701 | 0.7917 | 0.9900 |
| Random forest | 0.9882 | 0.9198 | 0.9831 | 0.9504 | 0.9986 |

The dummy classifier predicts the majority class for every hour and misses every proxy positive. Its 88.49% accuracy is therefore not useful evidence of risk detection. F1 balances precision and recall for the positive class; ROC AUC measures ranking across thresholds. Neither is a real accident-risk estimate here.

| Regression model | MAE (vehicles/hour) | R-squared |
|---|---:|---:|
| Dummy training mean | 1744.46 | -0.0039 |
| Linear regression | 825.75 | 0.7106 |
| Random forest | 246.26 | 0.9583 |

MAE is the mean magnitude of validation errors, not an error guarantee for every hour. Forests have the strongest first validation results under the predeclared selection rules: highest classification F1 and lowest regression MAE among learned models. They are validation candidates, not finalized deployed models. No extensive tuning, time-series cross-validation, final test evaluation, probability calibration or condition-specific error audit is claimed yet.

## Saved evidence and MLflow

| Location | Contents |
|---|---|
| data/processed/traffic_supervised.csv | Prepared data with explicit chronological split |
| results/dataset_metadata.json | Definitions, whitelist, source hashes, split counts and limitations |
| results/validation_metrics.csv | Six validation rows: two algorithms plus a dummy per task |
| results/validation_predictions.csv | 52,278 validation predictions, including proxy scores for classifiers |
| results/training_manifest.json | Candidate selection, model versions/hashes, software versions and actual run IDs |
| results/mlflow_evidence.json | Export of six real FINISHED runs, parameters, tags, metrics and artifact checks |
| models/&lt;name&gt;/ | Six persisted model.joblib files with feature/schema/metric metadata |
| dataset.log / train_baselines.log / verify_baselines.log | Preparation, training and evidence-verification events |
| mlflow/mlflow.db and mlflow/artifacts/ | Recreated local SQLite tracking store and per-run artifacts |

MLflow was enabled before fitting the first model. Each actual run records task, model version v1, split, seed, data hash, feature count, estimator settings, training time and validation metrics; it stores the model and input schema, dataset metadata and classifier confusion matrix. This supplies an audit trail and comparison evidence, rather than reconstructing experiments later. Persisted-model predictions match live predictions.

The local `mlflow/` directory is ignored to avoid committing a machine-specific database and duplicate binary artifacts. Portable `results/mlflow_evidence.json`, manifests, model files and logs remain trackable. MLflow artifact URIs in the local store are machine-specific; the export is a review snapshot, not a portable tracking server. Re-running training creates six additional real runs and replaces current model/CSV/manifest outputs; export evidence again after rerunning. This is local experiment tracking, not production model governance or deployment.

To inspect the current local runs, run the following from the repository root and open the localhost URL reported by MLflow:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///part3_machine_learning/mlflow/mlflow.db --port 5000
```

Three Part 3 tests pass: synthetic proxy/multi-label cases and weekday cycle, real-data source/preprocessing/feature/split checks, and independent recomputation of saved validation metrics. The evidence exporter verified six FINISHED runs, exact saved model hashes, matching metrics and validation-only dates. Existing Part 1/2 code is unchanged; their prior tests were not rerun without a new dependency on changed code.

## Remaining work and incremental commits

Supervised baseline comparisons and initial MLflow tracking are implemented. Final test evaluation, unsupervised clustering and association rules, a neural network with SHAP/LIME, recommendation logic, versioning/deployment/monitoring, condition-specific errors and responsible-AI reporting remain pending. MLflow work contributes to the advanced-technique and MLOps requirements but does not complete every requirement of either task.

Suggested user-controlled commits:

1. Add same-hour supervised dataset and proxy labels: dataset.py, package initializer, requirements.txt, dataset.log, dataset metadata, supervised CSV and the dataset-related tests.
2. Add tracked supervised baselines and validation evidence: train_baselines.py, verify_baselines.py, their logs, six model folders, remaining results/test changes, .gitignore and this README. Keep the test file together in this second batch if not using partial staging, because it also checks generated training results. Commit the root overview link with this documentation batch.

No stage/commit/push is performed by the assistant. No final report or model metric has been silently substituted into the already reviewed Part 1/2 PDFs or submission ZIP.

## K-means traffic regimes (2026-11-25)

Run `python -m part3_machine_learning.clustering`, then the existing Part 3 test command. The module compares k=2..6 with ten initializations, seed 42 and a maximum of 300 iterations. Every candidate fits all 25,329 training hours. Selection uses silhouette on the SAME 2,000 seeded training samples for each candidate; neither validation nor the 2018 final test selects k. Smaller k breaks an exact tie.

Four inputs are standardized with training-only StandardScaler: traffic_volume, hour_sin, hour_cos and adverse_weather. The last is a binary indicator of any Snow/Thunderstorm/Squall/Fog/Mist/Haze/Smoke flag, using the previously approved category assumptions; it is not measured visibility or a continuous severity scale. It is recomputed from the weather flags. high_risk and traffic_volume_quartile are not inputs. Traffic volume is allowed here because the purpose is to describe observed traffic regimes, not forecast it. Do not feed these traffic-dependent cluster IDs into same-hour supervised predictions: that would reintroduce the target indirectly.

| k | Sampled training silhouette |
|---|---:|
| 2 | 0.37837 |
| 3 | 0.43272 |
| 4 | 0.42269 |
| 5 | 0.45969 |
| 6 | 0.46131 |

The selected k=6 is best under this internal criterion within the compared range. It is only slightly higher than k=5 (difference about 0.00163); no claim of a uniquely correct or stable six-cluster structure is made. Silhouette measures relative cohesion/separation, typically from -1 to 1; it is not classification accuracy. Inertia generally falls with more clusters, so it is logged but not used alone to select k. All training clusters have at least 9.61% of training hours.

| Cluster ID | Post-hoc descriptive name | Training hours | Mean traffic |
|---|---|---:|---:|
| 0 | Evening, no selected adverse-weather flag | 4631 | 2515.10 |
| 1 | Afternoon, no selected adverse-weather flag | 5252 | 5104.66 |
| 2 | Morning to noon, no selected adverse-weather flag | 5116 | 4761.16 |
| 3 | Mostly night/early morning, selected adverse weather | 2595 | 1308.23 |
| 4 | Overnight low traffic, no selected adverse-weather flag | 5302 | 832.88 |
| 5 | Mostly daytime, selected adverse weather | 2433 | 4832.25 |

Names summarize the observed profiles, not hard clock boundaries or accident severity. Cluster IDs have no ordinal risk meaning. In this solution the standardized binary weather variable separates adverse and other hours completely, illustrating how feature choice and Euclidean scaling shape the result. An adverse flag of zero can still include rain or drizzle, so it must not be called universally good weather. Membership does not establish a causal weather effect.

Outputs in results/clustering/: k_comparison.csv, cluster_summary.csv, cluster_hour_profile.csv, cluster_assignments.csv (34,042 train/validation rows), cluster_centers.csv and clustering_manifest.json. Summary statistics and profiles separate training from validation. centers are in original feature units; inverse-scaled hour sine/cosine centroids need not be points on the unit circle. The selected scaler/model are in models/kmeans/. The figure figures/kmeans_selection_profiles.png uses training observations only and has been visually inspected.

Five actual FINISHED MLflow runs are exported in the clustering manifest with parameters, metrics and IDs; the selected run contains the saved model and result artifacts. The 2018 test period was not fitted, assigned or plotted. Source hash and saved/reloaded predictions match. Three new clustering checks plus the three existing supervised checks passed (six total): label-input isolation, training-only scaling, saved/reloaded assignments, counts, selection rule and model hash.

The updated plot can be regenerated from existing comparison/profile CSVs with the module's save_plot helper without retraining. A full rerun creates five additional real MLflow runs and replaces the current clustering outputs. Stage clustering.py, clustering.log, tests/test_clustering.py, models/kmeans/, results/clustering/, the PNG and this README as a separate descriptive clustering commit when reviewing the still-uncommitted Part 3 work. Association-rule mining remains the next unsupervised component; the full Task 2 is not yet complete.

## Association rules from training-hour transactions (2026-11-25)

Run `python -m part3_machine_learning.association_rules`. mlxtend 0.25.0 was actually imported and its FP-growth implementation executed. The module mines only the 25,329 training hours. It freezes rule IDs and training-lift ordering before calculating metrics on 8,713 validation hours. No 2018 transactions are built and no validation-based rule selection is performed.

### How one hour becomes a transaction

The first real row is 2012-10-02 09:00:00, Tuesday, Clouds, traffic 5545, training quartile 4. Its item set is:

```text
{time=morning_06_09, day=weekday, weather=Clouds, congestion=Severe}
```

Time bins are fixed, human-defined inclusive ranges: 00-05 night, 06-09 morning, 10-12 midday, 13-16 afternoon, 17-20 evening and 21-23 late evening. These are modelling choices, not learned time boundaries. Weekday/weekend comes from the timestamp. Every weather flag that equals 1 contributes an item, so Rain and Snow can both appear. Quartile 1/2/3/4 maps to Low/Medium/High/Severe using the frozen training thresholds 1204/3339/4932.

The item set is encoded into 23 Boolean columns (six time bins, two day types, 11 weather categories and four outcomes). Exactly one time bin, day type and outcome is true per hour; weather is multi-hot. transaction_examples.json and transaction_encoding_examples.csv show the first ten training hours. Raw traffic volume is displayed for review but is not a condition item. Outcome items are included to count co-occurrence, but can only occur on the RIGHT side of a retained rule. high_risk, cluster IDs, traffic-derived flags and continuous target values never appear in antecedents.

### Mining all four outcomes

FP-growth first finds frequent item combinations once. Code then retains combinations with exactly one congestion outcome and one to three context items. It calculates support=P(A AND B), confidence=P(B|A) and lift=P(B|A)/P(B). This naturally covers each of the four outcomes; it does not fit four classifiers or need a manual run for each level. Time/day values within one transaction are mutually exclusive; multiple weather items may coexist.

Predeclared training criteria: support >=1% (at least 254 joint hours), at least 500 antecedent hours, confidence >=30% and lift >=1.2; maximum itemset length four including the outcome. FP-growth produced 291 frequent itemsets, of which 83 context-to-congestion rules passed. Rules are ranked by training lift then support, with deterministic tie ordering. top_rules_per_level.csv contains at most three per outcome, still chosen from the training ranking, not from validation performance.

| Highest training-lift rule for each outcome | Training joint / antecedent | Training confidence | Training lift | 2017 confidence |
|---|---:|---:|---:|---:|
| Weekend AND 00-05 => Low | 1639 / 1827 | 89.71% | 3.5857 | 84.19% |
| Weekend AND 21-23 => Medium | 737 / 921 | 80.02% | 3.2025 | 77.78% |
| Weekend AND 13-16 AND Clouds => High | 491 / 572 | 85.84% | 3.4337 | 83.41% |
| Weekday AND 06-09 AND Clear => Severe | 689 / 810 | 85.06% | 3.4031 | 87.08% |

A simpler timing rule, Weekday AND 06-09 => Severe, has training confidence 2425/2960=81.93%, lift 3.2777, support 9.57% and 2017 confidence 879/1029=85.42%. This offers a more broadly supported timing description than some weather-specific refinements. It does not establish that time or weather causes congestion.

Outputs under results/association_rules/: rules_train_validation.csv (all 83), top_rules_per_level.csv, frequent_itemsets.csv, frozen_rules.json, transaction_examples.json, transaction_encoding_examples.csv, mining_manifest.json and mlflow_evidence.json. One real FINISHED MLflow run records the method, criteria, data hash, metrics and artifacts. The local store emitted a third-party SQLAlchemy noload deprecation warning during setup; the run still finished and no pipeline failure was recorded. No global warning suppression was introduced.

Three new rule tests plus six prior Part 3 tests pass with warnings treated as errors. They check inclusive time-bin boundaries, multi-weather encoding, exact synthetic metrics, zero-antecedent handling and every saved rule's train/validation counts/metrics against source-derived transactions. Fixed antecedents may have little validation support, so both support counts and ratios are saved; no automatic filtering on validation is applied.

Interpretation limits: rules are descriptive co-occurrences, not causal evidence or future forecasts. Confidence does not mean accident probability. High lift may coexist with modest support; overlapping/nested rules are not independent discoveries. The minimum-support requirement suppresses rare weather such as Squall/Smoke, so no rule does not mean safe. Time-bin boundaries, observational coverage and label definitions affect findings. There is no multiplicity-adjusted significance claim. Task 2 now has K-means and association-rule initial implementations; final integrated reporting and review remain pending.

### Rule-search size and threshold rationale audit

The 500-hour antecedent floor and 1% joint support were analyst-chosen first-pass settings, not course requirements, learned thresholds or statistical guarantees. The 500-hour floor is about 1.97% of 25,329 training hours; the 1% support implies ceil(253.29)=254 joint hours. They reduce tiny-sample results but do not establish independence, stability or significance; adjacent hours are correlated. No threshold sensitivity study or optimality claim has been made.

With 19 context items (6 time, 2 day type, 11 weather), 1-3 antecedent items and one of four outcomes, an unconstrained count is 4 x [C(19,1)+C(19,2)+C(19,3)] = 4636. Excluding antecedents containing two time bins or both day types leaves 19, 155 and 737 possible antecedents of sizes 1, 2 and 3, respectively: 911 x 4 = 3644 structurally compatible theoretical rules. This still includes weather combinations never observed. It is a search-space count, not the number FP-growth explicitly evaluated.

FP-growth returned 291 frequent itemsets of all types, including context-only sets and singletons. Of these, 164 give an eligible nonempty-context => one-outcome rule with joint support >=254 hours. Sequential filtering gives 164 -> 150 (antecedent >=500) -> 83 (confidence >=0.30) -> 83 (lift >=1.20). The last filter removes no additional rules in this run; the criteria overlap. The final 83 have 12 one-condition, 49 two-condition and 22 three-condition antecedents. Counts and assumptions are saved in rule_search_audit.json. This audit does not rerun mining, alter criteria or use validation/test to improve the rule count.

## Neural demand regression with direct model explanations (2026-11-25)

Run `python -m part3_machine_learning.neural_explain` in the same capstone environment; then run the Part 3 test command. TensorFlow 2.21.0 and SHAP 0.51.0 were actually imported and executed. Native Windows TensorFlow reported CPU-only operation; the experiment uses two CPU compute threads and a small tf.data thread pool. No GPU/WSL installation was attempted.

The model is a feed-forward multilayer perceptron (MLP), not an LSTM: 21 inputs, Dense(64, ReLU), Dense(32, ReLU), Dense(1), 3,521 trainable parameters. This implements the course's neural-network demand-prediction option without pretending that adjacent rows across missing periods form a continuous sequence. It estimates same-hour traffic volume using the same input whitelist and chronological partitions as the supervised baselines. No accident target or future-weather information is introduced.

Input and target StandardScalers fit only the 25,329 training hours. Target scaling improves numerical optimization; predictions and final MAE are converted back to vehicles/hour. Adam learning rate=0.001, batch size=256, seed=42, maximum 80 epochs. Early stopping monitors 2017 validation MSE with patience 8 and min_delta=1e-4 (standardized target units), restoring the best qualifying weights. It ran 51 epochs and restored epoch 43. Within-epoch training MAE in learning_history.csv is accumulated as weights change; the final fixed-model training MAE is separately evaluated after weight restoration. The final test is never fitted, predicted or scored. Using validation for early stopping means these validation metrics are model-selection evidence, not unbiased final evaluation.

| Model | Training MAE | 2017 validation MAE | Validation R-squared |
|---|---:|---:|---:|
| MLP 64/32 | 281.69 | 268.78 | 0.9548 |
| Existing random forest | Not recomputed this round | 246.26 | 0.9583 |
| Existing linear regression | Not recomputed this round | 825.75 | 0.7106 |

The MLP improves over linear and dummy baselines but is worse than the forest on this validation MAE. Complexity alone does not justify replacing the forest. This train/validation comparison does not show the classic low-training/high-validation error gap, but it is not proof that no overfitting or distribution risk exists. No hyperparameter search or test-based tuning is claimed. Baseline files and their original manifests are unchanged; the new comparison is results/neural/regression_comparison.csv.

### What SHAP explains

Kernel SHAP calls this actual neural model; no surrogate model was substituted. Background: 64 fixed seeded training hours. Explained sample: 24 fixed seeded 2017 validation hours, ordered chronologically. Each uses 256 coalition samples with the explicit 21-feature l1 setting. Values are estimated output contributions in vehicles/hour, relative to the mean neural prediction on the background. The background mean prediction (3168.43) is different from the dummy model's training-target mean (3252.51).

For the first sampled case (2017-01-12 04:00), the model predicts about 893.95 vehicles/hour versus actual 843. The expected model output plus all 21 signed SHAP contributions reconstructs the prediction within 0.01 vehicles/hour; the maximum observed discrepancy across all 24 cases is about 0.00095. A positive contribution increases that model output relative to the background; a negative contribution decreases it. This is an explanation of a fitted function, not a causal effect of changing weather or an accident-risk explanation.

The sampled mean-absolute SHAP ranking is led by hour_cos, hour_sin and is_weekend. It describes only these 24 cases and this chosen background, not all traffic hours. Correlated hour/weekday coordinates, holiday flags and overlapping weather indicators can share attribution. Masking may construct unrealistic feature combinations; finite coalition/background sampling adds approximation uncertainty. The additive reconstruction check is an accounting check, not proof that explanations are causal or robust.

### Outputs and checks

- models/neural_regressor/: model.keras, training-only preprocessing.joblib and metadata.json.
- results/neural/: learning_history.csv, validation_predictions.csv, regression_comparison.csv, shap_contributions.csv, shap_input_cases.csv, shap_reconstruction.csv, shap_sample_importance.csv, neural_manifest.json and MLflow evidence.
- figures/: neural_learning_curve.png, neural_shap_sample_importance.png, neural_shap_local.png. The local figure shows ten largest absolute contributions; reconstruction uses all 21. All three final figures were visually checked.
- neural_explain.log: epoch milestones, fixed-model metrics, SHAP and save events; no internal progress print(). Third-party SHAP internals are filtered below WARNING for the reviewed run. The first run and its actual artifacts remain in local MLflow history; a reviewed rerun after logging/plot fixes is separately recorded, rather than rewriting an old run.

Checks cover all 8,713 validation predictions, independent metric recomputation, scaler fit counts/means, model hashes, 64 background dates in training, 24 explained dates in validation and additive SHAP reconstruction. The saved .keras model reload produced identical validation predictions within tolerance. The updated Part 3 suite has 12 tests. The new experiment records architecture, optimizer, fit/validation sizes, seed, metrics, epochs/artifacts and data hash in MLflow.

The implementation portion of the neural/explainability task is now present, pending user review and integration in the final Part 3 report. Understanding of neural architecture, early stopping and SHAP is tracked separately from execution success. Recommendation logic, final evaluation, deployment/monitoring and responsible-AI reporting remain pending. The user continues to handle staging/commit/push; this is new uncommitted Part 3 work, separate from the reviewed Part 1/2 submission artifacts.

## Bounded feature and architecture refinement (2026-11-25)

Run `python -m part3_machine_learning.refinement` for the bounded search and frozen-configuration fit, `python -m part3_machine_learning.verify_neural_reload` for fresh-process model/scaler checks, and `python -m part3_machine_learning.plot_refinement` to regenerate the comparison figure without training. Re-running the search creates 14 new inner-fold MLflow runs and one final configuration run; it does not overwrite the original neural experiment outputs.

### Search design and leakage control

This is a two-stage, manually bounded search, not exhaustive AutoML. Stage 1 compares five feature groups with the same 64/32 MLP. Stage 2 compares 32, 64/32 and 128/64 hidden-layer layouts on the selected feature group, reusing its existing 64/32 fold results. Seed=42, Adam=.001, batch=256 and a 40-epoch limit are fixed. Inner-fold early stopping uses patience=5, min_delta=1e-4 and that fold's later period. Each configuration is ranked by the equal mean of two fold MAEs, not by 2017 performance:

| Fold | Training hours | Assessment hours | Assessment period |
|---|---:|---:|---|
| Through 2014 -> 2015 | 13898 | 3593 | Available 2015 observations |
| Through 2015 -> 2016 | 17491 | 7838 | Available 2016 observations |

Before each fit, flagged imputed weather values in the cleaned hourly table are reset to missing. Monthly/fallback medians and temperature/cloud scaling are estimated anew on that fold's training rows, then applied to its assessment rows. Neural input/target scalers also fit only the fold training rows. This avoids carrying the earlier 2016-fitted imputation into an assessment of 2015. Source aggregation rules are fixed; no synthetic observations fill missing months. Fold-fitted states are logged in actual MLflow runs and exported in fold_preprocessing.json.

### Feature-group results

| Group | Inputs | Mean inner-fold MAE | Eligible main candidate |
|---|---:|---:|---|
| Calendar/time only | 6 | 310.82 | Diagnostic only |
| Calendar/time + numeric weather | 10 | 327.44 | Diagnostic only |
| Calendar/time + weather category flags | 17 | 338.23 | Yes, selected |
| Full except weekend flag | 20 | 359.51 | Yes |
| Full initial set | 21 | 367.40 | Yes |

Calendar/time includes both hour and weekday sin/cos, weekend and holiday flags. Eligibility was predeclared to keep time, holiday and categorical weather in this round's main candidates; it is an analyst design constraint, not a claim that a time-only regression is universally invalid. Task 1's existing classification/regression common 21-feature comparisons remain unchanged. A neural model under Task 3 need not intrinsically use every weather feature; diagnostic groups are still reported, including their better scores. This comparison shows that extra inputs do not automatically help under this model/budget, not that weather can never be useful.

The selected 17-feature group's mean fold MAEs are 434.19 (32 units), 338.23 (64/32) and 331.19 (128/64). These are differences within a small, single-seed comparison, not proof of significance or global optimality. The 32-unit model reaches the 40-epoch limit in both folds; its score may reflect the training budget. Feature/architecture interactions are not exhaustively searched.

### Frozen configuration and 2017 check

The 17-input, 128/64 configuration is frozen BEFORE the 2017 check. Its two best inner epochs are 26 and 23; their rounded median gives a fixed 25 epochs for fitting all pre-2017 training hours. The final fit does not use 2017 early stopping. The source imputation, z-score state and input/target scalers are refitted on the complete 25,329-hour training period.

The resulting 2017 MAE is 288.45 vehicles/hour, R-squared about 0.9499. This is worse than the initial MLP's 268.78 and the forest's 246.26. The bounded search therefore does NOT establish an improved model for this later period, and does not replace the forest candidate. Inner-period selection can fail to transfer to a later distribution; missing coverage, limited seeds/trials and fixed budgets are material limitations. The initial MLP also used 2017 early stopping, so the comparison protocols differ and all are selection/development evidence, not pristine test scores. No further search is performed to chase this 2017 result. The 2018 final test remains unscored.

### Fresh-process bundle check and outputs

`verify_neural_reload` starts a separate process and loads both model.keras and preprocessing.joblib from disk for the initial and refined models. For the initial model it reads the saved 21 prepared validation features. For the refined model it also rebuilds features from hourly observations using the serialized training-only imputation/scaling state. Both make 8,713 2017 predictions, checked against the saved reference predictions at atol=1e-6, rtol=0. Maximum differences are about 5e-11. This supplements the earlier same-process model-only reload check; it is still not an HTTP API/load/stress test or proof of prediction accuracy.

- models/refined_neural/: model.keras, preprocessing.joblib, metadata.json with version/hashes and the input schema.
- results/refinement/: feature_ablation.csv, architecture_comparison.csv, inner_fold_results.csv, fold_preprocessing.json, validation_predictions.csv, refinement_manifest.json, mlflow_evidence.json and reload_evidence.json.
- results/neural/reload_evidence.json: fresh-process evidence for the original MLP.
- figures/refinement_comparison.png: mean inner-fold MAEs with diagnostic-only groups distinguished; visually checked.
- refinement.log, verify_neural_reload.log and plot_refinement.log: actual run and verification trails.

Four additional tests cover flag restoration/fold-specific imputation, pre-2017 selection and saved configuration, validation metrics/scaler fit counts, and fresh-process reload evidence. The Part 3 test suite now has 16 tests. The best model has not been selected using 2018 outcomes; feature selection and architecture tuning remain limited to the documented search scope.

SHAP artifacts for the original 21-input MLP remain labelled as explanations of that model; they are not silently reattributed to the refined 17-input MLP or the forest. If a different final model is adopted, explanations must be regenerated for that actual model. Recommendation/deployment work will use the selected model's own frozen feature contract. No Part 1/2 PDFs or submission ZIP were changed.

## Constrained departure-window recommendation prototype (2026-11-25)

Run the saved examples with `python -m part3_machine_learning.recommend_examples`. This freezes a self-contained bundle from the existing random_forest_regressor candidate and the 25,329 training support hours; it does not fit another model. The candidate is still based on 2017 development evidence, not final-test approval. Bundle metadata records the source model run ID/hash, data hash, frozen imputation/scaling cutoff, input whitelist and bundle SHA-256.

For a user request, run:

```powershell
python -m part3_machine_learning.recommend --request part3_machine_learning/results/recommendation/requests/weekday_clear.json
```

The CLI loads model and preprocessing/support data from models/recommendation/bundle.joblib, rather than relying on training-process objects or reading later observed outcomes. --build-bundle rebuilds the bundle only. Answers are UTF-8 JSON on stdout; named-logger status/errors go to stderr and recommend.log. Invalid inputs return code 2 with clear ERROR and no traceback. Insufficient support is a valid result with no recommendation, not a zero-traffic estimate.

### Input and selection procedure

Required fields: date (YYYY-MM-DD in the source calendar), start_hour (inclusive), end_hour (exclusive), window_hours, a nonempty weather category list, and an explicit integer is_holiday=0/1. Optional numerical weather: temp_c, clouds_all, rain_1h, snow_1h. Missing numeric values use the frozen training-month median/fallback and each assumption is returned; these are not measured or forecast weather. No external holiday calendar or timezone conversion is performed. All provided weather is assumed constant across the candidate hours.

1. Validate the permitted same-day hour interval and requested window length. A two-hour window means a flexible period in which to depart, not a two-hour journey.
2. Construct the original forest's 21 inputs from date/time, supplied holiday/weather and frozen scaling; round prepared values to eight decimals to match the training CSV convention. No actual current traffic or derived proxy label enters the model.
3. Predict each candidate hour in the permitted interval.
4. Count a training-only support cohort for each hour: same hour, weekday/weekend type, holiday flag and EXACT multi-hot weather label set. Numerical weather values are not matched. Support is an observational count, not a calibrated uncertainty estimate or validation of the joint feature distribution.
5. Consider every contiguous window of the requested length for which each hour passes the support floor. Rank by mean estimated hourly traffic; ties prefer the earlier start. Missing support is not bridged across gaps. Never recommend outside the permitted interval.
6. If no supported window exists, return insufficient_support and best_window=null. If a numeric weather value is outside the training min/max range, return outside_training_weather_range and no best window, while retaining estimates for inspection.

The default per-hour support floor of 30 is an analyst-chosen prototype setting, not a course requirement or a statistical reliability guarantee. It can be explicitly changed with min_support_hours. It is a different check from association mining's 500 antecedent hours: here cohorts are conditioned on a specific hour/day type/holiday/exact weather set. Neither threshold proves safety, causal validity or forecast performance.

### Controlled examples

All use a permitted interval 08:00-19:00, end exclusive, and a two-hour departure window. Historical calendar dates are used only to construct scenarios; the system does NOT read actual 2017 traffic outcomes. Clear examples specify 20 C, clouds 1%, no rain/snow. Rain specifies 15 C, clouds 90%, rain 1 mm, snow 0. These are different supplied contexts, not a causal intervention comparison.

| Scenario | Suggested window | Mean estimated traffic | Minimum per-hour training support |
|---|---|---:|---:|
| Weekday Clear (2017-06-05) | 10:00-12:00 | 4425.21 vehicles/hour | 205 |
| Weekend Clear (2017-06-10) | 08:00-10:00 | 2842.35 | 90 |
| Weekday Rain (2017-06-05) | 10:00-12:00 | 4304.86 | 45 |
| Squall scenario | No recommendation | Not promoted as a window | Insufficient |

Lower estimated vehicle volume does not guarantee lower journey time, no congestion or greater safety: speed, capacity, incidents and actual travel-time targets are absent. Differences between nearby windows may be smaller than model errors; the ranking is illustrative and does not constitute calibrated confidence in a unique best time. It is not a live traffic or future weather service, and no route search is performed for this single corridor.

### Outputs and acceptance

- recommend.py: request validation, serving feature construction, support checks, constrained ranking and CLI.
- recommend_examples.py: four recorded scenarios and one actual FINISHED MLflow application run. Logged counts are application-demo counts, not model accuracy or realized travel savings.
- models/recommendation/: compressed bundle.joblib and metadata.json (source model/data hashes and input/support contract).
- results/recommendation/: each request/result, recommendation_manifest.json, mlflow_evidence.json and fresh-process cli_acceptance.json.
- figures/recommendation_scenarios.png: conditional hourly estimates and selected windows for the three supported scenarios; visually inspected.

Four new recommendation tests plus the prior 16 Part 3 tests pass: interval/duration constraints, no bridging unsupported gaps, minimum-mean choice, invalid numeric/date/category inputs, exact multi-weather support, explicit median assumptions, out-of-range handling and saved-example reproducibility. Five fresh-process CLI cases verify the four scenario outputs and a clear invalid-input error; predictions agree within tolerance after loading the frozen bundle. The 2018 final test remains unscored. No earlier reports or submitted ZIPs were changed.

Task 5 now has an initial conditional travel-timing prototype. Next work is serving/monitoring simulation and final candidate freeze/evaluation, followed by responsible-AI reporting and integrated delivery. Do not label this prototype as a validated real-world travel advisory system. Stage/commit/push remain with the user; the accumulating Part 3 work should be split into functional commits.


## Local API and monitoring mock (2026-11-25)

Use the capstone Python environment. `python -m part3_machine_learning.serve` starts a Flask development mock on 127.0.0.1:8000. GET /health returns readiness and the candidate model run ID; POST /predict accepts the same JSON schema as recommend.py and returns hourly predictions and constrained recommendation status. Invalid scenario inputs return HTTP 400. Insufficient support remains HTTP 200 with best_window=null. The app loads the hash-checked saved bundle once at startup, never fits a model, and logs request status, latency and model run ID. This is a local development mock, not a production deployment or final-model approval. No authentication, production worker configuration, concurrency or load testing is claimed.

`python -W error -m part3_machine_learning.verify_api` starts a temporary real loopback HTTP server, loads the bundle from disk, checks health, four saved scenarios and three invalid inputs, compares every hourly prediction to saved evidence, checks that the bundle hash is unchanged, and stops the server. Evidence: results/serving/http_acceptance.json and verify_api.log. This is serving-contract verification, not accuracy evaluation.

`python -W error -m part3_machine_learning.monitoring` saves results/monitoring/simulation.json and monitoring.log. The monitor compares a batch's source-calendar hour distribution with the 25,329 training support hours using total variation distance: half the sum of absolute differences across 24 hourly proportions. TV=0 means equal proportions; TV=1 means disjoint support. An analyst prototype threshold of TV>0.20 triggers ALERT; fewer than 100 batch observations also triggers ALERT for insufficient evidence. These settings are not statistically calibrated. Empty/invalid hour batches are rejected. Reference replay is an intentionally identical synthetic PASS control; 1,000 repeated 08:00 hours are an intentionally shifted ALERT control. Five observations also demonstrate the small-batch alert. Saved batches make these checks reproducible.

Monitoring is an offline simulation of one feature only, not wired to collect live API requests. It does not monitor all weather/joint distributions, prediction error or operational availability. PASS only clears these checks. ALERT requests investigation of sampling, calendar mix, schema and upstream data; it does not establish deteriorating accuracy or automatically authorize retraining. Actual error monitoring requires later observed traffic aligned with predictions. The 2018 final test remains unscored. No model, Part 1/2 report or submission ZIP is rebuilt. Stage/commit/push remain user-owned.

The optional request field `weather_source` labels caller-supplied conditions as `forecast` or `assumption`. Omitted sources display as unspecified. The human-readable `recommendation` gives the date, candidate hours, source, categories, temperature, cloud cover, rain and snow before the result. Omitted numeric fields display the actual median value used with an imputation label. The forecast label describes the caller's input; the service does not retrieve a forecast.

## Responsible and sustainable AI

The [bias and fairness report](bias_and_fairness_report.md) covers sampling, proxy-label limitations, frozen holdout group errors, operational oversight and measured modelling trade-offs.

## Frozen 2018 holdout evaluation (2026-11-25)

The two preselected forests were evaluated on 6,533 observed hours from January 1 through September 30, using the 21 prepared same-hour features and historical weather. The selection basis and artifact hashes are in results/final_evaluation_plan.json. The evaluation leaves training and the frozen plan intact. The entry point is python -m part3_machine_learning.final_evaluation. It skips inference when results/final_2018 already exists. Use the saved predictions.csv and metrics.json there for further analysis.

Volume MAE was 241.54 vehicles/hour (2017 validation 246.26), with R² 0.9600. Weekday MAE was 223.34 (4,671 hours) and weekend MAE 287.19 (1,862 hours). Proxy classification yielded F1 0.9208, precision 0.8594, recall 0.9917 and ROC AUC 0.9964. There were 721 true positives, 6 false negatives, 118 false positives and 5,688 true negatives. The risky-weather flag participates in the proxy definition, making positives structurally absent in the other group. Undefined precision and recall are recorded as null. These results evaluate historical observed-weather scenarios on one corridor. They do not validate advance weather forecasts or accident probabilities. All targets and subgroup keys were matched to the source by timestamp and all reported metrics independently recalculated from the saved predictions.
