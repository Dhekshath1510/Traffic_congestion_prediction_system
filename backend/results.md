# Results Section — Research Paper

## 4. Results

### 4.1 Experimental Setup

Describe the experimental configuration used to evaluate the proposed traffic congestion prediction framework.

Include:

- Dataset used
- Number of traffic sensors
- Sampling interval
- Prediction horizon
- Historical input window
- Training/validation/test split
- Input variables
- Traffic variables used for prediction
- Congestion classification methodology
- Hardware/software environment
- Evaluation metrics

**Important:** Keep this subsection strictly descriptive. Do not interpret the significance of the results here.

---

### 4.2 Dataset and Traffic Characteristics

Present the characteristics of the processed traffic dataset.

Report:

- Number of sensors
- Number of observations
- Temporal resolution
- Training samples
- Validation samples
- Test samples
- Speed statistics
- Presence of very-low/near-zero speed observations
- Temporal variation in traffic speed
- Sensor-level variability

Suggested table:

**Table I — Dataset Characteristics**

| Property | Value |
|---|---:|
| Number of sensors | 207 |
| Sampling interval | 5 min |
| Historical sequence length | 12 |
| Training samples | 23,976 |
| Validation samples | 5,126 |
| Test samples | 5,128 |
| Input representation | 12 × 207 |
| Prediction target | Future traffic speed |

Suggested figures:

- Speed distribution
- Temporal speed trend
- Sensor-level speed heatmap

---

### 4.3 Traffic Speed Prediction Results

Evaluate the three forecasting approaches:

1. Persistence
2. Lightweight XGBoost
3. GRU

**Do not mention multiple versions/configurations of GRU. Refer to the model simply as GRU.**

Report results separately for validation and test datasets.

Suggested table:

**Table II — Traffic Speed Prediction Performance**

| Model | Dataset | MAE | RMSE | sMAPE |
|---|---|---:|---:|---:|
| Persistence | Validation | 4.0593 | 9.9222 | 11.18% |
| Lightweight XGBoost | Validation | 4.6567 | 9.5430 | 24.93% |
| GRU | Validation | 4.0459 | 9.8186 | 25.15% |
| Persistence | Test | 3.8795 | 9.5040 | 11.02% |
| Lightweight XGBoost | Test | 4.6272 | 9.2205 | 32.77% |
| GRU | Test | 3.9040 | 9.4304 | 33.01% |

Present:

- Actual versus predicted speed
- Model-wise metric comparison
- Error distribution
- Prediction examples across different traffic conditions

**Do not interpret why one model is better in this subsection.** Interpretation belongs in the Discussion/Conclusion portions of the paper.

---

### 4.4 Sensor-Specific Congestion Reference Speed

Describe the derivation of sensor-specific reference speeds.

The reference speed is calculated using the **85th percentile (P85)** of training-period traffic speeds for each sensor.

Report:

- Minimum reference speed
- Maximum reference speed
- Mean reference speed
- Median reference speed

Current values:

| Statistic | Reference Speed |
|---|---:|
| Minimum | 36.22 |
| Maximum | 70.00 |
| Mean | 65.57 |
| Median | 66.38 |

State the congestion index formulation:

\[
CI = \operatorname{clip}
\left(
1-\frac{v_{future}}{v_{reference}},
0,1
\right)
\]

where:

- \(v_{future}\) is the predicted future speed.
- \(v_{reference}\) is the sensor-specific P85 reference speed.

---

### 4.5 Congestion Classification Results

Describe the conversion of predicted traffic speed into three congestion states:

| Congestion Class | Congestion Index |
|---|---:|
| LOW | \(CI < 0.10\) |
| MODERATE | \(0.10 \leq CI < 0.30\) |
| SEVERE | \(CI \geq 0.30\) |

Report class distributions for each dataset.

**Table III — Congestion Class Distribution**

| Dataset | LOW | MODERATE | SEVERE |
|---|---:|---:|---:|
| Training | 68.30% | 14.18% | 17.52% |
| Validation | 67.52% | 12.85% | 19.63% |
| Test | 63.34% | 12.97% | 23.69% |

Include:

- Class distribution figure
- Congestion-index distribution
- Example sensor/time-period classifications

---

### 4.6 XGBoost Congestion Classification Performance

Present the performance of the shared multiclass XGBoost congestion classifier.

The classifier predicts:

- LOW
- MODERATE
- SEVERE

Report validation and test performance.

**Table IV — Congestion Classification Performance**

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.6762 | 0.6533 |
| Macro Precision | 0.5939 | 0.5950 |
| Macro Recall | 0.6745 | 0.6653 |
| Macro F1 | 0.6111 | 0.6040 |
| Weighted F1 | 0.6962 | 0.6706 |

---

### 4.7 Class-Level Classification Performance

Provide the detailed test-set classification results.

**Table V — Test-Set Class-Level Performance**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| LOW | 0.88 | 0.61 | 0.72 | 672,379 |
| MODERATE | 0.32 | 0.57 | 0.41 | 137,673 |
| SEVERE | 0.58 | 0.81 | 0.68 | 251,444 |

Include the normalized confusion matrix.

Discuss only the **observed numerical results** in this section:

- Per-class precision
- Per-class recall
- Per-class F1
- Confusion between classes

Avoid causal explanations such as *"this occurs because..."*. Those belong in the Discussion.

---

### 4.8 Explainability Results

Evaluate the explainability component using TreeSHAP.

Report:

- Number of evaluation samples
- Number of features
- Global feature importance
- Class-specific feature importance
- Representative prediction explanation

Current experiment:

- Evaluation samples: 1,061,496
- Features: 25
- Explainer: TreeSHAP
- Classes: LOW, MODERATE, SEVERE

**Table VI — Global SHAP Feature Importance**

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | low_speed_fraction | 0.256325 |
| 2 | sensor_index | 0.198116 |
| 3 | high_speed_fraction | 0.162975 |
| 4 | hour | 0.082578 |
| 5 | current_speed_mean | 0.072380 |
| 6 | recent_speed_std | 0.071810 |
| 7 | recent_speed_mean | 0.049941 |
| 8 | hour_cos | 0.043115 |
| 9 | current_speed_std | 0.032728 |
| 10 | moderate_speed_fraction | 0.030332 |

Include:

- Global SHAP importance plot
- Class-specific SHAP importance
- Representative local explanation

For the `sensor_index` feature, describe it accurately as a **sensor-identity feature capturing sensor-specific distributional differences**. Do not describe it as a spatial topology or graph feature.

---

### 4.9 Explainability and Confidence Evaluation

Report the reliability of model predictions and explanations.

**Table VII — Confidence and Calibration Results**

| Metric | Result |
|---|---:|
| Mean confidence | 0.6069 |
| Median confidence | 0.5795 |
| Accuracy | 0.6533 |
| High-confidence threshold | 0.80 |
| Low-confidence threshold | 0.50 |
| High-confidence fraction | 12.76% |
| Low-confidence fraction | 30.99% |
| High-confidence accuracy | 95.49% |
| Low-confidence accuracy | 47.34% |
| Expected Calibration Error | 0.0485 |
| Multiclass Brier score | 0.4590 |

Include:

- Confidence distribution
- Correct vs. incorrect confidence
- Reliability/calibration diagram
- Confidence by congestion class

---

### 4.10 Severe-Congestion Detection Performance

Because severe congestion is operationally important for a future traffic-management/speed-assistance system, report the severe class explicitly.

Report:

- Severe precision
- Severe recall
- Severe F1-score
- Severe support
- Severe-class confidence
- Severe-class confusion with LOW and MODERATE

Current test results:

- Precision: 0.58
- Recall: 0.81
- F1-score: 0.68
- Support: 251,444

Also report the severe-class mean confidence:

- Mean confidence: 0.7278

This subsection should remain results-oriented and should not claim that the model is already suitable for safety-critical deployment.

---

### 4.11 Sensor-Level Performance Variation

Report the variation in classification performance across sensors.

Include:

**Table VIII — Lowest-Performing Sensors**

| Sensor | Accuracy | Error Rate | Severe Recall |
|---:|---:|---:|---:|
| 21 | 0.3974 | 0.6026 | 0.7520 |
| 14 | 0.4052 | 0.5948 | 0.8181 |
| 10 | 0.4095 | 0.5905 | 0.7555 |
| 50 | 0.4216 | 0.5784 | 0.7590 |
| 56 | 0.4440 | 0.5560 | 0.4372 |
| 126 | 0.4499 | 0.5501 | 0.5669 |
| 17 | 0.4507 | 0.5493 | 0.8047 |
| 71 | 0.4550 | 0.5450 | 0.6725 |
| 22 | 0.4668 | 0.5332 | 0.7364 |
| 30 | 0.4748 | 0.5252 | 0.7445 |

Include a sensor-level error distribution figure.

Do not infer physical causes of poor sensor performance unless supported by the experimental data.

---

### 4.12 Error Analysis

Analyze the major classification error categories:

- LOW → MODERATE
- LOW → SEVERE
- MODERATE → LOW
- MODERATE → SEVERE
- SEVERE → LOW
- SEVERE → MODERATE

Use the generated SHAP error-analysis files:

- `shap_error_low_to_moderate.csv`
- `shap_error_severe_to_moderate.csv`
- `shap_error_moderate_to_low.csv`
- `shap_error_moderate_to_severe.csv`

For each major error category report:

1. Number of observations
2. Percentage of observations
3. Model confidence
4. Dominant SHAP features
5. Representative examples

This subsection should identify **what the model got wrong and what features were associated with those errors**, without claiming causal mechanisms.

---

### 4.13 Overall Experimental Summary

Provide one consolidated table containing the principal experimental results.

**Table IX — Overall System Results**

| Component | Metric | Result |
|---|---|---:|
| Persistence | Test MAE | 3.8795 |
| Persistence | Test RMSE | 9.5040 |
| Persistence | Test sMAPE | 11.02% |
| Lightweight XGBoost | Test MAE | 4.6272 |
| Lightweight XGBoost | Test RMSE | 9.2205 |
| Lightweight XGBoost | Test sMAPE | 32.77% |
| GRU | Test MAE | 3.9040 |
| GRU | Test RMSE | 9.4304 |
| GRU | Test sMAPE | 33.01% |
| Congestion Classifier | Test Accuracy | 65.33% |
| Congestion Classifier | Macro F1 | 0.6040 |
| Congestion Classifier | Severe Recall | 0.81 |
| XAI | ECE | 0.0485 |
| XAI | High-confidence Accuracy | 95.49% |

---

## 4.14 Results Narrative Guidelines

The Results section should:

- Report numerical findings.
- Refer to tables and figures.
- Describe observed performance differences.
- Report class-level performance.
- Report explainability findings.
- Report confidence/calibration.
- Report sensor-level variation.
- Report error patterns.

The Results section should **not**:

- Introduce new models.
- Introduce GNNs or other untested architectures.
- Discuss a second/versioned GRU.
- Claim causality from SHAP.
- Claim production/safety readiness.
- Make unsupported claims about real-world traffic behavior.
- Repeat the literature review.
- Present future work.
- Merge Results with Conclusion.
- Present limitations as conclusions unless they are directly observed experimental findings.

---

# Figure Plan

## Figure 1
**Traffic speed distribution across the dataset**

Purpose:
Show the overall distribution and variability of observed traffic speeds.

## Figure 2
**Temporal variation of traffic speed**

Purpose:
Show temporal traffic behavior across the observation period.

## Figure 3
**Sensor-level traffic speed heatmap**

Purpose:
Visualize variation across the 207 sensors and time.

## Figure 4
**Comparison of traffic speed prediction models**

Models:

- Persistence
- Lightweight XGBoost
- GRU

Metrics:

- MAE
- RMSE
- sMAPE

## Figure 5
**Actual versus predicted traffic speed**

Compare representative predictions produced by the evaluated forecasting models.

## Figure 6
**Congestion class distribution**

Classes:

- LOW
- MODERATE
- SEVERE

## Figure 7
**Congestion classification confusion matrix**

Show validation/test classification behavior.

## Figure 8
**Global SHAP feature importance**

Show the most influential features using mean absolute SHAP values.

## Figure 9
**Class-specific SHAP feature importance**

Compare influential features for:

- LOW
- MODERATE
- SEVERE

## Figure 10
**Prediction confidence analysis**

Compare confidence distributions for correct and incorrect predictions.

## Figure 11
**Calibration reliability diagram**

Show predicted confidence against empirical accuracy.

## Figure 12
**Sensor-level classification error distribution**

Show variation in classification performance across sensors.

---

# Recommended Results Writing Pattern

For every experiment use:

### Experiment
What was evaluated?

### Quantitative Result
What numerical result was obtained?

### Table/Figure Reference
Where is the result shown?

### Direct Observation
What does the result show numerically?

Avoid interpretation beyond the evidence.

Example structure:

> Table II presents the traffic speed prediction performance of Persistence, Lightweight XGBoost, and GRU on the validation and test datasets. On the test set, Persistence obtained an MAE of 3.8795 and RMSE of 9.5040, while Lightweight XGBoost obtained an MAE of 4.6272 and RMSE of 9.2205. GRU obtained an MAE of 3.9040 and RMSE of 9.4304. The corresponding sMAPE values were 11.02%, 32.77%, and 33.01%, respectively.

Then move to the next result.

**Do not add the interpretation/conclusion immediately after this paragraph.**

---

# Important Terminology Rules

Use these exact model names throughout the paper:

- **Persistence**
- **Lightweight XGBoost**
- **GRU**
- **XGBoost Congestion Classifier**
- **TreeSHAP**

Do NOT use:

- GRU v1
- GRU v2
- improved GRU
- enhanced GRU
- residual GRU
- optimized GRU

unless the final paper explicitly defines those as separate experimental models.

The forecasting comparison is:

\[
\boxed{\text{Persistence} \rightarrow \text{Lightweight XGBoost} \rightarrow \text{GRU}}
\]

The congestion pipeline is:

\[
\boxed{
\text{Future Speed}
\rightarrow
\text{Sensor-Specific Reference Speed}
\rightarrow
CI
\rightarrow
\text{LOW/MODERATE/SEVERE}
}
\]

The explainability pipeline is:

\[
\boxed{
\text{XGBoost Congestion Classifier}
\rightarrow
\text{TreeSHAP}
\rightarrow
\text{Global + Local Explanations}
}
\]

---

# Separation From Discussion and Conclusion

## Results

Answer:

> **What did the experiments produce?**

Use:
- metrics
- tables
- figures
- distributions
- confusion matrices
- SHAP values
- confidence
- calibration
- error statistics

## Discussion

Answer:

> **Why are these results important, and what do they mean?**

Discuss:
- why the models behaved differently
- why moderate congestion is difficult
- implications of severe-congestion recall
- significance of sensor variability
- interpretation of SHAP patterns
- comparison with prior research
- practical implications
- limitations

## Conclusion

Answer:

> **What is the final takeaway from the entire study?**

Summarize:
- principal contribution
- major findings
- research significance
- practical significance
- limitations at a high level
- future research direction

**Do not combine Results and Conclusion into one section.**
**Do not place future-work claims inside Results.**
**Do not use the Results section to argue that the proposed system is superior unless the experimental evidence directly supports that statement.**

---

# Final Results Section Checklist

- [ ] Dataset characteristics reported
- [ ] Train/validation/test setup reported
- [ ] Persistence evaluated
- [ ] Lightweight XGBoost evaluated
- [ ] GRU evaluated
- [ ] No second/versioned GRU terminology
- [ ] MAE reported
- [ ] RMSE reported
- [ ] sMAPE reported
- [ ] Sensor-specific P85 reference speed reported
- [ ] Congestion-index equation reported
- [ ] LOW/MODERATE/SEVERE thresholds reported
- [ ] Class distribution reported
- [ ] XGBoost classifier accuracy reported
- [ ] Macro precision/recall/F1 reported
- [ ] Class-level metrics reported
- [ ] Confusion matrix included
- [ ] SHAP global importance reported
- [ ] SHAP class-specific importance reported
- [ ] Confidence results reported
- [ ] Calibration results reported
- [ ] Severe-class performance reported
- [ ] Sensor-level variation reported
- [ ] Error analysis reported
- [ ] Tables and figures referenced
- [ ] Results separated from Discussion
- [ ] Results separated from Conclusion
- [ ] No unsupported causal claims
- [ ] No new models introduced
- [ ] No GNN introduced
- [ ] No future-work discussion in Results