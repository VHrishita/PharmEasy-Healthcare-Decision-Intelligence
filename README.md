Domain: Healthcare

# PharmEasy Healthcare Decision Intelligence

## How to run

### 1. Clone or open the repository

Open the project folder in VS Code.

git clone https://github.com/VHrishita/PharmEasy-Healthcare-Decision-Intelligence.git
cd PharmEasy-Healthcare-Decision-Intelligence

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python generate_data.py
```

Expected output:

```text
800 10 170 e26b46e82f1a1af44dcabc133050802e
```

### 5. Run the four parts in order

```bash
python part1.py
python part2.py
python part3.py
python part4.py
```

All generated results are saved in the `outputs/` directory.

---

## Part 1 findings

### R1.1 Estimation and hypothesis testing

The estimated mean care-plan cost is 8.9382 thousand INR, compared with the planning assumption of 8.0 thousand INR. The calculated z-statistic is 2.8930, which is greater than the critical value of 1.96. Therefore, the null hypothesis is rejected at the 5% significance level.

A Type I error in this context would mean concluding that the actual average care-plan cost differs from INR 8,000 when it is actually INR 8,000.

The estimated difference is 0.9382 thousand INR, equivalent to INR 938.20. Although statistically significant, this difference should also be considered in terms of its financial impact across a large number of patients.

The 95% confidence interval is 8.3026 to 9.5738 thousand INR.

### R1.2 Sample size requirement

The required sample size is 1,293 records, whereas the available dataset contains only 800 records. Therefore, the current dataset is insufficient to achieve the desired margin of error of INR 500 at 95% confidence.

### R1.3 Gradient descent

The gradient descent slope gap is 0.0000. Gradient descent reached the closed-form regression solution because the cost function is a convex quadratic function, and repeated full-batch updates converged to the same line.

### R1.4 Residual analysis

The raw regression model has a residual standard deviation ratio of 3.7525, indicating a funnel-shaped pattern where residual spread increases with fitted values.

After applying the log-log transformation, the ratio reduces to 1.0396. This indicates that the transformation substantially reduces the heteroscedasticity observed in the raw model.

### R1.5 Multiple regression

The predictors with coefficients having an absolute value greater than 0.05, excluding the constant, are:

- ln(X4): 1.1576
- X2: 0.0940
- X3: 0.1067
- X6: 0.1518

The adjusted R² of the multiple regression model is 0.7403, compared with 0.6561 for the log-log model. This indicates that the additional predictors improve the model's explanatory performance. The coefficient threshold is a mechanical selection rule and does not represent a ranking of predictor importance.

---

## Part 2 findings

### R2.1 Learning-rate comparison

The training losses after 200 updates are:

| Learning rate | Training loss |
|---|---:|
| 1.0 | 0.3703 |
| 0.1 | 0.3776 |
| 0.01 | 0.5091 |

A very small learning rate takes longer to converge, while a larger learning rate can reach a lower loss more quickly in this experiment.

### R2.2 Important predictors

The three predictors with the largest absolute logistic regression weights are:

- X3: 0.9013
- X1: -0.5554
- X2: 0.5505

The largest odds ratio belongs to X3, with a value of 2.4629. This means that a one-standard-deviation increase in copay burden ratio multiplies the estimated odds of a no-show by approximately 2.46, holding the other predictors constant. This is an association, not evidence of causation.

### R2.3 Threshold comparison

The blind model, which predicts no-show for nobody, has an accuracy of 0.7875. At a threshold of 0.5, logistic regression achieves an accuracy of 0.8042.

Accuracy alone is not sufficient because only 51 of the 240 test records are positive. A model can achieve apparently high accuracy while missing many actual no-shows.

Lowering the threshold from 0.5 to 0.3 changes the metrics as follows:

- Recall increases from 0.4706 to 0.7451.
- Precision decreases from 0.5455 to 0.4634.

This means more actual no-shows are identified, but more patients who would have attended are also flagged.

### R2.4 Logistic regression versus KNN

At a threshold of 0.3, logistic regression achieves an F1 score of 0.5714 and recall of 0.7451.

The KNN model achieves an F1 score of 0.3696 and recall of 0.3333.

Logistic regression performs better than KNN on both measures for this dataset and split.

### R2.5 Cost-based threshold selection

The selected logistic regression threshold is 0.15.

| Evaluation | Test cost |
|---|---:|
| Selected threshold: 0.15 | INR 6,050 |
| Threshold: 0.5 | INR 11,800 |
| Flag nobody | INR 20,400 |

The threshold was selected using training data rather than test data to avoid using the test set to make model decisions and to preserve an independent evaluation.

Based on the cost comparison, the logistic regression model with a threshold of 0.15 is the preferred alerting approach.

---

## Part 3 findings

### R3.1 K-means and elbow method

The successive WCSS reductions are:

| Comparison | WCSS reduction |
|---|---:|
| K=2 to K=3 | 659.8282 |
| K=3 to K=4 | 108.0739 |
| K=4 to K=5 | 171.0575 |
| K=5 to K=6 | 45.5466 |

The elbow criterion selects K=3 because the reduction from K=2 to K=3 is 659.8282, and the next reduction of 108.0739 is less than half of that value.

The reductions do not decrease steadily because the pinned starting centroids provide a reproducible solution but do not necessarily produce the global optimum for every K.

### R3.2 Patient segments

The three segments are ordered by increasing average copay burden ratio (X3).

**Segment A — Established and consistent users**

- Records: 322
- Average registration period: 47.5028 months
- Average missed visits: 1.9503
- Outcome rate: 1.86%
- Mean care-plan value: 3.5976 thousand INR

These users have longer registration periods, fewer missed visits and higher adherence scores. Routine service reminders and convenient refill scheduling can help maintain engagement.

**Segment B — High-friction, higher-bill users**

- Records: 255
- Average registration period: 28.6071 months
- Average missed visits: 6.0706
- Outcome rate: 23.53%
- Mean care-plan value: 17.1248 thousand INR

These users have the highest average missed-visit count and the highest mean care-plan value. Appointment confirmations and clearer billing communication could help improve service engagement.

**Segment C — High-copay, follow-up-risk users**

- Records: 223
- Average registration period: 14.3892 months
- Average missed visits: 4.2287
- Outcome rate: 46.64%
- Mean care-plan value: 7.2883 thousand INR

This segment has the highest copay burden ratio and the highest no-show outcome rate of 46.64%. Clear cost explanations, payment information and proactive appointment reminders could help improve engagement.

Segment C has the highest observed outcome rate and deserves particular attention from the scheduling operations team.

### R3.3 Hierarchical clustering

The clusters obtained at the dendrogram cut are:

**Single linkage — 4 clusters**

- E1
- E2+E3+E6
- E4
- E5

**Complete linkage — 2 clusters**

- E1+E4+E5
- E2+E3+E6

Single linkage uses the minimum distance between members of two clusters, so it can connect groups through close individual pairs. Complete linkage uses the maximum distance, requiring clusters to remain closer across all member pairs, which produces a different grouping for this table.

### R3.4 Principal Component Analysis

The mean vector after standardisation is approximately zero for all five predictors. The data is centred before calculating the covariance matrix and performing eigen-decomposition.

The PC1 loadings are:

| Predictor | PC1 loading |
|---|---:|
| X1 | +0.5317 |
| X2 | -0.4100 |
| X3 | -0.5187 |
| X4 | -0.2916 |
| X5 | +0.4418 |

PC1 is mainly driven by X1, X3 and X5, with X2 and X4 also contributing. The positive loadings of X1 and X5 contrast with the negative loadings of X2, X3 and X4.

The first two components explain 75.93% of the total variance. Three components are required to retain at least 80%, explaining 86.16%.

The mean PC1 scores of the segments are:

- Segment A: +1.7773
- Segment B: -0.7831
- Segment C: -1.6709

These values show that Segment A is separated from Segments B and C along the first principal component, while Segment C has the lowest mean PC1 score.

---

## Part 4 findings

### R4.1 Decision tree split

The best split is S4 (X1 ≤ 24), with an information gain of 0.1252 and a Gini gain of 0.0606. Both measures identify S4 as the best among the four candidate splits because it produces the greatest reduction in impurity.

Growing a decision tree until every leaf is pure can cause overfitting because the tree may learn patterns specific to the training records rather than general patterns. A maximum-depth restriction is one way to prevent a tree from growing too deep.

### R4.2 Model comparison and voting

The five models were evaluated on the test set.

| Model | F1 | Recall |
|---|---:|---:|
| Logistic regression | 0.5714 | 0.7451 |
| KNN | 0.3696 | 0.3333 |
| Decision stump | 0.5532 | 0.7647 |
| Hard voting | 0.5344 | 0.6863 |
| Soft voting | 0.5507 | 0.7451 |

Logistic regression is the preferred structured-data model because it achieves the highest F1 score of 0.5714.

Combining models through hard and soft voting did not improve F1 compared with logistic regression alone.

### R4.3 AdaBoost

After round 1 normalisation, points 3 and 7 are misclassified. Each receives a weight of 0.25, giving a combined weight of 0.50.

In round 2, h2 receives greater importance because its weighted error is lower at 0.1667 compared with 0.2500 for h1. Consequently, its alpha value of 0.8047 is greater than h1's alpha of 0.5493.

The final ensemble accuracy is 0.7500.

### R4.4 Neural network architecture

**Model A — Convolutional neural network**

The input is a 28 × 28 grayscale image with one channel.

| Layer | Output shape | Parameters |
|---|---|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv1 (5 × 5, 6 filters) | 28 × 28 × 6 | 156 |
| Max Pool1 (2 × 2) | 14 × 14 × 6 | 0 |
| Conv2 (5 × 5, 16 filters) | 10 × 10 × 16 | 2,416 |
| Max Pool2 (2 × 2) | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense1 | 32 | 12,832 |
| Dense2 (output) | 10 | 330 |

The total parameters for the fully connected network (MLP) are 13,002, while the convolutional network (CNN) has 15,734 parameters.

For the larger 128 × 128 × 3 input:

- Fully connected layer parameters: 50,332,672
- Convolutional layer parameters: 1,792

Convolution uses shared filters across spatial locations, which greatly reduces the number of parameters compared with connecting every input value to every neuron.

**Model B — Recurrent neural network**

The RNN uses a 64-dimensional input vector, a 128-dimensional hidden state and 16 output classes.

| Weight matrix | Shape |
|---|---|
| W_xh | 128 × 64 |
| W_hh | 128 × 128 |
| W_hy | 16 × 128 |

The total RNN parameter count, including biases, is 26,768.

The parameter count does not increase with sequence length because the same weight matrices are reused at every time step.

For a 10-class classification task, ReLU can be used in the hidden layers, with softmax and categorical cross-entropy at the output. For the 16-class RNN, the output softmax should cover all 16 classes.

In a very deep plain neural network, gradients can become extremely small as they travel backwards through many layers, making early layers learn very slowly. Skip connections provide an alternative path for information and gradients, helping deeper networks train more effectively.

---

## Model-choice memo

**To:** Head of Scheduling Operations  
**Subject:** Recommendation for structured-data no-show prediction

Based on the experiments, logistic regression is recommended as the initial structured-data model for scheduling operations.

It achieved an F1 score of 0.5714 and recall of 0.7451 at a threshold of 0.3, outperforming KNN, the decision stump and both voting approaches on F1. The selected KNN value was K=5, while the best decision-tree split was S4 (X1 ≤ 24), with an information gain of 0.1252.

For cost-based alerting, a threshold of 0.15 is recommended. At this threshold, the logistic regression model produced a test cost of INR 6,050, compared with INR 11,800 at a threshold of 0.5 and INR 20,400 when nobody is flagged.

The threshold was selected using training data to avoid test-set leakage. This is a promising initial result, but the evidence is limited because it comes from synthetic data and one train/test split.

Model A, the CNN for scan-image routing, is worth considering as a future project, but it should not take priority over validating and deploying the structured-data scheduling model.

---

## Limitations

1. **Synthetic dataset:** All 800 records are generated using a predefined script. They do not represent actual PharmEasy patient records or real operational performance.

2. **Single train/test split:** The classification models are evaluated using one stratified 70/30 split. Results may change with a different dataset split.

3. **Limited positive records:** The test set contains only 51 positive no-show records. This limits the reliability of some evaluation metrics.

4. **Model limitations:** Logistic regression and KNN are relatively simple models and may not capture every complex relationship in the data.

5. **Fixed clustering initialisation:** K-means uses the specified starting rows. The resulting clusters are reproducible but may not represent the globally optimal clustering.

6. **No real-world deployment validation:** The models have not been tested on live scheduling data, and the cost estimates are based on the assumptions provided in the assignment.

7. **Architecture estimates only:** The CNN and RNN parameter counts are theoretical calculations. These networks have not been trained or evaluated on actual images or referral-note data.

8. **Not clinical advice:** This project focuses only on operational scheduling, cost planning and data analysis. It does not diagnose, predict or recommend medical treatment.
