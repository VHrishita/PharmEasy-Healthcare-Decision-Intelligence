
import csv
import math
import os
import numpy as np

from part2_helpers import (
    stratified_split,
    standardize_data,
    train_logistic_regression,
    predict_logistic,
    knn_predict,
    knn_cross_validation,
    confusion_matrix_values,
    calculate_metrics
)


# ============================================================
# PART 4: MODEL CHOICE AND ARCHITECTURE
# ============================================================

os.makedirs("outputs", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

with open("data/records.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Seven predictors: X1 to X7
X = np.array([
    [
        float(row["months_registered"]),
        float(row["missed_visit_count_12m"]),
        float(row["copay_burden_ratio"]),
        float(row["avg_bill_inr_thousands"]),
        float(row["adherence_score"]),
        float(row["chronic_condition_flag"]),
        float(row["insured_flag"])
    ]
    for row in rows
], dtype=float)

y = np.array(
    [int(row["no_show"]) for row in rows],
    dtype=int
)


# ============================================================
# RECREATE PART 2 TRAIN/TEST SPLIT
# ============================================================

train_indices, test_indices = stratified_split(X, y)

X_train_raw = X[train_indices]
X_test_raw = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

X_train, X_test, train_mean, train_sd = standardize_data(
    X_train_raw,
    X_test_raw
)


# ============================================================
# RECREATE PART 2 LOGISTIC REGRESSION
# ============================================================

logistic_weights, logistic_bias, _ = train_logistic_regression(
    X_train,
    y_train,
    learning_rate=0.1,
    updates=1000
)

logistic_probabilities = predict_logistic(
    X_test,
    logistic_weights,
    logistic_bias
)


# ============================================================
# RECREATE PART 2 KNN AND SELECT BEST K
# ============================================================

k_values = list(range(1, 22, 2))

knn_cv_results = knn_cross_validation(
    X_train,
    y_train,
    k_values
)

# Highest mean F1; ties go to smaller K
best_k = sorted(
    knn_cv_results,
    key=lambda item: (-item[1], item[0])
)[0][0]

knn_predictions, knn_probabilities = knn_predict(
    X_train,
    y_train,
    X_test,
    best_k
)


# ============================================================
# 4.1 ENTROPY AND GINI
# ============================================================

def calculate_entropy(labels):

    if len(labels) == 0:
        return 0.0

    positive_rate = np.mean(labels)
    negative_rate = 1 - positive_rate

    entropy = 0.0

    for probability in [positive_rate, negative_rate]:

        if probability > 0:
            entropy -= probability * math.log2(probability)

    return entropy


def calculate_gini(labels):

    if len(labels) == 0:
        return 0.0

    positive_rate = np.mean(labels)
    negative_rate = 1 - positive_rate

    return 1 - positive_rate ** 2 - negative_rate ** 2


root_entropy = calculate_entropy(y_train)
root_gini = calculate_gini(y_train)


# Four candidate splits, using ORIGINAL unstandardised values
split_conditions = {
    "S1": X_train_raw[:, 5] == 1,
    "S2": X_train_raw[:, 6] == 1,
    "S3": X_train_raw[:, 2] > 0.5,
    "S4": X_train_raw[:, 0] <= 24
}


split_results = []

for split_name, yes_mask in split_conditions.items():

    no_mask = ~yes_mask

    y_yes = y_train[yes_mask]
    y_no = y_train[no_mask]

    n_yes = len(y_yes)
    n_no = len(y_no)

    positives_yes = int(np.sum(y_yes))
    positives_no = int(np.sum(y_no))

    weighted_entropy = (
        (n_yes / len(y_train)) * calculate_entropy(y_yes)
        + (n_no / len(y_train)) * calculate_entropy(y_no)
    )

    weighted_gini = (
        (n_yes / len(y_train)) * calculate_gini(y_yes)
        + (n_no / len(y_train)) * calculate_gini(y_no)
    )

    info_gain = root_entropy - weighted_entropy
    gini_gain = root_gini - weighted_gini

    split_results.append({
        "split": split_name,
        "n_yes": n_yes,
        "positives_yes": positives_yes,
        "n_no": n_no,
        "positives_no": positives_no,
        "info_gain": info_gain,
        "gini_gain": gini_gain
    })


# Highest gain; ties preserve S1, S2, S3, S4 order
best_info_split = max(
    split_results,
    key=lambda item: item["info_gain"]
)

best_gini_split = max(
    split_results,
    key=lambda item: item["gini_gain"]
)

best_split_by_info_gain = best_info_split["split"]
best_split_by_gini_gain = best_gini_split["split"]


# Write split CSV
with open("outputs/part4_splits.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "split",
        "n_yes",
        "positives_yes",
        "n_no",
        "positives_no",
        "info_gain",
        "gini_gain"
    ])

    for item in split_results:

        writer.writerow([
            item["split"],
            item["n_yes"],
            item["positives_yes"],
            item["n_no"],
            item["positives_no"],
            f'{item["info_gain"]:.4f}',
            f'{item["gini_gain"]:.4f}'
        ])


# ============================================================
# 4.2 DECISION STUMP AND VOTING
# ============================================================

# Use the best information-gain split
stump_condition = split_conditions[best_split_by_info_gain]

stump_yes_mask = stump_condition
stump_no_mask = ~stump_yes_mask

stump_rate_yes = float(np.mean(y_train[stump_yes_mask]))
stump_rate_no = float(np.mean(y_train[stump_no_mask]))


# Determine stump probabilities for test rows
# The split conditions are applied to original test values.
if best_split_by_info_gain == "S1":
    stump_test_mask = X_test_raw[:, 5] == 1

elif best_split_by_info_gain == "S2":
    stump_test_mask = X_test_raw[:, 6] == 1

elif best_split_by_info_gain == "S3":
    stump_test_mask = X_test_raw[:, 2] > 0.5

else:
    stump_test_mask = X_test_raw[:, 0] <= 24


stump_probabilities = np.where(
    stump_test_mask,
    stump_rate_yes,
    stump_rate_no
)


# Individual model predictions at threshold 0.3
logistic_predictions = (
    logistic_probabilities >= 0.3
).astype(int)

stump_predictions = (
    stump_probabilities >= 0.3
).astype(int)


# Hard voting: at least two of three models predict positive
hard_vote_predictions = (
    logistic_predictions
    + knn_predictions
    + stump_predictions
    >= 2
).astype(int)


# Soft voting: average probability >= 0.3
soft_vote_probabilities = (
    logistic_probabilities
    + knn_probabilities
    + stump_probabilities
) / 3

soft_vote_predictions = (
    soft_vote_probabilities >= 0.3
).astype(int)


voting_models = [
    ("logistic", logistic_predictions),
    ("knn", knn_predictions),
    ("stump", stump_predictions),
    ("hard_vote", hard_vote_predictions),
    ("soft_vote", soft_vote_predictions)
]


# Write voting CSV
with open("outputs/part4_voting.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "model",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1"
    ])

    for model_name, predictions in voting_models:

        tp, fp, tn, fn = confusion_matrix_values(
            y_test,
            predictions
        )

        metrics = calculate_metrics(tp, fp, tn, fn)

        writer.writerow([
            model_name,
            tp,
            fp,
            tn,
            fn,
            f'{metrics["precision"]:.4f}',
            f'{metrics["recall"]:.4f}',
            f'{metrics["f1"]:.4f}'
        ])


# ============================================================
# 4.3 ADABOOST: TWO ROUNDS
# ============================================================

true_classes = np.array([
    1, 1, 1, -1, -1, -1, 1, -1
])

h1 = np.array([
    1, 1, -1, -1, -1, -1, -1, -1
])

h2 = np.array([
    -1, 1, 1, -1, 1, -1, 1, -1
])

# Initial equal weights
weights = np.full(8, 1 / 8, dtype=float)

adaboost_rows = []


def adaboost_round(round_number, predictions, current_weights):

    incorrect = predictions != true_classes

    error = float(np.sum(current_weights[incorrect]))

    alpha = 0.5 * math.log((1 - error) / error)

    unnormalised_weights = (
        current_weights
        * np.exp(-alpha * true_classes * predictions)
    )

    normalised_weights = (
        unnormalised_weights / np.sum(unnormalised_weights)
    )

    for i in range(8):

        adaboost_rows.append([
            round_number,
            i + 1,
            f"{current_weights[i]:.4f}",
            f"{unnormalised_weights[i]:.4f}",
            f"{normalised_weights[i]:.4f}"
        ])

    return error, alpha, normalised_weights


# Round 1 uses h1
round1_error, round1_alpha, weights_after_round1 = adaboost_round(
    1,
    h1,
    weights
)

# Round 2 uses h2 and round 1 normalised weights
round2_error, round2_alpha, weights_after_round2 = adaboost_round(
    2,
    h2,
    weights_after_round1
)


# Combined ensemble prediction
ensemble_scores = (
    round1_alpha * h1
    + round2_alpha * h2
)

ensemble_predictions = np.where(
    ensemble_scores >= 0,
    1,
    -1
)

ensemble_accuracy = float(
    np.mean(ensemble_predictions == true_classes)
)


# Write AdaBoost CSV
with open("outputs/part4_adaboost.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "round",
        "point",
        "weight_before",
        "weight_unnormalised",
        "weight_after"
    ])

    writer.writerows(adaboost_rows)


# ============================================================
# 4.4 OUT-OF-BAG FRACTION
# ============================================================

oob_fraction = (1 - 1 / 560) ** 560


# ============================================================
# 4.5 NEURAL NETWORK ARCHITECTURE PARAMETERS
# ============================================================

# MODEL A: MLP
mlp_hidden1 = 784 * 16 + 16
mlp_hidden2 = 16 * 16 + 16
mlp_output = 16 * 10 + 10

mlp_total = (
    mlp_hidden1
    + mlp_hidden2
    + mlp_output
)


# MODEL A: CNN
# Conv1: 5x5, input depth 1, 6 filters
cnn_conv1 = (5 * 5 * 1 + 1) * 6

# After Conv1: 28 x 28 x 6
# After Pool1: 14 x 14 x 6

# Conv2: 5x5, input depth 6, 16 filters
cnn_conv2 = (5 * 5 * 6 + 1) * 16

# After Conv2: 10 x 10 x 16
# After Pool2: 5 x 5 x 16
# Flattened size: 400

cnn_dense1 = 400 * 32 + 32
cnn_dense2 = 32 * 10 + 10

cnn_total = (
    cnn_conv1
    + cnn_conv2
    + cnn_dense1
    + cnn_dense2
)


# Fully connected vs convolutional layer
fc_layer_params = 128 * 128 * 3 * 1024 + 1024

conv_layer_params = (3 * 3 * 3 + 1) * 64


# MODEL B: RNN
# W_xh: 128 x 64
# W_hh: 128 x 128
# W_hy: 16 x 128
# b_h: 128
# b_y: 16

rnn_total = (
    128 * 64
    + 128 * 128
    + 16 * 128
    + 128
    + 16
)


# Write architecture CSV
with open("outputs/part4_architecture.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["layer", "parameters"])

    writer.writerows([
        ["mlp_hidden1", mlp_hidden1],
        ["mlp_hidden2", mlp_hidden2],
        ["mlp_output", mlp_output],
        ["cnn_conv1", cnn_conv1],
        ["cnn_conv2", cnn_conv2],
        ["cnn_dense1", cnn_dense1],
        ["cnn_dense2", cnn_dense2]
    ])


# ============================================================
# WRITE PART 4 RESULTS
# ============================================================

results = {
    "root_entropy": root_entropy,
    "root_gini": root_gini,

    "best_split_by_info_gain": best_split_by_info_gain,
    "best_split_by_gini_gain": best_split_by_gini_gain,

    "stump_rate_yes": stump_rate_yes,
    "stump_rate_no": stump_rate_no,

    "round1_error": round1_error,
    "round1_alpha": round1_alpha,
    "round2_error": round2_error,
    "round2_alpha": round2_alpha,

    "ensemble_accuracy": ensemble_accuracy,

    "oob_fraction": oob_fraction,

    "mlp_total": mlp_total,
    "cnn_total": cnn_total,

    "fc_layer_params": fc_layer_params,
    "conv_layer_params": conv_layer_params,

    "rnn_total": rnn_total
}


with open("outputs/part4_results.txt", "w") as file:

    for key, value in results.items():

        if isinstance(value, str):
            file.write(f"{key} = {value}\n")

        elif isinstance(value, (int, np.integer)):
            file.write(f"{key} = {int(value)}\n")

        else:
            file.write(f"{key} = {value:.4f}\n")


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\nPART 4 COMPLETED SUCCESSFULLY")
print("-----------------------------------")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Best K:", best_k)
print("Best split:", best_split_by_info_gain)
print("AdaBoost ensemble accuracy:", round(ensemble_accuracy, 4))
print("MLP parameters:", mlp_total)
print("CNN parameters:", cnn_total)
print("RNN parameters:", rnn_total)
print("-----------------------------------")

print("Generated files:")
print("- outputs/part4_results.txt")
print("- outputs/part4_splits.csv")
print("- outputs/part4_voting.csv")
print("- outputs/part4_adaboost.csv")
print("- outputs/part4_architecture.csv")