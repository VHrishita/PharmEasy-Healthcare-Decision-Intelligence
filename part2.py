
import csv
from pathlib import Path

import numpy as np

from part2_helpers import (
    stratified_split,
    standardize_data,
    train_logistic_regression,
    predict_logistic,
    binary_cross_entropy,
    confusion_matrix_values,
    calculate_metrics,
    evaluate_threshold,
    knn_predict,
    knn_cross_validation
)


# ============================================================
# PHARMEASY HEALTHCARE DECISION INTELLIGENCE
# PART 2: CLASSIFICATION AND EVALUATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "records.csv"

OUTPUT_DIR = BASE_DIR / "outputs"

RESULTS_FILE = OUTPUT_DIR / "part2_results.txt"

THRESHOLDS_FILE = OUTPUT_DIR / "part2_thresholds.csv"

KNN_CV_FILE = OUTPUT_DIR / "part2_knn_cv.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    with open(DATA_FILE, "r", newline="") as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    feature_columns = [
        "months_registered",
        "missed_visit_count_12m",
        "copay_burden_ratio",
        "avg_bill_inr_thousands",
        "adherence_score",
        "chronic_condition_flag",
        "insured_flag"
    ]

    X = np.array([
        [
            float(row[column])
            for column in feature_columns
        ]
        for row in rows
    ])

    y = np.array([
        int(row["no_show"])
        for row in rows
    ])

    return X, y


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w") as file:

        for key, value in results.items():

            if isinstance(value, (bool, np.bool_)):

                formatted = "yes" if value else "no"

            elif isinstance(value, (int, np.integer)):

                formatted = str(value)

            else:

                formatted = f"{value:.4f}"

            file.write(f"{key} = {formatted}\n")


# ============================================================
# TASK 2.5: CLASSIFICATION EVALUATION
# ============================================================

def store_threshold_metrics(results, y, probabilities, threshold):

    tp, fp, tn, fn, metrics = evaluate_threshold(
        y, probabilities, threshold
    )

    prefix = f"tau_{threshold}"

    results[f"{prefix}_tp"] = tp
    results[f"{prefix}_fp"] = fp
    results[f"{prefix}_tn"] = tn
    results[f"{prefix}_fn"] = fn

    results[f"{prefix}_accuracy"] = metrics["accuracy"]
    results[f"{prefix}_precision"] = metrics["precision"]
    results[f"{prefix}_recall"] = metrics["recall"]
    results[f"{prefix}_f1"] = metrics["f1"]
    results[f"{prefix}_specificity"] = metrics["specificity"]


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    results = {}

    # Load dataset
    X, y = load_data()

    # ========================================================
    # TASK 2.1: STRATIFIED SPLIT AND STANDARDISATION
    # ========================================================

    train_indices, test_indices = stratified_split(X, y)

    X_train_original = X[train_indices]
    X_test_original = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    X_train, X_test, train_mean, train_sd = standardize_data(
        X_train_original,
        X_test_original
    )

    train_rows = len(y_train)
    train_positives = int(np.sum(y_train))

    test_rows = len(y_test)
    test_positives = int(np.sum(y_test))

    results["train_rows"] = train_rows
    results["train_positives"] = train_positives
    results["test_rows"] = test_rows
    results["test_positives"] = test_positives

    # ========================================================
    # TASK 2.2: LOGISTIC REGRESSION
    # ========================================================

    weights, bias, captured_losses = train_logistic_regression(
        X_train,
        y_train,
        learning_rate=0.1,
        updates=1000,
        capture_epochs=[1, 10, 100, 1000]
    )

    results["loss_epoch_0"] = captured_losses[0]
    results["loss_epoch_1"] = captured_losses[1]
    results["loss_epoch_10"] = captured_losses[10]
    results["loss_epoch_100"] = captured_losses[100]
    results["loss_epoch_1000"] = captured_losses[1000]

    for index, weight in enumerate(weights, start=1):

        results[f"weight_X{index}"] = weight

    results["bias"] = bias

    train_probabilities = predict_logistic(
        X_train,
        weights,
        bias
    )

    test_probabilities = predict_logistic(
        X_test,
        weights,
        bias
    )

    test_bce = binary_cross_entropy(
        y_test,
        test_probabilities
    )

    results["test_bce"] = test_bce

    # ========================================================
    # TASK 2.3: LEARNING-RATE EFFECT
    # ========================================================

    learning_rates = [0.01, 0.1, 1.0]

    for learning_rate in learning_rates:

        _, _, losses = train_logistic_regression(
            X_train,
            y_train,
            learning_rate=learning_rate,
            updates=200,
            capture_epochs=[200]
        )

        results[
            f"loss_epoch_200_eta_{learning_rate}"
        ] = losses[200]

    # ========================================================
    # TASK 2.4: ODDS RATIOS
    # ========================================================

    odds_ratios = np.exp(weights)

    for index, odds_ratio in enumerate(odds_ratios, start=1):

        results[f"odds_ratio_X{index}"] = odds_ratio

    # ========================================================
    # TASK 2.5: TEST EVALUATION AT 0.5 AND 0.3
    # ========================================================

    store_threshold_metrics(
        results,
        y_test,
        test_probabilities,
        0.5
    )

    store_threshold_metrics(
        results,
        y_test,
        test_probabilities,
        0.3
    )

    # Blind model: predicts zero for every record

    blind_predictions = np.zeros(
        len(y_test),
        dtype=int
    )

    blind_tp, blind_fp, blind_tn, blind_fn = (
        confusion_matrix_values(y_test, blind_predictions)
    )

    blind_metrics = calculate_metrics(
        blind_tp,
        blind_fp,
        blind_tn,
        blind_fn
    )

    results["blind_accuracy"] = blind_metrics["accuracy"]

    # ========================================================
    # TASK 2.6: THRESHOLD SWEEP
    # ========================================================

    threshold_rows = []

    for threshold in np.arange(0.1, 1.0, 0.1):

        threshold = round(float(threshold), 1)

        tp, fp, tn, fn, metrics = evaluate_threshold(
            y_test,
            test_probabilities,
            threshold
        )

        threshold_rows.append([
            threshold,
            tp,
            fp,
            tn,
            fn,
            metrics["recall"],
            metrics["fpr"]
        ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(THRESHOLDS_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "tau",
            "tp",
            "fp",
            "tn",
            "fn",
            "tpr",
            "fpr"
        ])

        for row in threshold_rows:

            writer.writerow([
                f"{row[0]:.4f}",
                row[1],
                row[2],
                row[3],
                row[4],
                f"{row[5]:.4f}",
                f"{row[6]:.4f}"
            ])

    # ========================================================
    # TASK 2.7: K-NEAREST NEIGHBOURS
    # ========================================================

    k_values = list(range(1, 22, 2))

    cv_results = knn_cross_validation(
        X_train,
        y_train,
        k_values
    )

    # Highest mean F1 wins.
    # Ties go to smaller K.

    knn_best_k = sorted(
        cv_results,
        key=lambda item: (-item[1], item[0])
    )[0][0]

    results["knn_best_k"] = knn_best_k

    # Save cross-validation results

    with open(KNN_CV_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "K",
            "mean_cv_f1"
        ])

        for k, mean_f1 in cv_results:

            writer.writerow([
                k,
                f"{mean_f1:.4f}"
            ])

    # Refit using all training rows

    knn_predictions, knn_probabilities = knn_predict(
        X_train,
        y_train,
        X_test,
        knn_best_k
    )

    knn_tp, knn_fp, knn_tn, knn_fn = (
        confusion_matrix_values(
            y_test,
            knn_predictions
        )
    )

    knn_metrics = calculate_metrics(
        knn_tp,
        knn_fp,
        knn_tn,
        knn_fn
    )

    results["knn_tp"] = knn_tp
    results["knn_fp"] = knn_fp
    results["knn_tn"] = knn_tn
    results["knn_fn"] = knn_fn

    results["knn_accuracy"] = knn_metrics["accuracy"]
    results["knn_precision"] = knn_metrics["precision"]
    results["knn_recall"] = knn_metrics["recall"]
    results["knn_f1"] = knn_metrics["f1"]

    # ========================================================
    # TASK 2.8: COST-BASED THRESHOLD SELECTION
    # ========================================================

    best_cost = float("inf")
    cost_tau = None

    for k in range(1, 20):

        threshold = k / 20

        predictions = (
            train_probabilities >= threshold
        ).astype(int)

        tp, fp, tn, fn = confusion_matrix_values(
            y_train,
            predictions
        )

        cost = 50 * fp + 400 * fn

        # Ascending search automatically preserves
        # the smaller threshold in case of ties.

        if cost < best_cost:

            best_cost = cost
            cost_tau = threshold

    results["cost_tau"] = cost_tau

    results["cost_train_at_tau"] = best_cost

    # Test cost using selected training threshold

    test_predictions_at_tau = (
        test_probabilities >= cost_tau
    ).astype(int)

    tp, fp, tn, fn = confusion_matrix_values(
        y_test,
        test_predictions_at_tau
    )

    cost_test_at_tau = 50 * fp + 400 * fn

    results["cost_test_at_tau"] = cost_test_at_tau

    # Test cost at 0.5

    test_predictions_at_05 = (
        test_probabilities >= 0.5
    ).astype(int)

    tp, fp, tn, fn = confusion_matrix_values(
        y_test,
        test_predictions_at_05
    )

    cost_test_at_05 = 50 * fp + 400 * fn

    results["cost_test_at_0.5"] = cost_test_at_05

    # Cost of flagging nobody

    cost_test_flag_nobody = 400 * test_positives

    results["cost_test_flag_nobody"] = cost_test_flag_nobody

    # ========================================================
    # SAVE PART 2 RESULTS
    # ========================================================

    save_results(results)

    print("=" * 60)
    print("PHARMEASY HEALTHCARE DECISION INTELLIGENCE")
    print("PART 2 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Training rows: {train_rows}")
    print(f"Training positives: {train_positives}")

    print(f"Testing rows: {test_rows}")
    print(f"Testing positives: {test_positives}")

    print("-" * 60)

    print(f"Logistic training loss: {captured_losses[1000]:.4f}")

    print(f"Test BCE: {test_bce:.4f}")

    print(f"Best K: {knn_best_k}")

    print(f"Cost-based threshold: {cost_tau:.2f}")

    print(f"Test cost at selected threshold: {cost_test_at_tau}")

    print("-" * 60)

    print("Generated files:")

    print(RESULTS_FILE)
    print(THRESHOLDS_FILE)
    print(KNN_CV_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()