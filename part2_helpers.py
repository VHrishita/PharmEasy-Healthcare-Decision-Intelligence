
import numpy as np


# ============================================================
# PART 2: REUSABLE CLASSIFICATION FUNCTIONS
# ============================================================


def stratified_split(X, y):
    """
    Stratified 70/30 split.

    First floor(7 * count / 10) records of each
    class go into training.

    Both sets preserve original file order.
    """

    train_indices = []
    test_indices = []

    for class_value in [0, 1]:

        class_indices = np.where(y == class_value)[0]

        split_point = (7 * len(class_indices)) // 10

        train_indices.extend(class_indices[:split_point])
        test_indices.extend(class_indices[split_point:])

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    return train_indices, test_indices


def standardize_data(X_train, X_test):
    """
    Standardise using training mean and sample SD.
    """

    mean = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0, ddof=1)

    X_train_scaled = (X_train - mean) / sd
    X_test_scaled = (X_test - mean) / sd

    return X_train_scaled, X_test_scaled, mean, sd


def sigmoid(z):
    """
    Numerically stable sigmoid function.
    """

    result = np.empty_like(z, dtype=float)

    positive = z >= 0

    result[positive] = 1 / (1 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])

    result[~positive] = exp_z / (1 + exp_z)

    return result


def binary_cross_entropy(y, probabilities):
    """
    Average binary cross-entropy.
    """

    probabilities = np.clip(
        probabilities, 1e-15, 1 - 1e-15
    )

    loss = -np.mean(
        y * np.log(probabilities)
        + (1 - y) * np.log(1 - probabilities)
    )

    return loss


def train_logistic_regression(
    X,
    y,
    learning_rate=0.1,
    updates=1000,
    capture_epochs=None
):
    """
    Logistic regression implemented from scratch.

    Starts all weights and bias at zero.
    Uses full-batch gradient descent.
    """

    n, number_of_features = X.shape

    weights = np.zeros(number_of_features)
    bias = 0.0

    if capture_epochs is None:
        capture_epochs = []

    captured_losses = {}

    probabilities = sigmoid(X @ weights + bias)

    captured_losses[0] = binary_cross_entropy(
        y, probabilities
    )

    for epoch in range(1, updates + 1):

        z = X @ weights + bias

        probabilities = sigmoid(z)

        errors = probabilities - y

        gradient_weights = (X.T @ errors) / n

        gradient_bias = np.sum(errors) / n

        weights -= learning_rate * gradient_weights

        bias -= learning_rate * gradient_bias

        if epoch in capture_epochs:

            probabilities = sigmoid(X @ weights + bias)

            captured_losses[epoch] = binary_cross_entropy(
                y, probabilities
            )

    return weights, bias, captured_losses


def predict_logistic(X, weights, bias):
    """
    Return predicted probabilities.
    """

    return sigmoid(X @ weights + bias)


def confusion_matrix_values(y_true, predictions):
    """
    Calculate TP, FP, TN and FN manually.
    """

    tp = int(np.sum((y_true == 1) & (predictions == 1)))

    fp = int(np.sum((y_true == 0) & (predictions == 1)))

    tn = int(np.sum((y_true == 0) & (predictions == 0)))

    fn = int(np.sum((y_true == 1) & (predictions == 0)))

    return tp, fp, tn, fn


def calculate_metrics(tp, fp, tn, fn):
    """
    Calculate classification metrics manually.
    """

    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0

    precision = tp / (tp + fp) if (tp + fp) else 0

    recall = tp / (tp + fn) if (tp + fn) else 0

    specificity = tn / (tn + fp) if (tn + fp) else 0

    fpr = fp / (fp + tn) if (fp + tn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1
    }


def evaluate_threshold(y_true, probabilities, threshold):
    """
    Predict positive when probability >= threshold.
    """

    predictions = (probabilities >= threshold).astype(int)

    tp, fp, tn, fn = confusion_matrix_values(
        y_true, predictions
    )

    metrics = calculate_metrics(tp, fp, tn, fn)

    return tp, fp, tn, fn, metrics


# ============================================================
# K-NEAREST NEIGHBOURS
# ============================================================

def knn_predict(
    X_reference,
    y_reference,
    X_query,
    k
):
    """
    KNN using Euclidean distance.

    Stable sorting preserves reference-row order
    when distances are tied.

    Returns:
    - Predicted classes
    - Positive-neighbour probabilities
    """

    predicted_classes = []
    predicted_probabilities = []

    required_votes = (k + 1) // 2

    for query in X_query:

        differences = X_reference - query

        distances = np.sqrt(
            np.sum(differences ** 2, axis=1)
        )

        neighbour_indices = np.argsort(
            distances, kind="stable"
        )[:k]

        neighbour_labels = y_reference[neighbour_indices]

        positive_count = int(np.sum(neighbour_labels))

        probability = positive_count / k

        prediction = (
            1 if positive_count >= required_votes else 0
        )

        predicted_classes.append(prediction)

        predicted_probabilities.append(probability)

    return (
        np.array(predicted_classes, dtype=int),
        np.array(predicted_probabilities, dtype=float)
    )


def knn_cross_validation(X, y, k_values):
    """
    Five-fold cross-validation.

    Fold assignment:
    training position modulo 5.

    Score:
    Mean of the five fold F1 scores.
    """

    results = []

    positions = np.arange(len(y))

    folds = positions % 5

    for k in k_values:

        fold_f1_scores = []

        for fold_number in range(5):

            validation_mask = folds == fold_number

            reference_mask = ~validation_mask

            X_reference = X[reference_mask]
            y_reference = y[reference_mask]

            X_validation = X[validation_mask]
            y_validation = y[validation_mask]

            predictions, _ = knn_predict(
                X_reference,
                y_reference,
                X_validation,
                k
            )

            tp, fp, tn, fn = confusion_matrix_values(
                y_validation,
                predictions
            )

            metrics = calculate_metrics(tp, fp, tn, fn)

            fold_f1_scores.append(metrics["f1"])

        mean_f1 = float(np.mean(fold_f1_scores))

        results.append((k, mean_f1))

    return results
