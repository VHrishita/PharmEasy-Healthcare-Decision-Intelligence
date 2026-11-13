
import csv
import math
from pathlib import Path

import numpy as np


# ============================================================
# PHARMEASY HEALTHCARE DECISION INTELLIGENCE
# PART 1: ESTIMATION, INFERENCE AND REGRESSION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "records.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "part1_results.txt"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    with open(DATA_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    x4 = np.array(
        [float(row["avg_bill_inr_thousands"]) for row in rows]
    )

    value = np.array(
        [float(row["care_plan_cost_inr_thousands"]) for row in rows]
    )

    x1 = np.array(
        [float(row["months_registered"]) for row in rows]
    )

    x2 = np.array(
        [float(row["missed_visit_count_12m"]) for row in rows]
    )

    x3 = np.array(
        [float(row["copay_burden_ratio"]) for row in rows]
    )

    x5 = np.array(
        [float(row["adherence_score"]) for row in rows]
    )

    x6 = np.array(
        [float(row["chronic_condition_flag"]) for row in rows]
    )

    x7 = np.array(
        [float(row["insured_flag"]) for row in rows]
    )

    return rows, x1, x2, x3, x4, x5, x6, x7, value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_regression(x, y):
    """
    Closed-form simple linear regression.

    slope = covariance(x, y) / variance(x)
    intercept = mean(y) - slope * mean(x)
    """

    n = len(x)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    covariance = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    variance = np.sum((x - mean_x) ** 2) / (n - 1)

    slope = covariance / variance
    intercept = mean_y - slope * mean_x

    fitted = slope * x + intercept
    residuals = y - fitted

    tss = np.sum((y - mean_y) ** 2)
    ess = np.sum((fitted - mean_y) ** 2)
    rss = np.sum(residuals ** 2)

    r2 = 1 - rss / tss
    rmse = math.sqrt(rss / n)
    mae = np.mean(np.abs(residuals))

    return {
        "slope": slope,
        "intercept": intercept,
        "fitted": fitted,
        "residuals": residuals,
        "tss": tss,
        "ess": ess,
        "rss": rss,
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    }


def residual_standard_deviation(residuals):
    return np.std(residuals, ddof=1)


def calculate_adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# ============================================================
# MAIN PART 1 EXECUTION
# ============================================================

def main():

    results = {}

    # Load dataset
    rows, x1, x2, x3, x4, x5, x6, x7, value = load_data()

    n = len(value)

    # ========================================================
    # TASK 1.1: ESTIMATION WITH CONFIDENCE INTERVAL
    # ========================================================

    mean_value = np.mean(value)
    sd_value = np.std(value, ddof=1)
    se_value = sd_value / math.sqrt(n)

    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    results["n"] = n
    results["mean_value"] = mean_value
    results["sd_value"] = sd_value
    results["se_value"] = se_value
    results["ci95_low"] = ci95_low
    results["ci95_high"] = ci95_high

    # ========================================================
    # TASK 1.2: HYPOTHESIS TESTING
    # ========================================================

    hypothesized_mean = 8.0

    z_stat = (mean_value - hypothesized_mean) / se_value

    reject_h0 = abs(z_stat) > 1.96

    results["z_stat"] = z_stat
    results["reject_h0"] = reject_h0

    # ========================================================
    # TASK 1.3: REQUIRED SAMPLE SIZE
    # ========================================================

    margin_of_error = 0.5

    n_required = math.ceil(
        (1.96 * sd_value / margin_of_error) ** 2
    )

    results["n_required"] = n_required

    # ========================================================
    # TASK 1.4: SIMPLE LINEAR REGRESSION
    # ========================================================

    regression = calculate_regression(x4, value)

    reg_slope = regression["slope"]
    reg_intercept = regression["intercept"]

    fitted = regression["fitted"]
    residuals = regression["residuals"]

    tss = regression["tss"]
    ess = regression["ess"]
    rss = regression["rss"]

    tss_equals_ess_plus_rss = (
        abs(tss - ess - rss) < 1e-6 * tss
    )

    r2 = regression["r2"]
    rmse = regression["rmse"]
    mae = regression["mae"]

    results["reg_slope"] = reg_slope
    results["reg_intercept"] = reg_intercept
    results["tss"] = tss
    results["ess"] = ess
    results["rss"] = rss
    results["tss_equals_ess_plus_rss"] = tss_equals_ess_plus_rss
    results["r2"] = r2
    results["rmse"] = rmse
    results["mae"] = mae

    # ========================================================
    # TASK 1.5: GRADIENT DESCENT
    # ========================================================

    mean_x4 = np.mean(x4)
    sd_x4 = np.std(x4, ddof=1)

    z = (x4 - mean_x4) / sd_x4

    m = 0.0
    c = 0.0

    learning_rate = 0.1

    def calculate_cost(m_value, c_value):
        predictions = m_value * z + c_value
        errors = value - predictions
        return np.mean(errors ** 2)

    gd_J0 = calculate_cost(m, c)

    gd_costs = {}

    for iteration in range(1, 501):

        predictions = m * z + c
        errors = value - predictions

        gradient_m = -(2 / n) * np.sum(z * errors)
        gradient_c = -(2 / n) * np.sum(errors)

        m = m - learning_rate * gradient_m
        c = c - learning_rate * gradient_c

        if iteration in [1, 10, 100, 500]:
            gd_costs[iteration] = calculate_cost(m, c)

    gd_J1 = gd_costs[1]
    gd_J10 = gd_costs[10]
    gd_J100 = gd_costs[100]
    gd_J500 = gd_costs[500]

    gd_slope = m / sd_x4

    gd_intercept = c - (m * mean_x4 / sd_x4)

    gd_slope_gap = abs(gd_slope - reg_slope)

    results["gd_J0"] = gd_J0
    results["gd_J1"] = gd_J1
    results["gd_J10"] = gd_J10
    results["gd_J100"] = gd_J100
    results["gd_J500"] = gd_J500
    results["gd_slope"] = gd_slope
    results["gd_intercept"] = gd_intercept
    results["gd_slope_gap"] = gd_slope_gap

    # ========================================================
    # TASK 1.6: RESIDUAL ANALYSIS AND LOG-LOG REGRESSION
    # ========================================================

    # Stable sorting by fitted values
    sorted_indices = np.argsort(fitted, kind="stable")

    lower_indices = sorted_indices[:400]
    upper_indices = sorted_indices[400:]

    resid_sd_lower_raw = residual_standard_deviation(
        residuals[lower_indices]
    )

    resid_sd_upper_raw = residual_standard_deviation(
        residuals[upper_indices]
    )

    resid_sd_ratio_raw = (
        resid_sd_upper_raw / resid_sd_lower_raw
    )

    # Log transformation
    log_x4 = np.log(x4)
    log_value = np.log(value)

    log_regression = calculate_regression(log_x4, log_value)

    loglog_intercept = log_regression["intercept"]
    loglog_slope = log_regression["slope"]

    log_fitted = log_regression["fitted"]
    log_residuals = log_regression["residuals"]

    loglog_r2 = log_regression["r2"]
    loglog_rmse = log_regression["rmse"]

    # Stable sorting using the log model fitted values
    log_sorted_indices = np.argsort(
        log_fitted, kind="stable"
    )

    log_lower_indices = log_sorted_indices[:400]
    log_upper_indices = log_sorted_indices[400:]

    resid_sd_lower_log = residual_standard_deviation(
        log_residuals[log_lower_indices]
    )

    resid_sd_upper_log = residual_standard_deviation(
        log_residuals[log_upper_indices]
    )

    resid_sd_ratio_log = (
        resid_sd_upper_log / resid_sd_lower_log
    )

    results["resid_sd_lower_raw"] = resid_sd_lower_raw
    results["resid_sd_upper_raw"] = resid_sd_upper_raw
    results["resid_sd_ratio_raw"] = resid_sd_ratio_raw

    results["loglog_intercept"] = loglog_intercept
    results["loglog_slope"] = loglog_slope
    results["loglog_r2"] = loglog_r2
    results["loglog_rmse"] = loglog_rmse

    results["resid_sd_lower_log"] = resid_sd_lower_log
    results["resid_sd_upper_log"] = resid_sd_upper_log
    results["resid_sd_ratio_log"] = resid_sd_ratio_log

    # ========================================================
    # TASK 1.7: MULTIPLE REGRESSION USING NORMAL EQUATION
    # ========================================================

    # Design matrix in the exact order specified:
    # 1. Constant
    # 2. ln(X4)
    # 3. X1
    # 4. X2
    # 5. X3
    # 6. X5
    # 7. X6
    # 8. X7

    design_matrix = np.column_stack([
        np.ones(n),
        np.log(x4),
        x1,
        x2,
        x3,
        x5,
        x6,
        x7
    ])

    target = np.log(value)

    # Normal equation:
    # beta = (X.T @ X)^(-1) @ X.T @ Y
    # Using solve instead of explicitly calculating inverse.

    xtx = design_matrix.T @ design_matrix
    xty = design_matrix.T @ target

    beta = np.linalg.solve(xtx, xty)

    # Predictions
    multiple_fitted = design_matrix @ beta

    # Calculate R-squared manually
    target_mean = np.mean(target)

    multiple_tss = np.sum((target - target_mean) ** 2)
    multiple_rss = np.sum((target - multiple_fitted) ** 2)

    mr_r2 = 1 - multiple_rss / multiple_tss

    mr_adj_r2 = calculate_adjusted_r2(
        mr_r2, n, 7
    )

    loglog_adj_r2 = calculate_adjusted_r2(
        loglog_r2, n, 1
    )

    for i in range(8):
        results[f"mr_beta_{i}"] = beta[i]

    results["mr_r2"] = mr_r2
    results["mr_adj_r2"] = mr_adj_r2
    results["loglog_adj_r2"] = loglog_adj_r2

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as file:

        for key, result in results.items():

            if isinstance(result, (bool, np.bool_)):
                formatted_value = "yes" if result else "no"

            elif isinstance(result, (int, np.integer)):
                formatted_value = str(result)

            else:
                formatted_value = f"{result:.4f}"

            file.write(
                f"{key} = {formatted_value}\n"
            )

    print("=" * 60)
    print("PHARMEASY HEALTHCARE DECISION INTELLIGENCE")
    print("PART 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Total records processed: {n}")
    print(f"Mean care-plan cost: {mean_value:.4f}")
    print(f"Required sample size: {n_required}")
    print(f"Regression R-squared: {r2:.4f}")
    print(f"Gradient descent slope gap: {gd_slope_gap:.8f}")

    print("-" * 60)
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()