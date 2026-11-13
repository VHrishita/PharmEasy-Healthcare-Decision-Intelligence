
import csv
import math
import os
import numpy as np


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

DATA_FILE = "data/records.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# READ DATA
# --------------------------------------------------

def load_data():
    with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    # X1 to X5: five continuous predictors
    X = np.array([
        [
            float(row["months_registered"]),
            float(row["missed_visit_count_12m"]),
            float(row["copay_burden_ratio"]),
            float(row["avg_bill_inr_thousands"]),
            float(row["adherence_score"])
        ]
        for row in rows
    ], dtype=float)

    outcome = np.array(
        [int(row["no_show"]) for row in rows],
        dtype=int
    )

    value = np.array(
        [float(row["care_plan_cost_inr_thousands"]) for row in rows],
        dtype=float
    )

    return rows, X, outcome, value


# --------------------------------------------------
# STANDARDISATION
# --------------------------------------------------

def standardize(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=1)

    Z = (X - means) / stds

    return Z, means, stds


# --------------------------------------------------
# TASK 3.1: K-MEANS FROM SCRATCH
# --------------------------------------------------

def run_kmeans(X, k):
    n = X.shape[0]

    # Pinned starting centroid rows
    step = n // k
    starting_indices = [
        i * step for i in range(k)
    ]

    centroids = X[starting_indices].copy()

    # Initially, no row has a cluster assignment
    labels = np.full(n, -1, dtype=int)

    while True:

        previous_labels = labels.copy()

        # Calculate squared Euclidean distances
        distances = np.zeros((n, k))

        for cluster in range(k):
            difference = X - centroids[cluster]
            distances[:, cluster] = np.sum(
                difference ** 2,
                axis=1
            )

        # np.argmin selects the lower cluster number
        # when distances are tied
        labels = np.argmin(distances, axis=1)

        # Update centroids
        for cluster in range(k):
            members = X[labels == cluster]

            if len(members) > 0:
                centroids[cluster] = np.mean(
                    members,
                    axis=0
                )

        # Stop when no assignment changes
        if np.array_equal(labels, previous_labels):
            break

    # Calculate final WCSS using final centroids
    wcss = 0.0

    for cluster in range(k):
        members = X[labels == cluster]

        if len(members) > 0:
            differences = members - centroids[cluster]
            wcss += np.sum(differences ** 2)

    return labels, centroids, wcss


# --------------------------------------------------
# TASK 3.3: PROFILE THREE SEGMENTS
# --------------------------------------------------

def create_segments(X, labels, outcome, value):

    profiles = []

    for cluster in range(3):

        indices = np.where(labels == cluster)[0]
        members = X[indices]

        profile = {
            "original_cluster": cluster,
            "indices": indices,
            "size": len(indices),
            "means": np.mean(members, axis=0),
            "outcome_rate": np.mean(outcome[indices]),
            "mean_value": np.mean(value[indices])
        }

        profiles.append(profile)

    # Order by ascending mean X3
    profiles.sort(
        key=lambda item: item["means"][2]
    )

    # Assign segment names A, B and C
    segment_labels = np.empty(len(labels), dtype=object)

    for position, profile in enumerate(profiles):
        segment_name = ["A", "B", "C"][position]

        profile["segment"] = segment_name
        segment_labels[profile["indices"]] = segment_name

    return profiles, segment_labels


# --------------------------------------------------
# TASK 3.4: HIERARCHICAL CLUSTERING
# --------------------------------------------------

def cluster_distance(cluster_a, cluster_b, matrix, linkage):

    distances = []

    for a in cluster_a:
        for b in cluster_b:
            distances.append(matrix[a][b])

    if linkage == "single":
        return min(distances)

    elif linkage == "complete":
        return max(distances)

    elif linkage == "average":
        return sum(distances) / len(distances)

    raise ValueError("Unknown linkage")


def run_hierarchical(matrix, linkage):

    # Each cohort begins as an individual cluster
    clusters = [[i] for i in range(6)]

    merge_records = []
    merge_heights = []

    for merge_no in range(1, 6):

        best_distance = float("inf")
        best_pair = None

        # Examine every pair of current clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):

                distance = cluster_distance(
                    clusters[i],
                    clusters[j],
                    matrix,
                    linkage
                )

                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        i, j = best_pair

        cluster_a = clusters[i]
        cluster_b = clusters[j]

        # Ensure cluster_a has the smaller minimum member
        if min(cluster_a) > min(cluster_b):
            cluster_a, cluster_b = cluster_b, cluster_a

        merged_cluster = sorted(cluster_a + cluster_b)

        merge_records.append({
            "linkage": linkage,
            "merge_no": merge_no,
            "cluster_a": "+".join(
                f"E{x + 1}" for x in cluster_a
            ),
            "cluster_b": "+".join(
                f"E{x + 1}" for x in cluster_b
            ),
            "height": best_distance
        })

        merge_heights.append(best_distance)

        # Remove old clusters and add merged cluster
        clusters.remove(cluster_a)
        clusters.remove(cluster_b)
        clusters.append(merged_cluster)

        clusters.sort(key=lambda c: min(c))

    # Find the largest gap between consecutive merge heights
    gaps = [
        merge_heights[i + 1] - merge_heights[i]
        for i in range(4)
    ]

    largest_gap_index = int(np.argmax(gaps))

    # Number of merges below the cut
    merges_below_cut = largest_gap_index + 1

    # Reconstruct clusters at the selected cut
    cut_clusters = [[i] for i in range(6)]

    for merge_index in range(merges_below_cut):

        record = merge_records[merge_index]

        members_a = [
            int(x[1:]) - 1
            for x in record["cluster_a"].split("+")
        ]

        members_b = [
            int(x[1:]) - 1
            for x in record["cluster_b"].split("+")
        ]

        cut_clusters.remove(members_a)
        cut_clusters.remove(members_b)

        cut_clusters.append(
            sorted(members_a + members_b)
        )

        cut_clusters.sort(key=lambda c: min(c))

    cut_description = ";".join(
        "+".join(f"E{x + 1}" for x in cluster)
        for cluster in cut_clusters
    )

    return merge_records, cut_description


# --------------------------------------------------
# TASK 3.5: PCA FROM SCRATCH
# --------------------------------------------------

def run_pca(X, segment_labels):

    # Step 1: Calculate the mean vector
    mean_vector = np.mean(X, axis=0)

    # Step 2: Centre the data
    centred = X - mean_vector

    # Step 3: Calculate covariance matrix
    n = centred.shape[0]

    covariance = (
        centred.T @ centred
    ) / (n - 1)

    # Step 4: Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    # Step 5: Sort eigenvalues in descending order
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Apply the required sign convention
    for column in range(eigenvectors.shape[1]):

        vector = eigenvectors[:, column]

        largest_index = np.argmax(np.abs(vector))

        if vector[largest_index] < 0:
            eigenvectors[:, column] *= -1

    # Step 6: Keep the top two eigenvectors
    W = eigenvectors[:, :2]

    # Step 7: Project data onto the two components
    projections = centred @ W

    eigenvalue_sum = np.sum(eigenvalues)

    explained_ratios = eigenvalues / eigenvalue_sum
    cumulative_ratios = np.cumsum(explained_ratios)

    components_for_80pct = int(
        np.argmax(cumulative_ratios >= 0.80) + 1
    )

    results = {}

    results["eigenvalue_sum"] = eigenvalue_sum

    for i in range(5):
        results[f"eigenvalue_{i + 1}"] = eigenvalues[i]
        results[f"explained_ratio_{i + 1}"] = explained_ratios[i]
        results[f"cumulative_ratio_{i + 1}"] = cumulative_ratios[i]

    results["components_for_80pct"] = components_for_80pct

    # PC1 and PC2 loadings
    for i in range(5):
        results[f"pc1_loading_X{i + 1}"] = W[i, 0]
        results[f"pc2_loading_X{i + 1}"] = W[i, 1]

    # First three data-row projections
    for row_index in range(3):
        results[f"row{row_index + 1}_pc1"] = projections[row_index, 0]
        results[f"row{row_index + 1}_pc2"] = projections[row_index, 1]

    # Mean PC1 score for each segment
    for segment in ["A", "B", "C"]:
        indices = np.where(segment_labels == segment)[0]

        results[f"mean_pc1_segment_{segment}"] = np.mean(
            projections[indices, 0]
        )

    return results


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    rows, X, outcome, value = load_data()

    # Standardise all five continuous predictors
    Z, means, stds = standardize(X)

    results = {}

    # ----------------------------------------------
    # TASK 3.1: K-MEANS FOR K = 2 TO 6
    # ----------------------------------------------

    kmeans_results = {}

    for k in range(2, 7):

        labels, centroids, wcss = run_kmeans(Z, k)

        kmeans_results[k] = {
            "labels": labels,
            "centroids": centroids,
            "wcss": wcss
        }

        results[f"wcss_k{k}"] = wcss

    # ----------------------------------------------
    # TASK 3.2: ELBOW METHOD
    # ----------------------------------------------

    elbow_k = None

    for k in [3, 4, 5]:

        previous_drop = (
            kmeans_results[k - 1]["wcss"]
            - kmeans_results[k]["wcss"]
        )

        current_drop = (
            kmeans_results[k]["wcss"]
            - kmeans_results[k + 1]["wcss"]
        )

        if current_drop < 0.5 * previous_drop:
            elbow_k = k
            break

    if elbow_k is None:
        elbow_k = 5

    results["elbow_k"] = elbow_k

    # ----------------------------------------------
    # TASK 3.3: USE K = 3 FOR SEGMENTATION
    # ----------------------------------------------

    labels_k3 = kmeans_results[3]["labels"]

    profiles, segment_labels = create_segments(
        X,
        labels_k3,
        outcome,
        value
    )

    for profile in profiles:
        segment = profile["segment"]
        results[f"size_segment_{segment}"] = profile["size"]

    # Write segment CSV
    segment_file = os.path.join(
        OUTPUT_DIR,
        "part3_segments.csv"
    )

    with open(segment_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "segment",
            "size",
            "mean_X1",
            "mean_X2",
            "mean_X3",
            "mean_X4",
            "mean_X5",
            "outcome_rate",
            "mean_value"
        ])

        for profile in profiles:

            writer.writerow([
                profile["segment"],
                profile["size"],
                *[f"{x:.4f}" for x in profile["means"]],
                f"{profile['outcome_rate']:.4f}",
                f"{profile['mean_value']:.4f}"
            ])

    # ----------------------------------------------
    # TASK 3.4: HIERARCHICAL CLUSTERING
    # ----------------------------------------------

    distance_matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
    ])

    all_hierarchical_records = []

    for linkage in ["single", "complete", "average"]:

        merge_records, cut_description = run_hierarchical(
            distance_matrix,
            linkage
        )

        all_hierarchical_records.extend(merge_records)

        results[f"{linkage}_clusters_at_cut"] = cut_description

    # Write hierarchical CSV
    hierarchical_file = os.path.join(
        OUTPUT_DIR,
        "part3_hierarchical.csv"
    )

    with open(hierarchical_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "linkage",
            "merge_no",
            "cluster_a",
            "cluster_b",
            "height"
        ])

        for record in all_hierarchical_records:

            writer.writerow([
                record["linkage"],
                record["merge_no"],
                record["cluster_a"],
                record["cluster_b"],
                f"{record['height']:.4f}"
            ])

    # ----------------------------------------------
    # TASK 3.5: PCA
    # ----------------------------------------------

    pca_results = run_pca(Z, segment_labels)

    results.update(pca_results)

    # ----------------------------------------------
    # WRITE RESULTS TEXT FILE
    # ----------------------------------------------

    results_file = os.path.join(
        OUTPUT_DIR,
        "part3_results.txt"
    )

    with open(results_file, "w", encoding="utf-8") as file:

        for key, value_result in results.items():

            if isinstance(value_result, (int, np.integer)):
                file.write(f"{key} = {value_result}\n")

            elif isinstance(value_result, str):
                file.write(f"{key} = {value_result}\n")

            else:
                file.write(f"{key} = {float(value_result):.4f}\n")

    # ----------------------------------------------
    # COMPLETION MESSAGE
    # ----------------------------------------------

    print("\nPART 3 COMPLETED SUCCESSFULLY")
    print("-----------------------------------")
    print("Total records:", len(rows))
    print("K-means checkpoint WCSS (K=2):",
          f"{results['wcss_k2']:.4f}")
    print("Selected elbow K:", results["elbow_k"])
    print("Segment sizes:",
          sum(profile["size"] for profile in profiles))
    print("PCA eigenvalue sum:",
          f"{results['eigenvalue_sum']:.4f}")
    print("-----------------------------------")
    print("Generated files:")
    print("- outputs/part3_results.txt")
    print("- outputs/part3_segments.csv")
    print("- outputs/part3_hierarchical.csv")


if __name__ == "__main__":
    main()