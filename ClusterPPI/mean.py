import numpy as np
from statistics import NormalDist
import statsmodels.stats.sandwich_covariance as sw
from ppi_py import ppi_mean_pointestimate, ppi_mean_ci
from ppi_py.utils import cov_cluster


theta = 0
cluster_size = 5
num_clusters_labeled = 300
num_clusters_unlabeled = 1000

rho = 0.3
# covariance matrix with ones on the diagonal and rho on the off-diagonal
sigma = (1 - rho) * np.eye(cluster_size) + rho * np.ones(
    (cluster_size, cluster_size)
)

ppi_correlation = 0.5


def cluster_ppi_mean_stats(
    Y,
    Y_hat,
    Y_hat_unlabeled,
    cluster_labels_labeled=None,
    cluster_labels_unlabeled=None,
):
    """
    Compute the cluster-robust PPI mean point estimate, standard error, and lambda hat.

    Returns:
        tuple: (point_estimate, standard_error, lambda_hat)
    """
    Y = np.asarray(Y).reshape(-1)
    Y_hat = np.asarray(Y_hat).reshape(-1)
    Y_hat_unlabeled = np.asarray(Y_hat_unlabeled).reshape(-1)

    n = Y.size
    N = Y_hat_unlabeled.size

    if n == 0 or N == 0:
        raise ValueError("Y and Y_hat_unlabeled must be non-empty.")
    if Y_hat.size != n:
        raise ValueError("Y and Y_hat must have the same length.")

    if cluster_labels_labeled is None:
        cluster_labels_labeled = np.arange(n)
    else:
        cluster_labels_labeled = np.asarray(cluster_labels_labeled).reshape(-1)
        if cluster_labels_labeled.size != n:
            raise ValueError(
                "cluster_labels_labeled must have the same length as Y."
            )

    if cluster_labels_unlabeled is None:
        cluster_labels_unlabeled = np.arange(N)
    else:
        cluster_labels_unlabeled = np.asarray(
            cluster_labels_unlabeled
        ).reshape(-1)
        if cluster_labels_unlabeled.size != N:
            raise ValueError(
                "cluster_labels_unlabeled must have the same length as Y_hat_unlabeled."
            )

    theta_hat = Y.mean()
    theta_f_hat = np.concatenate([Y_hat, Y_hat_unlabeled]).mean()

    e = np.column_stack((Y - theta_hat, Y_hat - theta_f_hat))
    e_tilde = (Y_hat_unlabeled - theta_f_hat).reshape(-1, 1)

    e_cov = sw.S_crosssection(e, cluster_labels_labeled)
    e_tilde_cov = sw.S_crosssection(e_tilde, cluster_labels_unlabeled)

    n2 = float(n**2)
    N2 = float(N**2)

    A = e_cov[0, 0] / n2
    B = e_cov[0, 1] / n2
    C = e_cov[1, 1] / n2 + e_tilde_cov[0, 0] / N2

    tol = 1e-12
    if C <= tol:
        raise ValueError(
            "Degenerate lambda denominator: cannot compute lambda_hat."
        )

    lambda_hat = B / C
    point_estimate = np.mean(Y - lambda_hat * Y_hat) + lambda_hat * np.mean(
        Y_hat_unlabeled
    )

    var_hat = A - (B**2) / C
    var_hat = max(var_hat, 0.0)
    standard_error = np.sqrt(var_hat)
    print(lambda_hat)
    return float(point_estimate), float(standard_error)


def simulate_clustered_data(
    theta=theta,
    cluster_size=cluster_size,
    num_clusters_labeled=num_clusters_labeled,
    num_clusters_unlabeled=num_clusters_unlabeled,
    rho=rho,
    ppi_correlation=ppi_correlation,
    seed=0,
):
    rng = np.random.default_rng(seed)

    sigma = (1 - rho) * np.eye(cluster_size) + rho * np.ones(
        (cluster_size, cluster_size)
    )
    mean = np.full(cluster_size, theta)

    # One multivariate Gaussian draw per cluster (dimension = cluster_size).
    Y_clustered = rng.multivariate_normal(
        mean=mean, cov=sigma, size=num_clusters_labeled
    )
    Y_unlabeled_clustered = rng.multivariate_normal(
        mean=mean, cov=sigma, size=num_clusters_unlabeled
    )

    noise_sd = np.sqrt(1 - ppi_correlation**2)
    Y_hat_clustered = ppi_correlation * Y_clustered + rng.normal(
        loc=0.0, scale=noise_sd, size=Y_clustered.shape
    )
    Y_hat_unlabeled_clustered = (
        ppi_correlation * Y_unlabeled_clustered
        + rng.normal(loc=0.0, scale=noise_sd, size=Y_unlabeled_clustered.shape)
    )

    # Flatten to observation-level vectors.
    Y = Y_clustered.reshape(-1)
    Y_hat = Y_hat_clustered.reshape(-1)
    Y_hat_unlabeled = Y_hat_unlabeled_clustered.reshape(-1)

    # Distinct cluster labels with one label per flattened observation.
    cluster_labels_labeled = np.repeat(
        np.arange(num_clusters_labeled), cluster_size
    )
    cluster_labels_unlabeled = np.repeat(
        np.arange(
            num_clusters_labeled, num_clusters_labeled + num_clusters_unlabeled
        ),
        cluster_size,
    )

    return {
        "Y": Y,
        "Y_hat": Y_hat,
        "Y_hat_unlabeled": Y_hat_unlabeled,
        "cluster_labels_labeled": cluster_labels_labeled,
        "cluster_labels_unlabeled": cluster_labels_unlabeled,
    }


def run_coverage_simulation(
    num_simulations=1000,
    alpha=0.05,
    theta_true=theta,
    simulation_seed=0,
):
    """
    Estimate CI coverage for cluster-robust SE vs naive SE (ignoring clustering).

    Returns:
        dict: Coverage rates plus per-simulation arrays of estimates.
    """
    z_value = NormalDist().inv_cdf(1 - alpha / 2)

    point_estimates_cluster = np.empty(num_simulations)
    standard_errors_cluster = np.empty(num_simulations)
    ci_lower_cluster = np.empty(num_simulations)
    ci_upper_cluster = np.empty(num_simulations)
    covered_cluster = np.empty(num_simulations, dtype=bool)

    point_estimates_naive = np.empty(num_simulations)
    standard_errors_naive = np.empty(num_simulations)
    ci_lower_naive = np.empty(num_simulations)
    ci_upper_naive = np.empty(num_simulations)
    covered_naive = np.empty(num_simulations, dtype=bool)

    for s in range(num_simulations):
        sim_data = simulate_clustered_data(seed=simulation_seed + s)

        Y = sim_data["Y"]
        Y_hat = sim_data["Y_hat"]
        Y_hat_unlabeled = sim_data["Y_hat_unlabeled"]
        cluster_labels_labeled = sim_data["cluster_labels_labeled"]
        cluster_labels_unlabeled = sim_data["cluster_labels_unlabeled"]

        point_estimate_c, standard_error_c = cluster_ppi_mean_stats(
            Y,
            Y_hat,
            Y_hat_unlabeled,
            cluster_labels_labeled,
            cluster_labels_unlabeled,
        )
        point_estimate_n, standard_error_n = cluster_ppi_mean_stats(
            Y,
            Y_hat,
            Y_hat_unlabeled,
            cluster_labels_labeled=None,
            cluster_labels_unlabeled=None,
        )

        lower_c = point_estimate_c - z_value * standard_error_c
        upper_c = point_estimate_c + z_value * standard_error_c
        lower_n = point_estimate_n - z_value * standard_error_n
        upper_n = point_estimate_n + z_value * standard_error_n

        point_estimates_cluster[s] = point_estimate_c
        standard_errors_cluster[s] = standard_error_c
        ci_lower_cluster[s] = lower_c
        ci_upper_cluster[s] = upper_c
        covered_cluster[s] = lower_c <= theta_true <= upper_c

        point_estimates_naive[s] = point_estimate_n
        standard_errors_naive[s] = standard_error_n
        ci_lower_naive[s] = lower_n
        ci_upper_naive[s] = upper_n
        covered_naive[s] = lower_n <= theta_true <= upper_n

    coverage_cluster = covered_cluster.mean()
    coverage_naive = covered_naive.mean()

    return {
        "coverage_cluster": coverage_cluster,
        "coverage_naive": coverage_naive,
        "point_estimates_cluster": point_estimates_cluster,
        "standard_errors_cluster": standard_errors_cluster,
        "ci_lower_cluster": ci_lower_cluster,
        "ci_upper_cluster": ci_upper_cluster,
        "covered_cluster": covered_cluster,
        "point_estimates_naive": point_estimates_naive,
        "standard_errors_naive": standard_errors_naive,
        "ci_lower_naive": ci_lower_naive,
        "ci_upper_naive": ci_upper_naive,
        "covered_naive": covered_naive,
    }


sim_data = simulate_clustered_data(seed=1)
Y = sim_data["Y"]
Y_hat = sim_data["Y_hat"]
Y_hat_unlabeled = sim_data["Y_hat_unlabeled"]
cluster_labels_labeled = sim_data["cluster_labels_labeled"]
cluster_labels_unlabeled = sim_data["cluster_labels_unlabeled"]

point_estimate, standard_error = cluster_ppi_mean_stats(
    Y,
    Y_hat,
    Y_hat_unlabeled,
    cluster_labels_labeled,
    cluster_labels_unlabeled,
)
print(
    f"Cluster-robust point estimate: {point_estimate:.3f}, SE: {standard_error:.3f}"
)

# grads = Y - np.mean(Y)
# cov_cluster_labeled = cov_cluster(grads, cluster_labels_labeled)
# cov_naive = cov_cluster(grads, None)
# print(cov_cluster_labeled.shape)
# print(cov_naive)

ppi_point_estimate = float(ppi_mean_pointestimate(
    Y,
    Y_hat,
    Y_hat_unlabeled,
    group=cluster_labels_labeled,
    group_unlabeled=cluster_labels_unlabeled,
))

ppi_point_estimate_naive = float(ppi_mean_pointestimate(
    Y,
    Y_hat,
    Y_hat_unlabeled,
    group=None,
    group_unlabeled=None,
))

print(f"PPI point estimate: {ppi_point_estimate:.3f}")
print(f"PPI point estimate (naive): {ppi_point_estimate_naive:.3f}")

conf_int = ppi_mean_ci(
    Y,
    Y_hat,
    Y_hat_unlabeled,
    alpha=0.05,
    group=cluster_labels_labeled,
    group_unlabeled=cluster_labels_unlabeled,
)

conf_int_naive = ppi_mean_ci(
    Y,
    Y_hat,
    Y_hat_unlabeled,
    alpha=0.05,
    group=None,
    group_unlabeled=None,
)
print(f"PPI 95% CI: [{float(conf_int[0]):.3f}, {float(conf_int[1]):.3f}]")
print(f"Other CI: [{point_estimate - 1.96 * standard_error:.3f}, {point_estimate + 1.96 * standard_error:.3f}]")
print(f"PPI 95% CI (naive): [{float(conf_int_naive[0]):.3f}, {float(conf_int_naive[1]):.3f}]")



# if __name__ == "__main__":
#     results = run_coverage_simulation(
#         theta_true=theta, num_simulations=1000, alpha=0.05
#     )
#     print(f"Cluster-robust coverage: {results['coverage_cluster']:.3f}")
#     print(f"Naive coverage: {results['coverage_naive']:.3f}")
#     print(
#         f"RMSE (cluster-robust): {(((results['point_estimates_cluster']-theta)**2).mean())**0.5:.3f}"
#     )
#     print(
#         f"RMSE (naive): {(((results['point_estimates_naive']-theta)**2).mean())**0.5:.3f}"
#     )
#     print(
#         f"Average se (cluster-robust): {results['standard_errors_cluster'].mean():.3f}"
#     )
#     print(f"Average se (naive): {results['standard_errors_naive'].mean():.3f}")
