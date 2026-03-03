import numpy as np
from statistics import NormalDist
import statsmodels.stats.sandwich_covariance as sw
from ppi_py.ppi import ppi_mean_pointestimate, ppi_mean_ci, _calc_lam_glm, sandwich_cov_glm
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
    var_hat = sw.S_crosssection(e[:, 0] - lambda_hat * e[:, 1], cluster_labels_labeled) / n2 + sw.S_crosssection(lambda_hat * e_tilde, cluster_labels_unlabeled) / N2
    standard_error = np.sqrt(var_hat)
    return float(point_estimate), float(standard_error), lambda_hat


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


sim_data = simulate_clustered_data(seed=10)
Y = sim_data["Y"]
Y_hat = sim_data["Y_hat"]
Y_hat_unlabeled = sim_data["Y_hat_unlabeled"]
cluster_labels_labeled = sim_data["cluster_labels_labeled"]
cluster_labels_unlabeled = sim_data["cluster_labels_unlabeled"]

theta_hat = Y.mean()
theta_f_hat = np.concatenate([Y_hat, Y_hat_unlabeled]).mean()

e = np.column_stack((Y - theta_hat, Y_hat - theta_f_hat))
e_tilde = (Y_hat_unlabeled - theta_f_hat).reshape(-1, 1)

e_cov = sw.S_crosssection(e, cluster_labels_labeled)
e_tilde_cov = sw.S_crosssection(e_tilde, cluster_labels_unlabeled)

print(e_cov)
print(e_tilde_cov)

e_cov_ppi = cov_cluster(e, cluster_labels_labeled)
e_tilde_cov_ppi = cov_cluster(e_tilde, cluster_labels_unlabeled)
print(e_cov_ppi)
print(e_tilde_cov_ppi)

point_estimate, standard_error, lambda_hat = cluster_ppi_mean_stats(
    Y,
    Y_hat,
    Y_hat_unlabeled,
    cluster_labels_labeled=cluster_labels_labeled,
    cluster_labels_unlabeled=cluster_labels_unlabeled,
)
print(f"Lambda Hat: {lambda_hat:.4f}")
print(f"Point Estimate: {point_estimate:.4f}")
print(f"Standard Error: {standard_error:.4f}")

ppi_point_estimate = ppi_mean_pointestimate(
    Y,
    Y_hat,
    Y_hat_unlabeled,
    group=cluster_labels_labeled,
    group_unlabeled=cluster_labels_unlabeled,
)

ppi_ci_lower, ppi_ci_upper = ppi_mean_ci(
    Y,
    Y_hat,
    Y_hat_unlabeled,
    group=cluster_labels_labeled,
    group_unlabeled=cluster_labels_unlabeled,
    alpha=0.05,
)
H = np.eye(1)

lam = _calc_lam_glm(
    Y,
    Y_hat,
    Y_hat_unlabeled,
    inv_hessian=H,
    group=cluster_labels_labeled,
    group_unlabeled=cluster_labels_unlabeled,
)
Sigma = sandwich_cov_glm(
    Y,
    Y_hat,
    Y_hat_unlabeled,
    lam = lambda_hat,
    inv_hessian=H,
    group=cluster_labels_labeled,
    group_unlabeled=cluster_labels_unlabeled,
)
print(f"Lambda GLM: {lam:.4f}")
print(f"PPI Point Estimate: {float(ppi_point_estimate):.4f}")
print(f"PPI 95% CI: [{float(ppi_ci_lower):.4f}, {float(ppi_ci_upper):.4f}]")
print(f"GLM Sandwich Covariance: {float(Sigma**0.5):.4f}")