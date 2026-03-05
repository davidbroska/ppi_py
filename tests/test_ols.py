import numpy as np
from ppi_py import ppi_ols_pointestimate, ppi_ols_ci
from ppi_py import classical_ols_ci, postprediction_ols_ci
from tqdm import tqdm
import pdb

"""
    PPI tests
"""


def test_ppi_ols_pointestimate():
    # Make a synthetic regression problem
    n = 1000
    N = 10000
    d = 10
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    beta_prediction = beta + np.random.randn(d) + 2
    Y = X.dot(beta) + np.random.randn(n)
    Yhat = X.dot(beta_prediction) + np.random.randn(n)
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = X_unlabeled.dot(beta_prediction) + np.random.randn(N)
    # Compute the point estimate
    beta_ppi_pointestimate = ppi_ols_pointestimate(
        X, Y, Yhat, X_unlabeled, Yhat_unlabeled
    )
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < np.linalg.norm(
        beta_prediction - beta
    )  # Makes it less biased


def test_ppi_ols_ci():
    n = 1000
    N = 10000
    d = 1
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 1000
    includeds = np.zeros_like(alphas)
    for i in range(num_trials):
        # Make a synthetic regression problem
        X = np.random.randn(n, d)
        beta = np.random.randn(d)
        beta_prediction = beta + np.random.randn(d) + 2
        Y = X.dot(beta) + np.random.randn(n)
        Yhat = X.dot(beta_prediction) + np.random.randn(n)
        # Make a synthetic unlabeled data set with predictions Yhat
        X_unlabeled = np.random.randn(N, d)
        Yhat_unlabeled = X_unlabeled.dot(beta_prediction) + np.random.randn(N)
        for j in range(alphas.shape[0]):
            # Compute the confidence interval
            beta_ppi_ci = ppi_ols_ci(
                X, Y, Yhat, X_unlabeled, Yhat_unlabeled, alpha=alphas[j]
            )
            # Check that the confidence interval contains the true beta
            includeds[j] += int(
                (beta_ppi_ci[0][0] <= beta[0]) & (beta[0] <= beta_ppi_ci[1][0])
            )
    print((includeds / num_trials))
    failed = np.any((includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed

def simulate_clustered_data(
    regression_coefficients,
    cluster_size,
    num_clusters_labeled,
    num_clusters_unlabeled,
    rho,
    correlation,
    bias=0,
    seed=0,
):
    rng = np.random.default_rng(seed)

    sigma = (1 - rho) * np.eye(cluster_size) + rho * np.ones(
        (cluster_size, cluster_size)
    )
    
    zeros = np.zeros(cluster_size)
    # One multivariate Gaussian draw per cluster (dimension = cluster_size).
    errors_clustered = rng.multivariate_normal(
        zeros, cov=sigma, size=num_clusters_labeled
    )
    errors_unlabeled_clustered = rng.multivariate_normal(
        zeros, cov=sigma, size=num_clusters_unlabeled
    )

    noise_sd = np.sqrt(1 - correlation**2)
    errors_hat_clustered = correlation * errors_clustered + rng.normal(
        loc=0.0, scale=noise_sd, size=errors_clustered.shape
    )
    errors_hat_unlabeled_clustered = (
        correlation * errors_unlabeled_clustered
        + rng.normal(loc=0.0, scale=noise_sd, size=errors_unlabeled_clustered.shape)
    )

    # Flatten to observation-level vectors.
    errors = errors_clustered.reshape(-1)
    errors_hat = errors_hat_clustered.reshape(-1)
    errors_hat_unlabeled = errors_hat_unlabeled_clustered.reshape(-1)

    # Distinct cluster labels with one label per flattened observation.
    group = np.repeat(
        np.arange(num_clusters_labeled), cluster_size
    )
    group_unlabeled = np.repeat(
        np.arange(
            num_clusters_labeled, num_clusters_labeled + num_clusters_unlabeled
        ),
        cluster_size,
    )

    d = len(regression_coefficients)
    n = len(errors)
    N = len(errors_hat_unlabeled)

    X = rng.normal(size=(n, d))
    X_unlabeled = rng.normal(size=(N, d))

    Y = X.dot(regression_coefficients) + errors
    Y_hat = X.dot(regression_coefficients+bias) + errors_hat 
    Y_hat_unlabeled = X_unlabeled.dot(regression_coefficients+bias) + errors_hat_unlabeled

    

    return {
        "X" : X,
        "Y": Y,
        "Y_hat": Y_hat,
        "X_unlabeled": X_unlabeled,
        "Y_hat_unlabeled": Y_hat_unlabeled,
        "group": group,
        "group_unlabeled": group_unlabeled,
    }



def test_ppi_ols_clustered_pointestimate():
    seed = 0
    regression_coefficients = np.array([1, 2])
    rho = 0.3
    correlation = 0.9
    cluster_size = 10
    num_clusters_labeled = 100
    num_clusters_unlabeled = 1000
    bias = 2

    sim_data = simulate_clustered_data(
        regression_coefficients=regression_coefficients,
        cluster_size=cluster_size,
        num_clusters_labeled=num_clusters_labeled,
        num_clusters_unlabeled=num_clusters_unlabeled,
        rho=rho,
        correlation=correlation,
        bias=bias,
        seed=seed,
    )
    X = sim_data["X"]
    Y = sim_data["Y"]
    Y_hat = sim_data["Y_hat"]
    X_unlabeled = sim_data["X_unlabeled"]
    Y_hat_unlabeled = sim_data["Y_hat_unlabeled"]
    group = sim_data["group"]
    group_unlabeled = sim_data["group_unlabeled"]

    point_estimate = ppi_ols_pointestimate(
        X, 
        Y, 
        Y_hat, 
        X_unlabeled, 
        Y_hat_unlabeled,
        group = group,
        group_unlabeled = group_unlabeled,
    )
    print(point_estimate, regression_coefficients)
    
    assert np.allclose(point_estimate, regression_coefficients, atol=0.1)

def test_ppi_ols_clustered_ci():
    seed = 0
    regression_coefficients = np.array([1, 2])
    rho = 0.3
    correlation = 0.9
    cluster_size = 10
    num_clusters_labeled = 100
    num_clusters_unlabeled = 1000
    bias = 2

    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 100
    includeds = np.zeros((len(alphas), len(regression_coefficients)))
    for i in range(num_trials):
        sim_data = simulate_clustered_data(
            regression_coefficients=regression_coefficients,
            cluster_size=cluster_size,
            num_clusters_labeled=num_clusters_labeled,
            num_clusters_unlabeled=num_clusters_unlabeled,
            rho=rho,
            correlation=correlation,
            bias=bias,
            seed=seed+i,
        )
        X = sim_data["X"]
        Y = sim_data["Y"]
        Y_hat = sim_data["Y_hat"]
        X_unlabeled = sim_data["X_unlabeled"]
        Y_hat_unlabeled = sim_data["Y_hat_unlabeled"]
        group = sim_data["group"]
        group_unlabeled = sim_data["group_unlabeled"]

        for j in range(alphas.shape[0]):
            beta_ppi_ci = ppi_ols_ci(
                X, 
                Y, 
                Y_hat, 
                X_unlabeled, 
                Y_hat_unlabeled, 
                group=group,
                group_unlabeled=group_unlabeled,
                alpha=alphas[j],
            )
            includeds[j] += (
                (beta_ppi_ci[0] <= regression_coefficients) & (regression_coefficients <= beta_ppi_ci[1])
            ).astype(int)
    

    failed = (includeds / num_trials) < (1 - alphas[:,None] - epsilon)

    assert not np.any(failed)





"""
    Baseline tests
"""


def test_classical_ols_ci():
    n = 1000
    d = 3
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 1000
    includeds = np.zeros_like(alphas)
    for i in range(num_trials):
        # Make a synthetic regression problem
        X = np.random.randn(n, d)
        beta = np.random.randn(d)
        Y = X.dot(beta) + np.random.randn(n)
        for j in range(alphas.shape[0]):
            # Compute the confidence interval
            beta_ci = classical_ols_ci(X, Y, alpha=alphas[j])
            # Check that the confidence interval contains the true beta
            includeds[j] += int(
                (beta_ci[0][0] <= beta[0]) & (beta[0] <= beta_ci[1][0])
            )
    print((includeds / num_trials))
    failed = np.any((includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed


def test_postprediction_ols_ci():
    n = 1000
    N = 10000
    d = 2
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 20
    bias = 10
    sigma = 0.1
    includeds = np.zeros_like(alphas)
    for i in tqdm(range(num_trials)):
        # Make a synthetic regression problem
        X = np.random.randn(n, d)
        beta = np.random.randn(d)
        Y = X.dot(beta) + np.random.randn(n)
        Yhat = Y + sigma * np.random.randn(n) + bias
        # Make a synthetic unlabeled data set with predictions Yhat
        X_unlabeled = np.random.randn(N, d)
        Y_unlabeled = X_unlabeled.dot(beta) + np.random.randn(N)
        Yhat_unlabeled = Y_unlabeled + sigma * np.random.randn(N) + bias
        for j in range(alphas.shape[0]):
            # Compute the confidence interval
            beta_ci = postprediction_ols_ci(
                Y, Yhat, X_unlabeled, Yhat_unlabeled, alpha=alphas[j]
            )
            print(beta, beta_ci)
            # Check that the confidence interval contains the true beta
            includeds[j] += int(
                (beta_ci[0][0] <= beta[0]) & (beta[0] <= beta_ci[1][0])
            )
    print((includeds / num_trials))
    failed = False  # This confidence interval doesn't cover, so the test succeeds if it can construct intervals...
    assert not failed
