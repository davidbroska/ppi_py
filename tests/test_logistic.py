import numpy as np
from scipy.special import expit
from ppi_py import (
    ppi_logistic_pointestimate,
    ppi_logistic_ci,
    ppi_logistic_pval,
)
from ppi_py import classical_logistic_ci
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
    PPI tests
"""


def test_ppi_logistic_pointestimate_debias():
    # Make a synthetic regression problem
    n = 100
    N = 1000
    d = 2
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    beta_prediction = beta + np.random.randn(d) + 2
    Y = expit(X.dot(beta) + np.random.randn(n))
    Yhat = expit(X.dot(beta_prediction) + np.random.randn(n))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = expit(
        X_unlabeled.dot(beta_prediction) + np.random.randn(N)
    )
    # Compute the point estimate
    beta_ppi_pointestimate = ppi_logistic_pointestimate(
        X,
        (Y > 0.5).astype(int),
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options={"gtol": 1e-3},
    )
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < np.linalg.norm(
        beta_prediction - beta
    )  # Makes it less biased


def test_ppi_logistic_pointestimate_recovers():
    # Make a synthetic regression problem
    n = 10000
    N = 100000
    d = 3
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    Y = np.random.binomial(1, expit(X.dot(beta)))
    Yhat = expit(X.dot(beta))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = expit(X_unlabeled.dot(beta))
    # Compute the point estimate
    beta_ppi_pointestimate = ppi_logistic_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options={"gtol": 1e-3},
    )
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < 0.2


def test_ppi_logistic_pval_makesense():
    # Make a synthetic regression problem
    n = 10000
    N = 100000
    d = 3
    X = np.random.randn(n, d)
    beta = np.array([0, 0, 1.0])

    Y = np.random.binomial(1, expit(X.dot(beta)))
    Yhat = expit(X.dot(beta))
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = expit(X_unlabeled.dot(beta))
    beta_ppi_pval = ppi_logistic_pval(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lam=0.5,
        optimizer_options={"gtol": 1e-3},
    )
    assert beta_ppi_pval[-1] < 0.1

    beta_ppi_pval = ppi_logistic_pval(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lam=None,
        optimizer_options={"gtol": 1e-3},
    )
    assert beta_ppi_pval[-1] < 0.1


def ppi_logistic_ci_subtest(i, alphas, n=1000, N=10000, d=1, epsilon=0.02):
    includeds = np.zeros(len(alphas))
    # Make a synthetic regression problem
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    beta_prediction = beta + np.random.randn(d) + 2
    Y = np.random.binomial(1, expit(X.dot(beta)))
    Yhat = expit(X.dot(beta_prediction))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = expit(X_unlabeled.dot(beta_prediction))
    # Compute the confidence interval
    for j in range(len(alphas)):
        beta_ppi_pointestimate = ppi_logistic_pointestimate(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            optimizer_options={"gtol": 1e-3},
        )
        beta_ppi_ci = ppi_logistic_ci(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            alpha=alphas[j],
            optimizer_options={"gtol": 1e-3},
        )
        # Check that the confidence interval contains the true beta
        includeds[j] += int(
            (beta_ppi_ci[0][0] <= beta[0]) & (beta[0] <= beta_ppi_ci[1][0])
        )
    return includeds


def test_ppi_logistic_ci_parallel():
    n = 1000
    N = 10000
    d = 2
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.1
    num_trials = 10

    total_includeds = np.zeros(len(alphas))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                ppi_logistic_ci_subtest, i, alphas, n, N, d, epsilon
            )
            for i in range(num_trials)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            total_includeds += future.result()

    print((total_includeds / num_trials))
    failed = np.any((total_includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed


"""
    Cluster tests
"""


def simulate_clustered_data(
    regression_coefficients,
    cluster_size,
    num_clusters_labeled,
    num_clusters_unlabeled,
    Yhat_accuracy,
    seed=0,
):
    rng = np.random.default_rng(seed)

    # Distinct cluster labels with one label per flattened observation.
    group = np.repeat(np.arange(num_clusters_labeled), cluster_size)
    group_unlabeled = np.repeat(
        np.arange(
            num_clusters_labeled, num_clusters_labeled + num_clusters_unlabeled
        ),
        cluster_size,
    )

    d = len(regression_coefficients)
    n = len(group)
    N = len(group_unlabeled)

    X = rng.normal(size=(n, d))
    X_unlabeled = rng.normal(size=(N, d))

    eta = X.dot(regression_coefficients)
    eta_unlabeled = X_unlabeled.dot(regression_coefficients)

    Y = np.random.binomial(1, expit(eta))
    Y_noise_cluster = np.random.binomial(
        1, p=Yhat_accuracy, size=num_clusters_labeled+num_clusters_unlabeled
    )
    Y_noise = Y_noise_cluster[group]
    Y_hat = Y * Y_noise + (1 - Y) * (1 - Y_noise)
    Y_unlabeled = np.random.binomial(1, expit(eta_unlabeled))
    Y_unlabeled_noise = Y_noise_cluster[group_unlabeled]
    Y_hat_unlabeled = Y_unlabeled * Y_unlabeled_noise + (1 - Y_unlabeled) * (
        1 - Y_unlabeled_noise
    )

    return {
        "X": X,
        "Y": Y,
        "Y_hat": Y_hat,
        "X_unlabeled": X_unlabeled,
        "Y_hat_unlabeled": Y_hat_unlabeled,
        "group": group,
        "group_unlabeled": group_unlabeled,
    }


def test_ppi_logistic_clustered_pointestimate():
    seed = 0
    regression_coefficients = np.array([1, 2])
    cluster_size = 5
    num_clusters_labeled = 1000
    num_clusters_unlabeled = 2000
    Yhat_accuracy = 0.8
    data = simulate_clustered_data(
        regression_coefficients=regression_coefficients,
        cluster_size=cluster_size,
        num_clusters_labeled=num_clusters_labeled,
        num_clusters_unlabeled=num_clusters_unlabeled,
        Yhat_accuracy=Yhat_accuracy,
        seed=seed,
    )
    X = data["X"]
    Y = data["Y"]
    Y_hat = data["Y_hat"]
    X_unlabeled = data["X_unlabeled"]
    Y_hat_unlabeled = data["Y_hat_unlabeled"]
    group = data["group"]
    group_unlabeled = data["group_unlabeled"]
    point_estimate = ppi_logistic_pointestimate(
        X,
        Y,
        Y_hat,
        X_unlabeled,
        Y_hat_unlabeled,
        group=group,
        group_unlabeled=group_unlabeled,
        optimizer_options={"gtol": 1e-3},
    )

    print(point_estimate, regression_coefficients)

    assert np.allclose(point_estimate, regression_coefficients, atol=0.1)


def ppi_logistic_clustered_ci_subtest(
    i,
    alphas,
    regression_coefficients,
    cluster_size,
    num_clusters_labeled,
    num_clusters_unlabeled,
    Yhat_accuracy,
    seed=0,
):
    data = simulate_clustered_data(
        regression_coefficients=regression_coefficients,
        cluster_size=cluster_size,
        num_clusters_labeled=num_clusters_labeled,
        num_clusters_unlabeled=num_clusters_unlabeled,
        Yhat_accuracy=Yhat_accuracy,
        seed=seed + i,
    )
    X = data["X"]
    Y = data["Y"]
    Y_hat = data["Y_hat"]
    X_unlabeled = data["X_unlabeled"]
    Y_hat_unlabeled = data["Y_hat_unlabeled"]
    group = data["group"]
    group_unlabeled = data["group_unlabeled"]
    includeds = np.zeros((len(alphas), len(regression_coefficients)))
    for j in range(len(alphas)):
        ci = ppi_logistic_ci(
            X,
            Y,
            Y_hat,
            X_unlabeled,
            Y_hat_unlabeled,
            group=group,
            group_unlabeled=group_unlabeled,
            alpha=alphas[j],
            optimizer_options={"gtol": 1e-3},
        )
        includeds[j] = (
            (ci[0] <= regression_coefficients)
            & (regression_coefficients <= ci[1])
        ).astype(int)
    return includeds


def test_ppi_logistic_clustered_ci_parallel():
    seed = 0
    regression_coefficients = np.array([1, 2])
    cluster_size = 5
    num_clusters_labeled = 1000
    num_clusters_unlabeled = 2000
    Yhat_accuracy = 0.8
    alphas = np.array([0.05, 0.1, 0.2, 0.5])
    epsilon = 0.1
    num_trials = 100

    total_includeds = np.zeros((len(alphas), len(regression_coefficients)))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                ppi_logistic_clustered_ci_subtest,
                i,
                alphas,
                regression_coefficients,
                cluster_size,
                num_clusters_labeled,
                num_clusters_unlabeled,
                Yhat_accuracy,
                seed,
            )
            for i in range(num_trials)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            total_includeds += future.result()

    print((total_includeds / num_trials))
    failed = (total_includeds / num_trials) < (1 - alphas[:, None] - epsilon)
    print(failed)
    assert not np.any(failed)


"""
    Baseline tests
"""


def classical_logistic_ci_subtest(i, alphas, n, d, epsilon):
    includeds = np.zeros(len(alphas))
    # Make a synthetic regression problem
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    Y = np.random.binomial(1, expit(X.dot(beta)))
    # Compute the confidence interval
    for j in range(len(alphas)):
        beta_ci = classical_logistic_ci(X, Y, alpha=alphas[j])
        # Check that the confidence interval contains the true beta
        includeds[j] += int(
            (beta_ci[0][0] <= beta[0]) & (beta[0] <= beta_ci[1][0])
        )
    return includeds


def test_classical_logistic_ci_parallel():
    n = 1000
    d = 2
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 200

    total_includeds = np.zeros(len(alphas))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                classical_logistic_ci_subtest, i, alphas, n, d, epsilon
            )
            for i in range(num_trials)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            total_includeds += future.result()

    print((total_includeds / num_trials))
    failed = np.any((total_includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed
