import numpy as np
from ppi_py import *

"""
    PPI tests
"""


def test_ppi_mean_pointestimate():
    Y = np.random.normal(0, 1, 100)
    Yhat = Y + 2
    Yhat_unlabeled = np.ones(10000) * 2
    theta_hat = ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled)
    assert np.isclose(theta_hat, 0, atol=1e-6)


def test_ppi_mean_ci():
    trials = 100
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    includeds = np.zeros_like(alphas)
    for i in range(trials):
        Y = np.random.normal(0, 1, 10000)
        Yhat = np.random.normal(-2, 1, 10000)
        Yhat_unlabeled = np.random.normal(-2, 1, 10000)
        for j in range(alphas.shape[0]):
            ci = ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=alphas[j])
            if ci[0] <= 0 and ci[1] >= 0:
                includeds[j] += 1
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed


def test_ppi_mean_multid():
    trials = 100
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    n_alphas = alphas.shape[0]
    n_dims = 5
    epsilon = 0.1
    includeds = np.zeros((n_alphas, n_dims))
    for _ in range(trials):
        Y = np.random.normal(0, 1, (10000, n_dims))
        Yhat = Y + np.random.normal(-2, 1, (10000, n_dims))
        Yhat_unlabeled = np.random.normal(-2, 2**0.5, (10000, n_dims))
        for j in range(alphas.shape[0]):
            ci = ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=alphas[j])

            included = (ci[0] <= 0) & (ci[1] >= 0)
            includeds[j] += included.astype(int)
    failed = (includeds / trials) < 1 - alphas[:,None] - epsilon
    assert not np.any(failed)


def test_ppi_mean_elem():
    alpha = 0.1
    Y = np.random.normal(0, 1, 10000)
    Yhat = np.random.normal(-2, 1, 10000)
    Yhat_unlabeled = np.random.normal(-2, 1, 10000)

    ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled, lam_optim_mode="element")
    ppi_mean_ci(
        Y, Yhat, Yhat_unlabeled, alpha=alpha, lam_optim_mode="element"
    )
    ppi_mean_pval(Y, Yhat, Yhat_unlabeled, lam_optim_mode="element")

    Y = np.random.normal(0, 1, (10000, 5))
    Yhat = np.random.normal(-2, 1, (10000, 5))
    Yhat_unlabeled = np.random.normal(-2, 1, (10000, 5))

    ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled, lam_optim_mode="element")
    ppi_mean_ci(
        Y, Yhat, Yhat_unlabeled, alpha=alpha, lam_optim_mode="element"
    )
    ppi_mean_pval(Y, Yhat, Yhat_unlabeled, lam_optim_mode="element")


def test_ppi_mean_pval():
    trials = 1000
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.05
    failed = False
    rejected = np.zeros_like(alphas)
    for i in range(trials):
        Y = np.random.normal(0, 1, 10000)
        Yhat = np.random.normal(-2, 1, 10000)
        Yhat_unlabeled = np.random.normal(-2, 1, 10000)
        pval = ppi_mean_pval(Y, Yhat, Yhat_unlabeled, null=0)
        rejected += pval < alphas
    failed = rejected / trials > alphas + epsilon
    assert not np.any(failed)


"""
    Classical tests
"""


def test_classical_mean_ci():
    trials = 1000
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.05
    includeds = np.zeros_like(alphas)
    for i in range(trials):
        Y = np.random.normal(0, 1, 10000)
        for j in range(alphas.shape[0]):
            ci = classical_mean_ci(Y, alpha=alphas[j])
            if ci[0] <= 0 and ci[1] >= 0:
                includeds[j] += 1
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed


def test_semisupervised_mean_ci():
    trials = 100
    K = 5
    d = 2  # Dimension of _features_ (not mean)
    n = 1000
    N = 10000
    sigma = 1
    mu = 10
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    includeds = np.zeros_like(alphas)
    for i in range(trials):
        X = np.random.normal(0, 1, size=(n, d))
        X_unlabeled = np.random.normal(0, 1, size=(N, d))
        theta = np.random.normal(0, 1, size=d)
        theta /= np.linalg.norm(theta)
        Y = X.dot(theta) + sigma * np.random.normal(0, 1, n) + mu
        for j in range(alphas.shape[0]):
            ci = semisupervised_mean_ci(X, Y, X_unlabeled, K, alpha=alphas[j])
            if ci[0] <= mu and ci[1] >= mu:
                includeds[j] += 1
    print(includeds / trials)
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed


def test_conformal_mean_ci():
    trials = 100
    n = 1000
    N = 10000
    bias = 5
    sigma = 10
    mu = 10
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    includeds = np.zeros_like(alphas)
    for i in range(trials):
        Y = np.random.normal(mu, 1, n)
        Yhat = Y + sigma * np.random.normal(0, 1, n) + bias
        Y_unlabeled = np.random.normal(mu, 1, N)
        Yhat_unlabeled = Y_unlabeled + sigma * np.random.normal(0, 1, N) + bias
        for j in range(alphas.shape[0]):
            ci = conformal_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=alphas[j])
            if ci[0] <= mu and ci[1] >= mu:
                includeds[j] += 1
    print(includeds / trials)
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed

def simulate_clustered_data(
    theta,
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
    mean = np.full(cluster_size, theta)

    # One multivariate Gaussian draw per cluster (dimension = cluster_size).
    Y_clustered = rng.multivariate_normal(
        mean=mean, cov=sigma, size=num_clusters_labeled
    )
    Y_unlabeled_clustered = rng.multivariate_normal(
        mean=mean, cov=sigma, size=num_clusters_unlabeled
    )

    noise_sd = np.sqrt(1 - correlation**2)
    Y_hat_clustered = correlation * Y_clustered + rng.normal(
        loc=bias, scale=noise_sd, size=Y_clustered.shape
    )
    Y_hat_unlabeled_clustered = (
        correlation * Y_unlabeled_clustered
        + rng.normal(loc=bias, scale=noise_sd, size=Y_unlabeled_clustered.shape)
    )

    # Flatten to observation-level vectors.
    Y = Y_clustered.reshape(-1)
    Y_hat = Y_hat_clustered.reshape(-1)
    Y_hat_unlabeled = Y_hat_unlabeled_clustered.reshape(-1)

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

    return {
        "Y": Y,
        "Y_hat": Y_hat,
        "Y_hat_unlabeled": Y_hat_unlabeled,
        "group": group,
        "group_unlabeled": group_unlabeled,
    }


def test_ppi_mean_clustered_pointestimate():
    seed = 0
    theta = 1
    rho = 0.3
    correlation = 0.9
    cluster_size = 10
    num_clusters_labeled = 100
    num_clusters_unlabeled = 1000
    bias = 2

    sim_data = simulate_clustered_data(
        theta=theta,
        cluster_size=cluster_size,
        num_clusters_labeled=num_clusters_labeled,
        num_clusters_unlabeled=num_clusters_unlabeled,
        rho=rho,
        correlation=correlation,
        bias=bias,
        seed=seed,
    )
    Y = sim_data["Y"]
    Y_hat = sim_data["Y_hat"]
    Y_hat_unlabeled = sim_data["Y_hat_unlabeled"]
    group = sim_data["group"]
    group_unlabeled = sim_data["group_unlabeled"]

    theta_hat = ppi_mean_pointestimate(
        Y, 
        Y_hat, 
        Y_hat_unlabeled, 
        group = group, 
        group_unlabeled = group_unlabeled
    )
    print(theta_hat)
    assert np.isclose(theta_hat, theta, atol=0.1)
    

def test_ppi_mean_clustered_ci():
    seed = 0
    theta = 0
    rho = 0.3
    correlation = 0.9
    cluster_size = 10
    num_clusters_labeled = 100
    num_clusters_unlabeled = 1000
    bias = 2

    ntrials = 100
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    includeds = np.zeros_like(alphas)
    for i in range(ntrials):
        sim_data = simulate_clustered_data(
            theta=theta,
            cluster_size=cluster_size,
            num_clusters_labeled=num_clusters_labeled,
            num_clusters_unlabeled=num_clusters_unlabeled,
            rho=rho,
            correlation=correlation,
            bias=bias,
            seed=seed + i,  # Vary seed across trials
        )
        Y = sim_data["Y"]
        Y_hat = sim_data["Y_hat"]
        Y_hat_unlabeled = sim_data["Y_hat_unlabeled"]
        group = sim_data["group"]
        group_unlabeled = sim_data["group_unlabeled"]

        ci = ppi_mean_ci(
            Y, 
            Y_hat, 
            Y_hat_unlabeled, 
            alpha=alphas, 
            group=group, 
            group_unlabeled=group_unlabeled
        )
        includeds += ((ci[0] <= theta) & (ci[1] >= theta)).astype(int)
    failed = np.any(includeds / ntrials < 1 - alphas - epsilon)
    assert not failed

def test_ppi_mean_clustered_pval():
    seed = 0
    theta = 0
    rho = 0.3
    correlation = 0.9
    cluster_size = 10
    num_clusters_labeled = 100
    num_clusters_unlabeled = 1000
    bias = 2

    ntrials = 1000
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.05
    rejected = np.zeros_like(alphas)
    for i in range(ntrials):
        sim_data = simulate_clustered_data(
            theta=theta,
            cluster_size=cluster_size,
            num_clusters_labeled=num_clusters_labeled,
            num_clusters_unlabeled=num_clusters_unlabeled,
            rho=rho,
            correlation=correlation,
            bias=bias,
            seed=seed + i,  # Vary seed across trials
        )
        Y = sim_data["Y"]
        Y_hat = sim_data["Y_hat"]
        Y_hat_unlabeled = sim_data["Y_hat_unlabeled"]
        group = sim_data["group"]
        group_unlabeled = sim_data["group_unlabeled"]

        pval = ppi_mean_pval(
            Y, 
            Y_hat, 
            Y_hat_unlabeled, 
            null=theta, 
            group=group, 
            group_unlabeled=group_unlabeled
        )
        rejected += (pval < alphas).astype(int)
    print(rejected / ntrials)
    failed = rejected / ntrials > alphas + epsilon
    assert not np.any(failed)