import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

# Number of observations
data_size = 100

# Number of iterations to average over when computing the mean squared error
iterations = 1000

# Number of samples used to plot histograms
hist_size = 10000


# Beta-Binomial


# Ground truth
p = 0.3
n = 10


# Three sets of hyperparameters
hyperparams = [(1, 1), (5, 5), (7, 3)]

fig, axarr = plt.subplots(nrows=1, ncols=len(hyperparams), figsize=[
                          18, 5], sharex=True, sharey=True)
priors = np.zeros(shape=[hist_size, len(hyperparams)])

for i, ((a, b), ax) in enumerate(zip(hyperparams, axarr)):
    priors[:, i] = np.random.beta(a=a, b=b, size=hist_size)
    ax.hist(priors[:, i], normed=1)
    ax.set_title("Prior: Beta(a={0}, b={1})".format(a, b))
    ax.set_ylabel("Probability")
    ax.set_xlabel("p")

plt.show()


bayes_squared_errors = []
ml_squared_errors = []

for _ in range(iterations):
    # Observations
    X = np.random.binomial(n, p, data_size)

    # Bayes estimate
    bayes_params = np.zeros(
        shape=[len(X) + 1, len(hyperparams) * len(hyperparams[0])])
    bayes_params[0, :] = list(chain(*hyperparams))

    for i, x in enumerate(X):
        # Update rule for binomial model with beta prior for p
        bayes_params[i + 1, :] = bayes_params[i, :] + np.tile([x, n - x], 3)

    # Expectation of Beta(a,b) = a/(a+b)
    # Also, throw away the prior estimate
    bayes_ests = (bayes_params[:, ::2] /
                  (bayes_params[:, ::2] +
                   bayes_params[:, 1::2]))[1:, :]

    # ML estimate
    ml_ests = np.zeros(shape=[len(X), 1])

    for i in range(len(X)):
        # Mean of observations
        ml_ests[i] = np.mean(X[:i + 1] / n)

    bayes_squared_errors.append((bayes_ests - p)**2)
    ml_squared_errors.append((ml_ests - p)**2)

mean_bayes_squared_error = np.dstack(bayes_squared_errors).mean(axis=2)
mean_ml_squared_error = np.dstack(ml_squared_errors).mean(axis=2)


fig, axarr = plt.subplots(nrows=1, ncols=len(
    hyperparams), figsize=[18, 5], sharex=True)

for i, (ax, (a, b)) in enumerate(zip(axarr, hyperparams)):
    ax.plot(mean_ml_squared_error, label='ML MSE')
    ax.plot(mean_bayes_squared_error[:, i], label='Bayes MSE')
    ax.set_title(
        "Mean Squared Error with Prior Beta(a={0}, b={1})".format(a, b))
    ax.legend()
    ax.set_ylabel("Mean Squared Error")
    ax.set_xlabel("Number of Observations")

plt.show()


fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=[
                          18, 10], sharex=True, sharey=True)
num_obs = [0, 1, 5, 10, 50, 100]

for i, (num, ax) in enumerate(zip(num_obs, axarr.flatten())):
    samp = np.random.beta(
        a=bayes_params[num, 0], b=bayes_params[num, 1], size=hist_size)
    ax.hist(samp, normed=True)
    ax.set_title("Posterior Given {} Observations".format(num))
    ax.set_ylabel("Probability")
    ax.set_xlabel("p")

plt.show()


# Gaussian with Known Variance


# Ground truth
mu = -4
sigma = 1


# Three sets of hyperparameters
hyperparams = [(0, 1), (-7, 2), (5, 3)]

fig, axarr = plt.subplots(nrows=1, ncols=len(hyperparams), figsize=[
                          18, 5], sharex=True, sharey=True)
priors = np.zeros(shape=[hist_size, len(hyperparams)])

for i, ((mu_0, sigma_0), ax) in enumerate(zip(hyperparams, axarr)):
    priors[:, i] = np.random.normal(loc=mu_0, scale=sigma_0, size=hist_size)
    ax.hist(priors[:, i], normed=True)
    ax.set_title("Prior: Normal(mu={0}, sigma={1})".format(mu_0, sigma_0))
    ax.set_ylabel("Probability")
    ax.set_xlabel("μ")

plt.show()


bayes_squared_errors = []
ml_squared_errors = []

for _ in range(iterations):
    # Observations
    X = np.random.normal(loc=mu, scale=sigma, size=data_size)

    # Bayes
    bayes_params = np.zeros(
        shape=[len(X) + 1, len(hyperparams) * len(hyperparams[0])])
    bayes_params[0, :] = list(chain(*hyperparams))

    for i, x in enumerate(X):
        # Update rule for normal model with normal prior for mu. Online
        # learning.
        coeff = sigma**2 / (bayes_params[i, 1::2]**2 + sigma**2)
        bayes_params[i + 1, ::2] = coeff * \
            bayes_params[i, ::2] + (1 - coeff) * x
        bayes_params[i +
                     1, 1::2] = np.sqrt(1 /
                                        (1 /
                                         (bayes_params[i, 1::2]**2) +
                                            1 /
                                            sigma**2))

    # Expectation of N(mu, sigma^2) = mu
    # Also, throw away the prior estimate
    bayes_ests = bayes_params[1:, ::2]

    # ML
    ml_ests = np.zeros(shape=[len(X), 1])

    for i in range(len(X)):
        ml_ests[i] = np.mean(X[:i + 1])

    bayes_squared_errors.append((bayes_ests - mu)**2)
    ml_squared_errors.append((ml_ests - mu)**2)

mean_bayes_squared_error = np.dstack(bayes_squared_errors).mean(axis=2)
mean_ml_squared_error = np.dstack(ml_squared_errors).mean(axis=2)


fig, axarr = plt.subplots(nrows=1, ncols=len(
    hyperparams), figsize=[18, 5], sharex=True)

for i, (ax, (mu_0, sigma_0)) in enumerate(zip(axarr, hyperparams)):
    ax.plot(mean_ml_squared_error, label='ML MSE')
    ax.plot(mean_bayes_squared_error[:, i], label='Bayes MSE')
    ax.set_title(
        "Mean Squared Error: Prior is Normal(mu={0}, sigma={1})".format(
            mu_0, sigma_0))
    ax.legend()
    ax.set_ylabel("Mean Squared Error")
    ax.set_xlabel("Number of Observations")

plt.show()


fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=[
                          18, 10], sharex=True, sharey=True)
num_obs = [0, 1, 5, 10, 50, 100]

for i, (num, ax) in enumerate(zip(num_obs, axarr.flatten())):
    samp = np.random.normal(
        loc=bayes_params[num, 0], scale=bayes_params[num, 1], size=hist_size)
    ax.hist(samp, normed=True)
    ax.set_title("Posterior Given {} Observations".format(num))
    ax.set_ylabel("Probability")
    ax.set_xlabel("μ")

plt.show()


# Gaussian with Known Mean (estimating precision $τ$)


# Ground Truth
mu = 3
tau = 1 / 4


# 3 sets of hyperparameters
hyperparams = [(1, 1), (4, 2), (3, 4)]

fig, axarr = plt.subplots(nrows=1, ncols=len(hyperparams), figsize=[
                          18, 5], sharex=True, sharey=True)
priors = np.zeros(shape=[hist_size, len(hyperparams)])

for i, ((a, b), ax) in enumerate(zip(hyperparams, axarr)):
    # NumPy uses k and theta and not alpha and beta. The 1/ is to compensate
    # for this
    priors[:, i] = np.random.gamma(shape=a, scale=1 / b, size=hist_size)
    ax.hist(priors[:, i], normed=True)
    ax.set_xlim([0, 2])
    ax.set_title("Prior: Gamma(a={0}, b={1})".format(a, b))
    ax.set_ylabel("Probability")
    ax.set_xlabel("τ")

plt.show()


bayes_squared_errors = []
ml_squared_errors = []

for _ in range(iterations):
    # Observations
    X = np.random.normal(loc=mu, scale=np.sqrt(1 / tau), size=data_size)

    # Bayes
    bayes_params = np.zeros(
        shape=[len(X) + 1, len(hyperparams) * len(hyperparams[0])])
    bayes_params[0, :] = list(chain(*hyperparams))

    for i, x in enumerate(X):
        # Update rule for Gaussian model with Gamma prior for tau. Online
        # learning.
        bayes_params[i + 1, ::2] = bayes_params[i, ::2] + 1 / 2
        bayes_params[i + 1, 1::2] = bayes_params[i, 1::2] + ((x - mu)**2) / 2

    # Expectation of Gamma(a,b) = a/b
    # Also, throw away the prior estimate
    bayes_ests = bayes_params[1:, ::2] / bayes_params[1:, 1::2]

    # ML
    ml_ests = np.zeros(shape=[len(X), 1])

    for i, x in enumerate(X):
        ml_ests[i] = 1 / (np.sum((X[:i + 1] - mu)**2) / (i + 1))

    bayes_squared_errors.append((bayes_ests - tau)**2)
    ml_squared_errors.append((ml_ests - tau)**2)

mean_bayes_squared_error = np.dstack(bayes_squared_errors).mean(axis=2)
mean_ml_squared_error = np.dstack(ml_squared_errors).mean(axis=2)


fig, axarr = plt.subplots(nrows=1, ncols=len(
    hyperparams), figsize=[18, 5], sharex=True)

for i, (ax, (a, b)) in enumerate(zip(axarr, hyperparams)):
    # Omit plotting the first 3 ML mean squared errors
    ax.plot(mean_ml_squared_error[2:], label='ML MSE')
    ax.plot(mean_bayes_squared_error[:, i], label='Bayes MSE')
    ax.set_title(
        "Mean Squared Error: Prior is Gamma(a={0}, b={1})".format(a, b))
    ax.legend()
    ax.set_ylabel("Mean Squared Error")
    ax.set_xlabel("Number of Observations")

plt.show()


fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=[
                          18, 10], sharex=True, sharey=True)
num_obs = [0, 1, 5, 10, 50, 100]

for i, (num, ax) in enumerate(zip(num_obs, axarr.flatten())):
    foo = np.random.beta(
        a=bayes_params[num, 0], b=bayes_params[num, 1], size=hist_size)
    ax.hist(foo, normed=True)
    ax.set_title("Posterior Given {} Observations".format(num))
    ax.set_ylabel("Probability")
    ax.set_xlabel("τ")

plt.show()
