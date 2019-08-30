import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def f_x(x):
    return 1/(1 + np.exp(-x))


def distribution(mu=0, sigma=1):
    # return probability given a value
    distribution = stats.norm(mu, sigma)
    return distribution


if __name__ == "__main__":
    # pre-setting
    n = 1000

    mu_target = 3.5
    sigma_target = 1
    mu_appro = 3
    sigma_appro = 1

    p_x = distribution(mu_target, sigma_target)
    q_x = distribution(mu_appro, sigma_appro)

    plt.figure(figsize=[10, 4])

    sns.distplot([np.random.normal(mu_target, sigma_target) for _ in range(3000)], label="distribution $p(x)$")
    sns.distplot([np.random.normal(mu_appro, sigma_appro) for _ in range(3000)], label="distribution $q(x)$")

    plt.title("Distributions", size=16)
    plt.legend()

    # value
    s = 0
    for i in range(n):
        # draw a sample
        x_i = np.random.normal(mu_target, sigma_target)
        s += f_x(x_i)
    print("simulate value", s / n)

    # calculate value sampling from a different distribution

    value_list = []
    for i in range(n):
        # sample from different distribution
        x_i = np.random.normal(mu_appro, sigma_appro)
        value = f_x(x_i) * (p_x.pdf(x_i) / q_x.pdf(x_i))

        value_list.append(value)

    print("average {} variance {}".format(np.mean(value_list), np.var(value_list)))

    # pre-setting different q(x)
    n = 5000

    mu_target = 3.5
    sigma_target = 1
    mu_appro = 1
    sigma_appro = 1

    p_x = distribution(mu_target, sigma_target)
    q_x = distribution(mu_appro, sigma_appro)

    plt.figure(figsize=[10, 4])

    sns.distplot([np.random.normal(mu_target, sigma_target) for _ in range(3000)], label="distribution $p(x)$")
    sns.distplot([np.random.normal(mu_appro, sigma_appro) for _ in range(3000)], label="distribution $q(x)$")

    plt.title("Distributions", size=16)
    plt.legend()

    # calculate value sampling from a different distribution

    value_list = []
    # need larger steps
    for i in range(n):
        # sample from different distribution
        x_i = np.random.normal(mu_appro, sigma_appro)
        value = f_x(x_i) * (p_x.pdf(x_i) / q_x.pdf(x_i))

        value_list.append(value)

    print("average {} variance {}".format(np.mean(value_list), np.var(value_list)))