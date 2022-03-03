from scipy.stats import norm, chi2, mstats, shapiro, kstest, chisquare
from scipy.stats import t as student_value
from mpmath import mp, ncdf
import math
import numpy as np
import matplotlib.pyplot as plt
mp.dps = 50


def calculate_sample_mean(sample):
    sample_sum = 0
    for x in sample:
        sample_sum += x
    return sample_sum / len(sample)


def calculate_sample_dispersion(sample):
    sample_mean = calculate_sample_mean(sample)

    dispersion = 0
    for x in sample:
        dispersion += (x - sample_mean) ** 2
    dispersion = dispersion / len(sample)
    return dispersion


def chi2_test(sorted_sample):
    k = 100
    b = max(abs(sorted_sample[0]), abs(sorted_sample[-1]))
    a = -b
    h = np.longdouble((b - a) / k)

    loc = calculate_sample_mean(sorted_sample)
    scale = math.sqrt(calculate_sample_dispersion(sorted_sample))

    n_arr = [0] * k
    i = 0
    for x in sorted_sample:
        x_i = a + h * (i + 1)
        while x > a + h * (i + 1) and i < k - 1:
            x_i = a + h * (i + 1)
            i += 1
        try:
            n_arr[i] += 1
        except IndexError:
            print(x_i, x, i)
            raise IndexError

    p = []
    for i in range(k):
        n_arr[i] = n_arr[i]
        p_i = len(sorted_sample)*(np.longdouble(norm.cdf(a + h * (i + 1), loc=loc, scale=scale)) -
                                  np.longdouble(norm.cdf(a + h * i, loc=loc, scale=scale)))
        if p_i == 0:
            p_i = 1e-21
        p.append(p_i)

    p[len(p)//2+1] += sum(n_arr) - sum(p)
    return chisquare(n_arr, p, ddof=2)


if __name__ == '__main__':
    chi2_results_norm = [0, 1]
    shapiro_results_norm = [0, 1]
    kstest_results_norm = [0, 1]
    norm_n = 10000
    for _ in range(norm_n):
        norm_sample = sorted(norm.rvs(size=100))
        chi2_results_norm.append(chi2_test(norm_sample).pvalue)
        shapiro_results_norm.append(shapiro(norm_sample).pvalue)
        kstest_results_norm.append(kstest(norm_sample, 'norm').pvalue)
    chi2_results_norm.sort()
    shapiro_results_norm.sort()
    kstest_results_norm.sort()

    norm_x = np.linspace(0, 1, norm_n + 2)
    fig, ax = plt.subplots(1, 1)
    ax.plot(chi2_results_norm, norm_x, linestyle=':')
    ax.plot(shapiro_results_norm, norm_x, linestyle='--')
    ax.plot(kstest_results_norm, norm_x, linestyle='-.')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

    chi2_results_student = [0, 1]
    shapiro_results_student = [0, 1]
    kstest_results_student = [0, 1]
    student_n = 5000
    for _ in range(student_n):
        student_sample = sorted(student_value.rvs(6, size=200))
        chi2_results_student.append(chi2_test(student_sample).pvalue)
        shapiro_results_student.append(shapiro(student_sample).pvalue)
        kstest_results_student.append(kstest(student_sample, 'norm').pvalue)
    chi2_results_student.sort()
    shapiro_results_student.sort()
    kstest_results_student.sort()

    x_student = np.linspace(0, 1, student_n + 2)
    fig, ax = plt.subplots(1, 1)
    ax.plot(chi2_results_student, x_student, linestyle=':')
    ax.plot(shapiro_results_student, x_student, linestyle='--')
    ax.plot(kstest_results_student, x_student, linestyle='-.')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.show()
