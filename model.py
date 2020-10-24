#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
 SVM Model using Sequential Minimal Optimization (SMO) Algo
'''

import numpy as np
import random as rnd


def get_rnd_int(a, b, z):
    # Getting a random number(i) between a and b such that i != z
    i = z
    cnt = 0
    while i == z and cnt < 1000:
        i = rnd.randint(a, b)
        cnt = cnt + 1
    return i


def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)


class SvmSmo:

    def __init__(self, C=1.0, epsilon=0.001):
        self.kernel = linear_kernel
        self.b = 0.0
        self.w = None
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):

        n, m = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        kernel = self.kernel
        count = 0
        while True:
            count += 1
            alpha_old = np.copy(alpha)
            for j in range(0, n):
                i = get_rnd_int(0, n - 1, j)  # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i, :], X[j, :], y[i], y[j]
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                alpha_new_j, alpha_new_i = alpha[j], alpha[i]
                (L, H) = self.get_U_V(self.C, alpha_new_j, alpha_new_i, y_j, y_i)

                # Calc w and b
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_new_j + float(y_j * (E_i - E_j)) / k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_new_i + y_i * y_j * (alpha_new_j - alpha[j])

            # Check convergence using epsilon closeness
            diff = np.linalg.norm(alpha - alpha_old)
            if diff < self.epsilon:
                break

        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        self.w = self.calc_w(alpha, y, X)


    def get_U_V(self, C, alpha_new_j, alpha_new_i, y_j, y_i):
        # upper and lower bounds
        if y_i != y_j:
            return max(0, alpha_new_j - alpha_new_i), min(C, C - alpha_new_i + alpha_new_j)
        else:
            return max(0, alpha_new_i + alpha_new_j - C), min(C, alpha_new_i + alpha_new_j)

    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha, y))

    def predict(self, X):
        return self.activation(X, self.w, self.b)

    def activation(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.activation(x_k, w, b) - y_k
