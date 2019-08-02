#! coding:utf-8

import numpy as np
from math import exp, log, sqrt

from logging import getLogger, StreamHandler, basicConfig, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

fmt = "[%(asctime)s][%(levelname)s][%(name)s]:%(message)s"
basicConfig(level=DEBUG, format=fmt)


class IRT:
    def __init__(self, num_problems, num_climbers, init_a, init_b, init_t):

        self.alpha = [init_a for _ in range(num_problems)]
        self.beta = [init_b for _ in range(num_problems)]
        self.theta = [init_t for _ in range(num_climbers)]

        self.tau = 21

        self.mu_a = log(1.7)
        self.sig_a = 1.0

        self.mu_b = 0.5
        self.sig_b = 2.0

        self.mu_t = 0
        self.sig_t = sqrt(2)

        self.epoch = 100000
        self.lr = 0.001

        self.thresh = 0.0001

    @classmethod
    def sigmoid(cls, a, b, x):
        n = -a * (x - b)
        if n > 709:
            return 10e-5
        else:
            return 1 / (1 + exp(n))

    def _get_derivation_for_alpha_beta(self, i, a, b, results, is_alpha):
        d = (-1 + 2 * (log(a) - self.mu_a)) / a if is_alpha else -b
        X, W = np.polynomial.hermite.hermgauss(self.tau)

        for result in results:
            r = result[i]
            for x, w in zip(X, W):
                if r == 1:
                    d += w * a * (1 - self.sigmoid(a, b, x))
                else:
                    d += w * -a * self.sigmoid(a, b, x)

        return d

    def gauss(self, x, mu, sig):
        return (1 / (sqrt(2 * np.pi) * sig)) * exp(-1 * (x-mu)**2 /(2 * (sig**2)))

    def get_derivation_for_alpha_beta(self, i, a, b, results, is_alpha):
        d = -1 * (a - self.mu_a) / (self.sig_a ** 2) if is_alpha else self.gauss(b, self.mu_b, self.sig_b) *(-1 *(b-self.mu_b)/(self.sig_b**2))
        #d = -1 * (a - self.mu_a) / (self.sig_a ** 2) if is_alpha else -1 * (b - self.mu_b) / (self.sig_b ** 2)
        t = self.theta[i]
        for result in results:
            r = result[i]
            # print(r, a, b, t, -a * (t - b), self.sigmoid(a, b, t), (b-t) * (1 - self.sigmoid(a, b, t)), (t-b) * self.sigmoid(a, b, t))
            if r == 1:
                if is_alpha:
                    d += (b-t) * (1 - self.sigmoid(a, b, t))
                else:
                    d += a * (1 - self.sigmoid(a, b, t))
            else:
                if is_alpha:
                    d += (t-b) * self.sigmoid(a, b, t)
                else:
                    d += -a * self.sigmoid(a, b, t)

        return d

    def _get_derivation_for_theta(self, t, result):
        d = -t / (self.sig_t ** 2)
        for a, b, r in zip(self.alpha, self.beta, result):
            if r == 1:
                d += a * (1 - self.sigmoid(a, b, t))
            else:
                d += -a * self.sigmoid(a, b, t)
        return d

    def get_derivation_for_theta(self, t, result):
        d = self.gauss(t, self.mu_t, self.sig_t) *(-1 *(t-self.mu_t)/(self.sig_t**2))
        for a, b, r in zip(self.alpha, self.beta, result):
            if r == 1:
                d += -a * (1 - self.sigmoid(a, b, t))
            else:
                d += a * self.sigmoid(a, b, t)
        return d

    def predict_alpha_beta(self, results):
        rand_index = np.random.permutation(list(range(len(self.alpha))))
        for i in rand_index:
            a, b = self.alpha[i], self.beta[i]

            # print(self.get_derivation_for_alpha_beta(i, a, b, results, is_alpha=True))
            # print(self.get_derivation_for_alpha_beta(i, a, b, results, is_alpha=False))

            self.alpha[i] = a + self.lr * self.get_derivation_for_alpha_beta(i, a, b, results, is_alpha=True) - log(max(a, 10e-5))
            self.beta[i] = b + self.lr * self.get_derivation_for_alpha_beta(i, a, b, results, is_alpha=False)


    def predict_theta(self, results):
        rand_index = np.random.permutation(list(range(len(self.theta))))
        for i in rand_index:
            t = self.theta[i]
            # print(results[i], self.get_derivation_for_theta(t, results[i]))
            self.theta[i] = t + self.lr * self.get_derivation_for_theta(t, results[i])

    def calc_log_likelihood(self, results):
        log_l = 0
        for t, result in zip(self.theta, results):
            for a, b, r in zip(self.alpha, self.beta, result):
                # print(r, a, b, t, -a * (t - b), self.sigmoid(a, b, t), 1-self.sigmoid(a, b, t))
                if r == 1:
                    log_l += log(self.sigmoid(a, b, t))
                else:
                    if 1 - self.sigmoid(a, b, t) == 0:
                        log_l += 0
                    else:
                        log_l += log(1 - self.sigmoid(a, b, t))
        return log_l

    def fit(self, results):
        past_l = None
        for i in range(self.epoch):
            logger.info(f'epoch: {i}')
            # thetaを周辺化して勾配降下法でalphaとbetaを推定(周辺化尤度最大化)
            self.predict_alpha_beta(results)

            # 推定したalpha, betaでthetaを推定(MAP推定)
            self.predict_theta(results)

            log_l = self.calc_log_likelihood(results)

            # logger.debug(f'alpha: {self.alpha}')
            # logger.debug(f'beta: {self.beta}')
            # logger.debug(f'theta: {self.theta}')
            logger.info(f'log_L: {log_l}')

            if i > 0:
                loss = abs(past_l - log_l)
                logger.info(f'loss: {loss}')
                if loss < self.thresh:
                    break

            past_l = log_l
