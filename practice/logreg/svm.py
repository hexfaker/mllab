import numpy as np
import cvxopt
from cvxopt import sparse, matrix, spmatrix
from cvxopt.solvers import qp
import oracles
from sklearn.base import BaseEstimator

import time


def linear_kernel(X, Y):
    return np.matmul(X, Y.T)


def compute_rbf_kernel(X, Y, gamma):
    dist = np.matmul(X, Y.T)
    dist *= -2
    x_sqr_sum = (X ** 2).sum(axis=1)
    y_sqr_sum = (Y ** 2).sum(axis=1)
    dist += x_sqr_sum[:, np.newaxis]
    dist += y_sqr_sum[np.newaxis, :]

    return np.exp(-gamma * dist)


def compute_polynomial_kernel(X, Y, degree, r=1, gamma=1):
    return np.power(gamma * (X @ Y.T) + r, degree)


class SVMSolver(BaseEstimator):
    """
    Класс с реализацией SVM через метод внутренней точки.
    """

    @staticmethod
    def _set_opts(tolerance, max_iter):
        cvxopt.solvers.options['reltol'] = tolerance
        cvxopt.solvers.options['abstol'] = tolerance
        cvxopt.solvers.options['maxiters'] = max_iter
        cvxopt.solvers.options['show_progress'] = False

    @staticmethod
    def _reset_opts():
        del cvxopt.solvers.options['reltol']
        del cvxopt.solvers.options['abstol']
        del cvxopt.solvers.options['maxiters']

    def __init__(self, C, method, kernel='linear', gamma=None, degree=None):
        """
        C - float, коэффициент регуляризации
        
        method - строка, задающая решаемую задачу, может принимать значения:
            'primal' - соответствует прямой задаче
            'dual' - соответствует двойственной задаче
        kernel - строка, задающая ядро при решении двойственной задачи
            'linear' - линейное
            'polynomial' - полиномиальное
            'rbf' - rbf-ядро
        Обратите внимание, что часть функций класса используется при одном методе решения,
        а часть при другом
        """
        self.C = C
        self._w = None
        self._w0 = None
        self._E = None
        self._lambda = None
        self._support_mask = None
        self._X = None
        self._y = None
        self.dual = method == 'dual'
        self.method = method
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

        if kernel == 'polynomial':
            self._K = compute_polynomial_kernel
        elif kernel == 'rbf':
            self._K = compute_rbf_kernel
        else:
            self._K = linear_kernel

    def _prepare_and_solve_primal(self, X, y):
        # Solution vector [{w},w_0,e_1,...e_i]
        # See:
        #   http://cvxopt.org/userguide/matrices.html#sparse-matrices
        #   http://cvxopt.org/userguide/coneprog.html#cvxopt.solvers.qp
        objects, features = X.shape
        opt_space_dim = features + 1 + objects
        P = spmatrix(1., range(features), range(features), (opt_space_dim, opt_space_dim))
        q = matrix(spmatrix(self.C / objects,
                            range(features + 1, opt_space_dim),
                            [0] * objects,
                            (opt_space_dim, 1)))
        y_column = -y.reshape(-1, 1)

        # G is also a block matrix
        Gx = sparse([[matrix(X * y_column)], [matrix(y_column)], [spmatrix(-1., range(objects), range(objects))]])
        Ge = spmatrix(
            -1.,
            range(objects),
            range(features + 1, opt_space_dim),
            (objects, opt_space_dim)
        )
        G = sparse([Gx, Ge])

        hx = spmatrix(-1., range(objects), [0] * objects)
        he = spmatrix(0, range(objects), [0] * objects)
        h = matrix(sparse([hx, he]))
        return qp(P, q, G, h)

    def _prepare_and_solve_dual(self, X, y):

        # x =  [λ_1, ..., λ_l]
        # despite of solving dual problem, we will solve it as primal

        n_samples, n_features = X.shape

        y = y.reshape(-1, 1).astype(float)

        P = matrix(y * self._K(X, X, **self.kernel_args()) * y.T)
        q = matrix(-1., (n_samples, 1))

        G = sparse([
            spmatrix(1, range(n_samples), range(n_samples)),
            spmatrix(-1, range(n_samples), range(n_samples))
        ])

        h = matrix(self.C / n_samples, (n_samples * 2, 1))
        h[n_samples:2 * n_samples] = 0

        A = matrix(y.T)
        b = matrix(0., (1, 1))

        return qp(P, q, G, h, A, b)

    def kernel_args(self):
        res = {}
        if self.gamma is not None:
            res['gamma'] = self.gamma
        if self.degree is not None:
            res['degree'] = self.degree
        return res

    def compute_primal_objective(self, X, y):
        """
        Метод для подсчета целевой функции SVM для прямой задачи

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """

        return self._prepare_and_solve_primal(X, y)['primal objective']

    def compute_dual_objective(self, X, y):
        """
        Метод для подсчёта целевой функции SVM для двойственной задачи
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """
        return self._prepare_and_solve_dual(X, y)['primal objective']

    def fit(self, X: np.ndarray, y, tolerance=1e-3, max_iter=1000):
        """
        Метод для обучения svm согласно выбранной в method задаче
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        tolerance - требуемая точность для метода обучения
        max_iter - максимальное число итераций в методе
        
        """
        self._set_opts(tolerance, max_iter)
        samples_n, features_n = X.shape

        if not self.dual:
            solution = self._prepare_and_solve_primal(X, y)['x']
            self._w = np.array(solution[:features_n]).reshape(-1)
            self._w0 = solution[features_n]
            self._E = np.array(solution[features_n + 1:features_n + 1 + samples_n]).reshape(-1)
            self._reset_opts()
        else:
            self._X = X
            self._y = y
            solution = self._prepare_and_solve_dual(X, y)
            self._lambda = np.array(solution['x']).reshape(-1)

            self._support_mask = ~np.isclose(self._lambda, 0, tolerance, tolerance)
            self._w0 = self._compute_w0(X, y, tolerance)

        return self

    def _compute_w0(self, X, y, tolerance=1e-5):
        boundary_mask = self._support_mask & ~np.isclose(self._lambda, self.C / X.shape[0], tolerance, tolerance)

        support_vectors = X.compress(self._support_mask, 0)
        boundary_vectors = X.compress(boundary_mask, 0)

        res = y[boundary_mask].sum() - self._lambda[self._support_mask].reshape(-1, 1) \
              * y[self._support_mask].reshape(-1, 1) \
              * self._K(support_vectors, boundary_vectors, **self.kernel_args())
        res /= boundary_mask.sum()
        return res

    def _compute_w(self, X, y):
        sv = self._support_mask

        return (X.compress(sv, 0) * y[sv] * self._lambda[sv]).sum(axis=0)

    def predict(self, X):
        """
        Метод для получения предсказаний на данных
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        """
        if self.dual:
            sv = self._support_mask
            M = (self._y[sv].reshape(-1, 1) * self._lambda[sv].reshape(-1, 1) *
                 self._K(self._X.compress(sv, 0), X, **self.kernel_args())).sum(0)
        else:
            M = X @ self._w + self._w0
        return np.sign(M)

    def get_w(self, X=None, y=None):
        """
        Получить прямые переменные (без учёта w_0)
        
        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y) 
        
        return: одномерный numpy array
        """
        if self._K == linear_kernel and self.dual and X is not None and y is not None:
            return self._compute_w(X, y)
        else:
            return self._w

    def get_w0(self, X=None, y=None):
        """
        Получить вектор сдвига
        
        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y) 
        
        return: float
        """
        if self._K == linear_kernel and self.dual and X is not None and y is not None:
            return self._compute_w0(X, y, self._compute_w(X, y))
        else:
            return self._w0

    def get_dual(self):
        """
        Получить двойственные переменные
        
        return: одномерный numpy array
        """
        return self._lambda


class LinearSVM(BaseEstimator):
    def __init__(self, C: float = 1., log_iter: int = 1, max_iter: int = -1,
                 mode='full', step=1e-3, tolerance: float = 1e-3):
        self.C = C
        self.log_iter = log_iter
        self.max_iter = max_iter
        self.is_full = mode == 'full'
        self.mode = mode
        self.tolerance = tolerance
        self.step = step

        self._w = None  # type: np.ndarray
        self._w0 = None  # type: float
        self._hist = None  # type: dict

    def fit(self, X: np.ndarray, y: np.ndarray, trace=False):
        oracle = oracles.BinaryHinge(self.C)
        n_samples, n_features = X.shape
        X = np.hstack((np.ones((n_samples, 1)), X))
        w0w = np.zeros(1 + n_features)
        last_L = oracle.func(X, y, w0w)
        best_w0w, best_L = w0w, last_L

        func_vals = []
        iteration_i = []
        times = []
        current_time = time.time()

        i = 0
        while True:
            if i == self.max_iter:
                break

            if self.is_full:
                grad = oracle.grad(X, y, w0w)
            else:
                sample_i = np.random.randint(0, n_samples - 1)
                grad = oracle.grad(X[sample_i].reshape(1, -1), np.array([y[sample_i]]), w0w)

            w0w -= self.step * grad

            if i % self.log_iter == 0 or i == self.max_iter - 1:
                current_L = oracle.func(X, y, w0w)
                if np.isclose(np.abs(current_L - last_L), 0, rtol=self.tolerance):
                    break

                if not self.is_full and best_L > current_L:
                    best_L = current_L
                    best_w0w = w0w

                last_L = current_L

                if trace:
                    func_vals.append(current_L)
                    next_time = time.time()
                    times.append(next_time - current_time)
                    current_time = next_time
                    iteration_i.append(i)

            i += 1

        self._w = best_w0w[1:]
        self._w0 = best_w0w[0]

        if trace:
            self._hist = {
                'iter': np.array(iteration_i),
                'func': np.array(func_vals),
                'time': np.array(times)
            }

        return self

    def predict(self, X: np.ndarray):
        return np.sign(X @ self._w + self._w0)
