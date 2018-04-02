import numpy as np
import time
from sklearn.base import BaseEstimator


class PEGASOSMethod(BaseEstimator):
    """
    Реализация метода Pegasos для решения задачи svm.
    """

    def __init__(self, step_lambda, batch_size, num_iter):
        """
        step_lambda - величина шага, соответствует 
        
        batch_size - размер батча
        
        num_iter - число итераций метода, предлагается делать константное
        число итераций 
        """
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.step_lambda = step_lambda

        self._w = None  # type: np.ndarray
        self._w0 = None  # type: float

    def _func(self, X, y, w0w):
        hinge = 1 - ((X * y.reshape(-1, 1)) @ w0w)
        hinge[hinge < 0] = 0

        return self.step_lambda / 2 * (w0w @ w0w) + hinge.mean()

    def fit(self, X, y, trace=False):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        n_samples, n_features = X.shape
        X = np.hstack((np.ones((n_samples, 1)), X))
        w0w = np.zeros(n_features + 1)

        func_vals = []
        times = []
        current_time = time.time()

        if trace:
            func_vals.append(self._func(X, y, w0w))
            times.append(0.)

        for i in range(self.num_iter):
            batch_elems = np.random.randint(0, n_samples - 1, self.batch_size)
            batchX = X.take(batch_elems, axis=0)
            batchy = y.take(batch_elems, axis=0)

            Xy = batchX * batchy.reshape(-1, 1)
            mask = (Xy @ w0w) < 1

            alpha = 1 / (i + 1) / self.step_lambda
            w0w = (1 - self.step_lambda * alpha) * w0w + alpha / self.batch_size * Xy.compress(mask, axis=0).sum(axis=0)

            scale = np.min([1, 1 / (np.sqrt(self.step_lambda) * np.linalg.norm(w0w))])
            w0w = scale * w0w

            if trace:
                func_vals.append(self._func(X, y, w0w))
                next_time = time.time()
                times.append(next_time - current_time)
                current_time = next_time

        self._w = w0w[1:]
        self._w0 = w0w[0]

        if trace:
            return {
                'func': np.array(func_vals),
                'time': np.array(times)
            }

    def predict(self, X):
        """
        Получить предсказания по выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        """
        return np.sign(X @ self._w + self._w0)
