import numpy as np
import scipy


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, X, y, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

        
class BinaryHinge(BaseSmoothOracle):
    """
    Оракул для задачи двухклассового линейного SVM.
    """
    
    @staticmethod
    def _compute_hinge_x(X, y, w):
        hinge_x = 1 - np.dot(y.reshape(-1, 1) * X, w)
        hinge_x[hinge_x <= 0] = 0
        return hinge_x
     
    def __init__(self, C):
        """
        Задание параметров оракула.
        """
        self._C = C

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        samples_n = X.shape[0]
        return 0.5 * np.dot(w[1:], w[1:]) + self._C / samples_n * self._compute_hinge_x(X, y, w).sum()

    def grad(self, X, y, w):
        """
        Вычислить субградиент функционала в точке w на выборке X с ответами y.
        Субгрдиент в точке 0 необходимо зафиксировать равным 0.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        hinge_x = self._compute_hinge_x(X, y, w)
        samples_n = X.shape[0]
        only_w = w.copy()
        only_w[0] = 0
        return only_w - self._C / samples_n * (X * y.reshape(-1, 1)).compress(hinge_x > 0, axis=0).sum(axis=0)
