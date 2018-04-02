import numpy as np
import scipy as sp
import scipy.special


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


# noinspection PyMethodOverriding
class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    """

    def _clean_w(self, merged_w: np.ndarray):
        if self.w0_exclude:
            w = merged_w.copy()
            w[0] = 0
            return w
        return merged_w

    def __init__(self, l2_coef=0., w0_exclude=False):
        """
        Задание параметров оракула.
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.w0_exclude = w0_exclude
        self.l2_lambda = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        clean_w = self._clean_w(w)
        regularizer = (self.l2_lambda / 2) * clean_w.dot(clean_w)
        return np.logaddexp(0, (-y) * X.dot(w)).mean() + regularizer

    def grad(self, X: np.ndarray, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        n_samples = X.shape[0]
        minus_y = -y
        sample_coef = (minus_y * sp.special.expit(minus_y * X.dot(w)))
        return (X.T.dot(sample_coef)) / n_samples + \
               self.l2_lambda * self._clean_w(w)


# noinspection PyMethodOverriding
class MulticlassLogistic(BaseSmoothOracle):
    """
    Оракул для задачи многоклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    
    w в этом случае двумерный numpy array размера (class_number, d),
    где class_number - количество классов в задаче, d - размерность задачи
    """

    def _clean_w(self, merged_w: np.ndarray):
        if self.w0_exclude:
            w = merged_w.copy()
            w[:, 0] = 0
            return w
        return merged_w

    def __init__(self, class_number=None, l2_coef=0., w0_exclude=False):
        """
        Задание параметров оракула.
        
        class_number - количество классов в задаче
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_lambda = l2_coef
        self.w0_exclude = w0_exclude

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - двумерный numpy array
        """
        n_samples, _ = X.shape
        n_classes = w.shape[0]
        margins = X.dot(w.T)  # type: np.ndarray
        margins -= margins.max(1).reshape(-1, 1)

        log_class_sum = sp.special.logsumexp(margins, 1).mean()
        class_idx = y - y.min()
        log_nom = margins[np.arange(n_samples), class_idx].mean()

        clean_w = self._clean_w(w)
        return log_class_sum - log_nom + self.l2_lambda / 2 * ((clean_w ** 2).sum())

    def grad(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - двумерный numpy array
        """
        n_classes = w.shape[0]
        n_samples, _ = X.shape

        class_idx = y - y.min()

        margins = X.dot(w.T)  # type: np.ndarray
        margins -= margins.max(1).reshape(-1, 1)
        margins_exp = np.exp(margins)

        samples_denom = margins_exp.sum(1)

        samples_mul = margins_exp.T
        class_equal_mask = class_idx.reshape(1, -1) == np.arange(n_classes).reshape(-1, 1)
        samples_mul -= class_equal_mask * samples_denom.reshape(1, -1)
        samples_mul /= samples_denom.reshape(1, -1) * n_samples

        t = X.T.dot(samples_mul.T).T
        w_ = self.l2_lambda * self._clean_w(w)
        return t + w_
