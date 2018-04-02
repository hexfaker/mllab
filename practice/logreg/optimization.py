from oracles import *
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import scipy as sp
import scipy.special
import scipy.sparse
import time


# noinspection PyPep8Naming
class GDClassifier(BaseEstimator):
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, l2_coef=0., optimize_bias=False, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия
                
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход 
        
        max_iter - максимальное число итераций     
        
        **kwargs - аргументы, необходимые для инициализации   
        """

        self.l2_coef = l2_coef
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.step_beta = step_beta
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.optimize_bias = optimize_bias
        self.kwargs = kwargs

        self._w0 = None
        self._w = None
        self._hist = None
        self._multiclass_map = None

    def _step_length(self, iter_no):
        return self.step_alpha / (iter_no + 1) ** self.step_beta

    def _make_orcale(self):
        param = {'l2_coef': self.l2_coef}

        if self.optimize_bias:
            param['w0_exclude'] = True

        if self.loss_function == 'binary_logistic':
            return BinaryLogistic(**param)
        elif self.loss_function == 'multinomial_logistic':
            return MulticlassLogistic(**param)

    def accuracy(self, X, y):
        pred_y = self.predict(X)
        return accuracy_score(y, pred_y)

    def fit(self, X, y, w_0=None, trace=False, X_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w_0 - начальное приближение в методе
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        oracle = self._make_orcale()
        n_samples, n_features = X.shape
        n_classes = np.unique(y).size

        if self.optimize_bias:
            n_features += 1
            const_feature = np.ones((n_samples, 1))
            if isinstance(X, sp.sparse.csr_matrix):
                X = sp.sparse.hstack((sp.sparse.csr_matrix(const_feature), X))
            else:
                X = np.hstack((const_feature, X))

        if self.loss_function == 'binary_logistic':
            w0w = np.zeros(n_features)
        else:
            w0w = np.zeros((n_classes, n_features))
            self._multiclass_map = np.unique(y)
            y = np.digitize(y, self._multiclass_map) - 1

        if w_0 is not None:
            w0w[1 if self.optimize_bias else 0:] = w_0

        current_loss = oracle.func(X, y, w0w)
        func, times = [current_loss], [0.]
        current_time = time.time()

        accuracy = []

        for i in range(self.max_iter):
            grad = oracle.grad(X, y, w0w)
            w0w -= self._step_length(i) * grad

            previous_loss = current_loss
            current_loss = oracle.func(X, y, w0w)

            if trace:
                next_time = time.time()
                times.append(next_time - current_time)
                current_time = next_time

                func.append(current_loss)

                if X_test is not None:
                    self._set_w_w0(w0w)
                    accuracy.append(self.accuracy(X_test, y_test))

            if np.abs(previous_loss - current_loss) < self.tolerance:
                break

        self._set_w_w0(w0w)

        if trace:
            self._hist = {
                'func': np.array(func),
                'time': np.array(times),
                'accuracy': np.array(accuracy)
            }

            return self._hist

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """
        if self.loss_function == 'binary_logistic':
            return np.sign(X.dot(self._w) + self._w0)

        class_idx = np.argmax(self.predict_proba(X), axis=1)
        return self._multiclass_map[class_idx]

    def _set_w_w0(self, w0w):
        if self.optimize_bias:
            self._w = w0w[1:]
            self._w0 = w0w[0]
        else:
            self._w = w0w
            self._w0 = 0

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k 
        """
        if self.loss_function == 'binary_logistic':
            p = sp.special.expit(X.dot(self._w)).reshape(-1, 1)
            return np.hstack((1 - p, p))
        else:
            margin = X.dot(self._w.T) + self._w0
            margin -= margin.max(1).reshape(-1, 1)
            margin_e = np.exp(margin)
            return margin_e / margin_e.sum(1).reshape(-1, 1)

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: float
        """
        return self._make_orcale().func(X, y, self._w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: numpy array, размерность зависит от задачи
        """
        return self._make_orcale().grad(X, y, self._w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self._w


# noinspection PyPep8Naming
class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, batch_size=1, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=153, l2_coef=0., optimize_bias=False, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход 
        
        
        max_iter - максимальное число итераций
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """
        super(SGDClassifier, self).__init__(loss_function, step_alpha, step_beta, tolerance,
                                            max_iter, l2_coef, optimize_bias, **kwargs)

        self.random_seed = random_seed
        self.batch_size = batch_size

    def fit(self, X, y, w_0=None, trace=False, log_freq=1, X_test=None, y_test=None, verbose=False):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        oracle = self._make_orcale()

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        n_samples, n_features = X.shape
        n_classes = np.unique(y).size

        if self.optimize_bias:
            n_features += 1
            const_feature = np.ones((n_samples, 1))
            if isinstance(X, sp.sparse.csr_matrix):
                X = sp.sparse.hstack((sp.sparse.csr_matrix(const_feature), X))
            else:
                X = np.hstack((const_feature, X))

        if self.loss_function == 'binary_logistic':
            w0w = np.zeros(n_features)
        else:
            w0w = np.zeros((n_classes, n_features))
            self._multiclass_map = np.unique(y)
            y = np.digitize(y, self._multiclass_map) - 1

        if w_0 is not None:
            w0w[1 if self.optimize_bias else 0:] = w_0

        current_loss = oracle.func(X, y, w0w)
        current_time = time.time()
        log_epoch = 0.
        log_w0w = w0w.copy()

        func, times = [current_loss], [0.]
        log_epochs = [0.]
        weight_diffs = [0.]
        accuracy = []

        feature_idx = np.arange(n_features)

        shuffled_idx = np.random.permutation(np.arange(n_samples))
        samples_processed = 0
        batch_start = 0
        for i in range(self.max_iter):
            batch_idx = shuffled_idx[batch_start: batch_start + self.batch_size]
            current_epoch = samples_processed / n_samples
            samples_processed += batch_idx.size
            batch_start += self.batch_size

            if batch_start >  n_samples:
                shuffled_idx = np.random.permutation(np.arange(n_samples))
                batch_start = 0       

            batch_X = X[batch_idx.reshape(-1, 1), feature_idx.reshape(1, -1)]
            batch_y = y[batch_idx]

            #if isinstance(batch_X, sp.sparse.csr_matrix):
            #    batch_X = np.asarray(batch_X.todense())

            grad = oracle.grad(batch_X, batch_y, w0w)
            w0w -= self._step_length(i) * grad

            if current_epoch - log_epoch > log_freq:
                previous_loss = current_loss
                current_loss = oracle.func(X, y, w0w)

                if verbose:
                    print('iteration={0} epoch={1} loss={2}'.format(i, current_epoch, current_loss))

                if trace:
                    next_time = time.time()
                    times.append(next_time - current_time)
                    current_time = next_time

                    func.append(current_loss)

                    weight_diffs.append(np.linalg.norm(log_w0w - w0w))
                    log_w0w = w0w

                    log_epochs.append(current_epoch)

                    if X_test is not None:
                        self._set_w_w0(w0w)
                        accuracy.append(self.accuracy(X_test, y_test))

                if np.abs(previous_loss - current_loss) < self.tolerance:
                    break

                log_epoch = current_epoch

        self._set_w_w0(w0w)

        if trace:
            self._hist = {
                'func': np.array(func),
                'time': np.array(times),
                'weights_diff': np.array(weight_diffs),
                'epoch_num': np.array(log_epochs),
                'accuracy': np.array(accuracy)
            }

            return self._hist
