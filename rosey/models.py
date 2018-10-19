import copy
import keras
import numpy as np
import keras.regularizers as kreg
from keras.models import Model
from keras.layers import Input, Dense
from keras.metrics import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity
from sklearn.utils import resample
from sklearn.exceptions import NotFittedError
from glmnet import ElasticNet, LogitNet
from .keras_utils import sklearn_mse, keras_r2


def _np_dropna(a):
    """Mimics pandas dropna"""
    return a[~np.isnan(a).any(axis=1)]


# noinspection PyPep8Naming
class PreTrainedVoteEnsemble(ClassifierMixin):
    """
    Binary Ensemble Classifier that only accepts pre-trained models as an input.
    """
    def __init__(self, trained_estimators):
        self.estimators = trained_estimators
        self.models_ = [model for _, model in trained_estimators]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), 1)

    def predict_proba(self, X):
        probabilities = np.asarray([est.predict_proba(X) for est in self.models_])
        return np.average(probabilities, axis=0)


# noinspection PyPep8Naming
class KerasSKWrappedBaggingClassifier(ClassifierMixin):
    """
    Implements bagging for Keras models that have been wrapped in an SKLearn wrapper

    :param bootstrap: if False then use Stratified K Fold CV to fit the ensemble
    """
    def __init__(self, base_estimator, n_estimators=10, bootstrap=True, random_state=None):
        assert n_estimators >= 2, 'Ensemble models needs 2 or more estimators'
        self.models_, self.classes_ = [], None
        self._base_model = base_estimator
        self._n_estimators = n_estimators
        self._bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        self.models_ = []
        if self._bootstrap:
            # Bootstrapping (Traditional bagging)
            for i in range(self._n_estimators):
                x_boot, y_boot = resample(X, y, random_state=self.random_state)
                model_i = copy.deepcopy(self._base_model)
                model_i.fit(x_boot, y_boot)
                self.models_.append(model_i)
        else:
            # Cross Validation (Ensures all data is used)
            k_fold = KFold(n_splits=self._n_estimators, shuffle=True, random_state=self.random_state)
            for train_indices, _ in k_fold.split(X, y):
                x_cv, y_cv = X[train_indices], y[train_indices]
                model_i = copy.deepcopy(self._base_model)
                model_i.fit(x_cv, y_cv)
                self.models_.append(model_i)
        return self

    def predict_proba(self, X):
        assert len(self.models_) > 0, 'Fit must be called first!'
        probabilities = np.asarray([est.predict_proba(X) for est in self.models_])
        return np.average(probabilities, axis=0)

    def predict(self, X):
        # TODO 1/3/2018 does this work for non-binary predictions?
        return np.argmax(self.predict_proba(X), 1)


# noinspection PyPep8Naming
class ProbabilityLogisticRegression(LinearRegression):
    """
    A Logistic Regression whose target variable is the prediction probability.

    >>> import numpy as np
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> x, y = load_breast_cancer(return_X_y=True)
    >>> logit = LogisticRegression().fit(x, y)
    >>> y_prob = logit.predict_proba(x)[:, 1]
    >>> plr = ProbabilityLogisticRegression().fit(x, y_prob)
    """
    @staticmethod
    def _validate_target_variable(y: np.ndarray):
        if np.squeeze(y).ndim != 1:
            raise AssertionError('`y` must be 1 dimensional')
        if 1.0 in y or 0.0 in y:
            raise AssertionError('`y` vector cannot contain probabilities equal to either 1 or 0')

    @staticmethod
    def _inverse_logit(p):
        return -np.log((1/p) - 1)

    @staticmethod
    def _logit(z):
        from scipy.special import expit
        return expit(z)

    def fit(self, X, y, sample_weight=None):
        self._validate_target_variable(y)
        super().fit(X, self._inverse_logit(y), sample_weight)
        return self

    def predict_proba(self, X):
        z = super().predict(X)
        return self._logit(z)

    def predict(self, X):
        return self.predict_proba(X).round()

    def score(self, X, y, sample_weight=None):
        """
        This returns a pseudo Rsq

        :param X:
        :param y:
        :param sample_weight:
        :return:
        """
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict_proba(X))


# noinspection PyPep8Naming
class BayesKDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian Classifier that uses Kernel Density Estimations to generate the joint distribution
    Parameters:
        - bandwidth: float
        - kernel: for scikit learn KernelDensity
    """
    def __init__(self, bandwidth=0.2, kernel='gaussian'):
        self.classes_, self.models_, self.priors_logp_ = [None] * 3
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(x_subset)
                        for x_subset in training_sets]

        self.priors_logp_ = [np.log(x_subset.shape[0] / X.shape[0]) for x_subset in training_sets]
        return self

    def predict_proba(self, X):
        logp = np.array([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logp + self.priors_logp_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class PolynomialRegression(object):
    def __init__(self, degree=1):
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        self.model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )


class L1Regressor(ElasticNet):
    """
    L1 Regressor wrapper for GLMNet
    """
    def __init__(self, lam, n_jobs=1):
        super().__init__(lambda_path=np.asarray([lam]), n_jobs=n_jobs)


class L1Classifier(LogitNet):
    """
    L1 Classifier wrapper for GLMNet
    """
    def __init__(self, lam, n_jobs=1):
        super().__init__(lambda_path=np.asarray([lam]), n_jobs=n_jobs)


# noinspection PyPep8Naming
class KerasL1OLSBase:
    def __init__(self, alpha, ols_indices, verbose=False):
        if not(isinstance(ols_indices, list) or isinstance(ols_indices, np.ndarray)):
            raise ValueError('ols_indices must be a list or np.array')

        self.alpha = alpha
        self.ols_idx = ols_indices
        self.verbose = verbose
        self.model = Model()
        self.hist = None

    def predict(self, X):
        if self.hist is None:
            NotFittedError('Model not fitted. First call .fit()')
        x_li, x_ols = self._split_x(X)
        return self.model.predict([x_li, x_ols])

    def _split_x(self, x):
        # Perfect solution
        ols_mask = np.zeros((x.shape[1],), dtype=bool)
        ols_mask[self.ols_idx] = True

        # X_l1, X_ols
        return x[:, ~ols_mask], x[:, ols_mask]

    def _create_model(self, l1_shape, ols_shape, is_regression: bool):
        # Inputs
        input_l1 = Input(shape=(l1_shape[1],), name='l1-inputs')
        input_ols = Input(shape=(ols_shape[1],), name='ols-inputs')

        # Model Specification
        coef_init = 'zeros'
        activations = 'linear' if is_regression else 'sigmoid'
        comp_l1 = Dense(
            1, use_bias=False, kernel_regularizer=kreg.l1(self.alpha), kernel_initializer=coef_init,
            activation=activations,
            name='l1-weights'
        )(input_l1)
        comp_ols = Dense(
            1, kernel_initializer=coef_init,
            activation=activations,
            name='ols-weights'
        )(input_ols)

        y_hat = keras.layers.add([comp_l1, comp_ols])

        self.model = Model([input_l1, input_ols], y_hat)

        if self.verbose:
            print(self.model.summary())


# noinspection PyPep8Naming
class KerasL1OLSClassification(KerasL1OLSBase, ClassifierMixin):
    """
    A model where only certain coefficients are subject to L1 penalty

    y = Xm + Zu + b
    """
    def fit(self, X, y, max_iter=500):
        x_l1, x_ols = self._split_x(X)
        self._create_model(x_l1.shape, x_ols.shape, is_regression=False)

        self.model.compile(loss=sklearn_mse, metrics=[keras_r2], optimizer='adam')
        self.hist = self.model.fit(
            [x_l1, x_ols], y,
            epochs=max_iter, verbose=int(self.verbose),
            callbacks=[
                EarlyStopping(monitor='loss', patience=15),
                ModelCheckpoint('clf-l1ols-model', monitor='loss', save_best_only=True)
            ]
        )
        return self


# noinspection PyPep8Naming
class KerasL1OLSRegressor(KerasL1OLSBase, RegressorMixin):
    """
    A model where only certain coefficients are subject to L1 penalty

    y = Xm + Zu + b
    Where
    Cost(m, u, b) = (1/2N) ||y - Xm + Zu + b||^2_2 + ||m||_1
    """
    def fit(self, X, y, max_iter=500):
        x_l1, x_ols = self._split_x(X)
        self._create_model(x_l1.shape, x_ols.shape, is_regression=True)

        self.model.compile(loss=binary_crossentropy, optimizer='adam')
        self.hist = self.model.fit(
            [x_l1, x_ols], y,
            epochs=max_iter, verbose=int(self.verbose),
            callbacks=[
                EarlyStopping(monitor='loss', patience=15),
                ModelCheckpoint('reg-l1ols-model', monitor='loss', save_best_only=True)
            ]
        )
        return self
