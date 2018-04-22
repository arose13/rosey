import copy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.utils import resample


def _np_dropna(a):
    """Mimics pandas dropna"""
    return a[~np.isnan(a).any(axis=1)]


def make_keras_picklable():
    """
    This must be called on top of the script
    """
    import tempfile
    import keras.models

    def __getstate__(self):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as temp:
            keras.models.save_model(self, temp.name, overwrite=True)
            model_str = temp.read()
        return {'model_str': model_str}

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as temp:
            temp.write(state['model_str'])
            temp.flush()
            model = keras.models.load_model(temp.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


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
