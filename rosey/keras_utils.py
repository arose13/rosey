# Check out the Loss Functions Notebook if you want to so some graphs
import keras.backend as K
from keras.optimizers import Optimizer
from keras.regularizers import Regularizer
from .Errors import IllegalArgumentsException


########################################################################################################################
# Metric Functions
########################################################################################################################
def keras_frac_var_unexplained(y_true, y_pred):
    """
    Compute the unexplained variance

    SumSq(y - y_hat) / SumSq(y - y_mean)
    """
    return K.sum(K.square(y_true - y_pred)) / K.sum(K.square(y_true - K.mean(y_true)))


def keras_r2(y_true, y_pred):
    """
    Computes the Rsq for Keras models
    """
    return 1 - keras_frac_var_unexplained(y_true, y_pred)


########################################################################################################################
# Loss Functions
########################################################################################################################
def sklearn_mse(y_true, y_pred):
    """
    This is so that SKLearn and Keras has the same MSE function
    a = y - f(x)
    0.5 * mse
    """
    return K.mean(K.square(y_pred - y_true), axis=-1) * 0.5


def huber_loss(y_true, y_pred, delta=1):
    """
    NOTE: only for Keras and TensorFlow
    a = y - f(x)
    0.5 * a ** 2 if np.abs(a) <= delta else (delta * np.abs(a)) - 0.5 * delta ** 2
    """
    a = y_true - y_pred
    cost_i = K.switch(
        a <= delta,
        0.5 * K.pow(a, 2),
        (delta * K.abs(a)) - 0.5 * delta ** 2
    )
    return K.mean(cost_i, axis=-1)


def pseudo_huber_loss(y_true, y_pred, delta=1):
    """
    a = y - f(x)
    (delta ** 2) * (np.sqrt(1 + (a / delta) ** 2) - 1)
    """
    return K.mean((delta ** 2) * (K.sqrt(1 + K.pow((y_true - y_pred) / delta, 2)) - 1))


def log_cosh_loss(y_true, y_pred, delta=1):
    """
    Log of the Hyperbolic Cosine. This is an approximation of the Pseudo Huber Loss
    """
    def _cosh(x):
        return (K.exp(x) + K.exp(-x)) / 2
    return K.mean(K.log(_cosh(y_pred - y_true)), axis=-1)


########################################################################################################################
# Regularisers
########################################################################################################################
class Fusion(Regularizer):
    """
    Base class for all of the fusion regularisers.

    Fused Lasso -> https://web.stanford.edu/group/SOL/papers/fused-lasso-JRSSB.pdf
    Absolute Fused Lasso -> http://www.kdd.org/kdd2016/papers/files/rpp0343-yangA.pdf
    """
    def __init__(self, l1: float=0, fusion: float=0, absolute_fusion: float=0):
        self.l1 = K.cast_to_floatx(l1)
        self.fuse = K.cast_to_floatx(fusion)
        self.abs_fuse = K.cast_to_floatx(absolute_fusion)

    def __call__(self, x):
        regularization = 0.

        x_rolled = self._roll_tensor(x)

        # Add components if they are given
        if self.l1:
            # \lambda ||x||
            regularization += self.l1 * K.sum(K.abs(x))
        if self.fuse:
            # \lambda \sum{ |x - x_+1| }
            regularization += self.fuse * K.sum(K.abs(x - x_rolled))
        if self.abs_fuse:
            # \lambda \sum{ ||x| - |x_+1|| }
            regularization += self.abs_fuse * K.sum(K.abs(K.abs(x) - K.abs(x_rolled)))

        return regularization

    def get_config(self):
        return {
            'l1': float(self.l1),
            'fusion': float(self.fuse),
            'abs_fusion': float(self.abs_fuse)
        }

    @staticmethod
    def _roll_tensor(x):
        vector_length = K.int_shape(x)[0]
        x_tile = K.tile(x, [2, 1])
        return x_tile[vector_length - 1:-1]

    @staticmethod
    def check_alpha_and_ratio(alpha, l1_ratio):
        assert alpha >= 0, 'alpha must be >= 0'
        assert 0.0 <= l1_ratio <= 1.0, 'l1_ratio must be between [0, 1]'

    @staticmethod
    def check_l1_and_fusion(l1, fuse):
        assert l1 is not None and fuse is not None, 'Both l1 and fuse must be given'


def fused_lasso(alpha=2.0, l1_ratio=0.5, l1=None, fuse=None) -> Regularizer:
    """
    Either alpha and l1_ratio must be given or l1 and fuse

    :param alpha: Regularization strength
    :param l1_ratio: Proportion of alpha to transfer to the l1 regularization term
    :param l1:
    :param fuse:
    :return:
    """
    if l1 is None and fuse is None:
        # Use the alpha and l1 ratio
        Fusion.check_alpha_and_ratio(alpha, l1_ratio)
        return Fusion(l1=alpha * l1_ratio, fusion=alpha * (1 - l1_ratio))

    elif l1 is not None or fuse is not None:
        Fusion.check_l1_and_fusion(l1, fuse)
        return Fusion(l1=l1, fusion=fuse)

    else:
        raise IllegalArgumentsException('`l1` and `fuse` must be given OR `alpha` and `l1_ratio`')


def absolute_fused_lasso(alpha=2.0, l1_ratio=0.5, l1=None, abs_fuse=None):
    """
    Either alpha and l1_ratio must be given or l1 and abs_fuse

    :param alpha: Regularization strength
    :param l1_ratio: Proportion of alpha to transfer to the l1 regularization term
    :param l1:
    :param abs_fuse:
    :return:
    """
    if l1 is None and abs_fuse is None:
        # Use the alpha and l1 ratio
        Fusion.check_alpha_and_ratio(alpha, l1_ratio)
        return Fusion(l1=alpha * l1_ratio, absolute_fusion=alpha * (1 - l1_ratio))

    elif l1 is not None or abs_fuse is not None:
        Fusion.check_l1_and_fusion(l1, abs_fuse)
        return Fusion(l1=l1, absolute_fusion=abs_fuse)

    else:
        raise IllegalArgumentsException('`l1` and `fuse` must be given OR `alpha` and `l1_ratio`')


########################################################################################################################
# Optimiser
########################################################################################################################
class SKLearnCoordinateDescent(Optimizer):
    # TODO 17/03/2018 Create this if you want L1 to have values that are exactly zero.
    def get_updates(self, loss, params):
        pass


########################################################################################################################
# Utils
########################################################################################################################
def make_keras_picklable():
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
