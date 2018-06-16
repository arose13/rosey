import arrow
import numpy as np
import pandas as pd
from functools import partial


fprint = partial(print, flush=True)


def suppress_warnings():
    import warnings
    warnings.simplefilter('ignore')


def coerce_string_to_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def timestamp(date_only=True):
    return arrow.utcnow().format('DD-MM-YYYY' if date_only else 'DD-MM-YYYY-HHmm')


def preprocess_line(l, astype=float):
    a = [coerce_string_to_float(x) for x in l]
    a = np.array(a)
    return a.astype(astype)


def find_nearest(array: np.ndarray, value: np.ndarray, return_idx=False):
    """
    Find the nearest value or index for a value in an array

    >>> a = np.array([1, 2, 3, 6, 7, 8])
    >>> find_nearest(a, 4)
    3
    >>> find_nearest(a, 4, return_idx=True)
    2

    :param array:
    :param value:
    :param return_idx:
    :return:
    """
    idx = (np.abs(array-value)).argmin()
    return idx if return_idx else array[idx]


def vec_to_array(vector: np.ndarray) -> np.ndarray:
    """
    Converts a 1D vector to 2D array with the length of the input vector

    for example
    >>> x = np.linspace(0, 1, 10)
    >>> x.shape
    (10,)
    >>> vec_to_array(x).shape
    (10, 1)

    :param vector:
    :return:
    """
    return vector.reshape((len(vector), 1))


def _np_min(a, b):
    return a if a < b else b


np_min = np.vectorize(_np_min, otypes=[float])


def time_func(f):
    """
    Prints the execution time of the function it decorates

    :param f:
    :return:
    """
    import time

    def wrap(*args):
        time_start = time.time()
        ret = f(*args)
        time_end = time.time()

        duration = time_end - time_start
        if duration < 60:
            duration_string = f'{duration} sec'
        elif 1 < (duration / 60) < 60:
            duration_string = f'{duration / 60} min'
        else:
            duration_string = f'{duration / 3600} hours'

        print(f'{f.__name__} function took {duration_string}')
        return ret
    return wrap


def invert_dict(dictionary):
    return {v: k for k, v in dictionary.items()}


def mutate_string(string, pos, change_to):
    """
    Fastest way I've found to mutate a string
    @ 736 ns

    >>> mutate_string('anthony', 0, 'A')
    'Anthony'

    :param string:
    :param pos:
    :param change_to:
    :return:
    """
    string_array = list(string)
    string_array[pos] = change_to
    return ''.join(string_array)


def verify_pvcf(df: pd.DataFrame, scaffold_col='chrom', position_col='pos'):
    if df[scaffold_col].dtype != np.object:
        raise TypeError('Ensure that the PVCF DataFrame was loaded with the `chrom` column as an object')

    if df[position_col].dtype != np.float and df[position_col].dtype != np.int:
        raise TypeError('Ensure that the PVCF DataFrame was loaded with the `pos` column as a number')


class Dataset(dict):
    """
    Container object for datasets.

    It's like a dictionary but does other things that makes it useful for machine learning.
    Also turns its keys into class attributes.

    >>> d = Dataset(a=1, b=2)
    >>> d['a'] == d.a
    True
    >>> d.a = d.a + d.b
    >>> d.a
    3
    >>> d.process()
    False
    >>> d.data = pd.DataFrame(data=np.zeros((5, 3)), columns=['y', 'x1', 'x2'])
    >>> d.process()
    """
    def __init__(self, **kwargs):
        super(Dataset, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass

    def process(self, x_columns=None, y_columns=None):
        """
        Creates a x and y attribute from a data (array) and a column indices. If the columns indices aren't given the
        first column is assumed to be the y variable.
        """
        if 'data' in self:
            assert type(self.data) is pd.DataFrame, 'data is not {}'.format(type(pd.DataFrame))

            if y_columns is None or x_columns is None:
                self['y'] = self.data[self.data.columns[0]]
                self['x'] = self.data[self.data.columns[1:]]
            else:
                self['y'] = self.data[y_columns]
                self['x'] = self.data[x_columns]
        else:
            return False


if __name__ == '__main__':
    import doctest
    doctest.testmod()
