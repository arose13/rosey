import warnings
import numpy as np
import pandas as pd
import numpy.linalg as la
from tqdm import tqdm
from multiprocessing import cpu_count
from scipy.sparse.linalg import eigsh
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, log_loss
from sklearn.feature_selection import f_regression, f_classif
from sklearn.preprocessing import KernelCenterer, scale
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from rosey.helpers import vec_to_array


def lambda_test(p_values, df=1):
    """
    Test for p-value inflation for hinting at population stratification or cryptic species.

    Paper: https://neurogenetics.qimrberghofer.edu.au/papers/Yang2011EJHG.pdf

    :param p_values: array of p-values
    :param df: Degrees of freedom for the Chi sq distribution
    :return: lambda gc
    """
    from scipy.stats import chi2
    assert np.max(p_values) <= 1 and np.min(p_values) >= 0, 'These do not appear to be p-values'

    chi_sq_scores = chi2.ppf(1 - p_values, df)
    return np.median(chi_sq_scores) / chi2.ppf(0.5, df)


def p_value_inflation_test(p_values):
    """
    Test for p-value inflation using the Kolmogorov-Smirnov test.
    However there are no assumptions to fail for this test.

    :param p_values: array of p-values
    :return: p-value (significant is bad)
    """
    from scipy.stats import ks_2samp
    h_null = np.random.uniform(0, 1, size=int(1e6))
    d, p_value = ks_2samp(p_values, h_null)
    return p_value, d


def compute_distance_matrix(input_df, metric_func, fast=True, is_ones=False) -> pd.DataFrame:
    """
    Compute a matrix that is symmetric down the diagonal. This does not compute the entire at once which is good on RAM

    :param is_ones: value for the diagonal
    :param input_df:
    :param metric_func: scipy style metric function
    :param fast: Whether to use multiprocessing or not.
    :return:
    """
    from tqdm import tqdm
    from scipy.misc import comb
    from itertools import combinations
    from sklearn.metrics.pairwise import pairwise_distances

    if fast:
        # Fast and multiprocessing
        output = pd.DataFrame(
            pairwise_distances(input_df.values, metric=metric_func, n_jobs=-1),
            columns=input_df.index,
            index=input_df.index
        )
    else:
        # Raw pandas with progress bar
        n = len(input_df)
        warnings.warn('I strongly recommend using `sklearn=True` as it is parallel and faster')
        output = pd.DataFrame(
            np.ones((n, n)) if is_ones else np.zeros((n, n)),
            columns=input_df.index,
            index=input_df.index
        )

        for u, v in tqdm(combinations(input_df.iterrows(), 2), total=int(comb(n, 2))):
            ui, ux = u
            vi, vx = v
            score = metric_func(ux, vx)

            # Store results on both sides of the matrix
            output.at[ui, vi] = score
            output.at[vi, ui] = score

    return output


def solve_n_bins(x):
    """
    Uses the Freedman Diaconis Rule for generating the number of bins required

    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule

    Bin Size = 2 IQR(x) / (n)^(1/3)
    """
    from scipy.stats import iqr

    x = np.asarray(x)
    hat = 2 * iqr(x) / (len(x) ** (1 / 3))

    if hat == 0:
        return int(np.sqrt(len(x)))
    else:
        return int(np.ceil((x.max() - x.min()) / hat))


def neg_log_p(p):
    """
    >>> neg_log_p(0.1)
    1.0
    >>> neg_log_p(0.01)
    2.0
    >>> neg_log_p(1e-6)
    6.0
    >>> neg_log_p(10)
    -1.0
    """
    return np.nan_to_num(-np.log10(p))


def minor_af(alt_af: np.ndarray):
    """
    Converts the alternate allele frequency to the minor allele frequency

    >>> a = np.array([0.01, 0.2, 0.7, 0.9])
    >>> res = minor_af(a)
    >>> np.allclose(res, np.array([0.01, 0.2 , 0.3 , 0.1 ]))
    True

    :param alt_af:
    :return:
    """
    maf = alt_af.copy()
    maf[np.where(maf > 0.5)] = 1 - maf[np.where(maf > 0.5)]
    return maf


def most_common(iterable):
    """
    >>> most_common([1, 2, 2, 3, 4, 4, 4, 4])
    4
    >>> most_common(list('Anthony'))
    'n'
    >>> most_common('Stephen')
    'e'

    Exceptionally quick! Benchmark at 12.1 us
    :param iterable:
    :return:
    """
    from collections import Counter

    data = Counter(iterable)
    return data.most_common(1)[0][0]


def compute_null_ss(y: np.ndarray):
    return ((y - y.mean()) ** 2).sum()


def pymc3_r2(y_true, y_pred_rv):
    """
    Allows you to get a trace of a model's r2

    :param y_true: array like
    :param y_pred_rv: PyMC3 Rv trace
    :return:
    """
    import pymc3.math as pmath
    return 1 - (pmath.sum(pmath.sqr(y_true - y_pred_rv)) / compute_null_ss(y_true))


def pymc3_mse(y_true, y_pred_rv):
    """
    Allows you to get a trace of a model's mean square error

    :param y_true: array like
    :param y_pred_rv: PyMC3 Rv trace
    :return:
    """
    import pymc3.math as pmath
    n = len(y_true)
    return pmath.sum(pmath.sqr(y_true - y_pred_rv)) / n


def r2_from_deviance(residuals_deviance, null_deviance):
    return 1 - (residuals_deviance / null_deviance)


def logit_model_loglikelihood(y_true, x, logit_model: LogisticRegression):
    """
    Computes the loglikelihood of a logistic regression model

    :param y_true:
    :param x:
    :param logit_model:
    :return:
    """
    from scipy.stats import logistic
    q = 2 * y_true - 1
    y = (x @ logit_model.coef_.flatten()) + logit_model.intercept_
    return logistic.logcdf(q * y).sum()


def r2_for_logit(y_true, x, logit_model: LogisticRegression):
    """
    Logistic Regression model is used to compute the pseudo R^2 using McFadden estimate

    pseudo R^2 = 1 - ln(L_model) / ln(L_null_model)
    """
    import copy

    assert len(y_true) == len(x), 'X and y must be properly paired'

    # Create null model
    null_model = copy.deepcopy(logit_model)
    null_model.coef_ = np.zeros(null_model.coef_.shape)

    # Compute log likelihood
    model_log_likelihood = log_loss(y_true, logit_model.predict_proba(x))
    null_model_log_likelihood = log_loss(y_true, null_model.predict_proba(x))

    # Pseudo Rsq
    return 1 - (model_log_likelihood / null_model_log_likelihood)


def adj_r2(r2: float, sample_size: int, n_features: int) -> float:
    """
    >>> round(adj_r2(0.8, 100, 5), 3)
    0.789
    >>> round(adj_r2(0.8, 20, 5), 3)
    0.729
    """
    return 1 - ((sample_size - 1) / (sample_size - n_features - 1)) * (1 - r2)


def proportion_ci(p, n, alpha=0.05):
    """
    Create credible intervals for percentage data

    :param p:
    :param n:
    :param alpha:
    :return:
    """
    from scipy.stats import binom
    lower, upper = binom.interval(1 - alpha, n, p)
    return lower / n, upper / n


def af_to_genotypes(alt_af):
    """
    Returns the Hardy Weinberg equilibrium for a given alternate pooled allele frequency

    :param alt_af: [0, 1]
    :return: [homozygous Reference, heterozygous, homozygous Alternate]
    """
    return (1 - alt_af) ** 2, 2 * alt_af * (1 - alt_af), alt_af ** 2


def baseline_classifier(x, y, metric, prob=False):
    from sklearn.dummy import DummyClassifier

    for strategy in ['stratified', 'most_frequent', 'prior', 'uniform']:
        dummy = DummyClassifier(strategy=strategy)
        dummy.fit(x, y)
        y_pred = dummy.predict_proba(x)[:, 1] if prob else dummy.predict(x)
        print('{} = {}'.format(
            strategy,
            metric(y, y_pred)
        ))


def replace_blank_with_zero(a):
    return 0 if a == '' else float(a)


def fix_proportions(a: np.array):
    """
    Compute the proportion fixed for reference allele and alternate allele
    """
    fix_ref = len(a[a == 0]) / len(a)
    fix_alt = len(a[a == 1]) / len(a)
    not_fix = 1 - fix_ref - fix_alt
    return not_fix, {'fixed ref': fix_ref, 'fixed alt': fix_alt}


def bonferroni(p_values, alpha=0.05):
    """
    Bonferroni Correction

    :param p_values:
    :param alpha:
    :return:
    """
    return alpha / len(p_values)


def bh_procedure(p_values: pd.Series, alpha=0.05):
    """
    Benjamini–Hochberg procedure

    https://en.wikipedia.org/wiki/False_discovery_rate#BH_procedure
    http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf
    """
    assert len(p_values) > 0, 'p-value list in empty'
    assert sum(np.isnan(p_values)) == 0, 'You cannot have nans in the p-value list'

    p_values, m = sorted(p_values), len(p_values)

    max_condition = 0
    for i, p in enumerate(p_values):
        rank = i + 1
        if p <= (rank * alpha) / m:
            max_condition = p
        else:
            break

    return max_condition


def ecdf(data):
    """
    Empirical CDF (x, y) generator
    """
    x = np.sort(data)
    cdf = np.linspace(0, 1, len(x))
    return cdf, x


def elbow_detection(data, threshold=1, tune=0.01, get_index=False):
    cdf, ordered = ecdf(data)
    data_2nd_deriv = np.diff(np.diff(ordered))
    elbow_point = np.argmax(data_2nd_deriv > threshold) - int(len(cdf) * tune)
    return elbow_point if get_index else ordered[elbow_point]


def get_n(vector):
    """
    Count non-nan values in a vector
    """
    return np.count_nonzero(~np.isnan(vector))


def rotate_feature(feature: np.ndarray, k_matrix: np.ndarray = None, eigen_vectors: np.ndarray = None):
    """
    Performs a spectral decomposition of K (k_matrix) and then uses that to rotate the feature to uncorrelate them.

    :param eigen_vectors: Use eigen vectors if given to save computation time
    :param feature:
    :param k_matrix:
    :return:
    """

    if k_matrix is None and eigen_vectors is None:
        raise ValueError('You must either give an GRM or the eigen vectors of that matrix')

    if eigen_vectors is None:
        _, eigen_vectors = la.eig(k_matrix)

    return eigen_vectors.T @ feature


def repeated_rbf(x, center, period, std=1):
    """
    Repeated Radial Basis Function

    Great for seasonal, weekly, etc prediction using linear models.

    >>> import numpy as np
    >>> data = np.array([0, 365, 2*365])
    >>> repeated_rbf(data, 0, 365, std=15)
    array([1., 1., 1.])

    :param x: time
    :param center: center in the period the peak occurs at
    :param period: how often the cycle repeats
    :param std: standard deviation (width of the RBF)
    :return:
    """
    span = x.max() - x.min()
    n_repeats = int(span // period) + 1
    cycles = np.ones(n_repeats) * np.arange(0, n_repeats) * period

    # Repeating peaks
    centers = cycles - (-center)
    centers = np.tile(centers, (len(x), 1))

    # RBF function. Get the .max() RBF across columns to get the closest peak
    delta = (x - centers.T).T
    return np.exp((-(delta ** 2)) / (2 * std ** 2)).max(axis=1)


def get_q_feature(target_name: str, random_effect: str, data: pd.DataFrame) -> pd.Series:
    """
    Performs the a transformation meant to approximate that done by the get_precise_q_feature(). There are situations
    (example, simpson's paradox) where this solution does NOT approximate the precise solution at all.

    :param target_name:
    :param random_effect:
    :param data:
    :return:
    """
    q_target = data[target_name] - data.groupby(random_effect).transform('mean')[target_name]
    assert np.isclose(q_target.mean(), 0), f'{target_name} could not be transformed to a N(0, sigma) vector'
    return q_target


def get_precise_q_feature(
        feature_name: str, random_effect: str, x_name: str, data: pd.DataFrame, verbose=True, return_rfx=False
):
    """
    Performs the a transformation to the feature (y) with respect to the random effect (Zu) that hopefully makes
    linear models approximately equal to linear mixed models.

    q = y - Zu

    y = Xm + Zu + b
    y - Zu = Xm + Zu + b - Zu
    q = Xm + b

    Therefore solving m here should yield the same value.

    :param random_effect:
    :param x_name: name of the x variable truly associated with feature to transformed
    :param data:
    :param verbose:
    :param return_rfx:
    :param feature_name:
    :return:
    """
    from bambi import Model as BayesModel

    data[random_effect] = data[random_effect].astype(str)

    rfx_model = BayesModel(data)
    rfx_results = rfx_model.fit(f'{feature_name} ~ {x_name}', random=[f'1|{random_effect}'], samples=2000)

    # Trace information
    rfx_trace = rfx_results[500::3].to_df(ranefs=True)
    random_cols = [r for r in rfx_trace if f'1|{random_effect}[' in r]

    # Compute u vector
    group_ids = [g.replace(f'1|{random_effect}[', '').replace(f']', '') for g in random_cols]
    group_rfx_dict = {gid: rfx_trace[gcol].mean() for gid, gcol in zip(group_ids, random_cols)}
    if verbose:
        print(group_rfx_dict)

    # Generate q feature
    q = data.apply(lambda x: x[feature_name] - group_rfx_dict[x[random_effect]], axis='columns')
    q = q - rfx_trace['Intercept'].mean()
    return q if return_rfx is False else (q, group_rfx_dict)


def normalise_percents(x: np.ndarray or pd.Series, should_impute=False) -> np.ndarray or pd.Series:
    """
    If the vector contains numbers whose numbers are [0, 1] then this is transformation that properly handles the
    difference in how the variance of the kind of a feature is computed.

    x_norm = (x - mu) / sqrt(mu * (1 - mu))

    :param should_impute: If True then nans are imputed with the mean of x
    :param x:
    :return:
    """
    # NOTE Currently python doesn't allow you to write this as 0 <= x <= 1
    assert (0 <= x[~np.isnan(x)]).all() and (x[~np.isnan(x)] <= 1).all(), 'Values must be [0, 1]'
    mu = np.nanmean(x)
    if should_impute:
        x[np.isnan(x)] = mu
    return (x - mu) / np.sqrt(mu * (1 - mu))


def autocorrelate(x, n_lags=100):
    from tqdm import tqdm
    import scipy.stats as stats

    n_lags = n_lags if n_lags <= len(x) else int(len(x) / 5)
    lags = np.arange(-n_lags, n_lags)

    # TODO this can be vectorized
    # Compute Autocorrelation
    corrs, p_values = [], []
    for i in tqdm(range(len(lags)), desc='Autocorrelating'):
        c, p = stats.pearsonr(x, np.roll(x, lags[i]))
        corrs.append(c)
        p_values.append(p)

    # Create result
    df = pd.DataFrame(lags, columns=['lags'])
    df['corr'] = corrs
    df['p'] = p_values
    return df


def _parallel_fit_models(estimator, x, y):
    return estimator.fit(x, y)


def _parallel_fit_dist(i, data):
    from scipy.stats import t
    return i, t.fit(data)


def sklearn_p_values(
        base_estimator, x, y, fitted_coefs, n_permutations=1000, verbose=0, n_jobs=-1, full_output=False
) -> np.ndarray:
    """
    Uses a permutation test to generate the p-values by permuting y

    Params are assumed to be Student T distributed

    :param base_estimator:
    :param x:
    :param y:
    :param fitted_coefs: hypothesis to test
    :param n_permutations:
    :param verbose:
    :param full_output: If True is returns the function for computing p values
    :return: Approximate p-values
    """
    import copy
    from scipy.stats import t
    from .helpers import np_min

    def is_bad_coef_array(input_coefs):
        return input_coefs.shape[0] != n_permutations or input_coefs.shape[1] != len(fitted_coefs)

    # Generate Null Hypothesis
    ys = [y.copy() for _ in range(n_permutations)]
    _ = [np.random.shuffle(ys[i]) for i in range(n_permutations)]

    # Fit Null Hypothesis
    null_models = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_parallel_fit_models)(copy.deepcopy(base_estimator), x, ys[i])
        for i in range(n_permutations)
    )

    # Get feature params
    coefs = np.vstack([est.coef_ for est in null_models])
    if is_bad_coef_array(coefs):
        # Maybe this will work but if not crash!
        coefs = np.vstack([est.coef_[0] for est in null_models])
        if is_bad_coef_array(coefs):
            raise ValueError(f'Cannot retrieve `coef_` as expected from base_estimator received {coefs}')

    # Approximate Parameters
    t_results = Parallel(n_jobs=-1, verbose=verbose)(
        delayed(_parallel_fit_dist)(i, coefs[:, i]) for i in range(coefs.shape[1])
    )
    t_results.sort(key=lambda res: res[0])
    t_params = [params_i for _, params_i in t_results]
    t_df, t_loc, t_scale = zip(*t_params)
    t_df, t_loc, t_scale = [np.array(a) for a in (t_df, t_loc, t_scale)]
    params = dict(df=t_df, loc=t_loc, scale=t_scale)

    # Compute p-values
    compute_p_values = lambda d: np_min(t.sf(d, **params), t.cdf(d, **params)) * 2
    p_values = compute_p_values(fitted_coefs)

    return (p_values, compute_p_values) if full_output else p_values


def bootstrapped_ci(base_estimator, x, y, n_resamples=100, is_regression=True, n_jobs=-1, verbose=0):
    """
    Use Bootstrapped models to get distributions of params to estimate credible distribution.
    Those credible intervals are then used to estimate the distance of the mean from 0.

    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_breast_cancer
    >>> x, y = load_breast_cancer(return_X_y=True)
    >>> _ = bootstrapped_ci(LinearRegression(), x, y)
    These are not p-values!
    >>> print('Done')
    Done

    :param n_jobs:
    :param verbose:
    :param base_estimator: SKLearn style linear model that has a coef_ attribute after fitting.
    :param x:
    :param y:
    :param n_resamples: How many Bootstrap draws to make
    :param is_regression:
    :return:
    """
    import scipy.stats as stats
    from rosey.helpers import np_min
    from sklearn.ensemble import BaggingRegressor, BaggingClassifier

    print('These are not p-values!')
    if is_regression:
        # Fit Models
        bootstrapped_models = BaggingRegressor(
            base_estimator, n_estimators=n_resamples,
            bootstrap=True, bootstrap_features=False,
            n_jobs=n_jobs, verbose=verbose
        )
        bootstrapped_models.fit(x, y)

        # Get params
        model_coefs = np.vstack([est.coef_ for est in bootstrapped_models.estimators_])
        coefs_mu = model_coefs.mean(axis=0)
        coefs_sd = model_coefs.std(axis=0)
        vec = coefs_mu.shape

        # Compute p values
        p_values = np_min(
            stats.norm.pdf(np.zeros(vec), loc=coefs_mu, scale=coefs_sd),
            stats.norm.cdf(np.zeros(vec), loc=coefs_mu, scale=coefs_sd)
        )
        if any(p_values > 1):
            warnings.warn('Bad p-values detected')

        return pd.DataFrame(np.vstack([coefs_mu, coefs_sd, p_values]).T, columns=['mu', 'sd', 'p'])
    else:
        raise NotImplementedError


def recover_ols_coef(coefs_ln, lam, method='l1'):
    """
    Recovers the OLS estimate from regularisation methods

    'Elements of Statistical Learning' (2009)

    :param coefs_ln:
    :param lam:
    :param method:
    :return:
    """
    method = method.lower()
    if method == 'l1':
        return coefs_ln + np.sign(coefs_ln) * lam
    elif method == 'l2':
        return coefs_ln / (1 + lam)


def l1_coef_is_probably_correct(coefs: np.ndarray, x_fitted: np.ndarray, return_result=False):
    """
    Tells you if an L1 model correctly found the non-zero coefs in the limit

    'Sharp Thresholds for High-Dimensional and Noisy Sparsity Recovery Using -Constrained Quadratic Programming'
    (Wainwright 2009)

    :param coefs: L1 coefs
    :param x_fitted: data used to fit the L1 Model
    :param return_result: If True then returns 1 for good, 0 for unsure and -1 for bad
    :return:
    """
    n, p = x_fitted.shape
    k = len(coefs) - np.isclose(coefs, 0).sum()

    pre = k * np.log(p - k)
    success_cond = 2 * pre
    failure_cond = 0.5 * pre

    if n > success_cond:
        if return_result:
            return 1
        else:
            print(f'L1 is likely to succeed! ({n} > {success_cond})')

    if n < failure_cond:
        if return_result:
            return 0
        else:
            print(f'L1 is likely to fail ({n} < {failure_cond})')

    if failure_cond < n < success_cond:
        if return_result:
            return -1
        else:
            print(f'Unsure of L1 power ({failure_cond} < {n} < {success_cond})')


def l1_permutation_test(
        x, y, alpha, estimated_coefs: np.ndarray, is_regression=True,
        n_permutations=20000, n_jobs=-1, verbose=0
):
    """
    L1 permutations test for p-values

    'Resampling-based tests for Lasso in genome-wide association studies' (Arbet 2017)

    >>> import warnings
    >>> from sklearn.datasets import load_boston, load_breast_cancer
    >>> from glmnet import ElasticNet, LogitNet
    >>> data, target = load_boston(return_X_y=True)
    >>> l1 = ElasticNet().fit(data, target)
    >>> warnings.simplefilter('ignore')
    >>> p_values = l1_permutation_test(data, target, l1.lambda_best_, l1.coef_, is_regression=True)
    >>> len(p_values) == len(l1.coef_)
    True
    >>> warnings.simplefilter('default')
    >>> data, target = load_breast_cancer(return_X_y=True)
    >>> l1 = LogitNet().fit(data, target)
    >>> warnings.simplefilter('ignore')
    >>> p_values = l1_permutation_test(data, target, l1.lambda_best_, l1.coef_, is_regression=False)
    >>> len(p_values) == len(np.squeeze(l1.coef_))
    True

    :param x:
    :param y:
    :param alpha: lambda or the regularisation strength
    :param estimated_coefs:
    :param is_regression:
    :param n_permutations:
    :param n_jobs:
    :param verbose: 0 is no output but the more positive the the more output
    :return:
    """
    from .models import L1Regressor, L1Classifier

    estimated_coefs = np.squeeze(estimated_coefs)
    if estimated_coefs.ndim != 1:
        raise ValueError('`estimated_coefs` could not be coerced into a vector. Check input!')

    # Permute y and fit L1 model to add get set of null models
    ys = [y.copy() for _ in range(n_permutations)]
    _ = [np.random.shuffle(ys[i]) for i in range(n_permutations)]

    L1 = L1Regressor if is_regression else L1Classifier

    warnings.simplefilter('ignore')
    null_models = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_parallel_fit_models)(L1(lam=alpha), x, ys[i]) for i in range(n_permutations)
    )
    permuted_coefs = np.vstack([np.square(est.coef_) for est in null_models])
    warnings.simplefilter('default')

    # Calculate permutation p-value
    coef_iterator = range(permuted_coefs.shape[1])
    if verbose != 0:
        coef_iterator = tqdm(coef_iterator)

    p_values = np.ones(permuted_coefs.shape[1])
    for i in coef_iterator:
        p_values[i] = ((np.abs(permuted_coefs[:, i]) >= np.abs(estimated_coefs[i])).sum() + 1) / (n_permutations + 1)

    if any(p_values > 1) or any(p_values < 0):
        warnings.warn('Bad p-values computed!')

    return p_values


def is_positive_definite(x):
    """
    If all eigenvalues are greater than 0 then the matrix is positive definite!
    """
    return np.all(np.linalg.eigvals(x) > 0)


def is_positive_semi_definite(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def _parallel_permute_count_nonzero_penalised_coefs(xp, yp, lam_path, penalties, norm_num, is_regression):
    from glmnet import ElasticNet, LogitNet
    np.random.shuffle(yp)

    params = dict(alpha=norm_num, lambda_path=lam_path)
    pm = ElasticNet(**params) if is_regression else LogitNet(**params)
    pm.fit(xp, yp, relative_penalties=penalties)

    return np.sign(np.abs(np.squeeze(pm.coef_path_)) * vec_to_array(penalties)).sum(axis=0)


def _resampled_model(x, y, model, relative_penalties):
    x_boot, y_boot = resample(x, y, replace=True)
    model.fit(x_boot, y_boot, relative_penalties=relative_penalties)
    coef_nonzero = np.sign(np.abs(np.squeeze(model.coef_)))
    return coef_nonzero


# noinspection PyPep8Naming
class PenalisedFDRControl:
    """
    Implemented 'yi' the 'FDR control using data permutation' method from
    Penalised Multimarker vs Single-Marker Regression Methods for Genome-Wide Association Studies of Quantitative Traits
    (Yi et al 2015)

    Also if method is 'arbet' then implements
    Implemented the type-1-error control from Arbet et al Permutations to select lambda for type-1-error control
    'Resampling-based tests for Lasso in genome-wide association studies'
    (Arbet et al 2017)

    >>> import numpy as np
    >>> from sklearn.datasets import load_breast_cancer, load_boston
    >>> x, y = load_boston(True)
    >>> reg_fdr = PenalisedFDRControl().fit(x, y)
    >>> np.isclose(reg_fdr.model.score(x, y), 0.618, atol=0.01)
    True
    >>> x, y = load_breast_cancer(True)
    >>> clf_fdr = PenalisedFDRControl(is_regression=False).fit(x, y)
    >>> np.isclose(clf_fdr.model.score(x, y), 0.984, atol=0.01)
    True

    """

    def __init__(
            self, penalty_free_indices=list(),
            min_lambda_ratio=1e-3, n_lambdas=250, cv=10, is_regression=True, norm_num=1
    ):
        from glmnet import ElasticNet, LogitNet

        if not (isinstance(penalty_free_indices, list) or isinstance(penalty_free_indices, np.ndarray)):
            raise ValueError('ols_indices must be a list or np.array')

        if is_regression:
            self.model = ElasticNet(norm_num, n_lambdas, min_lambda_ratio, n_splits=cv, n_jobs=cpu_count())
        else:
            self.model = LogitNet(norm_num, n_lambdas, min_lambda_ratio, n_splits=cv, n_jobs=cpu_count())

        self.norm_num = norm_num
        self.ols_idx = penalty_free_indices
        self.is_regression = is_regression

        self.n = None
        self.p = None
        self.coef_path = None
        self.lambdas = None
        self.fdr_grid = None
        self.fdr_analytic_grid = None
        self.n_nonzero_true_coefs = None
        self.mean_n_false_positive_coefs = None

    def _penalty_weights(self):
        penalty_weights = np.ones(self.p)
        penalty_weights[self.ols_idx] = 0
        return penalty_weights

    def plot_coef_path(self, only_penalised_coefs=True, complete=False, show_graph=False):
        import matplotlib.pyplot as graph

        if self.fdr_grid is None:
            raise NotFittedError

        coef_path = self.model.coef_path_ if complete else self.coef_path
        lambdas = self.model.lambda_path_ if complete else self.lambdas

        for f in range(coef_path.shape[0]):
            if np.isclose(coef_path[f, :], 0).all():
                continue
            if only_penalised_coefs and f in self.ols_idx:
                continue
            graph.plot(lambdas, coef_path[f, :], linewidth=2, alpha=0.25)
        graph.ylabel(r'$\beta$')
        graph.xlabel(r'$\lambda$')

        if show_graph:
            graph.show()

    def fit(self, X, y, n_permutations=500, n_jobs=-1, verbose=False):
        from scipy.stats import norm

        self.n, self.p = X.shape
        penalties = self._penalty_weights()

        # Fit real model (R)
        if verbose:
            print('Regression Model' if self.is_regression else 'Classification Model')
            print('Fitting y -> R(alpha)')
        self.model.fit(X, y, relative_penalties=self._penalty_weights())

        # Get lasso path
        subset_idx = np.argwhere(self.model.lambda_path_ >= self.model.lambda_best_ * 0.8).flatten()
        self.lambdas = self.model.lambda_path_[subset_idx]
        self.coef_path = np.squeeze(self.model.coef_path_)[:, subset_idx]

        # Compute R(alpha) and multiple by penalties to prevent counting OLS coefs
        self.n_nonzero_true_coefs = np.sign(np.abs(self.coef_path) * vec_to_array(penalties)).sum(axis=0)

        # About to fit single lambda path models, ignore RuntimeWarnings
        warnings.simplefilter('ignore', RuntimeWarning)

        # Compute Permutation FDR Control F(b, alpha)
        iter_perm = range(n_permutations)
        iter_perm = tqdm(iter_perm, desc='Computing permuted FDR y -> F(b, lambda)...') if verbose else iter_perm
        f_grid = Parallel(n_jobs)(
            delayed(_parallel_permute_count_nonzero_penalised_coefs)(
                X, y.copy(), self.lambdas, penalties, self.norm_num, self.is_regression
            ) for _ in iter_perm
        )
        self.mean_n_false_positive_coefs = np.vstack(f_grid).mean(axis=0)
        self.fdr_grid = self.mean_n_false_positive_coefs / self.n_nonzero_true_coefs
        self.fdr_grid[np.isnan(self.fdr_grid)] = 0

        # Compute Analytic FDR Control
        analytic_fdr = []
        prediction_lam = self.model.predict(X, self.lambdas)
        iter_lam = range(len(self.lambdas))
        iter_lam = tqdm(iter_lam, desc='Computing analytic FDR') if verbose else iter_lam
        for i in iter_lam:
            if self.is_regression is False:
                warnings.warn('Analytic FDR was not intended for classification')
            rej = np.sign(np.abs(self.coef_path[:, i]) * penalties).sum()
            residuals = y - prediction_lam[:, i]

            test_val = -((self.lambdas[i] * self.n) / np.sqrt(residuals.T @ residuals))
            probit = norm.cdf(test_val)

            fdr_hat = (2 * self.p * probit) / rej
            analytic_fdr.append(fdr_hat)
        self.fdr_analytic_grid = np.array(analytic_fdr).flatten()

        warnings.simplefilter('default', RuntimeWarning)
        return self

    def compute_coef_stability(self, X, y, penalty, n_samples=500, n_jobs=-1, verbose=False):
        """
        Uses resampling to compute the Prob(beta_i != 0 | penalty)

        :param X:
        :param y:
        :param penalty: lambda to check the
        :param n_samples: effectively controls the resolution as 1/n is the sample resolution
        :param n_jobs:
        :param verbose:
        :return:
        """
        from copy import copy
        from .models import L1Classifier, L1Regressor

        model = L1Regressor(lam=penalty) if self.is_regression else L1Classifier(lam=penalty)
        penalties = self._penalty_weights()
        iterator = range(n_samples)
        iterator = tqdm(iterator, desc=f'Prob(beta_i != 0 | lam={penalty:0.3f})') if verbose else iterator

        warnings.simplefilter('ignore')
        is_nonzero = Parallel(n_jobs)(delayed(_resampled_model)(X, y, copy(model), penalties) for _ in iterator)
        warnings.simplefilter('default')

        is_nonzero = np.vstack(is_nonzero)
        return is_nonzero.mean(axis=0)

    def estimate_fpr(self, penalty):
        """
        Estimates the FPR for any given lambda. This will return the Expected FDR at this rate.
        The approximate False Positive Rate is estimated using permutation testing.

        :param penalty: Lasso alpha param
        :return: The approximate FPR
        """
        from .helpers import find_nearest

        fpr = self.mean_n_false_positive_coefs / self.p
        fpr[np.isnan(fpr)] = 0
        return fpr[find_nearest(self.lambdas, penalty, return_idx=True)]

    def sharp_threshold(self, X: np.ndarray, verbose=False):
        """
        This method finds the lowest value for alpha where the coefficients are all extremely likely to be replicable.

        'Sharp Thresholds for High-Dimensional and Noisy Sparsity Recovery Using l1 Constrained Quadratic Programming'
        (Wainwright 2009)

        :param X:
        :param verbose:
        :return:
        """
        import matplotlib.pyplot as graph

        results = []
        for lambda_pos in range(self.coef_path.shape[1]):
            results.append(l1_coef_is_probably_correct(self.coef_path[:, lambda_pos], X, return_result=True))
        results = np.array(results)

        if verbose:
            graph.plot(self.lambdas, self.fdr_grid, label='FWER')
            graph.plot(self.lambdas, results, label='Thresholds')
            graph.show()

        lowest_alpha_idx = np.argmin(results >= 1)
        return self.lambdas[lowest_alpha_idx]

    def fdr_alpha(self, alpha, method='yi', verbose=False):
        """
        Returns the alpha for L1 regression that best controls for FDR at the rate requested

        :param alpha: FDR (Yi) or FPR (Arbet) or FDR (analytic)
        :param method:
            `yi` uses Yi's FDR = E(F)/R,
            `arbet` uses Arbet's FPR = N_fp / N_features,
            `analytic` uses Yi's analytic FDR control FDR ~= 2p N.cdf((-lam N) / sqrt(r.T @ r)) / R
        :param verbose:
        """
        from .helpers import find_nearest

        metric = ''
        method = method.lower()
        if method == 'yi':
            # Penalized Mutlimarker vs Single-Marker Regression Methods for Genome-Wide Association Studies
            fpr_grid = self.mean_n_false_positive_coefs / self.n_nonzero_true_coefs
            fpr_grid[np.isnan(fpr_grid)] = 0
            metric = 'FDR (False Discovery Rate)'

        elif method == 'arbet':
            # Resampling-based tests for Lasso in genome-wide association studies
            fpr_grid = self.mean_n_false_positive_coefs / self.p
            fpr_grid[np.isnan(fpr_grid)] = 0
            metric = 'FPR (Expected False Positives per Feature)'

        elif method == 'analytic':
            # Penalized Mutlimarker vs Single-Marker Regression Methods for Genome-Wide Association Studies (Analytic)
            fpr_grid = self.fdr_analytic_grid
            fpr_grid[np.isnan(fpr_grid)] = 0
            metric = 'aFDR (analytic False Discovery Rate)'

        else:
            fpr_grid = False
            ValueError('Only supported methods are `yi` `arbet` and `analytic`')

        approx_idx = find_nearest(fpr_grid, alpha, return_idx=True)
        if verbose:
            print(f'{metric} ~{fpr_grid[approx_idx]} @ alpha {self.lambdas[approx_idx]}')
        return self.lambdas[approx_idx]


# noinspection PyPep8Naming
class SupervisedPCA(BaseEstimator, TransformerMixin):
    """
    Supervised Principal component analysis (SPCA)

    Finally for Python 3

    Non-linear dimensionality reduction through the use of kernels.
    Parameters
    ----------
    n_components: int or None
        Number of components. If None, all non-zero components are kept.
    kernel: 'linear' | 'poly' | 'rbf' | 'sigmoid' | 'precomputed'
        Kernel.
        Default: 'linear'
    degree : int, optional
        Degree for poly, rbf and sigmoid kernels.
        Default: 3.
    gamma : float, optional
        Kernel coefficient for rbf and poly kernels.
        Default: 1/n_features.
    coef0 : float, optional
        Independent term in poly and sigmoid kernels.
    eigen_solver: string ['auto'|'dense'|'arpack']
        Select eigensolver to use.  If n_components is much less than
        the number of training samples, arpack may be more efficient
        than the dense eigensolver.
    tol: float
        convergence tolerance for arpack.
        Default: 0 (optimal value will be chosen by arpack)
    max_iter : int
        maximum number of iterations for arpack
        Default: None (optimal value will be chosen by arpack)
    Attributes
    ----------
    `lambdas_`, `alphas_`:
        Eigenvalues and eigenvectors of the centered kernel matrix
    """

    def __init__(self, n_components=None, kernel='linear', gamma=0, degree=3,
                 coef0=1, alpha=1.0, fit_inverse_transform=False,
                 eigen_solver='auto', tol=0, max_iter=None):

        self.n_components = n_components
        self.kernel = kernel.lower()
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.centerer = KernelCenterer()

    def transform(self, X):
        """
        Returns a new X, X_trans, based on previous self.fit() estimates
        """
        return X @ self.alphas_

    def fit(self, X, y):
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError('SPCA requires a target variable')
        self.fit(X, y)
        return X @ self.alphas_

    def _fit(self, X, y):
        # find kernel matrix of Y
        K = self.centerer.fit_transform(self._get_kernel(y))
        # scale X
        X_scale = scale(X)

        n_components = K.shape[0] if self.n_components is None else min(K.shape[0], self.n_components)

        # compute eigenvalues of X^TKX
        M = X.T @ K @ X
        if self.eigen_solver == 'auto':
            if M.shape[0] > 200 and n_components < 10:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self.eigen_solver

        if eigen_solver == 'dense':
            warnings.warn('`dense` is experimental! Please verify results or use < 10 components.')
            self.lambdas_, self.alphas_ = la.eigh(M)
        elif eigen_solver == 'arpack':
            self.lambdas_, self.alphas_ = eigsh(M, n_components, which='LA', tol=self.tol)

        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.alphas_ = self.alphas_[:, indices]

        # remove the zero/negative eigenvalues
        self.alphas_ = self.alphas_[:, self.lambdas_ > 0]
        self.lambdas_ = self.lambdas_[self.lambdas_ > 0]

        self.X_fit = X

    def _get_kernel(self, X, Y=None):
        params = {'gamma': self.gamma,
                  'degree': self.degree,
                  'coef0': self.coef0}
        try:
            return pairwise_kernels(X, Y, metric=self.kernel,
                                    filter_params=True, n_jobs=-1, **params)
        except AttributeError:
            raise ValueError(f'{self.kernel} is not a valid kernel. Valid kernels are: '
                             'rbf, poly, sigmoid, linear and precomputed.')


# noinspection PyPep8Naming
class BairSPCA(BaseEstimator, TransformerMixin):
    """
    Supervised Principal Components Analysis

    This is the one as described by 'Prediction by Supervised Principal Components' (Eric Bair, Trevor Hastie et al)
    https://stats.stackexchange.com/a/767/91928

    NOTE -> Use sklearn LinearRegression over statsmodels OLS because it is ~3x faster.

    Example below
    >>> from sklearn.datasets import load_boston, load_breast_cancer
    >>> require_dims = 3
    >>> data, target = load_boston(True)
    >>> bspca = BairSPCA(n_components=require_dims)
    >>> trans_a = bspca.fit_transform(data, target)
    >>> trans_b = bspca.transform(data)
    >>> trans_a.ndim
    2
    >>> trans_a.shape[1]
    3
    >>> np.isclose(trans_a, trans_b).all()
    True
    >>> data, target = load_breast_cancer(True)
    >>> lspca = BairSPCA(require_dims)
    >>> trans_a = lspca.fit_transform(data, target)
    >>> trans_b = lspca.transform(data)
    >>> trans_a.ndim
    2
    >>> trans_a.shape[1]
    3
    >>> np.isclose(trans_a, trans_b).all()
    True
    >>> print('Done')
    Done
    """

    def __init__(self, n_components=None, is_regression=True, cv=5,
                 threshold_samples=25, use_pvalues=False, verbose=False):
        self.pca = PCA(n_components=n_components, whiten=True)
        self.conditioner_model_ = LinearRegression() if is_regression else LogisticRegression()
        self.cv, self.n_thres = cv, threshold_samples
        self.n_components = n_components
        self.is_regression, self.use_pvalues = is_regression, use_pvalues
        self.cv_results, self.indices, self.best = 3 * [None]
        self.verbose = verbose

        if use_pvalues:
            warnings.warn('Using p-values could select spurious features as important!')

    def _check_is_fitted(self):
        if self.indices is None or self.best is None:
            raise NotFittedError

    def plot_learning_curve(self, show_graph=False):
        import matplotlib.pyplot as graph
        from .graphing import plot_learning_curve as plc
        self._check_is_fitted()

        plc(self.cv_results['mean'], self.cv_results['std'], self.cv_results['theta'], n=self.cv)
        graph.ylabel('R2 Score' if self.is_regression else 'Log loss')
        graph.xlabel(r'$\theta$')
        if show_graph:
            graph.show()

    def _univariate_regression(self, x, y):
        def model(x_i):
            lm_i = LinearRegression() if self.is_regression else LogisticRegression()
            lm_i.fit(vec_to_array(x_i), y)
            return lm_i.coef_[0]

        iterator = range(x.shape[1])
        if self.verbose:
            iterator = tqdm(iterator, desc='Computing Coefs')
        return np.array([model(x[:, i]) for i in iterator])

    def fit(self, X, y):
        # Step 1 -> Compute (univariate) standard regression coefficient for each feature
        if self.use_pvalues:
            _, thetas = f_regression(X, y) if self.is_regression else f_classif(X, y)
            grid_sweep = np.linspace(thetas.min(), 1, self.n_thres)
        else:
            # Compute the regression coef like it says in the paper
            if self.is_regression:
                y_centered = y - np.mean(y)
                thetas = self._univariate_regression(X, y_centered)
            else:
                thetas = self._univariate_regression(X, y)
            # noinspection PyTypeChecker
            grid_sweep = np.percentile(np.abs(thetas), np.linspace(0.01, 1, self.n_thres)[::-1] * 100)

        # Step 2 -> Form a reduced data matrix
        thetas = (thetas if self.use_pvalues else np.abs(thetas)).flatten()
        cv_results = []
        for thres in grid_sweep:
            select = np.squeeze(np.argwhere(thetas <= thres) if self.use_pvalues else np.argwhere(thetas >= thres))
            x_selected = X[:, select]
            try:
                comps = float('inf') if self.n_components is None else self.n_components
                u_selected = PCA(min(x_selected.shape[1], comps), whiten=True).fit_transform(x_selected)
            except (ValueError, IndexError):
                u_selected = x_selected

            kf, scores = KFold(n_splits=self.cv, shuffle=True), []
            for train_ind, val_ind in kf.split(u_selected):
                # Split
                x_train, x_val = u_selected[train_ind], u_selected[val_ind]
                y_train, y_val = y[train_ind], y[val_ind]

                # Fit
                if x_train.ndim == 1:
                    x_train, x_val = vec_to_array(x_train), vec_to_array(x_val)

                if self.is_regression:
                    lm = LinearRegression().fit(x_train, y_train)
                else:
                    lm = LogisticRegression().fit(x_train, y_train)

                # Score
                y_hat = lm.predict(x_val)
                score = r2_score(y_val, y_hat) if self.is_regression else log_loss(y_val, y_hat)

                # Test
                scores.append(score)

            # Score threshold
            scores = np.array(scores)
            cv_results.append((scores.mean(), scores.std()))
            if self.verbose:
                print(f'Theta -> {thres}', cv_results[-1])

        # Get best results
        self.cv_results = pd.DataFrame(cv_results, columns=['mean', 'std'])
        self.cv_results['theta'] = grid_sweep
        self.cv_results = self.cv_results.tail(len(self.cv_results) - 1)

        self.best = self.cv_results.sort_values(by='mean', ascending=False if self.is_regression else True).head(1)
        if self.use_pvalues:
            best_select = np.argwhere(thetas <= self.best['theta'].values)
        else:
            best_select = np.argwhere(thetas >= self.best['theta'].values)
        self.indices = np.squeeze(best_select)

        X = vec_to_array(X[:, self.indices]) if X[:, self.indices].shape[1] == 1 else X[:, self.indices]
        self.pca.fit(X)
        self.conditioner_model_.fit(self.pca.transform(X), y)

        return self

    def transform(self, X, y=None, **fit_params):
        self._check_is_fitted()

        # Step 3 -> Reduce X and then perform PCA
        x_reduced = X[:, self.indices]
        self.n_components = min(x_reduced.shape[1], float('inf') if self.n_components is None else self.n_components)
        return self.pca.transform(x_reduced)

    def fit_transform(self, X, y=None, **fit_params):
        assert y is not None
        if X.ndim == 1:
            raise ValueError('X cannot be a vector')
        elif X.shape[1] == 1:
            raise ValueError('X must have more than 1 feature')

        self.fit(X, y)
        return self.transform(X)

    def precondition(self, X):
        """
        This returns the preconditioned target variable (It predicts y from the input data)

        :param X:
        :return:
        """
        return self.conditioner_model_.predict(self.pca.transform(X[:, self.indices]))


if __name__ == '__main__':
    import doctest

    doctest.testmod()
