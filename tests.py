def test_plr():
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from rosey.models import ProbabilityLogisticRegression

    x, y = load_breast_cancer(True)

    logit = LogisticRegression().fit(x, y)
    y_prob = logit.predict_proba(x)[:, 1]

    plr = ProbabilityLogisticRegression().fit(x, y_prob)

    np.testing.assert_array_almost_equal(plr.coef_, np.squeeze(logit.coef_))
    print('test_plr() passed')


def test_loss_regression():
    import numpy as np
    from statsmodels.api import add_constant
    from sklearn.datasets import load_boston
    from sklearn.metrics import mean_squared_error, mean_squared_log_error
    from sklearn.linear_model import LinearRegression
    from rosey.models import LossRegression

    x, y = load_boston(return_X_y=True)
    x = add_constant(x)

    print((x @ np.ones(x.shape[1])).shape)

    gt = LinearRegression(fit_intercept=False)
    gt.fit(x, y)
    print(gt.coef_)

    clr = LossRegression()
    clr.fit(x, y)


def test_binary_logit():
    # TODO 11/3/2018 check that this test runs as expected
    from sklearn.datasets import load_breast_cancer
    from rosey.models import BinaryLogisticRegression

    x, y = load_breast_cancer(return_X_y=True)

    logit = BinaryLogisticRegression()
    logit.fit(x, y)
    print(logit.score(x, y))


def test_biplot():
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_breast_cancer, load_iris
    from rosey.graphing import plot_biplot
    import matplotlib.pyplot as graph

    data = load_breast_cancer()
    x, y = data.data, data.target
    pca = PCA(whiten=True)
    pca.fit(x)

    plot_biplot(pca, feature_names=data.feature_names, c=y)
    graph.show()

    plot_biplot(pca, data=x, feature_names=data.feature_names, c=y)
    graph.show()

if __name__ == '__main__':
    test_biplot()
    # test_loss_regression()
    # test_binary_logit()
    # test_plr()
