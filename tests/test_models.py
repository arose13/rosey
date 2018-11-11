def test_probability_logistic_regression():
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from rosey.models import ProbabilityLogisticRegression

    x, y = load_breast_cancer(return_X_y=True)

    logit = LogisticRegression()
    logit.fit(x, y)
    y_prob = logit.predict_proba(x)[:, 1]

    plr = ProbabilityLogisticRegression()
    plr.fit(x, y_prob)

    # Raises assertion error if it fails
    np.testing.assert_array_almost_equal(plr.coef_, np.squeeze(logit.coef_))


def test_binary_logit():
    # TODO 11/11/2018 finish writing this test!
    from sklearn.datasets import load_breast_cancer
    from rosey.models import BinaryLogisticRegression

    x, y = load_breast_cancer(return_X_y=True)

    binary_logit = BinaryLogisticRegression()
    binary_logit.fit(x, y)
    # TODO 11/11/2018 continue and compare with statsmodels result


def test_biplot():
    import matplotlib.pyplot as graph
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_iris
    from rosey.graphing import plot_biplot

    data = load_iris()
    x, y = data.data, data.target
    pca = PCA(whiten=True)
    pca.fit(x)

    plot_biplot(pca, feature_names=data.feature_names, c=y)
    graph.close()

    plot_biplot(pca, data=x, feature_names=data.feature_names, c=y)
    graph.close()
