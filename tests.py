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


if __name__ == '__main__':
    test_plr()
