def test_loss_regression():
    import numpy as np
    from statsmodels.api import add_constant
    from sklearn.datasets import load_boston
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
