import matplotlib.pyplot as graph


def plot_roc_curve(prediction_probability, true, label='', plot_curve_only=False, show_graph=False):
    from sklearn.metrics import roc_curve, roc_auc_score
    label = label if label == '' else label + ' '

    fpr, tpr, thres = roc_curve(true, prediction_probability)
    auc = roc_auc_score(true, prediction_probability)

    graph.plot(fpr, tpr, label='{}AUC = {}'.format(label, auc))
    if not plot_curve_only:
        graph.plot([0, 1], [0, 1], linestyle='--', color='k', label='Guessing')
    graph.xlim([0, 1])
    graph.ylim([0, 1])
    graph.legend(loc=0)
    graph.xlabel('False Positive Rate')
    graph.ylabel('True Positive Rate')

    if show_graph:
        graph.show()

    return {'fpr': fpr, 'tpr': tpr, 'threshold': thres}


def plot_learning_curve(means, stds, xs=None, n=None, show_graph=False):
    """
    Plot learning curve with confidence intervals

    :param xs: What the units on the x-axis should be
    :param n: sample size, usually the number of CV intervals
    :param means:
    :param stds:
    :param show_graph:
    :return:
    """
    import numpy as np
    xs = xs if xs is not None else np.arange(len(means))

    # If N is given, compute the standard error
    stds = stds / np.sqrt(n) if n is not None else stds
    ci95 = stds * 1.96

    graph.plot(xs, means)
    graph.fill_between(
        xs,
        means - ci95, means + ci95,
        alpha=0.4
    )
    if show_graph:
        graph.show()


def plot_confusion_matrix(y_true, y_pred, labels: list=None, axis=1, show_graph=False):
    """
    Normalised Confusion Matrix

    :param y_true:
    :param y_pred:
    :param labels:
    :param axis: 0 if you want to know the probabilities given a predication. 1 if you want to know class confusion.
    :param show_graph:
    :return:
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    labels = True if labels is None else labels

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=axis, keepdims=True)

    sns.heatmap(
        cm,
        annot=True, square=True, cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    graph.xlabel('Predicted')
    graph.ylabel('True')

    if show_graph:
        graph.show()


def plot_ecdf(x, plot_kwargs=None, show_graph=False):
    """
    Create the plot of the empricial distribution function

    >>> import numpy as np
    >>> import matplotlib.pyplot as graph
    >>> plot_ecdf(np.random.normal(100, 15, size=100), {'label': 'blah'})
    >>> graph.show()

    :param x:
    :param plot_kwargs:
    :param show_graph:
    :return:
    """
    from rosey.stats import ecdf

    cdf, x = ecdf(x)
    plot_kwargs = dict() if plot_kwargs is None else plot_kwargs

    graph.plot(x, cdf, **plot_kwargs)

    if show_graph:
        graph.show()


def plot_confusion_probability_matrix(
        y_true, y_pred, y_pred_proba,
        labels: list=None, figsize=(8, 8), rug_height=0.05, show_graph=False
):
    """
    Confusion matrix where you can see the histogram of the

    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> import matplotlib.pyplot as graph
    >>> x, y = load_breast_cancer(return_X_y=True)
    >>> model = LogisticRegression().fit(x, y)
    >>> ypp = model.predict_proba(x)[:, 1]
    >>> plot_confusion_probability_matrix(y, model.predict(x), model.predict_proba(x))
    >>> graph.show()
    >>> plot_confusion_probability_matrix(y, model.predict(x), model.predict_proba(x), labels=['Maligant', 'Benign'])
    >>> graph.show()

    :param y_true:
    :param y_pred:
    :param y_pred_proba:
    :param labels:
    :param figsize:
    :param rug_height:
    :param show_graph:
    :return:
    """
    import numpy as np
    from itertools import product
    from sklearn.metrics import confusion_matrix
    from rosey.stats import solve_n_bins

    n_classes = y_pred_proba.shape[1]
    labels = list(range(n_classes)) if labels is None else labels
    cm = confusion_matrix(y_true, y_pred)

    # Create subplots
    figure, box = graph.subplots(n_classes, n_classes, sharex='all', figsize=figsize)

    # Create histograms
    for i, j in product(range(n_classes), range(n_classes)):
        selection_mask = (y_true == i) & (y_pred == j)
        assert selection_mask.sum() == cm[i, j]

        subset_probabilities = y_pred_proba[selection_mask, i]
        box[i, j].set_title(f'N: {cm[i, j]}')
        box[i, j].hist(subset_probabilities, density=True, bins=solve_n_bins(subset_probabilities), alpha=0.7)
        box[i, j].plot(subset_probabilities, np.ones(len(subset_probabilities)) * rug_height, '|', alpha=0.7)
        box[i, j].set_yticks([])

    # Axis labels
    for k in range(n_classes):
        box[-1, k].set_xlabel(f'Pred = {labels[k]}')
        box[k, 0].set_ylabel(f'True = {labels[k]}')

    if show_graph:
        graph.show()


def plot_barplot_from_dict(d: dict, orient: str='h', show_graph=False):
    """
    Create bar plot from a dictionary that maps a name to the size of the bar

    >>> dictionary = {'dogs': 10, 'cats': 4, 'birbs': 8}
    >>> plot_barplot_from_dict(dictionary, show_graph=True)

    :param d:
    :param orient:
    :param show_graph:
    :return:
    """
    orient = orient.lower()

    if 'h' not in orient and 'v' not in orient:
        raise ValueError('`orient` must be either `h` for horizontal and `v` for veritcal')

    bar_plot = graph.barh if 'h' in orient else graph.bar
    bar_plot(
        range(len(d.values())),
        list(d.values()),
        tick_label=list(d.keys())
    )

    if show_graph:
        graph.show()


def plot_2d_histogram(x, y, bins=100, transform=lambda z: z, show_graph=False):
    """
    Creates a 2D histogram AND allows you to transform the colors of the histogram with the transform function

    Datashader like functionality without all the hassle
    :param x:
    :param y:
    :param bins:
    :param transform: function that takes 1 argument used to transform the histogram
    :param show_graph:
    :return:
    """
    import numpy as np

    h, *_ = np.histogram2d(x, y, bins=bins)
    h = np.rot90(h)

    graph.imshow(transform(h), aspect='auto')
    if show_graph:
        graph.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
