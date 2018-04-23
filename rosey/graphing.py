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
