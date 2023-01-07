import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_decision_regions(_x, _y, classifier, test_idx=None, resolution=0.02):
    """
    Plot the 2D-decision region of a classifier
    with matplotlib along its first two dimensions X[:,0] and X[:,1].

    Args:
        X (np.Array): (n,p) dataset to classify
        y (np.Array): (n,) array of labels.
        Works well up to 5 unique labels.
        classifier (sklearn): fitted sklearn classifier.
        test_idx (int, optional):  Index of test datapoints
        within X to display with a larger mark style. Defaults to None.
        resolution (float, optional): Resolution of the
        meshgrid used to colorize regions. Defaults to 0.02.
    """

    # setup marker generator and color map up for up to 5 classes
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(_y))])

    # plot the decision surface
    x1_min, x1_max = _x[:, 0].min() - 1, _x[:, 0].max() + 1
    x2_min, x2_max = _x[:, 1].min() - 1, _x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    _z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    _z = _z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, _z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot samples
    for idx, _cl in enumerate(np.unique(_y)):
        plt.scatter(x=_x[_y == _cl, 0], y=_x[_y == _cl, 1],
                    alpha=0.8, color=cmap(idx),
                    marker='x', label=_cl)

    # Plot test samples if they exist
    if not test_idx is None:
        X_test, y_test = _x[test_idx, :], _y[test_idx]
        for idx, _cl in enumerate(np.unique(y_test)):
            plt.scatter(x=X_test[y_test == _cl, 0], y=X_test[y_test == _cl, 1],
                        alpha=1, color=cmap(idx),
                        linewidths=1, marker='o', s=55, label=f'test {_cl}')
    plt.legend()
