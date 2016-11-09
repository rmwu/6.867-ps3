import pdb
import numpy as np
import pylab as pl

import matplotlib.pyplot as plt

# X is data matrix (each row is a data point)
# Y is desired output (1 or -1)
# scoreFn is a function of a data point
# values is a list of values to plot

def plotDecisionBoundary(X, Y, scoreFn, values, title="", save_path=None):
    # Plot the decision boundary. For that, we will asign a score to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
    zz = np.array([scoreFn(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    pl.figure()
    CS = pl.contour(xx, yy, zz, values, colors = 'cornflowerblue', linestyles = 'solid', linewidths = 1)
    pl.clabel(CS, fontsize=9, inline=1)
    # Plot the training points
    pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = "coolwarm", marker=".", edgecolors="none")
    pl.title(title)

    if save_path:
        pl.savefig(save_path, format='pdf')

    pl.axis('tight')
    
def plotCMPBoundary(X, Y, scoreFn, scoreFnCMP, values, title="", save_path=None):
    # Plot the decision boundary. For that, we will asign a score to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
    zz = np.array([scoreFn(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    
    zz2 = np.array([scoreFnCMP(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    zz2 = zz2.reshape(xx.shape)
    
    pl.figure()
    CS = pl.contour(xx, yy, zz, values, colors = 'red', linestyles = 'solid', linewidths = 1)
    pl.clabel(CS, fontsize=9, inline=1, fmt="L2 %1.3f")
    
    CS2 = pl.contour(xx, yy, zz2, values, colors = 'forestgreen', linestyles = 'solid', linewidths = 1)
    pl.clabel(CS2, fontsize=9, inline=1, fmt="L1 %1.3f")
    # Plot the training points
    pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = "coolwarm", marker=".", edgecolors="none")
    pl.title(title)

    if save_path:
        pl.savefig(save_path, format='pdf')

    pl.axis('tight')

def plot_weights(objs, filename):
    colors = ["blue","pink"]
    for delta_obj, color in list(zip(objs, colors)):
        weight_norms = [np.linalg.norm(vals[0]) for vals in delta_obj]
        iterations = len(weight_norms)

        plt.plot(np.arange(iterations), weight_norms, color=color)
    
    plt.title("||w|| over time")
    plt.xlabel("Number of iterations")
    
    y_axis_label = "||w||"
    plt.ylabel(y_axis_label)
    
    plt.savefig(filename, format='pdf')
    
    plt.show()