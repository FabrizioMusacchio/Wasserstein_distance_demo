"""
A script to demonstrate the Wasserstein metric as a measure of dissimilarity.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: July 20, 2023


ACKNOWLEDGEMENT:
The code of DEMO 1 is taken and modified from POT documentation:
https://pythonot.github.io/auto_examples/others/plot_screenkhorn_1D.html#screened-optimal-transport-screenkhorn
written by Author: Mokhtar Z. Alaya <mokhtarzahdi.alaya@gmail.com> (License: MIT License). 
Also the plot1D_mat() function of DEMO 1 is taken from the POT library.
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from ot.bregman import screenkhorn
from matplotlib import gridspec
from scipy.stats import wasserstein_distance
# check, whether there is a folder "images" in the current directory,otherwise create it:
import os
if not os.path.exists('images'):
    os.makedirs('images')
# %% DEMO 1

# generate the distributions:
n = 100  # nb bins
x = np.arange(n, dtype=np.float64) # bin positions
a = gauss(n, m=20, s=5)  # m= mean, s= std
b = gauss(n, m=60, s=10)

# calculate the cost/loss matrix:
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), metric='sqeuclidean')
"""
ot.dist() calculates the cost matrix, which is a matrix of all pairwise distances 
between the points in the source/target distributions. By default, the squared
Eucledian distance is used. When just computing Euclidean distance, the function
becomes equivalent to:

M = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
"""

M /= M.max()

def plot1D_mat(a, b, M, title=''):
    r""" Plot matrix :math:`\mathbf{M}`  with the source and target 1D distribution
    
    Creates a subplot with the source distribution :math:`\mathbf{a}` on the left and
    target distribution :math:`\mathbf{b}` on the top. The matrix :math:`\mathbf{M}` is shown in between.

    Modified function from the POT library.

    Parameters:
    ----------
    a : ndarray, shape (na,)
        Source distribution
    b : ndarray, shape (nb,)
        Target distribution
    M : ndarray, shape (na, nb)
        Matrix to plot
    """    
    na, nb = M.shape

    gs = gridspec.GridSpec(3, 3)

    xa = np.arange(na)
    xb = np.arange(nb)

    ax1 = plt.subplot(gs[0, 1:])
    plt.plot(xb, b, c="#E69F00", label='Target\ndistribution', lw=2)
    #plt.xticks(())
    # remove top axis:
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['bottom'].set_visible(False)
    # hide the xticks:
    #ax1.set_xticks(())
    # set the ylimit to the max of the two distributions:
    plt.ylim((0, max(max(a), max(b))))
    # make axis thicker:
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    plt.legend(fontsize=8)
    #plt.title(title)

    ax2 = plt.subplot(gs[1:, 0])
    plt.plot(a, xa, c="#0072B2",  label='Source\ndistribution', lw=2)
    plt.xlim((0, max(max(a), max(b))))
    # set the same y ticks as the other plot:
    plt.xticks(ax1.get_yticks())
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    #ax2.spines['left'].set_visible(False)
    #plt.xticks(())
    # make axis thicker:
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    plt.legend(fontsize=8)

    plt.subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)
    plt.imshow(M, interpolation='nearest', cmap="plasma")
    # show only bottom and right axis:
    ax = plt.gca()
    #ax.spines['top'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.yaxis.set_ticks_position('right')
    # show y axis on the right
    plt.axis('off')
    plt.text(xa[-1:], 0.5, title, horizontalalignment='right', verticalalignment='top', 
             color='white', fontsize=12, fontweight="bold")
    plt.xlim((0, nb))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0., hspace=0.2)

# plot distributions:
plt.figure(1, figsize=(6.4, 3))
plt.plot(x, a, c="#0072B2", label='Source distribution', lw=3)
plt.plot(x, b, c="#E69F00", label='Target distribution', lw=3)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# make axis thicker:
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
# make xticks thicker:
ax.tick_params(axis='x', which='major', width=2)
ax.tick_params(axis='y', which='major', width=2)
# make fontsize of ticks bold:
ax.tick_params(axis='both', which='major', labelsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('images/wasserstein_distributions.png', dpi=200)
plt.show()

# plot distributions and loss matrix:
plt.figure(2, figsize=(5, 5))
plot1D_mat(a, b, M, 'Cost matrix\nC$_{i,j}$')
plt.savefig('images/wasserstein_cost_matrix.png', dpi=200)
plt.show()

# solve transport plan problem:
G = ot.emd(a, b, M)

# solve Screenkhorn:
#lambd = 2e-03  # entropy parameter
#ns_budget = 30  # budget number of points to be keept in the source distribution
#nt_budget = 30  # budget number of points to be keept in the target distribution
#G = screenkhorn(a, b, M, lambd, ns_budget, nt_budget, uniform=False, restricted=True, verbose=True)

# solve Sinkhorn:
#epsilon = 1e-3
#G = ot.sinkhorn(a, b, M, epsilon, verbose=False)

plt.figure(3, figsize=(5, 5))
plot1D_mat(a, b, G, 'Optimal transport\nmatrix G$_{i,j}$')
plt.savefig('images/wasserstein_optimal_transport.png', dpi=200)
plt.show()

# the wasserstein distance is according W(P, Q) = \sum_i \sum_j (\gamma_{ij} * c_{ij}):
w_dist = np.sum(G * M)
print(f"Wasserstein distance W_1 (manual): {w_dist}")
# %% TEST DIFFERENT COST MATRICES
def plot_cost_and_transport_matrices(M, G, cost_metric=''):
    # plot the two matrices side by side:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(M, cmap='plasma')
    ax[0].axis('off')
    ax[0].text(M.shape[0]-0.5, 0.5, f"Cost matrix M\nbased on {cost_metric}", horizontalalignment='right', verticalalignment='top', 
                color='white', fontsize=16, fontweight="bold")
    ax[1].imshow(G, cmap='plasma')
    ax[1].text(M.shape[0]-0.5, 0.5, "Optimal transport\nmatrix G", horizontalalignment='right', verticalalignment='top', 
                color='white', fontsize=16, fontweight="bold")
    ax[1].axis('off')
    
    # set a global title for the entire figure:
    #plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f'images/wasserstein_{cost_metric}.png', dpi=200)

# squared Euclidean distance:
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), metric='sqeuclidean')
M /= M.max()
G = ot.emd(a, b, M)
plot_cost_and_transport_matrices(M, G, cost_metric='squared\nEuclidean distance')
w_dist = np.sum(G * M)
print(f"Wasserstein distance (squared Euclidean): {w_dist}")

# Euclidean distance:
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), metric='euclidean')
M /= M.max()
G = ot.emd(a, b, M)
plot_cost_and_transport_matrices(M, G, cost_metric='Euclidean distance')
w_dist = np.sum(G * M)
print(f"Wasserstein distance (Euclidean): {w_dist}")

# dice distance:
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), metric='jaccard')
M /= M.max()
G = ot.emd(a, b, M)
plot_cost_and_transport_matrices(M, G, cost_metric='Jaccard distance')
w_dist = np.sum(G * M)
print(f"Wasserstein distance (jaccard): {w_dist}")

# dice distance:
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), metric='canberra')
M /= M.max()
G = ot.emd(a, b, M)
plot_cost_and_transport_matrices(M, G, cost_metric='canberra distance')
w_dist = np.sum(G * M)
print(f"Wasserstein distance (Canberra): {w_dist}")
# %% WASSERSTEIN METRIC AS DISSIMILARITY MEASURE (CONTINUOUS GAUSSIAN DISTRIBUTIONS)
n=1000
x=np.linspace(-10, 10, n)

# define Gaussian function:
def my_gauss(x, m, s):
    return np.exp(-((x - m) ** 2) / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))

# define a function with two gaussian peaks:
def my_gauss_mixt(x, m1, m2, s1, s2):
    return 0.5*my_gauss(x, m1, s1)+0.5*my_gauss(x, m2, s2)

# define distribution plot function:
def plot_distributions(x, a, b, a_label="source distribution", 
                      b_label="target distribution", title="", plot_title="dist"):
    plt.figure(1, figsize=(6.4, 3))
    plt.plot(x, a, c="#0072B2", label=a_label, lw=3)
    plt.plot(x, b, c="#E69F00", label=b_label, lw=3, ls='--')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='x', which='major', width=2)
    ax.tick_params(axis='y', which='major', width=2)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig('images/'+plot_title+'.png', dpi=200)
    plt.show()

# define main executable function:
def calc_and_plot_distributions(x, m1=1, m2=1, s1=1, s2=1):
    """ m1, m2 = 20, 20
    s1, s2 = 5,5 """
    a = my_gauss(x, m=m1, s=s1)
    b = my_gauss(x, m=m2, s=s2)
    a_label = f"source ($\mu$={m1}, $\sigma$={s1})"
    b_label = f"target ($\mu$={m2}, $\sigma$={s2})"
    w_dist = wasserstein_distance(a, b)
    print(f"Wasserstein distance (scipy): {w_dist}")
    print(f"Wasserstein distance W_1 (POT): {ot.wasserstein_1d(a, b, p=1)}")
    plot_distributions(x, a, b, a_label, b_label, title=f"Wasserstein distance: {w_dist}",
                       plot_title=f"dist_m1_{m1}_m2_{m2}_s1_{s1}_s2_{s2}")

def calc_and_plot_distributions2(x, m1=1, m2=1, m3=1, s1=1, s2=1, s3=1):
    a = my_gauss(x, m=m1, s=s1)
    b = my_gauss_mixt(x, m1=m2, m2=m3, s1=s2, s2=s3)
    a_label = f"source ($\mu$={m1}, $\sigma$={s1})"
    b_label = f"target ($\mu_1$={m2}, $\sigma_1$={s2} & $\mu_2$={m3}, $\sigma_2$={s3})"
    w_dist = wasserstein_distance(a, b)
    print(f"Wasserstein distance (scipy): {w_dist}")
    print(f"Wasserstein distance W_1 (POT): {ot.wasserstein_1d(a, b, p=1)}")
    plot_distributions(x, a, b, a_label, b_label, title=f"Wasserstein distance: {w_dist}",
                       plot_title=f"dist_m1_{m1}_m2_{m2}_m3_{m3}_s1_{s1}_s2_{s2}_s3_{s3}")

# increasing mu:
calc_and_plot_distributions(x, m1=0, m2=0, s1=1, s2=1)
calc_and_plot_distributions(x, m1=0, m2=1, s1=1, s2=1)
calc_and_plot_distributions(x, m1=0, m2=2, s1=1, s2=1)
calc_and_plot_distributions(x, m1=0, m2=4, s1=1, s2=1)
calc_and_plot_distributions(x, m1=0, m2=5, s1=1, s2=1)
"""
The two distributions are identical, but shifted apart along the \mu/x-axis.
Here, the Wasserstein distance expresses the "dissimilarity" between the two distributions
with regard to this shift. Since the distributions are still quite similar (identical), 
the Wasserstein distance is very small, i.e., shifting two identical distributions
against each other doesn't change theirs Wasserstein similarity that much.
"""

# increasing sigma:
calc_and_plot_distributions(x, m1=0, m2=0, s1=1, s2=1)
calc_and_plot_distributions(x, m1=0, m2=0, s1=1, s2=2)
calc_and_plot_distributions(x, m1=0, m2=1, s1=1, s2=2)
calc_and_plot_distributions(x, m1=0, m2=2, s1=1, s2=2)
calc_and_plot_distributions(x, m1=0, m2=4, s1=1, s2=2)

calc_and_plot_distributions(x, m1=0, m2=0, s1=1, s2=1)
calc_and_plot_distributions(x, m1=0, m2=0, s1=1, s2=3)
calc_and_plot_distributions(x, m1=0, m2=1, s1=1, s2=3)
calc_and_plot_distributions(x, m1=0, m2=2, s1=1, s2=3)
calc_and_plot_distributions(x, m1=0, m2=4, s1=1, s2=3)
"""
The increase of the standard deviation of the target distribution increases 
the Wasserstein distance, while still the increase of \mu has no noticeable effect
on the distance.
"""


# two at first glance different distributions can have the same wasserstein distance:
calc_and_plot_distributions(x, m1=0, m2=0, s1=1, s2=2)
calc_and_plot_distributions2(x, m1=0, m2=-2, m3=2, s1=1, s2=1, s3=1)
"""
The Wasserstein distance between two normal Gaussian distributions with $\mu_1=\mu2$ and
$\sigma_2=2\sigma_1$ is the same as for one normal Gaussian ($\mu_1, \sigma_2$) and a 
double peaked Gaussian with $\sigma_2+\sigma_3=\sigma_1$. The reason for is that
we have created the two-peaked distribution by adding two normal distributions, while
still keeping the resulting distribution normalized (the area under the curve is 1 for
the chosen set of $\mu_i$ and $\sigma_i$). And this is independent of how far the
two peaks are apart from each other or how far the barycenter of the two peaks is
apart from $\mu_1$.
"""
calc_and_plot_distributions(x, m1=0, m2=0, s1=1, s2=2)
calc_and_plot_distributions2(x, m1=0, m2=-2+2, m3=2+2, s1=1, s2=1, s3=1)
calc_and_plot_distributions2(x, m1=0, m2=-3, m3=4, s1=1, s2=1, s3=1)
# %% 2D DISTRIBUTIONS

# generate some toy data:
n = 50  # nb samples
m1  = np.array([0, 0])
m2  = np.array([4, 4])
s_1 = 1
s_2 = 3
cov1 = np.array([[s_1, 0], [0, s_1]])
cov2 = np.array([[s_2, 0], [0, s_2]])
np.random.seed(0)
xs = ot.datasets.make_2D_samples_gauss(n, m1, cov1)
np.random.seed(0)
xt = ot.datasets.make_2D_samples_gauss(n, m2, cov2)
a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

# loss matrix:
M = np.sum((xs[:, np.newaxis, :] - xt[np.newaxis, :, :]) ** 2, axis=-1)
#M = ot.dist(xs, xt, metric='sqeuclidean')
"""Note, that ot.dist() introduces some rounding errors, which may lead to
miss interpretation of the results."""
M /= M.max()

# transport plan:
G0 = ot.emd(a, b, M)

# Wasserstein distance:
w_dist = np.sum(G0 * M)
print(f"Wasserstein distance: {w_dist}")

fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
# plot the distributions:
plt.subplot(1, 3, 1, aspect='equal')
ot.plot.plot2D_samples_mat(xs, xt, G0, c="lightsteelblue")
"""
Plot lines between source and target 2D samples with a color proportional to 
the value of the matrix G0 between samples.
"""
plt.plot(xs[:, 0], xs[:, 1], '+', label=f'Source (random normal,\n $\mu$={m1}, $\sigma$={s_1})')
plt.plot(xt[:, 0], xt[:, 1], 'x', label=f'Target (random normal,\n $\mu$={m2}, $\sigma$={s_2})')
plt.legend(loc=0, fontsize=8)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title(f'Source and target distributions\nWasserstein distance: {w_dist}')

# plot the loss/cost matrix:
plt.subplot(1, 3, 2)
plt.imshow(M, cmap='plasma')
plt.xlabel("i")
plt.ylabel("j")
plt.title('Cost matrix C$_{i,j}$')

# plot the optimal transport plan:
plt.subplot(1, 3, 3)
plt.imshow(G0, cmap='plasma')
plt.xlabel("i")
plt.ylabel("j")
plt.title('Optimal transport matrix G$_{i,j}$')

plt.tight_layout()
plt.savefig(f'images/wasserstein_2D_m1_{m1[0]}_{m1[1]}_m2_{m2[0]}_{m2[1]}_s1_{cov1[0,0]}_{cov1[0,1]}_s2_{cov2[0,0]}_{cov2[0,1]}.png', dpi=200)
plt.show()


# %% END
