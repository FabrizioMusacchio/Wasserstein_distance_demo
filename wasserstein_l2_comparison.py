"""
A script for comparing the Wasserstein distance between two 2D 
distributions with the L2 norm distance.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: July 25, 2023
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import norm
import ot
import os
if not os.path.exists('images'):
    os.makedirs('images')
from scipy.stats import multivariate_normal
import ot.plot
from scipy.stats import wasserstein_distance
# %% DISCRETE SAMPLES
# generate two 2D gaussian samples sets:
n = 50  # nb samples
m1  = np.array([0, 0])
m2  = np.array([1, 1])
s_1 = 1
s_2 = 1
cov1 = np.array([[s_1, 0], [0, s_1]])
cov2 = np.array([[s_2, 0], [0, s_2]])
np.random.seed(0)
xs = ot.datasets.make_2D_samples_gauss(n, m1, cov1)
np.random.seed(0)
xt = ot.datasets.make_2D_samples_gauss(n, m2, cov2)

# plot the distributions:
fig = plt.figure(figsize=(5, 5))
plt.plot(xs[:, 0], xs[:, 1], '+', label=f'Source (random normal,\n $\mu$={m1}, $\sigma$={s_1})')
plt.plot(xt[:, 0], xt[:, 1], 'x', label=f'Target (random normal,\n $\mu$={m2}, $\sigma$={s_2})')
plt.legend(loc=0, fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title(f'Source and target distributions')
plt.tight_layout()
plt.savefig('images/wasserstein_l2_comparison.png', dpi=200)
plt.show()

# loss matrix:
M = np.sum((xs[:, np.newaxis, :] - xt[np.newaxis, :, :]) ** 2, axis=-1)
M /= M.max()

# transport plan:
G0 = ot.emd(a, b, M)

# Wasserstein distance:
w_dist = np.sum(G0 * M)
print(f"Wasserstein distance (POT): {w_dist}")

# sliced Wasserstein distance:
n_projections=1000
a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
w_dist_sliced = ot.sliced_wasserstein_distance(xs, xt, a, b, n_projections, seed=0)
print(f"Sliced Wasserstein distance (POT): {w_dist_sliced}")

"""
`a` and `b` represent discrete probability distributions of the source and target, respectively. 
Both are required when using the POT library to calculate the Wasserstein distance or sliced 
Wasserstein distance. `a` and `b` are in our case uniform distributions, meaning each sample 
point in the source and target distributions are equally likely. The statement `np.ones((n,)) / n` 
creates an array of size `n` (the number of samples) where each entry is `1/n`. Since the sum of 
all probabilities in a probability distribution must equal 1, this represents a uniform distribution 
over `n` points. We use  uniform distributions because we are dealing with sets of samples where 
there is no reason to believe any sample is more likely than any other, i.e., we assume no additional
information about the distribution of the samples.  In any other case, where the sample points are 
not equally likely, `a` and `b` would be different and not uniform. These cases might occur, for 
example, when dealing with weighted samples from some underlying distribution.
"""

# calculate the L2 distance:
L2_dist = np.sqrt(norm(xs - xt))
print(f"L2 distance: {L2_dist}")

# plot the distributions again:
fig = plt.figure(figsize=(5, 5))
ot.plot.plot2D_samples_mat(xs, xt, G0, c="lightsteelblue")
plt.plot(xs[:, 0], xs[:, 1], '+', label=f'Source (random normal,\n $\mu$={m1}, $\sigma$={s_1})')
plt.plot(xt[:, 0], xt[:, 1], 'x', label=f'Target (random normal,\n $\mu$={m2}, $\sigma$={s_2})')
plt.legend(loc=0, fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title(f'Source and target distributions\nWasserstein distance: {w_dist}'+
          f'\nSliced Wasserstein distance: {w_dist_sliced}'+
          f'\nL2 distance: {L2_dist}')
plt.tight_layout()
plt.savefig('images/wasserstein_l2_comparison_w_info.png', dpi=200)
plt.show()
# %% DISCRETE SAMPLES (RUNNING m2)
# generate two 2D gaussian samples sets:
n = 50  # nb samples
m1  = np.array([0, 0])
s_1 = 1
cov1 = np.array([[s_1, 0], [0, s_1]])
np.random.seed(0)
xs = ot.datasets.make_2D_samples_gauss(n, m1, cov1)
a = np.ones((n,)) / n  # uniform distribution on samples

wasserstein_distances = []
wasserstein_distances_sliced = []
l2_distances = []
n_max=50
m2_values = np.linspace(0, n_max, 100)

for m2 in m2_values:
    m2 = np.array([m2, m2])
    s_2 = 1
    cov2 = np.array([[s_2, 0], [0, s_2]])
    xt = ot.datasets.make_2D_samples_gauss(n, m2, cov2)
    b = np.ones((n,)) / n  # uniform distribution on samples

    # loss matrix:
    M = np.sum((xs[:, np.newaxis, :] - xt[np.newaxis, :, :]) ** 2, axis=-1)
    M /= M.max()

    # transport plan:
    G0 = ot.emd(a, b, M)

    # Wasserstein distance:
    w_dist = np.sum(G0 * M)
    wasserstein_distances.append(w_dist)
    
    # sliced Wasserstein distance:
    n_projections=1000
    w_dist_sliced = ot.sliced_wasserstein_distance(xs, xt, a, a, n_projections, seed=0)
    wasserstein_distances_sliced.append(w_dist_sliced)
    
    # calculate the L2 distance:
    L2_dist = np.sqrt(norm(xs - xt))
    l2_distances.append(L2_dist)

# plot the distributions:
plt.figure(figsize=(6, 6), dpi=200)
ot.plot.plot2D_samples_mat(xs, xt, G0, c="lightsteelblue")
plt.plot(xs[:, 0], xs[:, 1], '+', label=f'Source (random normal,\n $\mu$={m1}, $\sigma$={s_1})')
plt.plot(xt[:, 0], xt[:, 1], 'x', label=f'Target (random normal,\n $\mu$={m2}, $\sigma$={s_2})')
plt.legend(loc=0, fontsize=8)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.xlim(-10, n_max+10)
plt.ylim(-10, n_max+10)
plt.title(f'Source and target distributions\nWasserstein distance: {w_dist}')

# plot the distances:
plt.figure(figsize=(10, 7))
plt.plot(m2_values, wasserstein_distances, label='Wasserstein distance')
plt.plot(m2_values, wasserstein_distances_sliced, label='sliced Wasserstein distance')
plt.plot(m2_values, l2_distances, label='L2 distance')
plt.legend()
plt.xlabel('m2')
plt.ylabel('Distance')
plt.title('Evolution of Wasserstein and L2 distances with increasing m2')
plt.grid(True)
plt.show()
# %% DISCRETE SAMPLES (RUNNING m2, ANIMATION)
# generate two 2D gaussian samples sets:
n = 50  # nb samples
m1  = np.array([0, 0])
s_1 = 1
cov1 = np.array([[s_1, 0], [0, s_1]])
np.random.seed(0)
xs = ot.datasets.make_2D_samples_gauss(n, m1, cov1)
a = np.ones((n,)) / n  # uniform distribution on samples

wasserstein_distances = []
wasserstein_distances_sliced = []
l2_distances = []
samples = []
n_max=50
m2_values = np.linspace(0, n_max, 100)

for m2 in m2_values:
    m2 = np.array([m2, m2])
    s_2 = 1
    cov2 = np.array([[s_2, 0], [0, s_2]])
    np.random.seed(0)
    xt = ot.datasets.make_2D_samples_gauss(n, m2, cov2)
    b = np.ones((n,)) / n  # uniform distribution on samples
    samples.append(xt)

    # loss matrix:
    M = np.sum((xs[:, np.newaxis, :] - xt[np.newaxis, :, :]) ** 2, axis=-1)
    M /= M.max()

    # transport plan:
    G0 = ot.emd(a, b, M)

    # Wasserstein distance:
    w_dist = np.sum(G0 * M)
    wasserstein_distances.append(w_dist)
    
    # sliced Wasserstein distance:
    n_projections=1000
    w_dist_sliced = ot.sliced_wasserstein_distance(xs, xt, a, a, n_projections, seed=0)
    wasserstein_distances_sliced.append(w_dist_sliced)
    
    # calculate the L2 distance:
    L2_dist = np.sqrt(norm(xs - xt))
    l2_distances.append(L2_dist)

# set up figure and axis_
fig, axs = plt.subplots(1, 2, figsize=(12, 6.7))

# update function for animation
def update(num):
    axs[0].clear()
    axs[1].clear()
    m2 = m2_values[num]
    m2 = np.array([m2, m2])

    xt = samples[num]
    
    w_dist = wasserstein_distances[num]
    w_dist_sliced = wasserstein_distances_sliced[num]
    L2_dist = l2_distances[num]

    # plot the distributions:
    axs[0].plot(xs[:, 0], xs[:, 1], '+', label=f'Source (random normal,\n $\mu_1$={m1}, $\sigma_1$={s_1})')
    axs[0].plot(xt[:, 0], xt[:, 1], 'x', label=f'Target (random normal,\n $\mu_2$={m2.round(2)}, $\sigma_2$={s_2})')
    axs[0].legend(loc="upper left", fontsize=10)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_xlim(-10, n_max+10)
    axs[0].set_ylim(-10, n_max+10)
    axs[0].set_title(f'Source and target distributions\nWasserstein distance: {w_dist}'+
                     f'\nSliced Wasserstein distance: {w_dist_sliced}'+
                     f'\nL2 distance: {L2_dist}')

    # plot the distances:
    axs[1].plot(m2_values[:num+1], wasserstein_distances[:num+1], label='Wasserstein distance')
    axs[1].plot(m2_values[:num+1], wasserstein_distances_sliced[:num+1], label='sliced Wasserstein distance')
    axs[1].plot(m2_values[:num+1], l2_distances[:num+1], label='L2 distance')
    axs[1].legend()
    axs[1].set_xlabel('$\mu_2$ (equal in both dimensions)')
    axs[1].set_ylabel('Distance')
    axs[1].set_title('Evolution of Wasserstein and L2 distances with increasing $\mu_2$')
    axs[1].grid(True)
    
    plt.tight_layout()

ani = animation.FuncAnimation(fig, update, frames=len(m2_values), repeat=False)
ani.save('images/wasserstein_l2_animation_m2.gif', writer='imagemagick', fps=10)
plt.close(fig)
# %% DISCRETE SAMPLES (RUNNING s2, ANIMATION)
# generate two 2D gaussian samples sets:
n = 50  # nb samples
m1  = np.array([0, 0])
s_1 = 1
cov1 = np.array([[s_1, 0], [0, s_1]])
np.random.seed(0)
xs = ot.datasets.make_2D_samples_gauss(n, m1, cov1)
a = np.ones((n,)) / n  # uniform distribution on samples

wasserstein_distances = []
wasserstein_distances_sliced = []
l2_distances = []
samples = []
n_max=50
s2_values = np.linspace(1, n_max, 100)

m2 = np.array([0, 0]) # Fix m2

for s_2 in s2_values:
    cov2 = np.array([[s_2, 0], [0, s_2]])
    np.random.seed(0)
    xt = ot.datasets.make_2D_samples_gauss(n, m2, cov2)
    b = np.ones((n,)) / n  # uniform distribution on samples
    samples.append(xt)

    # loss matrix:
    M = np.sum((xs[:, np.newaxis, :] - xt[np.newaxis, :, :]) ** 2, axis=-1)
    M /= M.max()

    # transport plan:
    G0 = ot.emd(a, b, M)

    # Wasserstein distance:
    w_dist = np.sum(G0 * M)
    wasserstein_distances.append(w_dist)
    
    # sliced Wasserstein distance:
    n_projections=1000
    w_dist_sliced = ot.sliced_wasserstein_distance(xs, xt, a, a, n_projections, seed=0)
    wasserstein_distances_sliced.append(w_dist_sliced)
    
    # calculate the L2 distance:
    L2_dist = np.sqrt(norm(xs - xt))
    l2_distances.append(L2_dist)

# set up figure and axis:
fig, axs = plt.subplots(1, 2, figsize=(12, 6.7))

# update function for animation
def update(num):
    axs[0].clear()
    axs[1].clear()
    s_2 = s2_values[num]

    xt = samples[num]
    
    w_dist = wasserstein_distances[num]
    w_dist_sliced = wasserstein_distances_sliced[num]
    L2_dist = l2_distances[num]

    # plot the distributions:
    axs[0].plot(xs[:, 0], xs[:, 1], '+', label=f'Source (random normal,\n $\mu_1$={m1}, $\sigma_1$={s_1})')
    axs[0].plot(xt[:, 0], xt[:, 1], 'x', label=f'Target (random normal,\n $\mu_2$={m2.round(2)}, $\sigma_2$={s_2.round(2)})')
    axs[0].legend(loc="upper left", fontsize=10)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_xlim(-25, 25)
    axs[0].set_ylim(-25, 25)
    axs[0].set_title(f'Source and target distributions\nWasserstein distance: {w_dist}'+
                     f'\nSliced Wasserstein distance: {w_dist_sliced}'+
                     f'\nL2 distance: {L2_dist}')

    # plot the distances:
    axs[1].plot(s2_values[:num+1], wasserstein_distances[:num+1], label='Wasserstein distance')
    axs[1].plot(s2_values[:num+1], wasserstein_distances_sliced[:num+1], label='sliced Wasserstein distance')
    axs[1].plot(s2_values[:num+1], l2_distances[:num+1], label='L2 distance')
    axs[1].legend()
    axs[1].set_xlabel('$\sigma_2$ (equal in both dimensions)')
    axs[1].set_ylabel('Distance')
    axs[1].set_title('Evolution of Wasserstein and L2 distances with increasing $\sigma_2$')
    axs[1].grid(True)
    
    plt.tight_layout()

ani = animation.FuncAnimation(fig, update, frames=len(s2_values), repeat=False)
ani.save('images/wasserstein_l2_animation_s2.gif', writer='imagemagick', fps=10)
plt.close(fig)
# %% # %% 2D GAUSSIAN DISTRIBUTIONS
"""
This here is some unfinished code for calculating the metrics between differently
generated 2D Gaussian distributions. It is not used in the final animation.
"""
# define the two distributions:
mean1 = np.array([0, 0])
cov1 = np.array([[1, 0], [0, 1]])
gauss1 = multivariate_normal(mean=mean1, cov=cov1)

mean2 = np.array([1, 1])
cov2 = np.array([[1, 0], [0, 1]])
gauss2 = multivariate_normal(mean=mean2, cov=cov2)

# Create the 2D grid
x = np.linspace(-3, 4, 100)
y = np.linspace(-3, 4, 100)
X, Y = np.meshgrid(x, y)

# compute the PDFs at each grid point:
pdf1 = gauss1.pdf(np.dstack((X, Y)))
pdf2 = gauss2.pdf(np.dstack((X, Y)))

# flatten and normalize the PDFs for Wasserstein distance calculation:
pdf1_flat = pdf1.flatten()
pdf2_flat = pdf2.flatten()
pdf1_flat /= pdf1_flat.sum()
pdf2_flat /= pdf2_flat.sum()

""" # Compute the cost matrix for Wasserstein distance calculation
M = ot.dist(np.column_stack((X.flatten(), Y.flatten())))
M /= M.max()
# Calculate the Wasserstein distance using the POT library
wasserstein_dist = ot.emd2(pdf1_flat, pdf2_flat, M)
print(f"Wasserstein distance: {wasserstein_dist}") """

# calculate the L2 distance (Euclidean norm) between the two PDFs:
l2_dist = norm(pdf1_flat - pdf2_flat)
"""equivalent to np.sqrt(np.sum((pdf1_flat-pdf2_flat)**2))"""
print(f"L2 distance: {l2_dist}")

# calculate sliced Wasserstein distance:
def sliced_wasserstein_distance(gauss1, gauss2, num_projections):
    total_wasserstein = 0
    for _ in range(num_projections):
        # draw a random 1D projection:
        theta = np.random.normal(0, 1, 2)
        theta /= np.linalg.norm(theta)

        # project the data onto the random 1D slice:
        projection_gauss1 = np.dot(gauss1, theta)
        projection_gauss2 = np.dot(gauss2, theta)

        # Compute the 1D Wasserstein distance
        total_wasserstein += wasserstein_distance(projection_gauss1, projection_gauss2)
    return total_wasserstein / num_projections
# draw random samples from each distribution:
samples1 = gauss1.rvs(5000)
samples2 = gauss2.rvs(5000)
wasserstein_dist_slice = sliced_wasserstein_distance(samples1, samples2, 100)
print(f"Sliced Wasserstein distance: {wasserstein_dist_slice}")


# set up the figure and axis for plotting:
xy_lim = 10
fig, ax = plt.subplots(figsize=(5, 5))
x_gauss1, y_gauss1 = np.mgrid[-xy_lim:xy_lim:.01, -xy_lim:xy_lim:.01]
pos_gauss1 = np.dstack((x_gauss1, y_gauss1))
ax.contourf(x_gauss1, y_gauss1, gauss1.pdf(pos_gauss1), cmap='Blues', alpha=1.00)
pos_gaus2 = np.dstack((x_gauss1, y_gauss1))
ax.contourf(x_gauss1, y_gauss1, gauss2.pdf(pos_gaus2), cmap='Reds', alpha=0.5)
ax.set_title('Two Distributions')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(-xy_lim, xy_lim)
ax.set_ylim(-xy_lim, xy_lim)
plt.show()
# %% 2D GAUSSIAN DISTRIBUTIONS (RUNNING m2)
"""
This here is some unfinished code for calculating the metrics between differently
generated 2D Gaussian distributions. It is not used in the final animation.
"""
""" def sliced_wasserstein_distance(gauss1, gauss2, num_projections):
    total_wasserstein = 0
    for _ in range(num_projections):
        # Draw a random 1D projection
        theta = np.random.normal(0, 1, 2)
        theta /= np.linalg.norm(theta)

        # Project the data onto the random 1D slice
        projection_gauss1 = np.dot(gauss1, theta)
        projection_gauss2 = np.dot(gauss2, theta)

        # Compute the 1D Wasserstein distance
        total_wasserstein += wasserstein_distance(projection_gauss1, projection_gauss2)
    return total_wasserstein / num_projections
 """
# Define the first Gaussian distribution
mean1 = np.array([0, 0])
cov1 = np.array([[1, 0], [0, 1]])
gauss1 = multivariate_normal(mean=mean1, cov=cov1)
samples1 = gauss1.rvs(5000)

# Define the range of means for the second Gaussian distribution
mean2_range = np.linspace(0, 10, 100)

# Initialize lists to hold the distances
l2_distances = []
sliced_wasserstein_distances = []

for mean2_val in mean2_range:
    # define the second Gaussian distribution with the current mean:
    mean2 = np.array([mean2_val, mean2_val])
    cov2 = np.array([[1, 0], [0, 1]])
    gauss2 = multivariate_normal(mean=mean2, cov=cov2)
    samples2 = gauss2.rvs(5000)

    # compute the L2 distance:
    l2_dist = np.linalg.norm(samples1 - samples2)
    l2_distances.append(l2_dist)

    # compute the sliced Wasserstein distance:
    wasserstein_dist_slice = sliced_wasserstein_distance(samples1, samples2, 100)
    sliced_wasserstein_distances.append(wasserstein_dist_slice)

# plot the distances:
fig, ax = plt.subplots(2,1, figsize=(6, 3))
ax[0].plot(mean2_range, l2_distances, label='L2 Distance')
ax[1].plot(mean2_range, sliced_wasserstein_distances, label='Sliced Wasserstein Distance')
ax[0].set_title('Distance vs Mean of Second Gaussian Distribution')
ax[0].set_xlabel('Mean of Second Gaussian Distribution')
ax[0].set_ylabel('Distance')
plt.show()
# %% END
