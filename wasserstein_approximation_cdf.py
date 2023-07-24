"""
A script to approximate the Wasserstein distance between two 1D distributions
using the cumulative distribution functions (CDFs) and the total "work" required.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: July 20, 2023

"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cumfreq
from scipy.stats import wasserstein_distance
from scipy.stats import norm
from scipy.interpolate import interp1d
import ot
# %% MAIN
# generate two 1D gaussian samples:
n=1000
x=np.linspace(-10, 10, n)
m1 = 0
m2 = 1
s1 = 1
s2 = 1
np.random.seed(2)
dist1 = norm.rvs(loc=m1, scale=s1, size=n)
np.random.seed(2)
dist2 = norm.rvs(loc=m2, scale=s2, size=n)
# the function norm.rvs() generates random samples from a normal distribution

""" # Generate example distributions (1D Gaussians):
# define Gaussian function:
def gauss(x, m, s):
    return np.exp(-((x - m) ** 2) / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))
# define a function with two gaussian peaks:
def gauss_mixt(x, m1, m2, s1, s2):
    return 0.5*gauss(x, m1, s1)+0.5*gauss(x, m2, s2)
n=1000
x=np.linspace(-10, 10, n)
dist1 = gauss(x, m=0, s=2)
#dist2 = gauss(x, m=1, s=2)
dist2 = gauss_mixt(x, m1=-5, s1=3, m2=5, s2=2) """

# plot the distributions:
plt.figure(figsize=(7, 3))
plt.plot(x, dist1, label=f"source ($\mu$={m1}, $\sigma$={s1})", alpha=1.00)
plt.plot(x, dist2, label=f"target ($\mu$={m2}, $\sigma$={s2})", alpha=0.55)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.savefig('images/wasserstein_approximation_distributions.png', dpi=200)
plt.show()

# compute the CDFs:
a = cumfreq(dist1, numbins=100)
b = cumfreq(dist2, numbins=100)

# compute the x-values for the CDFs:
x_a = a.lowerlimit + np.linspace(0, a.binsize*a.cumcount.size, a.cumcount.size)
x_b = b.lowerlimit + np.linspace(0, b.binsize*b.cumcount.size, b.cumcount.size)

# interpolate the CDFs to the same x-values:
f_a = interp1d(x_a, a.cumcount / a.cumcount[-1])
f_b = interp1d(x_b, b.cumcount / b.cumcount[-1])
x_common = np.linspace(max(x_a[0], x_b[0]), min(x_a[-1], x_b[-1]), 1000)
cdf_a_common = f_a(x_common)
cdf_b_common = f_b(x_common)

# calculate the PDF of the first distribution:
pdf_a = np.diff(cdf_a_common)
pdf_b = np.diff(cdf_b_common)

# plot the PDFs:
plt.figure(figsize=(7, 3))
plt.plot(pdf_a, label='source PDF')
plt.plot(pdf_b, label='target PDF')
plt.ylabel('probability density')
plt.legend()
plt.tight_layout()
plt.savefig('images/wasserstein_approximation_pdf.png', dpi=200)
plt.show()

# plot the CDFs:
plt.figure(figsize=(5.5, 5))
plt.plot(x_common, cdf_a_common, label='source CDF')
plt.plot(x_common, cdf_b_common, label='target CDF')
# plot the absolute difference between the CDFs:
plt.fill_between(x_common, cdf_a_common, cdf_b_common, color='gray', alpha=0.5, label='absolute difference')
plt.ylabel('cumulative frequency')
plt.legend()
plt.tight_layout()
plt.savefig('images/wasserstein_approximation_cdf.png', dpi=200)
plt.show()

# compute the absolute difference between the CDFs at each point:
diff = np.abs(cdf_a_common - cdf_b_common)

# compute the distance between the points:
dx = np.diff(x_common)

# compute the total "work" as the sum of the absolute differences, weighted by the distance between the points
# Note: We need to exclude the last point because np.diff returns an array that is one element shorter
total_work = np.sum(diff[:-1] * dx)
print(f"Total work of the transport: {total_work}")

# compute the Wasserstein distance using library functions:
print(f"Wasserstein distance (scipy): {wasserstein_distance(dist1, dist2)}")
print(f"Wasserstein distance W_1 (POT): {ot.wasserstein_1d(dist1, dist2, p=1)}")
print(f"Wasserstein distance W_2 (POT): {ot.wasserstein_1d(dist1, dist2, p=2)}")


# plot the distributions, PDF and CDFs plots from above in a single figure with subplots:
fig, axs = plt.subplots(1,3, figsize=(10, 3))
# plot the distributions:
axs[0].plot(x, dist1, label=f"source ($\mu$={m1}, $\sigma$={s1})", alpha=1.00)
axs[0].plot(x, dist2, label=f"target ($\mu$={m2}, $\sigma$={s2})", alpha=0.55)
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()
axs[0].set_title('Distributions')

# plot the PDFs:
axs[1].plot(pdf_a, label='source PDF')
axs[1].plot(pdf_b, label='target PDF')
axs[1].set_ylabel('probability density')
axs[1].legend()
axs[1].set_title('PDFs')

# plot the CDFs:
axs[2].plot(x_common, cdf_a_common, label='source CDF')
axs[2].plot(x_common, cdf_b_common, label='target CDF')
axs[2].fill_between(x_common, cdf_a_common, cdf_b_common, color='gray', alpha=0.5, label='absolute difference')
axs[2].set_ylabel('cumulative frequency')
axs[2].legend()
W1approx_str = "W_{1, approx}"
W1scipy_str = "W_{1, scipy}"
axs[2].set_title(f'CDFs\n${W1approx_str}$={total_work:.2f}, ${W1scipy_str}$={wasserstein_distance(dist1, dist2):.2f}')

plt.tight_layout()
plt.savefig(f'images/wasserstein_approximation_subplots_m1_{m1}_m2_{m2}_s1_{s1}_s2_{s2}.png', dpi=200)
plt.show()


# %% END