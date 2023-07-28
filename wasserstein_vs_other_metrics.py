# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
if not os.path.exists('images'):
    os.makedirs('images')
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
# %% TWO DISCRETE DISTRIBUTIONS
# generate two uniform, random sample sets, with adjustable mean and std:
mean1, std1 = 0, 1  # Distribution 1 parameters (mean and standard deviation)
mean2, std2 = 1, 1  # Distribution 2 parameters (mean and standard deviation)
n = 1000
np.random.seed(0)  # Set random seed for reproducibility
sample1 = np.random.normal(mean1, std1, n)
#sample1 = norm.rvs(loc=mean1, scale=std1, size=n)
np.random.seed(10)  # Set random seed for reproducibility
sample2 = np.random.normal(mean2, std2, n)
#sample2 = norm.rvs(loc=mean2, scale=std2, size=n)

# plot the two samples:
plt.figure(figsize=(6,3))
plt.plot(sample1, label='Distribution 1', lw=1.5)
plt.plot(sample2, label='Distribution 2', linestyle="--", lw=1.5)
plt.title('Samples')
plt.legend()
plt.tight_layout()
plt.savefig('images/wasserstein_vs_other_metrics_samples.png', dpi=200)
plt.show()

# calculate KDE for the samples:
x = np.linspace(-5, 7, 1000)  # X values for KDE
kde_sample1 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sample1[:, None])
pdf_sample1 = np.exp(kde_sample1.score_samples(x[:, None]))

kde_sample2 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sample2[:, None])
pdf_sample2 = np.exp(kde_sample2.score_samples(x[:, None]))

# we normalize the distributions to make sure they sum to 1:
pdf_sample1 /= np.sum(pdf_sample1)
pdf_sample2 /= np.sum(pdf_sample2)

#print(np.sum(pdf_sample1), np.sum(pdf_sample2))

# plot the distributions:
plt.figure(figsize=(6,3))
plt.plot(pdf_sample1, label='Distribution 1', lw=2.5)
plt.plot(pdf_sample2, label='Distribution 2', linestyle="--", lw=2.5)
plt.title('Probability Distributions')
plt.legend()
plt.tight_layout()
plt.savefig('images/wasserstein_vs_other_metrics_distributions.png', dpi=200)
plt.show()

# calculate the Wasserstein distance:
wasserstein_dist = wasserstein_distance(x, x, pdf_sample1, pdf_sample2)
print(f"Wasserstein Distance: {wasserstein_dist}")

# calculate the KL divergence between the two distributions:
epsilon = 1e-12
kl_divergence = entropy(pdf_sample1+epsilon, pdf_sample2+epsilon)
print(f"KL Divergence: {kl_divergence}")

# calculate the average distribution M:
pdf_avg = 0.5 * (pdf_sample1 + pdf_sample2)

# calculate the Jensen-Shannon divergence:
kl_divergence_p_m = entropy(pdf_sample1, pdf_avg)
kl_divergence_q_m = entropy(pdf_sample2, pdf_avg)
js_divergence = 0.5 * (kl_divergence_p_m + kl_divergence_q_m)
print(f"Jensen-Shannon Divergence: {js_divergence}")

# calculate the Total Variation distance:
tv_distance = 0.5 * np.sum(np.abs(pdf_sample1 - pdf_sample2))
print(f"Total Variation Distance: {tv_distance}")

# calculate the Bhattacharyya distance:
bhattacharyya_coefficient = np.sum(np.sqrt(pdf_sample1 * pdf_sample2))
bhattacharyya_distance = -np.log(bhattacharyya_coefficient)
print(f"Bhattacharyya Distance: {bhattacharyya_distance}")
# %% TWO DISCRETE DISTRIBUTIONS ANIMATION (mean2 variation)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import entropy, wasserstein_distance
from sklearn.neighbors import KernelDensity
from scipy.spatial import distance

# Generate two uniform, random sample sets, with adjustable mean and std
mean1, std1 = 0, 1  # Distribution 1 parameters (mean and standard deviation)
mean2_values = np.linspace(0, 8, 60)  # Vary mean2 from 0 to 4
std2 = 1  # Distribution 2 parameter (standard deviation)
x = np.linspace(-5, 12, 100)  # X values for KDE

# Arrays to store distances for different mean2 values
wasserstein_dist_arr = []
kl_divergence_arr = []
js_divergence_arr = []
tv_distance_arr = []
bhattacharyya_distance_arr = []
bhattacharyya_coefficient_arr = []
pdf_sample2_arr = []

# Loop over different mean2 values and calculate distances
for mean2 in mean2_values:
    np.random.seed(0)  # Set random seed for reproducibility
    sample1 = np.random.normal(mean1, std1, 1000)
    np.random.seed(10)  # Set random seed for reproducibility
    sample2 = np.random.normal(mean2, std2, 1000)

    # Calculate KDE for the samples
    kde_sample1 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sample1[:, None])
    pdf_sample1 = np.exp(kde_sample1.score_samples(x[:, None]))

    kde_sample2 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sample2[:, None])
    pdf_sample2 = np.exp(kde_sample2.score_samples(x[:, None]))
    
    pdf_sample1 /= np.sum(pdf_sample1)
    pdf_sample2 /= np.sum(pdf_sample2)
    
    pdf_sample2_arr.append(pdf_sample2)

    # Calculate the distances
    wasserstein_dist = wasserstein_distance(x, x, pdf_sample1, pdf_sample2)
    epsilon = 1e-12
    kl_divergence = entropy(pdf_sample1+epsilon, pdf_sample2+epsilon)
    pdf_avg = 0.5 * (pdf_sample1 + pdf_sample2)
    kl_divergence_p_m = entropy(pdf_sample1, pdf_avg)
    kl_divergence_q_m = entropy(pdf_sample2, pdf_avg)
    js_divergence = 0.5 * (kl_divergence_p_m + kl_divergence_q_m)
    tv_distance = 0.5 * np.sum(np.abs(pdf_sample1 - pdf_sample2))
    bhattacharyya_coefficient = np.sum(np.sqrt(pdf_sample1 * pdf_sample2))
    bhattacharyya_distance = -np.log(bhattacharyya_coefficient)
    
    # Append distances to the arrays
    wasserstein_dist_arr.append(wasserstein_dist)
    kl_divergence_arr.append(kl_divergence)
    js_divergence_arr.append(js_divergence)
    tv_distance_arr.append(tv_distance)
    bhattacharyya_distance_arr.append(bhattacharyya_distance)
    bhattacharyya_coefficient_arr.append(bhattacharyya_coefficient)

# Create a function for the animation
def update(frame):
    lw=2.5
    lw_axes=1.5
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(x, pdf_sample1, label='Distribution 1', lw=lw)
    plt.plot(x, pdf_sample2_arr[frame], label='Distribution 2', lw=lw)
    plt.xlabel('X', fontweight="normal", fontsize=14)
    plt.ylabel('Probability Density', fontweight="normal", fontsize=14)
    plt.title(f"Probability Distributions (mean2 = {mean2_values[frame]:.2f})", 
              fontsize=14, fontweight="normal")
    ax = plt.gca()
    ax.spines['top'].set_linewidth(lw_axes)
    ax.spines['right'].set_linewidth(lw_axes)
    ax.spines['bottom'].set_linewidth(lw_axes)
    ax.spines['left'].set_linewidth(lw_axes)
    ax.tick_params(axis='x', which='major', labelsize=14, width=lw_axes)
    ax.tick_params(axis='y', which='major', labelsize=14, width=lw_axes)
    plt.legend(loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(mean2_values[:frame+1], wasserstein_dist_arr[:frame+1], label='Wasserstein', lw=lw)
    plt.plot(mean2_values[:frame+1], kl_divergence_arr[:frame+1], label='KL Divergence', lw=lw)
    plt.plot(mean2_values[:frame+1], js_divergence_arr[:frame+1], label='JS Divergence', lw=lw)
    plt.plot(mean2_values[:frame+1], tv_distance_arr[:frame+1], label='Total Variation', lw=lw)
    plt.plot(mean2_values[:frame+1], bhattacharyya_distance_arr[:frame+1], label='Bhattacharyya', 
             c="violet", lw=lw)
    plt.plot(mean2_values[:frame+1], bhattacharyya_coefficient_arr[:frame+1], 
             label='Bhattacharyya coefficient', c="violet", linestyle="--", lw=lw)
    plt.xlabel('Mean2', fontweight="normal", fontsize=14)
    plt.ylabel('Distance', fontweight="normal", fontsize=14)
    plt.title('Distance Evolution', fontweight="normal", fontsize=14)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(lw_axes)
    ax.spines['right'].set_linewidth(lw_axes)
    ax.spines['bottom'].set_linewidth(lw_axes)
    ax.spines['left'].set_linewidth(lw_axes)
    ax.tick_params(axis='x', which='major', labelsize=14, width=lw_axes)
    ax.tick_params(axis='y', which='major', labelsize=14, width=lw_axes)
    plt.legend(loc="upper left")
    
    plt.tight_layout()

# Create the animation
fig = plt.figure(figsize=(12, 5.5))
#pdf_sample2_values = [(1 / (std2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean2) / std2) ** 2) for mean2 in mean2_values]
ani = FuncAnimation(fig, update, frames=len(mean2_values), interval=200)

# Save the animation as a gif
ani.save('images/distance_evolution_discrete_mean2.gif', writer='imagemagick')
plt.savefig('images/distance_evolution_discrete_mean2.png')
# Show the animation (this will not work on all environments)
plt.show()
# %% TWO DISCRETE DISTRIBUTIONS ANIMATION (std2 variation)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import entropy, wasserstein_distance
from sklearn.neighbors import KernelDensity

# Generate two uniform, random sample sets, with fixed mean2 and varying std2
mean2 = 0  # Distribution 2 parameter (mean)
std2_values = np.linspace(0.1, 10, 100)  # Vary std2 from 0.1 to 2
std1 = 1  # Distribution 1 parameter (standard deviation)
x = np.linspace(-7, 7, 100)  # X values for KDE

# Arrays to store distances for different std2 values
wasserstein_dist_arr = []
kl_divergence_arr = []
js_divergence_arr = []
tv_distance_arr = []
bhattacharyya_distance_arr = []
bhattacharyya_coefficient_arr = []
pdf_sample2_arr = []

# Loop over different std2 values and calculate distances
for std2 in std2_values:
    np.random.seed(0)  # Set random seed for reproducibility
    sample1 = np.random.normal(0, std1, 1000)
    np.random.seed(0)  # Set random seed for reproducibility
    sample2 = np.random.normal(mean2, std2, 1000)

    # Calculate KDE for the samples
    kde_sample1 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sample1[:, None])
    pdf_sample1 = np.exp(kde_sample1.score_samples(x[:, None]))

    kde_sample2 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sample2[:, None])
    pdf_sample2 = np.exp(kde_sample2.score_samples(x[:, None]))
    
    pdf_sample1 /= np.sum(pdf_sample1)
    pdf_sample2 /= np.sum(pdf_sample2)
    
    pdf_sample2_arr.append(pdf_sample2)

    # Calculate the distances
    wasserstein_dist = wasserstein_distance(x, x, pdf_sample1, pdf_sample2)
    epsilon = 1e-12
    kl_divergence = entropy(pdf_sample1+epsilon, pdf_sample2+epsilon)
    pdf_avg = 0.5 * (pdf_sample1 + pdf_sample2)
    kl_divergence_p_m = entropy(pdf_sample1, pdf_avg)
    kl_divergence_q_m = entropy(pdf_sample2, pdf_avg)
    js_divergence = 0.5 * (kl_divergence_p_m + kl_divergence_q_m)
    tv_distance = 0.5 * np.sum(np.abs(pdf_sample1 - pdf_sample2))
    bhattacharyya_coefficient = np.sum(np.sqrt(pdf_sample1 * pdf_sample2))
    bhattacharyya_distance = -np.log(bhattacharyya_coefficient)

    # Append distances to the arrays
    wasserstein_dist_arr.append(wasserstein_dist)
    kl_divergence_arr.append(kl_divergence)
    js_divergence_arr.append(js_divergence)
    tv_distance_arr.append(tv_distance)
    bhattacharyya_distance_arr.append(bhattacharyya_distance)
    bhattacharyya_coefficient_arr.append(bhattacharyya_coefficient)

# Create a function for the animation
def update(frame):
    lw=2.5
    lw_axes=1.5
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(x, pdf_sample1, label='Distribution 1', lw=lw)
    plt.plot(x, pdf_sample2_arr[frame], label='Distribution 2', lw=lw)
    plt.xlabel('X', fontweight="normal", fontsize=14)
    plt.ylabel('Probability Density', fontweight="normal", fontsize=14)
    plt.title(f"Probability Distributions (std2 = {std2_values[frame]:.2f})", 
              fontweight="normal", fontsize=14)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(lw_axes)
    ax.spines['right'].set_linewidth(lw_axes)
    ax.spines['bottom'].set_linewidth(lw_axes)
    ax.spines['left'].set_linewidth(lw_axes)
    ax.tick_params(axis='x', which='major', labelsize=14, width=lw_axes)
    ax.tick_params(axis='y', which='major', labelsize=14, width=lw_axes)
    plt.legend(loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(std2_values[:frame+1], wasserstein_dist_arr[:frame+1], label='Wasserstein', lw=lw)
    plt.plot(std2_values[:frame+1], kl_divergence_arr[:frame+1], label='KL Divergence', lw=lw)
    plt.plot(std2_values[:frame+1], js_divergence_arr[:frame+1], label='JS Divergence', lw=lw)
    plt.plot(std2_values[:frame+1], tv_distance_arr[:frame+1], label='Total Variation', lw=lw)
    plt.plot(std2_values[:frame+1], bhattacharyya_distance_arr[:frame+1], label='Bhattacharyya', 
             c="violet", lw=lw)
    plt.plot(std2_values[:frame+1], bhattacharyya_coefficient_arr[:frame+1], 
             label='Bhattacharyya coefficient', c="violet", linestyle="--", lw=lw)
    plt.xlabel('Std2', fontweight="normal", fontsize=14)
    plt.ylabel('Distance', fontweight="normal", fontsize=14)
    plt.title('Distance Evolution', fontweight="normal", fontsize=14)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(lw_axes)
    ax.spines['right'].set_linewidth(lw_axes)
    ax.spines['bottom'].set_linewidth(lw_axes)
    ax.spines['left'].set_linewidth(lw_axes)
    ax.tick_params(axis='x', which='major', labelsize=14, width=lw_axes)
    ax.tick_params(axis='y', which='major', labelsize=14, width=lw_axes)
    plt.legend(loc="upper left")
    
    plt.tight_layout()

# Create the animation
fig = plt.figure(figsize=(12, 5.5))
#pdf_sample2_values = [(1 / (std2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean2) / std2) ** 2) for std2 in std2_values]
ani = FuncAnimation(fig, update, frames=len(std2_values), interval=200)

# Save the animation as a gif
ani.save('images/distance_evolution_discrete_std2.gif', writer='imagemagick')
plt.savefig('images/distance_evolution_discrete_std2.png')
# Show the animation (this will not work on all environments)
plt.show()

# %% END