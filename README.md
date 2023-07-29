# Wasserstein metric

This repository contains the code for the blog posts on 

* [Wasserstein distance and optimal transport](https://www.fabriziomusacchio.com/blog/2023-07-22-wasserstein_distance)
* [Wasserstein distance via entropy regularization (Sinkhorn algorithm)](https://www.fabriziomusacchio.com/blog/2023-07-23-wasserstein_distance_skinhorn)
* [Approximating the Wasserstein distance with cumulative distribution functions ](https://www.fabriziomusacchio.com/blog/2023-07-24-wasserstein_distance_cdf_approximation/)
* [Comparing Wasserstein distance, sliced Wasserstein distance, and L2 norm ](https://www.fabriziomusacchio.com/blog/2023-07-26-wasserstein_vs_l2_norm/)
* [Probability distance metrics in machine learning](https://www.fabriziomusacchio.com/blog/2023-07-28-probability_density_metrics/)

For further details, please refer to this post.

For reproducibility:

```powershell
conda create -n wasserstein -y python=3.9
conda activate wasserstein
conda install mamba -y
mamba install -y numpy matplotlib scikit-learn scipy pot ipykernel
pip install POT
```


## Examples
Two example distributions (source and target):

![img](images/wasserstein_distributions.png)

The according distance (cost) matrix:

![img](images/wasserstein_cost_matrix.png)

And the resulting optimal transport plan:

![img](images/wasserstein_optimal_transport_linear_programming.png)

The corresponding Wasserstein distance is $W_1 = \sim0.1658$.

Comparing Wasserstein distance, sliced Wasserstein distance (SWD), and L2 norm:

![img](images/wasserstein_l2_animation_m2.gif)
![img](images/wasserstein_l2_animation_s2.gif)

Comparing various probability distance metrics:

![img](images/distance_evolution_discrete_mean2.gif)
![img](images/distance_evolution_discrete_std2.gif)




