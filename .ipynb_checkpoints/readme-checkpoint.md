# What it is?

This is a Python package for many-body approximation and its-based low-rank approximation that covers the following papers:

- Many-body approximation for non-negative tensors (Neurips2023) 
- Deformed many-body approximation for non-negative tensors (AISTATS2026)

Our code works on Python 3.12.7 with required packages:

```
Numpy 2.4.2, itertools, scipy 1.15.3
```

# What is the deformed decomposition?

For given tensor `P`, its deformed one-body, two-body, and three-body approximation can be given as follows.

$$
\begin{align}
P_{ijkl} \simeq Q_{ijkl} &= p_i^{(1)} p_j^{(2)}p_k^{(3)}p_l^{(4)} \\
P_{ijkl} \simeq Q_{ijkl} &= X_{ij}^{(12)}X_{ik}^{(13)}X_{il}^{(14)}X_{jk}^{(23)}X_{kl}^{(34)} \\
P_{ijkl} \simeq Q_{ijkl} &= \chi_{ijk}^{(123)} \chi_{ijl}^{(124)} \chi_{ikl}^{(134)} \chi_{jkl}^{(234)}
\end{align}
$$

where 

# Dual representation of the theta- and eta- coordinate system

In this library, we regard non-negative normalized tensors as discrete joint distributions and describe the tensor by natural parameter and expectation parameter of the deformed exponential family. Lets see an example. Firstly, we define a random rank-1 tensor

```
import sys
sys.path.append("src/")
import transform
import utils_test
import numpy as np

N = 3
T = utils_test.generate_low_rank_tensor( (N,N), 10, seed=None)
T = T / np.sum(T)
```

We can convert the tensor with natural parameters of the q-exponential family. 

```
theta = transform.theta_from_prob(T, chi="Tsallis", q=1)
theta
```

where `q=1` denotes a natural parameter without deformation. If we change the value of `q`, it distorts the coordinate system, which changes the parameters. We can also try Kaniadakis-defomration by option `chi="Kani"` with hyper-parameter `k`, where `k=0` recoves ordianly exponteail family without defomration. In the same way, we obtain a natural parameter:

```
eta = transform.eta_from_prob(T, chi="Tsallis", q=1)
eta
```

Our algorithm performs tensor decomposition, leveraging the theta and eta representations of the tensor as see blow.


# Examples of deformed many-body approximation.

Lets see an example of many-body approximation. We define the random rank-10 tensor.

```
import utils_test
N = 8
P = utils_test.generate_low_rank_tensor( (N,N,N,N), 10, seed=None)
P = P / np.sum(P)
```

Then, its three body approximation can be given as follows:
```
body = 3
Q, theta, eta, his = MBA(P, body, lr_search=True, Newton=True, max_iter=100, epsilon_auto=True,
                         chi="Tsallis", q=0.5, 
                         verbose_interval=1, verbose=True);
```



# Citation

If you use this source code in a scientific publication, please consider citing the following papers:

```
@inproceedings{Ghalamkari2023Many,
    Author = {Ghalamkari, K. and Sugiyama, M. and Kawahara, Y.},
    Title = {Many-body Approximation for Non-negative Tensors},
    Booktitle = {Advances in Neural Information Processing Systems},
    Volume = {36},
    Pages = {74077--74102},
    Address = {New Orleans, US},
    Month = {December},
    Year = {2023}}

@inproceedings{Ghalamkari2026Deformed,
    Author = {Ghalamkari, K., Taborsky, P., Mørup, M.},
    Title = {Deformed decomposition for non-negative tensors},
    Booktitle = {Proceedings of the 29th International Conference on Artificial Intelligence and Statistics},
    Volume = {--},
    Pages = {--},
    Address = {Tangier, Morocco},
    Month = {May},
    Year = {2026}}
```

# Further readings

- Convex Manifold Approximation for Tensors [[Theis]](https://ir.soken.ac.jp/records/6661)
- How to choose interaction automatically? [[arXiv]](https://arxiv.org/pdf/2410.11964) 
- Blind Source Separation via Legendre Transformation [[Paper]](https://proceedings.mlr.press/v161/luo21a.html) [[Code]](https://github.com/sjmluo/IGLLM?utm_source=catalyzex.com) [[Slide]](https://github.com/sjmluo/IGLLM/blob/master/IGBSS_NeurIPS2020_Poster.pdf)
- Relationship between many-body approximation and low-rank approximation [[arxiv]](https://arxiv.org/abs/2405.18220)
- Coordinate Descent Method for Log-linear Model on Posets, by Hayashi, S., Sugiyama, M., & Matsushima, S. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9260027)

# Links to other packages

Many-body approximation
- [Python implementation](https://github.com/kojima-r/pyLegendreDecomposition)
- [Julia implementation](https://github.com/gkazunii/IgTensors/)

Legendre decomposition
- [Python implementation by R. Kojima](https://github.com/kojima-r/pyLegendreDecomposition)
- [C++ implementation by M. Sugiyama](https://github.com/mahito-sugiyama/Legendre-decomposition)
- [Python implementation by Y. Kawakami](https://github.com/Yhkwkm/legendre-decomposition-python)

Tensor balancing
- [C++ implementation by M. Sugiyama](https://github.com/mahito-sugiyama/newton-balancing)
- [Julia implementation](https://github.com/k-kitai/TensorBalancing.jl) 

# Acknowledgement
This work was supported by the Danish Data Science Academy, which is funded by the Novo Nordisk Foundation, Grant Number NNF21SA0069429.

