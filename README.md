# transdim

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![repo size](https://img.shields.io/github/repo-size/xinychen/transdim.svg)](https://github.com/xinychen/transdim/archive/master.zip)
[![GitHub stars](https://img.shields.io/github/stars/xinychen/transdim.svg?logo=github&label=Stars&logoColor=white)](https://github.com/xinychen/transdim)

<h6 align="center">Made by Xinyu Chen • :globe_with_meridians: <a href="https://xinychen.github.io">https://xinychen.github.io</a></h6>

![logo](https://github.com/xinychen/transdim/blob/master/images/transdim_logo_large.png)

**Trans**portation **d**ata **im**putation (a.k.a., transdim). 

Machine learning models make important developments in the field of spatiotemporal data modeling - like how to forecast near-future traffic states of road networks. But what happens when these models are built on incomplete data commonly collected from real-world systems (e.g., transportation system)?

<br>

Table of Content
--------------

- [About this Project](https://github.com/xinychen/transdim?tab=readme-ov-file#about-this-project)
- [Tasks and Challenges](https://github.com/xinychen/transdim?tab=readme-ov-file#tasks-and-challenges)
- [Implementation](https://github.com/xinychen/transdim?tab=readme-ov-file#quick-start)
- [Quick Start](https://github.com/xinychen/transdim?tab=readme-ov-file#quick-start)
- [Documentation](https://github.com/xinychen/transdim?tab=readme-ov-file#documentation)
- [Publications](https://github.com/xinychen/transdim?tab=readme-ov-file#publications)
- [Contributors](https://github.com/xinychen/transdim?tab=readme-ov-file#collaborators)

<br>

About this Project
--------------

In the **transdim** project, we develop machine learning models to help address some of the toughest challenges of spatiotemporal data modeling - from missing data imputation to time series prediction. The strategic aim of this project is **creating accurate and efficient solutions for spatiotemporal traffic data imputation and prediction tasks**.

In a hurry? Please check out our contents as follows.

<br>

Tasks and Challenges
--------------

> Missing data are there, whether we like them or not. The really interesting question is how to deal with incomplete data.

<p align="center">
<img align="middle" src="https://github.com/xinychen/transdim/blob/master/images/missing.png" width="800" />
</p>

<p align = "center">
<b>Figure 1</b>: Two classical missing patterns in a spatiotemporal setting.
</p>

We create three missing data mechanisms on real-world data.

- **Missing data imputation** 🔥

  - Random missing (RM): Each sensor lost observations at completely random. (★★★)
  - Non-random missing (NM): Each sensor lost observations during several days. (★★★★)
  - Blockout missing (BM): All sensors lost their observations at several consecutive time points. (★★★★)

<p align="center">
<img src="https://github.com/xinychen/transdim/blob/master/images/framework.png" alt="drawing" width="800"/>
</p>

<p align = "center">
<b>Figure 2</b>: Tensor completion framework for spatiotemporal missing traffic data imputation.
</p>

- **Spatiotemporal prediction** 🔥
  - Forecasting without missing values. (★★★)
  - Forecasting with incomplete observations. (★★★★★)

<p align="center">
<img align="middle" src="https://github.com/xinychen/transdim/blob/master/images/predictor-explained.png" width="700" />
</p>

<p align = "center">
<b>Figure 3</b>: Illustration of our proposed Low-Rank Autoregressive Tensor Completion (LATC) imputer/predictor with a prediction window τ (green nodes: observed values; white nodes: missing values; red nodes/panel: prediction; blue panel: training data to construct the tensor).
</p>

<br>

Implementation
--------------

### Open data

In this project, we have adapted some publicly available data sets into our experiments. The original links for these data are summarized as follows,

- **Multivariate time series**
  - [Birmingham parking data set](https://archive.ics.uci.edu/ml/datasets/Parking+Birmingham)
  - [California PeMS traffic speed data set](https://doi.org/10.5281/zenodo.3939792) (large-scale)
  - [Guangzhou urban traffic speed data set](https://doi.org/10.5281/zenodo.1205228)
  - [Hangzhou metro passenger flow data set](https://doi.org/10.5281/zenodo.3145403)
  - [London urban movement speed data set](https://movement.uber.com/) (other cities are also available at [Uber movement project](https://movement.uber.com/))
  - [Portland highway traffic data set](https://portal.its.pdx.edu/home) (including traffic volume/speed/occupancy, see [data documentation](https://portal.its.pdx.edu/static/files/fhwa/Freeway%20Data%20Documentation.pdf))
  - [Seattle freeway traffic speed data set](https://github.com/zhiyongc/Seattle-Loop-Data)
- **Multidimensional time series**
  - [New York City (NYC) taxi data set](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  - [Pacific surface temperature data set](http://iridl.ldeo.columbia.edu/SOURCES/.CAC/)

For example, if you want to view or use these data sets, please download them at the [../datasets/](https://github.com/xinychen/transdim/tree/master/datasets) folder in advance, and then run the following codes in your Python console:

```python
import scipy.io

tensor = scipy.io.loadmat('../datasets/Guangzhou-data-set/tensor.mat')
tensor = tensor['tensor']
```

In particular, if you are interested in large-scale traffic data, we recommend **PeMS-4W/8W/12W** and [UTD19](https://utd19.ethz.ch/index.html). For PeMS data, you can download the data from [Zenodo](https://doi.org/10.5281/zenodo.3939792) and place them at the folder of datasets (data path example: `../datasets/California-data-set/pems-4w.csv`). Then you can use `Pandas` to open data:

```python
import pandas as pd

data = pd.read_csv('../datasets/California-data-set/pems-4w.csv', header = None)
```

For model evaluation, we mask certain entries of the "observed" data as missing values and then perform imputation for these "missing" values.

### Model implementation

In our experiments, we implemented some machine learning models mainly on `Numpy`, and written these Python codes with **Jupyter Notebook**. If you want to evaluate these models, please download and run these notebooks directly (prerequisite: **download the data sets** in advance). In the following implementation, we have improved Python codes (in Jupyter Notebook) in terms of both readiability and efficiency.

> Our proposed models are highlighted in bold fonts.

- **imputer** (imputation models)

| Notebook                                        | Guangzhou | Birmingham | Hangzhou | Seattle | London | NYC | Pacific |
| :----------------------------------------------------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [BPMF](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/imputer/BPMF.ipynb) |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   🔶   |   🔶   |
| [TRMF](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/imputer/TRMF.ipynb) |   ✅   |   🔶   |   ✅   |   ✅   |   ✅   |   🔶   |   🔶   |
| [BTRMF](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/imputer/BTRMF.ipynb) |   ✅   |   🔶   |   ✅   |   ✅   |   ✅   |   🔶   |   🔶   |
| [**BTMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/imputer/BTMF.ipynb) |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   🔶   |   🔶   |
| [**BGCP**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/imputer/BGCP.ipynb) |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
| [**BATF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/imputer/BATF.ipynb) |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
| [**BTTF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/imputer/BTTF.ipynb) |   🔶   |   🔶   |   🔶   |   🔶   |   🔶   |   ✅   |   ✅   |
| [HaLRTC](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/imputer/HaLRTC.ipynb) |   ✅   |   🔶   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
| [**LRTC-TNN**](https://nbviewer.org/github/xinychen/transdim/blob/master/imputer/LRTC-TNN.ipynb) |   ✅   |   🔶   |   ✅   |   ✅   |   🔶   |   🔶   |   🔶   |

- **predictor** (prediction models)

| Notebook                                        | Guangzhou | Birmingham | Hangzhou | Seattle | London | NYC | Pacific |
| :----------------------------------------------------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [TRMF](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/predictor/TRMF.ipynb) |   ✅   |   🔶   |   ✅   |   ✅   |   ✅   |   🔶   |   🔶   |
| [BTRMF](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/predictor/BTRMF.ipynb) |   ✅   |   🔶   |   ✅   |   ✅   |   ✅   |   🔶   |   🔶   |
| [BTRTF](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/predictor/BTRTF.ipynb) |   🔶   |   🔶   |   🔶   |   🔶   |   🔶   |   ✅   |   ✅   |
| [**BTMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/predictor/BTMF.ipynb) |   ✅   |   🔶   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
| [**BTTF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/predictor/BTTF.ipynb) |   🔶   |   🔶   |   🔶   |   🔶   |   🔶   |   ✅   |   ✅   |

* ✅ — Cover
* 🔶 — Does not cover
* 🚧 — Under development

> For the implementation of these models, we use both `dense_mat` and `sparse_mat` (or `dense_tensor` and `sparse_tensor`) as inputs. However, it is not necessary by doing so if you do not hope to see the imputation/prediction performance in the iterative process, you can remove `dense_mat` (or `dense_tensor`) from the inputs of these algorithms.

### Imputation/Prediction performance

- **Imputation example (on Guangzhou data)**

![example](https://github.com/xinychen/transdim/blob/master/images/estimated_series1.png)
  *(a) Time series of actual and estimated speed within two weeks from August 1 to 14.*

![example](https://github.com/xinychen/transdim/blob/master/images/estimated_series2.png)
  *(b) Time series of actual and estimated speed within two weeks from September 12 to 25.*

> The imputation performance of BGCP (CP rank r=15 and missing rate α=30%) under the fiber missing scenario with third-order tensor representation, where the estimated result of road segment #1 is selected as an example. In the both two panels, red rectangles represent fiber missing (i.e., speed observations are lost in a whole day).

- **Prediction example**

![example](https://github.com/xinychen/transdim/blob/master/images/prediction_hangzhou.png)

![example](https://github.com/xinychen/transdim/blob/master/images/prediction_nyc_heatmap.png)

![example](https://github.com/xinychen/transdim/blob/master/images/prediction_nyc.png)

<br>

Quick Start
--------------
This is an imputation example of Low-Rank Tensor Completion with Truncated Nuclear Norm minimization (LRTC-TNN). One notable thing is that unlike the complex equations in our paper, our Python implementation is extremely easy to work with.

- First, import some necessary packages:

```python
import numpy as np
from numpy.linalg import inv as inv
```

- Define the operators of tensor unfolding (`ten2mat`) and matrix folding (`mat2ten`) using `Numpy`:

```python
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')
```

```python
def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order = 'F'), 0, mode)
```

- Define Singular Value Thresholding (SVT) for Truncated Nuclear Norm (TNN) minimization:

```python
def svt_tnn(mat, tau, theta):
    [m, n] = mat.shape
    if 2 * m < n:
        u, s, v = np.linalg.svd(mat @ mat.T, full_matrices = 0)
        s = np.sqrt(s)
        idx = np.sum(s > tau)
        mid = np.zeros(idx)
        mid[:theta] = 1
        mid[theta:idx] = (s[theta:idx] - tau) / s[theta:idx]
        return (u[:,:idx] @ np.diag(mid)) @ (u[:,:idx].T @ mat)
    elif m > 2 * n:
        return svt_tnn(mat.T, tau, theta).T
    u, s, v = np.linalg.svd(mat, full_matrices = 0)
    idx = np.sum(s > tau)
    vec = s[:idx].copy()
    vec[theta:] = s[theta:] - tau
    return u[:,:idx] @ np.diag(vec) @ v[:idx,:]
```

- Define performance metrics (i.e., RMSE, MAPE):

```python
def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]
```

- Define LRTC-TNN:

```python
def LRTC(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, maxiter):
    """Low-Rank Tensor Completion with Truncated Nuclear Norm, LRTC-TNN."""
    
    dim = np.array(sparse_tensor.shape)
    pos_missing = np.where(sparse_tensor == 0)
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    dense_test = dense_tensor[pos_test]
    del dense_tensor
    
    X = np.zeros(np.insert(dim, 0, len(dim))) # \boldsymbol{\mathcal{X}}
    T = np.zeros(np.insert(dim, 0, len(dim))) # \boldsymbol{\mathcal{T}}
    Z = sparse_tensor.copy()
    last_tensor = sparse_tensor.copy()
    snorm = np.sqrt(np.sum(sparse_tensor ** 2))
    it = 0
    while True:
        rho = min(rho * 1.05, 1e5)
        for k in range(len(dim)):
            X[k] = mat2ten(svt_tnn(ten2mat(Z - T[k] / rho, k), alpha[k] / rho, np.int(np.ceil(theta * dim[k]))), dim, k)
        Z[pos_missing] = np.mean(X + T / rho, axis = 0)[pos_missing]
        T = T + rho * (X - np.broadcast_to(Z, np.insert(dim, 0, len(dim))))
        tensor_hat = np.einsum('k, kmnt -> mnt', alpha, X)
        tol = np.sqrt(np.sum((tensor_hat - last_tensor) ** 2)) / snorm
        last_tensor = tensor_hat.copy()
        it += 1
        if (it + 1) % 50 == 0:
            print('Iter: {}'.format(it + 1))
            print('MAPE: {:.6}'.format(compute_mape(dense_test, tensor_hat[pos_test])))
            print('RMSE: {:.6}'.format(compute_rmse(dense_test, tensor_hat[pos_test])))
            print()
        if (tol < epsilon) or (it >= maxiter):
            break

    print('Imputation MAPE: {:.6}'.format(compute_mape(dense_test, tensor_hat[pos_test])))
    print('Imputation RMSE: {:.6}'.format(compute_rmse(dense_test, tensor_hat[pos_test])))
    print()
    
    return tensor_hat
```

- Let us try it on Guangzhou urban traffic speed data set:

```python
import scipy.io

import scipy.io
import numpy as np
np.random.seed(1000)

dense_tensor = scipy.io.loadmat('../datasets/Guangzhou-data-set/tensor.mat')['tensor']
dim = dense_tensor.shape
missing_rate = 0.2 # Random missing (RM)
sparse_tensor = dense_tensor * np.round(np.random.rand(dim[0], dim[1], dim[2]) + 0.5 - missing_rate)
```

- Run the imputation experiment:

```python
import time
start = time.time()
alpha = np.ones(3) / 3
rho = 1e-5
theta = 0.30
epsilon = 1e-4
maxiter = 200
tensor_hat = LRTC(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, maxiter)
end = time.time()
print('Running time: %d seconds'%(end - start))
```

> This example is from [../imputer/LRTC-TNN.ipynb](https://nbviewer.org/github/xinychen/transdim/blob/master/imputer/LRTC-TNN.ipynb), you can check out this Jupyter Notebook for details.

<br>

Documentation
--------------

1. [Intuitive understanding of randomized singular value decomposition](https://towardsdatascience.com/intuitive-understanding-of-randomized-singular-value-decomposition-9389e27cb9de). July 1, 2020.
2. [Generating random numbers and arrays in Matlab and Numpy](https://towardsdatascience.com/generating-random-numbers-and-arrays-in-matlab-and-numpy-47dcc9997650). October 9, 2021.
3. [Reduced-rank vector autoregressive model for high-dimensional time series forecasting](https://towardsdatascience.com/reduced-rank-vector-autoregressive-model-for-high-dimensional-time-series-forecasting-bdd17df6c5ab). October 16, 2021.
4. [Dynamic mode decomposition for spatiotemporal traffic speed time series in Seattle freeway](https://towardsdatascience.com/dynamic-mode-decomposition-for-spatiotemporal-traffic-speed-time-series-in-seattle-freeway-b0ba97e81c2c#ce4e-5f7c3f01d622). October 29, 2021.
5. [Analyzing missing data problem in Uber movement speed data](https://medium.com/@xinyu.chen/analyzing-missing-data-problem-in-uber-movement-speed-data-208d7a126af5). February 14, 2022.
6. [Using conjugate gradient to solve matrix equations](https://medium.com/p/7f16cbae18a3). February 23, 2022.
7. [Inpainting fluid dynamics with tensor decomposition (NumPy)](https://medium.com/p/d84065fead4d). March 15, 2022.
8. [Temporal matrix factorization for multivariate time series forecasting](https://medium.com/p/b1c59faf05ea). March 20, 2022.
9. [Forecasting multivariate time series with nonstationary temporal matrix factorization](https://medium.com/p/4705df163fcf). April 25, 2022.
10. [Implementing Kronecker product decomposition with NumPy](https://medium.com/p/13f679f76347). June 20, 2022.
11. [Tensor autoregression: A multidimensional time series model](https://medium.com/p/21681f696d79). September 3, 2022.
12. [Reproducing dynamic mode decomposition on fluid flow data in Python](https://medium.com/@xinyu.chen/reproducing-dynamic-mode-decomposition-on-fluid-flow-data-in-python-94b8d7e1f203). September 6, 2022.
13. [Convolution nuclear norm minimization for time series modeling](https://medium.com/p/377c56e49962). October 3, 2022.
14. [Reinforce matrix factorization for time series modeling: Probabilistic sequential matrix factorization](https://medium.com/p/873f4ca344de). October 5, 2022.
15. [Discrete convolution and fast Fourier transform explained and implemented step by step](https://medium.com/p/83ff1809378d). October 19, 2022.
16. [Matrix factorization for image inpainting in Python](https://medium.com/p/d7300e6afbfd). December 8, 2022.
17. [Circulant matrix nuclear norm minimization for image inpainting in Python](https://medium.com/p/b98eb94d8e). December 9, 2022.
18. [Low-rank Laplacian convolution model for time series imputation and image inpainting](https://medium.com/p/a46dd88d107e). December 10, 2022.
19. [Low-rank Laplacian convolution model for color image inpainting](https://medium.com/p/e8c5cdb3cc73). December 17, 2022.
20. [Intuitive understanding of tensors in machine learning](https://medium.com/@xinyu.chen/intuitive-understanding-of-tensors-in-machine-learning-33635c64b596). January 20, 2023.
21. [Low-rank matrix and tensor factorization for speed field reconstruction](https://medium.com/p/bb4807cb93c5). March 9, 2023.
22. [Bayesian vector autoregression forecasting](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/toy-examples/Bayesian-VAR-forecasting.ipynb)
23. [Structured low-rank matrix completion](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/toy-examples/SLRMC.ipynb)

<br>

Publications
--------------

- Xinyu Chen, Zhanhong Cheng, HanQin Cai, Nicolas Saunier, Lijun Sun (2024). **Laplacian convolutional representation for traffic time series imputation**. IEEE Transactions on Knowledge and Data Engineering. 36 (11): 6490-6502. [[DOI](https://doi.org/10.1109/TKDE.2024.3419698)] [[Slides](https://xinychen.github.io/slides/LCR24.pdf)] [[Data & Python code](https://github.com/xinychen/LCR)]

- Xinyu Chen, Lijun Sun (2022). **Bayesian temporal factorization for multidimensional time series prediction**. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44 (9): 4659-4673. [[Preprint](https://arxiv.org/abs/1910.06366v2)] [[DOI](https://doi.org/10.1109/TPAMI.2021.3066551)] [[Slides](https://doi.org/10.5281/zenodo.4693404)] [[Data & Python code](https://github.com/xinychen/transdim)]

- Xinyu Chen, Mengying Lei, Nicolas Saunier, Lijun Sun (2022). **Low-rank autoregressive tensor completion for spatiotemporal traffic data imputation**. IEEE Transactions on Intelligent Transportation Systems, 23 (8): 12301-12310. [[Preprint](https://arxiv.org/abs/2104.14936)] [[DOI](https://doi.org/10.1109/TITS.2021.3113608)] [[Data & Python code](https://github.com/xinychen/transdim)] (Also accepted in part to [MiLeTS Workshop of KDD 2021](https://kdd-milets.github.io/milets2021/), see [workshop paper](https://kdd-milets.github.io/milets2021/papers/MiLeTS2021_paper_23.pdf))

- Xinyu Chen, Yixian Chen, Nicolas Saunier, Lijun Sun (2021). **Scalable low-rank tensor learning for spatiotemporal traffic data imputation**. Transportation Research Part C: Emerging Technologies, 129: 103226. [[Preprint](https://arxiv.org/abs/2008.03194)] [[DOI](https://doi.org/10.1016/j.trc.2021.103226)] [[Data](https://doi.org/10.5281/zenodo.3939792)] [[Python code](https://github.com/xinychen/transdim/tree/master/large-imputer)]

- Xinyu Chen, Jinming Yang, Lijun Sun (2020). **A nonconvex low-rank tensor completion model for spatiotemporal traffic data imputation**. Transportation Research Part C: Emerging Technologies, 117: 102673. [[Preprint](https://arxiv.org/abs/2003.10271v2)] [[DOI](https://doi.org/10.1016/j.trc.2020.102673)] [[Data & Python code](https://github.com/xinychen/transdim)]

- Xinyu Chen, Zhaocheng He, Yixian Chen, Yuhuan Lu, Jiawei Wang (2019). **Missing traffic data imputation and pattern discovery with a Bayesian augmented tensor factorization model**. Transportation Research Part C: Emerging Technologies, 104: 66-77. [[DOI](https://doi.org/10.1016/j.trc.2019.03.003)] [[Slides](https://doi.org/10.5281/zenodo.2632552)] [[Data](http://doi.org/10.5281/zenodo.1205229)] [[Matlab code](https://github.com/sysuits/BATF)] [[Python code](https://github.com/xinychen/transdim/blob/master/imputer/BATF.ipynb)]

- Xinyu Chen, Zhaocheng He, Lijun Sun (2019). **A Bayesian tensor decomposition approach for spatiotemporal traffic data imputation**. Transportation Research Part C: Emerging Technologies, 98: 73-84. [[Preprint](https://www.researchgate.net/publication/329177786_A_Bayesian_tensor_decomposition_approach_for_spatiotemporal_traffic_data_imputation)] [[DOI](https://doi.org/10.1016/j.trc.2018.11.003)] [[Data](http://doi.org/10.5281/zenodo.1205229)] [[Matlab code](https://github.com/lijunsun/bgcp_imputation)] [[Python code](https://github.com/xinychen/transdim/blob/master/experiments/Imputation-BGCP.ipynb)]

- Xinyu Chen, Zhaocheng He, Jiawei Wang (2018). **Spatial-temporal traffic speed patterns discovery and incomplete data recovery via SVD-combined tensor decomposition**. Transportation Research Part C: Emerging Technologies, 86: 59-77. [[DOI](http://doi.org/10.1016/j.trc.2017.10.023)] [[Data](http://doi.org/10.5281/zenodo.1205229)]

  >This project is from the above papers, please cite these papers if they help your research.

<br>

Collaborators
--------------

<table>
  <tr>
    <td align="center"><a href="https://github.com/xinychen"><img src="https://github.com/xinychen.png?size=80" width="80px;" alt="Xinyu Chen"/><br /><sub><b>Xinyu Chen</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=xinychen" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/yangjm67"><img src="https://github.com/yangjm67.png?size=80" width="80px;" alt="Jinming Yang"/><br /><sub><b>Jinming Yang</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=yangjm67" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/yxnchen"><img src="https://github.com/yxnchen.png?size=80" width="80px;" alt="Yixian Chen"/><br /><sub><b>Yixian Chen</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=yxnchen" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/MengyingLei"><img src="https://github.com/MengyingLei.png?size=80" width="80px;" alt="Mengying Lei"/><br /><sub><b>Mengying Lei</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=MengyingLei" title="Code">💻</a></td>
<!--     <td align="center"><a href="https://github.com/lijunsun"><img src="https://github.com/lijunsun.png?size=80" width="80px;" alt="Lijun Sun"/><br /><sub><b>Lijun Sun</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=lijunsun" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/HanTY"><img src="https://github.com/HanTY.png?size=80" width="80px;" alt="Tianyang Han"/><br /><sub><b>Tianyang Han</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=HanTY" title="Code">💻</a></td> -->
<!--   </tr>
  <tr>
    <td align="center"><a href="https://github.com/xxxx"><img src="https://github.com/xxxx.png?size=100" width="100px;" alt="xxxx"/><br /><sub><b>xxxx</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=xxxx" title="Code">💻</a></td> -->
  </tr>
</table>

- **Advisory Board**

<table>
  <tr>
    <td align="center"><a href="https://github.com/lijunsun"><img src="https://github.com/lijunsun.png?size=80" width="80px;" alt="Lijun Sun"/><br /><sub><b>Lijun Sun</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=lijunsun" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/nsaunier"><img src="https://github.com/nsaunier.png?size=80" width="80px;" alt="Nicolas Saunier"/><br /><sub><b>Nicolas Saunier</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=nsaunier" title="Code">💻</a></td>
  </tr>
</table>

> See the list of [contributors](https://github.com/xinychen/transdim/graphs/contributors) who participated in this project.

<br>

Supported by
--------------

<a href="https://ivado.ca/en">
<img align="middle" src="https://github.com/xinychen/tracebase/blob/main/graphics/ivado_logo.jpeg" alt="drawing" height="70" hspace="50">
</a>
<a href="https://www.cirrelt.ca/">
<img align="middle" src="https://github.com/xinychen/tracebase/blob/main/graphics/cirrelt_logo.png" alt="drawing" height="50">
</a>

<br>

License
--------------

This work is released under the MIT license.
