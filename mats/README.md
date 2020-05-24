# mats

**Ma**chine learning for multivariate **t**ime **s**eries with missing values.

> This folder includes our latest **imputer** and **predictor** for multivariate time series analysis.

-------------------------------------------

<h3 align='center'> Low-Rank Autoregressive Tensor Completion for Multivariate Time Series Forecasting<br>
    [<a href="https://arxiv.org/abs/2005">arXiv</a>] </h3>
   
<p align="center">
<img align="middle" src="https://github.com/xinychen/transdim/blob/master/images/predictor-explained.png" width="666" />
</p>

Building multivariate time series forecasting tool on the well-understood Low-Rank Tensor Completion (LRTC), we develop a **Low-Rank Autoregressive Tensor Completion** which takes into account:

- autoregressive process on the matrix structure to capture local temporal states,
- and low-rank assumption on the tensor structure to capture global low-rank patterns simultaneously.

Code for reproducing experiments is provided in this folder. Please check out `LATC-imputer.ipynb` and `LATC-predictor.ipynb` for details.

### Quick example

### Results

- **Ranking report (`imputer`)**

We create this report by evaluating our proposed imputation models on Guangzhou traffic speed data set and PeMS traffic speed data set with certain amount of missing values.

| Data Set |          No           | Model | MAPE (%) | RMSE | Running Time (sec.) |
| :------: | :-------------------: | :---- | :------: | :--: | :--------------: |
|Guangzhou (40%, RM) | 1 | LATC |   6.79   |   2.96   | - |
|                    | 2 | [LRTC-TNN](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-LRTC-TNN.ipynb)    |   7.32   |   3.17   | 40 |
|                    | 3 | [BTMF](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-BTMF.ipynb)    |   7.81   |   3.35   | - |
|                    | 4 | [BGCP](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-BGCP.ipynb)    |   8.29   |   3.59   | - |

| Data Set |          No           | Model | MAPE (%) | RMSE | Running Time (sec.) |
| :------: | :-------------------: | :---- | :------: | :--: | :--------------: |
|Guangzhou (40%, NM) | 1 | LATC |   9.51   |   4.07   | - |
|                    | 2 | [LRTC-TNN](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-LRTC-TNN.ipynb)    |   9.54   |   4.08   | 36 |
|                    | 3 | [BGCP](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-BGCP.ipynb)    |   10.25   |   4.32   | 265 |
|                    | 4 | [BTMF](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-BTMF.ipynb)    |   10.36   |   4.46   | 3885 |


- **Ranking report (`predictor`)**


### Citation

If you use these codes in your work, please cite our paper:

```bibtex
@article{chen2020lowrank,
    author={xx},
    title={{Low-Rank Autoregressive Tensor Completion for Multivariate Time Series Forecasting}},
    year={2020},
    journal={arXiv:2005.xxxx}
}
```
