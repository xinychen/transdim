# mats

**Ma**chine learning for multivariate **t**ime **s**eries with missing values.

> This folder includes our latest **imputer** and **predictor** for multivariate time series analysis.

-------------------------------------------

<h3 align='center'> Low-Rank Autoregressive Tensor Completion for Multivariate Time Series Forecasting<br>
    [<a href="https://arxiv.org/abs/2005">arXiv</a>] </h3>
   
<p align="center">
<img align="middle" src="./images/predictor-explained.png" width="666" />
</p>

Building multivariate time series forecasting tool on the well-understood Low-Rank Tensor Completion (LRTC), we develop a **Low-Rank Autoregressive Tensor Completion** which takes into account:

- autoregressive process on the matrix structure to capture local temporal states,
- and low-rank assumption on the tensor structure to capture global low-rank patterns simultaneously.

Code for reproducing experiments is provided in this folder. Please check out `LATC-imputer.ipynb` and `LATC-predictor.ipynb` for details.

### Quick example

### Results

- Ranking report (`imputer`)


- Ranking report (`predictor`)


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
