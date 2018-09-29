# *transdim*

**Trans**portation **d**ata **im**putation (**transdim**) covers the following topics:

 - ***Missing data imputation***
 
-- Scenario #1: **Random** missing where sensor lost their observations at completely random. (*simple task*)

-- Scenario #2: **Fiber** missing where sensor lost their observations during several complete days at random. (*difficult task*)

 - ***Rolling traffic prediction*** (short-term)
 
-- Forecasting **without missing values**. (*simple task*)

-- Forecasting **with incomplete observations**. (*difficult task*)

   >With the development and application of intelligent transportation systems, large quantities of urban traffic data are collected on a continuous basis from various sources, such as loop detectors, cameras, and floating vehicles. These data sets capture the underlying states and dynamics of transportation networks and the whole system and become beneficial to many traffic operation and management applications, including routing, signal control, travel time prediction, and so on. However, the missing data problem is inevitable when collecting traffic data from intelligent transportation systems.
 
![example](https://github.com/xinychen/transdim/blob/master/images/estimated_series1.png)
*Figure 1: An example of time series consisting of incomplete speed observations.*

![example](https://github.com/xinychen/transdim/blob/master/images/estimated_series2.png)
*Figure 2: An example of time series consisting of incomplete speed observations.*


# References

## Tensor regression

 - Guillaume Rabusseau, Hachem Kadri, 2016. [*Low-rank regression with tensor responses*](https://papers.nips.cc/paper/6302-low-rank-regression-with-tensor-responses.pdf). 30th Conference on Neural Information Processing Systems (*NIPS 2016*), Barcelona, Spain.
 - Rose Yu, Yan Liu, 2016. [*Learning from multiway data: simple and efficient tensor regression*](http://proceedings.mlr.press/v48/yu16.pdf). Proceedings of the 33rd International Conference on Machine Learning (*ICML 2016*), New York, NY, USA, 2016.
 - Rose Yu, Guangyu Li, Yan Liu, 2018. [*Tensor regression meets Gaussian processes*](http://proceedings.mlr.press/v84/yu18a/yu18a.pdf). Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (*AISTATS*) 2018, Lanzarote, Spain. PMLR: Volume 84.
 
# Publications

 - Xinyu Chen, Zhaocheng He, Jiawei Wang, 2018. [*Spatial-temporal traffic speed patterns discovery and incomplete data recovery via SVD-combined tensor decomposition*](https://doi.org/10.1016/j.trc.2017.10.023). Transportation Research Part C: Emerging Technologies, 86: 59-77. [[data](https://doi.org/10.5281/zenodo.1205228)]

 - Xinyu Chen, Zhaocheng He, Lijun Sun, 2018. *A Bayesian tensor decomposition approach for spatiotemporal traffic data imputation*. Transportation Research Part C: Emerging Technologies. (*under review*)

