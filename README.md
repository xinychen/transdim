# *transdim*

**Trans**portation **d**ata **im**putation (***transdim***) covers the following topics:

 - ***Missing data imputation***

	- **Random missing**: each sensor lost their observations at completely random. (*simple task*)

	- **Fiber missing**: each sensor lost their observations during several days. (*difficult task*)

 - ***Rolling traffic prediction*** (short-term)

	- Forecasting **without missing values**. (*simple task*)

	- Forecasting **with incomplete observations**. (*difficult task*)

>Note:
- add a **framework** indicating overall studies;
- list **main contributions** of these studies;
- define the **problems** clearly and describe the **core challenges** intuitively.

# Overview

   >With the development and application of intelligent transportation systems, large quantities of urban traffic data are collected on a continuous basis from various sources, such as loop detectors, cameras, and floating vehicles. These data sets capture the underlying states and dynamics of transportation networks and the whole system and become beneficial to many traffic operation and management applications, including routing, signal control, travel time prediction, and so on. However, the missing data problem is inevitable when collecting traffic data from intelligent transportation systems.

### [Urban traffic speed data set of Guangzhou, China](https://doi.org/10.5281/zenodo.1205228)

  >**Publicly available at our Zenodo repository!**


![example](https://github.com/xinychen/transdim/blob/master/images/estimated_series1.png)
  *(a) Time series of actual and estimated speed within two weeks from August 1 to 14.*

![example](https://github.com/xinychen/transdim/blob/master/images/estimated_series2.png)
  *(b) Time series of actual and estimated speed within two weeks from September 12 to 25.*

*Figure 1: The imputation performance of BGCP (CP rank r=15 and missing rate α=30%) under the fiber missing scenario with third-order tensor representation, where the estimated result of road segment #1 is selected as an example. In the both two panels, red rectangles represent fiber missing (i.e., speed observations are lost in a whole day).*

## Machine learning models

  - ***LocInt***: local interpolation.

  - ***BGCP***: Bayesian Gaussian CP decomposition. [[Matlab code is also available!](https://github.com/lijunsun/bgcp_imputation)]

  - ***BTMF***: Bayesian temporal matrix factorization. [[Matlab code is also available!](https://github.com/lijunsun/btmf)]

  - ***BPMF***: Bayesian probabilistic matrix factorization.

  - ***HaLRTC***

  - ***LSTM***

  - ***CNN***

  - ***GAIN***: Generative Adversarial Imputation Nets.


# Selected references

- **Missing traffic data imputation**

	- Li Qu, Li Li, Yi Zhang, Jianming Hu, 2009. [*PPCA-based missing data imputation for traffic flow volume: a systematical approach*](https://doi.org/10.1109/TITS.2009.2026312). IEEE Transactions on Intelligent Transportation Systems, 10(3): 512-522.

	- Li Li, Yuebiao Li, Zhiheng Li, 2013. [*Efficient missing data imputing for traffic flow by considering temporal and spatial dependence*](https://doi.org/10.1016/j.trc.2013.05.008). Transportation Research Part C: Emerging Technologies, 34: 108-120.

- **Generative Adversarial Nets**

  - Brandon Amos, 2016. [*Image completion with deep learning in TensorFlow*](http://bamos.github.io/2016/08/09/deep-completion/). blog post. [[github](https://github.com/bamos/dcgan-completion.tensorflow)]

  - Jinsun Yoon, James Jordon, Mihaela van der Schaar, 2018. [*GAIN: missing data imputation using Generative Adversarial Nets*](http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf). Proceedings of the 35th International Conference on Machine Learning (*ICML 2018*), Stockholm, Sweden. [[supplementary materials](http://medianetlab.ee.ucla.edu/papers/ICML_GAIN_Supp.pdf)]

- **Tensor regression**

  - Guillaume Rabusseau, Hachem Kadri, 2016. [*Low-rank regression with tensor responses*](https://papers.nips.cc/paper/6302-low-rank-regression-with-tensor-responses.pdf). 30th Conference on Neural Information Processing Systems (*NIPS 2016*), Barcelona, Spain.

  - Rose Yu, Yan Liu, 2016. [*Learning from multiway data: simple and efficient tensor regression*](http://proceedings.mlr.press/v48/yu16.pdf). Proceedings of the 33rd International Conference on Machine Learning (*ICML 2016*), New York, NY, USA.

  - Rose Yu, Guangyu Li, Yan Liu, 2018. [*Tensor regression meets Gaussian processes*](http://proceedings.mlr.press/v84/yu18a/yu18a.pdf). Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (*AISTATS 2018*), Lanzarote, Spain.

# Publications

  - Xinyu Chen, Zhaocheng He, Jiawei Wang, 2018. [*Spatial-temporal traffic speed patterns discovery and incomplete data recovery via SVD-combined tensor decomposition*](https://doi.org/10.1016/j.trc.2017.10.023). Transportation Research Part C: Emerging Technologies, 86: 59-77.

  - Xinyu Chen, Zhaocheng He, Lijun Sun, 2018. *A Bayesian tensor decomposition approach for spatiotemporal traffic data imputation*. Transportation Research Part C: Emerging Technologies. (*under review*)

  >Please consider citing our papers if they help your research.
