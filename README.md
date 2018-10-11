# *transdim*

**Trans**portation **d**ata **im**putation (***transdim***) covers the following topics:

 - ***Missing data imputation***

	- **Random missing**: each sensor lost their observations at completely random. (*simple task*)

	- **Fiber missing**: each sensor lost their observations during several days. (*difficult task*)

 - ***Rolling traffic prediction*** (short-term)

	- Forecasting **without missing values**. (*simple task*)

	- Forecasting **with incomplete observations**. (*difficult task*)


>**What I should do just now!**

- add a **framework** indicating overall studies;
- define the **problems** clearly;

    - Example: Traffic forecasting using matrix factorization models.

		![example](https://github.com/xinychen/transdim/blob/master/images/rolling_prediction_strategy.png)

> *Real experiment setting*: Observations with **0%, 20% and 40% fiber missing rates** during first **56 days are treated as stationary inputs**. Meanwhile, there are some rolling inputs for **forecasting traffic speed during last 5 days (from Monday to Friday)** in a rolling manner.

- describe the **core challenges** intuitively;
- list **main contributions** of these studies.

>**key words**

- data organization;
- temporal modeling.

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

	> Note: This model only considers *local information* from observations at the neighboring time slots of the missing values.

 - ***TRMF***: Temporal regularized matrix factorization. [[Matlab code is also available!](https://github.com/rofuyu/exp-trmf-nips16)]

 - ***BGCP***: Bayesian Gaussian CP decomposition. [[Matlab code is also available!](https://github.com/lijunsun/bgcp_imputation)]

 - ***BPMF***: Bayesian probabilistic matrix factorization.

 - ***HaLRTC***

 - ***LSTM***

 - ***CNN***

 - ***GAIN***: Generative Adversarial Imputation Nets.


# Selected references

- **Spatio-temporal forecasting**

  - Zheyi Pan, Yuxuan Liang, Junbo Zhang, Xiuwen Yi, Yong Yu, Yu Zheng, 2018. [*HyperST-Net: hypernetworks for spatio-temporal forecasting*](https://arxiv.org/pdf/1809.10889.pdf). arXiv.

  - Truc Viet Le, Richard Oentaryo, Siyuan Liu, Hoong Chuin Lau, 2017. [*Local Gaussian processes for efficient fine-grained traffic speed prediction*](https://arxiv.org/pdf/1708.08079.pdf). arXiv.

- **Principal component analysis (PCA)**

	- Li Qu, Li Li, Yi Zhang, Jianming Hu, 2009. [*PPCA-based missing data imputation for traffic flow volume: a systematical approach*](https://doi.org/10.1109/TITS.2009.2026312). IEEE Transactions on Intelligent Transportation Systems, 10(3): 512-522.

	- Li Li, Yuebiao Li, Zhiheng Li, 2013. [*Efficient missing data imputing for traffic flow by considering temporal and spatial dependence*](https://doi.org/10.1016/j.trc.2013.05.008). Transportation Research Part C: Emerging Technologies, 34: 108-120.

- **Guassian process (GP)**

	- Filipe Rodrigues, Kristian Henrickson, Francisco C. Pereira, 2018. [*Multi-output Gaussian processes for crowdsourced traffic data imputation*](https://doi.org/10.1109/TITS.2018.2817879). IEEE Transactions on Intelligent Transportation Systems, early access. [[Matlab code](http://fprodrigues.com/publications/multi-output-gaussian-processes-for-crowdsourced-traffic-data-imputation/)]

	- Nicolo Fusi, Rishit Sheth, Huseyn Melih Elibol, 2017. [*Probabilistic matrix factorization for automated machine learning*](https://arxiv.org/pdf/1705.05355.pdf). arXiv. [[Python code](https://github.com/elibol/amle)]

- **Matrix factorization**

  - Nikhil Rao, Hsiangfu Yu, Pradeep Ravikumar, Inderjit S Dhillon, 2015. [*Collaborative filtering with graph information: Consistency and scalable methods*](http://www.cs.utexas.edu/~rofuyu/papers/grmf-nips.pdf). Neural Information Processing Systems (*NIPS 2015*). [[Matlab code](http://bigdata.ices.utexas.edu/publication/collaborative-filtering-with-graph-information-consistency-and-scalable-methods/)]

  - Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon, 2016. [*Temporal regularized matrix factorization for high-dimensional time series prediction*](http://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf). 30th Conference on Neural Information Processing Systems (*NIPS 2016*), Barcelona, Spain. [[Matlab code](https://github.com/rofuyu/exp-trmf-nips16)]

  - Yongshun Gong, Zhibin Li, Jian Zhang, Wei Liu, Yu Zheng, Christina Kirsch, 2018. [*Network-wide crowd flow prediction of Sydney trains via customized online non-negative matrix factorization*](http://urban-computing.com/pdf/CIKM18-1121-Camera%20Ready.pdf). In The 27th ACM International Conference on Information and Knowledge Management (*CIKM 2018*), Torino, Italy.

- **Bayesian matrix/tensor factorization**

  - Ruslan Salakhutdinov, Andriy Mnih, 2008. [*Bayesian probabilistic matrix factorization using Markov chain Monte Carlo*](https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf). Proceedings of the 25th International Conference on Machine Learning (*ICML 2008*), Helsinki, Finland.

  - Nicolo Fusi, Rishit Sheth, Melih Huseyn Elibol, 2017. [*Probabilistic matrix factorization for automated machine learning*](https://arxiv.org/pdf/1705.05355.pdf). arXiv.

  - Liang Xiong, Xi Chen, Tzu-Kuo Huang, Jeff Schneider, Jaime G. Carbonell, 2010. [*Temporal collaborative filtering with Bayesian probabilistic tensor factorization*](https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Temporal_Collaborative_Filtering_With_Bayesian_Probabilidtic_Tensor_Factorization.pdf). Proceedings of the 2010 SIAM International Conference on Data Mining. SIAM, pp. 211-222.

  - Qibin Zhao, Liqing Zhang, Andrzej Cichocki, 2015. [*Bayesian CP factorization of incomplete tensors with automatic rank determination*](https://doi.org/10.1109/TPAMI.2015.2392756). IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(9): 1751-1763.

  - Qibin Zhao, Liqing Zhang, Andrzej Cichocki, 2015. [*Bayesian sparse Tucker models for dimension reduction and tensor completion*](https://arxiv.org/pdf/1505.02343.pdf). arXiv.

  - Piyush Rai, Yingjian Wang, Shengbo Guo, Gary Chen, David B. Dunsun,	Lawrence Carin, 2014. [*Scalable Bayesian low-rank decomposition of incomplete multiway tensors*](http://people.ee.duke.edu/~lcarin/mpgcp.pdf). Proceedings of the 31st International Conference on Machine Learning (*ICML 2014*), Beijing, China.

- **Low-rank tensor completion**

	- Ji Liu, Przemyslaw Musialski, Peter Wonka, Jieping Ye, 2013. [*Tensor completion for estimating missing values in visual data*](https://doi.org/10.1109/TPAMI.2012.39). IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(1): 208-220.

	- Bin Ran, Huachun Tan, Yuankai Wu, Peter J. Jin, 2016. [*Tensor based missing traffic data completion with spatial–temporal correlation*](https://doi.org/10.1016/j.physa.2015.09.105). Physica A: Statistical Mechanics and its Applications, 446: 54-63.

- **Generative Adversarial Nets (GAN)**

  - Brandon Amos, 2016. [*Image completion with deep learning in TensorFlow*](http://bamos.github.io/2016/08/09/deep-completion/). blog post. [[github](https://github.com/bamos/dcgan-completion.tensorflow)]

  - Jinsun Yoon, James Jordon, Mihaela van der Schaar, 2018. [*GAIN: missing data imputation using Generative Adversarial Nets*](http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf). Proceedings of the 35th International Conference on Machine Learning (*ICML 2018*), Stockholm, Sweden. [[supplementary materials](http://medianetlab.ee.ucla.edu/papers/ICML_GAIN_Supp.pdf)]

- **Tensor regression**

  - Guillaume Rabusseau, Hachem Kadri, 2016. [*Low-rank regression with tensor responses*](https://papers.nips.cc/paper/6302-low-rank-regression-with-tensor-responses.pdf). 30th Conference on Neural Information Processing Systems (*NIPS 2016*), Barcelona, Spain.

  - Rose Yu, Yan Liu, 2016. [*Learning from multiway data: simple and efficient tensor regression*](http://proceedings.mlr.press/v48/yu16.pdf). Proceedings of the 33rd International Conference on Machine Learning (*ICML 2016*), New York, NY, USA.

  - Masaaki Imaizumi, Kohei Hayashi, 2016. [*Doubly decomposing nonparametric tensor regression*](http://proceedings.mlr.press/v48/imaizumi16.pdf). Proceedings of the 33 rd International Conference on Machine Learning (*ICML 2016*), New York, NY, USA.

  - Rose Yu, Guangyu Li, Yan Liu, 2018. [*Tensor regression meets Gaussian processes*](http://proceedings.mlr.press/v84/yu18a/yu18a.pdf). Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (*AISTATS 2018*), Lanzarote, Spain.

- **Poisson matrix factorization**

  - Liangjie Hong, 2015. [*Poisson matrix factorization*](http://www.hongliangjie.com/2015/08/17/poisson-matrix-factorization/). blog post.

  - Prem Gopalan, Jake M. Hofman, David M. Blei, 2013. [*Scalable recommendation with Poisson factorization*](https://arxiv.org/pdf/1311.1704v1.pdf). arXiv.

  - Laurent Charlin, Rajesh Ranganath, James McInerney, David M. Blei, 2015. [*Dynamic Poisson factorization*](http://www.cs.toronto.edu/~lcharlin/papers/2015_CharlinRanganathMcInerneyBlei.pdf). arXiv.

# Publications

  - Xinyu Chen, Zhaocheng He, Jiawei Wang, 2018. [*Spatial-temporal traffic speed patterns discovery and incomplete data recovery via SVD-combined tensor decomposition*](https://doi.org/10.1016/j.trc.2017.10.023). Transportation Research Part C: Emerging Technologies, 86: 59-77.

  - Xinyu Chen, Zhaocheng He, Lijun Sun, 2018. *A Bayesian tensor decomposition approach for spatiotemporal traffic data imputation*. Transportation Research Part C: Emerging Technologies. (*under review*)

  >Please consider citing our papers if they help your research.

# Near-future plans

- First idea: present an interesting comparison between augmented Bayesian tensor factorization and GAN, and this will be an important illustration in the data imputation!

  - Xinyu Chen, Zhaocheng He, Yixian Chen, Yuhuan Lu, 2018. *Missing traffic data imputation and pattern discovery with an augmented Bayesian tensor factorization model*. Transportation Research Part C: Emerging Technologies. (*under review*)

	 - Contribution #1: propose an augmented model based on Bayesian tensor factorization and present robust imputation performance.

	 - Contribution #2: discover traffic patterns.

	 - Contribution #3: present comparison between GAN model (state-of-the-art model) and our proposed model.

Note: add an appendix for describing GAN model.

>most fantastic things: overcoming the **data noise** in GAN is a direction of further studies; placing **time series constriants** on discrete time slots is a direction of the future tensor factorization models; transforming from conventional tensor learning model to novel genrative models needs do more about data algebric structure.

- Second idea: Bayesian temporal matrix factorization + AR time series model (*a simple trick, but rather powerful!*)

- Third idea: Bayesian temporal matrix factorization + Gaussian process

  - Current work:
	 - factorize matrix into two sub-matrices (spatial and temporal factor matrices);
	 - then apply Gaussian process to temporal factor matrix for prediction;
	 - finally recover the data matrix by producting spatial factor matrix and temporal post-factor matrix.

- Fourth idea: GAN + matrix/tensor models

# License
This work is released under the MIT license.
