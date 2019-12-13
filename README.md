# transdim

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![repo size](https://img.shields.io/github/repo-size/xinychen/transdim.svg)](https://github.com/xinychen/transdim/archive/master.zip)
[![GitHub stars](https://img.shields.io/github/stars/xinychen/transdim.svg?logo=github&label=Stars&logoColor=white)](https://github.com/xinychen/transdim)

![logo](https://github.com/xinychen/transdim/blob/master/images/transdim_logo_large.png)

Machine learning models make important developments about spatiotemporal data modeling - like how to forecast near-future traffic states of road networks. But what happens when these models are built with incomplete data commonly collected in real-world systems?

About the Project
--------------

In the **transdim** (**trans**portation **d**ata **im**putation) project, we build machine learning models to help address some of the toughest challenges of spatiotemporal data modeling - from missing data imputation to time series prediction. The strategic aim of this project is **creating accurate and efficient solutions for spatiotemporal traffic data imputation and prediction tasks**.

In a hurry? Please check out our contents as follows.


Tasks and Challenges
--------------

> Missing data are there, whether we like them or not. The really interesting question is how to deal with incomplete data.

- **Missing data imputation** 🔥

  - Random missing (RM): Each sensor lost their observations at completely random. (★★★)
  - Non-random missing (NM): Each sensor lost their observations during several days. (★★★★)

<img src="https://github.com/xinychen/transdim/blob/master/images/framework.png" alt="drawing" width="800"/>

> **Example**: Tensor completion framework for multi-dimensional missing traffic data imputation.

- **Spatiotemporal prediction** 🔥
  - Forecasting without missing values. (★★★)
  - Forecasting with incomplete observations. (★★★★★)

<img src="https://github.com/xinychen/awesome-latex-drawing/blob/master/Framework/rolling_prediction.png" alt="drawing" width="500"/>

> **Example**: An illustration of single-step rolling prediction task under a matrix factorization framework.


Implementation
--------------

### Open data

In this repository, we have adapted the public data sets into our experiments. If you want to view these data sets, please run the following command in your Python console:

```python
import scipy.io

tensor = scipy.io.loadmat('../datasets/Guangzhou-data-set/tensor.mat')
tensor = tensor['tensor']
random_matrix = scipy.io.loadmat('../datasets/Guangzhou-data-set/random_matrix.mat')
random_matrix = random_matrix['random_matrix']
random_tensor = scipy.io.loadmat('../datasets/Guangzhou-data-set/random_tensor.mat')
random_tensor = random_tensor['random_tensor']
```

If you want to view the original data, please check out the following links:

- **Gdata**: [Guangzhou urban traffic speed data set](https://doi.org/10.5281/zenodo.1205228).
- **Bdata**: [Birmingham parking data set](https://archive.ics.uci.edu/ml/datasets/Parking+Birmingham).
- **Hdata**: [Hangzhou metro passenger flow data set](https://doi.org/10.5281/zenodo.3145403).
- **Ndata**: [NYC taxi data set](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- **Sdata**: [Seattle freeway traffic speed data set](https://github.com/zhiyongc/Seattle-Loop-Data).

### Model implementation

In our experiments, we have implemented the machine learning models mainly on `Numpy`, and written these Python codes with **Jupyter Notebook**. So, if you want to evaluate these models, please download and run these notebooks directly (prerequisite: **download the data sets** before evaluation).

| Task     | Jupyter Notebook link | Gdata | Bdata | Hdata | Sdata | Ndata |
| :---:    | :------------- | :------------: | :------------:  | :-----------: | :-----------: | :-------:|
|Missing Data Imputation| [**BTMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-BTMF.ipynb)           | ✅     | ✅     | ✅    | ✅ | 🔶 |
|          | [**BayesTRMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-BayesTRMF.ipynb) | ✅     | ✅     | ✅    | ✅ | 🔶 |
|          | [**TRMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-TRMF.ipynb) | ✅     | ✅     | ✅    | ✅ | 🔶 |
|          | [**BPMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-BPMF.ipynb) | ✅     | ✅     | ✅    | ✅ | 🔶 |
|          | [**BGCP**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-BGCP.ipynb) | ✅     | ✅     | ✅    | ✅ | ✅ |
|          | [**HaLRTC**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-HaLRTC.ipynb) | ✅     | ✅     | ✅    | ✅ | 🔶 |
|          | [**TF-ALS**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-TF-ALS.ipynb) | ✅     | ✅     | ✅    | ✅ | ✅ |
|          | [**BTTF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-BTTF.ipynb) | 🔶     | 🔶     | 🔶    | 🔶 | ✅ |
|          | [**BayesTRTF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-BayesTRTF.ipynb) | 🔶     | 🔶     | 🔶    | 🔶 | ✅ |
|          | [**BPTF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Imputation-BPTF.ipynb) | 🔶     | 🔶     | 🔶    | 🔶 | ✅ |
|Single-Step Prediction| [**BTMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-ST-Online-BTMF.ipynb) | ✅     | ✅     | ✅    | ✅ | 🔶 |
|          | [**BayesTRMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-ST-Online-BayesTRMF.ipynb) | ✅     | ✅     | ✅    | ✅ | 🔶 |
|          | [**TRMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-ST-Online-TRMF.ipynb) | ✅     | ✅     | ✅    | ✅ | 🔶 |
|          | [**BTTF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-ST-Online-BTTF.ipynb) | 🔶     | 🔶     | 🔶    | 🔶 | ✅ |
|          | [**BayesTRTF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-ST-Online-BayesTRTF.ipynb) | 🔶     | 🔶     | 🔶    | 🔶 | ✅ |
|          | [**TRTF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-ST-Online-TRTF.ipynb) | 🔶     | 🔶     | 🔶    | 🔶 | ✅ |
|Multi-Step Prediction| [**BTMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-Multi-BTMF.ipynb) | ✅     | ✅     | ✅    | ✅ | 🔶 |
|           | [**BayesTRMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-Multi-BayesTRMF.ipynb) | ✅     | ✅     | ✅    | ✅ | 🔶 |
|           | [**TRMF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-Multi-TRMF.ipynb) | ✅     | ✅     | ✅    | ✅ | 🔶 |
|          | [**BTTF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-Multi-BTTF.ipynb) | 🔶     | 🔶     | 🔶    | 🔶 | ✅ |
|          | [**BayesTRTF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-Multi-BayesTRTF.ipynb) | 🔶     | 🔶     | 🔶    | 🔶 | ✅ |
|          | [**TRTF**](https://nbviewer.jupyter.org/github/xinychen/transdim/blob/master/experiments/Prediction-Multi-TRTF.ipynb) | 🔶     | 🔶     | 🔶    | 🔶 | ✅ |


* ✅ — Cover
* 🔶 — Does not cover
* 🚧 — Under development

### Imputation/Prediction performance

- **Imputation example**

![example](https://github.com/xinychen/transdim/blob/master/images/estimated_series1.png)
  *(a) Time series of actual and estimated speed within two weeks from August 1 to 14.*

![example](https://github.com/xinychen/transdim/blob/master/images/estimated_series2.png)
  *(b) Time series of actual and estimated speed within two weeks from September 12 to 25.*

> The imputation performance of BGCP (CP rank r=15 and missing rate α=30%) under the fiber missing scenario with third-order tensor representation, where the estimated result of road segment #1 is selected as an example. In the both two panels, red rectangles represent fiber missing (i.e., speed observations are lost in a whole day).

- **Prediction example**

![example](https://github.com/xinychen/transdim/blob/master/images/prediction_hangzhou.png)

![example](https://github.com/xinychen/transdim/blob/master/images/prediction_nyc_heatmap.png)

![example](https://github.com/xinychen/transdim/blob/master/images/prediction_nyc.png)

References
--------------

- ### **Spatiotemporal forecasting**

  - Yuyang Wang, Alex Smola, Danielle C. Maddix, Jan Gasthaus, Dean Foster, Tim Januschowski, 2019. [*Deep Factors for Forecasting*](https://arxiv.org/pdf/1905.12417.pdf). ICML 2019. (★★★★★)
  
  - Danielle C. Maddix, Yuyang Wang, Alex Smola, 2018. [*Deep Factors with Gaussian Processes for Forecasting*](https://arxiv.org/pdf/1812.00098.pdf). arXiv.
  
  - Syama Sundar Rangapuram, Matthias Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, Tim Januschowski, 2018. [*Deep State Space Models for Time Series Forecasting*](http://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf?nsukey=CoqM71tHuwiqTTDwkPRsu40QKdc%2BlY%2FWWNEiuEJaWd6sw7dUXlAT3mU122dvYrMHOYfANParRFLBDLgraANFjHdggEDRxbT9Kk2Mam6nZfVkQT4E2cRCMULUTPTFmjCeuBKHjDSSEs6L4Ci7JuyRqB2ojJnBKDD%2FkcAgdhwhJQa%2BLZ5owi4oVwnQrgWX5pTr0vTORENMdC59F4mqtOQENA%3D%3D). NeurIPS 2018.

  - San Gultekin, John Paisley, 2019. [*Online Forecasting Matrix Factorization*](https://ieeexplore.ieee.org/document/8590686/). IEEE Transactions on Signal Processing, 67(5): 1223-1236. [[Python code](https://github.com/chloemnge/online_learning)]

  - Zheyi Pan, Yuxuan Liang, Junbo Zhang, Xiuwen Yi, Yong Yu, Yu Zheng, 2018. [*HyperST-Net: hypernetworks for spatio-temporal forecasting*](https://arxiv.org/pdf/1809.10889.pdf). arXiv.

  - Truc Viet Le, Richard Oentaryo, Siyuan Liu, Hoong Chuin Lau, 2017. [*Local Gaussian processes for efficient fine-grained traffic speed prediction*](https://arxiv.org/pdf/1708.08079.pdf). arXiv.

  - Yaguang Li, Cyrus Shahabi, 2018. [*A brief overview of machine learning methods for short-term traffic forecasting and future directions*](https://doi.org/10.1145/3231541.3231544). ACM SIGSPATIAL, 10(1): 3-9.

  - Bing Yu, Haoteng Yin, Zhanxing Zhu, 2017. [*Spatio-temporal graph convolutional networks: a deep learning framework for traffic forecasting*](https://arxiv.org/pdf/1709.04875.pdf). arXiv. ([appear in IJCAI 2018](https://www.ijcai.org/proceedings/2018/0505.pdf))

  - Feras A. Saad, Vikash K. Mansinghka, 2018. [*Temporally-reweighted Chinese Restaurant Process mixtures for clustering, imputing, and forecasting multivariate time series*](http://proceedings.mlr.press/v84/saad18a/saad18a.pdf). Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (*AISTATS 2018*), Lanzarote, Spain. PMLR: Volume 84.

  - Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu, 2018. [*Recurrent neural networks for multivariate time series with missing values*](https://doi.org/10.1038/s41598-018-24271-9). Scientific Reports, 8(6085).

  - Zhengping Che, Sanjay Purushotham, Guangyu Li, Bo Jiang, Yan Liu, 2018. [*Hierarchical deep generative models for multi-rate multivariate time series*](http://proceedings.mlr.press/v80/che18a/che18a.pdf). Proceedings of the 35th International Conference on Machine Learning (*ICML 2018*), PMLR 80:784-793, 2018.

  - Chuxu Zhang, Dongjin Song, Yuncong Chen, Xinyang Feng, Cristian Lumezanu, Wei Cheng, Jingchao Ni, Bo Zong, Haifeng Chen, Nitesh V. Chawla, 2018. [*A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data*](https://arxiv.org/abs/1811.08055). arXiv.
  - Wang, X., Chen, C., Min, Y., He, J., Yang, B., Zhang, Y., 2018. [*Efficient metropolitan traffic prediction based on graph recurrent neural network*](https://arxiv.org/pdf/1811.00740.pdf). arXiv.

  - Peiguang Jing, Yuting Su, Xiao Jin, Chengqian Zhang, 2018. [*High-order temporal correlation model learning for time-series prediction*](https://doi.org/10.1109/TCYB.2018.2832085). IEEE Transactions on Cybernetics, early access.

  - Oren Anava, Elad Hazan, Assaf Zeevi, 2015. [*Online time series prediction with missing data*](http://proceedings.mlr.press/v37/anava15.pdf). Proceedings of the 32nd International Conference on Machine Learning (*ICML 2015*), 37: 2191-2199.

  - Shanshan Feng, Gao Cong, Bo An, Yeow Meng Chee, 2017. [*POI2Vec: Geographical latent representation for predicting future visitors*](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14902/13749). Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence (*AAAI 2017*).

  - Yasuko Matsubara, Yasushi Sakurai, Christos Faloutsos, Tomoharu Iwata, Masatoshi Yoshikawa, 2012. [*Fast mining and forecasting of complex time-stamped events*](http://www.cs.kumamoto-u.ac.jp/~yasuko/PUBLICATIONS/kdd12-trimine.pdf). Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining (*KDD 2012*).

  - Yasuko Matsubara, Yasushi Sakurai, Willem G. van Panhuis, Christos Faloutsos, 2014. [*FUNNEL: automatic mining of spatially coevolving epidemics*](http://www.cs.cmu.edu/~christos/PUBLICATIONS/14-kdd-funnel.pdf). Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (*KDD 2014*).

  - Koh Takeuchi, Hisashi Kashima, Naonori Ueda, 2017. [*Autoregressive tensor factorization for spatio-temporal predictions*](https://doi.org/10.1109/ICDM.2017.146). 2017 IEEE International Conference on Data Mining (*ICDM 2017*).

  - Shun-Yao Shih, Fan-Keng Sun, Hung-yi Lee, 2018. [*Temporal pattern attention for multivariate time series forecasting*](https://arxiv.org/pdf/1809.04206v2.pdf). arXiv.
  
  - Dingxiong Deng, Cyrus Shahabi, Ugur Demiryurek, Linhong Zhu, Rose Yu, Yan Liu, 2016. [*Latent space model for road networks to predict time-varying traffic*](http://roseyu.com/Papers/kdd2016.pdf). Proceedings of the 22rd ACM SIGKDD international conference on Knowledge discovery and data mining (*KDD 2016*).

- ### **Principal component analysis**

  - Shigeyuki Oba, Masa-aki Sato, Ichiro Takemasa, Morito Monden, Ken-ichi Matsubara, Shin Ishii, 2003. [*A Bayesian missing value estimation method for gene expression profile data*](https://doi.org/10.1093/bioinformatics/btg287). Bioinformatics, 19: 2088-2096. [[Matlab code](http://ishiilab.jp/member/oba/tools/BPCAFill.html)]

  - Li Qu, Li Li, Yi Zhang, Jianming Hu, 2009. [*PPCA-based missing data imputation for traffic flow volume: a systematical approach*](https://doi.org/10.1109/TITS.2009.2026312). IEEE Transactions on Intelligent Transportation Systems, 10(3): 512-522.

  - Li Li, Yuebiao Li, Zhiheng Li, 2013. [*Efficient missing data imputing for traffic flow by considering temporal and spatial dependence*](https://doi.org/10.1016/j.trc.2013.05.008). Transportation Research Part C: Emerging Technologies, 34: 108-120.

- ### **Guassian process**

  - Michalis K. Titsias, Magnus Rattray, Neil D. Lawrence, 2009. [*Markov chain Monte Carlo algorithms for Gaussian processes*](http://www2.aueb.gr/users/mtitsias/papers/ILDMChapter09.pdf), Chapter.

  - Filipe Rodrigues, Kristian Henrickson, Francisco C. Pereira, 2018. [*Multi-output Gaussian processes for crowdsourced traffic data imputation*](https://doi.org/10.1109/TITS.2018.2817879). IEEE Transactions on Intelligent Transportation Systems, early access. [[Matlab code](http://fprodrigues.com/publications/multi-output-gaussian-processes-for-crowdsourced-traffic-data-imputation/)]

  - Nicolo Fusi, Rishit Sheth, Huseyn Melih Elibol, 2017. [*Probabilistic matrix factorization for automated machine learning*](https://arxiv.org/pdf/1705.05355.pdf). arXiv. [[Python code](https://github.com/elibol/amle)]

  - Tinghui Zhou, Hanhuai Shan, Arindam Banerjee, Guillermo Sapiro, 2012. [*Kernelized probabilistic matrix factorization: exploiting graphs and side information*](http://www.cs.cmu.edu/~tinghuiz/papers/sdm12_kpmf.pdf). [[slide](http://people.ee.duke.edu/~lcarin/Jorge6.4.2012.pdf)]

  - John Bradshaw, Alexander G. de G. Matthews, Zoubin Ghahramani, 2017. [*Adversarial examples, uncertainty, and transfer testing robustness in Gaussian process hybrid deep networks*](https://arxiv.org/pdf/1707.02476.pdf). arXiv.

  - David Salinas, Michael Bohlke-Schneider, Laurent Callot, Roberto Medico, Jan Gasthaus, 2019. [*High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes*](https://arxiv.org/pdf/1910.03002.pdf). arXiv. (★★★★)

- ### **Matrix factorization**

  - Nikhil Rao, Hsiangfu Yu, Pradeep Ravikumar, Inderjit S Dhillon, 2015. [*Collaborative filtering with graph information: Consistency and scalable methods*](http://www.cs.utexas.edu/~rofuyu/papers/grmf-nips.pdf). Neural Information Processing Systems (*NIPS 2015*). [[Matlab code](http://bigdata.ices.utexas.edu/publication/collaborative-filtering-with-graph-information-consistency-and-scalable-methods/)]

  - Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon, 2016. [*Temporal regularized matrix factorization for high-dimensional time series prediction*](http://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf). 30th Conference on Neural Information Processing Systems (*NIPS 2016*), Barcelona, Spain. [[Matlab code](https://github.com/rofuyu/exp-trmf-nips16)]

  - Yongshun Gong, Zhibin Li, Jian Zhang, Wei Liu, Yu Zheng, Christina Kirsch, 2018. [*Network-wide crowd flow prediction of Sydney trains via customized online non-negative matrix factorization*](http://urban-computing.com/pdf/CIKM18-1121-Camera%20Ready.pdf). In The 27th ACM International Conference on Information and Knowledge Management (*CIKM 2018*), Torino, Italy.

- ### **Bayesian matrix and tensor factorization**

  - Ruslan Salakhutdinov, Andriy Mnih, 2008. [*Bayesian probabilistic matrix factorization using Markov chain Monte Carlo*](https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf). Proceedings of the 25th International Conference on Machine Learning (*ICML 2008*), Helsinki, Finland. [[Matlab code (official)](https://www.cs.toronto.edu/~rsalakhu/BPMF.html)] [[Python code](https://github.com/LoryPack/BPMF)] [[Julia and C++ code](https://github.com/ExaScience/bpmf)] [[Julia code](https://github.com/RottenFruits/BPMF.jl)]

  - Neil D. Lawrence, Raquel Urtasun, 2009. [*Non-linear Matrix Factorization with Gaussian Processes*](http://people.ee.duke.edu/~lcarin/MatrixFactorization.pdf). ICML 2009. (★★★★★)

  - Ilya Sutskever, Ruslan Salakhutdinov, Joshua B. Tenenbaum, 2009. [*Modelling relational data using Bayesian clustered tensor factorization*](https://ece.duke.edu/~lcarin/pmfcrp.pdf). NIPS 2009.

  - kan Saha, Vikas Sindhwani, 2012. [*Learning evolving and emerging topics in social media: A dynamic NMF approach with temporal regularization*](http://people.cs.uchicago.edu/~ankans/Papers/wsdm227-saha.pdf). WSDM 2012. (★★★★)

  - Nicolo Fusi, Rishit Sheth, Melih Huseyn Elibol, 2017. [*Probabilistic matrix factorization for automated machine learning*](https://arxiv.org/pdf/1705.05355.pdf). arXiv.

  - Liang Xiong, Xi Chen, Tzu-Kuo Huang, Jeff Schneider, Jaime G. Carbonell, 2010. [*Temporal collaborative filtering with Bayesian probabilistic tensor factorization*](https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Temporal_Collaborative_Filtering_With_Bayesian_Probabilidtic_Tensor_Factorization.pdf). Proceedings of the 2010 SIAM International Conference on Data Mining. SIAM, pp. 211-222.

  - Qibin Zhao, Liqing Zhang, Andrzej Cichocki, 2015. [*Bayesian CP factorization of incomplete tensors with automatic rank determination*](https://doi.org/10.1109/TPAMI.2015.2392756). IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(9): 1751-1763.

  - Qibin Zhao, Liqing Zhang, Andrzej Cichocki, 2015. [*Bayesian sparse Tucker models for dimension reduction and tensor completion*](https://arxiv.org/pdf/1505.02343.pdf). arXiv.

  - Piyush Rai, Yingjian Wang, Shengbo Guo, Gary Chen, David B. Dunsun,	Lawrence Carin, 2014. [*Scalable Bayesian low-rank decomposition of incomplete multiway tensors*](http://people.ee.duke.edu/~lcarin/mpgcp.pdf). Proceedings of the 31st International Conference on Machine Learning (*ICML 2014*), Beijing, China.

  - Ömer Deniz Akyildiz, Theodoros Damoulas, Mark F. J. Steel, 2019. [*Probabilistic sequential matrix factorization*](https://arxiv.org/pdf/1910.03906.pdf). arXiv. (★★★★★)
  
- ### **Matrix completion on graphs**

  - Vassilis Kalofolias, Xavier Bresson, Michael Bronstein, Pierre Vandergheynst, 2014. [*Matrix completion on graphs*](https://arxiv.org/abs/1408.1717). arXiv. (appear in NIPS 2014)

  - Rianne van den Berg, Thomas N. Kipf, Max Welling, 2018. [*Graph convolutional matrix completion*](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_32.pdf). Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (*KDD 2018*), London, UK.

  - Federico Monti, Michael M. Bronstein, Xavier Bresson, 2017. [*Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks*](https://papers.nips.cc/paper/6960-geometric-matrix-completion-with-recurrent-multi-graph-neural-networks.pdf). NIPS 2017.

  - Tianyang Han, Kentaro Wada and Takashi Oguchi, 2019. [*Large-scale traffic data imputation using matrix completion on graphs*](http://doi.org/10.1109/ITSC.2019.8917365). IEEE Intelligent Transportation Systems Conference (ITSC), Auckland, New Zealand, 2019, pp. 2252-2258.

- ### **Low-rank tensor completion**

  - Ji Liu, Przemyslaw Musialski, Peter Wonka, Jieping Ye, 2013. [*Tensor completion for estimating missing values in visual data*](https://doi.org/10.1109/TPAMI.2012.39). IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(1): 208-220.

  - Bin Ran, Huachun Tan, Yuankai Wu, Peter J. Jin, 2016. [*Tensor based missing traffic data completion with spatial–temporal correlation*](https://doi.org/10.1016/j.physa.2015.09.105). Physica A: Statistical Mechanics and its Applications, 446: 54-63.

- ### **Generative Adversarial Nets**

  - Brandon Amos, 2016. [*Image completion with deep learning in TensorFlow*](http://bamos.github.io/2016/08/09/deep-completion/). blog post. [[github](https://github.com/bamos/dcgan-completion.tensorflow)]

  - Jinsun Yoon, James Jordon, Mihaela van der Schaar, 2018. [*GAIN: missing data imputation using Generative Adversarial Nets*](http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf). Proceedings of the 35th International Conference on Machine Learning (*ICML 2018*), Stockholm, Sweden. [[supplementary materials](http://medianetlab.ee.ucla.edu/papers/ICML_GAIN_Supp.pdf)] [[Python code](https://github.com/jsyoon0823/GAIN)]

  - Ian Goodfellow, 2016. [*NIPS 2016 tutorial: Generative Adversarial Networks*](https://arxiv.org/abs/1701.00160).

  - Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, Georg Langs, 2017. [*Unsupervised anomaly detection with generative adversarial networks to guide marker discovery*](https://arxiv.org/abs/1703.05921). arXiv.

  - Yonghong Luo, Xiangrui Cai, Ying Zhang, Jun Xu, Xiaojie Yuan, 2018. [*Multivariate time series imputation with generative adversarial networks*](https://papers.nips.cc/paper/7432-multivariate-time-series-imputation-with-generative-adversarial-networks). 32nd Conference on Neural Information Processing Systems (*NeurIPS 2018*), Montréal, Canada. [[Python code](https://github.com/Luoyonghong/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks)]

- ### **Variational Autoencoder**

  - Zhiwei Deng, Rajitha Navarathna, Peter Carr, Stephan Mandt, Yisong Yue, 2017. [*Factorized variational autoencoders for modeling audience reactions to movies*](http://openaccess.thecvf.com/content_cvpr_2017/papers/Deng_Factorized_Variational_Autoencoders_CVPR_2017_paper.pdf). 2017 IEEE Conference on Computer Vision and Pattern Recognition (*CVPR 2017*), Honolulu, HI, USA.

  - [*Graph autoencoder - GitHub*](https://github.com/tkipf/gae).

  - Haowen Xu, Wenxiao Chen, Nengwen Zhao, Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, Dan Pei, Yang Feng, Jie Chen, Zhaogang Wang, Honglin Qiao, 2018. [*Unsupervised anomaly detection via variational auto-encoder for seasonal KPIs in web applications*](https://arxiv.org/pdf/1802.03903.pdf). *WWW 2018*.

  - John T. McCoy, Steve Kroon, Lidia Auret, 2018. [*Variational Autoencoders for missing data imputation with application to a simulated milling circuit*](https://doi.org/10.1016/j.ifacol.2018.09.406). IFAC-PapersOnLine, 51(21): 141-146. [[Python code](https://github.com/ProcessMonitoringStellenboschUniversity/IFAC-VAE-Imputation)] [[VAE demo](https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/vae/vae_demo-2D.ipynb)]

  - Pierre-Alexandre Mattei, Jes Frellsen, 2018. [missingIWAE: Deep generative modelling and imputation of incomplete data](http://bayesiandeeplearning.org/2018/papers/100.pdf). Third workshop on Bayesian Deep Learning (*NeurIPS 2018*), Montréal, Canada. [[related slide](https://ai.ku.dk/ai-seminar-series/ai-seminar_jes-frellsen.pdf)]

- ### **Tensor regression**

  - Guillaume Rabusseau, Hachem Kadri, 2016. [*Low-rank regression with tensor responses*](https://papers.nips.cc/paper/6302-low-rank-regression-with-tensor-responses.pdf). 30th Conference on Neural Information Processing Systems (*NIPS 2016*), Barcelona, Spain.

  - Rose Yu, Yan Liu, 2016. [*Learning from multiway data: simple and efficient tensor regression*](http://proceedings.mlr.press/v48/yu16.pdf). Proceedings of the 33rd International Conference on Machine Learning (*ICML 2016*), New York, NY, USA.

  - Masaaki Imaizumi, Kohei Hayashi, 2016. [*Doubly decomposing nonparametric tensor regression*](http://proceedings.mlr.press/v48/imaizumi16.pdf). Proceedings of the 33 rd International Conference on Machine Learning (*ICML 2016*), New York, NY, USA.

  - Rose Yu, Guangyu Li, Yan Liu, 2018. [*Tensor regression meets Gaussian processes*](http://proceedings.mlr.press/v84/yu18a/yu18a.pdf). Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (*AISTATS 2018*), Lanzarote, Spain. [[Matlab code](https://github.com/yuqirose/MultilinearGP)]

  - Lifang He, Kun Chen, Wanwan Xu, Jiayu Zhou, Fei Wang, 2018. [*Boosted sparse and low-rank tensor regression*](https://papers.nips.cc/paper/7379-boosted-sparse-and-low-rank-tensor-regression.pdf). 32nd Conference on Neural Information Processing Systems (*NeurIPS 2018*), Montréal, Canada.

- ### **Poisson matrix factorization**

  - Liangjie Hong, 2015. [*Poisson matrix factorization*](http://www.hongliangjie.com/2015/08/17/poisson-matrix-factorization/). blog post.

  - Ali Taylan Cemgil, 2009. [*Bayesian inference for nonnegative matrix factorisation models*](http://downloads.hindawi.com/journals/cin/2009/785152.pdf). Computational intelligence and neuroscience.

  - Prem Gopalan, Jake M. Hofman, David M. Blei, 2015. [*Scalable recommendation with hierarchical poisson factorization*](http://www.cs.columbia.edu/~blei/papers/GopalanHofmanBlei2015.pdf). In UAI, 326-335. [[C++ code](https://github.com/premgopalan/hgaprec)]

  - Laurent Charlin, Rajesh Ranganath, James Mclnerney, 2015. [*Dynamic Poisson factorization*](http://www.cs.toronto.edu/~lcharlin/papers/2015_CharlinRanganathMcInerneyBlei.pdf). Proceedings of the 9th ACM Conference on Recommender Systems (*RecSys 2015*), Vienna, Italy. [[C++ code](https://github.com/blei-lab/DynamicPoissonFactorization)]

  - Seyed Abbas Hosseini, Keivan Alizadeh, Ali Khodadadi, Ali Arabzadeh, Mehrdad Farajtabar, Hongyuan Zha, Hamid R. Rabiee, 2017. [*Recurrent Poisson factorization for temporal recommendation*](https://dl.acm.org/citation.cfm?doid=3097983.3098197). Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (*KDD 2017*), Halifax, Nova Scotia Canada. [[Matlab code](https://github.com/AHosseini/RPF)]

  - Aaron Schein, Scott W. Linderman, Mingyuan Zhou, David M. Blei, Hanna Wallach, 2019. [*Poisson-Randomized Gamma Dynamical Systems*](https://arxiv.org/pdf/1910.12991.pdf). arXiv. (★★★★★)

- ### **Graph signal processing**

  - Arman Hasanzadeh, Xi Liu, Nick Duffield, Krishna R. Narayanan, Byron Chigoy, 2017. [*A graph signal processing approach for real-time traffic prediction in transportation networks*](https://arxiv.org/pdf/1711.06954.pdf). arXiv.

  - Antonio Ortega, Pascal Frossard, Jelena Kovačević, José M. F. Moura, Pierre Vandergheynst, 2018. [*Graph signal processing: overview, challenges, and applications*](https://doi.org/10.1109/JPROC.2018.2820126). Proceedings of the IEEE, 106(5): 808-828. [[slide](https://www.seas.upenn.edu/~gsp16/ortega.pdf)]

- ### **Graph neural network**

  - [*How to do Deep Learning on Graphs with Graph Convolutional Networks (Part 1: A High-Level Introduction to Graph Convolutional Networks)*](https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780). blog post.

  - [*Structured deep models: Deep learning on graphs and beyond*](https://tkipf.github.io/misc/SlidesCambridge.pdf). slide.

  - [*gcn: Implementation of Graph Convolutional Networks in TensorFlow*](https://github.com/tkipf/gcn). GitHub project.

  - [*gated-graph-neural-network-samples: Sample Code for Gated Graph Neural Networks*](https://github.com/Microsoft/gated-graph-neural-network-samples). GitHub project.

  - Xu Geng, Yaguang Li, Leye Wang, Lingyu Zhang, Qiang Yang, Jieping Ye, Yan Liu, 2019. [*Spatiotemporal multi-graph convolution network for ride-hailing demand forecasting*](http://www-scf.usc.edu/~yaguang/papers/aaai19_multi_graph_convolution.pdf). *AAAI 2019*.

  - Menglin Wang, Baisheng Lai, Zhongming Jin, Yufeng Lin, Xiaojia Gong, Jiangqiang Huang, Xiansheng Hua, 2018. [*Dynamic spatio-temporal graph-based CNNs for traffic prediction*](https://arxiv.org/pdf/1812.02019.pdf). arXiv.

- ### **Missing data imputation**

  -  Daniel J. Stekhoven, Peter Bühlmann, 2012. [*MissForest—non-parametric missing value imputation for mixed-type data*](https://doi.org/10.1093/bioinformatics/btr597). Bioinformatics, 28(1): 112–118. [[missingpy - PyPI](https://pypi.org/project/missingpy/)] or [[missingpy - GitHub](https://github.com/epsilon-machine/missingpy)]

  - [fancyimpute](https://github.com/iskandr/fancyimpute): A variety of matrix completion and imputation algorithms implemented in Python. [[homepage](https://pypi.org/project/fancyimpute/)]

  - Dimitris Bertsimas, Colin Pawlowski, Ying Daisy Zhuo, 2018. [*From predictive methods to missing data imputation: An optimization approach*](http://jmlr.org/papers/v18/17-073.html). Journal of Machine Learning Research, 18(196): 1-39.

  - Wei Cao, Dong Wang, Jian Li, Hao Zhou, Yitan Li, Lei Li, 2018. [*BRITS: Bidirectional Recurrent Imputation for Time Series*](https://papers.nips.cc/paper/7911-brits-bidirectional-recurrent-imputation-for-time-series.pdf). 32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montréal, Canada. [[Python code](https://github.com/caow13/BRITS)]

Our Publications
--------------

- Xinyu Chen, Lijun Sun (2019). **Bayesian temporal factorization for multidimensional time series prediction**. arxiv. 1910.06366. [[preprint](https://arxiv.org/abs/1910.06366)] [[slide](https://xinychen.github.io/paper/Bayesian-temporal-factorization-slide.pdf)] [[data & Python code](https://github.com/xinychen/transdim)]

- Xinyu Chen, Zhaocheng He, Yixian Chen, Yuhuan Lu, Jiawei Wang (2019). **Missing traffic data imputation and pattern discovery with a Bayesian augmented tensor factorization model**. Transportation Research Part C: Emerging Technologies, 104: 66-77. [[preprint](https://xinychen.github.io/paper/BATF.pdf)] [[doi](https://doi.org/10.1016/j.trc.2019.03.003)] [[slide](https://doi.org/10.5281/zenodo.2632552)] [[data](http://doi.org/10.5281/zenodo.1205229)] [[Matlab code](https://github.com/sysuits/BATF)]

- Xinyu Chen, Zhaocheng He, Lijun Sun (2019). **A Bayesian tensor decomposition approach for spatiotemporal traffic data imputation**. Transportation Research Part C: Emerging Technologies, 98: 73-84. [[preprint](https://www.researchgate.net/publication/329177786_A_Bayesian_tensor_decomposition_approach_for_spatiotemporal_traffic_data_imputation)] [[doi](https://doi.org/10.1016/j.trc.2018.11.003)] [[data](http://doi.org/10.5281/zenodo.1205229)] [[Matlab code](https://github.com/lijunsun/bgcp_imputation)] [[Python code](https://github.com/xinychen/transdim/blob/master/experiments/Imputation-BGCP.ipynb)]

- Xinyu Chen, Zhaocheng He, Jiawei Wang (2018). **Spatial-temporal traffic speed patterns discovery and incomplete data recovery via SVD-combined tensor decomposition**. Transportation Research Part C: Emerging Technologies, 86: 59-77. [[doi](http://doi.org/10.1016/j.trc.2017.10.023)] [[data](http://doi.org/10.5281/zenodo.1205229)]

  >This project is from our papers, please consider citing our papers if they help your research.

Collaborators
--------------

<table>
  <tr>
    <td align="center"><a href="https://github.com/xinychen"><img src="https://github.com/xinychen.png?size=80" width="80px;" alt="Xinyu Chen"/><br /><sub><b>Xinyu Chen</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=xinychen" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Vadermit"><img src="https://github.com/Vadermit.png?size=80" width="80px;" alt="Jinming Yang"/><br /><sub><b>Jinming Yang</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=Vadermit" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/yxnchen"><img src="https://github.com/yxnchen.png?size=80" width="80px;" alt="Yixian Chen"/><br /><sub><b>Yixian Chen</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=yxnchen" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/lijunsun"><img src="https://github.com/lijunsun.png?size=80" width="80px;" alt="Lijun Sun"/><br /><sub><b>Lijun Sun</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=lijunsun" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/HanTY"><img src="https://github.com/HanTY.png?size=80" width="80px;" alt="Tianyang Han"/><br /><sub><b>Tianyang Han</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=HanTY" title="Code">💻</a></td>
<!--   </tr>
  <tr>
    <td align="center"><a href="https://github.com/xxxx"><img src="https://github.com/xxxx.png?size=100" width="100px;" alt="xxxx"/><br /><sub><b>xxxx</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=xxxx" title="Code">💻</a></td> -->
  </tr>
</table>

- **Principal Investigator (PI)**

<table>
  <tr>
    <td align="center"><a href="https://github.com/lijunsun"><img src="https://github.com/lijunsun.png?size=80" width="80px;" alt="Lijun Sun"/><br /><sub><b>Lijun Sun</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=lijunsun" title="Code">💻</a></td>
  </tr>
</table>

See the list of [contributors](https://github.com/xinychen/transdim/graphs/contributors) who participated in this project.


Acknowledgements
--------------


If you have any suggestion, please feel free to contact Xinyu Chen (email: chenxy346@mail2.sysu.edu.cn) and send your suggestions. We would like to thank everyone who has helped this project in any way.

> Recommended email subject: Suggestions on transdim from [+ your name].



License
--------------

This work is released under the MIT license.
