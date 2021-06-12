# large-imputer

> This folder includes our latest **imputer** on large-scale spatiotemporal traffic data.

## Highlights

- Linear unitary transform.
- Temporal variation using quadratic time series autoregression.
- Large-scale spatiotemporal imputation problem.

## Data Sets

In this repository, we have adapted some publicly available data sets into our experiments. Thse data are summarized as follows,

- **PeMS-4W data set**
  - This data set contains freeway traffic speed collected from 11160 traffic measurement sensors over 4 weeks (the first 4 weeks in the year of 2018) with a 5-min time resolution (288 time intervals per day) in California, USA. It can be arranged in a matrix of size 11160 x 8064 or a tensor of size 11160 x 288 x 28 according to the spatial and temporal dimensions. Note that this data set contains about 90 million observations.

- **PeMS-8W data set**
  - This data set contains freeway traffic speed collected from 11160 traffic measurement sensors over 8 weeks (the first 8 weeks in the year of 2018) with a 5-min time resolution (288 time intervals per day) in California, USA. It can be arranged in a matrix of size 11160 x 16128 or a tensor of size 11160 x 288 x 56 according to the spatial and temporal dimensions. Note that this data set contains about 180 million observations.
  
In particular, if you are interested in large-scale traffic data, we recommend **PeMS-4W/8W/12W** and [UTD19](https://utd19.ethz.ch/index.html). For PeMS data, you can download the data from [Zenodo](https://doi.org/10.5281/zenodo.3939792) and place them at the folder of datasets (data path example: `../datasets/California-data-set/pems-4w.csv`). Then you can use `Pandas` to open data:

```python
import pandas as pd

data = pd.read_csv('../datasets/California-data-set/pems-4w.csv', header = None)
```

For model evaluation, we mask certain entries of the "observed" data as missing values and then perform imputation for these "missing" values.

- **London-1M data set**
  - This is London movement speed data set that created by [Uber movement project](https://movement.uber.com). This data set includes the average speed on a given road segment for each hour of each day over a whole month (April 2019). In this data set, there are about 220,000 road segments. Note that this data sets only includes the hours or a road segment with at least 5 unique trips in that hour. There are up to 73.09% missing values and most missing values occur during the night. We choose the subset of this raw data set and build a time series matrix of size 35912 x 720 (or a tensor of size 35912 x 24 x 30) in which each time series has at least 70% observations.

- **Guangzhou-2M data set**
  - This traffic speed data set was collected from 214 road segments over two months (61 days from August 1 to September 30, 2016) with a 10-min resolution (144 time intervals per day) in Guangzhou, China. It can be arranged in a matrix of size 214 x 8784 or a tensor of size 214 x 144 x 61.

## Python implementation

- [Low-tubal-rank smoothing tensor completion (LSTC-Tubal)](https://github.com/xinychen/transdim/blob/master/large-imputer/LSTC.ipynb)

Our Publication
--------------

- Xinyu Chen, Yixian Chen, Nicolas Saunier, Lijun Sun (2021). **Scalable low-rank tensor learning for spatiotemporal traffic data imputation**. Transportation Research Part C: Emerging Technologies. [[preprint](https://arxiv.org/abs/2008.03194)] [[DOI](https://doi.org/10.1016/j.trc.2021.103226)] [[data](https://doi.org/10.5281/zenodo.3939792)] [[Python code](https://github.com/xinychen/tensor-learning)]

  >This folder is for the above paper, please cite this paper if it help your research.
