There are seven data sets in this folder.

- Guangzhou urban traffic speed data set.

- Birmingham parking data set.

- Hangzhou metro passenger flow data set.

- Seattle freeway traffic speed data set.

- NYC taxi data set.

- PeMS traffic speed data set. [[PeMS Tutorial](https://people.eecs.berkeley.edu/~varaiya/papers_ps.dir/PeMSTutorial.pdf)]

- Electricity consumption data set. This data set collected electricity consumption from 370 clients and it is publicly available at [UCI data repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014). We use the 321-client subset of this hourly electricity data, which is available at [multivariate time series data](https://github.com/laiguokun/multivariate-time-series-data).



#### List of publicly available spatiotemporal data sets (e.g., transportation data).



| No | Name | Intro. | Download |
|---:|:------|:--------|:---------|
|  1 | **PeMS-BAY** | This data set contains the speed information of 325 sensors in the Bay Area, USA lasting for six months ranging from January 1, 2017 to June 30, 2017. The total number of observed traffic data is 16,941,600 and the time interval is 5-minute. | [GitHub](https://github.com/liyaguang/DCRNN), [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX), [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) |
|  2 | **Seattle Inductive Loop Detector Dataset** | The data is collected by the inductive loop detectors deployed on freeways in Seattle area. This data set contains spatiotemporal speed information of the freeway system within a whole year of 2015. The time interval of this data set is 5-minute. Here, Loop Adjacency Matrix describes the traffic network structure as a graph. Notably, the speed data is stored by a pickle file and can be loaded by `Pandas`: | [GitHub](https://github.com/zhiyongc/Seattle-Loop-Data), [Google Drive](https://drive.google.com/drive/folders/1XuK0fgI6lmSUzmToyDdHQy8CPunlm5yr?usp=sharing) |
```python
import pandas as pd
data = pd.pickle('../speed_matrix_2015')
```
| | | | |
|---:|:------|:--------|:---------|
|  3 | **Guangzhou Traffic Speed Dataset** | This data set contains the speed information of 214 anonymous road segments (mainly consist of urban expressways and arterials) in Guangzhou, China within two months (i.e., 61 days from August 1, 2016 to September 30, 2016) at 10-minute interval. | [Zenodo](https://zenodo.org/record/1205229) |
|  4 | **UTD19: Largest Multi-City Traffic Dataset** | UTD19 is a large-scale traffic dataset from over 23541 stationary detectors on urban roads in 40 cities worldwide making it the largest multi-city traffic dataset publically available. | [Data](https://utd19.ethz.ch/index.html) |
|  5 | **London Movement Speed Dataset** | This dataset is created by [Uber movement project](https://movement.uber.com/), which includes the average speed on a given road segment for each hour of each day in April 2019. Note that this dataset only includes road segments with at least 5 unique trips in that hour. As a consequence, there are about 73% missing values. | [Data](https://movement.uber.com/cities/london/downloads/speeds?lang=en-US&tp[y]=2019&tp[q]=2) |

> Example code for processing movement speed as a multivariate time series data. This is indeed a spatiotemporal matrix which consists of 200,000+ road segments (i.e., 70,000+ road semgents with different directions) and 30 x 24 time points. Note that this dataset is downloaded from [Uber movement project](https://movement.uber.com/). For getting the data file `movement-speeds-hourly-london-2019-4.csv`, you need to choose city as `London`, product type as `speeds`, and time period as `2019 Quarter 2`.


```python
import numpy as np
import pandas as pd

data = pd.read_csv('../datasets/London-data-set/movement-speeds-hourly-london-2019-4.csv')

road = data.drop_duplicates(['osm_way_id', 'osm_start_node_id', 'osm_end_node_id'])
road = road.drop(['year', 'month', 'day', 'hour', 'utc_timestamp', 'segment_id', 'start_junction_id', 
                  'end_junction_id', 'speed_mph_mean', 'speed_mph_stddev'], axis = 1)
tensor = np.zeros((road.shape[0], max(data.day.values), 24))
k = 0
for i in range(road.shape[0]):
    temp = data[(data['osm_way_id'] == road.osm_way_id.iloc[i]) 
                & (data['osm_start_node_id'] == road.osm_start_node_id.iloc[i]) 
                & (data['osm_end_node_id'] == road.osm_end_node_id.iloc[i])]
    for j in range(temp.shape[0]):
        tensor[k, temp.day.iloc[j] - 1, temp.hour.iloc[j]] = temp.speed_mph_mean.iloc[j]
    k += 1
    if (k % 1000) == 0:
        print(k)
mat = tensor.reshape([road.shape[0], max(data.day.values) * 24])
np.save('../datasets/London-data-set/hourly_speed_mat.npy', mat)

del data, road, tensor
```

```python
import numpy as np
import pandas as pd

data = pd.read_csv('../datasets/Temperature-data-set/data.tsv', sep = '\t', header = None)
mat = data.values

num_month = 399
tensor = np.zeros((30, 84, 399))
for t in range(num_month):
    tensor[:, :, t] = mat[t * 30 + 1 : (t + 1) * 30 + 1, 1 :]
    
np.save('../datasets/Temperature-data-set/tensor.npy', tensor)
```

- [PORTAL is the official transportation data archive for the Portland-Vancouver Metropolitan region.](https://portal.its.pdx.edu/home) PORTAL provides a centralized, electronic database that facilitates the collection, archiving, and sharing of data and information for public agencies within the region. The data stored in Portal includes 20-second granularity loop detector data from freeways in the Portland-Vancouver metropolitan region, arterial signal data, travel time data, weather data, incident data, VAS/VMS message data, truck volumes, transit data, and arterial signal data. Many of these data feeds are received by PORTAL in real time or on a daily basis and for most, the retrieval and archiving process is fully automated.


### Reading material

- [Data for the public good](https://nic.org.uk/app/uploads/Data-for-the-Public-Good-NIC-Report.pdf), National Infrastructure Commission report.

