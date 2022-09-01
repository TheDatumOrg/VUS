# Volume Under the Surface: new accuracy measures for abnormal subsequences detection in time series

The receiver operator characteristic (ROC) curve and the area under the curve (AUC) are widely used to compare the performance of different anomaly detectors. They mainly focus on point-based detection. However, the detection of collective anomalies concerns two factors: whether this outlier is detected and what percentage of this outlier is detected. The first factor is not reflected in the AUC. Another problem is the possible shift between the anomaly score and the real outlier due to the application of the sliding window. To tackle these problems, we incorporate the idea of range-based precision and recall, and suggest the range-based ROC and its counterpart in the precision-recall space, which provides a new evaluation for the collective anomalies. We finally introduce a new measure VUS (Volume Under the Surface) which corresponds to the averaged range-based measure when we vary the range size. We demonstrate in a large experimental evaluation that the proposed measures are significantly more robust to important criteria (such as lag and noise) and also significantly more useful to separate correctly the accurate from the the inaccurate methods.

<p align="center">
<img width="500" src="./docs/auc_volume.png"/>
</p>

## Publications

If you use VUS in your project or research, please cite our papers:

John Paparrizos, Yuhao Kang, Paul Boniol, Ruey S. Tsay, Themis Palpanas,
and Michael J. Franklin. TSB-UAD: An End-to-End Benchmark Suite for
Univariate Time-Series Anomaly Detection. PVLDB, 15(8): 1697 - 1711, 2022.
doi:10.14778/3529337.3529354

John Paparrizos, Paul Boniol, Themis Palpanas, Aaron Elmore,
and Michael J. Franklin. Volume Under the Surface: new accuracy measures for abnormal subsequences detection in time series. PVLDB, 15(X): X - X, 2022.
doi:X.X/X.X

## Datasets

Due to limitations in the upload size on GitHub, we host the datasets at a different location:

TSB-UAD benchmark: http://chaos.cs.uchicago.edu/tsb-uad/public.zip

Once the full dataset is dowmloaded, store it under the benchmark folder.

We include in the data folder few examples of datasets for examples and visualization purposes.

## Contributors

* John Paparrizos (University of Chicago)
* Paul Boniol (Université Paris Cité)


## Installation

The following tools are required to install VUS from source:

- git
- conda (anaconda or miniconda)

#### Steps

1. Clone this repository using git and change into its root directory.

```bash
git clone https://github.com/boniolp/VUS-temp.git
cd VUS-temp/
```

2. Create and activate a conda-environment 'VUS'.

```bash
conda env create --file environment.yml
conda activate VUS
```

3. Install VUS using setup.py:

```
python setup.py install
```
   
4. Install the dependencies from `requirements.txt`:
```
pip install -r requirements.txt
```


## repo organisation

This repo contains all the codes and examples in order to facilitate the reproductibility of our experimental evaluation. The latter is organized as follows:
- **data/**: Datasets used in the examples.
- **benchmark/**: datasets used in our experimental evaluation.
- **src/**: source code for the anoamly detection methods and accuracy measures.
- **experiments/**:
- - **robustness_analysis/**: 
- - - **robustness_example.ipynb**: Example of experiments conducted in our robustness evaluation. Application to the MBA(805) dataset.
- - - **robustness_benchmark_exp.py**: Script to run the full robustness experimental evaluation on the benchmark.
- - - **benchmark_robustness_result_analysis.ipynb**: Analysis of the full robustness experimental evaluation on the benchmark.
- - **separability_analysis/**:
- - - **benchmark_sep_exp_MBA.py**: script for the separability experimental evaluation on the MBA(805) and MBA(820) datasets.
- - - **benchmark_sep_exp_SED.py**: script for the separability experimental evaluation on the SED dataset.
- - - **separability_example.ipynb**: Example of the separability experimental evaluation conducted on one specific datasets (MBA(805)).
- - **entropyAnalysis/**:
- - - **acc_eval.py**: script to run the full accuracy evaluation.
- - - **Accuracy_evaluation.ipynb**: Analysis of the full accuracy evalution, including the entropy analysis
- - **visualization/**: Visualization notebooks of the different accuracy measures when applied on specific anomaly detection methods and datasets.
- **results/**: All the results from the previously enumerated scripts are stored in this folder.
- - **acc_eval/**: txt files for each time series of the TSB-UAD benchmark containing the accuracy values for every anomaly detection method and every accuracy measure. Results computed by the **experiments/entropyAnalysis/acc_eval.py** script.
- - **robustness_results/**: 
- - - **result_data_aggregated_lag/**: txt files for each time series of the TSB-UAD benchmark containing for every accuracy measure the average standard deviation when we apply lag on the anomaly scores of every anomaly detection (one value per accuracy measures). Results computed by the **experiments/robustness_analysis/robustness_benchmark_exp.py** script.
- - - **result_data_aggregated_noise/**: txt files for each time series of the TSB-UAD benchmark containing for every accuracy measure the average standard deviation when we inject noise on the anomaly scores of every anomaly detection (one value per accuracy measures). Results computed by the **experiments/robustness_analysis/robustness_benchmark_exp.py** script.
- - - **result_data_aggregated_percentage/**: txt files for each time series of the TSB-UAD benchmark containing for every accuracy measure the average standard deviation when we vary the normal/abnormal ratio of the time series (one value per accuracy measures). Results computed by the **experiments/robustness_analysis/robustness_benchmark_exp.py** script.
- - - **noise_lag_result.pickle**: Aggregation of the noise and lage results
- - - **ratio_result.pickle**: Aggregation of the ratio results
- - **separability_results/**: 
- - - **MBA_ECG805_data.out_z_test.pickle**: aggregated results of the **experiments/separability_analysis/benchmark_sep_exp_MBA.py** script
- - - **MBA_ECG820_data.out_z_test.pickle**: aggregated results of the **experiments/separability_analysis/benchmark_sep_exp_MBA.py** script
- - - **SED_data.out_z_test.pickle**: aggregated results of the **experiments/separability_analysis/benchmark_sep_exp_SED.py** script

## How to use

Here is an example on how to run Range-AUC and VUS measures for a given model.

```python


import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from sklearn.preprocessing import MinMaxScaler
from src.models.iforest import IForest
from src.analysis.robustness_eval import generate_curve
from src.utils.metrics import metricor

# Data Preprocessing
filepath = "path/to/time_series"
slidingWindow = 100 #the user-defined subsequence length 

df = pd.read_csv(filepath, header=None).to_numpy()
data = df[0:-1,0].astype(float)
label = df[0:-1,1]

#specific type of data structure for IF, LOF, and PCA
X_data = Window(window = slidingWindow).convert(data).to_numpy()


# Training the model and computing the score

# Matrix Profile
# clf = MatrixProfile(window = slidingWindow)
# x = data

# POLY
# clf = POLY(power=3, window = slidingWindow)
# x = data

# PCA
# clf = PCA()
# x = X_data

# LOF
# clf = LOF(n_neighbors=20, n_jobs=1)
# x = X_data

# Isolation Forest
clf = IForest(n_jobs=1)
x = X_data

clf.fit(x)
score = clf.decision_scores_

# Score normalization
score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))


# Computing RANGE_AUC_ROC and RANGE_AUC_PR
grader = metricor()
R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=label, score=score, window=slidingWindow, plot_ROC=True)

# Computing VUS_ROC and VUS_PR
_, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(label,score,2*slidingWindow)

```
