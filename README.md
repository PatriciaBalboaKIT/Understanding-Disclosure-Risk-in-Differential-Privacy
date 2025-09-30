# Understanding Disclosure Risk in Differential Privacy with Applications to Noise Calibration and Auditing
This repo holds the code and result files for all experiments performed for the paper "Understanding Disclosure Risk in Differential Privacy with Applications to Noise Calibration and Auditing" by Patricia Guerra-Balboa, Annika Sauer, Héber Hwang Arcolezi and Thorsten Strufe. In particular:

1.  `Long_version.pdf` is the extended version of the submission that contains all the formal proofs of the results presented in the paper.
2.  The datasets used in the experiments.
3.  The source code for the utility experiments, as well as for any related plots.
4.  The code to generate the plots of the formulas represented in the paper.



## Git LFS
This repository uses [Git Large File Storage (LFS)](https://git-lfs.github.com/) to manage large files, such as `.pkl` or `.csv`.  
To access the actual file contents (instead of just pointer files shown on GitHub), please install Git LFS before cloning:

```bash
git lfs install
git clone <repo-url>
```

## Bounds
To reproduce the plots that compare various bounds (Figures 1 and 2), refer to the `Bounds/visualization` directory. It holds
* `compare_theorems.py` which recreates the plot from Figure 1, and
* `plot_laplace.py` which recreates the plot from Figure 2.
* `plot_grr.py` which creates an equivalent plot to Figure 2 for the GRR mechanism
Activate the env
```bash
conda activate wb
```
Then, run
```bash
python -m Bounds.visualization.compare_theorems
```
or 
```bash
python -m Bounds.visualization.plot_laplace
```
The results are saved in the `Bounds/plots` folder.


## Attacks on DP-SGD
The code to run DP-SGD and compute ReRo was provided by Jamie Hayes based on the publication
```markdown
Hayes, J., Balle, B., & Mahloujifar, S. (2023). Bounding training data reconstruction in DP-SGD. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt & S. Levine (Eds.), Advances in Neural Information Processing Systems (Vol. 36, pp. 78696–78722). Curran Associates, Inc. [PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/f8928b073ccbec15d35f2a9d39430bfd-Paper-Conference.pdf)
```
A copy of the original shared code is provided in `DPSGD_noAux/shared_reconstruct_with_prior.ipynb`.

This part discusses how to set up and recreate the experiments for DP-SGD (folders `DPSGD_*/`). The folder structure is as follows
* `DPSGD_fullAux`: Membership inference attack (MIA) against DP-SGD under uniform prior.
* `DPSGD_fullAux_nonUnif`: MIA against DP-SGD under non-uniform prior.
* `DPSGD_partAux`: Attribute inference attack (AIA) against DP-SGD under uniform prior.
* `DPSGD_noAux`: Data reconstruction attack (DRA) againat DP-SGD under uniform prior.

### Environment
The environment is defined in `white_box_env.yml`. Create and activate with:
```bash
conda env create -f white_box_env.yml
conda activate wb
```

### Data Preprocessing
The scripts in `DPSGD_*` do not require additional preprocessing. The loading and splitting of the MNIST and Fashion MNIST datasets is done directly in the scripts (in `DPSGD_*/mnist_*.py` or `DPSGD_*/fashion_*.py`).

### Run Experiments
Activate the corresponding environment
```bash
conda activate wb
```
Each attack (MIA, AIA, DRA) has its own folder with the scripts `mnist_script.sh` and `fashion_script.sh` to run the attack on the respective dataset. Each script takes about 5 days to execute.

To run an attack, from the top-level project directory run
```bash
bash $FOLDER/mnist_script.sh
```
to run on the MNIST dataset or 
```bash
bash $FOLDER/fashion_script.sh
```
to run on the Fashion dataset. Folder is either `DPSGD_fullAux` for MIA under uniform prior, `DPSGD_fullAux_nonUnif` for MIA under non-uniform prior, `DPSGD_partAux` for AIA under uniform prior or `DPSGD_noAux` for DRA under uniform prior.

### Visualize results
Each folder has a script to process and visualize the results from the MNIST and Fashion datasets respectively. After running the respective experiments, from the top-level project-directory run
```bash
bash $FOLDER/process_results_mnist.sh
```
for the MNIST or 
```bash
bash $FOLDER/process_results_fashion.sh
```
for the Fashion datasets. The results can afterwards be found in `.csv`format under `$FOLDER/results/*.csv` and in `.png` format under `$FOLDER/plots/*.png`

## Imputation Attack
The code and Readme for the imputation attack has been adapted from https://github.com/bargavj/EvaluatingDPML/tree/master. The relevant code can be found in the folder `Blackbox/improved_ai`. The original publication is
```markdown
Bargav Jayaraman and David Evans. 2022. Are Attribute Inference Attacks Just Imputation? In Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security (CCS '22). Association for Computing Machinery, New York, NY, USA, 1569–1582. https://dl.acm.org/doi/abs/10.1145/3548606.3560663
```

### Environment
The environment for the imputation attack is defined in `black_box_env.yml` for the black-box attacks. Create and activate with:
```bash
conda env create -f black_box_env.yml
conda activate bb
```

### Data Preprocessing
For the imputation attack, we work with the following datasets
* Census: The Census19 dataset can be downloaded as .zip from https://github.com/bargavj/EvaluatingDPML/blob/master/dataset/census.zip or can be found as .zip in the `Blackbox/dataset/census` folder. 
* Texas-100 v2: `PUDF_base1q2006_tab.txt`, `PUDF_base2q2006_tab.txt`, `PUDF_base3q2006_tab.txt` and `PUDF_base4q2006_tab.txt` files can be downloaded from https://www.dshs.texas.gov/THCIC/Hospitals/Download.shtm and should be saved in the `Blackbox/dataset/texas_100_v2/` folder. They can also be found directly in `Blackbox/dataset/texas_100_v2/` folder.

The processed dataset files are in the form of two pickle files: $DATASET_feature.p and $DATASET_labels.p (where $DATASET is a placeholder for the data set file name). For Texas-100 v2, go to the `Blackbox/extra` folder and run
```bash
python preprocess_dataset.py texas_100_v2 --preprocess=1
```
For Census, extract the .zip and place the files `census_features.p`, `census_labels.p` and `census_feature_desc.p` into the `Blackbox/dataset` folder.

### Run Experiments
Activate the corresponding environment
```bash
conda activate bb
```
Then, from the `Blackbox/improved_ai` folder, run:
```bash
bash run_experiments.sh $DATASET
```
where $DATASET is either `census` or `texas_100_v2`. This script will take about 15min to execute. Results can then be found in `.csv` format under `Blackbox/improved_ai/results/` as `results_dataset$DATASET$.csv `

## DP Auditing
This section discusses how to set up and recreate the experiments for U-ReRo for DP Auditing. The code to implement the LDP protocols and attacks is taken from the public GitHub repository https://github.com/hharcolezi/ldp-audit.

### Environment
The environment for running the LDP experiments is defined in `dp_audit.yml`.  
Create and activate it with:
```bash
conda env create -f dp_audit.yml
conda activate ldp_audit
```

### Data Preprocessing
The DP Audit experiments work on the `Porto/porto_graph.pkl` and the `Geolife/beijing_graph.pkl` files.

The raw Porto taxi dataset is located at `Porto/Data/train.csv` or can be downloaded from https://www.kaggle.com/datasets/crailtap/taxi-trajectory and placed there.

To generate processed data, run:

```bash
python -m Porto.preprocessing
```

This produces the `porto_graph.pkl` (OSMnx road graph, extracted based on centerpoint and radius specified in `Porto/constants.py`).

The raw Geolife data is located under `Geolife/Data/` or can be downloaded from https://www.microsoft.com/en-us/download/details.aspx?id=52367 and placed in the same location. To generate processed data, run:
```bash
python -m Geolife.preprocessing
```
This produces the equivalent `.pkl` file as for the Porto dataset.

### Run Experiments
Ensure that the LDP audit environment is active.
```bash
conda activate ldp_audit
```
To run the experiments, run
```bash
bash run_audits.sh $DATASET
```
where $DATASET is either `porto` to run on the Porto taxi dataset or `beijing` to run on the Geolife dataset. The script will take about 5h to execute.

The resulting epsilon estimation can afterwards be found in `DP_Audit/[module]/results` in `.csv` format. The RAD result of each attack can also be found in `DP_Audit/[module]/results` in `.csv` format. In order to plot the resulting RAD of these attacks together with the theoretical bound, run
```bash
python -m DP_Audit.[module].plot_attack_porto
```
for the Porto dataset or
```bash
python -m DP_Audit.[module].plot_attack_beijing
```
for the Beijing dataset.
In order to plot the resulting epsilon estimate together with the estimate from LDP Auditor, run
```bash
python -m DP_Audit.[module].plot_audit_porto
```
for the Porto dataset or
```bash
python -m DP_Audit.[module].plot_audit_beijing
```
for the Beijing dataset.


### Visualize Experiments
In order to recreate the plot from Figure 5, run
```bash
python -m DP_Audit.visualization.plot_porto
```
for the Porto dataset or
```bash
python -m DP_Audit.visualization.plot_beijing
```
for the Beijing dataset. Find the resulting plot in `DP_Audit/plots`.

### Obtain Results from LDP Auditor
The code for LDP Auditor is taken directly from the author's GitHub: https://github.com/hharcolezi/ldp-audit. The original publication is
```markdown
 	Arcolezi, Héber H., and Sébastien Gambs. "Revealing the True Cost of Locally Differentially Private Protocols: An Auditing Perspective." Proceedings on Privacy Enhancing Technologies 4 (2024): 123-141. https://arxiv.org/abs/2309.01597.
```


We adapt only the parameter `k`, the size of the data domain, to the data domain sizes of our respective datasets, `k=3,052` and `k=5,356`.

To run LDP Auditor, do
```bash
conda activate ldp_audit
cd DP_Audit/LDP_Auditor
python experiment_1.py
```
Find the result documents `summary_Porto.csv` and `summary_Beijing.csv` in `LDP_Auditor/results`. The script takes about 2h to run.
