The repository presents scripts used in the paper "**Predicting allergy and postpartum depression from incomplete compositional microbiome**." for reproducibility and further usage.


<img width="1304" height="476" alt="image" src="https://github.com/user-attachments/assets/f6cb7fac-4ff7-4011-80b7-7f7ca509dd21" />


Scripts in Python 3.10 for log-transformations, imputation, and forecasting is "Transformation_Imputation_Forecasting.py". Find below instructions how to run the script and its parameters.

# Data

The datasets used in the study are publicly available.

- [Dataset 1](https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJEB62678) is available on NCBI under BioProject PRJEB62678. In the species profile, name the first column Sample_id. There are two files required to run the scripts for Dataset 1:
-  - BASIC_metadata_full.csv
   - Species_Profile_full.csv
- [Dataset 2](https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA730851) can be found on NCBI BioProject PRJNA730851. There are two files required to run the scripts for Dataset 2:
- - GMAP_metadata_public.csv
  - feature-table-not-filtered-l6.csv

# Installation
Follow the instructions below to run the scripts.

- Clone the GitHub repository
```
git clone https://github.com/AndreyShternshis/prediction-and-imputation-for-microbiome.git
```
- Copy dataset files in the folder prediction-and-imputation-for-microbiome
- Create a conda environment through Integrated Development Environments (VSCode/PyCharm) or in terminal:
```
conda create -n microb python=3.10
conda activate microb
```
- Import required packages in terminal

```
pip install -r requirements.txt
```

**Alternatively**, use graphical interface in [Anaconda-Navigator](https://www.anaconda.com/products/navigator)

```
Go to Enviroments (left tab)
Click "Import" (left bottom tab)
Choose  conda_environment.yaml located in the folder prediction-and-imputation-for-microbiome
From Home (left tab) run scripts launching JupyterLab/PyCharm/VSCode
```

# User-defined parameters

- Dataset_N: The number of dataset in the paper. Default is 2.

- Imputation_type: if imputation of missing values has to be done. "No" is for no imputation; "Imputation" (default) is for complementing a complete dataset with artificial values with both labels y=-1,1, "Oversampling" is for complementing a complete dataset with artificial values with only y=1.

- Imputation_by: type of model chosen for imputation. The options are linear, SVR, GPR,CVAE, or cGAN (default).

- Transoformation_type: type of log-transformation for Dataset 2. The options are Compositional (default), CLR, ALR, or PLR

- is_median: do we want to use median or mean when select optimal number of features. 1 is for median, 0 is for mean (default)

- is_balanced: do we want to use weights during classification? 0 is for equal weights (default), 1 is for weight depending on class size.


# Reproducibility

- To reproduce results from Table 9 run from terminal 

python Transformation_Imputation_Forecasting.py

- To reproduce results from Table 8 run from terminal

python Transformation_Imputation_Forecasting.py --Imputation_type="No" --Transoformation_type="ALR"

- To reproduce results from Section 4.2 by Oversampling run from terminal

python Transformation_Imputation_Forecasting.py --Dataset_N=1 --is_median=1 --Imputation_type="Oversampling" --Imputation_by="GPR"

Use --Imputation_type="Imputation" to include artificial data with y=-1 as well.

# Python notebooks

- Transformation_Imputation_Forecasting.py exists in notebook format .ipynb as well.

- Run Depression classification.ipynb to reproduce results from Table 4.

- Run Depression bacteria list.ipynb to reproduce results from Table 5.

- Run Depression forecast by depression.ipynb to forecast postpartum depression from EPDS during pregnancy.

- Run Depression forecast by Shannon.ipynb to forecast postpartum depression from Shannon entropy.

- Run Depression forecast by missingness.ipynb to forecast postpartum depression from the information about missingness only.

# Libraries

Folder libraries include libraries for transformation and imputation.

- Imputation_library provides functions for filling data points with artificial data instead of missing  time points.

- Transformation_library provide functions for logarithm transformations and softmax.



