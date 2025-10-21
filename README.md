The repository presents scripts used in the paper "**Predicting allergy and postpartum depression from incomplete compositional microbiome**." for reproducibility and further usage.

Python script for log-transformations, imputation, and forecasting is "Transformation_Imputation_Forecasting.py". Find below its parameters and instructions to run the script.

**User-defined parameters:**

- Dataset_N: The number of dataset in the paper. Default is 2.

- Imputation_type: if imputation of missing values has to be done. "No" is for no imputation; "Imputation" (default) is for complementing a complete dataset with artificial values with both labels y=-1,1, "Oversampling" is for complementing a complete dataset with artificial values with only y=1.

- Imputation_by: type of model chosen for imputation. The options are linear, SVR, GPR,CVAE, or cGAN (default).

- Transoformation_type: type of log-transformation for Dataset 2. The options are Compositional (default), CLR, ALR, or PLR

- is_median: do we want to use median or mean when select optimal number of features. 1 is for median, 0 is for mean (default)

- is_balanced: do we want to use weights during classification? 0 is for equal weights (default), 1 is for weight depending on class size.

**Reproducibility**

- To reproduce results from Table 8 run from terminal 

python Transformation_Imputation_Forecasting.py

- To reproduce results from Table 7 run from terminal

python Transformation_Imputation_Forecasting.py --Imputation_type="No" --Transoformation_type="ALR"

- To reproduce results from Section 4.2 by Oversampling run from terminal

python Transformation_Imputation_Forecasting.py --Dataset_N=1 --is_median=1 --Imputation_type="Oversampling" --Imputation_by="GPR"

Use --Imputation_type="Imputation" to include artificial data with y=-1 as well.

**Python notebooks**

- Transformation_Imputation_Forecasting.py exists in notebook format .ipynb as well.

- Run Depression classification.ipynb to reproduce results from Table 4.

- Run Depression bacteria list.ipynb to reproduce results from Table 5.

- Run Depression forecast by depression.ipynb to forecast postpartum depression from EPDS during pregnancy
- 
- Run Depression forecast by Shannon.ipynb to forecast postpartum depression from Shannon entropy

- Run Depression forecast by missingness.ipynb to forecast postpartum depression from the information about missingness only

**Libraries**

Folder libraries include libraries for transformation and imputation.

- Imputation_library provides functions for filling data points with artificial data instead of missing  time points.

- Transformation_library provide functions for logarithm transformations and softmax.

**Data**

To run the codes, 2 files are needed for each dataset: metadata and species profile.

- Dataset 1 is available on NCBI under BioProject PRJEB62678. In the species profile, name the first column Sample_id.
- Dataset 2 can be found on NCBI BioProject PRJNA730851.

**Requirements** 

The versions of all python modules are listed in requirements.txt


