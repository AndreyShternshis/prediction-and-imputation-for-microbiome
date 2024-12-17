Scripts used in the paper "Predicting postpartum depression and food allergy from incomplete microbiome data"

To run the codes, you need 2 files for each of dataset: metadata and species profile. For the latter of the depression dataset, name the first column Sample_id.

libraries include functions for transformation and imputation

"Depression classification" presents the results for the complete data points. "Bacteria list" shows what features are used for classification. We forecast depression by using 1) two bacteria only, 2) imputation tecniques 3) oversampling, that is imputing only positive labels 4) depression states during pregnancy 5) variables denoting missingness.

Prediction of food allergy is shown by applying 1) all log ratios 2) imputation
