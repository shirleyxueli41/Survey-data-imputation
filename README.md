# DAP survey data imputation
Material is written based on the github repo Autocomplete and a paper published in nature genetics. 
https://github.com/sriramlab/AutoComplete/tree/master

https://www.nature.com/articles/s41588-023-01558-w

AutoComplete is a deep-learning based imputation method capable of imputing continuous and binary values simultaneously.

## Getting Started

AutoComplete can run with most Python 3 versions, and defines neural nets using [pytorch](https://pytorch.org).
The dependencies can be found in `requirements.txt`

### Download the following scripts to your working directory and load Python
```
s1_phenotype_missingness_simulation.py
s2_fit.py
s3_bootstrap_r2_statistic.py
ac.py*
dataset.py*

use Python-3.9
```



## Imputation Demo: hyperparameter tuning

An example procedure for training, testing, and scoring an example survey dataset with missing values. 

Follow the following commands. 

### Step One
Artificial missing values are introduced to `example_survey_data.tsv` such that they can be withheld then scored after imputation.
This script also split the data into training/fitting set and testing set. 
66.6% will be in training set and 33.4% will be in testing set. 

```bash
python s1_phenotype_missingness_simulation.py \
--data_file example_survey_data.csv \
--id_name dog_id \
--simulate_missing 0.05 \
--output example_survey_data.MR  # The prefix of the two output files.

# output files:
# example_survey_data.MR.fit.csv
# example_survey_data.MR.test.csv

```

### Step Two
The method is trained/fitted on a training/fitting split of the data saved to `example_survey_data.MR.fit.csv`.
```bash
python s2_fit.py \
    --data_file example_survey_data.MR.fit.csv \
    --id_name dog_id \
    --save_model_path mymodel.pth \
    --batch_size 2048 \
    --epochs 30 \
    --device cpu:0
```

The fitted model is used to impute the testing split of the data which is `example_survey_data.MR.test.csv`.
```bash
python s2_fit.py \
    --data_file example_survey_data.MR.fit.csv \
    --id_name dog_id \
    --impute_using_saved mymodel.pth \
    --impute_data_file example_survey_data.MR.test.csv \
    --output example_survey_data.MR.test_imputed.csv \
    --device cpu:0
```
Note: "--data_file" This can't be removed, but this doesn't influence the result either. Because fitting and testing are written in the same script, when we ask the script to do tesing, the fitting section can't be silenced, the script has to go through the code that does the fitting and then go the testing section. 

### Step Three
Finally, the simulated missing values are scored against their originally observed values. The Pearson's r^2 correlation is used and a number of bootstrap replicates are used to obtain the point estimate of the accuracy and its standard error.
There are also two other statistics added. (1) F1 (2) Imputation accuracy. 

```bash
python s3_bootstrap_r2_statistic.py \
    --data_file example_survey_data.csv \
    --id_name dog_id \
    --simulated_data_file example_survey_data.MR.test.csv \
    --imputed_data_file example_survey_data.MR.test_imputed.csv \
    --num_bootstraps 10 \
    --saveas result_r2_phenotype_demo.csv
```



## Apply the model to the real dataset after hyperparameter tuning.
Note: We don't need to introduce missing data, so no need to run s1_phenotype_missingness_simulation.py


Generate a model based on the data itself with the best parameter identified from "hyperparameter tuning" step. 
```bash
python s2_fit.py \
    --data_file example_survey_data.csv \
    --id_name dog_id \
    --save_model_path myfinalmodel.pth \
    --batch_size 2048 \
    --epochs 50 \
    --device cpu:0
```

Apply the model back to the data, `example_survey_data_imputed.csv` will be the final result after imputation. 

```bash
python s2_fit.py \
    --data_file example_survey_data.csv \
    --id_name dog_id \
    --impute_using_saved myfinalmodel.pth  \
    --impute_data_file example_survey_data.csv \
    --output example_survey_data_imputed.csv \
    --device cpu:0
```
Note: The model learn from the dataset and then be applied back to the dataset. That's how autoencoder works. This will not cause overfitting. 
