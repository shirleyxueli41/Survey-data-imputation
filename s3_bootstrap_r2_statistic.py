#%%
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import f1_score
#%%
class args:
	data_file = 'datasets/phenotypes/data.csv'
	simulated_data_file = 'datasets/phenotypes/data_test.csv'
	imputed_data_file = 'datasets/phenotypes/imputed_data_test.csv'
	num_bootstraps = 100
#%%
parser = argparse.ArgumentParser(description='AutoComplete')
parser.add_argument('--data_file', type=str, help='CSV file where rows are samples and columns correspond to features. Fit set. ')
parser.add_argument('--id_name', type=str, default='ID', help='Column in CSV file which is the identifier for the samples.')
parser.add_argument('--simulated_data_file', type=str, help='Data with simulated missing values. This is required to check which values were simulated as missing.')
parser.add_argument('--imputed_data_file', type=str, help='Imputed data.')
parser.add_argument('--num_bootstraps', type=int, default=100, help='Number of times to bootstrap the test statistic.')
parser.add_argument('--saveas', type=str, default='results_r2.csv', help='Where to save the evaluation results.')
args = parser.parse_args()
#%%
# the original dataset
original_data = pd.read_csv(args.data_file).set_index(args.id_name)
original_data
#%%
# data with simulated missing values
simulated_data = pd.read_csv(args.simulated_data_file).set_index(args.id_name)
simulated_data
#%%
imputed_data = pd.read_csv(args.imputed_data_file).set_index(args.id_name)
imputed_data
#%%
assert simulated_data.shape == imputed_data.shape
assert simulated_data.index.tolist() == imputed_data.index.tolist()
assert imputed_data.isna().sum().sum() == 0
assert len(imputed_data.index.intersection(original_data.index)) == len(imputed_data)
#%%
ests = []
stds = []
ests_matching_percentages = []
stds_matching_percentages = []
ests_F1s = []
stds_F1s = []
missing_fracs = []
unique_values = []
nsize = len(imputed_data)
for pheno in imputed_data.columns:
	missing_frac = simulated_data[pheno].isna().sum() / nsize
	unique_value = len(np.unique(original_data[pheno]))

	est = np.nan
	stderr = np.nan
	est_matching_percentages = np.nan
	stderr_matching_percentages = np.nan
	est_F1 = np.nan
	stderr_F1 = np.nan
	if missing_frac != 0:
		stats = []
		f1s = []
		matching_percentages = []
		# for n in range(args.num_bootstraps):
		n = 0
		while n < args.num_bootstraps:
			boot_idx = np.random.choice(range(nsize), size=nsize, replace=True)
			boot_obs = original_data.loc[imputed_data.index][pheno].iloc[boot_idx]
			boot_imp = imputed_data[pheno].iloc[boot_idx]
		
			simulated_missing_inds = simulated_data[pheno].iloc[boot_idx].isna() & ~boot_obs.isna()

			if simulated_missing_inds.sum() > 0:
				r2 = np.corrcoef(
					boot_obs.values[simulated_missing_inds],
					boot_imp.values[simulated_missing_inds])[0, 1]**2
				    # Calculate the percentage of matching values
				boot_imp[simulated_missing_inds] = np.round(boot_imp[simulated_missing_inds]).astype(int)
				matches = (boot_obs[simulated_missing_inds] == boot_imp[simulated_missing_inds]).sum()
				total_simulated_missing = simulated_missing_inds.sum()
				matching_percentage = (matches / total_simulated_missing) * 100
				if unique_value <= 3:
					f1 = f1_score(boot_obs[simulated_missing_inds], boot_imp[simulated_missing_inds])
				else:
					f1 = np.nan

 				# Count unique values in the observed vector for simulated missing indices

			else:
				r2 =  np.nan
				matching_percentage = np.nan
				f1 = np.nan

			n += 1
			stats += [r2]
			f1s += [f1]
			matching_percentages += [matching_percentage]
		est = np.nanmean(stats)
		stderr = np.nanstd(stats)
		est_matching_percentages = np.nanmean(matching_percentages)
		stderr_matching_percentages = np.nanstd(matching_percentages)
		est_F1 = np.nanmean(f1s)
		stderr_F1 = np.nanstd(f1s)
		print(f'{pheno} ({missing_frac*100:.1f}%): {est:.4f} ({stderr:.4f})')
	else:
		print(f'{pheno} ({missing_frac*100:.1f}%)')
	missing_fracs += [missing_frac]
	unique_values += [unique_value]
	ests += [est]
	stds += [stderr]
	ests_matching_percentages += [est_matching_percentages]
	stds_matching_percentages += [stderr_matching_percentages]
	ests_F1s += [est_F1]
	stds_F1s += [stderr_F1]
# %%
results = pd.DataFrame(dict(pheno=imputed_data.columns, options=unique_values, missing_fraction=missing_fracs, estimates=ests, stderrs=stds, estimates_accuracy=ests_matching_percentages, stderrs_accuracy=stds_matching_percentages, F1=ests_F1s, std_F1=stds_F1s)).set_index('pheno')
results.to_csv(args.saveas)
results
# %%
