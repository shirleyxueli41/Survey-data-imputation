#%%
import pandas as pd
import numpy as np
import argparse

#%%
parser = argparse.ArgumentParser(description='IntroduceMissing')
parser.add_argument('--data_file', type=str, help='CSV file where rows are samples and columns correspond to features, raw file without introducing missing value(full path)')
parser.add_argument('--id_name', type=str, default='ID', help='Column in CSV file which is the identifier for the samples.')
parser.add_argument('--simulate_missing', help='Specifies the %% of original data to be simulated as missing for r^2 computation.', default=0.01, type=float)
parser.add_argument('--output', type=str, help='CSV file where rows are samples and columns correspond to features, fit and test split(full path)')

args = parser.parse_args()

#%%
db = pd.read_csv(args.data_file).set_index(args.id_name)
db
#%%
vmat = db.values
#%%
obs_level = lambda: (vmat.shape[0]*vmat.shape[1]) - np.sum(np.isnan(vmat))
otarget = obs_level() * (1-args.simulate_missing)
mcopy = 100
obs_level(), otarget
#%%
while obs_level() > otarget:
    randpos = np.random.randint(0, len(db), size=mcopy)
    maskpos = np.isnan(vmat[randpos, :])
    randpos = np.random.randint(0, len(db), size=mcopy)
    batch = vmat[randpos, :]
    batch[maskpos] = np.nan
    vmat[randpos, :] = batch
    print('\r{} > {}'.format(obs_level(), otarget), end='')
#%%
db[:] = vmat
# %%
data_inds = list(range(db.shape[0]))
np.random.shuffle(data_inds)
data_inds[:5]
# %%
split = len(db) // 3*2
fit_inds, test_inds = data_inds[:split], data_inds[split:]
len(fit_inds), len(test_inds)
# %%
fitdb = db.iloc[fit_inds]
testdb = db.iloc[test_inds]
# %%
fitdb.to_csv(f'{args.output}.fit.csv')
# %%
testdb.to_csv(f'{args.output}.test.csv')

# %%
print('done')
