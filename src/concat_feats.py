import pandas as pd

stat_feas = pd.read_pickle('./data/stat_feas.pkl')
lgb_feas = pd.read_pickle('./data/lgb_feas.pkl')

feas = stat_feas.join(lgb_feas)

feas.to_pickle('./data/feas.pkl')