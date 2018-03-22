import os
import pandas as pd


file_path = os.path.abspath(os.path.dirname(__file__)) + '/Compiled_csvs'

mc_df = pd.read_csv(file_path +'/MC_100.csv', index_col=0)

is_df = pd.read_csv(file_path +'/IS_new.csv', index_col=0)

out_file = os.path.abspath(os.path.dirname(__file__)) + '/new_stuff/paper_new.csv'

mc_df['fake_index'] = mc_df['T'].astype(str) + '_' + mc_df['mu'].astype(str)

mc_df.index = mc_df['fake_index']

is_df['fake_index'] = is_df['T'].astype(str) + '_' + is_df[' mu'].astype(str)

is_df.index = is_df['fake_index']

out_df = mc_df

out_df[['IS_Estimate','IS_Variance']] = is_df[[' IS_Estimate',' IS_Variance']]

###Add variance ratio

out_df = out_df.drop(['fake_index'], axis=1)

out_df = out_df.reset_index(drop= True)

out_df.to_csv(out_file, index=False)

#out_df.to_latex(out_file, index = False)