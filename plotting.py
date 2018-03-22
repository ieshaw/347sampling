import matplotlib.pyplot as plt
import pandas as pd
import os

#load in old df
file = os.path.abspath(os.path.dirname(__file__))  + '/Compiled_csvs/IS_new.csv'
var_col = ' IS_Variance'
prob_col = ' IS_Estimate'
#scheme is 'IS' or 'MC'
scheme = 'IS'

# #load in old df
# file = os.path.abspath(os.path.dirname(__file__))  + '/Compiled_csvs/MC_100.csv'
# var_col = 'Variance'
# prob_col = 'Estimate'
# #scheme is 'IS' or 'MC'
# scheme = 'MC'

#output stuf
out_file = os.path.abspath(os.path.dirname(__file__)) + '/new_plots/{}_plot'.format(scheme)

##
old_df = pd.read_csv(file)

#turn old df into var df
#index = mu_n
#cols = T
#entries: var_col
var_df = pd.DataFrame(index= old_df[' mu_n'].unique(), columns= old_df['T'].unique().astype(str))
for T in old_df['T'].unique():
    var_df['{}'.format(T)] = old_df[var_col].loc[old_df['T']==T].values

var_df = var_df.sort_index()

var_df.plot(logy= True)
plt.title('{} Variance for differing T'.format(scheme))
plt.xlabel('mu * n')
plt.ylabel('log(Variance)')
plt.savefig(out_file + '_variance.png')

#plot prob df
#index = mu_n
#cols = T
#entries: prob_col
prob_df = pd.DataFrame(index= old_df[' mu_n'].unique(), columns= old_df['T'].unique().astype(str))
for T in old_df['T'].unique():
    prob_df['{}'.format(T)] = old_df[prob_col].loc[old_df['T']==T].values

prob_df = prob_df.sort_index()

prob_df.plot()
plt.title('{} Probabilities for differing T'.format(scheme))
plt.xlabel('mu * n')
plt.ylabel('Probability')
plt.savefig(out_file + '_prob.png')
