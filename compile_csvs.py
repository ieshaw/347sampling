import os
import pandas as pd


output_file = os.path.abspath(os.path.dirname(__file__))  + '/Compiled_csvs/IS_new.csv'
#
# #loop through models in some directory
#
csv_dir = os.path.abspath(os.path.dirname(__file__))  + '/IS_csvs_new/'

i = 0

for file in os.listdir(csv_dir):

    if i == 0:

        out_df = pd.read_csv(csv_dir + file)

        i = 1

    else:

        new_df = pd.read_csv(csv_dir + file)

        out_df = out_df.append(new_df, ignore_index=True)


out_df.to_csv(output_file)



