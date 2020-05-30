import pandas as pd
import os
df = pd.DataFrame(columns=['dir', 'class'])
root_dir = '/home/liushuai/cleaned_images/train'
for d in range(1,7):
    cur_cat = str(d+5000)
    cur_dir = os.path.join(root_dir, cur_cat)
    for l in os.listdir(cur_dir):
        df.loc[len(df)] = [os.path.join(cur_dir, l), cur_cat]
df.to_csv(os.path.join(root_dir, 'description.csv'))