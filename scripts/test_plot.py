import sys 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

fname = sys.argv[1]
df = pd.read_csv(fname, skiprows=[0])
df['smoothed'] = df['r'].ewm(span=100, adjust=False).mean()
sns.relplot(data=df, x=np.arange(len(df)), y='smoothed',kind='line')
plt.show()
