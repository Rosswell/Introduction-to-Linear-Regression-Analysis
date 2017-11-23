import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('2017-11-21_230246.csv')
stats_df = df.drop(['city', 'state', 'zip'], 1)
stats_df = stats_df.dropna()
df = df.dropna()

print(f"Mean:\n{stats_df.mean()}")
print(f"Median:\n{stats_df.median()}")
print(f"Mode:\n{stats_df.mode().iloc[0]}")
print(f"Variance:\n{stats_df.var()}")
# outlier detection and exclusion, using 3 standard deviations as the cutoff point

df_sans_outliers = stats_df[(np.abs(stats.zscore(stats_df)) < 3).all(axis=1)]
outliers = stats_df[(np.abs(stats.zscore(stats_df)) > 3).all(axis=1)]

sns_pairplot = sns.pairplot(df_sans_outliers, size=2.5)
fig1 = sns_pairplot.fig
fig1.savefig("pairplot.png")

pyplot.clf()
le = LabelEncoder()
df['city'] = le.fit_transform(df['city'])
sns_heatmap = sns.heatmap(df.corr(), annot=True)
fig2 = sns_heatmap.get_figure()
fig2.savefig("heatmap.png")