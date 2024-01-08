import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df_path = 'data/emotic_train.csv'
test_df_path = 'data/emotic_test.csv'

train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)

print("Train dataset:")
print(f"Dataset is {train_df.shape[0]} rows by {train_df.shape[1]} columns")
print(f"Columns are : '{', '.join(list(train_df.columns))}'")
print(f"Types of columns: {train_df.dtypes}")
print(f"Columns example: {train_df.iloc[0]}")
print("histogram of the log of size:")
sns.histplot(train_df['size'], bins=100,log_scale=True)
plt.show()
print(train_df.describe())
print("Data are mostly of the same size, but there are some outliers")
print(f"Extension(s) of images: {train_df['extension'].unique()}")

print(f" random sample of 5 rows: {train_df['box'].sample(5)}")
print("box data have a bad quality, we will need to clean them")

print(f"Labels: {train_df['label'].unique()}")
print(f"Labels count: {train_df['label'].value_counts()}")
print(f"data are heavily unbalanced")
print(f"Labels pct: {(train_df['label'].value_counts()/train_df.shape[0]*100).round(3)}")
print(f"Nans: {train_df.isna().sum()}")