import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/Mayank Goel/OneDrive/Desktop/ML/Electric_Vehicle_Population_Data.csv")
# print(df)

# print first and last five entries
print(df.head(),'\n',df.tail(),'\n')

print(df.info(),'\n')
# print(df.describe())  
print(df.nunique(),'\n')

# get no.of null values with respect to each feature
print(df.isnull().sum())

# get percentage of null data in each column
print((df.isnull().sum()/(len(df)))*100,'\n')

# creating a new feature from the existing one
from datetime import date
date.today().year
df['Car_Age']=date.today().year-df['Model Year']
print(df.head())

# getting count and name of unique values for a particular column
print(df.Make.unique())
print(df.Make.nunique())

# get mean,max,min std deviation and such data for a particular column
print(df.describe().T)

# get mean,max,min std deviation and such data for all features/parameters
print(df.describe(include='all').T)

# separating the features based on categorical data and numerical data
cat_cols=df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)

# performing univariate analysis on numercial variable
print('Car_Age')
print('Skew :', round(df['Car_Age'].skew(), 2))
plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
df['Car_Age'].hist(grid=False)
plt.ylabel('count')
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Car_Age'])
plt.show()

# performing univariate analysis on categorical variable
fig, axes = plt.subplots(figsize = (18, 18))
fig.suptitle('Bar plot for categorical variables in the dataset')
sns.countplot( x = 'Electric Vehicle Type', data = df, color = 'blue', 
              order = df['Electric Vehicle Type'].value_counts().index)
plt.show()