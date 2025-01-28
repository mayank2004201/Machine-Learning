print('--Mayank Goel----1/22/FET/BCS/161---')
import pandas as pd
# 1.1 Introduction to Pandas
data = [10, 20, 30, 40, 50]
series = pd.Series(data)
print(series,'\n')   
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']}

df = pd.DataFrame(data)
print(df,'\n')

# 1.2 Data Selection and Indexing
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Score': [85, 90, 88, 95]
}
df = pd.DataFrame(data)
print(df.loc[0],'\n') 
print(df.loc[df['Age'] > 30],'\n')
print(df.iloc[3],'\n')  


# 1.3 Data Cleaning and Preprocessing
import numpy as np
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, np.nan, 35],
    'City': ['New York', np.nan, 'Los Angeles']
}
df = pd.DataFrame(data)
print(df.isnull(),'\n') 
print(df.isnull().sum(),'\n') 
df['Age'] = df['Age'].fillna(df['Age'].mean())  
print(df,'\n')
df_dropped = df.dropna()  
print(df_dropped,'\n')

# 1.4 Aggregation and Grouping
data = {
    'Department': ['HR', 'IT', 'IT', 'HR'],
    'Salary': [50000, 60000, 55000, 52000]
}
df = pd.DataFrame(data)
grouped = df.groupby('Department')['Salary'].mean()
print(grouped,'\n')
grouped_sum = df.groupby('Department')['Salary'].sum()
print(grouped_sum,'\n')
grouped_agg = df.groupby('Department')['Salary'].agg(['mean', 'sum', 'max', 'min'])
print(grouped_agg,'\n')

# 1.5 Data Merging and Joining
data1 = {
    'Employee': ['Alice', 'Bob'],
    'Department': ['HR', 'IT']
}
df1 = pd.DataFrame(data1)
data2 = {
    'Employee': ['Alice', 'Bob'],
    'Salary': [50000, 60000]
}
df2 = pd.DataFrame(data2)

print("DataFrame 1:")
print(df1,'\n')
print("\nDataFrame 2:")
print(df2,'\n')
merged_inner = pd.merge(df1, df2, on='Employee', how='inner')
print("\nMerged DataFrame (Inner Join):")
print(merged_inner,'\n')
concatenated_rows = pd.concat([df1, df2], axis=0)
print("\nConcatenated DataFrame (Rows):")
print(concatenated_rows,'\n')

# 1.6 Filtering and Sorting 
data = {
    'Employee': ['Alice', 'Bob'],
    'Department': ['HR', 'IT'],
    'Salary': [50000, 60000]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df,'\n')
filtered_df = df[df['Salary'] > 55000]
print("\nFiltered DataFrame (Salary > 55000):")
print(filtered_df,'\n')
sorted_df = df.sort_values(by='Salary', ascending=True)
print("\nSorted DataFrame (by Salary, Ascending):")
print(sorted_df,'\n')

# 1.7 Handling CSV and Excel Files
data = {
    'Employee': ['Alice', 'Bob'],
    'Salary': [50000, 60000]
}
df = pd.DataFrame(data)
df.to_csv('output.csv', index=False)  
df_csv = pd.read_csv('output.csv')
print("DataFrame from CSV:")
print(df_csv)

# 1.8 Advanced Data Manipulations
data = {
    'Employee': ['Alice', 'Bob'],
    'Department': ['HR', 'IT'],
    'Salary': [50000, 60000],
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df,'\n')
pivot_table = pd.pivot_table(df, values='Salary', index='Department', columns='Employee', aggfunc='mean')
print("\nPivot Table:")
print(pivot_table,'\n')
df_multi = df.set_index(['Department', 'Employee'])
print("\nMulti-Indexed DataFrame:")
print(df_multi,'\n')
df_melted = pd.melt(df, id_vars=['Employee'], value_vars=['Salary'])
print("\nReshaped Data with Melt:")
print(df_melted,'\n')




