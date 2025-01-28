print("------MAYANK Goel-----1/22/FET/BCS/161-----")
import pandas as pd 
# loading the dataset 
df = pd.read_excel("C:/Users/Mayank Goel/Desktop/ML/Student-List.xlsx")

# rename the given parameters i.e. rename the column name for all columns
df = df.rename(columns={"Branch /Course":"Course","University Registration No. (Issued by R&S Branch)":"Registration No.",
                        "Date of Admission in the Present Class/Course":"Admission Date",
                        "Spelling strictly as per Registration record in Capital Letters (Issued by R&S Branch)-SN":"Student Name",
                        "Spelling strictly as per Registration record in Capital Letters (Issued by R&S Branch)-FN":"Father Name",
                        "Aadhaar No. of Student":"Aadhaar No.","Updated Mobile No. of Student":"Mobile No.",
                        "Updated E-mail ID":"E-Mail ID","Total Due Fee for Current Semester / (Previous semester if pending)":"Due Fee Pending",
                        "Fee Paid for Current Semester / (Previous semester if pending)":"Fee Paid Pending"})

# count the total no.of rows and total no .of parameters in the dataset 
print(df)
rows, columns = df.shape
print(f"Total number of rows: {rows}")
print(f"Total number of columns (parameters): {columns}",'\n')

# find no. of males and females
gender_cnt = df["Gender"].value_counts()
print("Number of Males and Females:")
print(gender_cnt,'\n')

# state wise count 
state_cnt = df["State"].value_counts()
print("State wise count of students:")
print(state_cnt,'\n')

# find category count 
category_cnt = df["Category"].value_counts()
print("Category wise count of students:")
print(category_cnt,'\n')

# count null values with respect to each column 
null_cnt = df.isnull().sum()
print("Count of null vlues with respect to each column ")
print(null_cnt,'\n')

# total count of pending dues
pending_dues = df['Due Fee Pending'].value_counts()
print("Total count of pending dues:")
print(pending_dues,'\n')

# list of students with pending dues along with amount due 
df1 = df[['Student Name','Due Fee Pending']]
std_cnt = df1[df1['Due Fee Pending'] > 0]
print("List of students with pending dues along with amount due:")
print(std_cnt,'\n')

# extract specific columns like name , roll no , mentor name 
df2 = df[['Student Name','Roll No.','mentors']]
print(df2,'\n')

# list of student where aadhaar number is not available
if df['Aadhaar No.'].isnull().any():
    print("List of students where Aadhaar number is not available:")
    df3 = df[df['Aadhaar No.'].isnull()]['Student Name']
    print(df3,'\n')

# renaming the column names in the excel sheet 
df.to_excel("C:/Users/Mayank Goel/Desktop/ML/Student-List.xlsx", index=False)
print("Columns have been renamed and saved back to the Excel file.",'\n')