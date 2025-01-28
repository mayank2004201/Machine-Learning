print('--Mayank Goel----1/22/FET/BCS/161---')
import numpy as np 
#2.1 create 1D & 2D arrrays using various methods
arr1 = np.array([1,2,3,4,5])
print("1D Array using np.array():",arr1)
arr2 = np.array([[1,2],[3,4]])
print("2D Array using np.array():\n",arr2)

arr1= np.zeros(5)
print("\n1D Array using np.zeros():",arr1)
arr2=np.zeros((3,4))
print("2D Array using np.zeros():\n",arr2)

arr1= np.ones(5)
print("\n1D Array using np.ones():",arr1)
arr2=np.ones((3,4))
print("2D Array using np.ones():\n",arr2)

arr1= np.arange(0,10,2)
print("\n1D Array using np.arange():",arr1)
arr2= np.arange(1,13).reshape(3,4)
print("2D Array using np.arange():\n",arr2)

# 2.2. Array Indexing and Slicing
array = np.array([10,20,30,40,50])
print("\nArray:",array)
print("Array element at 3rd index:",array[2])
print("Array after performing slicing array by removing last three elements:",array[:2])
mask = array > 30
filtered_array = array[mask]
print("Boolean mask for the above array:",mask)
print("Filtered values (values greater than 30):",filtered_array)

# 2.3. Mathematical Operations on Arrays
array = np.array([10,20,30,40,50])
x = 5
add = array+x
sub = array - x
mult = array * x
div = array / x
print("\nElement wise addition:",add)
print("Element wise subtraction:",sub)
print("Element wise multiplication:",mult)
print("Element wise division:",div)

sum = np.sum(array)
mean = np.mean(array)
std_dev = np.std(array)
print("\nSum of array:",sum)
print("Mean of array:",mean)
print("Standard deviation of array:",std_dev)

# 2.4. Matrix Operations
array = np.array([[1,2,3,4,5],[6,7,8,9,10]])
T_array= array.T
print("Original array:\n",array)
print("Transpose of the array:\n",T_array)
multiplication = np.dot(array,T_array)
print("\nMultiplication Matrix:\n",multiplication)

# 2.5. Generate Random Data and Perform Distribution Operations
uniform_array = np.random.uniform(0, 10, 5)  
random_int_array = np.random.randint(1, 100, 10) 
print("Uniform array:",uniform_array)
print("Random array:",random_int_array)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.hist(uniform_array,bins=5,alpha=0.7,label="Uniform Distribution",color='blue')
plt.hist(random_int_array,bins=10,alpha=0.7,label="Random Integers",color='red')
plt.title('Histogram of Randomly Generated Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()  

# 2.6. Linear Algebra Operations
A = np.array([[3, 2], [1, 4]])
B = np.array([5, 6])
determinant = np.linalg.det(A)
print("Determinant of A:", determinant)
x = np.linalg.solve(A, B)
print("Solution to the system of equations (Ax = b):", x)

# 2.7. Correlation and Covariance Computation
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([5, 4, 3, 2, 1])
Covariance = np.cov(array1, array2)
print("Covariance Matrix:\n", Covariance)
Correlation = np.corrcoef(array1, array2)
print("\nCorrelation Matrix:\n", Correlation)

# 2.8. Sort and Search Elements in an Array
array = np.array([10, 5, 8, 12, 3, 15])
sorted_array = np.sort(array)
print("Sorted Array:", sorted_array)
limit = 10
mod_array = array[array> limit]
print("Elements greater than",limit,':',mod_array)