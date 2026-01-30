# Write a program to visualization of each species of iris dataset using Liner Regression Model.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].apply(lambda x: iris.target_names[x])

print("First five rows of the Iris dataset:")
print(iris_df.head())




#visualization 1: Sepal Lenghth vs Sepal Width
plt.figure(figsize=(10, 6))
sns.lmplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=iris_df, markers=['o', 's', 'D'])
plt.title('Sepal Length vs Sepal Width by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()


#visualization 2: Petal Length vs Petal Width
plt.figure(figsize=(10, 6))
sns.lmplot(x='petal length (cm)', y='petal width (cm)', hue='species', data=iris_df, markers=['o', 's', 'D'])       
plt.title('Petal Length vs Petal Width by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()