import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

data = pd.read_csv("dataset.csv")
data.head()

data['Target'] = data['Target'].map({
    'Dropout':0,
    'Enrolled':1,
    'Graduate':2
})

print(data["Target"].unique())

data.info()

data.corr()['Target']
print(data.corr()['Target'])

new_data = data.copy()
new_data = new_data.drop(columns=['Nacionality', 
                                  'Mother\'s qualification', 
                                  'Father\'s qualification', 
                                  'Educational special needs', 
                                  'International', 
                                  'Curricular units 1st sem (without evaluations)',
                                  'Unemployment rate', 
                                  'Inflation rate'], axis=1)
new_data.info()
      
new_data['Target'].value_counts()

correlations = data.corr()['Target']
top_7_features = correlations.abs().nlargest(8).index
top_7_corr_values = correlations[top_7_features]

plt.figure(figsize=(8, 11))
plt.bar(top_7_features, top_7_corr_values)
plt.xlabel('Features')
plt.ylabel('Correlation with Target')
plt.title('Top 7 Features with Highest Correlation to Target')
plt.xticks(rotation=45)
plt.show()


#Scatter plots of relationships to the target

#Age vs. Target
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Age at enrollment', data=new_data)
plt.xlabel('Target')
plt.ylabel('Age at enrollment')
plt.title('Relationship between Age at enrollment and Target')
plt.show()

#Curricular units 2nd semester (approved)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Curricular units 2nd sem (approved)', data=new_data)
plt.xlabel('Target')
plt.ylabel('Curricular units 2nd sem (approved)')
plt.title('Relationship between Curricular units 2nd sem (approved) and Target')
plt.show()

#Curricular units 2nd semester (grade)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Curricular units 2nd sem (grade)', data=new_data)
plt.xlabel('Target')
plt.ylabel('Curricular units 2nd sem (grade)')
plt.title('Relationship between Curricular units 2nd sem (grade) and Target')
plt.show()

#Curricular units 1st semester (approved)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Curricular units 1st sem (approved)', data=new_data)
plt.xlabel('Target')
plt.ylabel('Curricular units 1st sem (approved)')
plt.title('Relationship between Curricular units 1st sem (approved) and Target')
plt.show()

#Curricular units 1st semester (grade)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Curricular units 1st sem (grade)', data=new_data)
plt.xlabel('Target')
plt.ylabel('Curricular units 1st sem (grade)')
plt.title('Relationship between Curricular units 1st sem (grade) and Target')
plt.show()

#Tuition fees up to date
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Tuition fees up to date', data=new_data)
plt.xlabel('Target')
plt.ylabel('Tuition fees up to date')
plt.title('Relationship between Tuition fees up to date and Target')
plt.show()

#Scholarship holder
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Scholarship holder', data=new_data)
plt.xlabel('Target')
plt.ylabel('Scholarship holder')
plt.title('Relationship between Scholarship holder and Target')
plt.show()

#Data training
X = new_data.drop('Target', axis=1)
y = new_data['Target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
svm = svm.SVC(kernel='linear',probability=True)
knn.fit(X_train,y_train)
svm.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")
y_pred = svm.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")
