# INTRODUCTION TO KAGGLE & DATA PREPROCESSING

## NAME : ANTONY ABISHEK K

## REGISTER NO. : 212223240009

## EX. NO.1

## DATE : 16/02/26

## AIM :

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED :

Hardware – PCs

Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT :

**Kaggle :**

Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing :**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.

Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.

Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM : 

STEP 1:Importing the libraries<BR>

STEP 2:Importing the dataset<BR>

STEP 3:Taking care of missing data<BR>

STEP 4:Encoding categorical data<BR>

STEP 5:Normalizing the data<BR>

STEP 6:Splitting the data into test and train<BR>

##  PROGRAM :

```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))

```

## OUTPUT :

### DATASET :

<img width="1590" height="450" alt="image" src="https://github.com/user-attachments/assets/f4c13b6a-6897-4500-9f39-c9348d5ab0e4" />

### X-VALUES :

<img width="473" height="295" alt="image" src="https://github.com/user-attachments/assets/204086fa-aa34-4d40-8138-1e5edf78ea6a" />

### Y-VALUES :

<img width="302" height="141" alt="image" src="https://github.com/user-attachments/assets/625f1650-bd1b-49aa-bbf5-20c78a0c9f21" />

### NULL VALUES/MISSING VALUES :

<img width="231" height="382" alt="image" src="https://github.com/user-attachments/assets/595fabf3-f945-40d0-a988-b114fe333fd8" />
 
### REDUNDANT VALUES :

<img width="267" height="312" alt="image" src="https://github.com/user-attachments/assets/5f1fd931-83dd-4ba1-80f0-37bd9f72f2c9" />

### DATASET DESCRIPTION :

<img width="1557" height="310" alt="image" src="https://github.com/user-attachments/assets/fc7df001-3494-4e1e-832e-107f31305f48" />

### NORMALISED DATASET :

<img width="827" height="666" alt="image" src="https://github.com/user-attachments/assets/8c2bb567-d924-4d03-9231-db1fe264e019" />

### TRAINING DATA :

<img width="725" height="213" alt="image" src="https://github.com/user-attachments/assets/f69d0e81-f7b6-43d0-b246-28b417d62232" />

### TESTING DATA :

<img width="726" height="202" alt="image" src="https://github.com/user-attachments/assets/edc79241-7a53-44d6-8a5d-060e6bbf0b54" />

## RESULT :

Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


