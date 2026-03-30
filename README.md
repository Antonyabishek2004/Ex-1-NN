<H3>NAME : ANTONY ABISHEK K </H3> 
<H3>REGISTER NO : 212223240009</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 20/02/26</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

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

### Import Libraries

```
from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```

### Read the dataset

```
df=pd.read_csv("Churn_Modelling.csv")
```

### Checking Data

```
df.head()
df.tail()
df.columns
```

### Check the missing data

```
df.isnull().sum()
```

### Check for Duplicates

```
df.duplicated()
```

### Assigning Y
```
y = df.iloc[:, -1].values
print(y)
```

### Check for duplicates

```
df.duplicated()
```

### Check for outliers

```
df.describe()
```

### Dropping string values data from dataset

```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()
```

### Normalize the dataset

```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```

### Split the dataset

```
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```

### Training and testing model

```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT :

### Data checking :

<img width="509" height="71" alt="image" src="https://github.com/user-attachments/assets/88d7e9be-36ab-45ec-808d-6f817d15710c" />

### Missing Data :

<img width="163" height="218" alt="image" src="https://github.com/user-attachments/assets/7d4b7ce2-ca31-4d72-a768-26e67358414b" />

### Duplicates identification :

<img width="179" height="180" alt="image" src="https://github.com/user-attachments/assets/2d6c2c51-be72-434c-9f35-d300beeb31b2" />

### Values of 'Y' :

<img width="196" height="33" alt="image" src="https://github.com/user-attachments/assets/d92beb96-0813-4ae7-8b18-74838c858343" />

### Outliers :

<img width="1036" height="243" alt="image" src="https://github.com/user-attachments/assets/d8a120b5-3f50-4e8a-bc49-b9f1ebf65365" />

### Checking datasets after dropping string values data from dataset :

<img width="906" height="174" alt="image" src="https://github.com/user-attachments/assets/2245bccb-8ab0-4472-94ec-eb332940050d" />

### Normalize the dataset :

<img width="513" height="385" alt="image" src="https://github.com/user-attachments/assets/1cca5f33-b60b-4e0d-8e1e-3d3388148412" />

### Split the dataset :

<img width="301" height="128" alt="image" src="https://github.com/user-attachments/assets/fdb6365d-a32d-41ec-9264-f3f7e1894211" />

### Training and testing model :

<img width="343" height="339" alt="image" src="https://github.com/user-attachments/assets/d3a4605b-7b22-4c3b-9bd8-05f6495c2094" />

## RESULT :

Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


