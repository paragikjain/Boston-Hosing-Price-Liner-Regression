import numpy as np #scientific computation
import pandas as pd #high performance and easy to use data structure
import scipy.stats as stat #maths based library
import matplotlib.pyplot as plt #graph and chart based library
import sklearn #machine learning library
import statsmodels.api as sm #statistics based library
import seaborn as sns #high level visualization library
from matplotlib import rcParams
import sklearn.cross_validation



sns.set_style("whitegrid")
sns.set_context("poster")


###load data from sklearn.dataset
from sklearn.datasets import load_boston
boston=load_boston()


###data analysis
#print(boston.keys())
#print(boston.data.shape)
#print(boston.feature_names) #colume name for each kind of data
#print(boston.target) #this is price colume and we are also predicting price
#print(boston.data) #all value for data
#print(boston.DESCR) #description of data

##insert data to pandas dataframe

bos=pd.DataFrame(boston.data)
bos.columns=boston.feature_names## adding feature to pandas table
bos['price']=boston.target ##adding the target to the pandas table
#print(bos.head())
#print(bos.describe())

##split the dataset into train and test dataset
y=bos['price'];
x=bos.drop('price',axis=1)

##finally spliting data into train and test dataset
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(x,y, test_size = 0.33, random_state = 5)
#print(X_train)
#print(Y_train)


###Finally liner regression for the dataset

from sklearn.linear_model import LinearRegression
ln=LinearRegression()
ln.fit(X_train,Y_train) 
out=ln.predict(X_test)
plt.scatter(Y_test,out,color=red)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")



