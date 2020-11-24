# import libs
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
%matplotlib inline
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# load data
df = pd.read_csv('/Users/sebastianmedina/Desktop/Machine Learning Excercises/Real Estate Prediction/real_estate.csv')
df.head()

# data types
df.info()

# describe data
df.describe()

# check for missing values
print(df.isnull().sum())

# rename features
rename_dict = {
    "X1 transaction date": "transaction_dt",
    "X2 house age": "house_age",
    "X3 distance to the nearest MRT station": "near_mrt_station",
    "X4 number of convenience stores": "nr_stores",
    "X5 latitude": "lat",
    "X6 longitude": "lon",
    "Y house price of unit area": "price_unit_area"
}

for k,v in rename_dict.items():
    df.rename(columns = {k:v}, inplace = True)

df.head()

# histogram of features
df.hist(bins=50,figsize=(20,25))
plt.show()

# pearson's std feature correlation (r) coeff
corr_matrix = df.corr()
corr_matrix["price_unit_area"].sort_values(ascending=False)

# scatter matrix for features
# from pandas.plotting import scatter_matrix
# attributes = ["price_unit_area","nr_stores","lat","lon","transaction_dt","house_age"]    
# scatter_matrix(df[attributes], figsize=(16,10))

# pairplot of features
sns.pairplot(df)

# based on correlation coeff & scatter plot - most important features: nr_stores, lat, & lon

# create linear mdoel
sns.set(color_codes=True)
sns.lmplot(x='nr_stores', y='price_unit_area',data=df)

# create linear mdoel
sns.set(color_codes=True)
sns.lmplot(x='nr_stores', y='price_unit_area',data=df)

# train / test data
X = df[['nr_stores']]
y = df['price_unit_area']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train, y_train)

# coefficient / intercept
print('coefficient is: ', lm.coef_)
print('intercept is : ', lm.intercept_)

# predict test data
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.ylabel('Predicted')
plt.xlabel('Y test')

# evaluate the model
import sklearn.metrics as metrics
print('MAE: {}'.format(metrics.mean_absolute_error(y_test, predictions)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, predictions)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predictions))))

# explore residuals
sns.distplot((y_test-predictions))

print('unit area price ', lm.predict([[6]]))
# y = intercept + (coeff * x)
