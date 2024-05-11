import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
df =sns.load_dataset('mpg')
df.isnull().sum()
df.dropna(inplace=True)
x = df[['displacement','horsepower','weight','acceleration']]
y = df.mpg
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.15,random_state=42)
from ctypes import LibraryLoader
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x,y)
model.score(x_test,y_test)
df.isnull().sum()
from sklearn.tree import DecisionTreeRegressor
model2= DecisionTreeRegressor(criterion='poisson',random_state=0)
model2.fit(x_train,y_train)
model2.score(x_test,y_test)
import pickle
filename ='mpg_regression.sav'
pickle.dump(model,open(filename,'wb'))
