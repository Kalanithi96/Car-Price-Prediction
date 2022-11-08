
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import seaborn as sns
sns.set()


df=pd.read_csv("auto.csv")
df=pd.DataFrame(df)

plt.figure(figsize=(10,7))
plt.title('Distribution of Car Prices')
df['price'].hist()
plt.savefig('Graphs\Price Distribution.png')
def plots(data, fig):
  plt.subplot(4,2,fig)
  sns.countplot(x = data, data = df)
  plt.title(data+ ' Histogram')
  plt.savefig('Graphs\graphs.png')


plt.figure(figsize=(10,27))

plots('fueltype', 1)
plots('carbody', 2)
plots('aspiration', 3)
plots('enginelocation', 4)

# 1.   Most cars are use `gas` as their fuel source.
# 2.   A high number of the cars in our data are `sedans`.
# 3.   Majority of cars are `std` typed in aspiration.
# 4.   Most cars have their engines located that the `front`.

def box(dat, fig,flag=True):
  plt.subplot(4, 2, fig)
  sns.boxplot(x = dat, y = df['price'], data = df)
  plt.title(dat+ ' Vs Price')
  if flag:
      plt.savefig('Graphs\price vs .png')
  else:
      plt.savefig('Graphs\price(2) vs .png')


plt.figure(figsize=(15,20))
box('fueltype', 1)
box('carbody', 2)
box('cylindernumber', 3)
box('drivewheel', 4)

box('doornumber', 1,False)
box('fuelsystem', 2)
box('aspiration', 3)
box('enginelocation', 4)

# 1.   Diesel cars cost more than cars that use gas.
# 2.   Hardtop and convertible styled cars have the highest prices.
# 3.   Car prices seems to have a very strong correlation with the number of cylinders.
# 4.   rwd cars cost more than 4wd and fwd.
# 5.   The number of doors don't determine the price of cars.
# 6.   `ldi` and `mpfi` fuel systems have the highest prices.
# 7.    `turbo` cars cost more than `std` cars
# 8.    Cars with the engine located at the the rear cost more than those with their engines in the front.

def pairs(x, fig):
  plt.subplot(4, 2, fig)
  plt.scatter(df[x], df['price'])
  plt.title(x+ ' Vs Price')
  plt.xlabel(x)
  plt.ylabel('Price')
  plt.savefig('Graphs\pricevsperfomance.png')

plt.figure(figsize=(15,20))
pairs('horsepower', 1)
pairs('peakrpm', 2)
pairs('citympg', 3)
pairs('highwaympg', 4)
# 1.   `horsepower` has a strong positive relationship with car prices.
# 2.   `peak-rpm` has a weak relationship with car prices.
# 3.   `city-mpg` and `highway-mpg` have strong negative relationships with car prices.


df.loc[ df["doornumber"] == "two", "doornumber"] = 2
df.loc[ df["doornumber"] == "four", "doornumber"] = 4
df.loc[ df["drivewheel"] == "fwd" , "drivewheel"] = 0
df.loc[ df["drivewheel"] == "rwd" , "drivewheel"] = 0
df.loc[ df["drivewheel"] == "4wd", "drivewheel"] = 1
df.loc[df["cylindernumber"]=="four","cylindernumber"]=4
df.loc[df["cylindernumber"]=="six","cylindernumber"]=6
df.loc[df["cylindernumber"]=="twelve","cylindernumber"]=12
df.loc[df["cylindernumber"]=="three","cylindernumber"]=3
df.loc[df["cylindernumber"]=="five","cylindernumber"]=5
df.loc[df["cylindernumber"]=="eight","cylindernumber"]=8
df.loc[df["cylindernumber"]=="two","cylindernumber"]=2
df.loc[df["carbody"]=="convertible","carbody"]=2
df.loc[df["carbody"]=="hatchback","carbody"]=0
df.loc[df["carbody"]=="sedan","carbody"]=1
df.loc[df["carbody"]=="wagon","carbody"]=0.5
df.loc[df["carbody"]=="hardtop","carbody"]=1.5
df[['doornumber']] = df[['doornumber']].apply(pd.to_numeric)
df[['cylindernumber']] = df[['cylindernumber']].apply(pd.to_numeric)
df[['drivewheel']] = df[['drivewheel']].apply(pd.to_numeric)
df[['carbody']] = df[['carbody']].apply(pd.to_numeric)
x=df.drop(['car_ID','symboling','CarName','fueltype','aspiration','enginelocation','fuelsystem','price','enginetype'],axis=1)
y=df['price']
reg = linear_model.LinearRegression()
reg.fit(x,y)
y_pred=reg.predict(x)
df['Y prediction']=y_pred
print("R sqaured value is %.2f"%r2_score(y,y_pred))
