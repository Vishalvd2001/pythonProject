import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/MAHADEV/Downloads/tatasteel (1).csv")
print(df)

df["Date"] = pd.to_datetime(df["Date"])
df.drop(['PREV. CLOSE'], axis=1, inplace=True)

print(df)
print(df.isna().any())
print(df.columns)

print(df.info())
x = df[['OPEN', 'HIGH', 'LOW']]
y = df['close']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train)

mod = LinearRegression()
mod.fit(x_train, y_train)

pred = mod.predict(x_test)

print(pred)

print(pd.DataFrame({'ACTUAL_PRICE': y_test, 'Predict': pred}))

print(mod.score(x_test, y_test))

import matplotlib.pyplot as plt

#create figure
plt.figure()

#define width of candlestick elements
width = .4
width2 = .05

#define up and down prices
up = prices[prices.close>=prices.open]
down = prices[prices.close<prices.open]

#define colors to use
col1 = 'green'
col2 = 'red'

#plot up prices
plt.bar(up.index,up.close-up.open,width,bottom=up.open,color=col1)
plt.bar(up.index,up.high-up.close,width2,bottom=up.close,color=col1)
plt.bar(up.index,up.low-up.open,width2,bottom=up.open,color=col1)

#plot down prices
plt.bar(down.index,down.close-down.open,width,bottom=down.open,color=col2)
plt.bar(down.index,down.high-down.open,width2,bottom=down.open,color=col2)
plt.bar(down.index,down.low-down.close,width2,bottom=down.close,color=col2)

#rotate x-axis tick labels
plt.xticks(rotation=45, ha='right')

#display candlestick chart
plt.show()
