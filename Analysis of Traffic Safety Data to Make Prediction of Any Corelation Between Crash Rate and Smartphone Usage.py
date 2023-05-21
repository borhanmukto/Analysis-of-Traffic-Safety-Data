

import pandas as pd
import datetime as dt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

traffic = pd.read_csv("traffic.csv")

traffic.head(6)


print(traffic.dtypes)


# convert Date to date format

traffic["Date"] = pd.to_datetime(traffic["Date"])

# inspect data types

print(traffic.dtypes)


# create line plot

sns.lineplot(x="Date", y="Crashes_per_100k", data=traffic)


traffic = traffic[traffic.Date.dt.year !=2000]

sns.boxplot(x="Crashes_per_100k", y="Season", data=traffic)


smartphones = pd.read_csv("crashes_smartphones.csv")

print(smartphones.head())

smartphones["Smartphone_Survey_Date"] = pd.to_datetime(smartphones["Smartphone_Survey_Date"])

print(smartphones.dtypes)

smartphones["Smartphone_Survey_Date"] = pd.to_datetime(smartphones["Smartphone_Survey_Date"])

sns.lineplot(x="Smartphone_Survey_Date", y="Smartphone_usage", data=smartphones)

traffic = pd.read_csv("traffic.csv")
traffic["Date"] = pd.to_datetime(traffic["Date"])

smartphones = pd.read_csv("crashes_smartphones.csv")
smartphones["Smartphone_Survey_Date"] = pd.to_datetime(smartphones["Smartphone_Survey_Date"])

sns.regplot(x="Smartphone_usage", y="Crashes_per_100k", data=smartphones)

from scipy.stats import pearsonr

traffic = pd.read_csv("traffic.csv")
traffic["Date"] = pd.to_datetime(traffic["Date"])

smartphones = pd.read_csv("crashes_smartphones.csv")

smartphones["Smartphone_Survey_Date"] = pd.to_datetime(smartphones["Smartphone_Survey_Date"])
print(smartphones.head(10))

column1 = smartphones["Smartphone_usage"]
column2 = smartphones["Crashes_per_100k"]

corr, p = pearsonr(column1, column2)

r = column1.corr(column2)



# print corr and p
print("Pearson's r =",  round(corr,3))
print("p = ", round(p,3))


# convert columns to arrays
x = smartphones['Smartphone_usage'].to_numpy().reshape(-1, 1)
y = smartphones['Crashes_per_100k'].to_numpy().reshape(-1, 1)


# initiate the linear regression model

from sklearn.linear_model import LinearRegression


lm = LinearRegression()

x = smartphones['Smartphone_usage'].to_numpy().reshape(-1, 1)
y = smartphones['Crashes_per_100k'].to_numpy().reshape(-1, 1)

lm.fit(x,y)

coef = lm.coef_[0]
intercept = lm.intercept_

print(f"The co-efficient is= {coef}")
print(f"The intercept is= {intercept}")


smartphones_2019 = smartphones[smartphones["Smartphone_Survey_Date"].dt.year == 2019]

smartphone_usage_2019 = smartphones_2019

print(smartphone_usage_2019)




# predict the crash rate in 2020 using the regression equation

coef = lm.coef_[0]
intercept = lm.intercept_

print(f"The co-efficient is= {coef}")
print(f"The intercept is= {intercept}")

# model_predict = lm.predict(smartphones[smartphones["Smartphone_Survey_Date"].dt.year == 2020])

predict_new = smartphones[smartphones["Smartphone_Survey_Date"].dt.year == 2019]

print(predict_new)

x_new = predict_new["Smartphone_usage"].to_numpy().reshape(-1,1)

# print(x_new[0])

lm_predict_2020 = lm.predict(x_new)

print(f"The Crash rate is= {np.round(lm_predict_2020,3)[0]}")


traffic = pd.read_csv("traffic.csv")
# print(traffic.head())

traffic["Date"] = pd.to_datetime(traffic["Date"])

traffic_2020 = traffic[(traffic["Date"].dt.year == 2020) & (traffic["Date"].dt.month == 2)]

print(traffic_2020)


# recreate the regression plot we made earlier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression


traffic = pd.read_csv("traffic.csv")
# print(traffic.head())

traffic["Date"] = pd.to_datetime(traffic["Date"])


smartphones = pd.read_csv("crashes_smartphones.csv")

smartphones["Smartphone_Survey_Date"] = pd.to_datetime(smartphones["Smartphone_Survey_Date"])

sns.regplot(x = "Smartphone_usage", y = "Crashes_per_100k", data = smartphones)


# add a scatter plot layer to show the actual and predicted 2020 values
## YOUR CODE HERE ##
traffic_2020 = traffic[(traffic["Date"].dt.year == 2020) & (traffic["Date"].dt.month == 2)]
print(traffic_2020)

Assumed_smartphones_2020 = smartphones[(smartphones["Smartphone_Survey_Date"].dt.year == 2019) & (smartphones["Smartphone_Survey_Date"].dt.month == 2)]

print(Assumed_smartphones_2020)

actual_x = Assumed_smartphones_2020["Smartphone_usage"].to_numpy().reshape(-1,1)
actual_y = traffic_2020["Crashes_per_100k"].to_numpy().reshape(-1,1)

print(actual_x)
print(actual_y)

lm = LinearRegression()

x = smartphones['Smartphone_usage'].to_numpy().reshape(-1, 1)
y = smartphones['Crashes_per_100k'].to_numpy().reshape(-1, 1)

lm.fit(x,y)

coef = lm.coef_[0]
intercept = lm.intercept_

predicted_y = lm.predict(actual_x)

print(predicted_y)

plt.scatter(actual_x, actual_y, s=50, color = "green", marker= "*", label="Actual_2020")
plt.scatter(actual_x, predicted_y, s=60, color= "red", marker= "^", label= "Predicted_2020")


# add legend title
plt.legend()
plt.show()


# In[ ]:


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

By adding another layer to our regression plot, we can see the difference between the predicted and real crash rates in February 2020. This allows us to see how these values compare to the rest of the dataset. 

</details>

