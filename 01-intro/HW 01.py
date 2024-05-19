# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import root_mean_squared_error

# %%
GREEN_TAXI_JAN_2023 = "../data/yellow_tripdata_2023-01.parquet"
GREEN_TAXI_FEB_2023 = "../data/yellow_tripdata_2023-02.parquet"

# %%
df_jan = pd.read_parquet(GREEN_TAXI_JAN_2023)
df_feb = pd.read_parquet(GREEN_TAXI_FEB_2023)

# %% [markdown]
# ## Q1. Downloading the data
#
# We'll use [the same NYC taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
# but instead of "**Green** Taxi Trip Records", we'll use "**Yellow** Taxi Trip Records".
#
# Download the data for January and February 2023.
#
# Read the data for January. How many columns are there?
#
# * 16
# * 17
# * 18
# * 19

# %%
# how many columns
df_jan.shape

# 19

# %% [markdown]
# ## Q2. Computing duration
#
# Now let's compute the `duration` variable. It should contain the duration of a ride in minutes. 
#
# What's the standard deviation of the trips duration in January?
#
# * 32.59
# * 42.59
# * 52.59
# * 62.59

# %%
df_jan["duration_in_minutes"] = (df_jan["tpep_dropoff_datetime"]-df_jan["tpep_pickup_datetime"]).dt.total_seconds() / 60
df_jan["duration_in_minutes"].std()

# %% [markdown]
# ## Q3. Dropping outliers
#
# Next, we need to check the distribution of the `duration` variable. There are some outliers. Let's remove them and keep only the records where the duration was between 1 and 60 minutes (inclusive).
#
# What fraction of the records left after you dropped the outliers?
#
# * 90%
# * 92%
# * 95%
# * 98%
#

# %%
before_dropping = len(df_jan)

df_jan = df_jan.query("1 <= duration_in_minutes <= 60")

after_dropping = len(df_jan)

print(f"Records left: {after_dropping/before_dropping * 100:.2f}%")

# %% [markdown]
# ## Q4. One-hot encoding
#
# Let's apply one-hot encoding to the pickup and dropoff location IDs. We'll use only these two features for our model. 
#
# * Turn the dataframe into a list of dictionaries (remember to re-cast the ids to strings - otherwise it will 
#   label encode them)
# * Fit a dictionary vectorizer 
# * Get a feature matrix from it
#
# What's the dimensionality of this matrix (number of columns)?
#
# * 2
# * 155
# * 345
# * 515
# * 715
#

# %%
categorical = ["PULocationID", "DOLocationID"]

# recast the ids into strings
df_jan[categorical] = df_jan[categorical].astype(str)

# convert dataframe to list of dictionaries
categorical_dict = df_jan[categorical].to_dict(orient="records")

# fit a dictionary vectorizer
vec = DictVectorizer(sparse=True)
X_train = vec.fit_transform(categorical_dict)

# %%
X_train.shape

# %% [markdown]
# ## Q5. Training a model
#
# Now let's use the feature matrix from the previous step to train a model. 
#
# * Train a plain linear regression model with default parameters 
# * Calculate the RMSE of the model on the training data
#
# What's the RMSE on train?
#
# * 3.64
# * 7.64
# * 11.64
# * 16.64

# %%
target = "duration_in_minutes"
y_train = df_jan[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

RMSE = root_mean_squared_error(y_train, y_pred)

print(f"{RMSE:.2f}")

# %% [markdown]
# ## Q6. Evaluating the model
#
# Now let's apply this model to the validation dataset (February 2023). 
#
# What's the RMSE on validation?
#
# * 3.81
# * 7.81
# * 11.81
# * 16.81

# %%
# run the same transformation on validation data

df_feb["duration_in_minutes"] = (df_feb["tpep_dropoff_datetime"]-df_feb["tpep_pickup_datetime"]).dt.total_seconds() / 60
df_feb = df_feb.query("1 <= duration_in_minutes <= 60")

categorical = ["PULocationID", "DOLocationID"]

# recast the ids into strings
df_feb[categorical] = df_feb[categorical].astype(str)

# convert dataframe to list of dictionaries
categorical_dict = df_feb[categorical].to_dict(orient="records")

# fit a dictionary vectorizer
X_train = vec.transform(categorical_dict)

y_test = df_feb[target].values

y_pred = lr.predict(X_train)

RMSE = root_mean_squared_error(y_test, y_pred)

print(f"{RMSE:.2f}")

# %% [markdown]
# ## Submit the results
#
# * Submit your results here: https://courses.datatalks.club/mlops-zoomcamp-2024/homework/hw1
# * If your answer doesn't match options exactly, select the closest one
