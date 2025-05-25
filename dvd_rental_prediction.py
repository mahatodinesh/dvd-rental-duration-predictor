"""
DVD Rental Duration Prediction Project
--------------------------------------

Objective:
----------
Q-A DVD rental company wants to predict the number of days a customer will rent a DVD based on various features.
This helps in optimizing inventory planning and improving operational efficiency.

Goal:
-----
Develop a regression model that accurately predicts DVD rental durations and achieves a Mean Squared Error (MSE) of 3 or less on the test set.

Dataset:
--------
The dataset provided is `rental_info.csv` with the following columns:

- rental_date: The timestamp when the DVD was rented.
- return_date: The timestamp when the DVD was returned.
- amount: The total amount paid for the rental.
- amount_2: Square of the amount.
- rental_rate: Rate at which the DVD is rented.
- rental_rate_2: Square of the rental rate.
- release_year: The year the movie was released.
- length: Duration of the movie in minutes.
- length_2: Square of the movie length.
- replacement_cost: Cost to replace the DVD.
- special_features: Text data about special features (e.g., 'Deleted Scenes', 'Behind the Scenes').
- NC-17, PG, PG-13, R: Dummy variables for movie rating.

Approach:
---------
1. Feature engineering (e.g., computing rental duration, extracting special features).
2. Feature selection using Lasso regression.
3. Model building using OLS regression on selected features.
4. Model tuning and selection using Random Forest with hyperparameter optimization.

Outcome:
--------
- Random Forest Regressor with optimized hyperparameters achieved the best performance.
- Final MSE is printed to assess prediction accuracy.

"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Read in data
df_rental = pd.read_csv("rental_info.csv")

# Add information on rental duration
df_rental["rental_length"] = pd.to_datetime(df_rental["return_date"]) - pd.to_datetime(df_rental["rental_date"])
df_rental["rental_length_days"] = df_rental["rental_length"].dt.days

# Add dummy for deleted scenes
df_rental["deleted_scenes"] = np.where(df_rental["special_features"].str.contains("Deleted Scenes"), 1, 0)

# Add dummy for behind the scenes
df_rental["behind_the_scenes"] = np.where(df_rental["special_features"].str.contains("Behind the Scenes"), 1, 0)

# Choose columns to drop
cols_to_drop = ["special_features", "rental_length", "rental_length_days", "rental_date", "return_date"]

# Split into feature and target sets
X = df_rental.drop(cols_to_drop, axis=1)
y = df_rental["rental_length_days"]

# Further split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Create the Lasso model
lasso = Lasso(alpha=0.3, random_state=9)
lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_

# Perform feature selection by choosing columns with positive coefficients
X_lasso_train, X_lasso_test = X_train.iloc[:, lasso_coef > 0], X_test.iloc[:, lasso_coef > 0]

# Run OLS models on lasso-selected features
ols = LinearRegression()
ols.fit(X_lasso_train, y_train)
y_test_pred = ols.predict(X_lasso_test)
mse_lin_reg_lasso = mean_squared_error(y_test, y_test_pred)

# Random forest hyperparameter space
param_dist = {
    'n_estimators': np.arange(1, 101, 1),
    'max_depth': np.arange(1, 11, 1)
}

# Create a random forest regressor
rf = RandomForestRegressor()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, cv=5, random_state=9)
rand_search.fit(X_train, y_train)

# Get the best hyperparameters
hyper_params = rand_search.best_params_

# Run the random forest with the best hyperparameters
rf = RandomForestRegressor(n_estimators=hyper_params["n_estimators"], max_depth=hyper_params["max_depth"], random_state=9)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
mse_random_forest = mean_squared_error(y_test, rf_pred)

# Choose the best model
best_model = rf
best_mse = mse_random_forest

print("Best MSE:", best_mse)
