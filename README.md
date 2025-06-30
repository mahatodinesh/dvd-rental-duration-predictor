#  DVD Rental Duration Prediction Project

##  Objective

Q-A DVD rental company wants to predict the number of days a customer will rent a DVD based on various features.  
This helps in optimizing inventory planning and improving operational efficiency.

---

##  Goal

Develop a regression model that accurately predicts DVD rental durations and achieves a **Mean Squared Error (MSE) of 3 or less** on the test set.

---

## Dataset

The dataset provided is `rental_info.csv` with the following columns:

- `rental_date`: The timestamp when the DVD was rented.
- `return_date`: The timestamp when the DVD was returned.
- `amount`: The total amount paid for the rental.
- `amount_2`: Square of the amount.
- `rental_rate`: Rate at which the DVD is rented.
- `rental_rate_2`: Square of the rental rate.
- `release_year`: The year the movie was released.
- `length`: Duration of the movie in minutes.
- `length_2`: Square of the movie length.
- `replacement_cost`: Cost to replace the DVD.
- `special_features`: Text data about special features (e.g., 'Deleted Scenes', 'Behind the Scenes').
- `NC-17`, `PG`, `PG-13`, `R`: Dummy variables for movie rating.

---

##  Approach

1. Feature engineering (e.g., computing rental duration, extracting special features).
2. Feature selection using **Lasso regression**.
3. Model building using **OLS regression** on selected features.
4. Model tuning and selection using **Random Forest** with hyperparameter optimization.

---

## Outcome

- **Random Forest Regressor** with optimized hyperparameters achieved the best performance.
- Final **Mean Squared Error (MSE)** was printed to assess prediction accuracy and met the target of ≤ 3.

---

## Files

- `dvd_rental_prediction.py`: Main Python script for data preprocessing, modeling, and evaluation.
- `rental_info.csv`: Input dataset (if included).
- `README.md`: Project documentation (this file).

