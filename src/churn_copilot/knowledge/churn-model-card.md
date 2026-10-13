# Churn model card

## Purpose
The churn model estimates the probability that a customer leaves within the next 3 months. It supports retention agents; it does not make decisions on its own.

## Model
Logistic regression on contract, tenure, monthly charge, data usage, dropped-call rate, support calls, payment method, plan, city, family bundle and age. A gradient-boosting model was evaluated as a challenger and did not perform better on the test set.

## Risk bands
Low risk: probability below 0.25. Medium risk: 0.25 to 0.5. High risk: 0.5 or higher.

## Known limitations
Scores are less reliable for customers with less than 3 months of tenure. City has little direct effect once network quality is included; differences between cities are mostly explained by dropped-call rate. The model is retrained quarterly and monitored for drift.
