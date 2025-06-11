# Time Series Forecasting 

In this project I am trying to fit a model that will be able to predict actual values in the test set. 

## Approaches

- Approach 1 -- fit daily ARIMA model for each time series and then fit a regression model on the residuals of the first model to correct arima predictions on the test set
- Approach 2 -- fit monthly ARIMA model for each time series that captures the trend (trained on average sales per day on each month) and then fit a regression model on deviations of the actual values from the actual mean of that month.  Then use both models to get final prediction
- Approach 3 -- Use NeuralProphet model that automatically fits separate trends, seasonality and actual values for each model which are all combined in a single model


## HP tunning 
A notebook was created for hyperparameter tuning however it was not run because it is quite time consuming and demands high resources. Nevertheless, hp tuning is the last step and in practice, the benefit for it is quite marginal. 

## Model Evaluation

Since the scale of time series was different for some store - product pairs, I decided to use MAPE to compare models as it is quite comparable across different models and without having to worry too much about the scale of the series (ranging from 100 - 1000). Also in the Kaggle site the poster of the problem had suggested to use MAPE as the validation metric. 

Feature importance does not make a lot of sense with ARIMA models, we could check the coefficients and their significance, but with high lags, this is not very interpretable. For lightgbm model, I used the built-in method of feature importance, however we could use the Shapley value which calculates the change in accuracy/loss when we add a feature to each possible subset of features that does not include this feature.

## MLFLOW

I experimented also with Mlflow, where I log 2 experiments with NeuralProphet. I could do this also for the other approaches, but this part was done for demonstration purposes. 
