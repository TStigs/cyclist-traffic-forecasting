# ==============================================================================
# Script: 04_future_forecast.R
# Purpose: Generate true out-of-sample forecasts using OLS, ARIMAX, and XGBoost
# ==============================================================================

# Load required libraries
library(readr)
library(dplyr)
library(forecast)
library(xgboost)

# 1. Load Processed Data
# ------------------------------------------------------------------------------
bikes_weather <- read_csv("data/bikes_processed_weather.csv") %>%
  mutate(date = as.Date(date))

# Separate into full training set and the future forecast target set (where cyclists is NA)
prod_train <- bikes_weather %>% filter(!is.na(cyclists))
prod_test  <- bikes_weather %>% filter(is.na(cyclists))

# Define the full matrix of exogenous regressors
month_cols <- paste0("month_", sprintf("%02d", 1:12))
day_cols   <- paste0("day_", c("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"))
xreg_cols  <- c("mean_temp", "precip_mm", "wind_mean_kmh", 
                setdiff(month_cols, "month_06"), 
                setdiff(day_cols, "day_Monday"))

# Prepare matrices for production
X_train_full <- as.matrix(prod_train[, xreg_cols])
class(X_train_full) <- "numeric"

X_test_full  <- as.matrix(prod_test[, xreg_cols])
class(X_test_full)  <- "numeric"

# 2. Train Models on Full Historical Dataset
# ------------------------------------------------------------------------------
cat("Training production models on full historical dataset...\n")

# Model A: OLS Regression Baseline
fit_lm_prod <- lm(cyclists ~ mean_temp + precip_mm + wind_mean_kmh, data = prod_train)

# Model B: ARIMAX
train_ts_full <- ts(prod_train$cyclists, frequency = 7)
fit_arimax_prod <- Arima(
  train_ts_full,
  xreg     = X_train_full,
  order    = c(14, 1, 1),
  seasonal = c(1, 0, 0),
  method   = "CSS-ML"
)

# Model C: XGBoost
fit_xgb_prod <- xgboost(
  data      = X_train_full, 
  label     = prod_train$cyclists, 
  max_depth = 4, 
  eta       = 0.1, 
  nrounds   = 50, 
  objective = "reg:squarederror",
  verbose   = 0
)

# 3. Generate Future Forecasts for November 2025
# ------------------------------------------------------------------------------
prod_test$pred_OLS     <- predict(fit_lm_prod, newdata = prod_test)
prod_test$pred_ARIMAX  <- as.numeric(forecast(fit_arimax_prod, xreg = X_test_full, h = nrow(prod_test))$mean)
prod_test$pred_XGBoost <- predict(fit_xgb_prod, newdata = X_test_full)

# Lower bound floor to prevent negative counts from linear model calculations
prod_test <- prod_test %>%
  mutate(across(starts_with("pred_"), ~ pmax(.x, 0)))

# Save forecast outputs to CSV for the evaluation step
write_csv(prod_test %>% select(date, mean_temp, precip_mm, starts_with("pred_")), 
          "results/future_forecast_predictions.csv")

cat("Future forecasts successfully generated and saved to 'results/future_forecast_predictions.csv'!\n")


