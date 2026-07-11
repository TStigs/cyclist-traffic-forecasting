# ==============================================================================
# Script: 03_model_testing.R
# Purpose: Simple Validation Backtest (Single Split) for ARIMAX Model
# ==============================================================================

# Load required libraries
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(forecast)
library(xgboost)

# 1. Load Processed Data
# ------------------------------------------------------------------------------
bikes_weather <- read_csv("data/bikes_processed_weather.csv") %>%
  mutate(date = as.Date(date))

# Isolate historical data with known actuals
historical_data <- bikes_weather %>% filter(!is.na(cyclists))

# 2. Define Train/Validation Split (Hold out the last 30 days)
# ------------------------------------------------------------------------------
holdout_days <- 30
split_point  <- nrow(historical_data) - holdout_days

bt_train_data <- historical_data[1:split_point, ]
bt_val_data   <- historical_data[(split_point + 1):nrow(historical_data), ]

# Initialize time series object for training
train_ts <- ts(bt_train_data$cyclists, frequency = 7)

# 3. Define Exogenous Regressors
# ------------------------------------------------------------------------------
month_cols <- paste0("month_", sprintf("%02d", 1:12))
day_cols   <- paste0("day_", c("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"))
xreg_cols  <- c("mean_temp", "precip_mm", "wind_mean_kmh", 
                setdiff(month_cols, "month_06"), 
                setdiff(day_cols, "day_Monday"))

xreg_train <- as.matrix(bt_train_data[, xreg_cols])
class(xreg_train) <- "numeric"

xreg_val   <- as.matrix(bt_val_data[, xreg_cols])
class(xreg_val)   <- "numeric"

# 4. Fit ARIMAX Model
# ------------------------------------------------------------------------------
# Using method = "CSS-ML" for faster convergence and stability
fit_backtest <- Arima(
  train_ts,
  xreg = xreg_train,
  order = c(14, 1, 1),
  seasonal = c(1, 0, 0),
  method = "CSS-ML"
)

val_forecast <- forecast(fit_backtest, xreg = xreg_val, h = holdout_days)

# 5. Generate Alternative Baseline Models for Comparison
# ------------------------------------------------------------------------------
# --- Model A: XGBoost (Machine Learning Framework) ---
# Prepare matrix inputs required by xgboost
X_train <- as.matrix(bt_train_data[, xreg_cols])
y_train <- bt_train_data$cyclists
X_val   <- as.matrix(bt_val_data[, xreg_cols])

# Train the XGBoost model
fit_xgb <- xgboost(
  data      = X_train, 
  label     = y_train, 
  max_depth = 4, 
  eta       = 0.1, 
  nrounds   = 50, 
  objective = "reg:squarederror",
  verbose   = 0                  # Keeps the console clean
)

# Predict using XGBoost
val_pred_xgb <- predict(fit_xgb, newdata = X_val)




# --- Model B: Simple OLS Regression (Weather variables and day month )---
# Force March and Monday to be the reference (omitted) baselines

# choosing refernce month and day
bt_train_data$month <- relevel(factor(bt_train_data$month), ref = "03")
bt_train_data$day   ~ relevel(factor(bt_train_data$day), ref = "Monday")


fit_lm <- lm(cyclists ~ mean_temp + precip_mm + wind_mean_kmh + factor(month) + factor(day), data = bt_train_data)
val_pred_lm <- predict(fit_lm, newdata = bt_val_data)


# --- Model C: Your ARIMAX Model (From Step 4) ---
val_pred_arimax <- as.numeric(val_forecast$mean)

# Isolate Actuals
val_actuals <- bt_val_data$cyclists

# 6. Calculate Comparative Corporate Performance Metrics
# ------------------------------------------------------------------------------
compute_metrics <- function(actual, predicted) {
  mae  <- mean(abs(actual - predicted))
  rmse <- sqrt(mean((actual - predicted)^2))
  mape <- mean(abs((actual - predicted) / actual), na.rm = TRUE) * 100
  return(c(MAE = mae, RMSE = rmse, MAPE = mape))
}

metrics_summary <- rbind(
  "Linear Regression (OLS)"   = compute_metrics(val_actuals, val_pred_lm),
  "XGBoost (Machine Learning)" = compute_metrics(val_actuals, val_pred_xgb),
  "ARIMAX(14,1,1)(1,0,0)"     = compute_metrics(val_actuals, val_pred_arimax)
)

print(round(metrics_summary, 2))

# 7. Reshape and Plot All Models Simultaneously
# ------------------------------------------------------------------------------


plot_comparison <- data.frame(
  date    = bt_val_data$date,
  Actual  = val_actuals,
  OLS     = val_pred_lm,
  XGBoost = val_pred_xgb,
  ARIMAX  = val_pred_arimax
)

plot_long <- plot_comparison %>%
  pivot_longer(cols = -date, names_to = "Model", values_to = "Cyclists")

# Reorder levels for plotting hierarchy
plot_long$Model <- factor(plot_long$Model, levels = c("Actual", "OLS", "XGBoost", "ARIMAX"))

ggplot(plot_long, aes(x = date, y = Cyclists, group = Model)) +
  geom_line(aes(color = Model, linetype = Model, size = Model)) +
  scale_color_manual(values = c("Actual" = "black", "OLS" = "blue", "XGBoost" = "darkgreen", "ARIMAX" = "red")) +
  scale_linetype_manual(values = c("Actual" = "solid", "OLS" = "dotdash", "XGBoost" = "dashed", "ARIMAX" = "dashed")) +
  scale_size_manual(values = c("Actual" = 1.2, "OLS" = 0.8, "XGBoost" = 1, "ARIMAX" = 1)) +
  labs(
    title = "Model Evaluation Bake-Off (30-Day Out-of-Sample)",
    subtitle = "Comparing Classical Econometrics vs. Machine Learning Architectures",
    x = "Date",
    y = "Number of Cyclists",
    caption = "Portfolio Analysis: Out-of-Sample Validation"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.title = element_text(face = "bold"),
    legend.position = "top"
  )

# Save the multi-model chart
ggsave("results/model_bakeoff_comparison.png", width = 9, height = 6)





