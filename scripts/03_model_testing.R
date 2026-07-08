# ==============================================================================
# Script: 03_model_testing.R
# Purpose: Simple Validation Backtest (Single Split) for ARIMAX Model
# ==============================================================================

# Load required libraries
library(readr)
library(dplyr)
library(forecast)

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

# 5. Forecast and Evaluate Accuracy
# ------------------------------------------------------------------------------
val_forecast <- forecast(fit_backtest, xreg = xreg_val, h = holdout_days)

# Extract predictions
val_predictions <- as.numeric(val_forecast$mean)
val_actuals     <- bt_val_data$cyclists

# Calculate Error Metrics
mae  <- mean(abs(val_actuals - val_predictions))
rmse <- sqrt(mean((val_actuals - val_predictions)^2))

cat("\n--- Simple Backtest Results (30-Day Holdout) ---\n")
cat(sprintf("Mean Absolute Error (MAE): %.2f cyclists\n", mae))
cat(sprintf("Root Mean Squared Error (RMSE): %.2f cyclists\n", rmse))

# Save results for visualization
backtest_results <- data.frame(
  date = bt_val_data$date,
  Actual = val_actuals,
  Predicted = val_predictions
)
write_csv(backtest_results, "results/simple_backtest_results.csv")


# 6. Visualize Backtest Performance
# ------------------------------------------------------------------------------
library(ggplot2)

ggplot(backtest_results, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted, color = "Predicted"), size = 1, linetype = "dashed") +
  scale_color_manual(
    name = "Legend",
    values = c("Actual" = "black", "Predicted" = "red")
  ) +
  labs(
    title = "Out-of-Sample Backtest Validation",
    subtitle = "ARIMAX(14,1,1)(1,0,0) Performance Over 30-Day Holdout",
    x = "Date",
    y = "Number of Cyclists",
    caption = sprintf("Backtest Metrics: MAE = %.2f | RMSE = %.2f", mae, rmse)
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.title = element_text(face = "bold"),
    legend.position = "top"
  )

# Save the visualization for the portfolio/README
ggsave("results/backtest_validation_plot.png", width = 8, height = 5)
cat("Backtest validation plot saved successfully to 'results/backtest_validation_plot.png'\n")
