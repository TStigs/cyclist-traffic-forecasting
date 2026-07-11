# ==============================================================================
# Script: 02_model_training.R
# Purpose: Train ARIMAX model, evaluate accuracy, and output results
# ==============================================================================

# Load required libraries
library(readr)
library(dplyr)
library(forecast)
library(ggplot2)
library(modelsummary)

# 1. Load Processed Data
# ------------------------------------------------------------------------------
bikes_weather <- read_csv("data/bikes_processed_weather.csv") %>%
  mutate(date = as.Date(date))

# Define train/test split. 
# Training data has actual cyclist counts; Test data has NAs to forecast.
train_data <- bikes_weather %>% filter(!is.na(cyclists))
test_data  <- bikes_weather %>% filter(is.na(cyclists))

# Initialize time series object
bikes_ts <- ts(train_data$cyclists, frequency = 7)

# 2. Define Exogenous Regressors
# ------------------------------------------------------------------------------
# Exclude June and Monday to avoid multicollinearity (Coefficients will be 
# relative to the exlcuded month and excluded day)
month_cols <- paste0("month_", sprintf("%02d", 1:12))
day_cols   <- paste0("day_", c("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"))

reg_weather <- c("mean_temp", "precip_mm", "wind_mean_kmh")
reg_months  <- setdiff(month_cols, "month_06")
reg_days    <- setdiff(day_cols, "day_Monday")

xreg_cols <- c(reg_weather, reg_months, reg_days)

# Create matrices and force them to be strictly numeric
xreg_train <- as.matrix(train_data[, xreg_cols])
class(xreg_train) <- "numeric"

xreg_test  <- as.matrix(test_data[, xreg_cols])
class(xreg_test) <- "numeric"

# 3. Fit ARIMAX Model (Maximum Likelyhood)
# ------------------------------------------------------------------------------
fit_arimax <- Arima(
  bikes_ts,
  xreg = xreg_train,
  order = c(14, 1, 1),
  seasonal = c(1, 0, 0),
  method = "ML"
)

# Export model summary (Note: The non-positive definite variance-covariance 
# warning is normal here due to the high number of regressors and lags).
modelsummary(
  fit_arimax, 
  output = "results/forecast_summary.docx", 
  title = "ARIMAX (14,1,1)(1,0,0)"
)


### Diagnostic tests

## ACF plot
# New Section: Residual Diagnostics (ACF & PACF)
# ------------------------------------------------------------------------------
# Save a consolidated diagnostic plot containing residual series, ACF, and PACF
png("results/residual_diagnostics.png", width = 800, height = 600)
ggtsdisplay(residuals(fit_arimax), main = "ARIMAX Residual Diagnostics")
dev.off()

# Run a formal portmanteau test to check for remaining autocorrelation
checkresiduals(fit_arimax)


# 4. Forecasting & Evaluation
# ------------------------------------------------------------------------------
# We use nrow(test_data) which will now correctly be 24 days.
forecast_arimax <- forecast(fit_arimax, xreg = xreg_test, h = nrow(test_data))

# Append predictions back to the test dataset
test_data$bikePrediction <- as.numeric(forecast_arimax$mean)

# 5. Visualization
# ------------------------------------------------------------------------------
# Combine for plotting
plot_df <- bind_rows(
  train_data %>% select(date, cyclists) %>% mutate(bikePrediction = NA),
  test_data %>% select(date, cyclists, bikePrediction)
)

ggplot(plot_df, aes(x = date)) +
  geom_line(aes(y = cyclists), color = "black", size = 0.7) +
  geom_line(aes(y = bikePrediction), color = "red", size = 0.7, linetype = "dashed") +
  labs(
    title = "Actual vs Forecasted Cyclist Counts",
    subtitle = "ARIMAX Forecast Performance",
    x = "Date",
    y = "Number of Cyclists",
    caption = "Black = Actual | Red Dashed = Forecast"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.title = element_text(face = "bold")
  ) +
  coord_cartesian(xlim = c(as.Date("2025-09-01"), max(plot_df$date)))

ggsave("results/forecast_plot.png", width = 8, height = 5)



