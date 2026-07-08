# ==============================================================================
# Script: 05_final_production_evaluation.R
# Purpose: Evaluate November 2025 forecast predictions against true actuals
# ==============================================================================
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)

# 1. Load Data
# ------------------------------------------------------------------------------
predictions <- read_csv("results/future_forecast_predictions.csv")
actuals     <- read_csv("data/november_2025_actuals.csv")

# Join the predictions and actuals on date
evaluation_df <- inner_join(actuals, predictions, by = "date")

# 2. Calculate Evaluation Metrics
# ------------------------------------------------------------------------------
compute_metrics <- function(actual, predicted) {
  mae  <- mean(abs(actual - predicted))
  rmse <- sqrt(mean((actual - predicted)^2))
  mape <- mean(abs((actual - predicted) / actual), na.rm = TRUE) * 100
  return(c(MAE = mae, RMSE = rmse, MAPE = mape))
}

final_metrics <- rbind(
  "Linear Regression (OLS)"   = compute_metrics(evaluation_df$cyclists, evaluation_df$pred_OLS),
  "XGBoost (Machine Learning)" = compute_metrics(evaluation_df$cyclists, evaluation_df$pred_XGBoost),
  "ARIMAX(14,1,1)(1,0,0)"     = compute_metrics(evaluation_df$cyclists, evaluation_df$pred_ARIMAX)
)

cat("\n🏆 --- FINAL PRODUCTION EVALUATION RESULTS (NOVEMBER 2025) --- 🏆\n")
print(round(final_metrics, 2))

# Save production metrics to csv
write_csv(as.data.frame(final_metrics) %>% mutate(Model = rownames(final_metrics)), 
          "results/final_production_metrics.csv")

# 3. Create the Final Production Comparison Plot
# ------------------------------------------------------------------------------
plot_data <- evaluation_df %>%
  select(date, Actual = cyclists, OLS = pred_OLS, XGBoost = pred_XGBoost, ARIMAX = pred_ARIMAX) %>%
  pivot_longer(cols = -date, names_to = "Model", values_to = "Cyclists")

plot_data$Model <- factor(plot_data$Model, levels = c("Actual", "OLS", "XGBoost", "ARIMAX"))

ggplot(plot_data, aes(x = date, y = Cyclists, group = Model)) +
  geom_line(aes(color = Model, linetype = Model, size = Model)) +
  scale_color_manual(values = c("Actual" = "black", "OLS" = "blue", "XGBoost" = "darkgreen", "ARIMAX" = "red")) +
  scale_linetype_manual(values = c("Actual" = "solid", "OLS" = "dotdash", "XGBoost" = "dashed", "ARIMAX" = "dashed")) +
  scale_size_manual(values = c("Actual" = 1.2, "OLS" = 0.8, "XGBoost" = 1, "ARIMAX" = 1)) +
  labs(
    title = "Forecast Evaluation: Actual vs Predicted Cyclist Counts",
    subtitle = "November 2025 Out-of-Sample Model Comparison (Johnson St. Bridge)",
    x = "Date",
    y = "Number of Cyclists",
    caption = "Out-of-Sample Forecast Evaluation"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.title = element_text(face = "bold"),
    legend.position = "top"
  )

ggsave("results/final_production_evaluation_plot.png", width = 10, height = 6)
cat("\nFinal evaluation plot saved to 'results/final_production_evaluation_plot.png'\n")






# Grab September and October actuals as a historical baseline anchor
historical_tail <- prod_train %>%
  filter(date >= as.Date("2025-10-01") & date < as.Date("2025-11-01")) %>%
  select(date, Actual = cyclists)

# Prep November data
november_eval <- evaluation_df %>%
  select(date, Actual = cyclists, OLS = pred_OLS, XGBoost = pred_XGBoost, ARIMAX = pred_ARIMAX)

# Combine them cleanly
full_plot_df <- bind_rows(historical_tail, november_eval) %>%
  pivot_longer(cols = -date, names_to = "Model", values_to = "Cyclists") %>%
  drop_na(Cyclists)

# Reorder levels for plotting hierarchy
full_plot_df$Model <- factor(full_plot_df$Model, levels = c("Actual", "OLS", "XGBoost", "ARIMAX"))

# Plot with a vertical dashed line separating history from forecast
ggplot(full_plot_df, aes(x = date, y = Cyclists, group = Model)) +
  geom_line(aes(color = Model, linetype = Model, size = Model)) +
  geom_vline(xintercept = as.numeric(as.Date("2025-11-01")), linetype = "dotted", color = "gray40", size = 1) +
  annotate("text", x = as.Date("2025-10-20"), y = 3400, label = "Past Observations", fontface = "italic") +
  annotate("text", x = as.Date("2025-11-10"), y = 3200, label = "Forecast", fontface = "italic") +
  scale_color_manual(values = c("Actual" = "black", "OLS" = "blue", "XGBoost" = "darkgreen", "ARIMAX" = "red")) +
  scale_linetype_manual(values = c("Actual" = "solid", "OLS" = "dotdash", "XGBoost" = "dashed", "ARIMAX" = "dashed")) +
  scale_size_manual(values = c("Actual" = 1.1, "OLS" = 0.8, "XGBoost" = 0.9, "ARIMAX" = 0.9)) +
  theme_minimal(base_size = 14) +
  labs(title = "Daily Cyclist Counts: Past and Forecast", x = "Date", y = "Cyclists")


ggsave(
  filename = "results/forecast_horizon_plot.png",
  width    = 11,      # Gives the x-axis plenty of room to stretch chronologically
  height   = 6.5,     # Prevents the y-axis labels and legend from feeling crowded
  dpi      = 300,     # Essential: Sets dots-per-inch to print/retina display standard
  bg       = "white"  # Forces a solid white background (prevents transparent artifacting on GitHub dark mode)
)

cat("Optimized portfolio graphic saved to 'results/production_forecast_with_context.png'\n")


