# ==============================================================================
# Script: 01_data_preparation.R
# Purpose: Clean bikes.csv, engineer date features, and integrate weather data
# ==============================================================================

# Load required libraries
library(readr)
library(dplyr)
library(lubridate)
library(openmeteo)

# 1. Load Core Dataset
# ------------------------------------------------------------------------------
# Ensure the working directory is set to the project root before running
bikes <- read_csv("data/bikes.csv") %>%
  mutate(date = as.Date(date)) %>%
  arrange(date) 

# 2. Fetch Historical Weather Data
# ------------------------------------------------------------------------------
lat <- 48.4283
lon <- -123.3721
start_date <- min(bikes$date, na.rm = TRUE)
end_date <- max(bikes$date, na.rm = TRUE)

weather_hist <- weather_history(
  location = c(lat, lon),
  start    = start_date,
  end      = end_date,
  daily    = c("temperature_2m_mean", "precipitation_sum", "wind_speed_10m_mean"),
  timezone = "America/Vancouver"
)

weather_df <- weather_hist %>%
  transmute(
    date          = as.Date(date),
    mean_temp     = daily_temperature_2m_mean,
    precip_mm     = daily_precipitation_sum,
    wind_mean_kmh = daily_wind_speed_10m_mean
  )

# 3. Merge and Engineer Features
# ------------------------------------------------------------------------------
bikes_weather <- bikes %>%
  left_join(weather_df, by = "date") %>%
  mutate(
    month = format(date, "%m"),
    day   = weekdays(date)
  )

# Create fixed levels for dummy variables
month_levels <- sprintf("%02d", 1:12)
day_levels   <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# Generate one-hot encoded variables
month_dummies <- model.matrix(~ factor(month, levels = month_levels) - 1, data = bikes_weather)
colnames(month_dummies) <- paste0("month_", month_levels)

day_dummies <- model.matrix(~ factor(day, levels = day_levels) - 1, data = bikes_weather)
colnames(day_dummies) <- paste0("day_", day_levels)

# Bind features and finalize dataset
bikes_weather <- bind_cols(bikes_weather, as.data.frame(month_dummies), as.data.frame(day_dummies))

# Save processed dataset for modeling
write_csv(bikes_weather, "data/bikes_processed_weather.csv")