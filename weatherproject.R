#===============================================================================
# 0. import required modules

install.packages("rlang")
install.packages("tidymodels")
install.packages("tidyverse")

library(tidymodels)
library(tidyverse)
library(rlang)

#===============================================================================
# 1. Download NOAA Weather Dataset

url <- "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ENn4iRKnW2szuR-zPKslwg/noaa-weather-sample-data-tar.gz"
download.file(url, destfile = "noaa-weather-sample-data-tar.gz")
untar("noaa-weather-sample-data-tar.gz", tar = "internal")

#===============================================================================
# 2. Extract and Read into Project

weather_data <- read_csv("noaa-weather-sample-data/jfk_weather_sample.csv")

#display first few rows
head(weather_data)

#glimpse of dataset
glimpse(weather_data)

#===============================================================================
# 3. Select Subset of Columns (PREPROCESSING)

#select columns
weather_subset <- weather_data %>%
  select(HOURLYRelativeHumidity,
         HOURLYDRYBULBTEMPF,
         HOURLYPrecip,
         HOURLYWindSpeed,
         HOURLYStationPressure)

#first 10 rows of new dataframe
head(weather_subset, 10)

#===============================================================================
# 4. Clean Up Columns (PREPROCESSING)

#Inspect the unique values present in the column HOURLYPrecip
unique(weather_subset$HOURLYPrecip)

#Replace all the T values with "0.0" and remove 's'
weather_clean <- weather_subset %>%
  mutate(
    HOURLYPrecip, if_else(HOURLYPrecip == "T", "0.0", HOURLYPrecip),
    HOURLYPrecip, str_remove(HOURLYPrecip, pattern = "s$")
  )

#===============================================================================
# 5. Clean Up Columns (PREPROCESSING)

#Convert the column to numeric
weather_clean <- weather_clean %>%
  mutate(HOURLYPrecip = as.numeric(HOURLYPrecip))

#Verify the cleanup
unique(weather_clean$HOURLYPrecip)

#glimpse
glimpse(weather_clean)

#===============================================================================
# 6. Rename Columns (PREPROCESSING)

weather_final <- weather_clean %>%
  rename(
    relative_humidity = HOURLYRelativeHumidity,
    dry_bulb_temp_f = HOURLYDRYBULBTEMPF,
    precip = HOURLYPrecip,
    wind_speed = HOURLYWindSpeed,
    station_pressure = HOURLYStationPressure
  )

#CHECK
glimpse(weather_final)

#===============================================================================
# 7. Exploratory Data Analysis

#Split data + set random seed (ensures data is split same every time = reproducible results)
set.seed(1234)
weather_split <- initial_split(weather_final, prop = 0.8)
train_data <- training(weather_split)
test_data <- testing(weather_split)

train_data <- train_data %>%
  select(relative_humidity, dry_bulb_temp_f, precip, wind_speed, station_pressure)
test_data  <- test_data %>%
  select(relative_humidity, dry_bulb_temp_f, precip, wind_speed, station_pressure)


#plot histograms or box plots of the variables (relative_humidity, dry_bulb_temp_f, precip, wind_speed, station_pressure) for their distributions
train_data_long <- train_data %>%
  pivot_longer(
    cols = c(relative_humidity, dry_bulb_temp_f, precip, wind_speed, station_pressure),
    names_to = "variable",
    values_to = "value"
  )

ggplot(train_data_long, aes(x = value, fill = variable)) +
  geom_histogram(bins = 30, alpha = 0.6, color = "black") +
  facet_wrap(~ variable, scales = "free", ncol = 2) +
  labs(title = "Distributions of Weather Variables (Training Set)",
       x = "Value",
       y = "Frequency") +
  theme_minimal() +
  theme(legend.position = "none")

#===============================================================================
# 8. Linear Regression

#Create simple linear regression models 
lm_spec <- linear_reg() %>%
  set_engine("lm")

#LR (precip ~ relative_humidity)
lm_humidity <- lm_spec %>%
  fit(precip ~ relative_humidity, data = train_data)

#LR (precip ~ dry_bulb_temp_f)
lm_humidity <- lm_spec %>%
  fit(precip ~ dry_bulb_temp_f, data = train_data)

#LR (precip ~ wind_speed)
lm_humidity <- lm_spec %>%
  fit(precip ~ wind_speed, data = train_data)

#LR (precip ~ station_pressure)
lm_humidity <- lm_spec %>%
  fit(precip ~ station_pressure, data = train_data)

#Visualization
ggplot(data = train_data, mapping = aes(x = relative_humidity, y = precip)) +
  geom_point(alpha = 0.3, color = "blue") +
  geom_smooth(method = "lm", na.rm = TRUE, color = "red") +
  labs(title = "Precip vs Relative Huimidity", x = "Relative Humidity", y = "Precipitation")

ggplot(data = train_data, mapping = aes(x = dry_bulb_temp_f, y = precip)) +
  geom_point(alpha = 0.3, color = "blue") +
  geom_smooth(method = "lm", na.rm = TRUE, color = "red") +
  labs(title = "Precip vs Temperature", x = "Dry Bulb Temperature (F)", y = "Precipitation")

ggplot(data = train_data, mapping = aes(x = wind_speed, y = precip)) +
  geom_point(alpha = 0.3, color = "blue") +
  geom_smooth(method = "lm", na.rm = TRUE, color = "red") +
  labs(title = "Precip vs Wind Speed", x = "Wind Speed (mph)", y = "Precipitation")

ggplot(data = train_data, mapping = aes(x = station_pressure, y = precip)) +
  geom_point(alpha = 0.3, color = "blue") +
  geom_smooth(method = "lm", na.rm = TRUE, color = "red") +
  labs(title = "Precip vs Station Pressure", x = "Station Pressure (in Hg)", y = "Precipitation")

#===============================================================================
# 9. Improve the Model

# Multiple Linear Regression (MLR)

# Define and fit model
lm_multi_spec <- linear_reg() %>%
  set_engine("lm")

lm_multi_fit <- lm_multi_spec %>%
  fit(precip ~ relative_humidity + dry_bulb_temp_f + wind_speed + station_pressure,
      data = train_data)

# View model summary
summary(lm_multi_fit$fit)


# LASSO Regression

# Clean training data (remove missing values)
train_data <- train_data %>%
  drop_na(precip, relative_humidity, dry_bulb_temp_f, wind_speed, station_pressure)


# Define LASSO model (λ = 0.1)
lasso_spec <- linear_reg(penalty = 0.1, mixture = 1) %>%
  set_engine("glmnet")

# Create recipe
weather_recipe <- recipe(
  precip ~ relative_humidity + dry_bulb_temp_f + wind_speed + station_pressure,
  data = train_data_clean
)

# Build workflow
lasso_wf <- workflow() %>%
  add_recipe(weather_recipe) %>%
  add_model(lasso_spec)

# Fit LASSO model
lasso_fit <- lasso_wf %>%
  fit(data = train_data_clean)

# View coefficients
lasso_fit %>%
  extract_fit_parsnip() %>%
  tidy()


# Evaluate MLR and LASSO (Fixed λ)
# --- Multiple Linear Regression ---

train_results <- predict(lm_multi_fit, new_data = train_data) %>%
  bind_cols(train_data %>% select(precip))

test_results <- predict(lm_multi_fit, new_data = test_data) %>%
  bind_cols(test_data %>% select(precip))

rmse_train_mlr <- rmse(train_results, truth = precip, estimate = .pred)
rmse_test_mlr  <- rmse(test_results,  truth = precip, estimate = .pred)
rsq_train_mlr  <- rsq(train_results,  truth = precip, estimate = .pred)
rsq_test_mlr   <- rsq(test_results,   truth = precip, estimate = .pred)

# --- LASSO (Fixed Penalty) ---

lasso_train_results <- predict(lasso_fit, new_data = train_data_clean) %>%
  bind_cols(train_data_clean %>% select(precip))

lasso_test_results <- predict(lasso_fit, new_data = test_data) %>%
  bind_cols(test_data %>% select(precip))

rmse_train_lasso <- rmse(lasso_train_results, truth = precip, estimate = .pred)
rmse_test_lasso  <- rmse(lasso_test_results,  truth = precip, estimate = .pred)
rsq_train_lasso  <- rsq(lasso_train_results,  truth = precip, estimate = .pred)
rsq_test_lasso   <- rsq(lasso_test_results,   truth = precip, estimate = .pred)


# LASSO Regression (Tuned Penalty)

# Define tunable LASSO spec
tune_spec <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")
# (Optional but recommended) Clear old objects from memory to prevent recipe conflicts
rm(lasso_tune_wf, lasso_grid, final_lasso_fit)

# Define tunable LASSO specification
tune_spec <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")

# Create recipe with proper imputation and normalization
weather_recipe_tuned <- recipe(
  precip ~ relative_humidity + dry_bulb_temp_f + wind_speed + station_pressure,
  data = train_data
) %>%
  step_impute_mean(all_numeric_predictors()) %>%   # ✅ Only numeric predictors
  step_normalize(all_numeric_predictors())          # ✅ Normalize predictors

# Create workflow (fresh, clean version)
lasso_tune_wf <- workflow() %>%
  add_recipe(weather_recipe_tuned) %>%
  add_model(tune_spec)

# Cross-validation setup
set.seed(1234)
weather_cvfolds <- vfold_cv(train_data, v = 5)

# Define lambda grid (on log scale)
lambda_grid <- grid_regular(
  penalty(range = c(-3, 0.3)),   # 10^-3 to ~2
  levels = 50
)

# Tune lambda (this will now run without the 'missing precip' error)
set.seed(1234)
lasso_grid <- tune_grid(
  lasso_tune_wf,
  resamples = weather_cvfolds,
  grid = lambda_grid,
  metrics = metric_set(rmse, rsq),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# View best lambda by RMSE
best_lambda <- select_best(lasso_grid, metric = "rmse")
best_lambda

# Refit final model with best lambda
final_lasso <- finalize_workflow(lasso_tune_wf, best_lambda)

final_lasso_fit <- final_lasso %>%
  fit(data = train_data)


# Evaluate Tuned LASSO


lasso_tuned_results <- predict(final_lasso_fit, new_data = test_data) %>%
  bind_cols(test_data %>% select(precip))

rmse_lasso_tuned <- rmse(lasso_tuned_results, truth = precip, estimate = .pred)
rsq_lasso_tuned  <- rsq(lasso_tuned_results,  truth = precip, estimate = .pred)


# Combine Metrics into Comparison Table


comparison_df <- data.frame(
  Model = c("Multiple Linear", "LASSO (Fixed λ=0.1)", "LASSO (Tuned λ)"),
  RMSE  = round(c(rmse_test_mlr$.estimate,
                  rmse_test_lasso$.estimate,
                  rmse_lasso_tuned$.estimate), 4),
  R_Squared = round(c(rsq_test_mlr$.estimate,
                      rsq_test_lasso$.estimate,
                      rsq_lasso_tuned$.estimate), 4)
)

print(comparison_df)

#===============================================================================
# 10. Find Best Model

# Multiple Linear Regression
multi_results <- predict(lm_multi_fit, new_data = test_data) %>%
  bind_cols(test_data %>% select(precip))

# LASSO Regression
lasso_results <- predict(final_lasso_fit, new_data = test_data) %>%
  bind_cols(test_data %>% select(precip))

head(lasso_results)



# RMSE
multi_rmse <- rmse(multi_results, truth = precip, estimate = .pred)
lasso_rmse <- rmse(lasso_results, truth = precip, estimate = .pred)

# R-squared (optional)
multi_rsq <- rsq(multi_results, truth = precip, estimate = .pred)
lasso_rsq <- rsq(lasso_results, truth = precip, estimate = .pred)



# Create comparison table
model_names <- c("Multiple Linear", "LASSO")

test_rmse <- c(multi_rmse$.estimate, lasso_rmse$.estimate)
test_rsq  <- c(multi_rsq$.estimate,  lasso_rsq$.estimate)

comparison_df <- data.frame(
  Model = model_names,
  RMSE = round(test_rmse, 4),
  R_Squared = round(test_rsq, 4)
)

print(comparison_df)


