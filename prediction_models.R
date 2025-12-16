#Amazon Employee Access Kaggle Competition
library(tidyverse)
library(tidymodels)
library(vroom)
library(dplyr)
library(DataExplorer)
library(lubridate)  # for hour extraction
library(bonsai)
library(lightgbm)
library(vroom)
library(ggmosaic)
library(ggplot2)
library(embed)
library(discrim)
library(remotes)
library(tensorflow)
library(keras)

train <- vroom("/kaggle/input/amazon-employee-access-challenge/train.csv")
test <- vroom("/kaggle/input/amazon-employee-access-challenge/test.csv")

# DataExplorer::plot_correlation(train)
# DataExplorer::plot_histogram(train)
# DataExplorer::plot_bar(train)
# ggplot(data=train) + geom_mosaic(aes(x=product(RESOURCE), fill=ACTION))

## LOGISTIC REGRESSION -------------------

# train$ACTION <- as.factor(train$ACTION)  # Convert all predictors to factors

# amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
#   step_mutate_at(all_predictors(), fn=factor) %>%
#   step_other(all_factor_predictors(), threshold = 0.001, other = "other") %>%
#   step_dummy(all_factor_predictors()) %>%
#   step_normalize(all_numeric_predictors()) %>% step_pca(all_predictors(), threshold = .75)

# prepped <- prep(amazon_recipe, verbose = TRUE)
# new_data <- bake(prepped, new_data = NULL)
# # ncol(new_data)  # Should return 1050

# logRegModel <- logistic_reg() %>%
#   set_engine("glm") 

# log_wf <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel) %>%
#     fit(data=train)

# amazon_preds <- predict(log_wf,
#                         new_data=test,
#                         type="prob")

# final <- amazon_preds[2]
# colnames(final)[1] <- "ACTION" # Renames the first column

# kaggle_submission <- final %>%
#   bind_cols(test %>% select(id)) %>%  # Keep original test datetime
#   select(id, ACTION)              # Column order

# vroom_write(kaggle_submission, "log_v1.csv", delim = ",")

# PENALIZED REGRESSION ---------------------------

# train$ACTION <- as.factor(train$ACTION)  # Convert all predictors to factors

# amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
#   step_mutate_at(all_predictors(), fn=factor) %>%
#   step_other(all_factor_predictors(), threshold = 0.001, other = "other") %>%
#   step_dummy(all_factor_predictors()) %>%
#   step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>%
#   step_normalize(all_numeric_predictors())

# prepped <- prep(amazon_recipe, verbose = TRUE)
# new_data <- bake(prepped, new_data = NULL)
# # ncol(new_data)  # Should return 1050

# logRegModel <- logistic_reg(mixture = tune(), penalty = tune()) %>%
#   set_engine("glmnet") 

# amazon_workflow <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel)

# tuning_grid <- grid_regular(penalty(), mixture(), levels=10)

# folds <- vfold_cv(train, v=100, repeats=1)

# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#   grid=tuning_grid,
#   metrics=metric_set(roc_auc))

# bestTune <- CV_results %>% select_best(metric="roc_auc")

# final_wf <- amazon_workflow %>% 
#   finalize_workflow(bestTune) %>%
#   fit(data=train)

# final_wf %>% predict(new_data=test,
#                         type="prob")

# final <- final_wf[2]
# colnames(final)[1] <- "ACTION" # Renames the first column

# kaggle_submission <- final %>%
#   bind_cols(test %>% select(id)) %>%  # Keep original test datetime
#   select(id, ACTION)              # Column order

# vroom_write(kaggle_submission, "submission.csv", delim = ",")

# # RANDOM FOREST ---------------------------

train$ACTION <- as.factor(train$ACTION)  # Convert all predictors to factors

amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn=factor) %>%
  # step_other(all_factor_predictors(), threshold = 0.001, other = "other") %>%
  # step_dummy(all_factor_predictors()) %>%
  step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>%
  step_zv(all_predictors())%>%
  step_normalize(all_numeric_predictors())
# step_pca(all_predictors(), threshold = .8)

prepped <- prep(amazon_recipe, verbose = TRUE)
new_data <- bake(prepped, new_data = NULL)
# ncol(new_data)  # Should return 1050

logRegModel <- rand_forest(mtry = 1, 
                           min_n = 10, 
                           trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

amazon_workflow <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(logRegModel)%>%
  fit(data=train)

# tuning_grid <- grid_regular(
#   mtry(),
#   min_n(),
#   levels = 5
# )

# folds <- vfold_cv(train, v=5, repeats=1)

# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc))

# bestTune <- CV_results %>% select_best(metric="roc_auc")
# print(bestTune)

# final_wf <- amazon_workflow %>% 
#   finalize_workflow(bestTune) %>%
#   fit(data=train)

# Save predictions to a tibble
amazon_preds <- predict(amazon_workflow, new_data = test, type = "prob")

# # Write predictions to CSV
# vroom_write(amazon_preds, "preds.csv", delim = ",")

# Extract the positive class probability
final <- amazon_preds %>% select(.pred_1)
colnames(final)[1] <- "ACTION"  # Rename column to match Kaggle format

# Combine with test IDs and format for submission
kaggle_submission <- final %>%
  bind_cols(test %>% select(id)) %>%
  select(id, ACTION)

# Write final submission file
vroom_write(kaggle_submission, "submission.csv", delim = ",")

# # KNN CLASSIFIER ---------------------------

# train$ACTION <- as.factor(train$ACTION)  # Convert all predictors to factors

# amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
#   step_mutate_at(all_predictors(), fn=factor) %>%
#   step_other(all_factor_predictors(), threshold = 0.001, other = "other") %>%
#   step_dummy(all_factor_predictors()) %>%
#   step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>%
#   step_normalize(all_numeric_predictors())

# prepped <- prep(amazon_recipe, verbose = TRUE)
# new_data <- bake(prepped, new_data = NULL)
# # ncol(new_data)  # Should return 1050

# logRegModel <- nearest_neighbor(neighbors=tune()) %>%
#   set_engine("kknn") %>%
#   set_mode("classification")

# amazon_workflow <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel)

# # tuning_grid <- grid_regular(
# #   neighbors(range = c(1, 25)),
# #   levels = 10
# # )
# tuning_grid <- tibble(neighbors = seq(1, 15, by = 2))

# # folds <- vfold_cv(train, v=5, repeats=1)
# folds <- vfold_cv(train, v = 3)

# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc))

# bestTune <- CV_results %>% select_best(metric="roc_auc")
# print(bestTune)

# final_wf <- amazon_workflow %>% 
#   finalize_workflow(bestTune) %>%
#   fit(data=train)

# # Save predictions to a tibble
# amazon_preds <- predict(final_wf, new_data = test, type = "prob")

# # Write predictions to CSV
# vroom_write(amazon_preds, "preds.csv", delim = ",")

# # Extract the positive class probability
# final <- amazon_preds %>% select(.pred_1)
# colnames(final)[1] <- "ACTION"  # Rename column to match Kaggle format

# # Combine with test IDs and format for submission
# kaggle_submission <- final %>%
#   bind_cols(test %>% select(id)) %>%
#   select(id, ACTION)

# # Write final submission file
# vroom_write(kaggle_submission, "submission.csv", delim = ",")

# # NAIVE BAYES ---------------------------

# train$ACTION <- as.factor(train$ACTION)  # Convert all predictors to factors

# # amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
# #   step_mutate_at(all_predictors(), fn=factor) %>%
# #   step_other(all_factor_predictors(), threshold = 0.001, other = "other") %>%
# #   step_dummy(all_factor_predictors()) %>%
# #   step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>%
# #   step_normalize(all_numeric_predictors())

# amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
#   step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
#   step_mutate_at(all_nominal_predictors(), fn = factor)

# prepped <- prep(amazon_recipe, verbose = TRUE)
# new_data <- bake(prepped, new_data = NULL)
# # ncol(new_data)  # Should return 1050

# logRegModel <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
#   set_engine("naivebayes") %>%
#   set_mode("classification")

# amazon_workflow <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel)

# tuning_grid <- grid_regular(
#   Laplace(range = c(0, 1)),
#   smoothness(range = c(0, 1)),
#   levels = 5
# )

# # folds <- vfold_cv(train, v=5, repeats=1)
# folds <- vfold_cv(train, v = 3)

# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc))

# bestTune <- CV_results %>% select_best(metric="roc_auc")
# print(bestTune)

# final_wf <- amazon_workflow %>% 
#   finalize_workflow(bestTune) %>%
#   fit(data=train)

# # Save predictions to a tibble
# amazon_preds <- predict(final_wf, new_data = test, type = "prob")

# # Write predictions to CSV
# vroom_write(amazon_preds, "preds.csv", delim = ",")

# # Extract the positive class probability
# final <- amazon_preds %>% select(.pred_1)
# colnames(final)[1] <- "ACTION"  # Rename column to match Kaggle format

# # Combine with test IDs and format for submission
# kaggle_submission <- final %>%
#   bind_cols(test %>% select(id)) %>%
#   select(id, ACTION)

# # Write final submission file
# vroom_write(kaggle_submission, "submission.csv", delim = ",")

# NEURAL NET ---------------------------

# train$ACTION <- as.factor(train$ACTION)  # Convert all predictors to factors

# amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
#   step_mutate_at(all_predictors(), fn=factor) %>%
#   step_other(all_factor_predictors(), threshold = 0.001, other = "other") %>%
#   step_dummy(all_factor_predictors()) %>%
#   step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>%
#   step_normalize(all_numeric_predictors())%>% step_range(all_numeric_predictors(),min=0, max=1)

# # amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
# #   step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
# #   step_mutate_at(all_nominal_predictors(), fn = factor)

# prepped <- prep(amazon_recipe, verbose = TRUE)
# new_data <- bake(prepped, new_data = NULL)
# # ncol(new_data)  # Should return 1050

# logRegModel <- mlp(hidden_units=tune(),
#                   epochs=50) %>%
#   set_engine("keras") %>%
#   set_mode("classification")

# amazon_workflow <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel)

# tuning_grid <- grid_regular(hidden_units(range=c(1, 20)),
#                             levels = 5)

# # folds <- vfold_cv(train, v=5, repeats=1)
# folds <- vfold_cv(train, v = 5)

# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc))

# bestTune <- CV_results %>% select_best(metric="roc_auc", "accuracy")
# print(bestTune)

# final_wf <- amazon_workflow %>% 
#   finalize_workflow(bestTune) %>%
#   fit(data=train)

# # Save predictions to a tibble
# amazon_preds <- predict(final_wf, new_data = test, type = "prob")

# # Write predictions to CSV
# vroom_write(amazon_preds, "preds.csv", delim = ",")

# final_wf %>% collect_metrics() %>%
# filter(.metric=="accuracy") %>%
# ggplot(aes(x=hidden_units,y=mean)) + geom_line()

# vroom_write(final_wf, "tuned_nn.jpg")

# # Extract the positive class probability
# final <- amazon_preds %>% select(.pred_1)
# colnames(final)[1] <- "ACTION"  # Rename column to match Kaggle format

# # Combine with test IDs and format for submission
# kaggle_submission <- final %>%
#   bind_cols(test %>% select(id)) %>%
#   select(id, ACTION)

# # Write final submission file
# vroom_write(kaggle_submission, "submission.csv", delim = ",")