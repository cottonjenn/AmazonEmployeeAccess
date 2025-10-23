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
library(glmnet)

train <- vroom("train.csv")
test <- vroom("test.csv")

# DataExplorer::plot_correlation(train)
# DataExplorer::plot_histogram(train)
# DataExplorer::plot_bar(train)
# ggplot(data=train) + geom_mosaic(aes(x=product(RESOURCE), fill=ACTION))

## LOGISTIC REGRESSION -------------------

train$ACTION <- as.factor(train$ACTION)  # Convert all predictors to factors

amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn=factor) %>%
  step_other(all_factor_predictors(), threshold = 0.001, other = "other") %>%
  step_dummy(all_factor_predictors()) %>%
  # step_normalize(all_numeric_predictors())

prepped <- prep(amazon_recipe, verbose = TRUE)
new_data <- bake(prepped, new_data = NULL)
ncol(new_data)  # Should return 1050

# logRegModel <- logistic_reg() %>%
#   set_engine("glm") 
# 
# log_wf <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel) %>%
#     fit(data=train)
# 
# amazon_preds <- predict(log_wf,
#                         new_data=test,
#                         type="prob")
# 
# final <- amazon_preds[2]
# colnames(final)[1] <- "ACTION" # Renames the first column
# 
# kaggle_submission <- final %>%
#   bind_cols(test %>% select(id)) %>%  # Keep original test datetime
#   select(id, ACTION)              # Column order
# 
# vroom_write(kaggle_submission, "log_v1.csv", delim = ",")

# # PENALIZED REGRESSION ---------------------------
# 
# train$ACTION <- as.factor(train$ACTION)  # Convert all predictors to factors
# 
# amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
#   step_mutate_at(all_predictors(), fn=factor) %>%
#   step_other(all_factor_predictors(), threshold = 0.001, other = "other") %>%
#   step_dummy(all_factor_predictors()) %>%
#   step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>%
#   step_normalize(all_numeric_predictors())
#   
# prepped <- prep(amazon_recipe, verbose = TRUE)
# new_data <- bake(prepped, new_data = NULL)
# # ncol(new_data)  # Should return 1050
# 
# logRegModel <- logistic_reg(mixture = tune(), penalty = tune()) %>%
#   set_engine("glmnet") 
# 
# amazon_workflow <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel)
# 
# tuning_grid <- grid_regular(penalty(), mixture(), levels=5)
# 
# folds <- vfold_cv(train, v=5, repeats=1)
# 
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#   grid=tuning_grid,
#   metrics=metric_set(roc_auc))
# 
# bestTune <- CV_results %>% select_best(metric="roc_auc")
# 
# final_wf <- amazon_workflow %>% 
#   finalize_workflow(bestTune) %>%
#   fit(data=train)
# 
# final_wf %>% predict(new_data=test,
#                         type="prob")
# 
# final <- amazon_preds[2]
# colnames(final)[1] <- "ACTION" # Renames the first column
# 
# kaggle_submission <- final %>%
#   bind_cols(test %>% select(id)) %>%  # Keep original test datetime
#   select(id, ACTION)              # Column order
# 
# vroom_write(kaggle_submission, "penlog_v1.csv", delim = ",")

# RANDOM FOREST ---------------------------

# train$ACTION <- as.factor(train$ACTION)  # Convert all predictors to factors
# 
# amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
#   step_mutate_at(all_predictors(), fn=factor) %>%
#   step_other(all_factor_predictors(), threshold = 0.001, other = "other") %>%
#   step_dummy(all_factor_predictors()) %>%
#   step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>%
#   step_normalize(all_numeric_predictors())
# 
# prepped <- prep(amazon_recipe, verbose = TRUE)
# new_data <- bake(prepped, new_data = NULL)
# # ncol(new_data)  # Should return 1050
# 
# logRegModel <- rand_forest(mtry = tune(), 
#                            min_n = tune(), 
#                            trees = 500) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# amazon_workflow <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel)
# 
# tuning_grid <- grid_regular(
#   mtry(range = c(1, 10)),
#   min_n(range = c(2, 10)),
#   levels = 5
# )
# 
# folds <- vfold_cv(train, v=5, repeats=1)
# 
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc))
# 
# bestTune <- CV_results %>% select_best(metric="roc_auc")
# 
# final_wf <- amazon_workflow %>% 
#   finalize_workflow(bestTune) %>%
#   fit(data=train)
# 
# final_wf %>% predict(new_data=test,
#                      type="prob")
# 
# final <- amazon_preds[2]
# colnames(final)[1] <- "ACTION" # Renames the first column
# 
# kaggle_submission <- final %>%
#   bind_cols(test %>% select(id)) %>%  # Keep original test datetime
#   select(id, ACTION)              # Column order
# 
# vroom_write(kaggle_submission, "penlog_v1.csv", delim = ",")
