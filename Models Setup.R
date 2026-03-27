# Logistic Regression
churn_log_reg_mod <- logistic_reg() %>%  
  set_engine("glm") %>% 
  set_mode("classification")

churn_log_reg_workflow <- workflow() %>% 
  add_model(churn_log_reg_mod) %>% 
  add_recipe(churn_recipe)

churn_log_reg_fit <- fit(churn_log_reg_workflow, churn_train)

save(churn_log_reg_fit, churn_log_reg_workflow, 
     file = "/Users/User/OneDrive/Desktop/Final Project/RDA/churn_log_reg.rda")

# Linear Discriminant Analysis
churn_lda_mod <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

churn_lda_workflow <- workflow() %>% 
  add_model(churn_lda_mod) %>% 
  add_recipe(churn_recipe)

churn_lda_fit <- fit(churn_lda_workflow, data = churn_train)

save(churn_lda_fit, churn_lda_workflow, 
     file = "/Users/User/OneDrive/Desktop/Final Project/RDA/churn_lda.rda")

# Quadratic Discriminant Analysis
churn_qda_mod <- discrim_quad() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

churn_qda_workflow <- workflow() %>% 
  add_model(churn_qda_mod) %>% 
  add_recipe(churn_recipe)

churn_qda_fit <- fit(churn_qda_workflow, data = churn_train)

save(churn_qda_fit, churn_qda_workflow, 
     file = "/Users/User/OneDrive/Desktop/Final Project/RDA/churn_qda.rda")

# Lasso Regression
churn_lasso_mod <- 
  logistic_reg(penalty = tune(), mixture = 1) %>%  
  set_mode("classification") %>% 
  set_engine("glmnet") 

churn_lasso_wf <- workflow() %>% 
  add_recipe(churn_recipe) %>% 
  add_model(churn_lasso_mod)

churn_lasso_grid <- grid_regular(penalty(range = c(-5, 5)), levels = 10)

churn_lasso_tune <- tune_grid(
  churn_lasso_wf,
  resamples = churn_folds, 
  grid = churn_lasso_grid,
  metrics = metric_set(roc_auc)
)

churn_lasso_best <- select_best(churn_lasso_tune, metric = "roc_auc")

churn_lasso_final <- finalize_workflow(churn_lasso_wf, churn_lasso_best)

churn_lasso_fit <- fit(churn_lasso_final, data = churn_train)

save(churn_lasso_fit, churn_lasso_final, churn_lasso_tune,
     file = "/Users/User/OneDrive/Desktop/Final Project/RDA/churn_lasso.rda")

# Decision Tree
churn_dt_mod <- 
  decision_tree(cost_complexity = tune(),
                tree_depth = tune(),
                min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

churn_dt_wf <- workflow() %>%
  add_model(churn_dt_mod) %>%
  add_recipe(churn_recipe)

churn_dt_grid <- grid_regular(cost_complexity(),
                              tree_depth(),
                              min_n(),
                              levels = 4)

churn_dt_tune <- tune_grid(
  churn_dt_wf, 
  resamples = churn_folds, 
  grid = churn_dt_grid, 
  metrics = metric_set(roc_auc)
)

churn_dt_best <- select_best(churn_dt_tune, metric = "roc_auc")

churn_dt_final <- finalize_workflow(churn_dt_wf, churn_dt_best)

churn_dt_fit <- fit(churn_dt_final, data = churn_train)

save(churn_dt_fit, churn_dt_final, churn_dt_tune,
     file = "/Users/User/OneDrive/Desktop/Final Project/RDA/churn_decision_tree.rda")

# Random Forest
churn_rf_model <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

churn_rf_wf <- workflow() %>%
  add_model(churn_rf_model) %>%
  add_recipe(churn_recipe)

churn_rf_grid <- grid_regular(mtry(range = c(1, 5)),
                              trees(range = c(100, 500)),
                              min_n(range = c(1, 40)),
                              levels = 5)

churn_rf_tune <- tune_grid(
  churn_rf_wf,
  resamples = churn_folds,
  grid = churn_rf_grid,
  metrics = metric_set(roc_auc)
)

churn_rf_best <- select_best(churn_rf_tune, metric = "roc_auc")

churn_rf_final <- finalize_workflow(churn_rf_wf, churn_rf_best)

churn_rf_fit <- fit(churn_rf_final, data = churn_train)

save(churn_rf_fit, churn_rf_final, churn_rf_tune,
     file = "/Users/User/OneDrive/Desktop/Final Project/RDA/churn_random_forest.rda")

# K-Nearest Neighbors
churn_knn_mod <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

churn_knn_wf <- workflow() %>% 
  add_model(churn_knn_mod) %>% 
  add_recipe(churn_recipe)

churn_knn_grid <- grid_regular(neighbors(range = c(1, 30)), levels = 10)

churn_knn_tune <- tune_grid(
  churn_knn_wf,
  resamples = churn_folds,
  grid = churn_knn_grid,
  metrics = metric_set(roc_auc)
)

churn_knn_best <- select_best(churn_knn_tune, metric = "roc_auc")

churn_knn_final <- finalize_workflow(churn_knn_wf, churn_knn_best)

churn_knn_fit <- fit(churn_knn_final, data = churn_train)

save(churn_knn_fit, churn_knn_final, churn_knn_tune,
     file = "/Users/User/OneDrive/Desktop/Final Project/RDA/churn_knn.rda")