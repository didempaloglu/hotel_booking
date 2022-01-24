library(dplyr)
library(magrittr)
library(ggplot2)
library(gridExtra)
library(corrplot)
library(caret)
library(readr)
library(tibble)
library(purrr)
library(DescTools)
library(lmtest)
library(pROC)
library(DMwR)
library(Information)
library(janitor)
library(olsrr)
library(verification)
library(class)
library(kernlab)

setwd("C:\\Users\\user\\Desktop\\ML1_FinalProject")


booking <- read.csv("hotel_bookings.csv", sep = ";")
colnames(booking)

booking <- booking[!(booking$adr== 0), ] # we remove the rows with 0 adr

glimpse(booking)
summary(booking)

# integers into factor
booking$is_canceled <- as.factor(booking$is_canceled)
booking$is_repeated_guest <- as.factor(booking$is_repeated_guest)
booking$reservation_status_date <- as.Date.character(booking$reservation_status_date, tryFormats = c("%Y.%m.%d"))

colSums(is.na(booking)) %>% 
  sort()


### Descriptive Visualization

# Distribution of ADR

ggplot(booking, aes(adr)) +
  geom_boxplot() +
  ggtitle("Avearage Daily Rates of Bookings")

ggplot(booking,
       aes(x = adr)) +
  geom_histogram(fill = "blue",
                 bins = 100) +
  theme_bw()

# mean adr by room type       
ggplot(data = booking,
       aes(x=assigned_room_type, y=adr)) +
  stat_summary(fun="mean", geom="bar", fill="darkgreen")



# Cancelization by Hotel Type
ggplot(data = booking,aes(x = is_canceled, fill=is_canceled)) + 
  geom_bar() +
  xlab("Canceled Bookings") +
  ylab("Frequencies") +
  ggtitle("Cancelazation Frequencies by Hotel") +
  facet_grid(~hotel)

# Most frequent arrival months

booking %>%
  group_by(arrival_date_month) %>%
  summarise(Count = n()) %>%
  arrange(-Count) %>%
  ggplot(aes(x = reorder(arrival_date_month, Count), Count)) +
  xlab("Frequencies") +
  geom_bar(stat = 'identity',fill = "purple") + 
  coord_flip() + 
  geom_text(aes(x =arrival_date_month, y=0.02, label= Count),
            hjust=-1, vjust=0, size=4, 
            colour="white",
            angle=360)

# Stays Length by Hotel
ggplot(booking, aes(stays_in_week_nights, fill = hotel)) +
  geom_bar(aes(y=(..count..)), position="dodge") +
  ylab("Frequency") +
  ggtitle("Number of Stays in Weekdays by Hotel")

ggplot(booking, aes(stays_in_weekend_nights, fill = hotel)) +
  geom_bar(aes(y=(..count..)), position="dodge") +
  ylab("Frequency") +
  ggtitle("Number of Stays in Weekend by Hotel")


#Distribution Channel

data <- data.frame(a= booking$distribution_channel, b=1:length(booking$distribution_channel))
data <- data %>% 
  group_by(a) %>% 
  count() %>% 
  ungroup() %>% 
  mutate(per=`n`/sum(`n`)) %>% 
  arrange(desc(a))
data$label <- scales::percent(data$per)
ggplot(data=data)+
  geom_bar(aes(x="", y=per, fill=a), stat="identity", width = 1)+
  coord_polar("y", start=0)+
  theme_void()+
  geom_text(aes(x=1, y = cumsum(per) - per/2, label=label)) 

# data division

set.seed(987654321)

booking_which_training <- createDataPartition(booking$adr,
                                              p = 0.7, 
                                              list = FALSE) 
booking_train <- booking[booking_which_training,]
booking_test <- booking[-booking_which_training,]

# Numeric columns
num_vars <-sapply(booking, is.numeric) %>% 
  which() %>% 
  names() 

booking_correlations <-
  cor(booking_train[,num_vars],
      use = "pairwise.complete.obs")

booking_numeric_vars_order <- 
  # we take correlations with the Sale_Price
  booking_correlations[,"adr"] %>% 
  # sort them in the decreasing order
  sort(decreasing = TRUE) %>%
  # end extract just variables' names
  names()

corrplot.mixed(booking_correlations[booking_numeric_vars_order, 
                                    booking_numeric_vars_order],
               upper = "circle",
               lower = "circle",
               tl.col="black", 
               tl.pos = "lt",
               tl.cex=0.6) 

categ_vars <- 
  # check if variable is a factor
  sapply(booking, is.factor) %>% 
  # select those which are
  which() %>% 
  # and keep just their names
  names()

booking_F_anova <- function(categorical_vars) {
  anova_ <- aov(booking_train$adr ~ 
                  booking_train[[categorical_vars]]) 
  
  return(summary(anova_)[[1]][1, 4])
}

sapply(categ_vars,
       booking_F_anova) %>% 
  # in addition lets sort them
  # in the decreasing order of F
  #  and store as an object
  sort(decreasing = TRUE) -> booking_anova_all_categorical

booking_anova_all_categorical


( findLinearCombos(booking_train[, num_vars] ) ->
    booking_linearCombos ) 


booking_nzv_stats <- nearZeroVar(booking_train,
                                 saveMetrics = TRUE)


booking_nzv_stats %>%
  # we add rownames of the frame
  # (with names of variables)
  # as a new column in the data
  rownames_to_column("zmienna") %>%
  # and sort it in the descreasing order
  arrange(-zeroVar, -nzv, -freqRatio)

booking_variables_all <- names(booking_train)

booking_variables_nzv <- nearZeroVar(booking_train, 
                                     names = TRUE) 

booking_variables_selected <-
  booking_variables_all[!booking_variables_all %in% 
                          booking_variables_nzv]

booking_variables_selected

# Regression

options(contrasts = c("contr.treatment",  # for non-ordinal factors
                      "contr.treatment")) # for ordinal factors

booking_lm1 <- lm(adr ~ .,
                  data = booking_train)

summary(booking_lm1)
coef(booking_lm1)[is.na(coef(booking_lm1))]

booking_variables_selected <-
  booking_variables_selected[-which(booking_variables_selected %in% 
                                      c("reservation_status"))]
booking_lm2 <- lm(adr ~ ., 
                  data = booking_train %>% 
                    dplyr::select(all_of(booking_variables_selected))) # training data

summary(booking_lm2)

#elimination

ols_step_backward_p(booking_lm2,
                    
                    prem = 0.05,
                    # show progress
                    progress = TRUE) -> booking_lm2_backward_p

summary(booking_lm2_backward_p$model)

booking_lm2_backward_p$removed

ols_step_backward_aic(booking_lm2, 
                      progress =  TRUE) -> booking_lm2_backward_AIC

summary(booking_lm2_backward_AIC$model)

booking_lm2_backward_AIC$predictors

# Do both approaches give the same result?

coef(booking_lm1_backward_AIC$model) %>% 
  names() %>% 
  sort() -> coef_list_AIC

coef(booking_lm1_backward_p$model) %>% 
  names() %>% 
  sort() -> coef_list_p

identical(coef_list_AIC, 
          coef_list_p)

# YESSS!!!


booking_variables_selected <-
  booking_variables_selected[-which(booking_variables_selected %in% 
                                      c("reservation_status_date", 
                                        "arrival_date_week_number"))]


source("F_regression_metrics.R")


# lets apply it to our model

# regressionMetrics(real = booking_train$adr,
#                   predicted = predict(booking_lm3))

##### Feature Generation

ggplot(booking_train,
       aes(x = adr)) +
  geom_histogram(fill = "blue",
                 bins = 100) +
  theme_bw()

# clearly right-skewed distribution - let's see
# how it looks after log transformation
# we use log(x + 1) in case of zeros in x

ggplot(booking_train,
       aes(x = log(adr + 1))) +
  geom_histogram(fill="blue",
                 bins = 100) +
  theme_bw()

# booking_lm0_all <- lm(adr ~ .,
#                   data = booking_train %>% 
#                     dplyr::select(all_of(booking_variables_all))) # training data
# 
# booking_lm0_selected <- lm(adr ~ .,
#                       data = booking_train %>% 
#                         dplyr::select(all_of(booking_variables_selected))) # training data
# 
# booking_lm1_all_log <- lm(log(adr + 1) ~ .,
#                          data = booking_train %>% 
#                            dplyr::select(all_of(booking_variables_all)))
# 
# booking_lm1_selected_log <- lm(log(adr + 1) ~ .,
#                            data = booking_train %>% 
#                              dplyr::select(all_of(booking_variables_selected))) # training data
# 
# #####
# 
# booking_models_list <- list(booking_lm0_all = booking_lm0_all,
#                             booking_lm0_selected = booking_lm0_selected,
#                             booking_lm1_all_log = booking_lm1_all_log,
#                             booking_lm1_selected_log = booking_lm1_selected_log)
# 
# booking_models_list %>% 
#   sapply(function(x) predict(x, newdata = booking_train)) %>% 
#   data.frame() -> booking_fitted
# 
# head(booking_fitted)
# 
# 
# booking_fitted$booking_lm1_all_log <- 
#   exp(booking_fitted$booking_lm1_all_log) - 1
# 
# booking_fitted$booking_lm1_selected_log <-
#   exp(booking_fitted$booking_lm1_selected_log) - 1
# 
# head(booking_fitted)
# 
# 
# booking_models_list %>% 
#   sapply(function(x) predict(x, newdata = booking_test)) %>% 
#   data.frame() -> booking_forecasts
# 
# 
# 
# booking_forecasts$booking_lm1_all_log <- 
#   exp(booking_forecasts$booking_lm1_all_log) - 1
# 
# 
# booking_forecasts$booking_lm1_selected_log <-
#   exp(booking_forecasts$booking_lm1_selected_log) - 1
# 
# head(booking_forecasts)
# 
# sapply(booking_fitted,
#        function(x) regressionMetrics(
#                                      booking_train$adr, x)) %>% 
#   t()
# 
# sapply(booking_forecasts,
#        function(x) regressionMetrics(
#          booking_test$adr, x)) %>% 
#   t()
# 
# 

# construction of the model with CV

ctrl_cv5 <- trainControl(method = "cv",
                         number = 5)

# model with all variables
set.seed(987654321)

booking_lm0_all <- train(adr ~ .,
                         data = booking_train,
                         method  = "lm",
                         trControl = ctrl_cv5)

# model with selected variables
set.seed(987654321)
booking_lm0_selected <- train(adr ~ .,
                              data = booking_train %>%
                                dplyr::select(all_of(booking_variables_selected)),
                              method  = "lm",
                              trControl = ctrl_cv5)


# model with all variables and log transformation in target variable adr
set.seed(987654321)
booking_lm1_all_log <- train(log(adr + 1) ~ .,
                             data = booking_train,
                             method  = "lm",
                             trControl = ctrl_cv5)

# model with selected variables and log transformation in target variable adr
set.seed(987654321)

booking_lm1_selected_log <- train(log(adr + 1) ~ .,
                                  data = booking_train %>%
                                    dplyr::select(all_of(booking_variables_selected)),
                                  method  = "lm",
                                  trControl = ctrl_cv5)



booking_models_all <- list(booking_lm2_all_cv5 = booking_lm2_all_cv5,
                           booking_lm2_selected_cv5 = booking_lm2_selected_cv5,
                           booking_lm2_all_log_cv5 = booking_lm2_all_log_cv5,
                           booking_lm2_selected_log_cv5 = booking_lm2_selected_log_cv5)


booking_models_all %>% 
  sapply(function(x) predict(x, newdata = booking_train)) %>% 
  data.frame() -> booking_fitted_all


booking_fitted_all$booking_lm2_all_log_cv5 <- 
  exp(booking_fitted_all$booking_lm2_all_log_cv5) - 1


booking_fitted_all$booking_lm2_selected_log_cv5 <- 
  exp(booking_fitted_all$booking_lm2_selected_log_cv5) - 1

booking_models_all %>% 
  sapply(function(x) predict(x, newdata = booking_test)) %>% 
  data.frame() -> booking_forecast_all


booking_forecast_all$booking_lm2_all_log_cv5 <- 
  exp(booking_forecast_all$booking_lm2_all_log_cv5) - 1

booking_forecast_all$booking_lm2_selected_log_cv5 <- 
  exp(booking_forecast_all$booking_lm2_selected_log_cv5) - 1


booking_models_prediction <- list(booking_fitted_all = booking_fitted_all,
                                  booking_forecast_all = booking_forecast_all)

sapply(booking_fitted_all,
       function(x) regressionMetrics(booking_train$adr, x)) %>% 
  t()

sapply(booking_forecast_all,
       function(x) regressionMetrics(booking_test$adr, x)) %>% 
  t()


### Compare diffrent models

## knn
options(contrasts = c("contr.treatment",  # for non-ordinal factors
                      "contr.treatment")) # for ordinal factors

sqrt(nrow(booking_train))

k_value <- data.frame(k = 287)

set.seed(987654321)

booking_lm_knn <-train(log(adr + 1) ~.,
                       data = booking_train %>%
                         dplyr::select(all_of(booking_variables_selected)),
                       method  = "knn",
                       trControl = ctrl_cv5) 

booking_lm_knn_fitted <- predict(booking_lm_knn,
                                 booking_train)

booking_lm_knn_fitted <- exp(booking_lm_knn_fitted) -1


booking_lm_knn_forecast <- predict(booking_lm_knn,
                                   booking_test)

booking_lm_knn_forecast <- exp(booking_lm_knn_forecast) -1


regressionMetrics(booking_train$adr, booking_lm_knn_fitted )
regressionMetrics(booking_test$adr, booking_lm_knn_forecast )

#SVM

# linear

options(contrasts = c("contr.treatment",  # for non-ordinal factors
                      "contr.treatment")) # for ordinal factors

# parametersC <- data.frame(C = c(0.001, 0.01, 0.02, 0.05, 
#                                 0.1, 0.2, 0.5, 1, 2, 5))

set.seed(987654321)

booking.svm_Linear <- train(adr ~ ., 
                            data = booking_train %>%
                              dplyr::select(all_of(booking_variables_selected)), 
                            method = "svmLinear",
                            # tuneGrid = parametersC,
                            trControl = ctrl_cv5)

booking.svm_Linear

## poly

svm_parametersPoly <- expand.grid(C = c(0.001, 1),
                                  degree = 2:5, 
                                  scale = 1)

svm_parametersPoly

set.seed(987654321)

booking.svm_poly <- train(adr ~ ., 
                          data = booking_train %>%
                            dplyr::select(all_of(booking_variables_selected)), 
                          method = "svmPoly",
                          # tuneGrid = svm_parametersPoly,
                          trControl = ctrl_cv5)

booking.svm_poly

# radial

parametersC_sigma <- 
  expand.grid(C = c(0.01, 0.05, 0.1, 0.5, 1, 5),
              sigma = c(0.05, 0.1, 0.2, 0.5, 1))

set.seed(987654321)

booking.svm_Radial <- train(adr ~ ., 
                            data = booking_train %>%
                              dplyr::select(all_of(booking_variables_selected)),  
                            method = "svmRadial",
                            tuneGrid = parametersC_sigma,
                            trControl = ctrl_cv5)

plot(booking.svm_Radial$finalModel)

booking.svm_Radial_forecast <- predict(booking.svm_Radial, 
                                       data2_test)


regressionMetrics(booking_test$adr, booking.svm_Radial_forecast)



data2.svm_Radial1

# random forest

set.seed(987654321)

booking_rf_cv5 <- train(adr ~ ., 
                        data = booking_train %>%
                          dplyr::select(all_of(booking_variables_selected)),  
                        method = "rf",
                        trControl = ctrl_cv5)



# LASSO & RIDGE

#mixed method

parameters_elastic <- expand.grid(alpha = seq(0, 1, 0.2), 
                                  lambda = seq(10, 1e4, 10))

set.seed(987654321)

booking_elastic <- train(adr ~ .,
                         data = booking_train %>%
                           dplyr::select(all_of(booking_variables_selected)),
                         method = "glmnet", 
                         tuneGrid = parameters_elastic,
                         trControl = ctrl_cv5)


booking_elastic

# looks like here the ridge regression (alpha = 0)
# gives the best result

plot(booking_elastic)

# Mixing percentage = alpha
# Regularization parameter = lambda

elastic_fitted <- predict(booking_elastic, booking_train)


elastic_forecast <- predict(booking_elastic, booking_test)


regressionMetrics(real = booking_train$adr,
                  predicted = elastic_fitted)

regressionMetrics(real = booking_test$adr,
                  predicted = elastic_forecast)




