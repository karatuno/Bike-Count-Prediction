rm(list = ls())
#set current working directory
setwd("F:/analytics_basics/bike_prediction")
# importing all required library
required_library <- c('ggplot2', 'corrgram', 'corrplot', 'randomForest',
                      'caret', 'class', 'e1071', 'rpart', 'mlr','grid',
                      'DMwR','usdm','dplyr','caTools','LiblineaR')

# checking for each library whether installed or not
# if not install then installing it first and then attaching to file
for (lib in required_library){
  if(!require(lib, character.only = TRUE))
  {
    install.packages(lib)
    require(lib, character.only = TRUE)
  }
}

# removing extra variable
rm(required_library,lib)

data = read.csv("day.csv")



#     Data Interpretation and Visualizations       #


# cheking datatypes of all columns
str(data)

numeric_columns <- c('temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt')
cat_columns <- c('season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit')

### checking numerical variables ###
# Checking numerical statistics of numerical columns (Five point summary + mean of all column)
summary(data[,numeric_columns])

### Checking categorical variable ###
# unique values in each category
lapply(data[,cat_columns], function(feat) length(unique(feat)))

# counting of each unique values in categorical columns
lapply(data[,cat_columns], function(feature) table(feature))


#   Missing value analysis   #

# checking missing value for each column and storing counting in dataframe with column name
missing_val <- data.frame(lapply(data, function(feat) sum(is.na(feat))))


#   outlier analysis     #

# box_plot function to plot boxplot of numerical columns
box_plot <- function(column, dataset){
  dataset$x = 1
  ggplot(aes_string(x= 'x', y = column), data = dataset)+
    stat_boxplot(geom = 'errorbar', width = 0.5)+
    geom_boxplot(outlier.size = 2, outlier.shape = 18)+
    labs(y = "", x = column)+
    ggtitle(paste(" BP :",column))
}

# hist_plot function to plot histogram of numerical variable
hist_plot <- function(column, dataset){
  ggplot(aes_string(column), data = dataset)+
    geom_histogram(aes(y=..density..), fill = 'skyblue2')+
    geom_density()+
    labs(x = gsub('\\.', ' ', column))+
    ggtitle(paste(" Histogram :",gsub('\\.', ' ', column)))
}

# calling box_plot function and storing all plots in a list
all_box_plots <- lapply(c('temp', 'atemp', 'hum', 'windspeed'),box_plot, dataset = data)

# calling hist_plot function and storing all plots in a list
all_hist_plots <- lapply(c('temp', 'atemp', 'hum', 'windspeed'),hist_plot, dataset = data)

# printing all plots in one go
gridExtra::grid.arrange(all_box_plots[[1]],all_box_plots[[2]],all_box_plots[[3]],all_box_plots[[4]],
                        all_hist_plots[[1]],all_hist_plots[[2]],all_hist_plots[[3]],all_hist_plots[[4]],ncol=4,nrow=2)


#   Feature Engineering      #

# method which will plot barplot of a columns with respect to other column
plot_bar <- function(cat, y, fun){
  gp = aggregate(x = data[, y], by=list(cat=data[, cat]), FUN=fun)
  ggplot(gp, aes_string(x = 'cat', y = 'x'))+
    geom_bar(stat = 'identity')+
    labs(y = y, x = cat)+
    ggtitle(paste("Bar plot for",y,"wrt to",cat))
}

# plotting cnt with respect to month
plot_bar('mnth', 'cnt', 'sum')

# plotting cnt with respect to yr
plot_bar('yr', 'cnt', 'sum')

# plotting cnt with respect to yr
plot_bar('weekday', 'cnt', 'sum')

# making bins of mnth and weekday
# changing values of month 5th to 10th as 1 and others 0
data = transform(data, mnth = case_when(
  mnth <= 4 ~ 0, 
  mnth >= 11 ~ 0,
  TRUE   ~ 1 
))
colnames(data)[5] <- 'month_feat'

# changing values of weekday for day 0 and 1 the value will be 0
#and 1 for rest
data = transform(data, weekday = case_when(
  weekday < 2 ~ 0, 
  TRUE   ~ 1 
))
colnames(data)[7] <- 'week_feat'

#   Feature Selection        #

# correlation plot for numerical feature
corrgram(data[,numeric_columns], order = FALSE,
         upper.panel = panel.pie, text.panel = panel.txt,
         main = "Correlation Plot for bike data set")

# heatmap plot for numerical features
corrplot(cor(data[,numeric_columns]), method = 'color', type = 'lower')

cat_columns <- c('season', 'yr', 'month_feat', 'holiday', 'week_feat', 'workingday', 'weathersit')

# making every combination from cat_columns
combined_cat <- combn(cat_columns, 2, simplify = F)


# doing chi-square test for every combination
for(i in combined_cat){
  print(i)
  print(chisq.test(table(data[,i[1]], data[,i[2]])))
}


# creating another dataset with dropping outliers
data_wo <- data

# removing outliers from hum and windspeed columns
for (i in c('hum', 'windspeed')){
  out_value = data_wo[,i] [data_wo[,i] %in% boxplot.stats(data_wo[,i])$out]
  data_wo = data_wo[which(!data_wo[,i] %in% out_value),]
}

# checking dimension of both dataset
dim(data)
dim(data_wo)

# dropping unwanted columns
drop_col <- c('instant', 'dteday', 'holiday', 'atemp', 'casual', 'registered')
data[,drop_col]<- NULL
data_wo[,drop_col] <- NULL


#   Building models                  #

set.seed(1)
split = sample.split(data$cnt, SplitRatio = 0.80)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

split = sample.split(data_wo$cnt, SplitRatio = 0.80)
train_set_wo = subset(data_wo, split == TRUE)
test_set_wo = subset(data_wo, split == FALSE)

#Using R2 parameter with k fold cross validation in R
# making a function which will train model on training data and would show 
# K-fold R2 score , R2 score for test dataset and train dataset
fit.predict.show.performance <- function(method, train_data, test_data){
  reg_fit <- caret::train(cnt~., data = train_data, method = method)
  
  y_pred <- predict(reg_fit, test_data[,-10])
  print("R2 on test dataset")
  print(caret::R2(y_pred, test_data[,10])) 
  
  y_pred <- predict(reg_fit, train_data[,-10])
  print("R2 on train dataset")
  print(caret::R2(y_pred, train_data[,10]))
  
  # creating 10 folds of data
  ten_folds = createFolds(train_data$cnt, k = 10)
  ten_cv = lapply(ten_folds, function(fold) {
    training_fold = train_data[-fold, ]
    test_fold = train_data[fold, ]
    reg_fit <- caret::train(cnt~., data = training_fold, method = method)
    
    y_pred <- predict(reg_fit, test_fold[,-10])
    return(as.numeric(caret::R2(y_pred, test_fold[,10]))) 
  })
  sum = 0
  for(i in ten_cv){
    sum = sum + as.numeric(i)
  }
  print("K-fold (K =10) explained variance")
  print(sum/10)
}


#   Linear Regression   #


# building model for dataset 
fit.predict.show.performance('lm', train_set, test_set)

# building model for dataset  without  outliers
fit.predict.show.performance('lm', train_set_wo, test_set_wo)

#         KNN           #


# building model for dataset 
fit.predict.show.performance('knn', train_set, test_set)

# building model for dataset  without  outliers
fit.predict.show.performance('knn', train_set_wo, test_set_wo)


#        SVM            #

# building model for dataset 
fit.predict.show.performance('svmLinear3', train_set, test_set)

# building model for dataset  without  outliers
fit.predict.show.performance('svmLinear3', train_set_wo, test_set_wo)


# Decision Tree Regression  #


# building model for dataset 
fit.predict.show.performance('rpart2', train_set, test_set)

# building model for dataset  without  outliers
fit.predict.show.performance('rpart2', train_set_wo, test_set_wo)


#  Random Forest        #


# building model for dataset 
fit.predict.show.performance('rf', train_set, test_set)

# building model for dataset without  outliers
fit.predict.show.performance('rf', train_set_wo, test_set_wo)


#     XGBRegressor      #


# building model for dataset 
fit.predict.show.performance('xgbTree', train_set, test_set)

# building model for dataset  without  outliers
fit.predict.show.performance('xgbTree', train_set_wo, test_set_wo)


#        Hyperparameter tuning             #


# tuning Random Forest 

control <- trainControl(method="repeatedcv", number=10, repeats=3)
reg_fit <- caret::train(cnt~., data = train_set, method = "rf",trControl = control)
reg_fit$bestTune
y_pred <- predict(reg_fit, test_set[,-10])
print(caret::R2(y_pred, test_set[,10]))


#      tuning XGB       #


control <- trainControl(method="repeatedcv", number=10, repeats=3)
reg_fit <- caret::train(cnt~., data = train_set, method = "xgbTree",trControl = control)
reg_fit$bestTune
y_pred <- predict(reg_fit, test_set[,-10])
print(caret::R2(y_pred, test_set[,10]))


