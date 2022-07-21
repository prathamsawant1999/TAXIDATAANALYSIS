# Removing all global environment variables from memory
# Except the loaded dataset
rm(list=ls()[ls() != "DATASET_GLOBAL"])

# Clear Console Area
cat("\014")


install.packages("h2o")
install.packages("Metrics")
install.packages("data.table")
install.packages("rnn")
install.packages("bit64")
install.packages("parallel")
install.packages("future.apply")
install.packages("vars")
install.packages( "ggplot2")
install.packages("reshape2")
install.packages("tidyverse")


library("h2o")
library("Metrics")
library("data.table")
library("rnn")
library("bit64")
library("parallel")
library("future.apply")
library("vars")
library("ggplot2")
library("reshape2")
library("tidyverse")

#Constants

OUTPUT_FIELD_NAME <- "total_trips"
DATASET_URL <- "Taxi_Trips.csv"
DATASET_CLEANED_URL <- "Taxi_Trips_cleaned.csv"
DATASET_PREPROCESSED_URL <- "Dataset_Preprocessed.csv"
DATASET_SPLIT_THRESHOLD <- 0.46        #Splitting dataset threshold; 70% Training, 30% Testing
EVALUATION_METRICS_URL <- "Metrics.csv"
NO_OF_TRIPS_2019 <- "eda/2019/No_of_Trips_2019.csv"
TOTAL_MILES_2019 <- "eda/2019/TotalMiles_2019.csv"
TOTAL_FARE_2019 <- "eda/2019/TotalFare_2019.csv"
TOTAL_TIME_2019 <- "eda/2019/TotalTime_2019.csv"
AVG_TIME_2019 <- "eda/2019/AvgTime_2019.csv"
AVG_MILES_2019 <- "eda/2019/AvgMiles_2019.csv"
AVG_FARE_2019 <- "eda/2019/AvgFare_2019.csv"
PAYMENT_DF_2020 <-  "eda/2020/Payment_2020.csv"
PAYMENT_DF_2019 <-  "eda/2019/Payment_2019.csv"
NO_OF_TRIPS_2020 <- "eda/2020/No_of_Trips_2020.csv"
TOTAL_MILES_2020 <- "eda/2020/TotalMiles_2020.csv"
TOTAL_FARE_2020 <- "eda/2020/TotalFare_2020.csv"
TOTAL_TIME_2020 <- "eda/2020/TotalTime_2020.csv"
AVG_TIME_2020 <- "eda/2020/AvgTime_2020.csv"
AVG_MILES_2020 <- "eda/2020/AvgMiles_2020.csv"
AVG_FARE_2020 <- "eda/2020/AvgFare_2020.csv"
COMMUNITY_AREA_2019 <-  "eda/2019/Community_area_2019.csv"
COMMUNITY_AREA_2020 <-  "eda/2020/Community_area_2020.csv"
COMPANY_2019 <-  "eda/2019/Company_2019.csv"
COMPANY_2020 <-  "eda/2020/Company_2020.csv"
# No of rows per batch processing
BATCH_SPLIT <- 1000000   






#****************************************************
# MetricsWriter()
#
# Save the metrics into a csv file to persist evaluation
# metrics in case of system crash
#
# INPUT:  Data.frame: Evaluation Metrics
#
# OUTPUT: 
#
#****************************************************
MetricsWriter <- function(metrics){
 fwrite(metrics, EVALUATION_METRICS_URL)
}



#**************************************************
# RunExperiment()
# This function will run given modeling algorithms
# against the given dataset, and store performance
# metrics in a dataframe.
#
# INPUT: Data.frame <- Dataset
#
# OUTPUT: Data.frame <- Performance Metrics 
#                       for every modelling algorithm
#***************************************************
RunExperiment <- function(dataset){
  
  print("Calling RunExperiment()")
  
  h2o.init(nthreads = -1, max_mem_size="10G")
  h2o.removeAll()
  
  # Just making sure that there are no NA values
  # in the dataset
  dataset <- na.omit(dataset)
  
  # Coverting all columns into numeric
  dataset$community_area <- as.numeric(dataset$community_area)
  dataset$start_time_sin <- as.numeric(dataset$start_date_sin)
  dataset$time_bin <- as.numeric(dataset$time_bin)
  dataset$total_trips <- as.numeric(dataset$total_trips)
  dataset$avg_miles <- as.numeric(dataset$avg_miles)
  dataset$avg_fare <- as.numeric(dataset$avg_fare)
  
  metrics <- data.frame(model_algo = character(0), MSE = numeric(0), RMSE = numeric(0), MAE = numeric(0), community_area = integer(0))
  
  #Splitting the data into training and testing data
  #By 70-30 ratio rule
  
  training_dataset <- dataset[dataset$start_time_sin <= DATASET_SPLIT_THRESHOLD,]
  testing_dataset <- dataset[dataset$start_time_sin > DATASET_SPLIT_THRESHOLD,]
  
  community_areas <- unique(dataset$"community_area")
  
  for(com in community_areas) {
    
    print(paste("Going through community area ", com))
    #community_area <- as.numeric(community_area)
    
    stratified_dataset_train <- training_dataset[training_dataset$community_area == com,]
    stratified_dataset_test <- testing_dataset[testing_dataset$community_area == com,]
    
  
    #Training dataset on Vector Autoregression 
    
    VARModel <- VARTrainer(dataset = stratified_dataset_train)
    
    
    metrics <- rbind(metrics, VARPredictor(VARModel, stratified_dataset_test, com))
    
    
    #Training dataset on Multilayer Perceptron
    
    MLPModel <- MLPTrainer(dataset = stratified_dataset_train)
    
    
    metrics <- rbind(metrics, MLPPredictor(MLPModel, stratified_dataset_test, com)) 
    
    
    #Training dataset on Recurrent Neural Network
    
    
    RNNModel <- RNNTrainer(dataset = stratified_dataset_train)
    
    
    metrics <- rbind(metrics, RNNPredictor(RNNModel, stratified_dataset_test, com)) 
    
    
    MetricsWriter(metrics)
    
    } 
  
  PlotEvaluationMetrics(metrics)
  
}


#**************************************************
# PlotEvaluationMetrics()
# This function uses the evaluation metrics
# dataframe to plot graphs
# 
#
# INPUT: Data.frame <- Evaluation Metrics 
#
#***************************************************
PlotEvaluationMetrics <- function(metrics){
  
  print("Calling PlotEvaluationMetrics()")
  
  metrics_mse<-filter(metrics,community_area==c(14,44,69))
  metrics_mse<-subset(metrics,community_area==14)
  metrics_mse<-rbind(metrics_mse,subset(metrics,community_area==44))
  metrics_mse<-rbind(metrics_mse,subset(metrics,community_area==69))
  
  
  metrics_rmse<-filter(metrics,community_area==c(7,24,56))
  metrics_rmse<-subset(metrics,community_area==7)
  metrics_rmse<-rbind(metrics_rmse,subset(metrics,community_area==24))
  metrics_rmse<-rbind(metrics_rmse,subset(metrics,community_area==56))
  
  metrics_mae<-filter(metrics,community_area==c(56,77,6))
  metrics_mae<-subset(metrics,community_area==56)
  metrics_mae<-rbind(metrics_mae,subset(metrics,community_area==77))
  metrics_mae<-rbind(metrics_mae,subset(metrics,community_area==6))
  
  
  #tiff("Plot_MAE_bar_all_comm.tiff", units="in", width=20, height=10, res=300)
  plt <- ggplot(data=metrics, aes(x=community_area, y=MAE, fill=model_algo)) +
    geom_bar(stat="identity", color="black", position=position_dodge())+
    theme_minimal()+scale_x_discrete(limits=c(1:77))+
    ylim(0,100)+ggtitle("Community Area Vs MAE")+theme(plot.title = element_text(hjust = 0.5))
  #geom_text(aes(label=MSE), vjust=-1, size=3,position = position_dodge(0.9),hjust=1)
  #dev.off()
  print(plt)
  
  #tiff("Plot_RMSE_bar_all_comm.tiff", units="in", width=20, height=10, res=300)
  plt<-ggplot(data=metrics, aes(x=community_area, y=RMSE, fill=model_algo)) +
    geom_bar(stat="identity", color="black", position=position_dodge())+
    theme_minimal()+scale_x_discrete(limits=c(1:77))+
    ylim(0,100)+ggtitle("Community Area Vs RMSE")+theme(plot.title = element_text(hjust = 0.5))
  #geom_text(aes(label=MSE), vjust=-1, size=3,position = position_dodge(0.9),hjust=1)
  #dev.off()
  print(plt)
  
  #tiff("Plot_MSE_bar_all_comm.tiff", units="in", width=20, height=10, res=300)
  plt<-ggplot(data=metrics, aes(x=community_area, y=MSE, fill=model_algo)) +
    geom_bar(stat="identity", color="black", position=position_dodge())+
    theme_minimal()+scale_x_discrete(limits=c(1:77))+
    ylim(0,100)+ggtitle("Community Area Vs MSE")+theme(plot.title = element_text(hjust = 0.5))
  #geom_text(aes(label=MSE), vjust=-1, size=3,position = position_dodge(0.9),hjust=1)
  #dev.off()
  print(plt)
  
  #plt<-tiff("Plot_MAE_top.tiff", units="in", width=20, height=10, res=300)
  plt<-ggplot(data=metrics_mae, aes(x=community_area, y=MAE, fill=model_algo)) +
    geom_bar(stat="identity", color="black", position=position_dodge(),width=8)+
    theme_minimal()+scale_x_discrete(limits=c(56,77,6))+
    
    ggtitle("Community Area Vs MAE")+theme(plot.title = element_text(hjust = 0.5))
  # dev.off()
  print(plt)
  
  #plt<-tiff("Plot_MSE_top.tiff", units="in", width=20, height=10, res=300)
  plt<-ggplot(data=metrics_mse, aes(x=community_area, y=MSE, fill=model_algo)) +
    geom_bar(stat="identity", color="black", position=position_dodge(),width=8)+
    theme_minimal()+scale_x_discrete(limits=c(14,44,69))+
    
    ggtitle("Community Area Vs MSE")+theme(plot.title = element_text(hjust = 0.5))
  # dev.off()
  print(plt)
  
  #tiff("Plot_RMSE_top.tiff", units="in", width=20, height=10, res=300)
  plt<-ggplot(data=metrics_rmse, aes(x=community_area, y=RMSE, fill=model_algo)) +
    geom_bar(stat="identity", color="black", position=position_dodge(),width=8)+
    theme_minimal()+scale_x_discrete(limits=c(7,24,56))+
    
    ggtitle("Community Area Vs RMSE")+theme(plot.title = element_text(hjust = 0.5))
  # dev.off()
  print(plt)
  
}



#*************************************************
# MLPTrainer()
# Trains a Multilayer Perceptron on the 
# training dataset and outputs the model
#
# INPUT: Data.frame <- Dataset
#     
#
# OUTPUT: R Object <- Trained Model
#*************************************************
MLPTrainer <- function(dataset){
  
  mlp_learningrate <- 0.1
  mlp_hidden_layers <- c(20,20)
  mlp_iterations <- 10
  
  dataset <- as.h2o(dataset)
  
  MLPModel <- h2o.deeplearning(x=c(2,3,5,6),
                                        y=c(4),
                                        training_frame = dataset,
                                        hidden = mlp_hidden_layers,
                                        epochs = mlp_iterations,
                                        rate = mlp_learningrate)
  
  return(MLPModel)
  
}



#***********************************************
# MLPPredictor()
# Performs evaluation on testing dataset using 
# trained MLP model
#
# INPUT: R Object <- Trained MLP Model
#        Data.frame <- Testing Dataset
#
# OUTPUT: Data.frame <- Performance Metrics:
#                        Title, Model Title/ID
#                        MSE, (Mean Absolute Error)
#                        RMSE, (Root Mean Squared Error)
#                        MAE, (Mean Absolute Error)
#
#***********************************************
MLPPredictor <- function(model, testing_dataset, com){
  
  testing_dataset <- as.h2o(testing_dataset)
  
  model_performance <- h2o.performance(model=model, newdata=testing_dataset)
  
  metrics <- data.frame(model_algo = "MLP", MSE = model_performance@metrics$MSE,
                        RMSE = model_performance@metrics$RMSE,
                        MAE = model_performance@metrics$mae,
                        community_area = com)
  
  return(metrics)
  
}



#*************************************************
# VARTrainer()
# Trains a Vector AutoRegression Model on the 
# training dataset and outputs the model
#
# INPUT: Data.frame <- Dataset
#     
#
# OUTPUT: R Object <- Trained Model
#*************************************************
VARTrainer <- function(dataset){
  
  predictorFields <- dataset[,c("avg_miles", "avg_fare", "time_bin", "start_time_sin", "total_trips")]

  # Recommended Lag value is 15
  
  VARModel <- VAR(y = predictorFields, p = 15, type = "const", season = NULL) 
  
  return(VARModel)
  
}


#*************************************************
# VARPredictor()
# Evaluates a Vector AutoRegression Model on the 
# testing dataset and outputs the performance metrics
#
# INPUT: Data.frame <- Dataset
#        R Object <- Trained Model
#        character vector <- Title of the model 
#     
#
# OUTPUT: Data.frame <- Performance Metrics
VARPredictor <- function(model, testing_dataset, com){
  
     NO_OF_DAYS <- nrow(testing_dataset)
     
     actual <- testing_dataset$total_trips
     
     predicted <- predict(model, n.ahead = NO_OF_DAYS)$fcst$total_trips[,1]
     
     MSE <- mse(actual = actual, predicted = predicted)
     RMSE <- rmse(actual = actual, predicted = predicted)
     MAE <- mae(actual = actual, predicted = predicted)
     
     metrics <- data.frame(model_algo = "VAR", "MSE" = MSE, "RMSE" = RMSE, "MAE" = MAE, community_area = com)
     
     return(metrics)
  
}


#*************************************************
# RNNTrainer()
# This function trains a Recurrent Neural Network
# using H2O implementation
#
# INPUT: Data.frame <- Dataset
#        Integers <- Model Hyperparameters
#
# OUTPUT: R Object <- Trained Model
#*************************************************
RNNTrainer <- function(dataset){
  
  rnn_learningrate <- 0.1
  rnn_hidden_layers <- c(20,20)
  rnn_iterations <- 10
  
  
  #Time step size
  window_length <- 10
  
  window_length <- sin((2 * pi * window_length)/365)


  outputField <- array(dataset$total_trips, dim=c(1, nrow(dataset)))
  
  
  predictorFields <- array(c(dataset$avg_miles, dataset$avg_fare, dataset$time_bin, dataset$start_time_sin),
                           dim=c(1, nrow(dataset), 4))
    
    
  RNNModel <- trainr(X = predictorFields,
                     Y = outputField,
                     learningrate = rnn_learningrate,
                     hidden_dim = rnn_hidden_layers,
                     numepochs = rnn_iterations,
                     sigmoid = "logistic",
                     use_bias = FALSE,
                     seq_to_seq_unsync = FALSE)
                                 
    
  return(RNNModel)
  
}


#***********************************************
# RNNPredictor()
# Uses the trained RNN Model to predict on the
# testing dataset
#
# INPUT: R Object <- Trained RNN Model
#       Data.frame <- Testing Dataset
#
# OUTPUT: R Object <- Performance Metrics:
#                     R2, (R Squared)
#                     MSE, (Mean Absolute Error)
#                     RMSE, (Root Mean Squared Error)
#                     RMSLE, (Root Mean Squared Logarithmtic Error)
#                     MAE, (Mean Absolute Error)
#
#***********************************************
RNNPredictor <- function(model, testing_dataset, com){
  
   input <- array(c(testing_dataset$avg_miles, testing_dataset$avg_fare, testing_dataset$time_bin, testing_dataset$start_time_sin),
                  dim=c(1, nrow(testing_dataset), 4))
   actual <- array(testing_dataset$total_trips, dim=c(1, nrow(testing_dataset)))
   predicted <- predictr(model, X=input)
  
   MSE <- mse(actual = actual, predicted = predicted)
   RMSE <- rmse(actual = actual, predicted = predicted)
   MAE <- mae(actual = actual, predicted = predicted)
   
   metrics <- data.frame(model_algo = "RNN", "MSE" = MSE, "RMSE" = RMSE, 
                         "MAE" = MAE, community_area = com)
   
   return(metrics)
}

#*************************************************
# DatasetPreprocessing()
# Preprocesses the cleaned dataset for use in 
# modelling
#
# INPUT: Data.frame <- Cleaned Dataset
#
# OUTPUT: Data.frame <- Preprocessed Dataset
#*************************************************
DatasetPreprocessing <- function(dataset){
  
  print("Calling DatasetPreprocessing()")
  
  # Removing rows having null value in any of the columns
 
  dataset$trip_miles <- as.numeric(dataset$trip_miles)
  dataset$trip_total <- as.numeric(dataset$trip_total)
  
  dataset <- na.omit(dataset)
  
  #*************************************************
  #* HELPER FUNCTIONS FOR DATA PREPROCESSING
  #*************************************************
  
  # Function to find the sine value of time
  EncodeTime.Sin<-function(data){
   # print("Sin")
    day <- as.numeric(data[9])
    total_day <- 365
    a <- (2 * 3.14 * day)/ total_day
    return(sin(a)) 
  }
  
  # Function to to calculate time bins
  TimeBinning <- function(data){
    #print("Time")
    t <- as.numeric(data[10])
    
    if(t>= 0 && t<5) {return(1) }
    else if(t>= 5 && t<10) {return(2) }
    else if(t>= 10 && t<15) {return(3) }
    else if(t>= 15 && t<18) {return(4) }
    else {return(5) }  
    
  }
  
  #*************************************************
  #*************************************************
  
  
  dataset$start_date_sin <- apply(dataset, 1 , EncodeTime.Sin)
  dataset$time_bin <- apply(dataset, 1 , TimeBinning)
    
 
  Community_Area<-1
  Date_Sin <-1
  avg_fare <- 1
  avg_miles <- 1
  total_trips <- 1
  Total_Miles <- 1
  Total_Fare <- 1
  Time_bin <- 1
  count<-1
  
  preprocessed_dataset <- data.frame(matrix(0, nrow = 140526, ncol = 9))
  m <- data.frame(matrix(0, ncol = 30, nrow = 2))
  
  for (i in 1:77) {
    
    Subset1= (subset(dataset, pickup_community_area==i))
    Subset1 <- na.omit(Subset1)
    
    for ( j in 1:365)
    {
      Subset2= (subset(Subset1,start_date==j))
      
      for ( k in 1:5)
      {
        Subset3= (subset(Subset2,time_bin==k))
        trips  = nrow(Subset3)
        Miles = sum (Subset3$trip_miles)
        Fare = sum (Subset3$trip_total)
        Avgmiles = Miles/trips
        Avgfare = Fare / trips 
        preprocessed_dataset$community_area[count] = i  # Community Area
        preprocessed_dataset$start_date_sin[count] = Subset3$start_date_sin[1]
        preprocessed_dataset$time_bin[count] = k # Timebin 
        preprocessed_dataset$total_trips[count] = trips 
        preprocessed_dataset$avg_miles[count] = Avgmiles
        preprocessed_dataset$avg_fare[count] = Avgfare
        count = count+1
      }
    }
  }
  
  preprocessed_dataset[c(1:9)] <- NULL
  
  return(preprocessed_dataset)
  
}



#*************************************************
# DatasetCleaning()
# Cleans the dataset by removing unused columns and 
# splitting the timestamp into seperate components
#
# INPUT: Data.frame <- Dataset
#
# OUTPUT: Data.frame <- Cleaned Dataset
#*************************************************
DatasetCleaning <- function(dataset){
  
  print("Calling DatasetCleaning()")
  
  names(dataset)[names(dataset) == '\n                                    payment_type'] <- 'payment_type'
  
  required_columns <- c("trip_id", "trip_start_timestamp", "trip_seconds", "trip_miles",
                        "pickup_community_area", "trip_total", "company", "payment_type")
  
  cleaned_dataset <- dataset[, ..required_columns]
  
  cleaned_dataset <- cleaned_dataset[!is.na(cleaned_dataset$pickup_community_area),]
  
  
  # Batch processing 
  
  batches <- seq(BATCH_SPLIT, nrow(cleaned_dataset), BATCH_SPLIT)
  
  batches <- append(batches, nrow(cleaned_dataset))
  
  
  batch_index <- 1
  
  if(file.exists(DATASET_CLEANED_URL)) {
    
    DATASET_CLEANED_TEMP <- fread(DATASET_CLEANED_URL)
    
    previous_rows <- nrow(DATASET_CLEANED_TEMP) + 1
    
    if((previous_rows - 1) == nrow(DATASET_CLEANED_TEMP)){
      return(DATASET_CLEANED_TEMP)
    }
     
    print(paste("Resuming from row ", as.character(previous_rows)))
    
    batch_index <- match(previous_rows, batches) + 1
    
  }
  
  
  for(i in batch_index:length(batches))
  {
    print(paste("Going through Batch ", i))
    
    dataset_batch <- cleaned_dataset[(BATCH_SPLIT*(i-1)):(batches[i] - 1),]
    
    startTp <- as.list(strptime(dataset_batch$trip_start_timestamp, "%m/%d/%Y %I:%M:%S %p"))
    
    startDate_Y <- as.numeric(format(startTp, "%Y"))
    startDate_M <- as.numeric(format(startTp, "%m"))
    startDate_D <- as.numeric(format(startTp, "%d"))
    startTime_H <- as.numeric(format(startTp, "%H"))
   
    day_in_year_fn <- function(M, D) {
      
      day_in_year <- D
      
      if (M == 2){
        day_in_year <- day_in_year + 29 
      }
      else if (M %in% c(1,3,5,7,9,12)){
        day_in_year <- day_in_year + 30 * (M-1)
      }
      else {
        day_in_year <- day_in_year + 31 * (M-1)
      }
      
      return(day_in_year)
    }
    
    startDate_DY <- mapply(day_in_year_fn, startDate_M, startDate_D)
    
    dataset_batch$"start_date_year" <- startDate_Y
    
    dataset_batch$"start_date" <- startDate_DY
    
    dataset_batch$"start_time_hours" <- startTime_H
    
    dataset_batch$"start_date_month" <- startDate_M
    
    dataset_batch$trip_start_timestamp <- NULL
    
    fwrite(dataset_batch, DATASET_CLEANED_URL, append = TRUE)
    
  }
  
  rm(list = ls())
  
  cleaned_dataset <- fread(DATASET_CLEANED_URL)
  
  return(cleaned_dataset)
  
}


#*************************************************
# EDA()
# Main Code for Exploratory Data Analysis and 
# Plotting
#
# INPUT: Data.frame <- Cleaned Dataset
#
# OUTPUT: 
#*************************************************
EDA <- function(dataset, year) {  
  
  print("Calling EDA()")
  
  dataset <- na.omit(dataset)
  
  dataset$trip_seconds <- as.numeric(dataset$trip_seconds)
  dataset$trip_miles <- as.numeric(dataset$trip_miles)
  dataset$trip_total <- as.numeric(dataset$trip_total)
  
  No_of_Trips <- vector(mode = "list",12)
  
  TotalMiles<- vector(mode = "list",12)
  TotalFare<- vector(mode = "list",12)
  TotalTime<- vector(mode = "list",12)
  
  
  AvgTime <- vector(mode = "list",12)
  AvgMiles<- vector(mode = "list",12)
  AvgFare<- vector(mode = "list",12)
  i<-0
  
  for (i in 1:12){
    dff<-subset(dataset,start_date_month == i)
    dff<- na.omit(dff)# Subset for the respective month 
    Notrips<-nrow(dff) # No Of trips for that month 
    time<-sum(dff$trip_seconds)  # Sum of seconds
    Distance<-sum(dff$trip_miles)   # Sum of Distance
    fare<-sum(dff$trip_total)    # Sum of Fare
    avg_time <- time/Notrips   
    avg_mile <- Distance/Notrips
    avg_fare<-fare/Notrips
    No_of_Trips[i] <- Notrips
    TotalMiles[i] <- Distance
    TotalFare[i] <- fare
    TotalTime[i] <- time
    AvgTime[i] <- avg_time
    AvgMiles[i] <- avg_mile
    AvgFare[i] <- avg_fare
  }
  
  list <- unique(dataset$payment_type)
  Payment_mode<- vector(mode = "list",length(list))
  
  i<-1
  for ( i in 1:length(list))
  {
    Payment_mode[i] <- nrow(subset(dataset,payment_type== list[i]))
    
  }
  
  Payment_df <- data.frame(Type = unlist(list),Total = unlist(Payment_mode))
  
  if(year == 2019) {
    
    fwrite(No_of_Trips , NO_OF_TRIPS_2019)
    fwrite(TotalMiles , TOTAL_MILES_2019)
    fwrite(TotalFare , TOTAL_FARE_2019)
    fwrite(TotalTime , TOTAL_TIME_2019)
    fwrite(AvgTime , AVG_TIME_2019)
    fwrite(AvgMiles , AVG_MILES_2019)
    fwrite(AvgFare , AVG_FARE_2019)
    fwrite(Payment_df , PAYMENT_DF_2019)
    
    
   
    clean_taxids_19 <- dataset
    
    #top 5 Community areas w.r.t trips
    community_area <- vector(mode = "list")
    community_area <- unique(clean_taxids_19$pickup_community_area)
    
    community_area_trips <- vector(mode = "list")
    for (i in 1:length(community_area)){
      community_area_trips[i] = nrow(subset(clean_taxids_19, pickup_community_area == community_area[i]))
    }
    
    community_area_trips <- data.frame(matrix(unlist(community_area_trips), nrow=length(community_area_trips), byrow=TRUE))
    CA_TRIPS <- data.frame(Community_Area = community_area)
    CA_TRIPS$Total_Trips <- community_area_trips$matrix.unlist.community_area_trips...nrow...length.community_area_trips...
    write.csv(CA_TRIPS, COMMUNITY_AREA_2019)
    
    
    #top 5 Company w.r.t trips
    taxi_company <- vector(mode = "list")
    taxi_company <- unique(clean_taxids_19$company)
    
    taxi_company_trips <- vector(mode = "list")
    for (i in 1:length(taxi_company)){
      taxi_company_trips[i] = nrow(subset(clean_taxids_19, company == taxi_company[i]))
    }
    taxi_company_trips <- data.frame(matrix(unlist(taxi_company_trips), nrow=length(taxi_company_trips), byrow=TRUE))
    COMPANY_TRIPS <- data.frame(Taxi_Company = taxi_company)
    COMPANY_TRIPS$Total_Trips <- taxi_company_trips$matrix.unlist.taxi_company_trips...nrow...length.taxi_company_trips...
    write.csv(COMPANY_TRIPS, COMPANY_2019)
    
    
    
  } else {
    
    fwrite(No_of_Trips , NO_OF_TRIPS_2020)
    fwrite(TotalMiles , TOTAL_MILES_2020)
    fwrite(TotalFare , TOTAL_FARE_2020)
    fwrite(TotalTime , TOTAL_TIME_2020)
    fwrite(AvgTime , AVG_TIME_2020)
    fwrite(AvgMiles , AVG_MILES_2020)
    fwrite(AvgFare , AVG_FARE_2020)
    fwrite(Payment_df , PAYMENT_DF_2020)
    
    clean_taxids_20 <- dataset
    
    #top 5 Community areas w.r.t trips
    community_area_20 <- vector(mode = "list")
    community_area_20 <- unique(clean_taxids_20$pickup_community_area)
    
    community_area_trips_20 <- vector(mode = "list")
    for (i in 1:length(community_area_20)){
      community_area_trips_20[i] = nrow(subset(clean_taxids_20, pickup_community_area == community_area_20[i]))
    }
    
    community_area_trips_20 <- data.frame(matrix(unlist(community_area_trips_20), nrow=length(community_area_trips_20), byrow=TRUE))
    CA_TRIPS_20 <- data.frame(Community_Area = community_area_20)
    CA_TRIPS_20$Total_Trips <- community_area_trips_20$matrix.unlist.community_area_trips_20...nrow...length.community_area_trips_20...
    write.csv(CA_TRIPS_20, COMMUNITY_AREA_2020)
    
    
    #top 5 Company w.r.t trips
    taxi_company_20 <- vector(mode = "list")
    taxi_company_20 <- unique(clean_taxids_20$company)
    
    taxi_company_trips_20 <- vector(mode = "list")
    for (i in 1:length(taxi_company_20)){
      taxi_company_trips_20[i] = nrow(subset(clean_taxids_20, company == taxi_company_20[i]))
    }
    taxi_company_trips_20 <- data.frame(matrix(unlist(taxi_company_trips_20), nrow=length(taxi_company_trips_20), byrow=TRUE))
    COMPANY_TRIPS_20 <- data.frame(Taxi_Company = taxi_company_20)
    COMPANY_TRIPS_20$Total_Trips <- taxi_company_trips_20$matrix.unlist.taxi_company_trips_20...nrow...length.taxi_company_trips_20...
    write.csv(COMPANY_TRIPS_20, COMPANY_2020)
    
  }
}



EDAPlotting <- function(){
  
  mon <- c("January","February","March","April","May","June","July","August","September","October","November","December")
  
  PlotChicagoTaxi <- function(F2019, F2020, m){
    F2019<- as.list(F2019)
    F2019<-unlist(F2019)
    
    F2020<- as.list(F2020)
    F2020<-unlist(F2020)
    if(m== 1){
      Months <- mon
    }
    else{
      Months <- c(1,2,3,4,5,6,7,8,9,10,11,12) 
    }
    total <- data.frame(Months, F2019, F2020)
    total_df <- melt(total, id=c("Months"))
    return(total_df)
    
  }
  
  
  #-------------------------- Payments 2019 & 2020-----------------------------------#
  
  payments<-fread(PAYMENT_DF_2019)
  payments<-payments[order(payments$Total),]
  payments <- subset(payments, select = c(Type, Total))
  payments<-tail(payments,n=7)
  #payments <- payments

  slices <- payments$Total
  lbls <- payments$Type
  pct <- round(slices/sum(slices)*100)
  lbls <- paste(lbls, pct) # add percents to labels
  lbls <- paste(lbls,"%",sep="") # ad % to labels

  bp<- ggplot(payments, aes(x="", y=Total, fill=Type))+
    geom_bar(width = 1, stat = "identity")+ scale_y_continuous(labels = scales::comma) + ggtitle("Payment Method used in 2019")
  pie <- bp + coord_polar("y", start=0)
  print(pie)
  
  #-------------------------- 2020-----------------------------------#
  
  payments<-fread(PAYMENT_DF_2020)
  payments<-payments[order(payments$Total),]
  payments <- subset(payments, select = c(Type, Total))
  payments<-tail(payments,n=5)

  slices <- payments$Total
  lbls <- payments$Type
  pct <- round(slices/sum(slices)*100)
  lbls <- paste(lbls, pct) # add percents to labels
  lbls <- paste(lbls,"%",sep="") # ad % to labels

  bp<- ggplot(payments, aes(x="", y=Total, fill=Type))+
    geom_bar(width = 1, stat = "identity")+ scale_y_continuous(labels = scales::comma) + ggtitle("Payment Method used in 2020")
  pie <- bp + coord_polar("y", start=0)
  print(pie)


  # #-------------------------- Total Fare 2019 & 2020-----------------------------------#

  Fare_2019<-fread(TOTAL_FARE_2019)
  Fare_2020<-fread(TOTAL_FARE_2020)

  totalFare <- PlotChicagoTaxi(Fare_2019,Fare_2020,1)

  #tiff("Plot_totalFare_vs_Year.tiff", units="in", width=20, height=10, res=300)
  plt <- ggplot(totalFare) +
    geom_bar(aes(x = Months, y = value, fill = variable), 
    stat="identity", position = "dodge", width = 0.8) +
    geom_text(aes(x=Months,y=value,label = value, vjust = 1))+
    scale_fill_manual("Year\n", values = c("steelblue2","gold2"),labels = c(" 2019", " 2020")) +
    labs(x="\n Months",y="Total Fare\n") +
    theme_bw(base_size = 20)+ggtitle("Total Fare per month")+theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(angle = 90, hjust = 0.5, vjust = 0.5)) + 
    scale_x_discrete(limits = mon)+
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
    scale_y_continuous(labels = scales::comma)
  #dev.off()
  print(plt)
  
  
  #-------------------------- Total Miles 2019 & 2020-----------------------------------#
  
  Miles_2019<-fread(TOTAL_MILES_2019)
  Miles_2020<-fread(TOTAL_MILES_2020)
  
  
  totalMiles <- PlotChicagoTaxi(Miles_2019,Miles_2020,1)
  
  #tiff("Plot_totalMiles_vs_Year.tiff", units="in", width=20, height=10, res=300)
  plt<-ggplot(totalMiles) +
    geom_bar(aes(x = Months, y = value, fill = variable), 
    stat="identity", position = "dodge", width = 0.8) +
    geom_text(aes(x=Months,y=value,label = value, vjust = 1))+
    scale_fill_manual("Year\n", values = c("paleturquoise3","azure4"),labels = c(" 2019", " 2020")) +
    labs(x="\n Months",y="Total Miles\n") +
    theme_bw(base_size = 20)+ggtitle("Total Miles per month")+theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(angle = 90, hjust = 0.5, vjust = 0.5)) + 
    scale_x_discrete(limits = mon)+
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
    scale_y_continuous(labels = scales::comma)
  
  print(plt)
  #dev.off()
  
  
  
  #-------------------------- Total Trips 2019 & 2020-----------------------------------#
  
  Trips_2019<-fread(NO_OF_TRIPS_2019)
  Trips_2020<-fread(NO_OF_TRIPS_2020)
  
  
  totalTrips <- PlotChicagoTaxi(Trips_2019,Trips_2020,1)
  
  #tiff("Plot_totalTrips_vs_Year.tiff", units="in", width=20, height=10, res=300)
  plt<-ggplot(totalTrips) +
    geom_bar(aes(x = Months, y = value, fill = variable), 
    stat="identity", position = "dodge", width = 0.8) +
    geom_text(aes(x=Months,y=value,label = value, vjust = 1))+
    scale_fill_manual("Year\n", values = c("lightcyan3","rosybrown2"),labels = c(" 2019", " 2020")) +
    labs(x="\n Months",y="Total Trips\n") +
    theme_bw(base_size = 20)+ggtitle("Total Trips per month")+theme(plot.title = element_text(hjust = 0.5)) + 
    scale_x_discrete(limits=mon)+
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),axis.text.x = element_text(angle = 90, hjust = 0.5, vjust = 0.5))+
    scale_y_continuous(labels = scales::comma)
  
  print(plt)
  #dev.off()
  
  
  
  #-------------------------- Average Fare 2019 & 2020-----------------------------------#
  
  Avg_Fare_2019<-fread(AVG_FARE_2019)
  Avg_Fare_2020<-fread(AVG_FARE_2020)
  
  
  AvgFare <- PlotChicagoTaxi(Avg_Fare_2019,Avg_Fare_2020,0)
  
  #tiff("Plot_AverageFare_vs_Year.tiff", units="in", width=20, height=10, res=300)
  
  plt<-ggplot(AvgFare, aes(x= Months, y=value, fill=variable)) + 
    geom_area() + 
    labs(x="\n Months",y="Average Fare\n") +
    scale_fill_manual("Year\n", values = c("slateblue3","darkseagreen2"),labels = c(" 2019", " 2020")) +
    ggtitle("Average Fare per month") + theme(plot.title = element_text(hjust = 0.5)) +
    scale_x_discrete(limits=c(1:12))+
    scale_y_continuous(labels = scales::comma)
  
  
  print(plt)
  #dev.off()
  
  
  #-------------------------- Average Miles 2019 & 2020-----------------------------------#
  
  Avg_Miles_2019<-fread(AVG_MILES_2019)
  Avg_Miles_2020<-fread(AVG_MILES_2020)
  
  
  AvgMiles <- PlotChicagoTaxi(Avg_Miles_2019,Avg_Miles_2020,0)
  
  #tiff("Plot_AverageMiles_vs_Year.tiff", units="in", width=20, height=10, res=300)
  
  plt<-ggplot(AvgMiles, aes(x= Months, y=value, fill=variable)) + 
    geom_area() + 
    labs(x="\n Months",y="Average Miles\n") +
    scale_fill_manual("Year\n", values = c("sienna3","plum3"),labels = c(" 2019", " 2020")) +
    ggtitle("Average Miles per month") + theme(plot.title = element_text(hjust = 0.5)) +
    scale_x_discrete(limits=c(1:12))+
    scale_y_continuous(labels = scales::comma)
  
  print(plt)
  #dev.off()
  
  #-------------------------- Average Time 2019 & 2020-----------------------------------#
  
  Avg_Time_2019<-fread(AVG_TIME_2019)
  Avg_Time_2020<-fread(AVG_TIME_2020)
  
  
  AvgTime <- PlotChicagoTaxi(Avg_Time_2019,Avg_Time_2020,0)
  
  #tiff("Plot_AverageTripTime_vs_Year.tiff", units="in", width=20, height=10, res=300)
  
  plt<-ggplot(AvgTime, aes(x= Months, y=value, fill=variable)) + 
    geom_area() + 
    labs(x="\n Months",y="Average Trip Time\n") +
    scale_fill_manual("Year\n", values = c("cyan4","skyblue1"),labels = c(" 2019", " 2020")) +
    ggtitle("Average Trip Time per month") + theme(plot.title = element_text(hjust = 0.5)) +
    scale_x_discrete(limits=c(1:12))+
    scale_y_continuous(labels = scales::comma)
  print(plt)
  #dev.off()
  
  #-------------------------- Community Area and Trips 2019 & 2020-----------------------------------#
  
  ca_2019<-fread(COMMUNITY_AREA_2019)
  ca_2020<-fread(COMMUNITY_AREA_2020)
  ca_2019 <- na.omit(ca_2019)
  ca_2020 <- na.omit(ca_2020)
  
  Year <-rep(2019,times=77)
  df1<-data.frame(Year)
  Year <-rep(2020,times=77)
  df2<-data.frame(Year)
  
  # ca_2019_df <- cbind(select(ca_2019,c(2,3)),df1)
  # ca_2020_df <- cbind(select(ca_2020,c(2,3)),df2)
  ca_2019_df <- cbind(ca_2019, df1)
  ca_2019_df[,1] <- NULL
  ca_2020_df <- cbind(ca_2020, df2)
  ca_2020_df[,1] <- NULL
  ca_trips_total <- rbind(ca_2019_df,ca_2020_df)
  ca_trips_total$Year <- as.character( ca_trips_total$Year)
  
  # tiff("Plot_CAvsTrips.tiff", units="in", width=20, height=10, res=300)
   
  plt<-ggplot(ca_trips_total, aes(x=Community_Area, y=Total_Trips, group=Year, color=Year)) +
    geom_point(size=3)+
    geom_segment(aes(x=Community_Area, 
                     xend=Community_Area, 
                     y=0, 
                     yend=Total_Trips))+
    labs(x="\n Community Area",y="Total Trips\n") +
    scale_fill_manual("Year\n", values = c("red","greenyellow"),labels = c(" 2019", " 2020")) +
    ggtitle("Trips by all Community Areas in a year") + theme(plot.title = element_text(hjust = 0.5))+
    scale_x_discrete(limits=c(1:77))+
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_rect(fill = 'white', colour = 'red'))+
    scale_y_continuous(labels = scales::comma)
  
  print(plt)
  
  # dev.off()

  
  #-------------------------- Taxi Company and Trips 2019 & 2020-----------------------------------#
  
  company_2019<-fread(COMPANY_2019)
  company_2020<-fread(COMPANY_2020)
  
  
  company_2019_top <- company_2019[1:10,]
  company_2020_top <- company_2020[1:10,]
  
  Year <-rep(2019,times=nrow(company_2019))
  df1<-data.frame(Year)
  Year <-rep(2020,times=nrow(company_2020))
  df2<-data.frame(Year)
  com_2019_df <- cbind(company_2019, df1)
  com_2020_df <- cbind(company_2020, df2)
  
  com_full <- rbind(com_2019_df,com_2020_df)
  com_full$Year <- as.character( com_full$Year)
  
  #tiff("Plot_CompanyvsTrips.tiff", units="in", width=20, height=10, res=300)
  
  plt<-ggplot(com_full, aes(x=Taxi_Company, y=Total_Trips, group=Year, color=Year)) +
    geom_point(size=3)+
    geom_segment(aes(x=Taxi_Company, 
                     xend=Taxi_Company, 
                     y=0, 
                     yend=Total_Trips))+
    labs(x="\n Taxi Company",y="Total Trips\n") +
    scale_fill_manual("Year\n", values = c("blue","cyan"),labels = c(" 2019", " 2020")) +
    ggtitle("Trips by Taxi Companies") + theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(angle = 90, hjust = 1))+
    scale_y_continuous(labels = scales::comma)
  print(plt)
  
  #dev.off()
  
  
  company_2019_top <- company_2019[1:10,]
  company_2020_top <- company_2020[1:10,]
  
  Year <-rep(2019,times=10)
  dft1<-data.frame(Year)
  Year <-rep(2020,times=10)
  dft2<-data.frame(Year)
  
  com_top_2019_df <- cbind(company_2019_top,dft1)
  com_top_2020_df <- cbind(company_2020_top,dft2)
  
  com_top <- rbind(com_top_2019_df,com_top_2020_df)
  com_top$Year <- as.character( com_top$Year)
  
  #tiff("Plot_TopCompanyvsTrips.tiff", units="in", width=20, height=10, res=300)
  
  plt<-ggplot(com_top, aes(x=Taxi_Company, y=Total_Trips, group=Year, color=Year)) +
    geom_point(size=3)+
    geom_segment(aes(x=Taxi_Company, 
                     xend=Taxi_Company, 
                     y=0, 
                     yend=Total_Trips))+
    labs(x="\n Taxi Company",y="Total Trips\n") +
    scale_fill_manual("Year\n", values = c("blue","peru"),labels = c(" 2019", " 2020")) +
    ggtitle("Trips by Top 10 Taxi Companies") + theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(angle = 90, hjust = 1))+
    scale_y_continuous(labels = scales::comma)
  
  
  print(plt)
}

user.input <- function(prompt) {
  if (interactive()) {
    return(readline(prompt))
  } else {
    cat(prompt)
    return(readLines("stdin", n=1))
  }
}



#*********************************************************
# Main Code
# Call your functions here!
#
#*********************************************************

# H2O testing code

#args(h2o.deeplearning)
#help(h2o.deeplearning)
#example(h2o.deeplearning)
#demo(h2o.deeplearning)


print("Analysis & Predicting Chicago Taxi Demand ")
print("By: Group 1: R'vengers")
print("Sanchit Agarwal")
print("Kowshik Kesavarapu")
print("Aravind Vivekanandan") 
print("Kathik Madheswaran")
print("Visalakshi Abirami Meiyappan")
print("Debjoyti Saha")

print("What would you like the program to do")
print("1. Exploratory Data Analysis (Comparing pre-COVID and post-COVID trends")
print("2. Compare performance of predictive algorithms for predicting taxi demand")

choice <- user.input("Enter your answer {1,2}")

# EDA
if(choice == 1) {
  
    # A dirty hack to ensure that only
    # the fully cleaned dataset file is read
    if(!file.exists(DATASET_CLEANED_URL) || nrow(fread(DATASET_CLEANED_URL)) != 18843640){
      DATASET_GLOBAL <- fread(DATASET_URL)
      colnames(DATASET_GLOBAL) <- c("trip_id", "taxi_id", "trip_start_timestamp", 
                                    "trip_end_timestamp", "trip_seconds", "trip_miles",
                                    "pickup_census_tract", "dropoff_census_tract", 
                                    "pickup_community_area", "dropoff_community_area",
                                    "fare", "tips", "tolls", "extras","trip_total","
                                    payment_type", "company", "pickup_centroid_latitude",
                                    "pickup_centroid_longitude", "pickup_centroid_location",
                                    "dropoff_centroid_latitude","dropoff_centroid_longitude",
                                    "dropoff_centroid_location")
      
      DATASET_CLEANED <- DatasetCleaning(DATASET_GLOBAL)
    } else {
      DATASET_CLEANED <- fread(DATASET_CLEANED_URL)
    }
  
  # Checking if data aggregations for the EDA is done 
  # or not
  if(!file.exists(TOTAL_FARE_2020) && !file.exists(TOTAL_FARE_2020)) {
    dataset_2019 <- DATASET_CLEANED[DATASET_CLEANED$start_date_year == 2019]
    dataset_2020 <- DATASET_CLEANED[DATASET_CLEANED$start_date_year == 2020]
    
    EDA(dataset_2019, 2019)
    EDA(dataset_2020, 2020)
  }
  
  #Plotting Code
  EDAPlotting()
  
  
} else if(choice == 2){  
  # Modelling
   
    # DATASET_PREPROCESSED not written in a csv file
    # Need to preprocess
     if(!file.exists(EVALUATION_METRICS_URL)){
       
           if(!file.exists(DATASET_PREPROCESSED_URL)) {
           
           # DATASET_CLEANED not written in a csv file
           # Need to clean data
           if(!file.exists(DATASET_CLEANED_URL)){
             rm(list=ls())
             DATASET_GLOBAL <- fread(DATASET_URL)
             
             DATASET_CLEANED <- DatasetCleaning(DATASET_GLOBAL)
             fwrite(DATASET_CLEANED, DATASET_CLEANED_URL)
             rm(list=ls()[ls() != "DATASET_CLEANED"])
             
           } else {
             DATASET_CLEANED <- fread(DATASET_CLEANED_URL)
           }
           
           
           # Calling Data Preprocessing code
           DATASET_CLEANED <- DATASET_CLEANED[DATASET_CLEANED$start_date_year == 2020]
           DATASET_PREPROCESSED <- DatasetPreprocessing(DATASET_CLEANED)
           fwrite(DATASET_PREPROCESSED, DATASET_PREPROCESSED_URL)
           
         } else {
           
           DATASET_PREPROCESSED <- fread(DATASET_PREPROCESSED_URL)
         }
  
          # Calling Modelling code
          RunExperiment(DATASET_PREPROCESSED)
      
     } 
  
    Metrics <- fread(EVALUATION_METRICS_URL)
    PlotEvaluationMetrics(Metrics)
}
