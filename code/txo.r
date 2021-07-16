library(dplyr)
library(magrittr) #pipelines
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(purrr)

test <- function(input_data){
  mydf <- read.csv("my_data.csv", row.names = 1, header= TRUE)
  kmedoid.cluster <- pam(x= mydf, k = input_data, diss = TRUE)
  
  
  id <- kmedoid.cluster$id.med
  id <- paste(id,collapse=" ")
  med <- kmedoid.cluster$medoids
  med <- paste(med,collapse=" ")
  
  logFile = "med.txt"
  cat(id, file=logFile, append=FALSE, sep = "\n")
  cat(med, file=logFile, append=TRUE, sep = "\n")
  write.csv(kmedoid.cluster$clustering, file = "clustering.csv")
}



