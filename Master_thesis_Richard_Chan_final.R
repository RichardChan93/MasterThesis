#-------------------------------------
##### MASTER THESIS Richard Chan #####
#-------------------------------------

# Assessing the performance of Machine Learning for a Job Matching Algorithm
# A study of optimal skill granularity on an online gig work platform

#-------------------------------------

# University of Zurich
# Chair of Marketing
# Prof. Dr. Martin Natter

##### READ ME ####
# For more convenience, open this file with RStudio
# This is the code used to calculate the models and results in the thesis
# The code is split into three parts
# The first part loads all required packages and data, transforms 
# it to the format needed to train models, and creates features, plots and data summaries.
# The second and third part trains all models and evaluates the performance,
# as well as measuring the improvements to the algorithm.
# The second part trains the model on worker answers of requests.
# The third part trains the model on worker ratings.
# These parts train the base models separately, 
# before they're compared using different skill granularities.
# Finally, the final model predicts the validation data to measure improvements.
# Each section is commented with descriptions of what each chunk of code does.

#######################################
##### Load Libraries ######

# loads all needed packages (install manually if needed)
library("dplyr") 
library("stargazer") 
library("lmtest")
library("ggplot2")
library("utils")
library("gridExtra")
library("scales")
library("caret")
library("kableExtra")
library("pander")
library("fastDummies")
library("e1071")
library("tidyverse")
library("Hmisc")
library("pROC")
library("PRROC")
library("kernlab")
library("randomForest")
library("xgboost")
library("Matrix")
library("ROSE")
library("Metrics")
library("magick")
library("cutpointr")
library("papeR")
library("DMwR")

# load and install keras (might need restart of session)
# for more infos on keras and Tensorflow with R visit https://tensorflow.rstudio.com/keras/reference/install_keras.html
library("keras")
install_keras()

##### Load Data ####

# set working directory to zipped data
wd <- "/Users/richardchan/Dropbox/FS19/Master Thesis/MA_data"
setwd(wd)

file1 <- "/Users/richardchan/Dropbox/FS19/Master Thesis/MA_data/MA_worker.csv.zip"
file2 <- "/Users/richardchan/Dropbox/FS19/Master Thesis/MA_data/MA_skills.csv.zip"
file3 <- "/Users/richardchan/Dropbox/FS19/Master Thesis/MA_data/MA_company.csv.zip"
file4 <- "/Users/richardchan/Dropbox/FS19/Master Thesis/MA_data/MA_request.csv.zip"
file5 <- "/Users/richardchan/Dropbox/FS19/Master Thesis/MA_data/MA_ratings.csv.zip"
file6 <- "/Users/richardchan/Dropbox/FS19/Master Thesis/MA_data/MA_validationset.csv.zip"

# this extracts the data from the zipped files and removes the .csv file for more space
worker_data <- data.table::fread(utils::unzip(file1, "MA_worker.csv"))
skill_data <- data.table::fread(utils::unzip(file2,"MA_skills.csv"))
company_data <- data.table::fread(utils::unzip(file3,"MA_company.csv"))
validation_data <- data.table::fread(utils::unzip(file6,"MA_validationset.csv"),showProgress=getOption("datatable.showProgress", interactive()))
file.remove(c("MA_company.csv","MA_skills.csv","MA_worker.csv","MA_validationset.csv"))


rm(wd, file1, file2, file3, file4, file5, file6)



##### Data Transformations ####
# this section creates the data for the machine learning models
# data cleaning
# transforms


# remove all unwanted characters from data
skill_data$person_id<-as.character(skill_data$person_id)
skill_data$job_profile<-as.factor(skill_data$job_profile)
skill_data$edu_level<-as.factor(skill_data$edu_level)
skill_data$education<-as.factor(skill_data$education)
skill_data$education_group<-as.factor(skill_data$education_group)
skill_data<-skill_data%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "/", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "-", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "_", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\(.*\\)", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\>", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\<", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\+", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\.", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\.", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., ":", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "&", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "'", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., '"', ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., " ", ""))))%>%
  mutate(jp_educ=as.factor(paste0(as.character(job_profile),as.character(edu_level))))


worker_data$person_id<-as.character(worker_data$person_id)
worker_data<-worker_data%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "/", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "-", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "_", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\(.*\\)", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\>", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\<", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\+", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\.", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\.", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., ":", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "&", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "'", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., '"', ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., ',', ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., " ", ""))))



company_data$company_id<-as.character(company_data$company_id)
company_data<-company_data%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "/", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "-", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "_", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\(.*\\)", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\>", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\<", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\+", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\.", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\.", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., ":", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "&", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., " ", ""))))




validation_data$wj_id<-as.character(validation_data$wj_id)
validation_data$person_id<-as.character(validation_data$person_id)
validation_data$company_id<-as.character(validation_data$company_id)
validation_data$required_jpname<-as.factor(validation_data$required_jpname)
validation_data$required_educname<-as.factor(validation_data$required_educname)
validation_data$required_education<-as.factor(validation_data$required_education)
validation_data$required_education_group<-as.factor(validation_data$required_education_group)
validation_data<-validation_data%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "/", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "-", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "_", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\(.*\\)", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\>", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\<", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\+", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\.", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\.", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., ":", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "&", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "'", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., '"', ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., ',', ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., " ", ""))))%>%
  mutate(required_jp_educ=as.factor(paste0(as.character(required_jpname), as.character(required_educname))))
validation_data$required_educname[validation_data$required_educname=='']<-as.factor(as.character("Novice"))
validation_data$required_education_group[validation_data$required_education_group=='']<-as.factor(as.character("Novice"))
validation_data$job_duration[validation_data$job_duration==0]<-as.numeric(1)

prop.table(table(validation_data$answer))

# create outcome variable of request data
validation_data <- validation_data%>%
  mutate(accepted=as.factor(ifelse(validation_data$answer=="WORKER_ACCEPTED", "YES", "NO")), 
         reacted=as.factor(ifelse(validation_data$answer=="NO_ANSWER", "NO", "YES")))

# create integers from logicals
tempvars <- validation_data%>%select(starts_with("use_"))%>%mutate_all(funs(as.integer(as.logical(.))))
validation_data <- cbind(validation_data%>%select(-starts_with("use")), tempvars)


# create request data from validation_data
request_data <- validation_data%>%
  # filter(validation_data==0)%>%
  filter(reacted=="YES")
# prop.table(table(request_data$answer))

# create rating data from validation_data
rating_data <- validation_data%>%filter(is.na(worker_rating)==F)
# prop.table(table(rating_data$worker_rating))

validation_orig <- validation_data
validation_data <- validation_orig%>%filter(validation_set==1)

# normalize worker ratings by avg company rating
# this is reverted after prediction to get denormalized real errors
rating_data$worker_rating_orig <- rating_data$worker_rating
rating_data$worker_rating <- rating_data$worker_rating_orig/rating_data$avg_rating

# one-hot encode worker data, rest was one-hot encoded in the database itself
worker_data <- worker_data%>%rename(worker_age=age)%>%replace(., is.na(.), 0.0)
worker_data<-worker_data%>%
  dummy_cols(select_columns = c("gender","worker_language"),remove_first_dummy = TRUE)%>%
  select(-one_of(c("gender","worker_language")))


company_data<-company_data%>%dummy_cols(select_columns=c("company_industry"),remove_first_dummy = T)%>%select(-one_of(c("company_industry")))




rm(tempvars)

# sample for code testing
#request_data <- request_data%>%sample_n(100000)
#rating_data <- rating_data%>%sample_n(100000)
#validation_data <- validation_data%>%sample_n(100000)


##### Create Skill Features #####
# creates skill data features and calculates the matching score per granularity
# For each skill granularity, the skill data needs to be regrouped for the score
# Granularities are
# E191 Job profile with education levels (jp_educ)
# E91 Job profile without education levels (jp_name)
# E84 Education level without job profiles (educ_level)
# E31 Grouped education levels, small groups (education)
# E6 Grouped education levels, big groups (edugroup)
# I5 Industry of the job profile (jp_industry)
# I10 Subindustry or sector of the job profile (jp_subindustry)



# JP EDUC NAME
# job profile education names
jp_educ_name_per_worker<-skill_data%>%
  filter(jp_educ!='')%>%
  select(person_id, jp_educ)%>%
  dummy_cols(select_columns = c("jp_educ"))%>%
  select(-one_of(c("jp_educ")))%>%
  group_by(person_id)%>%
  summarise_all(list(sum))

# required jp educ skills
required_jp_educ_name<-rating_data%>%
  dplyr::select(wj_id, required_jp_educ, person_id)%>%
  dummy_cols(select_columns=c("required_jp_educ"))%>%
  dplyr::select(-one_of(c("required_jp_educ")))

requested_jp_educ_name<-request_data%>%
  dplyr::select(wj_id, required_jp_educ, person_id)%>%
  dummy_cols(select_columns=c("required_jp_educ"))%>%
  dplyr::select(-one_of(c("required_jp_educ")))


# since this granularity is the lowest and used for the current matching, the score is always 1 for each request, as otherwise worker wouldn't have received
# join required and provided skills per granularity
request_join_granularity <- requested_jp_educ_name%>%inner_join(jp_educ_name_per_worker, by="person_id")
rating_join_granularity <- required_jp_educ_name%>%inner_join(jp_educ_name_per_worker, by="person_id")

# levels of granularity
granularity_levels <- skill_data%>%
  filter(jp_educ!='')%>%
  select(jp_educ)%>%distinct()%>%mutate(jp_educ=as.character(jp_educ))
  

# max jps per level of granularity
max_per_level <- jp_educ_name_per_worker%>%summarise_all(funs(max))%>%select(-c("person_id"))

# calculate scores for the requests and the ratings
granule_score <- request_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  #i=35
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- request_join_granularity[grep(paste0(a,"$"), colnames(request_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
request_educname_score <- granule_score
sum(request_edugroup_score$scores)

granule_score <- rating_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- rating_join_granularity[grep(paste0(a,"$"), colnames(rating_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score)
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
rating_educname_score <- granule_score

# validation data
validation_jp_educ_name<-validation_data%>%
  dplyr::select(wj_id, required_jp_educ, person_id)%>%
  dummy_cols(select_columns=c("required_jp_educ"))%>%
  dplyr::select(-one_of(c("required_jp_educ")))

validation_join_granularity <- validation_jp_educ_name%>%inner_join(jp_educ_name_per_worker, by="person_id")

granule_score <- validation_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- validation_join_granularity[grep(paste0(a,"$"), colnames(validation_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
validation_educname_score <- granule_score
sum(validation_educname_score$scores)




# JP NAMES
# job profile names
jp_names_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, job_profile)%>%
  dummy_cols(select_columns = c("job_profile"))%>%
  select(-one_of(c("job_profile")))%>%
  group_by(person_id)%>%
  summarise_all(list(sum))

# required jp name skills
required_jp_name<-rating_data%>%
  dplyr::select(wj_id, required_jpname, person_id)%>%
  dummy_cols(select_columns=c("required_jpname"))%>%
  dplyr::select(-one_of(c("required_jpname")))

requested_jp_name<-request_data%>%
  dplyr::select(wj_id, required_jpname, person_id)%>%
  dummy_cols(select_columns=c("required_jpname"))%>%
  dplyr::select(-one_of(c("required_jpname")))

# join required and provided skills per granularity
request_join_granularity <- requested_jp_name%>%inner_join(jp_names_per_worker, by="person_id")
rating_join_granularity <- required_jp_name%>%inner_join(jp_names_per_worker, by="person_id")

# levels of granularity
granularity_levels <- skill_data%>%
  filter(job_profile!='')%>%
  select(job_profile)%>%distinct()%>%mutate(job_profile=as.character(job_profile))


# max jps per level of granularity
max_per_level <- jp_names_per_worker%>%summarise_all(funs(max))%>%select(-c("person_id"))

# calculate scores for the requests and the ratings
granule_score <- request_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- request_join_granularity[grep(paste0(a,"$"), colnames(request_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
request_jpname_score <- granule_score
sum(request_jpname_score$scores)

granule_score <- rating_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- rating_join_granularity[grep(paste0(a,"$"), colnames(rating_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score)
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
rating_jpname_score <- granule_score

# validation data
validation_jp_name<-validation_data%>%
  dplyr::select(wj_id, required_jpname, person_id)%>%
  dummy_cols(select_columns=c("required_jpname"))%>%
  dplyr::select(-one_of(c("required_jpname")))

validation_join_granularity <- validation_jp_name%>%inner_join(jp_names_per_worker, by="person_id")

granule_score <- validation_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  #i=35
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- validation_join_granularity[grep(paste0(a,"$"), colnames(validation_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
validation_jpname_score <- granule_score
sum(validation_jpname_score$scores)




# JP EDULEVEL
# jp edu_level
jp_edu_level_per_worker<-skill_data%>%
  filter(edu_level!='')%>%
  select(person_id, edu_level)%>%
  dummy_cols(select_columns = c("edu_level"))%>%
  select(-one_of(c("edu_level")))%>%
  group_by(person_id)%>%
  summarise_all(list(sum))

# required edu_level
required_jp_edu_level<-rating_data%>%
  dplyr::select(wj_id, required_educname, person_id)%>%
  dummy_cols(select_columns=c("required_educname"))%>%
  dplyr::select(-one_of(c("required_educname")))

requested_jp_edu_level<-request_data%>%
  dplyr::select(wj_id, required_educname, person_id)%>%
  dummy_cols(select_columns=c("required_educname"))%>%
  dplyr::select(-one_of(c("required_educname")))

# join required and provided skills per granularity
request_join_granularity <- requested_jp_edu_level%>%inner_join(jp_edu_level_per_worker, by="person_id")
rating_join_granularity <- required_jp_edu_level%>%inner_join(jp_edu_level_per_worker, by="person_id")

# levels of granularity
granularity_levels <- skill_data%>%
  filter(edu_level!='')%>%
  select(edu_level)%>%distinct()%>%mutate(edu_level=as.character(edu_level))


# max jps per level of granularity
max_per_level <- jp_edu_level_per_worker%>%summarise_all(funs(max))%>%select(-c("person_id"))

# calculate scores for the requests and the ratings
granule_score <- request_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- request_join_granularity[grep(paste0(a,"$"), colnames(request_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
request_edulevel_score <- granule_score
sum(request_edulevel_score$scores)

granule_score <- rating_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- rating_join_granularity[grep(paste0(a,"$"), colnames(rating_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score)
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
rating_edulevel_score <- granule_score

# validation data
validation_educname<-validation_data%>%
  dplyr::select(wj_id, required_educname, person_id)%>%
  dummy_cols(select_columns=c("required_educname"))%>%
  dplyr::select(-one_of(c("required_educname")))

validation_join_granularity <- validation_educname%>%inner_join(jp_edu_level_per_worker, by="person_id")

granule_score <- validation_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- validation_join_granularity[grep(paste0(a,"$"), colnames(validation_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
validation_edulevel_score <- granule_score
sum(validation_edulevel_score$scores)





# JP EDUCATIOIN
# jp education
jp_education_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, education)%>%
  rename(jp_education=education)%>%
  dummy_cols(select_columns = c("jp_education"))%>%
  select(-one_of(c("jp_education")))%>%
  group_by(person_id)%>%
  summarise_all(funs(sum))

# required education
required_jp_education<-rating_data%>%
  dplyr::select(wj_id, required_education, person_id)%>%
  dummy_cols(select_columns=c("required_education"))%>%
  dplyr::select(-one_of(c("required_education")))

requested_jp_education<-request_data%>%
  dplyr::select(wj_id, required_education, person_id)%>%
  dummy_cols(select_columns=c("required_education"))%>%
  dplyr::select(-one_of(c("required_education")))

# join required and provided skills per granularity
request_join_granularity <- requested_jp_education%>%inner_join(jp_education_per_worker, by="person_id")
rating_join_granularity <- required_jp_education%>%inner_join(jp_education_per_worker, by="person_id")

# levels of granularity
granularity_levels <- skill_data%>%
  filter(education!='')%>%
  select(education)%>%distinct()%>%mutate(education=as.character(education))


# max jps per level of granularity
max_per_level <- jp_education_per_worker%>%summarise_all(funs(max))%>%select(-c("person_id"))

# calculate scores for the requests and the ratings
granule_score <- request_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  #i<-22
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- request_join_granularity[grep(paste0(a,"$"), colnames(request_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
request_education_score <- granule_score
sum(request_education_score$scores)

granule_score <- rating_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- rating_join_granularity[grep(paste0(a,"$"), colnames(rating_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
rating_education_score <- granule_score
sum(rating_education_score$scores)

# validation data
validation_educname<-validation_data%>%
  dplyr::select(wj_id, required_education, person_id)%>%
  dummy_cols(select_columns=c("required_education"))%>%
  dplyr::select(-one_of(c("required_education")))

validation_join_granularity <- validation_educname%>%inner_join(jp_education_per_worker, by="person_id")

granule_score <- validation_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- validation_join_granularity[grep(paste0(a,"$"), colnames(validation_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
validation_education_score <- granule_score
sum(validation_education_score$scores)





# JP EDUGROUP
# jp education group
jp_educationgroup_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, education_group)%>%
  rename(jp_edugroup=education_group)%>%
  dummy_cols(select_columns = c("jp_edugroup"))%>%
  select(-one_of(c("jp_edugroup")))%>%
  group_by(person_id)%>%
  summarise_all(funs(sum))

# required jp education_group
required_jp_educationgroup<-rating_data%>%
  dplyr::select(wj_id, required_education_group, person_id)%>%
  rename(required_edugroup=required_education_group)%>%
  dummy_cols(select_columns=c("required_edugroup"))%>%
  dplyr::select(-one_of(c("required_edugroup")))

requested_jp_educationgroup<-request_data%>%
  dplyr::select(wj_id, required_education_group, person_id)%>%
  rename(required_edugroup=required_education_group)%>%
  dummy_cols(select_columns=c("required_edugroup"))%>%
  dplyr::select(-one_of(c("required_edugroup")))

# join required and provided skills per granularity
request_join_granularity <- requested_jp_educationgroup%>%inner_join(jp_educationgroup_per_worker, by="person_id")
rating_join_granularity <- required_jp_educationgroup%>%inner_join(jp_educationgroup_per_worker, by="person_id")

# levels of granularity
granularity_levels <- skill_data%>%
  filter(education_group!='')%>%
  select(education_group)%>%distinct()%>%mutate(education_group=as.character(education_group))


# max jps per level of granularity
max_per_level <- jp_educationgroup_per_worker%>%summarise_all(funs(max))%>%select(-c("person_id"))

# calculate scores for the requests and the ratings
granule_score <- request_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  #i<-22
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- request_join_granularity[grep(paste0(a,"$"), colnames(request_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
request_edugroup_score <- granule_score
sum(request_edugroup_score$scores)

granule_score <- rating_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- rating_join_granularity[grep(paste0(a,"$"), colnames(rating_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
rating_edugroup_score <- granule_score
sum(rating_education_score$scores)

# validation data
validation_edugroup<-validation_data%>%
  dplyr::select(wj_id, required_education_group, person_id)%>%
  dummy_cols(select_columns=c("required_education_group"))%>%
  dplyr::select(-one_of(c("required_education_group")))

validation_join_granularity <- validation_edugroup%>%inner_join(jp_educationgroup_per_worker, by="person_id")

granule_score <- validation_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- validation_join_granularity[grep(paste0(a,"$"), colnames(validation_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
validation_edugroup_score <- granule_score
sum(validation_edugroup_score$scores)






# INDUSTRY
# jp industry
jp_industry_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, industry)%>%
  rename(jp_industry=industry)%>%
  dummy_cols(select_columns = c("jp_industry"))%>% 
  select(-one_of(c("jp_industry")))%>%
  group_by(person_id)%>%
  summarise_all(funs(sum))

required_jp_industry<-rating_data%>%
  dplyr::select(wj_id, required_industry, person_id)%>%
  dummy_cols(select_columns=c("required_industry"))%>%
  dplyr::select(-one_of(c("required_industry")))

requested_jp_industry<-request_data%>%
  dplyr::select(wj_id, required_industry, person_id)%>%
  dummy_cols(select_columns=c("required_industry"))%>%
  dplyr::select(-one_of(c("required_industry")))

# join required and provided skills per granularity
request_join_granularity <- requested_jp_industry%>%inner_join(jp_industry_per_worker, by="person_id")
rating_join_granularity <- required_jp_industry%>%inner_join(jp_industry_per_worker, by="person_id")

# levels of granularity
granularity_levels <- skill_data%>%
  filter(industry!='')%>%
  select(industry)%>%distinct()%>%mutate(industry=as.character(industry))


# max jps per level of granularity
max_per_level <- jp_industry_per_worker%>%summarise_all(funs(max))%>%select(-c("person_id"))

# calculate scores for the requests and the ratings
granule_score <- request_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- request_join_granularity[grep(paste0(a,"$"), colnames(request_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
request_industry_score <- granule_score
sum(request_industry_score$scores)

granule_score <- rating_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- rating_join_granularity[grep(paste0(a,"$"), colnames(rating_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
rating_industry_score <- granule_score
sum(rating_education_score$scores)

# validation data
validation_industry <- validation_data%>%
  dplyr::select(wj_id, required_industry, person_id)%>%
  dummy_cols(select_columns=c("required_industry"))%>%
  dplyr::select(-one_of(c("required_industry")))

validation_join_granularity <- validation_industry%>%inner_join(jp_industry_per_worker, by="person_id")

granule_score <- validation_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- validation_join_granularity[grep(paste0(a,"$"), colnames(validation_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
validation_industry_score <- granule_score
sum(validation_industry_score$scores)





# SUBINDUSTRY
# jp subindustry
jp_subindustry_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, subindustry)%>%
  rename(jp_subindustry=subindustry)%>%
  dummy_cols(select_columns = c("jp_subindustry"))%>%
  select(-one_of(c("jp_subindustry")))%>%
  group_by(person_id)%>%
  summarise_all(funs(sum))

required_jp_subindustry<-rating_data%>%
  dplyr::select(wj_id, required_subindustry, person_id)%>%
  dummy_cols(select_columns=c("required_subindustry"))%>%
  dplyr::select(-one_of(c("required_subindustry")))

requested_jp_subindustry<-request_data%>%
  dplyr::select(wj_id, required_subindustry, person_id)%>%
  dummy_cols(select_columns=c("required_subindustry"))%>%
  dplyr::select(-one_of(c("required_subindustry")))

# join required and provided skills per granularity
request_join_granularity <- requested_jp_subindustry%>%inner_join(jp_subindustry_per_worker, by="person_id")
rating_join_granularity <- required_jp_subindustry%>%inner_join(jp_subindustry_per_worker, by="person_id")

# levels of granularity
granularity_levels <- skill_data%>%
  filter(subindustry!='')%>%
  select(subindustry)%>%distinct()%>%mutate(subindustry=as.character(subindustry))


# max jps per level of granularity
max_per_level <- jp_subindustry_per_worker%>%summarise_all(funs(max))%>%select(-c("person_id"))

# calculate scores for the requests and the ratings
granule_score <- request_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  #i<-22
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- request_join_granularity[grep(paste0(a,"$"), colnames(request_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
request_subindustry_score <- granule_score
sum(request_subindustry_score$scores)

granule_score <- rating_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- rating_join_granularity[grep(paste0(a,"$"), colnames(rating_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
rating_subindustry_score <- granule_score
sum(rating_subindustry_score$scores)

# validation data
validation_subindustry <- validation_data%>%
  dplyr::select(wj_id, required_subindustry, person_id)%>%
  dummy_cols(select_columns=c("required_subindustry"))%>%
  dplyr::select(-one_of(c("required_subindustry")))

validation_join_granularity <- validation_subindustry%>%inner_join(jp_subindustry_per_worker, by="person_id")

granule_score <- validation_join_granularity%>%select(wj_id)
granule_score$scores <- 0
for(i in 1:nrow(granularity_levels)) {
  a <- granularity_levels[i,]
  temp_max_score <- max_per_level[grep(paste0(a,"$"), colnames(max_per_level))]
  ifelse(as.numeric(temp_max_score)==0, temp_max_score<-as.data.frame(1), temp_max_score <- temp_max_score)
  temp_features <- validation_join_granularity[grep(paste0(a,"$"), colnames(validation_join_granularity))]
  ifelse(ncol(temp_features)==1,temp_features[,2]<-0, temp_features<-temp_features)
  scores <- temp_features[,1]*temp_features[,2]/as.numeric(temp_max_score[1,])
  granule_score$scores <- pmax(scores, granule_score$scores)
  print(i)
  rm(temp_max_score, a, temp_features,scores)
}
validation_subindustry_score <- granule_score
sum(validation_subindustry_score$scores)






# join granularity scores
request_granularity_scores <- request_data%>%select(wj_id)%>%
  inner_join(request_edugroup_score, by="wj_id")%>%rename(edugroup_score=scores)%>%
  inner_join(request_education_score, by="wj_id")%>%rename(education_score=scores)%>%
  inner_join(request_edulevel_score, by="wj_id")%>%rename(edulevel_score=scores)%>%
  inner_join(request_educname_score, by="wj_id")%>%rename(educname_score=scores)%>%
  inner_join(request_jpname_score, by="wj_id")%>%rename(jpname_score=scores)%>%
  inner_join(request_industry_score, by="wj_id")%>%rename(industry_score=scores)%>%
  inner_join(request_subindustry_score, by="wj_id")%>%rename(subindustry_score=scores)

rating_granularity_scores <- rating_data%>%select(wj_id)%>%
  inner_join(rating_edugroup_score, by="wj_id")%>%rename(edugroup_score=scores)%>%
  inner_join(rating_education_score, by="wj_id")%>%rename(education_score=scores)%>%
  inner_join(rating_edulevel_score, by="wj_id")%>%rename(edulevel_score=scores)%>%
  inner_join(rating_educname_score, by="wj_id")%>%rename(educname_score=scores)%>%
  inner_join(rating_jpname_score, by="wj_id")%>%rename(jpname_score=scores)%>%
  inner_join(rating_industry_score, by="wj_id")%>%rename(industry_score=scores)%>%
  inner_join(rating_subindustry_score, by="wj_id")%>%rename(subindustry_score=scores)
  
validation_granularity_scores <- validation_data%>%select(wj_id)%>%
  inner_join(validation_edugroup_score, by="wj_id")%>%rename(edugroup_score=scores)%>%
  inner_join(validation_education_score, by="wj_id")%>%rename(education_score=scores)%>%
  inner_join(validation_edulevel_score, by="wj_id")%>%rename(edulevel_score=scores)%>%
  inner_join(validation_educname_score, by="wj_id")%>%rename(educname_score=scores)%>%
  inner_join(validation_jpname_score, by="wj_id")%>%rename(jpname_score=scores)%>%
  inner_join(validation_industry_score, by="wj_id")%>%rename(industry_score=scores)%>%
  inner_join(validation_subindustry_score, by="wj_id")%>%rename(subindustry_score=scores)

rm(request_edugroup_score,request_edugroup
   ,request_education_score,request_education
   ,request_edulevel_score,request_edulevel
   ,request_educname_score,request_educname
   ,request_jpname_score,request_jpname
   ,request_industry_score,request_industry
   ,request_subindustry_score,request_subindustry
   ,rating_edugroup_score,rating_edugroup
   ,rating_education_score,rating_education
   ,rating_edulevel_score,rating_edulevel
   ,rating_educname_score,rating_educname
   ,rating_jpname_score,rating_jpname
   ,rating_industry_score,rating_industry
   ,rating_subindustry_score,rating_subindustry
   ,validation_edugroup_score,validation_edugroup
   ,validation_education_score,validation_education
   ,validation_edulevel_score,validation_edulevel
   ,validation_educname_score,validation_educname
   ,validation_jp_educ_name_score,validation_jp_educ_name
   ,validation_jp_name_score,validation_jp_name
   ,validation_industry_score,validation_industry
   ,validation_subindustry_score,validation_subindustry)

##### Create Other Skill Features #####

# This is the initial approach to process skill data
request_data$wj_id<-as.character(request_data$wj_id)
request_data$person_id<-as.character(request_data$person_id)
request_data$company_id<-as.character(request_data$company_id)

request_data$required_jpname<-as.factor(request_data$required_jpname)
request_data$required_educname<-as.factor(request_data$required_educname)
request_data$required_education<-as.factor(request_data$required_education)
request_data$required_education_group<-as.factor(request_data$required_education_group)

request_data<-request_data%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "/", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "-", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "_", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\(.*\\)", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\>", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\<", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\+", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\.", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "\\.", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., ":", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "&", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., "'", ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., '"', ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., ',', ""))))%>%
  mutate_if(is.factor,funs(as.factor(str_replace_all(., " ", ""))))%>%
  mutate(required_jp_educ=as.factor(paste0(as.character(required_jpname), as.character(required_educname))))
request_data$required_educname[request_data$required_educname=='']<-as.factor(as.character("Novice"))


# job profile education name hours
jp_educ_name_hours_per_worker<- skill_data%>%
  filter(job_profile!='')%>%
  mutate(jp_educ_hours=as.factor(paste(job_profile, edu_level)))%>%
  select(person_id, jp_educ_hours, hours)%>%
  dummy_cols( select_columns = c("jp_educ_hours"))%>%
  mutate_if(is.integer,function(col){.$hours*col})%>%
  select(-one_of(c("jp_educ_hours","hours")))%>%
  group_by(person_id)%>%
  summarise_if(is.numeric,list(sum))

# job profile education name rating
jp_educ_name_rating_per_worker<- skill_data %>%
  filter(job_profile!='' ) %>%
  mutate(jp_educ_rating=as.factor(paste(job_profile, edu_level)))%>%
  select(person_id, jp_educ_rating, average_rating)%>%
  dummy_cols(select_columns = c("jp_educ_rating"))%>%
  mutate_if(is.integer, function(col){.$average_rating*col})%>%
  select(-one_of(c("jp_educ_rating","average_rating")))%>%
  group_by(person_id)%>%
  summarise_if(is.numeric, list(max))



# job profile education name hours
jp_name_hours_per_worker<- skill_data%>%
  filter(job_profile!='')%>%
  rename(jp_name_hours=job_profile)%>%
  select(person_id, jp_name_hours, hours)%>%
  dummy_cols( select_columns = c("jp_name_hours"))%>%
  mutate_if(is.integer,function(col){.$hours*col})%>%
  select(-one_of(c("jp_name_hours","hours")))%>%
  group_by(person_id)%>%
  summarise_if(is.numeric,list(sum))

# job profile education name rating
jp_name_rating_per_worker<- skill_data %>%
  filter(job_profile!='' ) %>%
  rename(jp_name_rating=job_profile)%>%
  select(person_id, jp_name_rating, average_rating, hours)%>%
  dummy_cols(select_columns = c("jp_name_rating"))%>%
  mutate_if(is.integer, function(col){.$average_rating*.$hours*col})%>%
  select(-one_of(c("average_rating")))%>%
  group_by(person_id, jp_name_rating)%>%
  summarise_all(funs(sum))%>%
  ungroup()%>%
  mutate_if(is.numeric, function(col){(col/.$hours)})%>%
  select(-one_of(c("hours","jp_name_rating")))%>%
  group_by(person_id)%>%
  summarise_all(funs(max))%>%
  replace(., is.na(.), 0.0)%>%
  mutate_if(is.numeric, funs(ifelse(. == 0.0, 4.0, .)))




# jp hours education group
jp_hours_educationgroup_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, education_group, hours)%>%
  rename(jp_edugroup_hours=education_group)%>%
  dummy_cols(select_columns = c("jp_edugroup_hours"))%>% 
  mutate_if(is.integer, function(col){.$hours*col})%>%
  select(-one_of(c("jp_edugroup_hours","hours")))%>%
  group_by(person_id)%>%
  summarise_all(funs(sum))

# jp rating education group
jp_rating_educationgroup_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, education_group, average_rating, hours)%>%
  rename(jp_edugroup_rating=education_group)%>%
  dummy_cols(select_columns = c("jp_edugroup_rating"))%>% 
  mutate_if(is.integer, function(col){.$average_rating*.$hours*col})%>%
  select(-one_of(c("average_rating")))%>%
  group_by(person_id, jp_edugroup_rating)%>%
  summarise_all(funs(sum))%>%
  ungroup()%>%
  mutate_if(is.numeric, function(col){(col/.$hours)})%>%
  select(-one_of(c("hours","jp_edugroup_rating")))%>%
  group_by(person_id)%>%
  summarise_all(funs(max))%>%
  replace(., is.na(.), 0.0)%>%
  mutate_if(is.numeric, funs(ifelse(. == 0.0, 4.0, .)))
  


# jp hours education
jp_hours_education_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, education, hours)%>%
  rename(jp_education_hours=education)%>%
  dummy_cols(select_columns = c("jp_education_hours"))%>% 
  mutate_if(is.integer, function(col){.$hours*col})%>%
  select(-one_of(c("jp_education_hours","hours")))%>%
  group_by(person_id)%>%
  summarise_all(funs(sum))

# jp rating education
jp_rating_education_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, education, average_rating, hours)%>%
  rename(jp_education_rating=education)%>%
  dummy_cols(select_columns = c("jp_education_rating"))%>% 
  mutate_if(is.integer, function(col){.$average_rating*.$hours*col})%>%
  select(-one_of(c("average_rating")))%>%
  group_by(person_id, jp_education_rating)%>%
  summarise_all(funs(sum))%>%
  ungroup()%>%
  mutate_if(is.numeric, function(col){(col/.$hours)})%>%
  select(-one_of(c("hours","jp_education_rating")))%>%
  group_by(person_id)%>%
  summarise_all(funs(max))%>%
  replace(., is.na(.), 0.0)%>%
  mutate_if(is.numeric, funs(ifelse(. == 0.0, 4.0, .)))




# jp hours industry
jp_hours_industry_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, industry, hours)%>%
  rename(jp_industry_hours=industry)%>%
  dummy_cols(select_columns = c("jp_industry_hours"))%>% 
  mutate_if(is.integer, function(col){.$hours*col})%>%
  select(-one_of(c("jp_industry_hours")))%>%
  group_by(person_id)%>%
  summarise_all(funs(sum))

# jp rating industry
jp_rating_industry_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, industry, average_rating, hours)%>%
  rename(jp_industry_rating=industry)%>%
  dummy_cols(select_columns = c("jp_industry_rating"))%>% 
  mutate_if(is.integer, function(col){.$average_rating*.$hours*col})%>%
  select(-one_of(c("average_rating")))%>%
  group_by(person_id, jp_industry_rating)%>%
  summarise_all(funs(sum))%>%
  ungroup()%>%
  mutate_if(is.numeric, function(col){(col/.$hours)})%>%
  select(-one_of(c("hours","jp_industry_rating")))%>%
  group_by(person_id)%>%
  summarise_all(funs(max))%>%
  replace(., is.na(.), 0.0)%>%
  mutate_if(is.numeric, funs(ifelse(. == 0.0, 4.0, .)))






# jp hours subindustry
jp_hours_subindustry_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, subindustry, hours)%>%
  rename(jp_subindustry_hours=subindustry)%>%
  dummy_cols(select_columns = c("jp_subindustry_hours"))%>% 
  mutate_if(is.integer, function(col){.$hours*col})%>%
  select(-one_of(c("jp_subindustry_hours","hours")))%>%
  group_by(person_id)%>%
  summarise_all(funs(sum))

# jp rating subindustry
jp_rating_subindustry_per_worker<-skill_data%>%
  filter(job_profile!='')%>%
  select(person_id, subindustry, average_rating, hours)%>%
  rename(jp_subindustry_rating=subindustry)%>%
  dummy_cols(select_columns = c("jp_subindustry_rating"))%>% 
  mutate_if(is.integer, function(col){.$average_rating*.$hours*col})%>%
  select(-one_of(c("average_rating")))%>%
  group_by(person_id, jp_subindustry_rating)%>%
  summarise_all(funs(sum))%>%
  ungroup()%>%
  mutate_if(is.numeric, function(col){(col/.$hours)})%>%
  select(-one_of(c("hours","jp_subindustry_rating")))%>%
  group_by(person_id)%>%
  summarise_all(funs(max))%>%
  replace(., is.na(.), 0.0)%>%
  mutate_if(is.numeric, funs(ifelse(. == 0.0, 4.0, .)))


##### Missing Value Imputation ####


# check variables with NA's
company_nas<-colnames(company_data)[colSums(is.na(company_data)) > 0]
worker_nas<-colnames(worker_data)[colSums(is.na(worker_data)) > 0]
request_nas<-colnames(request_data)[colSums(is.na(request_data)) > 0]
ratings_nas<-colnames(rating_data)[colSums(is.na(rating_data)) > 0]
validation_nas<-colnames(validation_data)[colSums(is.na(validation_data)) > 0]


# impute mean for days_since_ll NA's 
validation_data_nas <- validation_data[is.na(validation_data$days_since_ll),]
a <- summary(validation_data$days_since_ll)
b <- as.data.frame(t(a[4:4]))$Freq
validation_data$days_since_ll[is.na(validation_data$days_since_ll)] <- b
rating_data$days_since_ll[is.na(rating_data$days_since_ll)] <- b
request_data$days_since_ll[is.na(request_data$days_since_ll)] <- b

#duration imputation
validation_data_nas <- validation_data[is.na(validation_data$job_duration),]
a <- summary(validation_data$job_duration)
b <- as.data.frame(t(a[4:4]))$Freq
validation_data$job_duration[is.na(validation_data$job_duration)] <- b
rating_data$job_duration[is.na(rating_data$job_duration)] <- b
request_data$job_duration[is.na(request_data$job_duration)] <- b

rm(a,b,company_nas,worker_nas,request_nas,ratings_nas,validation_nas,validation_data_nas)


##### Join Data Sets #####

rating_model_data <- rating_data%>%
 
  # join features from worker and company
  inner_join(worker_data, by="person_id")%>%
  inner_join(company_data, by="company_id")%>%
  inner_join(rating_granularity_scores, by="wj_id")
table(rating_model_data$worker_rating_orig)
prop.table(table(rating_model_data$worker_rating_orig))

# setwd("/Users/richardchan/Dropbox/FS19/Master Thesis")
# saveRDS(rating_model_data,file="rating_model_data.Rda")



# join request data
# prop.table(table(request_data$accepted))
request_model_data <- request_data%>%
  
  # join worker and company features
  inner_join(worker_data, by="person_id")%>%
  inner_join(company_data, by="company_id")%>%
  inner_join(request_granularity_scores, by="wj_id")

# setwd("/Users/richardchan/Dropbox/FS19/Master Thesis")
# saveRDS(request_model_data,file="request_model_data.Rda")
prop.table(table(request_model_data$accepted))


# join validation data

# join validation data
validation_model_data <- validation_data%>%
  # join worker and company features and granularity features
  inner_join(worker_data, by="person_id")%>%
  inner_join(company_data, by="company_id")%>%
  inner_join(validation_granularity_scores, by="wj_id")

# add best granularity to validation set when found


prop.table(table(validation_model_data$accepted))

prop.table(table(validation_model_data$accepted))



rm(jp_educ_name_per_worker, jp_educ_name_hours_per_worker, 
   jp_educ_name_rating_per_worker, required_jp_educ_name, requested_jp_educ_name,
   jp_edu_level_per_worker, required_jp_edu_level, requested_jp_edu_level,
   jp_names_per_worker, jp_name_rating_per_worker, 
   jp_name_hours_per_worker, required_jp_name, requested_jp_name,
   jp_educationgroup_per_worker, jp_hours_educationgroup_per_worker,
   jp_rating_educationgroup_per_worker, required_jp_educationgroup, 
   jp_education_per_worker, jp_hours_education_per_worker, requested_jp_educationgroup,
   jp_rating_education_per_worker, required_jp_education, requested_jp_education,
   jp_indu_per_worker, jp_hours_indu_per_worker, jp_rating_indu_per_worker, 
   required_jp_industry, requested_jp_industry,jp_subindu_per_worker, 
   jp_hours_subindu_per_worker, jp_rating_subindu_per_worker,
   required_jp_subindustry, requested_jp_subindustry)



##### Plots and Summaries for the Thesis Document #####
# Creates the plots and the data summary shown in the thesis
# Creates the numbers and data facts displayed in the thesis

plotwd <- "/Users/richardchan/Dropbox/FS19/Master Thesis/descriptive_data_analysis"

plotdata <- validation_orig%>%left_join(worker_data, by="person_id")%>%
              left_join(company_data, by="company_id")%>%
              left_join(rating_granularity_scores, by="wj_id")
# clean na's like before
plotdata_nas<-colnames(plotdata)[colSums(is.na(plotdata)) > 0]

plotdata$job_duration[is.na(plotdata$job_duration)] <- as.numeric(23.66)


a <- summary(validation_data$days_since_ll)
b <- as.data.frame(t(a[4:4]))$Freq
validation_data$days_since_ll[is.na(validation_data$days_since_ll)] <- b


#create summary table
names(plotdata)
plotdata_summary <- as.data.frame(summary(plotdata))
plotdata_summarize <- papeR::summarize(plotdata)
plotdata_summarize_t <- as.data.frame(t(plotdata_summarize), row.names=plotdata_summarize[1,])
final_summary <- cbind(colnames(plotdata_summarize),plotdata_summarize_t)%>%select(-c("V1"))
final_summary_f <- final_summary[-c(2,3,4,7),] 
write.csv(final_summary_f, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/data_summary.csv")


# rating plot
ratings <- plotdata%>%
  dplyr::select(worker_rating)%>%
  filter(is.na(worker_rating)==F)%>%
  group_by(worker_rating)%>%
  summarise(count=n())

rating_dist_plot <- ggplot(ratings, aes(x=worker_rating,y=count)) +
  geom_histogram(colour="black", fill="grey", stat="identity")+
  geom_text(aes(label=format(ratings$count, big.mark = "'", scientific = FALSE),vjust=-0.5),size=3.0) +
  labs(x="worker Rating", y="Requests", size=4.5)+
  scale_y_continuous(#labels = scales::comma
    labels=function(x){format(x, big.mark = "'", scientific = FALSE)})+
  theme(panel.background = element_rect(fill = "white",
                                    colour = "white",
                                    size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                    colour = "white"), 
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                    colour = "white"),
    axis.line =  element_line(size = 0.5, linetype = 'solid',
                              colour = "black"))

rating_dist_plot
ggsave("rating_dist_plot.pdf", plot=rating_dist_plot, width=17, height= 9, units="cm", path=plotwd)


# request plot
requests <- plotdata%>%
  dplyr::select(answer)%>%mutate(answer1=ifelse(answer=="WORKER_ACCEPTED","accepted",ifelse(answer=="WORKER_DECLINED","declined","no_answer")))%>%
  group_by(answer1)%>%
  summarise(count=n())
  
request_dist_plot <- ggplot(requests, aes(x=answer1,y=count)) +
  geom_histogram( colour="black", fill="grey", stat="identity")+
  labs(x="Worker Answer", y="Requests", size=3)+
  scale_y_continuous(labels = function(x){format(x, big.mark = "'", scientific = FALSE)})+
  geom_text(aes(label=format(requests$count, big.mark = "'", scientific = FALSE),vjust=-0.5),size=3.0)+
  theme(panel.background = element_rect(fill = "white",
                                        colour = "white",
                                        size = 0.5, linetype = "solid"),
        panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                        colour = "white"), 
        panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                        colour = "white"),
        axis.line =  element_line(size = 0.5, linetype = 'solid',
                                  colour = "black"))

request_dist_plot
ggsave("request_dist_plot.pdf",plot=request_dist_plot, width=17, height= 9, units="cm", path=plotwd)



# some numbers for the text in Section 3.3
total_requested_workers <- validation_orig%>%select(person_id)%>%summarise(worker=n_distinct(person_id))
avg_acceptance_rate_per_worker <- validation_orig%>%select(person_id, accepted)%>%group_by(person_id)%>%summarise(acc_rate=sum(ifelse(accepted=="YES",1,0))/n())
avg_acceptance_rate<-avg_acceptance_rate_per_worker%>%summarise(avg_rate=mean(acc_rate))
answer_rate <- validation_orig%>%select(person_id, answer)%>%mutate(answered=ifelse(answer=="NO_ANSWER",0,1))%>%
  group_by(person_id)%>%summarise(avg_answer_rate=mean(answered))
avg_answer_rate <- answer_rate%>%summarise(avg_rate=mean(avg_answer_rate))


##### Interaction Terms ####

# create some additional features after joining as interaction terms
request_model_data <- request_model_data%>%mutate(wage=as.numeric(salary/job_duration),
                                                  salary_diff = salary-mean_salary,
                                                  duration_diff = job_duration-mean_jobduration,
                                                  #distance_diff = distance_to_job-mean_distance,
                                                  mean_wage_diff = wage-mean_wage,
                                                  min_wage_diff = wage - min_wage)

rating_model_data <- rating_model_data%>%mutate(wage=as.numeric(salary/job_duration),
                                                  salary_diff = salary-mean_salary,
                                                  duration_diff = job_duration-mean_jobduration,
                                                  #distance_diff = distance_to_job-mean_distance,
                                                  mean_wage_diff = wage-mean_wage,
                                                  min_wage_diff = wage - min_wage)

validation_model_data <- validation_model_data%>%mutate(wage=as.numeric(salary/job_duration),
                                                salary_diff = salary-mean_salary,
                                                duration_diff = job_duration-mean_jobduration,
                                                #distance_diff = distance_to_job-mean_distance,
                                                mean_wage_diff = wage-mean_wage,
                                                min_wage_diff = wage - min_wage)

##### Data Normalization #####

# creates normalized data by substracting from the variable mean and deviding it by the variable standard deviation
request_model_data <- as.data.frame(scale(request_model_data))
rating_model_data <- as.data.frame(scale(request_model_data))
validation_model_data <- as.data.frame(scale(validation_model_data))

##### Data Splitting #####
# split data into a training set and a testing set
# check whether the proportion of outcomes was not changed

seed <- 123
set.seed(seed)
# for rating data
ratingtrainIndex <- createDataPartition(rating_model_data$worker_rating,p=.6,list=FALSE)
ratingtrainData <- rating_model_data[ratingtrainIndex,-c(1,2,3,4)]
ratingtestData  <- rating_model_data[-ratingtrainIndex,-c(1,2,3,4)]
# prop.table(table(ratingtrainData$worker_rating))
# prop.table(table(ratingtestData$worker_rating))

# for request data
requesttrainIndex <- createDataPartition(request_model_data$accepted,p=.6,list=FALSE)
requesttrainData <- request_model_data[requesttrainIndex,-c(1,2,3,4)]
requesttestData  <- request_model_data[-requesttrainIndex,-c(1,2,3,4)]
# prop.table(table(requesttrainData$accepted))
# prop.table(table(requesttestData$accepted))


rm(ratingtrainIndex,requesttrainIndex)



##### Feature Selection #####

# this section runs a wrapper selection algorithm called boruta 
# It selects all features considered as important
# the result of this section is a list of features to be used in all sections below
# Particullarly, features considered as unimportant, are not included in section "request Machine learning models" and "rating Machine learning models"

# install.packages("Boruta")
library(Boruta)
trainx <- as.data.frame(train_x_request_base%>%select(-c("job_duration","duration_diff","mean_wage_diff","min_wage_diff")))
typeof(trainx)

trainx <- as.data.frame(colnames(trainx))
boruta.train2 <- Boruta(x=trainx, y=train_y_request$accepted, doTrace=3)
print(boruta.train)

plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)
getSelectedAttributes(final.boruta, withTentative = F)
boruta.df <- attStats(final.boruta)
print(boruta.df)
write.csv(boruta.df, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/boruta_rating_varImp.csv")



#######################################
##### Request Machine learning models #####

# Creates the features to train the model using the BASE MODEL
# Which features to include is defined under "Feature Selection" above
# Skill granularity features are created in the section "Request Skill Granularity Models"

# create request model with selected features separated in
# worker features
columns.worker.request.features<-c(as.character(c("worker_age","gender_Female",
                                                  "receive_newsletter","count_cv","has_picture","photos",
                                                  "is_domestic","description_length","count_certificate",
                                                  "count_diploma","count_testimonial","count_drivinglicens",
                                                  "worker_language_DE","worker_language_FR",
                                                  "average_rating", "comments_count",
                                                  "writing_level_a","writing_level_b","writing_level_c","writing_level_l",
                                                  "speaking_level_a","speaking_level_b","speaking_level_c","speaking_level_l",
                                                  "mean_reactiontime"
                                                  )))

# company features
columns.company.request.features<-c(as.character(c("company_was_referred", "company_size","company_count_employers",
                                                   "company_favorite_workers","company_former_workers",
                                                   "company_days_to_first_shift"
                                                   )))

# and request features
columns.request.features <- c(as.character(c("is_favorite","is_former",
                                             "days_since_ll", "job_duration", "salary","distance_to_job",
                                             "language_skill_required","driving_skill_required_","uniform",
                                             "job_name_length","job_requirements_length","clothing_requrements_length",
                                             "meeting_point_length","additional_skills_required_count",
                                             "use_former_workers","use_public_workers","use_favorite_workers",
                                             "salary_diff", "duration_diff","mean_wage_diff","min_wage_diff"
                                             )))

# Create the training and test datasets with those features and outcomes
worker_features<-paste0(colnames(requesttrainData)[grep(paste(columns.worker.request.features, collapse="|"),colnames(requesttrainData))])
company_features<-paste0(colnames(requesttrainData)[grep(paste(columns.company.request.features,collapse="|"),colnames(requesttrainData))])
request_features<-paste0(colnames(requesttrainData)[grep(paste(columns.request.features,collapse="|"),colnames(requesttrainData))])

train_x_request_base<-as.data.frame(requesttrainData[,c(as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttrainData))),
                                                        as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttrainData))), 
                                                        as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttrainData))))])


train_y_request<-as.data.frame(as.factor(requesttrainData$accepted))%>%rename(accepted="as.factor(requesttrainData$accepted)")


# base model data
test_x_request_base<-as.data.frame(requesttestData[,c(as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttestData))),
                                                      as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttestData))),
                                                      as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttrainData))))])
test_y_request<-as.data.frame(as.factor(requesttestData$accepted))%>%rename(accepted="as.factor(requesttestData$accepted)")


# validation data - base model
# contains workers which have not answered the requests
validationset_x <- as.data.frame(validation_model_data[,c(as.numeric(grep(paste(worker_features,collapse="|"),colnames(validation_model_data))),
                                                        as.numeric(grep(paste(company_features,collapse="|"),colnames(validation_model_data))),
                                                        as.numeric(grep(paste(jp_edu_level_features,collapse="|"),colnames(validation_model_data))),
                                                        as.numeric(grep(paste(request_features,collapse="|"),colnames(validation_model_data))))])


validationset <- validationset_x
validationset$wj_id <- validation_model_data$wj_id
validationset$wa_id <- validation_model_data$wa_id
validationset$accepted <- as.numeric(validation_model_data$accepted)-1
validationset$answer <- validation_model_data$answer

# create result dataframe
final_results<-data.frame()
# or load if existing and only needs results update
file1 <- "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/final_request_results.csv"
final_results <- data.table::fread(file1)
final_results <- final_results%>%select(-c(V1))




##### Request LOG #####

# Request Logit Model


train_control <- trainControl(method="repeatedcv", 
                              number=5, 
                              repeats=5,
                              summaryFunction = twoClassSummary,
                              classProbs=T)
set.seed(123)

test <- cbind(train_x_request_base,train_y_request)
test_sample <- test
train_x_temp <- test_sample%>%select(-c(accepted))
train_y_temp <- test_sample%>%select(accepted)

model.train <- train(train_x_temp, train_y_temp$accepted,
                     trControl=train_control, 
                     preProc = c("center","scale"), # no effect if data was already normalized
                     method="glm",
                     family="binomial",
                     metric = "ROC",
                     tuneLength = 10)
logitrequest.model.train <- model.train # saves the model to environment

# predict test data
pred <- predict(model.train, test_x_request_base, type="prob")

#optimal balanced sensitivity-specificity threshold
fg_bg <- as.data.frame(cbind(pred$YES, as.numeric(test_y_request$accepted)-1))%>%
  rename(pred=V1,accepted=V2)
cp <- cutpointr(fg_bg, pred, accepted, pos_class=1,
                method = maximize_metric, metric = sum_sens_spec)
bal_cp <- cp$optimal_cutpoint

preds<- ifelse(pred$YES >= bal_cp,"YES","NO")

# Confusion Matrix
matrix<- caret::confusionMatrix(table(preds, test_y_request$accepted), positive="YES")
matrix

pred.acc<-matrix$overall['Accuracy']
pred.sens<-matrix$byClass['Sensitivity']
pred.spec<-matrix$byClass['Specificity']
pred.balaccs<-matrix$byClass['Balanced Accuracy']
pred.precision<-matrix$byClass['Pos Pred Value']

# roc curve
pred.rocauc <- roc.curve(predicted=pred$YES, response=test_y_request$accepted)

#precision recall curve
fg <- pred$YES[test_y_request$accepted == "YES"]
bg <- pred$YES[test_y_request$accepted == "NO"]
pred.prauc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)

plot(pred.prauc)


# store results
counter <- 1
final_results[counter,1] <- "LOG"
final_results[counter,2] <- pred.sens
final_results[counter,3] <- pred.spec
final_results[counter,4] <- pred.precision
final_results[counter,5] <- pred.acc
final_results[counter,6] <- pred.balaccs
final_results[counter,7] <- pred.rocauc$auc
final_results[counter,8] <- pred.prauc$auc.integral



# loop for sampling method
sampling_methods<-c("none",
                    "up", 
                    "down",
                    "smote"
                    # "rose"
                    )
counter=1
name="LOG"

# creat train and test set
test <- cbind(train_x_request_base,train_y_request)
test_sample <- test
train_x_temp <- test_sample%>%select(-c(accepted))
train_y_temp <- test_sample%>%select(accepted)

test <- cbind(test_x_request_base,test_y_request)
test_sample <- test
test_x_temp <- test_sample%>%select(-c(accepted))
test_y_temp <- test_sample%>%select(accepted)

for (i in sampling_methods) {
    # control for train
  if(i=="none") {
    train_control <- trainControl(method="repeatedcv", 
                                  number=5, 
                                  repeats=5,
                                  summaryFunction = twoClassSummary,
                                  classProbs=T)
    
    
  } else {
    if(i=="smote") {
      # creates smote'd data
      rosedata<-cbind(train_x_temp, train_y_temp)
      data.rose <- ROSE(accepted ~ ., data = rosedata, seed = 1)$data
      #prop.table(table(data.rose$accepted))
      #table(data.rose$accepted)
      train_x_temp <- data.rose%>%select(-c("accepted"))
      train_y_temp <- data.rose%>%select(accepted)
      train_control <- trainControl(method="repeatedcv", 
                                    number=5, 
                                    repeats=5,
                                    summaryFunction = twoClassSummary,
                                    classProbs=T)
      
    } else {
      # creates under or oversampling trainControl
      train_control <- trainControl(method="repeatedcv", 
                                  number=5,
                                  repeats=5, 
                                  summaryFunction = twoClassSummary,
                                  classProbs=T,
                                  savePredictions = TRUE,
                                  sampling=i # samples through over- and undersampling methods
    )
    }
    }  
  
  # trains the model
  set.seed(123)
  model.train <- train(train_x_temp, train_y_temp$accepted,
                       trControl=train_control, 
                       preProc = c("center","scale"),
                       method="glm",
                       family="binomial",
                       metric="ROC",
                       tuneLength = 10)
  
  # predict test set
  pred <- predict(model.train, test_x_temp, type="prob")
  fg_bg <- as.data.frame(cbind(pred$YES, as.numeric(test_y_temp$accepted)-1))%>%rename(pred=V1,accepted=V2)
  #optimal balanced sensitivity specificity threshold
  cp <- cutpointr(fg_bg, pred, accepted, pos_class=1,
                  method = maximize_metric, metric = sum_sens_spec)
  bal_cp <- cp$optimal_cutpoint
  
  preds<- ifelse(pred$YES >= bal_cp,"YES","NO")  
  
  # Confusion matrix
  matrix<-confusionMatrix(table(preds, test_y_temp$accepted), positive = "YES")
  pred.acc<-matrix$overall['Accuracy']
  pred.sens<-matrix$byClass['Sensitivity']
  pred.spec<-matrix$byClass['Specificity']
  pred.balaccs<-matrix$byClass['Balanced Accuracy']
  pred.precision<-matrix$byClass['Pos Pred Value']
  
  # roc curve
  pred.rocauc <- roc.curve(predicted=pred$YES, response=test_y_temp$accepted)
  # precision recall curve
  fg <- pred$YES[test_y_temp$accepted == "YES"]
  bg <- pred$YES[test_y_temp$accepted == "NO"]
  pred.prauc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  

  if(counter==1){request_results<-data.frame()} else {request_results <- request_results}
  
  # Store results
  request_results[counter,1]<-as.character(paste(name,i,sep="_"))
  
  request_results[counter,2] <- pred.sens
  request_results[counter,3] <- pred.spec
  request_results[counter,4] <- pred.precision
  request_results[counter,5] <- pred.acc
  request_results[counter,6] <- pred.balaccs
  request_results[counter,7] <- pred.rocauc$auc
  request_results[counter,8] <- pred.prauc$auc.integral
  
  
  if(counter==1){request_results <- request_results%>%
    rename(model=V1, TPrate=V2, TNrate=V3, PosPval=V4, acc=V5, balacc=V6,
           rocauc=V7, prauc=V8
           )}
  counter <- counter + 1
  print(counter)
  }
log_request_sampling_results<-request_results%>%rename(Model=model, "TP rate"=TPrate, "TN rate" = TNrate
                                                       ,Precision = PosPval, "Accuracy" = acc, "Bal. Acc." = balacc, ROCAUC=rocauc, PRAUC=prauc)
log_request_sampling_results[1,1]<-"LOG none"
log_request_sampling_results[2,1]<-"LOG over"
log_request_sampling_results[3,1]<-"LOG under"
log_request_sampling_results[4,1]<-"LOG smote"
log_request_sampling_results

write.csv(log_request_sampling_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/request_logit_resampling.csv")




##### Request KNN #####

# KNN request model with SMOTE rebalancing

rosedata<-cbind(test_x_temp, test_y_temp)
data.rose <- ROSE(accepted ~ ., data = rosedata, seed = 1)$data
#prop.table(table(data.rose$accepted))
#table(data.rose$accepted)

train_x_request_base_rose <- data.rose%>%select(-c("accepted"))
train_y_request_rose <- data.rose%>%select(accepted)
train_control <- trainControl(method="repeatedcv", 
                              number=10, 
                              repeats=10,
                              summaryFunction = twoClassSummary,
                              classProbs=T)
test_y_temp <- train_y_request_rose
test_x_temp <- train_x_request_base_rose

# train model
model.train <- train(
  train_x_temp
  ,train_y_temp$accepted
  ,trControl=train_control
  ,method="knn"
  ,preProc = c("center","scale")
  ,metric = "ROC"
  #,tuneLength = 10
  ,tuneGrid = expand.grid(k = c(10:15))
  )
model.train
# k=60, ROC = 0.7135348
# k=15, Roc=0.6962865
# k=20, ROC = 0.6909259
# K15, CV=10 & 10, roc =0.7175828
# K10, CV 10 & 10, roc = 0.7533202

test <- cbind(test_x_request_base,test_y_request)
test_sample <- test
test_x_temp <- test_sample%>%select(-c(accepted))
test_y_temp <- test_sample%>%select(accepted)

# predict test data
pred <- predict(model.train, test_x_temp, type="prob")
fg_bg <- as.data.frame(cbind(pred$YES, as.numeric(test_y_temp$accepted)-1))%>%rename(pred=V1,accepted=V2)

#optimal balanced sensitivity specificity threshold
cp <- cutpointr(fg_bg, pred, accepted, pos_class=1,
                method = maximize_metric, metric = sum_sens_spec)
bal_cp <- cp$optimal_cutpoint

preds <- ifelse(pred$YES > bal_cp,"YES","NO")  

matrix <- caret::confusionMatrix(table(preds, test_y_temp$accepted), positive="YES")
matrix

pred.acc<-matrix$overall['Accuracy']
pred.sens<-matrix$byClass['Sensitivity']
pred.spec<-matrix$byClass['Specificity']
pred.balaccs<-matrix$byClass['Balanced Accuracy']
pred.precision<-matrix$byClass['Pos Pred Value']

pred.rocauc <- roc.curve(predicted=pred$YES, response=test_y_temp$accepted)
#precision recall curve
fg <- pred$YES[test_y_temp$accepted == "YES"]
bg <- pred$YES[test_y_temp$accepted == "NO"]
pred.prauc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)

# store results
counter <- 2
final_results[counter,1] <- "KNN"
final_results[counter,2] <- pred.sens
final_results[counter,3] <- pred.spec
final_results[counter,4] <- pred.precision
final_results[counter,5] <- pred.acc
final_results[counter,6] <- pred.balaccs
final_results[counter,7] <- pred.rocauc$auc
final_results[counter,8] <- pred.prauc$auc.integral




# loop for sampling method
sampling_methods<-c("none",
                    "up", 
                    "down"
                    ,"smote"
                    # "rose"
)
counter=1
name="KNN"
test <- cbind(train_x_request_base,train_y_request)
test_sample <- test
train_x_temp <- test_sample%>%select(-c(accepted))
train_y_temp <- test_sample%>%select(accepted)

train_data <- train_y_temp
train_data_x <- train_x_temp

test <- cbind(test_x_request_base,test_y_request)
test_sample <- test
test_x_temp <- test_sample%>%select(-c(accepted))
test_y_temp <- test_sample%>%select(accepted)

for (i in sampling_methods) {

  if(i=="none") {
    train_control <- trainControl(method="repeatedcv", 
                                  number=10, 
                                  repeats=10,
                                  summaryFunction = twoClassSummary,
                                  classProbs=TRUE)
    
    
  } else {
    
    if(i=="smote") {
      rosedata<-cbind(test_x_temp, test_y_temp)
      data.rose <- ROSE(accepted ~ ., data = rosedata, seed = 1)$data
      #prop.table(table(data.rose$accepted))
      #table(data.rose$accepted)
      train_x_request_base_rose <- data.rose%>%select(-c("accepted"))
      train_y_request_rose <- data.rose%>%select(accepted)
      train_control <- trainControl(method="repeatedcv", 
                                    number=10, 
                                    repeats=10,
                                    summaryFunction = twoClassSummary,
                                    classProbs=T)
      test_y_temp <- train_y_request_rose
      test_x_temp <- train_x_request_base_rose
      
    } else {
      train_control <- trainControl(method="repeatedcv", 
                                    number=10,
                                    repeats=10, 
                                    summaryFunction = twoClassSummary,
                                    classProbs=T,
                                    #savePredictions = TRUE,
                                    sampling=i # samples through all sampling methods
      )
    }
  }
  
  
   set.seed(123)
   model.train <- train(x=test_x_temp, y=test_y_temp$accepted, 
                        trControl=train_control, 
                        method="knn",
                        preProc = c("center","scale"),
                        metric = "ROC",
                        tuneGrid = expand.grid(k = c(10:15)))
  
  # create results
  pred <- predict(model.train, test_x_temp, type="prob")
  fg_bg <- as.data.frame(cbind(pred$YES, as.numeric(test_y_temp$accepted)-1))%>%rename(pred=V1,accepted=V2)
  #optimal balanced sens spec threshold
  cp <- cutpointr(fg_bg, pred, accepted, pos_class=1,
                 method = maximize_metric, metric = sum_sens_spec)
  bal_cp <- cp$optimal_cutpoint
  
  preds<- ifelse(pred$YES > bal_cp,"YES","NO")  
  matrix<- caret::confusionMatrix(table(preds, test_y_temp$accepted), positive="YES")
  pred.acc<-matrix$overall['Accuracy']
  pred.sens<-matrix$byClass['Sensitivity']
  pred.spec<-matrix$byClass['Specificity']
  pred.balaccs<-matrix$byClass['Specificity']
  pred.precision<-matrix$byClass['Pos Pred Value']
  
  pred.rocauc <- roc.curve(predicted=pred$YES, response=test_y_temp$accepted)
  #precision recall curve
  fg <- pred$YES[test_y_temp$accepted == "YES"]
  bg <- pred$YES[test_y_temp$accepted == "NO"]
  pred.prauc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  
  
  
  if(counter==1){request_results<-data.frame()} else {request_results <- request_results}
  # store results
  request_results[counter,1]<-as.character(paste(name,i,sep="_"))
  
  request_results[counter,2] <- pred.sens
  request_results[counter,3] <- pred.spec
  request_results[counter,4] <- pred.precision
  request_results[counter,5] <- pred.acc
  request_results[counter,6] <- pred.balaccs
  request_results[counter,7] <- pred.rocauc$auc
  request_results[counter,8] <- pred.prauc$auc.integral
  
  

  counter <- counter + 1
  print(counter)
}
knn_request_results <- request_results%>%
  rename(Model=V1, "TP rate"=V2, "TN rate"=V3, Precision=V4, Accuracy=V5, "Bal. Acc."=V6,
         ROCAUC=V7, PRAUC=V8
  )
knn_request_results[1,1] <- "KNN no rebal."
knn_request_results[2,1] <- "KNN over"
knn_request_results[3,1] <- "KNN under"
knn_request_results[4,1] <- "KNN smote"
knn_request_results
# save results
write.csv(knn_request_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/request_knn_resampling.csv")




##### Request SVM #######

# simple svm request model
train_control <- trainControl(method="repeatedcv", 
                              number=10, 
                              repeats=10,
                              summaryFunction = twoClassSummary,
                              classProbs=TRUE
                              )
set.seed(123)
test <- cbind(train_x_request_base,train_y_request)
test_sample <- test
train_x_temp <- test_sample%>%select(-c(accepted))
train_y_temp <- test_sample%>%select(accepted)


# Final choice
grid <- expand.grid(sigma= 2^c(-7), C= c(0.1)) 
# grid<-expand.grid(sigma = c(.01, 0.5),C = c(0.75, 1,5))
# grid <- expand.grid(sigma= 2^c(-15,-7, -3), C= c(0.01, 0.05, 0.1,0.15))
# sigma = 0.01385817 and C = 0.25., ROC=0.503
# sigma = 0.01385817 and C = 0.25., ROC=0.65
#sigma = 0.0078125 (2^-7) and C = 0.1., ROC=0.7770955/.7709/
# sigma = 0.015625 and C = 0.09., ROC=0.7698018
# sigma = 0.0078125 and C = 0.15., ROC=0.7912834

model.train<-train( 
  train_x_temp, train_y_temp$accepted, 
                   trControl=train_control,
                   tuneGrid=grid,
                   preProcess=c("center","scale"),
                   metric="ROC",
                   method="svmRadial") 
svmrequest.model.train <- model.train
svmrequest.model.train
# linear svm: svmLinearWeights (e1071), lssvmLinear (kernlab)
# polinomial svm: lssvmPoly (kernlab), svmPoly (kernlab)
# raidal svm: svmRadial (kernlab)

test <- cbind(test_x_request_base,test_y_request)
test_sample <- test
test_x_temp <- test_sample%>%select(-c(accepted))
test_y_temp <- test_sample%>%select(accepted)

# predict test set
pred <- predict(svmrequest.model.train, test_x_temp, type="prob")

#optimal balanced sens spec threshold
fg_bg <- as.data.frame(cbind(pred$YES, as.numeric(test_y_temp$accepted)-1))%>%rename(pred=V1,accepted=V2)
cp <- cutpointr(fg_bg, pred, accepted, pos_class=1,
                method = maximize_metric, metric = sum_sens_spec)
bal_cp <- cp$optimal_cutpoint

preds<- ifelse(pred$YES > bal_cp,"YES","NO")  

matrix<- caret::confusionMatrix(table(preds, test_y_temp$accepted), positive="YES")
matrix

pred.acc<-matrix$overall['Accuracy']
pred.sens<-matrix$byClass['Sensitivity']
pred.spec<-matrix$byClass['Specificity']
pred.balaccs<-matrix$byClass['Balanced Accuracy']
pred.precision<-matrix$byClass['Pos Pred Value']

# roc curve
pred.rocauc <- roc.curve(predicted=pred$YES, response=test_y_temp$accepted)

#precision recall curve
fg <- pred$YES[test_y_temp$accepted == "YES"]
bg <- pred$YES[test_y_temp$accepted == "NO"]
pred.rocauc <- PRROC::roc.curve(scores.class0=fg, scores.class1=bg, curve = TRUE)

pred.prauc <- PRROC::pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
svm_roc <- pred.rocauc
svm_pr <- pred.prauc


# store results
counter <- 3
final_results[counter,1] <- "SVM"
final_results[counter,2] <- pred.sens
final_results[counter,3] <- pred.spec
final_results[counter,4] <- pred.precision
final_results[counter,5] <- pred.acc
final_results[counter,6] <- pred.balaccs
final_results[counter,7] <- pred.rocauc$auc
final_results[counter,8] <- pred.prauc$auc.integral



# loop for sampling method
sampling_methods<-c("none",
                    "up", 
                    "down"
                    ,"smote"
                    # "rose"
)

counter=1
name="SVM"
set.seed(123)
test <- cbind(train_x_request_base,train_y_request)
test_sample <- test
train_x_temp <- test_sample%>%select(-c(accepted))
train_y_temp <- test_sample%>%select(accepted)


test <- cbind(test_x_request_base,test_y_request)
test_sample <- test
test_x_temp <- test_sample%>%select(-c(accepted))
test_y_temp <- test_sample%>%select(accepted)


for (i in sampling_methods) {
  
  # control for train
  if(i=="none") {
    train_control <- trainControl(method="repeatedcv", 
                                  number=10, 
                                  repeats=10,
                                  classProbs=TRUE)
    
    
  } else {
    
    if(i=="smote") {
      rosedata<-cbind(train_x_temp, train_y_temp)
      data.rose <- ROSE(accepted ~ ., data = rosedata, seed = 1)$data
      #prop.table(table(data.rose$accepted))
      #table(data.rose$accepted)
      train_x_request_base_rose <- data.rose%>%select(-c("accepted"))
      train_y_request_rose <- data.rose%>%select(accepted)
      train_control <- trainControl(method="repeatedcv", 
                                    number=10, 
                                    repeats=10,
                                    summaryFunction = twoClassSummary,
                                    classProbs=T)
      train_y_temp <- train_y_request_rose
      train_x_temp <- train_x_request_base_rose
      
    } else {
      train_control <- trainControl(method="repeatedcv", 
                                    number=10,
                                    repeats=10, 
                                    summaryFunction = twoClassSummary,
                                    classProbs=T,
                                    savePredictions = TRUE,
                                    sampling=i # samples through all sampling methods
      )
    }
  }
  
  # manually adjust grid search parameters
  #grid <- expand.grid(sigma= 2^c(-15,-7, -3), C= c(0.01, 0.05, 0.1,0.15))
  grid <- expand.grid(sigma= 2^c(-7), C= c(0.1))
  model.train<-train(train_x_temp, train_y_temp$accepted, 
    trControl=train_control,
    tuneGrid=grid,
    preProcess=c("center","scale"),
    metric="ROC",
    method="svmRadial")


  # predict test set
  pred <- predict(model.train, test_x_temp, type="prob")
  
  #optimal balanced sens spec threshold
  fg_bg <- as.data.frame(cbind(pred$YES, as.numeric(test_y_temp$accepted)-1))%>%rename(pred=V1,accepted=V2)
  cp <- cutpointr(fg_bg, pred, accepted, pos_class=1,
                  method = maximize_metric, metric = sum_sens_spec)
  bal_cp <- cp$optimal_cutpoint
  
  preds<- ifelse(pred$YES > bal_cp,"YES","NO")  
  matrix<-confusionMatrix(table(preds, test_y_temp$accepted), positive="YES")
  
  pred.acc<-matrix$overall['Accuracy']
  pred.sens<-matrix$byClass['Sensitivity']
  pred.spec<-matrix$byClass['Specificity']
  pred.balaccs<-matrix$byClass['Balanced Accuracy']
  pred.precision<-matrix$byClass['Pos Pred Value']
  
  # roc and pr curve
  pred.rocauc <- roc.curve(predicted=pred$YES, response=test_y_temp$accepted)
  fg <- pred$YES[test_y_temp$accepted == "YES"]
  bg <- pred$YES[test_y_temp$accepted == "NO"]
  pred.prauc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)  
  
  
  if(counter==1){request_results<-data.frame()} else {request_results <- request_results}
  
  request_results[counter,1]<-as.character(paste(name,i,sep="_"))
  
  request_results[counter,2] <- pred.sens
  request_results[counter,3] <- pred.spec
  request_results[counter,4] <- pred.precision
  request_results[counter,5] <- pred.acc
  request_results[counter,6] <- pred.balaccs
  request_results[counter,7] <- pred.rocauc$auc
  request_results[counter,8] <- pred.prauc$auc.integral

  counter <- counter + 1
  print(counter)
}

svm_request_sampling_results<-request_results%>%rename(Model=V1, "TP rate"=V2, "TN rate"=V3, "Precision"=V4, "Accuracy"=V5, "Bal. Acc."=V6,
                                                       ROCAUC=V7, PRAUC=V8
)
svm_request_sampling_results[1,1] <- "SVM no rebal."
svm_request_sampling_results[2,1] <- "SVM over"
svm_request_sampling_results[3,1] <- "SVM under"
svm_request_sampling_results[4,1] <- "SVM smote"
svm_request_sampling_results
# save results
write.csv(svm_request_sampling_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/request_svm_resampling.csv")




##### Request RF  ###################

# simple random forest model
train_control <- trainControl(method="repeatedcv", 
                              number=10, 
                              repeats=5,
                              summaryFunction = twoClassSummary,
                              classProbs=TRUE
)
set.seed(123)
test <- cbind(train_x_request_base,train_y_request)
test_sample <- test
train_x_temp <- test_sample%>%select(-c(accepted))
train_y_temp <- test_sample%>%select(accepted)

#select tunegrid manually
tunegrid <- expand.grid(.mtry=c(5:8))
#tunegrid <- expand.grid(.mtry=c(1:20))


model.train <- train(
  train_x_temp, train_y_temp$accepted, 
                     trControl=train_control, 
                     method="rf",
                     metric="ROC",
                     preProcess=c("center","scale")
                     ,ntrees=10
                     ,tuneGrid = tunegrid
  )

rfrequest.model.train <- model.train
rfrequest.model.train
# ntrees 35, mtry=20: ROC=0.947134
# ntrees 20, mtry=20: ROC=0.951782
# ntrees 10, mtry=21: ROC=0.947656
# ntrees 15, mtry=15: ROC=0.948672
# ntrees 20, mtry=15: ROC=0.948672
# ntrees 20, mtry=11: ROC=0.953928
# ntrees 15, mtry=8: ROC=0.954808
# ntrees 15, mtry=7: ROC=0.953972
# ntrees 10, mtry=5: ROC=0.954940 => best model so far
# ntrees 5, mtry=6: ROC=0.951628
# => test models mtry between 5 and 8, holding ntrees at 10, with repeated CV


test <- cbind(test_x_request_base,test_y_request)
test_sample <- test
test_x_temp <- test_sample%>%select(-c(accepted))
test_y_temp <- test_sample%>%select(accepted)

# predict test set
pred <- predict(rfrequest.model.train, test_x_temp, type="prob")

#optimal balanced sens spec threshold
fg_bg <- as.data.frame(cbind(pred$YES, as.numeric(test_y_temp$accepted)-1))%>%rename(pred=V1,accepted=V2)
cp <- cutpointr(fg_bg, pred, accepted, pos_class=1,
                method = maximize_metric, metric = sum_sens_spec)
bal_cp <- cp$optimal_cutpoint

preds<- ifelse(pred$YES > bal_cp,"YES","NO")  

matrix<- caret::confusionMatrix(table(preds, test_y_temp$accepted), positive="YES")
matrix

pred.acc<-matrix$overall['Accuracy']
pred.sens<-matrix$byClass['Sensitivity']
pred.spec<-matrix$byClass['Specificity']
pred.balaccs<-matrix$byClass['Balanced Accuracy']
pred.precision<-matrix$byClass['Pos Pred Value']

pred.rocauc <- roc.curve(predicted=pred$YES, response=test_y_temp$accepted)
#plot(pred.rocauc)

#precision recall curve
fg <- pred$YES[test_y_temp$accepted == "YES"]
bg <- pred$YES[test_y_temp$accepted == "NO"]
pred.prauc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
pred.rocauc <- PRROC::roc.curve(scores.class0=fg, scores.class1=bg, curve = TRUE)
plot(pred.rocauc)
rf_roc <- pred.rocauc
rf_pr <- pred.prauc

plot(rf_roc, color="black", label="rf")

# store results
counter <- 4
final_results[counter,1] <- "RF"
final_results[counter,2] <- pred.sens
final_results[counter,3] <- pred.spec
final_results[counter,4] <- pred.precision
final_results[counter,5] <- pred.acc
final_results[counter,6] <- pred.balaccs
final_results[counter,7] <- pred.rocauc$auc
final_results[counter,8] <- pred.prauc$auc.integral




# loop through different sampling methods
sampling_methods<-c("none",
                    "up", 
                    "down"
                    ,"smote"
                    # "rose"
)

counter=1
name="RF"

test <- cbind(train_x_request_base,train_y_request)
test_sample <- test
train_x_temp <- test_sample%>%select(-c(accepted))
train_y_temp <- test_sample%>%select(accepted)

train_data <- train_y_temp
train_data_x <- train_x_temp

test <- cbind(test_x_request_base,test_y_request)
test_sample <- test
test_x_temp <- test_sample%>%select(-c(accepted))
test_y_temp <- test_sample%>%select(accepted)

# loop
for (i in sampling_methods) {
    # control for train
  if(i=="none") {
    train_control <- trainControl(method="repeatedcv", 
                                  number=10, 
                                  repeats=5,
                                  summaryFunction = twoClassSummary,
                                  classProbs=TRUE)
    
    
  } else {
    
    if(i=="smote") {
      # not sure whether this works
      rosedata<-cbind(train_x_temp, train_y_temp)
      data.rose <- ROSE(accepted ~ ., data = rosedata,
                        seed = 123)$data
      #prop.table(table(data.rose$accepted))
      #table(data.rose$accepted)
      train_x_request_base_rose <- data.rose%>%select(-c("accepted"))
      train_y_request_rose <- data.rose%>%select(accepted)
      train_y_temp <- train_y_request_rose
      train_x_temp <- train_x_request_base_rose
      
      train_control <- trainControl(method="repeatedcv", 
                                    number=10, 
                                    repeats=5,
                                    summaryFunction = twoClassSummary,
                                    classProbs=T)
      
    } else {
      train_control <- trainControl(method="repeatedcv", 
                                    number=10,
                                    repeats=5, 
                                    summaryFunction = twoClassSummary,
                                    classProbs=T,
                                    # savePredictions = TRUE,
                                    sampling=i # samples through all sampling methods
      )
    }
  }
  tunegrid <- expand.grid(.mtry=c(5:8))
  
  model.train <- train(
    train_x_temp, train_y_temp$accepted, 
    trControl=train_control, 
    method="rf",
    metric="ROC",
    ntrees=10,
    tuneGrid=tunegrid
    #,tuneLength = 5
    )
  
  
  # create results
  pred <- predict(model.train, test_x_temp, type="prob")
  fg_bg <- as.data.frame(cbind(pred$YES, as.numeric(test_y_temp$accepted)-1))%>%rename(pred=V1,accepted=V2)
  #optimal balanced sens spec threshold
  cp <- cutpointr(fg_bg, pred, accepted, pos_class="1",
                  method = maximize_metric, metric = sum_sens_spec)
  bal_cp <- cp$optimal_cutpoint
  
  preds<- ifelse(pred$YES > bal_cp,"YES","NO")  
  # preds <- ifelse (pred$YES > 0.5,"YES","NO")
  matrix <- caret::confusionMatrix(table(preds, test_y_temp$accepted), positive="YES")
  
  pred.acc <- matrix$overall['Accuracy']
  pred.sens <- matrix$byClass['Sensitivity']
  pred.spec <- matrix$byClass['Specificity']
  pred.balaccs <- matrix$byClass['Balanced Accuracy']
  pred.precision <- matrix$byClass['Pos Pred Value']
  
  pred.rocauc <- roc.curve(predicted=pred$YES, response=test_y_temp$accepted)
  #precision recall curve
  fg <- pred$YES[test_y_temp$accepted == "YES"]
  bg <- pred$YES[test_y_temp$accepted == "NO"]
  pred.prauc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)  
  
  
  if(counter==1){request_results <- data.frame()} else {request_results <- request_results}
  
  request_results[counter,1] <- as.character(paste(name,i,sep="_"))
  
  request_results[counter,2] <- pred.sens
  request_results[counter,3] <- pred.spec
  request_results[counter,4] <- pred.precision
  request_results[counter,5] <- pred.acc
  request_results[counter,6] <- pred.balaccs
  request_results[counter,7] <- pred.rocauc$auc
  request_results[counter,8] <- pred.prauc$auc.integral
  
  

  counter <- counter + 1
  print(counter)
}
rfrequest.model.train <- model.train
rf_request_sampling_results<-request_results%>%rename(Model=V1, "TP rate"=V2, "TN rate"=V3, "Precision"=V4, "Accuracy"=V5, "Bal. Acc."=V6,
                                                       ROCAUC=V7, PRAUC=V8
)
# Rename
rf_request_sampling_results[1,1] <- "RF no rebal."
rf_request_sampling_results[2,1] <- "RF over"
rf_request_sampling_results[3,1] <- "RF under"
rf_request_sampling_results[4,1] <- "RF smote"

rf_request_sampling_results
write.csv(rf_request_sampling_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/request_rf_resampling.csv")



# check on which variables are important selected by RF
# plot variable importance
rf.importance <- varImp(rf.under.acc, scale=T)$importance%>% 
  as.data.frame() %>%
  rownames_to_column() %>%
  arrange(Overall) %>% select(rowname, Overall)
plot(varImp(rf.under.acc, scale=T), top=35)
varImp(rf.raw, scale=F)$importance['is_former',]


# extract feature importance AUC for skill groups
# from the most predictive model
rf.jp.imp<-rf.importance%>%
  filter(grepl('jp_',rowname))

rf.jp_hours.imp<-rf.importance%>%
  filter(grepl('jp_hours',rowname))

rf.jp_hours_industry.imp<-rf.importance%>%
  filter(grepl('jp_hours_industry',rowname))

rf.jp_hours_subindustry.imp<-rf.importance%>%
  filter(grepl('jp_hours_subindustry',rowname))

rf.jp_rating.imp<-rf.importance%>%
  filter(grepl('jp_rating',rowname))

rf.jp_rating_industry.imp<-rf.importance%>%
  filter(grepl('jp_rating_industry',rowname))

rf.jp_rating_subindustry.imp<-rf.importance%>%
  filter(grepl('jp_rating_subindustry',rowname))


rf.label.imp<-rf.importance%>%
  filter(grepl('label_',rowname))



boxplot(rf.jp.imp$Overall,rf.jp_hours.imp$Overall,
        rf.jp_hours_industry.imp$Overall,rf.jp_hours_subindustry.imp$Overall,
        rf.jp_rating.imp$Overall,rf.jp_rating_industry.imp$Overall,rf.jp_rating_subindustry.imp$Overall,
        rf.label.imp$Overall)




##### Request FFNNET #############

# data into format for FFNNET
set.seed(123)
test <- cbind(train_x_request_base,train_y_request)
test_sample <- test
train_x_temp <- test_sample%>%select(-c(accepted))
train_y_temp <- test_sample%>%mutate(accepted0=ifelse(accepted=='YES',1,0))%>%select(accepted0)

train_x <- as.matrix(data.frame(train_x_temp))
train_y<-as.matrix(data.frame(train_y_temp$accepted0))

# Build layers and nodes/units
model <- keras_model_sequential() %>%
  
  # network architecture
  layer_dense(units = 50, activation = "relu", input_shape = ncol(train_x_temp),
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 30, activation = "relu",
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 20, activation = "relu",
              kernel_regularizer = regularizer_l1(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 10, activation = "relu",
              kernel_regularizer = regularizer_l1(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 1, activation=sigmoid) %>%
  
  # backpropagation
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = c('AUC')
  )

# train model
model.train <- model %>% fit(
  x=train_x,
  y=train_y,
  epochs = 50,
  batch_size = 32,
  validation_split = .3,
  verbose = FALSE,
  callbacks = list(
    callback_early_stopping(patience = 10),
    callback_reduce_lr_on_plateau()
  )
)

ffnetrequest.model.train <- model.train
plot(ffnetrequest.model.train)

# test set results
test <- cbind(test_x_request_base,test_y_request)
test_sample <- test
test_x_temp <- test_sample%>%select(-c(accepted))
test_y_temp <- test_sample%>%select(accepted)

# predict test set
predictions <- as.data.frame(model %>% predict_proba(as.matrix(test_x_temp)))%>%rename(YES=V1)

#optimal balanced sens spec threshold
fg_bg <- as.data.frame(cbind(predictions$YES, as.numeric(test_y_temp$accepted)-1))%>%rename(pred=V1,accepted=V2)
cp <- cutpointr(fg_bg, predictions, accepted, pos_class=1,
                method = maximize_metric, metric = sum_sens_spec)
bal_cp <- cp$optimal_cutpoint

preds<- ifelse(predictions$YES > bal_cp,"YES","NO")  

matrix<- caret::confusionMatrix(table(preds, test_y_temp$accepted), positive="YES")
matrix

pred.acc<-matrix$overall['Accuracy']
pred.sens<-matrix$byClass['Sensitivity']
pred.spec<-matrix$byClass['Specificity']
pred.balaccs<-matrix$byClass['Balanced Accuracy']
pred.precision<-matrix$byClass['Pos Pred Value']

pred.rocauc <- roc.curve(predicted=predictions$YES, response=test_y_temp$accepted, curve=T)
#precision recall curve & Roc curve
fg <- predictions$YES[test_y_temp$accepted == "YES"]
bg <- predictions$YES[test_y_temp$accepted == "NO"]
pred.prauc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
pred.rocauc <- PRROC::roc.curve(scores.class0=fg, scores.class1=bg, curve=T)

ffnnet_roc <- pred.rocauc
ffnnet_pr <- pred.prauc
proc_curve_ffnnet <- pROC::roc(test_y_request$accepted ~predictions$YES)

plot(pred.rocauc)
plot(pred.prauc)
# store results
counter <- 5
final_results[counter,1] <- "FFNNET"
final_results[counter,2] <- pred.sens
final_results[counter,3] <- pred.spec
final_results[counter,4] <- pred.precision
final_results[counter,5] <- pred.acc
final_results[counter,6] <- pred.balaccs
final_results[counter,7] <- pred.rocauc$auc
final_results[counter,8] <- pred.prauc$auc.integral





# loop through sampling methods
sampling_methods<-c("none",
                    "up", 
                    "down"
                    ,"smote"
                    # "rose"
)

counter=1
name="ffnnet"
test <- cbind(train_x_request_base,train_y_request)
test_sample <- test
train_x_temp <- test_sample%>%select(-c(accepted))
train_y_temp <- test_sample%>%select(accepted)

train_data <- train_y_temp%>%mutate(accepted0=ifelse(accepted=='YES',1,0))%>%select(accepted0)
train_data_x <- train_x_temp

test <- cbind(test_x_request_base,test_y_request)
test_sample <- test
test_x_temp <- test_sample%>%select(-c(accepted))
test_y_temp <- test_sample%>%select(accepted)

for (i in sampling_methods) {
  
  # manual resampling for ffnet 
  if(i=="none") {
    train_y <- as.matrix(train_data)
    train_x <- as.matrix(train_data_x)
    
    
  } else {
    
    if(i=="smote") {
      # not sure whether this works
      temp<-cbind(train_data_x, train_data)
      data.rose <- ROSE(accepted0 ~ ., data = temp, N = nrow(temp), seed = 1)$data
      train_x_request_base_rose <- data.rose%>%select(-c("accepted0"))
      train_y_request_rose <- data.rose%>%select(accepted0)
      train_y <- as.matrix(train_y_request_rose)
      train_x <- as.matrix(train_x_request_base_rose)
      rm(temp, train_x_request_base_rose,train_y_request_rose)
    } 
    if(i=="up") {
      # oversampling
      temp <- cbind(train_data_x, train_data)
      data.over <- ovun.sample(accepted0 ~ .
                               , data = temp
                               , method = "over"
                               , N=nrow(temp)
                               )$data
      train_x <- as.matrix(data.over%>%dplyr::select(-c("accepted0")))
      train_y <- as.matrix(train_data)
      rm(temp)
    } 
    if(i=="down") {
      # undersampling
      temp <- cbind(train_data_x, train_data)
      data.under <- ovun.sample(accepted0 ~ .
                                , data = temp
                                , method = "under"
                                , N=nrow(temp)
                                )$data
      train_x <- as.matrix(data.under%>%dplyr::select(-c("accepted0")))
      train_y <- as.matrix(train_data)
      rm(temp)
      
    }
  }
  
  # build models
  model <- keras_model_sequential() %>%
    
    # network architecture
    layer_dense(units = 50, activation = "relu", input_shape = ncol(train_x_temp),
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 30, activation = "relu",
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 20, activation = "relu",
                kernel_regularizer = regularizer_l1(0.001)) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 10, activation = "relu",
                kernel_regularizer = regularizer_l1(0.001)) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 1, activation=sigmoid) %>%
    
    # backpropagation
    compile(
      loss = 'binary_crossentropy',
      optimizer = 'adam',
      metrics = c('AUC')
    )
  
  model.train <- model %>% fit(
    x=train_x,
    y=train_y,
    epochs = 50,
    batch_size = 32,
    validation_split = .3,
    verbose = FALSE,
    callbacks = list(
      callback_early_stopping(patience = 10),
      callback_reduce_lr_on_plateau()
    )
  )
  
  
  # predict test data
  pred <- as.data.frame(model %>% predict_proba(as.matrix(test_x_temp)))%>%rename(YES=V1)
  # roc and pr curve
  fg_bg <- as.data.frame(cbind(pred$YES , test_y_temp$accepted))
  fg_bg<-fg_bg%>%rename(pred=V1,accepted=V2)
  cp <- cutpointr(fg_bg, pred, accepted, pos_class=1,
                  method = maximize_metric, metric = sum_sens_spec)
  bal_cp <- cp$optimal_cutpoint
  
  preds<- ifelse(pred$YES > bal_cp,"YES","NO") 
  matrix<-confusionMatrix(table(preds, test_y_temp$accepted), positive="YES")
  pred.acc<-matrix$overall['Accuracy']
  pred.sens<-matrix$byClass['Sensitivity']
  pred.spec<-matrix$byClass['Specificity']
  pred.balaccs<-matrix$byClass['Balanced Accuracy']
  pred.precision<-matrix$byClass['Pos Pred Value']
  
  pred.rocauc <- roc.curve(predicted=pred$YES, response=test_y_temp$accepted)
  #precision recall curve
  fg <- pred$YES[test_y_temp$accepted == "YES"]
  bg <- pred$YES[test_y_temp$accepted == "NO"]
  pred.prauc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)  
  
  
  if(counter==1){request_results<-data.frame()} else {request_results <- request_results}
  
  request_results[counter,1]<-as.character(paste(name,i,sep="_"))
  
  request_results[counter,2] <- pred.sens
  request_results[counter,3] <- pred.spec
  request_results[counter,4] <- pred.precision
  request_results[counter,5] <- pred.acc
  request_results[counter,6] <- pred.balaccs
  request_results[counter,7] <- pred.rocauc$auc
  request_results[counter,8] <- pred.prauc$auc.integral
  

  counter <- counter + 1
  print(counter)
}



ffnnet_request_sampling_results<-request_results
ffnnet_request_sampling_results
ffnnet_request_sampling_results<-request_results%>%rename(Model=V1, "TP rate"=V2, "TN rate"=V3, "Precision"=V4, "Accuracy"=V5, "Bal. Acc."=V6,
                                                       ROCAUC=V7, PRAUC=V8
)
ffnnet_request_sampling_results[1,1] <- "FFNNET no rebal."
ffnnet_request_sampling_results[2,1] <- "FFNNET over"
ffnnet_request_sampling_results[3,1] <- "FFNNET under"
ffnnet_request_sampling_results[4,1] <- "FFNNET smote"
ffnnet_request_sampling_results
write.csv(ffnnet_request_sampling_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/request_ffnnet_resampling.csv")






##### Request XGB ########

# extreme gradient boosting

# gather data
test_sample <- cbind(train_x_request_base,train_y_request)
train_x_temp <- test_sample%>%select(-c(accepted))
train_y_temp <- test_sample%>%select(accepted)


test_sample <- cbind(test_x_request_base,test_y_request)
test_x_temp <- test_sample%>%select(-c(accepted))
test_y_temp <- test_sample%>%select(accepted)

xgbdata<-cbind(train_x_temp, train_y_temp)
prop.table(table(xgbdata$accepted))
table(xgbdata$accepted)


data.smote <- SMOTE(accepted ~ ., data = xgbdata, seed = 1)
prop.table(table(data.rose$accepted))
table(data.rose$accepted)


data.over <- ovun.sample(accepted ~ ., data = xgbdata, method = "over")$data
prop.table(table(data.over$accepted))
table(data.over$accepted)

data.under <- ovun.sample(accepted ~ ., data = xgbdata, method = "under"
                         )$data
prop.table(table(data.under$accepted))
table(data.under$accepted)

# use best resampling method
dfr <- xgbdata
# dfr <- data.under
# dfr <- data.over
# dfr <- data.smote

sparse_matrix <- sparse.model.matrix(accepted~.-1, data = dfr)
output_vector = dfr[,"accepted"] == "YES"
traindata <- cbind(data = test_x_temp, label=test_y_temp)
#prop.table(table(traindata$accepted))
#table(traindata$accepted)

testdata <- as.data.frame(cbind(test_x_temp,test_y_temp))
test_matrix <- sparse.model.matrix(accepted~.-1, data = testdata)
output_test_vector = testdata[,"accepted"] == "YES"

dtrain <- xgb.DMatrix(data = as.matrix(sparse_matrix), label=as.matrix(output_vector))
dtest <- xgb.DMatrix(data = as.matrix(test_matrix), label=as.matrix(output_test_vector))

# train xgb model with watchlist for more convenience about how many rounds are needed
watchlist <- list(train=dtrain, test=dtest)
model.train <- xgb.train(data=dtrain
                  , max.depth=4
                  , eta=0.008
                  , nthread = 5
                  , nrounds=600
                  , watchlist=watchlist
                  , objective = "binary:logistic"
                  )
xgbrequest.model.train <- model.train

# predict test data
pred <- predict(model.train, as.matrix(test_matrix))

# optimal balanced sens spec threshold
fg_bg <- as.data.frame(cbind(pred, as.numeric(test_y_temp$accepted)-1))%>%rename(accepted=V2)
cp <- cutpointr(fg_bg, pred, accepted, pos_class=1,
                method = maximize_metric, metric = sum_sens_spec)
bal_cp <- cp$optimal_cutpoint

preds<- ifelse(pred > bal_cp,"YES","NO") 
matrix<- caret::confusionMatrix(table(preds, test_y_temp$accepted), positive="YES")

pred.acc<-matrix$overall['Accuracy']
pred.sens<-matrix$byClass['Sensitivity']
pred.spec<-matrix$byClass['Specificity']
pred.balaccs<-matrix$byClass['Balanced Accuracy']
pred.precision<-matrix$byClass['Pos Pred Value']

# roc curve
pred.rocauc <- roc.curve(predicted=pred, response=test_y_temp$accepted, curve = TRUE)
#precision recall curve
fg <- pred[test_y_temp$accepted == "YES"]
bg <- pred[test_y_temp$accepted == "NO"]
pred.rocauc <- PRROC::roc.curve(scores.class0=fg, scores.class1=bg, curve = TRUE)

pred.prauc <- PRROC::pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
xgb_roc <- pred.rocauc
xgb_pr <- pred.prauc
plot(xgb_roc)

# store results
counter <- 6 # sixth request model
final_results[counter,1] <- "XGB"
final_results[counter,2] <- pred.sens
final_results[counter,3] <- pred.spec
final_results[counter,4] <- pred.precision
final_results[counter,5] <- pred.acc
final_results[counter,6] <- pred.balaccs
final_results[counter,7] <- pred.rocauc$auc
final_results[counter,8] <- pred.prauc$auc.integral
final_results

#rename results
final_results <- final_results%>%rename(Model = V1
                                        ,"TP rate" = V2
                                        ,"TN rate" = V3
                                        ,"Precision" = V4
                                        ,"Accuracy" = V5
                                        ,"Bal. Acc." = V6
                                        ,"ROCAUC" = V7
                                        ,"PRAUC" = V8)

write.csv(final_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/final_request_results.csv")




# algo for different resampling methods
test_sample <- cbind(train_x_request_base,train_y_request)
train_x_temp <- test_sample%>%select(-c(accepted))
train_y_temp <- test_sample%>%select(accepted)


sampling_methods<-c("none",
                    "up", 
                    "down"
                    ,"smote"
                    # "rose"
)

counter=1
modelname="xgb"
train_data <- train_y_temp
train_data_x <- train_x_temp

for (i in sampling_methods) {
  
  # control for train
  if(i=="none") {
    xgbdata<-cbind(train_data_x, train_data)
    sparse_matrix <- sparse.model.matrix(accepted~.-1, data = xgbdata)
    output_vector = train_data[,"accepted"] == "YES"
    name <- "no resampling"
  } else {
    
    if(i=="smote") {
      rosedata<-cbind(train_data_x, train_data)
      data.rose <- SMOTE(accepted ~ ., data = rosedata, seed = 1)
      train_x_request_base_rose <- data.rose%>%select(-c("accepted"))
      train_y_request_rose <- data.rose%>%select(accepted)

      xgbdata<-cbind(train_x_request_base_rose, train_y_request_rose)
      sparse_matrix <- sparse.model.matrix(accepted~.-1, data = xgbdata)
      output_vector = train_y_request_rose[,"accepted"] == "YES"
      name <- "smote"
    }
    if(i=="down") {
      temp <- cbind(train_data_x, train_data)
      data.under <- ovun.sample(accepted ~ .
                                , data = temp
                                , method = "under"
                                
      )$data
      
      sparse_matrix <- sparse.model.matrix(accepted~.-1, data = data.under)
      output_vector = data.under[,"accepted"] == "YES"
      name <- "under"
    }
    if(i=="up"){
      temp <- cbind(train_data_x, train_data)
      data.over <- ovun.sample(accepted ~ .
                                , data = temp
                                , method = "over"
                                
      )$data
      
      sparse_matrix <- sparse.model.matrix(accepted~.-1, data = data.over)
      output_vector = data.over[,"accepted"] == "YES"
      name <- "over"
    }
  }
  # xgboost
  bst <- xgboost(data = sparse_matrix
                 , label = output_vector
                 , max.depth = 4
                 , eta = 0.008
                 , nthread = 5
                 , nrounds = 600
                 , objective = "binary:logistic")
  
  # prop.table(table(traindata$accepted))
  # table(traindata$accepted)
  
  
  testdata <- as.data.frame(cbind(test_x_request_base,test_y_request))
  test_matrix <- sparse.model.matrix(accepted~.-1, data = testdata)
  output_test_vector = testdata[,"accepted"] == "YES"
  
  
  # create results
  pred <- predict(bst, as.matrix(test_matrix))
  fg_bg <- as.data.frame(cbind(pred, as.numeric(test_y_request$accepted)-1))%>%rename(accepted=V2)
  #optimal balanced sens spec threshold
  cp <- cutpointr(fg_bg, pred, accepted, pos_class=1,
                  method = maximize_metric, metric = sum_sens_spec)
  bal_cp <- cp$optimal_cutpoint
  
  preds<- ifelse(pred > bal_cp,"YES","NO") 
  matrix<-confusionMatrix(table(preds, test_y_request$accepted), positive="YES")
  
  
  pred.acc<-matrix$overall['Accuracy']
  pred.sens<-matrix$byClass['Sensitivity']
  pred.spec<-matrix$byClass['Specificity']
  pred.balaccs<-matrix$byClass['Balanced Accuracy']
  pred.precision<-matrix$byClass['Pos Pred Value']
  
  # precision recall curve and roc curve
  fg <- pred[test_y_request$accepted == "YES"]
  bg <- pred[test_y_request$accepted == "NO"]
  pred.rocauc <- PRROC::roc.curve(scores.class0=fg, scores.class1=bg, curve = TRUE)
  
  pred.prauc <- PRROC::pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  
  
  if(counter==1){request_results<-data.frame()} else {request_results <- request_results}
  
  request_results[counter,1]<-as.character(paste(modelname,name,sep=" "))
  
  request_results[counter,2] <- pred.sens
  request_results[counter,3] <- pred.spec
  request_results[counter,4] <- pred.precision
  request_results[counter,5] <- pred.acc
  request_results[counter,6] <- pred.balaccs
  request_results[counter,7] <- pred.rocauc$auc
  request_results[counter,8] <- pred.prauc$auc.integral
  

  counter <- counter + 1
  
}

# save results
xgb_request_sampling_results<-request_results%>%rename(Model = V1
                                                       ,"TP rate" = V2
                                                       ,"TN rate" = V3
                                                       ,"Precision" = V4
                                                       ,"Accuracy" = V5
                                                       ,"Bal. Acc." = V6
                                                       ,"ROCAUC" = V7
                                                       ,"PRAUC" = V8)
xgb_request_sampling_results[1,1] <- "XGB no rebal."
xgb_request_sampling_results[2,1] <- "XGB over"
xgb_request_sampling_results[3,1] <- "XGB under"
xgb_request_sampling_results[4,1] <- "XGB smote"

file1 <- "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/request_xgb_resampling.csv"
final_results <- data.table::fread(file1)
xgb_request_sampling_results <- final_results%>%select(-c(V1))
write.csv(xgb_request_sampling_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/request_xgb_resampling.csv")



##### Create Request Plots #####

# create example figures for thesis

# roc curve
grid::grid.newpage()
pdf("Roccurve_example.pdf")
dev.control(displaylist="enable")

plot(xgb_roc, lwd=4, ann=F)
title(main="ROC Curve",xlab="False Positive Rate", ylab="True Positive Rate")

p1.base <- recordPlot()
invisible(dev.off())



# PR curve
grid::grid.newpage()
pdf("prcurve_example.pdf")
dev.control(displaylist="enable")
# plot roc curve, save as Roc_curve_requests.pdf in ./figures
plot(xgb_pr,  lwd=4, ann=FALSE)
title(main="Precision-Recall Curve",xlab="True Positive Rate", ylab="Positive Predictive Value")

p1.base <- recordPlot()
invisible(dev.off())

grid.arrange(img1, img2, ncol = 2)

# ROC & PR curves
# create and save plots for ROC and PR curves  
savewd <- "/Users/richardchan/Dropbox/FS19/Master Thesis/figures"
setwd(savewd)


# plot roc curve, save as Roc_curve_requests.pdf in ./figures
grid::grid.newpage()
pdf("Roc_curve_requests.pdf")
dev.control(displaylist="enable")
plot(ffnnet_roc, color="limegreen", lwd=2, ann=FALSE)
plot(xgb_roc, add=TRUE,lwd=2, color="darkorange")
plot(svm_roc, add=TRUE,lwd=2, color="dodgerblue")
plot(rf_roc, add=TRUE, lwd=2, color="black")
title(main="ROC curve",xlab="FP rate", ylab="TP rate",cex=1.5)
legend(0.7, 0.3, legend=c("FFNNET", "XGB", "SVM", "RF"),
       col=c("limegreen", "darkorange", "dodgerblue"),lty=c(1,1,1) , cex=1.2)
p1.base <- recordPlot()
invisible(dev.off())


# plot pr curve, save as Pr_curve_requests.pdf in ./figures
grid::grid.newpage()
pdf("Pr_curve_requests.pdf")
dev.control(displaylist="enable")
plot(ffnnet_pr, color="limegreen", lwd=2,lty=1, ann=FALSE)
plot(xgb_pr, add=TRUE, lwd=2,color="darkorange")
plot(svm_pr, add=TRUE, lwd=2, color="dodgerblue")
plot(rf_pr, add=TRUE, lwd=2, color="black")
title(main="PR curve ",
      xlab="TP rate", ylab="Positive predictive value", cex=1.5)
legend(0.7, 0.99, legend=c("FFNNET", "XGB", "SVM", "RF"),
       col=c("limegreen", "darkorange", "dodgerblue", "black"), lty=c(1,1,1,1), cex=1.2)

p2.base <- recordPlot()
invisible(dev.off())


xgb_request_roc <- pred.rocauc
xgb_request_pr <- pred.prauc
proc_curve_xgb <- pROC::roc(test_y_request$accepted ~pred)



##### Request Skill Granularity Models #####

# First, the data for each granularity is formed
# It takes best models and calculate auc with using different skill granularities
# create the request train, test and validation data separately for skill granularities

# skill granularities
jp_educ_name_features <- "educname_score"
jp_educationgroup_features <- "edugroup_score"
jp_education_features <- "education_score"
jp_edu_level_features <- "edulevel_score"
jp_name_features <- "jpname_score" 
jp_industry_features <- "industry_score"
jp_subindustry_features <- "subindustry_score"

requesttrainData_orig <- requesttrainData
requesttestData_orig <- requesttestData
#requesttrainData <- requesttrainData_orig
#requesttestData <- requesttestData_orig

requesttrainData <- requesttestData_orig
requesttestData <- requesttrainData_orig


# train and test data for each granularities
train_x_request_educ_name_granule<-as.data.frame(requesttrainData[,c(as.numeric(grep(paste(jp_educ_name_features,collapse="|"),colnames(requesttrainData))),
                                                                     as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttrainData))),
                                                                     as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttrainData))),
                                                                     as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttrainData)))
)])

train_x_request_name_granule<-as.data.frame(requesttrainData[,c(as.numeric(grep(paste(jp_name_features,collapse="|"),colnames(requesttrainData))),
                                                                as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttrainData))),
                                                                as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttrainData))),
                                                                as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttrainData)))
)])

train_x_request_edulevel_granule<-as.data.frame(requesttrainData[,c(as.numeric(grep(paste(jp_edu_level_features,collapse="|"),colnames(requesttrainData))),
                                                                as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttrainData))),
                                                                as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttrainData))),
                                                                as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttrainData)))
)])





train_x_request_educationgroup_granule<-as.data.frame(requesttrainData[,c(as.numeric(grep(paste(jp_educationgroup_features,collapse="|"),colnames(requesttrainData))),
                                                                                     as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttrainData))),
                                                                          as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttrainData))), 
                                                                          as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttrainData)))
                                                                          
)])


train_x_request_education_granule<-as.data.frame(requesttrainData[,c(as.numeric(grep(paste(jp_education_features,collapse="|"),colnames(requesttrainData))),
                                                                     as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttrainData))),
                                                                     as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttrainData))),
                                                                     as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttrainData)))
)])


train_x_request_industry_granule<-as.data.frame(requesttrainData[,c(as.numeric(grep(paste(jp_industry_features,collapse="|"),colnames(requesttrainData))),
                                                                     as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttrainData))),
                                                                     as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttrainData))),
                                                                     as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttrainData)))
)])

train_x_request_subindustry_granule<-as.data.frame(requesttrainData[,c(as.numeric(grep(paste(jp_subindustry_features,collapse="|"),colnames(requesttrainData))),
                                                                    as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttrainData))),
                                                                    as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttrainData))),
                                                                    as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttrainData)))
)])




test_x_request_educationgroup_granule<-as.data.frame(requesttestData[,c(as.numeric(grep(paste(jp_educationgroup_features,collapse="|"),colnames(requesttestData))),
                                                                        as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttestData))),
                                                                        as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttestData))),
                                                                        as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttestData)))
)])




test_x_request_education_granule<-as.data.frame(requesttestData[,c(as.numeric(grep(paste(jp_education_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttestData)))
)])

test_x_request_edulevel_granule<-as.data.frame(requesttestData[,c(as.numeric(grep(paste(jp_edu_level_features,collapse="|"),colnames(requesttrainData))),
                                                                    as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttrainData))),
                                                                    as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttrainData))),
                                                                    as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttrainData)))
)])



test_x_request_name_granule<-as.data.frame(requesttestData[,c(as.numeric(grep(paste(jp_name_features,collapse="|"),colnames(requesttestData))),
                                                              as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttestData))),
                                                              as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttestData))),
                                                              as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttestData)))
)])


test_x_request_educ_name_granule<-as.data.frame(requesttestData[,c(as.numeric(grep(paste(jp_educ_name_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttestData)))
)])

test_x_request_industry_granule<-as.data.frame(requesttestData[,c(as.numeric(grep(paste(jp_industry_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttestData)))
)])

test_x_request_subindustry_granule<-as.data.frame(requesttestData[,c(as.numeric(grep(paste(jp_subindustry_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(worker_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(company_features,collapse="|"),colnames(requesttestData))),
                                                                   as.numeric(grep(paste(request_features,collapse="|"),colnames(requesttestData)))
)])

train_y_request<-as.data.frame(as.factor(requesttrainData$accepted))%>%rename(accepted="as.factor(requesttrainData$accepted)")
test_y_request <- as.data.frame(as.factor(requesttestData$accepted))%>%rename(accepted="as.factor(requesttestData$accepted)")


# validation data - best granularity

validationset_x <- as.data.frame(validation_model_data[,c(as.numeric(grep(paste(jp_edu_level_features,collapse="|"),colnames(requesttestData))),
                                                          as.numeric(grep(paste(worker_features,collapse="|"),colnames(validation_model_data))),
                                                          as.numeric(grep(paste(company_features,collapse="|"),colnames(validation_model_data))),
                                                          as.numeric(grep(paste(request_features,collapse="|"),colnames(validation_model_data))))])
validationset_bg <- validationset_x
validationset_bg$wj_id <- validation_model_data$wj_id
validationset_bg$wa_id <- validation_model_data$wa_id
validationset_bg$accepted <- as.numeric(validation_model_data$accepted)-1




# Evaluate which granularity predicts best
# this is only for the best models found above

granularities<-c(
                 "educ_name_granule",
                 "name_granule",
                 "educlevel_granule",
                 "educationgroup_granule",
                 "education_granule",
                 "industry_granule",
                 "subindustry_granule"
)

# use 
models <- c("model1" 
  ,"model2"
  # ,"model3"
  )
counter<-1
for (j in models) {
  for (i in granularities) {
    
    # loops through granules
    set.seed(123)
    if(i=="request_base") {
      train_x <- data.frame(train_x_request_base)
      test_x <- data.frame(test_x_request_base)
      granule <- "no_score"
      }
    if(i=="educationgroup_granule") {
      train_x <- data.frame(train_x_request_educationgroup_granule)
      test_x <- data.frame(test_x_request_educationgroup_granule)
      granule <- "E31"
      }
    if(i=="education_granule") {
      train_x <- data.frame(train_x_request_education_granule)
      test_x <- data.frame(test_x_request_education_granule)
      granule <- "E6"
    }
    if(i=="educlevel_granule") {
      train_x <- train_x_request_edulevel_granule
      test_x <- test_x_request_edulevel_granule
      granule <- "E85"
    }
    if(i=="name_granule") {
      train_x <-  train_x_request_name_granule
      test_x <-  test_x_request_name_granule
      granule <- "E91"
      }
    if(i=="educ_name_granule") {
      train_x <- train_x_request_educ_name_granule
      test_x <- test_x_request_educ_name_granule
      granule <- "E191"
    }
    if(i=="industry_granule") {
      train_x <- train_x_request_industry_granule
      test_x <- test_x_request_industry_granule
      granule <- "I5"
    }
    if(i=="subindustry_granule") {
      train_x <- train_x_request_subindustry_granule
      test_x <- test_x_request_subindustry_granule
      granule <- "I10"
    }
    # output vector stays the same for all granules
    train_y <- train_y_request
    test_y <- test_y_request
    
    # train models using granularity data sets
    if(j == "model1") {
      
      # model 1: RF
      name <- "RF"
      train_control <- trainControl(method="repeatedcv", 
                                    number=10,
                                    classProbs = T,
                                    summaryFunction = twoClassSummary,
                                    repeats=5
                                    )

      tunegrid <- expand.grid(.mtry=c(5:8))
      
      model.train <- train(
        train_x, train_y$accepted, 
        trControl=train_control, 
        method="rf",
        metric="ROC",
        preProcess=c("center","scale")
        ,ntrees=10
        ,tuneGrid = tunegrid
      )
      
      # predict test set
      pred <- predict(model.train, test_x, type="prob")$YES
    
    } else {
      if(j == "model2") {
        
        # Model 2: XGB
        name <- "xgb"
        
        xgbdata<-cbind(train_x, train_y)
        sparse_matrix <- sparse.model.matrix(accepted~.-1, data = xgbdata)
        output_vector = train_y[,"accepted"] == "YES"
        
        temp_x <- test_x
        temp_y <- test_y%>%select(accepted)
        
        testdata <- cbind(temp_x ,temp_y)
        test_matrix <- sparse.model.matrix(accepted~.-1, data = testdata)
        output_test_vector = test_y[,"accepted"] == "YES"
        
        dtrain <- xgb.DMatrix(data = as.matrix(sparse_matrix), label=as.matrix(output_vector))
        dtest <- xgb.DMatrix(data = as.matrix(test_matrix), label=as.matrix(output_test_vector))
        
        # train xgb model
        watchlist <- list(train=dtrain, test=dtest)
        # Model 2: extreme gradient boosting
        watchlist <- list(train=dtrain, test=dtest)
        model.train <- xgb.train(data=dtrain
                                 , max.depth=4
                                 , eta=0.008
                                 , nthread = 5
                                 , nrounds=600
                                 , watchlist=watchlist
                                 , objective = "binary:logistic"
        )
        
        # predict test set

        pred <- predict(model.train, as.matrix(test_matrix))
        
        
      } 
      
      if(j == "model3") {
        
        # Model 3: Feed forward neural network (not shown in thesis, just for fun)
        name <- "ffnnet"
        train_y_nnet <- as.matrix(data.frame(train_y%>%mutate(accepted0=ifelse(accepted=='YES',1,0))%>%select(accepted0)))
        train_x_nnet <- as.matrix(data.frame(train_x))
        
        
        model <- keras_model_sequential() %>%
          
          # network architecture
          layer_dense(units = 50, activation = "relu", input_shape = ncol(train_x),
                      kernel_regularizer = regularizer_l2(0.001)) %>%
          layer_batch_normalization() %>%
          layer_dropout(rate = 0.3) %>%
          layer_dense(units = 30, activation = "relu",
                      kernel_regularizer = regularizer_l1(0.001)) %>%
          layer_batch_normalization() %>%
          layer_dropout(rate = 0.1) %>%
          layer_dense(units = 10, activation = "relu",
                      kernel_regularizer = regularizer_l1(0.001)) %>%
          layer_batch_normalization() %>%
          layer_dropout(rate = 0.1) %>%
          layer_dense(units = 1, activation=sigmoid) %>%
          
          # backpropagation
          compile(
            loss = 'binary_crossentropy',
            optimizer = 'adam',
            metrics = c('AUC')
          )
        
        
        model.train <- model %>% fit(
          x=train_x_nnet,
          y=train_y_nnet,
          epochs = 10,
          batch_size = 32,
          validation_split = .33,
          verbose = FALSE,
          callbacks = list(
            callback_early_stopping(patience = 3),
            callback_reduce_lr_on_plateau()
          )
        )
        
        pred <- model %>% predict_proba(as.matrix(test_x))
        
      }

    }
    
    # create results
    fg_bg <- as.data.frame(cbind(pred, as.numeric(test_y_request$accepted)-1))%>%rename(accepted=V2)
    #optimal balanced sens spec threshold
    cp <- cutpointr(fg_bg, pred, accepted, pos_class=1,
                    method = maximize_metric, metric = sum_sens_spec)
    bal_cp <- cp$optimal_cutpoint
    
    preds<- ifelse(pred > bal_cp,"YES","NO") 
    matrix<-confusionMatrix(table(preds, test_y$accepted), positive="YES")
    pred.acc<-matrix$overall['Accuracy']
    pred.sens<-matrix$byClass['Sensitivity']
    pred.spec<-matrix$byClass['Specificity']
    pred.balaccs<-matrix$byClass['Balanced Accuracy']
    pred.precision<-matrix$byClass['Pos Pred Value']
    #ROC
    pred.rocauc <- roc.curve(predicted=pred, response=test_y$accepted)
    #precision recall curve
    fg <- pred[test_y$accepted == "YES"]
    bg <- pred[test_y$accepted == "NO"]
    pred.prauc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
    
    if(counter==1){request_results<-data.frame()} else {request_results <- request_results}
    
    request_results[counter,1]<-as.character(paste(name,granule,sep="_"))
    
    request_results[counter,2] <- pred.sens
    request_results[counter,3] <- pred.spec
    request_results[counter,4] <- pred.precision
    request_results[counter,5] <- pred.acc
    request_results[counter,6] <- pred.balaccs
    request_results[counter,7] <- pred.rocauc$auc
    request_results[counter,8] <- pred.prauc$auc.integral
    

    counter <- counter + 1
  }
}
request_granules_results<-request_results%>%
  rename(Model=V1, "TP rate"=V2, "TN rate"=V3,"Precision"=V4, "Accuracy"=V5, "Bal. Acc."=V6, ROCAUC=V7,
         PRAUC=V8
  )

# Store results
request_granules_results
write.csv(request_granules_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/request_granules_results.csv")
requ_gran_res <- data.table::fread("/Users/richardchan/Dropbox/FS19/Master Thesis/figures/request_granules_results.csv")
requ_gran_res <- requ_gran_res%>%select(-c(V1))
requ_gran_res[1,1]<-"RF E191"
requ_gran_res[2,1]<-"RF E91"
requ_gran_res[3,1]<-"RF E85"
requ_gran_res[4,1]<-"RF E31"
requ_gran_res[5,1]<-"RF E6"
requ_gran_res[6,1]<-"RF I5"
requ_gran_res[7,1]<-"RF I10"
requ_gran_res[8,1]<-"XGB E191"
requ_gran_res[9,1]<-"XGB E91"
requ_gran_res[10,1]<-"XGB E85"
requ_gran_res[11,1]<-"XGB E31"
requ_gran_res[12,1]<-"XGB E6"
requ_gran_res[13,1]<-"XGB I5"
requ_gran_res[14,1]<-"XGB I10"

write.csv(requ_gran_res, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/request_granules_results.csv")
# It shows that XGB has highest ROC with the E85 granule


##### Request Validation Results #####

# Calculates the final model using the train and test set
# and predicts the validation set

# combine train and test data with the E85 granule
train_x <- rbind(train_x_request_edulevel_granule, test_x_request_edulevel_granule)
train_y <- rbind(train_y_request, test_y_request)


# validationset is the set to predict
# adjust granularities under "Request skill granularities" for the validation set
collumnnames <- colnames(train_x)
test_x <- validationset%>%select(collumnnames)

# Prepare data for train
test_y_request <- as.data.frame(validationset$accepted)%>%rename(accepted="validationset$accepted")
dfr <- cbind(train_x, train_y)
sparse_matrix <- sparse.model.matrix(accepted~.-1, data = dfr)
output_vector = dfr[,"accepted"] == "YES"


testdata <- as.data.frame(cbind(test_x,test_y_request))
test_matrix <- sparse.model.matrix(accepted~.-1, data = testdata)
output_test_vector = testdata[,"accepted"] == "YES"

dtrain <- xgb.DMatrix(data = as.matrix(sparse_matrix), label=as.matrix(output_vector))
dtest <- xgb.DMatrix(data = as.matrix(test_matrix), label=as.matrix(output_test_vector))
watchlist <- list(train=dtrain, test=dtest)

# Train final model
model.train <- xgb.train(data=dtrain
                  , max.depth=4
                  , eta=0.008
                  , nthread = 5
                  , nrounds=600
                  , watchlist=watchlist
                  , objective = "binary:logistic"
                  #, eval.metric = "error"
                  #, eval.metric = "logloss"
)

# predicts data
pred <- predict(model.train, test_matrix, type="prob")


# predict validation data with the best model and the optimal skill allocation
validationset.pred <- validationset%>%select(-c("wj_id","wa_id","accepted"))
validationset$validation_preds <- pred

# create results from validation set with th=0.5
# validation set performance measure 
# only from those that did not answer
val_performance <- validationset%>%filter(answer!="NO_ANSWER")
val_perf.preds <- ifelse(val_performance$validation_preds > 0.5, 1, 0)
val_perf_ref <- as.numeric(val_performance$accepted)
val_perf.matrix<- caret::confusionMatrix(table(val_perf.preds, val_perf_ref), positive = "1")
pred.acc<-val_perf.matrix$overall['Accuracy']
pred.sens<-val_perf.matrix$byClass['Sensitivity']
pred.spec<-val_perf.matrix$byClass['Specificity']
pred.balaccs<-val_perf.matrix$byClass['Specificity']
#ROC
pred.rocauc <- roc.curve(predicted=val_performance$validation_preds, response=val_performance$accepted, curve = T)
#precision recall curve
fg <- val_performance$validation_preds[val_performance$accepted == "1"]
bg <- val_performance$validation_preds[val_performance$accepted == "0"]
pred.prauc <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
pred.rocauc <- PRROC::roc.curve(scores.class0=fg, scores.class1=bg, curve = TRUE)
plot(pred.rocauc)
plot(pred.prauc)



# calculate optimal cutpoints/thresholds
fg_bg <- val_performance

#optimal balanced sens spec threshold
cp <- cutpointr(fg_bg, validation_preds, answer, pos_class="WORKER_ACCEPTED",
                method = maximize_metric, metric = sum_sens_spec)
bal_cp <- cp$optimal_cutpoint
bal_cp
cp$AUC
plot_roc(cp)


# optimal sensitivity threshold
cp_sens <- cutpointr(fg_bg, validation_preds, answer, pos_class="WORKER_ACCEPTED",
                method = maximize_metric, metric = sens_constrain)
sens_cp <- cp_sens$optimal_cutpoint
sens_cp
cp_sens$AUC
plot_roc(cp_sens)

# optimal specificity threshold
cp_spec <- cutpointr(fg_bg, validation_preds, answer, pos_class="WORKER_ACCEPTED",
                method = maximize_metric, metric = spec_constrain)
spec_cp <- cp_spec$optimal_cutpoint
spec_cp
cp_spec$AUC
plot_roc(cp_spec)


# Calculate improvements

# optimal threshold
th <- bal_cp
val.preds<- ifelse (validationset$validation_preds > th, 1, 0)
val.ref  <- as.numeric(validationset$accepted)
val.matrix<- caret::confusionMatrix(table(val.preds, val.ref), positive = "1")

# amount of predicted requests sent
predicted.requests<-sum(val.preds)
# all requests
val.allrequests <- nrow(validationset)
# percentage of improvements in % of all requests
general.improvement <- (1-predicted.requests/val.allrequests)*100


# Hi!
# Make sure to pay extra attention to the form of the confusion matrix, as it's not in the same format as in the thesis ;)

# amount of not sent requests but have accepted
falsenegative <- val.matrix$table[1,2]
# percentage of wrongly excluded requests = (fn/fn+tp)*100
falsenegrate <- falsenegative/(falsenegative+val.matrix$table[2,2])*100

# amount of predicted sent requests which have declined
falsepos <- val.matrix$table[2,1]
# percentage of wrongly included requests = (fp/fp+tn)*100
falseposrate <- falsepos/(falsepos+val.matrix$table[1,1])*100


# per job: compared to above, how many requests saved with threshold?
validationset$val.perjob.preds<- ifelse (validationset$validation_preds > th, 1, 0)
perjob.results.to.5 <- validationset%>%
  select(wa_id, accepted, val.perjob.preds)%>%
  mutate(falseneg = ifelse(val.perjob.preds==0&accepted==1,1,0),
         trueneg = ifelse(val.perjob.preds==0&accepted==0,1,0),
         falsepos = ifelse(val.perjob.preds==1&accepted==0,1,0),
         truepos = ifelse(val.perjob.preds==1&accepted==1,1,0))%>%
  group_by(wa_id)%>%
  dplyr::summarize(requests=as.numeric(n()),
            predrequ=sum(val.perjob.preds),
            sumfalseneg=sum(falseneg),
            savedrequests=sum(falseneg)+sum(trueneg),
            general.improvement = (1-sum(val.perjob.preds)/n())*100,
            falsenegrate = ifelse(is.na(sum(falseneg)/sum(falseneg+truepos))==TRUE, 0, sum(falseneg)/sum(falseneg+truepos)*100),
            falseposrate = ifelse(is.na(sum(falsepos)/sum(falsepos+trueneg))==TRUE, 100, sum(falsepos)/sum(falsepos+trueneg)*100))%>%
  ungroup()%>%dplyr::summarize(avg.general.improvement=mean(general.improvement), avg.falsenegrate=mean(falsenegrate), avg.falseposrate=mean(falseposrate))



# store results
# performance results 
validation_results <- data.frame()
validation_results[1,1]$model<- "Final XGB E85"
validation_results$ROCAUC <- pred.rocauc$auc
validation_results$PRAUC <- pred.prauc$auc.integral

# improvements 
validation_results$general.improvement <- general.improvement
validation_results$general.falseneg <- falsenegrate
validation_results$general.falsepos <-falseposrate
validation_results$perjob.improvement <- perjob.results.to.5$avg.general.improvement
validation_results$perjob.falseneg <- perjob.results.to.5$avg.falsenegrate
validation_results$perjob.falspos <- perjob.results.to.5$avg.falseposrate



# upper bound
# general: how many requests can be saved with a threshold of .5?
th <- sens_cp
val.preds<- ifelse (validationset$validation_preds > th, 1, 0)
val.ref  <- as.numeric(validationset$accepted)
val.matrix<- caret::confusionMatrix(table(val.preds, val.ref), positive = "1")

# amount of predicted requests sent
predicted.requests<-sum(val.preds)
# all requests
val.allrequests <- nrow(validationset)
# percentage of improvements in % of all requests
general.improvement <- (1-predicted.requests/val.allrequests)*100

# amount of not sent requests but would have accepted
falsenegative <- val.matrix$table[1,2]
# percentage of wrongly excluded requests = (fn/fn+tp)*100
falsenegrate <- falsenegative/(falsenegative+val.matrix$table[2,2])*100

# amount of predicted sent requests which have been declined
falsepos <- val.matrix$table[2,1]
# percentage of wrongly included requests = (fp/fp+tn)*100
falseposrate <- falsepos/(falsepos+val.matrix$table[1,1])*100



# per job: compared to above, how many requests saved with threshold of .5?
validationset$val.perjob.preds<- ifelse (validationset$validation_preds > th, 1, 0)
perjob.results.to.lb <- validationset%>%
  select(wa_id, accepted, val.perjob.preds)%>%
  mutate(falseneg = ifelse(val.perjob.preds==0&accepted==1,1,0),
         trueneg = ifelse(val.perjob.preds==0&accepted==0,1,0),
         falsepos = ifelse(val.perjob.preds==1&accepted==0,1,0),
         truepos = ifelse(val.perjob.preds==1&accepted==1,1,0))%>%
  group_by(wa_id)%>%
  dplyr::summarise(requests=n(),
            predrequ=sum(val.perjob.preds),
            sumfalseneg=sum(falseneg),
            savedrequests=sum(falseneg+trueneg),
            general.improvement = (1-sum(val.perjob.preds)/n())*100,
            falsenegrate = ifelse(is.na(sum(falseneg)/sum(falseneg+truepos))==TRUE, 0, sum(falseneg)/sum(falseneg+truepos)*100),
            falseposrate = ifelse(is.na(sum(falsepos)/sum(falsepos+trueneg))==TRUE, 100, sum(falsepos)/sum(falsepos+trueneg)*100))%>%
  ungroup()%>%dplyr::summarise(avg.general.improvement=mean(general.improvement), avg.falsenegrate=mean(falsenegrate), avg.falseposrate=mean(falseposrate))


# improvements lower bound
validation_results$general.improvement.lb <- general.improvement
validation_results$general.falseneg.lb <- falsenegrate
validation_results$general.falsepos.lb <-falseposrate
validation_results$perjob.improvement.lb <- perjob.results.to.lb$avg.general.improvement
validation_results$perjob.falseneg.lb <- perjob.results.to.lb$avg.falsenegrate
validation_results$perjob.falspos.lb <- perjob.results.to.lb$avg.falseposrate




# lower bound
# general: how many requests can be saved
th <- spec_cp
val.preds<- ifelse (validationset$validation_preds > th, 1, 0)
val.ref  <- as.numeric(validationset$accepted)
val.matrix<- caret::confusionMatrix(table(val.preds, val.ref), positive = "1")

# amount of predicted requests sent
predicted.requests<-sum(val.preds)
# all requests
val.allrequests <- nrow(validationset)
# percentage of improvements in % of all requests
general.improvement <- (1-predicted.requests/val.allrequests)*100

# amount of not sent requests but would have accepted
falsenegative <- val.matrix$table[1,2]
# percentage of wrongly excluded requests = (fn/fn+tp)*100
falsenegrate <- falsenegative/(falsenegative+val.matrix$table[2,2])*100

# amount of predicted sent requests which have been declined
falsepos <- val.matrix$table[2,1]
# percentage of wrongly included requests = (fp/fp+tn)*100
falseposrate <- falsepos/(falsepos+val.matrix$table[1,1])*100



# per job: compared to above, how many requests saved with threshold
validationset$val.perjob.preds<- ifelse (validationset$validation_preds > th, 1, 0)
perjob.results.to.ub <- validationset%>%
  select(wa_id, accepted, val.perjob.preds)%>%
  mutate(falseneg = ifelse(val.perjob.preds==0&accepted==1,1,0),
         trueneg = ifelse(val.perjob.preds==0&accepted==0,1,0),
         falsepos = ifelse(val.perjob.preds==1&accepted==0,1,0),
         truepos = ifelse(val.perjob.preds==1&accepted==1,1,0))%>%
  group_by(wa_id)%>%
  dplyr::summarise(requests=n(),
            predrequ=sum(val.perjob.preds),
            sumfalseneg=sum(falseneg),
            savedrequests=sum(falseneg+trueneg),
            general.improvement = (1-sum(val.perjob.preds)/n())*100,
            falsenegrate = ifelse(is.na(sum(falseneg)/sum(falseneg+truepos))==TRUE, 0, sum(falseneg)/sum(falseneg+truepos)*100), 
            falseposrate = ifelse(is.na(sum(falsepos)/sum(falsepos+trueneg))==TRUE, 100, sum(falsepos)/sum(falsepos+trueneg)*100))%>%
  ungroup()%>%dplyr::summarise(avg.general.improvement=mean(general.improvement), avg.falsenegrate=mean(falsenegrate), avg.falseposrate=mean(falseposrate))


validation_results$general.improvement.ub <- general.improvement
validation_results$general.falseneg.ub <- falsenegrate
validation_results$general.falsepos.ub <-falseposrate
validation_results$perjob.improvement.ub <- perjob.results.to.ub$avg.general.improvement
validation_results$perjob.falseneg.ub <- perjob.results.to.ub$avg.falsenegrate
validation_results$perjob.falspos.ub <- perjob.results.to.ub$avg.falseposrate



# store results
validation_results
names(validation_results)
val_res <- data.frame()
val_res[1,1]$Boundary <- "optimum"
val_res[2,1] <- "upper"
val_res[3,1] <- "lower"

val_res[1,2]$ROCAUC <- validation_results$ROCAUC
val_res[2,2] <- validation_results$ROCAUC
val_res[3,2] <- validation_results$ROCAUC

val_res[1,3]$PRAUC <- validation_results$PRAUC
val_res[2,3] <- validation_results$PRAUC
val_res[3,3] <- validation_results$PRAUC

val_res[1,4]$genImp <- validation_results$general.improvement
val_res[2,4] <- validation_results$general.improvement.lb
val_res[3,4] <- validation_results$general.improvement.ub

val_res[1,5]$genFN <- validation_results$general.falseneg
val_res[2,5] <- validation_results$general.falseneg.lb
val_res[3,5] <- validation_results$general.falseneg.ub

val_res[1,6]$genFP <- validation_results$general.falsepos
val_res[2,6] <- validation_results$general.falsepos.lb
val_res[3,6] <- validation_results$general.falsepos.ub

val_res[1,7]$pjImp <- validation_results$perjob.improvement
val_res[2,7] <- validation_results$perjob.improvement.lb
val_res[3,7] <- validation_results$perjob.improvement.ub

val_res[1,8]$pjFN <- validation_results$perjob.falseneg
val_res[2,8] <- validation_results$perjob.falseneg.lb
val_res[3,8] <- validation_results$perjob.falseneg.ub

val_res[1,9]$pjFP <- validation_results$perjob.falspos
val_res[2,9] <- validation_results$perjob.falspos.lb
val_res[3,9] <- validation_results$perjob.falspos.ub
val_res
write.csv(val_res, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/request_validation_results.csv")




validation_results[1,1]$model <- "final_model"
validation_results$ROCAUC <- pred.rocauc$auc
validation_results$PRAUC <- pred.prauc$auc.integral

#######################################
##### Rating Machine learning models #####


# create rating model data with features from entities


# worker features
columns.worker.rating.features<-c(as.character(c("worker_age","gender_Female",
                                                 "receive_newsletter","count_cv","has_picture","photos",
                                                 "is_domestic","description_length","count_certificate",
                                                 "count_diploma","count_testimonial","count_drivinglicens",
                                                 "worker_language_DE","worker_language_FR",
                                                 "average_rating", "comments_count",
                                                 "writing_level_a","writing_level_b","writing_level_c","writing_level_l",
                                                 "speaking_level_a","speaking_level_b","speaking_level_c","speaking_level_l",
                                                 "mean_reactiontime"
)))

# company features
columns.company.rating.features<-c(as.character(c("company_was_referred", "company_size","company_count_employers",
                                                  "company_favorite_workers","company_former_workers",
                                                  "company_days_to_first_shift", "average_rating"
)))

# rating data features
columns.rating.features <- c(as.character(c("is_favorite","is_former","minutes_to_reaction", 
                                            "days_since_ll","job_duration", "salary","distance_to_job",
                                            "language_skill_required","driving_skill_required_","uniform",
                                            "job_name_length","job_requirements_length","clothing_reuqrements_length",
                                            "meeting_point_length","additional_skills_required_count",
                                            "use_former_workers","use_public_workers","use_favorite_workers",
                                            "salary_diff", "duration_diff","mean_wage_diff","min_wage_diff"
)))


# create datasets
worker_features<-paste0(colnames(ratingtrainData)[grep(paste(columns.worker.rating.features, collapse="|"),colnames(ratingtrainData))])
company_features<-paste0(colnames(ratingtrainData)[grep(paste(columns.company.rating.features,collapse="|"),colnames(ratingtrainData))])
rating_features <- paste0(colnames(ratingtrainData)[grep(paste(columns.rating.features,collapse="|"),colnames(ratingtrainData))])


train_x_rating_base<-as.data.frame(ratingtrainData[,c(as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtrainData))),
                                                      as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtrainData))), 
                                                      as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtrainData))))])

train_y_rating<-as.data.frame(as.numeric(ratingtrainData$worker_rating))%>%rename(worker_rating="as.numeric(ratingtrainData$worker_rating)")





test_x_rating_base<-as.data.frame(ratingtestData[,c(as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtestData))),
                                                    as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtestData))), 
                                                    as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtestData))))])


test_y_rating<-as.data.frame(as.numeric(ratingtestData$worker_rating))%>%rename(worker_rating="as.numeric(ratingtestData$worker_rating)")

# validation data 
validationset_x <- as.data.frame(validation_model_data[,c(as.numeric(grep(paste(worker_features,collapse="|"),colnames(validation_model_data))),
                                                          as.numeric(grep(paste(company_features,collapse="|"),colnames(validation_model_data))),
                                                          as.numeric(grep(paste(rating_features,collapse="|"),colnames(validation_model_data))))])
validationset <- validationset_x
validationset$wj_id <- validation_model_data$wj_id
validationset$wa_id <- validation_model_data$wa_id
validationset$worker_rating_orig <- as.numeric(validation_model_data$worker_rating)
validationset$worker_rating<-validationset$worker_rating_orig/validation_model_data$avg_rating

# results dataframe
final_rating_results <- data.frame()
# or load if existing
file1 <- "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/final_rating_results.csv"
final_rating_results <- data.table::fread(file1)
final_rating_results <- final_rating_results%>%select(-c(V1))



# check correlation matrix of the features (not in thesis)
# correlation matrix
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

matrixdata<-ratingtrainData[grep(paste(columns.jp.educ.name.granule,collapse="|"),colnames(ratingtrainData))]

corrmatrixdata<-na.omit(matrixdata)%>%dplyr::select_if(is.numeric)
corrmatrixdata_factors<-matrixdata%>%select_if(is.factor)
corrmatrixdata_nas<-corrmatrixdata[rowSums(is.na(corrmatrixdata)) > 0,]

res2<-rcorr(as.matrix(corrmatrixdata[,]))
cormatrix<-flattenCorrMatrix(res2$r, res2$P)%>%filter(abs(cor)>0.75)
cormatrix
rating_model_data
# all good to go!


##### Rating FFNNET #####


# train models through for loop of granularities
test_sample <- cbind(train_x_rating_base, train_y_rating)
train_x_temp <- test_sample%>%select(-c(worker_rating))
train_y_temp <- test_sample%>%select(worker_rating)


granularities <- c("base_model"
)
counter <- 1
for (i in granularities) {
  
  set.seed(123)
  if(i=="base_model") {train_x <- data.frame(train_x_temp)}
  if(i=="educationgroup_granule") {train_x <- data.frame(train_x_educationgroup_granule)}
  if(i=="education_granule") { train_x <- data.frame(train_x_education_granule)}
  if(i=="name_granule") {train_x <-  train_x_name_granule}
  if(i=="educ_name_granule") {train_x <- train_x_educ_name_granule}
  
  
  
  # model network architecture with 5 layers with rectified linear unit
  # performance decreases with less layers and more units per layer
  model <- keras_model_sequential() %>%
    
    layer_dense(units = 50, activation = "relu", input_shape = ncol(train_x)
              ) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 30, activation = "relu",
                kernel_regularizer = regularizer_l1_l2(0.01)) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 10, activation = "relu",
                kernel_regularizer = regularizer_l1_l2(0.01)) %>%
    #layer_batch_normalization() %>%
    #layer_dropout(rate = 0.5) %>%
    #layer_dense(units = 15, activation = "relu",
    #            kernel_regularizer = regularizer_l1_l2(0.01)) %>%
    #layer_batch_normalization() %>%
    #layer_dropout(rate = 0.5) %>%
    #layer_dense(units = 8, activation = "relu",
    #            kernel_regularizer = regularizer_l2(0.01)) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 1) %>%
    
    # backpropagation
    compile(
      optimizer = "rmsprop",
      loss = "mse",
      metrics = c("mae")
    )
  
  train_x_mat <- as.matrix(train_x)
  train_y_mat<-as.matrix(train_y_temp)
  model.train <- model %>% fit(
    x=train_x_mat,
    y=train_y_mat,
    epochs = 75, # 100, # 150
    batch_size = 64,
    validation_split = .33,
    verbose = FALSE,
    callbacks = list(
      callback_early_stopping(patience = 10)
      , callback_reduce_lr_on_plateau()
     )
  )
  
  
  if(counter==1){ffnnet_results<-data.frame()} else {ffnnet_results <- ffnnet_results}
  if(i == "base_model"){test_x <- data.frame(test_x_rating_base)}
  if(i=="educationgroup_granule") {test_x <- data.frame(test_x_educationgroup_granule)}
  if(i=="education_granule") { test_x <- data.frame(test_x_education_granule)}
  if(i=="name_granule") {test_x <-  test_x_name_granule}
  if(i=="educ_name_granule") {test_x <- test_x_educ_name_granule}
  test_y<-as.matrix(test_y_rating)
  test_x<-as.matrix(test_x)

  # predict test set
  predictions <- model %>% predict(test_x)
  
  
  #predicted_ratings_denorm <- predictions
  ratingtestData$predicted_ratings_denorm <- predictions*ratingtestData$avg_rating
  # create results
  test_rmse <- rmse(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
  test_mae <- mae(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
  # PCC2
  test_rsq <- as.numeric(cor(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm) ^ 2)

  
  # R2 calculation traditional
  squared_errors <- as.numeric((ratingtestData$worker_rating_orig - ratingtestData$predicted_ratings_denorm)^2)
  rss <- sum(squared_errors)
  squared_val <- (ratingtestData$worker_rating_orig - mean(ratingtestData$worker_rating_orig)) ^ 2
  tss <- sum(squared_val)
  R2 <- 1 - (rss/tss)
  R2
  
  # adjusted R squared
  n <- nrow(test_y)
  p <- ncol(test_x)
  adjRsq = 1 - ((1-test_rsq)*(n - 1)/(n - p))
  adjRsq
  adjR2 = 1 - ((1-R2)*(n - 1)/(n - p))
  adjR2
  
  ffnnet_results[counter,1]<-as.character(paste("ffnnet",i,sep="_"))
  
  ffnnet_results[counter,2]<-test_rmse
  ffnnet_results[counter,3]<-test_mae
  ffnnet_results[counter,4]<-test_rsq
  ffnnet_results[counter,5]<-R2
  ffnnet_results[counter,6]<-adjRsq
  ffnnet_results[counter,7]<-adjR2
  
  
 counter <- counter + 1

}
ffnnet_results
ffnnet_results_set <- ffnnet_results
ffnnet_results <- ffnnet_results%>%
  rename(model=V1,RMSE=V2, MAE=V3, R2corr=V4, R2tra=V5, adjR2c=V6, adjR2t=V7)

# store results
counter <- 1

final_rating_results[counter,1] <- "FFNNET"
final_rating_results[counter,2] <- ffnnet_results$RMSE
final_rating_results[counter,3] <- ffnnet_results$MAE
final_rating_results[counter,4] <- ffnnet_results$R2corr
final_rating_results[counter,5] <- ffnnet_results$R2tra
final_rating_results[counter,6] <- ffnnet_results$adjR2c
final_rating_results[counter,7] <- ffnnet_results$adjR2t
final_rating_results[counter,6] <- ffnnet_results$adjR2t
final_rating_results

##### Rating RF #####

set.seed(123)
test_sample_rating <- cbind(train_x_rating_base,train_y_rating)
train_x <- test_sample_rating%>%select(-c(worker_rating))
train_y <- test_sample_rating%>%select(worker_rating)


train_control_raw <- trainControl(method="repeatedcv", 
                                  number=10, repeats=5)
tunegrid  <- expand.grid(.mtry=c(5:8))

rf.rating.model <- train(x = train_x,
                y = train_y$worker_rating,
                trControl = train_control_raw, 
                preProc = c("center", "scale"),
                method ="rf"  # method = "parRF", # for faster parallel 
                ,ntree = 10
                #,tuneLength = 3
                ,tuneGrid = tunegrid
                )
rf.rating.model


predictions <- predict(rf.rating.model, test_x_rating_base)
# re-scale predictions (denormalize)
ratingtestData$predicted_ratings_denorm <- predictions*ratingtestData$avg_rating
# create results
test_rmse <- rmse(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
test_mae <- mae(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
# PCC2
test_rsq <- as.numeric(cor(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm) ^ 2)


# R2 calculation traditional
squared_errors <- as.numeric((ratingtestData$worker_rating_orig - ratingtestData$predicted_ratings_denorm)^2)
rss <- sum(squared_errors)
squared_val <- (ratingtestData$worker_rating_orig - mean(ratingtestData$worker_rating_orig)) ^ 2
tss <- sum(squared_val)
R2 <- 1 - (rss/tss)
R2

# adjusted R squared
n <- nrow(test_y)
p <- ncol(test_y)
adjRsq = 1 - ((1-test_rsq)*(n - 1)/(n - p))
adjRsq
adjR2 = 1 - ((1-R2)*(n - 1)/(n - p))
adjR2

i <- "base_model"


# store results
counter <- 2
final_rating_results[counter,1] <- "RF"
final_rating_results[counter,2] <- test_rmse
final_rating_results[counter,3] <- test_mae
final_rating_results[counter,4] <- test_rsq
final_rating_results[counter,5] <- R2
final_rating_results[counter,6] <- adjR2
final_rating_results[counter,7] <- adjR2
final_rating_results

##### Rating LM #####

set.seed(123)
test_sample_rating <- cbind(train_x_rating_base,train_y_rating)
train_x <- test_sample_rating%>%select(-c(worker_rating))
train_y <- test_sample_rating%>%select(worker_rating)
  
# train control
train_control <- trainControl(method="repeatedcv", 
                              number=5, repeats=5)
  
# train model
model.train.lm <- train(train_x, train_y$worker_rating,
                     trControl=train_control, 
                     method="lm",
                     metric="RMSE"
                     )
  

test_sample_rating <- cbind(test_x_rating_base,train_y_rating)
train_x <- test_sample_rating%>%select(-c(worker_rating))
train_y <- test_sample_rating%>%select(worker_rating)

predictions <- predict(model.train.lm, test_x_rating_base)
# re-scale predictions (denormalize)
ratingtestData$predicted_ratings_denorm <- predictions*ratingtestData$avg_rating
# create results
test_rmse <- rmse(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
test_mae <- mae(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
# PCC2
test_rsq <- as.numeric(cor(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm) ^ 2)
  
  
# R2 calculation traditional
squared_errors <- as.numeric((ratingtestData$worker_rating_orig - ratingtestData$predicted_ratings_denorm)^2)
rss <- sum(squared_errors)
squared_val <- (ratingtestData$worker_rating_orig - mean(ratingtestData$worker_rating_orig)) ^ 2
tss <- sum(squared_val)
R2 <- 1 - (rss/tss)
R2
  
# adjusted R squared
n <- nrow(ratingtestData)
p <- ncol(ratingtestData)
adjRsq = 1 - ((1-test_rsq)*(n - 1)/(n - p))
adjRsq
adjR2 = 1 - ((1-R2)*(n - 1)/(n - p))
adjR2
  
i <- "base_model"
rating_results <- data.frame()
counter <- 1
if(counter==1){rating_results<-data.frame()} else {rating_results <- rating_results}
  
rating_results[counter,1]<-as.character(paste("lm",i,sep="_"))
  
rating_results[counter,2]<-test_rmse
rating_results[counter,3]<-test_mae
rating_results[counter,4]<-test_rsq
rating_results[counter,5]<-R2
rating_results[counter,6]<-adjRsq
rating_results[counter,7]<-adjR2
  
rating_results <- rating_results%>%
rename(model=V1,RMSE=V2, MAE=V3, R2corr=V4, R2tra=V5, adjR2c=V6, adjR2t=V7)
  
lm_rating_results <- rating_results
lm_rating_results

# store results
counter <- 3
final_rating_results[counter,1] <- "LM"
final_rating_results[counter,2] <- lm_rating_results$RMSE
final_rating_results[counter,3] <- lm_rating_results$MAE
final_rating_results[counter,4] <- lm_rating_results$R2corr
final_rating_results[counter,5] <- lm_rating_results$R2tra
final_rating_results[counter,6] <- lm_rating_results$adjR2c
final_rating_results[counter,7] <- lm_rating_results$adjR2t
final_rating_results



##### Rating KNN #####

# KNN model

set.seed(123)
test_sample_rating <- cbind(train_x_rating_base,train_y_rating)
train_x <- test_sample_rating%>%select(-c(worker_rating))
train_y <- test_sample_rating%>%select(worker_rating)


train_control_raw <- trainControl(method="repeatedcv", 
                                  number=10, repeats=10)

tunegrid <- expand.grid(k=c(9:12))

knn.model.fit <- train(x = train_x,
                y = train_y$worker_rating,
                trControl = train_control_raw, 
                preProc = c("center", "scale"),
                method ="knn",
                metric="MAE"
                # ,tuneLength = 3
                ,tuneGrid = tunegrid
                )
knn.model.fit

predictions <- predict(knn.model.fit, test_x_rating_base)
# re-scale predictions (denormalization)
ratingtestData$predicted_ratings_denorm <- predictions*ratingtestData$avg_rating
# create results
test_rmse <- rmse(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
test_mae <- mae(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
# PCC2
test_rsq <- as.numeric(cor(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm) ^ 2)


# R2 calculation traditional
squared_errors <- as.numeric((ratingtestData$worker_rating_orig - ratingtestData$predicted_ratings_denorm)^2)
rss <- sum(squared_errors)
squared_val <- (ratingtestData$worker_rating_orig - mean(ratingtestData$worker_rating_orig)) ^ 2
tss <- sum(squared_val)
R2 <- 1 - (rss/tss)
R2

# adjusted R squared
n <- nrow(ratingtestData)
p <- ncol(ratingtestData)
adjRsq = 1 - ((1-test_rsq)*(n - 1)/(n - p))
adjRsq
adjR2 = 1 - ((1-R2)*(n - 1)/(n - p))
adjR2

i <- "base_model"
counter <- 1
if(counter==1){rating_results<-data.frame()} else {rating_results <- rating_results}
rating_results[counter,1]<-as.character(paste("rf",i,sep="_"))

rating_results[counter,2]<-test_rmse
rating_results[counter,3]<-test_mae
rating_results[counter,4]<-test_rsq
rating_results[counter,5]<-R2
rating_results[counter,6]<-adjRsq
rating_results[counter,7]<-adjR2

rating_results <- rating_results%>%
  rename(model=V1,RMSE=V2, MAE=V3, R2corr=V4, R2tra=V5, adjR2c=V6, adjR2t=V7)

knn_rating_results <- rating_results
knn_rating_results

# store results
counter <- 4
final_rating_results[counter,1] <- "KNN"
final_rating_results[counter,2] <- knn_rating_results$RMSE
final_rating_results[counter,3] <- knn_rating_results$MAE
final_rating_results[counter,4] <- knn_rating_results$R2corr
final_rating_results[counter,5] <- knn_rating_results$R2tra
final_rating_results[counter,6] <- knn_rating_results$adjR2t
final_rating_results[counter,7] <- knn_rating_results$adjR2c

final_rating_results

##### Rating SVM #####

# support vector machines



test_sample_rating <- cbind(train_x_rating_base,train_y_rating)
train_x <- test_sample_rating%>%select(-c(worker_rating))
train_y <- test_sample_rating%>%select(worker_rating)

train_control_raw <- trainControl(method="repeatedcv", 
                                  number=3, repeats=2)


grid<-expand.grid( sigma = 2^c(-12:-8),
  C = c(0.75,1,2)
)

svm.poly <- train(x=train_x,
                   y=train_y$worker_rating,
                   trControl=train_control_raw,
                   metric="Rsquared",
                   preProcess=c("center","scale"),
                    tuneGrid=grid,
                   # tuneLength = 3,
                   method="svmRadial")
svm.poly

# predict results
predictions <- predict(svm.poly, test_x_rating_base)
# re-scale predictions (denormalize)

ratingtestData$predicted_ratings_denorm <- predictions*ratingtestData$avg_rating
# create results
test_rmse <- rmse(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
test_mae <- mae(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
# PCC2
test_rsq <- as.numeric(cor(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm) ^ 2)


# R2 calculation traditional
squared_errors <- as.numeric((ratingtestData$worker_rating_orig - ratingtestData$predicted_ratings_denorm)^2)
rss <- sum(squared_errors)
squared_val <- (ratingtestData$worker_rating_orig - mean(ratingtestData$worker_rating_orig)) ^ 2
tss <- sum(squared_val)
R2 <- 1 - (rss/tss)
R2

# adjusted R squared
n <- nrow(test_y)
p <- ncol(test_y)
adjRsq = 1 - ((1-test_rsq)*(n - 1)/(n - p))
adjRsq
adjR2 = 1 - ((1-R2)*(n - 1)/(n - p))
adjR2

i <- "base_model"
counter <- 1
if(counter==1){rating_results<-data.frame()} else {rating_results <- rating_results}
rating_results[counter,1]<-as.character(paste("svm",i,sep="_"))

rating_results[counter,2]<-test_rmse
rating_results[counter,3]<-test_mae
rating_results[counter,4]<-test_rsq
rating_results[counter,5]<-R2
rating_results[counter,6]<-adjRsq
rating_results[counter,7]<-adjR2

rating_results <- rating_results%>%
  rename(model=V1,RMSE=V2, MAE=V3, R2corr=V4, R2tra=V5, adjR2c=V6, adjR2t=V7)

svm_rating_results <- rating_results
svm_rating_results


# store results
counter <- 5
final_rating_results[counter,1] <- "SVM"
final_rating_results[counter,2] <- svm_rating_results$RMSE
final_rating_results[counter,3] <- svm_rating_results$MAE
final_rating_results[counter,4] <- svm_rating_results$R2corr
final_rating_results[counter,5] <- svm_rating_results$R2tra
final_rating_results[counter,6] <- svm_rating_results$adjR2t
final_rating_results[counter,7] <- svm_rating_results$adjR2c

final_rating_results

##### Rating XGB #####


set.seed(123)
test_sample_rating <- cbind(train_x_rating_base,train_y_rating)
train_x <- test_sample_rating%>%select(-c(worker_rating))
train_y <- test_sample_rating%>%select(worker_rating)


# input data
xgbdata<-cbind(train_x, train_y)
dfr<-test_sample_rating

sparse_matrix <- sparse.model.matrix(worker_rating~.-1, data = dfr)

output_vector = dfr[,"worker_rating"] 

testdata <- as.data.frame(cbind(test_x_rating_base,test_y_rating))
test_matrix <- sparse.model.matrix(worker_rating~.-1, data = testdata)
output_test_vector = testdata[,"worker_rating"]

dtrain <- xgb.DMatrix(data = as.matrix(sparse_matrix), label=as.matrix(output_vector))
dtest <- xgb.DMatrix(data = as.matrix(test_matrix), label=as.matrix(output_test_vector))



# xgboost using watchlist
# the test set is not used to train the model, only to watch the test error during training
# Watch how it grows, adjust parameters such that 
# the difference between train and test error gets smaller (overfit)
# and the errors itself is smallest
watchlist <- list(train=dtrain, test=dtest)
model.train <- xgb.train(data=dtrain,
                  max.depth=4,
                  eta=0.008, 
                  nthread = 5, 
                  nrounds=600, 
                  watchlist=watchlist,
                  eval.metric = "rmse")
model.train

# predict test set
ratingtestData$predictions <- predict(model.train, as.matrix(test_matrix))
# re-scale predictions (denormalize)
ratingtestData$predicted_ratings_denorm <- ratingtestData$predictions*ratingtestData$avg_rating


# create results
test_rmse <- rmse(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
test_mae <- mae(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
# PCC2
test_rsq <- as.numeric(cor(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm) ^ 2)
# R2 calculation traditional
squared_errors <- as.numeric((ratingtestData$worker_rating_orig - ratingtestData$predicted_ratings_denorm)^2)
rss <- sum(squared_errors)
squared_val <- (ratingtestData$worker_rating_orig - mean(ratingtestData$worker_rating_orig)) ^ 2
tss <- sum(squared_val)
R2 <- 1 - (rss/tss)
R2
# adjusted R squared
n <- nrow(test_y)
p <- ncol(test_y)
adjRsq = 1 - ((1-test_rsq)*(n - 1)/(n - p))
adjRsq
adjR2 = 1 - ((1-R2)*(n - 1)/(n - p))
adjR2

i <- "base_model"
counter <- 1
if(counter==1){rating_results<-data.frame()} else {rating_results <- rating_results}
rating_results[counter,1]<-as.character(paste("xgb",i,sep="_"))

rating_results[counter,2]<-test_rmse
rating_results[counter,3]<-test_mae
rating_results[counter,4]<-test_rsq
rating_results[counter,5]<-R2
rating_results[counter,6]<-adjRsq
rating_results[counter,7]<-adjR2

rating_results <- rating_results%>%
  rename(model=V1,RMSE=V2, MAE=V3, R2cor=V4, R2tra=V5, adjR2c=V6, adjR2t=V7)

xgb_rating_results <- rating_results
xgb_rating_results

# store results
counter <- 6
final_rating_results[counter,1] <- "XGB"
final_rating_results[counter,2] <- xgb_rating_results$RMSE
final_rating_results[counter,3] <- xgb_rating_results$MAE
final_rating_results[counter,4] <- xgb_rating_results$R2cor
final_rating_results[counter,5] <- xgb_rating_results$R2tra
final_rating_results[counter,6] <- xgb_rating_results$adjR2t
final_rating_results[counter,7] <- xgb_rating_results$adjR2c
final_rating_results

final_rating_results <- final_rating_results%>%
  select(V1, V2, V3, V4, V5, V7)%>%
  rename(model=V1,RMSE=V2, MAE=V3, PCC2=V4, R2=V5, AdjR2=V7)
 
final_rating_results
write.csv(final_rating_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/final_rating_results.csv")

##### Rating Skill Granularity Models #####

# take best rating models using different skill granularities

# skill granularity models
jp_industry_features <- "industry_score"
jp_subindustry_features <- "subindustry_score"
jp_education_features <- "education_score"
jp_edugroup_features <- "edugroup_score"
jp_edu_level_features <- "edulevel_score"
jp_name_features<-"jpname_score"
jp_educ_name_features<-"educname_score"


ratingtrainData_orig <- ratingtrainData
ratingtestData_orig <- ratingtestData


# create datasets with features
train_x_rating_educ_name_granule<-as.data.frame(ratingtrainData[,c(as.numeric(grep(paste(jp_educ_name_features,collapse="|"),colnames(ratingtrainData))),
                                                                   as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtrainData))),
                                                                   as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtrainData))),
                                                                   as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtrainData)))
)])

train_x_rating_name_granule<-as.data.frame(ratingtrainData[,c(as.numeric(grep(paste(jp_name_features,collapse="|"),colnames(ratingtrainData))),
                                                              as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtrainData))),
                                                              as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtrainData))),
                                                              as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtrainData)))
)])


train_x_rating_edulevel_granule<-as.data.frame(ratingtrainData[,c(as.numeric(grep(paste(jp_edu_level_features,collapse="|"),colnames(ratingtrainData))),
                                                                  as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtrainData))),
                                                                  as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtrainData))),
                                                                  as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtrainData)))
)])



train_x_rating_educationgroup_granule<-as.data.frame(ratingtrainData[,c(as.numeric(grep(paste(jp_edugroup_features,collapse="|"),colnames(ratingtrainData))),
                                                                 as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtrainData))),
                                                                 as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtrainData))),
                                                                 as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtrainData)))
)])


train_x_rating_education_granule<-as.data.frame(ratingtrainData[,c(as.numeric(grep(paste(jp_education_features,collapse="|"),colnames(ratingtrainData))),
                                                            as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtrainData))),
                                                            as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtrainData))),
                                                            as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtrainData)))
)])

train_x_rating_industry_granule<-as.data.frame(ratingtrainData[,c(as.numeric(grep(paste(jp_industry_features,collapse="|"),colnames(ratingtrainData))),
                                                                   as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtrainData))),
                                                                   as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtrainData))),
                                                                   as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtrainData)))
)])

train_x_rating_subindustry_granule<-as.data.frame(ratingtrainData[,c(as.numeric(grep(paste(jp_subindustry_features,collapse="|"),colnames(ratingtrainData))),
                                                                   as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtrainData))),
                                                                   as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtrainData))),
                                                                   as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtrainData)))
)])



# test data
test_x_rating_educ_name_granule<-as.data.frame(ratingtestData[,c(as.numeric(grep(paste(jp_educ_name_features,collapse="|"),colnames(ratingtestData))),
                                                         as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtestData))),
                                                         as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtestData))),
                                                         as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtestData)))
)])

test_x_rating_name_granule<-as.data.frame(ratingtestData[,c(as.numeric(grep(paste(jp_name_features,collapse="|"),colnames(ratingtestData))),
                                                            as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtestData))),
                                                            as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtestData))),
                                                            as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtestData)))
)])

test_x_rating_edulevel_granule<-as.data.frame(ratingtestData[,c(as.numeric(grep(paste(jp_edu_level_features,collapse="|"),colnames(ratingtestData))),
                                                                  as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtestData))),
                                                                  as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtestData))),
                                                                  as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtestData)))
)])

test_x_rating_education_granule<-as.data.frame(ratingtestData[,c(as.numeric(grep(paste(jp_education_features,collapse="|"),colnames(ratingtestData))),
                                                                 as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtestData))),
                                                                 as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtestData))),
                                                                 as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtestData)))
)])

test_x_rating_educationgroup_granule<-as.data.frame(ratingtestData[,c(as.numeric(grep(paste(jp_edugroup_features,collapse="|"),colnames(ratingtestData))),
                                                                      as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtestData))),
                                                                      as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtestData))),
                                                                      as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtestData)))
)])

test_x_rating_industry_granule<-as.data.frame(ratingtestData[,c(as.numeric(grep(paste(jp_industry_features,collapse="|"),colnames(ratingtestData))),
                                                                      as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtestData))),
                                                                      as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtestData))),
                                                                      as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtestData)))
)])
test_x_rating_subindustry_granule<-as.data.frame(ratingtestData[,c(as.numeric(grep(paste(jp_subindustry_features,collapse="|"),colnames(ratingtestData))),
                                                                      as.numeric(grep(paste(worker_features,collapse="|"),colnames(ratingtestData))),
                                                                      as.numeric(grep(paste(company_features,collapse="|"),colnames(ratingtestData))),
                                                                      as.numeric(grep(paste(rating_features,collapse="|"),colnames(ratingtestData)))
)])



train_y_rating<-as.data.frame(as.numeric(ratingtrainData$worker_rating))%>%rename(worker_rating="as.numeric(ratingtrainData$worker_rating)")
test_y_rating <- as.data.frame(as.numeric(ratingtestData$worker_rating))%>%rename(worker_rating="as.numeric(ratingtestData$worker_rating)")



rm(columns.worker.rating.features,columns.company.rating.features,columns.jp.industry.granule,
   columns.jp.subindustry.granule,columns.jp.education.granule,columns.jp.educationgroup.granule,
   columns.jp.name.granule,columns.jp.educ.name.granule, columns.request.features)




# then evaluate which granularity predicts best
# this is only for the best models found above
granularities<-c(
                 "educ_name_granule",
                 "name_granule",
                 "educlevel_granule",
                 "educationgroup_granule",
                 "education_granule",
                 "industry_granule",
                 "subindustry_granule"
                 )

# models <- c("model1", "model2","model3")
models <- c("model2","model3") # RF and XGB

counter<-1
for (j in models) {
  for (i in granularities) {
  
    # loops through granules
    set.seed(123)
    if(i=="request_base") {
      train_x <- data.frame(train_x_rating_base)
      test_x <- data.frame(test_x_rating_base)
      granule <- "no_score"
    }
    if(i=="educationgroup_granule") {
      train_x <- data.frame(train_x_rating_educationgroup_granule)
      test_x <- data.frame(test_x_rating_educationgroup_granule)
      granule <- "E31"
    }
    if(i=="educlevel_granule") {
      train_x <- data.frame(train_x_rating_edulevel_granule)
      test_x <- data.frame(test_x_rating_edulevel_granule)
      granule <- "E84"
    }
    if(i=="education_granule") {
      train_x <- data.frame(train_x_rating_education_granule)
      test_x <- data.frame(test_x_rating_education_granule)
      granule <- "E6"
    }
    if(i=="name_granule") {
      train_x <-  train_x_rating_name_granule
      test_x <-  test_x_rating_name_granule
      granule <- "E91"
    }
    if(i=="educ_name_granule") {
      train_x <- train_x_rating_educ_name_granule
      test_x <- test_x_rating_educ_name_granule
      granule <- "E191"
    }
    if(i=="industry_granule") {
      train_x <- train_x_rating_industry_granule
      test_x <- test_x_rating_industry_granule
      granule <- "I5"
    }
    if(i=="subindustry_granule") {
      train_x <- train_x_rating_subindustry_granule
      test_x <- test_x_rating_subindustry_granule
      granule <- "I10"
    }
    
    # output vector stays the same for all granules
    train_y <- train_y_rating
    test_y <- test_y_rating
    
    # take best resampling method for all the models from above
    if(j == "model1") {
      
      # model 1: SVM
      name <- "SVM"
      train_control <- trainControl(method="repeatedcv", 
                                    number=5,
                                    repeats=5
      )
      
      grid<-expand.grid( sigma = 2^c(-12:-8),
                         C = c(0.75,1,2) # play around with values above 0.25
      )
      
      model.train <- train(x=train_x,
                        y=train_y$worker_rating,
                        trControl=train_control_raw,
                        metric="Rsquared",
                        preProcess=c("center","scale"),
                        tuneGrid=grid,
                        # tuneLength = 3,
                        method="svmRadial")
      
      predictions <- predict(model.train, test_x)
      
      
      
    } else {
      if(j == "model2") {
        
        # Model 2: extreme gradient boosting
        name <- "XGB"
        xgbdata<-cbind(train_x, train_y)
        dfr<-xgbdata
        
        sparse_matrix <- sparse.model.matrix(worker_rating~.-1, data = dfr)
        
        output_vector = dfr[,"worker_rating"] 
        
        testdata <- as.data.frame(cbind(test_x,test_y))
        test_matrix <- sparse.model.matrix(worker_rating~.-1, data = testdata)
        output_test_vector = testdata[,"worker_rating"]
        
        dtrain <- xgb.DMatrix(data = as.matrix(sparse_matrix), label=as.matrix(output_vector))
        dtest <- xgb.DMatrix(data = as.matrix(test_matrix), label=as.matrix(output_test_vector))
        
        
        watchlist <- list(train=dtrain, test=dtest)
        model.train <- xgb.train(data=dtrain,
                                 max.depth=4,
                                 eta=0.008, 
                                 nthread = 5, 
                                 nrounds=600, 
                                 watchlist=watchlist,
                                 eval.metric = "rmse")
        
        predictions <- predict(model.train, as.matrix(test_matrix))
        
        
      } 
      
      if(j == "model3") {
        
        # Model 3: Feed forward neural network
        name <- "RF"
        tunegrid  <- expand.grid(.mtry=c(25:27))
        
        model.train <- train(x = train_x,
                                 y = train_y$worker_rating,
                                 trControl = train_control_raw, 
                                 preProc = c("center", "scale"),
                                 method ="rf"  # method = "parRF", # for faster parallel 
                                 ,ntree = 30
                                 #,mtry=26
                                 #,tuneLength = 3
                                 ,tuneGrid = tunegrid
        )
        
        predictions <- predict(model.train, test_x)
        
        
      }
      
      
    }

      
    # denormalize predictions
    ratingtestData$predicted_ratings_denorm <- predictions*ratingtestData$avg_rating
    # create results
    test_rmse <- rmse(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
    test_mae <- mae(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm)
    # PCC2
    test_rsq <- as.numeric(cor(ratingtestData$worker_rating_orig, ratingtestData$predicted_ratings_denorm) ^ 2)
    
    
    # R2 calculation traditional
    squared_errors <- as.numeric((ratingtestData$worker_rating_orig - ratingtestData$predicted_ratings_denorm)^2)
    rss <- sum(squared_errors)
    squared_val <- (ratingtestData$worker_rating_orig - mean(ratingtestData$worker_rating_orig)) ^ 2
    tss <- sum(squared_val)
    R2 <- 1 - (rss/tss)
    R2
    
    # adjusted R squared
    n <- nrow(test_y)
    p <- ncol(test_y)-5 # five y- variables 
    adjRsq = 1 - ((1-test_rsq)*(n - 1)/(n - p))
    adjRsq
    adjR2 = 1 - ((1-R2)*(n - 1)/(n - p))
    adjR2
    
    if(counter==1){rating_results<-data.frame()} else {rating_results <- rating_results}
    
    rating_results[counter,1]<-as.character(paste(name,granule,sep=" "))
    
    rating_results[counter,2]<-test_rmse
    rating_results[counter,3]<-test_mae
    rating_results[counter,4]<-test_rsq
    rating_results[counter,5]<-R2
    rating_results[counter,6]<-adjRsq
    rating_results[counter,7]<-adjR2
    print(i)
    print(j)
    
    counter <- counter + 1
  }
}

rating_results

rating_granule_results <- rating_results%>%
  select(V1, V2, V3, V4, V5, V7)%>%
  rename(Model=V1,RMSE=V2, MAE=V3, PCC2=V4, R2=V5, AdjR2=V7
  )

rating_granule_results
write.csv(rating_granule_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/rating_granule_results.csv")





##### Rating Validation Results ####

# retrain the model using the combination of test and train set of ratings, such that validationset can be predicted
# use XGB with the I5 granularity
train_data <- rbind(train_x_industry_granule, test_x_industry_granule)
train_y_data <- rbind(train_y_rating, test_y_rating)

dfr<-cbind(train_data, train_y_data)
sparse_matrix <- sparse.model.matrix(worker_rating~.-1, data = dfr)
output_vector = dfr[,"worker_rating"] 

# get validationdata
col_names <- colnames(train_data)
val_set <- validationset%>%filter(is.na(worker_rating)==F)

test_x <- val_set%>%select(col_names)
test_y <- val_set%>%select("worker_rating")
testdata <- as.data.frame(cbind(test_x,test_y))
test_matrix <- sparse.model.matrix(worker_rating~.-1, data = testdata)
output_test_vector = testdata[,"worker_rating"]

dtrain <- xgb.DMatrix(data = as.matrix(sparse_matrix), label=as.matrix(output_vector))
dtest <- xgb.DMatrix(data = as.matrix(test_matrix), label=as.matrix(output_test_vector))


# xgboost using watchlist
watchlist <- list(train=dtrain, test=dtest)
final_model <- xgb.train(data=dtrain,
                         max.depth=1,
                         eta=0.008, 
                         nthread = 5, 
                         nrounds=600, 
                         watchlist=watchlist,
                         eval.metric = "rmse")


# create validation data with I5 granularity scores
validation_I5<-as.data.frame(validation_data[,c(as.numeric(grep(paste(worker_features,collapse="|"),colnames(validation_data))),
                                                    as.numeric(grep(paste(company_features,collapse="|"),colnames(validation_data))), 
                                                    as.numeric(grep(paste(jp_industry_features,collapse="|"),colnames(validation_data))),
                                                    as.numeric(grep(paste(rating_features,collapse="|"),colnames(validation_data))))])


test_y_validation <- cbind(validation_I5, validation_data$avg_rating)


# prepare prediction data
validationset$avg_rating <- validation_model_data$avg_rating
pred_set <- validationset

test_matrix <- sparse.model.matrix(worker_rating~.-1, data = validationset)
output_test_vector = testdata[,"worker_rating"]


# predict validation data with the best model and the optimal skill allocation
pred_set$worker_rating_orig <- validationset$worker_rating_orig
pred_set$predicted_ratings <- predict(final_model, as.matrix(test_x1))

# rescale predicted ratings
pred_set$pred_ratings_rescaled<-pred_set$predicted_ratings*pred_set$avg_rating


# Histogram of only rated in prediction set => looks like outcome distribution!
preeds <- pred_set%>%dplyr::filter(is.na(pred_set$worker_rating_orig)==F)
density_hist <- ggplot(preeds, aes(x=pred_ratings_rescaled)) +
  geom_histogram(binwidth=.05, colour="black", fill="white") 
density_hist

density_hist_plot_rated <- density_hist + 
  labs(x="Predicted worker ratings for observed ratings", y="Requests", size=4.5)+
  scale_y_continuous(#labels = scales::comma
    labels=function(x){format(x, big.mark = "'", scientific = FALSE)})+
  theme(panel.background = element_rect(fill = "white",
                                        colour = "white",
                                        size = 0.5, linetype = "solid"),
        panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                        colour = "white"), 
        panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                        colour = "white"),
        axis.line =  element_line(size = 0.5, linetype = 'solid',
                                  colour = "black"))

density_hist_plot_rated
# ggsave("predicted_ratings_rated_histogram.pdf", plot=density_hist_plot_rated, width=17, height= 9, units="cm", path="/Users/richardchan/Dropbox/FS19/Master Thesis/figures")

# plot all predicted requests histrogram
density_hist <- ggplot(pred_set, aes(x=pred_ratings_rescaled)) +
  geom_histogram(binwidth=.1, colour="black", fill="white") 
density_hist

density_hist_plot <- density_hist + 
  labs(x="Predicted worker rating", y="Requests", size=4.5)+
  scale_y_continuous(#labels = scales::comma
    labels=function(x){format(x, big.mark = "'", scientific = FALSE)})+
  theme(panel.background = element_rect(fill = "white",
                                        colour = "white",
                                        size = 0.5, linetype = "solid"),
        panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                        colour = "white"), 
        panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                        colour = "white"),
        axis.line =  element_line(size = 0.5, linetype = 'solid',
                                  colour = "black"))

density_hist_plot
# ggsave("predicted_ratings_histogram.pdf", plot=density_hist_plot, width=17, height= 9, units="cm", path="/Users/richardchan/Dropbox/FS19/Master Thesis/figures")

# store plots
final_hists <- grid.arrange(density_hist_plot,density_hist_plot_rated, nrow=1)
ggsave("final_hists.pdf", plot=final_hists, width=17, height= 9, units="cm", path="/Users/richardchan/Dropbox/FS19/Master Thesis/figures")



# cumulative density
cum_density <- ggplot(pred_set, aes(pred_ratings_rescaled)) + stat_ecdf(geom = "step", pad = FALSE)+ 
  labs(x="Predicted worker ratings", y="Cummulative Density", size=4.5)+
  scale_y_continuous(#labels = scales::comma
    labels=function(x){format(x, big.mark = "'", scientific = FALSE)})+
  theme(panel.background = element_rect(fill = "white",
                                        colour = "white",
                                        size = 0.5, linetype = "solid"),
        panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                        colour = "grey"), 
        panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                        colour = "white"),
        axis.line =  element_line(size = 0.5, linetype = 'solid',
                                  colour = "black"))
cum_density
ggsave("cum_density.pdf", plot=cum_density, width=17, height= 9, units="cm", path="/Users/richardchan/Dropbox/FS19/Master Thesis/figures")


# compare only rated workers
rated_preds <- pred_set%>%filter(is.na(worker_rating_orig)==F)


# create results
test_rmse <- rmse(rated_preds$worker_rating_orig, rated_preds$pred_ratings_rescaled)
test_mae <- mae(rated_preds$worker_rating_orig, rated_preds$pred_ratings_rescaled)
# PCC2
test_rsq <- as.numeric(cor(rated_preds$worker_rating_orig, rated_preds$pred_ratings_rescaled) ^ 2)
# R2 calculation traditional
squared_errors <- as.numeric((rated_preds$worker_rating_orig - rated_preds$pred_ratings_rescaled)^2)
rss <- sum(squared_errors)
squared_val <- (rated_preds$worker_rating_orig - mean(rated_preds$worker_rating_orig)) ^ 2
tss <- sum(squared_val)
R2 <- 1 - (rss/tss)
R2
# adjusted R squared
n <- nrow(test_y)
p <- ncol(test_y)-5 # five y- variables 
adjRsq = 1 - ((1-test_rsq)*(n - 1)/(n - p))
adjRsq
adjR2 = 1 - ((1-R2)*(n - 1)/(n - p))
adjR2

counter <- 1
rating_results<-data.frame()
rating_results[counter,1]<-"Final XGB"

rating_results[counter,2]<-test_rmse
rating_results[counter,3]<-test_mae
rating_results[counter,4]<-test_rsq
rating_results[counter,5]<-R2
rating_results[counter,6]<-adjRsq
rating_results[counter,7]<-adjR2

rating_results <- rating_results%>%select(V1,V2,V3,V4,V5,V7)%>%
  rename(Model=V1,RMSE=V2, MAE=V3, PCC2=V4, R2=V5, AdjR2=V7)

xgb_rating_results <- rating_results
xgb_rating_results
write.csv(xgb_rating_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/rating_validation_results.csv")



# These results here are not representative enough!

# denormalize predicted ratings
validationset$pred_ratings_rescaled
validation_model_data$worker_rating
test_y_validation$ratingtestData$predicted_ratings_denorm <- test_y_validation$predicted_ratings*test_y_validation$avg_rating
validationset$worker_rating
# validation error measures for continuous variables
# only for those that can be measured

val_data <- as.data.frame(cbind(validationset$pred_ratings_rescaled,
                 validation_model_data$worker_rating))%>%rename(predicted_ratings=V1, worker_rating=V2)
val_set <- val_data%>%filter(is.na(worker_rating)==F)
validation_rmse <- rmse(val_set$worker_rating, val_set$predicted_ratings)
validation_mae <- mae(val_set$worker_rating, val_set$predicted_ratings)
validation_rsq <- as.numeric(cor(val_set$worker_rating, val_set$predicted_ratings) ^ 2)



# measure algo improvements

resultset <- val_data%>%rename(predicted_ratings_denorm=predicted_ratings)
threshold <- 2
#summary(resultset%>%filter(worker_rating>0)%>%select(predicted_ratings_denorm))
all_requests <- nrow(resultset)
requests_below <- nrow(resultset%>%filter(predicted_ratings_denorm <= threshold))
percentage_saved <- requests_below/all_requests

rated_below <- nrow(resultset%>%filter(worker_rating>0 & predicted_ratings_denorm <= threshold))
percentage_rated_below <- rated_below/requests_below

false_rated_below <- nrow(resultset%>%filter(worker_rating>threshold & predicted_ratings_denorm <= threshold))
false_rated_rate <- false_rated_below/rated_below

# measure per job
perjobval <- cbind(resultset, validation_model_data$wa_id)
perjobval<-perjobval%>%rename(wa_id="validation_model_data$wa_id")
perjob_results <- perjobval%>%select(wa_id, worker_rating, predicted_ratings_denorm)%>%
  group_by(wa_id)%>%
  summarise(
    avg_improvements = count(predited_ratings%>%filter(predicted_ratings > threshold))/count(predicted_ratings),
    avg_error = count(predicted_ratings%>%filter(predicted_ratings < threshold & worker_rating > threshold))
  )

# per job: compared to above, how many requests saved with threshold of .5?
validationset$val.perjob.preds<- ifelse (validationset$validation_preds > 0.5, 1, 0)
perjob.results.to.5 <- validationset%>%
  select(wa_id, accepted, val.perjob.preds, accepted)%>%
  mutate(falseneg = ifelse(val.perjob.preds==0&accepted=="YES",1,0),
         trueneg = ifelse(val.perjob.preds==0&accepted=="NO",1,0))%>%
  group_by(wa_id)%>%
  summarise(requests=n(),predrequ=sum(val.perjob.preds),sumfalseneg=sum(falseneg),
            savedrequests=sum(falseneg+trueneg),
            general.improvement = (1-sum(val.perjob.preds)/n())*100,
            falsenegrate = ifelse(is.na(sum(falseneg)/sum(falseneg+trueneg))==TRUE, 100, sum(falseneg)/sum(falseneg+trueneg)*100))%>%
  ungroup()%>%summarise(avg.general.improvement=mean(general.improvement), avg.falsenegrate=mean(falsenegrate))

# test<-validationset%>%filter(wa_id=="0148c7cd-6561-4c51-84a3-f68b4fe93d24")



# per job: what is the avg threshold to not exclude any accepted and how many request can be saved 
# first, find out what is the avg. threshold for which no one accepted was predicted to be not requested
b <- validationset%>%filter(accepted=="YES")%>%group_by(wa_id)%>%summarise(min.prob=min(validation_preds))%>%ungroup()%>%summarise(mean=mean(min.prob))
# then predict the results with new threshold
validationset$val.perjob.preds<- ifelse (validationset$validation_preds > b$mean, 1, 0)
perjob.results.best.to <- validationset%>%
  select(wa_id, accepted, val.perjob.preds, accepted)%>%
  mutate(falseneg = ifelse(val.perjob.preds==0&accepted=="YES",1,0),
         trueneg = ifelse(val.perjob.preds==0&accepted=="NO",1,0))%>%
  group_by(wa_id)%>%
  summarise(requests=n(),predrequ=sum(val.perjob.preds),sumfalseneg=sum(falseneg),
            savedrequests=sum(falseneg+trueneg),
            general.improvement = (1-sum(val.perjob.preds)/n())*100,
            falsenegrate = ifelse(is.na(sum(falseneg)/sum(falseneg+trueneg))==TRUE, 100, sum(falseneg)/sum(falseneg+trueneg)*100))%>%
  ungroup()%>%summarise(avg.general.improvement=mean(general.improvement), avg.falsenegrate=mean(falsenegrate))
val.pj.preds<- ifelse (validationset$validation_preds > b$mean, 1, 0)
val.pj.ref  <- as.numeric(validationset$accepted)-1
val.pj.matrix<- caret::confusionMatrix(table(val.pj.preds, val.pj.ref), positive = "1")
val.pj.matrix

# not representative!!
validation_results$RMSE <- validation_rmse
validation_results$MAE <- validation_mae
validation_results$R2 <- validation_rsq
validation_results$general_improvement <-percentage_saved
validation_results$general_upper_fnegrate <-false_rated_rate
validation_results$general_lower_fnegrate <-percentage_rated_below

validation_results$avg_perjob_improvemet <- perjob_results$avg.general.improvement
validation_results$avg_perjob_fnegrate <- perjob_results$avg.falsenegrate
validation_results$perjob_upperth_improvement <- perjob.results.to.5$avg.falsenegrate
validation_results$perjob_upper_fnegrate <- perjob.results.to.5$avg.general.improvement
validation_results$perjob_lowerth_improvement <- val.pj.matrix$byClass['Specificity']
validation_results$perjob_lower_fnegrate <- as.numeric((val.pj.matrix$table[1,2]+val.pj.matrix$table[2,2])/(val.pj.matrix$table[1,2]+val.pj.matrix$table[2,2]+val.pj.matrix$table[1,1]+val.pj.matrix$table[2,1]))


validation_results
write.csv(validation_results, "/Users/richardchan/Dropbox/FS19/Master Thesis/figures/rating_validation_results.csv")


# Thanks for reading! 

#######################################

# Hey!
# Thanks for being curious about the last piece of code
# this is for convenience only
# save and load sessions at any time needed
save.image(file='yoursession.RData')
load('yoursession.RData')
