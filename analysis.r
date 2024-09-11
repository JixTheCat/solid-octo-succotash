library(randomForest)
library(mlbench)
library(caret)
library(rpart.plot)

# The csv file used is the state of the dataframe used in
#  the python analysis. After variables are made multiclass and the limit of 10 classes is used.
df <- read.csv("dftree.csv")
df[, "Profitable"] <- factor(df[, "Profitable"])
df[, "Water.Source"] <- factor(df[, "Water.Source"])
df[, "Cover.Crop.Type"] <- factor(df[, "Cover.Crop.Type"])
df[, "Irrigation.System.Type"] <- factor(df[, "Irrigation.System.Type"])
df[, "Irrigation.Energy.Type"] <- factor(df[, "Irrigation.Energy.Type"])
df[, "Year"] <- factor(df[, "Year"])
df[, "Region"] <- factor(df[, "Region"])
df[, "Disease.Presence"] <- factor(df[, "Disease.Presence"])

# Training function

trControl <- trainControl(
    method = "repeatedcv"
    , number = 10
    , repeats = 10
    , verboseIter = TRUE
)

###########################
#       GI region         #
###########################

model_tree <- train(Region ~ .
    , data = df[, colnames(df)[
        !(colnames(df) %in% c(
            "Profit",
            "Profitable",
            "Operating.Costs"))]]
    , method = "rpart"
    , metric = "Accuracy"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)

pdf("region.pdf")
rpart.plot(model_tree$finalModel
    , extra = 102
    , legend.x = -100
    , box.palette = "auto"
)
dev.off()

sink("region.txt")
print(model_tree)
sink()
###########################
#       Year              #
###########################

model_tree <- train(Year ~ .
    , data = df[, colnames(df)[
        !(colnames(df) %in% c(
            "Profit",
            "Profitable",
            "Operating.Costs"))]]
    , method = "rpart"
    , metric = "Accuracy"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)

pdf("year.pdf")
rpart.plot(model_tree$finalModel
    , extra = 102
    , legend.x = -100
    , box.palette = "auto"
)
dev.off()


sink("year.txt")
print(model_tree)
sink()
#########################
#       profitable      #
#########################

model_tree <- train(Profitable ~ .
    , data = df[, colnames(df)[
        !(colnames(df) %in% c(
            "Profit",
            "Operating.Costs"))]]
    , method = "rpart"
    , metric = "Accuracy"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)

pdf("profitable.pdf")
rpart.plot(model_tree$finalModel
    , extra = 102
    , legend.x = -100
    , box.palette = "auto"
)
dev.off()

sink("profitable.txt")
print(model_tree)
sink()
#####################
#       profit      #
#####################

model_tree <- train(Profit ~ .
    , data = df[, colnames(df)[
        !(colnames(df) %in% c(
            "Profitable",
            "Operating.Costs"))]]
    , method = "rpart"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)

# Each node shows
# - the predicted value (the average value of the response).
# Note that each terminating node is a regression model
# - the percentage of observations in the node

pdf("profit.pdf")
rpart.plot(model_tree$finalModel
    , legend.x = -100
    , box.palette = "auto"
)
dev.off()

sink("profit.txt")
print(model_tree)
sink()
#############################
#       operating costs     #
#############################

model_tree <- train(Operating.Costs ~ .
    , data = df[, colnames(df)[
        !(colnames(df) %in% c(
            "Profit",
            "Profitable"))]]
    , method = "rpart"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)

# Each node shows
# - the predicted value (the average value of the response).
# Note that each terminating node is a regression model
# - the percentage of observations in the node

pdf("operating_costs.pdf")
rpart.plot(model_tree$finalModel
    , legend.x = -100
    , box.palette = "auto"
)
dev.off()

sink("operating_costs.txt")
print(model_tree)
sink()
########################

######################
#       Revenue      #
######################

model_tree <- train(Revenue ~ .
    , data = df[, colnames(df)[
        !(colnames(df) %in% c(
            "Profitable",
            "Profit",
            "Operating.Costs"))]]
    , method = "rpart"
    , trControl = trControl
    , na.action = na.omit
    , tuneLength = 10
)

# Each node shows
# - the predicted value (the average value of the response).
# Note that each terminating node is a regression model
# - the percentage of observations in the node

pdf("revenue.pdf")
rpart.plot(model_tree$finalModel
    , legend.x = -100
    , box.palette = "auto"
)
dev.off()

sink("revenue.txt")
print(model_tree)
sink()