# Title: "Project: Plant Classification"
# Authors: David Swinkels (920714820090) and Ping Zhou (900405987130)

#--------------------------preparation---------------------------------------------

## Installing Packages if necessary
list.of.packages <- c("MASS", "class","tree","randomForest","gbm","e1071")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

## Loading libraries (MASS, tree, randomforest,gbm)
library(MASS)
library(class)
library(tree)
library(randomForest)
library(gbm)
library(e1071)

# Clear console, workspace, and plots:
cat("\014")
rm(list=ls())
graphics.off()

## Set Filenames for Reading data
filenameTrain = c('training_data/training_data-K_010.csv','training_data/training_data-K_020.csv','training_data/training_data-K_040.csv','training_data/training_data-K_080.csv','training_data/training_data-K_160.csv')
filenameTest = c('test_data/test_data-K_010.csv','test_data/test_data-K_020.csv','test_data/test_data-K_040.csv','test_data/test_data-K_080.csv','test_data/test_data-K_160.csv')
filenameVali = c('validation_data/validation_data-K_010.csv','validation_data/validation_data-K_020.csv','validation_data/validation_data-K_040.csv','validation_data/validation_data-K_080.csv','validation_data/validation_data-K_160.csv') 

## For-loop to loop through all K-values
for (k in 1:5){
  print(paste("Starting with Dataset:",filenameTrain[k]))
  ## Loading Data
  dataTrainRead = read.csv(file=filenameTrain[k])[, -1]
  dataTestRead = read.csv(file=filenameTest[k])[, -1]
  dataValidateRead = read.csv(file=filenameVali[k])[, -1]
  
  ## Creating Empty Test Error lists
  GLMFul = c()
  GLMFor = c()
  GLMBac = c()
  
  LDAFul = c()
  LDAFor = c()
  LDABac = c()
  
  # QDAFul = c()
  QDAFor = c()
  QDABac = c()
  
  KNNFul = c()
  KNNFor = c()
  KNNBac = c()
  
  TREFul = c()
  TREPru = c()
  TRERaF = c()
  TREGBM = c()
  
  SVMLin = c()
  SVMPol = c()
  SVMRad = c()
  
  ## Creating Empty Training Error lists
  TrainGLMFul=c()
  TrainGLMFor=c()
  TrainGLMBac=c()
  
  TrainLDAFul=c()
  TrainLDAFor=c()
  TrainLDABac=c()
  
  # TrainQDAFul = c()
  TrainQDAFor=c()
  TrainQDABac=c()
  
  TrainKNNFul=c()
  TrainKNNFor=c()
  TrainKNNBac=c()
  
  TrainTREFul=c()
  TrainTREPru=c()
  TrainTRERaF=c()
  TrainTREGBM=c()
  
  TrainSVMLin=c()
  TrainSVMPol=c()
  TrainSVMRad=c()
  
  
  ## Setup For-Loop to Test Classification Methods for different random samples
  for (i in 1:2){
    print(paste("Starting with Random Sample:",as.character(i)))
    ## Create Random Sample
    set.seed(i)
    test = sample(dim(dataValidateRead)[1],dim(dataValidateRead)[1]-65)
    dataTest = rbind(dataValidateRead[test,],dataTestRead[-test,])
    dataTest = dataTest[order(dataTest$Class),]
    dataTrain = dataTrainRead
    
    ##-----------------------------------------------------------------------------------------
    ## Logistic Regression
    ##-----------------------------------------------------------------------------------------
    
    fullmod=suppressWarnings(glm( Class ~ ., data = dataTrain , family = binomial))
    nothing=suppressWarnings(glm( Class ~ 1, data = dataTrain , family = binomial))
    
    backwards = suppressWarnings(step(fullmod,trace=0))
    forwards = suppressWarnings(step(nothing,scope=list(lower=formula(nothing),upper=formula(fullmod)), direction="forward", trace=0))
    
    bestformod=suppressWarnings(glm(formula(backwards), data = dataTrain, family = binomial))
    bestbackmod=suppressWarnings(glm(formula(forwards), data = dataTrain, family = binomial))
    
    ## Training Error GLM Full Model
    glm.probs=predict(fullmod,type ="response")
    glm.pred=rep("p",dim(dataTrain)[1])
    glm.pred[glm.probs>=.5]="s"
    TrainGLMFul[i]= 1 - mean(glm.pred==dataTrain$Class)
    
    ## Test Error GLM Full Model
    glm.probs=suppressWarnings(predict(fullmod,newdata=dataTest,type ="response"))
    glm.pred=rep("p",dim(dataTest)[1])
    glm.pred[glm.probs>=.5]="s"
    GLMFul[i] = 1 - mean(glm.pred==dataTest$Class)
    
    ## Training Error GLM Forwards Model
    glm.probs=predict(bestformod,type ="response")
    glm.pred=rep("p",dim(dataTrain)[1])
    glm.pred[glm.probs>=.5]="s"
    TrainGLMFor[i]= 1 - mean(glm.pred==dataTrain$Class)
    
    ## Test Error GLM Forwards Model
    glm.probs=suppressWarnings(predict(bestformod,newdata=dataTest,type ="response"))
    glm.pred=rep("p",dim(dataTest)[1])
    glm.pred[glm.probs>=.5]="s"
    GLMFor[i] = 1 - mean(glm.pred==dataTest$Class)
    
    ## Training Error GLM Backwards Model
    glm.probs=predict(bestbackmod,type ="response")
    glm.pred=rep("p",dim(dataTrain)[1])
    glm.pred[glm.probs>=.5]="s"
    TrainGLMBac[i]= 1- mean(glm.pred==dataTrain$Class)
    
    ## Test Error GLM Backwards Model
    glm.probs=suppressWarnings(predict(bestbackmod,newdata=dataTest,type ="response"))
    glm.pred=rep("p",dim(dataTest)[1])
    glm.pred[glm.probs>=.5]="s"
    GLMBac[i] = 1 - mean(glm.pred==dataTest$Class)
    
    ##-----------------------------------------------------------------------------------------
    ## Linear Discriminant Analysis
    ##-----------------------------------------------------------------------------------------
    
    ## Test Error LDA Full Model
    lda.fit=lda(formula(fullmod),data=dataTrain)
    lda.pred=predict(lda.fit,newdata=dataTest)
    lda.class=lda.pred$class
    LDAFul[i] = 1 - mean(lda.class==dataTest$Class)
    
    ## Training Error LDA Full Model
    lda.pred=predict(lda.fit,type="response")
    TrainLDAFul[i]= 1- mean(lda.pred$class==dataTrain$Class)
    
    ## Test Error LDA Forward Model
    lda.fit=lda(formula(bestformod),data=dataTrain)
    lda.pred=predict(lda.fit,newdata=dataTest)
    LDAFor[i] = 1 - mean(lda.pred$class==dataTest$Class)
    
    ## Training Error LDA Forward Model
    lda.pred=predict(lda.fit,type="response")
    TrainLDAFor[i]= 1- mean(lda.pred$class==dataTrain$Class)
    
    ## Test Error LDA Backward Model
    lda.fit=lda(formula(bestbackmod),data=dataTrain)
    lda.pred=predict(lda.fit,newdata=dataTest)
    lda.class=lda.pred$class
    LDABac[i] = 1 - mean(lda.class==dataTest$Class)
    
    ## Training Error LDA Backward Model
    lda.pred=predict(lda.fit,type="response")
    TrainLDABac[i]= 1- mean(lda.pred$class==dataTrain$Class)
    
    ##-----------------------------------------------------------------------------------------
    ## Quadratic Discriminant Analysis
    ##-----------------------------------------------------------------------------------------
    # ## Test Error QDA Full Model
    # As QDA only works for data that has fewer predictors than observations.In this case, only 135
    # observations are available while the number of predictors could reach 160 at most, so we choose 
    # not to use QDA FULL here.
    
    ## Test Error QDA Forward Model
    qda.fit=qda(formula(bestformod),data=dataTrain)
    qda.pred=predict(qda.fit,newdata=dataTest)
    QDAFor[i] = 1 - mean(qda.pred$class==dataTest$Class)
    
    ## Training Error QDA Forward Model
    qda.pred=predict(qda.fit,type="response")
    TrainQDAFor[i]= 1- mean(qda.pred$class==dataTrain$Class)
    
    ## Test Error QDA Backward Model
    qda.fit=qda(formula(bestbackmod),data=dataTrain)
    qda.pred=predict(qda.fit,newdata=dataTest)
    QDABac[i] = 1 - mean(qda.pred$class==dataTest$Class)
    
    ## Training Error QDA Backward Model
    qda.pred=predict(qda.fit,type="response")
    TrainQDABac[i]= 1- mean(qda.pred$class==dataTrain$Class)
    
    ##-----------------------------------------------------------------------------------------
    ## K-Nearest Neighbors
    ##-----------------------------------------------------------------------------------------
    ## Test and Train Error KNN Full Model
    range = 1:(dim(dataTrain)[1]-1)
    result=data.frame(range,ncol=0,train=0)
    colnames(result)<-c("Kvalue","TrainError","TestError")
    train.X=dataTrain[,-1]
    test.X=dataTest[,-1]
    
    for(kvalue in range){
      set.seed(1)
      knn.predTrain=knn(train.X,train.X,dataTrain$Class,k=kvalue)
      knn.predTest=knn(train.X,test.X,dataTrain$Class,k=kvalue)
      result[kvalue,]=c(kvalue,1-mean(knn.predTrain==dataTrain$Class),1-mean(knn.predTest==dataTest$Class))
    }
    KNNFul[i]=min(result$TestError)
    TrainKNNFul[i]= result$TrainError[which.min(result$TestError)]
    
    ## Test and Train Error KNN Forwards Model
    range = 1:(dim(dataTrain)[1]-1)
    result=data.frame(range,ncol=0,train=0)
    colnames(result)<-c("Kvalue","TrainError","TestError")
    
    # Use predictors we got from the bestformod
    train.X=dataTrain[,-1]
    train.X=train.X[names(bestformod$coefficients)[-1]]
    test.X=dataTest[,-1]
    test.X=test.X[names(bestformod$coefficients)[-1]]
    
    for(kvalue in range){
      set.seed(1)
      knn.predTrain=knn(train.X,train.X,dataTrain$Class,k=kvalue)
      knn.predTest=knn(train.X,test.X,dataTrain$Class,k=kvalue)
      result[kvalue,]=c(kvalue,1-mean(knn.predTrain==dataTrain$Class),1-mean(knn.predTest==dataTest$Class))
    }
    KNNFor[i]=min(result$TestError)
    TrainKNNFor[i]= result$TrainError[which.min(result$TestError)]
    
    ## Test and Train Error KNN Backwards Model
    range = 1:(dim(dataTrain)[1]-1)
    result=data.frame(range,ncol=0,train=0)
    colnames(result)<-c("Kvalue","TrainError","TestError")
    
    # Use predictors we got from the bestbackmod
    train.X=dataTrain[,-1]
    train.X=train.X[names(bestbackmod$coefficients)[-1]]
    test.X=dataTest[,-1]
    test.X=test.X[names(bestbackmod$coefficients)[-1]]
    
    for(kvalue in range){
      set.seed(1)
      knn.predTrain=knn(train.X,train.X,dataTrain$Class,k=kvalue)
      knn.predTest=knn(train.X,test.X,dataTrain$Class,k=kvalue)
      result[kvalue,]=c(kvalue,1-mean(knn.predTrain==dataTrain$Class),1-mean(knn.predTest==dataTest$Class))
    }
    KNNBac[i]=min(result$TestError)
    TrainKNNBac[i]= result$TrainError[which.min(result$TestError)]   
    
    ##-----------------------------------------------------------------------------------------
    ## Classification Trees
    ##-----------------------------------------------------------------------------------------
    
    ## Test Error Trees Full Model
    tree.plant=tree(Class~.,data = dataTrain)
    tree.pred = predict(tree.plant, newdata = dataTest, type="class")
    TREFul[i] = 1 - mean(tree.pred==dataTest$Class)
    
    ## Train Error Trees Full Model
    tree.plant=tree(Class~.,data = dataTrain)
    tree.pred = predict(tree.plant, type = "class")
    TrainTREFul[i] = 1 - mean(tree.pred==dataTest$Class)
    
    ## Test Error Pruned Tree
    tree.plant=tree(Class~.,data = dataTrain)
    cv.plant = cv.tree(tree.plant,FUN=prune.misclass)
    numnodes = cv.plant$size[which.min(cv.plant$dev)]
    prune.plant=prune.misclass(tree.plant,best=numnodes)
    tree.pred = predict(prune.plant,newdata=dataTest,type="class")
    TREPru[i] = 1 - mean(tree.pred==dataTest$Class)
    
    ## Train Error Pruned Tree
    tree.plant=tree(Class~.,data = dataTrain)
    cv.plant = cv.tree(tree.plant,FUN=prune.misclass)
    numnodes = cv.plant$size[which.min(cv.plant$dev)]
    prune.plant=prune.misclass(tree.plant,best=numnodes)
    tree.pred = predict(prune.plant, type="class")
    TrainTREPru[i] = 1 - mean(tree.pred==dataTest$Class)
    
    ## Test Error RandomForest Tree
    result=data.frame(1:50,train=0)
    colnames(result)<-c("TestError","TrainError")
    for (ntree in 1:50){
      set.seed(1)
      bag.plant =randomForest(Class~.,data=dataTrain,mtry=sqrt(dim(dataTrain)[2]-1),ntree=ntree,importance=TRUE)
      yhat.bag=predict(bag.plant,newdata=dataTest)
      yhat.bag2=predict(bag.plant,newdata=dataTrain)
      result[ntree]=c(1-mean(yhat.bag==dataTest$Class),1-mean(yhat.bag2==dataTest$Class))
    }
    TRERaF[i] = min(result$TestError)
    TrainTRERaF[i] = min(result$TrainError)
    
    ## Convert Dataset
    dataTrain2 = dataTrain
    dataTrain2$Class01[dataTrain$Class %in% "p"]<-1
    dataTrain2$Class01[dataTrain$Class %in% "s"]<-0
    dataTrain2=dataTrain2[,-1]
    
    dataTest2 = dataTest
    dataTest2$Class01[dataTest$Class %in% "p"]<-1
    dataTest2$Class01[dataTest$Class %in% "s"]<-0
    dataTest2=dataTest2[,-1]
    
    ## Test Error Boosted Classification Tree
    set.seed(1)
    boost.plant=gbm(Class01~.,data=dataTrain2,distribution="bernoulli",n.trees=5000,cv.folds = 5)
    best.iter = gbm.perf(boost.plant,method="cv",plot.it=FALSE)
    f.predict = predict(boost.plant,newdata=dataTest2,best.iter,type="response")
    
    predictResult=rep(0,dim(dataTest2)[1])
    predictResult[f.predict>=0.5]<-1
    predictResult[f.predict<0.5]<-0
    TREGBM[i] = 1 - mean(predictResult==dataTest2$Class01)
    
    ## Test Error Boosted Classification Tree
    set.seed(1)
    boost.plant=gbm(Class01~.,data=dataTrain2,distribution="bernoulli",n.trees=5000,cv.folds = 5)
    best.iter = gbm.perf(boost.plant,method="cv",plot.it=FALSE)
    f.predict = predict(boost.plant,newdata=dataTrain2,best.iter,type="response")
    predictResult=rep(0,dim(dataTrain2)[1])
    predictResult[f.predict>=0.5]<-1
    predictResult[f.predict<0.5]<-0
    TrainTREGBM[i] = 1 - mean(predictResult==dataTest2$Class01)
    
    ##-----------------------------------------------------------------------------------------
    ## Support Vector Machines
    ##-----------------------------------------------------------------------------------------
    ## Parameters options of SVM's were limited to reduce computation time
    ## If only SVM is computed, more parameters can be used to tune the classifier even better
    costlist=c(0.001,0.01,0.1,1,10,100,1000)
    gammalist=c(0.001,0.01,0.1,1)
    degreelist=c(2,3,4,5)
    
    ## Test Error Linear Kernel
    tune.out=tune(svm,Class~.,data=dataTrain,kernel="linear",ranges=list(cost=costlist))
    confmatrix = table(true=dataTest$Class,pred=predict(tune.out$best.model,newx=dataTest))
    TrainSVMLin[i] = tune.out$best.performance
    SVMLin[i] = 1 - (confmatrix[1,1]+confmatrix[2,2])/sum(confmatrix)
    
    ## Test Error Polynomial Kernel
    tune.out=tune(svm,Class~.,data=dataTrain,kernel="polynomial",ranges=list(cost=costlist,degree=degreelist))
    confmatrix = table(true=dataTest$Class,pred=predict(tune.out$best.model,newx=dataTest))
    TrainSVMPol[i] = tune.out$best.performance
    SVMPol[i] = 1 - (confmatrix[1,1]+confmatrix[2,2])/sum(confmatrix)
    
    ## Test Error Radial Kernel
    tune.out=tune(svm,Class~.,data=dataTrain,kernel="radial",ranges=list(cost=costlist,gamma=gammalist))
    confmatrix = table(true=dataTest$Class,pred=predict(tune.out$best.model,newx=dataTest))
    TrainSVMRad[i] = tune.out$best.performance
    SVMRad[i] = 1 - (confmatrix[1,1]+confmatrix[2,2])/sum(confmatrix)
    
    TrainErrorList = data.frame(TrainGLMFul,TrainGLMFor,TrainGLMBac,TrainLDAFul,TrainLDAFor,TrainLDABac,TrainQDAFor,TrainQDABac,TrainKNNFul,TrainKNNFor,TrainKNNBac,TrainTREFul,TrainTREPru,TrainTRERaF,TrainTREGBM,TrainSVMLin,TrainSVMPol,TrainSVMRad)
    TestErrorList = data.frame(GLMFul,GLMFor,GLMBac,LDAFul,LDAFor,LDABac,QDAFor,QDABac,KNNFul,KNNFor,KNNBac,TREFul,TREPru,TRERaF,TREGBM,SVMLin,SVMPol,SVMRad)
    
  }
  filenamelistTrain=c("TrainErrorK010.csv","TrainErrorK020.csv","TrainErrorK040.csv","TrainErrorK080.csv","TrainErrorK160.csv")
  filenamelistTest=c("TestErrorK010.csv","TestErrorK020.csv","TestErrorK040.csv","TestErrorK080.csv","TestErrorK160.csv")
  write.csv2(TrainErrorList,file=filenamelistTrain[k],row.names=FALSE)
  write.csv2(TestErrorList,file=filenamelistTest[k],row.names=FALSE)
  
}




#-------Boxplot Test Error Rate against methods for 5 different datasets---------------
# Read Test Error Rate Files into Memory 
TestErrorK010 = read.csv2("TestErrorK010.csv")
TestErrorK020 = read.csv2("TestErrorK020.csv")
TestErrorK040 = read.csv2("TestErrorK040.csv")
TestErrorK080 = read.csv2("TestErrorK080.csv")
TestErrorK160 = read.csv2("TestErrorK160.csv")

TrainErrorK010 = read.csv2("TrainErrorK010.csv")
TrainErrorK020 = read.csv2("TrainErrorK020.csv")
TrainErrorK040 = read.csv2("TrainErrorK040.csv")
TrainErrorK080 = read.csv2("TrainErrorK080.csv")
TrainErrorK160 = read.csv2("TrainErrorK160.csv")

# Generate a color vector for 5 different datasets(10,20,40,80,160 variables)
color_vec <- c(rgb(8,81,156,maxColorValue = 255),rgb(26,150,65,maxColorValue = 255),rgb(150,150,150,maxColorValue = 255),rgb(253,174,97,maxColorValue = 255),rgb(215,25,28,maxColorValue = 255))

# Boxplot Test Errors For Classification Methods
par(mfrow = c(3,2))
boxplot(TestErrorK010,las=2,ylab="Test Error", col = color_vec[1], main = "K10",cex.main=1.6,cex.lab=1.3, cex.axis=1.1)
boxplot(TestErrorK020,las=2,ylab="Test Error", col = color_vec[2], main = "K20",cex.main=1.6,cex.lab=1.3, cex.axis=1.1)
boxplot(TestErrorK040,las=2,ylab="Test Error", col = color_vec[3], main = "K40",cex.main=1.6,cex.lab=1.3, cex.axis=1.1)
boxplot(TestErrorK080,las=2,ylab="Test Error", col = color_vec[4], main = "K80",cex.main=1.6,cex.lab=1.3, cex.axis=1.1)
boxplot(TestErrorK160,las=2,ylab="Test Error", col = color_vec[5], main = "K160",cex.main=1.6,cex.lab=1.3, cex.axis=1.1)

#---------Plot overfitting and underfitting between test and train error rate-------------------


#Only the first row is used in this case
#Make a data frame 
train_test.K010 <- (TrainErrorK010[1,] - TestErrorK010[1,])
train_test.K020 <- (TrainErrorK020[1,] - TestErrorK020[1,])
train_test.K040 <- (TrainErrorK040[1,] - TestErrorK040[1,])
train_test.K080 <- (TrainErrorK080[1,] - TestErrorK080[1,])
train_test.K160 <- (TrainErrorK160[1,] - TestErrorK160[1,])

train_test <- rbind(train_test.K010, train_test.K020,train_test.K040,train_test.K080,train_test.K160)
row.names(train_test) <- c("K010","K020","K040","K080","K160")

# Make the column names shorter
methods_name <- colnames(train_test)
sub_name <- gsub('.*Train', '', methods_name)
colnames(train_test) <- sub_name

# Plot Overview Fitting Models
plot(x = as.factor(colnames(train_test)), y =  train_test[1,],col = color_vec[1], pch = 16,xaxt ="n",xlab="", ylab = "Train Error - Test Error", ylim = range(-0.5,0.5,0.1), main = "Comparison of Classification Methods for Model Fits")
points(x = as.factor(colnames(train_test)), y =  train_test[2,], col = color_vec[2], pch = 16,cex = 1)
points(x = as.factor(colnames(train_test)), y =  train_test[3,], col = color_vec[3], pch = 16)
points(x = as.factor(colnames(train_test)), y =  train_test[4,], col = color_vec[4], pch = 16)
points(x = as.factor(colnames(train_test)), y =  train_test[5,], col = color_vec[5], pch = 16)
axis(side=1, at=1:18, labels = colnames(train_test), las=2)
abline(h = 0.0)
legend("top",legend = rownames(train_test),col = color_vec, pch=16, cex = 1,horiz = T, title = "Number of Variables")



