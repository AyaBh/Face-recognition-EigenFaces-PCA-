classify <- function(image) {
  # Predicts the person in the database that appears in the image
  # using PCA and kNN
  #
  # Args:
  #   image: character string of value in the working directory
  #
  # Returns:
  #   result: ID of the person in the database
  
  load(file = "eigenfaces.RData")
  
  test <- as.vector(readImage(image))
  
  train.scaled <- data - pca_train$mean
  cum.var  <- cumsum(pca_train$D)
  threshold    <- min(which(cum.var > 0.95))
  
  eigenfaces <- t(train.scaled)%*%pca_train$P[,1:threshold]
  
  train.projection <- train.scaled%*%eigenfaces
  
  test.scaled <- as.matrix(test) - pca_train$mean
  test.projection <- t(test.scaled)%*%eigenfaces
  
  dmatrix <- dist(rbind(test.projection,train.projection),
                  method = "manhattan", diag = TRUE, upper = TRUE)
  dmatrix <- as.matrix(dmatrix)
  dmatrix <- dmatrix[1,2:(nrow(train.projection)+1)]
  sorted <- sort(dmatrix,index.return=TRUE,decreasing=FALSE)
  
  k <- 6
  labels_sel <- labels[sorted$ix][1:k]
  if (sorted$x[1] < 53827.58) {
    uniqv <- unique(labels_sel)
    #If they are two values with the same frecuency, it returs the first
    result <- uniqv[which.max(tabulate(match(labels_sel, uniqv)))]
  } else {
    result <- 0
  }
  
  return(result)
}

classifyD <- function(image) {
  # Predicts the person in the database that appears in the image
  # using kNN
  #
  # Args:
  #   image: character string of value in the working directory
  #
  # Returns:
  #   result: ID of the person in the database
  
  load(file = "eigenfaces.RData")
  
  test <- as.vector(readImage(image))
  
  dmatrix <- dist(rbind(t(as.matrix(test)), data),
                  method = "manhattan", diag = TRUE, upper = TRUE)
  dmatrix <- as.matrix(dmatrix)
  dmatrix <- dmatrix[1,2:(nrow(data)+1)]
  sorted <- sort(dmatrix,index.return=TRUE,decreasing=FALSE)
  
  k <- 6
  labels_sel <- labels[sorted$ix][1:k]
  uniqv <- unique(labels_sel)
  #If they are two values with the same frecuency, it returs the first
  result <- uniqv[which.max(tabulate(match(labels_sel, uniqv)))]
  
  return(result)
}
