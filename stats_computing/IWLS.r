################################################################
   ################################################################
   #
   # This function fits an exponential regression model using the
   # using the IWLS algorithm.
   #
   # This function expects a vector/matrix of size n x p containing
   # the covariates and a vector with the responses of size n.
   # Optionally one cann specifiy a vector with initial coefficient
   # (startval) values of size p. If nothing is specified it uses
   # the default initialisation of 0. Further, one can specify whether
   # diagnositic plots should be created when the function returns.
   # This is done trough the 'plot' flag. By default plot = TRUE.
   #
   # Function logic:
   #
   #  1. First the function makes sure that the input data is of
   #     the right data type.
   #  2. Retrieving shape of the input data.
   #  3. Inital error handling block that controlls for faulty
   #     input data.
   #  4. IWLS procedure.
   #  5. Computation of diagnostic metrics.
   #  6. Plotting the diagnostic plots.
   #  7. Bundeling output and return function.
   #
   ################################################################
   ################################################################


   erm <- function(y, X, startval=rep(0, dim(X)[2]), plot=TRUE) {

      # Making sure that the input values are of the right data type.
      X <- as.matrix(X)
      y <- as.vector(y)

      # Retriving dimentions of covariates and response variables.
      x_dim <- dim(X)
      y_len <- length(y)

      ###################### Error Handling ######################
      #
      # Check that y and X have compatible dimentions and throwing
      # an error if not.

      if(x_dim[1] != y_len){
         stop("input shapes are not equal. ")
      }
      # Checking whether missing values are in the supplied data set.
      else if(sum(is.nan(y)) > 0 | sum(is.nan(X)) > 0) {
         stop("NaN found in the repsonse variabel.")
      }
      # Checking whether the support regions of y confrim to the ones
      # of an exponential distribution.
      else if(sum(y < 0 ) > 0) {
         stop("y value not within the support region of an exponential distribution.")
      }

      # Populating the intial betahat vector with the startvalues supplied
      # in the function argument.
      betahat <- startval

      # Store an inital value in the score vector variable U in order to
      # enter the while loop.
      U <- 10

      # To keep track of the iterations we create a seperate counter
      # variable that stores the value of the current iteration.
      iter <- 0

      ###################### IWLS Procedure ######################

      # Start while loop.
      while(any(abs(U)>1e-6)){

         # Compute the vector eta which is simply the product of the
         # beta hats and the explanatory variables.
         eta <- as.vector(X%*%betahat)

         # Because we are using the exponential link function, we
         # can compute the mean by exponentiating -eta.
         mu <- exp(-eta)

         # The variance is simply the square of the mean. Please
         # see report for a more detailed description.
         V <- mu^2

         # Computing the diagonal W matrix.
         W <- mu^2/V

         # Calulating the update step vector z.
         z <- eta + ((y-mu)/-mu)

         XW <- t(W*X)
         XWX <- solve(XW%*%X)
         XWz <- XW%*%z

         # Computing the new score vector.
         U <- XW%*%(z-eta)

         # Updating the new coefficence.
         betahat <- XWX%*%XWz

         # Increment the iter variable by one.
         iter <- iter + 1

         # Condition that checks whether the score vector contains a infite value.
         # In this case the algorithm does not converge and the the function will
         # throw an error.
         if (any(is.nan(U) == TRUE)) {
            stop("Model is not converging. Maybe try different startvalues.")
         }

      }
      # Setting the dispersion parameter to 1.
      phi <- 1

      # Computing the standard error of the coefficients.
      beta_se <- sqrt(diag(XWX))

      # Fitting values with
      fitted_values = exp(-(X %*% betahat))

      # Computing the projection matrix (hat matrix).
      H <- diag(X%*%solve(t(X)%*%X)%*%t(X))

      # Computing the residual deviance.
      deviance <- 2 * sum(y/mu - 1- log(y) + log(mu))

      ###################### Residuals ######################
      #
      # Computing the pearson residuals.
      pearson_residuals <- (y - mu)/(sqrt(V))

      # Computing the standardised Pearson residuals.
      standardised_pearson_residuals <- pearson_residuals/sqrt(1-H)

      # Compute the deviance residuals.
      deviance_residuals <- sign(y-mu)*sqrt(2 * (y/mu - 1- log(y) + log(mu)))

      # Computing standardised deviance residuals.
      standardised_deviance_residuals <-  deviance_residuals/sqrt(1-H)

      ###################### Preparing Output ######################
      #
      # Computing the residual degrees of freedom.
      resid_dof <- x_dim[1] - x_dim[2]

      # Computing the model degrees of freedom.
      model_dof <- x_dim[1]

      # Computing the p-values.
      p_values <- 2*(1-pnorm(abs((betahat)/beta_se)))

      # Putting all summary statistics into a joined data frame.
      mle_table <- data.frame(betahat, beta_se, betahat/beta_se, p_values)

      # Naming the columns of the data frame according to its content.
      colnames(mle_table) <- c("Estimate", "Std. Errors", "z-value", "p-value")

      # Nameing the columns dynamically to match the number of coefficients.
      rownames(mle_table) <- paste(rep("X", x_dim[2]), 1:x_dim[2], sep = "")

      # Creating a list with all values that will be return from this function. Using
      # a list allows for easy access of the individual items through the $ notation.
      output <- list("y" = y, "fitted" = fitted_values,"betahat" = betahat, "sebeta" = beta_se,
                     "cov.beta" = XWX, "df.model" = model_dof, "df.residual" = resid_dof, "deviance" = deviance,
                     "summary" = mle_table)

      # Print general summary to the console.
      cat("------------------------ GLM ----------------------- \n\n")
      print(mle_table)
      cat(" \nModel df: ", output$df.model, "\nResidual df:", output$df.residual, "\n\n")
      cat("Deviance: ", output$deviance, "\n")

      ###################### Plotting ######################
      #
      # Making the plotting feature conditional is a pure
      # convinience. This might be an advantage where the glm function is used
      # in a loop and the user doesnt want to create a plot for each iteration.

      #Checks whether the user specified plotting as true.
      if(plot == TRUE) {

         # Creating the Cooks Distance plot.
         par(mfrow=c(2,2))

         # Computing the Cooks Distance.
         cooks_distance <- (H/(1-H)) * (pearson_residuals^2 /x_dim[2])

         # Plotting the Cooks distance.
         plot(cooks_distance, type="h", xlab="Observations", ylab="Cooks Distance", main = "Cooks Distance")

         # Computing the 3 largest Cooks distances.
         # First we sort the cooks distance largest to smallest
         # and retrive the first three values. Im reversing the vector again to
         # have the smalles value at index 1.
         largest <- rev(sort(cooks_distance, decreasing = TRUE)[1:3])

         # Here we are retrieving the indices of the largest values.
         # We set the 3rd largest value to be the threshold.
         idx <- which(cooks_distance >= largest[1])

         # Knowing the vlue and index of these values we can now plot them
         # as labels onto the graph.
         text(idx, largest, labels=round(largest, 3), cex= 0.7, pos=2)

         # Creating qqplot for the standardised deviance residuals.
         qqnorm(standardised_deviance_residuals, ylab="Std. Deviance Residuals")

         # Drawing the qq-line in order to be able to conduct
         # comparison.
         qqline(standardised_deviance_residuals)

         # Creating a deviance residuals vs fitted plot.
         plot(deviance_residuals, xlab="Fitted", ylab = "Deviance Residuals", main="Deviance Residuals vs. Fitted")

         # Adding a line to the deviance vs. fitted plot.
         abline(h= 0, lty=2, lwd=1)

         # Plotting scale location plot.
         plot(sqrt(abs(standardised_deviance_residuals)), xlab = "Fitted", ylab =expression(sqrt("Std. Deviance Residual")) ,main ="Scale-Location")
      }
      return(output)
   }
