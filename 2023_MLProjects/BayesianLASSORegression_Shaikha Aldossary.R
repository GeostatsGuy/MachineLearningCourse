# PGE 383 Project: Bayesian LASSO Regression Using Rstudio
# Shaikha Aldossary
# The Hildebrand Department of Petroleum and Geosystems Engineering, Cockrell School of Engineering

# Subsurface Machine Learning Course, The University of Texas at Austin
  #Hildebrand Department of Petroleum and Geosystems Engineering, Cockrell School of Engineering

# Workflow supervision and review by:
  # Instructor: Prof. Michael Pyrcz, Ph.D., P.Eng., Professor, The University of Texas at Austin
  # Course TA: Lei Liu, Graduate Student, The University of Texas at Austin

# Bayesian LASSO Regression Project
  #The project was inspired by Prof. Pyrcz lecture. The project was supposed to be done using Python. However, Theano is no longer available. Therefore, I used Rstudio to run Bayesian LASSO regression. I will be demonstrating another regression method called Bayesian LASSO regression and comparing it with frequentist LASSO regression. I will be using the data shared by Prof. Michael Pyrcz which is unconv_MV_v5.csv.
  #The Bayesian LASSO model in this workflow is not using Gibbs sampling or Metropolis-Hastings sampling directly. Instead, it uses a more advanced approach based on Hamiltonian dynamics, which allows for more efficient exploration of the parameter captured through stan. Stan employs a variant of Hamiltonian Monte Carlo (HMC) called the No-U-Turn Sampler (NUTS) for efficient sampling from the posterior distribution. HMC is a type of Markov Chain Monte Carlo (MCMC) method.

# Executive Summary
  #Bayesian statistics were implemented in the LASSO regression to perform this project. The parameters were used to estimate the posterior. In this project, a single predictor was used (porosity) and the response feature is production. The results were compared with frequentist LASSO performance using the same predictor and response feature. For lambda = 10, Bayesian LASSO performance is similar to frequentist LASSO regression. Both models show a good fit with the data. However, Bayesian LASSO regression illustrates the regularization and uncertainty of the model parameters. Further exploration can be done by evaluating the performance of Bayesian LASSO and frequentist LASSO regressions at a different lambda.
  
# Project Workflow:

# 1. Install and Load Packages
  #These are the required packages for Bayesian LASSO regression in R that provide additional functionality and tools for data analysis and visualization.
install.packages("brms")
install.packages("ggplot2")

library(brms)
#Helps specify and estimate Bayesian regression models using stan.

library(rstan)
#Provides an interface to the stan programming language, enabling the estimation of complex Bayesian models.

library(ggplot2)
#Offers a versatile and expressive tool set for creating high-quality data visualizations, making it easier to explore and communicate insights from the data.

#Load the glmnet package for the frequentist LASSO
library(glmnet)
#This package is designed for fitting generalized linear models (GLMs) with regularization, particularly for problems like LASSO (L1 regularization) and Ridge (L2 regularization).



# 2. Load Data
#The following workflow applies the .csv file "unconv_MV_v5.csv" shared by Prof. Pyrcz for the subsurface machine learning course.
# Set the working directory
setwd("/Users/shaikhaaldossary/downloads")

# Read in the dataset (I called it ProjectDataset)
ProjectDataset <- read.csv("unconv_MV_v5.csv")    
#I renamed Por and Prod to its full name, "Porosity" and "Production".

# Check the structure of the dataset
head(ProjectDataset)

# Define response and predictor variables
response_variable <- "Production"
predictor_variables <- "Porosity"

# Create a scatter plot of response Vs predictor 
print(ggplot(ProjectDataset, aes(x = Porosity, y = Production)) +
        geom_point() +
        labs(x = "Porosity", y = "Production", title = "Scatter Plot of Production vs Porosity"))

#The scatter plot visually displays the relationship between the predictor variable (porosity) and the response variable (production), providing insights into the overall pattern and distribution of the data.


# 3. Building a Bayesian LASSO Regression Model
#Check the data before building the Bayesian LASSO regression model.
# Check the structure of the data frame
str(ProjectDataset)

# Check for missing values
sum(is.na(ProjectDataset$Porosity) | is.na(ProjectDataset$Production))

# Methodology Used to Model Bayesian LASSO Regression:
#  a. Data Block:
      #N is the number of observations.
      #y is a vector of length N representing the response variable.
      #x is a vector of length N representing the predictor variable.
      #my_lambda is a real number representing the regularization parameter.
#  b. Parameters Block:
      #alpha is the intercept parameter.
      #beta is the coefficient parameter (slope).
      #sigma_sq is the variance parameter.
#  c. Model Block:
      #y ~ normal(alpha + beta * x, sqrt(sigma_sq));: This line specifies the likelihood function. It assumes a normal (Gaussian) distribution for the observed response variable y given the linear predictor alpha + beta * x and a standard deviation of sqrt(sigma_sq). This is the core of the linear regression model.
      #beta ~ double_exponential(0, sqrt(sigma_sq) / my_lambda);: This line specifies a double-exponential (Laplace) prior distribution for the regression coefficient beta. The double-exponential distribution is used for regularization (L1 regularization) to encourage sparsity in the coefficients. The my_lambda parameter controls the scale of the prior, influencing the strength of regularization.
      #alpha ~ uniform(-10, 10);: This line specifies a uniform prior distribution for the intercept parameter alpha between -10 and 10.
      #sigma_sq ~ inv_gamma(1, 10);: This line specifies an inverse gamma prior distribution for the variance parameter sigma_sq with shape parameter 1 and scale parameter 10.

#The Bayesian LASSO model assumes a normal distribution for the response variable with a linear predictor. The double-exponential prior on the coefficient beta introduces L1 regularization, encouraging some coefficients to be exactly zero. The other parameters have specified prior distributions to complete the Bayesian model. The my_lambda parameter allows to control the strength of the L1 regularization, influencing the sparsity of the model.

# Bayesian LASSO Regression Implementation
#Building Bayesian LASSO regression by using the project data and defining important parameters.

# Set a lambda value for Bayesian LASSO regression I called it my_lambda for this project, I will set it to 10
my_lambda <- 10


# Define custom stan model code with my_lambda specified
stan_model_code <- "
  data {
    int<lower=0> N;
    vector[N] y;
    vector[N] x;
    real my_lambda;
  }
  parameters {
    real alpha;               # Prior for intercept
    real beta;                # Prior for coefficient
    real<lower=0> sigma_sq;   # Prior for variance (sigma square)
  }
  model {
    y ~ normal(alpha + beta * x, sqrt(sigma_sq));      # Likelihood 
    beta ~ double_exponential(0, sqrt(sigma_sq) / my_lambda);  # Prior for coefficient
    alpha ~ uniform(-10, 10);                               # Prior for intercept
    sigma_sq ~ inv_gamma(1, 10);                        # Prior for variance (sigma square)
  }
"


# Save stan model code to a file
stan_model_file <- tempfile(fileext = ".stan")
writeLines(stan_model_code, con = stan_model_file)
#Saving the stan model code to a file serves a practical purpose when using the brms package in R to fit Bayesian models. The brm function in brms requires a file path to the stan model code and providing it in a separate file is a common practice. Then, fit the Bayesian LASSO model as follows:

# Fit Bayesian LASSO regression model
bayesian_lasso_model <- brm(
  formula = Production ~ Porosity,
  data = ProjectDataset,
  file = stan_model_file,
  cores = 4,
  iter = 5000,
  chains = 4,
  control = list(adapt_delta = 0.95),
  sample_prior = "yes"
)

#In this code:
  #formula = Production ~ Porosity: this specifies the formula for the Bayesian model. It indicates that the response variable (production) is modeled as a function of the predictor variable (porosity) using a linear relationship.
  #data = ProjectDataset: this specifies the data frame (ProjectDataset) that contains the variables used in the formula.
  #file = stan_model_file: this specifies the path to the stan model file. The stan model code is written in the file specified by stan_model_file.
  #cores = 4: this specifies the number of CPU cores to be used for parallel computing. In this case, it's set to 4 cores.
  #iter = 5000: this specifies the total number of iterations for the Markov chain Monte Carlo (MCMC) sampling. In Bayesian analysis, MCMC is used to approximate the posterior distribution of the model parameters. Here, it's set to 5000 iterations.
  #chains = 4: this specifies the number of chains to be run in parallel during MCMC sampling. In this case, it's set to 4 chains.
  #control = list(adapt_delta = 0.95): this provides additional control parameters for the MCMC algorithm. adapt_delta controls the target acceptance rate during the adaptation phase. A value of 0.95 is specified here.
  #sample_prior = "yes": this indicates whether to sample from the prior distribution. Setting it to "yes" means that prior draws are included in the output.

#The following summary provides a concise overview of various aspects of the model, allowing to assess its performance and understand the estimated parameters.
# Display summary 
summary(bayesian_lasso_model)

#The following plots are designed to help evaluate the performance of the Markov Chain Monte Carlo (MCMC) sampling, check for convergence, and assess the validity of the model assumptions.
  #a. Trace plots show the values of the estimated parameters over iterations of the MCMC sampling process. Each chain is typically displayed in a different color. These plots help visually inspect whether the chains have converged and if there are any trends or patterns.
  #b. Density plots provide the posterior distribution of each estimated parameter. These plots give an idea of the uncertainty associated with each parameter. They can reveal skewness, multimodality, or other important features of the distribution.
plot(bayesian_lasso_model)


#Generate trace plots for the parameters estimated in a Bayesian model. Trace plots are a common diagnostic tool in Bayesian statistics, providing a visual representation of how the parameter values evolve over the iterations of the Markov Chain Monte Carlo (MCMC) sampling
  #a. Each trace plot represents the MCMC chain for a specific parameter in the Bayesian LASSO regression model.
  #b. The x-axis represents the iteration number (MCMC step), and the y-axis represents the value of the parameter at each iteration.
  #c. Multiple chains are typically displayed in different colors to assess convergence.
bayesplot::mcmc_trace(bayesian_lasso_model)


#Perform posterior predictive checks plot: a technique to assess the fit of a Bayesian model by comparing simulated data generated from the posterior predictive distribution to the observed data. This command generates a series of plots to facilitate this comparison.
pp_check(bayesian_lasso_model)

#Then, extract and store the posterior samples of the model coefficients from a Bayesian LASSO regression model.
coef_samples <- as.data.frame(as.matrix(bayesian_lasso_model))
plot(coef_samples[, c("b_Porosity", "b_Intercept")])
#This scatter plot matrix of the posterior samples has two coefficients: the 'Porosity' coefficient (b_Porosity) and the intercept (b_Intercept). This type of plot is useful for visually examining the joint distribution and relationships between pairs of variables.


# To visualize Bayesian LASSO with predictor and response features

  # Extract posterior samples of predictions
yhat_bayesian_lasso_samples <- posterior_predict(bayesian_lasso_model)
  #This is using the posterior_predict function to generate posterior predictive samples of the response variable from the Bayesian LASSO regression model fitted using the brm function.

  # Calculate median prediction
yhat_bayesian_lasso_median <- apply(yhat_bayesian_lasso_samples, 2, median)
  #This is using the apply function to calculate the median across the columns (dimension 2) of the matrix.

  # Convert data to a data frame for ggplot
plot_data <- data.frame(
  Porosity = ProjectDataset$Porosity,
  Production = ProjectDataset$Production,
  yhatBayesianLasso = yhat_bayesian_lasso_median
)
  #The lines of code are creating a data frame named plot_data from the original data (ProjectDataset) and the median predictions (yhat_bayesian_lasso_median) obtained from the Bayesian LASSO regression model.

  # Plot scatter plot with Bayesian regression line
combined_plot <- ggplot(plot_data, aes(x = Porosity, y = Production)) +
  geom_point() +
  geom_line(aes(y = yhatBayesianLasso, color = "Bayesian LASSO"), linetype = "dashed", size = 1.2) +
  labs(title = "Scatter Plot with Bayesian LASSO Regression Line") +
  scale_color_manual(values = c("Bayesian LASSO" = "blue")) +
  theme_minimal() +
  theme(legend.position = "top")
  #This creates a combined plot that includes a scatter plot of the observed data and a dashed line representing the regression line based on the median predictions from the Bayesian LASSO regression model. 
  # Display combined plot
print(combined_plot)

#The plot is a visual representation of how well the Bayesian LASSO regression line fits the observed data. The scatter plot allows us to compare the distribution of observed data points with the central tendency represented by the Bayesian LASSO regression line. It suggests a good fit of the model because the line closely follows the pattern of the data points.



# 4. Building a Frequentist LASSO for Comparison

#Check the data before building frequentist LASSO model.
  #a. Check the structure of the data frame
str(ProjectDataset)
  #b. Check for missing values
sum(is.na(ProjectDataset$Porosity) | is.na(ProjectDataset$Production))


#Convert data to matrix format using the same predictor and response features.
X <- as.matrix(ProjectDataset$Porosity)
#This code is converting a column of the data frame ProjectDataset into a matrix and assigning it to the variable X. In this case, X is the predictor feature "porosity".
y <- as.matrix(ProjectDataset$Production)
#This code is converting a column of the data frame ProjectDataset into a matrix and assigning it to the variable y. In this case, y is the response feature "production".
# Convert X to a matrix with multiple columns
X <- cbind(X, rep(1, nrow(X)))  
#In this example, the modified X matrix includes a column for "porosity" and a column of ones, making it suitable for fitting a regression model with both a slope coefficient for "porosity" and an intercept.


#Fitting a LASSO regression model using the glmnet package in R.
# Fit the LASSO regression model with a specific lambda
lasso_model <- glmnet(X, y, alpha = 1, lambda = my_lambda)
  #alpha = 1: This specifies the type of regularization to be applied. In this case, alpha = 1 indicates LASSO regularization
  #lambda = my_lambda: This parameter controls the strength of the regularization. A higher lambda results in a more heavily regularized model, which tends to shrink coefficients towards zero.

# Extract the coefficients
lasso_coef <- coef(lasso_model)
#This code is extracting the coefficients from the fitted LASSO regression model stored in the lasso_model object and assigning them to the variable lasso_coef.
# Display the coefficients
print(lasso_coef)     
#This code is used to display the coefficients of the LASSO regression model stored in the lasso_coef variable.

# Make predictions using the LASSO model
yhatLasso <- predict(lasso_model, newx = X, s = my_lambda)
#This code is using the fitted LASSO regression model (lasso_model) to make predictions on new data (X) for a specific value of the regularization parameter (my_lambda)
#extract the coefficients as a numeric vector
lasso_coef_numeric <- as.numeric(lasso_coef)
#This code is converting the coefficients obtained from the LASSO regression model (stored in lasso_coef) into a numeric vector and assigning them to the variable lasso_coef_numeric.
print(lasso_coef_numeric)
#This will display the numeric vector containing the coefficients obtained from the LASSO regression model. The output would look like a simple numeric vector with the values of the coefficients.


# Comparison Plot Bayesian LASSO and Frequentist LASSO

# Convert data to a data frame for ggplot
plot_data <- data.frame(
  Porosity = ProjectDataset$Porosity,
  Production = ProjectDataset$Production,
  yhatBayesianLasso = yhat_bayesian_lasso_median,
  yhatFrequentistLasso = as.vector(yhatLasso)
)
#Plot_data will be a data frame containing columns for the original "porosity" and "production" variables, as well as the predicted values from the Bayesian LASSO and frequentist LASSO models.

# Plot scatter plot with Bayesian and frequentist LASSO regression lines
combined_plot <- ggplot(plot_data, aes(x = Porosity, y = Production)) +
  geom_point() +
  geom_line(aes(y = yhatBayesianLasso, color = "Bayesian LASSO"), linetype = "dashed", size = 1.2) +
  geom_line(aes(y = yhatFrequentistLasso, color = "Frequentist LASSO"), linetype = "dashed", size = 1.2) +
  labs(title = "Scatter Plot with Bayesian and Frequentist LASSO Regression Lines") +
  scale_color_manual(values = c("Bayesian LASSO" = "blue", "Frequentist LASSO" = "red")) +
  theme_minimal() +
  theme(legend.position = "top")
#This code creates a combined plot using the ggplot2 package in R to visualize the scatter plot with both Bayesian and frequentist LASSO regression lines.

# Display combined plot
print(combined_plot)

#This plot shows predictor feature "porosity" Vs. response feature "production". The blue line shows Bayesian LASSO and the red line shows frequentist LASSO regressions. Both models indicate a good fit with the data.


# 5. Adding the Sample Intercept and Beta to the Combined Plot

#Extract posterior samples of intercept and beta.
posterior_samples <- as.data.frame(as.matrix(bayesian_lasso_model))
#This code is converting the posterior samples obtained from the Bayesian LASSO regression model (bayesian_lasso_model) into a data frame for further analysis or visualization.
sample_intercept <- posterior_samples$b_Intercept
#This code is extracting the posterior samples of the intercept parameter (b_Intercept) from the data frame posterior_samples and assigning them to the variable sample_intercept.
sample_beta <- posterior_samples$b_Porosity
#This code is extracting the posterior samples of the regression coefficient for the predictor variable "Porosity" (b_Porosity) from the data frame posterior_samples and assigning them to the variable sample_beta.


# Create data frame for plotting
plot_data <- data.frame(
  Porosity = ProjectDataset$Porosity,
  Production = ProjectDataset$Production,
  yhatBayesianLasso = yhat_bayesian_lasso_median,
  yhatFrequentistLasso = as.vector(yhatLasso),
  Sample_Intercept = sample_intercept,
  Sample_Beta = sample_beta
)
# Adding sample_intercept and sample_beta to plot_data for further analysis and visualization

# Plot scatter plot with Bayesian and frequentist LASSO regression lines, and sample intercept/beta
combined_plot <- ggplot(plot_data, aes(x = Porosity, y = Production)) +
  geom_point() +
  geom_line(aes(y = yhatBayesianLasso, color = "Bayesian LASSO"), linetype = "dashed", size = 1.2) +
  geom_line(aes(y = yhatFrequentistLasso, color = "Frequentist LASSO"), linetype = "dashed", size = 1.2) +
  geom_point(aes(y = Sample_Intercept + Sample_Beta * Porosity, color = "Sample Intercept and Beta"), size = 2) +
  labs(
    title = "Scatter Plot with Bayesian and Frequentist LASSO Regression Lines",
    x = "Porosity",
    y = "Production"
  ) +
  scale_color_manual(
    values = c("Bayesian LASSO" = "blue", "Frequentist LASSO" = "red", "Sample Intercept and Beta" = "green"),
    name = "Lines and Points"
  ) +
  theme_minimal() +
  theme(
    legend.position = "top",
    plot.title = element_text(hjust = 0.5)
  )
#This code extends the previous combined_plot by adding another layer to the plot which is sample_intercept and sample_beta.

# Display combined plot
print(combined_plot)

#The plot shows the scatter plot of observed values along with the Bayesian and frequentist LASSO regression lines, as well as points based on the sample intercept and sample beta. This visualization helps in assessing how well the model predictions align with the actual data points.


# Results
#The workflow involves modeling Bayesian LASSO regression, comparing it with frequentist LASSO regression, and visualizing the results. The results show a good fit using both models (Bayesian LASSO regression and frequentist LASSO). sample_intercept and sample_beta are also in agreement with both models and the observed data. For lambda = 10, Bayesian LASSO performance is similar to frequentist LASSO regression. However, Bayesian LASSO regression illustrates the regularization and uncertainty of the model parameters. Further exploration can be done by evaluating the performance of Bayesian LASSO and frequentist LASSO regressions at a different lambda.

# Want to Work Together
#I hope this project is helpful to those who want to learn more about modeling Bayesian LASSO regression. I will be glad to discuss this workflow and answer your questions. You can reach me at shaikha.aldossary@gmail.com.

