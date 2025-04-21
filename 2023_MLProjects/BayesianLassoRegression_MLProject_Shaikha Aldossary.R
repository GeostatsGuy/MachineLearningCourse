#PGE383 GRADUATE STUDENT PROJECT TEMPLATE
#Shaikha Aldossary, PhD student, University of Texas at Austin
#NOV 17, 2023
###############################################################
#Bayesian LASSO Regression Project
#The project was inspired by Prof. Pyrcz lecture
#The project was supposed to be done using Python however Theano is no longer available. Therefore, i used Rstudio to run Bayesian Lasso regression
#Executive Summary: 
#Bayesian statistics was implemented in the LASSO regression to perform this Project.
#The parameters was used to estimate the posterior
#In this project a single predictor was used (Porosity) and the response feature is Production.
#The results were compared with Frequentist LASSO performance using same predictor and response feature.
#Bayesian LASSO regression illustrate regularization and uncertainty of the model parameters.
###############################################################
#Bayesian LASSO Regression workflow using Rstudio
#Shaikha Aldossary
#The Hildebrand Department of Petroleum and Geosystems Engineering
#I will be demonstrating another Regression method called Bayesian Lasso Regression and compare it with frequentist Lasso Regression.
#I will be using the dataset shared by Prof. Michael Pyrcz which is unconv_MV_v5.csv
#For the purpose of evaluating and visualizing Baysian Lasso regression, I will be using only porosity and production data.
#The result of Bayesian Lasso Regression should show posterior distribution as a point estimate for the parameters. Using the same predictor and target features for comparing frequentist LASSO Regression and Bayesian Lasso Regression.
#Also, Introducing Bayesian to Lasso regression will assess regularization and uncertainty in the model parameters as well. 
#The notebook should act as an educational guide for beginners to understand how to use Bayesian Lasso Regression and frequentist LASSO Regression. 
#In addition, discuss the similarity and differences between two Regression methods.

#FIRST STEP: Install, load packages and read data
# Install and load the required packages for Bayesian Lasso regression in R
install.packages("brms")
install.packages("ggplot2")

# Load necessary libraries to model Bayesian lasso regression
library(brms)
library(rstan)
library(ggplot2)

# Set the working directory
setwd("/Users/shaikhaaldossary/downloads")

# Read in the dataset (I called it ProjectDataset)
ProjectDataset <- read.csv("unconv_MV_v5.csv")
#I renamed Por and Prod to full name Porosity and Production from "unconv_MV_v5.csv" Provided by Prof.Pyrcz

# Check the structure of the dataset
head(ProjectDataset)

# Define response and predictor variables
response_variable <- "Production"
predictor_variables <- "Porosity"

# Create a scatter plot of Response Vs Predictor 
print(ggplot(ProjectDataset, aes(x = Porosity, y = Production)) +
        geom_point() +
        labs(x = "Porosity", y = "Production", title = "Scatter Plot of Production vs Porosity"))
##################################################################################################

# SECOND STEP: Building Bayesian Lasso Regression Model

# Check the structure of the data frame
str(ProjectDataset)

# Check for missing values
sum(is.na(ProjectDataset$Porosity) | is.na(ProjectDataset$Production))

# Set lambda value for Bayesian Lasso Regression i called it my_lambda for this project i will set it to 10
my_lambda <- 10


# Define custom Stan model code with my_lambda specified
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
#y ~ normal(alpha + beta * x, sqrt(sigma_sq)): This line specifies the likelihood function, assuming a normal (Gaussian) distribution for the observed response variable y given the linear predictor alpha + beta * x and a variance of sigma_sq.
#beta ~ double_exponential(0, sqrt(sigma_sq) / my_lambda): This line specifies a double-exponential (Laplace) prior distribution for the regression coefficient beta. The double-exponential distribution is used for regularization (L1 regularization) to encourage sparsity in the coefficients. The my_lambda parameter controls the scale of the prior.
#alpha ~ uniform(-10, 10): This line specifies a uniform prior distribution for the intercept parameter alpha between -10 and 10.
#sigma_sq ~ inv_gamma(1, 10): This line specifies an inverse gamma prior distribution for the variance parameter sigma_sq with shape parameter 1 and scale parameter 10.
#In summary, this Stan model represents a linear regression model with Bayesian inference, where the coefficients are regularized using a double-exponential prior (beta ~ double_exponential) to induce sparsity in the model. The other parameters have specified prior distributions to complete the Bayesian model. The my_lambda parameter allows you to control the strength of the L1 regularization.

# Save Stan model code to a file
stan_model_file <- tempfile(fileext = ".stan")
writeLines(stan_model_code, con = stan_model_file)

# Fit Bayesian Lasso regression model
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

# Display summary and diagnostic plots
summary(bayesian_lasso_model)
plot(bayesian_lasso_model)
bayesplot::mcmc_trace(bayesian_lasso_model)
pp_check(bayesian_lasso_model)
coef_samples <- as.data.frame(as.matrix(bayesian_lasso_model))
plot(coef_samples[, c("b_Porosity", "b_Intercept")])


#To visualize Bayesian Lasso with predictor and response features

# Extract posterior samples of predictions
yhat_bayesian_lasso_samples <- posterior_predict(bayesian_lasso_model)

# Calculate median prediction
yhat_bayesian_lasso_median <- apply(yhat_bayesian_lasso_samples, 2, median)

# Convert data to a data frame for ggplot
plot_data <- data.frame(
  Porosity = ProjectDataset$Porosity,
  Production = ProjectDataset$Production,
  yhatBayesianLasso = yhat_bayesian_lasso_median
)

# Plot scatter plot with Bayesian and Frequentist Lasso regression lines
combined_plot <- ggplot(plot_data, aes(x = Porosity, y = Production)) +
  geom_point() +
  geom_line(aes(y = yhatBayesianLasso, color = "Bayesian LASSO"), linetype = "dashed", size = 1.2) +
  labs(title = "Scatter Plot with Bayesian and Frequentist LASSO Regression Lines") +
  scale_color_manual(values = c("Bayesian LASSO" = "blue")) +
  theme_minimal() +
  theme(legend.position = "top")

# Display combined plot
print(combined_plot)

#############################################################################

# THIRD STEP: Building Frequentist Lasso for comparison
# Check the structure of the data frame
str(ProjectDataset)

# Check for missing values
sum(is.na(ProjectDataset$Porosity) | is.na(ProjectDataset$Production))

# Load the glmnet package for frequentist Lasso 
library(glmnet)

# Convert data to matrix format using same predictor and response features
X <- as.matrix(ProjectDataset$Porosity)
y <- as.matrix(ProjectDataset$Production)


# Convert X to a matrix with multiple columns
X <- cbind(X, rep(1, nrow(X)))  # Adding a column of 1s as an example, This is adjustable based on needs

# Fit the LASSO regression model with a specific lambda
lasso_model <- glmnet(X, y, alpha = 1, lambda = my_lambda)

# Extract the coefficients
lasso_coef <- coef(lasso_model)

# Display the coefficients
print(lasso_coef)

# Make predictions using the LASSO model
yhatLasso <- predict(lasso_model, newx = X, s = my_lambda)

#extract the coefficients as a numeric vector
lasso_coef_numeric <- as.numeric(lasso_coef)
print(lasso_coef_numeric)


# Comparison plot Bayesian LASSO and Frequentist LASSO

# Convert data to a data frame for ggplot
plot_data <- data.frame(
  Porosity = ProjectDataset$Porosity,
  Production = ProjectDataset$Production,
  yhatBayesianLasso = yhat_bayesian_lasso_median,
  yhatFrequentistLasso = as.vector(yhatLasso)
)

# Plot scatter plot with Bayesian and Frequentist Lasso regression lines
combined_plot <- ggplot(plot_data, aes(x = Porosity, y = Production)) +
  geom_point() +
  geom_line(aes(y = yhatBayesianLasso, color = "Bayesian LASSO"), linetype = "dashed", size = 1.2) +
  geom_line(aes(y = yhatFrequentistLasso, color = "Frequentist LASSO"), linetype = "dashed", size = 1.2) +
  labs(title = "Scatter Plot with Bayesian and Frequentist LASSO Regression Lines") +
  scale_color_manual(values = c("Bayesian LASSO" = "blue", "Frequentist LASSO" = "red")) +
  theme_minimal() +
  theme(legend.position = "top")

# Display combined plot
print(combined_plot)

###########################################################################################################

# FOURTH STEP: Adding sample intercept and beta to the plot

# Extract posterior samples of intercept and beta
posterior_samples <- as.data.frame(as.matrix(bayesian_lasso_model))
sample_intercept <- posterior_samples$b_Intercept
sample_beta <- posterior_samples$b_Porosity


# Create data frame for plotting
plot_data <- data.frame(
  Porosity = ProjectDataset$Porosity,
  Production = ProjectDataset$Production,
  yhatBayesianLasso = yhat_bayesian_lasso_median,
  yhatFrequentistLasso = as.vector(yhatLasso),
  Sample_Intercept = sample_intercept,
  Sample_Beta = sample_beta
)

# Plot scatter plot with Bayesian and Frequentist Lasso regression lines, and sample intercept/beta
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

# Display combined plot
print(combined_plot)
##################################################################################################################