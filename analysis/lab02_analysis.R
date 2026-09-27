# ANLY 735 - Replication Laboratory #2
# Controlled Biomedical Learning Experiment
# Student: Umesh Nepali

# ---------------------------------------------------------
# 1. Experimental setup
# ---------------------------------------------------------

# Set a random seed so the experiment is reproducible
set.seed(735)

# Number of synthetic patients in each condition
n_patients <- 1000

# Number of learning epochs
n_epochs <- 40

# Learning rate for the logistic regression model
learning_rate <- 0.05

# Display experiment settings
cat("Number of patients:", n_patients, "\n")
cat("Learning epochs:", n_epochs, "\n")
cat("Learning rate:", learning_rate, "\n")



# ---------------------------------------------------------
# 2. Generate synthetic biomedical data
# ---------------------------------------------------------

# Condition A: Initial patient population
age_A <- rnorm(n_patients, mean = 65, sd = 8)
biomarker_A <- rnorm(n_patients, mean = 0, sd = 1)
inflammation_A <- rnorm(n_patients, mean = 0, sd = 1)
metabolic_A <- rnorm(n_patients, mean = 0, sd = 1)

# Condition B: Patient population after conditions change
age_B <- rnorm(n_patients, mean = 65, sd = 8)
biomarker_B <- rnorm(n_patients, mean = 0, sd = 1)
inflammation_B <- rnorm(n_patients, mean = 0, sd = 1)
metabolic_B <- rnorm(n_patients, mean = 0, sd = 1)