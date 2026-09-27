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
# ---------------------------------------------------------
# 3. Define disease outcomes under changing conditions
# ---------------------------------------------------------

# Standardize age so it is on a similar scale as the biomarkers
age_A_z <- (age_A - 65) / 8
age_B_z <- (age_B - 65) / 8

# CONDITION A
# Disease risk is strongly associated with the biomarker
risk_A <- -0.5 +
  0.4 * age_A_z +
  1.5 * biomarker_A +
  0.8 * inflammation_A +
  0.4 * metabolic_A

prob_A <- 1 / (1 + exp(-risk_A))

disease_A <- rbinom(
  n_patients,
  size = 1,
  prob = prob_A
)

# CONDITION B
# The biomarker-disease relationship changes
risk_B <- -0.5 +
  0.4 * age_B_z -
  1.5 * biomarker_B +
  0.8 * inflammation_B +
  0.4 * metabolic_B

prob_B <- 1 / (1 + exp(-risk_B))

disease_B <- rbinom(
  n_patients,
  size = 1,
  prob = prob_B
)

# Check disease prevalence in both conditions
cat("Disease prevalence - Condition A:",
    round(mean(disease_A), 3), "\n")

cat("Disease prevalence - Condition B:",
    round(mean(disease_B), 3), "\n")

# ---------------------------------------------------------
# 4. Prepare data for model learning
# ---------------------------------------------------------

# Predictor matrix for Condition A
X_A <- cbind(
  1,
  age_A_z,
  biomarker_A,
  inflammation_A,
  metabolic_A
)

# Predictor matrix for Condition B
X_B <- cbind(
  1,
  age_B_z,
  biomarker_B,
  inflammation_B,
  metabolic_B
)

# Give the columns meaningful names
colnames(X_A) <- c(
  "Intercept",
  "Age",
  "Biomarker",
  "Inflammation",
  "Metabolic"
)

colnames(X_B) <- colnames(X_A)

# Check dimensions
cat("Condition A dimensions:", dim(X_A), "\n")
cat("Condition B dimensions:", dim(X_B), "\n")

# ---------------------------------------------------------
# 5. Define logistic regression learning functions
# ---------------------------------------------------------

# Convert a model score into a probability
sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

# Perform one learning update
update_weights <- function(weights, X, y, learning_rate) {
  
  # Calculate predicted disease probabilities
  probabilities <- sigmoid(X %*% weights)
  
  # Calculate the prediction error gradient
  gradient <- colMeans(
    X * as.vector(probabilities - y)
  )
  
  # Update model weights
  new_weights <- weights - learning_rate * gradient
  
  return(new_weights)
}

# Calculate classification accuracy
calculate_accuracy <- function(weights, X, y) {
  
  probabilities <- sigmoid(X %*% weights)
  
  predictions <- ifelse(
    probabilities >= 0.5,
    1,
    0
  )
  
  mean(predictions == y)
}
exists("update_weights")

# Perform one learning update
update_weights <- function(weights, X, y, learning_rate) {
  
  # Calculate predicted disease probabilities
  probabilities <- sigmoid(X %*% weights)
  
  # Calculate the prediction error gradient
  gradient <- colMeans(
    X * as.vector(probabilities - y)
  )
  
  # Update model weights
  new_weights <- weights - learning_rate * gradient
  
  return(new_weights)
}
exists("update_weights")



# ---------------------------------------------------------
# 6. Train the model on Condition A
# ---------------------------------------------------------

# Start with all model weights equal to zero
weights_A <- rep(0, ncol(X_A))

# Store accuracy after each learning epoch
accuracy_A <- numeric(n_epochs)

# Train the model for 40 epochs
for (epoch in 1:n_epochs) {
  
  weights_A <- update_weights(
    weights = weights_A,
    X = X_A,
    y = disease_A,
    learning_rate = learning_rate
  )
  
  accuracy_A[epoch] <- calculate_accuracy(
    weights_A,
    X_A,
    disease_A
  )
}

# Display learning performance
cat(
  "Condition A accuracy before learning:",
  round(calculate_accuracy(rep(0, ncol(X_A)), X_A, disease_A), 3),
  "\n"
)

cat(
  "Condition A accuracy after learning:",
  round(accuracy_A[n_epochs], 3),
  "\n"
)

# Display final learned weights
print(
  round(
    setNames(weights_A, colnames(X_A)),
    3
  )
)


# ---------------------------------------------------------
# 7. Introduce the change: evaluate on Condition B
# ---------------------------------------------------------

# Save the model learned under Condition A
weights_before_change <- weights_A

# Evaluate the trained model on the new Condition B
accuracy_B_before_learning <- calculate_accuracy(
  weights_before_change,
  X_B,
  disease_B
)

cat(
  "Condition A accuracy before change:",
  round(accuracy_A[n_epochs], 3),
  "\n"
)

cat(
  "Condition B accuracy immediately after change:",
  round(accuracy_B_before_learning, 3),
  "\n"
)
accuracy_B_before_learning


# ---------------------------------------------------------
# 8. Continue learning after the change
# ---------------------------------------------------------

# Start Condition B with the weights learned from Condition A
weights_continued <- weights_before_change

# Store performance during post-change learning
accuracy_B_continued <- numeric(n_epochs)
retention_A <- numeric(n_epochs)

for (epoch in 1:n_epochs) {
  
  # Continue training using Condition B
  weights_continued <- update_weights(
    weights = weights_continued,
    X = X_B,
    y = disease_B,
    learning_rate = learning_rate
  )
  
  # Measure ability to learn the new condition
  accuracy_B_continued[epoch] <- calculate_accuracy(
    weights_continued,
    X_B,
    disease_B
  )
  
  # Measure retention of the original condition
  retention_A[epoch] <- calculate_accuracy(
    weights_continued,
    X_A,
    disease_A
  )
}

cat(
  "Condition B accuracy before continued learning:",
  round(accuracy_B_before_learning, 3),
  "\n"
)

cat(
  "Condition B accuracy after 40 additional epochs:",
  round(accuracy_B_continued[n_epochs], 3),
  "\n"
)

cat(
  "Condition A retention after learning Condition B:",
  round(retention_A[n_epochs], 3),
  "\n"
)

print(
  round(
    setNames(weights_continued, colnames(X_B)),
    3
  )
)



# ---------------------------------------------------------
# 9. Train a fresh model directly on Condition B
# ---------------------------------------------------------

# Initialize a new model with no prior learning
weights_fresh <- rep(0, ncol(X_B))

# Store its learning trajectory
accuracy_B_fresh <- numeric(n_epochs)

for (epoch in 1:n_epochs) {
  
  weights_fresh <- update_weights(
    weights = weights_fresh,
    X = X_B,
    y = disease_B,
    learning_rate = learning_rate
  )
  
  accuracy_B_fresh[epoch] <- calculate_accuracy(
    weights_fresh,
    X_B,
    disease_B
  )
}

cat(
  "Fresh model Condition B accuracy after 40 epochs:",
  round(accuracy_B_fresh[n_epochs], 3),
  "\n"
)


# ---------------------------------------------------------
# 10. Compare post-change learning trajectories
# ---------------------------------------------------------

# Create selected checkpoints
checkpoints <- c(1, 5, 10, 20, 30, 40)

comparison_table <- data.frame(
  Epoch = checkpoints,
  Continued_Model = round(accuracy_B_continued[checkpoints], 3),
  Fresh_Model = round(accuracy_B_fresh[checkpoints], 3),
  Old_Task_Retention = round(retention_A[checkpoints], 3)
)

print(comparison_table)

# Improvement from the first to final post-change epoch
continued_improvement <-
  accuracy_B_continued[n_epochs] - accuracy_B_continued[1]

fresh_improvement <-
  accuracy_B_fresh[n_epochs] - accuracy_B_fresh[1]

cat(
  "Continued model improvement:",
  round(continued_improvement, 3),
  "\n"
)

cat(
  "Fresh model improvement:",
  round(fresh_improvement, 3),
  "\n"
)



# ---------------------------------------------------------
# 11. Plot post-change learning trajectories
# ---------------------------------------------------------

plot(
  1:n_epochs,
  accuracy_B_continued,
  type = "l",
  lwd = 2,
  ylim = c(0.4, 0.8),
  xlab = "Post-Change Learning Epoch",
  ylab = "Classification Accuracy",
  main = "Learning After the Biomedical Condition Changes"
)

lines(
  1:n_epochs,
  accuracy_B_fresh,
  lty = 2,
  lwd = 2
)

lines(
  1:n_epochs,
  retention_A,
  lty = 3,
  lwd = 2
)

legend(
  "right",
  legend = c(
    "Previously Trained Model - Condition B",
    "Fresh Model - Condition B",
    "Previously Trained Model - Condition A Retention"
  ),
  lty = c(1, 2, 3),
  lwd = 2,
  bty = "n"
)