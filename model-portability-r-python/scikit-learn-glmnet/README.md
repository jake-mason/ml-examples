# Porting glmnet-like models between R and Python

I would like to train glmnet-like models in Python and use them to predict in R, and vice-versa.

## Trained in Python, scored in R

```python
import numpy as np
np.random.seed(100)
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

data = load_iris()
X = data.data
y = np.where(data.target == 1, 1, 0)

idxs = np.random.randint(0, X.shape[0], X.shape[0])
train_idxs = idxs[:100]
test_idxs = idxs[100:]
X_train, X_test = X[train_idxs, :], X[test_idxs, :]
y_train, y_test = y[train_idxs], y[test_idxs]

model = LogisticRegression(
	penalty='elasticnet',
	# Scaling the features would help speed up covergence of the 'saga' solver,
	# but let's try it without
	solver='saga',
	warm_start=True,
	l1_ratio=0.5,
	max_iter=1000
)
model.fit(X_train, y_train)

y_proba_train = model.predict_proba(X_train)[:, 1]
y_proba_test = model.predict_proba(X_test)[:, 1]

print(roc_auc_score(y_train, y_proba_train))
print(roc_auc_score(y_test, y_proba_test))

coef_df = pd.DataFrame(
	{
		'feature_name': list(data.feature_names) + ['intercept'],
		'coef': model.coef_.ravel().tolist() + model.intercept_.tolist()
	}
)

coef_df.to_csv('coef.csv', index=False)

# Save indices for equal comparison in R
with open('train_idxs.txt', 'w') as train_idxs_f, open('test_idxs.txt', 'w') as test_idxs_f:
	train_idxs_f.write('\n'.join(map(str, train_idxs)))
	test_idxs_f.write('\n'.join(map(str, test_idxs)))

# Save predicted probabilities
with open('y_proba_train_python.txt', 'w') as y_proba_train_f, open('y_proba_test_python.txt', 'w') as y_proba_test_f:
	y_proba_train_f.write('\n'.join(map(str, y_proba_train)))
	y_proba_test_f.write('\n'.join(map(str, y_proba_test)))
```

Now we'll try to predict in R:

```R
library(glmnet)
library(dplyr)

data(iris)

# Add 1 because of R's indexing
train_idxs <- as.integer(read.table('train_idxs.txt')$V1) + 1
test_idxs <- as.integer(read.table('test_idxs.txt')$V1) + 1

coef_df <- read.csv('coef.csv', stringsAsFactors = FALSE)
coef_X <- as.matrix(coef_df[1:4, 2])
intercept <- coef_df[5, 2]

# Note the data(iris) and load_iris() datasets between R and Python, respectively, are ordered the same way
X_train <- as.matrix(iris[train_idxs, 1:4])
X_test <- as.matrix(iris[test_idxs, 1:4])
# Target name index at 1 in sklearn's load_iris() is versicolor
y_train <- ifelse(iris[train_idxs, 5] == 'versicolor', 1, 0)
y_test <- ifelse(iris[test_idxs, 5] == 'versicolor', 1, 0)

model <- cv.glmnet(
	X_train,
	as.factor(y_train),
	family = 'binomial',
	alpha = 0.5
)

tmp_coeffs <- coef(model, s = 'lambda.min')
r_coef_df <- data.frame(name = tmp_coeffs@Dimnames[[1]][tmp_coeffs@i + 1], coefficient = tmp_coeffs@x) %>% arrange(coefficient)

y_proba_train <- predict(model, X_train, type = 'response', s = 'lambda.min')
y_proba_test <- predict(model, X_test, type = 'response', s = 'lambda.min')

# From here https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/linear_model/_logistic.py#L1619
# we see getting to a predicted probability from an input feature vector isn't as easy as just doing a dot product.
# We need to do a dot product of X * weights, which gives us our "decision function" (as sklearn refers to it), then
# apply the sigmoid transformation to that output
decision_train_python <- (X_train %*% c(coef_X)) + intercept
decision_test_python <- (X_test %*% c(coef_X)) + intercept

sigmoid <- function(x){
	return(1 / (1 + exp(-x)))
}

# Apply sigmoid function to transform decision function to predicted probabilities
y_proba_train_python_r <- sigmoid(decision_train_python)
y_proba_test_python_r <- sigmoid(decision_test_python)

auroc <- function(y_true, y_proba) {
	n1 <- sum(!y_true)
	n2 <- sum(y_true)
	U  <- sum(rank(y_proba)[!y_true]) - n1 * (n1 + 1) / 2
	return(1 - U / n1 / n2)
}

# Compare the performance of the model generated in R to the Python model's coefficients
print(auroc(y_train, y_proba_train))
print(auroc(y_test, y_proba_test))
print(auroc(y_train, y_proba_train_python_r))
print(auroc(y_test, y_proba_test_python_r))

# We can even compare the predicted probabilities to ensure they're basically equal
y_proba_train_python <- as.numeric(read.table('y_proba_train_python.txt')$V1)
y_proba_test_python <- as.numeric(read.table('y_proba_test_python.txt')$V1)

mean(y_proba_train_python_r - y_proba_train_python)
mean(y_proba_test_python_r - y_proba_test_python)
# These are both *very* small numbers and is likely only due
# to the fact the predicted probabilities from Python were saved only up to the 7th decimal place,
# whereas the probabilities in R are unrounded (and go out about twice as many decimal places)
```

Now let's see if we can manipulate things in a way such that we can use R's native `predict` function to, well, make a prediction:

```R
customLogisticRegression <- function(coef, intercept = 0, ...){
	model <- structure(
		list(
			coef = coef,
			intercept = intercept
		),
		class = 'customLogisticRegression'
	)
	return(model)
}

# create a method for function print for class myClassifierClass
predict.customLogisticRegression <- function(model_object, newdata){
	decision <- (newdata %*% c(model_object$coef)) + model_object$intercept
	return(sigmoid(decision))
}

model <- customLogisticRegression(coef_X, intercept)
y_proba_test_custom <- predict(model, X_test)
mean(y_proba_test_python_r - y_proba_test_custom)
# 0 -> there is no difference between the two
```