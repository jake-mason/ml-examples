# Porting XGBoost models between R and Python

I would like to train XGBoost models in Python and use them to predict in R, and vice-versa.

## Trained in Python, scored in R

```python
import xgboost
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score

data = load_iris()
X = data.data
y = np.where(data.target == 1, 1, 0)

idxs = np.random.randint(0, X.shape[0], X.shape[0])
train_idxs = idxs[:100]
test_idxs = idxs[100:]
X_train, X_test = X[train_idxs, :], X[test_idxs, :]
y_train, y_test = y[train_idxs], y[test_idxs]

dtrain = xgboost.DMatrix(X_train, y_train)
dtest = xgboost.DMatrix(X_test, y_test)

params = {
	'learning_rate': 0.1,
	'max_depth': 3,
	'objective': 'binary:logistic',
}

model = xgboost.train(
	params,
	dtrain
)

y_proba_train = model.predict(dtrain)
y_proba_test = model.predict(dtest)

print(roc_auc_score(y_train, y_proba_train))
print(roc_auc_score(y_test, y_proba_test))

model.save_model('model.model')

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
library(xgboost)

data(iris)

# Add 1 because of R's indexing
train_idxs <- as.integer(read.table('train_idxs.txt')$V1) + 1
test_idxs <- as.integer(read.table('test_idxs.txt')$V1) + 1

model <- xgboost::xgb.load('model.model')

# Note the data(iris) and load_iris() datasets between R and Python, respectively, are ordered the same way
X_train <- iris[train_idxs, 1:4]
X_test <- iris[test_idxs, 1:4]
# Target name index at 1 in sklearn's load_iris() is versicolor
y_train <- ifelse(iris[train_idxs, 5] == 'versicolor', 1, 0)
y_test <- ifelse(iris[test_idxs, 5] == 'versicolor', 1, 0)

dtrain <- xgboost::xgb.DMatrix(
	data = as.matrix(X_train),
	label = y_train
)
dtest <- xgboost::xgb.DMatrix(
	data = as.matrix(X_test),
	label = y_test
)

y_proba_train <- predict(model, dtrain)
y_proba_test <- predict(model, dtest)

auroc <- function(y_true, y_proba) {
	n1 <- sum(!y_true)
	n2 <- sum(y_true)
	U  <- sum(rank(y_proba)[!y_true]) - n1 * (n1 + 1) / 2
	return(1 - U / n1 / n2)
}

print(auroc(y_train, y_proba_train))
print(auroc(y_test, y_proba_test))
# 0.9989149 and 1, respectively, which is exactly what we saw in Python

# We can even compare the predicted probabilities to ensure they're basically equal
y_proba_train_python <- as.numeric(read.table('y_proba_train_python.txt')$V1)
y_proba_test_python <- as.numeric(read.table('y_proba_test_python.txt')$V1)

mean(y_proba_train - y_proba_train_python)
mean(y_proba_test - y_proba_test_python)
# These are both around 1.434159e-09, which is a *very* small number and is likely only due
# to the fact the predicted probabilities from Python were saved only up to the 7th decimal place,
# whereas the probabilities in R are unrounded (and go out about twice as many decimal places)
```

As we see above, we can be sure porting an XGBoost model from Python to R will result in the same predicted probabilities, assuming we apply the same input data transformation logic across both platforms.

## Trained in R, scored in Python

```R
library(xgboost)

data(iris)

# Use same indices from Python run above
train_idxs <- as.integer(read.table('train_idxs.txt')$V1) + 1
test_idxs <- as.integer(read.table('test_idxs.txt')$V1) + 1

# Note the data(iris) and load_iris() datasets between R and Python, respectively, are ordered the same way
X_train <- iris[train_idxs, 1:4]
X_test <- iris[test_idxs, 1:4]
# Target name index at 1 in sklearn's load_iris() is versicolor
y_train <- ifelse(iris[train_idxs, 5] == 'versicolor', 1, 0)
y_test <- ifelse(iris[test_idxs, 5] == 'versicolor', 1, 0)

dtrain <- xgboost::xgb.DMatrix(
	data = as.matrix(X_train),
	label = y_train
)
dtest <- xgboost::xgb.DMatrix(
	data = as.matrix(X_test),
	label = y_test
)

params <- list(
	learning_rate = 0.1,
	max_depth = 3,
	objective = 'binary:logistic'
)

model <- xgboost::xgb.train(
	params,
	dtrain,
	nrounds = 10
)

y_proba_train <- predict(model, dtrain)
y_proba_test <- predict(model, dtest)

print(auroc(y_train, y_proba_train))
print(auroc(y_test, y_proba_test))

xgboost::xgb.save(model, 'model.model')
```

Now to load the trained-in-R model into Python and predict:

```python
import xgboost
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score

data = load_iris()
X = data.data
y = np.where(data.target == 1, 1, 0)

train_idxs = np.array([int(idx) for idx in open('train_idxs.txt')])
test_idxs = np.array([int(idx) for idx in open('test_idxs.txt')])

X_train, X_test = X[train_idxs, :], X[test_idxs, :]
y_train, y_test = y[train_idxs], y[test_idxs]

dtrain = xgboost.DMatrix(X_train, y_train)
dtest = xgboost.DMatrix(X_test, y_test)

model = xgboost.Booster()
model.load_model('model.model')

y_proba_train = model.predict(dtrain)
y_proba_test = model.predict(dtest)

print(roc_auc_score(y_train, y_proba_train))
print(roc_auc_score(y_test, y_proba_test))
# Again, we get the same performance as we observed in R
```