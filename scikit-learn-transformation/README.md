# Building a robust raw data transformation pipeline

The goal of this tutorial is to demonstrate how to go from a raw, flat file to an analytic dataset using scikit-learn's transformation and pipeline utilities.

This tutorial uses scikit-learn version 0.20, and I don't think the version of pandas used will make any difference. Note that some of the scikit API features used are noted as "experimental" and could be subject to substantial change in the future.

Let's start with importing the right packages:

```python
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
```

Now, let's define a "raw" dataset that we want to transform:

```python
df = pd.DataFrame(
    {
            'a': [1, 'a', 1, np.nan, 'b'],
            'b': [1, 2, 3, 4, 5],
            'c': list('abcde'),
            'd': list('aaabb'),
            'e': [0, 1, 1, 0, 1],
    }
)
```

Next, we'll define our five "X" columns' strategies in a JSON-like format:

```python
strategies = [
	{
		'name': 'a',
		'kind': 'categorical',
		'default': None
	},
	{
		'name': 'b',
		'kind': 'continuous',
		'default': None
	},
	{
		'name': 'c',
		'kind': 'categorical',
		'default': None
	},
	{
		'name': 'd',
		'kind': 'categorical',
		'default': None
	},
	{
		'name': 'e',
		'kind': 'continuous',
		'default': None
	}
]
```

Interestingly, although the `dtype` of column `'a'` is `object`, the numbers within that Series aren't actually converted to string. This creates problems in later pipeline steps, so a fine way to circumvent this issue is to cast all `object` columns to `str`.

```python
for col in df.select_dtypes('object'):
	df[col] = df[col].astype(str)
```

Now, let's create our custom `ColumnTransformer` instance. We'll define strategies for both categorical and continuous predictors. Note that scikit supports the `'passthrough'` option for features not requiring any transforming, but still needing to be included in the final dataset. The default strategy for dealing with continuous predictors in this example is to do nothing, so we'll just pass through these ones.

```python
categorical_columns = [strategy['name'] for strategy in strategies if strategy['kind'] == 'categorical']
continuous_columns = [strategy['name'] for strategy in strategies if strategy['kind'] == 'continuous']

categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
continuous_transformer = 'passthrough'

column_transformer = ColumnTransformer(
	[
		('categorical', categorical_transformer, categorical_columns),
		('continuous', continuous_transformer, continuous_columns),
	]
	,
	sparse_threshold=0.,
	n_jobs=-1
)
```

Let's see how it works!

```python
column_transformer.fit_transform(df)
array([[1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0.],
       [0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 2., 1.],
       [1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 3., 1.],
       [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 4., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 5., 1.]])
```

## Using the transformer in production

Now let's simulate using this same transformer - used to transform the raw training data - in a scoring scenario. I'm going to add some new categories to columns `'a'` and `'c'`, and a new value to column `'e'`. Let's see how the `ColumnTransformer` from above handles this:

```python
score_df = pd.DataFrame(
    {
            'a': [1, 'a', 2, np.nan, 'c'],
            'b': [1, 2, 3, 4, 5],
            'c': list('abcze'),
            'd': list('aaabb'),
            'e': [0, 1, 1, 0, 2],
    }
)

# Just cast all columns parsed as objects to string
for col in score_df.select_dtypes('object'):
	score_df[col] = score_df[col].astype(str)

column_transformer.transform(score_df)
# array([[1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0.],
#        [0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 2., 1.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 3., 1.],
#        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 4., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 5., 2.]])
```

Notice how the transformer ignores the new categories and values, because we specified `handle_unknown='ignore'` in the initialization of the `OneHotEncoder` from above. What happens if we set `handle_unknown='error'`?

```python
categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='error')
continuous_transformer = 'passthrough'

column_transformer = ColumnTransformer(
	[
		('categorical', categorical_transformer, categorical_columns),
		('continuous', continuous_transformer, continuous_columns),
	]
	,
	sparse_threshold=0.,
	n_jobs=-1
)

column_transformer.fit_transform(df)
# array([[1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0.],
#        [0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 2., 1.],
#        [1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 3., 1.],
#        [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 4., 0.],
#        [0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 5., 1.]])

column_transformer.transform(score_df)
```

The follow `ValueError` should have been raised:

```python
ValueError: Found unknown categories ['2', 'c'] in column 0 during transform
```

So, depending on how you want to handle new categories, this is a really important option. I could envision two nearly identical pipelines to monitor for new values: one with `handle_unknown='ignore'` for ensuring that predictions are consistently provided; another with `handle_unknown='error'`, with try-catch logic for logging any caught exceptions for later inspection.

## Saving the pipeline for future use

We should be able to save this transformer for use in production, just like we would with any other scikit-learn estimator.

```python
import pickle

pickle.dump(column_transformer, open('column_transformer.pkl', 'wb'))

loaded_column_transformer = pickle.load(open('column_transformer.pkl', 'rb'))
loaded_column_transformer.transform(score_df)
# array([[1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0.],
#        [0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 2., 1.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 3., 1.],
#        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 4., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 5., 2.]])
```

## Useful links

* https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/
* https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
* https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
* https://scikit-learn.org/stable/modules/compose.html#column-transformer
