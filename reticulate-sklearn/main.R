library(reticulate)
library(stringr)
library(glue)
library(dplyr)

# Create virtualenv for R to use. Note: I suppose you could also use this https://rstudio.github.io/reticulate/articles/python_packages.html
system('mkdir ~/.virtualenvs')
system('python3 -m venv ~/.virtualenvs/venv')
system('source ~/.virtualenvs/venv/bin/activate && pip3 install pandas numpy scikit-learn xgboost && deactivate')

use_virtualenv('venv')

pd <- import('pandas')
np <- import('numpy')

sk_compose <- import('sklearn.compose')
sk_preprocessing <- import('sklearn.preprocessing')

data(mtcars)

df <- mtcars

feature_mapping <- data.frame(
	parent_feature = c('vs', 'mpg', 'gear'),
	kind = c('binary', 'continuous', 'categorical'),
	stringsAsFactors = FALSE
)

create_column_transformer <- function(feature_mapping){
	transformers <- list()
	for(i in 1:nrow(feature_mapping)){
		parent_feature <- feature_mapping[i, 'parent_feature']
		kind <- feature_mapping[i, 'kind']
		if(kind == 'binary'){
			transformer <- 'passthrough'
		}else if(kind == 'categorical'){
			transformer <- sk_preprocessing$OneHotEncoder()
		}else if(kind == 'continuous'){
			transformer <- sk_preprocessing$StandardScaler()
		}else{
			stop(glue('kind {kind} not supported'))
		}
		transformers[[i]] <- c(parent_feature, transformer, tuple(parent_feature))
	}
	return(sk_compose$ColumnTransformer(transformers))
}

column_transformer <- create_column_transformer(feature_mapping)
column_transformer$fit_transform(df)

get_transformer_column_order <- function(column_transformer){
	cols <- c()
	for(i in 1:length(column_transformer$transformers_)){
		cols <- c(cols, unlist(column_transformer$transformers_[i])[1])
	}
	cols <- unlist(cols)
	cols <- cols[cols != 'remainder']
	return(cols)
}

# Decompose `column_transformer` object into something purely native to R
# works only if transformers are specified on a per-column basis
decompose_column_transformer <- function(column_transformer){
	strategies <- list()
	for(col in get_transformer_column_order(column_transformer)){
		trans <- column_transformer$named_transformers_[[col]]
		trans_class <- as.character(trans)

		if(trans_class == 'OneHotEncoder()'){
			strategies[[col]] <- list(
				kind = 'categorical',
				strategy = 'one-hot-encode',
				levels = unlist(trans$categories_),
				feature_names = str_replace_all(unlist(trans$get_feature_names()), 'x0_', glue('{col}_'))
			)
		}else if(trans_class == 'StandardScaler()'){
			strategies[[col]] <- list(
				kind = 'continuous',
				strategy = 'scale',
				mean = trans$mean_,
				scale = trans$scale_,
				var = trans$var_,
				n_samples_seen = trans$n_samples_seen_,
				feature_names = c(col)
			)
		}else if(trans_class == 'passthrough'){
			strategies[[col]] <- list(
				kind = NA,
				strategy = 'passthrough',
				feature_names = c(col)
			)
		}
	}
	return(strategies)
}

strategies <- decompose_column_transformer(column_transformer)

column_transform <- function(df, strategies){
	X <- data.frame(to_drop = 1:nrow(df))
	for(col in names(strategies)){
		base_series <- df %>% pull(col)
		strategy <- strategies[[col]]$strategy
		if(strategy == 'one-hot-encode'){
			levels <- strategies[[col]]$levels
			feature_names <- strategies[[col]]$feature_names
			for(i in  1:length(levels)){
				level <- levels[i]
				feature_name <- feature_names[i]
				X[, feature_name] <- ifelse(base_series == level, 1, 0)
			}
		}else if(strategy == 'scale'){
			feature_name <- strategies[[col]]$feature_names
			X[, feature_name] <- (base_series - c(strategies[[col]]$mean)) / c(strategies[[col]]$scale)
		}else if(strategy == 'passthrough'){
			feature_name <- strategies[[col]]$feature_names
			X[, feature_name] <- base_series
		}
	}
	X <- X %>% select(-to_drop)
	return(X)
}

R_result_X <- column_transform(df, strategies)
python_result_X <- column_transformer$transform(df)

all(R_result_X == python_result_X)
