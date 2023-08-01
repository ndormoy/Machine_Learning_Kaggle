# INTERMEDIATE CONCEPTS OF MACHINE LEARNING

## I) Missing Values : Three Approaches

### 1) A Simple Option: Drop Columns with Missing Values

The simplest option is to ***drop*** columns with missing values.  
The model loses access to a lot of (potentially useful!) information with this approach.

### 2) A Better Option: Imputation

Imputation ***fills in the missing values*** with some number. For instance, we can fill in the mean value along each column.

The imputed value won't be exactly right in most cases, but it usually leads to more accurate models than you would get from dropping the column entirely.

### 3) An Extension To Imputation

In this approach, we impute the missing values, as before. And, additionally, for each column with missing entries in the original dataset, we add a new column that shows the location of the imputed entries.

In some cases, this will meaningfully improve results. In other cases, it doesn't help at all.

### Examples for the three approaches

#### Base
```
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
```

#### Define Function to Measure Quality of Each Approach
We define a function score_dataset() to compare different approaches to dealing with missing values.  
This function reports the mean absolute error (MAE) from a random forest model.

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```

### 1bis) Drop Columns with Missing Values

```
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```
- axis=1 drops the entire column.
- La méthode drop() de pandas ne modifie pas le DataFrame d'origine (X_train dans ce cas), mais renvoie une copie du DataFrame avec les colonnes 

MAE from Approach 1 (Drop columns with missing values):
183550.22137772635


### 2bis) Imputation

```
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```
- La classe SimpleImputer permet de remplacer les valeurs manquantes par une valeur spécifiée ou par une statistique (moyenne, médiane, etc.) calculée à partir des autres valeurs présentes dans la colonne.

- my_imputer.fit_transform(X_train) : Le résultat est un nouveau DataFrame, imputed_X_train, qui contient les valeurs imputées pour les données d'entraînement.

- Le résultat est un nouveau DataFrame, imputed_X_valid, qui contient les valeurs imputées pour les données de validation.

MAE from Approach 2 (Imputation):
178166.46269899711

### 3bis) An Extension To Imputation

```
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
```
MAE from Approach 3 (An Extension to Imputation):
178927.503183954


## II) Categorical Variables

#### Get list of categorical variables :

```
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
```

### 1) Drop Categorical Variables

Remove them from the dataset. This approach will only work well if the columns did not contain useful information.

We drop the object columns with the select_dtypes() method.


```
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
```

### 2) Ordinal Encoding

Ordinal encoding assigns each unique value to a different integer.
This approach assumes an ordering of the categories: 
- "Never" (0) 
- < "Rarely" (1)
- < "Most days" (2)
- < "Every day" (3).

Scikit-learn has a OrdinalEncoder class that can be used to get ordinal encodings.  
We loop over the categorical variables and apply the ordinal encoder separately to each column.

```
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
```

### 3) One-Hot Encoding

One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data.  
To understand this, we'll work through an example.

|Color   |
|--------|
| Red    |
| Red    |
| Yellow |
| Green  |
| Yellow |

To

| Red   | Yellow | Green |
|-------|--------|-------|
| 1     | 0      | 0     |
| 1     | 0      | 0     |
| 0     | 1      | 0     |
| 0     | 0      | 1     |
| 0     | 1      | 0     |


We use the OneHotEncoder class from scikit-learn to get one-hot encodings. There are a number of parameters that can be used to customize its behavior.

- set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data, 
- setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).  
To use the encoder, we supply only the categorical columns that we want to be one-hot encoded.

```
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
```

In general, one-hot encoding will typically perform best, and dropping the categorical columns typically performs worst, but it varies on a case-by-case basis.

## III) Pipelines

Pipelines are a simple way to keep your data preprocessing and modeling code organized.  
Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.

- ***Cleaner Code*** : Accounting for data at each step of preprocessing can get messy. With a pipeline, you won't need to manually keep track of your training and validation data at each step.
- ***Fewer Bugs*** : There are fewer opportunities to misapply a step or forget a preprocessing step.
- ***Easier to Productionize***: It can be surprisingly hard to transition a model from a prototype to something deployable at scale. We won't go into the many related concerns here, but pipelines can help.
- ***More Options*** for Model Validation: You will see an example in the next tutorial, which covers cross-validation.

### 1) Define Preprocessing Steps

```
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```

- imputes missing values in numerical data, and
- imputes missing values and applies a one-hot encoding to categorical data.

### 2) Define the Model

```
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
```

### 3) Create and Evaluate the Pipeline

```
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```

- With the pipeline, we preprocess the training data and fit the model in a single line of code
- With the pipeline, we supply the unprocessed features in X_valid to the predict() command, and the pipeline automatically preprocesses the features before generating predictions.

## IV) Cross-Validation

### 1) What is Cross-Validation?

In cross-validation, we run our modeling process on different subsets of the data to get multiple measures of model quality.

![img](elements/crossvalidation.png)

### 2) When Should You Use Cross Validation?

Cross-validation gives a more accurate measure of model quality --> but longer to run.

- For small datasets, where extra computational burden isn't a big deal, you should run cross-validation.
- For larger datasets, a single validation set is sufficient.  
Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.

```
import pandas as pd
# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')
# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
# Select target
y = data.Price
```

define pipeline :

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])
```

Use ***cross_val_score()*** to obtain the MAE for your pipeline.

```
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
```
- cv = 5 for 5-fold cross-validation (4 folds for training and 1 fold for validation)

output :
```
MAE scores:
 [301628.7893587  303164.4782723  287298.331666   236061.84754543
 260383.45111427]
```


We typically want a single measure of model quality to compare alternative models. So we take the average across experiments.
```
print("Average MAE score (across experiments):")
print(scores.mean())
```
output :
```
Average MAE score (across experiments):
277707.3795913405
```

## V) XGBoost

### **Gradient Boosting**

***Gradient boosting*** is a method that goes through cycles to iteratively add models into an ensemble.

![img](elements/Gradient_boosting.png)

Here we use ***XGBoost library*** to build the model.

```
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)
```

We also make predictions and evaluate the model.

```
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
```
output :
```
Mean Absolute Error: 241041.5160392121
```

### ***Parameter Tuning***

#### a) ***n_estimators***

Specifies how many times to go through the modeling cycle described above.  
It is equal to the number of models that we include in the ensemble.

- ***Too low a value causes underfitting***, which leads to inaccurate predictions on both training data and test data.
- ***Too high a value causes overfitting***, which causes accurate predictions on training data, but inaccurate predictions on test data (which is what we care about).

Typical values range from 100-1000, though this depends a lot on the learning_rate

```
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)
```

#### b) **early_stopping_rounds**

early_stopping_rounds offers a way to ***automatically find the ideal value for n_estimators***.
Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping --> early_stopping_rounds=5 is a reasonable.

```
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
```

#### c) **learning_rate**

We can multiply the predictions from each model by a small number (known as the learning rate) before adding them in.  
This means each tree we add to the ensemble helps us less. So, ***we can set a higher value for n_estimators without overfitting***.  
If we use early stopping, the appropriate ***number of trees will be determined automatically***.  
In general, a ***small learning rate and large number of estimators will yield more accurate*** XGBoost models

```
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```

#### d) **n_jobs**

On larger datasets where runtime is a consideration, you can use parallelism ***to build your models faster***.  
It's common to set the parameter ***n_jobs equal to the number of cores on your machine***. On smaller datasets, this won't help.  
It's ***useful in large datasets*** where you would otherwise spend a long time waiting during the fit command.

```
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```

## VI) Data Leakage

***Data leakage*** (or leakage) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction.  
This leads to high performance on the training set (and possibly even the validation data), but ***the model will perform poorly in production***.

### 1) Target leakage

Target leakage occurs when your predictors include data that will ***not be available at the time you make predictions***.  
It is important to think about target leakage in terms of the timing or chronological order that data becomes available, not merely whether a feature helps make good predictions.

![img](elements/target_leakage.png)

#### Find target leakage

If we have a Cross-validation too much high score, it is a sign of target leakage.
Then we have to find the features that are highly correlated with the target.
Drop the features that are highly correlated with the target to avoid target leakage.

### 2) Train-Test Contamination

A different type of leak occurs when you aren't careful to distinguish training data from validation data.
For example, imagine you run preprocessing (like fitting an imputer for missing values) before calling train_test_split().  
The end result? Your model may get good validation scores, giving you great confidence in it, but perform poorly when you deploy it to make decisions.
