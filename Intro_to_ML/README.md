# BASIC CONCEPTS OF MACHINE LEARNING

## I) Introduction, in-sample predictions

### 1) Prepare the data, and split it into training and validation data

```
# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
```

We can print columns of dataset using `home_data.columns`
```
print(home_data.columns)
```

We have to predict the `SalePrice` column. We can select the column we want to predict using `y = home_data.SalePrice`, this is the target
```
y = home_data.SalePrice
```

Now you will create a DataFrame called X holding the predictive features.
```
# Create the list of features below
feature_names = ["LotArea", "LotArea", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
​
# Select data corresponding to features in feature_names
X = home_data[feature_names]
```

To review Data we can use :
```
# print description or statistics from X
print(X.describe())

# print the top few lines
print(X.head())
```

### 2) Specify and Fit the Model

We can use `scikit-learn` to create our model. Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames.
We split the dataset into training and testing sets, create a DecisionTreeRegressor model, and train it on the training data. Then, we use the trained model to make predictions on the test data and calculate the mean squared error to evaluate the model's performance.able of performing multiple regression on a set of features.

```
from sklearn.tree import DecisionTreeRegressor
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model

iowa_model = DecisionTreeRegressor(random_state = 1)

# Fit the model

iowa_model.fit(X, y)
```

### 3) Make Predictions and verify the model

Make predictions with the model's predict command using X as the data. Save the results to a variable called predictions.

```
predictions = iowa_model.predict(X)
print(predictions)
```

"first in-sample predictions" refer to the predictions made by a model on the same dataset it was trained on. In other words, the model uses the training data to make predictions, and the predicted values are compared to the actual target values (ground truth) in the training dataset.

Compare the predictions to the actuals home values :
```
print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())
```

output:
```
First in-sample predictions: [208500. 181500. 223500. 140000. 250000.]
Actual target values for those homes: [208500, 181500, 223500, 140000, 250000]
```

We see that we have exactly the same values, because we used the same data to train and test the model.

## II) Model Validation

### 1) Split Your Data

```
# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
```

### 2) Specify and Fit the Model

```
# Specify the model
iowa_model = DecisionTreeRegressor(random_state = 1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)
```

### 3) Make Predictions with Validation data

```
val_predictions = iowa_model.predict(val_X)
```

We print the top few validation predictions :
```
print("Top Predictions from Validation Data:")
print(val_predictions[:num_rows_to_display])

print("Top Actual Target Values for Validation Data:")
print(val_y.head(num_rows_to_display).tolist())
```
output :
```Top Predictions from Validation Data:
[186500. 184000. 130000.  92000. 164500.]
Top Actual Target Values for Validation Data:
[231500, 179500, 122000, 84500, 142000]
```

We see that the predictions are different from the actual values, because we used different data to train and test the model.