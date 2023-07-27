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

We can use `scikit-learn` to create our model. Scikit-learn is easily the most popular  
library for modeling the types of data typically stored in DataFrames.
We split the dataset into training and testing sets, create a DecisionTreeRegressor model, and train it on the training data.  
Then, we use the trained model to make predictions on the test data and calculate the mean  
squared error to evaluate the model's performance.able of performing multiple regression on a set of features.

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

## II) Model Validation / Validation data / Mean Absolute Error

### 1) Split Your Data

The purpose of splitting the dataset into training and testing sets is to evaluate the model's performance on data it has never seen before.  
If the model is evaluated on the same data it was trained on (in-sample evaluation),  
it may give an overly optimistic performance measure and not provide a realistic indication  
of how well it will perform on new, unseen data (out-of-sample evaluation).

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
# Choose the number of rows you want to display, for example, 5 rows
num_rows_to_display = 5 
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

4) Calculate the Mean Absolute Error in Validation Data

Mean Absolute Error (MAE) is a metric used to measure the accuracy of a predictive model, particularly in regression tasks.  
It quantifies the average absolute difference between the predicted values and the actual values in a dataset.  
The smaller the MAE, the closer the model's predictions are to the actual values, indicating a more accurate model.

```
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)
```

## III) Underfitting and Overfitting

#### Overfitting :

- Where a model matches the training data almost perfectly, but does poorly in validation and other new data,  
that it negatively impacts the performance of the model on new data.
- The primary goal of a machine learning model is to make accurate predictions on new and unseen data, not just on the data it was trained on.  
Overfitting occurs when the model becomes overly complex and tries to fit the noise and individual data points in the training set,  
resulting in poor generalization to new data.

#### Signs of Overfitting:

- The model's performance is excellent on the training set but significantly worse on the test set (or validation set).
- The model has too many parameters or features relative to the amount of training data, leading to overemphasis on specific data points and patterns.
- The model captures outliers and noise in the training data, leading to unrealistic predictions on new data.
- The model's performance degrades significantly when evaluated on unseen data compared to the training data.

#### Preventing Overfitting:

- Use a larger dataset if possible. More data can help the model learn more generalized patterns.
- Feature selection and engineering: Choose relevant and important features while avoiding irrelevant ones.
- Cross-validation: Use techniques like k-fold cross-validation to evaluate the model's performance on multiple subsets of the data.
- Regularization: Apply techniques like L1 or L2 regularization to penalize complex models and prevent them from fitting noise.
- Hyperparameter tuning: Tune the model's hyperparameters to find the right balance between model complexity and generalization.

#### Underfitting :

- At an extreme, if a tree divides houses into only 2 or 4, each group still has a wide variety of houses.  
Resulting predictions may be far off for most houses, even in the training data (and it will be bad in validation too for the same reason).  
When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called underfitting.

#### max leaf nodes :

- max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting.

### 1) Define a function to get MAE

```
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

### 2) Compare Different Tree Sizes

```
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for max_leaf_nodes in candidate_max_leaf_nodes:
    print(f"max_leaf_nodes = {max_leaf_nodes} | mae =  {get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)}")

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = 100
```

output :
```
max_leaf_nodes = 5 | mae =  35044.51299744237
max_leaf_nodes = 25 | mae =  29016.41319191076
max_leaf_nodes = 50 | mae =  27405.930473214907
max_leaf_nodes = 100 | mae =  27282.50803885739
max_leaf_nodes = 250 | mae =  27893.822225701646
max_leaf_nodes = 500 | mae =  29454.18598068598
```
### 3) Fit Model Using All Data

Fit with the ideal value of max_leaf_nodes. In the fit step, use all of the data in the dataset

```
# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state = 1)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)
```

## IV) Random Forests

- Random forests are a type of ensemble learning method, where a group of weak models combine to form a powerful model.
- The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree.
- It generally has much better predictive accuracy than a single decision tree and it works well with default parameters.

We build a random forest model similarly to how we built a decision tree in scikit-learn - this time using the RandomForestRegressor class instead of DecisionTreeRegressor.

### Define the model using Random Forest

```
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state = 1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(rf_model.predict(val_X), val_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
```

output :
```
Validation MAE for Random Forest Model: 21857.15912981083
```

## V) Machine Learning Competitions

### Train the model for competition

The code cell above trains a Random Forest model on train_X and train_y.

Use the code cell below to build a Random Forest model and train it on all of X and y.

```
# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state = 1)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# Run the code to save predictions in the format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
```

