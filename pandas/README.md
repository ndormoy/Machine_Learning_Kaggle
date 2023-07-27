# **Learning Pandas**

## I) **Pandas Intro**

### 1) **DataFrames**

We can create Pandas dataframe even if its not commun.

```
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits = pd.DataFrame({"Apples": [30], "Bananas" : [21]})
print(fruits)
```
|0  | Apples | Bananas |
|---|--------|---------|
| \- |   30   |   21    |

```
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = pd.DataFrame({"Apples": [35, 41], "Bananas": [21, 34]}, index=["2017 Sales", "2018 Sales"])
```

|             | Apples | Bananas |
|-------------|--------|---------|
| 2017 Sales  | 35     | 21      |
| 2018 Sales  | 41     | 34      |


### 2) **Series**

Create a variable ingredients with a Series that looks like:

Flour     4 cups
Milk       1 cup
Eggs     2 large
Spam       1 can
Name: Dinner, dtype: object

```
quantities = ['4 cups', '1 cup', '2 large', '1 can']
items = ['Flour', 'Milk', 'Eggs', 'Spam']
recipe = pd.Series(quantities, index=items, name='Dinner')
```

### 3) **Read from a CSV file**

- The **index_col** parameter in pd.read_csv() is used to specify which column from the CSV file should be used as the index of the DataFrame.  
- The index is used to uniquely label each row of the DataFrame and provides a way to access and reference rows based on their index values. 

Read the following csv dataset of wine reviews into a DataFrame called reviews.  
The filepath to the csv file is ../input/wine-reviews/winemag-data_first150k.csv.  


```
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col = 0)
```

### 4) **Create a CSV file with DataFrame :**

- We can create a CSV file with **DataFrame.to_csv()** method.

write code to save this DataFrame to disk as a csv file with the name cows_and_goats.csv.

```
animals.to_csv("cows_and_goats.csv")
```

## II) **Indexing, Selecting & Assigning**

We can view some data easily with pandas :
- **head()** : first 5 rows
- **df.mycol** : select a column --> same as df['mycol']
- **df['mycol'][0]** --> first row of column mycol


### 1) **Indexing**

- We use iloc to select rows and columns by position like in numpy.

```
df.iloc[0] # first row
df.iloc[:, 0] # first column
df.iloc[:3, 0] # first 3 rows of first column
df.iloc[[0, 1, 2], 0] # first 3 rows of first column
df.iloc[-5:] # last 5 rows
...
```

### 2) **Label-based selection**

- We use **loc** to select rows and columns by label.
- it's the data index value, not its position, which matters.

For example, to get the first entry in reviews, we would now do the following:
```
reviews.loc[0, 'country'] # Same as reviews.iloc[0, 0]
```

### 2bis) **Difference between loc and iloc**

In pandas, both loc and iloc are used for accessing rows and columns in a DataFrame, but they use different methods of indexing:

***loc*: Label-based indexing**

- loc is used for selection based on row and column labels. It allows you to select data by specifying the row labels and column labels explicitly.
- When using loc, the row and column labels are inclusive (i.e., both start and end indices are included in the selection).
- The syntax for loc is: df.loc[row_label, column_label]

***iloc*: Integer-based indexing**

- iloc is used for selection based on integer positions (i.e., numerical index positions) of rows and columns.
- When using iloc, the row and column integer positions are exclusive for the end index (i.e., the end index is not included in the selection).
- The syntax for iloc is: df.iloc[row_index, column_index]

**loc :**

```
# Access row with label 1 and column with label 'B'
print(df.loc[1, 'B'])
# Output: 200

# Access rows with labels 2 to 4 and columns with labels 'A' and 'C'
print(df.loc[2:4, ['A', 'C']])
# Output:
#     A  C
# 2  30  Z
# 3  40  W
# 4  50  V
```

**iloc:**

```
# Access row at integer position 1 and column at integer position 1
print(df.iloc[1, 1])
# Output: 200

# Access rows at integer positions 2 to 4 and columns at integer positions 0 and 2
print(df.iloc[2:5, [0, 2]])
# Output:
#     A  C
# 2  30  Z
# 3  40  W
# 4  50  V
```

### 3) **Manipulating the index**

- The index we use is not immutable. We can manipulate the index in any way we see fit.

The ***set_index()*** method can be used to do the job. Here is what happens when we set_index to the title field:
```
reviews.set_index("title")
```

### 4) **Conditional selection**

For example, suppose that we're interested specifically in better-than-average wines produced in Italy.

We can start by checking if each wine is Italian or not:

```
reviews.country == 'Italy' 
```
This operation produced a Series of True/False booleans based on the country of each record
We can use it into loc :
```
reviews.loc[reviews.country == 'Italy'] # Show all the italian wines
```

Create a variable df containing the country, province, region_1, and region_2 columns of the records with the index labels 0, 1, 10, and 100.  
In other words, generate the following DataFrame:
```
cols = ['country', 'province', 'region_1', 'region_2']
indices = [0, 1, 10, 100]
df = reviews.loc[indices, cols]
# or -->  df = reviews.loc[[0, 1, 10, 100], ["country", "province", "region_1", "region_2"]]
```

- We can use the ampersand (&) to bring the two questions together:
```
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
```
- We can use the pipe | character to bring the two questions together.

Pandas comes with a few built-in conditional selectors, two of which we will highlight here.

- isin is lets you select data whose value "is in" a list of values.
For example, here's how we can use it to select wines only from Italy or France:
```
reviews.loc[reviews.country.isin(['Italy', 'France'])]
```
- isnull, notnull


Create a DataFrame top_oceania_wines containing all reviews with at least 95 points (out of 100) for wines from Australia or New Zealand.

```
top_oceania_wines = reviews.loc[(reviews.country.isin(["Australia", "New Zealand"])) & (reviews.points >= 95)]
```

### 5) **Assigning data**

We can assign data to a DataFrame in many ways :

- assign either a ***constant*** value:
```
reviews['critic'] = 'everyone'
```
- assign an ***iterable*** of values:
```
reviews['index_backwards'] = range(len(reviews), 0, -1)
```

## III) **Summary functions and maps**

### 1) **Summary functions**

- **describe** : df.columnname.describe() --> return a ***series*** with some statistics about the column
- **mean** : df.columnname.mean() --> return the ***mean*** of the column
- **unique** : df.columnname.unique() --> return an ***array*** with all the ***unique values*** of the column
- **value_counts** : df.columnname.value_counts() --> return a series with all the unique values of the column and the ***number of times they appear***

### 2) **Maps**

the map() function is used to ***transform values*** in a Series or DataFrame based on a specified mapping or function.  
It allows you to replace each value with another value according to a defined mapping.

The syntax for using **map()** in pandas depends on whether you are using it with a Series or a DataFrame:

#### a) **Serie**
```
import pandas as pd
# Original Series
fruits = pd.Series(['apple', 'orange', 'banana', 'apple'])
# Define the mapping dictionary
mapping = {'apple': 'red', 'orange': 'orange', 'banana': 'yellow'}
# Use map() to transform the Series
new_fruits = fruits.map(mapping)
print(new_fruits)
```
output:
```
0      red
1   orange
2   yellow
3      red
dtype: object
```
#### b) **DataFrame**
```
import pandas as pd

# Original DataFrame
data = {
    'Fruit': ['apple', 'orange', 'banana', 'apple'],
    'Color': ['red', 'orange', 'yellow', 'red']
}

df = pd.DataFrame(data)

# Define the mapping dictionary
mapping = {'apple': 'red', 'orange': 'orange', 'banana': 'yellow'}

# Use map() to transform the 'Fruit' column
df['Fruit'] = df['Fruit'].map(mapping)

print(df)
```
output:
```
    Fruit    Color
0     red      red
1  orange   orange
2  yellow   yellow
3     red      red
```

#### c) **apply()**

apply() is the equivalent method if we want to transform a whole DataFrame by calling a custom method on each row.
```
def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')
```
The **axis** parameter in apply() can take two possible values:

- ***axis=0*** or ***axis='index'***: Apply the function to each column (operate on rows).
- ***axis=1*** or ***axis='columns'***: Apply the function to each row (operate on columns).


### d) **idxmax()**

Is used to find the index (row label) of the first occurrence of the ***maximum value*** in a Series or DataFrame.  
It returns the label (index) of the row where the maximum value occurs.

```
import pandas as pd

# Create a DataFrame
data = {
    'A': [10, 25, 15, 30, 20],
    'B': [5, 15, 8, 12, 10]
}
df = pd.DataFrame(data)
# Find the index of the maximum value for each column in the DataFrame
max_index = df.idxmax()
print("Index of maximum value for each column:")
print(max_index)
```

### e) **.loc**

***.loc*** is a label-based indexing method that is used to access rows and columns in a DataFrame by their labels (row and column names).  
It allows you to select data based on the row and column labels, rather than using numerical index positions (which is done with .iloc).

The basic syntax for using **.loc** is:

```
df.loc[row_label, column_label]
```
- df: The pandas DataFrame you want to access data from.
- row_label: The label (name) of the row you want to access.
- column_label: The label (name) of the column you want to access.
