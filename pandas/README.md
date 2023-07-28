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


#### d) **idxmax()**

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

#### e) **.loc**

***.loc*** is a label-based indexing method that is used to access rows and columns in a DataFrame by their labels (row and column names).  
It allows you to select data based on the row and column labels, rather than using numerical index positions (which is done with .iloc).

The basic syntax for using **.loc** is:

```
df.loc[row_label, column_label]
```
- df: The pandas DataFrame you want to access data from.
- row_label: The label (name) of the row you want to access.
- column_label: The label (name) of the column you want to access.

## IV) **Grouping and sorting**

### 1) **Groupwise analysis**

- **groupby** : df.groupby('columnname').method() --> return a ***groupby object*** that can be used to group together rows based off of a column and then apply a function to each group separately. 
We can use multiple functions on a groupby :
- groupby().size
- groupby().count
- groupby().min
- groupby().max
- [...]

For an example, here's how we would pick out the best wine by country and province:
```
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
```

- **agg** : df.groupby('columnname').agg() --> return a ***dataframe*** , which lets you run a bunch of different functions on your DataFrame simultaneously.

For example, we can generate a simple statistical summary of the dataset as follows:
```
reviews.groupby(['country']).price.agg([len, min, max])
```

| country   | len | min | max   |
|-----------|-----|-----|-------|
| Argentina | 3800 | 4.0 | 230.0 |
| Armenia   | 2   | 14.0 | 15.0  |
| ...       | ... | ... | ...   |
| Ukraine   | 14  | 6.0 | 13.0  |
| Uruguay   | 109 | 10.0 | 130.0 |

### 2) **Multi-indexes**

Instead of having a single row index (like a standard DataFrame), a ***multi-index*** DataFrame has two or more levels of row indices, allowing you to store and access data in a hierarchical structure.

- A multi-index allows you to efficiently organize and manage complex datasets with multiple dimensions or categories.
- It enables you to perform ***advanced indexing*** and slicing operations along each level of the index, making data retrieval more flexible.
- It is particularly useful when dealing with panel data, time series, or any data with ***hierarchical relationships***.

for example with this we have multi-indexes:
```
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
```

However, in general the multi-index method you will use most often is the one for converting back to a regular index, the ***reset_index()*** method:
```
countries_reviewed.reset_index()
```

### 3) **Sorting**

Looking again at countries_reviewed we can see that grouping returns data in index order, not in value order. That is to say, when outputting the result of a groupby, the order of the rows is dependent on the values in the index, not in the data.

To get data in the order want it in we can sort it ourselves. The ***sort_values()*** method is handy for this.

```
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')
```

| country | province        | len |
|--------|-----------------|-----|
| Greece | Muscat of Kefallonian | 1 |
| Greece | Sterea Ellada        | 1 |
| ...    | ...                 | ... |
| US     | Washington          | 8639 |
| US     | California          | 36247 |

- sort_values() defaults to an ascending sort, where the lowest values go first.  
However, most of the time we want a descending sort, where the higher numbers go first. That goes thusly:

```
countries_reviewed.sort_values(by='len', ascending=False)
```

Finally, know that you can sort by more than one column at a time:
```
countries_reviewed.sort_values(by=['country', 'len'])
```

Parameters:

- by: This parameter specifies the column(s) based on which the sorting should be performed. It can be a column name (string or list of strings) or a list of column names if you want to sort by multiple columns.
- axis: The axis along which the sorting should be performed. It can take two possible values:
- axis=0 (default): Sorts the rows based on the values in the specified column(s).
- axis=1: Sorts the columns based on the values in the specified row(s).
- ascending:  If True (default), the data is sorted in ascending order. If False, the data is sorted in - descending order.
- inplace: By default, it is False, which means the function returns a new DataFrame. If you set it to True, the sorting is done in place, and the original DataFrame is modified.
- ignore_index: A boolean value that determines whether to reset the index of the sorted DataFrame. By default, it is False, which means the index of the original DataFrame is preserved in the sorted DataFrame. If you set it to True, a new default integer index is assigned to the sorted DataFrame.

example:
What is the best wine I can buy for a given amount of money?  
Create a Series whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review.  
Sort the values by price, ascending (so that 4.0 dollars is at the top and 3300.0 dollars is at the bottom).

```
best_rating_per_price = reviews.groupby("price")["points"].max().sort_index()
```

In this code, we first group the DataFrame reviews by the "price" column using groupby("price"). Then, we select the "points" column by specifying ["points"] after the groupby operation.  
Finally, we apply the max() function to the "points" column to get the maximum points for each price level, and the result is sorted based on the "price" values using sort_index().  
This will give you the desired output, which is the maximum points for each unique price level.

Other example :
What are the minimum and maximum prices for each variety of wine? Create a DataFrame whose index is the variety category from the dataset and whose values are the min and max values thereof.

```
price_extremes = reviews.groupby("variety")["price"].agg([min, max])
```


#### **To summarize:**

**groupby() multi-index**
```
df.groupby("columname")["othercolumn"].method()
```
**groupby()**
```
df.groupby(["columnname", "othercolumnname"]).othercolumn.method()
```

## V) **Data types and missing values**

### 1) **Dtypes**

The data type for a column in a DataFrame or a Series is known as the dtype.

```
reviews.price.dtype
```

Alternatively, the dtypes property returns the dtype of every column in the DataFrame:
```
reviews.dtypes
```

#### convert type

```
reviews.points.astype('float64')
```

### 2) **Missing data**

- NaN values are always of the float64 dtype.
- To select NaN entries you can use pd.isnull() (or its companion pd.notnull())

example :
```
reviews[pd.isnull(reviews.country)]
```

#### a) **Replacing missing values**

**fillna()**

For example, we can simply replace each NaN with an "Unknown":

```
reviews.region_2.fillna("Unknown")
```

#### b) **Replacing existing values**

**replace()**

For example, suppose that since this dataset was published, reviewer Kerin O'Keefe has changed her Twitter handle from @kerinokeefe to @kerino

```
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
```

## VI) **Renaming and combining**

### 1) **Renaming**

**rename()**
which lets you change index names and/or column names.  
For example, to change the points column in our dataset to score, we would do:

```
reviews.rename(columns={'points': 'score'})
```

- rename() lets you rename index or column values by specifying a index or column keyword parameter, respectively.  
- It supports a variety of input formats, but usually a Python dictionary is the most convenient.

Here is an example using it to rename some elements of the index.
```
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})
```
You'll probably rename columns very often, but rename index values very rarely. For that, ***set_index()*** is usually more convenient.

Both the row index and the column index can have their own name attribute. The complimentary ***rename_axis()*** method may be used to change these names. For example:
```
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')
```

### 2) **Combining**

When performing operations on a dataset, we will sometimes need to combine different DataFrames and/or Series in non-trivial ways.  
Pandas has three core methods for doing this.  
In order of increasing complexity, these are ***concat(), join(), and merge()***

#### a) **concat()**

The simplest combining method is ***concat()***.  
Given a list of elements, this function will smush those elements together along an axis.

If we want to study multiple countries simultaneously, we can use concat() to smush them together:
```
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")
pd.concat([canadian_youtube, british_youtube])
```

#### b) **join()**

Lets you combine different DataFrame objects which have an index in common.

For example, to pull down videos that happened to be trending on the same day in both Canada and the UK, we could do the following:

```
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')
```

The ***lsuffix*** and ***rsuffix*** parameters are necessary here because the data has the same column names in both British and Canadian datasets.  
If this wasn't true (because, say, we'd renamed them beforehand) we wouldn't need them.