
# Decision Trees

Implementation of Decision Trees by finding the best (feature, threshold) pair when splitting a node.

The split criterion is based on maximizing the information gain.

Since the features are numeric values, the algorithm also needs input threshold mechanisms which act as halting conditions beyond which the split does not take place.

## Documentation

### Load the spambase dataset and split into train and test


```python
from Datasets import spambase
filename = "data/Spambase dataset/spambase.data"
train, test = spambase(filename)
```

### Setup model (following parameters are default)


```python
from DecisionTree import DecisionTree
model = DecisionTree(entropy_threshold = 0.3, rowcount_threshold = 10, depth_threshold = 7)
```

### Train model


```python
model.fit(train, model)
```

### Predict new observations


```python
prediction = model.predict_row(model, test.iloc[[0]])
```

### Testing the accuracy of the model on the train and test set


```python
test_predictions = model.predict(model, test)
test_accuracy = model.accuracy(test, test_predictions)

train_predictions = model.predict(model, train)
train_accuracy = model.accuracy(train, train_predictions)
```
