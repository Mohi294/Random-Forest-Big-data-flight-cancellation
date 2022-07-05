# flight-cancellation: a classification using google colab
this code is trying to predict whether a flight will be cancelled or not using random forest algorithm.

The documentations of main packages are available in following links:
- PySpark: https://spark.apache.org/docs/latest/api/python/
- hadoop: https://hadoop.apache.org/
- spark: https://spark.apache.org/documentation.html/

# Getting started
you can either upload data to drive from the beginning or download it directly from some sites that your datas are to the google drive using request for example:

```python
file_url = "some data url"

r = requests.get(file_url, stream = True) # stream=True let the data to come as chunks.

with open("/content/gdrive/My Drive/"your data information", "wb") as file: 
	for block in r.iter_content(chunk_size = 1024): 
		if block: 
			file.write(block)
```

then it is needed to load google drive so that google colaboratory recognizes it.

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

when the data is read, only the numerical data will be extracted and the train will be done on those data:
```python
feature_cols = ['_c0', 'Year', 'Month', 'DayofMonth',
                'DayOfWeek', 'CRSDepTime', 'CRSArrTime',
                'FlightNum', 'Diverted']

df = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(df)
```

the data must be splited into train and test so that calculating accuracy will be made possible:
```python
(trainingData, testData) = df.randomSplit([TRAINING_DATA_RATIO, 1 - TRAINING_DATA_RATIO])
```
 after that the model will be trained and used on test to get accuracy:
```python
# Train model
model = pipeline.fit(trainingData)

# Make predictions
predictions = model.transform(testData)
```

the accurancy will be calculated by;
```python
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Error = {(1.0 - accuracy):g}")
print(f"Accuracy = {accuracy:g}")
```





