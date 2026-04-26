# CompSci-kNN-Activity-Group_2-Guleng_Rico_Hadjinor

## Analysis and Reflection 
Strengths of KNN
KNN is one of the easiest machine learning algorithms to understand. The idea is simple: classify a new record based on what its nearest neighbors are. You do not need complex math or assumptions about the data. KNN also works well when the boundary between classes is curved or irregular, since it learns directly from the data without fitting a fixed formula. It has no real training step – it just stores the data and computes distances when a prediction is needed, which means adding new data is as easy as adding rows to the dataset.

## Limitations of KNN
KNN becomes slow when the dataset is large because it must compute the distance to every training record for each prediction. It is also very sensitive to scaling – if features have different ranges and you skip standardization, the large-range features will control the distances and the small-range features become useless. Another weakness is that KNN stores all training data in memory, which uses a lot of RAM. Finally, if the dataset has more records from one class (like 500 non-diabetic vs 268 diabetic here), KNN tends to predict the majority class more often.

## When KNN Is and Is Not a Good Choice
Use KNN when:

•	The dataset is small or medium in size.

•	You need a simple model to start with before trying more complex ones.

•	The relationship between features and the outcome is not linear.


## Do not use KNN when:
•	The dataset has millions of records – prediction will be too slow.

•	You need fast real-time predictions in a live application.

•	The data has many features without feature selection.

•	Memory is limited, since KNN must store all training data.


## Observations from the Experiment
The KNN model reached its best accuracy of 73.38% at K=5. The key findings from this experiment are:

4.	Insulin had the worst missing data problem with 374 zeros (48.7% of the dataset). After replacing these with the median value of 125, the mean of Insulin rose from 79.80 to 140.67.
5.	Standardization was the most important preprocessing step. Without it, Insulin (range 14–846) would have been roughly 360 times more influential than DiabetesPedigreeFunction (range 0.08–2.42) in the distance formula.
6.	K=5 gave the best balance. K=3 was slightly noisy and K=7 was slightly over-smoothed.
7.	The model is better at finding non-diabetic patients (TN=74–75) than diabetic ones (TP=35–38). This is partly because non-diabetic records outnumber diabetic ones in the dataset (500 vs 268).
8.	For medical use, missed diabetic diagnoses (False Negatives) are the most dangerous type of error. At K=5, only 17 diabetic patients were missed.
 
