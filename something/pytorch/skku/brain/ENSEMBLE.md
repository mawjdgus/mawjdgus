REFERENCE : https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/

# A Comprehensive Guide to Ensemble Learning ( with Python codes )

## Introduction

Combine the decisions from multiple models to improve the overall performance. 

## Table of Contents
1. Introduction to Ensemble Learning
2. Basic Ensemble Techniques
  2.1. Max Voting
  2.2. Averageing
  2.3. Weighted Average
3. Advanced Ensemble Techniques
  3.1 Stacking
  3.2. Blending
  3.3. Bagging
  3.4. Boosting
4. Algorithms based on Bagging and Boosting
  4.1. Bagging meta-estimator
  4.2. Random Forest
  4.3. AdaBoost
  4.4. GBM
  4.5. XGB
  4.6. Light GBM
  4.7. CatBoost
  
## 1. Introduction to Ensemble Learning

You can infer that a diverse group of people are likely to make better decisions as compared to individuals.
Similar is true for a diverse set of models in comparison to single models. This diversification in Machine Learning is achieved by a technique called Ensemble Learning.

## 2. Simple Ensemble Techniques

1. Max Voting
2. Averaging
3. Weighted Averaging

## 2.1 Max Voting

The max voting method is generally used for classification problems. In this technique, multiple models are used to make predictions for each data point.
The predictions by each model are considered as a 'vote'. The predictions which we get from the majority of the models are used as the final prediction.

For example, when you asked 5 of your colleagues to rate your movie (out of 5); we’ll assume three of them rated it as 4 while two of them gave it a 5. Since the majority gave a rating of 4, the final rating will be taken as 4. You can consider this as taking the mode of all the predictions.

The result of max voting would be something like this:

Colleague 1	Colleague 2	Colleague 3	Colleague 4	Colleague 5	Final rating

5          	4	          5          	4         	4        	4

```python
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict(x_test)
pred2=model2.predict(x_test)
pred3=model3.predict(x_test)

final_pred = np.array([])
for i in range(0,len(x_test)):
    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))
```

Alternatively, you can use "VotingClassifier" module in sklearn as follows:
```python
from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(x_train,y_train)
model.score(x_test,y_test)
```

## 2.2 Averaging

Similar to the max voting technique, multiple predictions are made for each data point in averaging. In this method, we take an average of predictions from all the models and use it to make the final prediction. Averaging can be used for making predictions in regression problems or while calculating probabilities for classification problems.

For example, in the below case, the averaging method would take the average of all the values.

i.e. (5+4+5+4+4)/5 = 4.4

Colleague 1	Colleague 2	Colleague 3	Colleague 4	Colleague 5	Final rating

5	4	5	4	4	4.4

```python
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1+pred2+pred3)/3
```

## 2.3 Weighted Average

This is an extension of the averaging method. All models are assigned different weights defining the importance of each model for prediction. For instance, if two of your colleagues are critics, while others have no prior experience in this field, then the answers by these two friends are given more importance as compared to the other people.

The result is calculated as [(5*0.23) + (4*0.23) + (5*0.18) + (4*0.18) + (4*0.18)] = 4.41.

Colleague 1	Colleague 2	Colleague 3	Colleague 4	Colleague 5	Final rating

weight	0.23	0.23	0.18	0.18	0.18

rating	5	4	5	4	4	4.41

```python
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)
```
