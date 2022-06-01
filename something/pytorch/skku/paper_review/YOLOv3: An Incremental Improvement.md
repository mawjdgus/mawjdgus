# YOLOv3 : An Incremental Improvement

## Abstract

Update at YOLO. Authors made a bunch of little design changes to make it better. They also trained this new network that's pretty sweel. It's a little bigger than last time but more accurate.
It's still fast though, At 320 x 320 YOLOv3 runs in **22 ms at 28.2 mAP, as accurate as SSD but three times faster**.

## 1. Introduction

1. Tell you what the deal is with YOLOv3.
2. Tell you how they do
3. Tell you about some things we tried that didn't work.
4. Complete what this all means.

## 2. The Deal

![image](https://user-images.githubusercontent.com/67318280/135716142-90fe4cfc-abcc-426e-a841-9a1397742a5c.png)

### 2.1. Bounding Box Prediction

![image](https://user-images.githubusercontent.com/67318280/135716180-ba1a1360-806e-471b-ad79-e5ffaf434209.png)

### 2.2. Class Prediction

Each Box predicts the classes the bounding box may contain using multilabel classification. They do not use a softmax as they have found it is unnecessary for good performance, instead they simply use independent logistic classifiers.
During training they use binary cross-entropy loss for the class predictions.<br>
This formulation helps when they move to more complex domains like the Open Images Dataset.
In this datset there are many overlapping labels (i.e. Woman and Person).
Using a softmax imposes the assumption that each box has exactly one class which is often not the case. A multilabel approach better models the data.

