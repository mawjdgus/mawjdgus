REFERENCE :https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

# mAP (mean Averag Precision) for object Detection

AP (Average precision) is a popular metric in measuring the accuracy of object detectors like Faster R-CNN, SSD(Single Shot MultiBox Detector), etc.
Average precision computes the average precision value for recall value over 0 to 1.
It sounds complicated but actually pretty simple as we illustrate it with an example.
But befor that, we will do a quick recap on precision, recall, and IoU first.

**Precision & recall**

**Precision** measures how accurate is your predictions. i,e. the precentage of your predictions are correct.

**Recall** measures how good you find all the positives. For example, we can find 80% of the possible positive cases in out top K predictions.

Here are their mathematical definitions:
![image](https://user-images.githubusercontent.com/67318280/135710232-1bf2421d-9de8-4d8c-91f9-2de9ba5e6a57.png)

For example, in the testing for cancer:
![image](https://user-images.githubusercontent.com/67318280/135710238-24b1ab28-c9e9-456e-889d-1e8d87c1362b.png)

**IoU (Intersection over union)**

IoU measures the overlap between 2 boundaries. We use that to measure how much our predicted boundary overlaps with the ground truth(the real object boundary).
In some datasets, we predefine an IoU threshold(say 0.5) in classifying whether the prediction is a true positive or a false positive.

![image](https://user-images.githubusercontent.com/67318280/135710332-028fc53e-5d64-475f-a29e-4f7486fde0e6.png)

**AP**

Let's create an over-simplified example in demonstrating the calculation of the average precision.
In this example, the whole dataset contains 5 apples only.
We collect all the predictions made for apples in all the images and rank it in descending order according to the predicted confidence level.
The second column indicates whether the prediction is correct or not.
In this example, the prediction is correct if IoU >= 0.5.

![image](https://user-images.githubusercontent.com/67318280/135710397-35b1a274-0142-4ab4-8ac4-0b784999ca88.png)


Let's take the row with rank#3 and demonstrate how precision and recall are calculated first.

**Precision** is the proportion of TP = 2/3 = 0.67.

**Recall** is the proportion of TP out of the possible positives = 2/5 = 0.4.

Recall values increase as we go down the prediction ranking. However, precision has a zigzag pattern -- it goes down with false positives and goes up agin with true positives.

![image](https://user-images.githubusercontent.com/67318280/135710467-056c38b7-9481-451e-9501-933e20a22d49.png)

Let's plot the precision against the recall value to see the zig-zag pattern.

![image](https://user-images.githubusercontent.com/67318280/135710475-0ac2a990-1a56-4f29-b31c-17fa7d5ba8fb.png)

The general definition for the Average Precision (AP) is finding the area under the precision-recall curve above.

![image](https://user-images.githubusercontent.com/67318280/135710489-088c4e7d-7335-40f9-a5b2-a8421147d592.png)

Precision and recall are always between 0 and 1. Therefore, AP falls within 0 and 1 also.
Before calculating AP for the object detection, we often smooth out the zigzag pattern first.

![image](https://user-images.githubusercontent.com/67318280/135710515-e63a71f1-fa7f-4d4c-b422-c222a858be35.png)

Graphically, at each recall level, we replace each precision value with the maximum precision value to the right of that recall level.

![image](https://user-images.githubusercontent.com/67318280/135710532-d0974544-1910-47c6-9ba8-398cad5b1ae9.png)

So the orange line is transformed into the green lines and the curve will decrease monotonically instead of the zigzag pattern.
The calculated AP value will be less suspectable to small variations in the ranking.
Mathematically, we replace the precision value for recall ȓ with the maximum precision for any recall >= ȓ. 

![image](https://user-images.githubusercontent.com/67318280/135710559-92130ef4-9012-463c-8527-c19a08e0d388.png)

**Interpolated AP**

PASCAL VOC is a popular dataset for object detection. For the PASCAL VOC challenge, a prediction is positive if IoU >= 0.5.
Also, if multiple detections of the same object are detected, it counts the first one as a positive while the rest as negatives.

In Pascal VOC2008, an average for the 11-point interpolated AP is calculated.

![image](https://user-images.githubusercontent.com/67318280/135710661-4b8734bc-29dc-4837-b2b4-0f737670aa86.png)

First, we divide the recall value from 0 to 1.0 into 11 point --0., 0.1, 0.2, ..., 0.9 and 1.0.
Next, we compute the average of maximum precision value for these 11 recall values.

![image](https://user-images.githubusercontent.com/67318280/135710674-64c5b6fa-35fc-4ab0-b13e-044a79dcb634.png)

In our example, AP = (5 x 1.0 + 4 x 0.57 + 2 x 0.5)/11

Here are the more precise mathematical definitions.

![image](https://user-images.githubusercontent.com/67318280/135710689-6eeda1e7-d688-4516-bab2-497e30d420ec.png)

When APᵣ turns extremely small, we can assume the remaining terms to be zero. 
i.e. we don’t necessarily make predictions until the recall reaches 100%.
If the possible maximum precision levels drop to a negligible level, we can stop.
For 20 different classes in PASCAL VOC, we compute an AP for every class and also provide an average for those 20 AP results.

According to the original researcher, the intention of using 11 interpolated point in calculating AP is

### The intention in interpolating the precision/recall curve in this way is to reduce the impact of the “wiggles” in the precision/recall curve, caused by small variations in the ranking of examples.

However, this interpolated method is an approximation which suffers two issues.
It is less precise. Second, it lost the capability in measuring the difference for methods with low AP.
Therefore, a different AP calculation is adopted after 2008 for PASCAL VOC.

## AP (Area under curve AUC)

For later Pascal VOC competitions, VOC2010-2012 samples the curve at all unizue recall values (r₁, r₂, …), whenever the maximum precision value drops.
With this change, we are measuring the exact area under the precision-recall curve after the zigzags are removed.

![image](https://user-images.githubusercontent.com/67318280/135710858-9ff0ee03-b91b-44cf-8495-4b57cf5739b7.png)

No approximation or interpolation is needed. Instead of sampling 11 points, we sample p(rᵢ) whenever it drops and computes AP as the sum of the rectangular blocks.

![image](https://user-images.githubusercontent.com/67318280/135710871-569b5e08-a550-4d84-8bd4-c8f0be401efa.png)

This definition is called the Area Under Curve (AUC). As shown below, as the interploated points do not cover where the precision drops, both methods will diverge.

![image](https://user-images.githubusercontent.com/67318280/135710882-3b375147-e30f-4d05-8d14-0f8dba6baf37.png)

## COCO mAP

Latest resarch papers tend to give results for the COCO dataset only. In COCO mAP, a 101-point interpolated AP definition is used in the calculation.
For COCO, AP is the average over multiple IoU (the minimum IoU to consider a positive match). **AP@[.5:95]** corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05.
For the COCO competition, AP is the average over 10 IoU levels on 80 categories (AP@[.50:.05:.95]: start from 0.5 to 0.95 with a step size of 0.05).
The following are some other metrics collected for the COCO dataset.

![image](https://user-images.githubusercontent.com/67318280/135710922-4dc48f17-911e-4362-a662-ba3cae56c724.png)

And, this is the AP result for the YOLOv3 detector.

![image](https://user-images.githubusercontent.com/67318280/135710930-90f2fc27-e0e9-48e9-94d3-bb1c5ca40a52.png)

In the figure above, AP@.75 means the AP with IoU=0.75.

mAP (mean average precision) is the average of AP. In some context, we compute the AP for each class and average them.
But in some context, they mean the same thing.
For example, under the COCO context, there is no difference between AP and mAP. Here is the direct quote from COCO:

### AP is averaged over all categories. Traditionally, this is called “mean average precision” (mAP). We make no distinction between AP and mAP (and likewise AR and mAR) and assume the difference is clear from context.

In ImageNet, the AUC method is used. So even all of them follow the same principle in measurement AP, the exact calculation may vary according to the datasets. Fortunately, development kits are available in calculating this metric.


