REFERENCE : https://www.analyticsvidhya.com/blog/2018/10/a-step-by-step-introduction-to-the-basic-object-detection-algorithms-part-1/

## Introduction

Start with the algorithms belonging to RCNN family, i.e. RCNN, Fast RCNN and Faster RCNN.

## Table of Contents
1. A Simple Way of Solving an Object Detection Task (using Deep Learning)
2. Understanding Region-Based Convolutional Neural Networks
  1. Intuition of RCNN
  2. Problems with RCNN
3. Understanding Fast RCNN
  1. Intuition of Fast RCNN
  2. Problems with Fast RCNN
4. Understanding Faster RCNN
  1. Intuition of Faster RCNN
  2. Problems with Faster RCNN
5. Summary of the Algorithms coverd

## 1. A Simple Way of Solving an Object Detection Task (using Deep Learning)

For each input image, we get a corresponding class as an output. Can we use this technique to detect various objects in an image?

1. First, we take an image as input:

![image](https://user-images.githubusercontent.com/67318280/136321523-b3ba33a1-63a7-42bb-b77a-6597e8f557b1.png)

2. Then we divide the image into various regions:

![image](https://user-images.githubusercontent.com/67318280/136321550-c0b77d0e-b30f-40b8-bd20-f4e2441f6a28.png)


3. We will then consider each region as a separate image.
4. Pass all these regions (images) to the CNN and classify them into various classes.
5. Once we have divided each region into its corresponding class, we can combine all these regions to get the original image with the detected objects:

![image](https://user-images.githubusercontent.com/67318280/136321562-51508841-39b5-4c33-84c6-b536a8a59421.png)

The problem with using this approach is that the objects in the image can have different aspect ratios and spatial locations. For instance, in some cases the object might be covering most of the image, while in others the object might only be covering a small percentage of the image. The shapes of the objects might also be different (happens a lot in real-life use cases).

As a result of these factors, we would require a very large number of regions resulting in a huge amount of computational time. So to solve this problem and reduce the number of regions, we can use region-based CNN, which selects the regions using a proposal method.

## 2. Understanding Region-Based Convolutional Neural Network
## 2.1 Intuition of RCNN

Instead of working on a massive number of regions, the RCNN algorithm proposes a bunch of boxes in the image and checks if any of these boxes contain any object. **RCNN uses selective search to extract these boxes from an image (these boxes are called regions).**

Let’s first understand what selective search is and how it identifies the different regions. There are basically four regions that form an object: varying scales, colors, textures, and enclosure. Selective search identifies these patterns in the image and based on that, proposes various regions. Here is a brief overview of how selective search works:

- If first takes an image as input:

![image](https://user-images.githubusercontent.com/67318280/136373258-a0afb6fd-d66a-4580-8486-92cd2e404291.png)

- Then, it geneartes initial sub-segmentations so that we have multiple regions from this image:

![image](https://user-images.githubusercontent.com/67318280/136373310-f640ce66-ae1d-4861-bb35-427c1a383654.png)

- The technique then combines the similar regions to form a larger region (based on color similarity, texture similarity, size similarity, and shape compatibility):

![image](https://user-images.githubusercontent.com/67318280/136373404-3ff52d2b-a87d-4948-84e1-025a81471bc5.png)

- Finally, these regions then produce the final object locations (Region of Interest).

Below is a succint summary of the convolutional neural network.
1. We first take a pre-trained convolutional neural network.
2. Then, this model is retrained. We train the last layer of the network based on the number of classes that need to be detected.
3. The third step is to get the Region of Interest for each image. We then reshape all these regions so that they can match the CNN input size.
4. After getting the regions, we train SVM to classify objects and background. For each class, we train one binary SVM.
5. Finally, we train a linear regression model to generate tighter bounding boxes for each identified object in the image.

- First, an image is taken as an input:

![image](https://user-images.githubusercontent.com/67318280/136373877-174946e0-bd53-4fc6-86ff-4f5a06c33643.png)

- Then, we get the Regions of Interest (ROI) using some proposal method (for example, selective search as seen above):

![image](https://user-images.githubusercontent.com/67318280/136373955-22f3e85c-50a2-4bc6-be63-c303672f0033.png)

- All these regions are then reshaped as per the input of the CNN, and each region is passed to the ConvNet:

![image](https://user-images.githubusercontent.com/67318280/136374020-24d242f3-4fc6-4c6c-8cf2-2f100bbf79e2.png)

- CNN then extracts features for each region and SVMs are used to divide these regions into different classes:

![image](https://user-images.githubusercontent.com/67318280/136374116-9f9fbc86-af43-4ec6-b8ea-74e77b5b1068.png)

- Finally, a bounding box regression (Bbox reg) is used to predict the bounding boxes for each identified region:

![image](https://user-images.githubusercontent.com/67318280/136374503-0bda16ac-4139-42b7-8d7d-2795cb9054d7.png)

And this, in a nutshell, is how an RCNN helps us to detect objects.
