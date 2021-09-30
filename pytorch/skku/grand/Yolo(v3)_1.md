REFERENCE : https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/

# How to implement a YOLO(v3) object detector from scratch in PyTorch:Part1

Object detection is a **domain that  has benefited immensely from the recent developments in deep learning**.<br>
Recent years seen people develop many algorithms for object detection, some of which include **YOLO, SSD, Mask RCNN and RetinaNet.**

We will use PyTorch to implement an object detector based on YOLO v3, one of the faster object detection algorithms out there.

The code for this tutorial is designed to run on Python 3.5, and PyTorch 0.4.

This tutorial is broken into 5 parts:
1. Part1:Understanding How YOLO works
2. Part2:Creating the layers of the network architecture
3. Part3:Implementing the forward pass of the network
4. Part4:Objectness score thresholding and Non-maximum suppression
5. Part5:Designing the input and the output pipelines

## What is YOLO?

YOLO stands for You Only Look Once. It's an object detector that uses featrues learned by a deep convolutional neural network to detect an object.<br>
Before we get out hands dirty with code, we must understand how YOLO works.

## A Fully Convolutional Neural Network

YOLO makes use of only convolutional layers, making it a fully convolutional network(FCN).<br>
It has **75 convolutional layers, with skip connections and upsampling layers**.<br>
**No form of pooling** is used, and **a convolutional layer with stride 2 is used to downsample the feature maps**.<br>
This helps in **preventing loss of low-level features often attributed to polling.** **이해 못한 부분**

Being a FCN, YOLO is invariant to the size of the input image.<br>
However, in practice, we might want to stick to a constant size du to various problems that only show their heads when we are implementing the algorithm.

A big one amongst these problems is that **if we want to process out images in batches**(images in batches can be processed in parallel by the GPU, leading to speed boosts), we need to have all images of fixed height and width.<br>
This is needed to concatenate multiple images into a large batch (concatenating many PyTorch tensors into one)

The network downsamples the image by a factor called the **stride** of the network.<br>
For example, if the stride of the network is 32, then an input image of size 416x416 will yield an output of size 13x13.<br>
**Generally, stride of any layer in the network is equal to the factor by which the output of the layer is smaller than the input image to the network.** **이해 못한 부분**

## Interpreting the output

Typically, (as is the case for all object detectors) the features learned by the convolutional layers are passed onto a classifier/regressor which makes the detection prediction (coordinates of the bounding boxes, the class label.. etc)

In YOLO, the prediction is done by using a convolutional layer which uses **1x1 convolutions**.

Now, the first thing to notice is out output is a **feature map.**<br>
Since we have used 1x1 convolutions, the sieze of the prediction map is exactly the size of the feature map before it.<br>
In YOLO v3, the way you interpret this prediction map is that each cell can predict a fixed number of bounding boxes.**이해못함**

### " Though the technically correct term to describe a unit in the feature map would be a neuron, calling it a cell makes it more intuitive in out context."

**Depth-wise, we have(Bx(5+C)) entries in the feature map.**<br>
B represents the number of bounding boxes each cell can predict.<br>
According to the paper, each of these B bounding boxes may specialize in detecting a certain kind of object.<br>
Each of the bounding boxes may specialize in detecting a certain kind of object.<br>
Each of the bounding boxes have (5+C) attributes, which describe the center coordinates, the dimensions, the objectness score and C class confidneces for each bounding box. **이해 못함**<br>
YOLO v3 predicts 3 bounding boxes for every cell.

**You expect each cell of the feature map to predict an object through one of it's bounding boxes if the center of the object falls in the receptive field of that cell**.<br>
feature map의 각 cell은 객체의 중심이 해당 cell의 receptive field에 맞아 떨어지면, 바운딩 박스 중 하나를 통해 객체를 예측합니다.
(receptive feild는 cell이 볼 수 있는 입력 이미지의 영역입니다.)

This how to do with how YOLO is trained, **where only one bounding box is responsible for detecting any given object.** <br>
First, we must ascertain which of the cells this bounding box belongs to.

To do that, we divide the input image into a grid of dimensions equal to that of the final feature map.

Let us consider an example below, where the input image is 416x416, and stride of the network is 32.<br>
As pointed earlier, the dimensions of the feature map will be 13x13.<br>
We then divide the input image into 13x13 cells.

![image](https://user-images.githubusercontent.com/67318280/135371613-7a2b14e1-05fb-4a5f-8a7d-97031470bff3.png)

Then, the cell(on the input image) containing the center of the ground truth box of an object is chosen to be the one responsible for predicting the object.<br>
In the image, it is the cell which marked red, which contains the center of the ground truth box(marked yellow).

Now, the red cell is the 7th row on the grid. We now assign the 7th cell in the 7th row on the feature map as the one responsible for detecting the dog.

Now, this cell can predict three bounding boxes.<br>
Which one will be assigned to the dog's ground truth label?<br>
In order to understand that, we must wrap out head around the concept of anchors.

### " Note that the cell we're talking about here is a cell on the prediction feature map. We divide the input image into a grid just to determine which cell of the prediction feature map is responsible for prediction."

## Anchor Boxes

It might make sense to predict the width and the height of the bounding box, but in practice, that leads to unstable gradients during training.<br>
Instead, most of the modern object detectors predict log-space transforms, or simply offsets to pre-defined default bounding boxes called anchors.

Then, these transforms are applied to the anchor boxes to obtain the prediction.<br>
YOLO v3 has three anchors, which result in prediction of three bounding boxes per cell.

Coming back to our earlier question, the bounding box responsible for detecting the dog will be the one whose anchor has the highest **IoU** with the ground truth box.

## Making Predictions

The following formulae describe how the network output is transformed to obtain bounding box predictions.
![image](https://user-images.githubusercontent.com/67318280/135372351-ffdaa776-4b91-41f1-b55a-7ea7bca77626.png)

bx, by, bw, bh are the x,y center co-ordinates, width and height of our prediction. tx, ty, tw, th is what the network outputs. cx and cy are the top-left co-ordinates of the grid. <br>
pw and ph are anchors dimensions for the box.

**Center Coordinates**


