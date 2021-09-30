REFERENCE : https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/

# How to implement a YOLO(v3) object detector from scratch in PyTorch:Part1

Object detection is a **domain that  has benefited immensely from the recent developments in deep learning**.<br>
Recent years seen people develop many algorithms for object detection, some of which include **YOLO, SSD, Mask RCNN and RetinaNet.**

We will use PyTorch to implement an object detector based on YOLO v3, one of the faster object detection algorithms out there.

The code for this tutorial is designed to run on Python 3.5, and PyTorch 0.4.

This tutorial is broken into 5 parts:
- 1. Part1:Understanding How YOLO works
- 2. Part2:Creating the layers of the network architecture
- 3. Part3:Implementing the forward pass of the network
- 4. Part4:Objectness score thresholding and Non-maximum suppression
- 5. Part5:Designing the input and the output pipelines

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
**Generally, stride of any layer in the network is equal to the factor by which the output of the layer is smaller than the input image to the network.**

## Interpreting the output

