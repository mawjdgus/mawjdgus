REFERENCE : https://medium.com/swlh/fully-connected-vs-convolutional-neural-networks-813ca7bc6ee5

# Fully Connected vs Convolutional Neural Networks

## Fully connected neural netwrok

- A fully connected neural network consists of a series of fully connected layers that **connect every neuron in one layer to every neuron in the other layer**.
- The major advantage of fully connected networks is that they are **"structure agnostic"** i.e. there are no special assumptions needed to be made about the input.
- While being structure agnostic makes fully connected networks very **broadly applicable**, such networks to tend to have weaker performance than special-purpose networks tuned to the structure of a problem space.

![image](https://user-images.githubusercontent.com/67318280/136320081-65446cc3-9e21-482b-9ee6-e357007cd1d7.png)

## Convolutional Neural Network

- CNN architectures make the explicit assumption that the inputs are images, which allows encoding certain properties into the model architecture.
- A simple CNN is a sequence of layers, and every layer of a CNN transforms one volume of activations to another through a differentiable function. Three main types of layers are used to build CNN architecture:
Convolutional Layer, Pooling Layer, and Fully-Connected LAyer.

![image](https://user-images.githubusercontent.com/67318280/136320558-8e635850-cda4-400a-87fb-0a83d58fded3.png)
