REFERENCE :https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275

# Understanding Dropout with the Simplified Math behind it

In spite of the groundbreaking results reported, little is known about Dropout from a theoretical standpoint.<br>
Likewise, the importace of Dropout rate as 0.5 and how it should be changed with layers are not evidently clear.<br>

Also, can we generalize Dropout to other approaches?

The following will provide some explanations.<br>
Deep Learning architectures are now becoming deeper and wider.<br>
With these bigger networks, we are able to achieve good accuracies.<br>

**However, this was not the case about a decade ago.**

Deep Learning was, in fact, infamous due to overfitting issue.

![image](https://user-images.githubusercontent.com/67318280/135192547-5b380f24-2855-4205-a9a1-09d0e9dd8814.png)

Then, around 2012, the idea of Dropout emerged.

The concept revolutionized Deep Learning.

Much of the success that we have with Deep Learning is attributed to Dropout.

## Quick recap: What is Dropout?

- Dropout changed the concept of learning all the weights together to learning a fraction of the weights in the network in each training iteration.

![image](https://user-images.githubusercontent.com/67318280/135192663-a475d794-3b67-4783-95d5-5a91251c7939.png)

- This issue resolved the overfitting issue in large networks. And suddenly bigger and more accurate Deep Learning architectures became possible.


Before Dropout, a major research area was **regularization**.

Introduction of regularization methods in neural networks, such as L1 and L2 weight penalties, started from the early 2000s [1]. However, these regularizations did not completely solve the overfitting issue.

The reason was Co-adaptation.

## Co-adaptation in Neural Network

![image](https://user-images.githubusercontent.com/67318280/135192935-3cd77225-1dfc-40e3-877c-46982dcbc085.png)

One major issue in learning large networks is **co-adaptaion**.<br>
In such a network, if all the weights are learned together it is common that **some of the connections will have more predictive capability than the others.**<br>
In such a scenario, as the network is trained iteratively **these powerful connections are learned more while the weaker ones are ignored.**<br>
Over many iterations, **only a fraction of the node connections is trained.**<br>
And the rest stop participating.<br>

This phenomenon is called **co-adaptaion**. <br>
This coult not be prevented with the traditional regularization, like the L1 and L2.<br>
**The reason is they also regularize based on the predictive capability of the connections.**<br>

Due to this, they become close to deterministic in choosing and rejecting weights.<br>
And,thus again, the strong gets stronger and the weak gets weaker.<br>
A major fallout of this was: **expanding the neural network size would not help**.<br>

Consequently, neural networks'size and, thus, accuracy became limited.<br>
Then came Dropout.<br>
A now regularization approach.<br>

**It resolved the co-adaptation**.

## Math behind Dropout

Consider a single layer linear unit in a network as shown in Figure 4 below.
Refer [2] for details.

![image](https://user-images.githubusercontent.com/67318280/135193535-fdbe9fde-1efa-4247-8173-8460d8402b5e.png)

This is called linear because of the linear activation, f(x) = x.<br>
As we can see in Fugure 4, the output of the layer is a linear weighted sum of the inputs.<br>
We are considering this simplified case for a mathematical explanation.<br>
The results (empirically) hold for the usual non-linear networks.<br>

For model estimation, we minimize a loss function.<br>
For this linear layer, we will look at the ordinary least square loss,<br>

![image](https://user-images.githubusercontent.com/67318280/135193719-80bf9684-3e2e-4544-a88d-23827ecf7679.png)

Eq. 1 shows loss for a regular network and Eq. 2 for a dropout network.

In Eq. 2, the dropout rate is 𝛿, where 𝛿 ~ Bernoulli(p).

This means 𝛿 is equal to 1 with probability p and 0 otherwise.

The backpropagation for network training uses a gradient desent approach.<br>
We will, therefor, first look at the gradient of the dropout network in Eq.2, and then come to the regular network in Eq.1

![image](https://user-images.githubusercontent.com/67318280/135194130-1280ced4-9d32-48ac-bfff-30c4e4f009e3.png)

Now, we will try to find a relationship between this gradient and the gradient of the regular network.<br>
To that end, suppose we make w’ = p* w in Eq.1.Therefore,

![image](https://user-images.githubusercontent.com/67318280/135196022-9283e817-9cbf-4203-a391-cbb80073cab6.png)

![image](https://user-images.githubusercontent.com/67318280/135196033-d99df3fd-b296-46a0-bfd6-cd69101321da.png)

![image](https://user-images.githubusercontent.com/67318280/135196038-90d95067-c69b-497e-bf67-3f93dd3412ce.png)

