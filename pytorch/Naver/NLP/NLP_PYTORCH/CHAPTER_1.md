# Introduction

## The Supervised Learning Paradigm

![image](https://user-images.githubusercontent.com/67318280/135713030-8db11973-23cc-4151-a5a6-7ba3443cc680.png)

**TRAINING USING (STOCHASTIC) GRADIENT DESCENT**

For large datasets, implementation of traditional
gradient descent over the entire dataset is usually impossible due to memory constraints, and very
slow due to the computational expense. Instead, an approximation for gradient descent called
stochastic gradient descent (SGD) is usually employed. In the stochastic case, a data point or a
subset of data points are picked at random, and the gradient is computed for that subset. When a
single data point is used, the approach is called pure SGD, and when a subset of (more than one)
data points are used, we refer to it as minibatch SGD.In practice, pure SGD is
rarely used because it results in very slow convergence due to noisy updates. There are different
variants of the general SGD algorithm, all aiming for faster convergence.

## Observation and Target Encoding

![image](https://user-images.githubusercontent.com/67318280/135713117-3596c4e8-8e1f-4ee5-840b-d657e4353274.png)

***Figure 1­2. Observation and target encoding: The targets and observations from igure 1­1 are represented
numerically as vectors, or tensors. This is collectively known as input “encoding.”***

A simple way to represent text is as a numerical vector. There are innumerable ways to perform this
mapping/representation. In fact, much of this book is dedicated to learning such representations for a
task from data. However, we begin with some simple count­based representations that are based on
heuristics. Though simple, they are incredibly powerful as they are and can serve as a starting point for
richer representation learning. All of these count­based representations start with a vector of fixed
dimension.

**One-Hot RePresentation**

```
Time flies like an arrow.
Fruit flies like a banana.
```

Tokenizing the sentences, ignoring punctuation, and treating everything as lowercase, will yield a
vocabulary of size 8: {time, fruit, flies, like, a, an, arrow, banana}. So, we
can represent each word with an eight­dimensional one­hot vector. In this book, we use 1 w to mean
one­hot representation for a token/word w.

The collapsed one­hot representation for a phrase, sentence, or a document is simply a logical OR of
the one­hot representations of its constituent words. Using the encoding shown in igure 1­3, the one­
hot representation for the phrase “like a banana” will be a 3×8 matrix, where the columns are the
eight­dimensional one­hot vectors. It is also common to see a “collapsed” or a binary encoding where
the text/phrase is represented by a vector the length of the vocabulary, with 0s and 1s to indicate
absence or presence of a word. The binary encoding for “like a banana” would then be: [0, 0, 0,
1, 1, 0, 0, 1].

![image](https://user-images.githubusercontent.com/67318280/135713247-b3e03b6b-acdf-4d2e-befd-ce95caec6e65.png)

