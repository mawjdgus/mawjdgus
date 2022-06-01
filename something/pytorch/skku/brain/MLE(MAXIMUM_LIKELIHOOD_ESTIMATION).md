REFERENCE : https://towardsdatascience.com/maximum-likelihood-vs-bayesian-estimation-dd2eb4dfda8a
REFERENCE : https://angeloyeo.github.io/2020/07/17/MLE.html

# MAXIMUM LIKELIHOOD ESTIMATION

### Sample problem: Suppose you want to know the distribution of tree’s heights in a forest as a part of an longitudinal ecological study of tree health, but the only data available to you for the current year is a sample of 15 trees a hiker recorded. The question you wish to answer is: “With what distribution can we model the entire forest’s trees’ heights?”

![image](https://user-images.githubusercontent.com/67318280/135740313-0733b3af-6dbe-4488-b753-725ee9610305.png)

**Quick note on notation**
- θ is the unknown variables, in our Gaussian case, θ = (μ,σ²)
- D is all observed data, where D = (x_1, x_2,…,x_n)

## Likelihood Function

The (pretty much only) commonality shared by MLE and Bayesian estimation is their dependence on the likelihood of seen data (in our case, the 15 samples). The likelihood describes the chance that each possible parameter value produced the data we observed, and is given by:

![image](https://user-images.githubusercontent.com/67318280/135740327-8b6f851b-5a24-41d5-961e-13a2e46e5ad9.png)

Thanks to the wonderful i.i.d. assumption, all data samples are considered independent and thus we are able to forgo messy conditional probabilities.

Let’s return to our problem. All this entails is knowing the values of our 15 samples, what are the probabilities that each combination of our unknown parameters (μ,σ²) produced this set of data? By using the Gaussian distribution function, our likelihood function is:

![image](https://user-images.githubusercontent.com/67318280/135740347-916c52ad-36e6-4167-b5e4-5fbf3fa48b27.png)

## Maximum Likelihood Estimation (MLE)

Awesome. Now that you know the likelihood function, calculating the maximum likelihood solution is really easy. It’s in the name. To get our estimated parameters (𝜃̂), all we have to do is find the parameters that yield the maximum of the likelihood function. In other words, what combination of (μ,σ²) give us that brightest yellow point at the top of the likelihood function pictured above?

To find this value, we need to apply a bit of calculus and derivatives:

![image](https://user-images.githubusercontent.com/67318280/135740375-4e20c4a0-505a-4d15-adb9-b654f2f6f900.png)

As you may have noticed, we run into a problem. Taking derivatives of products can get really complex and we want to avoid this. Luckily, we have a way around this issue: to instead use the log likelihood function. Recall that (1) the log of products is the sum of logs, and (2) taking the log of any function may change the values, but does not change where the maximum of that function occurs, and therefore will give us the same solution.

![image](https://user-images.githubusercontent.com/67318280/135740395-1e9483fc-60b3-4dd1-8a25-7c09aa80a545.png)

It turns out that for a Gaussian random variable, the MLE solution is simply the mean and variance of the observed data. Therefore, for our problem, the MLE solution modeling the distribution of tree heights is a Gaussian distribution with μ=152.62 and σ²=11.27.
