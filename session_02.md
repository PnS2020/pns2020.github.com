---
layout: default
---

Welcome to the second session of Deep Learning on the Raspberry Pi.

In this section, we are going to discuss some core concepts in Machine
Learning (ML). We will dive into two historically very influential learning models: Linear Regression and Logistic Regression. We will also discuss Stochastic Gradient Descent (SGD) and its variants. 

## What is a "Learning Algorithm"?

A broadly adopted definition of the _learning algorithm_ is given by Tom M. Mitchell in his classical book _Machine Learning_ in 1997:

---

"A computer program is said to __learn__ from

+   experience $$E$$ with respect to
+   some class of tasks $$T$$ and
+   performance measure $$P$$,

if its performance at tasks in $$T$$, as measured by $$P$$, improves with experience $$E$$ " (Mitchell, 1997)

---

__Remark__: This book is enjoyable to read and introduced many ML algorithms that were very popular back then. It reflects how researchers thought about and did ML in the 1980s and 1990s.

Many popular machine learning textbooks have in-depth discussions of this
definition (Mitchell, 1997; Murphy, 2012; Goodfellow et al., 2016).

### The task $$T$$

The task $$T$$ is usually a specific problem that cannot be easily solved by conventional methods (e.g, array sorting problem can be easily solved by using quick sort). Here we provide some canonical examples of Machine Learning tasks:

+ __Classification__ specifies which $$k$$ categories some input belongs to. ($$f:\mathbb{R}^{n}\rightarrow\{1,\ldots, K\}$$)

+ __Regression__ predicts a numerical value given some input. ($$f: \mathbb{R}^{n}\rightarrow\mathbb{R}$$)

+ __Transcription__ outputs a sequence of symbols, rather than a category code. (similar to classification, e.g., speech recognition, machine translation, image captioning)

+ __Denoising__ predicts clean samples $$\mathbf{x}$$ from _corrupted_ samples $$\tilde{\mathbf{x}}$$. (estimate $$\Pr(\mathbf{x}\vert\tilde{\mathbf{x}})$$)

### The performance measure $$P$$

The performance measure $$P$$ is usually specific to the task $$T$$ (e.g. accuracy to a classification task). Unlike optimization, the learning algorithm is evaluated based on _previously unseen_ data. We often have a set of _validation_ data to conduct this evaluation. The design of a measure $$P$$ can be very subtle. The measure should be effective so that we can anticipate how well the learning algorithm would perform after deployment.

### The experience $$E$$

Experience is what learning algorithms are allowed to have during the learning process. The experience is usually an _dataset_ that is a collection of _samples. Normally, we call this dataset as the _training dataset_. Here we give the structure of the training dataset to Unsupervised Learning and Supervised Learning algorithms.

+ _Unsupervised Learning algorithms_ experience a dataset containing many features, learning useful structure of the dataset (estimate $$\Pr(\mathbf{x})$$).

+ _Supervised Learning algorithms_ experience a dataset containing features, but each example is also associated with a _label_ or _target_ (estimate $$\Pr(\mathbf{y}\vert\mathbf{x})$$).

__Remark__: In most real world cases, we would not have access to the _testing dataset_. In the absence of the _validation dataset_, we usually split 20% of the training dataset to be the _validation dataset_.

### Hypothesis Function

Mathematically, this computer program with respect to the learning task $$T$$ can be defined
as a hypothesis function that takes an input $$\mathbf{x}$$ and transforms it to
an output $$\mathbf{y}$$ (e.g., in a translation task, $$\mathbf{x}$$ = "danke", $$\mathbf{y}$$ = "thanks").

$$\mathbf{y}=f(\mathbf{x}; \theta)$$

The function may be parameterized by a group of parameters $$\theta$$.
Note that $$\theta$$ includes both trainable and non-trainable parameters.
All the DNN architectures discussed in this module can be formulated in this paradigm.

Strictly speaking, the hypothesis function defines a large family of functions that could be the solution to the task $$T$$. At the end of training, the hypothesis function is expected to be parameterized by a set of optimal parameters $$\theta^{\star}$$ that yields the highest performance according to the performance measure $$P$$ of the given task. Conventionally, we call the hypothesis function that equips the optimal parameters $$\theta^{\star}$$ the
_trained model_.

__Remark__: Interested readers can take a look at _The Elements of Statistical Learning_ by Jerome H. Friedman, Robert Tibshirani, and Trevor Hastie, which offers in-depth discussion on this topic.

### Cost Function

A cost function $$J$$ is selected according to the objective(s) of the hypothesis function in which it defines the constraints. The cost function is minimized during the training so that the hypothesis function can be optimized and exhibits the desired behaviors (e.g., classify images, predict houshold value, text-to-speech). The cost function reflects the performance measure $$P$$ directly or indirectly. In most cases, the performance of a learning algorithm gets higher
when the cost function $$J$$ becomes lower.

When the cost function is differentiable (such as in DNNs presented in this module), a class of _Gradient-Based Optimization_ algorithms can be applied to minimize the cost function $$J$$. Thanks to specialized hardware such as GPUs and TPUs, these algorithms can be computed very efficiently.

Particularly, Gradient Descent (Cauchy, 1847) and its variants, such as
RMSprop (Tieleman & Hinton, 2012), Adagrad (Duchi et al., 2011), Adadelta
(Zeiler, 2012), Adam (Kingma & Ba, 2014) are surprisingly good at training
Deep Learning models and have dominated the development of training algorithms. Software libraries have automated the process of computing the gradient (the most difficult part of applying gradient descent) using
a symbolic computation graph. This automation enables the researchers to
design and train arbitrary learning models.

__Remark__: in this module, we use the term "cost function", "objective function" and "loss function" interchangeably. Usually, we also use the terms $$J$$ and $$\mathcal{L}$$ interchangeably.

### Ingredients to Solve a Machine Learning Task

Given a Machine Learning task, you need to have the following ingredients for solving the task:

+ A hypothesis function that maps the input features to outputs.
+ A loss function that defines the objectives.
+ A training algorithm that can optimize the loss function.
+ A training dataset that contains a collection of training examples.
+ A validation dataset that is used to evaluate the performance of the trained model.
+ (Optional) A testing dataset that evaluates the performance of the trained model after deployment.

In this module, we will identify these ingredients while solving different tasks.

## Linear Regression

[![CoLab](https://img.shields.io/badge/Reproduce%20in-CoLab-yellow.svg?style=flat-square)](https://colab.research.google.com/drive/1XI407WXf9efxRqoR__X23tt93V5530m0)

Regression is a task of Supervised Learning. The goal is to take an input vector $$\mathbf{x}\in\mathbb{R}^{n}$$ (a.k.a, features) and predict a target value $$y\in\mathbb{R}$$. In this section, we will learn how to implement _Linear Regression_.

As the name suggests, Linear Regression has a hypothesis function which is a linear function. The goal is to find a linear relationship between the input features and the target value:

$$
\begin{aligned}
y^{(i)}=f(\mathbf{x}^{(i)};\{\mathbf{w}, b\})=&w_{1}\cdot x_{1}+w_{2}\cdot x_{2}+\ldots+w_{i}\cdot x_{i}+\ldots+w_{n}\cdot x_{n}+b \\
=&\sum_{i=1}^{n}w_{i}\cdot x_{i}+b = \mathbf{w}^{\top}\mathbf{x}^{(i)}+b
\end{aligned}
$$

Note that $$\{\mathbf{x}^{(i)}, y^{(i)}\}$$ is the $$i$$-th sample in the dataset $$\{\mathcal{X}, \mathbf{y}\}$$ that has $$N$$ data points. The parameters $$\theta=\{\mathbf{w}, b\}$$ consists of weights $$\mathbf{w}$$ and a bias $$b$$.

Suppose that the target value is a scalar ($$y^{(i)}\in\mathbb{R}$$), we can easily define such a model in PyTorch:

```python
import torch
from torch.nn import Linear

# the model has 10-dimensional input and 1 dimensional output
model = Linear(in_features=10, out_features=1, bias=True)
```

To find a linear relationship that has $$y^{(i)}\approx f(\mathbf{x}^{(i)};\theta)$$, we need to find a set of parameters $$\theta^{\star}$$ from the parameter space $$\theta$$ where the optimized function $$f(\mathbf{x};\theta^{\star})$$ generates the smallest possible error. Supposing we have a cost function $$J$$ that measures the error made by the hypothesis function, our goal can be formulated into:

$$
\theta^{\star}=\arg\min_{\theta}J(\theta)
$$

For Linear Regression, one possible formulation of the cost function is Mean-Squared Error (MSE). This cost function measures the mean error caused by each data sample:

$$
J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\left(y^{(i)}-f(\mathbf{x}^{(i)};\theta)\right)^{2}
$$

By minimizing this cost function via training algorithms such as Stochastic Gradient Descent (SGD), we hope that the trained model $$f(\mathbf{x}; \theta^{\star})$$ can perform well on unseen examples in the testing dataset.

__Remark__: there are other cost functions for regression tasks, such as Mean Absolute Error (MAE) and Root-Mean-Square-Error (RMSE). Interested readers are encouraged to find out what they are.

__Remark__: The math in this module chooses to use a column-vector based system, which means each vector is assumed to be a column vector. Many books and tutorials also apply this convention. However, in practice, most `ndarray` packages use the row-vector based system because the first dimension of a multi-dimensional array is for the row axis. For example,

```python
A = np.array([1, 2, 3, 4, 5])
```

The array `A` is a row vector and has only the row axis. We assume that the readers know this fact and can modify the code accordingly.

## Logistic Regression

[![CoLab](https://img.shields.io/badge/Reproduce%20in-CoLab-yellow.svg?style=flat-square)](https://colab.research.google.com/drive/1cXKA_57f9V4UWGmKlJ90HMAs6wWgjfWR)

In this section, we discuss the solution to another Supervised Learning task - _Binary Classification_. Instead of predicting continuous values (e.g., how many pairs of shoes you have), we wish to decide whether the input feature $$\mathbf{x}$$ belongs to some category. In the case of Binary Classification, we have only two classes (e.g., to be or not to be, shoe or skirt).  _Logistic Regression_ is a simple learning algorithm that solves this kind of tasks.

__Remark__: Usually, we call a learning algorithm that solves binary classification a _Binary Classifier_.

Suppose our input feature $$\mathbf{x}$$ is an $$n$$-dimensional vector and the output class label $$y\in\{0, 1\}$$ (0 and 1 are abstract labels, we can associate meanings for these labels, such as 0 is shoe and 1 is skirt). The Logistic Regression constructs a hypothesis function that assigns the probability that $$\mathbf{x}$$ belongs to the class $$y=1$$. Specifically, the Logistic Regression uses the "logistic function". The hypothesis function is then described as follows:

$$
\sigma(\mathbf{x}; \theta) = \frac{1}{1+\exp(-(\mathbf{w}^{\top}\mathbf{x}+b))}
$$

<hr>
<div align="center">
<img src="./images/sigmoid.jpeg">
<p>The logistic function.</p>
</div>
<hr>

Here, we use the symbol $$\sigma(\cdot)$$ to represent the logistic function. Furthermore, $$\sigma(\cdot)$$ is often called the "sigmoid" function as well. The logistic function has a nice property where it can map the input $$\mathbf{w}^{\top}\mathbf{x}+b$$ into the range $$(0, 1)$$ so that we can interpret the output of this function as a probability:

$$
\begin{aligned}
\Pr(y=1|\mathbf{x}) =& \sigma(\mathbf{x}; \theta) \\
\Pr(y=0|\mathbf{x}) =& 1-\sigma(\mathbf{x}; \theta)
\end{aligned}
$$

Now, we need to design a cost function. The desired function for measuring the quality of the prediction is _binary cross-entropy_:

$$
J(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\left(y^{(i)} \log(\Pr(y=1|\mathbf{x}^{(i)}))+(1-y^{(i)})\log(\Pr(y=0|\mathbf{x}^{(i)}))\right)
$$

Intuitively, when the model makes a correct decision (suppose the true label is 1), then the $$\Pr(y=1|\mathbf{x})$$ is also high, this generates a lower cost than when the model makes a wrong decision and the $$\Pr(y=1|\mathbf{x})$$ is low. From the information theoretic point of view, the _cross-entropy_ between a "true" distribution $$p$$ and an estimated distribution $$q$$ measures the "similarity" between two distributions. Ideally, when the number of samples $$N\rightarrow\infty$$
and cost function $$J(\theta)=0$$, we cannot distinguish the estimation distribution from the "true" distribution.

Our learning algorithm is expected to find a best set of parameters that minimizes the cost function $$J(\theta)$$:

$$
\theta^{\star}=\arg\min_{\theta}J(\theta)
$$

Note that there is a close tie between the Logistic Regression and the Linear Regression. The Logistic Regression is nothing more than adding a non-linear function on top of the linear function. Here is an example of logistic regression in Keras:

```python
import torch
from torch.nn import Linear, Sigmoid, Module

class LR(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LR, self).__init__()
        self.layer = Linear(in_features=in_features, out_features=out_features,
                            bias=bias)
        
        self.act = Sigmoid()
    
    def forward(self, x):
        out = self.layer(x)
        out = self.act(x)

        return out

# there are 10 input features
# and 1 output feature
model = LR(in_features=10, out_features=1, bias=True)
```

__Remark__: we will revisit the logistic function in Session 3 when we introduce the first neural network model: the Multi-layer Perceptron.

Logistic Regression is designed to solve Binary Classification tasks. The above formulation can be generalized to solve Multi-class Classification tasks. The following equation defines the hypothesis function for the extension of the Logistic Regression - _Softmax Regression_:

$$
\text{softmax}(\mathbf{x})=\Pr(y=k|\mathbf{x}, \theta) = \frac{\exp(\mathbf{W}^{k\top}\mathbf{x}+b_{k})}{\sum_{j=1}^{K}\exp(\mathbf{W}^{(j)\top}\mathbf{x}+b_{k})}
$$

where $$\mathbf{W}^{k}$$ is the $$k$$-th column of a weight matrix $$\mathbf{W}\in\mathbb{R}^{n\times k}$$ and $$b_{k}$$ is the corresponding bias value.

The cost function is defined with _categorical cross-entropy_ function:

$$
J(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}\mathbf{1}\{y^{(i)}=k\}\log\Pr(y^{(i)}=k|\mathbf{x}^{(i)}, \theta)
$$

$$\mathbf{1}\{\cdot\}$$ is the "indicator function" so that $$\mathbf{1}\{\text{a true statement}\}=1$$ and $$0$$ otherwise.

Note that we do not explain this loss function here in detail. The _Deep Learning_ book has a very nice explanation of Softmax function in [Section 6.2.2.3](http://www.deeplearningbook.org/contents/mlp.html).

The optimization algorithm finds a set of parameters $$\theta^{\star}$$ that minimizes the cost function:

$$
\theta^{\star}=\arg\min_{\theta}J(\theta)
$$

Here is a Keras example

```python
import torch
from torch.nn import Linear, Softmax, Module

class SoftmaxRegression(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SoftmaxRegression, self).__init__()
        self.layer = Linear(in_features=in_features, out_features=out_features,
                            bias=bias)
        
        self.act = Softmax(dim=-1)
    
    def forward(self, x):
        out = self.layer(x)
        out = self.act(x)

        return out

# suppose there are 5 classes
model = SoftmaxRegression(in_features=10, out_features=5, bias=True)
```

__Remark__: Although we do not explain the Softmax Regression in detail, in fact, the function is widely used by many modern deep learning systems for solving fundamental problems such as classification, and complicated tasks such as neural machine translation.

__Remark__: Note that in the loss functions of the Logistic Regression and
the Softmax Regression computes __log__ likelihood. Hence, in order
to make less numerical error, we normally do not apply the sigmoid function
or the softmax function during the training. And the network outputs without
applying these functions are called _logits_.

## Stochastic Gradient Descent and its Variants

Previous sections define the learning models for Regression and Binary Classification tasks. We now need a training algorithm that minimizes the cost function described in the above sections. In this section, we introduce the most famous set of _Gradient-based Optimization_ algorithms -- Stochastic Gradient Descent (SGD) and its variants.

Almost all modern Deep Learning models are trained by the variants of SGD. In some particular cases, there are deep learning models  that are trained with second-order gradient-based methods (e.g., Hessian optimization).

To describe SGD, we first need to understand its parent method - Gradient Descent. The idea of Gradient Descent is very simple: Suppose that we need to iteratively refine the parameters $$\theta$$ so that we can minimize the cost function $$J$$. The _best_ direction that we should take follows the direction of the steepest descent. This direction can be computed by evaluating the _gradient_ of the loss function. The core idea of the Gradient Descent can be described as follows:

$$\hat{\theta}=\theta-\alpha\frac{\partial J(\theta)}{\partial \theta}$$

where $$\hat{\theta}$$ is the updated parameters, $$\frac{\partial J(\theta)}{\partial \theta}$$ computes the updating directions and the learning rate $$\alpha$$ controls the step size that is taken at the current update. Note that the cost function $$J$$ is a data-driven function where the training data is used to calculate the update. We sometimes call the above formulation "vanilla" Gradient Descent.

The learning rate $$\alpha$$ is arguably the most important _hyperparameter_ in training Deep Learning models. If you set the learning rate too large, then the step update may overshoot and leads to worse performance. On the other hand, if you set the learning rate too small, then the training process may take a longer time to complete.

__Remark__: Hyperparameters are settings that control the behavior of the learning algorithm. Usually we choose them empirically.

<hr>
<div align="center">
<img src="./images/stepsize.jpg">
<p>Visualizing the effect of step size. The white arrow points to the steepest direction. Image credit: <a href="http://cs231n.github.io/optimization-1/">CS231n</a></p>
</div>
<hr>

In most cases, it is not feasible to use Gradient Descent because the training dataset is too large to evaluate. Instead, a common practice is to compute the gradient over _batches_ of training examples. The parameters are updated after evaluating each batch of data. The reason this technique works is that the samples in the training dataset are correlated. However, this technique also introduces stochasticity into the parameter update. Hence this type of training algorithm is
called _Stochastic Gradient Descent_ (SGD). And because we most commonly use mini-batches, sometimes people also refer this training algorithm as _mini-batch SGD_.

Gradient Descent is guaranteed to converge to the global minimum if the cost function is a convex function. In most cases, the cost functions are non-convex functions.
In these cases, Gradient Descent can only find the local-minima.

### Momentum SGD

The vanilla SGD can easily lead to traps around a local minimum point. For example, in such a case, all the directions around the local region seem steep. The parameters then oscillate around this local region during training. To get out of the local minima, we can use a variant of the vanilla SGD - Momentum SGD. The formulation is as follows:

$$
\begin{aligned}
\hat{\mathbf{v}}=&\mu\mathbf{v}+\alpha\nabla_{\theta} J(\theta) \\
\hat{\theta}=&\theta-\hat{\mathbf{v}}
\end{aligned}
$$

The basic idea is that if we allow the gradient update to accumulate, at some point, the energy of the update is powerful enough to jump out from the local minima. With momentum, we gain faster convergence and less oscillatory behavior. Empirically, the momentum parameter $$\mu$$ is set to 0.9 or 0.99.

__Remark__: $$\nabla_{\theta} J(\theta)=\frac{\partial J(\theta)}{\partial \theta}$$.

### Adaptive SGD

Choosing the learning rate $$\alpha$$ for SGD is mainly empirical. Therefore, we will have to perform a manual search from a list of possible learning rates. This process is usually costly and time-consuming. In recent years, researchers developed a set of SGD variants that adjust the learning rate $$\alpha$$ automatically.

The notable examples are RMSprop (Tieleman & Hinton, 2012), Adagrad (Duchi et al., 2011), Adadelta (Zeiler, 2012), Adam (Kingma & Ba, 2014).

Note that the motivation of having these different variants is not entirely because of the dissatisfaction of the SGD and brute-force search for the learning rate. For example, RMSprop is proposed to deal with _the vanishing gradient problem_ where some very deep networks cannot be trained with standard SGD.

Empirically, one should use Adam optimizer as a starting point.


<hr>
<div align="center">
    <p><img src="./images/opt1.gif" width="49%">
    <img src="./images/opt2.gif" width="49%"></p>
    <p>Comparing different gradient optimization methods. Left: SGD optimization on saddle point. Right: SGD optimization on loss surface contours.  Image credit: <a href="https://twitter.com/alecrad">Alec Radford</a></p>
</div>
<hr>

### Learning Rate Scheduling

The above sections discuss the methods that have a fixed initial learning rate. The learning rate is either static or adjusted by the training algorithm itself. Recent research suggests that instead of using these optimizers, it is better to schedule the learning rate throughout the training. Normally, this involves even more expensive parameter searching because the researcher has to predefine the "schedule" of the use of learning rate at the different stages of training.

Fortunately, over the years, some empirical training schedules have been developed that work well across the tasks. We do not discuss this in detail since this is out of scope of this module. However, we do encourage readers to find some recent papers in large scale image recognition and natural language processing papers where they employ such training schedules.

__Remark__: SGD and its variants represent the most popular group of training algorithms. However, there are other optimization algorithms available and extensively studied by Machine Learning researchers, such as energy based models, evolutionary algorithms, genetic algorithms, Bayesian optimization.

__Remark__: Sebastian Ruder surveyed almost all popular variants of SGDs in a [blog post](http://ruder.io/optimizing-gradient-descent/index.html).

### Training a model in PyTorch

#### Setting up the dataset

To train a model, we need to prepare the data pipeline for feeding
the training dataset and the validation dataset.
With PyTorch, we can first define the data fetching for a single sample.

```python
from torch.utils.data import Dataset

class LRDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
```

Then, we will need to use a Data Loader to schedule for batching data:

```python
from torch.utils.data import DataLoader

# note that we shuffle the training data
train_dataset = DataLoader(LRDataset(train_x, train_y), batch_size=32,
                           shuffle=True, num_workers=4)
# note that we do not shuffle the validation data
valid_dataset = DataLoader(LRDataset(valid_x, valid_y), batch_size=32,
                           shuffle=False, num_workers=4)
```

#### Define Optimization

After you have defined a model (see above model examples for linear regression, logistic regression, and softmax regression), you will need to identify the optimizer, the loss function and the metric:

```python
from torch.nn import BCELoss
from torch.optim import SGD

# define optimizer
optimizer = SGD(params=model.parameters(), lr=0.001)
# Binary cross-entropy loss for Logistic Regression
loss_fn = BCELoss()

# define the accuracy as a function
def Accuracy(y, y_pred):
    num_samples = y.shape[0]
    
    accuracy = (y==pred).sum()/float(num_samples)

    return accuracy
```

The above example sets up a Logistic Regression model that uses BCE as the loss function and the default SGD as the optimizer. The performance measure $$P$$ here is called `metrics`. You can have different metrics for evaluating the performance of the model. We use accuracy as the metric for evaluation.
In the case of regression tasks, one could use MSE for both the loss function and the metric.

#### Training in epochs

First, you need to specify your training inputs `x` and training target `y`. And then you will need to define the mini-batch size and number of epochs. The `fit` API will run for a number of `epochs` and at each step of a epoch, the function will fetch a batch of training examples (in this case, 64) and then use them to compute the gradient update. The parameters are updated after the
gradient is computed. Finally, we can supply a set of validation data (in this case, `(test_X, test_y)`. After each training epoch, Keras evaluates the model's performance using the validation data.

The following code block demonstrates the training procedure.
Note that we omitted the log printing and loss statistics collection
for its simplicity.

```python
num_epochs = 10
for epoch_i in range(num_epochs):
    for batch_i, (batch_x, batch_y) in enumerate(train_dataset):
        # turn on the model training mode
        # parameters can be updated
        model.train()

        # prepare the optimizer
        optimizer.zero_grad()

        # calculate the prediction
        batch_y_pred = model(batch_x)
        # calculate the training loss for this batch
        batch_loss_output = loss_fn(batch_y_pred, batch_y)
        # calculate the training accuracy for this batch
        batch_metric_output = Accuracy(batch_y, (batch_y_pred > 0.5))
        
        # compute the gradients respect to the parameters
        batch_loss_output.backward()

        # the optimizer updates the parameters
        optimizer.step()

    for batch_i, (batch_x, batch_y) in enumerate(valid_dataset):
        # turn on the evaluation mode
        model.eval()

        # make sure no gradients are calculated
        with torch.no_grad():
            # compute the prediction
            batch_y_pred = model(batch_x)
            # compute the validation loss for this batch
            batch_loss_output = loss_fn(batch_y_pred, batch_y)
            # compute the validation accuracy for this batch
            batch_metric_output = Accuracy(batch_y, (batch_y_pred > 0.5))
```

## Generalization

Typically, we expect that the trained model can be used to predict _new, unseen samples_. This ability to perform well on previously unobserved data is called _generalization_. With a performance measure $$P$$, we can compute the error that is made over the training dataset. This error is called _training error_. We can also compute the _testing error_ with the validation dataset or the testing dataset if available. This _testing error_ quantifies the level of
generalization of a certain model. Our goal is to minimize this _testing error_ and improve the generalization.

Because we can only observe the training dataset during training, our trained model may suffer from either _overfitting_ or _underfitting_. Overfitting occurs when the gap between the training error and testing error is increasing. Underfitting occurs when the model is not able to obtain a sufficiently low error value on the training dataset.

We can control whether a model is more likely to overfit or underfit by altering its _capacity_. Informally, a model's capacity is its ability to fit a wide variety of functions. Ideally, we want to find a model that has the optimal capacity where the model generates the smallest gap between the training error and the testing error.

Usually, Deep Learning models have tremendous capacity. These models almost always overfit the dataset given enough training time. Strangely, these models also exhibit strong generalization over testing dataset with the help of different training techniques. One straightforward technique is _Early Stopping_. The Early Stopping technique stops the training when the model starts to overfit.

## No Free Lunch Theorem and Curse of Dimensionality

### No Free Lunch Theorem

To solve one particular task, we can choose from many different models and learning algorithms. Empirically, we can evaluate the performance of these models and algorithms using the validation dataset. However, there is no universally best model -- this is the famous _no free lunch theorem_ (Wolpert, 1996). The reasons for this is that different models make different assumptions. These assumptions may work well in one domain and work poorly in another domain. As a consequence,
rather than focusing on developing the universal model, we seek the solution that performs well in a practical scenario for some relevant domains or distributions.

### Curse of Dimensionality

---

Many machine learning problems become exceedingly difficult when the number of dimensions in the data is high. The phenomenon is known as the _curse of dimensionality_. Of particular concern is that the number of possible distinct configurations of the variables of interest increases __exponentially__ as the dimensionality increases.

---

__Remark__: For an elaborated description of these two challenges, please refer to [Chapter 5](http://www.deeplearningbook.org/contents/ml.html) of the Deep Learning book and Chapter 1 of the book [Machine Learning: A probabilistic perspective](https://mitpress.mit.edu/books/machine-learning-0).

## Exercises

In this exercise, you will need to implement Logistic Regression to distinguish two classes from the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

If you decide to use Colaboratory to complete the exercise, please follow the description at [here](./colab-intro.md).

1. We provide a [template script](./res/pt-code/logistic-regression-with-keras-layers-template.py) that has the barebone structure of implementing Logistic Regression in PyTorch. You will need to complete the script and get it running. You are expected to define a Logistic Regression model, compile the model with binary cross-entropy loss and an optimizer, and train the model. If you can successfully train the model, try to change the choice of optimizer, what do you observe?

2. (Optional) In this exercise, you will implement the Logistic Regression from scratch. We provide a [template script](./res/pt-code/logistic-regression-basic-template.py) that contains necessary setup code, you need to complete the code and run.
You are expected to write a training loop that can loop over the data for some number of epochs.
