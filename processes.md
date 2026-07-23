# Loss + Gradient Derivation
## Loss Function
#### Why not use standard linear regression loss function?
If we were to use the standard linear regression loss function, $L(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$ with our non-linear sigmoid function $\hat{y} = \frac{1}{1+e^{-z}}$, upon optimizing the wine weights, we would find our cost surface being riddled with local minimums. This is bad, as gradient descent would get trapped in one of these local minimums (unless it gets lucky).

#### So what must we do?
First, to answer that, we begin with two assumptions:
1. The probability of a wine being good (which this state we'll denote with $y=1$), will be equal to the predicted probability of the wine being good (denoted with $\hat{y}$), depending on our 11 input features containing fixed acidity, volatile acidity, etc. These features will be denoted with $x$.

2. Furthermore, the probability of a wine being of bad quality (which will be denoted with $y=0$), will be equal to 1 - our predicted probability, or $1-\hat{y}$.

Thus, $P(y=0 | x) = 1 - \hat{y}$ and $P(y=1 | x) = \hat{y}$.

#### How can we simplify these two statements into one equation?
Since $y$ only has two states, being 0 and 1 (bad and good), we can use exponents to represent $y$'s binary state. 
1. $P(y | x) = \hat{y}$ if and only if $y=1$, so we may re-write this part of the equation as $\hat{y}^y$. This works as when $y=1$, $\hat{y}$ is the result, which is what we want, and when $y=0, 1$ is the result, which gives us an opportunity to insert the second part of the equation, when $y=0$.

2. $P(y | x) = 1 - \hat{y}$ if and only if $y=0$. Using the same strategy as before, we can rewrite this as $(1 - \hat{y})^{(1 - y)}$. When $y = 0$, the exponent is $1$ and thus the proper term is returned $(1 - \hat{y})$, and when $y = 1$, the exponent is $0$ and thus returns $1$.

To combine the two, we'll multiply them together since when y is not equal 0 or 1, the associated equation will return 1, in which case multiplying by 1 does nothing to the final result. Then, the other part of the equation which has not turned into 1 will be our desired result. Essentially, we have turned an if statement into an equation.

$P(y | x) = (\hat{y})^{y}(1-\hat{y})^{1-y}$

For the sake of differentiability, we'll use the natural logarithm on the entire equation to turn exponents into multiplicands and multiplication into addition:

$\ln(P(y|x)) = y\ln(\hat{y}) + (1-y)\ln(1-\hat{y})$

#### One additional coefficient
Now, testing this equation with actual data points, let's say we have a wine with $y=1$ (truly good wine). We plug in an accurate prediction of 0.9. This gives us $(1)\ln(0.9) + (1-(1))\ln(1-0.9) \approx -1.05$. 

However, for an inaccurate prediction of 0.1, we get $(1)\ln(0.1) + (1-(1))\ln(1-0.1) \approx -2.30$. The accurate prediction has greater value than the inaccurate prediction! 

In our case, we want our loss function to minimize loss, i.e., less accurate predictions. So, we must add a $(-)$ coefficient to the final equation, making our loss function:

$L(\hat{y},y) = -[y\ln(\hat{y}) + (1-y)\ln(1-\hat{y})]$

And thus, our cost function, being an aggregate of all data points, is this:

$J(\theta) = \frac{1}{m} \sum_{i=1}^{m}L(\hat{y}^{(i)},y^{(i)}) = -\frac{1}{m} \sum_{i=1}^{m}[y^{(i)}\ln(\hat{y}^{(i)}) + (1-y^{(i)})\ln(1-\hat{y}^{(i)})]$

where $\theta$ represents the wine weights and bias, $m$ is the number of data points, $\hat{y}^{(i)}$ is our predicted probability of the wine being good $(y=1)$, and $y^{(i)}$ is the actual quality of the wine for the $i^{th}$ wine

## Gradient Math
#### Using the loss function

With our loss function, we can derive the gradient with a bit of partial differentation. So, we will find the derivative of $L(\hat{y},y)$ with respect to $w$ and $b$.

We have three equations:
1. $L(\hat{y},y) = -[y\ln(\hat{y}) + (1-y)\ln(1-\hat{y})]$ (loss function)
2. $\hat{y} = \sigma(z) = \frac{1}{1+e^{-z}}$ (sigmoid function)
3. $z = w \cdot x^T+b$ (linear combination of features). For this equation, we can do a little trick to combine $w$ and $b$ into one vector by first inserting $b$ into $w$, so the size becomes $[1, 12]$, and then additionally inserting $1$ in the 0th column index of $x$.
   - $\theta=[b,w_1,w_2,...,w_{11}]$
   - $\hat{x}^T = [1, x_1, x_2, ..., x_{11}]^T$
   - $z = \theta \cdot \hat{x}^T$

And according to the chain rule, $\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial \theta}$. Lets start calculating!

1. $\frac{\partial L}{\partial \hat{y}} = -[\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}(-1)]=\frac{1-y}{1-\hat{y}} - \frac{y}{\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$

2. $\frac{\partial \hat{y}}{\partial z} = -(1+e^{-z})^{-2} \cdot (e^{-z}) \cdot (-1) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} \cdot (\frac{1 + e^{-z}}{1+e^{-z}} - \frac{1}{1+e^{-z}}) = \hat{y}(1-\hat{y})$ since $\hat{y} = \frac{1}{1+e^{-z}}$ (Wow!)

3. $\frac{\partial z}{\partial \theta} = \hat{x}$

After multiplying the three, we get:

$\frac{\partial L}{\partial \theta} = (\hat{y}-y) \cdot \hat{x}$ or $\hat{x}^T \cdot (\hat{y}-y)$

Generalizing to the gradient, we get:

$\nabla J(\theta) = \frac{1}{m} X^T (\hat{y}-y)$
 - $X^T= \begin{bmatrix} 1 & 1 & \dots & 1 \\
  x_1^{(1)} & x_1^{(2)} & \dots & x_1^{(m)} \\
  \vdots & \vdots & \ddots & \vdots\\
  x_{11}^{(1)} & x_{11}^{(2)} & \dots & x_{11}^{(m)}
  \end{bmatrix} \quad (12 \times m)$

 - $(\hat{y}-y)= \begin{bmatrix} \hat{y}^{(1)}-y^{(1)} \\
  \hat{y}^{(2)}-y^{(2)} \\
  \vdots \\
  \hat{y}^{(m)}-y^{(m)}
  \end{bmatrix} \quad (m \times 1)$







