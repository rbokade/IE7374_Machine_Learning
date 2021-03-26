---
title: HW-2
author: Rohit Bokade (NUID 001280767) 
output: 
  pdf_document:
    fig_caption: true
    keep_tex: true
    citation_package: biblatex
bibliography: references.bib
header-includes: |
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{amssymb}
---

1.

(a) The best decision function can be given by $f^{*}(x) = x$. Thus, the expected loss or "risk" would be 0 $R(f) = \mathbb{E} \mathcal{l}(f^{*}(x), y) = 0$. 

(b) Approximation function: $R(f_{F}) - R(f^{*})$. The best constant function would be $f(x) = \mathbb{E}[X] = 5.5$. Thus the approximation error would be $\mathbb{E}[(f^{*}(X) - 5.5)^{2}] = \mathbb{E}[(Y - 5.5)^{2}] = Var(Y) = \frac{33}{4} = 8.25$.

(c)

(i) Hypothesis space $F$ of affine functions $f(x) = a + bx$. The best estimation function within this hypothesis space would have the risk of 0. And so, the approximation error would also be 0. [Reference][1(c)(i)]

(ii) 
$$
\begin{aligned}
    \hat{f}(x) & = x + 1 \\
    R(\hat{f}) - R(f_{F}) & = \mathbb{E}[(Y - X + 1)^{2}] - 0 = \mathbb{E}[1] = 1
\end{aligned}
$$

2.

(a) Since, $\epsilon_{i} \sim \mathcal{N}(0, \sigma^{2})$, $y \sim \mathcal{N}(w^{T}x + b, \sigma^{2})$. Thus, we can write 
$$
\begin{aligned}
    p(y = y_{i} | x_{i}, w_{i}, b) & = \prod_{i=1}^{N} \frac{1}{\sqrt{2\pi\sigma}} \exp \left( -\frac{1}{2} \left( \frac{y_{i} - w_{i}x_{i} - b}{\sigma} \right)^{2} \right) 
\end{aligned}
$$ 
[Reference][2(a)]

(b)
$$
\begin{aligned}
    P(y | \beta) & = \prod_{i=1}^{N} p(y_{i} | x_{i}, w_{i}, b) \\
    & = \left( \frac{1}{\sqrt{2\pi\sigma}} \right)^{N} \exp -\frac{1}{2} \left( \frac{\sum_{i=1}^{n}(y_{i} - w_{i}x_{i} - b)^{2}}{\sigma^{2}} \right) \\
    & \text{Taking log on both sides} \\
    \ln P(y | \beta) & = -n \log \sqrt{2\pi\sigma} - \frac{1}{2\sigma^{2}} \sum_{i=1}^{n} (y_{i} - w_{i}x_{i} - b)^{2}
\end{aligned}
$$

(c)
$$
\begin{aligned}
    ln P(y | \beta) & = -n \log \sqrt{2\pi\sigma} - \frac{1}{2\sigma^{2}} \sum_{i=1}^{n} (y_{i} - w_{i}x_{i} - b)^{2} \\
\end{aligned}
$$
Maximizing the log likelihood is equivalent to minimizing the "summation" in the second term}
$$
\begin{aligned}
    \max_{\beta} ln P(y | \beta) & = \min_{\beta}(\sum_{i=1}^{n} (y_{i} - w_{i}x_{i} - b)^{2}) \\
    & = arg\min_{\beta}(y - x^{T}\beta)^{T}(y - x^{T}\beta)
\end{aligned}
$$
Other terms are constant.

(d) To derive the values of coeffcients $\beta$, we can take derivative of the log likelihood with respect to $\beta$ and equating it to zero.
$$
\begin{aligned}
    J(\beta) & = (y - X\beta)^{T}((y - X\beta)) \\
    & \text{Taking derivative with respect to} \beta \\
    \frac{\partial}{\partial\beta} J(\beta) & = 2X^{T}(X\beta - y) = 0 \\
    & \therefore X^{T}(X\beta - y) = 0 \\
    & \therefore X^{T}X\beta - X^{T}y = 0 \\
    & \therefore \beta = (X^{T}X)^{-1}X^{T}y
\end{aligned}
$$

3.

(a)
$$
\begin{aligned}
    \hat{\beta} & = arg\min_{\beta} \{ (Y - X\beta)^{T}(Y - X\beta) + \lambda \beta^{T}\beta \} \\
    \frac{\partial\hat{\beta}}{\partial\beta} & = \frac{\partial(Y - X\beta)^{T}(Y - X\beta)}{\partial\beta} + \frac{\partial\lambda\beta^{T}\beta}{\partial\beta} \\
    & = \frac{Y^{T}Y - 2\beta^{T}X^{T}Y + \beta^{T}X^{T}X\beta}{\partial\beta} + \frac{\lambda\beta^{T}\beta}{\partial\beta} \\
    & = -2X^{T}(Y - \beta^{T}X) + 2\lambda\beta \\
    \therefore \beta & = (X^{T}X + \lambda \mathbb{I})^{-2}X^{T}Y \\
    & = (A^{T}A + \lambda \mathbb{I})^{-2}A^{T}Y
\end{aligned}
$$
[Reference][3(a)]

(b)

First of all, $A^{T}A$ is a symmetric matrix. Let $A^{T}A$ have dimensions $n \times n$. Therefore, for it to be full rank, the rank of the matrix should be $n$. Further, $A^{T}A$ will always be a semi-definite matrix and all of its __eigen values would be greater than zero__ $v \geq 0 \ \forall \ v \in V$. 

Next, any eigen vector $\textbf{v}_{i}$ of $A^{T}A$ will be the eigen vector of $(A^{T}A + \lambda\mathbb{I})$ scaled as $v_{i} + \lambda$.

$$
\begin{aligned}
    (A^{T}A + \lambda\mathbb{I})\textbf{v}_{i} & = \underbrace{A^{T}A\textbf{v}_{i}}_{A\textbf{v} = u\textbf{v}} + \lambda\mathbb{I} \textbf{v}_{i} \\
    & = (v_{i} + \lambda)\textbf{v}_{i} \\
\end{aligned}
$$

Thus, we can show that the eigen values of $(A^{T}A + \lambda\mathbb{I}) \geq0$ and therefore it is full rank and invertible.
[Reference][3(b)]

4.

(a) 

We can write $\mu$ in terms of the training data $x = (x_{1}, x_{2}, \cdots, x_{n})$ using Bayes' rule

$$
\begin{aligned}
    P(\mu | x) & = \frac{\overbrace{P(x | \mu)}^{\mathcal{N(\mu, \sigma)}}\overbrace{P(\mu)}^{\mathcal{N}(\mu_{0}, \sigma_{0})}}{\underbrace{P(x)}_{constant}} \\
    P(x | \mu) & = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^{2}}} \exp\left( - \frac{(x_{i} - \mu)^{2}}{2\sigma^{2}} \right) \\
    P(\mu) & = \frac{1}{\sqrt{2\pi\sigma_{0}^{2}}}\exp{ \left( -\frac{(\mu - \mu_{0})^{2}}{2\sigma_{0}^{2}} \right) }
\end{aligned}
$$
Next, taking log on both sides, we get
$$
\begin{aligned}
    \log(P(\mu | x)) & = \left( \sum_{i=1}^{n} -\log \left( \sqrt{2\pi\sigma^{2}} \right) \right) - \log \left( \sqrt{2\pi\sigma_{0}^{2}} \right) - \frac{(\mu - \mu_{0})^{2}}{2\sigma_{0}^{2}}
\end{aligned}
$$
We can take derivative with respect to $\mu$ and equate it to 0.
$$
\begin{aligned}
    \frac{\partial\log(P(\mu|x))}{\partial\mu} & = \left( \sum_{i=1}^{n} \frac{x_{i} - \mu}{\sigma^{2}} \right) - \frac{\mu - \mu_{0}}{\sigma_{0}^{2}} = 0 \\
    \therefore \mu & = \frac{\sigma^{2}\mu_{0} + \sigma_{0}^{2}\sum_{i=1}^{n}x_{i}}{\sigma^{2} + n\sigma_{0}^{2}}
\end{aligned}
$$

(b) 

Let $\bar{x}$ be sampled from Gaussian distribution $x_{i} \sim \mathcal{N}(\mu, \sigma)$. The likelihood function can be written as
$$
\begin{aligned}
    p(\bar{x_{i}} | \mu, \sigma) & = \prod_{i=1}^{n}f(x_{i}; \mu, \sigma) \\
    & = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^{2}}} \exp\left( - \frac{(x_{i} - \mu)^{2}}{2\sigma^{2}} \right) \\    
\end{aligned}
$$
Next, taking log on both sides, we get
$$
\begin{aligned}
    p(\bar{x_{i}} | \mu, \sigma) & = \sum_{i=1}^{n} \left( - \log{\sqrt{2\pi\sigma^{2}}} - \frac{(x_{i} - \mu)^{2}}{2\sigma^{2}} \right) \\
    & = \sum_{i=1}^{n} \left( - \frac{1}{2} \log{2\pi\sigma^{2}} - \frac{1}{2} \frac{(x_{i} - \mu)^{2}}{2\sigma^{2}} \right) \\
    & = \frac{n}{2} \log(2\pi\sigma^{2}) + \sum_{i=1}^{n} -\frac{1}{2\sigma^{2}} (x_{i} - \mu)^{2} \\
\end{aligned}
$$
To obtain the estimate for $\mu$ we can derivate with respect to $\mu$ and equate it to zero.
$$
\begin{aligned}
    & \frac{\partial}{\partial\mu} \left( \frac{n}{2} \log(2\pi\sigma^{2}) + \sum_{i=1}^{n} -\frac{1}{2\sigma^{2}} (x_{i} - \mu)^{2} \right) \\
    & = \frac{\partial}{\partial\mu} \left( -\frac{n}{2} \log (2\pi\sigma^{2}) \right) - \frac{\partial}{\partial\mu} \left( \frac{1}{2\sigma^{2}} \sum_{i=1}^{n} (x_{i} - \mu)^{2} \right) \\
    & = \frac{\partial}{\partial\mu} \left( - \frac{1}{2\sigma^{2}} \sum_{i=1}^{n} (x_{i} - \mu)^{2} \right) \\
    & = - \frac{1}{2\sigma^{2}} \frac{\partial}{\partial\mu} \left( \sum_{i=1}^{n} (x_{i} - \mu)^{2} \right) \\
    & = \frac{1}{\sigma^{2}} \sum_{i=1}^{n} (x_{i} - \mu)^{2} = 0 \\
    \therefore \mu & = \frac{1}{n} \sum_{i=1}^{n} x_{i}
\end{aligned}
$$
Next, to obtain the estimate for $\sigma^{2}$, we can follow similar procedure.
$$
\begin{aligned}
    & \frac{\partial}{\partial\sigma^{2}} \left( \frac{n}{2} \log(2\pi\sigma^{2}) + \sum_{i=1}^{n} -\frac{1}{2\sigma^{2}} (x_{i} - \mu)^{2} \right) \\
    & = \frac{\partial}{\partial\sigma^{2}} \left( -\frac{n}{2} \log (2\pi\sigma^{2}) \right) - \frac{\partial}{\partial\sigma^{2}} \left( \frac{1}{2\sigma^{2}} \sum_{i=1}^{n} (x_{i} - \mu)^{2} \right) \\
    & = -\frac{n}{2} \frac{\partial}{\partial\sigma^{2}} (\log(2\pi\sigma^{2})) + \frac{\partial}{\partial\sigma^{2}} \left( -\frac{1}{2\sigma^{2}} \sum_{i=1}^{n} (x_{i} - \mu)^{2} \right) \\
    & = -\frac{n}{\sigma^{2}} + \frac{1}{2\sigma^{4}} \sum_{i=1}^{n} (x_{i} - \mu)^{2} \\
    & = \frac{1}{2\sigma^{2}} \left( -n + \frac{1}{\sigma^{2}} \sum_{i=1}^{n} (x_{i} - \mu)^{2} \right) \\
    \therefore \sigma^{2} & =  \frac{1}{n} \sum_{i=1}^{n}(x_{i} - \mu)^{2}
\end{aligned}
$$
Thus, we can show that the likelihood of a Gaussian sample also follows a Gaussian distribution. <span style="color:blue"> [Reference] </span>
[4(b)]


[1(c)(i)]: https://davidrosenberg.github.io/ml2017/#lectures
[2(a)]: https://stats.stackexchange.com/questions/327427/how-is-y-normally-distributed-in-linear-regression
[3(a)]: https://stats.stackexchange.com/questions/69205/how-to-derive-the-ridge-regression-solution
[3(b)]: https://statisticaloddsandends.wordpress.com/2018/01/31/xtx-is-always-positive-semidefinite/
[4(b)]: https://jrmeyer.github.io/machinelearning/2017/08/18/mle.html