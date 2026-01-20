
<!-- Language switcher badges -->

[![English](https://img.shields.io/badge/English-blue)](./README.md)
[![日本語](https://img.shields.io/badge/日本語-red)](./README.ja.md)



# Volatility Decomposition

This repository contains code written for a part of my Master's degree in Finance \& Investments. This code represents an algorithm used for my thesis where realised variance, computed from intra-day logarithmic returns, is decomposed into a continuous and jump component.

For speed optimization, the computational heavy parts are written as C-extensions and consequently called from a Python wrapper. The final code is tightly put together in a user-friendly pipeline.

# Algorithm Explanation
## Step 1: Realised Variance (RV)

We assume that the asset price evolution is governed by a jump-diffusion process. In traditional asset pricing, the independent jumps are driven by a Poisson process. Hence, we define the following dynamics for the log-process of the stock price, $X(t) \coloneqq \log S(t)$, under real-world measure $\mathbb{P}$ [[1]](#1):



$$
	\text{d} X(t) = \mu\text{d}t + \sigma \text{d}W^{\mathbb{P}}(t) + J\text{d}X_{\mathcal{P}}(t),
$$

where $\mu$ is the drift term, $\sigma$ is the stochastic volatility process, $W(t)$ is a standard Geometric Brownian Motion, and $X_{\mathcal{P}}(t)$ is a Poisson process in which $J$ gives the jump magnitude. Realised volatility of the asset price is given by the cumulative squared intraday logarithmic returns. Let $r_{t+j,\delta} = X_{t+j,\delta} - X_{t+(j-1),\delta}$ denote the intraday return for process $X(t)$, where the log-return $r_{t+j,\delta}$ refers to the $j$*-th* realised price change on trading day $t$. Realised Variance ($RV$) is then defined by the summation of the $1/\delta$ intraday returns [[2]](#2):

$$
	RV_{t,t+1}(\delta) = \sum\limits_{j=1}^{1/\delta} r_{t+j,\delta}^2
$$

## Step 2: Bi-Power Variation (BV)
[[3]](#3), [[4]](#4) show that realised variance converges in probability to quadratic variation as $\delta$ tend to 0 (i.e., when the intraday sampling frequency increases). This results in the following properties of $RV$ [[2]](#2):

$$
 	RV_{t,t+1}(\delta) \rightarrow \underbrace{\int_{t}^{t+1} \sigma^2(s)\text{d}s}_{C_{t+1}} + \underbrace{\sum\limits_{t < s \le t+1} J^{2}(s)}_{J_{t+1}}
$$

As a result, realised variance consists of two components: the continuous sample path ($C$) and a jump ($J$) component. Realised Bi-Power Variation ($BV$) allows for separating the components [[2]](#2), [[5]](#5):

$$
	BV_{t+1}(\delta) = \mu_{1}^{-2} \sum\limits_{j=2}^{1/\delta} |{r_{t+j,\delta}}| |{r_{t+(j-1),\delta}}|,
$$

where $\mu_{1} = \sqrt{2/\pi}$. $BV$ converges to $C$ as $\delta$ goes to zero. $BV$ is smoothed by summing the absolute adjacent log-returns, and hence, is robust to infrequent jumps.

## Step 3: The Jump Component (J)
The jump component, $J$, is then computed as the difference between $RV$ and $BV$, and additionally imposed with a non-negativity constraint [[2]](#2):

$$
	J_{t+1} = \text{max}(RV_{t+1}(\delta) - BV_{t+1}(\delta), 0 ) \cdot 1_{ ( Z_{t+1}>\phi_{1-\alpha} ) },
$$

where $1_{( Z_{t}>\phi_{1-\alpha} ) }$ is an indicator function equal to 1 if the jump is significant, and 0 otherwise. $\phi_{\alpha}$ is the $\alpha$ quantile of the Gaussian distribution. The $Z$-statistic is defined as [[6]](#6), [[7]](#7):

$$
	Z_{t+1}(\delta) = \frac{ (RV_{t+1}(\delta) - BV_{t+1}(\delta)) RV_{t+1}(\delta)^{-1} }{ \sqrt{ ( (\mu_{1}^{-4} + 2\mu_{1}^{-2} - 5) \cdot \text{max}(1,TQ_{t+1}(\delta) BV_{t+1}^{-2}(\delta) ) ) } },
$$

 where $TQ_{t+1}(\delta)$ refers to Tri-Power Quarticity:

$$
 	TQ_{t+1}(\delta) = \mu_{4/3}^{-3} \cdot \delta^{-1} \cdot \sum\limits_{j=2}^{1/\delta} |{r_{t+j,\delta}}^{4/3}| |{r_{t+(j-1),\delta}}^{4/3}| |{r_{t+(j-2),\delta}}^{4/3}|,
$$

where $\mu_{p} = \pi^{-1/2}2^{p/2}\Gamma \Big(\frac{p+1}{2}\Big)$, and $\Gamma (z)$ is the gamma function. The suggested significance level for the jump test is $\alpha = .0001$ to construct the $J$ and $C$ time series.

## Step 4: The Continous Component (C)
The continuous component, $C$, is the remainder of $RV$ after a significant jump, i.e.,

$$
	C_{t+1} = RV_{t+1} - J_{t+1}
$$

## Step 5: Finalisation
The observations should cover non-overlapping periods. Thus, for the components described above ($RV$, $C$, and $J$), one can compute the $k$-day realised observation with the general formula below, using $k=22$ for monthly measures [[2]](#2). Note that, under stationarity, we have $\mathbb{E}\left[RV_{t,t+k}\right] = 252 \cdot \mathbb{E}\left[RV_{t+1}\right]$. In this process, we assume that the underlying series are not autocorrelated.

$$
	\hat{x}_{t} = 252\cdot k^{-1} ( \hat{x}_{t+1} + \ldots + \hat{x}_{t+k} )
$$

# References
<a id="1">[1]</a>
Oosterlee, C. W., \& Grzelak, L. A. (2019). *Mathematical Modeling and Computation in Finance*. World Scientific Publishing Europe Ltd.

<a id="2">[2]</a>
Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2003). Some like it smooth, and some like it rough: Untangling continuous and jump components in measuring, modeling, and forecasting asset return volatility. *Modeling, and Forecasting Asset Return Volatility (September 2003).*

<a id="3">[3]</a>
Barndorff-Nielsen, O. E., & Shephard, N. (2002a). Econometric analysis of realized volatility and its use in estimating stochastic volatility models. *Journal of the Royal Statistical Society: Series B (Statistical Methodology), 64*(2), 253–280.

<a id="4">[4]</a>
Barndorff-Nielsen, O. E., & Shephard, N. (2002b). Estimating quadratic variation using realized variance. *Journal of Applied Econometrics, 17*(5), 457–477.

<a id="5">[5]</a>
Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and bipower variation with stochastic volatility and jumps. *Journal of Financial Econometrics, 2*(1), 1–37.

<a id="6">[6]</a>
Nolte, I., & Xu, Q. (2015). The economic value of volatility timing with realized jumps. *Journal of Empirical Finance, 34*, 45–59.

<a id="7">[7]</a>
Huang, X., & Tauchen, G. (2005). The relative contribution of jumps to total price variance. *Journal of Financial Econometrics, 3*(4), 456–499.

# Getting Started
Clone the repository
```bash
git clone https://github.com/justkroft/vol_decomposition.git
cd vol_decomposition
```

and setup your virtual environment using `uv`.

```bash
uv venv  # create environment
uv sync  # sync all dependencies from toml file
```

Then, activate the environment:

```bash
source .venv/bin/activate  # on Mac
./venv/Scripts/activate    # on Windows
```

Once the environment is setup and all the dependencies are installed, compile the C-extenstions in your terminal:

```bash
python setup.py build_ext --inplace
```

Please take a look at [the example notebook](example.ipynb) for an example of how to use the code, including generating fake data for testing, as well as plotting the results.

# License
[MIT](LICENSE) License © 2026-PRESENT [justkroft](https://github.com/justkroft)
