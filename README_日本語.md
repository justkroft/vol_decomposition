# ボラティリティ分解

本コードは、金融・投資修士課程の卒業論文で用いられるアルゴリズムを示している。計算速度の最適化のため、計算負荷の高い部分はCで実装され、Pythonラッパーから呼び出す構成となっている。最終的なコードは、使用しやすい形で一貫したパイプラインとして統合されている

# アルゴリズムの解説
## 第一： 実現分散 (RV)
資産価格の時間的移動がジャンプ・ディフゥージョン過程に従うものとする。資産価格理論においては独立なジャンプはポアソン課程によって記述される。従って、株価も対数過程　$X(t) \coloneqq \log S(t)$　に対して、以下のダイナミクスを定義する[[1]](#1)。

$$
	\text{d} X(t) = \mu\text{d}t + \sigma \text{d}W^{\mathbb{P}}(t) + J\text{d}X_{\mathcal{P}}(t),
$$

上記の数式で、$\mu$ はドリフト項、$\sigma$ は確率的ボラティリティ過程を表し、$W(t)$ は標準ブラウン運動である。又、$X_{\mathcal{P}}(t)$ はポアソン過程を表し、$J$ はジャンプの大きさを与える。資産価格の実現ボラティリティは、日中に観測される対数リターンの二乗和として定義される。過程 $X(t)$ に対する日中リターンを $r_{t+j,\delta} = X_{t+j,\delta} - X_{t+(j-1),\delta}$ と定義し、ここで対数リターン $r_{t+j,\delta}$ は、取引日 $t$ における $j$*番目*の現実価格変化を表すものとする。それに伴い、実現分散 (Realised Variance, $RV$) は、$1/\delta$ 個の日中リターンの総和として定義される[[2]](#2)。

$$
	RV_{t,t+1}(\delta) = \sum\limits_{j=1}^{1/\delta} r_{t+j,\delta}^2
$$

## 第二： バイパワー変動 (BV)
[[3]](#3)、[[4]](#4)は、$\delta \to 0$（すなわち日中のサンプリング頻度が無限大に近づく場合）において、実現分散が確率収束の意味で quadratic variation に収束することを示している。その結果、$RV$ は以下の性質を持つことが知られている[[2]](#2):

$$
 	RV_{t,t+1}(\delta) \rightarrow \underbrace{\int_{t}^{t+1} \sigma^2(s)\text{d}s}_{C_{t+1}} + \underbrace{\sum\limits_{t < s \le t+1} J^{2}(s)}_{J_{t+1}}
$$

すなわち、実現分散は連続な連続なサンプルパス成分 ($C$) とジャンプ成分 ($J$) の二つから構成される。実現バイパワー変動 (Realised Bi-Power Variation, $BV$) は、これら二つの成分を分離することを可能にする[[2]](#2)、[[5]](#5):

$$
	BV_{t+1}(\delta) = \mu_{1}^{-2} \sum\limits_{j=2}^{1/\delta} |{r_{t+j,\delta}}| |{r_{t+(j-1),\delta}}|,
$$

上記の数式で、$\mu_{1} = \sqrt{2/\pi}$ である。$\delta \to 0$の極限において、$BV$ は連続成分 $C$ に収束することが示されている。$BV$ は隣接する対数リターンの絶対値の積を加算することによって平滑化されており、その結果として、低頻度で発生するジャンプに対してロバストな性質を持つ。


## 第三：ジャンプ成分 (J)
ジャンプ成分 $J$ は、$RV$ と $BV$ の差として計算され、加えて推定値が負とならないよう非負制約が課されている[[2]](#2)。

$$
	J_{t+1} = \text{max}(RV_{t+1}(\delta) - BV_{t+1}(\delta), 0 ) \cdot 1_{ ( Z_{t+1}>\phi_{1-\alpha} ) },
$$

ここで、 $1_{( Z_{t}>\phi_{1-\alpha} ) }$ はジャンプが統計的に有意である場合に１、それ以外の場合に０をとる指示関数である。又、 $\phi_{\alpha}$ は正規分布の $\alpha$ 分位点を表す。$Z$ 統計量は、以下のように定義される[[6]](#6)、[[7]](#7)。

$$
	Z_{t+1}(\delta) = \frac{ (RV_{t+1}(\delta) - BV_{t+1}(\delta)) RV_{t+1}(\delta)^{-1} }{ \sqrt{ ( (\mu_{1}^{-4} + 2\mu_{1}^{-2} - 5) \cdot \text{max}(1,TQ_{t+1}(\delta) BV_{t+1}^{-2}(\delta) ) ) } },
$$

ここで、 $TQ_{t+1}(\delta)$ はトライパワー・クォーティシティ (Tri-Power Quarticity) を表し、次式のように定義定義される：

$$
 	TQ_{t+1}(\delta) = \mu_{4/3}^{-3} \cdot \delta^{-1} \cdot \sum\limits_{j=2}^{1/\delta} |{r_{t+j,\delta}}^{4/3}| |{r_{t+(j-1),\delta}}^{4/3}| |{r_{t+(j-2),\delta}}^{4/3}|,
$$

$\mu_{p} = \pi^{-1/2}2^{p/2}\Gamma \Big(\frac{p+1}{2}\Big)$ と $\Gamma (z)$ はガンマ関数である。ジャンプ検定において推奨される有意水準は $\alpha = .0001$ であり、これを用いてジャンプ成分 $J$ および連続成分 $C$ の時系列を構築する。

## 第四：連続成分 (C)
連続成分 $C$ は、有意なジャンプを除いた後に残る実現分散として定義され、すなわち次式で与えられる：

$$
	C_{t+1} = RV_{t+1} - J_{t+1}
$$

## 第五：完了
観測値は、互いに重ならない期間を対象とする必要がある。従って、上述した各成分 ($RV$, $C$, および $J$) について、以下の一般式を用いることで $k$ 日間の実現値を算出することができる。月次指標を構築する場合には、$k=22$ を用いる[[2]](#2)。また、定常性の仮定の下では $\mathbb{E}\left[RV_{t,t+k}\right] = 252 \cdot \mathbb{E}\left[RV_{t+1}\right]$ が成り立つことに留意されたい。本手続きにおいては、基礎となる時系列が自己相関を持たないものと仮定する。

$$
	\hat{x}_{t} = 252\cdot k^{-1} ( \hat{x}_{t+1} + \ldots + \hat{x}_{t+k} )
$$

# 参考文献
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

# License
[MIT](LICENSE) License © 2026-PRESENT [justkroft](https://github.com/justkroft)
