## 第4章 潜在因子の推定方法

本章では、因子分析における潜在因子の推定方法を数理的な展開とともに詳細に説明します。特に、途中式の展開や各ステップの意味について丁寧に解説し、読者が推定過程を理解しやすいように工夫しています。

### 4.1 因子分析の基本モデル

因子分析では、観測変数ベクトル $\mathbf{x}$ を次の線形モデルで表します。

$$
\mathbf{x} = \Lambda \mathbf{f} + \boldsymbol{\epsilon},
$$

- **$\mathbf{x}$**: $p \times 1$ の観測変数ベクトル  
- **$\Lambda$**: $p \times m$ の因子負荷行列  
- **$\mathbf{f}$**: $m \times 1$ の潜在因子ベクトル  
- **$\boldsymbol{\epsilon}$**: $p \times 1$ の特有因子（誤差）ベクトル

#### 4.1.1 仮定

1. **潜在因子の分布**  
   潜在因子は平均0、分散共分散行列が単位行列 $ \mathbf{I}_m $ と仮定します。

   $$
   \mathbf{f} \sim N(\mathbf{0}, \mathbf{I}_m).
   $$

2. **特有因子の分布**  
   特有因子は平均0、対角行列 $\Psi$（各変数の固有分散）を持つと仮定します。

   $$
   \boldsymbol{\epsilon} \sim N(\mathbf{0}, \Psi) \quad \text{で、} \quad \Psi = \operatorname{diag}(\psi_1, \psi_2, \dots, \psi_p).
   $$

3. **独立性**  
   潜在因子と特有因子は互いに独立であると仮定します。

これらの仮定の下、観測変数 $\mathbf{x}$ の分散共分散行列 $\Sigma$ は以下のように導かれます。

$$
\Sigma = \operatorname{Cov}(\mathbf{x}) = \Lambda \Lambda^\top + \Psi.
$$

---

### 4.2 最尤法によるパラメータ推定

最尤法は、観測データが多変量正規分布に従うと仮定し、モデルパラメータ $\Lambda$ と $\Psi$ を推定する手法です。

#### 4.2.1 尤度関数の定式化

$ n $ 個の独立な観測データ $\{ \mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n \}$ に対し、各 $\mathbf{x}_i$ は次の多変量正規分布に従うとします。

$$
\mathbf{x}_i \sim N(\mathbf{0}, \Sigma), \quad \text{ただし} \quad \Sigma = \Lambda\Lambda^\top + \Psi.
$$

各観測の確率密度関数は、

$$
p(\mathbf{x}_i \mid \Lambda, \Psi) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} \mathbf{x}_i^\top \Sigma^{-1} \mathbf{x}_i\right).
$$

全サンプルの尤度関数 $L(\Lambda, \Psi)$ は、独立性から

$$
L(\Lambda, \Psi) = \prod_{i=1}^{n} \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} \mathbf{x}_i^\top \Sigma^{-1} \mathbf{x}_i\right).
$$

#### 4.2.2 対数尤度の導出

対数をとると、計算が容易になります。  
まず、対数を取ると、

$$
\begin{aligned}
\ell(\Lambda, \Psi) &= \ln L(\Lambda, \Psi) \\
&= \sum_{i=1}^{n} \ln \left[ \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} \mathbf{x}_i^\top \Sigma^{-1} \mathbf{x}_i\right) \right] \\
&= -\frac{np}{2} \ln (2\pi) - \frac{n}{2} \ln |\Sigma| - \frac{1}{2} \sum_{i=1}^{n} \mathbf{x}_i^\top \Sigma^{-1} \mathbf{x}_i.
\end{aligned}
$$

さらに、サンプル共分散行列 $ S $ を

$$
S = \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i \mathbf{x}_i^\top
$$

と定義すると、次の性質を利用できます。

$$
\sum_{i=1}^{n} \mathbf{x}_i^\top \Sigma^{-1} \mathbf{x}_i = n\, \operatorname{tr}(S \Sigma^{-1}).
$$

よって、対数尤度は以下のように整理されます。

$$
\ell(\Lambda, \Psi) = -\frac{n}{2} \left[ \ln |\Sigma| + \operatorname{tr}(S \Sigma^{-1}) \right] + \text{定数}.
$$

#### 4.2.3 パラメータ推定の目的

最尤法では、この対数尤度 $\ell(\Lambda, \Psi)$ を最大化する $\Lambda$ と $\Psi$ を求めます。  
ただし、解析的に閉じた解を得るのは困難なため、数値的な最適化手法（EMアルゴリズムやNewton-Raphson法）を利用して反復的に推定を行います。

---

### 4.3 EMアルゴリズムによる推定

EMアルゴリズムは、潜在変数 $ \mathbf{f} $ を「欠測データ」とみなして推定を行う方法です。ここでは、各ステップの概要と途中式の意味を解説します。

#### 4.3.1 Eステップ（期待値計算）

現在のパラメータ $\Lambda^{(t)}$ と $\Psi^{(t)}$ を用いて、各観測 $\mathbf{x}_i$ に対する潜在因子の条件付き分布を求めます。  
条件付き分布は

$$
\mathbf{f}_i \mid \mathbf{x}_i \sim N\left( \mathbf{M} \mathbf{x}_i, \, \mathbf{V} \right)
$$

と表され、ここで

- $\mathbf{M} = \Lambda^\top (\Lambda\Lambda^\top + \Psi)^{-1}$
- $\mathbf{V} = \mathbf{I}_m - \Lambda^\top (\Lambda\Lambda^\top + \Psi)^{-1}\Lambda$

となります。

このステップでは、以下の2点を計算します。

- **条件付き期待値:**  
  $ E(\mathbf{f}_i \mid \mathbf{x}_i) = \mathbf{M} \mathbf{x}_i $

- **条件付き共分散:**  
  $ \operatorname{Var}(\mathbf{f}_i \mid \mathbf{x}_i) = \mathbf{V} $

#### 4.3.2 Mステップ（パラメータ更新）

Eステップで得られた条件付き期待値と共分散を用い、完全データ（観測値と潜在変数を含む）の対数尤度の期待値を最大化する形で、パラメータ $\Lambda$ と $\Psi$ を更新します。  
更新式は以下のようになります。

$$
\Lambda^{(t+1)} = \left[ \sum_{i=1}^{n} \mathbf{x}_i \, E(\mathbf{f}_i^\top \mid \mathbf{x}_i) \right] \left[ \sum_{i=1}^{n} E(\mathbf{f}_i \mathbf{f}_i^\top \mid \mathbf{x}_i) \right]^{-1},
$$

$$
\Psi^{(t+1)} = \operatorname{diag}\left\{ \frac{1}{n} \sum_{i=1}^{n} \left( \mathbf{x}_i \mathbf{x}_i^\top - \Lambda^{(t+1)} \, E(\mathbf{f}_i \mathbf{x}_i^\top \mid \mathbf{x}_i) \right) \right\}.
$$

これらの式は、**最小二乗推定**の考え方に基づき、観測値 $\mathbf{x}_i$ と潜在因子の条件付き期待値との「ズレ」を最小化する形になっています。  
EステップとMステップを交互に繰り返すことで、対数尤度が収束するまでパラメータの更新が行われます。

---

### 4.4 その他の推定方法

#### 4.4.1 主軸因子法（Principal Axis Factoring）

1. **共通性の推定:**  
   各変数の多重相関係数などを用いて、初期の共通性 $ h_i^2 $（変数 $ x_i $ の共通部分の分散）を推定します。

2. **固有値分解:**  
   サンプルの相関行列 $ R $ に対して固有値分解を行い、固有値が1以上の因子数を選定します。  
   選定された因子の固有ベクトルを基に、因子負荷量の初期推定値を決定します。

#### 4.4.2 最小残差法（Minimum Residual Method）

実際のサンプル共分散行列 $ S $ と、モデルで再現される共分散行列 $\hat{\Sigma} = \Lambda \Lambda^\top + \Psi$ との差（残差）を最小化する方法です。  
目的関数として、フロベニウスノルムを用いる場合、

$$
\min_{\Lambda, \Psi} \| S - (\Lambda \Lambda^\top + \Psi) \|_F^2,
$$

を解く形になります。なお、$\Psi$ は対角行列であるという制約があります。

---

### 4.5 潜在因子（因子得点）の推定

パラメータ $\Lambda$ と $\Psi$ の推定が完了した後、各観測 $ \mathbf{x} $ に対する潜在因子（因子得点）を推定する必要があります。以下に代表的な2つの方法を示します。

#### 4.5.1 バートレット法（Bartlett's Method）

バートレット法では、誤差の影響を最小化する重み付けを行い、次の式で因子得点を求めます。

$$
\hat{\mathbf{f}} = \left( \Lambda^\top \Psi^{-1} \Lambda \right)^{-1} \Lambda^\top \Psi^{-1} \mathbf{x}.
$$

この式は、観測値 $\mathbf{x}$ から潜在因子の影響を逆算する形となっています。

#### 4.5.2 回帰法（Regression Method）

回帰法は、観測変数から因子空間への射影を最小二乗的に行う方法です。  
推定式は以下の通りです。

$$
\hat{\mathbf{f}} = \Lambda^\top \left( \Lambda \Lambda^\top + \Psi \right)^{-1} \mathbf{x}.
$$

この方法では、観測値に対してパラメータ $\Lambda$ と $\Psi$ に基づいた線形回帰を行い、潜在因子の得点を算出します。

---

### 4.6 まとめ

- **基本モデル:**  
  観測変数は、$\mathbf{x} = \Lambda \mathbf{f} + \boldsymbol{\epsilon}$ というモデルに基づき、分散共分散行列は $\Sigma = \Lambda \Lambda^\top + \Psi$ と表されます。

- **最尤法:**  
  多変量正規分布を仮定し、対数尤度

  $$
  \ell(\Lambda, \Psi) = -\frac{n}{2} \left[ \ln |\Sigma| + \operatorname{tr}(S \Sigma^{-1}) \right] + \text{定数}
  $$

  を最大化することでパラメータを推定します。解析的解が得にくいため、EMアルゴリズムやNewton-Raphson法などの反復法が用いられます。

- **その他の手法:**  
  主軸因子法や最小残差法など、データや解析目的に応じた代替手法も存在します。

- **因子得点の推定:**  
  バートレット法や回帰法を用いて、推定されたパラメータから各観測の潜在因子を算出します。
