# 線形代数学 講義ノート：第27回

## 1. 講義情報と予習ガイド

**講義回**: 第27回  
**テーマ**: 内積と正規直交基底・グラムシュミット直交化法  
**関連項目**: ベクトル空間、内積空間、直交基底、グラムシュミット直交化法  
**予習内容**: ベクトル空間の基本的な性質（第26回の内容）を復習しておくこと。特に基底の概念と線形独立性について理解しておくことが重要です。

## 2. 学習目標

1. ベクトルの内積と内積空間の定義を理解し、その性質を説明できる
2. 内積を用いてベクトル間の角度を計算できる
3. 直交基底と正規直交基底の違いを理解し、その性質を説明できる
4. グラムシュミット直交化法を用いて、与えられた基底から正規直交基底を構成できる
5. 内積空間の幾何学的解釈を理解し、データサイエンスにおける応用例を説明できる

## 3. 基本概念

### 3.1 ベクトルの内積

#### 3.1.1 内積の定義

> **定義（内積）**: ベクトル空間 $V$ 上の内積とは、任意の $\boldsymbol{u}, \boldsymbol{v}, \boldsymbol{w} \in V$ と任意のスカラー $c$ に対して以下の性質を満たす二項演算 $\langle \cdot, \cdot \rangle: V \times V \rightarrow \mathbb{R}$ である。
> 
> 1. 対称性: $\langle \boldsymbol{u}, \boldsymbol{v} \rangle = \langle \boldsymbol{v}, \boldsymbol{u} \rangle$
> 2. 線形性: $\langle c\boldsymbol{u} + \boldsymbol{v}, \boldsymbol{w} \rangle = c\langle \boldsymbol{u}, \boldsymbol{w} \rangle + \langle \boldsymbol{v}, \boldsymbol{w} \rangle$
> 3. 正定値性: $\langle \boldsymbol{v}, \boldsymbol{v} \rangle \geq 0$ かつ $\langle \boldsymbol{v}, \boldsymbol{v} \rangle = 0 \Leftrightarrow \boldsymbol{v} = \boldsymbol{0}$

実ベクトル空間 $\mathbb{R}^n$ における標準内積は、以下のように定義されます：

$$\langle \boldsymbol{u}, \boldsymbol{v} \rangle = \boldsymbol{u}^T \boldsymbol{v} = \sum_{i=1}^{n} u_i v_i$$

ここで、$\boldsymbol{u} = (u_1, u_2, \ldots, u_n)^T$ および $\boldsymbol{v} = (v_1, v_2, \ldots, v_n)^T$ です。

#### 3.1.2 内積とノルム

内積を用いてベクトルのノルム（長さ）を定義することができます：

> **定義（ノルム）**: ベクトル $\boldsymbol{v}$ のノルムは以下のように定義される。
> 
> $$\|\boldsymbol{v}\| = \sqrt{\langle \boldsymbol{v}, \boldsymbol{v} \rangle}$$

$\mathbb{R}^n$ における標準内積によるノルムはユークリッドノルムと呼ばれます：

$$\|\boldsymbol{v}\| = \sqrt{\langle \boldsymbol{v}, \boldsymbol{v} \rangle} = \sqrt{\sum_{i=1}^{n} v_i^2}$$

#### 3.1.3 内積と角度

内積を用いて2つのベクトル間の角度を計算できます：

> **定理**: 非零ベクトル $\boldsymbol{u}, \boldsymbol{v}$ の間の角度 $\theta$ は以下の式で求められる。
> 
> $$\cos \theta = \frac{\langle \boldsymbol{u}, \boldsymbol{v} \rangle}{\|\boldsymbol{u}\| \|\boldsymbol{v}\|}$$

この関係はコーシー・シュワルツの不等式から導かれます：

> **定理（コーシー・シュワルツの不等式）**: 任意のベクトル $\boldsymbol{u}, \boldsymbol{v}$ に対して以下が成り立つ。
> 
> $$|\langle \boldsymbol{u}, \boldsymbol{v} \rangle| \leq \|\boldsymbol{u}\| \|\boldsymbol{v}\|$$
> 
> 等号は $\boldsymbol{u}$ と $\boldsymbol{v}$ が線形従属であるとき、かつそのときに限る。

### 3.2 内積空間

> **定義（内積空間）**: 内積が定義されたベクトル空間を内積空間という。

内積空間では、ベクトル間の角度や距離が定義できるため、幾何学的な議論が可能になります。

### 3.3 直交性

> **定義（直交）**: 二つのベクトル $\boldsymbol{u}, \boldsymbol{v}$ が直交するとは、$\langle \boldsymbol{u}, \boldsymbol{v} \rangle = 0$ が成り立つことである。

直交するベクトルは、幾何学的には互いに垂直であることを意味します。

> **定義（直交補空間）**: 部分空間 $W$ に対する直交補空間 $W^{\perp}$ は以下のように定義される。
> 
> $$W^{\perp} = \{\boldsymbol{v} \in V \mid \langle \boldsymbol{v}, \boldsymbol{w} \rangle = 0 \text{ for all } \boldsymbol{w} \in W\}$$

### 3.4 直交基底と正規直交基底

> **定義（直交基底）**: ベクトル空間 $V$ の基底 $\{\boldsymbol{v}_1, \boldsymbol{v}_2, \ldots, \boldsymbol{v}_n\}$ が直交基底であるとは、任意の $i \neq j$ に対して $\langle \boldsymbol{v}_i, \boldsymbol{v}_j \rangle = 0$ が成り立つことである。

> **定義（正規直交基底）**: 直交基底 $\{\boldsymbol{v}_1, \boldsymbol{v}_2, \ldots, \boldsymbol{v}_n\}$ が正規直交基底であるとは、すべての $i$ に対して $\|\boldsymbol{v}_i\| = 1$ が成り立つことである。つまり、$\langle \boldsymbol{v}_i, \boldsymbol{v}_j \rangle = \delta_{ij}$ （クロネッカーのデルタ）が成り立つ。

正規直交基底は、計算上の利便性や数値的安定性の観点から、多くの数学的および応用的文脈で重要です。

## 4. 理論と手法

### 4.1 内積の具体例と計算

**例題1**: $\mathbb{R}^3$ において、ベクトル $\boldsymbol{u} = (1, 2, 3)^T$ と $\boldsymbol{v} = (4, 5, 6)^T$ の内積を計算しなさい。

**解答**:
標準内積の定義により、
$$\langle \boldsymbol{u}, \boldsymbol{v} \rangle = \boldsymbol{u}^T \boldsymbol{v} = \sum_{i=1}^{3} u_i v_i = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32$$

**例題2**: $\mathbb{R}^3$ において、ベクトル $\boldsymbol{u} = (1, 2, 3)^T$ と $\boldsymbol{v} = (4, 5, 6)^T$ のなす角度を計算しなさい。

**解答**:
まず、ノルムを計算します。
$$\|\boldsymbol{u}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{14}$$
$$\|\boldsymbol{v}\| = \sqrt{4^2 + 5^2 + 6^2} = \sqrt{77}$$

内積は上の例より $\langle \boldsymbol{u}, \boldsymbol{v} \rangle = 32$ です。

コサイン公式より、
$$\cos \theta = \frac{\langle \boldsymbol{u}, \boldsymbol{v} \rangle}{\|\boldsymbol{u}\| \|\boldsymbol{v}\|} = \frac{32}{\sqrt{14} \cdot \sqrt{77}} = \frac{32}{\sqrt{1078}} \approx 0.9747$$

したがって、$\theta \approx 0.2268$ ラジアン、または約 $13.0$ 度です。

### 4.2 直交基底と正規直交基底の例

**例題3**: $\mathbb{R}^3$ における以下のベクトル集合が直交基底かどうか判定しなさい。また、正規直交基底でない場合は、正規直交基底に変換しなさい。

$$\boldsymbol{v}_1 = (1, 1, 1)^T, \boldsymbol{v}_2 = (1, -1, 0)^T, \boldsymbol{v}_3 = (1, 1, -2)^T$$

**解答**:
まず、内積を計算して直交性を確認します。

$$\langle \boldsymbol{v}_1, \boldsymbol{v}_2 \rangle = 1 \cdot 1 + 1 \cdot (-1) + 1 \cdot 0 = 0$$
$$\langle \boldsymbol{v}_1, \boldsymbol{v}_3 \rangle = 1 \cdot 1 + 1 \cdot 1 + 1 \cdot (-2) = 0$$
$$\langle \boldsymbol{v}_2, \boldsymbol{v}_3 \rangle = 1 \cdot 1 + (-1) \cdot 1 + 0 \cdot (-2) = 0$$

すべての内積が0なので、この集合は直交基底です。

次に、各ベクトルのノルムを計算します。

$$\|\boldsymbol{v}_1\| = \sqrt{1^2 + 1^2 + 1^2} = \sqrt{3}$$
$$\|\boldsymbol{v}_2\| = \sqrt{1^2 + (-1)^2 + 0^2} = \sqrt{2}$$
$$\|\boldsymbol{v}_3\| = \sqrt{1^2 + 1^2 + (-2)^2} = \sqrt{6}$$

ノルムが1でないので、正規直交基底ではありません。

正規直交基底に変換するには、各ベクトルをそのノルムで割ります。

$$\boldsymbol{e}_1 = \frac{\boldsymbol{v}_1}{\|\boldsymbol{v}_1\|} = \frac{1}{\sqrt{3}}(1, 1, 1)^T = \left(\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}\right)^T$$

$$\boldsymbol{e}_2 = \frac{\boldsymbol{v}_2}{\|\boldsymbol{v}_2\|} = \frac{1}{\sqrt{2}}(1, -1, 0)^T = \left(\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}}, 0\right)^T$$

$$\boldsymbol{e}_3 = \frac{\boldsymbol{v}_3}{\|\boldsymbol{v}_3\|} = \frac{1}{\sqrt{6}}(1, 1, -2)^T = \left(\frac{1}{\sqrt{6}}, \frac{1}{\sqrt{6}}, -\frac{2}{\sqrt{6}}\right)^T$$

これで $\{\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3\}$ は $\mathbb{R}^3$ の正規直交基底となりました。

### 4.3 グラムシュミット直交化法

グラムシュミット直交化法は、与えられた線形独立なベクトル集合から直交基底を構成するアルゴリズムです。

> **アルゴリズム（グラムシュミット直交化法）**:
> 
> 入力: 線形独立なベクトル集合 $\{\boldsymbol{v}_1, \boldsymbol{v}_2, \ldots, \boldsymbol{v}_n\}$
> 
> 出力: 直交基底 $\{\boldsymbol{u}_1, \boldsymbol{u}_2, \ldots, \boldsymbol{u}_n\}$
> 
> 1. $\boldsymbol{u}_1 = \boldsymbol{v}_1$
> 2. $k = 2, 3, \ldots, n$ に対して、以下を計算する:
>    
>    $$\boldsymbol{u}_k = \boldsymbol{v}_k - \sum_{i=1}^{k-1} \frac{\langle \boldsymbol{v}_k, \boldsymbol{u}_i \rangle}{\langle \boldsymbol{u}_i, \boldsymbol{u}_i \rangle} \boldsymbol{u}_i$$

このアルゴリズムでは、各ステップで新しいベクトル $\boldsymbol{v}_k$ から、それまでに得られた直交ベクトル $\boldsymbol{u}_1, \boldsymbol{u}_2, \ldots, \boldsymbol{u}_{k-1}$ の方向への成分を取り除いています。

直交化後、各ベクトルをそのノルムで割ることで、正規直交基底を得ることができます：

$$\boldsymbol{e}_k = \frac{\boldsymbol{u}_k}{\|\boldsymbol{u}_k\|}$$


**例題4**: $\mathbb{R}^3$ における以下のベクトル集合にグラムシュミット直交化法を適用し、正規直交基底を求めなさい。

$$\boldsymbol{v}_1 = (2, 0, 1)^T, \boldsymbol{v}_2 = (1, 1, 0)^T, \boldsymbol{v}_3 = (0, 1, 2)^T$$

**解答**:
グラムシュミット直交化法のアルゴリズムに従って計算します。

#### ステップ1: $\boldsymbol{u}_1$ の計算

最初のベクトルはそのまま採用します：

$$\boldsymbol{u}_1 = \boldsymbol{v}_1 = (2, 0, 1)^T$$

#### ステップ2: $\boldsymbol{u}_2$ の計算

$\boldsymbol{u}_2$を求めるために、$\boldsymbol{v}_2$から$\boldsymbol{u}_1$方向への射影成分を引きます。

まず、$\boldsymbol{v}_2$ と $\boldsymbol{u}_1$ の内積を計算します：
$$\langle \boldsymbol{v}_2, \boldsymbol{u}_1 \rangle = 1 \cdot 2 + 1 \cdot 0 + 0 \cdot 1 = 2$$

次に、$\boldsymbol{u}_1$ のノルムの二乗を計算します：
$$\langle \boldsymbol{u}_1, \boldsymbol{u}_1 \rangle = 2^2 + 0^2 + 1^2 = 4 + 0 + 1 = 5$$

$\boldsymbol{u}_1$方向への射影係数は次のようになります：
$$\frac{\langle \boldsymbol{v}_2, \boldsymbol{u}_1 \rangle}{\langle \boldsymbol{u}_1, \boldsymbol{u}_1 \rangle} = \frac{2}{5}$$

$\boldsymbol{u}_1$方向への射影ベクトルは：
$$\frac{\langle \boldsymbol{v}_2, \boldsymbol{u}_1 \rangle}{\langle \boldsymbol{u}_1, \boldsymbol{u}_1 \rangle} \boldsymbol{u}_1 = \frac{2}{5} \cdot (2, 0, 1)^T = \left(\frac{4}{5}, 0, \frac{2}{5}\right)^T$$

これを$\boldsymbol{v}_2$から引いて$\boldsymbol{u}_2$を求めます：
$$\begin{align*}
\boldsymbol{u}_2 &= \boldsymbol{v}_2 - \frac{\langle \boldsymbol{v}_2, \boldsymbol{u}_1 \rangle}{\langle \boldsymbol{u}_1, \boldsymbol{u}_1 \rangle} \boldsymbol{u}_1 \\
&= (1, 1, 0)^T - \left(\frac{4}{5}, 0, \frac{2}{5}\right)^T \\
&= \left(1 - \frac{4}{5}, 1 - 0, 0 - \frac{2}{5}\right)^T \\
&= \left(\frac{5 - 4}{5}, 1, -\frac{2}{5}\right)^T \\
&= \left(\frac{1}{5}, 1, -\frac{2}{5}\right)^T
\end{align*}$$

したがって、$\boldsymbol{u}_2 = \left(\frac{1}{5}, 1, -\frac{2}{5}\right)^T$ となります。

#### ステップ3: $\boldsymbol{u}_3$ の計算

$\boldsymbol{u}_3$を求めるために、$\boldsymbol{v}_3$から$\boldsymbol{u}_1$方向と$\boldsymbol{u}_2$方向への射影成分を引きます。

まず、$\boldsymbol{v}_3$ と $\boldsymbol{u}_1$ の内積を計算します：
$$\langle \boldsymbol{v}_3, \boldsymbol{u}_1 \rangle = 0 \cdot 2 + 1 \cdot 0 + 2 \cdot 1 = 0 + 0 + 2 = 2$$

$\boldsymbol{u}_1$方向への射影係数は：
$$\frac{\langle \boldsymbol{v}_3, \boldsymbol{u}_1 \rangle}{\langle \boldsymbol{u}_1, \boldsymbol{u}_1 \rangle} = \frac{2}{5}$$

$\boldsymbol{u}_1$方向への射影ベクトルは：
$$\frac{\langle \boldsymbol{v}_3, \boldsymbol{u}_1 \rangle}{\langle \boldsymbol{u}_1, \boldsymbol{u}_1 \rangle} \boldsymbol{u}_1 = \frac{2}{5} \cdot (2, 0, 1)^T = \left(\frac{4}{5}, 0, \frac{2}{5}\right)^T$$

次に、$\boldsymbol{v}_3$ と $\boldsymbol{u}_2$ の内積を計算します：
$$\begin{align*}
\langle \boldsymbol{v}_3, \boldsymbol{u}_2 \rangle &= 0 \cdot \frac{1}{5} + 1 \cdot 1 + 2 \cdot \left(-\frac{2}{5}\right) \\
&= 0 + 1 + \left(-\frac{4}{5}\right) \\
&= 1 - \frac{4}{5} \\
&= \frac{5 - 4}{5} \\
&= \frac{1}{5}
\end{align*}$$

次に、$\boldsymbol{u}_2$ のノルムの二乗を計算します：
$$\begin{align*}
\langle \boldsymbol{u}_2, \boldsymbol{u}_2 \rangle &= \left(\frac{1}{5}\right)^2 + 1^2 + \left(-\frac{2}{5}\right)^2 \\
&= \frac{1}{25} + 1 + \frac{4}{25} \\
&= \frac{1}{25} + \frac{25}{25} + \frac{4}{25} \\
&= \frac{1 + 25 + 4}{25} \\
&= \frac{30}{25} \\
&= \frac{6}{5}
\end{align*}$$

$\boldsymbol{u}_2$方向への射影係数は：
$$\frac{\langle \boldsymbol{v}_3, \boldsymbol{u}_2 \rangle}{\langle \boldsymbol{u}_2, \boldsymbol{u}_2 \rangle} = \frac{\frac{1}{5}}{\frac{6}{5}} = \frac{1}{6}$$

$\boldsymbol{u}_2$方向への射影ベクトルは：
$$\begin{align*}
\frac{\langle \boldsymbol{v}_3, \boldsymbol{u}_2 \rangle}{\langle \boldsymbol{u}_2, \boldsymbol{u}_2 \rangle} \boldsymbol{u}_2 &= \frac{1}{6} \cdot \left(\frac{1}{5}, 1, -\frac{2}{5}\right)^T \\
&= \left(\frac{1}{6} \cdot \frac{1}{5}, \frac{1}{6} \cdot 1, \frac{1}{6} \cdot \left(-\frac{2}{5}\right)\right)^T \\
&= \left(\frac{1}{30}, \frac{1}{6}, -\frac{2}{30}\right)^T \\
&= \left(\frac{1}{30}, \frac{1}{6}, -\frac{1}{15}\right)^T
\end{align*}$$

これらの射影ベクトルを$\boldsymbol{v}_3$から引いて$\boldsymbol{u}_3$を求めます：
$$\begin{align*}
\boldsymbol{u}_3 &= \boldsymbol{v}_3 - \frac{\langle \boldsymbol{v}_3, \boldsymbol{u}_1 \rangle}{\langle \boldsymbol{u}_1, \boldsymbol{u}_1 \rangle} \boldsymbol{u}_1 - \frac{\langle \boldsymbol{v}_3, \boldsymbol{u}_2 \rangle}{\langle \boldsymbol{u}_2, \boldsymbol{u}_2 \rangle} \boldsymbol{u}_2 \\
&= (0, 1, 2)^T - \left(\frac{4}{5}, 0, \frac{2}{5}\right)^T - \left(\frac{1}{30}, \frac{1}{6}, -\frac{1}{15}\right)^T
\end{align*}$$

第1成分：
$$\begin{align*}
0 - \frac{4}{5} - \frac{1}{30} &= -\frac{4}{5} - \frac{1}{30} \\
&= -\frac{24}{30} - \frac{1}{30} \\
&= -\frac{25}{30} \\
&= -\frac{5}{6}
\end{align*}$$

第2成分：
$$\begin{align*}
1 - 0 - \frac{1}{6} &= 1 - \frac{1}{6} \\
&= \frac{6}{6} - \frac{1}{6} \\
&= \frac{5}{6}
\end{align*}$$

第3成分：
$$\begin{align*}
2 - \frac{2}{5} - \left(-\frac{1}{15}\right) &= 2 - \frac{2}{5} + \frac{1}{15} \\
&= \frac{30}{15} - \frac{6}{15} + \frac{1}{15} \\
&= \frac{30 - 6 + 1}{15} \\
&= \frac{25}{15} \\
&= \frac{5}{3}
\end{align*}$$

したがって、$\boldsymbol{u}_3 = \left(-\frac{5}{6}, \frac{5}{6}, \frac{5}{3}\right)^T$ となります。

### 直交基底

これで直交基底 $\{\boldsymbol{u}_1, \boldsymbol{u}_2, \boldsymbol{u}_3\}$ が得られました：

$$\boldsymbol{u}_1 = (2, 0, 1)^T, \boldsymbol{u}_2 = \left(\frac{1}{5}, 1, -\frac{2}{5}\right)^T, \boldsymbol{u}_3 = \left(-\frac{5}{6}, \frac{5}{6}, \frac{5}{3}\right)^T$$

### 正規直交基底の計算

次に、各ベクトルをそのノルムで割って正規化します。

$\boldsymbol{u}_1$のノルムを計算します：
$$\begin{align*}
\|\boldsymbol{u}_1\| &= \sqrt{2^2 + 0^2 + 1^2} \\
&= \sqrt{4 + 0 + 1} \\
&= \sqrt{5}
\end{align*}$$

$\boldsymbol{e}_1$を計算します：
$$\begin{align*}
\boldsymbol{e}_1 &= \frac{\boldsymbol{u}_1}{\|\boldsymbol{u}_1\|} \\
&= \frac{1}{\sqrt{5}}(2, 0, 1)^T \\
&= \left(\frac{2}{\sqrt{5}}, 0, \frac{1}{\sqrt{5}}\right)^T
\end{align*}$$

$\boldsymbol{u}_2$のノルムを計算します：
$$\begin{align*}
\|\boldsymbol{u}_2\| &= \sqrt{\left(\frac{1}{5}\right)^2 + 1^2 + \left(-\frac{2}{5}\right)^2} \\
&= \sqrt{\frac{1}{25} + 1 + \frac{4}{25}} \\
&= \sqrt{\frac{1 + 25 + 4}{25}} \\
&= \sqrt{\frac{30}{25}} \\
&= \sqrt{\frac{6}{5}}
\end{align*}$$

$\boldsymbol{e}_2$を計算します：
$$\begin{align*}
\boldsymbol{e}_2 &= \frac{\boldsymbol{u}_2}{\|\boldsymbol{u}_2\|} \\
&= \frac{1}{\sqrt{\frac{6}{5}}}\left(\frac{1}{5}, 1, -\frac{2}{5}\right)^T \\
&= \sqrt{\frac{5}{6}}\left(\frac{1}{5}, 1, -\frac{2}{5}\right)^T
\end{align*}$$

第1成分：
$$\begin{align*}
\sqrt{\frac{5}{6}} \cdot \frac{1}{5} &= \frac{\sqrt{5}}{\sqrt{6}} \cdot \frac{1}{5} \\
&= \frac{\sqrt{5}}{5\sqrt{6}} \\
&= \frac{1}{5} \cdot \frac{\sqrt{5}}{\sqrt{6}} \\
&= \frac{1}{5} \cdot \frac{\sqrt{30}}{\sqrt{36}} \\
&= \frac{1}{5} \cdot \frac{\sqrt{30}}{6} \\
&= \frac{\sqrt{30}}{30}
\end{align*}$$

第2成分：
$$\begin{align*}
\sqrt{\frac{5}{6}} \cdot 1 &= \frac{\sqrt{5}}{\sqrt{6}} \\
&= \frac{\sqrt{5}}{\sqrt{6}} \\
&= \frac{\sqrt{30}}{\sqrt{36}} \\
&= \frac{\sqrt{30}}{6}
\end{align*}$$

第3成分：
$$\begin{align*}
\sqrt{\frac{5}{6}} \cdot \left(-\frac{2}{5}\right) &= -\frac{2}{5} \cdot \frac{\sqrt{5}}{\sqrt{6}} \\
&= -\frac{2\sqrt{5}}{5\sqrt{6}} \\
&= -\frac{2\sqrt{30}}{5\sqrt{36}} \\
&= -\frac{2\sqrt{30}}{5 \cdot 6} \\
&= -\frac{2\sqrt{30}}{30}
\end{align*}$$

したがって、$\boldsymbol{e}_2 = \left(\frac{\sqrt{30}}{30}, \frac{\sqrt{30}}{6}, -\frac{2\sqrt{30}}{30}\right)^T$ となります。

$\boldsymbol{u}_3$のノルムを計算します：
$$\begin{align*}
\|\boldsymbol{u}_3\| &= \sqrt{\left(-\frac{5}{6}\right)^2 + \left(\frac{5}{6}\right)^2 + \left(\frac{5}{3}\right)^2} \\
&= \sqrt{\frac{25}{36} + \frac{25}{36} + \frac{25}{9}} \\
&= \sqrt{\frac{25}{36} \cdot 2 + \frac{25}{9}} \\
&= \sqrt{\frac{25}{36} \cdot 2 + \frac{100}{36}} \\
&= \sqrt{\frac{50}{36} + \frac{100}{36}} \\
&= \sqrt{\frac{150}{36}} \\
&= \sqrt{\frac{25 \cdot 6}{36}} \\
&= \frac{5}{\sqrt{6}}
\end{align*}$$

$\boldsymbol{e}_3$を計算します：
$$\begin{align*}
\boldsymbol{e}_3 &= \frac{\boldsymbol{u}_3}{\|\boldsymbol{u}_3\|} \\
&= \frac{1}{\frac{5}{\sqrt{6}}}\left(-\frac{5}{6}, \frac{5}{6}, \frac{5}{3}\right)^T \\
&= \frac{\sqrt{6}}{5}\left(-\frac{5}{6}, \frac{5}{6}, \frac{5}{3}\right)^T
\end{align*}$$

第1成分：
$$\begin{align*}
\frac{\sqrt{6}}{5} \cdot \left(-\frac{5}{6}\right) &= -\frac{5\sqrt{6}}{5 \cdot 6} \\
&= -\frac{\sqrt{6}}{6}
\end{align*}$$

第2成分：
$$\begin{align*}
\frac{\sqrt{6}}{5} \cdot \frac{5}{6} &= \frac{5\sqrt{6}}{5 \cdot 6} \\
&= \frac{\sqrt{6}}{6}
\end{align*}$$

第3成分：
$$\begin{align*}
\frac{\sqrt{6}}{5} \cdot \frac{5}{3} &= \frac{5\sqrt{6}}{5 \cdot 3} \\
&= \frac{\sqrt{6}}{3}
\end{align*}$$

したがって、$\boldsymbol{e}_3 = \left(-\frac{\sqrt{6}}{6}, \frac{\sqrt{6}}{6}, \frac{\sqrt{6}}{3}\right)^T$ となります。

#### 正規直交基底

以上の計算から、得られた正規直交基底 $\{\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3\}$ は以下の通りです：

$$\boldsymbol{e}_1 = \left(\frac{2}{\sqrt{5}}, 0, \frac{1}{\sqrt{5}}\right)^T$$

$$\boldsymbol{e}_2 = \left(\frac{\sqrt{30}}{30}, \frac{\sqrt{30}}{6}, -\frac{2\sqrt{30}}{30}\right)^T$$

$$\boldsymbol{e}_3 = \left(-\frac{\sqrt{6}}{6}, \frac{\sqrt{6}}{6}, \frac{\sqrt{6}}{3}\right)^T$$

#### 検証

これらのベクトルが正規直交基底であることを確認するために、各ペアの内積が0であり、各ベクトルのノルムが1であることを確認します。

$$\langle \boldsymbol{e}_1, \boldsymbol{e}_2 \rangle = \frac{2}{\sqrt{5}} \cdot \frac{\sqrt{30}}{30} + 0 \cdot \frac{\sqrt{30}}{6} + \frac{1}{\sqrt{5}} \cdot \left(-\frac{2\sqrt{30}}{30}\right) = 0$$

$$\langle \boldsymbol{e}_1, \boldsymbol{e}_3 \rangle = \frac{2}{\sqrt{5}} \cdot \left(-\frac{\sqrt{6}}{6}\right) + 0 \cdot \frac{\sqrt{6}}{6} + \frac{1}{\sqrt{5}} \cdot \frac{\sqrt{6}}{3} = 0$$

$$\langle \boldsymbol{e}_2, \boldsymbol{e}_3 \rangle = \frac{\sqrt{30}}{30} \cdot \left(-\frac{\sqrt{6}}{6}\right) + \frac{\sqrt{30}}{6} \cdot \frac{\sqrt{6}}{6} + \left(-\frac{2\sqrt{30}}{30}\right) \cdot \frac{\sqrt{6}}{3} = 0$$

$$\|\boldsymbol{e}_1\|^2 = \left(\frac{2}{\sqrt{5}}\right)^2 + 0^2 + \left(\frac{1}{\sqrt{5}}\right)^2 = \frac{4}{5} + \frac{1}{5} = 1$$

$$\|\boldsymbol{e}_2\|^2 = \left(\frac{\sqrt{30}}{30}\right)^2 + \left(\frac{\sqrt{30}}{6}\right)^2 + \left(-\frac{2\sqrt{30}}{30}\right)^2 = 1$$

$$\|\boldsymbol{e}_3\|^2 = \left(-\frac{\sqrt{6}}{6}\right)^2 + \left(\frac{\sqrt{6}}{6}\right)^2 + \left(\frac{\sqrt{6}}{3}\right)^2 = 1$$

これらの計算結果から、$\{\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3\}$ は $\mathbb{R}^3$ の正規直交基底であることが確認できました。

#### まとめ

この例では、$\mathbb{R}^3$ における線形独立なベクトル集合に対してグラムシュミット直交化法を適用し、各ステップの詳細な計算過程を示しました。このプロセスは以下の通りです：

1. 最初のベクトル$\boldsymbol{v}_1$を採用して$\boldsymbol{u}_1$とする
2. $\boldsymbol{v}_2$から$\boldsymbol{u}_1$方向への射影を除去して$\boldsymbol{u}_2$を得る
3. $\boldsymbol{v}_3$から$\boldsymbol{u}_1$と$\boldsymbol{u}_2$方向への射影を除去して$\boldsymbol{u}_3$を得る
4. 各$\boldsymbol{u}_i$をそのノルムで割って$\boldsymbol{e}_i$を得る

このようにして得られた基底$\{\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3\}$は互いに直交し、すべてのベクトルの長さが1であるため、正規直交基底となっています。

## 5. Pythonによる実装と可視化

### 5.1 内積と角度の計算

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ベクトルの定義
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# 内積の計算
dot_product = np.dot(u, v)
print(f"内積: {dot_product}")

# ノルムの計算
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)
print(f"u のノルム: {norm_u}")
print(f"v のノルム: {norm_v}")

# 角度の計算
cos_theta = dot_product / (norm_u * norm_v)
theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 数値誤差対策のためclip
theta_deg = np.degrees(theta_rad)
print(f"ベクトル間の角度: {theta_rad:.4f} ラジアン ({theta_deg:.2f} 度)")

# 3Dプロットの作成
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 原点
origin = np.zeros(3)

# ベクトルの可視化
ax.quiver(*origin, *u, color='r', label='u = [1, 2, 3]')
ax.quiver(*origin, *v, color='b', label='v = [4, 5, 6]')

# グラフの設定
ax.set_xlim([0, 6])
ax.set_ylim([0, 6])
ax.set_zlim([0, 6])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('3D ベクトルの可視化')

plt.tight_layout()
plt.show()
```

### 5.2 グラムシュミット直交化法の実装

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gram_schmidt(vectors):
    """
    グラムシュミット直交化法の実装
    
    Parameters:
    vectors -- 線形独立なベクトル集合 (各ベクトルは行ベクトルとして格納)
    
    Returns:
    orthogonal -- 直交基底
    orthonormal -- 正規直交基底
    """
    n = len(vectors)
    orthogonal = np.zeros_like(vectors, dtype=float)
    
    for i in range(n):
        orthogonal[i] = vectors[i].copy()
        for j in range(i):
            # vectors[i] から orthogonal[j] 方向への成分を引く
            projection = np.dot(vectors[i], orthogonal[j]) / np.dot(orthogonal[j], orthogonal[j])
            orthogonal[i] = orthogonal[i] - projection * orthogonal[j]
    
    # 正規化
    orthonormal = np.zeros_like(orthogonal)
    for i in range(n):
        norm = np.linalg.norm(orthogonal[i])
        orthonormal[i] = orthogonal[i] / norm
    
    return orthogonal, orthonormal

# 例題4のベクトル集合
vectors = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1]
])

# グラムシュミット直交化法の適用
orthogonal, orthonormal = gram_schmidt(vectors)

# 結果の表示
print("元のベクトル集合:")
for i, v in enumerate(vectors):
    print(f"v{i+1} = {v}")

print("\n直交基底:")
for i, u in enumerate(orthogonal):
    print(f"u{i+1} = {u}")
    
print("\n正規直交基底:")
for i, e in enumerate(orthonormal):
    print(f"e{i+1} = {e}")

# 基底ベクトルの内積を確認
print("\n正規直交基底の内積行列:")
inner_products = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        inner_products[i, j] = np.dot(orthonormal[i], orthonormal[j])
print(inner_products)

# 3Dプロットの作成
fig = plt.figure(figsize=(15, 5))

# 元のベクトル
ax1 = fig.add_subplot(131, projection='3d')
origin = np.zeros(3)
for i, v in enumerate(vectors):
    ax1.quiver(*origin, *v, label=f'v{i+1}')
ax1.set_xlim([-0.5, 1.5])
ax1.set_ylim([-0.5, 1.5])
ax1.set_zlim([-0.5, 1.5])
ax1.legend()
ax1.set_title('元のベクトル')

# 直交基底
ax2 = fig.add_subplot(132, projection='3d')
for i, u in enumerate(orthogonal):
    ax2.quiver(*origin, *u, label=f'u{i+1}')
ax2.set_xlim([-0.5, 1.5])
ax2.set_ylim([-0.5, 1.5])
ax2.set_zlim([-0.5, 1.5])
ax2.legend()
ax2.set_title('直交基底')

# 正規直交基底
ax3 = fig.add_subplot(133, projection='3d')
for i, e in enumerate(orthonormal):
    ax3.quiver(*origin, *e, label=f'e{i+1}')
ax3.set_xlim([-0.5, 1.5])
ax3.set_ylim([-0.5, 1.5])
ax3.set_zlim([-0.5, 1.5])
ax3.legend()
ax3.set_title('正規直交基底')

plt.tight_layout()
plt.show()
```

### 5.3 応用例：健康データからの主成分分析に向けた基底の準備

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 架空の健康データの生成
np.random.seed(42)
n_samples = 100

# 血圧（収縮期、拡張期）と心拍数、体重のデータを生成
# 相関を持たせるため、共通の潜在変数を使用
latent = np.random.normal(0, 1, n_samples)
weight = 70 + 10 * np.random.normal(0, 1, n_samples) + 5 * latent
systolic_bp = 120 + 15 * np.random.normal(0, 1, n_samples) + 8 * latent
diastolic_bp = 80 + 10 * np.random.normal(0, 1, n_samples) + 5 * latent
heart_rate = 75 + 12 * np.random.normal(0, 1, n_samples) - 3 * latent

# データフレームの作成
health_data = pd.DataFrame({
    'Weight': weight,
    'SystolicBP': systolic_bp,
    'DiastolicBP': diastolic_bp,
    'HeartRate': heart_rate
})

print(health_data.head())

# データの標準化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(health_data)
data_scaled_df = pd.DataFrame(data_scaled, columns=health_data.columns)

# 共分散行列を計算
cov_matrix = np.cov(data_scaled.T)

# 共分散行列から固有値と固有ベクトルを計算
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 固有値の大きさでソート
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 結果の表示
print("\n共分散行列:")
print(cov_matrix)
print("\n固有値:")
print(eigenvalues)
print("\n固有ベクトル (列):")
print(eigenvectors)

# 固有ベクトルの正規直交性を確認
print("\n固有ベクトルの内積行列:")
inner_products = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        inner_products[i, j] = np.dot(eigenvectors[:, i], eigenvectors[:, j])
print(np.round(inner_products, 10))

# 主成分への投影
principal_components = np.dot(data_scaled, eigenvectors)
principal_df = pd.DataFrame(
    principal_components, 
    columns=['PC1', 'PC2', 'PC3', 'PC4']
)

# 最初の2つの主成分によるプロット
plt.figure(figsize=(10, 8))
plt.scatter(principal_df['PC1'], principal_df['PC2'], alpha=0.7)
plt.xlabel('第1主成分')
plt.ylabel('第2主成分')
plt.title('健康データの主成分分析（PC1 vs PC2）')
plt.grid(True)

# 元の特徴量の方向を主成分空間に投影して可視化
features = health_data.columns
for i, feature in enumerate(features):
    plt.arrow(0, 0, eigenvectors[i, 0]*3, eigenvectors[i, 1]*3, 
              head_width=0.2, head_length=0.2, fc='red', ec='red')
    plt.text(eigenvectors[i, 0]*3.2, eigenvectors[i, 1]*3.2, feature, 
             color='red', fontsize=12)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.show()

# 分散説明率の計算と可視化
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, label='個別寄与率')
plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid', label='累積寄与率')
plt.ylabel('寄与率')
plt.xlabel('主成分')
plt.title('主成分の寄与率')
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

このPythonコードでは、健康データ（体重、収縮期血圧、拡張期血圧、心拍数）をシミュレーションし、そのデータに対して主成分分析（PCA）の基礎となる計算を行っています。まず共分散行列を計算し、その固有値と固有ベクトルを求めています。固有ベクトルは互いに直交する基底ベクトルであり、これがグラムシュミット直交化法の応用結果と同等の役割を果たします。

## 6. 演習問題

### 6.1 基本問題

**問題1**: ベクトル $\boldsymbol{u} = (2, -1, 3)^T$ と $\boldsymbol{v} = (1, 2, 2)^T$ について以下を求めなさい。
(a) 内積 $\langle \boldsymbol{u}, \boldsymbol{v} \rangle$
(b) ノルム $\|\boldsymbol{u}\|$ と $\|\boldsymbol{v}\|$
(c) ベクトル間の角度 $\theta$

**問題2**: $\mathbb{R}^3$ におけるベクトル集合 $\{\boldsymbol{v}_1, \boldsymbol{v}_2, \boldsymbol{v}_3\}$ が直交基底かどうか判定しなさい。ただし、
$$\boldsymbol{v}_1 = (1, 1, 1)^T, \boldsymbol{v}_2 = (1, -2, 1)^T, \boldsymbol{v}_3 = (1, 0, -1)^T$$

**問題3**: 問題2のベクトル集合が直交基底でない場合は、グラムシュミット直交化法を適用して直交基底を求めなさい。また、正規直交基底も求めなさい。

**問題4**: 次のベクトル $\boldsymbol{a}$ を、直交する二つのベクトル $\boldsymbol{v}_1$ と $\boldsymbol{v}_2$ によって張られる空間への射影ベクトル $\boldsymbol{p}$ と、それに直交するベクトル $\boldsymbol{r}$ に分解しなさい。
$$\boldsymbol{a} = (3, 1, 2)^T, \boldsymbol{v}_1 = (1, 0, 1)^T, \boldsymbol{v}_2 = (0, 1, 0)^T$$

### 6.2 応用問題

**問題5**: $n$次元ベクトル空間における正規直交基底 $\{\boldsymbol{e}_1, \boldsymbol{e}_2, \ldots, \boldsymbol{e}_n\}$ と、任意のベクトル $\boldsymbol{v}$ について、フーリエ係数 $c_i = \langle \boldsymbol{v}, \boldsymbol{e}_i \rangle$ を用いて $\boldsymbol{v} = \sum_{i=1}^{n} c_i \boldsymbol{e}_i$ と表せることを証明しなさい。

**問題6**: あるデータセットに主成分分析を適用する前に、グラムシュミット直交化法を使って新しい基底を構築することが、どのようにデータの理解に役立つか説明しなさい。具体的な例を挙げて説明すること。

**問題7**: 心拍変動（HRV）データを分析する際に、複数の特徴量（SDNN、RMSSD、pNN50など）の線形結合で新たな指標を作ることを考えます。この際、特徴量間の直交性が重要である理由を説明し、グラムシュミット直交化法をどのように応用できるか具体的に提案しなさい。

## 7. よくある質問と解答

### Q1: 内積とノルムの違いは何ですか？
**A1**: 内積は2つのベクトル間の演算で、ベクトルの方向の類似性を測る指標です。一方、ノルムはベクトル1つの「長さ」を測る指標です。数学的には、ノルムは内積の特殊なケースで、ベクトル自身との内積の平方根として定義されます（$\|\boldsymbol{v}\| = \sqrt{\langle \boldsymbol{v}, \boldsymbol{v} \rangle}$）。

### Q2: 正規直交基底を使うメリットは何ですか？
**A2**: 正規直交基底を使うメリットは多々あります。計算が簡略化される点が最大の利点です。例えば、ベクトルの座標変換、射影計算、内積計算が簡単になります。また、数値計算上の安定性が向上し、様々なアルゴリズム（特異値分解や主成分分析など）においても重要な役割を果たします。

### Q3: グラムシュミット直交化法はなぜ機能するのですか？
**A3**: グラムシュミット直交化法は、各ステップで新しいベクトルからそれまでに構築した直交ベクトルの方向成分を逐次的に取り除くことで機能します。このプロセスにより、残ったベクトルはすでに構築した直交ベクトルすべてに直交するようになります。数学的には、部分空間への射影と、その射影の残差を計算していると解釈できます。

### Q4: 内積空間の実例には、どのようなものがありますか？
**A4**: 内積空間の実例としては以下のようなものがあります：
- $\mathbb{R}^n$（標準内積を持つn次元実ベクトル空間）
- 連続関数の空間$C[a,b]$（内積は$\langle f,g \rangle = \int_a^b f(t)g(t)dt$）
- 確率変数の空間（内積は共分散や相関係数に関連）
- 量子力学におけるヒルベルト空間
- 信号処理におけるフーリエ基底を持つ関数空間

### Q5: 主成分分析と内積の関係は何ですか？
**A5**: 主成分分析（PCA）では、データの分散を最大化する方向（主成分）を見つけます。この計算過程で、データの共分散行列の固有ベクトルを求めますが、これらの固有ベクトルは互いに直交しています。つまり、PCAsは内積空間における直交基底の探索と見なすことができます。主成分は内積を用いてデータを新しい座標系に変換する際の基底ベクトルとして機能し、この基底は互いに無相関（直交）となるように選ばれています。

### Q6: データサイエンスにおける内積空間の重要性は何ですか？
**A6**: データサイエンスにおいて、内積空間の概念は以下のような側面で重要です：
1. 特徴量間の相関や類似度を測る（内積やコサイン類似度）
2. 次元削減手法（PCA、SVDなど）の理論的基盤
3. 分類アルゴリズム（SVMなど）でのカーネル関数の定義
4. クラスタリングにおける距離や類似度の定義
5. 信号処理や画像処理における基底変換（ウェーブレット変換など）
6. 推薦システムにおける項目やユーザーの類似度計算
