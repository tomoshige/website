# 線形代数学 第28回 講義ノート：正規直交基底と射影

## 1. 講義情報と予習ガイド

**講義回**: 第28回  
**テーマ**: 正規直交基底と射影  
**関連項目**: ベクトル空間、内積、直交基底、線形回帰モデル  
**予習項目**:
- 第26回のベクトル空間の基礎
- 第27回の内積と正規直交基底・グラムシュミット直交化法

## 2. 学習目標

本講義の終了時には、以下のことができるようになります：

1. ベクトル空間の直和分解と直交直和分解の概念を理解し、説明できる
2. 直交補空間の定義と性質を理解し、具体例で計算できる
3. 直交射影と正規直交基底の関係を理解し、射影を計算できる
4. 射影行列・直交射影行列の性質を理解し、構成できる
5. 線形回帰モデルと直交射影行列の関連を説明できる

## 3. 基本概念

### 3.1 正規直交基底の復習

前回学習した正規直交基底について簡単に復習します。

> **定義：正規直交基底**  
> ベクトル空間 $V$ の基底 $\{v_1, v_2, \ldots, v_n\}$ が以下の条件を満たすとき、これを正規直交基底と呼びます。
> 1. 各ベクトルの長さが1である：$\|v_i\| = 1$ （正規性）
> 2. 異なるベクトル同士が直交する：$v_i \cdot v_j = 0$ （$i \neq j$）

正規直交基底の重要な性質：
- 計算が簡単になる
- ベクトルの座標表示が直感的になる
- 射影の計算が簡単になる

**例**: $\mathbb{R}^3$ における標準基底 $\{e_1, e_2, e_3\}$ は正規直交基底です。
$$e_1 = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, e_2 = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, e_3 = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$

### 3.2 ベクトル空間の直和分解

> **定義：部分空間の直和**  
> ベクトル空間 $V$ の2つの部分空間 $U$ と $W$ について：
> 1. $V = U + W$ （$U$ と $W$ で $V$ を生成できる）
> 2. $U \cap W = \{0\}$ （共通部分は零ベクトルのみ）
> 
> このとき、$V$ は $U$ と $W$ の直和であるといい、$V = U \oplus W$ と表記します。

直和の重要な性質：
- $V$ の任意のベクトル $v$ は、$v = u + w$ （$u \in U, w \in W$）と一意に分解できる
- $\dim(V) = \dim(U) + \dim(W)$

**具体例1**: $\mathbb{R}^3$ において、xy平面を表す部分空間 $U = \{(x, y, 0) \mid x, y \in \mathbb{R}\}$ とz軸を表す部分空間 $W = \{(0, 0, z) \mid z \in \mathbb{R}\}$ を考えます。

この場合：
- $\dim(U) = 2$（基底は $\{(1,0,0), (0,1,0)\}$）
- $\dim(W) = 1$（基底は $\{(0,0,1)\}$）
- $U \cap W = \{(0,0,0)\}$（共通部分は原点のみ）

任意のベクトル $v = (a, b, c) \in \mathbb{R}^3$ は、$v = (a, b, 0) + (0, 0, c)$ と一意に分解できます。よって $\mathbb{R}^3 = U \oplus W$ です。

**具体例2**: $\mathbb{R}^2$ において、x軸を表す部分空間 $U = \{(x, 0) \mid x \in \mathbb{R}\}$ とy軸を表す部分空間 $W = \{(0, y) \mid y \in \mathbb{R}\}$ を考えます。

このとき：
- $\dim(U) = 1$（基底は $\{(1,0)\}$）
- $\dim(W) = 1$（基底は $\{(0,1)\}$）
- $U \cap W = \{(0,0)\}$（共通部分は原点のみ）

任意のベクトル $v = (a, b) \in \mathbb{R}^2$ は、$v = (a, 0) + (0, b)$ と一意に分解できます。よって $\mathbb{R}^2 = U \oplus W$ です。

**直和でない例**: $\mathbb{R}^2$ において、x軸を表す部分空間 $U = \{(x, 0) \mid x \in \mathbb{R}\}$ と直線 $W = \{(x, x) \mid x \in \mathbb{R}\}$ を考えます。

この場合：
- $U \cap W = \{(0,0)\}$（共通部分は原点のみ）
- しかし、ベクトル $(0, 1) \in \mathbb{R}^2$ は $U$ と $W$ の和として表現できません
- よって $\mathbb{R}^2 \neq U + W$ であり、$U \oplus W$ ではありません

また、直線 $U = \{(t, 0, 0) \mid t \in \mathbb{R}\}$ と平面 $W = \{(s, t, 0) \mid s, t \in \mathbb{R}\}$ を考えると、$U \subset W$ なので $U \cap W = U \neq \{0\}$ であり、$U \oplus W$ ではありません。

### 3.3 直交補空間

> **定義：直交補空間**  
> ベクトル空間 $V$ の部分空間 $U$ に対して、$U$ のすべてのベクトルと直交するベクトルの集合を $U$ の直交補空間といい、$U^{\perp}$ と表記します。
> 
> $U^{\perp} = \{v \in V \mid \langle v, u \rangle = 0 \text{ for all } u \in U\}$

直交補空間の重要な性質：
- $U^{\perp}$ も $V$ の部分空間である
- $(U^{\perp})^{\perp} = U$
- $\dim(U) + \dim(U^{\perp}) = \dim(V)$
- $V = U \oplus U^{\perp}$ （直交直和分解）

**具体例1**: $\mathbb{R}^3$ において、xy平面を表す部分空間 $U = \{(x, y, 0) \mid x, y \in \mathbb{R}\}$ の直交補空間を求めます。

$U$ の基底は $\{(1,0,0), (0,1,0)\}$ です。

$U$ の直交補空間 $U^{\perp}$ は、これらの基底ベクトルと直交するベクトル全体の集合です。つまり：
- $(1,0,0) \cdot (a,b,c) = a = 0$
- $(0,1,0) \cdot (a,b,c) = b = 0$

を満たす $(a,b,c)$ は $(0,0,c)$ の形をとります。よって $U^{\perp} = \{(0, 0, z) \mid z \in \mathbb{R}\}$ であり、これはz軸を表します。また：
- $\dim(U) = 2$
- $\dim(U^{\perp}) = 1$
- $\dim(U) + \dim(U^{\perp}) = 2 + 1 = 3 = \dim(\mathbb{R}^3)$

が成り立ちます。

**具体例2**: $\mathbb{R}^3$ において、直線 $U = \{(t, t, t) \mid t \in \mathbb{R}\}$ の直交補空間を求めます。

$U$ の基底は $\{(1,1,1)\}$ です。

$U$ の直交補空間 $U^{\perp}$ は、この基底ベクトルと直交するベクトル全体の集合です。つまり：
- $(1,1,1) \cdot (a,b,c) = a + b + c = 0$

を満たす $(a,b,c)$ です。これは平面 $a + b + c = 0$ を表し、例えば基底として $\{(1,-1,0), (1,0,-1)\}$ を取ることができます。また：
- $\dim(U) = 1$
- $\dim(U^{\perp}) = 2$
- $\dim(U) + \dim(U^{\perp}) = 1 + 2 = 3 = \dim(\mathbb{R}^3)$

が成り立ちます。

**具体例3**: $\mathbb{R}^4$ において、部分空間 $U = \{(x, y, 0, 0) \mid x, y \in \mathbb{R}\}$ の直交補空間を求めます。

$U$ の基底は $\{(1,0,0,0), (0,1,0,0)\}$ です。

$U$ の直交補空間 $U^{\perp}$ は、これらの基底ベクトルと直交するベクトル全体の集合です。つまり：
- $(1,0,0,0) \cdot (a,b,c,d) = a = 0$
- $(0,1,0,0) \cdot (a,b,c,d) = b = 0$

を満たす $(a,b,c,d)$ は $(0,0,c,d)$ の形をとります。よって $U^{\perp} = \{(0, 0, z, w) \mid z, w \in \mathbb{R}\}$ であり、また：
- $\dim(U) = 2$
- $\dim(U^{\perp}) = 2$
- $\dim(U) + \dim(U^{\perp}) = 2 + 2 = 4 = \dim(\mathbb{R}^4)$

が成り立ちます。

### 3.4 直交直和分解

> **定義：直交直和分解**  
> ベクトル空間 $V$ の2つの部分空間 $U$ と $W$ が以下を満たすとき、$V$ は $U$ と $W$ の直交直和であるといい、$V = U \oplus^{\perp} W$ と表記します。
> 1. $V = U \oplus W$ （直和）
> 2. $U \perp W$ （直交性：$\langle u, w \rangle = 0$ for all $u \in U, w \in W$）

特に重要なのは、部分空間 $U$ とその直交補空間 $U^{\perp}$ による直交直和分解 $V = U \oplus^{\perp} U^{\perp}$ です。

**具体例1**: $\mathbb{R}^3$ において、xy平面 $U = \{(x, y, 0) \mid x, y \in \mathbb{R}\}$ とz軸 $W = \{(0, 0, z) \mid z \in \mathbb{R}\}$ を考えます。

既に確認したように $W = U^{\perp}$ であり、任意のベクトル $v = (a, b, c) \in \mathbb{R}^3$ は $v = (a, b, 0) + (0, 0, c)$ と分解できます。ここで $(a, b, 0) \in U$ と $(0, 0, c) \in W$ は直交しています（内積が0）。よって $\mathbb{R}^3 = U \oplus^{\perp} W$ は直交直和です。

この直交直和分解の意味は、空間内の任意の点を、まずxy平面に垂直に射影し、そこからz軸に沿って移動することで到達できるということです。

**具体例2**: $\mathbb{R}^3$ において、直線 $U = \{(t, t, t) \mid t \in \mathbb{R}\}$ とその直交補空間 $U^{\perp} = \{(x, y, z) \mid x + y + z = 0\}$ を考えます。

任意のベクトル $v = (a, b, c) \in \mathbb{R}^3$ は以下のように分解できます：

まず、$v$ を $U$ へ射影するために、$U$ の正規化された方向ベクトル $u_0 = \frac{1}{\sqrt{3}}(1, 1, 1)$ を用いて：
$$\text{proj}_U(v) = \langle v, u_0 \rangle u_0 = \frac{a + b + c}{3}(1, 1, 1)$$

次に、残りの成分は $U^{\perp}$ に属し：
$$\text{proj}_{U^{\perp}}(v) = v - \text{proj}_U(v) = (a, b, c) - \frac{a + b + c}{3}(1, 1, 1)$$

この分解は直交しており（内積が0）、一意です。よって $\mathbb{R}^3 = U \oplus^{\perp} U^{\perp}$ は直交直和です。

この直交直和分解の幾何学的意味は、空間内の任意の点を、まず直線 $U$ （原点を通り $(1,1,1)$ 方向の直線）への垂直射影と、そこからの垂直方向の移動に分解できるということです。

**具体例3**: $\mathbb{R}^4$ の部分空間 $U = \text{span}\{(1,0,1,0), (0,1,0,1)\}$ とその直交補空間 $U^{\perp}$ を考えます。

$U^{\perp}$ を求めるには、$U$ の基底ベクトルと直交する条件を立てます：
- $(1,0,1,0) \cdot (a,b,c,d) = a + c = 0$
- $(0,1,0,1) \cdot (a,b,c,d) = b + d = 0$

これより $U^{\perp} = \{(a, b, -a, -b) \mid a, b \in \mathbb{R}\}$ となり、$U^{\perp}$ の基底は $\{(1, 0, -1, 0), (0, 1, 0, -1)\}$ です。

ここで：
- $\dim(U) = 2$
- $\dim(U^{\perp}) = 2$
- $\dim(U) + \dim(U^{\perp}) = 2 + 2 = 4 = \dim(\mathbb{R}^4)$

任意のベクトル $v = (a, b, c, d) \in \mathbb{R}^4$ は次のように直交分解できます：
$$v = \frac{1}{2}(a+c, b+d, a+c, b+d) + \frac{1}{2}(a-c, b-d, c-a, d-b)$$

ここで第1項は $U$ の要素、第2項は $U^{\perp}$ の要素であり、これらは直交します。よって $\mathbb{R}^4 = U \oplus^{\perp} U^{\perp}$ は直交直和です。

**直交直和でない例**: $\mathbb{R}^2$ において、直線 $U = \{(t, t) \mid t \in \mathbb{R}\}$ と直線 $W = \{(t, 0) \mid t \in \mathbb{R}\}$ を考えます。

$U$ と $W$ は共通部分が原点のみなので $\mathbb{R}^2 = U \oplus W$ は直和ですが、$U$ の方向ベクトル $(1,1)$ と $W$ の方向ベクトル $(1,0)$ の内積は $1 \neq 0$ なので、$U \perp W$ ではありません。よって $\mathbb{R}^2 = U \oplus W$ は直和ですが、直交直和 $U \oplus^{\perp} W$ ではありません。

## 4. 理論と手法

### 4.1 直交射影と正規直交基底の関係

ベクトル空間 $V$ の部分空間 $U$ への射影は、ベクトル $v \in V$ を $U$ の直交直和分解 $v = u + w$ ($u \in U, w \in U^{\perp}$) における $u$ に対応させる写像です。

> **定義：直交射影**  
> ベクトル空間 $V$ の部分空間 $U$ への直交射影 $P_U: V \rightarrow U$ は、各ベクトル $v \in V$ を $v$ の $U$ への成分 $u$ に写す線形写像です。
> 
> ここで、$v = u + w$ ($u \in U, w \in U^{\perp}$) は $v$ の直交直和分解です。

$U$ の正規直交基底 $\{u_1, u_2, \ldots, u_k\}$ を用いると、$v$ の $U$ への射影は以下のように計算できます：

$$P_U(v) = \sum_{i=1}^{k} \langle v, u_i \rangle u_i$$

**計算例1**: $\mathbb{R}^3$ において、xy平面 $U = \text{span}\{e_1, e_2\}$ への射影を考えます。

$U$ の正規直交基底は $\{e_1, e_2\} = \{(1,0,0), (0,1,0)\}$ です。

ベクトル $v = (3, 4, 5)$ の $U$ への射影は：
$$P_U(v) = \langle v, e_1 \rangle e_1 + \langle v, e_2 \rangle e_2 = 3 \cdot (1,0,0) + 4 \cdot (0,1,0) = (3, 4, 0)$$

残りの成分 $v - P_U(v) = (3,4,5) - (3,4,0) = (0,0,5)$ は $U^{\perp}$ に属し、$P_U(v)$ と直交していることが確認できます。

**計算例2**: $\mathbb{R}^3$ において、直線 $U = \text{span}\{(1,1,1)\}$ への射影を考えます。

$U$ の正規直交基底は $\{u_1\} = \{\frac{1}{\sqrt{3}}(1,1,1)\}$ です。

ベクトル $v = (2, 3, 4)$ の $U$ への射影は：
$$P_U(v) = \langle v, u_1 \rangle u_1 = \frac{2+3+4}{\sqrt{3}} \cdot \frac{1}{\sqrt{3}}(1,1,1) = 3 \cdot (1,1,1) = (3,3,3)$$

残りの成分 $v - P_U(v) = (2,3,4) - (3,3,3) = (-1,0,1)$ は $U^{\perp}$ に属し、$P_U(v)$ と直交していることが確認できます（内積 $\langle (3,3,3), (-1,0,1) \rangle = -3 + 0 + 3 = 0$）。

**幾何学的解釈**: 直交射影は、$v$ から $U$ への「最短距離」を与えるポイントを求めることと同等です。言い換えると、$\|v - P_U(v)\|$ が最小となるような $U$ 上の点 $P_U(v)$ を見つけることです。このことから、$v - P_U(v)$ は $U$ に対して垂直（直交）になるという性質が導かれます。

### 4.2 射影行列

直交射影は線形写像なので、行列で表現できます。

> **定義：射影行列**  
> 部分空間 $U$ への直交射影を表す行列 $P$ を射影行列といいます。$U$ の正規直交基底 $\{u_1, u_2, \ldots, u_k\}$ を用いると：
> 
> $$P = \sum_{i=1}^{k} u_i u_i^T$$

射影行列の性質：
1. **冪等性**: $P^2 = P$ （射影を2回適用しても同じ）
2. **対称性**: $P^T = P$ （直交射影の場合）
3. **固有値**: 0または1のみ
4. **トレース**: $\text{tr}(P) = \dim(U)$

**計算例1**: $\mathbb{R}^3$ において、xy平面 $U = \text{span}\{e_1, e_2\}$ への射影行列を求めます。

$U$ の正規直交基底は $\{e_1, e_2\} = \{(1,0,0), (0,1,0)\}$ です。

射影行列は：
$$P = e_1 e_1^T + e_2 e_2^T = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} \begin{pmatrix} 1 & 0 & 0 \end{pmatrix} + \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix} \begin{pmatrix} 0 & 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

この射影行列の性質を確認します：
- $P^2 = P$ （冪等性）
- $P^T = P$ （対称性）
- 固有値は $\{1, 1, 0\}$ （0と1のみ）
- $\text{tr}(P) = 2 = \dim(U)$

**計算例2**: $\mathbb{R}^3$ において、直線 $U = \text{span}\{(1,1,1)\}$ への射影行列を求めます。

$U$ の正規直交基底は $\{u_1\} = \{\frac{1}{\sqrt{3}}(1,1,1)\}$ です。

射影行列は：
$$P = u_1 u_1^T = \frac{1}{3}\begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} \begin{pmatrix} 1 & 1 & 1 \end{pmatrix} = \frac{1}{3}\begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix}$$

この射影行列の性質を確認します：
- $P^2 = P$ （冪等性）
- $P^T = P$ （対称性）
- 固有値は $\{1, 0, 0\}$ （0と1のみ）
- $\text{tr}(P) = 1 = \dim(U)$

**計算例3**: $\mathbb{R}^3$ において、平面 $U = \{(x, y, z) \mid x + y + z = 0\}$ への射影行列を求めます。

$U$ の基底は例えば $\{(1,-1,0), (1,0,-1)\}$ ですが、これは正規直交基底ではありません。グラム・シュミット法で正規直交化します：

$u_1 = \frac{(1,-1,0)}{\|(1,-1,0)\|} = \frac{1}{\sqrt{2}}(1,-1,0)$

$u_2$ は $u_1$ と $(1,0,-1)$ から求めます：
$u_2' = (1,0,-1) - \langle (1,0,-1), u_1 \rangle u_1 = (1,0,-1) - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1,-1,0) = (1,0,-1) - \frac{1}{2}(1,-1,0) = (\frac{1}{2}, \frac{1}{2}, -1)$

$u_2 = \frac{u_2'}{\|u_2'\|} = \frac{(\frac{1}{2}, \frac{1}{2}, -1)}{\sqrt{(\frac{1}{2})^2 + (\frac{1}{2})^2 + (-1)^2}} = \frac{1}{\sqrt{\frac{1}{4} + \frac{1}{4} + 1}} \cdot (\frac{1}{2}, \frac{1}{2}, -1) = \frac{1}{\sqrt{\frac{6}{4}}} \cdot (\frac{1}{2}, \frac{1}{2}, -1) = \frac{1}{\sqrt{\frac{3}{2}}} \cdot (\frac{1}{2}, \frac{1}{2}, -1)$

$u_2 = \frac{1}{\sqrt{6}}(1, 1, -2)$

よって射影行列は：
$$P = u_1 u_1^T + u_2 u_2^T = \frac{1}{2}\begin{pmatrix} 1 \\ -1 \\ 0 \end{pmatrix} \begin{pmatrix} 1 & -1 & 0 \end{pmatrix} + \frac{1}{6}\begin{pmatrix} 1 \\ 1 \\ -2 \end{pmatrix} \begin{pmatrix} 1 & 1 & -2 \end{pmatrix}$$

これを計算すると：
$$P = \frac{1}{2}\begin{pmatrix} 1 & -1 & 0 \\ -1 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix} + \frac{1}{6}\begin{pmatrix} 1 & 1 & -2 \\ 1 & 1 & -2 \\ -2 & -2 & 4 \end{pmatrix} = \frac{1}{6}\begin{pmatrix} 4 & -2 & -2 \\ -2 & 4 & -2 \\ -2 & -2 & 4 \end{pmatrix}$$

この射影行列の性質を確認します：
- $P^2 = P$ （冪等性）
- $P^T = P$ （対称性）
- 固有値は $\{1, 1, 0\}$ （0と1のみ）
- $\text{tr}(P) = 2 = \dim(U)$

### 4.3 直交射影行列

> **定義：直交射影行列**  
> 直交射影を表す行列 $P$ は以下の性質を持ちます：
> 1. $P^2 = P$ （冪等性）
> 2. $P^T = P$ （対称性）

直交射影行列を用いると、部分空間 $U$ への射影は $P_U(v) = Pv$ と表せます。また、$U^{\perp}$ への射影は $P_{U^{\perp}}(v) = (I-P)v$ となります。

**計算例1**: $\mathbb{R}^2$ において、直線 $U = \text{span}\{(1, 1)\}$ への射影行列を求めます。

正規化すると $u_1 = \frac{1}{\sqrt{2}}(1, 1)$ となり、射影行列は：
$$P = u_1 u_1^T = \frac{1}{2}\begin{pmatrix} 1 \\ 1 \end{pmatrix} \begin{pmatrix} 1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

ベクトル $v = (3, 1)$ の $U$ への射影を計算します：
$$Pv = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 3 \\ 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 4 \\ 4 \end{pmatrix} = \begin{pmatrix} 2 \\ 2 \end{pmatrix}$$

$U^{\perp}$ への射影は：
$$(I-P)v = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 3 \\ 1 \end{pmatrix} - \begin{pmatrix} 2 \\ 2 \end{pmatrix} = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

実際に $Pv$ と $(I-P)v$ が直交していることを確認します：
$$\langle Pv, (I-P)v \rangle = \langle (2,2), (1,-1) \rangle = 2 \cdot 1 + 2 \cdot (-1) = 0$$

また、$v$ が $U$ と $U^{\perp}$ の直交直和として表されることも確認できます：
$$v = Pv + (I-P)v = (2,2) + (1,-1) = (3,1)$$

**計算例2**: $\mathbb{R}^3$ において、平面 $U = \{(x, y, z) \mid x + 2y + 3z = 0\}$ への射影行列を求めます。

平面の法線ベクトルは $n = (1, 2, 3)$ です。正規化すると $n_0 = \frac{1}{\sqrt{14}}(1, 2, 3)$ となります。

$U^{\perp} = \text{span}\{n_0\}$ なので、$U^{\perp}$ への射影行列は：
$$P_{U^{\perp}} = n_0 n_0^T = \frac{1}{14}\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} \begin{pmatrix} 1 & 2 & 3 \end{pmatrix} = \frac{1}{14}\begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{pmatrix}$$

$U$ への射影行列は：
$$P_U = I - P_{U^{\perp}} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} - \frac{1}{14}\begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{pmatrix} = \frac{1}{14}\begin{pmatrix} 13 & -2 & -3 \\ -2 & 10 & -6 \\ -3 & -6 & 5 \end{pmatrix}$$

例えばベクトル $v = (1, 1, 1)$ の $U$ への射影を計算します：
$$P_U v = \frac{1}{14}\begin{pmatrix} 13 & -2 & -3 \\ -2 & 10 & -6 \\ -3 & -6 & 5 \end{pmatrix} \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} = \frac{1}{14}\begin{pmatrix} 8 \\ 2 \\ -4 \end{pmatrix} = \begin{pmatrix} \frac{4}{7} \\ \frac{1}{7} \\ -\frac{2}{7} \end{pmatrix}$$

実際にこの点が平面上にあることを確認します：
$$1 \cdot \frac{4}{7} + 2 \cdot \frac{1}{7} + 3 \cdot (-\frac{2}{7}) = \frac{4}{7} + \frac{2}{7} - \frac{6}{7} = 0$$

### 4.4 線形回帰モデルと直交射影行列

線形回帰モデルは、データを特定の部分空間に射影することで近似するモデルと解釈できます。

線形回帰モデル: $y \approx X\beta$
- $y$: 目的変数（$n$次元ベクトル）
- $X$: 説明変数の行列（$n \times p$行列）
- $\beta$: 回帰係数（$p$次元ベクトル）

最小二乗法による $\beta$ の推定は、$y$ を $X$ の列空間への射影として解釈できます：

$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

このとき、予測値 $\hat{y} = X\hat{\beta} = X(X^TX)^{-1}X^Ty$ は、射影行列 $P = X(X^TX)^{-1}X^T$ を用いて $\hat{y} = Py$ と表せます。

**具体例**: 簡単な具体例で線形回帰と射影の関係を見てみましょう。

データ点が $(x_1, y_1) = (1, 2)$, $(x_2, y_2) = (2, 3)$, $(x_3, y_3) = (3, 5)$ の3点があり、直線 $y = \beta_0 + \beta_1 x$ へのフィッティングを考えます。

このとき：
$$X = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}, \quad y = \begin{pmatrix} 2 \\ 3 \\ 5 \end{pmatrix}$$

$X^TX$ と $(X^TX)^{-1}$ を計算します：
$$X^TX = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix} = \begin{pmatrix} 3 & 6 \\ 6 & 14 \end{pmatrix}$$

$$(X^TX)^{-1} = \frac{1}{3 \cdot 14 - 6 \cdot 6} \begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix} = \frac{1}{6} \begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix}$$

これを用いて回帰係数を求めます：
$$\hat{\beta} = (X^TX)^{-1}X^Ty = \frac{1}{6} \begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix} \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{pmatrix} \begin{pmatrix} 2 \\ 3 \\ 5 \end{pmatrix}$$

$$= \frac{1}{6} \begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix} \begin{pmatrix} 10 \\ 23 \end{pmatrix} = \frac{1}{6} \begin{pmatrix} 14 \cdot 10 - 6 \cdot 23 \\ -6 \cdot 10 + 3 \cdot 23 \end{pmatrix} = \frac{1}{6} \begin{pmatrix} 140 - 138 \\ -60 + 69 \end{pmatrix} = \frac{1}{6} \begin{pmatrix} 2 \\ 9 \end{pmatrix} = \begin{pmatrix} \frac{1}{3} \\ \frac{3}{2} \end{pmatrix}$$

よって回帰直線は $y = \frac{1}{3} + \frac{3}{2}x$ となります。

次に射影行列を計算します：
$$P = X(X^TX)^{-1}X^T = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix} \frac{1}{6} \begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix} \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{pmatrix}$$

これは計算が煩雑になるため、直接 $\hat{y} = X\hat{\beta}$ を計算します：
$$\hat{y} = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix} \begin{pmatrix} \frac{1}{3} \\ \frac{3}{2} \end{pmatrix} = \begin{pmatrix} \frac{1}{3} + \frac{3}{2} \cdot 1 \\ \frac{1}{3} + \frac{3}{2} \cdot 2 \\ \frac{1}{3} + \frac{3}{2} \cdot 3 \end{pmatrix} = \begin{pmatrix} \frac{11}{6} \\ \frac{7}{2} \\ \frac{29}{6} \end{pmatrix} \approx \begin{pmatrix} 1.83 \\ 3.50 \\ 4.83 \end{pmatrix}$$

実際の $y$ と予測値 $\hat{y}$ の差（残差）は：
$$y - \hat{y} = \begin{pmatrix} 2 \\ 3 \\ 5 \end{pmatrix} - \begin{pmatrix} 1.83 \\ 3.50 \\ 4.83 \end{pmatrix} = \begin{pmatrix} 0.17 \\ -0.50 \\ 0.17 \end{pmatrix}$$

この残差ベクトルは、$X$ の列空間と直交していることが確認できます：
$$X^T(y - \hat{y}) = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{pmatrix} \begin{pmatrix} 0.17 \\ -0.50 \\ 0.17 \end{pmatrix} = \begin{pmatrix} 0.17 - 0.50 + 0.17 \\ 0.17 - 1.00 + 0.51 \end{pmatrix} \approx \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

これは、$y$ を $X$ の列空間に射影したものが $\hat{y}$ であり、残差 $y - \hat{y}$ は $X$ の列空間の直交補空間に属することを示しています。つまり線形回帰は、データ点 $y$ を説明変数によって張られる部分空間に射影するものと解釈できるのです。

**応用例**: 多変量データにおいて、目的変数 $y$ に対する複数の説明変数 $X_1, X_2, \ldots, X_p$ の影響を分析する際、各説明変数の寄与は射影の観点から解釈できます。

例えば、健康データ分析において血圧（$y$）を年齢（$X_1$）とBMI（$X_2$）で予測する場合、射影行列 $P = X(X^TX)^{-1}X^T$ は、血圧データを年齢とBMIで説明可能な部分空間へ射影します。残差分析（$y - \hat{y}$）は、年齢とBMIで説明できない変動を表し、他の要因（例：塩分摂取量、運動量など）の影響を示唆します。

## 5. Pythonによる実装と可視化

### 5.1 直交補空間の計算

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 部分空間の基底を定義
U_basis = np.array([[1, 0, 0], [0, 1, 0]]).T  # xy平面の基底

# 直交補空間を計算（ヌル空間を求める）
U_perp = np.linalg.null_space(U_basis.T)

print("U_basis =\n", U_basis)
print("U_perp =\n", U_perp)

# 3D空間で可視化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# xy平面の表示
xx, yy = np.meshgrid(range(-2, 3), range(-2, 3))
z = np.zeros_like(xx)
ax.plot_surface(xx, yy, z, alpha=0.2, color='b')

# 直交補空間のベクトルを表示
ax.quiver(0, 0, 0, 0, 0, 1, color='r', arrow_length_ratio=0.1, length=2)

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('xy平面(青)とその直交補空間(赤)')

plt.tight_layout()
plt.show()
```

### 5.2 射影の計算と可視化

```python
import numpy as np
import matplotlib.pyplot as plt

# ベクトルとその部分空間への射影を可視化
def plot_projection(v, u):
    # uを正規化
    u_norm = u / np.linalg.norm(u)
    
    # vのuへの射影
    proj = np.dot(v, u_norm) * u_norm
    
    # vのu_perpへの射影（残差）
    perp = v - proj
    
    # プロット
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 原点と各ベクトルをプロット
    ax.quiver(0, 0, u_norm[0], u_norm[1], angles='xy', scale_units='xy', scale=1, color='r', label='基底ベクトル')
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='ベクトルv')
    ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, color='g', label='射影')
    ax.quiver(proj[0], proj[1], perp[0], perp[1], angles='xy', scale_units='xy', scale=1, color='m', label='残差')
    
    # 射影の線を点線で表示
    ax.plot([v[0], proj[0]], [v[1], proj[1]], 'k--')
    
    # プロットの範囲設定と軸の調整
    max_val = max(np.max(np.abs(v)), np.max(np.abs(u_norm))) * 1.2
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # タイトルと凡例
    ax.set_title('ベクトルの射影')
    ax.legend()
    
    return proj, perp

# 例：ベクトルvと部分空間の基底u
v = np.array([3, 4])
u = np.array([1, 0])  # x軸方向

# 射影を計算し可視化
proj, perp = plot_projection(v, u)

print(f"ベクトルv = {v}")
print(f"基底ベクトルu = {u}")
print(f"射影 = {proj}")
print(f"残差 = {perp}")
plt.show()
```

### 5.3 射影行列の実装と確認

```python
import numpy as np

# 射影行列の構築と性質確認
def projection_matrix_properties(basis):
    # 基底の正規直交化
    Q, _ = np.linalg.qr(basis)
    
    # 射影行列の構築
    P = Q @ Q.T
    
    # 性質の確認
    print("射影行列P =\n", P)
    print("\n冪等性 P^2 = P?")
    print(np.allclose(P @ P, P))
    
    print("\n対称性 P^T = P?")
    print(np.allclose(P.T, P))
    
    print("\n固有値:")
    eigvals = np.linalg.eigvals(P)
    print(np.round(eigvals, 10))
    
    print("\nトレース:", np.trace(P))
    
    return P, Q

# 例：R^3の部分空間（xy平面）の基底
basis = np.array([[1, 0, 0], [0, 1, 0]]).T

P, Q = projection_matrix_properties(basis)

# 射影の例
v = np.array([1, 2, 3])
projection = P @ v
residual = v - projection

print("\nベクトルv =", v)
print("射影 =", projection)
print("残差 =", residual)
print("射影と残差は直交しているか？", np.allclose(np.dot(projection, residual), 0))
```

### 5.4 線形回帰と射影の関係

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# データの生成
np.random.seed(42)
X = np.random.rand(50, 1) * 10
y = 2 * X.squeeze() + 1 + np.random.randn(50) * 2

# 線形回帰
X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # 切片のための列を追加
beta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
y_pred = X_with_bias @ beta

# 射影行列
P = X_with_bias @ np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T
y_proj = P @ y

# 射影行列の性質確認
print("P @ P ≈ P?", np.allclose(P @ P, P))
print("P.T ≈ P?", np.allclose(P.T, P))
print("P @ y ≈ y_pred?", np.allclose(P @ y, y_pred))

# 可視化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='データ')
plt.plot(X, y_pred, color='red', label='回帰直線')

# データ点から回帰直線への射影線を描画
for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], y_pred[i]], 'k--', alpha=0.3)

plt.xlabel('X')
plt.ylabel('y')
plt.title('線形回帰と射影の関係')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 残差の可視化
plt.figure(figsize=(10, 6))
residuals = y - y_pred
plt.stem(residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('データ点')
plt.ylabel('残差')
plt.title('残差（y - P@y）')
plt.grid(alpha=0.3)
plt.show()
```

## 6. 演習問題

### 基本問題

1. $\mathbb{R}^3$ において、ベクトル $v = (2, 3, 4)$ の以下の部分空間への射影を求めなさい。
   a) $U_1 = \text{span}\{(1, 0, 0)\}$ （x軸）
   b) $U_2 = \text{span}\{(1, 1, 0), (0, 0, 1)\}$

2. $\mathbb{R}^3$ における次の部分空間 $U = \text{span}\{(1, 1, 1), (1, 2, 0)\}$ について：
   a) $U$ の正規直交基底を求めなさい。
   b) $U$ の射影行列 $P$ を求めなさい。
   c) $U^{\perp}$ の基底と射影行列 $I-P$ を求めなさい。

3. 次の射影行列 $P = \begin{pmatrix} 2/3 & 1/3 & 2/3 \\ 1/3 & 2/3 & 1/3 \\ 2/3 & 1/3 & 2/3 \end{pmatrix}$ について：
   a) $P$ が射影行列であることを確認しなさい。
   b) $P$ の階数（ランク）を求めなさい。
   c) $P$ の像空間（列空間）を基底で表しなさい。

### 応用問題

4. 健康データ分析において、3つの健康指標（血圧、コレステロール値、血糖値）が測定されたデータがあります。これらの指標は標準化されており、血圧と血糖値には強い相関があることがわかっています。
   
   a) 血圧と血糖値で張られる部分空間 $U$ を考え、この部分空間への射影行列を求めなさい。
   b) コレステロール値が $U$ と直交するように、3つの健康指標の正規直交基底を構成しなさい。
   c) あるデータポイント $(0.8, 0.5, 0.3)$ の $U$ への射影と $U^{\perp}$ への射影を求めなさい。
   d) この射影の健康データ分析における解釈を説明しなさい。

5. 線形回帰モデル $y = X\beta + \varepsilon$ において、$X$ は $n \times p$ 行列、$\beta$ は $p$ 次元パラメータベクトル、$\varepsilon$ は誤差項とします。
   
   a) 最小二乗推定量 $\hat{\beta} = (X^TX)^{-1}X^Ty$ が、$y$ の $X$ の列空間への射影を意味することを示しなさい。
   b) 残差 $e = y - X\hat{\beta}$ が $X$ の列空間と直交することを示しなさい。
   c) モデルに新しい説明変数を追加すると、$R^2$ 値（決定係数）が常に増加または維持される理由を、射影の観点から説明しなさい。

## 7. よくある質問と解答

### Q1: 直和分解と直交直和分解の違いは何ですか？

**A1**: 直和分解 $V = U \oplus W$ は、$V$ の任意のベクトルが $U$ と $W$ の成分に一意に分解できることを意味します。直交直和分解 $V = U \oplus^{\perp} W$ は、直和分解の条件に加えて、$U$ と $W$ が互いに直交している（$\langle u, w \rangle = 0$ for all $u \in U, w \in W$）という条件を満たす場合です。直交直和分解の場合、射影が計算しやすくなり、幾何学的解釈も明確になります。

### Q2: 射影行列の固有値はなぜ0と1だけなのですか？

**A2**: 射影行列 $P$ は冪等性 $P^2 = P$ を持ちます。$P$ の固有値を $\lambda$ とすると、固有ベクトル $v$ に対して $Pv = \lambda v$ です。ここで $P(Pv) = P^2v = Pv = \lambda v$ より、$\lambda^2 v = \lambda v$ となります。$v \neq 0$ なので、$\lambda^2 = \lambda$ となり、これを解くと $\lambda = 0$ または $\lambda = 1$ が得られます。つまり、射影行列の固有値は0と1だけです。

### Q3: 線形回帰における射影の意味は何ですか？

**A3**: 線形回帰モデル $y \approx X\beta$ において、最小二乗法で得られる $\hat{y} = X\hat{\beta} = X(X^TX)^{-1}X^Ty$ は、$y$ を $X$ の列空間に射影したものと解釈できます。つまり、線形回帰は目的変数 $y$ を説明変数 $X$ で張られる部分空間に最も近づける（二乗誤差を最小化する）ように射影しているのです。残差 $y - \hat{y}$ は $X$ の列空間と直交しており、これは誤差が説明変数と無相関であることを意味します。

### Q4: 正規直交基底を使うメリットは何ですか？

**A4**: 正規直交基底を使うと以下のメリットがあります：
1. 座標の計算が簡単になる（内積で直接求まる）
2. 射影の計算が簡単になる
3. 変換行列が直交行列となり、逆行列が転置行列と等しくなる
4. 数値計算上の安定性が向上する
5. 幾何学的解釈が明確になる

### Q5: ベクトルの直交分解は一意に決まりますか？

**A5**: ベクトル $v \in V$ の直交分解 $v = u + w$ （$u \in U, w \in U^{\perp}$）は一意に決まります。これは直交補空間 $U^{\perp}$ が一意に定まることと、直交直和分解 $V = U \oplus^{\perp} U^{\perp}$ における分解の一意性によります。しかし、異なる部分空間 $U_1, U_2$ に関しては、同じベクトル $v$ の直交分解は一般に異なります。