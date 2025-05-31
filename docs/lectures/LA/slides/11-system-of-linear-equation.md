---
marp: true
size: 16:9
paginate: true
theme: gaia
backgroundColor: #fff
math: katex
---

<!-- header: 'T. Nakamura | Juntendo Univ.' -->
<!-- footer: '2025/02/08' -->
<style>
section { 
    font-size: 20px; 
}
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
/* ページ番号 */
section::after {
    content: attr(data-marpit-pagination) ' / ' attr(data-marpit-pagination-total);
    fonr-size: 60%;
}
/* 発表会名 */
header {
    width: 100%;
    position: absolute;
    top: unset;
    bottom: 21px;
    left: 0;
    text-align: center;
    font-size: 60%;
}
/* 日付 */
footer {
    text-align: center;
    font-size: 15px;
}
</style>


<!--
_class: lead
_paginate: false
-->

![bg left:50% 80%](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*7c3wpgkwcxbHgvZtaTL7gg.png)

# 線形代数学 I: 第11回講義
## 連立1次方程式の解の探索

### 中村 知繁

---

## 1. 講義情報と予習ガイド

**講義回**: 第11回
**テーマ**: ガウスの消去法と解の探索
**関連項目**: 連立一次方程式、行列の基本変形、ガウスの消去法

**予習すべき内容**:
- 第10回の内容（連立一次方程式の行列表現）
- 行列の基本操作の概念

---

## 2. 学習目標

本講義の終了時には以下の能力を身につけることを目標とします:

1.  連立一次方程式の行列表現と拡大係数行列の概念を理解する
2.  行列の基本変形の種類と性質を理解する
3.  ガウスの消去法のアルゴリズムを理解し実行できる
4.  ガウスの消去法と連立方程式の消去法の関係を理解する

---

## 3. 基本概念

### 3.1 連立一次方程式の行列表現と拡大係数行列（復習）

連立一次方程式:
$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
$$

行列形式:
$$\mathbf{A}\mathbf{x} = \mathbf{b}$$

---

### 3.1 連立一次方程式の行列表現と拡大係数行列（続き）

ここで、
$$\mathbf{A} =
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}, \quad
\mathbf{x} =
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{pmatrix}, \quad
\mathbf{b} =
\begin{pmatrix}
b_1 \\
b_2 \\
\vdots \\
b_m
\end{pmatrix}
$$

**拡大係数行列**（augmented matrix）:
$$\left(\mathbf{A} | \mathbf{b}\right) =
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} & | & b_1 \\
a_{21} & a_{22} & \cdots & a_{2n} & | & b_2 \\
\vdots & \vdots & \ddots & \vdots & | & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} & | & b_m
\end{pmatrix}
$$

---

### 3.2 行列の基本変形

連立方程式の解を変えずに行列の形を変える操作です。

> **行列の基本変形の定義**:
> 1.  **行の交換（Row Exchange）**: 2つの行を入れ替える
> 2.  **行のスカラー倍（Row Scaling）**: ある行の全ての要素を0でない定数倍する
> 3.  **行の加減（Row Addition）**: ある行の定数倍を別の行に加える

これらの基本変形は連立方程式の解を変えないという重要な性質があります。

---

#### 行列の基本変形の具体例 (1/3)

1.  **行の交換（Row Exchange）**:
    行列の2つの行を入れ替える操作。
    $$
    \begin{pmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
    \end{pmatrix}
    \rightarrow
    \begin{pmatrix}
    4 & 5 & 6 \\
    1 & 2 & 3 \\
    7 & 8 & 9
    \end{pmatrix}
    $$
    方程式の順序を入れ替えただけで、方程式の内容自体は変わらない。

---

#### 行列の基本変形の具体例 (2/3)

2.  **行のスカラー倍（Row Scaling）**:
    行列のある行の全ての要素を0でない定数倍する操作。
    $$
    \begin{pmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
    \end{pmatrix}
    \rightarrow
    \begin{pmatrix}
    1 & 2 & 3 \\
    8 & 10 & 12 \\
    7 & 8 & 9
    \end{pmatrix}
    $$
    （第2行を2倍）
    連立方程式では、ある方程式の両辺を定数倍することに相当。
    例: $4x + 5y + 6z = 10 \quad \rightarrow \quad 8x + 10y + 12z = 20$
    解は変わらない。

---

#### 行列の基本変形の具体例 (3/3)

3.  **行の加減（Row Addition）**:
    ある行の定数倍を別の行に加える操作。
    
    第1行の(-4)倍を第2行に加える:
    
    $$
    \begin{pmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
    \end{pmatrix}
    \rightarrow
    \begin{pmatrix}
    1 & 2 & 3 \\
    0 & -3 & -6 \\
    7 & 8 & 9
    \end{pmatrix}
    $$
    
    連立方程式では:
    
    $$
    \begin{cases}
    x + 2y + 3z = 10 \\
    4x + 5y + 6z = 20
    \end{cases}
    \quad \rightarrow \quad
    \begin{cases}
    x + 2y + 3z = 10 \\
    -3y - 6z = -20
    \end{cases}
    $$
    
    この操作により変数が消去されるが、解は同じ。

---

#### 行列の基本変形の応用例

拡大係数行列:

$$
\begin{pmatrix}
1 & 3 & | & 5 \\
2 & 7 & | & 11
\end{pmatrix}
$$

第1行の(-2)倍を第2行に加える:

$$
\begin{pmatrix}
1 & 3 & | & 5 \\
0 & 1 & | & 1
\end{pmatrix}
$$
元の連立方程式:
$$
\begin{cases}
x + 3y = 5 \\
2x + 7y = 11
\end{cases}
$$
変形後:
$$
\begin{cases}
x + 3y = 5 \\
y = 1
\end{cases}
$$
後退代入で簡単に解ける。

---

### 3.3 ガウスの消去法の概念

基本変形を用いて拡大係数行列を特定の形（上三角形または階段形）に変形することで連立方程式を解くアルゴリズム。

> **ガウスの消去法**:
> 拡大係数行列に基本変形を施し、係数行列を上三角行列または階段行列に変形する方法。

ガウスの消去法には主に二つの段階があります:
1.  **前進消去（Forward Elimination）**: 係数行列を上三角行列に変形する
2.  **後退代入（Back Substitution）**: 上三角行列の形の連立方程式を解く

---

## 4. 理論と手法

### 4.1 ガウスの消去法の手順

1.  **拡大係数行列の作成**: 連立方程式から拡大係数行列を作成
2.  **前進消去**:
    -   左上から始め、その列の対角成分（ピボット）を基準にする
    -   ピボット以下の要素をすべて0にするように基本変形を行う
    -   次の列に移動し同様の操作を繰り返す
3.  **後退代入**: 変形された方程式を最後の変数から順に解いていく

---

### 4.2 ガウスの消去法の詳細アルゴリズム (前進消去)

$i = 1$ から $n-1$ まで以下の操作を繰り返す:
-   ピボット要素 $a_{ii}$ を選ぶ
-   $j = i + 1$ から $m$ まで以下の操作を行う:
    -   係数 $\mu_{ji} = a_{ji}/a_{ii}$ を計算
    -   行 $j$ から 行 $i$ の $\mu_{ji}$ 倍を引く: $\text{row}_j \leftarrow \text{row}_j - \mu_{ji} \cdot \text{row}_i$

---

### 4.3 後退代入の手順

上三角行列に変形された連立方程式を解く:

1.  最後の変数 $x_n$ を計算: $x_n = b_n/a_{nn}$
2.  $i = n-1$ から $1$ まで逆順に以下を計算:
    $$x_i = \frac{1}{a_{ii}}\left(b_i - \sum_{j=i+1}^{n}a_{ij}x_j\right)$$

---

### 4.4 計算例: 3変数の連立方程式

以下の連立方程式を考えます:
$$
\begin{cases}
2x + y - z = 8 \\
-3x - y + 2z = -11 \\
-2x + y + 2z = -3
\end{cases}
$$

**ステップ 1**: 拡大係数行列を作成
$$
\begin{pmatrix}
2 & 1 & -1 & | & 8 \\
-3 & -1 & 2 & | & -11 \\
-2 & 1 & 2 & | & -3
\end{pmatrix}
$$

---

#### 計算例: ステップ 2: 前進消去 (1/2)

第1列の第1行以下の要素を0に:
- 第2行に第1行の $\frac{3}{2}$ 倍を加える ($R_2 \leftarrow R_2 + \frac{3}{2}R_1$):
$$
\begin{pmatrix}
2 & 1 & -1 & | & 8 \\
0 & \frac{1}{2} & \frac{1}{2} & | & 1 \\
-2 & 1 & 2 & | & -3
\end{pmatrix}
$$
- 第3行に第1行の $1$ 倍を加える ($R_3 \leftarrow R_3 + R_1$):
$$
\begin{pmatrix}
2 & 1 & -1 & | & 8 \\
0 & \frac{1}{2} & \frac{1}{2} & | & 1 \\
0 & 2 & 1 & | & 5
\end{pmatrix}
$$

---

#### 計算例: ステップ 2: 前進消去 (2/2)

第2列の第2行以下の要素を0に:
- 第3行から第2行の $4$ 倍を引く ($R_3 \leftarrow R_3 - 4R_2$):
$$
\begin{pmatrix}
2 & 1 & -1 & | & 8 \\
0 & \frac{1}{2} & \frac{1}{2} & | & 1 \\
0 & 0 & -1 & | & 1
\end{pmatrix}
$$
前進消去完了！

---

#### 計算例: ステップ 3: 後退代入

変形された連立方程式:
$$
\begin{cases}
2x + y - z = 8 \\
\frac{1}{2}y + \frac{1}{2}z = 1 \\
-z = 1
\end{cases}
$$
1.  最後の式から: $-z = 1 \implies \mathbf{z = -1}$
2.  2番目の式に $z=-1$ を代入: $\frac{1}{2}y + \frac{1}{2}(-1) = 1 \implies \frac{1}{2}y - \frac{1}{2} = 1 \implies \frac{1}{2}y = \frac{3}{2} \implies \mathbf{y = 3}$
3.  最初の式に $y=3, z=-1$ を代入: $2x + 3 - (-1) = 8 \implies 2x + 4 = 8 \implies 2x = 4 \implies \mathbf{x = 2}$

従って、解は $(x, y, z) = (2, 3, -1)$ です。

---

## 5. 演習問題

### 5.1 基本問題

**問題1**: 以下の連立方程式をガウスの消去法を用いて手計算で解きなさい:
$$
\begin{cases}
3x + 2y - z = 10 \\
-x + 3y + 2z = 5 \\
x - y + z = 0
\end{cases}
$$

---

**問題2**: 以下の拡大係数行列にガウスの消去法を適用し、上三角行列の形に変形しなさい:
$$
\begin{pmatrix}
1 & 2 & 3 & | & 4 \\
2 & 5 & 3 & | & 7 \\
1 & 0 & 8 & | & 9
\end{pmatrix}
$$

---

**問題3**: 次の連立方程式をガウスの消去法で解き、解が $(x,y,z) = (1,2,3)$ であることを確認しなさい:
$$
\begin{cases}
2x - y + z = 3 \\
x + y + z = 6 \\
x - y + 2z = 8
\end{cases}
$$

---

# 質問はありますか？