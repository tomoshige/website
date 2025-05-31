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

# 線形代数学 I: 第10回講義
## 連立1次方程式

### 中村 知繁

---

## 1. 講義情報と予習ガイド

- **講義回**: 第10回
- **関連項目**: 連立一次方程式、行列表現、解法の基礎
- **予習すべき内容**:
    - 行列の基本演算（第4-6回）
    - ベクトルと行列の関係（第7-8回）

---

## 2. 学習目標

本講義の終了時には、以下のことができるようになることを目指します：

1.  連立一次方程式の行列表現ができるようになる
2.  連立一次方程式の幾何学的意味を理解する
3.  連立一次方程式の基本的な解法を習得する
4.  連立一次方程式の解の種類とその特徴を理解する

---

## 3. 基本概念

### 3.1 連立一次方程式とは

> **定義**: 連立一次方程式とは、複数の一次方程式（変数の次数が全て1次である方程式）を同時に満たす解を求める問題です。

一般的な形式：
$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1\\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2\\
\vdots\\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
$$
ここで、$a_{ij}$は係数、$x_j$は未知変数、$b_i$は定数項です。
$m$は方程式の数、$n$は未知変数の数を表します。

---

### 3.1 連立一次方程式とは (続き)

**例**: 次の連立方程式を考えてみましょう。
$$
\begin{cases}
2x + 3y = 8\\
4x - y = 2
\end{cases}
$$
この連立方程式には2つの未知変数（$x$と$y$）と2つの方程式があります。

---

### 3.2 連立一次方程式の行列表現

連立一次方程式は行列とベクトルを用いて、より簡潔に表現することができます。

> **定義**: $m$個の方程式と$n$個の未知数からなる連立一次方程式は、次の行列方程式で表すことができます。
> $$A\mathbf{x} = \mathbf{b}$$
> ここで、$A$は$m \times n$の係数行列、$\mathbf{x}$は$n$次元の未知数ベクトル、$\mathbf{b}$は$m$次元の定数項ベクトルです。

---

### 3.2 連立一次方程式の行列表現 (続き)

具体的には：
$$
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n}\\
a_{21} & a_{22} & \cdots & a_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}
\begin{pmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{pmatrix}
=
\begin{pmatrix}
b_1\\
b_2\\
\vdots\\
b_m
\end{pmatrix}
$$

**例**: 先ほどの連立方程式を行列表現すると：
$$
\begin{pmatrix}
2 & 3\\
4 & -1
\end{pmatrix}
\begin{pmatrix}
x\\
y
\end{pmatrix}
=
\begin{pmatrix}
8\\
2
\end{pmatrix}
$$
これにより、複雑な連立方程式も簡潔に表現できます。

---

### 3.3 拡大係数行列

連立一次方程式を解く際には、「拡大係数行列」という概念が役立ちます。

> **定義**: 拡大係数行列とは、係数行列$A$に定数項ベクトル$\mathbf{b}$を右側に追加した行列です。
> $$[A|\mathbf{b}] =
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} & | & b_1\\
a_{21} & a_{22} & \cdots & a_{2n} & | & b_2\\
\vdots & \vdots & \ddots & \vdots & | & \vdots\\
a_{m1} & a_{m2} & \cdots & a_{mn} & | & b_m
\end{pmatrix}$$

**例**: 拡大係数行列の例
$$[A|\mathbf{b}] =
\begin{pmatrix}
2 & 3 & | & 8\\
4 & -1 & | & 2
\end{pmatrix}$$
拡大係数行列は、連立方程式を解く際のガウスの消去法などで特に重要になります。

---

### 4.1 連立一次方程式の解法

連立一次方程式を解くための基本的な方法には、以下のものがあります：

1.  **代入法**: 一つの方程式から変数を他の変数で表し、それを他の方程式に代入する方法
2.  **加減法**: 複数の方程式を加えたり引いたりして変数を消去する方法
3.  **行列の逆行列を用いる方法**: $A\mathbf{x} = \mathbf{b}$ から $\mathbf{x} = A^{-1}\mathbf{b}$ を求める方法
4.  **ガウスの消去法**: 行の基本変形を用いて上三角行列または行簡約形に変形する方法
    （※ガウスの消去法の詳細は次回扱います）

ここでは、1～3の方法について具体例を用いて説明します。

---

#### 4.1.1 代入法

**例**: 次の連立方程式を代入法で解きます。
$$
\begin{cases}
2x + 3y = 8\\
4x - y = 2
\end{cases}
$$
**解法**:
1. 2番目の方程式から$y$について解きます：$y = 4x - 2$
2. この$y$の式を1番目の方程式に代入します：$2x + 3(4x - 2) = 8$
3. 整理します：$2x + 12x - 6 = 8$
4. さらに整理：$14x = 14$
5. よって$x = 1$
6. $x = 1$を$y = 4x - 2$に代入：$y = 4 \cdot 1 - 2 = 2$
7. 解は$(x, y) = (1, 2)$です。

---

#### 4.1.2 加減法

**例**: 同じ連立方程式を加減法で解きます。
$$
\begin{cases}
2x + 3y = 8 \quad \ldots (1)\\
4x - y = 2 \quad \ldots (2)
\end{cases}
$$
**解法**:
1. 方程式(1)の両辺を2倍します：$4x + 6y = 16 \quad \ldots (1')$
2. 方程式(1')から方程式(2)を引きます：$(4x + 6y) - (4x - y) = 16 - 2 \Rightarrow 7y = 14$
3. よって$y = 2$
4. $y = 2$を方程式(2)に代入：$4x - 2 = 2 \Rightarrow 4x = 4$
5. よって$x = 1$
6. 解は$(x, y) = (1, 2)$です。

---

#### 4.1.3 行列の逆行列を用いる方法

**例**: 同じ連立方程式を行列の逆行列を用いて解きます。
$A\mathbf{x} = \mathbf{b}$
ここで、
$$A = \begin{pmatrix} 2 & 3\\ 4 & -1 \end{pmatrix}, \quad
\mathbf{x} = \begin{pmatrix} x\\ y \end{pmatrix}, \quad
\mathbf{b} = \begin{pmatrix} 8\\ 2 \end{pmatrix}$$
**解法**:
1. $A$の逆行列$A^{-1}$を求めます。
   $\det(A) = 2 \cdot (-1) - 3 \cdot 4 = -2 - 12 = -14$
   $$A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} -1 & -3\\ -4 & 2 \end{pmatrix} = \frac{1}{-14} \begin{pmatrix} -1 & -3\\ -4 & 2 \end{pmatrix} = \begin{pmatrix} \frac{1}{14} & \frac{3}{14}\\ \frac{4}{14} & -\frac{2}{14} \end{pmatrix}$$

2. $\mathbf{x} = A^{-1}\mathbf{b}$より：

   $$\mathbf{x} = \begin{pmatrix} \frac{1}{14} & \frac{3}{14}\\ \frac{4}{14} & -\frac{2}{14} \end{pmatrix}\begin{pmatrix} 8\\ 2 \end{pmatrix}= \begin{pmatrix} \frac{1}{14} \cdot 8 + \frac{3}{14} \cdot 2\\ \frac{4}{14} \cdot 8 + (-\frac{2}{14}) \cdot 2 \end{pmatrix}= \begin{pmatrix} \frac{8 + 6}{14}\\ \frac{32 - 4}{14} \end{pmatrix} = \begin{pmatrix} \frac{14}{14}\\ \frac{28}{14} \end{pmatrix} = \begin{pmatrix} 1\\ 2 \end{pmatrix}$$

3. 解は$(x, y) = (1, 2)$です。

---

#### 4.1.3 行列の逆行列を用いる方法 (続き)

*注意* : この方法は係数行列 $A$ が正方行列で、かつ逆行列 $A^{-1}$ が存在する場合にのみ適用可能です。*

---

### 4.2 連立一次方程式の幾何学的解釈

#### 4.2.1 2元連立1次方程式の幾何学的解釈

2元連立1次方程式は平面上の直線として解釈できます。
- 各方程式は平面上の1本の直線を表します
- 連立方程式の解は、これらの直線の交点に対応します

**例**: 先ほどの連立方程式の幾何学的解釈
$$
\begin{cases}
2x + 3y = 8 \quad \ldots \text{直線1}\\
4x - y = 2 \quad \ldots \text{直線2}
\end{cases}
$$
この2つの直線は点$(1, 2)$で交わります。 (図を挿入推奨)

---

#### 4.2.1 2元連立1次方程式の幾何学的解釈 (続き)

連立方程式の解の存在については、次の3つのケースがあります：

1.  **唯一解**: 2つの直線が1点で交わる（一般的なケース）
    ![height:150px](https://via.placeholder.com/300x150/E8E8E8/B0B0B0?text=Intersecting+Lines)
2.  **無数の解**: 2つの直線が一致する（無限個の解）
    ![height:150px](https://via.placeholder.com/300x150/E8E8E8/B0B0B0?text=Coincident+Lines)
3.  **解なし**: 2つの直線が平行で交わらない
    ![height:150px](https://via.placeholder.com/300x150/E8E8E8/B0B0B0?text=Parallel+Lines)

---

#### 4.2.2 3元連立1次方程式の幾何学的解釈

3元連立1次方程式は3次元空間内の平面として解釈できます。

- 各方程式は3次元空間内の1つの平面を表します
- 連立方程式の解は、これらの平面の共通部分（交点や交線）に対応します

連立方程式の解の存在については、次のケースなどがあります：
1.  **唯一解**: 3つの平面が1点で交わる
2.  **無数の解**: 3つの平面が一直線または平面で交わる
3.  **解なし**: 3つの平面に共通部分がない

---

## 5. 健康データにおける連立一次方程式の応用

健康データ分析において、連立一次方程式はさまざまな場面で活用されます。

### 5.1 薬物動態モデル

薬物動態学では、薬物が体内でどのように吸収、分布、代謝、排泄されるかを数学的にモデル化します。最も基本的なコンパートメントモデルでは、その定常状態解析に連立一次方程式が必要になります。

**例**: 2コンパートメントモデル
ある薬物が血液（コンパートメント1）と組織（コンパートメント2）の間で移動する定常状態モデル：
$$
\begin{cases}
k_{12}x_1 - k_{21}x_2 = 0\\
x_1 + x_2 = D
\end{cases}
$$
$k_{12}$: 血液から組織への移行速度定数
$k_{21}$: 組織から血液への移行速度定数
$D$: 投与された薬物の総量


-----

### 5.2 栄養素バランスの最適化

栄養計画では、複数の栄養素要件を満たす食事の組み合わせを求める問題があります。これは連立一次方程式（または制約条件によっては連立一次不等式）として表現できます。

**例**: 3種類の食品から2種類の栄養素要件を満たす問題

$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + a_{13}x_3 = b_1\\
a_{21}x_1 + a_{22}x_2 + a_{23}x_3 = b_2
\end{cases}
$$

$a_{ij}$: 食品$j$に含まれる栄養素$i$の量
$x_j$: 食品$j$の摂取量
$b_i$: 栄養素$i$の必要量


-----

### 基本問題

1.  次の連立方程式を解きなさい。

$$
\begin{cases}
3x + 2y = 7\\
x - 4y = 9
\end{cases}
$$

2.  次の連立方程式を行列表現しなさい。

    $$\begin{cases}
    2x - 3y + z = 7\\
    5x + y - 2z = 4\\
    -x + 4y + 3z = 10
    \end{cases}

3.  次の連立方程式を解きなさい。

    $$\begin{cases}
    x + 2y + 3z = 14\\
    2x - y + z = 4\\
    3x + y - z = 2
    \end{cases}
    $$

---

4.  次の行列方程式を解きなさい。

    $$
    \begin{pmatrix} 2 & 5\\ 1 & 3 \end{pmatrix} \begin{pmatrix} x\\ y \end{pmatrix}
    = \begin{pmatrix}
    16\\
    10
    \end{pmatrix}$$
    
-----

### 応用問題

5.  次の連立方程式の解の存在と一意性を判定し、解が存在する場合は求めなさい。

    $$\begin{cases}
    2x + 4y = 6\\
    x + 2y = 3
    \end{cases}
    $$

6.  次の連立方程式の解の存在と一意性を判定し、解が存在する場合は求めなさい。

    $$\begin{cases}
    2x + 4y = 6\\
    x + 2y = 4
    \end{cases}
    $$

---

## まとめと次回予告

**今回のまとめ**

  - 連立一次方程式の定義と行列表現
  - 拡大係数行列
  - 基本的な解法（代入法、加減法、逆行列）
  - 解の幾何学的解釈
  - 健康データへの応用例