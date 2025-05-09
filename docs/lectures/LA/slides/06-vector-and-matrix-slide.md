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

# 線形代数学 I: 第6回講義
## データサイエンスに必要な行列

### 中村 知繁

---

## 1. 講義情報と予習ガイド

-   **講義回**: 第6回
-   **関連項目**: 行列の特殊形、行列の性質
-   **予習すべき内容**:
    -   行列の定義（第4回）
    -   行列の和とスカラー倍（第4回）
    -   行列の積（第5回）

---

## 2. 学習目標

1.  **単位行列**の定義と性質を理解し、応用できる
2.  **転置行列**の定義と性質を理解し、応用できる
3.  **対称行列**の定義と性質を理解し、応用できる
4.  行列の特殊形が**データサイエンス**でどのように活用されるかを理解する

---

## 3. 基本概念

今回学ぶ行列の特殊な形：

1.  **単位行列 (Identity Matrix)**
2.  **転置行列 (Transpose Matrix)**
3.  **対称行列 (Symmetric Matrix)**

---

### 3.1 単位行列（Identity Matrix） - 定義と例

> **定義**: $n$次の単位行列 $I_n$ は、主対角線上の要素がすべて1で、それ以外の要素がすべて0である正方行列である。
>
> $$
> I_n = \begin{pmatrix}
> 1 & 0 & \cdots & 0 \\
> 0 & 1 & \cdots & 0 \\
> \vdots & \vdots & \ddots & \vdots \\
> 0 & 0 & \cdots & 1
> \end{pmatrix}
> $$

**例**:
$$
I_2 = \begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix}, \quad
I_3 = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

---

### 3.1 単位行列（Identity Matrix） - 性質と数値例

**性質**:

1.  任意の行列 $A$ に対して: $AI = IA = A$
2.  対称行列である: $I^T = I$
3.  逆行列は自身である: $I^{-1} = I$
4.  ランクは $n$ である（$I$ が $n \times n$ 行列の場合）

**数値例**: $A = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix}$ のとき

$$
A \cdot I_2 = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix} = A
$$
同様に $I_2 A = A$

---

### 3.2 転置行列（Transpose Matrix） - 定義と例

> **定義**: 行列 $A$ の転置行列 $A^T$ は、$A$ の行と列を入れ替えた行列である。
> ($m \times n$ 行列 $A=(a_{ij})$ なら、$A^T=(a_{ji})$ は $n \times m$ 行列)

**例**:
$$
A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}
\quad \implies \quad
A^T = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}
$$

---

### 3.2 転置行列（Transpose Matrix） - 性質

1.  $(A^T)^T = A$
2.  $(A + B)^T = A^T + B^T$
3.  $(cA)^T = cA^T$ （$c$ はスカラー）
4.  $(AB)^T = B^T A^T$ （**積の順序が逆転！**）
5.  $\text{rank}(A) = \text{rank}(A^T)$

---

### 3.2 転置行列（Transpose Matrix） - 数値例 (性質2)

$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$ のとき、$(A + B)^T = A^T + B^T$ を確認。

-   $A + B = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix} \implies (A + B)^T = \begin{pmatrix} 6 & 10 \\ 8 & 12 \end{pmatrix}$
-   $A^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}$, $B^T = \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix}$
-   $A^T + B^T = \begin{pmatrix} 1+5 & 3+7 \\ 2+6 & 4+8 \end{pmatrix} = \begin{pmatrix} 6 & 10 \\ 8 & 12 \end{pmatrix}$

両者は一致する。

---

### 3.3 対称行列（Symmetric Matrix） - 定義と例

> **定義**: 正方行列 $A$ が対称行列であるとは、$A = A^T$ が成り立つことである。（つまり $a_{ij} = a_{ji}$）

**例**: 主対角線について要素が対称になっている。
$$
A = \begin{pmatrix}
\mathbf{1} & \color{red}{2} & \color{blue}{3} \\
\color{red}{2} & \mathbf{4} & \color{green}{5} \\
\color{blue}{3} & \color{green}{5} & \mathbf{6}
\end{pmatrix}
$$
$A^T = A$ を満たす。

---

### 3.3 対称行列（Symmetric Matrix） - 性質

1.  対角要素 $a_{ii}$ は任意（実数行列なら実数）。
2.  固有値はすべて実数（後述）。
3.  異なる固有値に対応する固有ベクトルは直交（後述）。
4.  **任意の行列 $A$ に対して、$A^T A$ と $A A^T$ は常に対称行列**。
5.  対称行列同士の和も対称行列。

---

### 3.3 対称行列（Symmetric Matrix） - 数値例 (性質4)

任意の行列 $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ から $A^T A$ を作る。

$A^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}$

$$
A^T A = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 1(1)+3(3) & 1(2)+3(4) \\ 2(1)+4(3) & 2(2)+4(4) \end{pmatrix}
$$

$$
= \begin{pmatrix} 10 & 14 \\ 14 & 20 \end{pmatrix}
$$
この結果は $a_{12}=a_{21}=14$ であり、対称行列になっている。

---

## 4. 演習問題

ここから演習問題です。

---

## 演習問題 1

$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$ とする。$AB$ を計算せよ。

---

## 演習問題 2

$A = \begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix}, B = \begin{pmatrix} 3 & 1 \\ -1 & 0 \end{pmatrix}$ とする。$BA$ を計算せよ。

---

## 演習問題 3

$A = \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}, B = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$ とする。$AB$ と $BA$ を計算せよ。

---

## 演習問題 4

$A = \begin{pmatrix} 1 & 0 & -1 \\ 2 & 1 & 0 \end{pmatrix}, B = \begin{pmatrix} 1 & 1 \\ 0 & 2 \\ 3 & 0 \end{pmatrix}$ とする。$AB$ を計算せよ。

---

## 演習問題 5

$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, C = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$ とする。$AC$ と $CA$ はそれぞれ定義されるか。定義される場合は計算し、定義されない場合はその理由を述べよ。

---

## 演習問題 6

$A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ とする。$A^2$ と $A^3$ を計算せよ。（$A^2 = AA$, $A^3 = AAA$）

---

## 演習問題 7

$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$, $E_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $E_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$ とする。$E_2 A$ と $A E_3$ を計算し、結果が $A$ と一致することを確認せよ。

---

## 演習問題 8

$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ と $2 \times 2$ の単位行列 $E$ について、$AE = EA = A$ となることを計算により確認せよ。

---

## 演習問題 9

$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$ の転置行列 $A^T$ を求めよ。

---

## 演習問題 10

$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, B = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ とする。$(A+B)^T$ と $A^T + B^T$ をそれぞれ計算し、両者が等しいことを確認せよ。

---

## 演習問題 11

問10の行列 $A, B$ について、$(AB)^T$ と $B^T A^T$ をそれぞれ計算し、両者が等しいことを確認せよ。

---

## 演習問題 12

次の行列の中から、(i) 対称行列、(ii) 反対称行列（交代行列）をすべて選べ。

(a) $$ \begin{pmatrix} 1 & 2 \\ 2 & 3 \end{pmatrix} $$
(b) $$ \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} $$
(c) $$ \begin{pmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{pmatrix} $$
(d) $$ \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix} $$
(e) $$ \begin{pmatrix} 0 & -2 & 3 \\ 2 & 0 & -1 \\ -3 & 1 & 0 \end{pmatrix} $$
(f) $$ \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} $$

---

## 演習問題 13

$A = \begin{pmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{pmatrix}$ とする。$A^T A$ を計算し、この結果が対称行列になることを確認せよ。

---

## 演習問題 14

$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ とする。$S = \frac{1}{2}(A + A^T)$ と $K = \frac{1}{2}(A - A^T)$ をそれぞれ計算せよ。そして、$S$ が対称行列、$K$ が反対称行列であり、かつ $A = S + K$ が成り立つことを確認せよ。

---

## 演習問題 15

$A, B$ を $n \times n$ 行列とする。$X = A^T B A$ とおく。

(a) $X^T$ を $A, B, A^T, B^T$ を用いて表せ。
(b) $B$ が対称行列である場合、$X = A^T B A$ も対称行列になることを示せ。

---

## 演習問題 16

**対称・反対称部分への分解と積**
任意の $n \times n$ 正方行列 $A$ に対して、$S = \frac{1}{2}(A+A^T)$（対称部分）、$K = \frac{1}{2}(A-A^T)$（反対称部分）とおく。

(a) $A = S+K$ および $A^T = S-K$ であることを確認せよ。
(b) $A A^T$ を $S$ と $K$ を用いて表せ。（ヒント: $A A^T = (S+K)(S-K)$ を展開する）

---

## 演習問題 17

**対称行列の決定と性質**
行列 $A = \begin{pmatrix} 1 & x & y \\ 2 & 3 & z \\ 0 & 1 & 4 \end{pmatrix}$ が対称行列であるように、$x, y, z$ の値を定めよ。

---

## 演習問題 18

**転置と双線形形式**
$A$ を $n \times n$ 行列、$x, y$ を $n \times 1$ の列ベクトルとする。$s = x^T A y$ はスカラー（$1 \times 1$ 行列）である。

(a) $s^T$ を $x, y, A$ の転置を用いて表せ。（スカラーの転置は元のスカラーと同じ）
(b) $A$ が対称行列のとき、$x^T A y = y^T A x$ が成り立つことを示せ。

---

## 演習問題 19

**条件を満たす対称行列**
次の3つの条件をすべて満たす $2 \times 2$ 行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ を$b$を用いて表せ。

(i) $A$ は対称行列である。($c=b$)
(ii) $A$ の対角成分の和（トレース）は 5 である ($a+d=5$)。
(iii) $A$ の行列式は 4 である ($ad-bc=4$)。