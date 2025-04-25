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

# 線形代数学 I: 第5回講義
## ベクトル - 定義と基本操作
### 中村 知繁
---

## 1. 講義情報と予習ガイド

*   **講義回**: 第5回
*   **テーマ**: 行列の積
*   **関連項目**: 行列積の定義、計算方法、注意点、逆行列の導入
*   **予習すべき内容**: 第4回の内容（行列の定義、行列の和、行列のスカラー倍）

---

## 2. 学習目標

本講義の終了時には、以下のことができるようになることを目指します：

1.  行列の積の定義を理解し、正確に計算できる
2.  行列の積の性質（結合法則、分配法則など）を説明できる
3.  行列の積の非可換性を理解し、その意味を説明できる
4.  逆行列の概念を理解し、2次の正則行列の逆行列を計算できる
5.  データサイエンスにおける行列積の意味と応用例を説明できる

---

## 3. 基本概念

### 3.1 行列積の定義

> **定義 3.1.1（行列積）**
> $A$ を $m \times n$ 行列、$B$ を $n \times p$ 行列とする。このとき、$A$ と $B$ の積 $AB$ は $m \times p$ の行列であり、その $(i,j)$ 成分は以下のように定義される：
>
> $$ (AB)_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj} = a_{i1}b_{1j} + a_{i2}b_{2j} + \cdots + a_{in}b_{nj} $$

*   **重要**: 積 $AB$ が定義されるためには、$A$ の**列数**と $B$ の**行数**が一致する必要がある。
*   結果の行列 $AB$ のサイズは、$A$ の行数 $\times$ $B$ の列数 ($m \times p$) となる。

---

### 行列積の定義 - 例 3.1.1

$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ ($2 \times \color{red}{2}$) と $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$ ($\color{red}{2} \times 2$) の積 $AB$ を計算する。
($A$ の列数 = $B$ の行数 = 2 なので計算可能。結果は $2 \times 2$ 行列)

$(AB)_{11} = a_{11}b_{11} + a_{12}b_{21} = 1 \times 5 + 2 \times 7 = 5 + 14 = 19$
$(AB)_{12} = a_{11}b_{12} + a_{12}b_{22} = 1 \times 6 + 2 \times 8 = 6 + 16 = 22$
$(AB)_{21} = a_{21}b_{11} + a_{22}b_{21} = 3 \times 5 + 4 \times 7 = 15 + 28 = 43$
$(AB)_{22} = a_{21}b_{12} + a_{22}b_{22} = 3 \times 6 + 4 \times 8 = 18 + 32 = 50$

よって、$AB = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$

---

### 行列積の定義 - 例 3.1.2

$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$ ($3 \times \color{red}{2}$) と $B = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}$ ($\color{red}{2} \times 3$) の積 $AB$ を計算する。
($A$ の列数 = $B$ の行数 = 2 なので計算可能。結果は $3 \times 3$ 行列)

$(AB)_{11} = 1 \times 7 + 2 \times 10 = 27$
$(AB)_{12} = 1 \times 8 + 2 \times 11 = 30$
$(AB)_{13} = 1 \times 9 + 2 \times 12 = 33$
$(AB)_{21} = 3 \times 7 + 4 \times 10 = 61$
$(AB)_{22} = 3 \times 8 + 4 \times 11 = 68$
$(AB)_{23} = 3 \times 9 + 4 \times 12 = 75$
$(AB)_{31} = 5 \times 7 + 6 \times 10 = 95$
$(AB)_{32} = 5 \times 8 + 6 \times 11 = 106$
$(AB)_{33} = 5 \times 9 + 6 \times 12 = 117$

よって、$AB = \begin{pmatrix} 27 & 30 & 33 \\ 61 & 68 & 75 \\ 95 & 106 & 117 \end{pmatrix}$

---

### 行列積の定義 - 例 3.1.3 (非可換性の例)

例 3.1.2 の $A$ と $B$ を使って、積 $BA$ を計算する。
$B = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}$ ($2 \times \color{red}{3}$) と $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$ ($\color{red}{3} \times 2$)
($B$ の列数 = $A$ の行数 = 3 なので計算可能。結果は $2 \times 2$ 行列)

$(BA)_{11} = 7 \times 1 + 8 \times 3 + 9 \times 5 = 7 + 24 + 45 = 76$
$(BA)_{12} = 7 \times 2 + 8 \times 4 + 9 \times 6 = 14 + 32 + 54 = 100$
$(BA)_{21} = 10 \times 1 + 11 \times 3 + 12 \times 5 = 10 + 33 + 60 = 103$
$(BA)_{22} = 10 \times 2 + 11 \times 4 + 12 \times 6 = 20 + 44 + 72 = 136$

よって、$BA = \begin{pmatrix} 76 & 100 \\ 103 & 136 \end{pmatrix}$

**注意**: $AB = \begin{pmatrix} 27 & 30 & 33 \\ 61 & 68 & 75 \\ 95 & 106 & 117 \end{pmatrix}$ ($3 \times 3$) と $BA = \begin{pmatrix} 76 & 100 \\ 103 & 136 \end{pmatrix}$ ($2 \times 2$) は、サイズも成分も異なる。一般に $AB \neq BA$ である（非可換性）。

---

### 3.2 行列積の幾何学的解釈

*   行列積 $AB$ は、線形変換の**合成**に対応する。
*   ベクトル $\mathbf{x}$ に行列 $B$ を作用させると $B\mathbf{x}$ ( $B$ による変換)。
*   その結果にさらに行列 $A$ を作用させると $A(B\mathbf{x})$ ( $A$ による変換)。
*   これは、合成された変換を表す行列 $AB$ を $\mathbf{x}$ に作用させるのと同じ： $(AB)\mathbf{x} = A(B\mathbf{x})$。
    *   **注意**: 変換の順序は右から左へ ($B$ が先、次に $A$)。

---

### 行列積の幾何学的解釈 - 例 3.2.1 (回転)

角度 $\theta$ の回転を表す行列: $A = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$
ベクトル $\mathbf{x} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ (x軸上の点) に $A$ を作用させる：

$$ A\mathbf{x} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\begin{pmatrix} 1\\ 0 \end{pmatrix} = \begin{pmatrix} \cos\theta \\ \sin\theta \end{pmatrix} $$
これは、$(1,0)$ を原点の周りに $\theta$ 回転させた点。

さらに $A$ を作用させる ($A^2\mathbf{x}$)：
$$ A^2\mathbf{x} = A(A\mathbf{x}) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\begin{pmatrix} \cos\theta \\ \sin\theta \end{pmatrix} = \begin{pmatrix} \cos^2\theta - \sin^2\theta \\ \sin\theta\cos\theta + \cos\theta\sin\theta \end{pmatrix} $$
$$ = \begin{pmatrix} \cos(2\theta) \\ \sin(2\theta) \end{pmatrix} $$
これは $\mathbf{x}$ を $2\theta$ 回転させた点。行列 $A^2$ は $2\theta$ の回転を表す。$A^2 = \begin{pmatrix} \cos(2\theta) & -\sin(2\theta) \\ \sin(2\theta) & \cos(2\theta) \end{pmatrix}$

---

## 4. 行列積の性質

### 4.1 行列積の基本的な性質

行列 $A, B, C$ とスカラー $c$ に対して、以下の性質が成り立つ（ただし、各演算が定義されるサイズであるとする）。

> **性質 4.1.1（結合法則）**
> $(AB)C = A(BC)$
> (掛ける順番は変えられないが、どこから計算しても結果は同じ)

> **性質 4.1.2（分配法則）**
> $A(B+C) = AB + AC$
> $(A+B)C = AC + BC$

> **性質 4.1.3（スカラー倍との関係）**
> $c(AB) = (cA)B = A(cB)$

---

### 行列積の基本的な性質 - 例 4.1.1 (分配法則の確認)

$A = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}, B = \begin{pmatrix} 4 & 1 \\ 2 & 5 \end{pmatrix}, C = \begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix}$ で $A(B+C) = AB + AC$ を確認する。

*   **左辺 $A(B+C)$:**
    $B+C = \begin{pmatrix} 6 & 1 \\ 3 & 8 \end{pmatrix}$
    $A(B+C) = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix} \begin{pmatrix} 6 & 1 \\ 3 & 8 \end{pmatrix} = \begin{pmatrix} 1(6)+2(3) & 1(1)+2(8) \\ 0(6)+3(3) & 0(1)+3(8) \end{pmatrix} = \begin{pmatrix} 12 & 17 \\ 9 & 24 \end{pmatrix}$

*   **右辺 $AB + AC$:**
    $AB = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix} \begin{pmatrix} 4 & 1 \\ 2 & 5 \end{pmatrix} = \begin{pmatrix} 8 & 11 \\ 6 & 15 \end{pmatrix}$
    $AC = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix} \begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix} = \begin{pmatrix} 4 & 6 \\ 3 & 9 \end{pmatrix}$
    $AB + AC = \begin{pmatrix} 8 & 11 \\ 6 & 15 \end{pmatrix} + \begin{pmatrix} 4 & 6 \\ 3 & 9 \end{pmatrix} = \begin{pmatrix} 12 & 17 \\ 9 & 24 \end{pmatrix}$

*   **比較:** 左辺 = 右辺 となり、分配法則が成り立っている。

---

### 4.2 行列積の非可換性

*   実数の掛け算では $ab = ba$ (可換性)。
*   行列の積では、一般に $AB \neq BA$ (非可換性)。
    *   積 $AB$ が定義できても $BA$ が定義できるとは限らない。
    *   両方定義できても、サイズが異なるとは限らない（例 3.1.2, 3.1.3）。
    *   両方定義でき、サイズが同じでも、成分が異なるとは限らない。

**例:** $A = \begin{pmatrix} 1 & 2 \\ 3 & 0 \end{pmatrix}, B = \begin{pmatrix} 4 & 1 \\ 2 & 5 \end{pmatrix}$
$$AB = \begin{pmatrix} 1(4)+2(2) & 1(1)+2(5) \\ 3(4)+0(2) & 3(1)+0(5) \end{pmatrix} = \begin{pmatrix} 8 & 11 \\ 12 & 3 \end{pmatrix}$$
$$BA = \begin{pmatrix} 4(1)+1(3) & 4(2)+1(0) \\ 2(1)+5(3) & 2(2)+5(0) \end{pmatrix} = \begin{pmatrix} 7 & 8 \\ 17 & 4 \end{pmatrix}$$
明らかに $AB \neq BA$ である。

---

### 4.3 特殊な行列と行列積 - 単位行列

> **定義 4.3（単位行列）**
> $n$ 次の単位行列 $I_n$ (または単に $I$) は、主対角線上の成分がすべて $1$ で、それ以外の成分がすべて $0$ である $n \times n$ の正方行列。
> $$ I_n = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{pmatrix} $$

*   **性質**: 任意の $m \times n$ 行列 $A$ に対して、$I_m A = A$ かつ $A I_n = A$。
*   単位行列は、行列の積における「単位元」（実数での $1$ のような役割）。

**例 4.3**: $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, I_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$
$I_2 A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 1(1)+0(3) & 1(2)+0(4) \\ 0(1)+1(3) & 0(2)+1(4) \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = A$
$A I_2 = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1(1)+2(0) & 1(0)+2(1) \\ 3(1)+4(0) & 3(0)+4(1) \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = A$

---

### 4.4 零行列

*   **定義**: すべての成分が $0$ である行列を零行列と呼び、$O$ で表す。サイズは文脈による。
    例: $O_{2 \times 2} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$, $O_{2 \times 3} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$
*   **性質**:
    *   和について: $A + O = O + A = A$ (サイズが $A$ と同じ $O$)
    *   積について: $AO = O$, $OA = O$ (積が定義できるサイズの $O$)
        *   注意: 積の結果の $O$ は、元の $O$ とサイズが異なる場合がある。
        *   例: $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$, $\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$
        *   例: $\begin{pmatrix} 1 & 2 \end{pmatrix} \begin{pmatrix} 0 \\ 0 \end{pmatrix} = (0)$ ( $1 \times 1$ 零行列)
        *   例: $\begin{pmatrix} 0 \\ 0 \end{pmatrix} \begin{pmatrix} 1 & 2 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ ( $2 \times 2$ 零行列)

---

### 4.5 逆行列

> **定義 4.5（逆行列）**
> $n$ 次正方行列 $A$ に対して、$AB = BA = I_n$ を満たす $n$ 次正方行列 $B$ が存在するとき、$B$ を $A$ の**逆行列**といい、$A^{-1}$ と表す。
> $$ AA^{-1} = A^{-1}A = I_n $$

*   逆行列が存在する行列 $A$ を**正則行列** (regular) または**可逆行列** (invertible) という。
*   逆行列が存在しない正方行列は**特異行列** (singular) または**非可逆行列** (non-invertible) という。
*   逆行列は正方行列に対してのみ定義される。

---

### 4.5.1 2次の行列の逆行列の計算

> 2次の行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ に対して、
>
> *   $ad - bc \neq 0$ ならば、$A$ は正則であり、逆行列 $A^{-1}$ が存在する。
> *   逆行列は次の式で与えられる：
>     $$ A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix} $$
> *   $ad - bc = 0$ ならば、$A$ は特異行列であり、逆行列は存在しない。

---

### 2次の行列の逆行列の計算 - 例 4.5.1

$A = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix}$ の逆行列を求める。

1.  **行列式を計算**:
    $ad - bc = 3 \times 2 - 1 \times 2 = 6 - 2 = 4$
2.  **逆行列の存在確認**:
    $ad - bc = 4 \neq 0$ なので、$A$ は正則であり、逆行列が存在する。
3.  **公式を適用**:
    $$ A^{-1} = \frac{1}{4} \begin{pmatrix} 2 & -1 \\ -2 & 3 \end{pmatrix} = \begin{pmatrix} \frac{2}{4} & -\frac{1}{4} \\ -\frac{2}{4} & \frac{3}{4} \end{pmatrix} = \begin{pmatrix} \frac{1}{2} & -\frac{1}{4} \\ -\frac{1}{2} & \frac{3}{4} \end{pmatrix} $$

**検算**:
$$A A^{-1} = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix} \begin{pmatrix} \frac{1}{2} & -\frac{1}{4} \\ -\frac{1}{2} & \frac{3}{4} \end{pmatrix} = \begin{pmatrix} 3(\frac{1}{2})+1(-\frac{1}{2}) & 3(-\frac{1}{4})+1(\frac{3}{4}) \\ 2(\frac{1}{2})+2(-\frac{1}{2}) & 2(-\frac{1}{4})+2(\frac{3}{4}) \end{pmatrix} = \begin{pmatrix} \frac{3}{2}-\frac{1}{2} & -\frac{3}{4}+\frac{3}{4} \\ 1-1 & -\frac{2}{4}+\frac{6}{4} \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I_2$$
$A^{-1}A$ も同様に $I_2$ となる（各自確認）。

---

### 4.6 2次行列に対するケーリ・ハミルトンの定理

> **定理 4.5.1 （ケーリ・ハミルトンの定理, Cayley-Hamilton Theorem, CHT）**
> $2 \times 2$ 行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ について、以下の関係式が成り立つ。
> $$ A^2 - (a+d)A + (ad-bc)I = O $$
> ここで、$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ は単位行列、$O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ は零行列。


---

### ケーリ・ハミルトンの定理 - 例 4.5.1

$A = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix}$ でケーリ・ハミルトンの定理を確認する。

*   $a+d = 3+2 = 5$
*   $ad-bc = 3(2)-1(2) = 4$

計算：
$A^2 = AA = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix} = \begin{pmatrix} 3(3)+1(2) & 3(1)+1(2) \\ 2(3)+2(2) & 2(1)+2(2) \end{pmatrix} = \begin{pmatrix} 11 & 5 \\ 10 & 6 \end{pmatrix}$

$(a+d)A = 5A = 5 \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix} = \begin{pmatrix} 15 & 5 \\ 10 & 10 \end{pmatrix}$

$4I = 4 \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 4 & 0 \\ 0 & 4 \end{pmatrix}$

$$A^2 - 5A + 4I = \begin{pmatrix} 11 & 5 \\ 10 & 6 \end{pmatrix} - \begin{pmatrix} 15 & 5 \\ 10 & 10 \end{pmatrix} + \begin{pmatrix} 4 & 0 \\ 0 & 4 \end{pmatrix} = \begin{pmatrix} 11-15+4 & 5-5+0 \\ 10-10+0 & 6-10+4 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix} = O$$

確かに成り立っている。

---

### 問題 1-1

以下の行列の積を計算しなさい。

(a) $\begin{pmatrix} 2 & 1 \\ 1 & -2 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 2 & 3 \end{pmatrix}$

(b) $\begin{pmatrix} 1 & -1 \\ 0 & 2 \\ 3 & 1 \end{pmatrix} \begin{pmatrix} 4 & 1 \\ 0 & 5 \end{pmatrix}$

(c) $\begin{pmatrix} 1 & 0 & 2 \\ -1 & 1 & 0 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ -2 & 1 \\ -1 & 0 \end{pmatrix}$

---

### 問題 1-2

以下の行列の積を計算しなさい。

(d) $\begin{pmatrix} 1 & 2 \\ 0 & -1 \\ 3 & 1 \end{pmatrix}\begin{pmatrix} 1 & -1 & 2 \\ 0 & 3 & 1 \end{pmatrix}$

(e) $\begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix}\begin{pmatrix} 1 & 0 & 4 \end{pmatrix}$

(f) $\begin{pmatrix} 1 & 0 & 4 \end{pmatrix}\begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix}$


---

### 問題 2

$A = \begin{pmatrix} 1 & 0 \\ -2 & 3 \end{pmatrix}$, $B = \begin{pmatrix} 1 & 2 & 3 \\ 0 & -1 & 1 \end{pmatrix}$, $C = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$ とする。以下の行列の積のうち、計算可能なものをすべて計算せよ。計算不可能な場合はその理由を述べよ。

(a) $AB$
(b) $BA$
(c) $AC$
(d) $CA$
(e) $BC$

---

### 問題 3

$G = \begin{pmatrix} 5 & -2 \\ 1 & 3 \end{pmatrix}$ とする。単位行列 $I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ について、$GI$ および $IG$ を計算しなさい。

---

### 問題 4

$H = \begin{pmatrix} 1 & 0 \\ 6 & 7 \end{pmatrix}$ とする。零行列 $O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ について、$HO$ および $OH$ を計算しなさい。

---

### 問題 5

行列 $A = \begin{pmatrix} 1 & 3 \\ -1 & 2 \end{pmatrix}$ について、$A^2$ ($=A A$) を計算しなさい。

---

### 問題 6

行列 $P = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, Q = \begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix}$ について、積 $PQ$ と $QP$ をそれぞれ計算し、$PQ=QP$が成り立つか確かめなさい。

---

### 問題 7

行列 $X = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 2 & 0 \\ 1 & 0 & 1 \end{pmatrix}, Y = \begin{pmatrix} 0 & 1 & 0 \\ 3 & 0 & -1 \\ 0 & 1 & 0 \end{pmatrix}$ について、積 $XY$ を計算しなさい。

---

### 問題 8

行列 $A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}, B = \begin{pmatrix} 2 & -6 \\ -1 & 3 \end{pmatrix}$ について、積 $AB$ を計算しなさい。(結果は何になるか？ $A, B$ は零行列ではないことに注意)

---

### 問題 9

行列 $A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, B = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ について、$(A+B)(A-B)$ と $A^2 - B^2$ をそれぞれ計算し、結果を比較しなさい。（実数の場合 $(a+b)(a-b)=a^2-b^2$ ですが、行列ではどうなるでしょうか？）

---

### 問題 10

行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ が正則（逆行列を持つ）ための必要十分条件は $\det(A) = ad - bc \neq 0$ です。この条件を満たさない（つまり $\det(A)=0$ となる）$A \neq O$ の例を挙げ、その行列が逆行列を持たないことを、$AB=I$ となる $B$ が存在しないことを示すことで（あるいは他の方法で）確認しなさい。

---

### 問題 11

2次の正方行列 $A, B$ が
$A+B = \begin{pmatrix} 3 & 3 \\ -1 & 0 \end{pmatrix}$,
$A-B = \begin{pmatrix} -1 & -3 \\ 1 & -2 \end{pmatrix}$
を満たすとき、
$A^2 - B^2$ を求めよ。
（ヒント: 問題9の結果を考慮すること。まず $A$ と $B$ を求める必要がある。）

---

### 問題 12

2つの正方行列
$A = \begin{pmatrix} 3 & 2 \\ a & b \end{pmatrix}$,
$B = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$
について、$AB = BA$ が成り立つとき、$a$ と $b$を求めなさい。

---

### 問題 13

行列
$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$,
$B = \begin{pmatrix} x & -1 \\ y & 1 \end{pmatrix}$
に対して、$AB = BA$ が成り立つとき、$x$ と $y$ を求めよ。

---

### 問題 14

行列 $A = \begin{pmatrix} x & 5 \\ -3 & y \end{pmatrix}$ が $A = A^{-1}$ を満たすとき、$x, y$ を求めよ。
（ヒント: $A=A^{-1} \iff A^2 = I$）

---

### 問題 15

行列 $A = \begin{pmatrix} a & 3 \\ 3 & 1 \end{pmatrix}$ の逆行列が $A^{-1} = \begin{pmatrix} 1 & x \\ b & y \end{pmatrix}$ であるとする。$x, y$を求めなさい。
（ヒント: $A$ の逆行列の公式を使う）

---

### 問題 16 （ケーリ・ハミルトン）

$2 \times 2$ 行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ について考える。ここで、$a, b, c, d$ は実数とする。また、$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ を $2 \times 2$ 単位行列、$O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ を $2 \times 2$ 零行列とする。
$$ A^2 - (a+d)A + (ad-bc)I = O $$
が成り立つことを、左辺を実際に計算して示しなさい。

---

### 問題 17 （ケーリ・ハミルトン）

行列 $A = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix}$ とする。$A^3$ を $pA + qI$ (ただし $p, q$ は実数) の形で表しなさい。また、$A^3$ を求めなさい。
（ヒント: まずケーリ・ハミルトンの定理を使って $A^2$ を $A$ と $I$ で表す）

---

### 問題 18 （ケーリ・ハミルトン）

行列 $A = \begin{pmatrix} a & 1 \\ 1 & -a \end{pmatrix}$ について、$A^{10}$ の $(2,2)$成分を求めなさい。
（ヒント: ケーリ・ハミルトンの定理から $A^2$ を求める）

---

### 問題 19

$A = \begin{pmatrix} 9 & 4 & 8 \\ -8 & -3 & -8 \\ 4 & 2 & 5 \end{pmatrix}$ とし、$E$ を 3 次の単位行列とする。このとき、次の問いに答えよ。

(1) $A^2 - 10A = -9E$ であることを示せ。

(2) $AB = \begin{pmatrix} -3 & 4 & -18 \\ 5 & -1 & 18 \\ -4 & 1 & -9 \end{pmatrix}$ を満たす行列 $B$ を求めよ。
    （ヒント: (1) の式を利用して $A$ の逆行列（のようなもの）を考える）

---

### 問題 20 (ア) （逆行列の有無）

$A$ は 2 次の正方行列であり、$A^2 + A - 2E = O$ を満たす。$A$ は逆行列をもつか。理由を付して答えよ。
（ヒント: $A^2 + A - 2E = O$ を $A( ... ) = E$ または $( ... )A = E$ の形に変形できないか考える）

---

### 問題 20 (イ) （逆行列の有無）

$B$ は 2 次の正方行列であり、$B^2 = O$ を満たす。$B$ は逆行列をもつか。理由を付して答えよ。
（ヒント: もし $B$ が逆行列 $B^{-1}$ を持つと仮定するとどうなるか？）

---

### 問題 20 (ウ) （逆行列の有無）

$C = E + B$ とする。ここで $B$ は問題 20 (イ) の行列（$B^2=O$）とする。$C$ は逆行列をもつか。理由を付して答えよ。
（ヒント: $(E+B)( ... ) = E$ となる行列を探す。二項展開のような考え方が使えるか？）

---

## 演習問題（標準）

---

### 標準問題 1

**可換な行列の探求:**

行列 $A = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}$ と可換な（つまり $AB = BA$ を満たす） $2 \times 2$ 行列 $B = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ の $a, b, c, d$ が満たすべき条件を求めなさい。

---

### 標準問題 2

**冪零行列:**

$N = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$ とする。

(a) $N^2$ と $N^3$ を計算しなさい。

(b) $(I-N)(I+N+N^2)$ を計算し、結果が $I$ になることを示しなさい。

---

### 標準問題 3

**行列のトレースと積の性質:**

行列 $M = \begin{pmatrix}x & y \\ z & w \end{pmatrix}$ のトレース $\mathrm{tr}(M)$ とは、その対角成分の和 $\mathrm{tr}(M) = x + w$ のことである。

$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, B = \begin{pmatrix} p & q \\ r & s \end{pmatrix}$ を一般的な $2 \times 2$ 行列とする。

積 $AB$ と $BA$ を計算し、 $\mathrm{tr}(AB) = \mathrm{tr}(BA)$ が成り立つことを示しなさい。（この性質は一般の $n \times n$ 行列でも成り立ちます）。

---

### 標準問題 4 (a)

**行列の $n$ 乗の規則性 (回転行列):**

$A = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ （回転行列）とする。

(a) $A^2$ および $A^3$ を計算しなさい。（三角関数の加法定理を用いるとよいでしょう）

(b) $A^n$ ($n$ は正の整数) の形を推測し、数学的帰納法を用いてその推測が正しいことを証明しなさい。

---

### 標準問題 5

**射影行列（Idempotent Matrix）の性質:**

ある正方行列 $X$ が $X^2 = X$ を満たすとき、$X$ は冪等行列（idempotent matrix）または射影行列と呼ばれる。
$P$ が冪等行列であるとき、以下のことを証明しなさい。

(a) $I-P$ も冪等行列である（つまり $(I-P)^2 = I-P$ を示す）。

(b) $P(I-P) = (I-P)P = O$ （$O$ は零行列）である。


---

### 標準問題 6

**行列の$n$乗の探求 1:**

$A = \begin{pmatrix} 2 & -1 \\ 1 & 0 \end{pmatrix}$ とする。$A^2, A^3$ を計算しても $A^n$ の規則性は見つけにくい。
しかし、$A=I+N$ となる $N = \begin{pmatrix} 1 & -1 \\ 1 & -1 \end{pmatrix}$ を考えると、$N^2=O$（零行列）となることを示せ。これを用いて $(I+N)^n$ を二項展開し、$A^n$ を求めよ。

（ヒント: $(I+N)^n = \binom{n}{0}I^n N^0 + \binom{n}{1}I^{n-1} N^1 + \binom{n}{2}I^{n-2} N^2 + \dots$ だが、$N^2=O$ なので...）

---

### 標準問題 7

**行列の$n$乗の探求 2:**

$A = \begin{pmatrix} \lambda & 1 \\ 0 & \lambda \end{pmatrix}$ ($\lambda$ はスカラー) とする。$A^n$ ($n$ は自然数) を推測し、数学的帰納法などを用いて証明せよ。
（ヒント: $A = \lambda I + N$ の形に分解してみる）

---

### 標準問題 8 (a)

**行列の$n$乗の探求 3 (対角化の利用):**

$A = \begin{pmatrix} -1 & -3 \\ 4 & 6 \end{pmatrix}, U=\begin{pmatrix} 1 & 3 \\ -1 & -4 \end{pmatrix}$ に対して次の問いに答えなさい。

(a) 行列 $U$ の逆行列 $U^{-1}$ を求めなさい。

---

### 標準問題 8 (b)

**行列の$n$乗の探求 3 (対角化の利用):**

$A = \begin{pmatrix} -1 & -3 \\ 4 & 6 \end{pmatrix}, U=\begin{pmatrix} 1 & 3 \\ -1 & -4 \end{pmatrix}, U^{-1}=\begin{pmatrix} 4 & 3 \\ -1 & -1 \end{pmatrix}$

(b) $B = U^{-1}AU$ とおくとき、$B$ を計算し、自然数 $n$ に対して、行列 $B^n$ を求めなさい。

---

### 標準問題 8 (c)

**行列の$n$乗の探求 3 (対角化の利用):**

$A = \begin{pmatrix} -1 & -3 \\ 4 & 6 \end{pmatrix}, U=\begin{pmatrix} 1 & 3 \\ -1 & -4 \end{pmatrix}, U^{-1}=\begin{pmatrix} 4 & 3 \\ -1 & -1 \end{pmatrix}, B = U^{-1}AU = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$

(c) 自然数 $n$ に対して、行列 $A^n$ を求めなさい。
    （ヒント: $B = U^{-1}AU$ より $A = UBU^{-1}$。よって $A^n = (UBU^{-1})^n = U B^n U^{-1}$）

---

## 演習問題（少し難しい）

---

### 少し難しい問題 1

行列 $B = \begin{pmatrix} 4 & -1 \\ 3 & 2 \end{pmatrix}$ とする。$B$ の逆行列 $B^{-1}$ を $pB + qI$ (ただし $p, q$ は実数) の形で表しなさい。
（ヒント: ケーリ・ハミルトンの定理を利用する）

---

### 少し難しい問題 2

行列 $C = \begin{pmatrix} 1 & 3 \\ -1 & 0 \end{pmatrix}$ とする。$f(C) = C^4 - C^3 + 4C^2 - 2C + 5I$ の値を計算しなさい。
（ヒント: ケーリ・ハミルトンの定理を利用して、高次の項を次数下げする）

---

### 少し難しい問題 3

行列 $D = \begin{pmatrix} 2 & 4 \\ -1 & -1 \end{pmatrix}$ とする。$A^6 + 2 A^4 + 2 A^3 + 2 A^2 + 2 A + I$ の値を計算しなさい。
（ヒント: ケーリ・ハミルトンの定理と、多項式の割り算を利用する）

---

### 少し難しい問題 4

行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ が $A^2 - 97A + 2010I = O$ を満たすとき、$a+d$, $ad-bc$ の値の組をすべて求めよ。ただし、$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ とする。
（ヒント: ケーリ・ハミルトンの定理と比較する）

---

### 少し難しい問題 5

$a$ を実数とする。行列 $X = \begin{pmatrix} x & -y \\ y & x \end{pmatrix}$ が $X^2 - 2X + aI = O$ をみたすような実数 $x, y$ を求めよ。ただし、$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ とする。
（ヒント: $X$ についてケーリ・ハミルトンの定理を適用し、与式と比較する）

---

### 少し難しい問題 6 (a)

$N$ を $n \times n$ 行列とし、$N^k = O$ (零行列) となる正の整数 $k$ が存在するとする（このような行列 $N$ を冪零行列といいます）。

(a) $I - N$ が正則行列（逆行列を持つ行列）であることを証明しなさい。

(b) $(I - N)^{-1}$ を $I, N, N^2, \dots, N^{k-1}$ を用いて具体的に表しなさい。

（ヒント: $(I-N)(I+N+N^2+\dots+N^{k-1})$ を計算してみる）

---

### 少し難しい問題 6 (b)

$N$ を $n \times n$ 行列とし、$N^k = O$ となる正の整数 $k$ が存在する。


---

### 少し難しい問題 7

$a, b, c, d$ は実数とする。行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ に対して、等式 $A^2 + BA = 4I$, $AB + B^2 = 12I$ をみたす行列 $B$ が存在するとき、$a+d$ と $ad-bc$ の値を求めよ。ただし、$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ とする。

---

### 少し難しい問題 8

2次元ベクトル $A_n (n=1, 2, 3, \dots)$ が以下の関係式を満たすとき、$A_n$ を求めよ。

$A_1 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$, $A_2 = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$,
$$ A_{n+2} = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} A_{n+1} + \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} A_n \quad (n=1, 2, 3, \dots) $$

---

## 演習問題（データサイエンス応用 +$\beta$）

---

### DS 問題 1 (a)

**線形回帰モデルによる予測値の計算**
予測モデル: 予測価格 = $150 + 2.5 \times (\text{広さ}) - 1.5 \times (\text{築年数})$
特徴量行列 $X$ (切片項, 広さ, 築年数) と係数ベクトル $\beta$:
$$ X = \begin{pmatrix} 1 & 80 & 10 \\ 1 & 120 & 5 \\ 1 & 70 & 20 \end{pmatrix}, \quad \beta = \begin{pmatrix} 150 \\ 2.5 \\ -1.5 \end{pmatrix} $$
(a) 行列の積 $Y_{pred} = X\beta$ を計算しなさい。

---

### DS 問題 1 (b)

**線形回帰モデルによる予測値の計算**
$$ X = \begin{pmatrix} 1 & 80 & 10 \\ 1 & 120 & 5 \\ 1 & 70 & 20 \end{pmatrix}, \quad \beta = \begin{pmatrix} 150 \\ 2.5 \\ -1.5 \end{pmatrix} $$
$$ Y_{pred} = X\beta = \begin{pmatrix} 335 \\ 442.5 \\ 225 \end{pmatrix} \quad (\text{from (a)}) $$
(b) 計算結果のベクトル $Y_{pred}$ の各要素は何を表しているか説明してください。

---

### DS 問題 2 (a)

**ソーシャルネットワークにおける「友達の友達」の数**
4人のユーザー (A, B, C, D) のフォロー関係を表す隣接行列 $A$:
($i$ 行 $j$ 列は $i \rightarrow j$ のフォロー関係)
$$ A = \begin{pmatrix} 0 & 1 & 1 & 0 \\ 0 & 0 & 1 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix} \quad \begin{matrix} \leftarrow A \\ \leftarrow B \\ \leftarrow C \\ \leftarrow D \end{matrix} $$
(a) 行列の積 $A^2 = AA$ を計算しなさい。

---

### DS 問題 2 (b)

**ソーシャルネットワークにおける「友達の友達」の数**
$$ A = \begin{pmatrix} 0 & 1 & 1 & 0 \\ 0 & 0 & 1 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}, \quad A^2 = \begin{pmatrix} 1 & 0 & 1 & 1 \\ 1 & 1 & 0 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 1 & 1 \end{pmatrix} \quad (\text{from (a)}) $$
(b) $A^2$ の $(i,j)$成分 $(A^2)_{ij}$ は、ユーザー $i$ からユーザー $j$ へ、ちょうど1人の他のユーザー $k$ を経由していく経路 ($i \rightarrow k \rightarrow j$) の数を表します。
    *   $(A^2)_{14}$ の値は何を意味しますか？
    *   $(A^2)_{22}$ の値は何を意味しますか？

---

### DS 問題 3 (a)

**顧客のプラン移行予測 (マルコフ連鎖)**
プラン移行確率行列 $T$ (行 $i$ から 列 $j$ への移行確率):
$$ T = \begin{pmatrix} 0.6 & 0.3 & 0.1 \\ 0.1 & 0.7 & 0.2 \\ 0.0 & 0.1 & 0.9 \end{pmatrix} \quad \begin{matrix} \leftarrow \text{無料} \\ \leftarrow \text{基本} \\ \leftarrow \text{プレミアム} \end{matrix} $$
初期状態分布 $p_0 = \begin{pmatrix} 0.5 & 0.3 & 0.2 \end{pmatrix}$ (無料: 50%, 基本: 30%, プレミアム: 20%)

(a) 1ヶ月後の顧客の状態分布 $p_1 = p_0 T$ を計算しなさい。

---

### DS 問題 3 (b)

**顧客のプラン移行予測 (マルコフ連鎖)**
$$ T = \begin{pmatrix} 0.6 & 0.3 & 0.1 \\ 0.1 & 0.7 & 0.2 \\ 0.0 & 0.1 & 0.9 \end{pmatrix}, \quad p_0 = \begin{pmatrix} 0.5 & 0.3 & 0.2 \end{pmatrix} $$
$$ p_1 = p_0 T = \begin{pmatrix} 0.33 & 0.38 & 0.29 \end{pmatrix} \quad (\text{from (a)}) $$
(b) 2ヶ月後の顧客の状態分布 $p_2 = p_1 T = p_0 T^2$ を計算しなさい。

---

## 8. よくある質問と解答

### Q1: 行列の積が定義されるための条件は何ですか？
**A1**: 行列 $A$ と $B$ の積 $AB$ が定義されるためには、$A$ の**列数**と $B$ の**行数**が一致している必要があります。
すなわち、$A$ が $m \times n$ 行列で $B$ が $p \times q$ 行列のとき、$n = p$ であれば積 $AB$ が定義でき、結果は $m \times q$ 行列になります。

### Q2: 行列が正則であることと逆行列が存在することは同じ意味ですか？
**A2**: はい、同じ意味です。
**正方行列** $A$ が**正則**（regular）または**可逆**（invertible）であるとは、その**逆行列** $A^{-1}$ ($AA^{-1}=A^{-1}A=I$ を満たす行列) が存在することを意味します。
2次の行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ の場合、$\det(A) = ad - bc \neq 0$ が正則であるための必要十分条件です。
$\det(A) = 0$ の場合、その行列は**特異**（singular）または**非可逆**（non-invertible）と呼ばれ、逆行列は存在しません。
