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

## 講義情報と予習ガイド

- **講義回**: 第5回  
- **テーマ**: 行列の積  
- **関連項目**: 行列積の定義、計算方法、注意点、逆行列の導入  
- **予習すべき内容**: 第4回の内容（行列の定義、行列の和、行列のスカラー倍）

---

## 学習目標

本講義の終了時には、以下のことができるようになることを目指します：

1. 行列の積の定義を理解し、正確に計算できる
2. 行列の積の性質（結合法則、分配法則など）を説明できる
3. 行列の積の非可換性を理解し、その意味を説明できる
4. 逆行列の概念を理解し、2次の正則行列の逆行列を計算できる
5. データサイエンスにおける行列積の意味と応用例を説明できる

---

## 基本概念: 行列積の定義

**定義 3.1.1（行列積）**  
$A$ を $m \times n$ 行列、$B$ を $n \times p$ 行列とする。このとき、$A$ と $B$ の積 $AB$ は $m \times p$ の行列であり、その $(i,j)$ 成分は以下のように定義される：

$(AB)_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj} = a_{i1}b_{1j} + a_{i2}b_{2j} + \cdots + a_{in}b_{nj}$

**重要**: 行列の積 $AB$ が定義されるためには、左側の行列 $A$ の列数と右側の行列 $B$ の行数が一致していなければならない。

---

## 行列積の計算例 (1)

**例 3.1.1**：$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$（$2 \times 2$ 行列）と $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$（$2 \times 2$ 行列）の積


---

## 行列積の計算例 (1)

**例 3.1.1**： $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$（$2 \times 2$ 行列）と $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$（$2 \times 2$ 行列）の積

$(AB)_{11} = a_{11}b_{11} + a_{12}b_{21} = 1 \times 5 + 2 \times 7 = 5 + 14 = 19$

$(AB)_{12} = a_{11}b_{12} + a_{12}b_{22} = 1 \times 6 + 2 \times 8 = 6 + 16 = 22$

$(AB)_{21} = a_{21}b_{11} + a_{22}b_{21} = 3 \times 5 + 4 \times 7 = 15 + 28 = 43$

$(AB)_{22} = a_{21}b_{12} + a_{22}b_{22} = 3 \times 6 + 4 \times 8 = 18 + 32 = 50$

よって、$AB = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$

---

## 行列積の計算例 (2)

**例 3.1.2**： $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$（$3 \times 2$ 行列）と $B = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}$（$2 \times 3$ 行列）の積を計算してみましょう。


---

## 行列積の計算例 (2) 続き

**例 3.1.2**： $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$（$3 \times 2$ 行列）と $B = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}$（$2 \times 3$ 行列）の積を計算してみましょう。

$(AB){11} = a_{11}b_{11} + a_{12}b_{21} = 1 \times 7 + 2 \times 10 = 7 + 20 = 27$

$(AB){12} = a_{11}b_{12} + a_{12}b_{22} = 1 \times 8 + 2 \times 11 = 8 + 22 = 30$

$(AB){13} = a_{11}b_{13} + a_{12}b_{23} = 1 \times 9 + 2 \times 12 = 9 + 24 = 33$

$(AB){21} = a_{21}b_{11} + a_{22}b_{21} = 3 \times 7 + 4 \times 10 = 21 + 40 = 61$

$(AB){22} = a_{21}b_{12} + a_{22}b_{22} = 3 \times 8 + 4 \times 11 = 24 + 44 = 68$

$(AB){23} = a_{21}b_{13} + a_{22}b_{23} = 3 \times 9 + 4 \times 12 = 27 + 48 = 75$


---

## 行列積の計算例 (2) 続き

**例 3.1.2**： $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$（$3 \times 2$ 行列）と $B = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}$（$2 \times 3$ 行列）の積を計算してみましょう。


$(AB){31} = a_{31}b_{11} + a_{32}b_{21} = 5 \times 7 + 6 \times 10 = 35 + 60 = 95$

$(AB){32} = a_{31}b_{12} + a_{32}b_{22} = 5 \times 8 + 6 \times 11 = 40 + 66 = 106$

$(AB){33} = a_{31}b_{13} + a_{32}b_{23} = 5 \times 9 + 6 \times 12 = 45 + 72 = 117$

よって、$AB = \begin{pmatrix} 27 & 30 & 33 \\ 61 & 68 & 75 \\ 95 & 106 & 117 \end{pmatrix}$ となります。


---

## 行列積の計算 (3)

**例 3.1.3（続き）**： 次に、$B$ と $A$ の積 $BA$ を計算してみましょう。

$B = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}$（$2 \times 3$ 行列）と $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$（$3 \times 2$ 行列）の積：

---

## 行列積の計算 (3)

**例 3.1.3（続き）**： 次に、$B$ と $A$ の積 $BA$ を計算してみましょう。

$B = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}$（$2 \times 3$ 行列）と $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$（$3 \times 2$ 行列）の積：

$(BA)_{11} = b_{11}a_{11} + b_{12}a_{21} + b_{13}a_{31} = 7 \times 1 + 8 \times 3 + 9 \times 5 = 7 + 24 + 45 = 76$

$(BA)_{12} = b_{11}a_{12} + b_{12}a_{22} + b_{13}a_{32} = 7 \times 2 + 8 \times 4 + 9 \times 6 = 14 + 32 + 54 = 100$

$(BA)_{21} = b_{21}a_{11} + b_{22}a_{21} + b_{23}a_{31} = 10 \times 1 + 11 \times 3 + 12 \times 5 = 10 + 33 + 60 = 103$

$(BA)_{22} = b_{21}a_{12} + b_{22}a_{22} + b_{23}a_{32} = 10 \times 2 + 11 \times 4 + 12 \times 6 = 20 + 44 + 72 = 136$

よって、$BA = \begin{pmatrix} 76 & 100 \\ 103 & 136 \end{pmatrix}$ となります。

$AB$ は $3 \times 3$ 行列、$BA$ は $2 \times 2$ 行列となり、サイズが異なります。これは行列の積の順序によって結果の次元が変わることを示しています。


---

## 行列積の幾何学的解釈

行列積は線形変換の合成として幾何学的に解釈できる。

- 行列 $A$ と行列 $B$ がそれぞれ線形変換を表すとき、$AB$ はまず $B$ による変換を行い、次に $A$ による変換を行うという合成変換を表す
- ベクトル $\mathbf{x}$ に対して行列 $A$ を作用させると、$A\mathbf{x}$ は $\mathbf{x}$ を線形変換した結果を表す

---

## 回転行列の例

**例:3.2.1**: 角度 $0 \leq \theta \leq 2\pi$ に対して、行列
$A = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$を考える。

ベクトル $\mathbf{x}=(1,0)^T$ に対して：

$$Ax = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\begin{pmatrix} 1\\ 0 \end{pmatrix} = \begin{pmatrix} \cos\theta \\ \sin\theta \end{pmatrix}$$

$$A^2 x = A(Ax) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\begin{pmatrix} \cos\theta \\ \sin\theta \end{pmatrix} = \begin{pmatrix} \cos 2\theta \\ \sin 2\theta \end{pmatrix}$$

行列 $A$ は回転変換を表す。

---

## 行列積の基本的な性質

**性質 4.1.1（結合法則）**  
行列 $A$, $B$, $C$ に対して、$(AB)C = A(BC)$ が成り立つ（ただし、それぞれの積が定義されるとする）。

**性質 4.1.2（分配法則）**  
行列 $A$, $B$, $C$ に対して、$A(B+C) = AB + AC$ および $(A+B)C = AC + BC$ が成り立つ（ただし、それぞれの和と積が定義されるとする）。

**性質 4.1.3（スカラー倍との関係）**  
スカラー $c$ と行列 $A$, $B$ に対して、$c(AB) = (cA)B = A(cB)$ が成り立つ。

---

## 分配法則の例

**例 4.1.1**：
$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$, $C = \begin{pmatrix} 9 & 10 \\ 11 & 12 \end{pmatrix}$

$A + B = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix}$

$(A+B)C = \begin{pmatrix} 142 & 156 \\ 222 & 244 \end{pmatrix}$

$AC = \begin{pmatrix} 31 & 34 \\ 71 & 78 \end{pmatrix}$

$BC = \begin{pmatrix} 111 & 122 \\ 151 & 166 \end{pmatrix}$

$AC + BC = \begin{pmatrix} 142 & 156 \\ 222 & 244 \end{pmatrix}$

$(A+B)C = AC + BC$ が確認できた。

---

## 行列積の非可換性

行列の積には、一般に交換法則が成り立たない。すなわち、$AB \neq BA$ となる場合がある。

**例 4.2.1**：
$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ と $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$ について：

$AB = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$

$BA = \begin{pmatrix} 23 & 34 \\ 31 & 46 \end{pmatrix}$

$AB \neq BA$ であるため、行列の積は一般に可換ではない。

この非可換性は、行列が表す線形変換の順序が重要であることを示している。

---

## 特殊な行列と行列積：単位行列

**定義 4.3.1（単位行列）**  
$n$ 次の単位行列 $I_n$ は、主対角線上の成分がすべて $1$ で、それ以外の成分がすべて $0$ である $n \times n$ の正方行列：

$I_n = \begin{pmatrix} 
1 & 0 & \cdots & 0 \\ 
0 & 1 & \cdots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
0 & 0 & \cdots & 1 
\end{pmatrix}$

単位行列は、任意の $n \times m$ 行列 $A$ に対して、$I_n A = A$ かつ $A I_m = A$ を満たす。
この性質から、単位行列は行列の積に関する「単位元」と呼ばれる。

---

## 単位行列の例

**例 4.3.1**：
$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ と $2$ 次の単位行列 $I_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ について：

$I_2 A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = A$

$A I_2 = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = A$

---

## 零行列

すべての成分が $0$ である行列を零行列と呼び、$O$ で表す。

任意の行列 $A$ に対して以下が成り立つ：
- $A + O = A$
- $A \times O = O \times A = O$

---

## 逆行列

**定義 4.4.1（逆行列）**  
$n$ 次正方行列 $A$ に対して、$AB = BA = I_n$ を満たす $n$ 次正方行列 $B$ が存在するとき、$B$ を $A$ の逆行列といい、$A^{-1}$ と表す。

- 逆行列が存在する行列を**正則行列**（または**可逆行列**）と呼ぶ
- 逆行列が存在しない行列は**特異行列**（または**非可逆行列**）と呼ばれる

---

## 2次の行列の逆行列の計算

2次の行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ に対して、$\det(A) = ad - bc \neq 0$ であれば、$A$ は正則であり、その逆行列は次の式で与えられる：

$A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$

---

## 逆行列の計算例

**例 4.4.1**：
$A = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix}$ の逆行列を求める。

$\det(A) = 3 \times 2 - 1 \times 2 = 6 - 2 = 4 \neq 0$ なので、$A$ は正則。

$A^{-1} = \frac{1}{4} \begin{pmatrix} 2 & -1 \\ -2 & 3 \end{pmatrix} = \begin{pmatrix} \frac{1}{2} & -\frac{1}{4} \\ -\frac{1}{2} & \frac{3}{4} \end{pmatrix}$

検算：
$A A^{-1} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I_2$

---

## 演習問題：基本問題

1. 以下の行列の積を計算しなさい。
   (a) $\begin{pmatrix} 2 & 3 \\ 4 & 5 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 2 & 3 \end{pmatrix}$
   (b) $\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}$

2. 以下の行列 $A = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$ の逆行列を求め、$A A^{-1} = I$ であることを確認しなさい。

3. 行列 $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ と $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$ に対して、$AB$ と $BA$ を計算し、$AB \neq BA$ であることを確認しなさい。

---

## 演習問題：応用問題

4. 行列 $A = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$ について、$A^2$, $A^3$, $A^4$ を計算しなさい。規則性を見つけ、$A^n$ の一般形を予想しなさい。

5. 行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ が正則であるための必要十分条件は $\det(A) = ad - bc \neq 0$ です。この条件を満たさない例を挙げ、その行列が逆行列を持たないことを確認しなさい。

---

## データサイエンス応用問題

6. 3人の患者の血圧、血糖値、コレステロール値の標準化スコアが以下の行列 $X$ で表されるとします：
   
   $X = \begin{pmatrix} 
   0.5 & 1.2 & -0.3 \\
   -0.8 & 0.4 & 0.2 \\
   1.3 & -0.1 & 0.7
   \end{pmatrix}$
   
   心疾患リスクの関連を表す重み係数が $w = \begin{pmatrix} 0.4 \\ 0.3 \\ 0.5 \end{pmatrix}$ のとき：
   
   (a) 行列の積 $Xw$ を計算し、各患者の心疾患リスクスコアを求めなさい。
   (b) どの患者が最もリスクが高いですか？

---

## よくある質問と解答

**Q1: 行列の積が定義されるための条件は何ですか？**
A1: 行列 $A$ と $B$ の積 $AB$ が定義されるためには、$A$ の列数と $B$ の行数が一致している必要があります。すなわち、$A$ が $m \times n$ 行列で $B$ が $p \times q$ 行列のとき、$n = p$ であれば積 $AB$ が定義でき、結果は $m \times q$ 行列になります。

**Q2: 行列が正則であることと逆行列が存在することは同じ意味ですか？**
A2: はい、同じ意味です。正方行列 $A$ が正則（可逆）であるとは、その逆行列 $A^{-1}$ が存在することを意味します。2次の行列の場合、$\det(A) \neq 0$ が正則であるための必要十分条件です。

