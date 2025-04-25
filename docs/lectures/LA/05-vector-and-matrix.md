# 線形代数学 I - 講義ノート 第5回

## 1. 講義情報と予習ガイド

**講義回**: 第5回  

**テーマ**: 行列の積  

**関連項目**: 行列積の定義、計算方法、注意点、逆行列の導入  

**予習すべき内容**: 第4回の内容（行列の定義、行列の和、行列のスカラー倍）

**スライド**: [スライド](./slides/05-vector-and-matrix-slide.pdf)

## 2. 学習目標

本講義の終了時には、以下のことができるようになることを目指します：

1. 行列の積の定義を理解し、正確に計算できる
2. 行列の積の性質（結合法則、分配法則など）を説明できる
3. 行列の積の非可換性を理解し、その意味を説明できる
4. 逆行列の概念を理解し、2次の正則行列の逆行列を計算できる
5. データサイエンスにおける行列積の意味と応用例を説明できる

## 3. 基本概念

### 3.1 行列積の定義

> **定義 3.1.1（行列積）**  
> $A$ を $m \times n$ 行列、$B$ を $n \times p$ 行列とする。このとき、$A$ と $B$ の積 $AB$ は $m \times p$ の行列であり、その $(i,j)$ 成分は以下のように定義される：
> 
> $(AB)_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj} = a_{i1}b_{1j} + a_{i2}b_{2j} + \cdots + a_{in}b_{nj}$

ここで重要なのは、行列の積 $AB$ が定義されるためには、左側の行列 $A$ の列数と右側の行列 $B$ の行数が一致していなければならないということです。

**例 3.1.1**：
$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$（$2 \times 2$ 行列）と $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$（$2 \times 2$ 行列）の積を計算してみましょう。

$(AB)_{11} = a_{11}b_{11} + a_{12}b_{21} = 1 \times 5 + 2 \times 7 = 5 + 14 = 19$

$(AB)_{12} = a_{11}b_{12} + a_{12}b_{22} = 1 \times 6 + 2 \times 8 = 6 + 16 = 22$

$(AB)_{21} = a_{21}b_{11} + a_{22}b_{21} = 3 \times 5 + 4 \times 7 = 15 + 28 = 43$

$(AB)_{22} = a_{21}b_{12} + a_{22}b_{22} = 3 \times 6 + 4 \times 8 = 18 + 32 = 50$

よって、$AB = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$ となります。

**例 3.1.2**： $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$（$3 \times 2$ 行列）と $B = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}$（$2 \times 3$ 行列）の積を計算してみましょう。

$(AB){11} = a_{11}b_{11} + a_{12}b_{21} = 1 \times 7 + 2 \times 10 = 7 + 20 = 27$

$(AB){12} = a_{11}b_{12} + a_{12}b_{22} = 1 \times 8 + 2 \times 11 = 8 + 22 = 30$

$(AB){13} = a_{11}b_{13} + a_{12}b_{23} = 1 \times 9 + 2 \times 12 = 9 + 24 = 33$

$(AB){21} = a_{21}b_{11} + a_{22}b_{21} = 3 \times 7 + 4 \times 10 = 21 + 40 = 61$

$(AB){22} = a_{21}b_{12} + a_{22}b_{22} = 3 \times 8 + 4 \times 11 = 24 + 44 = 68$

$(AB){23} = a_{21}b_{13} + a_{22}b_{23} = 3 \times 9 + 4 \times 12 = 27 + 48 = 75$

$(AB){31} = a_{31}b_{11} + a_{32}b_{21} = 5 \times 7 + 6 \times 10 = 35 + 60 = 95$

$(AB){32} = a_{31}b_{12} + a_{32}b_{22} = 5 \times 8 + 6 \times 11 = 40 + 66 = 106$

$(AB){33} = a_{31}b_{13} + a_{32}b_{23} = 5 \times 9 + 6 \times 12 = 45 + 72 = 117$

よって、$AB = \begin{pmatrix} 27 & 30 & 33 \\ 61 & 68 & 75 \\ 95 & 106 & 117 \end{pmatrix}$ となります。

**例 3.1.3（続き）**： 次に、$B$ と $A$ の積 $BA$ を計算してみましょう。

$B = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}$（$2 \times 3$ 行列）と $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$（$3 \times 2$ 行列）の積：

$(BA){11} = b_{11}a_{11} + b_{12}a_{21} + b_{13}a_{31} = 7 \times 1 + 8 \times 3 + 9 \times 5 = 7 + 24 + 45 = 76$

$(BA){12} = b_{11}a_{12} + b_{12}a_{22} + b_{13}a_{32} = 7 \times 2 + 8 \times 4 + 9 \times 6 = 14 + 32 + 54 = 100$

$(BA){21} = b_{21}a_{11} + b_{22}a_{21} + b_{23}a_{31} = 10 \times 1 + 11 \times 3 + 12 \times 5 = 10 + 33 + 60 = 103$

$(BA){22} = b_{21}a_{12} + b_{22}a_{22} + b_{23}a_{32} = 10 \times 2 + 11 \times 4 + 12 \times 6 = 20 + 44 + 72 = 136$

よって、$BA = \begin{pmatrix} 76 & 100 \\ 103 & 136 \end{pmatrix}$ となります。

$AB$ は $3 \times 3$ 行列、$BA$ は $2 \times 2$ 行列となり、サイズが異なります。これは行列の積の順序によって結果の次元が変わることを示しています。



### 3.2 行列積の幾何学的解釈

行列積は線形変換の合成として幾何学的に解釈できます。行列 $A$ と行列 $B$ がそれぞれ線形変換を表すとき、$AB$ はまず $B$ による変換を行い、次に $A$ による変換を行うという合成変換を表します。特に、ベクトル $\mathbf{x}$ に対して行列 $A$ を作用させると、$A\mathbf{x}$ は $\mathbf{x}$ を線形変換した結果を表します。


**例:3.2.1**: 角度 $0 \leq \theta \leq 2\pi$ に対して、行列
$A = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$を考えます。このとき、$\mathbf{x}=(1,0)^T$ に対して

$$Ax = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\begin{pmatrix} 1\\ 0 \end{pmatrix} = \begin{pmatrix} \cos\theta \\ \sin\theta \end{pmatrix}$$

$$A^2 x = A(Ax) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\begin{pmatrix} \cos\theta \\ \sin\theta \end{pmatrix} = \begin{pmatrix} \cos^2\theta - \sin^2\theta \\ 2\cos\theta\sin\theta \end{pmatrix} = \begin{pmatrix} \cos 2\theta \\ \sin 2\theta \end{pmatrix}$$

行列$A$が回転を表すことがわかる。


## 4. 行列積の性質

### 4.1 行列積の基本的な性質

行列積には以下のような重要な性質があります：

> **性質 4.1.1（結合法則）**  
> 行列 $A$, $B$, $C$ に対して、$(AB)C = A(BC)$ が成り立つ（ただし、それぞれの積が定義されるとする）。

> **性質 4.1.2（分配法則）**  
> 行列 $A$, $B$, $C$ に対して、$A(B+C) = AB + AC$ および $(A+B)C = AC + BC$ が成り立つ（ただし、それぞれの和と積が定義されるとする）。

> **性質 4.1.3（スカラー倍との関係）**  
> スカラー $c$ と行列 $A$, $B$ に対して、$c(AB) = (cA)B = A(cB)$ が成り立つ。

**例 4.1.1**：

$2 \times 2$ 行列を用いて、分配法則を確認してみましょう。

以下の行列を定義します。
$A = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}$
$B = \begin{pmatrix} 4 & 1 \\ 2 & 5 \end{pmatrix}$
$C = \begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix}$

**1. 分配法則 $A(B+C) = AB + AC$ の確認**

* **左辺 $A(B+C)$ の計算:**

    * まず $B+C$ を計算します。
        $B+C = \begin{pmatrix} 4 & 1 \\ 2 & 5 \end{pmatrix} + \begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix} = \begin{pmatrix} 4+2 & 1+0 \\ 2+1 & 5+3 \end{pmatrix} = \begin{pmatrix} 6 & 1 \\ 3 & 8 \end{pmatrix}$

    * 次に $A(B+C)$ を計算します。
        $A(B+C) = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix} \begin{pmatrix} 6 & 1 \\ 3 & 8 \end{pmatrix} = \begin{pmatrix} (1 \times 6)+(2\times 3) & (1 \times 1)+(2 \times 8) \\ (0 \times 6)+(3 \times 3) & (0 \times 1)+(3 \times 8) \end{pmatrix} = \begin{pmatrix} 6+6 & 1+16 \\ 0+9 & 0+24 \end{pmatrix} = \begin{pmatrix} 12 & 17 \\ 9 & 24 \end{pmatrix}$

* **右辺 $AB + AC$ の計算:**

    * まず $AB$ を計算します。
        $AB = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix} \begin{pmatrix} 4 & 1 \\ 2 & 5 \end{pmatrix} = \begin{pmatrix} 4+4 & 1+10 \\ 0+6 & 0+15 \end{pmatrix} = \begin{pmatrix} 8 & 11 \\ 6 & 15 \end{pmatrix}$
    * 次に $AC$ を計算します。
        $AC = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix} \begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix} = \begin{pmatrix} 2+2 & 0+6 \\ 0+3 & 0+9 \end{pmatrix} = \begin{pmatrix} 4 & 6 \\ 3 & 9 \end{pmatrix}$
    * 最後に $AB + AC$ を計算します。
        $AB + AC = \begin{pmatrix} 8 & 11 \\ 6 & 15 \end{pmatrix} + \begin{pmatrix} 4 & 6 \\ 3 & 9 \end{pmatrix} = \begin{pmatrix} 8+4 & 11+6 \\ 6+3 & 15+9 \end{pmatrix} = \begin{pmatrix} 12 & 17 \\ 9 & 24 \end{pmatrix}$

* **比較:**
    左辺 $A(B+C) = \begin{pmatrix} 12 & 17 \\ 9 & 24 \end{pmatrix}$ と 右辺 $AB + AC = \begin{pmatrix} 12 & 17 \\ 9 & 24 \end{pmatrix}$ は等しくなりました。
    これにより、分配法則がこの例で成り立つことが確認できました。

このように、具体的な行列を使って計算することで、行列の積の分配法則が成り立つことを視覚的に理解できます。


### 4.2 行列積の非可換性

行列の積には、一般に交換法則が成り立ちません。実数の掛け算では $a \times b = b \times a$ が常に成り立ちますが（可換性）、行列の積では積の順序を入れ替えると結果が変わるのが一般的です。これを**非可換性**といいます。

**例:**
以下の $2 \times 2$ 行列 $A$ と $B$ を考えます。

$A = \begin{pmatrix} 1 & 2 \\ 3 & 0 \end{pmatrix}$
$B = \begin{pmatrix} 4 & 1 \\ 2 & 5 \end{pmatrix}$

**1. 積 $AB$ の計算**

$AB = \begin{pmatrix} 1 & 2 \\ 3 & 0 \end{pmatrix} \begin{pmatrix} 4 & 1 \\ 2 & 5 \end{pmatrix} = \begin{pmatrix} 4+4 & 1+10 \\ 12+0 & 3+0 \end{pmatrix} = \begin{pmatrix} 8 & 11 \\ 12 & 3 \end{pmatrix}$

**2. 積 $BA$ の計算**

$BA = \begin{pmatrix} 4 & 1 \\ 2 & 5 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 3 & 0 \end{pmatrix} = \begin{pmatrix} 4+3 & 8+0 \\ 2+15 & 4+0 \end{pmatrix} = \begin{pmatrix} 7 & 8 \\ 17 & 4 \end{pmatrix}$

**3. 結果の比較**

計算結果を比較すると、
$AB = \begin{pmatrix} 8 & 11 \\ 12 & 3 \end{pmatrix} \neq \begin{pmatrix} 7 & 8 \\ 17 & 4 \end{pmatrix} =BA$
となるため、行列の積は順序を交換すると結果が変わります。

### 4.3 特殊な行列と行列積

> **定義 4.3（単位行列）**  
> $n$ 次の単位行列 $I_n$ は、主対角線上の成分がすべて $1$ で、それ以外の成分がすべて $0$ である $n \times n$ の正方行列です：
> 
> $I_n = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{pmatrix}$

単位行列は、任意の $n \times m$ 行列 $A$ に対して、$I_n A = A$ かつ $A I_m = A$ を満たします。この性質から、単位行列は行列の積に関する「単位元」と呼ばれます。

**例 4.3**：
$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ と $2$ 次の単位行列 $I_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ について、$I_2 A$ と $A I_2$ を計算してみましょう。

$I_2 A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 1 \times 1 + 0 \times 3 & 1 \times 2 + 0 \times 4 \\ 0 \times 1 + 1 \times 3 & 0 \times 2 + 1 \times 4 \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = A$

$A I_2 = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 \times 1 + 2 \times 0 & 1 \times 0 + 2 \times 1 \\ 3 \times 1 + 4 \times 0 & 3 \times 0 + 4 \times 1 \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = A$

よって、$I_2 A = A I_2 = A$ が確認できました。

### 4.4 零行列

すべての成分が $0$ である行列を零行列と呼び、$O$ で表します。任意の行列 $A$ に対して、$A + O = A$ および $A \times O = O \times A = O$ が成り立ちます。

### 4.5 逆行列

> **定義 4.5（逆行列）**  
> $n$ 次正方行列 $A$ に対して、$AB = BA = I_n$ を満たす $n$ 次正方行列 $B$ が存在するとき、$B$ を $A$ の逆行列といい、$A^{-1}$ と表します。

逆行列が存在する行列を**正則行列**（または**可逆行列**）と呼びます。逆行列が存在しない行列は**特異行列**（または**非可逆行列**）と呼ばれます。

#### 4.5.1 2次の行列の逆行列の計算

> 2次の行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ に対して、$\det(A) = ad - bc \neq 0$ であれば、$A$ は正則であり、その逆行列は次の式で与えられます：

> $$A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

**例 4.5.1**：
$A = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix}$ の逆行列を求めましょう。

まず、$\det(A) = 3 \times 2 - 1 \times 2 = 6 - 2 = 4 \neq 0$ なので、$A$ は正則です。

$$A^{-1} = \frac{1}{4} \begin{pmatrix} 2 & -1 \\ -2 & 3 \end{pmatrix} = \begin{pmatrix} \frac{1}{2} & -\frac{1}{4} \\ -\frac{1}{2} & \frac{3}{4} \end{pmatrix}$$

検算として、$A A^{-1}$ を計算してみましょう：

$$A A^{-1} = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix} \begin{pmatrix} \frac{1}{2} & -\frac{1}{4} \\ -\frac{1}{2} & \frac{3}{4} \end{pmatrix} = \begin{pmatrix} \frac{3}{2} - \frac{1}{2} & -\frac{3}{4} + \frac{3}{4} \\ 1 - 1 & -\frac{1}{2} + \frac{3}{2} \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I_2$$

よって、求めた逆行列が正しいことが確認できました。

### 4.6 2次行列に対するケーリハミルトンの定理

> **定理 4.5.1 （ケーリ・ハミルトンの定理, KC定理）**  
> $2 \times 2$ 行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ について考える。ここで、$a, b, c, d$ は実数とする。また、$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ を $2 \times 2$ 単位行列、$O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ を $2 \times 2$ 零行列とする。行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ について、以下の関係式が成り立つ。

> $$A^2 - (a+d)A + (ad-bc)I = O$$ 

**例 4.5.1**：
$A = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix}$ に対して、ケーリハミルトンの定理が成り立つかを確認します。

$$A^2 = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix}\begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix} = \begin{pmatrix} 11 & 5 \\ 10 & 6 \end{pmatrix}$$

$$(a+d)A = 5\begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix} = \begin{pmatrix} 15 & 5 \\ 10 & 10 \end{pmatrix}$$

ここで、$(ad-bc) = 4$ なので、

$$A^2 - (a+d)A = \begin{pmatrix} -4 & 0 \\ 0 & -4 \end{pmatrix} = -(ad-bc)I$$

であるから、

$$A^2 - (a+d)A + (ad-bc)I = O$$ 

が成り立つ。

## 5. 演習問題

### 問題（基礎）

1. 以下の行列の積を計算しなさい。
      
   (a) $\begin{pmatrix} 2 & 1 \\ 1 & -2 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 2 & 3 \end{pmatrix}$

   (b) $\begin{pmatrix} 1 & -1 \\ 0 & 2 \\ 3 & 1 \end{pmatrix} \begin{pmatrix} 4 & 1 \\ 0 & 5 \end{pmatrix}$
      
   (c) $\begin{pmatrix} 1 & 0 & 2 \\ -1 & 1 & 0 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ -2 & 1 \\ -1 & 0 \end{pmatrix}$

   (d) $\begin{pmatrix} 1 & 2 \\ 0 & -1 \\ 3 & 1 \end{pmatrix}\begin{pmatrix} 1 & -1 & 2 \\ 0 & 3 & 1 \end{pmatrix}$

   (e) $\begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix}\begin{pmatrix} 1 & 0 & 4 \end{pmatrix}$

   (f) $\begin{pmatrix} 1 & 0 & 4 \end{pmatrix}\begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix}$

2.  $A = \begin{pmatrix} 1 & 0 \\ -2 & 3 \end{pmatrix}$, $B = \begin{pmatrix} 1 & 2 & 3 \\ 0 & -1 & 1 \end{pmatrix}$, $C = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$ とする。以下の行列の積のうち、計算可能なものをすべて計算せよ。計算不可能な場合はその理由を述べよ。
    
    (a) $AB$
    
    (b) $BA$
    
    (c) $AC$
    
    (d) $CA$
    
    (e) $BC$

3.  $G = \begin{pmatrix} 5 & -2 \\ 1 & 3 \end{pmatrix}$ とする。単位行列 $I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ について、$GI$ および $IG$ を計算しなさい。

4.  $H = \begin{pmatrix} 1 & 0 \\ 6 & 7 \end{pmatrix}$ とする。零行列 $O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ について、$HO$ および $OH$ を計算しなさい。

5.  行列 $A = \begin{pmatrix} 1 & 3 \\ -1 & 2 \end{pmatrix}$ について、$A^2$ ($=A A$) を計算しなさい。

6.  行列 $P = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, Q = \begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix}$ について、積 $PQ$ と $QP$ をそれぞれ計算し、$PQ=QP$が成り立つか確かめなさい。

7.  行列 $X = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 2 & 0 \\ 1 & 0 & 1 \end{pmatrix}, Y = \begin{pmatrix} 0 & 1 & 0 \\ 3 & 0 & -1 \\ 0 & 1 & 0 \end{pmatrix}$ について、積 $XY$ を計算しなさい。

8.  行列 $A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}, B = \begin{pmatrix} 2 & -6 \\ -1 & 3 \end{pmatrix}$ について、積 $AB$ を計算しなさい。

9. 行列 $A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, B = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ について、$(A+B)(A-B)$ と $A^2 - B^2$ をそれぞれ計算し、結果を比較しなさい。（実数の場合 $(a+b)(a-b)=a^2-b^2$ ですが、行列ではどうなるでしょうか？）

10. 行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ が正則（逆行列を持つ）ための必要十分条件は $\det(A) = ad - bc \neq 0$ です。この条件を満たさない例を挙げ、その行列が逆行列を持たないことを確認しなさい。

11. 2次の正方行列 $A, B$ が
$A+B = \begin{pmatrix} 3 & 3 \\ -1 & 0 \end{pmatrix}$,
$A-B = \begin{pmatrix} -1 & -3 \\ 1 & -2 \end{pmatrix}$
を満たすとき、
$A^2 - B^2$ を求めよ。 

12. 2つの正方行列
$A = \begin{pmatrix} 3 & 2 \\ a & b \end{pmatrix}$,
$B = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$
について、$AB = BA$ が成り立つとき、$a$ と $b$を求めなさい。

13. 行列
$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$,
$B = \begin{pmatrix} x & -1 \\ y & 1 \end{pmatrix}$
に対して、$AB = BA$ が成り立つとき、$x$ と $y$ を求めよ。

14. 行列 $A = \begin{pmatrix} x & 5 \\ -3 & y \end{pmatrix}$ が $A = A^{-1}$ を満たすとき、$x, y$ を求めよ。

15. 行列 $A = \begin{pmatrix} a & 3 \\ 3 & 1 \end{pmatrix}$ の逆行列が $A^{-1} \begin{pmatrix} 1 & x \\ b & y \end{pmatrix}$とする。$x, y$を求めなさい。

16. （ケーリ・ハミルトン） $2 \times 2$ 行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ について考える。ここで、$a, b, c, d$ は実数とする。また、$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ を $2 \times 2$ 単位行列、$O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ を $2 \times 2$ 零行列とする。$A^2 - (a+d)A + (ad-bc)I = O$ 成り立つことを計算で示しなさい。

17. （ケーリ・ハミルトン） 行列 $A = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix}$ とする。$A^3$ を $pA + qI$ (ただし $p, q$ は実数) の形で表しなさい。また、$A^3$ を求めなさい。

18. （ケーリ・ハミルトン）行列 $A = \begin{pmatrix} a & 1 \\ 1 & -a \end{pmatrix}$ について、$A^{10}$ の $(2,2)$成分を求めなさい。

19. $A = \begin{pmatrix} 9 & 4 & 8 \\ -8 & -3 & -8 \\ 4 & 2 & 5 \end{pmatrix}$ とし、$E$ を 3 次の単位行列とする。このとき、次の問いに答えよ。

    (1) $A^2 - 10A = -9E$ であることを示せ。

    (2) $AB = \begin{pmatrix} -3 & 4 & -18 \\ 5 & -1 & 18 \\ -4 & 1 & -9 \end{pmatrix}$ を満たす行列 $B$ を求めよ。

20. （逆行列の有無）$A, B, C$ はいずれも 2 次の正方行列であり、下記の性質を満たす $A, B, C$ のうち逆行列をもつものはどれか。理由を付して答えよ。

    (a) $A$ は $A^2 + A - 2E = O$ を満たす。

    (b) $B$ は $B^2 = O$ を満たす。

    (c) $C = E + B$

### 問題（標準）

1.  **可換な行列の探求:**

    行列 $A = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}$ と可換な（つまり $AB = BA$ を満たす） $2 \times 2$ 行列 $B = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ の $a, b, c, d$ が満たすべき条件を求めなさい。

2.  **冪零行列と恒等行列:**

    $N = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$ とする。

    (a) $N^2$ と $N^3$ を計算しなさい。
    
    (b) $I$ を $3 \times 3$ の単位行列とする。 $(I-N)(I+N+N^2)$ を計算し、結果が $I$ になることを示しなさい（この問題が後の問題のヒントになります）。

3.  **行列のトレースと積の性質:**

    行列 $M = \begin{pmatrix}x & y \\ z & w \end{pmatrix}$ のトレース $\mathrm{tr}(M)$ とは、その対角成分の和のことで、 $\mathrm{tr}(M) = x + w$ です。

    $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, B = \begin{pmatrix} p & q \\ r & s \end{pmatrix}$ を一般的な $2 \times 2$ 行列とする。
    積 $AB$ と $BA$ を計算し、 $\mathrm{tr}(AB) = \mathrm{tr}(BA)$ が成り立つことを示しなさい。（この性質は一般の $n \times n$ 行列でも成り立ちます）。

4.  **行列の $n$ 乗の規則性:**

    $A = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ （回転行列）とする。
    
    (a) $A^2$ および $A^3$ を計算しなさい。（三角関数の加法定理を用いるとよいでしょう）
    
    (b) $A^n$ ($n$ は正の整数) の形を推測し、数学的帰納法を用いてその推測が正しいことを証明しなさい。

5.  **射影行列（Idempotent Matrix）の性質:**

    ある正方行列 $X$ が $X^2 = X$ を満たすとき、$X$ は冪等行列（idempotent matrix）または射影行列と呼ばれます。
    $P$ が冪等行列であるとき、以下のことを証明しなさい。

    (a) $I-P$ も冪等行列である（$(I-P)^2 = (I-P)$を示す）。
    
    (b) $P(I-P) = (I-P)P = O$ （$O$ は零行列）である。

6.  **行列の$n$乗の探求1**

    $A = \begin{pmatrix} 2 & -1 \\ 1 & 0 \end{pmatrix}$ とする。$A^2, A^3$ を計算し、$A^n$ の規則性を見つけるのは難しい。しかし、$A=I+N$ となる $N = \begin{pmatrix} 1 & -1 \\ 1 & -1 \end{pmatrix}$ を考えると、$N^2=O$（ゼロ行列）となることを示せ。これを用いて $(I+N)^n$ を二項展開し、$A^n$ を求めよ。

7.  **行列の$n$乗の探求2**

    $A = \begin{pmatrix} \lambda & 1 \\ 0 & \lambda \end{pmatrix}$ ($\lambda$ はスカラー) とする。$A^n$ ($n$ は自然数) を推測し証明せよ。

8.  **行列の$n$乗の探求3**

    $A = \begin{pmatrix} -1 & -3 \\ 4 & 6 \end{pmatrix},  U=\begin{pmatrix} 1 & 3 \\ -1 & -4 \end{pmatrix}$ に対して次の問いに答えなさい。

    (a) 行列 $U$ の逆行列 $U^{-1}$ を求めなさい

    (b) $B = U^{-1}AU$ とおくとき、自然数 $n$ に対して、行列 $B^n$ を求めなさい

    (c) 自然数 $n$ に対して、行列 $A^n$ を求めなさい。

### 問題 （少し難しい）

1.  行列 $B = \begin{pmatrix} 4 & -1 \\ 3 & 2 \end{pmatrix}$ とする。$B$ の逆行列 $B^{-1}$ を $pB + qI$ (ただし $p, q$ は実数) の形で表しなさい。

2.  行列 $C = \begin{pmatrix} 1 & 3 \\ -1 & 0 \end{pmatrix}$ とする。$f(C) = C^4 - C^3 + 4C^2 - 2C + 5I$ の値を計算しなさい。

3.  行列 $D = \begin{pmatrix} 2 & 4 \\ -1 & -1 \end{pmatrix}$ とする。$A^6 + 2 A^4 + 2 A^3 + 2 A^2 + 2 A + I$ の値を計算しなさい。

4.  行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ が $A^2 - 97A + 2010E = O$ を満たすとき、$a+d$, $ad-bc$ の値の組をすべて求めよ。ただし、$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ とする。

5.  $a$ を実数とする。行列 $X = \begin{pmatrix} x & -y \\ y & x \end{pmatrix}$ が $X^2 - 2X + aE = O$ をみたすような実数 $x, y$ を求めよ。ただし、$E = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ とする。

6.  $N$ を $n \times n$ 行列とし、$N^k = O$ (零行列) となる正の整数 $k$ が存在するとする（このような行列 $N$ を冪零行列といいます）。

    (a) $I - N$ が正則行列（逆行列を持つ行列）であることを証明しなさい。
    
    (b) $(I - N)^{-1}$ を $I, N, N^2, \dots, N^{k-1}$ を用いて具体的に表しなさい。

7.  $a, b, c, d$ は実数とする。行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ に対して、等式 $A^2 + BA = 4E$, $AB + B^2 = 12E$ をみたす行列 $B$ が存在するとき、$a+d$ と $ad-bc$ の値を求めよ。ただし、$E = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ とする。

8.  2次元ベクトル $A_n (n=1, 2, 3, \dots)$ が以下の関係式を満たすとき、$A_n$ を求めよ。

    $A_1 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$,　$A_2 = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$,
    
$$A_{n+2} = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} A_{n+1} + \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} A_n \quad (n=1, 2, 3, \dots)$$

### 問題 （データサイエンス応用 +$\beta$）

1.  **線形回帰モデルによる予測値の計算**
    ある地域の家の価格を、その広さ（平方メートル）と築年数（年）から予測する線形回帰モデルが構築されたとします。予測モデルは以下の式で表されます。

    予測価格 = $150 + 2.5 \times (\text{広さ}) - 1.5 \times (\text{築年数})$

    3軒の家について、以下の特徴量データがあるとします。

    * 家1: 広さ 80 m², 築年数 10 年
    * 家2: 広さ 120 m², 築年数 5 年
    * 家3: 広さ 70 m², 築年数 20 年

    この情報を以下の行列とベクトルで表現します。

    * 特徴量行列 $X$ (各行が家、列が切片項、広さ、築年数):
        $$X = \begin{pmatrix} 1 & 80 & 10 \\ 1 & 120 & 5 \\ 1 & 70 & 20 \end{pmatrix}$$
    * 係数ベクトル $\beta$:
        $$\beta = \begin{pmatrix} 150 \\ 2.5 \\ -1.5 \end{pmatrix}$$

    (a) 行列の積 $X\beta$ を計算しなさい。
    (b) 計算結果のベクトル $Y_{pred} = X\beta$ の各要素は何を表しているか説明してください

2. **ソーシャルネットワークにおける「友達の友達」の数**

    4人のユーザー (A, B, C, D) からなる小さなソーシャルネットワークを考えます。ユーザー間のフォロー関係を以下の隣接行列 $A$ で表します。$A_{ij}=1$ はユーザー $i$ がユーザー $j$ をフォローしていることを意味し、$A_{ij}=0$ はフォローしていないことを意味します（自分自身はフォローしないとします）。

    $A = \begin{pmatrix} 0 & 1 & 1 & 0 \\ 0 & 0 & 1 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix} \quad \begin{matrix} \leftarrow A \\ \leftarrow B \\ \leftarrow C \\ \leftarrow D \end{matrix}$
    ($i$ 行 $j$ 列が $i \rightarrow j$ のフォロー関係)

    (a) 行列の積 $A^2 = AA$ を計算しなさい。
    (b) 計算結果 $(A^2)_{ij}$ は、$A^2$ の $(i,j)$要素です。ユーザー $i$ からユーザー $j$ へ、ちょうど1人の他のユーザーを経由していく経路（$i \rightarrow k \rightarrow j$ となる経路）の数を表します。例えば、$(A^2)_{14}$ の値は何を意味しますか？また、$(A^2)_{22}$ の値は何を意味しますか？

3. **顧客のプラン移行予測 (マルコフ連鎖)**

    あるサブスクリプションサービスでは、顧客が「無料プラン」「基本プラン」「プレミアムプラン」のいずれかの状態にあるとします。1ヶ月後に顧客がどのプランに移行するか（または留まるか）の確率を行列 $T$ で表します。

    $T = \begin{pmatrix} 0.6 & 0.3 & 0.1 \\ 0.1 & 0.7 & 0.2 \\ 0.0 & 0.1 & 0.9 \end{pmatrix} \quad \begin{matrix} \leftarrow \text{無料} \\ \leftarrow \text{基本} \\ \leftarrow \text{プレミアム} \end{matrix}$
    (行 $i$ から 列 $j$ への移行確率)

    例えば、$T_{12}=0.3$ ($T_{ij}$は行列$T$の$(i,j)$要素は、無料プランの顧客が1ヶ月後に基本プランに移行する確率が 0.3 であることを意味します。

    現在、顧客の分布が「無料プラン: 50%, 基本プラン: 30%, プレミアムプラン: 20%」であるとします。この初期状態分布をベクトル $p_0$ で表します。

    $p_0 = \begin{pmatrix} 0.5 & 0.3 & 0.2 \end{pmatrix}$

    (a) 1ヶ月後の顧客の状態分布 $p_1 = p_0 T$ を計算しなさい。
    (b) 2ヶ月後の顧客の状態分布 $p_2 = p_1 T = p_0 T^2$ を計算しなさい。


## 8. よくある質問と解答

### Q1: 行列の積が定義されるための条件は何ですか？
A1: 行列 $A$ と $B$ の積 $AB$ が定義されるためには、$A$ の列数と $B$ の行数が一致している必要があります。すなわち、$A$ が $m \times n$ 行列で $B$ が $p \times q$ 行列のとき、$n = p$ であれば積 $AB$ が定義でき、結果は $m \times q$ 行列になります。

### Q2: 行列が正則であることと逆行列が存在することは同じ意味ですか？
A3: はい、同じ意味です。正方行列 $A$ が正則（可逆）であるとは、その逆行列 $A^{-1}$ が存在することを意味します。2次の行列の場合、$\det(A) \neq 0$ が正則であるための必要十分条件です。




