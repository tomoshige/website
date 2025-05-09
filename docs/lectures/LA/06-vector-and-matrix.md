# 線形代数学 I / 基礎 / II
## 第6回 講義ノート：データサイエンスに必要な行列


## 1. 講義情報と予習ガイド

-   **講義回**: 第6回
-   **関連項目**: 行列の特殊形、行列の性質
-   **予習すべき内容**:
    -   行列の定義（第4回）
    -   行列の和とスカラー倍（第4回）
    -   行列の積（第5回）

- **スライド**: [スライド](./slides/06-vector-and-matrix-slide.pdf)


## 2. 学習目標

1.  単位行列の定義と性質を理解し、応用できる
2.  転置行列の定義と性質を理解し、応用できる
3.  対称行列の定義と性質を理解し、応用できる


## 3. 基本概念

### 3.1 単位行列（Identity Matrix）

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

**例**: 2次単位行列と3次単位行列

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

**単位行列の性質**:

1.  任意の行列 $A$ に対して: $AI = IA = A$ （ただし $I$ は適切なサイズの単位行列）
2.  単位行列 $I$ は対称行列である
3.  単位行列 $I$ の逆行列は $I$ 自身である: $I^{-1} = I$
4.  単位行列 $I$ のランクは $n$ である（$I$ が $n \times n$ 行列の場合）

**数値例**:

ある $2 \times 2$ 行列 $A = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix}$ と単位行列 $I_2$ との積を計算してみましょう。

$$
A \cdot I_2 = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix} \cdot \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 3 \cdot 1 + 1 \cdot 0 & 3 \cdot 0 + 1 \cdot 1 \\ 2 \cdot 1 + 4 \cdot 0 & 2 \cdot 0 + 4 \cdot 1 \end{pmatrix} = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix} = A
$$

同様に $I_2 \cdot A$ も計算すると結果は $A$ になります。

### 3.2 転置行列（Transpose Matrix）

> **定義**: 行列 $A$ の転置行列 $A^T$ は、$A$ の行と列を入れ替えた行列である。
>
> $A$ が $m \times n$ 行列の場合、$A^T$ は $n \times m$ 行列となる。
>
> 具体的には、$A = (a_{ij})$ に対して、$A^T = (a_{ji})$ である。

**例**: 行列 $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$ の転置行列は:

$$
A^T = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}
$$

**転置行列の性質**:

1.  $(A^T)^T = A$
2.  $(A + B)^T = A^T + B^T$
3.  $(cA)^T = cA^T$ （$c$ はスカラー）
4.  $(AB)^T = B^T A^T$ （行列の積の転置は、転置の積の順序を逆にしたものに等しい）

**数値例**:

行列 $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ と $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$ について、$(A + B)^T$ と $A^T + B^T$ が等しいことを確認します。

$$
A + B = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} + \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix}
$$

$$
(A + B)^T = \begin{pmatrix} 6 & 10 \\ 8 & 12 \end{pmatrix}
$$

$$
A^T + B^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} + \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} = \begin{pmatrix} 6 & 10 \\ 8 & 12 \end{pmatrix}
$$

したがって、$(A + B)^T = A^T + B^T$ が成り立ちます。

### 3.3 対称行列（Symmetric Matrix）

> **定義**: 正方行列 $A$ が対称行列であるとは、$A = A^T$ が成り立つことである。つまり、$a_{ij} = a_{ji}$ がすべての $i, j$ について成り立つ。

**例**:

$$
A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{pmatrix}
$$

この行列では、$a_{12} = a_{21} = 2$, $a_{13} = a_{31} = 3$, $a_{23} = a_{32} = 5$ となっており、主対角線に関して対称な位置にある要素が等しいため、対称行列です。

**対称行列の性質**:

1.  対称行列の対角要素 $a_{ii}$ は実数である（複素行列の場合、エルミート行列の対角要素は実数）

2.  任意の行列 $A$ に対して、$A^T A$ と $A A^T$ は常に対称行列である

3.  対称行列同士の和も対称行列になる

**数値例**:

任意の行列から対称行列を作る方法として、$A^T A$ を計算してみましょう。

$$
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
$$

$$
A^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}
$$

$$
A^T A = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} \cdot \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 1 \cdot 1 + 3 \cdot 3 & 1 \cdot 2 + 3 \cdot 4 \\ 2 \cdot 1 + 4 \cdot 3 & 2 \cdot 2 + 4 \cdot 4 \end{pmatrix} = \begin{pmatrix} 10 & 14 \\ 14 & 20 \end{pmatrix}
$$

結果の行列が対称行列になっていることを確認できます。


## 4. 演習問題

**行列の積**

1.  $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$ とする。$AB$ を計算せよ。

2.  $A = \begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix}, B = \begin{pmatrix} 3 & 1 \\ -1 & 0 \end{pmatrix}$ とする。$BA$ を計算せよ。

3.  $A = \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}, B = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$ とする。$AB$ と $BA$ を計算せよ。

4.  $A = \begin{pmatrix} 1 & 0 & -1 \\ 2 & 1 & 0 \end{pmatrix}, B = \begin{pmatrix} 1 & 1 \\ 0 & 2 \\ 3 & 0 \end{pmatrix}$ とする。$AB$ を計算せよ。

5.  $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, C = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$ とする。$AC$ と $CA$ はそれぞれ定義されるか。定義される場合は計算し、定義されない場合はその理由を述べよ。

6.  $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ とする。$A^2$ と $A^3$ を計算せよ。（$A^2 = AA$, $A^3 = AAA$）

7.  $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$、$E_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$、$E_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$ とする。$E_2 A$ と $A E_3$ を計算し、結果が $A$ と一致することを確認せよ。

8.  $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ と $2 \times 2$ の単位行列 $E$ について、$AE = EA = A$ となることを計算により確認せよ。

9.  $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$ の転置行列 $A^T$ を求めよ。

10. $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, B = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ とする。$(A+B)^T$ と $A^T + B^T$ をそれぞれ計算し、両者が等しいことを確認せよ。

11. 問10の行列 $A, B$ について、$(AB)^T$ と $B^T A^T$ をそれぞれ計算し、両者が等しいことを確認せよ。

12. 次の行列の中から、(i) 対称行列、(ii) 反対称行列（$B^T = -B$となる行列）をすべて選べ。
    (a) $$ \begin{pmatrix} 1 & 2 \\\ 2 & 3 \end{pmatrix} $$
    (b) $$ \begin{pmatrix} 0 & 1 \\\ -1 & 0 \end{pmatrix} $$
    (c) $$ \begin{pmatrix} 1 & 0 & 0 \\\ 0 & 2 & 0 \\\ 0 & 0 & 3 \end{pmatrix} $$
    (d) $$ \begin{pmatrix} 1 & 1 & 1 \\\ 1 & 1 & 1 \end{pmatrix} $$
    (e) $$ \begin{pmatrix} 0 & -2 & 3 \\\ 2 & 0 & -1 \\\ -3 & 1 & 0 \end{pmatrix} $$
    (f) $$ \begin{pmatrix} 1 & 2 \\\ 3 & 4 \end{pmatrix} $$

13. $A = \begin{pmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{pmatrix}$ とする。$A^T A$ を計算し、この結果が対称行列になることを確認せよ。

14. $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ とする。$S = \frac{1}{2}(A + A^T)$ と $K = \frac{1}{2}(A - A^T)$ をそれぞれ計算せよ。そして、$S$ が対称行列、$K$ が反対称行列であり、かつ $A = S + K$ が成り立つことを確認せよ。

15. $A, B$ を $n \times n$ 行列とする。$X = A^T B A$ とおく。
    (a) $X^T$ を $A, B, A^T, B^T$ を用いて表せ。
    (b) $B$ が対称行列である場合、$X = A^T B A$ も対称行列になることを示せ。

16. **対称・反対称部分への分解と積**
    任意の $n \times n$ 正方行列 $A$ に対して、$S = \frac{1}{2}(A+A^T)$（対称部分）、$K = \frac{1}{2}(A-A^T)$（反対称部分）とおく。
    (a) $A = S+K$ および $A^T = S-K$ であることを確認せよ。
    (b) $A A^T$ を $S$ と $K$ を用いて表せ。（ヒント: $A A^T = (S+K)(S-K)$ を展開する）

17. **対称行列の決定と性質**
    行列 $A = \begin{pmatrix} 1 & x & y \\ 2 & 3 & z \\ 0 & 1 & 4 \end{pmatrix}$ が対称行列であるように、$x, y, z$ の値を定めよ。

18. **転置と双線形形式**
    $A$ を $n \times n$ 行列、$x, y$ を $n \times 1$ の列ベクトルとする。$s = x^T A y$ はスカラー（$1 \times 1$ 行列）である。
    (a) $s^T$ を $x, y, A$ の転置を用いて表せ。（スカラーの転置は元のスカラーと同じであることを思い出そう）
    (b) $A$ が対称行列のとき、$x^T A y = y^T A x$ が成り立つことを示せ。

19. **条件を満たす対称行列**
    次の3つの条件をすべて満たす $2 \times 2$ 行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ を$b$を用いて表せ。
    (i) $A$ は対称行列である。($c=b$)
    (ii) $A$ の対角成分の和（トレース）は 5 である ($a+d=5$)。
    (iii) $A$ の行列式は 4 である ($ad-bc=4$)。