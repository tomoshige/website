# 線形代数学 I - 講義ノート 第5回

## 1. 講義情報と予習ガイド

**講義回**: 第5回  
**テーマ**: 行列の積  
**関連項目**: 行列積の定義、計算方法、注意点、逆行列の導入  
**予習すべき内容**: 第4回の内容（行列の定義、行列の和、行列のスカラー倍）

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
$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$, $C = \begin{pmatrix} 9 & 10 \\ 11 & 12 \end{pmatrix}$ について、$(A+B)C$ と $AC + BC$ を計算し、分配法則を確認しましょう。

$A + B = \begin{pmatrix} 1+5 & 2+6 \\ 3+7 & 4+8 \end{pmatrix} = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix}$

$(A+B)C = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix} \begin{pmatrix} 9 & 10 \\ 11 & 12 \end{pmatrix} = \begin{pmatrix} 6 \times 9 + 8 \times 11 & 6 \times 10 + 8 \times 12 \\ 10 \times 9 + 12 \times 11 & 10 \times 10 + 12 \times 12 \end{pmatrix} = \begin{pmatrix} 54 + 88 & 60 + 96 \\ 90 + 132 & 100 + 144 \end{pmatrix} = \begin{pmatrix} 142 & 156 \\ 222 & 244 \end{pmatrix}$

一方、
$AC = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 9 & 10 \\ 11 & 12 \end{pmatrix} = \begin{pmatrix} 31 & 34 \\ 71 & 78 \end{pmatrix}$

$BC = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} \begin{pmatrix} 9 & 10 \\ 11 & 12 \end{pmatrix} = \begin{pmatrix} 111 & 122 \\ 151 & 166 \end{pmatrix}$

$AC + BC = \begin{pmatrix} 31 & 34 \\ 71 & 78 \end{pmatrix} + \begin{pmatrix} 111 & 122 \\ 151 & 166 \end{pmatrix} = \begin{pmatrix} 142 & 156 \\ 222 & 244 \end{pmatrix}$

よって、$(A+B)C = AC + BC$ であることが確認できました。

### 4.2 行列積の非可換性

行列の積には、一般に交換法則が成り立ちません。すなわち、$AB \neq BA$ となる場合があります。

**例 4.2.1**：
$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ と $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$ について、$AB$ と $BA$ を計算してみましょう。

$AB = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$（先ほど計算した通り）

$BA = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 5 \times 1 + 6 \times 3 & 5 \times 2 + 6 \times 4 \\ 7 \times 1 + 8 \times 3 & 7 \times 2 + 8 \times 4 \end{pmatrix} = \begin{pmatrix} 5 + 18 & 10 + 24 \\ 7 + 24 & 14 + 32 \end{pmatrix} = \begin{pmatrix} 23 & 34 \\ 31 & 46 \end{pmatrix}$

$AB \neq BA$ であるため、行列の積は一般に可換ではないことがわかります。この非可換性は、行列が表す線形変換の順序が重要であることを示しています。

### 4.3 特殊な行列と行列積

#### 4.3.1 単位行列

> **定義 4.3.1（単位行列）**  
> $n$ 次の単位行列 $I_n$ は、主対角線上の成分がすべて $1$ で、それ以外の成分がすべて $0$ である $n \times n$ の正方行列です：
> 
> $I_n = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{pmatrix}$

単位行列は、任意の $n \times m$ 行列 $A$ に対して、$I_n A = A$ かつ $A I_m = A$ を満たします。この性質から、単位行列は行列の積に関する「単位元」と呼ばれます。

**例 4.3.1**：
$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ と $2$ 次の単位行列 $I_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ について、$I_2 A$ と $A I_2$ を計算してみましょう。

$I_2 A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 1 \times 1 + 0 \times 3 & 1 \times 2 + 0 \times 4 \\ 0 \times 1 + 1 \times 3 & 0 \times 2 + 1 \times 4 \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = A$

$A I_2 = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 \times 1 + 2 \times 0 & 1 \times 0 + 2 \times 1 \\ 3 \times 1 + 4 \times 0 & 3 \times 0 + 4 \times 1 \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = A$

よって、$I_2 A = A I_2 = A$ が確認できました。

#### 4.3.2 零行列

すべての成分が $0$ である行列を零行列と呼び、$O$ で表します。任意の行列 $A$ に対して、$A + O = A$ および $A \times O = O \times A = O$ が成り立ちます。

### 4.4 逆行列

> **定義 4.4.1（逆行列）**  
> $n$ 次正方行列 $A$ に対して、$AB = BA = I_n$ を満たす $n$ 次正方行列 $B$ が存在するとき、$B$ を $A$ の逆行列といい、$A^{-1}$ と表します。

逆行列が存在する行列を**正則行列**（または**可逆行列**）と呼びます。逆行列が存在しない行列は**特異行列**（または**非可逆行列**）と呼ばれます。

#### 4.4.1 2次の行列の逆行列の計算

2次の行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ に対して、$\det(A) = ad - bc \neq 0$ であれば、$A$ は正則であり、その逆行列は次の式で与えられます：

$A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$

**例 4.4.1**：
$A = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix}$ の逆行列を求めましょう。

まず、$\det(A) = 3 \times 2 - 1 \times 2 = 6 - 2 = 4 \neq 0$ なので、$A$ は正則です。

$A^{-1} = \frac{1}{4} \begin{pmatrix} 2 & -1 \\ -2 & 3 \end{pmatrix} = \begin{pmatrix} \frac{1}{2} & -\frac{1}{4} \\ -\frac{1}{2} & \frac{3}{4} \end{pmatrix}$

検算として、$A A^{-1}$ を計算してみましょう：

$A A^{-1} = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix} \begin{pmatrix} \frac{1}{2} & -\frac{1}{4} \\ -\frac{1}{2} & \frac{3}{4} \end{pmatrix} = \begin{pmatrix} 3 \times \frac{1}{2} + 1 \times (-\frac{1}{2}) & 3 \times (-\frac{1}{4}) + 1 \times \frac{3}{4} \\ 2 \times \frac{1}{2} + 2 \times (-\frac{1}{2}) & 2 \times (-\frac{1}{4}) + 2 \times \frac{3}{4} \end{pmatrix} = \begin{pmatrix} \frac{3}{2} - \frac{1}{2} & -\frac{3}{4} + \frac{3}{4} \\ 1 - 1 & -\frac{1}{2} + \frac{3}{2} \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I_2$

よって、求めた逆行列が正しいことが確認できました。

## 5. Pythonによる実装と可視化

NumPyを使用して行列の積と逆行列を計算してみましょう。

```python
import numpy as np
import matplotlib.pyplot as plt

# 行列の定義
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("行列A:\n", A)
print("行列B:\n", B)

# 行列の積
C = np.dot(A, B)  # または C = A @ B （Python 3.5以降）
print("A×B:\n", C)

# 行列積の非可換性を確認
D = np.dot(B, A)
print("B×A:\n", D)
print("A×B = B×Aは", np.array_equal(C, D))

# 単位行列
I = np.eye(2)  # 2次の単位行列
print("単位行列I:\n", I)
print("I×A:\n", np.dot(I, A))

# 逆行列の計算
A_inv = np.linalg.inv(A)
print("Aの逆行列:\n", A_inv)

# 逆行列の検証
print("A×A^(-1):\n", np.dot(A, A_inv))
print("A^(-1)×A:\n", np.dot(A_inv, A))
```

### 5.1 線形変換としての行列積の可視化

行列が線形変換を表すことを可視化してみましょう。

```python
import numpy as np
import matplotlib.pyplot as plt

# 2x2の行列（線形変換）を定義
A = np.array([[1, 0.5], [0.5, 1]])
B = np.array([[0, -1], [1, 0]])  # 90度回転
C = np.dot(A, B)  # 合成変換

# 可視化のための格子点を生成
x = np.linspace(-3, 3, 7)
y = np.linspace(-3, 3, 7)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.flatten(), Y.flatten()])

# 各変換を適用
points_A = np.dot(A, points)
points_B = np.dot(B, points)
points_C = np.dot(C, points)

# 可視化
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(points_A[0], points_A[1], c='red', s=50)
plt.scatter(points[0], points[1], c='blue', s=20)
plt.grid(True)
plt.title('変換A')
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.scatter(points_B[0], points_B[1], c='red', s=50)
plt.scatter(points[0], points[1], c='blue', s=20)
plt.grid(True)
plt.title('変換B（90度回転）')
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.scatter(points_C[0], points_C[1], c='red', s=50)
plt.scatter(points[0], points[1], c='blue', s=20)
plt.grid(True)
plt.title('合成変換C = A×B')
plt.axis('equal')

plt.tight_layout()
plt.show()
```

## 6. データサイエンスにおける応用例

### 6.1 線形回帰における行列演算

線形回帰モデルでは、説明変数と目的変数の関係を行列で表現します。$n$個のデータポイントと$p$個の特徴量があるとき、データ行列 $X$ は $n \times p$ の行列、目的変数ベクトル $\mathbf{y}$ は $n$ 次元ベクトルとなります。線形回帰の係数ベクトル $\mathbf{\beta}$ は、正規方程式 $X^T X \mathbf{\beta} = X^T \mathbf{y}$ の解として与えられ、これは $(X^T X)^{-1} X^T \mathbf{y}$ で計算できます。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# サンプルデータの生成
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_matrix = np.column_stack((np.ones(X.shape[0]), X))  # 切片項を追加

# 正規方程式を使った線形回帰
beta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y
print("回帰係数:", beta)

# 結果の可視化
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.7)
plt.plot(X, beta[0] + beta[1] * X, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('線形回帰モデル')
plt.grid(True)
plt.show()
```

### 6.2 健康データ分析における行列演算の応用

医療画像処理や生体信号処理などの健康データ分析では、行列演算が重要な役割を果たします。例えば、医療画像の変換、フィルタリング、特徴抽出などには行列の積が頻繁に使用されます。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 手書き数字データの読み込み（医療画像の代わりとして）
digits = load_digits()
image = digits.images[0]

# 行列を使った画像の操作（例：エッジ検出フィルタ）
edge_filter = np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]])

# フィルタリング処理（畳み込み）
def convolve2d(image, kernel):
    output = np.zeros_like(image)
    padding = kernel.shape[0] // 2
    padded_image = np.pad(image, padding, mode='constant')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(
                padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel
            )
    return output

filtered_image = convolve2d(image, edge_filter)

# 結果の可視化
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('原画像')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('エッジ検出後')
plt.axis('off')

plt.tight_layout()
plt.show()
```

## 7. 演習問題

### 基本問題

1. 以下の行列の積を計算しなさい。
      
   (a) $\begin{pmatrix} 2 & 3 \\ 4 & 5 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 2 & 3 \end{pmatrix}$
      
   (b) $\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}$

2. 以下の行列 $A = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$ の逆行列を求め、$A A^{-1} = I$ であることを確認しなさい。


3. 行列 $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ と $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$ に対して、$AB$ と $BA$ を計算し、$AB \neq BA$ であることを確認しなさい。

5. 行列 $A = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$ について、$A^2 = AA$, $A^3 = AAA$, $A^4 = AAAA$ を計算しなさい。規則性を見つけ、$A^n$ の一般形を予想しなさい。

6. 行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ が正則であるための必要十分条件は $\det(A) = ad - bc \neq 0$ です。この条件を満たさない例を挙げ、その行列が逆行列を持たないことを確認しなさい。


### 応用問題


7. ある会社の製品 A, B, C の3つの成分 X, Y, Z の含有量（単位: g）は以下の通りです：
   - 製品 A: $X = 2, Y = 1, Z = 3$
   - 製品 B: $X = 1, Y = 2, Z = 2$
   - 製品 C: $X = 3, Y = 1, Z = 1$
   
   各成分の単価（円/g）は $X = 100, Y = 200, Z = 150$ です。行列の積を使って、各製品の原価を計算しなさい。

8. **健康データサイエンス応用問題**：3人の患者（患者1, 2, 3）の血圧、血糖値、コレステロール値の標準化スコア（平均0、標準偏差1に変換したもの）が以下の行列 $X$ で表されるとします（行は患者1,2,3を表し、列は1:血圧、2:血糖値、3:コレステロール値を表す）：
   
   $X = \begin{pmatrix} 
   0.5 & 1.2 & -0.3 \\
   -0.8 & 0.4 & 0.2 \\
   1.3 & -0.1 & 0.7
   \end{pmatrix}$
   
   ここで、行は患者、列は順に血圧、血糖値、コレステロール値を表します。
   
   また、これらの値と心疾患リスクの関連を表す重み係数が $w = \begin{pmatrix} 0.4 \\ 0.3 \\ 0.5 \end{pmatrix}$ で与えられているとします。
   
   (a) 行列の積 $Xw$ を計算し、各患者の心疾患リスクスコアを求めなさい。
   
   (b) どの患者が最もリスクが高いでしょうか？
   
   (c) 各測定値（血圧、血糖値、コレステロール値）がリスクスコアにどの程度寄与しているかを分析しなさい。

## 8. よくある質問と解答

### Q1: 行列の積が定義されるための条件は何ですか？
A1: 行列 $A$ と $B$ の積 $AB$ が定義されるためには、$A$ の列数と $B$ の行数が一致している必要があります。すなわち、$A$ が $m \times n$ 行列で $B$ が $p \times q$ 行列のとき、$n = p$ であれば積 $AB$ が定義でき、結果は $m \times q$ 行列になります。

### Q2: 行列が正則であることと逆行列が存在することは同じ意味ですか？
A3: はい、同じ意味です。正方行列 $A$ が正則（可逆）であるとは、その逆行列 $A^{-1}$ が存在することを意味します。2次の行列の場合、$\det(A) \neq 0$ が正則であるための必要十分条件です。