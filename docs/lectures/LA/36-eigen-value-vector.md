# 線形代数学 I / 基礎 / II　講義ノート

## 第36回: データサイエンスのための2次形式と基礎

### 1. 講義情報と予習ガイド

**講義回**: 第36回  
**関連項目**: 2次形式、正定値・半正定値行列、分散共分散行列  
**予習すべき内容**: 行列の対角化、固有値と固有ベクトル、対称行列

### 2. 学習目標

本講義の終了時には、以下のことができるようになることを目指します：

1. 2次形式の定義を理解し、行列表現ができる
2. 正定値・半正定値行列の概念と判定法を理解する
3. 正定値性と固有値の関係を理解し、判定に応用できる
4. データサイエンスにおける2次形式と正定値行列の役割を理解する
5. 分散共分散行列の正定値性を理解し、データ分析に応用できる

### 3. 基本概念

#### 3.1 2次形式の定義

> **定義**: $n$個の変数 $x_1, x_2, \ldots, x_n$ に関する2次の多項式で、各項が変数の2乗または積の形をしているものを**2次形式**という。一般的に、$n$次元ベクトル $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$ に対して、$n \times n$ 行列 $A$ を用いて $Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ と表現される。

2次形式は、$n$個の変数 $x_1, x_2, \ldots, x_n$ の2次の多項式で、以下の形で表されます：

$$
Q(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}\sum_{j=1}^{n} a_{ij}x_i x_j
$$

ここで、$a_{ij}$ は係数です。行列表記では：

$$
Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}
$$

ここで、$A = (a_{ij})$ は $n \times n$ の係数行列です。

**重要なポイント**：行列 $A$ は常に対称行列として扱うことができます。なぜなら、任意の2次形式において $a_{ij}x_i x_j + a_{ji}x_j x_i = (a_{ij} + a_{ji})x_i x_j$ となるため、$A$ を対称行列 $\frac{1}{2}(A + A^T)$ に置き換えても、同じ2次形式を表すからです。

#### 3.2 2次形式の標準形と対角化

対称行列 $A$ は直交行列 $P$ を用いて対角化できるため：

$$
A = PDP^T
$$

ここで、$D$ は固有値を対角成分にもつ対角行列です。これを2次形式に適用すると：

$$
Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \mathbf{x}^T PDP^T \mathbf{x} = (P^T\mathbf{x})^T D (P^T\mathbf{x}) = \mathbf{y}^T D \mathbf{y}
$$

ここで、$\mathbf{y} = P^T\mathbf{x}$ は変数の直交変換です。つまり、適切な変数変換により、2次形式は以下の標準形で表せます：

$$
Q(\mathbf{y}) = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots + \lambda_n y_n^2
$$

ここで、$\lambda_1, \lambda_2, \ldots, \lambda_n$ は行列 $A$ の固有値です。

### 4. 理論と手法

#### 4.1 正定値・半正定値行列

> **定義**: 対称行列 $A$ が**正定値**であるとは、任意の非零ベクトル $\mathbf{x} \neq \mathbf{0}$ に対して $\mathbf{x}^T A \mathbf{x} > 0$ が成り立つことをいう。
> 
> **定義**: 対称行列 $A$ が**半正定値**であるとは、任意のベクトル $\mathbf{x}$ に対して $\mathbf{x}^T A \mathbf{x} \geq 0$ が成り立ち、かつある非零ベクトル $\mathbf{x} \neq \mathbf{0}$ に対して $\mathbf{x}^T A \mathbf{x} = 0$ となることをいう。

正定値行列は、2次形式がどのような非零入力に対しても常に正の値を取ることを意味します。これは、多変量データ分析において重要な性質となります。

**例**:
行列 $A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$ を考えます。この行列が正定値かどうかを調べるために、任意のベクトル $\mathbf{x} = (x, y)^T$ に対して $\mathbf{x}^T A \mathbf{x}$ を計算します：

$$
\mathbf{x}^T A \mathbf{x} = (x, y) \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = 2x^2 + 2xy + 3y^2
$$

これが任意の非零ベクトル $(x,y) \neq (0,0)$ に対して正であることを示す必要があります。

#### 4.2 正定値・半正定値と固有値の関係

> **定理**: 対称行列 $A$ が正定値であることと、$A$ のすべての固有値が正であることは同値である。
> 
> **定理**: 対称行列 $A$ が半正定値であることと、$A$ のすべての固有値が非負であることは同値である。

この定理により、行列の正定値性の判定は、固有値を計算することで行えます。

**証明の概略**:
対称行列 $A$ が直交行列 $P$ で対角化できることを利用します。$A = PDP^T$ とすると、任意のベクトル $\mathbf{x}$ に対して：

$$
\mathbf{x}^T A \mathbf{x} = \mathbf{x}^T PDP^T \mathbf{x} = \mathbf{y}^T D \mathbf{y} = \sum_{i=1}^{n} \lambda_i y_i^2
$$

ここで、$\mathbf{y} = P^T\mathbf{x}$ と置きました。$\mathbf{x} \neq \mathbf{0}$ のとき $\mathbf{y} \neq \mathbf{0}$ であり、$\mathbf{x}^T A \mathbf{x} > 0$ であるためには、すべての $\lambda_i > 0$ が必要十分条件となります。

#### 4.3 正定値性の判定法

正定値行列を判定する方法はいくつかあります：

1. **固有値による判定**：すべての固有値が正であれば、行列は正定値です。

2. **主座小行列式による判定（シルベスターの判定法）**：
   $n \times n$ 行列 $A$ について、左上の $k \times k$ 主座小行列式を $D_k$ とするとき、すべての $D_k > 0$ ($k = 1,2,\ldots,n$) であれば、$A$ は正定値です。

   例えば、$A = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$ の場合：
   - $D_1 = a_{11} > 0$
   - $D_2 = \begin{vmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{vmatrix} > 0$
   - $D_3 = \det(A) > 0$

3. **コレスキー分解**：正定値行列 $A$ は、下三角行列 $L$ を用いて $A = LL^T$ と分解できます。

#### 4.4 分散共分散行列と正定値性

データサイエンスでは、多変量データの分散共分散行列が正定値であることが重要です。$n$次元の確率変数 $\mathbf{X} = (X_1, X_2, \ldots, X_n)^T$ に対する分散共分散行列 $\Sigma$ は以下のように定義されます：

$$
\Sigma = \begin{pmatrix} 
\text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_n) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_n, X_1) & \text{Cov}(X_n, X_2) & \cdots & \text{Var}(X_n)
\end{pmatrix}
$$

理論的には、分散共分散行列は常に半正定値です。変数間に完全な線形関係がなければ、正定値となります。

### 5. Pythonによる実装と可視化

以下のコードは、2次形式の計算と可視化、および行列の正定値性の判定を行うものです。

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import eigh

# 1. 2次形式の計算と可視化
def visualize_quadratic_form(A, title="2次形式の可視化"):
    # 対称行列であることを確認
    A = (A + A.T) / 2
    
    # メッシュグリッドの作成
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # 2次形式の計算 Q(x,y) = [x y] A [x; y]
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            xy = np.array([X[i, j], Y[i, j]])
            Z[i, j] = xy.dot(A).dot(xy)
    
    # 3Dプロット
    fig = plt.figure(figsize=(12, 6))
    
    # 3D曲面プロット
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Q(x,y)')
    ax1.set_title(title)
    
    # 等高線プロット
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, 20, cmap=cm.coolwarm)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title("等高線図")
    plt.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    # 固有値と固有ベクトルの計算
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    print("固有値:", eigenvalues)
    print("固有ベクトル:")
    for i in range(len(eigenvalues)):
        print(f"λ_{i+1} = {eigenvalues[i]:.4f}の固有ベクトル: {eigenvectors[:, i]}")
    
    # 行列の正定値性の判定
    if np.all(eigenvalues > 0):
        print("行列は正定値です")
    elif np.all(eigenvalues >= 0):
        print("行列は半正定値です")
    else:
        print("行列は正定値でも半正定値でもありません")

# 2. 様々な行列での2次形式の可視化
# 正定値行列の例
A_positive_definite = np.array([[2, 1], [1, 3]])
visualize_quadratic_form(A_positive_definite, "正定値行列の2次形式")

# 半正定値行列の例
A_positive_semidefinite = np.array([[1, 1], [1, 1]])
visualize_quadratic_form(A_positive_semidefinite, "半正定値行列の2次形式")

# 不定値行列の例
A_indefinite = np.array([[1, 2], [2, -3]])
visualize_quadratic_form(A_indefinite, "不定値行列の2次形式")

# 3. データの分散共分散行列と正定値性
def analyze_covariance_matrix(X):
    # データの中心化
    X_centered = X - np.mean(X, axis=0)
    
    # 分散共分散行列の計算
    n = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / n
    
    print("分散共分散行列:")
    print(cov_matrix)
    
    # 固有値と固有ベクトルの計算
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    print("\n固有値:", eigenvalues)
    
    # 正定値性の判定
    if np.all(eigenvalues > 1e-10):  # 数値誤差を考慮して判定
        print("分散共分散行列は正定値です")
    elif np.all(eigenvalues >= -1e-10):
        print("分散共分散行列は半正定値です")
    else:
        print("分散共分散行列は正定値でも半正定値でもありません")
    
    # 2次形式の可視化
    visualize_quadratic_form(cov_matrix, "分散共分散行列の2次形式")
    
    return cov_matrix, eigenvalues, eigenvectors

# ランダムなデータを生成
np.random.seed(42)
# 2次元の相関のあるデータを生成
X = np.random.multivariate_normal(mean=[0, 0], cov=[[2, 1], [1, 3]], size=100)

# データをプロット
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title("2次元のランダムなデータ")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.grid(True)
plt.show()

# 分散共分散行列の分析
cov_matrix, eigenvalues, eigenvectors = analyze_covariance_matrix(X)

# 主軸の可視化
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)

# データの中心
mean = np.mean(X, axis=0)

# 固有ベクトルをスケーリングして主軸として表示
for i in range(2):
    plt.arrow(mean[0], mean[1], 
              eigenvalues[i] * eigenvectors[0, i], 
              eigenvalues[i] * eigenvectors[1, i],
              head_width=0.1, head_length=0.1, fc='red', ec='red')

plt.title("データと主軸（固有ベクトル*固有値）")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.grid(True)
plt.show()
```

このコードでは、以下のことを行っています：

1. 様々な種類の行列（正定値、半正定値、不定値）の2次形式を計算し、3D曲面と等高線図で可視化
2. 実際のデータから分散共分散行列を計算し、その正定値性を判定
3. 分散共分散行列の固有値・固有ベクトルを計算し、データの主軸として可視化

### 6. データサイエンスにおける2次形式と正定値行列の応用

#### 6.1 多変量正規分布と分散共分散行列

多変量正規分布の確率密度関数は以下のように表されます：

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

ここで、$\boldsymbol{\mu}$ は平均ベクトル、$\Sigma$ は分散共分散行列です。分散共分散行列 $\Sigma$ が正定値であることは、確率分布が適切に定義されるために重要です。

#### 6.2 マハラノビス距離と異常検出

マハラノビス距離は、多変量空間でのデータポイントの距離を測る指標で、次のように定義されます：

$$
d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}
$$

これは、データの共分散構造を考慮した距離尺度であり、異常検出などに利用されます。この式の中心部分 $(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})$ は2次形式であり、$\Sigma^{-1}$ が正定値であることが重要です。

#### 6.3 最適化問題における正定値行列

凸最適化問題では、目的関数のヘッセ行列が正定値であることが、その関数が凸関数である十分条件です。例えば、最小二乗法による回帰分析では、目的関数のヘッセ行列が正定値であれば、一意的な最小値が存在します。

#### 6.4 主成分分析（PCA）と固有値

主成分分析では、データの分散共分散行列の固有値・固有ベクトルを計算します。分散共分散行列は半正定値であり、その固有値は非負です。固有値の大きさは、対応する主成分の重要度（説明される分散の量）を表します。

### 7. 演習問題

#### 基本問題

1. 次の行列が正定値であるか、半正定値であるか、または不定値であるかを判定しなさい。
   
   (a) $A = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$
   
   (b) $B = \begin{pmatrix} 1 & 2 & 0 \\ 2 & 5 & 1 \\ 0 & 1 & 3 \end{pmatrix}$
   
   (c) $C = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix}$
   
   (d) $D = \begin{pmatrix} 2 & -1 \\ -1 & 0 \end{pmatrix}$

2. 行列 $A = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$ について：
   
   (a) 固有値と固有ベクトルを求めなさい。
   
   (b) 2次形式 $Q(x,y) = 3x^2 + 2xy + 2y^2$ を標準形（固有値を用いた形）に変換しなさい。
   
   (c) この2次形式が表す曲面を分類しなさい。

3. 次の2次形式について、対応する対称行列を求め、その固有値から2次形式の性質を判定しなさい。
   
   (a) $Q(x,y,z) = 2x^2 + 3y^2 + 4z^2 + 2xy + 4yz$
   
   (b) $Q(x,y) = x^2 - y^2 + 4xy$

4. シルベスターの判定法を用いて、次の行列が正定値であるかどうかを判定しなさい。
   
   $A = \begin{pmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{pmatrix}$

#### 応用問題

5. 次の分散共分散行列からデータを生成し、視覚化しなさい。また、その分散共分散行列の固有値と固有ベクトルを求め、データの主軸として可視化しなさい。
   
   $\Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$

6. 2次元のデータポイント $(1,2)$, $(2,3)$, $(3,5)$, $(4,6)$, $(5,8)$ について：
   
   (a) 分散共分散行列を計算しなさい。
   
   (b) 計算した分散共分散行列の固有値と固有ベクトルを求めなさい。
   
   (c) 各データポイントを、固有ベクトルを軸とする新しい座標系に変換しなさい。
   
   (d) 元のデータと変換後のデータを可視化し、比較しなさい。

7. 健康データ科学応用問題：心拍数と血圧の関係を調査するために、10人の被験者から測定したデータがあります。各被験者の安静時心拍数（拍/分）と収縮期血圧（mmHg）は以下の通りです：

   | 被験者 | 心拍数 | 収縮期血圧 |
   |-------|--------|-----------|
   | 1     | 62     | 120       |
   | 2     | 65     | 124       |
   | 3     | 68     | 130       |
   | 4     | 70     | 135       |
   | 5     | 71     | 132       |
   | 6     | 72     | 138       |
   | 7     | 74     | 140       |
   | 8     | 76     | 142       |
   | 9     | 78     | 145       |
   | 10    | 80     | 148       |

   (a) このデータの分散共分散行列を計算しなさい。
   
   (b) 分散共分散行列の固有値と固有ベクトルを求めなさい。
   
   (c) マハラノビス距離を用いて、各被験者のデータポイントが異常値かどうかを判定しなさい。（閾値として、マハラノビス距離の平方 > 5.99 を異常値とする）
   
   (d) 健康評価指標として、心拍数と血圧の線形結合で表される新しい変数を考えます。分散を最大化するような線形結合の係数を求め、その指標の意味を考察しなさい。

### 8. よくある質問と解答

#### Q1: 正定値行列と正則行列（可逆行列）は同じですか？
A1: いいえ、正定値行列は必ず正則ですが、逆は必ずしも成り立ちません。例えば、$A = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ は正則ですが、正定値ではありません（固有値の一つが負です）。

#### Q2: 半正定値行列とはどのような行列ですか？
A2: 半正定値行列は、2次形式 $\mathbf{x}^T A \mathbf{x} \geq 0$ が任意のベクトル $\mathbf{x}$ に対して成り立つ対称行列です。例えば、0ではない行列 $B$ に対して $A = B^T B$ とすると、$A$ は半正定値になります。固有値は全て非負です。

#### Q3: 実際のデータ分析で分散共分散行列が正定値でない場合はどうすればいいですか？
A3: これは「多重共線性」の問題として知られています。解決策としては：
1. 問題のある変数を取り除く
2. 正則化手法（リッジ回帰など）を適用する
3. 主成分分析などの次元削減を行う
などがあります。

#### Q4: 2次形式 $Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ が表す幾何学的な形状は何ですか？
A4: 2次元の場合、$A$ の固有値の符号によって以下のように分類されます：
- 両方の固有値が正 → 楕円（正定値）
- 両方の固有値が負 → 楕円（負定値）
- 一方が正、一方が負 → 双曲線（不定値）
- 一方が0、一方が非0 → 放物線（半定値）

#### Q5: 分散共分散行列はなぜ半正定値なのですか？
A5: 任意のベクトル $\mathbf{a}$ に対して、$\mathbf{a}^T \Sigma \mathbf{a}$ は線形結合 $\sum_i a_i X_i$ の分散を表します。分散は常に非負であるため、分散共分散行列は半正定値です。さらに、完全な線形従属関係がない場合は正定値になります。

### 9. 参考文献

1. Gilbert Strang, "Linear Algebra and Its Applications", 4th Edition, 2006.
2. Roger A. Horn and Charles R. Johnson, "Matrix Analysis", 2nd Edition, Cambridge University Press, 2012.
3. Trevor Hastie, Robert Tibshirani, and Jerome Friedman, "The Elements of Statistical Learning", 2nd Edition, Springer, 2009.
4. David A. Harville, "Matrix Algebra From a Statistician's Perspective", Springer, 1997.

この講義ノートは、データサイエンスにおける線形代数の重要な概念である2次形式と正定値行列について詳説しました。次回の講義では、これらの概念を発展させ、主成分分析の理論的基礎について学びます。