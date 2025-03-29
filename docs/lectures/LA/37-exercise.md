# 線形代数学 I / 基礎 / II
## 第37回講義：固有値と固有ベクトルの総合演習

### 1. 講義情報と予習ガイド

**講義回**: 第37回
**関連項目**: 固有値、固有ベクトル、対角化、2次形式、正定値行列
**予習すべき内容**: 
- 第33回「固有値・固有ベクトル」
- 第34回「特性方程式と対角化」
- 第35回「対称行列と直交行列による対角化」
- 第36回「データサイエンスのための2次形式と基礎」

### 2. 学習目標

本講義では以下の能力を身につけることを目標とします：

1. 固有値と固有ベクトルの計算を正確に行えるようになる
2. 行列の対角化を実行し、その性質を理解できるようになる
3. 対角化を利用した行列のべき乗計算や数列の一般項の導出ができるようになる
4. 2次形式と正定値・半正定値行列の性質を理解し判定できるようになる
5. これらの概念のデータサイエンスにおける応用を理解する

### 3. 基本概念の復習

#### 3.1 固有値と固有ベクトル

> **定義**: 正方行列 $A$ に対して、ベクトル $\mathbf{v} \neq \mathbf{0}$ と数 $\lambda$ が
> 
> $$A\mathbf{v} = \lambda\mathbf{v}$$
> 
> を満たすとき、$\lambda$ を行列 $A$ の**固有値**、$\mathbf{v}$ を対応する**固有ベクトル**と呼びます。

**固有値の求め方**:
1. 特性方程式 $\det(A - \lambda I) = 0$ を立てる
2. 特性方程式を解いて固有値 $\lambda$ を求める
3. 各固有値について $A\mathbf{v} = \lambda\mathbf{v}$ を満たす固有ベクトル $\mathbf{v}$ を求める

**重要な性質**:
- $n$ 次正方行列は最大 $n$ 個の固有値を持つ
- 固有値の総和は行列のトレース（対角成分の和）に等しい: $\sum_{i=1}^n \lambda_i = \mathrm{tr}(A)$
- 固有値の積は行列式に等しい: $\prod_{i=1}^n \lambda_i = \det(A)$

#### 3.2 行列の対角化

> **定義**: 正方行列 $A$ が対角化可能であるとは、逆行列が存在する行列 $P$ と対角行列 $D$ が存在して、
> 
> $$P^{-1}AP = D$$
> 
> と表せることを言います。このとき、$D$ の対角成分は $A$ の固有値であり、$P$ の各列ベクトルは対応する固有ベクトルです。

**対角化の手順**:
1. 行列 $A$ の固有値 $\lambda_1, \lambda_2, \ldots, \lambda_n$ を求める
2. 各固有値 $\lambda_i$ に対応する固有ベクトル $\mathbf{v}_i$ を求める
3. 固有ベクトルを列とする行列 $P = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n]$ を作る
4. 対角行列 $D = \mathrm{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ を作る

**対角化可能であるための条件**:
- 固有値に対応する固有ベクトルが $n$ 個の線形独立なベクトルを形成すること
- 特に、異なる固有値に対応する固有ベクトルは必ず線形独立
- 重複固有値を持つ場合、その固有値に対応する固有空間の次元が固有値の重複度と一致すれば対角化可能

#### 3.3 対称行列の対角化

> **定理**: 実対称行列 $A$ （つまり $A^T = A$）は常に対角化可能であり、さらに直交行列 $Q$ （つまり $Q^T Q = QQ^T = I$）によって対角化できます。
> 
> $$Q^T A Q = D$$

**対称行列の性質**:
- すべての固有値は実数
- 異なる固有値に対応する固有ベクトルは互いに直交
- 同じ固有値に対応する固有ベクトルからグラム・シュミット直交化法によって正規直交基底を構成できる

#### 3.4 2次形式と正定値行列

> **定義**: $n$ 次元ベクトル $\mathbf{x}$ と $n \times n$ 実対称行列 $A$ に対して、
> 
> $$f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$$
> 
> の形の関数を**2次形式**と呼びます。

**行列の正定値・半正定値**:
- $A$ が**正定値**であるとは、任意の $\mathbf{x} \neq \mathbf{0}$ に対して $\mathbf{x}^T A \mathbf{x} > 0$ が成り立つこと
- $A$ が**半正定値**であるとは、任意の $\mathbf{x}$ に対して $\mathbf{x}^T A \mathbf{x} \geq 0$ が成り立つこと
- 正定値・半正定値の判定は、行列 $A$ のすべての固有値が正・非負であることと同値

### 4. 理論と手法

#### 4.1 対角化を利用した行列のべき乗計算

行列 $A$ が対角化可能で $P^{-1}AP = D$ であるとき、$A$ の $k$ 乗は以下のように計算できます：

$$A^k = PD^kP^{-1}$$

ここで $D$ は対角行列なので、そのべき乗 $D^k$ は対角成分をそれぞれ $k$ 乗したものになります：

$$D^k = \begin{pmatrix}
\lambda_1^k & 0 & \cdots & 0 \\
0 & \lambda_2^k & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_n^k
\end{pmatrix}$$

これにより、$A^k$ の計算が著しく簡単になります。

#### 4.2 数列の一般項の求め方

漸化式で定義される数列の一般項を行列のべき乗を用いて求める方法を考えます。

線形漸化式 $a_{n+2} = pa_{n+1} + qa_n$ （$p$, $q$ は定数）は以下の行列を用いて表現できます：

$$\begin{pmatrix} a_{n+2} \\ a_{n+1} \end{pmatrix} = \begin{pmatrix} p & q \\ 1 & 0 \end{pmatrix} \begin{pmatrix} a_{n+1} \\ a_n \end{pmatrix}$$

これを $n$ 回繰り返すと：

$$\begin{pmatrix} a_{n+1} \\ a_n \end{pmatrix} = \begin{pmatrix} p & q \\ 1 & 0 \end{pmatrix}^n \begin{pmatrix} a_1 \\ a_0 \end{pmatrix}$$

ここで行列のべき乗を対角化によって計算することで、数列の一般項を求めることができます。

#### 4.3 2次形式の標準形

実対称行列 $A$ が直交行列 $Q$ によって対角化されるとき（$Q^T A Q = D$）、2次形式は以下のように標準形に変換できます：

$$\mathbf{x}^T A \mathbf{x} = \mathbf{y}^T D \mathbf{y} = \sum_{i=1}^n \lambda_i y_i^2$$

ここで $\mathbf{y} = Q^T \mathbf{x}$ です。この標準形を使うと、2次形式の幾何学的意味が明確になります。例えば2次元の場合、楕円、双曲線、放物線などのグラフを表現できます。

### 5. Pythonによる実装と可視化

#### 5.1 固有値と固有ベクトルの計算

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

# 例として対称行列を生成
A = np.array([[4, 1], 
              [1, 3]])

# 固有値と固有ベクトルを計算
eigenvalues, eigenvectors = LA.eig(A)

print("行列A:")
print(A)
print("\n固有値:")
print(eigenvalues)
print("\n固有ベクトル（各列が各固有値に対応）:")
print(eigenvectors)

# 検証: A*v = λ*v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    Av = A @ v
    lv = eigenvalues[i] * v
    print(f"\n固有値 λ_{i+1} = {eigenvalues[i]:.4f} の検証:")
    print(f"A・v_{i+1} = {Av}")
    print(f"λ_{i+1}・v_{i+1} = {lv}")
    print(f"差の絶対値: {np.abs(Av - lv).sum():.10f}")
```

#### 5.2 固有ベクトルの可視化

```python
def plot_eigenvectors(A):
    eigenvalues, eigenvectors = LA.eig(A)
    
    # 原点から始まるベクトルを表示するための準備
    origin = np.zeros(2)
    
    # グリッドの作成
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    
    # 2次形式の値を計算
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = point @ A @ point
    
    plt.figure(figsize=(10, 8))
    
    # 2次形式の等高線をプロット
    contour = plt.contour(X, Y, Z, levels=10, cmap='viridis')
    plt.colorbar(contour, label='2次形式の値')
    
    # 固有ベクトルの表示（長さは固有値に比例）
    for i in range(len(eigenvalues)):
        scaled_vec = eigenvectors[:, i] * np.sqrt(abs(eigenvalues[i]))
        plt.arrow(origin[0], origin[1], scaled_vec[0], scaled_vec[1], 
                 head_width=0.1, head_length=0.2, fc=f'C{i}', ec=f'C{i}',
                 label=f'固有ベクトル {i+1}, 固有値={eigenvalues[i]:.2f}')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title('行列の固有ベクトルと2次形式の等高線')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

# 例として正定値行列を可視化
A_positive = np.array([[2, 0.5], 
                       [0.5, 1]])
plot_eigenvectors(A_positive)

# 例として不定値行列を可視化
A_indefinite = np.array([[1, 2], 
                         [2, -1]])
plot_eigenvectors(A_indefinite)
```

#### 5.3 行列のべき乗計算

```python
def matrix_power_by_diagonalization(A, k):
    """対角化による行列のべき乗計算"""
    eigenvalues, P = LA.eig(A)
    
    # 対角行列D^kを作成
    D_k = np.diag(eigenvalues ** k)
    
    # A^k = P D^k P^(-1)
    A_k = P @ D_k @ LA.inv(P)
    
    return A_k

# 例として
A = np.array([[3, 1], 
              [1, 3]])
k = 10

# 対角化によるA^kの計算
A_k_diag = matrix_power_by_diagonalization(A, k)
print(f"対角化による A^{k}:")
print(A_k_diag)

# 直接計算によるA^kの計算（比較用）
A_k_direct = np.linalg.matrix_power(A, k)
print(f"\n直接計算による A^{k}:")
print(A_k_direct)

# 差の確認
print(f"\n両者の差の絶対値の総和: {np.abs(A_k_diag - A_k_direct).sum():.10f}")
```

#### 5.4 数列の一般項の計算例：フィボナッチ数列

```python
def fibonacci_matrix(n):
    """行列によるフィボナッチ数列のn項目の計算"""
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # フィボナッチ数列の漸化式を表す行列
    A = np.array([[1, 1], 
                  [1, 0]])
    
    # 固有値と固有ベクトルを計算
    eigenvalues, P = LA.eig(A)
    
    # 対角行列D^(n-1)を作成
    D_pow = np.diag(eigenvalues ** (n-1))
    
    # A^(n-1) = P D^(n-1) P^(-1)
    A_pow = P @ D_pow @ LA.inv(P)
    
    # [F_n, F_{n-1}] = A^(n-1) [F_1, F_0]
    result = A_pow @ np.array([1, 0])
    
    return result[0]

# フィボナッチ数列の計算
for n in range(20):
    fib_n = fibonacci_matrix(n)
    print(f"F_{n} = {fib_n:.0f}")
```

#### 5.5 2次形式と正定値行列の判定

```python
def check_matrix_definiteness(A):
    """行列の正定値性、半正定値性などを判定"""
    # 対称化（対称でない場合のため）
    A_sym = (A + A.T) / 2
    
    # 固有値を計算
    eigenvalues = LA.eigvals(A_sym)
    
    print(f"行列:\n{A}")
    print(f"固有値: {eigenvalues}")
    
    if np.all(eigenvalues > 0):
        print("判定結果: 正定値行列")
    elif np.all(eigenvalues >= 0):
        print("判定結果: 半正定値行列")
    elif np.all(eigenvalues < 0):
        print("判定結果: 負定値行列")
    elif np.all(eigenvalues <= 0):
        print("判定結果: 半負定値行列")
    else:
        print("判定結果: 不定値行列")
    
    return eigenvalues

# いくつかの行列で試す
matrices = [
    np.array([[2, 0], [0, 3]]),                  # 正定値
    np.array([[1, 0], [0, 0]]),                  # 半正定値
    np.array([[-2, 0], [0, -3]]),                # 負定値
    np.array([[0, 0], [0, -1]]),                 # 半負定値
    np.array([[1, 0], [0, -1]])                  # 不定値
]

for i, A in enumerate(matrices):
    print(f"\n例 {i+1}:")
    check_matrix_definiteness(A)
```

### 6. 演習問題

#### 基本問題

**問題1**: 以下の行列の固有値と固有ベクトルを求めよ。
$$A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$$

**問題2**: 以下の行列が対角化可能かどうかを判定し、可能ならば対角化せよ。
$$B = \begin{pmatrix} 2 & 1 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{pmatrix}$$

**問題3**: 以下の行列の4乗 $C^4$ を対角化を利用して求めよ。
$$C = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

**問題4**: 以下の2次形式が正定値であるか、半正定値であるか、不定値であるかを判定せよ。
$$f(x, y) = 2x^2 + 2xy + 2y^2$$

**問題5**: 漸化式 $a_{n+2} = 5a_{n+1} - 6a_n$ について、初期値 $a_0 = 1$, $a_1 = 2$ が与えられたとき、$a_{10}$ の値を対角化を利用して求めよ。

#### 応用問題

**問題6**: 以下の行列 $D$ は特異（$\det(D) = 0$）であり、ランクが2である。その2つの非ゼロ固有値と対応する固有ベクトルを求め、特異値分解の概念を使って $D$ を分解せよ。
$$D = \begin{pmatrix} 1 & 2 & 2 \\ 2 & 1 & -1 \\ 2 & -1 & 1 \end{pmatrix}$$

**問題7**: 以下の漸化式で定義される数列 $\{a_n\}$ の一般項を求めよ。
$$a_{n+3} = 3a_{n+2} - 3a_{n+1} + a_n, \quad a_0 = 1, a_1 = 2, a_2 = 3$$

**問題8**: 主成分分析において、データの分散共分散行列の固有値と固有ベクトルが果たす役割について説明せよ。特に、第一主成分と固有ベクトルの関係を述べよ。

**問題9 (健康データサイエンス応用)**: 
ある健康診断データでは、患者の身長、体重、血圧、血糖値、コレステロールの5つの指標を測定している。これらのデータを標準化した後の分散共分散行列の固有値が [2.5, 1.2, 0.8, 0.3, 0.2] だったとする。主成分分析を適用する場合、何個の主成分を採用するべきか、その理由と共に述べよ。また、固有値から計算される累積寄与率を求めよ。

### 7. よくある質問と解答

**Q1: 行列の固有値・固有ベクトルが重複する場合、どのように対角化すればよいですか？**

A1: 固有値が重複する場合（多重固有値）、その固有空間の次元が固有値の重複度と一致すれば対角化可能です。具体的には、各固有値 $\lambda_i$ に対して、$(A - \lambda_i I)\mathbf{v} = \mathbf{0}$ の一般解を求め、線形独立なベクトルを重複度分だけ見つける必要があります。これらのベクトルを対角化行列 $P$ の列として使用します。

**Q2: 対称行列でない行列も対角化できますか？**

A2: はい、対角化の条件は「固有ベクトルが基底を形成すること」です。対称行列は必ず対角化可能ですが、非対称行列でも条件を満たせば対角化できます。ただし、複素固有値が現れる場合は、複素数の計算が必要になります。また、対称行列と違って直交行列による対角化ができない場合もあります。

**Q3: 対角化を使って行列のべき乗を計算する利点は何ですか？**

A3: 通常の行列のべき乗計算は計算量が $O(n^3 \cdot k)$ （$n$ は行列の次元、$k$ はべき数）ですが、対角化を使えば $O(n^3 + n \cdot k)$ に削減できます。特に $k$ が大きい場合、計算効率が格段に向上します。また、数列の一般項や漸化式の解を求める際にも非常に有用です。

**Q4: データサイエンスで固有値と固有ベクトルはどのように使われますか？**

A4: データサイエンスでは、固有値と固有ベクトルは以下のような場面で使われます：
- 主成分分析（PCA）: データの分散を最大化する方向（固有ベクトル）を見つける
- スペクトラルクラスタリング: グラフラプラシアンの固有ベクトルを用いたクラスタリング
- 推薦システム: 行列分解による潜在因子の抽出
- 画像処理: 特徴抽出や圧縮（固有顔など）
- 自然言語処理: 単語ベクトルの次元削減や潜在意味解析

**Q5: 正定値行列の応用例にはどのようなものがありますか？**

A5: 正定値行列は以下のような応用があります：
- 機械学習における損失関数の凸性保証
- 最適化問題の制約条件
- 統計学での分散共分散行列（常に半正定値）
- カーネル法におけるグラム行列（半正定値）
- 物理学での安定性解析

### 8. 課題の解答例

**問題1の解答**: 
行列 $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$ の特性方程式は：

$$\det(A - \lambda I) = \begin{vmatrix} 3-\lambda & 1 \\ 1 & 3-\lambda \end{vmatrix} = (3-\lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = 0$$

これを解くと、$\lambda = 2, 4$ が固有値として得られます。

固有値 $\lambda = 2$ に対する固有ベクトルは：
$$(A - 2I)\mathbf{v} = \mathbf{0} \Rightarrow \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

これから $v_1 + v_2 = 0$ となるので、例えば $\mathbf{v}_1 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$ が固有ベクトルとなります。

同様に、固有値 $\lambda = 4$ に対する固有ベクトルは：
$$(A - 4I)\mathbf{v} = \mathbf{0} \Rightarrow \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

これから $v_1 = v_2$ となるので、例えば $\mathbf{v}_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ が固有ベクトルとなります。

**問題3の解答**: 
行列 $C = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ の固有値・固有ベクトルを求めます。

特性方程式：$\det(C - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = 0$

解くと $\lambda = 1, 3$ が固有値として得られます。

固有ベクトルを求めると、$\lambda = 1$ に対して $\mathbf{v}_1 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$、$\lambda = 3$ に対して $\mathbf{v}_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ が得られます。

対角化行列は $P = \begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$ となり、
$P^{-1} = \frac{1}{2}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$ です。

よって、$C^4 = P D^4 P^{-1} = P \begin{pmatrix} 1^4 & 0 \\ 0 & 3^4 \end{pmatrix} P^{-1} = P \begin{pmatrix} 1 & 0 \\ 0 & 81 \end{pmatrix} P^{-1}$

計算すると、$C^4 = \begin{pmatrix} 41 & 40 \\ 40 & 41 \end{pmatrix}$ となります。

**問題4の解答**: 
2次形式 $f(x, y) = 2x^2 + 2xy + 2y^2$ に対応する対称行列は
$$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

この行列の固有値を求めるため、特性方程式を立てます：
$$\det(A - \lambda I) = \begin{vmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{vmatrix} = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = 0$$

これを解くと、$\lambda = 1, 3$ が固有値として得られます。

両方の固有値が正であるため、行列 $A$ は正定値であり、2次形式 $f(x, y)$ も正定値です。これは幾何学的には、原点を中心とする楕円を表しています。

**問題5の解答**: 
漸化式 $a_{n+2} = 5a_{n+1} - 6a_n$ を行列形式で表現します：
$$\begin{pmatrix} a_{n+2} \\ a_{n+1} \end{pmatrix} = \begin{pmatrix} 5 & -6 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} a_{n+1} \\ a_n \end{pmatrix}$$

行列 $A = \begin{pmatrix} 5 & -6 \\ 1 & 0 \end{pmatrix}$ の固有値と固有ベクトルを求めます。

特性方程式：$\det(A - \lambda I) = (5-\lambda)(0-\lambda) - (-6)(1) = \lambda^2 - 5\lambda + 6 = 0$

解くと $\lambda = 2, 3$ が固有値として得られます。

固有値 $\lambda = 2$ に対する固有ベクトルは：
$$(A - 2I)\mathbf{v} = \mathbf{0} \Rightarrow \begin{pmatrix} 3 & -6 \\ 1 & -2 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

これから $3v_1 - 6v_2 = 0$ となるので、$v_1 = 2v_2$ です。例えば $\mathbf{v}_1 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$ が固有ベクトルとなります。

固有値 $\lambda = 3$ に対しても同様に計算すると、固有ベクトル $\mathbf{v}_2 = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$ が得られます。

これらから対角化行列 $P = \begin{pmatrix} 2 & 3 \\ 1 & 1 \end{pmatrix}$ が得られます。
また、$P^{-1} = \begin{pmatrix} -1 & 3 \\ 1 & -2 \end{pmatrix}$ です。

初期値から始めて $n$ 回の遷移を考えると：
$$\begin{pmatrix} a_{n+1} \\ a_n \end{pmatrix} = A^n \begin{pmatrix} a_1 \\ a_0 \end{pmatrix} = P D^n P^{-1} \begin{pmatrix} a_1 \\ a_0 \end{pmatrix} = P \begin{pmatrix} 2^n & 0 \\ 0 & 3^n \end{pmatrix} P^{-1} \begin{pmatrix} 2 \\ 1 \end{pmatrix}$$

計算すると：
$$\begin{pmatrix} a_{n+1} \\ a_n \end{pmatrix} = P \begin{pmatrix} 2^n & 0 \\ 0 & 3^n \end{pmatrix} \begin{pmatrix} -1 \cdot 2 + 3 \cdot 1 \\ 1 \cdot 2 - 2 \cdot 1 \end{pmatrix} = P \begin{pmatrix} 2^n & 0 \\ 0 & 3^n \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = P \begin{pmatrix} 2^n \\ 0 \end{pmatrix} = \begin{pmatrix} 2 \cdot 2^n \\ 1 \cdot 2^n \end{pmatrix}$$

よって、$a_n = 2^n$ となります。特に $a_{10} = 2^{10} = 1024$ です。