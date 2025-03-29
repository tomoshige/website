# 線形代数学 I / データサイエンス基礎 講義ノート

## 第34回：特性方程式と対角化

### 講義情報
- **講義回**: 第34回
- **関連項目**: 固有値と固有ベクトル
- **予習内容**: 固有値と固有ベクトルの定義、線形変換の基礎

### 学習目標
1. 行列の対角化の定義と意味を理解する
2. 特性方程式を解いて固有値と固有ベクトルを求める方法を習得する
3. 行列が対角化可能であるための条件を理解する
4. 対角化の計算手順を実行できるようになる
5. 対角化のデータサイエンスにおける応用意義を理解する

## 1. 基本概念：行列の対角化

### 1.1 対角化の定義

> **定義**: $n \times n$行列$A$が対角化可能であるとは、可逆行列$P$と対角行列$D$が存在して、$P^{-1}AP = D$と表せることをいう。

対角化とは、行列$A$に対して、適切な基底変換を行うことで、より単純な形（対角行列）に変形することです。対角行列は対角成分以外がすべて0である行列で、計算や性質の分析が非常に簡単になります。

### 1.2 対角化の意味

対角化は以下のような意味を持ちます：

1. **線形変換の単純化**: 行列$A$を線形変換として見たとき、適切な基底を選ぶことで、各基底ベクトルはその方向にだけ伸縮される（回転を伴わない）変換になる
2. **計算の簡略化**: 行列の冪乗$A^k$や関数$f(A)$の計算が容易になる
3. **システムの解析**: 動的システムや微分方程式の解析が簡単になる

### 1.3 対角行列の性質

対角行列$D = \text{diag}(d_1, d_2, \ldots, d_n)$は以下の性質を持ちます：

1. $D^k = \text{diag}(d_1^k, d_2^k, \ldots, d_n^k)$
2. $\det(D) = d_1 \cdot d_2 \cdot \ldots \cdot d_n$
3. $\text{tr}(D) = d_1 + d_2 + \ldots + d_n$

## 2. 理論と手法：特性方程式と固有値

### 2.1 特性方程式

> **定義**: $n \times n$行列$A$の特性方程式（固有方程式）とは、$\det(A - \lambda I) = 0$のことである。

この方程式の解$\lambda$が行列$A$の固有値となります。

特性多項式$p_A(\lambda) = \det(A - \lambda I)$は$\lambda$に関する$n$次多項式になり、最大で$n$個の固有値を持ちます（重複も含む）。

### 2.2 固有値と固有ベクトルの計算手順

1. **特性方程式を立てる**: $\det(A - \lambda I) = 0$
2. **特性方程式を解く**: 多項式の根として固有値$\lambda_1, \lambda_2, \ldots, \lambda_n$を求める
3. **各固有値に対する固有空間を求める**: 各$\lambda_i$に対して、$(A - \lambda_i I)x = 0$を満たす非零ベクトル$x$を求める
4. **固有ベクトルの基底を構成する**: 各固有空間の基底ベクトルを求める

### 2.3 固有値の代数的重複度と幾何的重複度

- **代数的重複度**: 特性多項式における固有値の重複度
- **幾何的重複度**: 対応する固有空間の次元

例えば、$\lambda = 3$が特性多項式の二重根（代数的重複度2）であるとき、対応する固有空間の次元（幾何的重複度）は1または2になりえます。

## 3. 対角化の理論

### 3.1 対角化可能条件

> **定理**: $n \times n$行列$A$が対角化可能であるための必要十分条件は、$A$の固有ベクトルで$n$次元空間の基底が構成できることである。

これは次の条件と同値です：

1. $A$のすべての固有値に対して、代数的重複度と幾何的重複度が等しい
2. $A$の固有ベクトルが$n$個線形独立に存在する

### 3.2 対角化の手順

1. $A$の固有値$\lambda_1, \lambda_2, \ldots, \lambda_n$を求める
2. 各固有値$\lambda_i$に対して、固有ベクトル$v_i$を求める
3. 固有ベクトルを列に並べた行列$P = [v_1, v_2, \ldots, v_n]$を構成する
4. 対角行列$D = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$を構成する
5. $P^{-1}AP = D$を確認する

### 3.3 対角化が可能でない場合

以下の場合は対角化できません：

1. 固有ベクトルの数が足りない（線形独立な固有ベクトルが$n$個ない）
2. 複素固有値を持つ実行列（この場合、実行列による対角化はできないが、複素行列による対角化は可能）

## 4. 具体例による対角化の計算

### 例題1：2×2行列の対角化

行列 $A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$ を対角化しましょう。

**Step 1**: 特性方程式を立てる
$\det(A - \lambda I) = \det\begin{bmatrix} 3-\lambda & 1 \\ 1 & 3-\lambda \end{bmatrix} = (3-\lambda)^2 - 1 = (3-\lambda)^2 - 1 = 0$

**Step 2**: 特性方程式を解く
$(3-\lambda)^2 = 1$
$3-\lambda = \pm 1$
$\lambda = 3 \pm 1 = 2, 4$

従って、固有値は $\lambda_1 = 2$ と $\lambda_2 = 4$ です。

**Step 3**: 各固有値に対応する固有ベクトルを求める

$\lambda_1 = 2$ の場合：
$(A - 2I)v = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$

これは $v_1 + v_2 = 0$ を意味します。従って、$v_1 = -v_2$ となり、例えば $v_1 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$ が固有ベクトルとなります。

$\lambda_2 = 4$ の場合：
$(A - 4I)v = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$

これは $-v_1 + v_2 = 0$ を意味します。従って、$v_1 = v_2$ となり、例えば $v_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ が固有ベクトルとなります。

**Step 4**: 対角化行列 $P$ と対角行列 $D$ を構成する

$P = \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}$, $D = \begin{bmatrix} 2 & 0 \\ 0 & 4 \end{bmatrix}$

**Step 5**: 検証

$P^{-1}AP = D$ を確認します。

$P^{-1} = \frac{1}{\det(P)}\begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix} = \frac{1}{2}\begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}$

$P^{-1}AP = \frac{1}{2}\begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 4 \end{bmatrix} = D$

以上により、行列 $A$ は対角化され、$P^{-1}AP = D$ が成り立つことが確認できました。

### 例題2：3×3行列の対角化

行列 $A = \begin{bmatrix} 4 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 0 & 4 \end{bmatrix}$ を対角化しましょう。

**Step 1**: 特性方程式を立てる
$\det(A - \lambda I) = \det\begin{bmatrix} 4-\lambda & 0 & 1 \\ 0 & 1-\lambda & 0 \\ 1 & 0 & 4-\lambda \end{bmatrix}$

これを展開すると：
$= (1-\lambda)[(4-\lambda)(4-\lambda) - 1]$
$= (1-\lambda)[(4-\lambda)^2 - 1]$
$= (1-\lambda)[(16 - 8\lambda + \lambda^2) - 1]$
$= (1-\lambda)(15 - 8\lambda + \lambda^2)$

**Step 2**: 特性方程式を解く
$(1-\lambda)(15 - 8\lambda + \lambda^2) = 0$

従って、$\lambda_1 = 1$ と $(15 - 8\lambda + \lambda^2) = 0$ が得られます。

二次方程式 $\lambda^2 - 8\lambda + 15 = 0$ を解くと：
$\lambda = \frac{8 \pm \sqrt{64 - 60}}{2} = \frac{8 \pm 2}{2} = 4 \pm 1$

よって、固有値は $\lambda_1 = 1$, $\lambda_2 = 3$, $\lambda_3 = 5$ です。

**Step 3**: 各固有値に対応する固有ベクトルを求める

$\lambda_1 = 1$ の場合：
$(A - I)v = \begin{bmatrix} 3 & 0 & 1 \\ 0 & 0 & 0 \\ 1 & 0 & 3 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$

これは $3v_1 + v_3 = 0$ および $v_1 + 3v_3 = 0$ を意味しますが、これらは同じ条件です。また、$v_2$ は自由に選べます。
従って、$v_1 = -v_3$, $v_2$ は任意となり、例えば $v_1 = \begin{bmatrix} -1 \\ 1 \\ 1 \end{bmatrix}$ が固有ベクトルとなります。

$\lambda_2 = 3$ の場合：
$(A - 3I)v = \begin{bmatrix} 1 & 0 & 1 \\ 0 & -2 & 0 \\ 1 & 0 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$

これより、$v_1 + v_3 = 0$, $-2v_2 = 0$, $v_1 + v_3 = 0$ となります。従って、$v_2 = 0$, $v_1 = -v_3$ となり、例えば $v_2 = \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}$ が固有ベクトルとなります。

$\lambda_3 = 5$ の場合：
$(A - 5I)v = \begin{bmatrix} -1 & 0 & 1 \\ 0 & -4 & 0 \\ 1 & 0 & -1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$

これより、$-v_1 + v_3 = 0$, $-4v_2 = 0$, $v_1 - v_3 = 0$ となります。従って、$v_2 = 0$, $v_1 = v_3$ となり、例えば $v_3 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$ が固有ベクトルとなります。

**Step 4**: 対角化行列 $P$ と対角行列 $D$ を構成する

$P = \begin{bmatrix} -1 & 1 & 1 \\ 1 & 0 & 0 \\ 1 & -1 & 1 \end{bmatrix}$, $D = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 5 \end{bmatrix}$

**Step 5**: 検証

$P^{-1}AP = D$ を確認します（計算は省略）。

## 5. 対角化の応用

### 5.1 行列の冪乗計算

対角化された行列 $A = PDP^{-1}$ に対して、$A^k = PD^kP^{-1}$ が成り立ちます。対角行列の冪乗計算は各対角成分を冪乗するだけなので、複雑な行列の冪乗計算が簡単になります。

**例**: $A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$ の10乗を計算する。

先ほどの例題1より、$A = PDP^{-1}$ で $P = \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}$, $D = \begin{bmatrix} 2 & 0 \\ 0 & 4 \end{bmatrix}$ です。

$A^{10} = PD^{10}P^{-1} = P\begin{bmatrix} 2^{10} & 0 \\ 0 & 4^{10} \end{bmatrix}P^{-1}$
$= P\begin{bmatrix} 1024 & 0 \\ 0 & 1048576 \end{bmatrix}P^{-1}$

$P^{-1} = \frac{1}{2}\begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}$

計算すると、$A^{10} = \begin{bmatrix} 524800 & 523776 \\ 523776 & 524800 \end{bmatrix}$ が得られます。

### 5.2 線形漸化式の一般項

$a_{n+1} = c_1a_n + c_2a_{n-1} + \ldots + c_ka_{n-k+1}$ という線形漸化式は、適切な行列 $A$ を定義することで、$\vec{x}_{n+1} = A\vec{x}_n$ という形に書き換えられます。対角化を用いることで、一般項を簡単に求めることができます。

### 5.3 微分方程式の解法

微分方程式系 $\frac{d\vec{x}}{dt} = A\vec{x}$ の解は、$A$ が対角化可能なら $\vec{x}(t) = Pe^{Dt}P^{-1}\vec{x}(0)$ と表せます。ここで $e^{Dt} = \text{diag}(e^{\lambda_1 t}, e^{\lambda_2 t}, \ldots, e^{\lambda_n t})$ です。

## 6. Pythonによる実装と可視化

以下に、対角化の計算とその可視化をPythonで実装した例を示します。

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv

# 行列の定義
A = np.array([[3, 1], [1, 3]])

# 固有値と固有ベクトルを計算
eigenvalues, eigenvectors = eig(A)

print("固有値:")
print(eigenvalues)

print("\n固有ベクトル（列ベクトルとして表示）:")
print(eigenvectors)

# 対角化の検証
P = eigenvectors
D = np.diag(eigenvalues)
P_inv = inv(P)

# P^(-1) * A * P = D を検証
result = P_inv @ A @ P
print("\n検証: P^(-1) * A * P")
print(result)
print("これは対角行列 D に近似的に等しい")

# 対角化を用いた行列の冪乗計算
def matrix_power_diag(A, power):
    eigenvalues, eigenvectors = eig(A)
    P = eigenvectors
    D_power = np.diag(eigenvalues ** power)
    P_inv = inv(P)
    return P @ D_power @ P_inv

# A^10 を計算
A_power_10 = matrix_power_diag(A, 10)
print("\nA^10 (対角化を使用):")
print(A_power_10)

# 直接計算で検証
A_power_10_direct = np.linalg.matrix_power(A, 10)
print("\nA^10 (直接計算):")
print(A_power_10_direct)

# 対角化の幾何学的解釈を可視化
def plot_transformation(A):
    # 単位円上の点を生成
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    circle_points = np.vstack((circle_x, circle_y))
    
    # 行列変換を適用
    transformed_points = A @ circle_points
    
    # プロット
    plt.figure(figsize=(10, 5))
    
    # 元の単位円
    plt.subplot(1, 2, 1)
    plt.plot(circle_x, circle_y, 'b-')
    plt.grid(True)
    plt.axis('equal')
    plt.title('Original Unit Circle')
    
    # 変換後
    plt.subplot(1, 2, 2)
    plt.plot(transformed_points[0], transformed_points[1], 'r-')
    plt.grid(True)
    plt.axis('equal')
    plt.title('Transformed by Matrix A')
    
    # 固有ベクトルも描画
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        transformed_v = A @ v
        plt.subplot(1, 2, 1)
        plt.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.1, fc='green', ec='green')
        plt.subplot(1, 2, 2)
        plt.arrow(0, 0, transformed_v[0], transformed_v[1], head_width=0.05, head_length=0.1, fc='green', ec='green')
    
    plt.tight_layout()
    plt.show()

# 変換を可視化
plot_transformation(A)
```

## 7. 演習問題

### 基本問題

1. 次の行列を対角化せよ。
   a) $\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$
   b) $\begin{bmatrix} 1 & 1 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix}$
   c) $\begin{bmatrix} 2 & 0 & 0 \\ 1 & 2 & 0 \\ 0 & 1 & 2 \end{bmatrix}$

2. 次の行列が対角化可能かどうかを判定し、可能ならば対角化せよ。
   a) $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$
   b) $\begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{bmatrix}$

3. 行列 $A = \begin{bmatrix} 1 & 2 \\ 4 & 3 \end{bmatrix}$ を対角化し、$A^{20}$ を求めよ。

4. 対角化を用いて、漸化式 $a_{n+2} = 6a_{n+1} - 9a_n$ ($a_0 = 1$, $a_1 = 3$) の一般項を求めよ。

### 応用問題

5. 正方行列 $A$ が対角化可能であるとき、線形写像 $f(x) = e^{At}x$ は何を表しているか考察せよ。また、$A$ が以下の行列の場合、$t=1$ での写像を可視化せよ。
   $A = \begin{bmatrix} -1 & 1 \\ -1 & -1 \end{bmatrix}$

6. 3×3の対称行列 $A = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{bmatrix}$ について、対角化を行い、その幾何学的意味を考察せよ。具体的には、この行列が表す二次形式 $q(x,y,z) = x^TAx$ の形を求めよ。

7. **健康データ分析の応用問題**：患者の血圧データからなるデータセット $X$ について、共分散行列 $\Sigma = X^TX$ を考える。この共分散行列を対角化することで、データの主要な変動方向を求める方法を説明せよ。また、共分散行列の固有値と固有ベクトルが持つ意味を解釈せよ。

## 8. よくある質問と解答

### Q1: 行列が対角化可能かどうかを簡単に判定する方法はありますか？

**A1**: 一般的には固有値と固有空間の次元を調べる必要がありますが、以下のケースでは判定が簡単になります：
- 対称行列は常に実対角化可能
- 固有値がすべて異なる行列は対角化可能
- 三角行列の場合、対角成分が固有値になるので、対角成分に重複がなければ対角化可能
- 冪零行列（$A^n = 0$ となるような行列）は一般に対角化不可能

### Q2: 固有値がすべて同じ値になる行列は対角化可能ですか？

**A2**: 固有値がすべて同じ値 $\lambda$ になる行列は、以下の2つの可能性があります：
1. その行列が $\lambda I$ の形（スカラー行列）の場合：既に対角形式なので対角化可能
2. $\lambda I$ でない場合：一般に対角化不可能（ヨルダン標準形が必要）

例えば $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ は固有値が両方とも1ですが対角化できません。

### Q3: 実行列で複素固有値がある場合はどうすればよいですか？

**A3**: 実行列で複素固有値が現れる場合、複素固有値は共役ペアで現れます。この場合：
1. 複素数体上では通常通り対角化できます
2. 実数体上では「実ヨルダン標準形」という別の形に変換できます（ブロック対角形式）
3. 実対称行列の場合は常に実固有値のみを持つので、この問題は発生しません

### Q4: 対角化と主成分分析（PCA）の関係は何ですか？

**A4**: 主成分分析（PCA）では、データの共分散行列を対角化します。共分散行列は対称行列なので常に対角化可能です。対角化した際の：
- 固有値：主成分の分散（重要度）を表す
- 固有ベクトル：主成分の方向（新しい座標軸）を表す

PCAは本質的に、データの変動を最もよく表現する新しい座標系を見つける手法です。

### Q5: 対角化できない行列はどのように扱いますか？

**A5**: 対角化できない行列に対しては、より一般的なヨルダン標準形を用います。または、特定の用途に応じて：
1. 三角行列化（シュール分解）
2. 特異値分解（SVD）
3. 実ヨルダン標準形（実行列の場合）
などの方法が使えます。

## 9. 参考文献と追加リソース

1. Gilbert Strang. "Linear Algebra and Its Applications"
2. David C. Lay. "Linear Algebra and Its Applications"
3. Python NumPy公式ドキュメント: [https://numpy.org/doc/stable/reference/routines.linalg.html](https://numpy.org/doc/stable/reference/routines.linalg.html)
4. Khan Academy線形代数講座: [https://www.khanacademy.org/math/linear-algebra](https://www.khanacademy.org/math/linear-algebra)

次回の授業では、対称行列の直交対角化について学びます。対称行列は常に対角化可能であり、さらに直交行列によって対角化できる特別な性質を持っています。これはデータサイエンスにおいて非常に重要な性質で、特に主成分分析（PCA）の基礎となります。