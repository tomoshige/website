# 線形代数学 I 講義ノート - 第9回：総合演習

## 1. 講義情報と予習ガイド

- **講義回**: 第9回
- **講義内容**: ベクトルと行列の基本概念の総合演習
- **関連項目**: ベクトル演算、行列演算、内積、データの統計量
- **予習内容**: 第1回～第8回の講義内容を復習しておくこと

## 2. 学習目標

本講義の終了時には、以下のことができるようになることを目指します：

1. ベクトルの基本操作（和、スカラー倍、内積、ノルム）を正確に理解し計算できる
2. 行列の基本操作（和、スカラー倍、積、転置）を正確に理解し計算できる
3. 行列とベクトルを用いてデータの統計量（平均、分散、共分散、相関係数）を計算できる
4. データサイエンスにおける線形代数の基礎概念を実データに適用できる
5. 線形代数の視点からデータの構造を理解し解釈できる

## 3. 基本概念の復習

### 3.1 ベクトルの基本概念

> **定義: ベクトル**
> 
> n次元の実ベクトルは、n個の実数を縦に並べたものであり、以下のように表記される：
> 
> $$\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}$$
> 
> ここで、$x_i$は実数である。

#### ベクトルの基本演算

1. **和**: $\mathbf{x} + \mathbf{y} = \begin{pmatrix} x_1 + y_1 \\ x_2 + y_2 \\ \vdots \\ x_n + y_n \end{pmatrix}$

2. **スカラー倍**: $\alpha \mathbf{x} = \begin{pmatrix} \alpha x_1 \\ \alpha x_2 \\ \vdots \\ \alpha x_n \end{pmatrix}$

3. **内積**: $\mathbf{x} \cdot \mathbf{y} = x_1y_1 + x_2y_2 + \cdots + x_ny_n = \sum_{i=1}^n x_i y_i$

4. **ノルム**: $\|\mathbf{x}\| = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2} = \sqrt{\mathbf{x} \cdot \mathbf{x}}$

### 3.2 行列の基本概念

> **定義: 行列**
> 
> m行n列の実行列は、mn個の実数を長方形状に配置したものであり、以下のように表記される：
> 
> $$A = \begin{pmatrix} 
> a_{11} & a_{12} & \cdots & a_{1n} \\
> a_{21} & a_{22} & \cdots & a_{2n} \\
> \vdots & \vdots & \ddots & \vdots \\
> a_{m1} & a_{m2} & \cdots & a_{mn}
> \end{pmatrix}$$
> 
> ここで、$a_{ij}$は実数である。

#### 行列の基本演算

1. **和**: $(A + B)_{ij} = a_{ij} + b_{ij}$

2. **スカラー倍**: $(\alpha A)_{ij} = \alpha a_{ij}$

3. **積**: $(AB)_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$

4. **転置**: $(A^T)_{ij} = a_{ji}$

### 3.3 特殊な行列とその性質

1. **単位行列**:
   $$I_n = \begin{pmatrix}
   1 & 0 & \cdots & 0 \\
   0 & 1 & \cdots & 0 \\
   \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & \cdots & 1
   \end{pmatrix}$$
   
   性質: 任意の行列 $A$ に対して $AI = IA = A$

2. **転置行列**:
   $A^T$ は $A$ の転置

   性質: 
   - $(A^T)^T = A$
   - $(A + B)^T = A^T + B^T$
   - $(AB)^T = B^T A^T$

3. **対称行列**:
   $A = A^T$ を満たす行列

   性質:
   - 対角要素以外の要素が対称的に配置される
   - 固有値が実数になる

### 3.4 データと行列・ベクトルの関係

データ行列 $X$ が $n$ 個のサンプルと $p$ 個の変数を持つ場合：

$$X = \begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{pmatrix}$$

ここで、$x_{ij}$ はサンプル $i$ の変数 $j$ の値を表す。

## 4. データの統計計算とベクトル・行列演算

### 4.1 平均ベクトル

n個のサンプルのp次元データの平均ベクトルは以下のように計算される：

$$\bar{\mathbf{x}} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i = \frac{1}{n} \begin{pmatrix} \sum_{i=1}^n x_{i1} \\ \sum_{i=1}^n x_{i2} \\ \vdots \\ \sum_{i=1}^n x_{ip} \end{pmatrix}$$

行列表記では：

$$\bar{\mathbf{x}} = \frac{1}{n}X^T\mathbf{1}_n$$

ここで、$\mathbf{1}_n$ は全ての要素が1の$n$次元ベクトルである。

### 4.2 分散・共分散行列

データの分散・共分散行列は以下のように計算される：

$$S = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$$

行列表記では：

$$S = \frac{1}{n}(X - \mathbf{1}_n\bar{\mathbf{x}}^T)^T(X - \mathbf{1}_n\bar{\mathbf{x}}^T)$$

ここで、$S$ の対角要素 $s_{jj}$ は変数 $j$ の分散、非対角要素 $s_{jk}$ は変数 $j$ と変数 $k$ の共分散を表す。

### 4.3 相関係数行列

相関係数行列 $R$ は以下のように計算される：

$$r_{jk} = \frac{s_{jk}}{\sqrt{s_{jj}s_{kk}}}$$

ここで、$r_{jk}$ は変数 $j$ と変数 $k$ の相関係数を表す。

行列表記では：

$$R = D^{-1/2}SD^{-1/2}$$

ここで、$D$ は $S$ の対角要素を対角に持つ対角行列である。

## 5. Pythonによる実装と可視化

### 5.1 ベクトルと行列の基本操作

```python
import numpy as np
import matplotlib.pyplot as plt

# ベクトルの定義
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# ベクトルの和
z = x + y
print("ベクトルの和:", z)  # [5 7 9]

# スカラー倍
alpha = 2
scaled_x = alpha * x
print("スカラー倍:", scaled_x)  # [2 4 6]

# 内積
dot_product = np.dot(x, y)
print("内積:", dot_product)  # 32

# ノルム
norm_x = np.linalg.norm(x)
print("ノルム:", norm_x)  # 3.7416573867739413

# 行列の定義
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 行列の和
C = A + B
print("行列の和:\n", C)  # [[6 8], [10 12]]

# 行列のスカラー倍
scaled_A = 2 * A
print("行列のスカラー倍:\n", scaled_A)  # [[2 4], [6 8]]

# 行列の積
D = np.matmul(A, B)
print("行列の積:\n", D)  # [[19 22], [43 50]]

# 行列の転置
A_transpose = A.T
print("行列の転置:\n", A_transpose)  # [[1 3], [2 4]]
```

### 5.2 データの統計計算

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# サンプルデータの生成（身長と体重のデータを想定）
np.random.seed(42)
height = 170 + 10 * np.random.randn(100)  # 平均170cm、標準偏差10cmの身長
weight = 60 + 0.6 * (height - 170) + 5 * np.random.randn(100)  # 身長と相関のある体重

# データ行列の作成
X = np.column_stack((height, weight))
print("データ行列の形状:", X.shape)  # (100, 2)

# 平均ベクトルの計算
mean_vector = np.mean(X, axis=0)
print("平均ベクトル:", mean_vector)

# 分散共分散行列の計算
cov_matrix = np.cov(X, rowvar=False)
print("分散共分散行列:\n", cov_matrix)

# 相関係数行列の計算
corr_matrix = np.corrcoef(X, rowvar=False)
print("相関係数行列:\n", corr_matrix)

# データの可視化
plt.figure(figsize=(10, 6))

# 散布図
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('身長 (cm)')
plt.ylabel('体重 (kg)')
plt.title('身長と体重の散布図')
plt.grid(True)
plt.show()
```

## 6. 演習問題

### 6.1 基本問題

**問題1**: 以下のベクトルについて、ベクトルの和、差、内積、ノルムを計算せよ。
$$\mathbf{a} = \begin{pmatrix} 3 \\ 1 \\ 4 \end{pmatrix}, \mathbf{b} = \begin{pmatrix} 2 \\ 5 \\ 0 \end{pmatrix}$$

**問題2**: 以下の行列の積を計算せよ。
$$A = \begin{pmatrix} 2 & 1 \\ 3 & 4 \end{pmatrix}, B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$$

**問題3**: 以下の行列 $A$ について、$A^TA$ と $AA^T$ を計算せよ。
$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$$

**問題4**: 次のデータの平均ベクトルを計算せよ。
$$X = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$$

**問題5**: 以下の行列が対称行列かどうかを判定せよ。
$$C = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{pmatrix}$$

**問題6**: 2つのベクトル $\mathbf{a} = (1, 2, 3)^T$ と $\mathbf{b} = (4, 5, 6)^T$ のなす角度（ラジアン）を計算せよ。（ヒント：$\cos \theta = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|}$）

### 6.2 応用問題

**問題7**: ある健康調査では、5人の患者から体温(℃)と脈拍(bpm)のデータを収集しました。

```
患者1: 体温=36.5, 脈拍=72
患者2: 体温=37.2, 脈拍=85
患者3: 体温=36.8, 脈拍=78
患者4: 体温=37.0, 脈拍=80
患者5: 体温=36.7, 脈拍=75
```

このデータを行列として表現し、体温と脈拍の相関係数を計算せよ。

**問題8**: 3人の学生のテストスコア（数学、物理、英語）が以下のように与えられています。

```
学生A: 数学=85, 物理=78, 英語=92
学生B: 数学=90, 物理=85, 英語=88
学生C: 数学=75, 物理=80, 英語=85
```

このデータを行列 $X$ として表し、以下の問いに答えよ。
(a) 各科目の平均点を求めよ。
(b) 数学と物理の相関係数を求めよ。

**問題9**: 2次元データ点 $(1, 2)$, $(3, 4)$, $(5, 6)$ について、これらの点の重心（平均）からの距離を計算せよ。

**問題10**: 行列 $A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$ を用いて、ベクトル $\mathbf{v} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$ を変換した結果を求めよ。また、変換前後のベクトルのノルムを比較せよ。

## 7. よくある質問と解答

### 7.1 ベクトルと行列の基本

**Q1: ベクトルと行列の違いは何ですか？**

A1: ベクトルは一次元の配列であり、大きさと方向を持つ量を表します。一方、行列は二次元の配列であり、複数のデータや線形変換を表現するために使用されます。数学的には、ベクトルは特殊な行列（1列行列または1行行列）と考えることもできます。

**Q2: 行列の積が可換でない理由は何ですか？**

A2: 行列の積 $AB$ は、行列 $A$ の列数と行列 $B$ の行数が一致する場合にのみ定義されます。このとき、結果の行列は $A$ の行数と $B$ の列数を持ちます。一般に、$A$ と $B$ の次元が異なる場合、$AB$ と $BA$ は次元が異なるか、一方が定義されないため、等しくなりません。また、同じ次元の正方行列でも、行列の積の定義から、一般には $AB \neq BA$ となります。

### 7.2 データ解析と線形代数

**Q3: なぜデータ分析において行列が重要なのですか？**

A3: 多変量データは自然に行列として表現できます。各行がサンプル、各列が変数を表すデータ行列は、多くのデータ分析手法の基礎となります。また、線形代数の演算を用いることで、データの変換、次元削減、パターン発見などの複雑な分析を効率的に行うことができます。例えば、主成分分析や因子分析などの多変量解析手法は、行列の固有値分解や特異値分解に基づいています。

**Q4: 相関係数と共分散の違いは何ですか？**

A4: 共分散は2つの変数の線形関係の強さを示す指標ですが、変数のスケールに依存します。一方、相関係数は共分散を各変数の標準偏差で割って標準化したもので、-1から1の範囲の値をとります。相関係数は変数のスケールに依存せず、2つの変数の線形関係の強さと方向を表す指標です。

### 7.3 計算と実装

**Q5: 大きな行列の計算を効率よく行うにはどうすればよいですか？**

A5: 大きな行列の計算には、NumPyやSciPyなどの最適化された数値計算ライブラリを使用することをお勧めします。これらのライブラリは、最適化された低レベルライブラリを利用しており、大規模な行列計算を非常に効率的に行うことができます。

**Q6: データの標準化と正規化の違いは何ですか？**

A6: 標準化はデータを平均0、分散1に変換する処理で、$(x - \mu) / \sigma$ で計算されます。これはデータの相対的な位置を保ちながら、異なるスケールの変数を比較可能にします。一方、正規化はデータを特定の範囲（通常は[0,1]）に収める処理で、$(x - \min) / (\max - \min)$ で計算されます。

## 8. 演習問題解答例

### 基本問題の解答

**問題1**:
ベクトルの和: $\mathbf{a} + \mathbf{b} = \begin{pmatrix} 3 \\ 1 \\ 4 \end{pmatrix} + \begin{pmatrix} 2 \\ 5 \\ 0 \end{pmatrix} = \begin{pmatrix} 5 \\ 6 \\ 4 \end{pmatrix}$

ベクトルの差: $\mathbf{a} - \mathbf{b} = \begin{pmatrix} 3 \\ 1 \\ 4 \end{pmatrix} - \begin{pmatrix} 2 \\ 5 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ -4 \\ 4 \end{pmatrix}$

内積: $\mathbf{a} \cdot \mathbf{b} = 3 \times 2 + 1 \times 5 + 4 \times 0 = 6 + 5 + 0 = 11$

ノルム:
$\|\mathbf{a}\| = \sqrt{3^2 + 1^2 + 4^2} = \sqrt{9 + 1 + 16} = \sqrt{26} \approx 5.10$
$\|\mathbf{b}\| = \sqrt{2^2 + 5^2 + 0^2} = \sqrt{4 + 25 + 0} = \sqrt{29} \approx 5.39$

**問題2**:
$AB = \begin{pmatrix} 2 & 1 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 2 \times 5 + 1 \times 7 & 2 \times 6 + 1 \times 8 \\ 3 \times 5 + 4 \times 7 & 3 \times 6 + 4 \times 8 \end{pmatrix} = \begin{pmatrix} 17 & 20 \\ 43 & 50 \end{pmatrix}$

**問題3**:
$A^TA = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 1 \times 1 + 3 \times 3 & 1 \times 2 + 3 \times 4 \\ 2 \times 1 + 4 \times 3 & 2 \times 2 + 4 \times 4 \end{pmatrix} = \begin{pmatrix} 10 & 14 \\ 14 & 20 \end{pmatrix}$

$AA^T = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} = \begin{pmatrix} 1 \times 1 + 2 \times 2 & 1 \times 3 + 2 \times 4 \\ 3 \times 1 + 4 \times 2 & 3 \times 3 + 4 \times 4 \end{pmatrix} = \begin{pmatrix} 5 & 11 \\ 11 & 25 \end{pmatrix}$

**問題4**:
平均ベクトル:
$\bar{\mathbf{x}} = \frac{1}{3} \begin{pmatrix} 1 + 2 + 3 \\ 4 + 5 + 6 \end{pmatrix} = \begin{pmatrix} 2 \\ 5 \end{pmatrix}$

**問題5**:
行列 $C$ が対称行列かどうかを判定するには、$C = C^T$ が成り立つかを確認します。

$C^T = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{pmatrix}^T = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{pmatrix}$

$C = C^T$ が成り立つので、行列 $C$ は対称行列です。

**問題6**:
2つのベクトル $\mathbf{a} = (1, 2, 3)^T$ と $\mathbf{b} = (4, 5, 6)^T$ のなす角度を計算します。

内積: $\mathbf{a} \cdot \mathbf{b} = 1 \times 4 + 2 \times 5 + 3 \times 6 = 4 + 10 + 18 = 32$

ノルム:
$\|\mathbf{a}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{1 + 4 + 9} = \sqrt{14} \approx 3.74$
$\|\mathbf{b}\| = \sqrt{4^2 + 5^2 + 6^2} = \sqrt{16 + 25 + 36} = \sqrt{77} \approx 8.78$

なす角のコサイン: $\cos \theta = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|} = \frac{32}{3.74 \times 8.78} \approx \frac{32}{32.84} \approx 0.974$

なす角: $\theta = \arccos(0.974) \approx 0.23$ ラジアン（約13度）

### 応用問題の解答

**問題7**:
まず、データ行列 $X$ を作成します。

$$X = \begin{pmatrix} 
36.5 & 72 \\
37.2 & 85 \\
36.8 & 78 \\
37.0 & 80 \\
36.7 & 75
\end{pmatrix}$$

次に、平均ベクトルを計算します。

$$\bar{\mathbf{x}} = \begin{pmatrix} 
\frac{36.5 + 37.2 + 36.8 + 37.0 + 36.7}{5} \\
\frac{72 + 85 + 78 + 80 + 75}{5}
\end{pmatrix} = \begin{pmatrix} 
36.84 \\
78
\end{pmatrix}$$

次に、中心化データ行列を計算します。

$$X - \mathbf{1}\bar{\mathbf{x}}^T = \begin{pmatrix} 
36.5 - 36.84 & 72 - 78 \\
37.2 - 36.84 & 85 - 78 \\
36.8 - 36.84 & 78 - 78 \\
37.0 - 36.84 & 80 - 78 \\
36.7 - 36.84 & 75 - 78
\end{pmatrix} = \begin{pmatrix} 
-0.34 & -6 \\
0.36 & 7 \\
-0.04 & 0 \\
0.16 & 2 \\
-0.14 & -3
\end{pmatrix}$$

分散共分散行列を計算します。

$$S = \frac{1}{5}(X - \mathbf{1}\bar{\mathbf{x}}^T)^T(X - \mathbf{1}\bar{\mathbf{x}}^T) = \frac{1}{5}\begin{pmatrix} 
(-0.34)^2 + 0.36^2 + (-0.04)^2 + 0.16^2 + (-0.14)^2 & (-0.34)(-6) + 0.36 \times 7 + (-0.04) \times 0 + 0.16 \times 2 + (-0.14)(-3) \\
(-0.34)(-6) + 0.36 \times 7 + (-0.04) \times 0 + 0.16 \times 2 + (-0.14)(-3) & (-6)^2 + 7^2 + 0^2 + 2^2 + (-3)^2
\end{pmatrix}$$

$$S = \frac{1}{5}\begin{pmatrix} 
0.3112 & 5.66 \\
5.66 & 118
\end{pmatrix} = \begin{pmatrix} 
0.06224 & 1.132 \\
1.132 & 23.6
\end{pmatrix}$$

各変数の標準偏差を計算します。

$$\sigma_1 = \sqrt{0.06224} \approx 0.2495$$
$$\sigma_2 = \sqrt{23.6} \approx 4.8580$$

相関係数を計算します。

$$r = \frac{s_{12}}{\sigma_1 \sigma_2} = \frac{1.132}{0.2495 \times 4.8580} \approx \frac{1.132}{1.212} \approx 0.934$$

したがって、体温と脈拍の間には強い正の相関関係（相関係数 ≈ 0.934）があることがわかります。

**問題8**:
データ行列 $X$ を作成します。

$$X = \begin{pmatrix} 
85 & 78 & 92 \\
90 & 85 & 88 \\
75 & 80 & 85
\end{pmatrix}$$

(a) 各科目の平均点を求めます。

$$\bar{\mathbf{x}} = \begin{pmatrix} 
\frac{85 + 90 + 75}{3} \\
\frac{78 + 85 + 80}{3} \\
\frac{92 + 88 + 85}{3}
\end{pmatrix} = \begin{pmatrix} 
83.33 \\
81 \\
88.33
\end{pmatrix}$$

したがって、数学の平均点は83.33点、物理の平均点は81点、英語の平均点は88.33点です。

(b) 数学と物理の相関係数を求めます。

まず、中心化データ行列を計算します。

$$X - \mathbf{1}\bar{\mathbf{x}}^T = \begin{pmatrix} 
85 - 83.33 & 78 - 81 & 92 - 88.33 \\
90 - 83.33 & 85 - 81 & 88 - 88.33 \\
75 - 83.33 & 80 - 81 & 85 - 88.33
\end{pmatrix} = \begin{pmatrix} 
1.67 & -3 & 3.67 \\
6.67 & 4 & -0.33 \\
-8.33 & -1 & -3.33
\end{pmatrix}$$

次に、数学と物理の列に関する共分散を計算します。
$$s_{12} = \frac{1}{3}[1.67 \times (-3) + 6.67 \times 4 + (-8.33) \times (-1)] = \frac{1}{3}[-5.01 + 26.68 + 8.33] = \frac{30}{3} = 10$$

数学の分散:
$$s_{11} = \frac{1}{3}[1.67^2 + 6.67^2 + (-8.33)^2] = \frac{1}{3}[2.79 + 44.49 + 69.39] = \frac{116.67}{3} \approx 38.89$$

物理の分散:
$$s_{22} = \frac{1}{3}[(-3)^2 + 4^2 + (-1)^2] = \frac{1}{3}[9 + 16 + 1] = \frac{26}{3} \approx 8.67$$

相関係数:
$$r_{12} = \frac{s_{12}}{\sqrt{s_{11}s_{22}}} = \frac{10}{\sqrt{38.89 \times 8.67}} = \frac{10}{\sqrt{337.18}} = \frac{10}{18.36} \approx 0.54$$

したがって、数学と物理の間には中程度の正の相関関係（相関係数 ≈ 0.54）があることがわかります。

**問題9**:
2次元データ点 $(1, 2)$, $(3, 4)$, $(5, 6)$ の重心（平均）を計算します。

$$\bar{\mathbf{x}} = \frac{1}{3} \begin{pmatrix} 1 + 3 + 5 \\ 2 + 4 + 6 \end{pmatrix} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$$

次に、各点から重心までの距離を計算します。

点 $(1, 2)$ から重心までの距離:
$$d_1 = \sqrt{(1 - 3)^2 + (2 - 4)^2} = \sqrt{4 + 4} = \sqrt{8} \approx 2.83$$

点 $(3, 4)$ から重心までの距離:
$$d_2 = \sqrt{(3 - 3)^2 + (4 - 4)^2} = \sqrt{0 + 0} = 0$$

点 $(5, 6)$ から重心までの距離:
$$d_3 = \sqrt{(5 - 3)^2 + (6 - 4)^2} = \sqrt{4 + 4} = \sqrt{8} \approx 2.83$$

**問題10**:
行列 $A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$ によるベクトル $\mathbf{v} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$ の変換結果を求めます。

$$A\mathbf{v} = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix} \begin{pmatrix} 2 \\ 1 \end{pmatrix} = \begin{pmatrix} 2 \times 2 + 1 \times 1 \\ 1 \times 2 + 3 \times 1 \end{pmatrix} = \begin{pmatrix} 5 \\ 5 \end{pmatrix}$$

変換前のベクトル $\mathbf{v}$ のノルム:
$$\|\mathbf{v}\| = \sqrt{2^2 + 1^2} = \sqrt{4 + 1} = \sqrt{5} \approx 2.24$$

変換後のベクトル $A\mathbf{v}$ のノルム:
$$\|A\mathbf{v}\| = \sqrt{5^2 + 5^2} = \sqrt{25 + 25} = \sqrt{50} \approx 7.07$$

変換によってベクトルのノルムは約3.16倍になりました。

## 9. まとめ

この講義では、線形代数学の基本概念である「ベクトル」と「行列」の基本操作から始めて、それらを用いたデータ解析の基礎について学びました。ベクトルの和・差・内積・ノルム、行列の和・差・積・転置などの基本操作を理解し、実際のデータに適用できることが重要です。

特に、多次元データの統計量（平均、分散、共分散、相関係数）を行列とベクトルを用いて表現・計算する方法を学びました。これらの概念は、次回以降の講義で学ぶ線形回帰モデルや主成分分析などのデータサイエンス手法の基礎となります。

演習問題を通じて、基本的な計算スキルを磨くとともに、実際のデータ分析の文脈での応用方法についても理解を深めました。線形代数は、データサイエンスの基盤となる重要な数学的ツールです。

次回の講義では、連立一次方程式とその解法に焦点を当て、線形代数とデータサイエンスの関連をさらに深く探求していきます。この総合演習を通じて、これまでに学んだ概念を整理・強化し、今後の応用に備えましょう。