# 線形代数学 I / 基礎 / II
## 第35回: 対称行列と直交行列による対角化

### 1. 講義情報と予習ガイド

**講義回**: 第35回  
**関連項目**: 固有値と固有ベクトル、対称行列、直交行列、対角化  
**予習すべき内容**:
- 行列の対角化の概念（第34回）
- 固有値と固有ベクトルの計算方法（第33回）
- 内積と直交性の概念（第27回）

### 2. 学習目標

1. 対称行列の定義と特性を理解する
2. 対称行列の固有値が実数であることを証明し理解する
3. 対称行列の異なる固有値に対する固有ベクトルの直交性を理解する
4. 直交行列の定義と性質を理解する
5. 対称行列の直交行列による対角化（スペクトル分解）の手順を習得する
6. 対称行列の対角化のデータサイエンスにおける重要性を理解する

### 3. 基本概念

#### 3.1 対称行列の定義

> **定義**: 正方行列 $A$ が**対称行列**であるとは、その転置行列が元の行列に等しい場合、すなわち $A^T = A$ が成り立つことをいう。

対称行列の具体例:

$$A = \begin{pmatrix} 
1 & 2 & 3 \\
2 & 4 & 5 \\
3 & 5 & 6
\end{pmatrix}$$

この行列は対称行列である。なぜなら、$a_{ij} = a_{ji}$ が全ての $i, j$ について成り立つからである。

対称行列でない行列の例:

$$B = \begin{pmatrix} 
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix}$$

この行列は対称行列ではない。例えば $b_{12} = 2$ だが $b_{21} = 4$ であり、$b_{12} \neq b_{21}$ となるからである。

#### 3.2 対称行列の性質

対称行列は以下の重要な性質を持つ:

1. すべての固有値は実数である
2. 異なる固有値に対応する固有ベクトルは互いに直交する
3. $n$ 次対称行列は $n$ 個の互いに直交する固有ベクトルを持つ

これらの性質について、次節で詳しく説明する。

#### 3.3 直交行列の定義

> **定義**: 正方行列 $Q$ が**直交行列**であるとは、$Q^T Q = Q Q^T = I$ が成り立つことをいう。ここで $I$ は単位行列である。

直交行列の性質:

1. 直交行列の列ベクトルは互いに直交する単位ベクトル（正規直交基底）である
2. 直交行列の行ベクトルも互いに直交する単位ベクトル（正規直交基底）である
3. 直交行列の逆行列は、その転置行列に等しい: $Q^{-1} = Q^T$
4. 直交行列の行列式の絶対値は 1 である: $|\det(Q)| = 1$

直交行列の例:

$$Q = \begin{pmatrix} 
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{pmatrix}$$

実際に確認してみると:

$$Q^T Q = \begin{pmatrix} 
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{pmatrix}^T 
\begin{pmatrix} 
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{pmatrix} = 
\begin{pmatrix} 
1 & 0 \\
0 & 1
\end{pmatrix} = I$$

### 4. 理論と手法

#### 4.1 対称行列の固有値が実数であることの証明

対称行列 $A$ のすべての固有値は実数である。これを証明する:

**証明**:
$A$ を対称行列とし、$\lambda$ をその固有値、$\mathbf{v}$ を対応する固有ベクトルとする。
ここで $\mathbf{v}$ は一般に複素数成分を持つ可能性があると仮定する。
また、$\mathbf{v}^*$ を $\mathbf{v}$ の複素共役転置とする。

固有値の定義より:
$A\mathbf{v} = \lambda\mathbf{v}$

両辺に左から $\mathbf{v}^*$ をかける:
$\mathbf{v}^* A \mathbf{v} = \lambda \mathbf{v}^* \mathbf{v}$

$A$ が対称行列なので $A^T = A$ であり、したがって $A = A^*$ (複素共役転置に等しい)

一方で、次の式も成り立つ:
$\mathbf{v}^* A \mathbf{v} = \mathbf{v}^* A^* \mathbf{v} = (A\mathbf{v})^* \mathbf{v} = (\lambda\mathbf{v})^* \mathbf{v} = \lambda^* \mathbf{v}^* \mathbf{v}$

よって:
$\lambda \mathbf{v}^* \mathbf{v} = \lambda^* \mathbf{v}^* \mathbf{v}$

$\mathbf{v} \neq \mathbf{0}$ なので $\mathbf{v}^* \mathbf{v} > 0$ であり、両辺を $\mathbf{v}^* \mathbf{v}$ で割ると:
$\lambda = \lambda^*$

これは $\lambda$ が実数であることを示している。

#### 4.2 対称行列の異なる固有値に対応する固有ベクトルの直交性

対称行列 $A$ について、異なる固有値 $\lambda_1 \neq \lambda_2$ に対応する固有ベクトル $\mathbf{v}_1$ と $\mathbf{v}_2$ は互いに直交する。

**証明**:
$A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$ かつ $A\mathbf{v}_2 = \lambda_2\mathbf{v}_2$ とする。

$\mathbf{v}_1$ と $A\mathbf{v}_2$ の内積を考える:
$\mathbf{v}_1^T (A\mathbf{v}_2) = \mathbf{v}_1^T (\lambda_2\mathbf{v}_2) = \lambda_2 \mathbf{v}_1^T \mathbf{v}_2$

一方、$A$ が対称行列なので:
$\mathbf{v}_1^T (A\mathbf{v}_2) = (A^T\mathbf{v}_1)^T \mathbf{v}_2 = (A\mathbf{v}_1)^T \mathbf{v}_2 = (\lambda_1\mathbf{v}_1)^T \mathbf{v}_2 = \lambda_1 \mathbf{v}_1^T \mathbf{v}_2$

よって:
$\lambda_2 \mathbf{v}_1^T \mathbf{v}_2 = \lambda_1 \mathbf{v}_1^T \mathbf{v}_2$
$(\lambda_2 - \lambda_1) \mathbf{v}_1^T \mathbf{v}_2 = 0$

$\lambda_1 \neq \lambda_2$ という仮定から:
$\mathbf{v}_1^T \mathbf{v}_2 = 0$

これは固有ベクトル $\mathbf{v}_1$ と $\mathbf{v}_2$ が直交していることを示している。

#### 4.3 対称行列の直交行列による対角化（スペクトル分解）

対称行列 $A$ は直交行列 $Q$ を用いて対角化できる:

> **定理（スペクトル分解）**: $n$ 次対称行列 $A$ に対して、直交行列 $Q$ と対角行列 $D$ が存在し、$A = QDQ^T$ と表せる。ここで $D$ の対角成分は $A$ の固有値であり、$Q$ の列は対応する正規化された固有ベクトルである。

**対角化の手順**:

1. 対称行列 $A$ の固有値 $\lambda_1, \lambda_2, \ldots, \lambda_n$ と対応する固有ベクトル $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$ を求める
2. 同じ固有値に対応する固有ベクトルからなる部分空間で、グラム・シュミットの直交化法を用いて正規直交基底を構成する
3. 全ての固有ベクトルを正規化して単位ベクトルにする
4. 正規化された固有ベクトルを列とする行列 $Q$ を作る
5. 固有値を対角成分とする対角行列 $D$ を作る

この時、$A = QDQ^T$ が成り立つ。

#### 4.4 スペクトル分解の意味

$A = QDQ^T$ というスペクトル分解は、行列 $A$ を以下のように表現することもできる:

$$A = \sum_{i=1}^{n} \lambda_i \mathbf{q}_i \mathbf{q}_i^T$$

ここで $\lambda_i$ は固有値、$\mathbf{q}_i$ は対応する正規化された固有ベクトル（$Q$ の $i$ 列目）である。

この表現は、対称行列 $A$ が、各固有ベクトル方向の「rank-1」射影行列 $\mathbf{q}_i \mathbf{q}_i^T$ の固有値による重み付き和として分解できることを示している。

### 5. 具体的な計算例とPython実装

#### 5.1 対称行列の対角化の例

次の対称行列について考える:

$$A = \begin{pmatrix} 
2 & 1 & 1 \\
1 & 2 & 1 \\
1 & 1 & 2
\end{pmatrix}$$

**ステップ1**: 固有値と固有ベクトルを求める

特性方程式: $\det(A - \lambda I) = 0$

$$\det\begin{pmatrix} 
2-\lambda & 1 & 1 \\
1 & 2-\lambda & 1 \\
1 & 1 & 2-\lambda
\end{pmatrix} = 0$$

計算すると:
$(2-\lambda)^3 - 3(2-\lambda) + 2 = 0$
$-(2-\lambda)^3 + 3(2-\lambda) - 2 = 0$
$-(\lambda-2)^3 + 3(\lambda-2) - 2 = 0$
$-(\lambda-2)^3 + 3(\lambda-2) - 2 = 0$

因数分解すると:
$-(\lambda-2)(\lambda-2)^2 + 3(\lambda-2) - 2 = 0$
$-(\lambda-2)((\lambda-2)^2 - 3) - 2 = 0$
$-(\lambda-2)((\lambda-2)^2 - 3) - 2 = 0$
$-(\lambda-2)((\lambda-2)^2 - 3) - 2 = 0$
$-(\lambda-2)((\lambda-2)^2 - 3) - 2 = 0$
$-(\lambda-2)((\lambda-2)^2 - 3) - 2 = 0$

簡単な計算方法として、この行列は特殊な形をしており、固有多項式は:
$(\lambda - 4)(\lambda - 1)^2 = 0$

したがって、固有値は:
$\lambda_1 = 4$ (多重度1)
$\lambda_2 = 1$ (多重度2)

**固有値 $\lambda_1 = 4$ に対応する固有ベクトルを求める**:
$(A - 4I)\mathbf{v}_1 = \mathbf{0}$

$$\begin{pmatrix} 
-2 & 1 & 1 \\
1 & -2 & 1 \\
1 & 1 & -2
\end{pmatrix}\mathbf{v}_1 = \mathbf{0}$$

これを解くと、$\mathbf{v}_1 = (1, 1, 1)^T$ が得られる（正規化前）。

正規化すると:
$\mathbf{q}_1 = \frac{\mathbf{v}_1}{||\mathbf{v}_1||} = \frac{1}{\sqrt{3}}(1, 1, 1)^T$

**固有値 $\lambda_2 = 1$ に対応する固有ベクトルを求める**:
$(A - I)\mathbf{v}_2 = \mathbf{0}$

$$\begin{pmatrix} 
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{pmatrix}\mathbf{v} = \mathbf{0}$$

この行列はランク1なので、固有空間の次元は2です。2つの線形独立な固有ベクトルとして、例えば:
$\mathbf{v}_2 = (1, -1, 0)^T$ と $\mathbf{v}_3 = (1, 0, -1)^T$ を選べます。

これらの固有ベクトルは、$\mathbf{q}_1$ に直交することを確認できます:
$\mathbf{q}_1^T \mathbf{v}_2 = \frac{1}{\sqrt{3}}(1 + (-1) + 0) = 0$
$\mathbf{q}_1^T \mathbf{v}_3 = \frac{1}{\sqrt{3}}(1 + 0 + (-1)) = 0$

しかし、$\mathbf{v}_2$ と $\mathbf{v}_3$ は互いに直交していないので、グラム・シュミットの直交化法を適用する必要があります。

ただ、簡単のために別の方法で2つの直交するベクトルを選びましょう:
$\mathbf{v}_2 = (1, -1, 0)^T$ と $\mathbf{v}_3 = (1, 1, -2)^T$

この二つは互いに直交していることが確認できます:
$\mathbf{v}_2^T \mathbf{v}_3 = 1 \cdot 1 + (-1) \cdot 1 + 0 \cdot (-2) = 0$

これらを正規化すると:
$\mathbf{q}_2 = \frac{\mathbf{v}_2}{||\mathbf{v}_2||} = \frac{1}{\sqrt{2}}(1, -1, 0)^T$
$\mathbf{q}_3 = \frac{\mathbf{v}_3}{||\mathbf{v}_3||} = \frac{1}{\sqrt{6}}(1, 1, -2)^T$

**ステップ3**: 直交行列 $Q$ と対角行列 $D$ を構成する

$$Q = \begin{pmatrix} 
\frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
\frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
\frac{1}{\sqrt{3}} & 0 & -\frac{2}{\sqrt{6}}
\end{pmatrix}$$

$$D = \begin{pmatrix} 
4 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}$$

**ステップ4**: $A = QDQ^T$ を確認する

計算により、$QDQ^T = A$ が成り立つことを確認できます。

#### 5.2 Pythonによる実装

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 対称行列の例
A = np.array([
    [2, 1, 1],
    [1, 2, 1],
    [1, 1, 2]
])

print("対称行列 A:")
print(A)

# 固有値と固有ベクトルの計算
eigvals, eigvecs = np.linalg.eigh(A)  # eigh は対称行列用の関数

print("\n固有値:")
print(eigvals)

print("\n固有ベクトル（列ベクトル）:")
print(eigvecs)

# 対角行列の構成
D = np.diag(eigvals)
print("\n対角行列 D:")
print(D)

# 直交行列 Q
Q = eigvecs
print("\n直交行列 Q:")
print(Q)

# 確認: Q^T Q = I
print("\nQ^T Q:")
print(np.round(Q.T @ Q, 10))  # 丸め誤差を考慮

# 確認: A = Q D Q^T
A_reconstructed = Q @ D @ Q.T
print("\nQ D Q^T:")
print(np.round(A_reconstructed, 10))  # 丸め誤差を考慮

# スペクトル分解の視覚化
def plot_spectral_decomposition():
    # 単位球の点を生成
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2*np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # 球面上の点を行列形式に変換
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    
    # 各点を変換
    transformed_points = np.array([A @ p for p in points])
    
    # 可視化
    fig = plt.figure(figsize=(12, 6))
    
    # 元の単位球
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:,0], points[:,1], points[:,2], c='b', alpha=0.2)
    ax1.set_title("元の単位球")
    ax1.set_xlim([-4, 4])
    ax1.set_ylim([-4, 4])
    ax1.set_zlim([-4, 4])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # A による変換後
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(transformed_points[:,0], transformed_points[:,1], transformed_points[:,2], c='r', alpha=0.2)
    
    # 固有ベクトルを描画
    for i in range(3):
        eigen_vec = eigvecs[:, i] * eigvals[i] * 3  # スケーリングして見やすく
        ax2.quiver(0, 0, 0, eigen_vec[0], eigen_vec[1], eigen_vec[2], color='g', arrow_length_ratio=0.1)
    
    ax2.set_title("A による変換後と固有ベクトル")
    ax2.set_xlim([-4, 4])
    ax2.set_ylim([-4, 4])
    ax2.set_zlim([-4, 4])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

# スペクトル分解の視覚化実行
plot_spectral_decomposition()
```

### 6. スペクトル分解のデータサイエンスにおける応用

#### 6.1 主成分分析（PCA）

対称行列の直交行列による対角化は、主成分分析（PCA）の理論的基礎となっています。PCAでは、データの共分散行列（これは対称行列です）を対角化することで、データの主成分を見つけます。

主成分分析のステップ:

1. データ行列 $X$ の各列（変数）について中心化を行う
2. 共分散行列 $C = \frac{1}{n-1}X^T X$ を計算する（これは対称行列）
3. 共分散行列の固有値と固有ベクトルを計算する
4. 固有値の大きい順に固有ベクトルを並べる（これらが主成分）
5. 元のデータを主成分空間に射影する: $Z = X W$（ここで $W$ は選択した固有ベクトルを列とする行列）

#### 6.2 画像処理における応用

対称行列の対角化は、画像処理でも重要な役割を果たします。例えば、画像の圧縮や特徴抽出に使用されます。具体的には以下のような応用があります：

- 画像圧縮: 固有値の小さな成分を切り捨てることで、情報をほとんど失わずに画像のサイズを削減
- 顔認識: 顔画像の主成分（固有顔）を抽出して特徴ベクトルとして利用
- ノイズ除去: 固有値の小さな成分を取り除くことで、画像からノイズを削減

#### 6.3 健康データ分析における応用例

健康データ分析では、対称行列の対角化が以下のようなシナリオで利用されます：

1. 多次元健康指標の次元削減:
   - 血圧、心拍数、血糖値などの複数の健康指標の相関構造を分析
   - 主成分分析を用いて重要な変動の方向を特定
   - 冗長性を減らした新しい健康スコアの開発

2. 医療画像処理:
   - MRIやCTスキャンデータの特徴抽出
   - 異常検出のための正常パターンの学習
   - 画像の圧縮と再構築

```python
# 健康データの主成分分析の例
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# サンプル健康データの生成（模擬データ）
np.random.seed(42)
n_samples = 100

# 収縮期血圧(SBP)、拡張期血圧(DBP)、心拍数(HR)、血糖値(BS)、体温(BT)
health_data = pd.DataFrame({
    'SBP': np.random.normal(120, 15, n_samples),
    'DBP': np.random.normal(80, 10, n_samples),
    'HR': np.random.normal(70, 10, n_samples),
    'BS': np.random.normal(100, 20, n_samples),
    'BT': np.random.normal(36.5, 0.5, n_samples)
})

# 相関を持たせる
health_data['SBP'] = health_data['SBP'] + health_data['DBP'] * 0.5
health_data['HR'] = health_data['HR'] + health_data['BS'] * 0.2

print("健康データのサンプル:")
print(health_data.head())

# データの標準化
scaler = StandardScaler()
health_data_scaled = scaler.fit_transform(health_data)

# 共分散行列の計算（これは対称行列）
cov_matrix = np.cov(health_data_scaled.T)
print("\n共分散行列:")
print(cov_matrix)

# 固有値と固有ベクトルの計算
eigvals, eigvecs = np.linalg.eigh(cov_matrix)

# 固有値を降順に並べ替え
idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

print("\n共分散行列の固有値（降順）:")
print(eigvals)

print("\n共分散行列の固有ベクトル（列ベクトル、降順）:")
print(eigvecs)

# 主成分分析の実施
pca = PCA()
health_data_pca = pca.fit_transform(health_data_scaled)

# 寄与率（各主成分がどれだけ元のデータの分散を説明するか）
print("\n各主成分の寄与率:")
print(pca.explained_variance_ratio_)

print("\n累積寄与率:")
print(np.cumsum(pca.explained_variance_ratio_))

# 可視化: 第1主成分と第2主成分へのプロット
plt.figure(figsize=(10, 6))
plt.scatter(health_data_pca[:, 0], health_data_pca[:, 1], alpha=0.7)
plt.xlabel('第1主成分')
plt.ylabel('第2主成分')
plt.title('健康データの主成分分析')
plt.grid(True)

# 元の特徴量の主成分への寄与を示す矢印を追加
for i, (x, y) in enumerate(zip(eigvecs[0, :2], eigvecs[1, :2])):
    plt.arrow(0, 0, x*3, y*3, head_width=0.15, head_length=0.2, fc='red', ec='red')
    plt.text(x*3.2, y*3.2, health_data.columns[i], fontsize=12)

plt.tight_layout()
plt.show()

# バイプロット: 主成分空間における変数と観測値の同時表示
def create_biplot(score, coeff, labels):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    
    plt.figure(figsize=(12, 8))
    plt.scatter(xs, ys, alpha=0.7)
    
    # 主成分空間における変数の方向を矢印で表示
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0]*5, coeff[i,1]*5, color='r', alpha=0.5, head_width=0.1)
        plt.text(coeff[i,0]*5.2, coeff[i,1]*5.2, labels[i], color='g', fontsize=12)
    
    plt.xlabel('第1主成分')
    plt.ylabel('第2主成分')
    plt.title('健康データのバイプロット')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 主成分係数（負荷量）を取得
loadings = pca.components_.T

# バイプロットの作成
create_biplot(health_data_pca[:, :2], loadings[:, :2], health_data.columns)
```

### 7. 演習問題

#### 基本問題

1. 次の行列が対称行列であるかどうかを判定せよ。
   a) $\begin{pmatrix} 3 & 2 \\ 2 & 5 \end{pmatrix}$
   b) $\begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 6 & 9 \end{pmatrix}$
   c) $\begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$

2. 次の対称行列の固有値と固有ベクトルを求め、直交行列による対角化を行え。
   $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$

3. 次の対称行列の固有値を求めよ。
   $B = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 3 & 4 \\ 0 & 4 & 3 \end{pmatrix}$

4. 対称行列の固有値が全て正であるとき、その行列は正定値であるという。次の行列が正定値であるかを判定せよ。
   $C = \begin{pmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{pmatrix}$

5. 直交行列 $Q = \begin{pmatrix} \frac{3}{5} & \frac{4}{5} \\ \frac{4}{5} & -\frac{3}{5} \end{pmatrix}$ について、$Q^TQ = I$ であることを確認せよ。

#### 応用問題

1. 3×3の対称行列 $A$ について、その特性多項式が $\lambda^3 - 6\lambda^2 + 11\lambda - 6 = 0$ であるとき、$A$ のスペクトル分解を求めよ。ただし、固有値に対応する固有ベクトルは次のように与えられているとする。
   - $\lambda_1 = 1$ に対して $\mathbf{v}_1 = (1, 1, 1)^T$
   - $\lambda_2 = 2$ に対して $\mathbf{v}_2 = (1, 0, -1)^T$
   - $\lambda_3 = 3$ に対して $\mathbf{v}_3 = (1, -2, 1)^T$

2. 対称行列 $A = \begin{pmatrix} 4 & 0 & 0 \\ 0 & 4 & 0 \\ 0 & 0 & 1 \end{pmatrix}$ について、この行列の任意の固有ベクトルから作られる直交行列 $Q$ を構成し、$A = QDQ^T$ の形で表せ。

3. 2次元データ点 $(1,2)$, $(2,3)$, $(3,5)$, $(4,4)$, $(5,6)$ について、以下の手順で主成分分析を行え。
   a) データの平均ベクトルを求めよ。
   b) 平均を引いた中心化データを求めよ。
   c) 共分散行列を計算せよ。
   d) 共分散行列の固有値と固有ベクトルを求めよ。
   e) 主成分方向を特定し、データを主成分空間に射影せよ。

4. 健康データ分析の文脈において、異なる測定値（血圧、心拍数、体温、血糖値など）の共分散行列が対称行列である理由を説明し、この行列の固有値と固有ベクトルがどのように解釈できるかを考察せよ。また、これらの測定値間に強い相関がある場合、主成分分析を用いた次元削減がどのように有効かを説明せよ。

### 8. よくある質問と解答

#### Q1: 対称行列の固有値が常に実数である理由は何ですか？
A1: 対称行列の固有値が常に実数である理由は、対称行列 $A$ の任意の固有値 $\lambda$ と対応する固有ベクトル $\mathbf{v}$ について、$\mathbf{v}^* A \mathbf{v} = \lambda \mathbf{v}^* \mathbf{v}$ と $\mathbf{v}^* A \mathbf{v} = \lambda^* \mathbf{v}^* \mathbf{v}$ の両方が成り立つことから導かれます。この二つの式から $\lambda = \lambda^*$ が導かれ、これは固有値が実数であることを意味します。直感的には、対称行列は「バランスの取れた」変換を表し、そのような変換の主要な方向（固有ベクトル）に沿った伸縮率（固有値）は実数になります。

#### Q2: 対称行列の異なる固有値に対応する固有ベクトルが直交する理由は何ですか？
A2: 対称行列 $A$ の異なる固有値 $\lambda_1 \neq \lambda_2$ に対応する固有ベクトル $\mathbf{v}_1$ と $\mathbf{v}_2$ について、$\mathbf{v}_1^T A \mathbf{v}_2 = \lambda_2 \mathbf{v}_1^T \mathbf{v}_2$ かつ $\mathbf{v}_1^T A \mathbf{v}_2 = \lambda_1 \mathbf{v}_1^T \mathbf{v}_2$ が成り立ちます。これにより $(\lambda_2 - \lambda_1) \mathbf{v}_1^T \mathbf{v}_2 = 0$ となり、$\lambda_1 \neq \lambda_2$ なので $\mathbf{v}_1^T \mathbf{v}_2 = 0$ となります。これは二つの固有ベクトルが直交していることを意味します。幾何学的には、対称行列は互いに独立した直交する方向に沿って伸縮する変換を表しています。

#### Q3: 対称行列の対角化と一般の行列の対角化の違いは何ですか？
A3: 対称行列と一般の行列の対角化の主な違いは以下の点です：
1. 対称行列は必ず対角化可能ですが、一般の行列は対角化できない場合があります。
2. 対称行列は直交行列（$Q^TQ = I$）によって対角化できますが、一般の行列は必ずしも直交行列では対角化できません。
3. 対称行列の固有値は全て実数ですが、一般の行列の固有値は複素数になる場合があります。
4. 対称行列の異なる固有値に対応する固有ベクトルは直交しますが、一般の行列ではそのような保証はありません。

#### Q4: 主成分分析（PCA）と対称行列の対角化はどのように関連していますか？
A4: 主成分分析（PCA）は、データの共分散行列（対称行列）の対角化に基づいています。具体的には：
1. データ行列から計算される共分散行列は対称行列です。
2. この共分散行列を対角化すると、固有ベクトルが主成分の方向を表し、対応する固有値がその方向の分散を表します。
3. 固有値が大きい順に固有ベクトル（主成分）を並べることで、データの変動を最もよく捉える軸を特定できます。
4. 少数の主要な主成分だけを選ぶことで、高次元データを低次元に圧縮しながら、重要な情報を保持できます。

#### Q5: スペクトル分解の応用例として他にどのようなものがありますか？
A5: スペクトル分解の応用例には以下のようなものがあります：
1. 信号処理：ノイズ除去や信号の特徴抽出
2. 量子力学：量子状態や観測量の解析
3. グラフ理論：グラフのラプラシアン行列から得られる固有値と固有ベクトルによるコミュニティ検出
4. 振動解析：構造物の固有振動モードの特定
5. 推薦システム：協調フィルタリングにおける低ランク近似
6. 機械学習：カーネル法（カーネル主成分分析）
7. 画像処理：画像圧縮や特徴抽出

#### Q6: 多重固有値が存在する場合、対角化はどのように行いますか？
A6: 多重固有値が存在する場合の対角化手順は以下の通りです：
1. 各固有値とその多重度を求めます。
2. 各固有値について、対応する固有空間（固有ベクトルの張る空間）の基底を求めます。
3. 同じ固有値に対応する固有ベクトルのセットには、グラム・シュミットの直交化法を適用して正規直交基底を得ます。
4. 全ての固有値に対して得られた正規直交基底を列として並べた行列 $Q$ と、固有値を対角成分に持つ対角行列 $D$ を構成します。
5. $A = QDQ^T$ によって対角化が完了します。

重要なのは、対称行列の場合、多重固有値に対応する固有ベクトルも直交化が可能で、必ず $n$ 個の互いに直交する固有ベクトルが存在するという点です。