# 線形代数学 I / 基礎 / II：第42回講義ノート

## 1. 講義情報と予習ガイド

**講義回**: 第42回  
**テーマ**: 主成分分析の結果の解釈と次元削減  
**関連項目**: 固有値・固有ベクトル、共分散行列、次元削減  
**予習内容**: 第41回の主成分分析の導入、固有値・固有ベクトルの概念、正定値対称行列の性質

## 2. 学習目標

本講義の終了時には、以下のことができるようになります：

1. 主成分分析の結果を線形代数の観点から厳密に解釈できる
2. スクリープロットとバイプロットを作成し、数学的背景に基づいて解釈できる
3. マハラノビス距離の数理的定義を理解し、外れ値検出に応用できる
4. 主成分回帰の線形代数的基礎と理論的特性を説明できる
5. 健康・医療データに対する主成分分析の応用例を線形変換の観点から理解できる

## 3. 基本概念

### 3.1 主成分分析の数理的基礎

主成分分析（Principal Component Analysis, PCA）は、線形代数学における固有値分解（または特異値分解）に基づく次元削減と特徴抽出のための手法です。前回の講義で学んだ理論を復習しましょう。

> **主成分分析の数学的定義**
> 
> $n$個の$p$次元データポイント $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$ があるとき、これらのデータの平均ベクトル $\boldsymbol{\mu}$ と共分散行列 $\boldsymbol{\Sigma}$ は以下のように定義されます：
> 
> $\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i$
> 
> $\boldsymbol{\Sigma} = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T$
> 
> 共分散行列 $\boldsymbol{\Sigma}$ は対称行列であり、非負の固有値 $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_p \geq 0$ と、それに対応する正規直交固有ベクトル $\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_p$ を持ちます。
>
> 第 $j$ 主成分は、固有ベクトル $\mathbf{e}_j$ 方向への射影として定義され、データポイント $\mathbf{x}_i$ の第 $j$ 主成分スコアは以下で計算されます：
>
> $z_{ij} = \mathbf{e}_j^T(\mathbf{x}_i - \boldsymbol{\mu})$

主成分分析の目的は、データの分散を最大限に保持しながら、データの次元を削減することです。線形代数的には、これは高次元データを低次元の部分空間に射影することに相当します。第$j$主成分方向の分散は固有値$\lambda_j$で与えられ、主成分は互いに直交しています。

### 3.2 共分散行列と固有値分解の関係

共分散行列 $\boldsymbol{\Sigma}$ は正定値対称行列（または半正定値対称行列）であるため、線形代数学の重要な定理により、以下の固有値分解が可能です：

> **共分散行列の固有値分解**
>
> $\boldsymbol{\Sigma} = \mathbf{E} \boldsymbol{\Lambda} \mathbf{E}^T$
>
> ここで、$\mathbf{E} = [\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_p]$ は固有ベクトルを列とする直交行列であり、$\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_p)$ は固有値を対角成分とする対角行列です。

この固有値分解は、データの分散構造を最もよく表現する直交基底（固有ベクトル）を見つけ、各基底方向のデータの散らばり具合（分散）を固有値として定量化しています。

## 4. 理論と手法

### 4.1 主成分分析の結果の数理的解釈

#### 4.1.1 固有値と分散の関係

固有値は各主成分の重要度を表しますが、より厳密には、各固有値は対応する主成分方向のデータの分散を表しています。

> **固有値と分散の関係**
>
> 元のデータの全分散は、共分散行列の対角和（トレース）に等しく、これは全固有値の和に等しくなります：
>
> $\text{全分散} = \text{tr}(\boldsymbol{\Sigma}) = \sum_{j=1}^{p} \lambda_j$
>
> 第 $j$ 主成分の分散 = $\lambda_j$

**例題**：あるデータセットの共分散行列の固有値が $\lambda = [5.2, 2.1, 0.7, 0.3, 0.1]$ であった場合、各主成分の説明する分散の割合を求めなさい。

**解答**：
全分散 = $\sum_{j=1}^{5} \lambda_j = 5.2 + 2.1 + 0.7 + 0.3 + 0.1 = 8.4$

第1主成分：$\frac{\lambda_1}{\sum_{j=1}^{5} \lambda_j} = \frac{5.2}{8.4} = 0.619 = 61.9\%$  
第2主成分：$\frac{\lambda_2}{\sum_{j=1}^{5} \lambda_j} = \frac{2.1}{8.4} = 0.250 = 25.0\%$  
第3主成分：$\frac{\lambda_3}{\sum_{j=1}^{5} \lambda_j} = \frac{0.7}{8.4} = 0.083 = 8.3\%$  
第4主成分：$\frac{\lambda_4}{\sum_{j=1}^{5} \lambda_j} = \frac{0.3}{8.4} = 0.036 = 3.6\%$  
第5主成分：$\frac{\lambda_5}{\sum_{j=1}^{5} \lambda_j} = \frac{0.1}{8.4} = 0.012 = 1.2\%$  

#### 4.1.2 寄与率と累積寄与率

寄与率は各主成分が全分散のうちどれだけを説明しているかを表します。数学的には以下のように定義されます：

> **寄与率の数学的定義**
>
> 第 $j$ 主成分の寄与率 = $\frac{\lambda_j}{\sum_{i=1}^{p} \lambda_i}$
>
> 第 $j$ 主成分までの累積寄与率 = $\frac{\sum_{i=1}^{j} \lambda_i}{\sum_{i=1}^{p} \lambda_i}$

累積寄与率は、選択した主成分がデータの分散をどれだけ説明しているかを示します。一般的に、累積寄与率が70%〜90%になるように主成分数を選択します。

上記の例題の累積寄与率は以下のようになります：
第1主成分まで：61.9%  
第2主成分まで：61.9% + 25.0% = 86.9%  
第3主成分まで：86.9% + 8.3% = 95.2%  
第4主成分まで：95.2% + 3.6% = 98.8%  
第5主成分まで：98.8% + 1.2% = 100%  

#### 4.1.3 主成分負荷量の線形代数的解釈

主成分負荷量は、各主成分と元の変数との相関係数です。線形代数的には、主成分負荷量は固有ベクトルと固有値を用いて計算されます。

> **主成分負荷量の数学的定義**
>
> 第 $j$ 主成分の第 $i$ 変数に対する負荷量 = $\sqrt{\lambda_j} \times e_{ij}$
>
> ここで、$e_{ij}$ は第 $j$ 固有ベクトル $\mathbf{e}_j$ の第 $i$ 成分です。

主成分負荷量の行列 $\mathbf{L}$ は以下のように表されます：

$\mathbf{L} = \mathbf{E} \boldsymbol{\Lambda}^{1/2}$

ここで、$\boldsymbol{\Lambda}^{1/2} = \text{diag}(\sqrt{\lambda_1}, \sqrt{\lambda_2}, \ldots, \sqrt{\lambda_p})$ です。

**解釈例**：
- 負荷量が正の大きな値 → 主成分と変数は正の相関
- 負荷量が負の大きな値 → 主成分と変数は負の相関
- 負荷量が0に近い → 主成分と変数にはほとんど関連がない

#### 4.1.4 主成分スコアと座標変換

主成分スコアは、各データポイントを主成分空間に投影した値です。線形代数的には、これはデータの座標変換に相当します。

> **主成分スコアの数学的定義**
>
> 中心化されたデータ行列 $\mathbf{X}_c = [\mathbf{x}_1 - \boldsymbol{\mu}, \mathbf{x}_2 - \boldsymbol{\mu}, \ldots, \mathbf{x}_n - \boldsymbol{\mu}]^T$ に対して、主成分スコア行列 $\mathbf{Z}$ は以下のように計算されます：
>
> $\mathbf{Z} = \mathbf{X}_c \mathbf{E} = \mathbf{X}_c [\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_p]$
>
> 各行が一つのデータポイントの主成分スコアを表し、各列が一つの主成分に対応します。

これは、元のデータ空間から新しい直交基底（固有ベクトル）によって定義される主成分空間への線形変換です。この変換により、データの共分散行列は対角化され、主成分間の共分散はゼロになります。

### 4.2 可視化ツールと数理的背景

#### 4.2.1 スクリープロットと固有値の減衰率

スクリープロットは固有値の大きさを降順に並べたグラフです。線形代数的な観点からは、固有値の減衰パターンはデータの本質的な次元性を示唆します。

> **スクリープロットの数理的解釈**
>
> 固有値の降下が急な部分は、データの主要な分散方向を示します。一方、固有値が急激に減少した後の平坦な部分は、データのノイズや冗長性に対応する方向を示します。
>
> 「肘」の位置は、データの本質的な次元数を示唆し、この点までの主成分を選択することで、情報の損失を最小限に抑えつつ次元削減が可能です。

**「肘」の数学的考察**：スクリープロットにおいて、固有値の減少率（$\frac{\lambda_j - \lambda_{j+1}}{\lambda_j}$）が急激に小さくなる点を「肘」と考えることができます。

#### 4.2.2 バイプロットと変数の射影

バイプロットは主成分スコアと主成分負荷量を同時に表示するプロットです。線形代数的には、これは元の変数を主成分空間に射影したものと見なせます。

> **バイプロットの数理的基礎**
>
> バイプロットでは、データポイントは主成分スコア行列 $\mathbf{Z}$ の最初の2列（通常は第1主成分と第2主成分）を用いてプロットされます。
>
> 変数ベクトルは、主成分負荷量行列 $\mathbf{L}$ の最初の2列を用いて描かれ、各変数が第1主成分と第2主成分にどのように寄与しているかを示します。
>
> 数学的には、変数ベクトルの向きと長さは、元の変数の基底ベクトルを主成分空間に射影したものに対応します。

**バイプロットの幾何学的解釈**：
- 変数ベクトルが同じ方向を向いている→これらの変数は正の相関関係にある
- 変数ベクトルが反対方向を向いている→これらの変数は負の相関関係にある
- 変数ベクトルが直交している→これらの変数はほぼ無相関である
- ベクトルの長さは、その変数が主成分平面上で表現される程度を示す

### 4.3 マハラノビス距離の数理的基礎

マハラノビス距離は、データの共分散構造を考慮した距離測度です。線形代数的には、データ空間の計量を定義しています。

> **マハラノビス距離の厳密な定義**
>
> データポイント $\mathbf{x}$ に対するマハラノビス距離の二乗は以下で与えられます：
>
> $D^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$
>
> ここで、$\boldsymbol{\mu}$ はデータの平均ベクトル、$\boldsymbol{\Sigma}$ は共分散行列、$\boldsymbol{\Sigma}^{-1}$ はその逆行列です。

共分散行列の固有値分解 $\boldsymbol{\Sigma} = \mathbf{E} \boldsymbol{\Lambda} \mathbf{E}^T$ を用いると、マハラノビス距離は主成分空間で以下のように表されます：

> **主成分空間でのマハラノビス距離**
>
> $D^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{E} \boldsymbol{\Lambda}^{-1} \mathbf{E}^T (\mathbf{x} - \boldsymbol{\mu}) = \sum_{j=1}^{p} \frac{z_j^2}{\lambda_j}$
>
> ここで、$z_j = \mathbf{e}_j^T(\mathbf{x} - \boldsymbol{\mu})$ は第 $j$ 主成分スコアです。

この式から、マハラノビス距離は主成分スコアを対応する固有値で標準化し、その二乗和として計算できることがわかります。これは、各主成分方向の分散の違いを考慮した標準化ユークリッド距離と解釈できます。

**統計的特性**：$p$ 変量正規分布に従うデータの場合、マハラノビス距離の二乗は自由度 $p$ のカイ二乗分布に従います。これを利用して、外れ値の統計的検定が可能です。

### 4.4 主成分回帰の線形代数的基礎

主成分回帰（Principal Component Regression, PCR）は、主成分分析と線形回帰を組み合わせた手法です。線形代数的には、これは回帰問題を主成分空間で解くことに相当します。

> **主成分回帰の数学的定式化**
>
> 説明変数 $\mathbf{X}$ と目的変数 $\mathbf{y}$ があるとき、主成分回帰は以下の手順で行われます：
>
> 1. $\mathbf{X}$ に対して PCA を適用し、固有ベクトル行列 $\mathbf{E}$ と主成分スコア行列 $\mathbf{Z} = \mathbf{X}_c \mathbf{E}$ を得る
> 2. 上位 $k$ 個の主成分を選択し、$\mathbf{Z}_k$ を得る
> 3. $\mathbf{Z}_k$ を説明変数とする線形回帰モデルを構築：$\mathbf{y} = \mathbf{Z}_k \boldsymbol{\gamma} + \boldsymbol{\epsilon}$
> 4. 元の説明変数での係数 $\boldsymbol{\beta}$ は以下で求められる：$\boldsymbol{\beta} = \mathbf{E}_k \boldsymbol{\gamma}$（ここで $\mathbf{E}_k$ は最初の $k$ 列の固有ベクトル）

主成分回帰の理論的特性：
1. 多重共線性の問題を解決（主成分は互いに直交）
2. 分散の小さい方向（ノイズや冗長情報）を除外することでモデルを安定化
3. 高次元データの効率的な回帰モデル構築（次元の呪いの緩和）

**例題**：健康データ分析において、10種類のバイオマーカーを用いて患者の予後スコアを予測する問題を考える。バイオマーカー間に強い相関がある場合、どのように主成分回帰を適用するか、線形代数の観点から説明せよ。

**解答**：
1. 10種類のバイオマーカーからなるデータ行列 $\mathbf{X}$ を中心化し、共分散行列 $\boldsymbol{\Sigma} = \frac{1}{n} \mathbf{X}_c^T \mathbf{X}_c$ を計算
2. 共分散行列の固有値分解 $\boldsymbol{\Sigma} = \mathbf{E} \boldsymbol{\Lambda} \mathbf{E}^T$ を行い、固有値 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{10}$ と対応する固有ベクトル $\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_{10}$ を求める
3. スクリープロットや累積寄与率に基づいて、情報の大部分（例：85%）を説明する主成分数 $k$ を決定
4. 主成分スコア行列 $\mathbf{Z} = \mathbf{X}_c \mathbf{E}$ を計算し、最初の $k$ 列 $\mathbf{Z}_k$ を抽出
5. $\mathbf{Z}_k$ を説明変数として線形回帰モデル $\mathbf{y} = \mathbf{Z}_k \boldsymbol{\gamma} + \boldsymbol{\epsilon}$ を構築
6. 得られた係数 $\boldsymbol{\gamma}$ を元の説明変数に変換：$\boldsymbol{\beta} = \mathbf{E}_k \boldsymbol{\gamma}$

この手法により、バイオマーカー間の相関による多重共線性問題を回避し、ノイズの影響を減らした安定した予測モデルを構築できます。各主成分は互いに直交しているため、回帰係数の推定が安定し、予測の分散も低減されます。

## 5. Pythonによる実装と可視化

### 5.1 健康データの主成分分析と結果の解釈

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2

# サンプル健康データの生成（実際の授業ではCSVファイルを読み込む）
np.random.seed(42)
n_samples = 100
# 血圧、心拍数、体温、BMI、血糖値の5つの指標
data = np.random.randn(n_samples, 5)
# 相関を持たせる
data[:, 0] = data[:, 0] + data[:, 1] * 0.8  # 血圧と心拍数に相関
data[:, 3] = data[:, 3] + data[:, 4] * 0.6  # BMIと血糖値に相関

# データフレームの作成
columns = ['血圧', '心拍数', '体温', 'BMI', '血糖値']
df = pd.DataFrame(data, columns=columns)

# データの標準化（平均0、分散1）
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# PCAの実行（共分散行列の固有値分解を内部で実行）
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# 固有値（分散）と寄与率の計算
eigenvalues = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 結果の表示
print("固有値（各主成分の分散）:", eigenvalues)
print("寄与率:", explained_variance_ratio)
print("累積寄与率:", cumulative_variance_ratio)
print("\n固有ベクトル（主成分の方向）:")
print(pca.components_)

# 主成分負荷量の計算 L = E * Λ^(1/2)
loadings = pca.components_.T * np.sqrt(eigenvalues)
loadings_df = pd.DataFrame(loadings, index=columns, 
                           columns=[f'PC{i+1}' for i in range(len(eigenvalues))])
print("\n主成分負荷量:")
print(loadings_df)

# スクリープロットの作成
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
plt.xlabel('主成分番号')
plt.ylabel('固有値（分散）')
plt.title('スクリープロット')
plt.grid(True)

# 固有値の減少率を計算してプロット
plt.subplot(1, 2, 2)
eigen_ratio = [1.0] + [(eigenvalues[i] - eigenvalues[i+1])/eigenvalues[i] 
                        for i in range(len(eigenvalues)-1)]
plt.plot(range(1, len(eigenvalues) + 1), eigen_ratio, 'ro-')
plt.xlabel('主成分番号')
plt.ylabel('固有値の減少率')
plt.title('固有値の減少率')
plt.grid(True)
plt.tight_layout()
plt.show()

# 累積寄与率のプロット
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'go-')
plt.xlabel('主成分番号')
plt.ylabel('累積寄与率')
plt.title('累積寄与率')
plt.axhline(y=0.8, color='r', linestyle='--', label='80%閾値')
plt.grid(True)
plt.legend()
plt.show()

# バイプロットの作成
plt.figure(figsize=(10, 8))
# データポイントのプロット（主成分スコア）
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)

# 変数ベクトルのプロット（主成分負荷量）
for i, (x, y) in enumerate(zip(loadings[:, 0], loadings[:, 1])):
    plt.arrow(0, 0, x * 3, y * 3, head_width=0.1, head_length=0.1, fc='r', ec='r')
    plt.text(x * 3.1, y * 3.1, columns[i], color='r')

plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel(f'第1主成分 ({explained_variance_ratio[0]:.2%})')
plt.ylabel(f'第2主成分 ({explained_variance_ratio[1]:.2%})')
plt.title('PCAバイプロット')
plt.grid(True)
plt.tight_layout()
plt.show()

# マハラノビス距離による外れ値検出
def mahalanobis_distance(pca_result, eigenvalues):
    # 各データポイントの主成分スコアを固有値で割って二乗し、合計する
    md_squared = np.sum((pca_result ** 2) / eigenvalues, axis=1)
    return md_squared

# 主成分空間でのマハラノビス距離計算
md_squared = mahalanobis_distance(pca_result, eigenvalues)

# カイ二乗分布の閾値（自由度は特徴量の数、有意水準0.01）
threshold = chi2.ppf(0.99, df=5)

# 外れ値の検出
outliers = np.where(md_squared > threshold)[0]
print("\n外れ値のインデックス:", outliers)

# マハラノビス距離のプロット
plt.figure(figsize=(10, 6))
plt.scatter(range(len(md_squared)), md_squared, alpha=0.7)
plt.axhline(y=threshold, color='r', linestyle='--', 
            label=f'閾値 (99%): {threshold:.2f}')
plt.xlabel('データポイント')
plt.ylabel('マハラノビス距離の二乗')
plt.title('マハラノビス距離による外れ値検出')
plt.legend()
plt.grid(True)
plt.show()
```

### 5.2 主成分回帰の詳細実装と線形代数的解釈

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# サンプルデータの生成（健康データと健康スコア）
np.random.seed(42)
n_samples = 100
n_features = 5
X = np.random.randn(n_samples, n_features)
# 相関を持たせる
X[:, 0] = X[:, 0] + X[:, 1] * 0.8
X[:, 3] = X[:, 3] + X[:, 4] * 0.6

# 真の係数ベクトル
true_coef = np.array([0.2, -0.1, 0.05, -0.3, -0.25])
# 健康スコア = X * true_coef + ノイズ
y = X.dot(true_coef) + np.random.randn(n_samples) * 0.1

# データの標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 共分散行列の計算と固有値分解の演示
cov_matrix = np.cov(X_train.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
# 固有値と固有ベクトルを降順に並び替え
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("共分散行列:")
print(cov_matrix)
print("\n固有値:")
print(eigenvalues)
print("\n固有ベクトル（列）:")
print(eigenvectors)

# 各主成分数での主成分回帰の性能評価
n_components_range = range(1, X_scaled.shape[1] + 1)
pcr_results = []
pcr_r2 = []
ols_mse = None  # 通常の最小二乗法のMSE
ols_r2 = None   # 通常の最小二乗法のR²

plt.figure(figsize=(12, 10))

for i, n_components in enumerate(n_components_range):
    # PCAによる次元削減
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # 線形回帰モデルの訓練
    lr = LinearRegression()
    lr.fit(X_train_pca, y_train)
    
    # テストデータでの予測
    y_pred = lr.predict(X_test_pca)
    
    # モデル評価（平均二乗誤差とR²）
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pcr_results.append(mse)
    pcr_r2.append(r2)
    
    # 主成分回帰の結果を可視化
    plt.subplot(2, 3, i+1)
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('実測値')
    plt.ylabel('予測値')
    plt.title(f'主成分数: {n_components}, R²: {r2:.3f}')
    
    # 通常の最小二乗法（OLS）
    if i == 0:
        lr_ols = LinearRegression()
        lr_ols.fit(X_train, y_train)
        y_pred_ols = lr_ols.predict(X_test)
        ols_mse = mean_squared_error(y_test, y_pred_ols)
        ols_r2 = r2_score(y_test, y_pred_ols)

plt.tight_layout()
plt.show()

# MSEとR²の比較
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(n_components_range, pcr_results, 'bo-')
plt.axhline(y=ols_mse, color='r', linestyle='--', label=f'OLS: {ols_mse:.4f}')
plt.xlabel('主成分数')
plt.ylabel('平均二乗誤差 (MSE)')
plt.title('主成分回帰の性能評価 (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(n_components_range, pcr_r2, 'go-')
plt.axhline(y=ols_r2, color='r', linestyle='--', label=f'OLS: {ols_r2:.4f}')
plt.xlabel('主成分数')
plt.ylabel('決定係数 (R²)')
plt.title('主成分回帰の性能評価 (R²)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 最適な主成分数での主成分回帰モデル
best_n_components = np.argmin(pcr_results) + 1
print(f"\n最適な主成分数: {best_n_components}")

# 最適なモデルの構築
pca_best = PCA(n_components=best_n_components)
X_train_pca_best = pca_best.fit_transform(X_train)
X_test_pca_best = pca_best.transform(X_test)

lr_best = LinearRegression()
lr_best.fit(X_train_pca_best, y_train)

# 各主成分の重要度（回帰係数）
print("\n主成分回帰モデルの係数（主成分空間）:")
print(lr_best.coef_)

# 元の変数空間での係数の計算
# β = Eₖ × γ（元の空間での係数 = 固有ベクトル行列 × 主成分空間での係数）
original_coef = pca_best.components_.T.dot(lr_best.coef_)
print("\n元の変数空間での係数（変換後）:")
print(original_coef)
print("\n真の係数:")
print(true_coef)

# 主成分回帰の数学的解釈の可視化
plt.figure(figsize=(10, 8))
# 第1主成分と第2主成分のみで可視化
if n_features >= 2:
    # データポイントの散布図
    plt.scatter(X_train_pca_best[:, 0], X_train_pca_best[:, 1] if best_n_components > 1 else np.zeros_like(X_train_pca_best[:, 0]), 
                c=y_train, cmap='viridis', alpha=0.7)
    
    # 主成分空間での回帰平面の可視化
    if best_n_components > 1:
        x_range = np.linspace(X_train_pca_best[:, 0].min(), X_train_pca_best[:, 0].max(), 50)
        y_range = np.linspace(X_train_pca_best[:, 1].min(), X_train_pca_best[:, 1].max(), 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        Z_grid = lr_best.intercept_ + lr_best.coef_[0] * X_grid + lr_best.coef_[1] * Y_grid
        
        plt.contourf(X_grid, Y_grid, Z_grid, alpha=0.2, cmap='viridis')
        
    plt.colorbar(label='健康スコア')
    plt.xlabel('第1主成分')
    plt.ylabel('第2主成分' if best_n_components > 1 else '')
    plt.title('主成分空間における回帰モデルの可視化')
    plt.grid(True)
    plt.show()
```

## 6. 演習問題

### 6.1 基本問題

1. あるデータセットの固有値が $\lambda = [4.2, 1.8, 0.6, 0.3, 0.1]$ であるとき、以下を計算せよ：

   a) 各主成分の寄与率（数式と計算過程を示すこと）

   b) 累積寄与率（数式と計算過程を示すこと）

   c) 80%の分散を説明するために必要な主成分数（計算過程を示すこと）

2. 次の主成分負荷量行列を線形代数の観点から解釈せよ：
   ```
           PC1    PC2
   身長    0.85   0.10
   体重    0.90   0.05
   BMI     0.80   0.15
   血圧    0.20   0.75
   心拍数  0.15   0.80
   ```
   特に、各主成分がどのような意味を持つか、また変数間の関係性について考察せよ。

3. マハラノビス距離を以下の観点から説明せよ：

   a) 線形代数的な定義（数式で表現）

   b) ユークリッド距離との違い（数学的に説明）
   
   c) 共分散行列の固有値分解を用いた効率的な計算方法
   
   d) 多変量正規分布におけるマハラノビス距離の統計的性質
   
   e) 外れ値検出への応用方法（理論的根拠を含む）

4. 主成分回帰について以下の問いに答えよ：

   a) 主成分回帰の数学的定式化（行列表記を用いて）

   b) $k$個の主成分を用いた場合の元の説明変数に対する回帰係数の導出

   c) 主成分回帰が多重共線性問題を解決できる理論的根拠

   d) 通常の最小二乗法と比較した場合の理論的な特性（バイアスとバリアンスのトレードオフの観点から）

### 6.2 応用問題

1. ある健康調査データには10個のバイオマーカー（血圧、心拍数、コレステロール値など）が含まれている。これらの変数間には強い相関がある。このデータに主成分分析を適用した結果、最初の3つの主成分で全分散の85%が説明された。主成分負荷量は以下の通りである：

   ```
              PC1    PC2    PC3
   血圧        0.82   0.15  -0.10
   心拍数      0.78   0.20  -0.15
   総コレステロール  0.25   0.75   0.10
   LDLコレステロール 0.20   0.80   0.15
   HDLコレステロール -0.10  -0.25   0.85
   中性脂肪    0.30   0.70   0.20
   空腹時血糖  0.75  -0.15   0.30
   BMI         0.80  -0.10   0.15
   ウエスト周囲径   0.85  -0.05   0.10
   体脂肪率    0.75  -0.10   0.20
   ```

   a) 各主成分の意味を解釈せよ（線形代数的根拠を示しながら）
   
   b) この結果から得られる健康状態の主要な因子について考察せよ
   
   c) このデータの次元削減による臨床的な利点について論じよ
   
   d) 次元削減後のデータを用いて患者をグループ化する方法を提案せよ

2. 糖尿病リスク予測のための主成分回帰モデルを考える。説明変数には、BMI、空腹時血糖値、HbA1c、年齢、血圧など15の変数があり、強い多重共線性が疑われる。以下の手順に従って分析を行い、各ステップの数学的根拠を詳細に説明せよ：

   a) 主成分分析の理論的背景と、なぜ多重共線性の問題を解決できるかを説明
   
   b) 最適な主成分数を決定するための統計的手法を3つ挙げ、その数理的基礎
   を説明
   
   c) 主成分回帰モデルの推定量の統計的性質（バイアス、分散、一貫性など）を導出
   
   d) 主成分回帰と他の正則化手法（リッジ回帰、Lasso）との理論的関係を説明
   
   e) 以下のPythonコードのテンプレートを完成させ、主成分回帰モデルを実装せよ：

   ```python
   def principal_component_regression(X, y, n_components=None, test_size=0.3):
       """
       主成分回帰を実装する関数
       
       パラメータ:
       X: 説明変数の行列
       y: 目的変数のベクトル
       n_components: 使用する主成分の数（Noneの場合は最適な数を自動選択）
       test_size: テストデータの割合
       
       戻り値:
       best_model: 最適な主成分回帰モデル
       original_coef: 元の変数空間での回帰係数
       best_n_components: 選択された主成分の数
       """
       # ここにコードを記述
       pass
   ```

3. バイプロットの数理的基礎と解釈を以下の観点から詳細に説明せよ：

   a) バイプロットの行列表現（データ点と変数ベクトルの両方）

   b) 変数ベクトルの方向と長さがどのような線形代数的意味を持つか

   c) 変数ベクトル間の角度と変数間の相関係数の関係（数学的証明を含む）

   d) データポイントの分布パターンから主成分空間における構造をどのように読み取るか
   
   e) 多次元データの視覚化におけるバイプロットの限界と、その問題を克服するための方法

## 7. よくある質問と解答

**Q1: 主成分数はどのように決定すればよいですか？**

A1: 主成分数の決定には主に以下の3つの方法があり、それぞれに数学的根拠があります：

1. **スクリープロットの「肘」の位置**：固有値の減少率 $(\lambda_j - \lambda_{j+1})/\lambda_j$ が急激に小さくなる点を特定します。これは、追加の主成分がデータの構造をほとんど説明しなくなる点を示しています。この点は、固有値のグラフが「肘」のような形状を示す位置に対応します。

2. **カイザー基準（固有値>1）**：相関行列を用いた場合、固有値が1未満の主成分は元の変数1つ分よりも情報量が少ないため除外します。数学的には、標準化された変数の分散は1であるため、固有値が1より大きい主成分は、少なくとも1つの変数分の情報を持っていることを意味します。

3. **累積寄与率による閾値**：$\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{p} \lambda_i} \geq \alpha$ となる最小の $k$ を選択します（$\alpha$ は通常0.8〜0.9）。これは、選択された主成分が全体の分散の少なくとも $\alpha \times 100\%$ を説明することを保証します。

これらの方法を組み合わせて、分析の目的や計算リソースを考慮して決定します。また、交差検証や情報量基準（AIC、BIC）などの統計的手法も利用できます。

**Q2: 主成分の解釈が難しい場合はどうすればよいですか？**

A2: 主成分の解釈が難しい場合は以下のアプローチが有効です：

1. **主成分負荷量の構造的パターン**：主成分負荷量行列 $\mathbf{L} = \mathbf{E}\boldsymbol{\Lambda}^{1/2}$ を詳細に分析し、各主成分に対して最も高い（絶対値の）負荷量を持つ変数のグループを特定します。これらのグループは、潜在的な因子構造を示唆しています。

2. **バリマックス回転などの回転手法**：固有ベクトル行列 $\mathbf{E}$ に直交回転行列 $\mathbf{R}$ を適用し、解釈しやすい構造を得ます：$\mathbf{E}' = \mathbf{E}\mathbf{R}$。これにより、各主成分が少数の変数と強く関連するようになります。

3. **バイプロットの幾何学的解釈**：バイプロットでは、変数ベクトルのクラスタリングパターンが変数間の関係性を視覚化します。同じ方向を向いたベクトルグループは、関連する変数グループを示唆します。

4. **ドメイン知識の活用**：健康データの場合、第1主成分が「全体的な代謝健康」を表し、第2主成分が「心血管系の健康状態」を表すというように、専門知識に基づいて解釈を行います。

また、解釈が難しい場合は、「因子分析」など、潜在変数の解釈可能性を重視した他の手法を検討することも有効です。

**Q3: 主成分分析は質的変数にも適用できますか？**

A3: 主成分分析は基本的に量的変数（連続変数）のために設計されており、その数学的基礎は共分散行列または相関行列の固有値分解に基づいています。質的変数（カテゴリ変数）を含む場合は、以下の理論的アプローチを検討してください：

1. **二値変数の場合**：二値変数（0/1）は、テトラコリック相関係数や点二列相関係数を用いて相関行列を構築し、その後でPCAを適用できます。

2. **多値カテゴリカル変数の場合**：
   - **多重対応分析（MCA）**：質的変数専用の次元削減手法で、カテゴリカル変数をダミー変数化し、特殊な標準化を適用した後に対応分析（Correspondence Analysis）を行います。
   - **最適尺度法（Optimal Scaling）**：カテゴリカル変数を数値変数に変換し、その後PCAを適用する方法です。

3. **混合データの場合**：
   - **因子分析混合データ法（FAMD）**：量的変数と質的変数の両方を同時に扱える手法で、量的変数には標準化を、質的変数には多重対応分析の原理を適用します。
   - **非線形PCA**：最適尺度法を用いて質的変数を量的変数に変換し、通常のPCAを適用する手法です。

これらの方法はいずれも、通常のPCAの線形代数的枠組みを拡張または修正したものであり、適切な変換と標準化を通じて質的変数を扱えるようにしています。

**Q4: 異なるスケールの変数がある場合、PCAを適用する前に標準化すべきですか？**

A4: 異なるスケールの変数がある場合は通常、PCAを適用する前に標準化することが重要です。線形代数的には、以下の理由があります：

1. **共分散行列vs相関行列**：標準化しない場合、PCAは共分散行列 $\boldsymbol{\Sigma} = \frac{1}{n}\mathbf{X}_c^T\mathbf{X}_c$ の固有値分解に基づきます。標準化すると、PCAは相関行列 $\mathbf{R} = \mathbf{D}^{-1/2}\boldsymbol{\Sigma}\mathbf{D}^{-1/2}$ の固有値分解に基づきます（$\mathbf{D}$ は共分散行列の対角成分を対角成分とする対角行列）。

2. **変数の重み付け**：標準化しない場合、大きな分散を持つ変数が主成分に不釣り合いに影響を与えます。数学的には、共分散行列の固有値と固有ベクトルは、分散の大きな変数に支配されます。

3. **スケール不変性**：標準化によって、PCAの結果は変数の測定単位に依存しなくなります。これは、固有ベクトルが変数の相対的な重要性を反映するようになるためです。

ただし、以下の場合は標準化を行わないことも考慮できます：
- すべての変数が同じ単位で測定されている場合
- 変数間の絶対的な分散の違いが重要な情報である場合
- 物理的な意味のある主成分を抽出したい場合（例：形状分析）

決定は、データの性質と分析の目的に基づいて行うべきです。

**Q5: マハラノビス距離と主成分分析はどのように関連していますか？**

A5: マハラノビス距離と主成分分析は線形代数的に密接に関連しています：

1. **共通の数学的基礎**：両方とも共分散行列 $\boldsymbol{\Sigma}$ の構造を利用しています。PCAは共分散行列の固有値分解 $\boldsymbol{\Sigma} = \mathbf{E}\boldsymbol{\Lambda}\mathbf{E}^T$ に基づき、マハラノビス距離はその逆行列 $\boldsymbol{\Sigma}^{-1}$ を使用します。

2. **主成分空間での簡略化**：共分散行列の固有値分解を用いると、マハラノビス距離の二乗は以下のように表現できます：
   
   $D^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$
   $= (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{E} \boldsymbol{\Lambda}^{-1} \mathbf{E}^T (\mathbf{x} - \boldsymbol{\mu})$
   $= \mathbf{z}^T \boldsymbol{\Lambda}^{-1} \mathbf{z} = \sum_{j=1}^{p} \frac{z_j^2}{\lambda_j}$
   
   ここで、$\mathbf{z} = \mathbf{E}^T(\mathbf{x} - \boldsymbol{\mu})$ は主成分スコアです。

3. **幾何学的解釈**：PCAは、データの分散が最大となる方向（固有ベクトル）を特定し、マハラノビス距離は、その分散の大きさ（固有値）で標準化された距離を計算します。つまり、マハラノビス距離は主成分空間での標準化されたユークリッド距離として解釈できます。

4. **次元削減との関係**：実際の応用では、小さな固有値に対応する次元を無視することで、安定したマハラノビス距離の計算が可能になります。これは、PCAによる次元削減の原理と一致しています。

主成分分析後のマハラノビス距離計算は特に外れ値検出に有用で、統計的に正確な多変量データの異常検知を可能にします。理論的には、このアプローチは多変量正規分布の等確率曲面に基づいています。