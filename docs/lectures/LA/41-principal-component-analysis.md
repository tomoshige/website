# 線形代数学 第41回 講義ノート：主成分分析（PCA）の導入

## 1. 講義情報と予習ガイド

**講義回**: 第41回  
**テーマ**: 主成分分析（PCA）の導入  
**関連項目**: 固有値・固有ベクトル、分散・共分散、相関係数、対称行列の対角化  
**予習すべき内容**: 
- 第33-35回の固有値・固有ベクトルと対角化の内容を復習
- 第7-8回の分散・共分散・相関係数の計算方法を確認
- 第36回の2次形式と正定値・半正定値の概念を理解

## 2. 学習目標

1. 主成分分析の目的と基本的な考え方を理解する
2. 分散共分散行列と相関行列の性質を理解し、それらの固有値・固有ベクトルの意味を説明できる
3. 主成分の数学的な導出方法を理解し、計算できるようになる
4. 分散共分散行列と相関行列に基づく主成分分析の違いを理解する
5. Pythonを用いて主成分分析を実装し、結果を視覚的に解釈できるようになる

## 3. 基本概念

### 3.1 主成分分析（PCA）とは

> **定義**: 主成分分析（Principal Component Analysis, PCA）は、多次元データの情報をできるだけ保持しながら、より少ない次元に圧縮する手法である。具体的には、データの分散が最大となる方向（主成分）を順次見つけることで、データの主要な構造を抽出する。

主成分分析は以下のような目的で利用されます：

1. **次元削減**：高次元データを低次元に圧縮し、可視化や計算効率の向上を図る
2. **情報抽出**：データの最も重要な特徴や変動を捉える
3. **ノイズ除去**：データに含まれるノイズの影響を軽減する
4. **多重共線性の解消**：相関の高い変数間の問題を解決する

### 3.2 データの表現と前処理

$n$個のサンプルと$p$個の変数からなるデータ行列を次のように表します：

$$X = \begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{pmatrix}$$

PCを行う前に、通常はデータを中心化（各変数の平均を0にする）します：

$$X_{centered} = X - \mathbf{1}\boldsymbol{\mu}^T$$

ここで、$\mathbf{1}$は全ての要素が1の$n$次元ベクトル、$\boldsymbol{\mu}$は各変数の平均値を含む$p$次元ベクトルです。

さらに、各変数の単位が異なる場合は標準化（各変数の標準偏差を1にする）を行います：

$$X_{standardized} = (X - \mathbf{1}\boldsymbol{\mu}^T)D^{-1}$$

ここで、$D$は各変数の標準偏差を対角成分に持つ対角行列です。

### 3.3 分散共分散行列と相関行列

> **定義（分散共分散行列）**: データ行列$X$（中心化済み）の分散共分散行列$S$は次のように定義される：
> $$S = \frac{1}{n-1}X^TX$$

> **定義（相関行列）**: データ行列$X$（標準化済み）の相関行列$R$は次のように定義される：
> $$R = \frac{1}{n-1}X^TX$$

分散共分散行列の対角成分は各変数の分散、非対角成分は変数間の共分散を表します。相関行列の対角成分はすべて1、非対角成分は変数間の相関係数を表します。

分散共分散行列と相関行列は以下の性質を持ちます：

1. 対称行列である（$S = S^T$, $R = R^T$）
2. 半正定値行列である（すべての固有値は非負）
3. データが標準化されていれば、分散共分散行列は相関行列に一致する

## 4. 理論と手法

### 4.1 主成分の数学的導出

主成分分析の目的は、データの分散が最大となる方向（主成分）を見つけることです。

まず、中心化されたデータ$X$に対して、単位ベクトル$\mathbf{w}$方向への射影を考えます：

$$\mathbf{z} = X\mathbf{w}$$

この射影$\mathbf{z}$の分散は次のように表されます：

$$\text{Var}(\mathbf{z}) = \frac{1}{n-1}\mathbf{z}^T\mathbf{z} = \frac{1}{n-1}\mathbf{w}^TX^TX\mathbf{w} = \mathbf{w}^TS\mathbf{w}$$

ここで、$S$は分散共分散行列です。

第一主成分を求めるには、$\mathbf{w}$の長さが1という制約の下で、$\mathbf{w}^TS\mathbf{w}$を最大化する問題を解きます：

$$\max_{\mathbf{w}} \mathbf{w}^TS\mathbf{w} \quad \text{subject to} \quad \mathbf{w}^T\mathbf{w} = 1$$

この最適化問題はラグランジュ乗数法で解くことができます。ラグランジアンは次のようになります：

$$L(\mathbf{w}, \lambda) = \mathbf{w}^TS\mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

$\mathbf{w}$で偏微分して0とおくと：

$$\frac{\partial L}{\partial \mathbf{w}} = 2S\mathbf{w} - 2\lambda\mathbf{w} = 0$$

これを整理すると：

$$S\mathbf{w} = \lambda\mathbf{w}$$

これは、$\mathbf{w}$が$S$の固有ベクトルであり、$\lambda$が対応する固有値であることを意味します。$\mathbf{w}^TS\mathbf{w} = \lambda$なので、分散を最大化するには最大の固有値に対応する固有ベクトルを選ぶ必要があります。

したがって、第一主成分は分散共分散行列$S$の最大固有値に対応する固有ベクトルとなります。

同様に、第二主成分以降は、それぞれ次に大きい固有値に対応する固有ベクトルとなり、それらは互いに直交します。

### 4.2 分散共分散行列と相関行列に基づくPCA

主成分分析は、分散共分散行列に基づく場合と相関行列に基づく場合の2種類があります：

**分散共分散行列に基づくPCA**：
- 元のデータの分散をそのまま反映する
- 分散の大きい変数が主成分に強く影響する
- 変数の単位が同じ場合や、単位の違いが意味を持つ場合に適している

**相関行列に基づくPCA**：
- すべての変数を標準化してから分析を行う
- すべての変数が等しく扱われる
- 変数の単位が異なる場合に適している

相関行列を用いる主な理由：
1. 変数の単位が異なる場合、大きな値を持つ変数が主成分に支配的な影響を与えてしまう
2. 変数間のスケールの違いを排除し、純粋な相関関係に基づいて分析できる
3. 結果の解釈がより直感的になることが多い

### 4.3 主成分の幾何学的解釈

2次元の例で考えてみましょう。次の図は、2変数のデータに対する主成分の方向を示しています：

![主成分の幾何学的解釈](https://i.imgur.com/example.png)

- 第一主成分（PC1）：データの分散が最大となる方向
- 第二主成分（PC2）：第一主成分と直交し、残りの分散が最大となる方向

多次元データの場合、主成分はデータの「主軸」を表し、それぞれの主成分はデータの変動の一部を説明します。固有値の大きさは、各主成分が説明する分散の量を表します。

## 5. Pythonによる実装と可視化

### 5.1 基本的なPCA実装

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# サンプルデータの生成（2つの相関のある変数）
np.random.seed(42)
n_samples = 100
x = np.random.normal(0, 1, n_samples)
y = x * 0.8 + np.random.normal(0, 0.6, n_samples)
data = np.column_stack((x, y))

# データの可視化
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.7)
plt.xlabel('変数1')
plt.ylabel('変数2')
plt.title('元のデータ')
plt.grid(True)
plt.show()

# 手動でPCAを実装
# 1. データを中心化
data_centered = data - np.mean(data, axis=0)

# 2. 分散共分散行列の計算
cov_matrix = np.cov(data_centered, rowvar=False)
print("分散共分散行列:")
print(cov_matrix)

# 3. 固有値と固有ベクトルの計算
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\n固有値:")
print(eigenvalues)
print("\n固有ベクトル（列ベクトルとして）:")
print(eigenvectors)

# 4. 固有値を降順にソート
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 5. 主成分の方向を可視化
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.7)

# 第一主成分
plt.arrow(np.mean(data[:, 0]), np.mean(data[:, 1]), 
          eigenvectors[0, 0] * eigenvalues[0], 
          eigenvectors[1, 0] * eigenvalues[0],
          head_width=0.1, head_length=0.1, fc='red', ec='red', 
          label='第一主成分')

# 第二主成分
plt.arrow(np.mean(data[:, 0]), np.mean(data[:, 1]), 
          eigenvectors[0, 1] * eigenvalues[1], 
          eigenvectors[1, 1] * eigenvalues[1],
          head_width=0.1, head_length=0.1, fc='green', ec='green', 
          label='第二主成分')

plt.xlabel('変数1')
plt.ylabel('変数2')
plt.title('主成分の方向')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

# 6. 主成分への射影
pc_scores = np.dot(data_centered, eigenvectors)

plt.figure(figsize=(10, 6))
plt.scatter(pc_scores[:, 0], pc_scores[:, 1], alpha=0.7)
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.title('主成分空間に射影したデータ')
plt.grid(True)
plt.axis('equal')
plt.show()
```

### 5.2 scikit-learnを用いたPCA

```python
# scikit-learnを使用したPCA
# データの標準化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# PCAの実行
pca = PCA()
pca_result = pca.fit_transform(data_scaled)

# 結果の確認
print("\nscikit-learnによる分散説明率:")
print(pca.explained_variance_ratio_)
print("\nscikit-learnによる累積分散説明率:")
print(np.cumsum(pca.explained_variance_ratio_))

# 結果の可視化
plt.figure(figsize=(12, 5))

# スクリープロット
plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_, 'o-', linewidth=2)
plt.title('スクリープロット')
plt.xlabel('主成分番号')
plt.ylabel('分散説明率')
plt.grid(True)

# 累積分散説明率
plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'o-', linewidth=2)
plt.title('累積分散説明率')
plt.xlabel('主成分番号')
plt.ylabel('累積分散説明率')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### 5.3 健康データへの応用例

```python
# 健康データのサンプル（BMI、血圧、血糖値、コレステロールなど）
np.random.seed(42)
n_samples = 100

# データの生成（相関を持たせる）
bmi = np.random.normal(22, 3, n_samples)
systolic_bp = 120 + bmi * 2 + np.random.normal(0, 10, n_samples)
diastolic_bp = 80 + bmi * 1.5 + np.random.normal(0, 8, n_samples)
blood_sugar = 100 + bmi * 1.8 + np.random.normal(0, 15, n_samples)
cholesterol = 200 + bmi * 3 + blood_sugar * 0.2 + np.random.normal(0, 20, n_samples)

# データフレームの作成
health_data = pd.DataFrame({
    'BMI': bmi,
    '収縮期血圧': systolic_bp,
    '拡張期血圧': diastolic_bp,
    '血糖値': blood_sugar,
    'コレステロール': cholesterol
})

print("健康データのサンプル:")
print(health_data.head())

# 相関行列の確認
correlation_matrix = health_data.corr()
print("\n相関行列:")
print(correlation_matrix)

# データの標準化
scaler = StandardScaler()
health_data_scaled = scaler.fit_transform(health_data)

# PCAの実行
pca = PCA()
health_pca_result = pca.fit_transform(health_data_scaled)

# 結果の確認
print("\n主成分の分散説明率:")
print(pca.explained_variance_ratio_)
print("\n累積分散説明率:")
print(np.cumsum(pca.explained_variance_ratio_))

# 主成分の負荷量（各変数の寄与度）
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_df = pd.DataFrame(
    loadings, 
    columns=[f'PC{i+1}' for i in range(loadings.shape[1])],
    index=health_data.columns
)
print("\n主成分負荷量:")
print(loading_df)

# バイプロットの作成
plt.figure(figsize=(10, 8))
# データポイントのプロット
plt.scatter(health_pca_result[:, 0], health_pca_result[:, 1], alpha=0.7)

# 変数ベクトルのプロット
for i, (name, row) in enumerate(loading_df.iloc[:, :2].iterrows()):
    plt.arrow(0, 0, row['PC1']*3, row['PC2']*3, head_width=0.1, head_length=0.1, fc='red', ec='red')
    plt.text(row['PC1']*3.2, row['PC2']*3.2, name, color='red')

plt.xlabel(f'第一主成分 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'第二主成分 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('健康データの主成分分析：バイプロット')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()
```

## 6. 演習問題

### 6.1 基本問題

1. 次のデータに対して主成分分析を手計算で行い、第一主成分と第二主成分を求めなさい。

$$X = \begin{pmatrix}
1 & 3 \\
2 & 5 \\
3 & 4 \\
4 & 6
\end{pmatrix}$$

2. 主成分分析において、固有値が次のように得られた：λ₁ = 4.2, λ₂ = 1.8, λ₃ = 0.7, λ₄ = 0.3。このとき、各主成分の分散説明率と累積分散説明率を求めなさい。

3. 以下の相関行列に対して、固有値と固有ベクトルを求め、第一主成分と第二主成分の方向を特定しなさい。

$$R = \begin{pmatrix}
1.0 & 0.7 & 0.3 \\
0.7 & 1.0 & 0.5 \\
0.3 & 0.5 & 1.0
\end{pmatrix}$$

4. 分散共分散行列と相関行列に基づくPCAの違いについて説明し、どのような場合にどちらを選ぶべきか述べなさい。

### 6.2 応用問題

1. 以下のPythonコードによって生成される3次元データに対して主成分分析を行い、結果を可視化してください。第一主成分と第二主成分の分散説明率は何％ですか？また、負荷量から各主成分の意味を解釈してください。

```python
import numpy as np

np.random.seed(42)
n = 100
x = np.random.normal(0, 1, n)
y = x * 0.8 + np.random.normal(0, 0.5, n)
z = x * 0.6 + y * 0.5 + np.random.normal(0, 0.4, n)
data_3d = np.column_stack((x, y, z))
```

2. 健康データサイエンスの文脈で、1000人分の身長、体重、BMI、ウエスト/ヒップ比、体脂肪率のデータがあるとします。これらの変数は互いに相関があります。このデータに主成分分析を適用する意義と、得られる可能性のある主成分の解釈について説明してください。さらに、この分析結果が健康リスク評価にどのように活用できるか考察しなさい。

3. ある遺伝子発現データに対して主成分分析を行った結果、多くの主成分が必要となりました（20個の主成分でようやく80%の分散を説明）。一方、別の生理学的測定データでは2つの主成分で90%以上の分散を説明できました。これらの結果からそれぞれのデータの性質についてどのようなことが言えるでしょうか？データの複雑さと主成分の数の関係について考察しなさい。

## 7. よくある質問と解答

### Q1: 主成分分析はどのような場合に使うべきですか？
A1: 主成分分析は以下のような場合に特に有用です：
- 多次元データを視覚化したい場合（2次元や3次元に圧縮）
- 変数間に強い相関があり、次元削減が可能と思われる場合
- データのノイズを減らしたい場合
- 線形回帰などの分析で多重共線性の問題がある場合
- データの主要な変動要因を特定したい場合

### Q2: 主成分の数はどのように決めれば良いですか？
A2: 主成分の数を決める一般的な方法には以下があります：
- 累積分散説明率が一定の閾値（例えば80%や90%）を超える点で切る
- スクリープロット（固有値のプロット）の「ひじ」の位置で切る
- カイザー基準：固有値が1.0より大きい主成分のみを選ぶ（相関行列を使用した場合）
- 交差検証などの統計的手法で評価する
- 解釈可能性も考慮する（選択した主成分が意味のある解釈が可能か）

### Q3: 分散共分散行列と相関行列のどちらを使うべきですか？
A3: 一般的には以下のガイドラインがあります：
- 変数の単位が同じ（例：全て長さの測定値）で、それらの分散の大きさが意味を持つ場合は分散共分散行列を使う
- 変数の単位が異なる（例：身長、体重、血圧など）場合は相関行列を使う
- 変数のスケールが大きく異なり、結果に不当な影響を与える可能性がある場合は相関行列を使う
- より保守的なアプローチとしては、両方の方法で分析を行い、結果を比較することも有効

### Q4: 主成分分析の結果をどのように解釈すれば良いですか？
A4: 主成分分析の結果の解釈には以下の観点が重要です：
- 各主成分の分散説明率：その主成分がデータの変動をどの程度説明しているか
- 主成分負荷量：各変数が主成分にどの程度寄与しているか
- 主成分スコア：各データポイントの主成分空間における位置
- バイプロット：データポイントと変数ベクトルの両方を表示し、関係性を視覚化

特に主成分負荷量が大きい変数に注目すると、各主成分が表す潜在的な概念や特徴を解釈しやすくなります。

### Q5: 主成分分析の限界は何ですか？
A5: 主成分分析には以下のような限界があります：
- 線形な次元削減手法であり、非線形な関係を捉えることができない
- 分散の大きさに基づいて主成分を選ぶため、分類などのタスクに直接役立つとは限らない
- 外れ値に敏感である
- 結果の解釈が必ずしも直感的でない場合がある
- 変数間の関係が複雑な場合、多くの主成分が必要になり、次元削減の効果が限定的になる

これらの限界を克服するために、カーネルPCAや因子分析などの代替手法や、より高度な非線形次元削減法（t-SNE、UMAPなど）も検討する価値があります。