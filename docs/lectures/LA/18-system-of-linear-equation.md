# 第18回：連立1次方程式と2つの説明変数がある場合の線形回帰モデル

## 1. 講義情報と予習ガイド

**講義回**: 第18回  
**関連項目**: 連立1次方程式、複数説明変数を持つ線形回帰モデル  
**予習すべき内容**: 第16回で学んだ単回帰モデルの概念、第10回〜第15回での連立1次方程式の解法

## 2. 学習目標

1. 複数説明変数を持つ線形回帰モデルの数学的表現を理解する
2. 行列とベクトルを用いて線形回帰モデルを表現できるようになる
3. 最小二乗法によるパラメータ推定の原理と計算方法を習得する
4. 正規方程式の導出と解法を理解する
5. 複数説明変数を持つ線形回帰モデルの幾何学的解釈ができるようになる

## 3. 基本概念

### 3.1 線形回帰モデルの復習

単回帰モデルでは、1つの説明変数を用いて反応変数を予測します。これを数式で表すと：

> **単回帰モデル**:  
> $y = \beta_0 + \beta_1 x + \varepsilon$
>
> ここで：  
> $y$: 反応変数（予測したい変数）  
> $x$: 説明変数  
> $\beta_0$: 切片  
> $\beta_1$: 傾き（回帰係数）  
> $\varepsilon$: 誤差項  

このモデルは2次元平面上では「直線」として表現されます。

### 3.2 複数説明変数を持つ線形回帰モデル

2つの説明変数がある場合、モデルは以下のように拡張されます：

> **2変数線形回帰モデル**:  
> $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon$
>
> ここで：  
> $y$: 反応変数  
> $x_1, x_2$: 2つの説明変数  
> $\beta_0$: 切片  
> $\beta_1, \beta_2$: 各説明変数の回帰係数  
> $\varepsilon$: 誤差項  

このモデルは3次元空間では「平面」として表現されます。

より一般的に、$p$個の説明変数がある場合：

> **多変量線形回帰モデル**:  
> $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon$

### 3.3 行列表記による表現

データが$n$個の観測値を持ち、各観測値に対して$p$個の説明変数がある場合、行列表記を用いると：

> **行列形式の線形回帰モデル**:  
> $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$
>
> ここで：  
> $\mathbf{y}$: $n \times 1$の反応変数ベクトル  
> $\mathbf{X}$: $n \times (p+1)$のデザイン行列（説明変数行列）  
> $\boldsymbol{\beta}$: $(p+1) \times 1$のパラメータベクトル  
> $\boldsymbol{\varepsilon}$: $n \times 1$の誤差ベクトル  

具体的に、2つの説明変数を持つモデルの場合：

$$\mathbf{X} = 
\begin{bmatrix} 
1 & x_{11} & x_{12} \\
1 & x_{21} & x_{22} \\
\vdots & \vdots & \vdots \\
1 & x_{n1} & x_{n2}
\end{bmatrix}$$

$$\boldsymbol{\beta} = 
\begin{bmatrix} 
\beta_0 \\
\beta_1 \\
\beta_2
\end{bmatrix}$$

最初の列の1は切片$\beta_0$に対応する列です。

## 4. 理論と手法

### 4.1 最小二乗法による推定

線形回帰モデルのパラメータ推定には最小二乗法を用います。誤差の二乗和：

$$S(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - (\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2}))^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$$

を最小化する$\boldsymbol{\beta}$を求めます。

行列表記で表すと：

$$S(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

### 4.2 正規方程式の導出

$S(\boldsymbol{\beta})$を最小化するために、$\boldsymbol{\beta}$に関する偏微分を0とおきます：

$$\frac{\partial S(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}$$

これを整理すると：

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

これが**正規方程式**です。

### 4.3 正規方程式の解

正規方程式から$\boldsymbol{\beta}$の推定値を求めると：

> **最小二乗推定量**:  
> $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

ただし、$\mathbf{X}^T\mathbf{X}$が正則（逆行列が存在する）ことが必要です。これは説明変数間に完全な線形関係がない（多重共線性がない）場合に成立します。

### 4.4 幾何学的解釈

行列$\mathbf{P} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$は**射影行列**と呼ばれ、$\mathbf{y}$を$\mathbf{X}$の列空間に射影する役割を持ちます。

予測値$\hat{\mathbf{y}}$は：

$$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} = \mathbf{P}\mathbf{y}$$

実際のデータ$\mathbf{y}$と予測値$\hat{\mathbf{y}}$の差（残差）$\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$は$\mathbf{X}$の列空間に直交します。

### 4.5 2つの説明変数がある場合の幾何学的イメージ

2つの説明変数を持つ線形回帰モデルは3次元空間における平面を表します：

$$z = \beta_0 + \beta_1 x + \beta_2 y$$

各観測点$(x_i, y_i, z_i)$があり、回帰平面はこれらの点からの垂直距離の二乗和を最小にする平面です。

## 5. 具体例

### 5.1 最小二乗法による推定

線形回帰モデルのパラメータ推定には最小二乗法を用います。誤差の二乗和：

$$S(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - (\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2}))^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$$

を最小化する$\boldsymbol{\beta}$を求めます。

行列表記で表すと：

$$S(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

**具体例**：
以下の5つのデータポイントを考えてみましょう：

| i | $x_{i1}$ | $x_{i2}$ | $y_i$ |
|---|----------|----------|-------|
| 1 | 2        | 3        | 10    |
| 2 | 1        | 5        | 12    |
| 3 | 3        | 2        | 11    |
| 4 | 4        | 4        | 18    |
| 5 | 5        | 1        | 15    |

このとき、

$$\mathbf{y} = \begin{bmatrix} 10 \\ 12 \\ 11 \\ 18 \\ 15 \end{bmatrix}, \quad
\mathbf{X} = \begin{bmatrix} 
1 & 2 & 3 \\
1 & 1 & 5 \\
1 & 3 & 2 \\
1 & 4 & 4 \\
1 & 5 & 1 
\end{bmatrix}, \quad
\boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \end{bmatrix}$$

最小化したい誤差二乗和は：
$$S(\boldsymbol{\beta}) = (10 - \beta_0 - 2\beta_1 - 3\beta_2)^2 + (12 - \beta_0 - \beta_1 - 5\beta_2)^2 + \cdots + (15 - \beta_0 - 5\beta_1 - \beta_2)^2$$

### 5.2 正規方程式の導出

$S(\boldsymbol{\beta})$を最小化するために、$\boldsymbol{\beta}$に関する偏微分を0とおきます：

$$\frac{\partial S(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}$$

これを整理すると：

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

これが**正規方程式**です。

**計算例**：
先ほどの例で$\mathbf{X}^T\mathbf{X}$と$\mathbf{X}^T\mathbf{y}$を計算してみましょう。

$$\mathbf{X}^T\mathbf{X} = \begin{bmatrix} 
1 & 1 & 1 & 1 & 1 \\
2 & 1 & 3 & 4 & 5 \\
3 & 5 & 2 & 4 & 1
\end{bmatrix} \cdot
\begin{bmatrix} 
1 & 2 & 3 \\
1 & 1 & 5 \\
1 & 3 & 2 \\
1 & 4 & 4 \\
1 & 5 & 1 
\end{bmatrix} = 
\begin{bmatrix} 
5 & 15 & 15 \\
15 & 55 & 37 \\
15 & 37 & 55
\end{bmatrix}$$

$$\mathbf{X}^T\mathbf{y} = \begin{bmatrix} 
1 & 1 & 1 & 1 & 1 \\
2 & 1 & 3 & 4 & 5 \\
3 & 5 & 2 & 4 & 1
\end{bmatrix} \cdot
\begin{bmatrix} 10 \\ 12 \\ 11 \\ 18 \\ 15 \end{bmatrix} = 
\begin{bmatrix} 
66 \\
219 \\
178
\end{bmatrix}$$

正規方程式は以下のようになります：
$$\begin{bmatrix} 
5 & 15 & 15 \\
15 & 55 & 37 \\
15 & 37 & 55
\end{bmatrix} \cdot
\begin{bmatrix} 
\beta_0 \\
\beta_1 \\
\beta_2
\end{bmatrix} = 
\begin{bmatrix} 
66 \\
219 \\
178
\end{bmatrix}$$

### 5.3 正規方程式の解

正規方程式から$\boldsymbol{\beta}$の推定値を求めると：

> **最小二乗推定量**:  
> $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

ただし、$\mathbf{X}^T\mathbf{X}$が正則（逆行列が存在する）ことが必要です。これは説明変数間に完全な線形関係がない（多重共線性がない）場合に成立します。

**計算例の続き**：
先ほどの正規方程式の解を求めるには、まず$(\mathbf{X}^T\mathbf{X})^{-1}$を計算する必要があります。

$$(\mathbf{X}^T\mathbf{X})^{-1} = \begin{bmatrix} 
5 & 15 & 15 \\
15 & 55 & 37 \\
15 & 37 & 55
\end{bmatrix}^{-1}$$

行列の逆行列を計算すると：
$$(\mathbf{X}^T\mathbf{X})^{-1} \approx \begin{bmatrix} 
2.12 & -0.39 & -0.29 \\
-0.39 & 0.16 & -0.03 \\
-0.29 & -0.03 & 0.13
\end{bmatrix}$$

これを$\mathbf{X}^T\mathbf{y}$に乗じて$\hat{\boldsymbol{\beta}}$を求めます：
$$\hat{\boldsymbol{\beta}} = \begin{bmatrix} 
2.12 & -0.39 & -0.29 \\
-0.39 & 0.16 & -0.03 \\
-0.29 & -0.03 & 0.13
\end{bmatrix} \cdot
\begin{bmatrix} 
66 \\
219 \\
178
\end{bmatrix} \approx
\begin{bmatrix} 
3.24 \\
2.15 \\
1.63
\end{bmatrix}$$

したがって、$\hat{\beta}_0 \approx 3.24$、$\hat{\beta}_1 \approx 2.15$、$\hat{\beta}_2 \approx 1.63$となり、回帰式は：
$$\hat{y} = 3.24 + 2.15x_1 + 1.63x_2$$

この結果から、$x_1$が1単位増加すると$y$は約2.15単位増加し、$x_2$が1単位増加すると$y$は約1.63単位増加することがわかります。

#### 5.4 幾何学的解釈

行列$\mathbf{P} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$は**射影行列**と呼ばれ、$\mathbf{y}$を$\mathbf{X}$の列空間に射影する役割を持ちます。

予測値$\hat{\mathbf{y}}$は：

$$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} = \mathbf{P}\mathbf{y}$$

実際のデータ$\mathbf{y}$と予測値$\hat{\mathbf{y}}$の差（残差）$\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$は$\mathbf{X}$の列空間に直交します。

**具体例の続き**：
先ほどの例で射影行列$\mathbf{P}$を計算してみましょう：

$$\mathbf{P} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$$

これを用いて予測値$\hat{\mathbf{y}}$を求めると：

$$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = 
\begin{bmatrix} 
1 & 2 & 3 \\
1 & 1 & 5 \\
1 & 3 & 2 \\
1 & 4 & 4 \\
1 & 5 & 1 
\end{bmatrix} \cdot
\begin{bmatrix} 
3.24 \\
2.15 \\
1.63
\end{bmatrix} = 
\begin{bmatrix} 
11.13 \\
13.39 \\
11.67 \\
17.56 \\
14.25
\end{bmatrix}$$

実際の$\mathbf{y}$との差（残差）は：

$$\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = 
\begin{bmatrix} 
10 \\
12 \\
11 \\
18 \\
15
\end{bmatrix} - 
\begin{bmatrix} 
11.13 \\
13.39 \\
11.67 \\
17.56 \\
14.25
\end{bmatrix} = 
\begin{bmatrix} 
-1.13 \\
-1.39 \\
-0.67 \\
0.44 \\
0.75
\end{bmatrix}$$

この残差ベクトル$\mathbf{e}$は$\mathbf{X}$の列空間に直交しているため、以下が成り立ちます：

$$\mathbf{X}^T\mathbf{e} = \mathbf{0}$$

実際に確認すると：

$$\mathbf{X}^T\mathbf{e} = 
\begin{bmatrix} 
1 & 1 & 1 & 1 & 1 \\
2 & 1 & 3 & 4 & 5 \\
3 & 5 & 2 & 4 & 1
\end{bmatrix} \cdot
\begin{bmatrix} 
-1.13 \\
-1.39 \\
-0.67 \\
0.44 \\
0.75
\end{bmatrix} \approx
\begin{bmatrix} 
0 \\
0 \\
0
\end{bmatrix}$$

これは数値計算の誤差を除けば0ベクトルとなり、残差ベクトル$\mathbf{e}$が$\mathbf{X}$の列空間に直交していることを示しています。

#### 5.5 2つの説明変数がある場合の幾何学的イメージ

2つの説明変数を持つ線形回帰モデルは3次元空間における平面を表します：

$$z = \beta_0 + \beta_1 x + \beta_2 y$$

各観測点$(x_i, y_i, z_i)$があり、回帰平面はこれらの点からの垂直距離の二乗和を最小にする平面です。

**具体例**：
先ほどの5つのデータポイントを3次元空間にプロットすると：
- 点1: $(2, 3, 10)$
- 点2: $(1, 5, 12)$
- 点3: $(3, 2, 11)$
- 点4: $(4, 4, 18)$
- 点5: $(5, 1, 15)$

求めた回帰平面は：$z = 3.24 + 2.15x + 1.63y$

この平面は3次元空間内の各データポイントから垂直距離の二乗和が最小になるように配置されています。各点から平面への垂直距離は残差に対応し、その二乗和：

$$\sum_{i=1}^5 e_i^2 = (-1.13)^2 + (-1.39)^2 + (-0.67)^2 + (0.44)^2 + (0.75)^2 = 4.54$$

が最小となっています。

異なる$\beta_0$, $\beta_1$, $\beta_2$の値で定義される他のどの平面でも、この平方和は4.54より大きくなります。これが最小二乗法の本質です。

## 6. Pythonによる実装と可視化

### 6.1 2変数線形回帰モデルの実装

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# サンプルデータの生成
np.random.seed(42)
n = 100
X = np.random.rand(n, 2) * 10
beta_true = np.array([5, 2, -1])  # 真のパラメータ: 切片, beta_1, beta_2
y = beta_true[0] + beta_true[1] * X[:, 0] + beta_true[2] * X[:, 1] + np.random.randn(n) * 2

# デザイン行列の作成
X_design = np.column_stack([np.ones(n), X])

# 正規方程式による解
beta_hat = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
print("最小二乗法による推定値:")
print(f"β₀ (切片): {beta_hat[0]:.4f}")
print(f"β₁: {beta_hat[1]:.4f}")
print(f"β₂: {beta_hat[2]:.4f}")

# scikit-learnによる解
model = LinearRegression()
model.fit(X, y)
print("\nscikit-learnによる推定値:")
print(f"β₀ (切片): {model.intercept_:.4f}")
print(f"β₁: {model.coef_[0]:.4f}")
print(f"β₂: {model.coef_[1]:.4f}")

# 予測値の計算
y_hat = X_design @ beta_hat

# 決定係数（R²）の計算
SS_total = np.sum((y - np.mean(y))**2)
SS_residual = np.sum((y - y_hat)**2)
r_squared = 1 - SS_residual / SS_total
print(f"\n決定係数 (R²): {r_squared:.4f}")
```

出力例:
```
最小二乗法による推定値:
β₀ (切片): 5.3361
β₁: 1.9629
β₂: -1.0743

scikit-learnによる推定値:
β₀ (切片): 5.3361
β₁: 1.9629
β₂: -1.0743

決定係数 (R²): 0.8956
```

### 6.2 3D可視化

```python
# 3Dプロットによる可視化
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 元データのプロット
ax.scatter(X[:, 0], X[:, 1], y, c='blue', marker='o', alpha=0.6, label='観測データ')

# 回帰平面のプロット
x_surf = np.linspace(0, 10, 20)
y_surf = np.linspace(0, 10, 20)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = beta_hat[0] + beta_hat[1] * x_surf + beta_hat[2] * y_surf
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.3, color='red')

# 実際の点から平面への垂線を描画
for i in range(0, n, 10):  # 10点おきに表示
    z_plane = beta_hat[0] + beta_hat[1] * X[i, 0] + beta_hat[2] * X[i, 1]
    ax.plot([X[i, 0], X[i, 0]], [X[i, 1], X[i, 1]], [y[i], z_plane], 'k-', alpha=0.2)

ax.set_xlabel('X₁')
ax.set_ylabel('X₂')
ax.set_zlabel('Y')
ax.set_title('2変数線形回帰モデル: 3D可視化')
ax.legend()
plt.tight_layout()
plt.show()
```

### 6.3 実データを用いた分析例

```python
# Boston住宅価格データセットを使用した実例
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# データの読み込み
boston = load_boston()
X = boston.data[:, [5, 12]]  # RM（部屋数）とLSTAT（低所得者割合）
y = boston.target

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# モデルの学習
model = LinearRegression()
model.fit(X_train, y_train)

# 結果の表示
print("回帰係数:")
print(f"切片: {model.intercept_:.4f}")
print(f"RM (部屋数): {model.coef_[0]:.4f}")
print(f"LSTAT (低所得者割合): {model.coef_[1]:.4f}")

# モデルの評価
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# 3Dプロットによる可視化
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# データのプロット
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='blue', marker='o', alpha=0.6, label='テストデータ')

# 回帰平面のプロット
x_surf = np.linspace(min(X[:, 0]), max(X[:, 0]), 20)
y_surf = np.linspace(min(X[:, 1]), max(X[:, 1]), 20)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.3, color='red')

ax.set_xlabel('部屋数 (RM)')
ax.set_ylabel('低所得者割合 (LSTAT)')
ax.set_zlabel('住宅価格 (MEDV)')
ax.set_title('Boston住宅価格: 2変数線形回帰モデル')
ax.legend()
plt.tight_layout()
plt.show()
```

## 7. 演習問題

### 7.1 基本問題

1. **計算問題**: 以下のデータに対して、2つの説明変数を持つ線形回帰モデルを正規方程式を使って求めなさい。

   | ID | x₁ | x₂ | y |
   |----|----|----|---|
   | 1  | 1  | 2  | 5 |
   | 2  | 2  | 1  | 6 |
   | 3  | 3  | 3  | 8 |
   | 4  | 4  | 2  | 10|
   | 5  | 5  | 4  | 12|

   解析的に$\mathbf{X}^T\mathbf{X}$、$\mathbf{X}^T\mathbf{y}$、$(\mathbf{X}^T\mathbf{X})^{-1}$を計算し、最終的に$\hat{\boldsymbol{\beta}}$を求めなさい。

2. **理論問題**: 2つの説明変数$x_1$と$x_2$の間に完全な線形関係（例えば$x_2 = 2x_1$）がある場合、線形回帰モデルのパラメータ推定にどのような問題が生じるか、行列$\mathbf{X}^T\mathbf{X}$の観点から説明しなさい。

3. **概念問題**: 単回帰モデルでは、最小二乗推定量は平面上の最適な「直線」を表しますが、2変数線形回帰モデルでは何を表すのか説明しなさい。これを幾何学的に解釈しなさい。

### 7.2 応用問題

1. **応用計算問題**: 以下のデータは、年齢($x_1$)と教育年数($x_2$)から年収($y$, 単位:万円)を予測するものです。

   | ID | 年齢($x_1$) | 教育年数($x_2$) | 年収($y$) |
   |----|------------|---------------|---------|
   | 1  | 25         | 12            | 300     |
   | 2  | 30         | 16            | 450     |
   | 3  | 35         | 12            | 400     |
   | 4  | 40         | 14            | 500     |
   | 5  | 45         | 18            | 650     |
   | 6  | 50         | 12            | 550     |

   a) 2変数線形回帰モデル$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon$のパラメータを正規方程式を用いて求めなさい。  
   b) 年齢が33歳で教育年数が14年の人の年収を予測しなさい。  
   c) 年齢と年収の関係性、教育年数と年収の関係性をそれぞれ解釈しなさい。

2. **健康データサイエンス応用問題**: 以下のデータは患者の体重($x_1$, kg)と年齢($x_2$, 歳)から血圧値($y$, mmHg)を予測するものです。

   | ID | 体重($x_1$) | 年齢($x_2$) | 血圧($y$) |
   |----|------------|------------|---------|
   | 1  | 55         | 25         | 110     |
   | 2  | 60         | 30         | 115     |
   | 3  | 65         | 40         | 120     |
   | 4  | 70         | 45         | 125     |
   | 5  | 75         | 50         | 130     |
   | 6  | 80         | 55         | 135     |
   | 7  | 85         | 60         | 140     |
   | 8  | 90         | 65         | 145     |

   a) 2変数線形回帰モデルのパラメータを求めなさい。  
   b) 体重75kg、年齢35歳の患者の血圧を予測しなさい。  
   c) 体重と血圧の関係性、年齢と血圧の関係性をそれぞれ解釈しなさい。  
   d) 健康管理の観点から、このモデルの限界点を3つ挙げなさい。

3. **多変量データ分析問題**: 以下の4つの変数からなるデータセットを使用します：
   - $x_1$: 運動時間（分/日）
   - $x_2$: 睡眠時間（時間/日）
   - $y$: 疲労度スコア（0-100）
   
   複数の線形回帰モデルを考えます：
   - モデル1: $y = \beta_0 + \beta_1 x_1 + \varepsilon$
   - モデル2: $y = \beta_0 + \beta_2 x_2 + \varepsilon$
   - モデル3: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon$

   以下の問いに答えなさい：

   a) 決定係数$R^2$の観点から、モデル3はモデル1およびモデル2よりも必ず高い値を示すか。理由を説明しなさい。  
   b) 多変量モデルにおける多重共線性の問題とその対処法について説明しなさい。  
   c) 健康データ分析の観点から、疲労度を予測する際に考慮すべき他の変数を3つ提案し、それぞれがどのように疲労度に影響すると予想されるか説明しなさい。

## 8. よくある質問と解答

### Q1: 単回帰と多変量回帰の大きな違いは何ですか？
**A1**: 単回帰では1つの説明変数のみを使用し、2次元平面上の直線でモデル化します。多変量回帰では複数の説明変数を使用するため、より高次元の空間（2変数なら3次元空間内の平面、3変数なら4次元空間内の超平面）でモデル化します。数学的には、パラメータベクトル$\boldsymbol{\beta}$の次元が増加し、モデルの説明力が向上する可能性がありますが、過学習のリスクも高まります。

### Q2: 正規方程式を解く際に逆行列が存在しない場合はどうすれば良いですか？
**A2**: 逆行列が存在しない場合、これは$\mathbf{X}^T\mathbf{X}$が特異（singular）であることを意味し、通常は説明変数間に完全な線形関係（多重共線性）があることが原因です。この場合、以下の方法が考えられます：
1. 相関の高い変数のうち一方を除外する
2. 主成分分析などで次元を削減する
3. 正則化（リッジ回帰やLasso回帰）を適用する
4. ムーア・ペンローズの擬似逆行列を用いる

### Q3: 複数の説明変数がある場合、各変数の重要度はどのように判断すればよいですか？
**A3**: 変数の重要度を判断するには以下の方法があります：
1. 標準化回帰係数（各変数を標準化してから回帰分析を行い、得られた係数を比較）
2. t値やp値（各係数の統計的有意性を評価）
3. 変数選択法（ステップワイズ法、AIC、BICなど）の適用
4. 部分的決定係数（各変数の寄与度を個別に評価）

### Q4: 実際のデータ分析では、どのような場合に単回帰より多変量回帰が適切ですか？
**A4**: 以下のような場合に多変量回帰が適切です：
1. 反応変数に影響を与える要因が複数ある場合
2. 交絡因子（confounding factor）の影響を制御したい場合
3. モデルの予測精度を向上させたい場合
4. 複数の要因の相対的な影響を比較したい場合

### Q5: 多変量回帰モデルの精度評価にはどのような指標が使われますか？
**A5**: 主な評価指標には以下があります：
1. 決定係数（$R^2$）：モデルによって説明される分散の割合
2. 調整済み決定係数（adjusted $R^2$）：変数の数を考慮した$R^2$
3. 平均二乗誤差（MSE）や平均絶対誤差（MAE）
4. 情報量規準（AIC、BIC）：モデルの複雑さとフィットのバランスを評価
5. クロスバリデーションによる汎化性能の評価

### Q6: 健康データ分析における多変量回帰の典型的な応用例を教えてください。
**A6**: 健康データ分析では以下のような応用例があります：
1. 複数のバイオマーカーから疾患リスクを予測
2. 生活習慣要因（運動、食事、睡眠など）から健康指標を予測
3. 処方薬の複合効果の分析
4. 年齢・性別・遺伝的要因などを考慮した疾患進行予測
5. 環境要因と健康アウトカムの関連分析