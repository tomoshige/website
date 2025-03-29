# 線形代数学 第16回 講義ノート

## 1. 講義情報と予習ガイド

**講義回**: 第16回  
**テーマ**: 連立1次方程式と単回帰モデル  
**関連項目**: 連立1次方程式、最小二乗法、単回帰モデル、正規方程式  
**予習内容**: 
- 連立1次方程式の行列表現（第10回〜第15回）の復習
- 基本的な統計量（平均、分散）の概念

## 2. 学習目標

1. 統計モデルの基本概念と単回帰モデルの理解
2. 説明変数と反応変数の関係性の理解
3. 行列・ベクトルを用いた単回帰モデルの表現方法の習得
4. 最小二乗法と正規方程式による回帰係数の推定方法の習得
5. 単回帰モデルの適用と解釈の方法の理解

## 3. 基本概念

### 3.1 統計モデルとは

> **定義**: 統計モデルとは、観測データの背後にある確率的な構造を数学的に表現したものである。データの生成過程を確率的に記述し、未知のパラメータを含む数学的な関数として表される。

統計モデルは、データの特性や傾向を捉え、将来の予測や現象の理解に役立てるために用いられます。モデルは単純なものから複雑なものまで様々ありますが、今回は最も基本的な「単回帰モデル」について学びます。

### 3.2 単回帰モデルとは

> **定義**: 単回帰モデルとは、1つの説明変数 $x$ と1つの反応変数 $y$ の間の関係を、直線的な関数 $y = \beta_0 + \beta_1 x + \varepsilon$ で表現するモデルである。ここで、$\beta_0$ は切片、$\beta_1$ は回帰係数、$\varepsilon$ は誤差項を表す。

単回帰モデルは最も基本的な統計モデルの一つで、2つの変数間の線形関係を表現します。例えば：
- 勉強時間（$x$）と試験の点数（$y$）の関係
- 運動量（$x$）と体重減少量（$y$）の関係
- 薬の投与量（$x$）と血圧低下（$y$）の関係

### 3.3 説明変数と反応変数

> **定義**:
> - **説明変数（独立変数、predictor）**: モデルにおいて、他の変数に影響を与えると考えられる変数。$x$ で表されることが多い。
> - **反応変数（従属変数、response）**: 説明変数の影響を受けると考えられる変数。$y$ で表されることが多い。

単回帰モデルでは、説明変数 $x$ が反応変数 $y$ にどのように影響を与えるかを調査します。例えば、運動時間（説明変数）が消費カロリー（反応変数）にどう影響するかを分析する場合などです。

### 3.4 切片と回帰係数

> **定義**:
> - **切片 $\beta_0$**: $x = 0$ のときの $y$ の値（y切片）
> - **回帰係数 $\beta_1$**: $x$ が1単位増加したときの $y$ の平均的な変化量（直線の傾き）

これらのパラメータは通常、データから推定する必要があります。データから最も当てはまりの良い直線を見つけることが、回帰分析の目的です。

### 3.5 データと回帰直線

$n$個のデータポイント $(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)$ があるとき、単回帰モデルは以下のように表されます：

$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i \quad (i = 1, 2, \ldots, n)$$

ここで、$\varepsilon_i$ は誤差項で、モデルで説明できない変動を表します。

## 4. 理論と手法

### 4.1 ベクトルと行列を用いた単回帰モデルの表現

単回帰モデルは、ベクトルと行列を用いて以下のように表現できます：

$$\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

ここで：
- $\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}$ は $n \times 1$ の反応変数ベクトル
- $\mathbf{X} = \begin{pmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{pmatrix}$ は $n \times 2$ のデザイン行列
- $\boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \end{pmatrix}$ は $2 \times 1$ のパラメータベクトル
- $\boldsymbol{\varepsilon} = \begin{pmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{pmatrix}$ は $n \times 1$ の誤差ベクトル

この行列表記は、連立1次方程式の行列表現と類似していますが、重要な違いがあります：**連立1次方程式の場合、解が存在することが期待されますが、回帰モデルでは通常、完全な解は存在せず、最良の近似解を見つける必要があります。**

### 4.2 最小二乗法による推定

最小二乗法は、モデルが予測する値と実際の観測値との差（残差）の二乗和を最小化する手法です。

> **定義**: 最小二乗法では、残差平方和 $RSS = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2$ を最小にするパラメータ $\beta_0$ と $\beta_1$ を求める。

残差平方和を行列形式で表すと：

$$RSS = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

これを $\boldsymbol{\beta}$ で微分して0とおくことで、最適なパラメータを求めることができます：

$$\frac{\partial RSS}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}$$

### 4.3 正規方程式

上記の最小二乗法の条件から、以下の方程式（正規方程式）が導かれます：

> **定理**: 最小二乗法による推定値 $\hat{\boldsymbol{\beta}}$ は、正規方程式 $\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$ の解である。

この方程式を解くことで、最適なパラメータ推定値 $\hat{\boldsymbol{\beta}}$ が得られます：

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

ただし、$\mathbf{X}^T\mathbf{X}$ が可逆であることを前提としています（通常の単回帰モデルではこの条件は満たされます）。

### 4.4 単回帰の場合の具体的な計算式

単回帰モデルの場合、正規方程式から以下の具体的な計算式が導けます：

$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

ここで：
- $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ （$x$の平均）
- $\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$ （$y$の平均）
- $S_{xy} = \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$ （$x$と$y$の共分散に比例）
- $S_{xx} = \sum_{i=1}^{n} (x_i - \bar{x})^2$ （$x$の分散に比例）

### 4.5 決定係数 $R^2$

回帰モデルの当てはまりの良さを測る指標として、決定係数 $R^2$ が用いられます：

> **定義**: 決定係数 $R^2$ は、反応変数の全変動のうち、回帰モデルによって説明される割合を表す。
> $$R^2 = \frac{回帰による変動}{全変動} = 1 - \frac{残差変動}{全変動} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

$R^2$ は0から1の間の値をとり、1に近いほどモデルの当てはまりが良いことを示します。

はい、線形回帰モデルの具体的な計算例を追加するのは非常に有益です。特に学生が理解しやすいように、数値を用いた計算のステップバイステップの解説を含めましょう。以下のような実例を講義ノートに追加することをお勧めします：

### 4.6 線形回帰モデルの具体的な計算例

以下に、単回帰モデルの計算プロセスを実データを用いて詳細に解説します。

#### 例題：運動時間と消費カロリーの関係

5人の被験者から得られた運動時間（分）と消費カロリー（kcal）のデータを用いて、単回帰モデルを作成してみましょう。

| 被験者 | 運動時間 $x$ (分) | 消費カロリー $y$ (kcal) |
|:------:|:-----------------:|:------------------------:|
|    1   |        20         |           100            |
|    2   |        30         |           150            |
|    3   |        40         |           190            |
|    4   |        50         |           230            |
|    5   |        60         |           280            |

#### ステップ1：データの要約統計量を計算する

まず、$x$と$y$の平均を計算します：
- $\bar{x} = \frac{20 + 30 + 40 + 50 + 60}{5} = \frac{200}{5} = 40$
- $\bar{y} = \frac{100 + 150 + 190 + 230 + 280}{5} = \frac{950}{5} = 190$

次に、計算に必要な各種の和を求めます：
- $\sum x_i = 200$
- $\sum y_i = 950$
- $\sum x_i^2 = 20^2 + 30^2 + 40^2 + 50^2 + 60^2 = 8,500$
- $\sum x_i y_i = 20 \times 100 + 30 \times 150 + 40 \times 190 + 50 \times 230 + 60 \times 280 = 41,000$

#### ステップ2：回帰係数 $\beta_1$ を計算する

回帰係数 $\beta_1$ の公式は以下の通りです：

$$\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}$$

この式は、以下のように書き直すこともできます：

$$\hat{\beta}_1 = \frac{n\sum x_i y_i - \sum x_i \sum y_i}{n\sum x_i^2 - (\sum x_i)^2}$$

これに値を代入します：

$$\hat{\beta}_1 = \frac{5 \times 41,000 - 200 \times 950}{5 \times 8,500 - 200^2} = \frac{205,000 - 190,000}{42,500 - 40,000} = \frac{15,000}{2,500} = 6$$

#### ステップ3：切片 $\beta_0$ を計算する

切片 $\beta_0$ は以下の式で求められます：

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

値を代入します：

$$\hat{\beta}_0 = 190 - 6 \times 40 = 190 - 240 = -50$$

#### ステップ4：回帰式を得る

以上の計算から、回帰式は以下のようになります：

$$\hat{y} = -50 + 6x$$

この式は、「運動時間が1分増えるごとに、平均して6カロリー余分に消費される」ということを意味します。また、切片が-50であることは、この回帰モデルが非常に短い運動時間（x < 8.33分）の場合に現実的な予測を与えない可能性があることを示唆しています。

#### ステップ5：予測値と残差を計算する

各データポイントについて、モデルからの予測値と実際の値との差（残差）を計算します：

| 被験者 | $x$ | $y$ | $\hat{y} = -50 + 6x$ | 残差 $y - \hat{y}$ |
|:------:|:---:|:---:|:--------------------:|:-----------------:|
|    1   |  20 | 100 |          70          |        30         |
|    2   |  30 | 150 |         130          |        20         |
|    3   |  40 | 190 |         190          |         0         |
|    4   |  50 | 230 |         250          |       -20         |
|    5   |  60 | 280 |         310          |       -30         |

#### ステップ6：モデルの評価（決定係数 $R^2$ の計算）

決定係数 $R^2$ を計算するために、以下の量を求めます：

- 全変動（TSS）：$\sum (y_i - \bar{y})^2 = (100-190)^2 + (150-190)^2 + (190-190)^2 + (230-190)^2 + (280-190)^2 = 17,000$
- 残差変動（RSS）：$\sum (y_i - \hat{y}_i)^2 = 30^2 + 20^2 + 0^2 + (-20)^2 + (-30)^2 = 2,600$
- 回帰による変動（ESS）：$\sum (\hat{y}_i - \bar{y})^2 = TSS - RSS = 17,000 - 2,600 = 14,400$

決定係数 $R^2$ は以下のように計算されます：

$$R^2 = \frac{ESS}{TSS} = \frac{14,400}{17,000} = 0.847 \approx 0.85$$

この結果は、回帰モデルが消費カロリーの変動の約85%を説明できることを示しています。

#### ステップ7：正規方程式による解法の確認

正規方程式は以下の形式で表されます：

$$\begin{pmatrix} n & \sum x_i \\ \sum x_i & \sum x_i^2 \end{pmatrix} \begin{pmatrix} \hat{\beta}_0 \\ \hat{\beta}_1 \end{pmatrix} = \begin{pmatrix} \sum y_i \\ \sum x_i y_i \end{pmatrix}$$

数値を代入すると：

$$\begin{pmatrix} 5 & 200 \\ 200 & 8,500 \end{pmatrix} \begin{pmatrix} \hat{\beta}_0 \\ \hat{\beta}_1 \end{pmatrix} = \begin{pmatrix} 950 \\ 41,000 \end{pmatrix}$$

この連立方程式を解くと：

$$5\hat{\beta}_0 + 200\hat{\beta}_1 = 950 \tag{1}$$
$$200\hat{\beta}_0 + 8,500\hat{\beta}_1 = 41,000 \tag{2}$$

方程式(1)から $\hat{\beta}_0$ について解くと：

$$\hat{\beta}_0 = \frac{950 - 200\hat{\beta}_1}{5} = 190 - 40\hat{\beta}_1 \tag{3}$$

方程式(3)を方程式(2)に代入：

$$200(190 - 40\hat{\beta}_1) + 8,500\hat{\beta}_1 = 41,000$$
$$38,000 - 8,000\hat{\beta}_1 + 8,500\hat{\beta}_1 = 41,000$$
$$38,000 + 500\hat{\beta}_1 = 41,000$$
$$500\hat{\beta}_1 = 3,000$$
$$\hat{\beta}_1 = 6$$

そして、この値を方程式(3)に代入することで $\hat{\beta}_0$ を求めます：

$$\hat{\beta}_0 = 190 - 40 \times 6 = 190 - 240 = -50$$

これは先ほどの計算と同じ結果になりました。

#### ステップ8：新しいデータポイントの予測

運動時間が45分の場合の消費カロリーを予測してみましょう：

$$\hat{y} = -50 + 6 \times 45 = -50 + 270 = 220 \text{ kcal}$$

このように、回帰モデルを使用すれば、新しいデータポイントに対する予測が可能になります。

この詳細な計算例は、学生が線形回帰モデルの各ステップを具体的に理解するのに役立ちます。特に、統計学や線形代数の概念が実際のデータ分析においてどのように適用されるかを示しています。

## 5. Pythonによる実装と可視化

### 5.1 単回帰モデルの実装

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# サンプルデータの生成（勉強時間と試験成績の例）
np.random.seed(42)
hours_studied = np.random.uniform(1, 10, 30)  # 1〜10時間の勉強時間（30人分）
noise = np.random.normal(0, 5, 30)  # ランダムなノイズ
test_score = 50 + 5 * hours_studied + noise  # 勉強時間と得点の関係 (y = 50 + 5x + ε)

# データフレームの作成
data = pd.DataFrame({
    'hours': hours_studied,
    'score': test_score
})

# 単回帰モデルの手動実装
def manual_simple_regression(x, y):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # 分母: x の偏差平方和
    ss_xx = np.sum((x - mean_x) ** 2)
    # 分子: x と y の偏差積和
    ss_xy = np.sum((x - mean_x) * (y - mean_y))
    
    # 回帰係数と切片の計算
    beta_1 = ss_xy / ss_xx
    beta_0 = mean_y - beta_1 * mean_x
    
    # 予測値の計算
    y_pred = beta_0 + beta_1 * x
    
    # 決定係数 R^2 の計算
    ss_total = np.sum((y - mean_y) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    return beta_0, beta_1, y_pred, r_squared

# 手動で単回帰分析を実行
beta_0, beta_1, y_pred, r_squared = manual_simple_regression(data['hours'], data['score'])

print(f"手動計算による結果:")
print(f"切片 (β₀): {beta_0:.4f}")
print(f"回帰係数 (β₁): {beta_1:.4f}")
print(f"決定係数 (R²): {r_squared:.4f}")

# scikit-learnを用いた単回帰分析
model = LinearRegression()
X = data['hours'].values.reshape(-1, 1)  # 説明変数をnumpy配列に変換
y = data['score'].values  # 反応変数をnumpy配列に変換

model.fit(X, y)
y_pred_sklearn = model.predict(X)

print("\nscikit-learnによる結果:")
print(f"切片 (β₀): {model.intercept_:.4f}")
print(f"回帰係数 (β₁): {model.coef_[0]:.4f}")
print(f"決定係数 (R²): {r2_score(y, y_pred_sklearn):.4f}")

# 結果の可視化
plt.figure(figsize=(10, 6))
plt.scatter(data['hours'], data['score'], color='blue', alpha=0.7, label='データ点')
plt.plot(data['hours'], y_pred, color='red', linewidth=2, label='回帰直線')
plt.xlabel('勉強時間 (時間)')
plt.ylabel('テスト得点')
plt.title('勉強時間とテスト得点の単回帰分析')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

### 5.2 正規方程式による解法

```python
# 正規方程式を使用した解法
def normal_equation(X, y):
    # デザイン行列に定数項（1）の列を追加
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # 正規方程式を解く: β = (X^T X)^(-1) X^T y
    beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    return beta

# 正規方程式で解く
X_array = data['hours'].values.reshape(-1, 1)
y_array = data['score'].values
beta_norm_eq = normal_equation(X_array, y_array)

print("\n正規方程式による結果:")
print(f"切片 (β₀): {beta_norm_eq[0]:.4f}")
print(f"回帰係数 (β₁): {beta_norm_eq[1]:.4f}")
```

### 5.3 健康データを用いた単回帰の例

```python
# 健康データのシミュレーション（日々の運動時間と心拍数の変化）
np.random.seed(123)
exercise_minutes = np.random.uniform(10, 60, 40)  # 10〜60分の運動時間
noise = np.random.normal(0, 3, 40)  # ランダムなノイズ
heart_rate_decrease = 0.3 * exercise_minutes + noise  # 運動時間と心拍数低下の関係

health_data = pd.DataFrame({
    'exercise_min': exercise_minutes,
    'heart_rate_decrease': heart_rate_decrease
})

# 単回帰モデルの適用
beta_0_health, beta_1_health, y_pred_health, r_squared_health = manual_simple_regression(
    health_data['exercise_min'], health_data['heart_rate_decrease']
)

print("\n健康データの単回帰分析結果:")
print(f"切片 (β₀): {beta_0_health:.4f}")
print(f"回帰係数 (β₁): {beta_1_health:.4f}")
print(f"決定係数 (R²): {r_squared_health:.4f}")

# 結果の可視化
plt.figure(figsize=(10, 6))
plt.scatter(health_data['exercise_min'], health_data['heart_rate_decrease'], 
            color='green', alpha=0.7, label='データ点')
plt.plot(health_data['exercise_min'], y_pred_health, 
         color='purple', linewidth=2, label='回帰直線')
plt.xlabel('運動時間 (分)')
plt.ylabel('安静時心拍数の低下 (bpm)')
plt.title('運動時間と心拍数低下の単回帰分析')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 推定モデルを使った予測
new_exercise_time = 45  # 新しい運動時間: 45分
predicted_decrease = beta_0_health + beta_1_health * new_exercise_time
print(f"\n運動時間が{new_exercise_time}分の場合、予測される心拍数低下: {predicted_decrease:.2f} bpm")
```

## 6. 演習問題

### 基本問題

1. 次のデータを用いて、手計算で単回帰モデルのパラメータ $\hat{\beta}_0$ と $\hat{\beta}_1$ を求めなさい。
   
   | $x$ | 1 | 2 | 3 | 4 | 5 |
   |-----|---|---|---|---|---|
   | $y$ | 3 | 5 | 7 | 8 | 11 |

2. 次の式は、単回帰モデルにおける正規方程式です。この式を展開して、$\hat{\beta}_0$ と $\hat{\beta}_1$ を求める一般式を導出せよ。
   
   $$\begin{pmatrix} n & \sum x_i \\ \sum x_i & \sum x_i^2 \end{pmatrix} \begin{pmatrix} \hat{\beta}_0 \\ \hat{\beta}_1 \end{pmatrix} = \begin{pmatrix} \sum y_i \\ \sum x_i y_i \end{pmatrix}$$

3. 単回帰モデル $y = \beta_0 + \beta_1 x + \varepsilon$ において、$\hat{\beta}_1 = 2.5$ と推定された。この時、説明変数 $x$ が1単位増加すると、反応変数 $y$ はどのように変化すると予測されるか説明せよ。

4. 決定係数 $R^2 = 0.85$ を持つ単回帰モデルがある。この値は、モデルの当てはまりの良さについて何を意味するか説明せよ。

### 応用問題

5. あるデータセットについて、以下の情報が得られている：
   - $\sum_{i=1}^{10} x_i = 50$
   - $\sum_{i=1}^{10} y_i = 150$
   - $\sum_{i=1}^{10} x_i^2 = 300$
   - $\sum_{i=1}^{10} x_i y_i = 800$
   
   このデータに対する単回帰モデルのパラメータ $\hat{\beta}_0$ と $\hat{\beta}_1$ を正規方程式を用いて求めよ。

6. 健康データサイエンスに関連する応用問題：
   
   ある医学研究では、患者の血中コレステロール値（mg/dL）が、1日の運動時間（分）とどのように関連しているかを調査している。20人の患者から得られたデータポイントに単回帰モデルを適用したところ、以下の結果が得られた：
   
   $\hat{\beta}_0 = 240$、$\hat{\beta}_1 = -0.8$、$R^2 = 0.65$
   
   (a) この回帰モデルの式を書き、運動時間と血中コレステロール値の関係について解釈せよ。
   
   (b) 1日30分の運動をしている患者の予測血中コレステロール値を求めよ。
   
   (c) この回帰モデルの限界点を2つ挙げよ。

## 7. よくある質問と解答

### Q1: 単回帰モデルと連立1次方程式の違いは何ですか？

A1: 連立1次方程式は、方程式の数と未知数の数が同じ場合に厳密な解を持ちますが、単回帰モデルでは通常、データ点の数（観測数 $n$）がパラメータの数（2つ、$\beta_0$ と $\beta_1$）よりも多いため、一般的に厳密な解を持ちません。そのため、最小二乗法などを用いて「最も良い近似解」を求めます。

### Q2: 最小二乗法はなぜ二乗を使うのですか？絶対値ではだめですか？

A2: 二乗を使う理由はいくつかあります：
1. 微分可能であるため、解析的に最適解を求めやすい
2. 大きな誤差を重く罰則するため、外れ値の影響を強く受ける（これは時に欠点にもなりますが）
3. 正と負の誤差を対称に扱うことができる
絶対値を使う手法（最小絶対偏差法など）もありますが、微分可能でない点があるため最適化が技術的に難しくなります。

### Q3: 説明変数が複数ある場合はどうなりますか？

A3: 複数の説明変数がある場合、それを「重回帰モデル（multiple regression model）」と呼びます。数式は $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + \varepsilon$ となり、行列形式での表現や正規方程式による解法は本質的に同じですが、計算がより複雑になります。これについては第18回の講義で学びます。

### Q4: 正規方程式による解法と勾配降下法による解法の違いは何ですか？

A4: 正規方程式は解析的に一度で最適解を求める方法である一方、勾配降下法は繰り返し計算によって徐々に解に近づける数値的な方法です。正規方程式は小〜中規模のデータセットでは効率的ですが、大規模データでは計算コストが高くなります。勾配降下法は反復計算が必要ですが、大規模データでもスケールしやすいという利点があります。現代の機械学習では両方の手法が状況に応じて使い分けられています。

### Q5: 回帰モデルの仮定は何ですか？

A5: 回帰モデルの主な仮定には以下があります：
1. 線形性: 説明変数と反応変数の間には線形の関係がある
2. 独立性: 誤差項は互いに独立している
3. 等分散性: 誤差の分散はすべての観測値で一定である
4. 正規性: 誤差項は正規分布に従う
これらの仮定が満たされているかを確認するために、残差分析などの診断手法が用いられます。

## 8. まとめ

本講義では、連立1次方程式の知識を基に、データサイエンスの基本的手法である単回帰モデルについて学びました。単回帰モデルは、1つの説明変数から1つの反応変数を予測するための統計モデルで、最小二乗法を用いてパラメータを推定します。

主な学習内容：
- 統計モデルと単回帰モデルの概念
- 説明変数と反応変数の関係
- 行列・ベクトルを用いた単回帰モデルの表現
- 最小二乗法と正規方程式によるパラメータ推定
- Pythonを用いた単回帰モデルの実装と可視化
- 決定係数 $R^2$ によるモデル評価

単回帰モデルは、データサイエンスにおける様々な分析手法の基礎となるものです。次回の講義では、ベクトルの1次独立性と逆行列の関係について学び、その後、複数の説明変数を持つ重回帰モデルへと発展させていきます。