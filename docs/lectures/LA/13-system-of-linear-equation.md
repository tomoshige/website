# 線形代数学 第13回：連立方程式の解の種類

## 1. 講義情報と予習ガイド

**講義回**: 第13回  
**関連項目**: 連立一次方程式の解の分類、行列のランク  
**予習内容**: 
- 第11回「ガウスの消去法と解の探索」の内容を復習しておくこと
- 第12回「ランクの概念とその計算」の内容を確認すること

## 2. 学習目標

本講義の終了時点で、以下のことができるようになることを目指します：

1. 連立一次方程式の解の種類（唯一解、無数の解、解なし）を理解し、区別できる
2. 具体的な連立一次方程式の解の存在条件を判定できる
3. 連立一次方程式の解の幾何学的解釈ができる
4. 行列のランクと連立一次方程式の解の関係を説明できる
5. Google Colabを用いて連立一次方程式の解を視覚化できる

## 3. 基本概念

### 3.1 連立一次方程式の復習

$n$個の未知数 $x_1, x_2, \ldots, x_n$ に関する $m$個の一次方程式の集まりを考えます：

$$
\begin{align}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1\\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2\\
\vdots\\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
\end{align}
$$

これを行列とベクトルを用いて表現すると：

$$A\mathbf{x} = \mathbf{b}$$

ここで、$A$は$m \times n$の係数行列、$\mathbf{x}$は$n$次元の未知数ベクトル、$\mathbf{b}$は$m$次元の右辺ベクトルです。

### 3.2 解の定義と種類

> **定義**: 連立一次方程式 $A\mathbf{x} = \mathbf{b}$ の「解」とは、方程式を満たす未知数ベクトル $\mathbf{x}$ の値のことです。

連立一次方程式の解は、以下の3つのカテゴリーに分類されます：

1. **唯一解 (Unique solution)**: 解がただ1つだけ存在する場合
2. **無数の解 (Infinitely many solutions)**: 解が無限に存在する場合
3. **解なし (No solution)**: 方程式を満たす解が存在しない場合

## 4. 理論と手法

### 4.1 方程式と未知数の数による解の関係

$m$個の方程式と$n$個の未知数がある場合：

1. $m < n$: 方程式の数より未知数の数が多い場合（**不足決定系**）
   - 通常、解は無数に存在する（ただし解がない場合もある）

2. $m = n$: 方程式の数と未知数の数が等しい場合（**完全決定系**）
   - 係数行列$A$が正則（行列式$\det(A) \neq 0$）なら唯一解が存在
   - 係数行列$A$が特異（行列式$\det(A) = 0$）なら解なしまたは無数の解

3. $m > n$: 方程式の数が未知数の数より多い場合（**過剰決定系**）
   - 通常、解は存在しない（ただし特殊な場合は解があることもある）

### 4.2 行列のランクと解の関係

連立一次方程式 $A\mathbf{x} = \mathbf{b}$ の解の存在と種類は、係数行列 $A$ のランク $\text{rank}(A)$ と拡大係数行列 $[A|\mathbf{b}]$ のランク $\text{rank}([A|\mathbf{b}])$ によって決定されます。

> **定理**: 連立一次方程式 $A\mathbf{x} = \mathbf{b}$ （$A$は$m \times n$行列）について：
>
> 1. $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) < n$ の場合：**無数の解**が存在する
> 2. $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) = n$ の場合：**唯一解**が存在する
> 3. $\text{rank}(A) < \text{rank}([A|\mathbf{b}])$ の場合：**解なし**

### 4.3 簡約階段行列を用いた解の判定

ガウスの消去法によって得られた簡約階段行列の形から、解の種類を判定することができます：

1. **唯一解**：各変数に対応する主成分（leading entry）があり、矛盾する方程式がない
2. **無数の解**：フリー変数（自由変数）が存在し、矛盾する方程式がない
3. **解なし**：`0 = 非ゼロ定数`という矛盾する行が存在する

## 5. 具体例と解説

### 5.1 唯一解を持つ連立一次方程式の例

$$
\begin{align}
x + 2y &= 5\\
2x - y &= 0
\end{align}
$$

この連立方程式を行列形式で表すと：

$$
\begin{bmatrix}
1 & 2 \\
2 & -1
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
=
\begin{bmatrix}
5 \\
0
\end{bmatrix}
$$

ガウスの消去法で解くと：

$$
\begin{bmatrix}
1 & 2 & 5 \\
2 & -1 & 0
\end{bmatrix}
\rightarrow
\begin{bmatrix}
1 & 2 & 5 \\
0 & -5 & -10
\end{bmatrix}
\rightarrow
\begin{bmatrix}
1 & 2 & 5 \\
0 & 1 & 2
\end{bmatrix}
$$

後退代入により、$y = 2$, $x + 2(2) = 5$ より $x = 1$ が得られます。

したがって、解は$x = 1$, $y = 2$の唯一解です。

係数行列$A$と拡大係数行列$[A|b]$のランクを考えると：
- $\text{rank}(A) = 2$
- $\text{rank}([A|b]) = 2$
- 未知数の数$n = 2$

$\text{rank}(A) = \text{rank}([A|b]) = n$なので、唯一解が存在します。

### 5.2 無数の解を持つ連立一次方程式の例

$$
\begin{align}
x + 2y - z &= 5\\
2x - y + z &= 0\\
3x + y &= 5
\end{align}
$$

行列形式で：

$$
\begin{bmatrix}
1 & 2 & -1 \\
2 & -1 & 1 \\
3 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
=
\begin{bmatrix}
5 \\
0 \\
5
\end{bmatrix}
$$

ガウス消去法を適用すると：

$$
\begin{bmatrix}
1 & 2 & -1 & 5 \\
2 & -1 & 1 & 0 \\
3 & 1 & 0 & 5
\end{bmatrix}
\rightarrow
\begin{bmatrix}
1 & 2 & -1 & 5 \\
0 & -5 & 3 & -10 \\
0 & -5 & 3 & -10
\end{bmatrix}
\rightarrow
\begin{bmatrix}
1 & 2 & -1 & 5 \\
0 & 1 & -\frac{3}{5} & 2 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

3行目はすべて0となり、方程式の数が実質的に減少しました。$z$を自由変数（パラメータ）とすると：

$y - \frac{3}{5}z = 2$ より $y = 2 + \frac{3}{5}z$

$x + 2y - z = 5$ より $x = 5 - 2y + z = 5 - 2(2 + \frac{3}{5}z) + z = 5 - 4 - \frac{6}{5}z + z = 1 - \frac{1}{5}z$

したがって、解は $z$をパラメータとして：
$x = 1 - \frac{1}{5}z$, $y = 2 + \frac{3}{5}z$, $z = z$（任意の値）

係数行列$A$と拡大係数行列$[A|b]$のランクを考えると：
- $\text{rank}(A) = 2$
- $\text{rank}([A|b]) = 2$
- 未知数の数$n = 3$

$\text{rank}(A) = \text{rank}([A|b]) < n$なので、無数の解が存在します。

### 5.3 解がない連立一次方程式の例

$$
\begin{align}
x + 2y &= 5\\
2x + 4y &= 6
\end{align}
$$

行列形式で：

$$
\begin{bmatrix}
1 & 2 \\
2 & 4
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
=
\begin{bmatrix}
5 \\
6
\end{bmatrix}
$$

ガウス消去法を適用すると：

$$
\begin{bmatrix}
1 & 2 & 5 \\
2 & 4 & 6
\end{bmatrix}
\rightarrow
\begin{bmatrix}
1 & 2 & 5 \\
0 & 0 & -4
\end{bmatrix}
$$

2行目が $0 = -4$ となり、これは矛盾します。したがって、この連立方程式は解を持ちません。

係数行列$A$と拡大係数行列$[A|b]$のランクを考えると：
- $\text{rank}(A) = 1$
- $\text{rank}([A|b]) = 2$

$\text{rank}(A) < \text{rank}([A|b])$なので、解は存在しません。

## 6. 幾何学的解釈

### 6.1 2次元での解の幾何学的意味

2元連立1次方程式の場合、各方程式は平面上の直線を表します。

1. **唯一解**: 2つの直線が1点で交わる場合
2. **無数の解**: 2つの直線が重なる場合（同一直線）
3. **解なし**: 2つの直線が平行で交わらない場合

### 6.2 3次元での解の幾何学的意味

3元連立1次方程式の場合、各方程式は3次元空間内の平面を表します。

1. **唯一解**: 3つの平面が1点で交わる場合
2. **無数の解**: 
   - 3つの平面が1本の直線で交わる場合（無限に多くの点）
   - 2つ以上の平面が重なり、残りの平面とその重なった平面が交わる場合
3. **解なし**: 
   - 2つ以上の平面が平行で交わらない場合
   - 3つの平面が共通点を持たずに交わる場合（3つの平面が三角柱のような形で交わる）

## 7. Pythonによる実装と可視化

### 7.1 行列のランクと解の判定

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 連立方程式の解の種類を判定する関数
def check_solution_type(A, b):
    A_matrix = np.array(A, dtype=float)
    b_vector = np.array(b, dtype=float)
    
    # 係数行列のランク
    rank_A = np.linalg.matrix_rank(A_matrix)
    
    # 拡大係数行列のランク
    augmented = np.column_stack((A_matrix, b_vector))
    rank_augmented = np.linalg.matrix_rank(augmented)
    
    # 未知数の数
    n = A_matrix.shape[1]
    
    # 解の種類を判定
    if rank_A < rank_augmented:
        return "解なし"
    elif rank_A == rank_augmented and rank_A < n:
        return "無数の解"
    else:  # rank_A == rank_augmented == n
        return "唯一解"

# 例1: 唯一解の場合
A1 = [[1, 2], [2, -1]]
b1 = [5, 0]
print("例1の解の種類:", check_solution_type(A1, b1))

# 例2: 無数の解の場合
A2 = [[1, 2, -1], [2, -1, 1], [3, 1, 0]]
b2 = [5, 0, 5]
print("例2の解の種類:", check_solution_type(A2, b2))

# 例3: 解なしの場合
A3 = [[1, 2], [2, 4]]
b3 = [5, 6]
print("例3の解の種類:", check_solution_type(A3, b3))
```

### 7.2 2次元での連立方程式の可視化

```python
def plot_2d_system(A, b, x_range=(-10, 10)):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(x_range[0], x_range[1], 1000)
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for i in range(len(b)):
        if A[i][1] != 0:  # y係数が0でない場合
            y = (b[i] - A[i][0] * x) / A[i][1]
            ax.plot(x, y, colors[i % len(colors)], label=f'式{i+1}: {A[i][0]}x + {A[i][1]}y = {b[i]}')
        else:  # y係数が0の場合（垂直線）
            if A[i][0] != 0:
                x_val = b[i] / A[i][0]
                ax.axvline(x=x_val, color=colors[i % len(colors)], label=f'式{i+1}: {A[i][0]}x = {b[i]}')
    
    # グラフの設定
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlim(x_range)
    ax.set_ylim(x_range)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('連立方程式の幾何学的解釈')
    
    plt.show()

# 例1: 唯一解 (x=1, y=2)
plot_2d_system(A1, b1)

# 例3: 解なし (平行線)
plot_2d_system(A3, b3)
```

### 7.3 3次元での連立方程式の可視化

```python
def plot_3d_system(A, b, ranges=(-5, 5)):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 平面上の点をグリッドで生成
    xx, yy = np.meshgrid(np.linspace(ranges[0], ranges[1], 10),
                         np.linspace(ranges[0], ranges[1], 10))
    
    # 各方程式の平面をプロット
    colors = ['b', 'g', 'r']
    for i in range(min(len(b), 3)):  # 最大3つの平面まで表示
        if A[i][2] != 0:  # z係数が0でない場合
            z = (b[i] - A[i][0] * xx - A[i][1] * yy) / A[i][2]
            surf = ax.plot_surface(xx, yy, z, alpha=0.6, color=colors[i], 
                                  label=f'平面{i+1}')
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d
    
    # グラフの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(ranges)
    ax.set_ylim(ranges)
    ax.set_zlim(ranges)
    ax.set_title('連立方程式の3D表現')
    ax.legend()
    
    plt.show()

# 例2: 無数の解の3D可視化
plot_3d_system(A2, b2)
```

## 8. データサイエンスでの応用例

### 8.1 線形回帰における解の存在

線形回帰では、データ点$(x_i, y_i)$を最もよく表す直線$y = \beta_0 + \beta_1 x$を求める問題を考えます。一般に、データ点の数$n$が2より大きい場合、通常は完全な解は存在せず（過剰決定系）、最小二乗法によって近似解を求めます。

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# サンプルデータ生成
np.random.seed(42)
x = np.random.rand(20) * 10
y = 2 * x + 1 + np.random.randn(20) * 2

# データフレーム作成
data = pd.DataFrame({'x': x, 'y': y})

# 線形回帰モデル
model = LinearRegression()
model.fit(data[['x']], data['y'])

# 回帰直線のパラメータ
beta_0 = model.intercept_
beta_1 = model.coef_[0]

print(f'推定された直線: y = {beta_0:.4f} + {beta_1:.4f}x')

# プロット
plt.figure(figsize=(10, 6))
plt.scatter(data['x'], data['y'], color='blue', label='データ点')
plt.plot(data['x'], model.predict(data[['x']]), color='red', label='回帰直線')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('線形回帰 - 過剰決定系の近似解')
plt.legend()
plt.show()
```

### 8.2 健康データにおける体組成の推定

健康データサイエンスの例として、体組成（体脂肪率、筋肉量など）を身長、体重、年齢から推定するモデルを考えます。このようなモデルは通常、過剰決定系の連立方程式となり、完全な解ではなく最適な近似解を求めることになります。

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 仮想的な健康データを生成
np.random.seed(42)
n_samples = 100

# 説明変数: 身長(cm)、体重(kg)、年齢(years)
height = np.random.normal(170, 10, n_samples)
weight = np.random.normal(70, 15, n_samples)
age = np.random.randint(20, 70, n_samples)

# 応答変数: 体脂肪率(%) - 身長、体重、年齢の関数として生成（ノイズ付き）
body_fat = 10 + 0.3 * (weight - 70) - 0.1 * (height - 170) + 0.2 * (age - 40) + np.random.normal(0, 3, n_samples)

# データフレーム作成
health_data = pd.DataFrame({
    'height': height,
    'weight': weight,
    'age': age,
    'body_fat': body_fat
})

# 説明変数と応答変数
X = health_data[['height', 'weight', 'age']]
y = health_data['body_fat']

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 線形回帰モデルの構築
model = LinearRegression()
model.fit(X_train, y_train)

# モデルの係数
print(f'切片: {model.intercept_:.4f}')
print(f'身長の係数: {model.coef_[0]:.4f}')
print(f'体重の係数: {model.coef_[1]:.4f}')
print(f'年齢の係数: {model.coef_[2]:.4f}')

# テストデータでの予測
y_pred = model.predict(X_test)

# 予測精度の評価
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'平均二乗誤差 (MSE): {mse:.4f}')
print(f'決定係数 (R²): {r2:.4f}')

# 実測値と予測値の散布図
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('実測体脂肪率 (%)')
plt.ylabel('予測体脂肪率 (%)')
plt.title('体脂肪率の実測値と予測値の比較')
plt.grid(True)
plt.show()
```

## 9. 演習問題

### 基本問題

1. 次の連立方程式の解を求め、解の種類（唯一解、無数の解、解なし）を判定しなさい。
    
    (a) $\begin{cases}
    2x + 3y = 8 \\
    -4x - 6y = -16
    \end{cases}$
    
    (b) $\begin{cases}
    2x + y - z = 3 \\
    x - y + z = 2 \\
    3x + 2y = 4
    \end{cases}$
    
    (c) $\begin{cases}
    x + 2y - z = 5 \\
    2x + 4y - 2z = 6 \\
    3x + 6y - 3z = 15
    \end{cases}$

2. 次の行列$A$と右辺ベクトル$\mathbf{b}$について、連立方程式$A\mathbf{x} = \mathbf{b}$の解の種類を判定しなさい。
    
    (a) $A = \begin{bmatrix}
    1 & 3 & 2 \\
    2 & 6 & 4 \\
    3 & 9 & 6
    \end{bmatrix}$, $\mathbf{b} = \begin{bmatrix}
    4 \\
    8 \\
    12
    \end{bmatrix}$
    
    (b) $A = \begin{bmatrix}
    1 & 3 & 2 \\
    2 & 6 & 4 \\
    3 & 9 & 6
    \end{bmatrix}$, $\mathbf{b} = \begin{bmatrix}
    4 \\
    8 \\
    10
    \end{bmatrix}$

3. 行列$A$のランクが2、拡大係数行列$[A|\mathbf{b}]$のランクが3、未知数の数が4の場合、連立方程式$A\mathbf{x} = \mathbf{b}$の解の種類はどうなるか。理由も説明しなさい。

### 応用問題

4. $\lambda$をパラメータとする連立方程式
    $\begin{cases}
    x + y + z = 6 \\
    x + 2y + \lambda z = 10 \\
    x + 3y + \lambda^2 z = \lambda + 12
    \end{cases}$
    
    において、$\lambda$の値によって解の種類がどのように変化するか調べなさい。

5. 3次元平面上の3点 $P_1(1, 2, 3)$, $P_2(2, 3, 1)$, $P_3(3, 1, 2)$ を通る平面の方程式を求めなさい。この問題をどのように連立一次方程式として定式化できるか説明し、解き方を示しなさい。

6. 健康データサイエンスにおける応用として、ある研究では5名の被験者の年齢（年）、体重（kg）、運動時間（時間/週）、および血圧値（mmHg）が以下のように記録されています：
   
   | 被験者 | 年齢 | 体重 | 運動時間 | 血圧 |
   |-----:|-----:|-----:|--------:|-----:|
   | 1 | 25 | 70 | 3 | 120 |
   | 2 | 45 | 85 | 1 | 140 |
   | 3 | 35 | 65 | 5 | 115 |
   | 4 | 55 | 75 | 2 | 145 |
   | 5 | 30 | 80 | 4 | 125 |
   
   血圧（$y$）を年齢（$x_1$）、体重（$x_2$）、運動時間（$x_3$）の線形関数 $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3$ として表そうとしています。
   
   a. この問題を連立一次方程式として定式化しなさい。
   b. この連立方程式の解の種類を判定しなさい。
   c. 最小二乗法を用いてパラメータ（$\beta_0, \beta_1, \beta_2, \beta_3$）を推定しなさい。
   d. 40歳、75kg、週に3時間運動する人の予想血圧を計算しなさい。

## 10. よくある質問と解答

### Q1: 連立方程式の解がないというのは、数学的にどういう意味ですか？
A1: 連立方程式の解がないということは、すべての方程式を同時に満たす変数の値が存在しないということです。幾何学的には、例えば2次元の場合、2つの直線が平行で交わらない状況に対応します。これは方程式間に矛盾があることを意味します。

## 10. よくある質問と解答（続き）

### Q2: 解が無数にあるとき、一般解はどのように表現すればよいですか？
A2: 解が無数にある場合、一般解はパラメータ（自由変数）を用いて表現します。例えば、3元連立方程式で解が無数にある場合、1つまたは複数の変数をパラメータとして選び、残りの変数をそのパラメータの関数として表します。例：$z$をパラメータとして、$x = 1 - \frac{1}{5}z$, $y = 2 + \frac{3}{5}z$のように表現します。

### Q3: 行列のランクと解の関係をどのように覚えておけばよいですか？
A3: 次の関係を覚えておくと便利です：
1. $\text{rank}(A) < \text{rank}([A|\mathbf{b}])$ → 解なし
2. $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) = n$ → 唯一解
3. $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) < n$ → 無数の解
ここで、$n$は未知数の数、$A$は係数行列、$[A|\mathbf{b}]$は拡大係数行列です。

### Q4: 過剰決定系（方程式の数 > 未知数の数）の場合、なぜ通常は解がないのですか？
A4: 過剰決定系では、方程式の数が未知数の数より多いため、各方程式が課す条件をすべて同時に満たすことは一般的に困難です。幾何学的には、例えば3つの平面が1点で交わる確率は低く、通常は共通点を持ちません。ただし、方程式間に線形従属関係がある場合（例：1つの方程式が他の方程式の線形結合で表せる場合）は解が存在する可能性があります。

### Q5: データサイエンスでは、解がない連立方程式はどのように扱われますか？
A5: データサイエンスでは、特に回帰分析などにおいて、完全な解が存在しない過剰決定系の連立方程式がよく現れます。そのような場合、最小二乗法を用いて「最良の近似解」を求めます。これは、すべての方程式を完全に満たすのではなく、誤差の二乗和を最小化する解を求める方法です。実際のデータには誤差やノイズが含まれるため、このアプローチが現実的かつ有用です。

## 11. 演習問題の解答例

### 基本問題の解答

**問題1(a)** 
$\begin{cases}
2x + 3y = 8 \\
-4x - 6y = -16
\end{cases}$

第2式を$-1/2$倍すると：
$\begin{cases}
2x + 3y = 8 \\
2x + 3y = 8
\end{cases}$

両方の式が同じになるため、2つの方程式は線形従属であり、第2式は第1式から導出できます。したがって、解は無数に存在します。

$2x + 3y = 8$を解くと：$x = (8 - 3y)/2 = 4 - 3y/2$

よって一般解は、$y$をパラメータとして：$x = 4 - \frac{3}{2}y$, $y$は任意の実数

**問題1(b)**
$\begin{cases}
2x + y - z = 3 \\
x - y + z = 2 \\
3x + 2y = 4
\end{cases}$

ガウスの消去法を適用すると：
$\begin{bmatrix}
2 & 1 & -1 & 3 \\
1 & -1 & 1 & 2 \\
3 & 2 & 0 & 4
\end{bmatrix}$

第1行と第2行を入れ替え：
$\begin{bmatrix}
1 & -1 & 1 & 2 \\
2 & 1 & -1 & 3 \\
3 & 2 & 0 & 4
\end{bmatrix}$

第2行から第1行の2倍を引く：
$\begin{bmatrix}
1 & -1 & 1 & 2 \\
0 & 3 & -3 & -1 \\
3 & 2 & 0 & 4
\end{bmatrix}$

第3行から第1行の3倍を引く：
$\begin{bmatrix}
1 & -1 & 1 & 2 \\
0 & 3 & -3 & -1 \\
0 & 5 & -3 & -2
\end{bmatrix}$

第2行を3で割る：
$\begin{bmatrix}
1 & -1 & 1 & 2 \\
0 & 1 & -1 & -\frac{1}{3} \\
0 & 5 & -3 & -2
\end{bmatrix}$

第3行から第2行の5倍を引く：
$\begin{bmatrix}
1 & -1 & 1 & 2 \\
0 & 1 & -1 & -\frac{1}{3} \\
0 & 0 & 2 & -\frac{1}{3}
\end{bmatrix}$

第3行を2で割る：
$\begin{bmatrix}
1 & -1 & 1 & 2 \\
0 & 1 & -1 & -\frac{1}{3} \\
0 & 0 & 1 & -\frac{1}{6}
\end{bmatrix}$

後退代入すると：
$z = -\frac{1}{6}$
$y - z = -\frac{1}{3}$ より $y = -\frac{1}{3} + z = -\frac{1}{3} - \frac{1}{6} = -\frac{1}{2}$
$x - y + z = 2$ より $x = 2 + y - z = 2 - \frac{1}{2} + \frac{1}{6} = \frac{10}{6} = \frac{5}{3}$

よって、唯一解 $x = \frac{5}{3}$, $y = -\frac{1}{2}$, $z = -\frac{1}{6}$ が存在します。

**問題1(c)**
$\begin{cases}
x + 2y - z = 5 \\
2x + 4y - 2z = 6 \\
3x + 6y - 3z = 15
\end{cases}$

第2式は第1式の2倍、第3式は第1式の3倍なので：
$\begin{cases}
x + 2y - z = 5 \\
2(x + 2y - z) = 6 \\
3(x + 2y - z) = 15
\end{cases}$

第2式：$2 \cdot 5 = 10 \neq 6$ となり矛盾
第3式：$3 \cdot 5 = 15$ は整合

この連立方程式は矛盾を含むため解を持ちません。

**問題2(a)**
$A = \begin{bmatrix}
1 & 3 & 2 \\
2 & 6 & 4 \\
3 & 9 & 6
\end{bmatrix}$, $\mathbf{b} = \begin{bmatrix}
4 \\
8 \\
12
\end{bmatrix}$

まず、係数行列$A$のランクを調べます。行基本変形を適用すると：
$\begin{bmatrix}
1 & 3 & 2 \\
2 & 6 & 4 \\
3 & 9 & 6
\end{bmatrix}$
→
$\begin{bmatrix}
1 & 3 & 2 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}$

よって $\text{rank}(A) = 1$

次に、拡大係数行列 $[A|\mathbf{b}]$ のランクを調べます：
$\begin{bmatrix}
1 & 3 & 2 & 4 \\
2 & 6 & 4 & 8 \\
3 & 9 & 6 & 12
\end{bmatrix}$
→
$\begin{bmatrix}
1 & 3 & 2 & 4 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}$

よって $\text{rank}([A|\mathbf{b}]) = 1$

未知数の数 $n = 3$ で、$\text{rank}(A) = \text{rank}([A|\mathbf{b}]) < n$ なので、この連立方程式は無数の解を持ちます。

**問題2(b)**
$A = \begin{bmatrix}
1 & 3 & 2 \\
2 & 6 & 4 \\
3 & 9 & 6
\end{bmatrix}$, $\mathbf{b} = \begin{bmatrix}
4 \\
8 \\
10
\end{bmatrix}$

$\text{rank}(A) = 1$ であることは前問で確認しました。

拡大係数行列 $[A|\mathbf{b}]$ のランクを調べます：
$\begin{bmatrix}
1 & 3 & 2 & 4 \\
2 & 6 & 4 & 8 \\
3 & 9 & 6 & 10
\end{bmatrix}$

第1行を用いて他の行を消去すると：
$\begin{bmatrix}
1 & 3 & 2 & 4 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & -2
\end{bmatrix}$

第3行は $0 = -2$ という矛盾を含むため、$\text{rank}([A|\mathbf{b}]) = 2$

$\text{rank}(A) < \text{rank}([A|\mathbf{b}])$ なので、この連立方程式は解を持ちません。

**問題3**
行列$A$のランクが2、拡大係数行列$[A|\mathbf{b}]$のランクが3、未知数の数が4の場合：

$\text{rank}(A) = 2 < \text{rank}([A|\mathbf{b}]) = 3$ なので、連立方程式$A\mathbf{x} = \mathbf{b}$は解を持ちません。これは、拡大係数行列のランクが係数行列のランクより大きいということは、右辺ベクトル$\mathbf{b}$が係数行列$A$の列空間に含まれていないことを意味するためです。つまり、どのような未知数の値を選んでも、右辺ベクトル$\mathbf{b}$を係数行列$A$の列ベクトルの線形結合として表すことができません。

### 応用問題の一部解答例

**問題4**
$\begin{cases}
x + y + z = 6 \\
x + 2y + \lambda z = 10 \\
x + 3y + \lambda^2 z = \lambda + 12
\end{cases}$

係数行列$A$と拡大係数行列$[A|\mathbf{b}]$を書き出します：

$A = \begin{bmatrix}
1 & 1 & 1 \\
1 & 2 & \lambda \\
1 & 3 & \lambda^2
\end{bmatrix}$, $[A|\mathbf{b}] = \begin{bmatrix}
1 & 1 & 1 & 6 \\
1 & 2 & \lambda & 10 \\
1 & 3 & \lambda^2 & \lambda + 12
\end{bmatrix}$

ガウスの消去法を適用します：

$\begin{bmatrix}
1 & 1 & 1 & 6 \\
0 & 1 & \lambda-1 & 4 \\
0 & 2 & \lambda^2-1 & \lambda + 6
\end{bmatrix}$

第3行から第2行の2倍を引くと：

$\begin{bmatrix}
1 & 1 & 1 & 6 \\
0 & 1 & \lambda-1 & 4 \\
0 & 0 & \lambda^2-1-2(\lambda-1) & \lambda + 6 - 2 \cdot 4
\end{bmatrix}$

$\begin{bmatrix}
1 & 1 & 1 & 6 \\
0 & 1 & \lambda-1 & 4 \\
0 & 0 & \lambda^2-2\lambda+1 & \lambda - 2
\end{bmatrix}$

$\lambda^2-2\lambda+1 = (\lambda-1)^2$

$\lambda$の値によって解の種類が変わります：

1. $\lambda \neq 1$ の場合：
   $(\lambda-1)^2 \neq 0$ なので、第3行は $(\lambda-1)^2 z = \lambda - 2$ となり、$z = \frac{\lambda-2}{(\lambda-1)^2}$ と決まります。後退代入により唯一解が求まります。

2. $\lambda = 1$ の場合：
   $(\lambda-1)^2 = 0$ となり、第3行は $0 \cdot z = \lambda - 2 = 1 - 2 = -1$ となります。これは $0 = -1$ という矛盾を含むため、$\lambda = 1$ のときは解が存在しません。

したがって、$\lambda = 1$ のときは解なし、$\lambda \neq 1$ のときは唯一解が存在します。

**問題6(a)**
血圧（$y$）を年齢（$x_1$）、体重（$x_2$）、運動時間（$x_3$）の線形関数 $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3$ として表す連立方程式は：

$\begin{cases}
\beta_0 + 25\beta_1 + 70\beta_2 + 3\beta_3 = 120 \\
\beta_0 + 45\beta_1 + 85\beta_2 + 1\beta_3 = 140 \\
\beta_0 + 35\beta_1 + 65\beta_2 + 5\beta_3 = 115 \\
\beta_0 + 55\beta_1 + 75\beta_2 + 2\beta_3 = 145 \\
\beta_0 + 30\beta_1 + 80\beta_2 + 4\beta_3 = 125
\end{cases}$

これは、4つの未知数（$\beta_0, \beta_1, \beta_2, \beta_3$）と5つの方程式からなる過剰決定系です。

**問題6(b)**
この連立方程式は過剰決定系（方程式の数 > 未知数の数）なので、一般的には解を持ちません。実際のデータに完全に適合するパラメータ値を見つけることは通常できないため、最小二乗法などの近似手法を用いて最適なパラメータ値を推定します。

（残りの問題の解答省略）

## 12. まとめ

本講義では、連立一次方程式の解の種類とその判定方法について学びました。主なポイントは以下の通りです：

1. 連立一次方程式の解は「唯一解」「無数の解」「解なし」の3種類に分類される
2. 解の種類は係数行列と拡大係数行列のランクによって判定できる
3. 幾何学的には、解の種類は直線や平面の交わり方に対応する
4. データサイエンスでは過剰決定系の方程式がよく現れ、最小二乗法などで近似解を求める

次回の講義では、連立一次方程式の解の存在条件についてさらに詳しく学び、行列のランクと解の関係性についての理解を深めていきます。