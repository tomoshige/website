# 線形代数学 I / 基礎 / II：第32回講義ノート

## 1. 講義情報と予習ガイド

**講義回**: 第32回  
**テーマ**: 固有値と固有ベクトルのための線形変換の基礎  
**関連項目**: 線形変換、行列による変換、固有値・固有ベクトルの導入  
**予習内容**: 行列の演算、ベクトル空間の基本的性質、正規直交基底、内積の概念

## 2. 学習目標

1. 線形変換の概念と数学的表現を理解する
2. 行列による線形変換の表現方法を習得する
3. 代表的な線形変換（回転、拡大・縮小、せん断など）とその行列表現を学ぶ
4. 線形変換の幾何学的解釈ができるようになる
5. 固有値と固有ベクトルの学習に必要な線形変換の基礎知識を身につける

## 3. 基本概念

### 3.1 線形変換の定義と性質

> **定義 3.1 (線形変換)**  
> ベクトル空間 $V$ から ベクトル空間 $W$ への写像 $T: V \rightarrow W$ が次の2つの条件を満たすとき、$T$ を**線形変換**という：
> 1. 加法性: すべての $\mathbf{u}, \mathbf{v} \in V$ に対して $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
> 2. 斉次性: すべての $\mathbf{v} \in V$ と任意のスカラー $c$ に対して $T(c \mathbf{v}) = c T(\mathbf{v})$

これは、線形変換が「ベクトルの和」と「スカラー倍」という基本的な演算を保存することを意味します。言い換えると、線形変換は次のような性質を持ちます：

- ベクトルを足してから変換しても、変換してから足しても結果は同じ
- ベクトルをスカラー倍してから変換しても、変換してからスカラー倍しても結果は同じ

### 3.2 線形変換の例

線形変換の例として以下のものが挙げられます：

1. **恒等変換** $I(\mathbf{v}) = \mathbf{v}$：ベクトルを変化させない
2. **零変換** $O(\mathbf{v}) = \mathbf{0}$：すべてのベクトルを零ベクトルに写す
3. **スカラー倍** $T(\mathbf{v}) = c\mathbf{v}$：すべてのベクトルを定数 $c$ 倍する
4. **反転** $T(\mathbf{v}) = -\mathbf{v}$：すべてのベクトルの向きを反対にする
5. **射影** $P(\mathbf{v})$：ベクトルを部分空間に射影する

以下の変換も線形変換です：

6. **回転**：ベクトルを特定の角度で回転させる
7. **拡大・縮小**：ベクトルの長さを特定の比率で拡大・縮小する
8. **せん断**：ベクトルをある方向に沿ってずらす

線形ではない変換の例：

- $T(\mathbf{v}) = \mathbf{v} + \mathbf{b}$ (ただし $\mathbf{b} \neq \mathbf{0}$)：平行移動は線形変換ではない
- $T(\mathbf{v}) = \mathbf{v}^2$：二乗は線形変換ではない

### 3.3 線形性の検証

変換が線形かどうかを確認するには、加法性と斉次性を確認します。

**例 3.1**：$\mathbb{R}^2$ 上の変換 $T(\begin{bmatrix} x \\ y \end{bmatrix}) = \begin{bmatrix} 2x \\ 3y \end{bmatrix}$ が線形かどうかを確認しましょう。

1. 加法性：
   $T(\mathbf{u} + \mathbf{v}) = T(\begin{bmatrix} u_1 \\ u_2 \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}) = T(\begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}) = \begin{bmatrix} 2(u_1 + v_1) \\ 3(u_2 + v_2) \end{bmatrix} = \begin{bmatrix} 2u_1 + 2v_1 \\ 3u_2 + 3v_2 \end{bmatrix} = \begin{bmatrix} 2u_1 \\ 3u_2 \end{bmatrix} + \begin{bmatrix} 2v_1 \\ 3v_2 \end{bmatrix} = T(\mathbf{u}) + T(\mathbf{v})$

2. 斉次性：
   $T(c\mathbf{v}) = T(c\begin{bmatrix} v_1 \\ v_2 \end{bmatrix}) = T(\begin{bmatrix} cv_1 \\ cv_2 \end{bmatrix}) = \begin{bmatrix} 2(cv_1) \\ 3(cv_2) \end{bmatrix} = \begin{bmatrix} c(2v_1) \\ c(3v_2) \end{bmatrix} = c\begin{bmatrix} 2v_1 \\ 3v_2 \end{bmatrix} = cT(\mathbf{v})$

よって、この変換は線形です。

**例 3.2**：$\mathbb{R}^2$ 上の変換 $T(\begin{bmatrix} x \\ y \end{bmatrix}) = \begin{bmatrix} x + 1 \\ y - 2 \end{bmatrix}$ が線形かどうかを確認しましょう。

加法性を確認します：
$T(\mathbf{u} + \mathbf{v}) = T(\begin{bmatrix} u_1 \\ u_2 \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}) = T(\begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}) = \begin{bmatrix} (u_1 + v_1) + 1 \\ (u_2 + v_2) - 2 \end{bmatrix} = \begin{bmatrix} u_1 + v_1 + 1 \\ u_2 + v_2 - 2 \end{bmatrix}$

一方、
$T(\mathbf{u}) + T(\mathbf{v}) = \begin{bmatrix} u_1 + 1 \\ u_2 - 2 \end{bmatrix} + \begin{bmatrix} v_1 + 1 \\ v_2 - 2 \end{bmatrix} = \begin{bmatrix} u_1 + v_1 + 2 \\ u_2 + v_2 - 4 \end{bmatrix}$

$T(\mathbf{u} + \mathbf{v}) \neq T(\mathbf{u}) + T(\mathbf{v})$ なので、この変換は線形ではありません。

これは平行移動を含む変換であり、一般に平行移動は線形変換ではないことがわかります。

## 4. 理論と手法

### 4.1 線形変換と行列表現

有限次元ベクトル空間における線形変換は、**行列**を使って表現できます。これは線形代数の最も重要な概念の一つです。

> **定理 4.1**  
> $V$ を $n$ 次元ベクトル空間、$W$ を $m$ 次元ベクトル空間として、$T: V \rightarrow W$ を線形変換とします。$V$ の基底 $\{v_1, v_2, \ldots, v_n\}$ と $W$ の基底 $\{w_1, w_2, \ldots, w_m\}$ に関して、$T$ は一意的に $m \times n$ 行列 $A$ で表現できます。

具体的には、まず基底に対する変換結果を計算します：

$T(v_1) = a_{11}w_1 + a_{21}w_2 + \cdots + a_{m1}w_m$
$T(v_2) = a_{12}w_1 + a_{22}w_2 + \cdots + a_{m2}w_m$
$\vdots$
$T(v_n) = a_{1n}w_1 + a_{2n}w_2 + \cdots + a_{mn}w_m$

このとき、変換 $T$ を表す行列 $A$ は：

$A = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$

となります。

特に $V = W = \mathbb{R}^n$ で標準基底を使う場合、行列 $A$ の $j$ 列目は $T(e_j)$ の座標となります（$e_j$ は $j$ 番目の標準基底ベクトル）。

### 4.2 標準的な線形変換とその行列表現

$\mathbb{R}^2$ における標準的な線形変換の行列表現を見てみましょう。

#### 1. 恒等変換

ベクトルを変化させない変換です。
$I(\mathbf{v}) = \mathbf{v}$

行列表現:
$I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$

#### 2. 拡大・縮小変換

$x$ 方向に $a$ 倍、$y$ 方向に $b$ 倍する変換。
$T(\begin{bmatrix} x \\ y \end{bmatrix}) = \begin{bmatrix} ax \\ by \end{bmatrix}$

行列表現:
$A = \begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix}$

#### 3. 回転変換

原点を中心に角度 $\theta$ だけ反時計回りに回転させる変換。
$T(\begin{bmatrix} x \\ y \end{bmatrix}) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$

行列表現:
$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$

#### 4. せん断変換 (シアー変換)

$x$ 軸方向に $y$ の $k$ 倍だけずらす変換。
$T(\begin{bmatrix} x \\ y \end{bmatrix}) = \begin{bmatrix} x + ky \\ y \end{bmatrix}$

行列表現:
$S = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$

#### 5. 対称変換 (反射)

ある軸に関して対称に移す変換。例えば $x$ 軸に関する反射は：
$T(\begin{bmatrix} x \\ y \end{bmatrix}) = \begin{bmatrix} x \\ -y \end{bmatrix}$

行列表現:
$F_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$

$y$ 軸に関する反射は：
$T(\begin{bmatrix} x \\ y \end{bmatrix}) = \begin{bmatrix} -x \\ y \end{bmatrix}$

行列表現:
$F_y = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$

直線 $y = x$ に関する反射は：
$T(\begin{bmatrix} x \\ y \end{bmatrix}) = \begin{bmatrix} y \\ x \end{bmatrix}$

行列表現:
$F_{y=x} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$

### 4.3 合成変換

2つの線形変換 $S: V \rightarrow W$ と $T: W \rightarrow U$ があるとき、それらの合成 $T \circ S: V \rightarrow U$ も線形変換となります。行列表現では、合成変換は**行列の積**として表されます。

$S$ の行列表現が $A$、$T$ の行列表現が $B$ のとき、$T \circ S$ の行列表現は $BA$ となります。

**例 4.1**：$\mathbb{R}^2$ 上で、まず角度 $\theta$ の回転を行い、次に $x$ 方向に $a$ 倍、$y$ 方向に $b$ 倍する変換を考えます。

回転行列：$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$
拡大縮小行列：$S = \begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix}$

合成変換の行列：$SR_\theta = \begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix} \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} = \begin{bmatrix} a\cos\theta & -a\sin\theta \\ b\sin\theta & b\cos\theta \end{bmatrix}$

注意点として、行列の積は一般に交換法則が成り立ちません。つまり、$AB \neq BA$ となる場合があります。これは変換の順序が結果に影響することを意味します。

### 4.4 線形変換の核と像

線形変換 $T: V \rightarrow W$ に対して、重要な部分空間が2つあります。

> **定義 4.2 (核)**  
> 線形変換 $T: V \rightarrow W$ の**核** (kernel) または**零空間** (null space) は以下で定義される $V$ の部分空間です：
> $\text{Ker}(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}$

> **定義 4.3 (像)**  
> 線形変換 $T: V \rightarrow W$ の**像** (image) または**値域** (range) は以下で定義される $W$ の部分空間です：
> $\text{Im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}$

これらの概念は固有値・固有ベクトルの学習に重要です。特に、核の次元と像の次元には次の関係があります。

> **定理 4.2 (ランク・零度定理)**  
> $V$ を $n$ 次元ベクトル空間、$T: V \rightarrow W$ を線形変換とすると：
> $\dim(\text{Ker}(T)) + \dim(\text{Im}(T)) = \dim(V) = n$

行列 $A$ が線形変換 $T$ を表すとき、$\dim(\text{Ker}(T))$ は $A$ の零度 (nullity)、$\dim(\text{Im}(T))$ は $A$ のランク (rank) と呼ばれます。

### 4.5 線形変換と行列の関係のまとめ

線形変換と行列の関係をまとめると：

1. $n$ 次元ベクトル空間から $m$ 次元ベクトル空間への線形変換は $m \times n$ 行列で表される
2. 行列による演算 $A\mathbf{x}$ は線形変換を表す
3. 線形変換の合成は行列の積に対応する
4. 線形変換の核は連立方程式 $A\mathbf{x} = \mathbf{0}$ の解空間
5. 線形変換の像は行列 $A$ の列ベクトルの張る空間
6. 逆変換の存在は行列が逆行列を持つことと同値

## 5. Pythonによる実装と可視化

### 5.1 基本的な線形変換の実装と可視化

以下のコードは、線形変換とその効果を視覚化するためのPythonの実装例です。

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_transformation(A, points=None, title="Linear Transformation"):
    """
    行列Aで表される線形変換を可視化する関数
    """
    # デフォルトの点（単位正方形の頂点）
    if points is None:
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    
    # 変換前の点を保存
    original_points = points.copy()
    
    # 変換後の点を計算
    transformed_points = np.array([A @ point for point in points])
    
    # プロット領域の設定
    max_val = max(np.max(np.abs(original_points)), np.max(np.abs(transformed_points)))
    limit = max(3, max_val * 1.2)
    
    # 図の作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 変換前
    ax1.grid(True)
    ax1.set_xlim(-limit, limit)
    ax1.set_ylim(-limit, limit)
    ax1.set_aspect('equal')
    ax1.set_title('Original')
    
    # x軸とy軸
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 単位ベクトルを描画
    ax1.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1, fc='r', ec='r')
    ax1.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1, fc='g', ec='g')
    
    # 変換前の図形を描画
    ax1.plot(original_points[:, 0], original_points[:, 1], 'b-', alpha=0.7)
    ax1.fill(original_points[:, 0], original_points[:, 1], 'b', alpha=0.2)
    
    # 変換後
    ax2.grid(True)
    ax2.set_xlim(-limit, limit)
    ax2.set_ylim(-limit, limit)
    ax2.set_aspect('equal')
    ax2.set_title(f'After Transformation: {title}')
    
    # x軸とy軸
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 変換後の単位ベクトルを描画
    e1 = A @ np.array([1, 0])
    e2 = A @ np.array([0, 1])
    ax2.arrow(0, 0, e1[0], e1[1], head_width=0.1, head_length=0.1, fc='r', ec='r')
    ax2.arrow(0, 0, e2[0], e2[1], head_width=0.1, head_length=0.1, fc='g', ec='g')
    
    # 変換後の図形を描画
    ax2.plot(transformed_points[:, 0], transformed_points[:, 1], 'b-', alpha=0.7)
    ax2.fill(transformed_points[:, 0], transformed_points[:, 1], 'b', alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# 変換の例

# 1. 恒等変換
I = np.array([[1, 0], [0, 1]])
plot_transformation(I, title="Identity")

# 2. 拡大・縮小変換
S = np.array([[2, 0], [0, 0.5]])
plot_transformation(S, title="Scaling (x2, y/2)")

# 3. 回転変換 (90度)
theta = np.pi/2
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
plot_transformation(R, title="Rotation (90°)")

# 4. せん断変換
Sh = np.array([[1, 1], [0, 1]])
plot_transformation(Sh, title="Shear")

# 5. 反射変換（x軸について）
F = np.array([[1, 0], [0, -1]])
plot_transformation(F, title="Reflection (x-axis)")
```

### 5.2 行列による複合変換

複数の変換を組み合わせた例を見てみましょう。

```python
# 回転してから拡大する複合変換
theta = np.pi/4  # 45度回転
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
S = np.array([[2, 0], [0, 0.5]])

# 複合変換の行列（拡大してから回転）
T1 = R @ S
plot_transformation(T1, title="Rotation after Scaling")

# 複合変換の行列（回転してから拡大）
T2 = S @ R
plot_transformation(T2, title="Scaling after Rotation")
```

### 5.3 線形変換のアニメーション

線形変換を連続的に可視化するアニメーションを作成してみましょう。

```python
from matplotlib.animation import FuncAnimation

def animate_transformation(A, frames=50, interval=100):
    """
    単位正方形に対する線形変換をアニメーション化する関数
    """
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    I = np.eye(2)
    
    # 図の初期設定
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(True)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title('Linear Transformation Animation')
    
    # x軸とy軸
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 初期状態の図形
    line, = ax.plot([], [], 'b-', alpha=0.7)
    fill = ax.fill([], [], 'b', alpha=0.2)[0]
    
    # 単位ベクトル用の矢印
    arrow_e1, = ax.plot([], [], 'r-', lw=2)
    arrow_e2, = ax.plot([], [], 'g-', lw=2)
    
    def init():
        line.set_data([], [])
        fill.set_xy(np.zeros((5, 2)))
        arrow_e1.set_data([], [])
        arrow_e2.set_data([], [])
        return line, fill, arrow_e1, arrow_e2
    
    def update(frame):
        t = frame / frames
        # 恒等変換から目標の変換へと徐々に変化
        current_matrix = I + t * (A - I)
        
        # 現在の変換を適用
        current_points = np.array([current_matrix @ point for point in points])
        
        # 図形の更新
        line.set_data(current_points[:, 0], current_points[:, 1])
        fill.set_xy(current_points)
        
        # 単位ベクトルの更新
        e1 = current_matrix @ np.array([1, 0])
        e2 = current_matrix @ np.array([0, 1])
        arrow_e1.set_data([0, e1[0]], [0, e1[1]])
        arrow_e2.set_data([0, e2[0]], [0, e2[1]])
        
        return line, fill, arrow_e1, arrow_e2
    
    ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                        interval=interval, blit=True)
    plt.close()  # IPython上でアニメーションが二重表示されるのを防ぐ
    
    return ani

# 45度回転のアニメーション
theta = np.pi/4  # 45度
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
ani_rotation = animate_transformation(R)

# 拡大縮小のアニメーション
S = np.array([[2, 0], [0, 0.5]])
ani_scaling = animate_transformation(S)

# せん断変換のアニメーション
Sh = np.array([[1, 1], [0, 1]])
ani_shear = animate_transformation(Sh)

# HTML出力でアニメーションを表示
from IPython.display import HTML
HTML(ani_rotation.to_jshtml())
```

### 5.4 データサイエンスでの線形変換の応用例

健康データに対する線形変換の応用例を見てみましょう。

```python
# 健康データの例：(体重, 身長)のデータセット
np.random.seed(42)
height = 170 + np.random.normal(0, 5, 100)  # 身長 (cm)
weight = 0.5 * height - 35 + np.random.normal(0, 5, 100)  # 体重 (kg)

# 正規化のための線形変換
height_mean, height_std = np.mean(height), np.std(height)
weight_mean, weight_std = np.mean(weight), np.std(weight)

# 正規化変換行列 (対角行列)
normalize_matrix = np.array([[1/height_std, 0], [0, 1/weight_std]])

# 生データをプロット
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(height, weight, alpha=0.7)
plt.title('Original Data: Height vs Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.grid(True)

# データを行列形式に変換
data = np.vstack((height, weight)).T

# 平均を引いてから正規化変換を適用
centered_data = data - np.array([height_mean, weight_mean])
normalized_data = np.array([normalize_matrix @ point for point in centered_data])

# 正規化後のデータをプロット
plt.subplot(1, 2, 2)
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], alpha=0.7)
plt.title('Normalized Data')
plt.xlabel('Normalized Height')
plt.ylabel('Normalized Weight')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. 演習問題

### 6.1 基本問題 (演習時間: 約20分)

1. 以下の写像が線形変換かどうかを判定せよ。線形変換でない場合は、線形性のどの性質が満たされないかを示せ。
   a) $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2, T\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 3x \\ 2y-1 \end{pmatrix}$

   b) $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2, T\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x+y \\ x-y \end{pmatrix}$
   
   c) $T: \mathbb{R}^2 \rightarrow \mathbb{R}, T\begin{pmatrix} x \\ y \end{pmatrix} = xy$
   
   d) $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2, T\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x^2 \\ y \end{pmatrix}$

2. 以下の線形変換の行列表現を求めよ。
   a) $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ で、ベクトルを $x$ 軸に関して反射させる変換
   b) $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ で、原点を中心に $60^\circ$ 反時計回りに回転させる変換
   c) $T: \mathbb{R}^3 \rightarrow \mathbb{R}^3$ で、$z$ 軸方向に $2$ 倍に拡大し、$x$ 軸と $y$ 軸方向はそのままにする変換

3. 次の行列で表される線形変換の核と像を求めよ。
   $A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$

4. 行列 $A = \begin{pmatrix} 2 & 1 \\ -1 & 3 \end{pmatrix}$ と $B = \begin{pmatrix} 0 & 2 \\ 1 & -1 \end{pmatrix}$ に対して、以下を計算せよ。
   a) $AB$ と $BA$
   b) $A$ を行列とする線形変換を $T_A$、$B$ を行列とする線形変換を $T_B$ としたとき、合成変換 $T_B \circ T_A$ と $T_A \circ T_B$ の行列表現

### 6.2 応用問題 (演習時間: 約30分)

1. $\mathbb{R}^2$ 上の線形変換 $T$ が行列 $A = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix}$ で表されるとき、ベクトル $\begin{pmatrix} 2 \\ -1 \end{pmatrix}$ の像を求めよ。また、$T$ の核を求めよ。

2. $\mathbb{R}^2$ において、点 $(0, 0)$, $(1, 0)$, $(0, 1)$, $(1, 1)$ からなる単位正方形があるとする。以下の線形変換を順番に適用したときの、この正方形の変換後の頂点の座標を求めよ。
   a) まず $x$ 軸方向に $2$ 倍に拡大し、$y$ 軸方向に $0.5$ 倍に縮小する
   b) 次に原点を中心に $45^\circ$ 反時計回りに回転させる

3. 次の行列で表される線形変換について、その幾何学的意味を説明せよ。
   a) $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$
   b) $B = \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}$
   c) $C = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$

4. 健康データ分析の応用問題：ある研究で、患者の血圧（収縮期、拡張期）のデータが集められている。このデータに対して、以下の線形変換を考える。
   
   $T\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} \frac{x-120}{10} \\ \frac{y-80}{5} \end{pmatrix}$
   
   ここで $x$ は収縮期血圧（mmHg）、$y$ は拡張期血圧（mmHg）である。
   
   a) この変換の幾何学的意味を説明せよ
   b) 正常血圧（120/80 mmHg）、軽度高血圧（140/90 mmHg）、重度高血圧（160/100 mmHg）が変換後どのような点に写像されるかを計算せよ
   c) この変換を用いることで、血圧データを分析する際にどのような利点があるか説明せよ

## 7. よくある質問と解答

### Q1: 線形変換と行列の関係を直感的に理解するには？

**A1:** 線形変換とは、ベクトルの足し算とスカラー倍の構造を保存する写像です。行列による演算はまさにこの性質を持っています。行列とベクトルの積 $A\mathbf{v}$ を計算するとき、実は行列 $A$ の列ベクトルを基底ベクトルの像と考えることができます。例えば、$2 \times 2$ 行列の場合、第1列は標準基底ベクトル $(1,0)$ の変換後の位置、第2列は標準基底ベクトル $(0,1)$ の変換後の位置を表します。つまり、行列はベクトル空間の基底がどのように変換されるかを記述しているのです。

### Q2: なぜ行列の積は一般に可換ではないのですか？

**A2:** 行列の積が可換でない（$AB \neq BA$）のは、変換の順序が結果に影響するという事実を反映しています。例えば、「まず回転してから拡大する」という操作と、「まず拡大してから回転する」という操作は異なる結果を生じます。幾何学的な変換を順番に適用するとき、その順序が重要なのは直感的にも理解できるでしょう。

### Q3: 線形変換の核と像はデータサイエンスでどのように役立ちますか？

**A3:** データサイエンスでは、高次元データを扱うことが多いです。線形変換の核は、「変換によって同じ点に写像される入力の集合」を表し、変換によって失われる情報の方向を示します。一方、像は「変換によって到達可能な出力の集合」を表し、データの本質的な構造や特徴を捉える部分空間を表すことがあります。例えば、主成分分析（PCA）では、データの分散が最大になる方向（主成分）を見つけることで、データの重要な特徴を抽出しますが、これは線形変換の像を最適化する問題と見なすことができます。

### Q4: 回転行列はなぜ $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ という形になるのですか？

**A4:** 回転行列の形は、単位ベクトルが回転後にどのような座標になるかを考えると導けます。標準基底ベクトル $(1,0)$ を角度 $\theta$ だけ回転させると、新しい座標は $(\cos\theta, \sin\theta)$ になります。同様に、もう一つの標準基底ベクトル $(0,1)$ を回転させると、新しい座標は $(-\sin\theta, \cos\theta)$ になります。これらが回転行列の第1列と第2列になります。回転行列の重要な特徴は、それが直交行列（$R^TR = I$）であることで、これは回転が長さと角度を保存することを意味します。

### Q5: 線形変換と線形性の概念は今後の固有値・固有ベクトルの学習にどう関連していますか？

**A5:** 固有値と固有ベクトルは、線形変換の特別な挙動を記述します。固有ベクトルとは、線形変換によって方向が変わらないベクトル（つまり、変換後のベクトルが元のベクトルの定数倍になる）のことです。固有値はその「定数倍」の値を指します。線形変換の概念を理解していれば、固有値・固有ベクトルは線形変換の「不変方向」を見つけるものだという直感を得ることができます。これは主成分分析などのデータ分析手法の理論的基盤となっており、データの重要な特徴や構造を抽出するのに役立ちます。

### Q6: 行列で表せない線形変換はありますか？

**A6:** 有限次元ベクトル空間間の線形変換は、適切な基底を選べば必ず行列で表現できます。しかし、無限次元のベクトル空間（関数空間など）における線形変換は、有限の行列では表現できないことがあります。これらは線形作用素と呼ばれ、微分や積分などの演算がその例です。ただし、データサイエンスの実践では通常、有限次元の問題を扱うため、行列表現で十分です。

### Q7: 線形性の条件（加法性と斉次性）はなぜ重要なのですか？

**A7:** 線形性の条件は、変換の挙動を単純化し予測可能にするために重要です。これらの条件により、線形変換は少数のベクトル（基底）の変換結果だけを知れば、任意のベクトルの変換結果を予測できるという強力な性質を持ちます。データサイエンスでは、複雑なシステムを近似するために線形モデルが広く使われますが、これは線形モデルが理解しやすく、計算が効率的で、多くの場合に十分な精度を提供するためです。