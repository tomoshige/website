# 線形代数学 I 第12回 講義ノート

## 1. 講義情報と予習ガイド

**講義回**: 第12回  
**テーマ**: ランクの概念とその計算  
**関連項目**: 階段行列、簡約階段行列、行基本変形、行列のランク  
**予習すべき内容**: 連立一次方程式の行列表現、行列の基本変形、ガウスの消去法

## 2. 学習目標

1. 階段行列と簡約階段行列の定義と性質を理解する
2. 行基本変形を用いて階段行列と簡約階段行列を導出できるようになる
3. 行列のランクの概念を理解し、その意味を説明できるようになる
4. 行列のランクを計算できるようになる
5. ランクとベクトルの一次独立性の関係を理解する

## 3. 基本概念

### 3.1 階段行列の定義

> **定義**: 階段行列（echelon form）とは、以下の条件を満たす行列のことです。
> 1. すべてのゼロ行（要素がすべて0の行）は行列の下部に集められている
> 2. 各行の先頭の非ゼロ要素（先頭係数）は、その上の行の先頭非ゼロ要素よりも右にある
> 3. 先頭係数の列より左の列はすべて0である

階段行列は「階段のような形」をしています。次の行列は階段行列の例です：

$$
\begin{pmatrix}
2 & 3 & 1 & 7 \\
0 & 4 & -3 & 2 \\
0 & 0 & 1 & 5 \\
0 & 0 & 0 & 0
\end{pmatrix}
$$

この行列では：
- 1行目の先頭係数は第1列の2
- 2行目の先頭係数は第2列の4
- 3行目の先頭係数は第3列の1
- 4行目はゼロ行

このように、先頭係数が右下へと階段状に並んでいるため「階段行列」と呼ばれます。

### 3.2 簡約階段行列の定義

> **定義**: 簡約階段行列（reduced echelon form）とは、以下の追加条件を満たす階段行列のことです。
> 1. 各行の先頭係数は1である
> 2. 先頭係数のある列において、その先頭係数以外の要素はすべて0である

簡約階段行列の例：

$$
\begin{pmatrix}
1 & 0 & 0 & 2 \\
0 & 1 & 0 & -1 \\
0 & 0 & 1 & 3 \\
0 & 0 & 0 & 0
\end{pmatrix}
$$

この行列では：
- すべての先頭係数が1
- 先頭係数のある列（第1, 2, 3列）では、先頭係数以外の要素がすべて0

特に、各行の先頭係数が単位行列の形になっている部分があることに注目してください。

### 3.3 行列のランクの定義

> **定義**: 行列Aのランク（rank）とは、Aを階段行列（または簡約階段行列）に変形したときの非ゼロ行の数です。これは、（後で学ぶ）行列内の線形独立な行（または列）ベクトルの最大数を表します。

ランクは次のような記号で表します：
$\text{rank}(A)$ または $\text{r}(A)$

重要な性質：
- m×n行列のランクrは、r ≤ min(m, n) を満たす
- 行列のランクは、行基本変形によって変わらない
- 行列のランクは、その行ベクトルの最大線形独立数と列ベクトルの最大線形独立数に等しい

例えば、以下の階段行列のランクは3です（非ゼロ行が3行あるため）：

$$
\begin{pmatrix}
1 & 3 & 2 & 7 \\
0 & 1 & -3 & 2 \\
0 & 0 & 1 & 5 \\
0 & 0 & 0 & 0
\end{pmatrix}
$$

## 4. 理論と手法

### 4.1 行基本変形による階段行列の導出方法

行基本変形によって任意の行列を階段行列に変形できます。行基本変形には以下の3種類があります：

1. 行の入れ替え
2. 行のスカラー倍
3. ある行の定数倍を別の行に加える

階段行列への変形手順（ガウスの消去法）：

1. 左端の列から順に処理する
2. 処理中の列で、まだ処理していない行の中から先頭係数を選ぶ
3. 必要なら行を入れ替えて、その先頭係数を上に移動させる
4. その先頭係数を含む行を使って、同じ列の他の要素をすべて0にする
5. 次の列に移り、同様の処理を繰り返す

### 4.2 簡約階段行列への変形方法

階段行列から簡約階段行列への変形手順：

1. 最下行から順に上へ処理する
2. 各行の先頭係数を1にするため、行全体をその係数で割る
3. その先頭係数の列において、他のすべての要素を0にする
4. 前の行に移り、同様の処理を繰り返す

### 4.3 ランクの計算方法

行列のランクを計算する手順：

1. 行基本変形を用いて行列を階段行列に変形する
2. 非ゼロ行の数を数える

または：

1. 行基本変形を用いて行列を簡約階段行列に変形する
2. 非ゼロ行の数を数える

どちらの方法でも同じ結果が得られます。

### 4.4 行列のランクと一次独立性の関係

ランクには以下の重要な解釈があります：

- 行列Aのランクは、Aの行ベクトルの中で一次独立なものの最大数
- 同様に、Aのランクは、Aの列ベクトルの中で一次独立なものの最大数

これにより、ランクはベクトル空間の「次元」の概念と密接に関係します。ランクが持つ重要な性質：

- $\text{rank}(A) = \text{rank}(A^T)$（転置行列のランクは元の行列と同じ）
- $\text{rank}(A) = \text{rank}(AA^T) = \text{rank}(A^TA)$
- $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$（行列の積のランク）

## 5. 具体的な計算例

### 例題1: 階段行列への変形

行列Aを階段行列に変形してみましょう。

$$A = 
\begin{pmatrix}
2 & 1 & 3 & 7 \\
4 & -1 & 2 & 10 \\
6 & 3 & 9 & 18
\end{pmatrix}
$$

**解答**:

ステップ1: 第1列を処理します。先頭係数として第1行の2を選びます。

$$
\begin{pmatrix}
2 & 1 & 3 & 7 \\
4 & -1 & 2 & 10 \\
6 & 3 & 9 & 18
\end{pmatrix}
$$

第2行の要素を0にするため、第1行の(-2)倍を第2行に加えます。
$R_2 \leftarrow R_2 + (-2)R_1$

$$
\begin{pmatrix}
2 & 1 & 3 & 7 \\
0 & -3 & -4 & -4 \\
6 & 3 & 9 & 18
\end{pmatrix}
$$

第3行の要素を0にするため、第1行の(-3)倍を第3行に加えます。
$R_3 \leftarrow R_3 + (-3)R_1$

$$
\begin{pmatrix}
2 & 1 & 3 & 7 \\
0 & -3 & -4 & -4 \\
0 & 0 & 0 & -3
\end{pmatrix}
$$

ステップ2: 第2列を処理します。先頭係数として第2行の-3を選びます。

第3行は既に第2列が0なので処理する必要はありません。

ステップ3: 第3列を処理します。第3行は全て0になってしまっているため、第3列の処理は必要ありません。

結果として、階段行列は次のようになります：

$$
\begin{pmatrix}
2 & 1 & 3 & 7 \\
0 & -3 & -4 & -4 \\
0 & 0 & 0 & -3
\end{pmatrix}
$$

この行列は非ゼロ行が3行あるため、元の行列Aのランクは3です。

### 例題2: 簡約階段行列への変形

先ほどの階段行列を簡約階段行列に変形しましょう。

$$
\begin{pmatrix}
2 & 1 & 3 & 7 \\
0 & -3 & -4 & -4 \\
0 & 0 & 0 & -3
\end{pmatrix}
$$

**解答**:

ステップ1: 最下行から処理します。第3行の先頭係数を1にするため、行全体を-3で割ります。
$R_3 \leftarrow R_3 / (-3)$

$$
\begin{pmatrix}
2 & 1 & 3 & 7 \\
0 & -3 & -4 & -4 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

ステップ2: 第2行に移ります。先頭係数を1にするため、行全体を-3で割ります。
$R_2 \leftarrow R_2 / (-3)$

$$
\begin{pmatrix}
2 & 1 & 3 & 7 \\
0 & 1 & 4/3 & 4/3 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

第4列の第2行の要素を0にするため、第3行の(-4/3)倍を第2行に加えます。
$R_2 \leftarrow R_2 + (-4/3)R_3$

$$
\begin{pmatrix}
2 & 1 & 3 & 7 \\
0 & 1 & 4/3 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

ステップ3: 第1行に移ります。先頭係数を1にするため、行全体を2で割ります。
$R_1 \leftarrow R_1 / 2$

$$
\begin{pmatrix}
1 & 1/2 & 3/2 & 7/2 \\
0 & 1 & 4/3 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

第2列の第1行の要素を0にするため、第2行の(-1/2)倍を第1行に加えます。
$R_1 \leftarrow R_1 + (-1/2)R_2$

$$
\begin{pmatrix}
1 & 0 & 5/6 & 7/2 \\
0 & 1 & 4/3 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

第4列の第1行の要素を0にするため、第3行の(-7/2)倍を第1行に加えます。
$R_1 \leftarrow R_1 + (-7/2)R_3$

$$
\begin{pmatrix}
1 & 0 & 5/6 & 0 \\
0 & 1 & 4/3 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

これが簡約階段行列であり、非ゼロ行が3行あるので、ランクは3です。

### 例題3: ランクと連立方程式の解の関係

連立方程式が次のように与えられています：

$$
\begin{align}
2x + y + 3z &= 7 \\
4x - y + 2z &= 10 \\
6x + 3y + 9z &= 18
\end{align}
$$

係数行列と拡大係数行列のランクを求め、この連立方程式の解について考察してください。

**解答**:

係数行列Aと拡大係数行列[A|b]は以下のようになります：

$$A = 
\begin{pmatrix}
2 & 1 & 3 \\
4 & -1 & 2 \\
6 & 3 & 9
\end{pmatrix}, \quad [A|b] = 
\begin{pmatrix}
2 & 1 & 3 & 7 \\
4 & -1 & 2 & 10 \\
6 & 3 & 9 & 18
\end{pmatrix}
$$

ガウスの消去法を用いて[A|b]を階段行列に変形します（上の例題1の計算と同じです）：

$$
\begin{pmatrix}
2 & 1 & 3 & 7 \\
0 & -3 & -4 & -4 \\
0 & 0 & 0 & -3
\end{pmatrix}
$$

この行列から、係数行列Aのランクは2、拡大係数行列[A|b]のランクは3であることがわかります。

$\text{rank}(A) = 2 \neq \text{rank}([A|b]) = 3$

ランクの不一致から、この連立方程式は解を持たない（不能）ことがわかります。第3行が $0x + 0y + 0z = -3$ という矛盾した式になっているためです。

## 6. Pythonによる実装と可視化

### 6.1 階段行列と簡約階段行列の計算

まず、NumPyを使って行列を階段行列と簡約階段行列に変換する関数を実装します。

```python
import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 行列を見やすく表示する関数
def print_matrix(A, precision=4):
    for row in A:
        print('[', end=' ')
        for elem in row:
            if isinstance(elem, Fraction):
                print(f"{float(elem):.{precision}f}", end=' ')
            else:
                print(f"{elem:.{precision}f}", end=' ')
        print(']')
    print()

# ガウスの消去法により階段行列を求める関数
def to_echelon_form(A):
    A = A.copy().astype(float)
    m, n = A.shape
    r = 0  # 現在の行
    
    for c in range(n):  # 列について処理
        # 現在の列でピボットとなる非ゼロ要素を探す
        for i in range(r, m):
            if abs(A[i, c]) > 1e-10:  # 数値計算の誤差を考慮
                # 行を入れ替え
                A[[r, i]] = A[[i, r]]
                # この行をピボットとして使用
                pivot = A[r, c]
                # 他の行からこの行の倍数を引いて0にする
                for j in range(r+1, m):
                    factor = A[j, c] / pivot
                    A[j] = A[j] - factor * A[r]
                r += 1
                break
        
        if r == m:  # すべての行を処理したら終了
            break
    
    return A

# 簡約階段行列を求める関数
def to_reduced_echelon_form(A):
    A = to_echelon_form(A)
    m, n = A.shape
    
    # 非ゼロ行を特定
    nonzero_rows = []
    for i in range(m):
        for j in range(n):
            if abs(A[i, j]) > 1e-10:
                nonzero_rows.append(i)
                break
    
    # 下から上に処理
    for i in reversed(nonzero_rows):
        # 先頭の非ゼロ要素（ピボット）を見つける
        pivot_col = -1
        for j in range(n):
            if abs(A[i, j]) > 1e-10:
                pivot_col = j
                break
        
        if pivot_col == -1:
            continue
        
        # ピボットを1にする
        A[i] = A[i] / A[i, pivot_col]
        
        # 他の行のピボット列の要素を0にする
        for k in range(i):
            factor = A[k, pivot_col]
            A[k] = A[k] - factor * A[i]
    
    return A

# ランクを計算する関数
def rank(A):
    echelon = to_echelon_form(A)
    r = 0
    for i in range(echelon.shape[0]):
        if any(abs(echelon[i, j]) > 1e-10 for j in range(echelon.shape[1])):
            r += 1
    return r

# サンプル行列
A = np.array([
    [2, 1, 3, 7],
    [4, -1, 2, 10],
    [6, 3, 9, 18]
], dtype=float)

print("元の行列:")
print_matrix(A)

echelon = to_echelon_form(A)
print("階段行列:")
print_matrix(echelon)

reduced_echelon = to_reduced_echelon_form(A)
print("簡約階段行列:")
print_matrix(reduced_echelon)

print(f"行列のランク: {rank(A)}")
```

### 6.2 ランクの幾何学的意味の可視化

ランクの幾何学的意味を可視化するために、3次元空間での列ベクトルと、それらのなす部分空間を描画します。

```python
def visualize_rank(A):
    # 3次元までの行列のみ対応
    if A.shape[1] > 3:
        A = A[:, :3]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 原点
    ax.scatter([0], [0], [0], color='black', s=100, label='Origin')
    
    # 列ベクトルを描画
    colors = ['r', 'g', 'b']
    for i in range(A.shape[1]):
        vec = A[:, i]
        if len(vec) < 3:
            vec = np.append(vec, [0] * (3 - len(vec)))
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=colors[i], arrow_length_ratio=0.1, 
                  label=f'Column {i+1}')
    
    # ランクに基づいて次元を表示
    r = rank(A)
    
    if r == 1:  # 線を描画
        # 最初の非ゼロ列ベクトルを取得
        col_idx = 0
        for i in range(A.shape[1]):
            if np.linalg.norm(A[:, i]) > 1e-10:
                col_idx = i
                break
        
        vec = A[:, col_idx]
        if len(vec) < 3:
            vec = np.append(vec, [0] * (3 - len(vec)))
        
        t = np.linspace(-2, 2, 100)
        x = t * vec[0]
        y = t * vec[1]
        z = t * vec[2]
        ax.plot(x, y, z, 'k--', alpha=0.5)
        plt.title(f'Rank = {r}: Vectors span a line')
    
    elif r == 2:  # 平面を描画
        # 線形独立な2つの列ベクトルを見つける
        cols = []
        for i in range(A.shape[1]):
            if len(cols) == 0:
                if np.linalg.norm(A[:, i]) > 1e-10:
                    cols.append(i)
            elif len(cols) == 1:
                temp = A[:, [cols[0], i]]
                if rank(temp) == 2:
                    cols.append(i)
                    break
        
        if len(cols) >= 2:
            vec1 = A[:, cols[0]]
            vec2 = A[:, cols[1]]
            
            if len(vec1) < 3:
                vec1 = np.append(vec1, [0] * (3 - len(vec1)))
            if len(vec2) < 3:
                vec2 = np.append(vec2, [0] * (3 - len(vec2)))
            
            # 平面を描画
            xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
            
            # 平面の法線ベクトル
            normal = np.cross(vec1, vec2)
            
            if np.linalg.norm(normal) > 1e-10:
                d = 0  # 平面の方程式 ax + by + cz + d = 0
                z = (-normal[0] * xx - normal[1] * yy - d) / max(normal[2], 1e-10)
                ax.plot_surface(xx, yy, z, alpha=0.2, color='cyan')
                plt.title(f'Rank = {r}: Vectors span a plane')
    
    elif r == 3:
        plt.title(f'Rank = {r}: Vectors span the full 3D space')
    
    # グラフの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    plt.legend()
    plt.grid(True)
    plt.show()

# 異なるランクの行列を可視化
# ランク1の例（列ベクトルが線形従属）
A1 = np.array([
    [1, 2],
    [2, 4],
    [3, 6]
])

# ランク2の例（列ベクトルが平面を張る）
A2 = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
])

# ランク3の例（列ベクトルが3次元空間を張る）
A3 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

print("ランク1の行列:")
print_matrix(A1)
visualize_rank(A1)

print("ランク2の行列:")
print_matrix(A2)
visualize_rank(A2)

print("ランク3の行列:")
print_matrix(A3)
visualize_rank(A3)
```

### 6.3 ランクと連立方程式の関係

ランクと連立方程式の解の存在条件の関係を視覚化します。

```python
def analyze_system(A, b):
    # 係数行列
    coef_matrix = A
    # 拡大係数行列
    augmented_matrix = np.column_stack((A, b))
    
    # ランクを計算
    r_A = rank(coef_matrix)
    r_Aug = rank(augmented_matrix)
    
    print(f"係数行列のランク: {r_A}")
    print(f"拡大係数行列のランク: {r_Aug}")
    
    # 解の存在と種類を判定
    if r_A != r_Aug:
        print("解なし（不能）")
        solution_type = "no solution"
    elif r_A < A.shape[1]:
        print(f"無数の解（不定）- 自由変数の数: {A.shape[1] - r_A}")
        solution_type = "infinitely many solutions"
    else:
        print("唯一解")
        solution_type = "unique solution"
    
    # 簡約階段行列を表示
    reduced = to_reduced_echelon_form(augmented_matrix)
    print("簡約階段行列:")
    print_matrix(reduced)
    
    return r_A, r_Aug, solution_type

# 例1: 唯一解を持つ系
A1 = np.array([
    [2, 1],
    [1, 1]
])
b1 = np.array([[4], [3]])

# 例2: 無数の解を持つ系
A2 = np.array([
    [1, 2, 3],
    [2, 4, 6]
])
b2 = np.array([[6], [12]])

# 例3: 解を持たない系
A3 = np.array([
    [1, 1],
    [2, 2]
])
b3 = np.array([[2], [5]])

print("例1: ")
analyze_system(A1, b1)
print("\n例2: ")
analyze_system(A2, b2)
print("\n例3: ")
analyze_system(A3, b3)
```

### 6.4 データセットにおけるランクと次元削減

実際のデータ分析では、ランクの概念は次元削減と密接に関連しています。ここでは簡単な例として、高次元データの主成分分析（PCA）の前処理としてのランク計算を示します。

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

# アヤメのデータセットを読み込む
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# データフレームとして表示
df = pd.DataFrame(X, columns=feature_names)
print("Iris データセット（最初の5行）:")
print(df.head())

# データ行列のランクを計算
r = rank(X)
print(f"\nデータ行列のランク: {r}")

# 相関行列を計算
X_centered = X - X.mean(axis=0)
corr_matrix = np.corrcoef(X.T)
print("\n特徴量間の相関行列:")
print_matrix(corr_matrix)

# 相関行列のランク
r_corr = rank(corr_matrix)
print(f"相関行列のランク: {r_corr}")

# 主成分分析を実行
pca = PCA()
pca.fit(X)

# 累積寄与率を計算
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# 結果をプロット
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.6, label='Individual explained variance')
plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid', label='Cumulative explained variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA: Explained Variance by Components')
plt.legend()
plt.grid(True)
plt.show()

print("\n各主成分の寄与率:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"主成分{i+1}: {ratio:.4f} ({ratio*100:.22f}%)")
print("\n累積寄与率:")
for i, ratio in enumerate(cumulative_variance_ratio):
print(f"主成分{i+1}まで: {ratio:.4f} ({ratio*100:.2f}%)")
```

実行すると、Irisデータセットの4つの特徴量の間の相関関係を分析し、データ行列のランクと主成分分析の結果を表示します。このように、ランクの概念はデータの本質的な次元数を把握するのに役立ちます。

## 7. 演習問題

### 7.1 基本問題

1. 以下の行列を階段行列に変形し、ランクを求めなさい。
   $$A = \begin{pmatrix}
   1 & 2 & 3 & 4 \\
   2 & 4 & 6 & 8 \\
   3 & 6 & 8 & 10
   \end{pmatrix}$$

2. 以下の行列を簡約階段行列に変形し、ランクを求めなさい。
   $$B = \begin{pmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
   \end{pmatrix}$$

3. 以下の行列のランクを求め、列ベクトルの線形独立性について考察しなさい。
   $$C = \begin{pmatrix}
   2 & 4 & 2 \\
   1 & 2 & 1 \\
   3 & 6 & 3
   \end{pmatrix}$$

4. 以下の行列のランクと、その転置行列のランクを求め、両者が等しいことを確認しなさい。
   $$D = \begin{pmatrix}
   1 & 0 & 2 & -1 \\
   0 & 1 & 3 & 2 \\
   2 & 1 & 7 & 0
   \end{pmatrix}$$

5. 以下の連立方程式を行列で表し、係数行列と拡大係数行列のランクから、解の存在と種類について議論しなさい。
   $$
   \begin{align}
   x + 2y - z &= 5 \\
   2x + 4y - 2z &= 10 \\
   3x + 6y - 3z &= 20
   \end{align}
   $$

### 7.2 応用問題

1. 行列 $A$ と $B$ について、$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$ となることを、次の例を用いて確認しなさい。
   $$A = \begin{pmatrix}
   1 & 2 \\
   3 & 4 \\
   5 & 6
   \end{pmatrix}, B = \begin{pmatrix}
   1 & 0 & 2 \\
   0 & 1 & 1
   \end{pmatrix}$$

2. 行列 $A$ が次のとき、$\text{rank}(A^T A)$ と $\text{rank}(A A^T)$ を計算し、両者の関係について考察しなさい。
   $$A = \begin{pmatrix}
   1 & 0 & 2 \\
   0 & 1 & 1 \\
   0 & 0 & 0
   \end{pmatrix}$$

3. 正方行列 $A$ について、$\text{rank}(A) = n$ であることと、$A$ が正則（逆行列が存在する）であることが同値であることを、次の例を用いて確認しなさい。
   $$A_1 = \begin{pmatrix}
   1 & 2 \\
   3 & 4
   \end{pmatrix}, A_2 = \begin{pmatrix}
   1 & 2 \\
   2 & 4
   \end{pmatrix}$$

4. 健康データの分析において、複数の生体指標（血圧、心拍数、体温など）を含むデータセットがあるとき、これらの指標間に強い相関がある場合、データ行列のランクはデータの次元数よりも小さくなることがあります。この状況を以下の行列を例にして説明しなさい。
   $$
   X = \begin{pmatrix}
   120 & 72 & 36.5 & 96 \\
   115 & 70 & 36.7 & 94 \\
   125 & 75 & 36.3 & 98 \\
   130 & 78 & 36.2 & 100 \\
   118 & 71 & 36.6 & 95
   \end{pmatrix}
   $$
   
   ここで各列は、収縮期血圧、拡張期血圧、体温、心拍数を表します。もし収縮期血圧と拡張期血圧に強い相関がある場合、データ行列のランクはどのようになるか考察しなさい。

## 8. よくある質問と解答

### Q1: 行列のランクと次元の違いは何ですか？

A1: 行列の「次元」は通常、行数×列数の形式で表される行列のサイズを指します（例：3×4行列）。一方、「ランク」は行列内の線形独立な行または列ベクトルの最大数です。ランクは行列が表現できる部分空間の次元を表します。

m×n行列のランクrは常に r ≤ min(m, n) を満たします。ランクが最大（r = min(m, n)）のとき、フルランク行列と呼びます。

### Q2: 階段行列と簡約階段行列の違いは何ですか？

A2: 階段行列と簡約階段行列の主な違いは以下の通りです：

- 階段行列：
  - 非ゼロ行は上部に集められる
  - 各行の先頭の非ゼロ要素（先頭係数）は、前の行の先頭係数より右にある
  
- 簡約階段行列は階段行列の条件に加えて：
  - 各行の先頭係数は1である
  - 先頭係数のある列では、その他の要素はすべて0である

簡約階段行列はより厳しい条件を持ち、連立方程式の解を直接読み取りやすい形になっています。

### Q3: なぜ行基本変形は行列のランクを変えないのですか？

A3: 行基本変形（行の入れ替え、行のスカラー倍、ある行の定数倍を別の行に加える）は、行ベクトルの線形結合を取る操作です。これらの操作は行ベクトルの線形従属性を保存します。つまり、変形前に線形従属だった行ベクトルの集合は、変形後も線形従属です。同様に、線形独立だった行ベクトルの集合も線形独立性を保ちます。

したがって、最大線形独立ベクトル数（ランク）は行基本変形によって変わりません。

### Q4: ランク落ち（rank deficient）行列とは何ですか？

A4: ランク落ち行列とは、そのランクが行数または列数（小さい方）より小さい行列のことです。つまり、m×n行列において、rank(A) < min(m, n) を満たす行列です。

ランク落ち行列は、行ベクトルまたは列ベクトルの間に線形従属関係があることを意味します。これは連立方程式の文脈では、方程式の間に冗長性があるか、または矛盾があることを示唆します。

### Q5: ランクと連立方程式の解の関係を簡単に説明してください。

A5: 連立一次方程式 Ax = b において、係数行列Aと拡大係数行列[A|b]のランクの関係から、解の存在と一意性について次のことがわかります：

1. rank(A) = rank([A|b]) < n（nは未知数の数）：
   - 解は無数に存在する（不定）
   - 自由変数の数は n - rank(A)

2. rank(A) = rank([A|b]) = n：
   - 唯一の解が存在する

3. rank(A) < rank([A|b])：
   - 解は存在しない（不能）

この関係は、連立方程式の解の分類において非常に重要です。

### Q6: 主成分分析（PCA）においてランクはどのような役割を果たしますか？

A6: 主成分分析では、データ行列のランクは本質的に必要な主成分の数を示唆します。データ行列Xのランクがrである場合、r個の主成分でデータの全分散を説明できます。

実際のデータ分析では、ノイズや測定誤差によって小さな固有値が生じるため、理論上のランクよりも少ない主成分数を選ぶことが一般的です。しかし、ランクの概念は、データに内在する本質的な次元数を把握するのに役立ちます。