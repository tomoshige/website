# 線形代数学 I / 基礎 / II - 第26回講義ノート

## 1. 講義情報と予習ガイド

**講義回**: 第26回  
**テーマ**: ベクトル空間の基礎  
**関連項目**: ベクトル空間、部分空間、1次独立性、基底、次元  
**予習すべき内容**: ベクトルの1次結合、1次独立・1次従属（第17回講義内容）

## 2. 学習目標

本講義の終了時には、以下のことができるようになります：

1. ベクトル空間の定義を理解し、具体例を挙げることができる
2. 部分ベクトル空間の条件を確認し、与えられた集合が部分空間であるかを判定できる
3. ベクトル空間の基底と次元を理解し、計算することができる
4. 基底の変換を行うことができる
5. ベクトル空間の概念をデータ解析の文脈で理解できる

## 3. 基本概念

### 3.1 ベクトル空間とは

これまでの講義では、主に$\mathbb{R}^n$（n次元実数ベクトル空間）を扱ってきました。今回は、より一般的な「ベクトル空間」の概念について学びます。

> **定義: ベクトル空間**
> 
> 体$F$（例えば実数体$\mathbb{R}$）上のベクトル空間$V$とは、加法演算$+: V \times V \rightarrow V$とスカラー倍演算$\cdot: F \times V \rightarrow V$が定義された集合であり、任意の$\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$と任意のスカラー$a, b \in F$に対して以下の性質を満たすものです：

1. **加法の結合法則**: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
2. **加法の交換法則**: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
3. **加法の単位元**: ゼロベクトル$\mathbf{0} \in V$が存在し、$\mathbf{v} + \mathbf{0} = \mathbf{v}$
4. **加法の逆元**: 各$\mathbf{v} \in V$に対して、$\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$となる$-\mathbf{v} \in V$が存在する
5. **スカラー倍の結合法則**: $a(b\mathbf{v}) = (ab)\mathbf{v}$
6. **スカラー倍の分配法則(1)**: $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$
7. **スカラー倍の分配法則(2)**: $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$
8. **スカラー倍の単位元**: $1\mathbf{v} = \mathbf{v}$

簡単に言えば、ベクトル空間は「ベクトルの足し算」と「スカラー倍」が自然な形で定義できる空間です。

### 3.2 ベクトル空間の例

以下に代表的なベクトル空間の例を示します：

1. **$\mathbb{R}^n$**: $n$次元実数ベクトル空間
   例: $\mathbb{R}^2 = \{(x,y) \mid x,y \in \mathbb{R}\}$

2. **$M_{m,n}(\mathbb{R})$**: $m \times n$実数行列の集合
   例: $2 \times 2$行列の集合 $M_{2,2}(\mathbb{R})$

3. **$P_n(\mathbb{R})$**: 次数が$n$以下の多項式の集合
   例: $P_2(\mathbb{R}) = \{a_0 + a_1x + a_2x^2 \mid a_0, a_1, a_2 \in \mathbb{R}\}$

4. **$C[a,b]$**: 区間$[a,b]$上の連続関数の集合
   例: $C[0,1] = \{f:[0,1] \rightarrow \mathbb{R} \mid f \text{は連続}\}$

5. **$\mathbb{R}^\infty$**: 無限次元実数空間（各成分が実数である無限列の集合）

### 3.3 部分ベクトル空間

> **定義: 部分ベクトル空間**
> 
> ベクトル空間$V$の部分集合$W$が$V$の部分ベクトル空間であるとは、$W$自体がベクトル空間となることです。つまり、以下の条件を満たす必要があります：
> 
> 1. $W$は空集合ではない（少なくともゼロベクトルを含む）
> 2. $\mathbf{u}, \mathbf{v} \in W$ならば$\mathbf{u} + \mathbf{v} \in W$（加法に関して閉じている）
> 3. $a \in F, \mathbf{v} \in W$ならば$a\mathbf{v} \in W$（スカラー倍に関して閉じている）

実はこれらは以下の１条件にまとめることもできます：

> **部分空間の判定条件**
> 
> 空でない部分集合$W$が部分ベクトル空間であるための必要十分条件は：
> 
> 任意の$\mathbf{u}, \mathbf{v} \in W$と任意のスカラー$a, b \in F$に対して、
> $a\mathbf{u} + b\mathbf{v} \in W$が成り立つこと（線形結合に関して閉じている）

#### 部分空間の例

1. $\mathbb{R}^3$の部分空間の例：
   - 原点を通る平面: $\{(x,y,z) \in \mathbb{R}^3 \mid ax + by + cz = 0\}$（$a,b,c$は定数）
   - 原点を通る直線: $\{t(a,b,c) \mid t \in \mathbb{R}\}$（$a,b,c$は定数）
   - $\mathbb{R}^3$自体と$\{\mathbf{0}\}$（零ベクトルのみの集合）

2. $P_n(\mathbb{R})$の部分空間の例：
   - 偶関数の多項式: $\{a_0 + a_2x^2 + a_4x^4 + \cdots \mid a_i \in \mathbb{R}\}$
   - 奇関数の多項式: $\{a_1x + a_3x^3 + a_5x^5 + \cdots \mid a_i \in \mathbb{R}\}$

### 3.4 部分空間であるかの判定方法

集合$W$が部分空間かどうかを判定するには以下の手順を踏みます：

1. $\mathbf{0} \in W$かどうかを確認（ゼロベクトルを含まない場合は部分空間ではない）
2. 加法に関して閉じているかを確認（$\mathbf{u}, \mathbf{v} \in W \Rightarrow \mathbf{u} + \mathbf{v} \in W$）
3. スカラー倍に関して閉じているかを確認（$a \in F, \mathbf{v} \in W \Rightarrow a\mathbf{v} \in W$）

**例**: $W = \{(x,y) \in \mathbb{R}^2 \mid x + y = 1\}$は$\mathbb{R}^2$の部分空間か？

**解答**: $W$はゼロベクトル$(0,0)$を含まない（$(0,0)$は$x + y = 1$を満たさない）ので、$W$は部分空間ではない。

**例**: $W = \{(x,y,z) \in \mathbb{R}^3 \mid 2x - 3y + z = 0\}$は$\mathbb{R}^3$の部分空間か？

**解答**: 
1. $\mathbf{0} = (0,0,0) \in W$（$2 \cdot 0 - 3 \cdot 0 + 0 = 0$）
2. $\mathbf{u} = (u_1,u_2,u_3), \mathbf{v} = (v_1,v_2,v_3) \in W$とすると、
   $2u_1 - 3u_2 + u_3 = 0$および$2v_1 - 3v_2 + v_3 = 0$
   よって$\mathbf{u} + \mathbf{v} = (u_1+v_1, u_2+v_2, u_3+v_3)$に対して、
   $2(u_1+v_1) - 3(u_2+v_2) + (u_3+v_3)$
   $= (2u_1 - 3u_2 + u_3) + (2v_1 - 3v_2 + v_3) = 0 + 0 = 0$
   したがって$\mathbf{u} + \mathbf{v} \in W$
3. $a \in \mathbb{R}, \mathbf{u} = (u_1,u_2,u_3) \in W$とすると、
   $2u_1 - 3u_2 + u_3 = 0$
   よって$a\mathbf{u} = (au_1, au_2, au_3)$に対して、
   $2(au_1) - 3(au_2) + (au_3) = a(2u_1 - 3u_2 + u_3) = a \cdot 0 = 0$
   したがって$a\mathbf{u} \in W$

以上より、$W$は部分空間である。

## 4. 理論と手法

### 4.1 ベクトルの線形結合と線形包

> **定義: 線形結合**
> 
> ベクトル$\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k \in V$の線形結合とは、
> $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$
> という形の式であり、ここで$c_1, c_2, \ldots, c_k$はスカラー（実数）です。

> **定義: 線形包（スパン）**
> 
> ベクトル$\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k \in V$の線形包（スパン）とは、
> これらのベクトルのすべての線形結合の集合であり、次のように表記します：
> 
> $\text{span}\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\} = \{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k \mid c_1, c_2, \ldots, c_k \in \mathbb{R}\}$

**重要な性質**: $\text{span}\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$は常に$V$の部分空間となります。

**例**: $\mathbf{v}_1 = (1,0,0), \mathbf{v}_2 = (0,1,0) \in \mathbb{R}^3$のスパンを考える。

$\text{span}\{\mathbf{v}_1, \mathbf{v}_2\} = \{c_1(1,0,0) + c_2(0,1,0) \mid c_1, c_2 \in \mathbb{R}\} = \{(c_1, c_2, 0) \mid c_1, c_2 \in \mathbb{R}\}$

これは$xy$平面を表しており、$\mathbb{R}^3$の部分空間です。

### 4.2 線形独立性と線形従属性

> **定義: 線形独立と線形従属**
> 
> ベクトル$\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k \in V$が**線形独立**であるとは、
> $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$
> という式が$c_1 = c_2 = \cdots = c_k = 0$のときのみ成り立つことです。
> 
> そうでない場合、すなわち少なくとも一つの$c_i \neq 0$が存在して上の式が成り立つ場合、これらのベクトルは**線形従属**であるといいます。

**線形従属の別の見方**: ベクトル群が線形従属であるということは、そのうちの少なくとも1つのベクトルが他のベクトルの線形結合として表せることを意味します。

#### 線形独立性の判定方法

ベクトル群が線形独立かどうかを判定するには、以下の方法が使えます：

1. **定義に基づく方法**: $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$という式を解き、唯一の解が$c_1 = c_2 = \cdots = c_k = 0$であれば線形独立。

2. **行列のランクを用いる方法**: ベクトル$\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$を並べた行列$A = [\mathbf{v}_1 \; \mathbf{v}_2 \; \cdots \; \mathbf{v}_k]$を考え、$\text{rank}(A) = k$であれば線形独立。

**例**: ベクトル$\mathbf{v}_1 = (1,2,1), \mathbf{v}_2 = (2,3,1), \mathbf{v}_3 = (1,-1,-1) \in \mathbb{R}^3$が線形独立かどうかを調べよ。

**解答**: $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + c_3\mathbf{v}_3 = \mathbf{0}$という式を考える。

$c_1(1,2,1) + c_2(2,3,1) + c_3(1,-1,-1) = (0,0,0)$

これは連立方程式
$c_1 + 2c_2 + c_3 = 0$
$2c_1 + 3c_2 - c_3 = 0$
$c_1 + c_2 - c_3 = 0$
を解くことに等しい。

行列の形で表すと：
$\begin{bmatrix} 1 & 2 & 1 \\ 2 & 3 & -1 \\ 1 & 1 & -1 \end{bmatrix} \begin{bmatrix} c_1 \\ c_2 \\ c_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$

この行列をガウスの消去法で簡約すると：
$\begin{bmatrix} 1 & 2 & 1 \\ 0 & -1 & -3 \\ 0 & -1 & -2 \end{bmatrix} \rightarrow \begin{bmatrix} 1 & 2 & 1 \\ 0 & -1 & -3 \\ 0 & 0 & 1 \end{bmatrix}$

これより、$c_3 = 0, c_2 = 0, c_1 = 0$となるので、与えられたベクトル群は線形独立である。

### 4.3 基底と次元

> **定義: 基底**
> 
> ベクトル空間$V$の**基底**とは、以下の2つの条件を満たすベクトルの集合$\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\}$のことです：
> 
> 1. $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$は線形独立である
> 2. $\text{span}\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\} = V$（つまり、これらのベクトルはVを生成する）

基底の重要な性質は、ベクトル空間の任意のベクトルが**唯一**の方法で基底ベクトルの線形結合として表せることです。

> **定義: 次元**
> 
> ベクトル空間$V$の**次元**（記号: $\dim V$）とは、$V$の基底に含まれるベクトルの個数です。

**重要な定理**:
1. ベクトル空間の任意の基底はすべて同じ個数のベクトルを持つ
2. 有限次元ベクトル空間$V$の任意の線形独立な集合は、$V$の基底に拡張できる
3. 有限次元ベクトル空間$V$の任意の生成系は、$V$の基底を部分集合として含む

#### 標準基底

多くの一般的なベクトル空間には、標準（正準）基底と呼ばれる自然な基底があります：

1. $\mathbb{R}^n$の標準基底: $\{\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n\}$
   ここで$\mathbf{e}_i$は$i$番目の成分だけが1で他はすべて0のベクトル

2. $P_n(\mathbb{R})$の標準基底: $\{1, x, x^2, \ldots, x^n\}$

3. $M_{m,n}(\mathbb{R})$の標準基底: $\{E_{ij} \mid 1 \leq i \leq m, 1 \leq j \leq n\}$
   ここで$E_{ij}$は$(i,j)$成分が1で他はすべて0の行列

**例**: $\mathbb{R}^2$の標準基底は$\{\mathbf{e}_1, \mathbf{e}_2\} = \{(1,0), (0,1)\}$です。$\mathbb{R}^2$内の任意のベクトル$(a,b)$は、
$(a,b) = a(1,0) + b(0,1) = a\mathbf{e}_1 + b\mathbf{e}_2$
と表すことができます。

### 4.4 座標表現

> **定義: 座標表現**
> 
> ベクトル空間$V$の基底$B = \{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\}$に関する
> ベクトル$\mathbf{v} \in V$の**座標表現**とは、
> $\mathbf{v} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n$
> を満たすスカラー$c_1, c_2, \ldots, c_n$を用いて、
> $[\mathbf{v}]_B = \begin{bmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{bmatrix}$
> と表記することです。

異なる基底を選ぶと、同じベクトルでも座標表現が変わります。これが「基底の変換」の考え方につながります。

#### 基底の変換

$V$の2つの基底$B = \{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\}$と$B' = \{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_n\}$があるとき、
基底$B$に関するベクトル$\mathbf{v}$の座標から、基底$B'$に関する座標への変換は線形変換として表されます。

この変換は「基底変換行列」$P_{B' \leftarrow B}$によって行われ、
$[\mathbf{v}]_{B'} = P_{B' \leftarrow B} [\mathbf{v}]_B$
となります。

基底変換行列$P_{B' \leftarrow B}$は、基底$B$のベクトルを基底$B'$で表したときの座標を列ベクトルとして並べた行列です：
$P_{B' \leftarrow B} = \begin{bmatrix} [\mathbf{v}_1]_{B'} & [\mathbf{v}_2]_{B'} & \cdots & [\mathbf{v}_n]_{B'} \end{bmatrix}$

**例**: $\mathbb{R}^2$で、標準基底$B = \{(1,0), (0,1)\}$と別の基底$B' = \{(1,1), (1,-1)\}$を考える。
ベクトル$\mathbf{v} = (3,1)$の基底$B'$に関する座標を求める。

まず、基底変換行列$P_{B' \leftarrow B}$を求める。
$(1,0) = \frac{1}{2}(1,1) + \frac{1}{2}(1,-1)$なので$[(1,0)]_{B'} = \begin{bmatrix} \frac{1}{2} \\ \frac{1}{2} \end{bmatrix}$
$(0,1) = \frac{1}{2}(1,1) - \frac{1}{2}(1,-1)$なので$[(0,1)]_{B'} = \begin{bmatrix} \frac{1}{2} \\ -\frac{1}{2} \end{bmatrix}$

よって、
$P_{B' \leftarrow B} = \begin{bmatrix} \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2} & -\frac{1}{2} \end{bmatrix}$

ベクトル$\mathbf{v} = (3,1)$の基底$B$に関する座標は$[\mathbf{v}]_B = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$
したがって、基底$B'$に関する座標は
$[\mathbf{v}]_{B'} = P_{B' \leftarrow B} [\mathbf{v}]_B = \begin{bmatrix} \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2} & -\frac{1}{2} \end{bmatrix} \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

これは$(3,1) = 2(1,1) + 1(1,-1)$を意味します。

## 5. Pythonによる実装と可視化

### 5.1 NumPyを用いたベクトル空間の基本操作

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ベクトルの定義
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

# 線形結合
a, b, c = 2, 3, -1
linear_combination = a * v1 + b * v2 + c * v3
print("線形結合:", linear_combination)

# 線形独立性の判定
vectors = np.array([v1, v2, v3])  # 各行がベクトル
rank = np.linalg.matrix_rank(vectors)
print("ランク:", rank)
print("線形独立" if rank == len(vectors) else "線形従属")

# 基底の変換
# 標準基底
standard_basis = np.eye(3)  # 3x3の単位行列
print("標準基底:\n", standard_basis)

# 新しい基底
new_basis = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
])

# ベクトルv = (2, 3, 4)を新しい基底で表現する
v = np.array([2, 3, 4])

# 基底変換行列（新しい基底を標準基底で表したもの）の逆行列
transformation_matrix = np.linalg.inv(new_basis)

# 変換実行
v_new_coords = transformation_matrix @ v
print("新しい基底での座標:", v_new_coords)

# 逆変換で確認
v_reconstructed = new_basis @ v_new_coords
print("元の座標に戻した結果:", v_reconstructed)
```

### 5.2 ベクトル空間の視覚化

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3Dプロットの設定
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 原点
origin = np.zeros(3)

# 標準基底ベクトル
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# 基底ベクトルを矢印で表示（標準基底）
ax.quiver(*origin, *e1, color='r', label='e1')
ax.quiver(*origin, *e2, color='g', label='e2')
ax.quiver(*origin, *e3, color='b', label='e3')

# 新しい基底ベクトル
v1 = np.array([1, 1, 0])
v2 = np.array([1, 0, 1])
v3 = np.array([0, 1, 1])

# 新しい基底ベクトルを矢印で表示
ax.quiver(*origin, *v1, color='r', linestyle='dashed', label='v1')
ax.quiver(*origin, *v2, color='g', linestyle='dashed', label='v2')
ax.quiver(*origin, *v3, color='b', linestyle='dashed', label='v3')

# テスト用ベクトル
test_vector = np.array([2, 3, 4])
ax.quiver(*origin, *test_vector, color='k', label='test vector')

# プロットの設定
ax.set_xlim([-1, 3])
ax.set_ylim([-1, 3])
ax.set_zlim([-1, 3])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('ベクトル空間と基底')
ax.legend()

plt.tight_layout()
plt.show()
```

### 5.3 部分空間の視覚化

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3Dプロットの設定
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 原点
origin = np.zeros(3)

# グリッドポイントの生成
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x, y)

# 平面 ax + by + cz = 0 のデータ点
a, b, c = 1, 1, 1  # 例: x + y + z = 0
Z = -(a * X + b * Y) / c

# 平面のプロット
surf = ax.plot_surface(X, Y, Z, alpha=0.5, color='cyan')

# 標準基底ベクトル
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

ax.quiver(*origin, *e1, color='r', label='e1')
ax.quiver(*origin, *e2, color='g', label='e2')
ax.quiver(*origin, *e3, color='b', label='e3')

# 平面上のベクトル（基底）を表示
v1 = np.array([1, 0, -1])  # x + z = 0 を満たす
v2 = np.array([0, 1, -1])  # y + z = 0 を満たす

ax.quiver(*origin, *v1, color='m', linestyle='dashed', label='v1')
ax.quiver(*origin, *v2, color='y', linestyle='dashed', label='v2')

# プロットの設定
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('部分空間（平面 x + y + z = 0）とその基底')
ax.legend()

plt.tight_layout()
plt.show()
```

### 5.4 基底変換の視覚化

```python
import numpy as np
import matplotlib.pyplot as plt

# 2次元空間での基底変換を視覚化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 原点
origin = np.zeros(2)

# 標準基底
e1 = np.array([1, 0])
e2 = np.array([0, 1])

# 新しい基底
v1 = np.array([1, 1])
v2 = np.array([1, -1])

# テストベクトル
test_vector = np.array([3, 1])

# 標準基底での表示（左側）
ax1.quiver(*origin, *e1, color='r', angles='xy', scale_units='xy', scale=1, label='e1')
ax1.quiver(*origin, *e2, color='g', angles='xy', scale_units='xy', scale=1, label='e2')
ax1.quiver(*origin, *test_vector, color='k', angles='xy', scale_units='xy', scale=1, label='v=(3,1)')

# 標準基底でのテストベクトルの成分
ax1.quiver(*origin, test_vector[0], 0, color='r', alpha=0.3, angles='xy', scale_units='xy', scale=1)
ax1.quiver(*np.array([test_vector[0], 0]), 0, test_vector[1], color='g', alpha=0.3, angles='xy', scale_units='xy', scale=1)

# 新しい基底での座標を計算
P = np.column_stack([v1, v2])  # 基底変換行列
P_inv = np.linalg.inv(P)
new_coords = P_inv @ test_vector
print(f"新しい基底での座標: {new_coords}")

# 新しい基底での表示（右側）
ax2.quiver(*origin, *v1, color='r', angles='xy', scale_units='xy', scale=1, label='v1')
ax2.quiver(*origin, *v2, color='g', angles='xy', scale_units='xy', scale=1, label='v2')
ax2.quiver(*origin, *test_vector, color='k', angles='xy', scale_units='xy', scale=1, label='v=(3,1)')

# 新しい基底でのテストベクトルの成分
ax2.quiver(*origin, *new_coords[0] * v1, color='r', alpha=0.3, angles='xy', scale_units='xy', scale=1)
ax2.quiver(*new_coords[0] * v1, *new_coords[1] * v2, color='g', alpha=0.3, angles='xy', scale_units='xy', scale=1)

# プロットの設定
for ax in [ax1, ax2]:
    ax.set_xlim([-2, 4])
    ax.set_ylim([-2, 4])
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()

ax1.set_title('標準基底での表現')
ax2.set_title('新しい基底での表現')

plt.tight_layout()
plt.show()
```

### 5.5 データサイエンスにおけるベクトル空間の例

以下の例では、データを行列として表現し、その列空間や行空間を考えることで、データの持つ構造を理解する方法を示します。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# ボストン住宅価格データセットを読み込む
boston = load_boston()
X = boston.data
feature_names = boston.feature_names

# 最初の5つの特徴量だけを使用
X_subset = X[:, :5]

# 行列のランクを計算
rank = np.linalg.matrix_rank(X_subset)
print(f"行列のランク: {rank}")

# 特異値分解（SVD）
U, S, Vt = np.linalg.svd(X_subset, full_matrices=False)

# 特異値をプロット
plt.figure(figsize=(10, 6))
plt.bar(range(len(S)), S)
plt.title('データ行列の特異値')
plt.xlabel('成分')
plt.ylabel('特異値')
plt.grid(True)
plt.show()

# 特異値から寄与率を計算
explained_variance_ratio = S**2 / np.sum(S**2)
print(f"各成分の寄与率: {explained_variance_ratio}")
print(f"累積寄与率: {np.cumsum(explained_variance_ratio)}")

# 右特異ベクトル（V）を基底とみなして特徴量の関係を可視化
plt.figure(figsize=(10, 6))
for i, feature in enumerate(feature_names[:5]):
    plt.arrow(0, 0, Vt[0, i], Vt[1, i], head_width=0.05, head_length=0.05, fc='k', ec='k')
    plt.text(Vt[0, i]*1.1, Vt[1, i]*1.1, feature)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.gca().set_aspect('equal')
plt.title('特徴量の第1・第2主成分空間での表現')
plt.show()
```

## 6. 演習問題

### 6.1 基本問題（概念理解の確認）

1. 以下の集合が$\mathbb{R}^3$の部分空間であるかどうかを判定せよ。
   (a) $W_1 = \{(x, y, z) \in \mathbb{R}^3 \mid x - 2y + 3z = 0\}$
   (b) $W_2 = \{(x, y, z) \in \mathbb{R}^3 \mid x - 2y + 3z = 1\}$
   (c) $W_3 = \{(x, y, z) \in \mathbb{R}^3 \mid xy = 0\}$

2. $\mathbb{R}^3$の以下のベクトル集合が線形独立かどうかを判定せよ。
   (a) $\{(1,1,1), (1,2,3), (2,3,4)\}$
   (b) $\{(1,0,1), (0,1,1), (1,1,2)\}$

3. $\mathbb{R}^3$において、平面$x + 2y - z = 0$の基底を求めよ。

4. $\mathbb{R}^2$における基底$B = \{(1,1), (1,-1)\}$について、ベクトル$v = (3,2)$の$B$に関する座標を求めよ。

5. 多項式空間$P_2(\mathbb{R})$の基底$\{1, 1+x, 1+x+x^2\}$について、多項式$p(x) = 2 + 3x + x^2$の座標表示を求めよ。

### 6.2 応用問題（応用能力の確認）

1. $\mathbb{R}^4$の部分空間$W = \text{span}\{(1,2,0,1), (0,1,1,1), (2,5,1,3)\}$の次元と基底を求めよ。

2. $\mathbb{R}^3$における2つの直線
   $L_1 = \{t(1,2,3) \mid t \in \mathbb{R}\}$と
   $L_2 = \{s(2,1,0) + (1,1,1) \mid s \in \mathbb{R}\}$について、
   (a) $L_1$と$L_2$が交わるかどうかを判定せよ。
   (b) 交わる場合はその交点を、交わらない場合は$L_1$と$L_2$の最短距離を求めよ。

3. $\mathbb{R}^3$における3つのベクトル$v_1 = (1,2,3)$, $v_2 = (2,3,4)$, $v_3 = (3,5,7)$について、
   (a) $\text{span}\{v_1, v_2, v_3\}$の次元を求めよ。
   (b) $v_3$を$v_1$と$v_2$の線形結合で表せ。

4. $\mathbb{R}^3$における標準基底$E = \{e_1, e_2, e_3\}$から別の基底$B = \{(1,1,1), (1,1,0), (1,0,0)\}$への基底変換行列を求め、ベクトル$v = (2,3,4)$を基底$B$に関する座標で表せ。

5. 健康データ分析の応用問題：
   あるフィットネストラッカーから収集された3つの変数（歩数、心拍数、睡眠時間）のデータがあります。日々の測定値は3次元ベクトル空間の点として表すことができます。
   
   以下のデータ行列$X$の各行は1日分のデータを表しています（標準化済み）：
   ```
   X = [
       [ 1.2,  0.8, -0.5],
       [ 0.9,  0.6, -0.3],
       [ 0.6,  0.4, -0.2],
       [-1.5, -1.0,  0.6],
       [-1.2, -0.8,  0.4]
   ]
   ```
   
   (a) この行列の列ベクトルが線形独立かどうかを判定せよ。
   (b) データ点が主に存在する部分空間の次元と基底を求めよ。
   (c) このデータから健康状態に関してどのような洞察が得られるか考察せよ。

## 7. よくある質問と解答

### Q1: ベクトル空間と部分空間の違いは何ですか？
**A**: ベクトル空間は、ベクトルの加法とスカラー倍が定義された集合であり、8つの公理を満たすものです。部分空間は、元のベクトル空間に含まれる集合であり、それ自体もベクトル空間になっているものを指します。部分空間は必ず原点（零ベクトル）を含み、ベクトルの加法とスカラー倍に関して閉じている必要があります。

### Q2: 線形独立と基底の関係を教えてください。
**A**: 基底は線形独立なベクトルの集合であり、かつそれらのベクトルが空間全体を生成（スパン）するものです。つまり、基底は「必要最小限」のベクトル集合であり、1つも無駄なく（線形独立）、かつ十分にベクトル空間をカバー（生成）するという特徴があります。

### Q3: ベクトル空間の次元とは何ですか？
**A**: ベクトル空間の次元とは、その空間の基底に含まれるベクトルの数です。例えば、$\mathbb{R}^3$の次元は3、$P_2(\mathbb{R})$（2次以下の多項式空間）の次元は3です。次元は、そのベクトル空間を表現するのに最低限必要なパラメータの数と考えることもできます。

### Q4: 線形従属なベクトル集合から基底を取り出すにはどうすればいいですか？
**A**: 線形従属なベクトル集合から基底を取り出すには、ガウス・ジョルダン消去法を用いて行簡約形（または列簡約形）に変換し、ピボット列（または行）に対応するベクトルを選べばよいです。あるいは、ベクトルを1つずつ考慮していき、既に選んだベクトルと線形独立なものだけを追加していく方法もあります。

### Q5: データサイエンスにおけるベクトル空間の応用例を教えてください。
**A**: データサイエンスでのベクトル空間の応用は多岐にわたります：
- **次元削減**: 主成分分析（PCA）は高次元データを低次元の部分空間に射影します
- **特徴抽出**: データの特徴を表現する新しい基底（主成分など）を見つけます
- **文書ベクトル化**: 文書をベクトル空間モデル（VSM）で表現し、類似度を計算します
- **画像処理**: 画像をベクトルとして扱い、特徴空間で分析します
- **推薦システム**: ユーザーの好みを表すベクトル空間を構築し、類似性を測定します

これらの応用では、データを適切なベクトル空間で表現することで、パターンの発見や予測モデルの構築が可能になります。