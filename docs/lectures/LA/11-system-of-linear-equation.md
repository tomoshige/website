# 線形代数学 第11回 講義ノート

## 1. 講義情報と予習ガイド

**講義回**: 第11回  
**テーマ**: ガウスの消去法と解の探索  
**関連項目**: 連立一次方程式、行列の基本変形、ガウスの消去法  
**予習すべき内容**: 
- 第10回の内容（連立一次方程式の行列表現）
- 行列の基本操作の概念

## 2. 学習目標

本講義の終了時には以下の能力を身につけることを目標とします:

1. 連立一次方程式の行列表現と拡大係数行列の概念を理解する
2. 行列の基本変形の種類と性質を理解する
3. ガウスの消去法のアルゴリズムを理解し実行できる
4. ガウスの消去法と連立方程式の消去法の関係を理解する
5. Google Colaboratoryを用いて連立一次方程式を解くことができる

## 3. 基本概念

### 3.1 連立一次方程式の行列表現と拡大係数行列（復習）

連立一次方程式は行列と係数ベクトルを用いて簡潔に表すことができます。一般的な形式は以下の通りです:

$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
$$

この連立方程式は行列形式で次のように表現できます:

$$\mathbf{A}\mathbf{x} = \mathbf{b}$$

ここで、

$$\mathbf{A} = 
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}, \quad
\mathbf{x} = 
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{pmatrix}, \quad
\mathbf{b} = 
\begin{pmatrix}
b_1 \\
b_2 \\
\vdots \\
b_m
\end{pmatrix}
$$

連立方程式を解く際には、係数行列と定数項をまとめた**拡大係数行列**（augmented matrix）を考えると便利です:

$$\left(\mathbf{A} | \mathbf{b}\right) = 
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} & | & b_1 \\
a_{21} & a_{22} & \cdots & a_{2n} & | & b_2 \\
\vdots & \vdots & \ddots & \vdots & | & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} & | & b_m
\end{pmatrix}
$$

### 3.2 行列の基本変形

行列の基本変形は、連立方程式の解を変えずに行列の形を変える操作です。主に3種類の基本変形があります:

> **行列の基本変形の定義**:
> 1. **行の交換（Row Exchange）**: 2つの行を入れ替える
> 2. **行のスカラー倍（Row Scaling）**: ある行の全ての要素を0でない定数倍する
> 3. **行の加減（Row Addition）**: ある行の定数倍を別の行に加える

これらの基本変形は連立方程式の解を変えないという重要な性質があります。

#### 行列の基本変形の具体例

1. **行の交換（Row Exchange）**:
   行列の2つの行を入れ替える操作です。例えば、以下の行列の第1行と第2行を交換すると:

   $$
   \begin{pmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
   \end{pmatrix}
   \rightarrow
   \begin{pmatrix}
   4 & 5 & 6 \\
   1 & 2 & 3 \\
   7 & 8 & 9
   \end{pmatrix}
   $$

   このような操作は連立方程式の解には影響しません。なぜなら、方程式の順序を入れ替えただけで、方程式の内容自体は変わっていないからです。

2. **行のスカラー倍（Row Scaling）**:
   行列のある行の全ての要素を0でない定数倍する操作です。例えば、以下の行列の第2行を2倍すると:

   $$
   \begin{pmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
   \end{pmatrix}
   \rightarrow
   \begin{pmatrix}
   1 & 2 & 3 \\
   8 & 10 & 12 \\
   7 & 8 & 9
   \end{pmatrix}
   $$

   これは連立方程式では、2番目の方程式の両辺を2倍することに相当します。例えば:

   $$4x + 5y + 6z = 10 \quad \rightarrow \quad 8x + 10y + 12z = 20$$

   方程式の両辺を同じ定数倍しても、解は変わりません。

3. **行の加減（Row Addition）**:
   ある行の定数倍を別の行に加える操作です。例えば、第1行の(-4)倍を第2行に加えると:

   $$
   \begin{pmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
   \end{pmatrix}
   \rightarrow
   \begin{pmatrix}
   1 & 2 & 3 \\
   0 & -3 & -6 \\
   7 & 8 & 9
   \end{pmatrix}
   $$

   これは連立方程式では、第1方程式の(-4)倍を第2方程式に加えることに相当します:

   $$
   \begin{cases}
   x + 2y + 3z = 10 \\
   4x + 5y + 6z = 20
   \end{cases}
   \quad \rightarrow \quad
   \begin{cases}
   x + 2y + 3z = 10 \\
   -3y - 6z = -20
   \end{cases}
   $$

   この操作により、第2方程式からx変数が消去されました。しかし、連立方程式全体としては同じ解を持ちます。

これらの基本変形を組み合わせることで、行列を望ましい形（上三角形や対角形など）に変形することができます。例えば、以下の拡大係数行列を考えます:

$$
\begin{pmatrix}
1 & 3 & | & 5 \\
2 & 7 & | & 11
\end{pmatrix}
$$

連立方程式の解を変えずに、第1行の(-2)倍を第2行に加えると:

$$
\begin{pmatrix}
1 & 3 & | & 5 \\
0 & 1 & | & 1
\end{pmatrix}
$$

これは元の連立方程式:

$$
\begin{cases}
x + 3y = 5 \\
2x + 7y = 11
\end{cases}
$$

が以下に変形されたことを意味します:

$$
\begin{cases}
x + 3y = 5 \\
y = 1
\end{cases}
$$

この変形により、連立方程式を後退代入で簡単に解くことができます。

### 3.3 ガウスの消去法の概念

ガウスの消去法は、基本変形を用いて拡大係数行列を特定の形（上三角形または階段形）に変形することで連立方程式を解くアルゴリズムです。

> **ガウスの消去法**:
> 拡大係数行列に基本変形を施し、係数行列を上三角行列または階段行列に変形する方法。

ガウスの消去法には主に二つの段階があります:
1. **前進消去（Forward Elimination）**: 係数行列を上三角行列に変形する
2. **後退代入（Back Substitution）**: 上三角行列の形の連立方程式を解く

## 4. 理論と手法

### 4.1 ガウスの消去法の手順

ガウスの消去法のアルゴリズムは以下の通りです:

1. **拡大係数行列の作成**: 連立方程式から拡大係数行列を作成
2. **前進消去**: 
   - 左上から始め、その列の対角成分（ピボット）を基準にする
   - ピボット以下の要素をすべて0にするように基本変形を行う
   - 次の列に移動し同様の操作を繰り返す
3. **後退代入**: 変形された方程式を最後の変数から順に解いていく

### 4.2 ガウスの消去法の詳細アルゴリズム

より数学的に厳密に記述すると、ガウスの消去法の前進消去段階は以下のようになります:

1. $i = 1$ から $n-1$ まで以下の操作を繰り返す:
   - ピボット要素 $a_{ii}$ を選ぶ
   - $j = i + 1$ から $m$ まで以下の操作を行う:
     - 係数 $\mu_{ji} = a_{ji}/a_{ii}$ を計算
     - 行 $j$ から 行 $i$ の $\mu_{ji}$ 倍を引く: $\text{row}_j \leftarrow \text{row}_j - \mu_{ji} \cdot \text{row}_i$

### 4.3 後退代入の手順

上三角行列に変形された連立方程式を解くための後退代入は以下のように行います:

1. 最後の変数 $x_n$ を計算: $x_n = b_n/a_{nn}$
2. $i = n-1$ から $1$ まで逆順に以下を計算:
   $$x_i = \frac{1}{a_{ii}}\left(b_i - \sum_{j=i+1}^{n}a_{ij}x_j\right)$$

### 4.4 計算例: 3変数の連立方程式

以下の連立方程式を考えます:

$$
\begin{cases}
2x + y - z = 8 \\
-3x - y + 2z = -11 \\
-2x + y + 2z = -3
\end{cases}
$$

**ステップ 1**: 拡大係数行列を作成

$$
\begin{pmatrix}
2 & 1 & -1 & | & 8 \\
-3 & -1 & 2 & | & -11 \\
-2 & 1 & 2 & | & -3
\end{pmatrix}
$$

**ステップ 2**: 前進消去を行う

まず、第1列の第1行以下の要素を0にします:

- 第2行に第1行の$\frac{3}{2}$倍を加える:

$$
\begin{pmatrix}
2 & 1 & -1 & | & 8 \\
0 & \frac{1}{2} & \frac{1}{2} & | & 1 \\
-2 & 1 & 2 & | & -3
\end{pmatrix}
$$

- 第3行に第1行の1倍を加える:

$$
\begin{pmatrix}
2 & 1 & -1 & | & 8 \\
0 & \frac{1}{2} & \frac{1}{2} & | & 1 \\
0 & 2 & 1 & | & 5
\end{pmatrix}
$$

次に、第2列の第2行以下の要素を0にします:

- 第3行から第2行の4倍を引く:

$$
\begin{pmatrix}
2 & 1 & -1 & | & 8 \\
0 & \frac{1}{2} & \frac{1}{2} & | & 1 \\
0 & 0 & -1 & | & 1
\end{pmatrix}
$$

**ステップ 3**: 後退代入を行う

変形された連立方程式は:

$$
\begin{cases}
2x + y - z = 8 \\
\frac{1}{2}y + \frac{1}{2}z = 1 \\
-z = 1
\end{cases}
$$

まず $z = -1$ を求めます。
次に $y$ を求めます: $\frac{1}{2}y + \frac{1}{2} \cdot (-1) = 1$ より $y = \frac{1}{2} \cdot 2 + \frac{1}{2} = 1.5$
最後に $x$ を求めます: $2x + 1.5 - (-1) = 8$ より $2x = 8 - 1.5 - 1 = 5.5$ となるので $x = 2.75$

従って、解は $(x, y, z) = (2.75, 1.5, -1)$ です。

### 4.5 部分ピボット選択（Partial Pivoting）

計算の精度を高めるため、実際のアルゴリズムでは「ピボット選択」と呼ばれる技術を使います。最も単純なものは部分ピボット選択で、各ステップで列内の最大絶対値の要素をピボットとして選ぶ方法です:

1. 現在の列における最大絶対値の要素を見つける
2. その要素がある行と現在の行を交換
3. 通常のガウスの消去法を続行

これにより数値計算上の誤差を小さくできます。

## 5. Pythonによる実装と可視化

### 5.1 NumPyを用いたガウスの消去法の実装

NumPyを使って手動でガウスの消去法を実装してみましょう:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaussian_elimination(A, b):
    """
    ガウスの消去法を用いて連立方程式 Ax = b を解く関数
    
    引数:
    A -- 係数行列 (n x n)
    b -- 定数ベクトル (n)
    
    戻り値:
    x -- 解ベクトル (n)
    """
    # 拡大係数行列を作成
    n = len(b)
    augmented = np.column_stack((A, b))
    
    # 前進消去
    for i in range(n):
        # 部分ピボット選択
        max_row = i + np.argmax(abs(augmented[i:, i]))
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
        
        # 現在の行以下の行について消去を行う
        for j in range(i+1, n):
            factor = augmented[j, i] / augmented[i, i]
            augmented[j, i:] -= factor * augmented[i, i:]
    
    # 後退代入
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (augmented[i, -1] - np.dot(augmented[i, i+1:n], x[i+1:])) / augmented[i, i]
    
    return x

# 例題の連立方程式
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

solution = gaussian_elimination(A, b)
print("解: ", solution)

# NumPyの内蔵関数を使った解と比較
numpy_solution = np.linalg.solve(A, b)
print("NumPyの解: ", numpy_solution)
```

### 5.2 連立方程式の解の可視化

3変数の連立一次方程式は3次元空間の平面の交点として解釈できます。それぞれの方程式は3次元空間の平面を表し、その交点が連立方程式の解となります。

```python
def plot_linear_system_3d(A, b, solution):
    """
    3変数連立一次方程式の平面と解を3Dプロットする関数
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # グリッドポイントの生成
    x = np.linspace(-5, 5, 10)
    y = np.linspace(-5, 5, 10)
    X, Y = np.meshgrid(x, y)
    
    # 各平面をプロット
    colors = ['r', 'g', 'b']
    for i in range(len(b)):
        # Z = (b - A[0]*X - A[1]*Y) / A[2] の形で平面の方程式を解く
        Z = (b[i] - A[i, 0] * X - A[i, 1] * Y) / A[i, 2]
        ax.plot_surface(X, Y, Z, alpha=0.3, color=colors[i])
    
    # 解をプロット
    ax.scatter([solution[0]], [solution[1]], [solution[2]], 
               color='black', s=100, label='Solution')
    
    # 軸ラベル
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3変数連立一次方程式の幾何学的表現')
    ax.legend()
    
    plt.show()

# 例題の連立方程式の平面と解を可視化
plot_linear_system_3d(A, b, solution)
```

### 5.3 拡大係数行列の変形過程の可視化

拡大係数行列がガウスの消去法によってどのように変形されていくかを可視化します:

```python
def visualize_gaussian_elimination(A, b):
    """
    ガウスの消去法の各ステップを可視化する関数
    """
    n = len(b)
    augmented = np.column_stack((A, b))
    steps = [augmented.copy()]
    
    # 前進消去
    for i in range(n):
        # 部分ピボット選択
        max_row = i + np.argmax(abs(augmented[i:, i]))
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
            steps.append(augmented.copy())
        
        # 現在の行以下の行について消去を行う
        for j in range(i+1, n):
            factor = augmented[j, i] / augmented[i, i]
            augmented[j, i:] -= factor * augmented[i, i:]
            steps.append(augmented.copy())
    
    # 各ステップをヒートマップで表示
    fig, axes = plt.subplots(1, len(steps), figsize=(4*len(steps), 4))
    if len(steps) == 1:
        axes = [axes]
    
    for i, step in enumerate(steps):
        im = axes[i].imshow(step, cmap='coolwarm')
        axes[i].set_title(f'Step {i}')
        
        # 行と列のラベル
        row_labels = [f'Equation {j+1}' for j in range(n)]
        col_labels = [f'x{j+1}' for j in range(n)] + ['b']
        
        # ラベルを表示
        axes[i].set_xticks(np.arange(len(col_labels)))
        axes[i].set_yticks(np.arange(len(row_labels)))
        axes[i].set_xticklabels(col_labels)
        axes[i].set_yticklabels(row_labels)
        
        # マス目の中に値を表示
        for ii in range(n):
            for jj in range(n+1):
                axes[i].text(jj, ii, f'{step[ii, jj]:.2f}',
                           ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.show()

# 例題の連立方程式のガウスの消去法を可視化
visualize_gaussian_elimination(A, b)
```

### 5.4 NumPyの線形代数ライブラリを使用した解法

実際のデータ分析では、通常NumPyの内蔵関数を使うことが多いです:

```python
def solve_with_numpy(A, b):
    """
    NumPyを使って連立一次方程式を解く関数
    """
    # numpy.linalg.solve を使用
    solution = np.linalg.solve(A, b)
    print("NumPyによる解: ", solution)
    
    # 解の検証: Ax = b を満たすかどうか
    verification = np.allclose(np.dot(A, solution), b)
    print("解の検証: ", "正しい" if verification else "誤差あり")
    
    return solution

# 例題の連立方程式をNumPyで解く
solve_with_numpy(A, b)
```

## 6. 演習問題

### 6.1 基本問題

**問題1**: 以下の連立方程式をガウスの消去法を用いて手計算で解きなさい:
$$
\begin{cases}
3x + 2y - z = 10 \\
-x + 3y + 2z = 5 \\
x - y + z = 0
\end{cases}
$$

**問題2**: 以下の拡大係数行列にガウスの消去法を適用し、上三角行列の形に変形しなさい:
$$
\begin{pmatrix}
1 & 2 & 3 & | & 4 \\
2 & 5 & 3 & | & 7 \\
1 & 0 & 8 & | & 9
\end{pmatrix}
$$

**問題3**: 次の連立方程式をガウスの消去法で解き、解が $(x,y,z) = (1,2,3)$ であることを確認しなさい:
$$
\begin{cases}
2x - y + z = 3 \\
x + y + z = 6 \\
x - y + 2z = 8
\end{cases}
$$

### 6.2 応用問題

**問題4**: 次の連立方程式をガウスの消去法で解いてみなさい:
$$
\begin{cases}
0.003x + 59.14y = 59.17 \\
5.291x - 6.130y = 46.78
\end{cases}
$$

数値的に問題が生じる可能性について考察しなさい。

**問題5**: 健康データ分析の文脈で、患者の体重（kg）、身長（cm）、年齢（歳）から肺活量（L）を予測する線形モデルを考えます。以下のデータからモデルのパラメータを求めなさい。

| 体重(kg) | 身長(cm) | 年齢(歳) | 肺活量(L) |
|---------|---------|---------|----------|
| 70      | 175     | 30      | 4.2      |
| 60      | 165     | 45      | 3.5      |
| 80      | 180     | 35      | 4.7      |

モデルの形式は以下の通りです:
$$\text{肺活量} = \beta_0 + \beta_1 \cdot \text{体重} + \beta_2 \cdot \text{身長} + \beta_3 \cdot \text{年齢}$$

このモデルのパラメータ $\beta_0, \beta_1, \beta_2, \beta_3$ を求めるために連立方程式を立て、ガウスの消去法で解きなさい。

## 7. よくある質問と解答

**Q1: ガウスの消去法と、中学校で習った連立方程式の解法は何が違うのでしょうか？**

A1: 中学校で習った消去法はガウスの消去法の特別な場合と考えることができます。中学校では主に2元または3元の連立方程式を扱い、直感的に変数を消去していきますが、ガウスの消去法はより体系的で、任意の次元の連立方程式に適用できるアルゴリズムです。また、コンピュータでの実装に適した形式になっています。

**Q2: 拡大係数行列はどのようなときに便利ですか？**

A2: 拡大係数行列は行列の基本変形を行う際に、連立方程式全体を一つの行列として扱えるため便利です。特に、ガウスの消去法などのアルゴリズムを適用する際、係数と定数項を一緒に変形できるため、計算ミスを減らし効率的に解を求められます。

**Q3: ピボット要素が0になった場合はどうすればよいですか？**

A3: ピボット要素が0の場合、その列で0でない要素を持つ下の行と交換します（部分ピボット選択）。すべての行の同じ列要素が0の場合、次の列に移動してピボット選択を行います。この場合、解が一意に定まらない可能性があります。

**Q4: ガウスの消去法の計算量はどのくらいですか？**

A4: n次の連立方程式に対してガウスの消去法の計算量は $O(n^3)$ です。これは3重のループ（i,j,k）があるためで、大規模な連立方程式では計算コストが高くなります。しかし、多くの場合、この計算量は避けられず、最適なアルゴリズムとなっています。

**Q5: データサイエンスでガウスの消去法はどのように使われますか？**

A5: データサイエンスでは、線形回帰モデル、最小二乗法による曲線当てはめ、主成分分析など多くの手法の計算基盤としてガウスの消去法が使われます。例えば線形回帰の正規方程式を解く際に、連立一次方程式を解く必要があり、その背後ではガウスの消去法が利用されています。

**Q6: ガウスの消去法と行列式の関係は？**

A6: ガウスの消去法における行の基本変形は行列式の値に影響を与えますが、特定のパターンで影響します。例えば、行の交換は行列式の符号を反転させ、行のスカラー倍はそのスカラー倍だけ行列式を変化させます。ガウスの消去法を用いて上三角行列に変形すれば、行列式は対角成分の積として簡単に計算できます。
