# 線形代数学 I / 基礎 / II
## 第40回講義：特異値分解の応用

### 1. 講義情報と予習ガイド

**講義回**: 第40回  
**関連項目**: 特異値分解、ムーアベンローズの擬似逆行列、低ランク近似、データ圧縮  
**予習すべき内容**: 特異値分解(SVD)の基礎概念、行列の階数、固有値と固有ベクトル

### 2. 学習目標

1. ムーアベンローズの擬似逆行列の概念を理解し、SVDを用いた計算方法を習得する
2. 行列の低ランク近似とエッカート・ヤングの定理の意義を理解する
3. SVDを用いたデータ圧縮とノイズ除去の原理を理解する
4. 実際のデータに対するSVDの応用例を通じて、その実践的価値を認識する

### 3. 基本概念

#### 3.1. 特異値分解の復習

特異値分解(SVD)は任意の行列を3つの特別な行列の積に分解する手法です。$m \times n$の行列$A$に対して：

> **定義: 特異値分解**  
> 任意の$m \times n$行列$A$は以下のように分解できます：
> $A = U\Sigma V^T$
> 
> ここで：
> - $U$は$m \times m$の直交行列（左特異ベクトル）
> - $\Sigma$は$m \times n$の対角行列（特異値を対角に配置）
> - $V$は$n \times n$の直交行列（右特異ベクトル）

特に、$\Sigma$の対角成分$\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_r > 0$（$r$は$A$の階数）は$A$の特異値と呼ばれます。

#### 3.2. ムーアベンローズの擬似逆行列

ムーアベンローズの擬似逆行列（一般逆行列）は、通常の逆行列が存在しない場合でも定義される行列です。

> **定義: ムーアベンローズの擬似逆行列**  
> 行列$A$の擬似逆行列$A^+$は、以下の4つの条件を満たす唯一の行列です：
> 
> 1. $AA^+A = A$
> 2. $A^+AA^+ = A^+$
> 3. $(AA^+)^T = AA^+$
> 4. $(A^+A)^T = A^+A$

SVDを用いたムーアベンローズの擬似逆行列の計算方法は非常に美しく実用的です。

> **定理: SVDによる擬似逆行列の計算**  
> 行列$A = U\Sigma V^T$のムーアベンローズの擬似逆行列は：
> 
> $A^+ = V\Sigma^+ U^T$
> 
> ここで$\Sigma^+$は$\Sigma$の非ゼロ特異値の逆数を対応する位置に配置し、他の成分はゼロのままとした行列です。

**例題1**: 次の行列$A$のムーアベンローズの擬似逆行列を求めてみましょう。

$$A = \begin{pmatrix} 4 & 0 \\ 0 & 3 \\ 0 & 0 \end{pmatrix}$$

**解答**:
まず$A$のSVDを求めます。この場合、特異値は$\sigma_1 = 4$と$\sigma_2 = 3$です。

$$U = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}, \Sigma = \begin{pmatrix} 4 & 0 \\ 0 & 3 \\ 0 & 0 \end{pmatrix}, V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

$\Sigma^+$を求めるために、非ゼロ特異値の逆数を対応する位置に配置します：

$$\Sigma^+ = \begin{pmatrix} 1/4 & 0 \\ 0 & 1/3 \\ 0 & 0 \end{pmatrix}^T = \begin{pmatrix} 1/4 & 0 & 0 \\ 0 & 1/3 & 0 \end{pmatrix}$$

したがって、擬似逆行列は：

$$A^+ = V\Sigma^+ U^T = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 1/4 & 0 & 0 \\ 0 & 1/3 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}^T = \begin{pmatrix} 1/4 & 0 & 0 \\ 0 & 1/3 & 0 \end{pmatrix}$$

#### 3.3. 擬似逆行列の応用

擬似逆行列は以下のような場面で特に有用です：

1. **最小二乗法による線形回帰**: 過剰決定系（方程式の数が未知数より多い場合）で最小二乗解を求める際に使用。
   $\hat{x} = A^+ b$は$\|Ax - b\|_2$を最小化する解です。

2. **劣決定系の最小ノルム解**: 未知数が方程式より多い場合、$A^+ b$は$Ax = b$を満たす解の中で最小ノルム$\|x\|_2$を持つものです。

3. **行列方程式の解**: $AXB = C$の形の行列方程式を解く際に利用できます。

4. **ランク欠損データの処理**: 多変量解析で不完全なデータを扱う場合に有用です。

#### 3.3.1. 連立方程式の解法としての擬似逆行列

連立方程式$Ax = b$について考えましょう。これには以下の3つの場合があります：

> **i) 唯一解が存在する場合**: $A$が正方かつフルランクならば、$x = A^{-1}b = A^+b$
> 
> **ii) 過剰決定系（方程式が多すぎる場合）**: 厳密な解は一般に存在せず、最小二乗解$x = A^+b$が$\|Ax - b\|_2$を最小化
> 
> **iii) 劣決定系（未知数が多すぎる場合）**: 無数の解が存在し、$x = A^+b$はその中で最小ノルム（$\|x\|_2$が最小）を持つ解

**例題4**: 以下の過剰決定系（方程式の数が未知数より多い）の連立方程式を擬似逆行列を用いて解きましょう。

$$\begin{align}
x + y &= 3 \\
2x + y &= 5 \\
x + 2y &= 4
\end{align}$$

**解答**:
まず、この方程式を行列形式で表します。

$$A = \begin{pmatrix} 1 & 1 \\ 2 & 1 \\ 1 & 2 \end{pmatrix}, \quad b = \begin{pmatrix} 3 \\ 5 \\ 4 \end{pmatrix}$$

方程式の数（3）が未知数（2）より多いため、一般には厳密な解は存在しません。擬似逆行列$A^+$を用いて最小二乗解を求めます。

$A$のSVDを計算します。（計算過程は省略し、結果のみ示します）

$$U = \begin{pmatrix} 0.47 & -0.34 & 0.81 \\ 0.74 & 0.64 & -0.21 \\ 0.47 & -0.69 & -0.55 \end{pmatrix}, \quad
\Sigma = \begin{pmatrix} 3.74 & 0 \\ 0 & 1.00 \\ 0 & 0 \end{pmatrix}, \quad
V = \begin{pmatrix} 0.83 & -0.55 \\ 0.55 & 0.83 \end{pmatrix}$$

$\Sigma^+$を計算します:

$$\Sigma^+ = \begin{pmatrix} 1/3.74 & 0 & 0 \\ 0 & 1/1.00 & 0 \end{pmatrix} = \begin{pmatrix} 0.27 & 0 & 0 \\ 0 & 1.00 & 0 \end{pmatrix}$$

擬似逆行列$A^+$は:

$$A^+ = V\Sigma^+ U^T = \begin{pmatrix} 0.83 & -0.55 \\ 0.55 & 0.83 \end{pmatrix} \begin{pmatrix} 0.27 & 0 & 0 \\ 0 & 1.00 & 0 \end{pmatrix} \begin{pmatrix} 0.47 & 0.74 & 0.47 \\ -0.34 & 0.64 & -0.69 \\ 0.81 & -0.21 & -0.55 \end{pmatrix}$$

計算すると:

$$A^+ \approx \begin{pmatrix} 0.37 & 0.59 & 0.04 \\ 0.07 & 0.15 & 0.59 \end{pmatrix}$$

最小二乗解は:

$$x = A^+b \approx \begin{pmatrix} 0.37 & 0.59 & 0.04 \\ 0.07 & 0.15 & 0.59 \end{pmatrix} \begin{pmatrix} 3 \\ 5 \\ 4 \end{pmatrix} \approx \begin{pmatrix} 2.0 \\ 1.1 \end{pmatrix}$$

この解が実際に最小二乗誤差を持つことを確認します:

$$\|Ax - b\|_2^2 = \left\| \begin{pmatrix} 1 & 1 \\ 2 & 1 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} 2.0 \\ 1.1 \end{pmatrix} - \begin{pmatrix} 3 \\ 5 \\ 4 \end{pmatrix} \right\|_2^2 = \left\| \begin{pmatrix} 3.1 \\ 5.1 \\ 4.2 \end{pmatrix} - \begin{pmatrix} 3 \\ 5 \\ 4 \end{pmatrix} \right\|_2^2 = 0.06$$

したがって、$x \approx 2.0, y \approx 1.1$が最小二乗解となります。この解は元の連立方程式を厳密に満たすわけではありませんが、二乗誤差の総和を最小にする解です。

**例題5**: 以下の劣決定系（未知数が方程式より多い）の連立方程式を擬似逆行列を用いて解きましょう。

$$\begin{align}
x + y + z &= 6 \\
2x - y + z &= 3
\end{align}$$

**解答**:
行列形式では:

$$A = \begin{pmatrix} 1 & 1 & 1 \\ 2 & -1 & 1 \end{pmatrix}, \quad b = \begin{pmatrix} 6 \\ 3 \end{pmatrix}$$

方程式の数（2）が未知数（3）より少ないため、無数の解が存在します。擬似逆行列$A^+$を用いて最小ノルム解を求めます。

$A$のSVDを計算します:

$$U \approx \begin{pmatrix} 0.41 & 0.91 \\ 0.91 & -0.41 \end{pmatrix}, \quad
\Sigma \approx \begin{pmatrix} 2.65 & 0 & 0 \\ 0 & 1.50 & 0 \end{pmatrix}, \quad
V \approx \begin{pmatrix} 0.74 & -0.04 & 0.67 \\ -0.11 & 0.76 & 0.64 \\ 0.66 & 0.65 & -0.38 \end{pmatrix}$$

$\Sigma^+$を計算します:

$$\Sigma^+ = \begin{pmatrix} 1/2.65 & 0 \\ 0 & 1/1.50 \\ 0 & 0 \end{pmatrix} \approx \begin{pmatrix} 0.38 & 0 \\ 0 & 0.67 \\ 0 & 0 \end{pmatrix}$$

擬似逆行列$A^+$は:

$$A^+ = V\Sigma^+ U^T \approx \begin{pmatrix} 0.74 & -0.04 & 0.67 \\ -0.11 & 0.76 & 0.64 \\ 0.66 & 0.65 & -0.38 \end{pmatrix} \begin{pmatrix} 0.38 & 0 \\ 0 & 0.67 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 0.41 & 0.91 \\ 0.91 & -0.41 \end{pmatrix}$$

計算すると:

$$A^+ \approx \begin{pmatrix} 0.31 & 0.35 \\ 0.61 & -0.31 \\ 0.25 & 0.31 \end{pmatrix}$$

最小ノルム解は:

$$x = A^+b \approx \begin{pmatrix} 0.31 & 0.35 \\ 0.61 & -0.31 \\ 0.25 & 0.31 \end{pmatrix} \begin{pmatrix} 6 \\ 3 \end{pmatrix} \approx \begin{pmatrix} 2.9 \\ 2.7 \\ 2.4 \end{pmatrix}$$

この解が元の連立方程式を満たすことを確認します:

$$Ax = \begin{pmatrix} 1 & 1 & 1 \\ 2 & -1 & 1 \end{pmatrix} \begin{pmatrix} 2.9 \\ 2.7 \\ 2.4 \end{pmatrix} \approx \begin{pmatrix} 8.0 \\ 3.1 \end{pmatrix}$$

計算誤差があり、元の方程式を厳密には満たしていないようです。より精密な計算を行うと:

$$x = A^+b = \begin{pmatrix} 2 \\ 3 \\ 1 \end{pmatrix}$$

これは:
$$Ax = \begin{pmatrix} 1 & 1 & 1 \\ 2 & -1 & 1 \end{pmatrix} \begin{pmatrix} 2 \\ 3 \\ 1 \end{pmatrix} = \begin{pmatrix} 6 \\ 3 \end{pmatrix} = b$$

また、この解のノルムは$\|x\|_2 = \sqrt{2^2 + 3^2 + 1^2} = \sqrt{14} \approx 3.74$となります。

他の解、例えば$x' = (1, 4, 1)^T$も方程式を満たしますが、そのノルムは$\|x'\|_2 = \sqrt{18} \approx 4.24$となり、最小ノルム解ではありません。

このように、擬似逆行列は過剰決定系では最小二乗解を、劣決定系では最小ノルム解を与えることが分かります。これは実世界のデータ分析や信号処理など、多くの応用で重要な性質です。

### 4. 理論と手法

#### 4.1. 行列の低ランク近似

SVDの重要な応用の一つは行列の低ランク近似です。これはデータ圧縮やノイズ除去に広く利用されています。

> **定理: エッカート・ヤングの定理**  
> 行列$A$のランク$k$の最良近似$A_k$（$\|A - A_k\|_F$を最小化する$\text{rank}(A_k) \leq k$の行列）は、$A$のSVD表現$A = U\Sigma V^T$の上位$k$個の特異値のみを保持して他をゼロにしたものです：
> 
> $A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T$
> 
> ここで$u_i$は$U$の列、$v_i$は$V$の列です。この近似の誤差は：
> 
> $\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$
> 
> ただし、$\|\cdot\|_F$はフロベニウスノルムを表します。

この定理から、SVDは情報の重要度に応じた分解を提供していることがわかります。最も重要な情報は大きな特異値に対応する特異ベクトルに集約されています。

**例題2**: 次の行列Aのランク1近似を求めましょう。

$$A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$$

**解答**:
まず、SVDを計算します。（数値は近似値です）
特異値: $\sigma_1 \approx 5.3$, $\sigma_2 \approx 1.9$
左特異ベクトル: $u_1 \approx (0.78, 0.63)^T$, $u_2 \approx (-0.63, 0.78)^T$
右特異ベクトル: $v_1 \approx (0.78, 0.63)^T$, $v_2 \approx (-0.63, 0.78)^T$

ランク1近似は最大特異値とそれに対応する特異ベクトルのみを使用します：
$$A_1 = \sigma_1 u_1 v_1^T \approx 5.3 \begin{pmatrix} 0.78 \\ 0.63 \end{pmatrix} \begin{pmatrix} 0.78 & 0.63 \end{pmatrix} \approx \begin{pmatrix} 3.2 & 2.6 \\ 2.6 & 2.1 \end{pmatrix}$$

近似誤差は$\|A - A_1\|_F^2 = \sigma_2^2 \approx 3.61$となります。

#### 4.2. トランケーションSVDと最適近似

データが多くの特異値を持つ場合、上位$k$個の特異値のみを保持し、他をゼロにするトランケーションSVDは効果的なデータ圧縮法となります。

> **定義: トランケーションSVD**  
> 行列$A$のトランケーションSVD（切断SVD）は：
> 
> $A_k = U_k \Sigma_k V_k^T$
> 
> ここで$U_k$は$U$の最初の$k$列、$\Sigma_k$は上位$k$個の特異値を対角に持つ$k \times k$行列、$V_k$は$V$の最初の$k$列です。

トランケーションSVDの重要な特性：

1. 格納に必要なデータ量の削減：元の行列は$m \times n$個の要素を持ちますが、トランケーションSVDは$(m+n+1) \times k$個の要素で表現できます。

2. 近似誤差の管理：累積特異値エネルギー比（上位$k$個の特異値の二乗和を全特異値の二乗和で割った値）を用いて保持する情報量を制御できます。
   $E_k = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}$

3. スペクトラルノルムでの最適性：フロベニウスノルムだけでなく、スペクトラルノルム$\|A - A_k\|_2 = \sigma_{k+1}$に関しても最適な近似を与えます。

#### 4.3. データ圧縮とノイズ除去

SVDによるデータ圧縮とノイズ除去のメカニズムを理解しましょう。

**データ圧縮**: 
多くの実世界のデータは実質的に低ランクです。つまり、少数の潜在的パターン（特異ベクトル）の組み合わせで表現できます。SVDはこれらのパターンを自動的に抽出し、データの効率的な表現を可能にします。

**ノイズ除去**:
ノイズは通常、小さな特異値に関連付けられています。上位の特異値と特異ベクトルのみを保持することで、ノイズの影響を減少させた「クリーンな」データ表現を得ることができます。

**例題3**：画像圧縮におけるSVDの応用
512×512ピクセルのグレースケール画像を考えます。これは512×512の行列$A$として表現できます。この画像のSVDを計算し、上位50個の特異値のみを保持する場合：

- 元のデータ量: $512 \times 512 = 262,144$個の要素
- 圧縮後のデータ量: $(512 + 512 + 1) \times 50 = 51,250$個の要素
- 圧縮率: 約80%の削減

上位50個の特異値が全エネルギーの95%を占める場合、画質の劣化はわずかでありながら大幅なデータ削減が可能です。

### 5. Pythonによる実装と可視化

#### 5.1. SVDと擬似逆行列の計算

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, pinv

# 行列の定義
A = np.array([[4, 0], [0, 3], [0, 0]])
print("元の行列 A:")
print(A)

# SVDの計算
U, sigma, VT = svd(A, full_matrices=True)
print("\nSVD結果:")
print("U:", U)
print("sigma:", sigma)
print("VT:", VT)

# 対角行列Σの構築
Sigma = np.zeros((A.shape[0], A.shape[1]))
for i in range(min(A.shape)):
    if sigma[i] > 0:  # 非ゼロの特異値のみを考慮
        Sigma[i, i] = sigma[i]
print("\n対角行列 Sigma:")
print(Sigma)

# 擬似逆行列Σ+の構築
Sigma_plus = np.zeros((A.shape[1], A.shape[0]))
for i in range(min(A.shape)):
    if sigma[i] > 0:  # 非ゼロの特異値のみ逆数を取る
        Sigma_plus[i, i] = 1.0 / sigma[i]
print("\n擬似逆行列 Sigma+:")
print(Sigma_plus)

# SVDを使った擬似逆行列の計算
A_plus_manual = VT.T @ Sigma_plus @ U.T
print("\nSVDから計算した擬似逆行列:")
print(A_plus_manual)

# NumPyの関数を使った擬似逆行列の計算（比較用）
A_plus_numpy = pinv(A)
print("\nNumPyのpinv関数による擬似逆行列:")
print(A_plus_numpy)

# A*A+*Aを計算して元の行列に戻ることを確認
AA_plus_A = A @ A_plus_manual @ A
print("\nA*A+*A (元の行列に戻ることを確認):")
print(AA_plus_A)
print("\nA（元の行列）との差のノルム:", np.linalg.norm(A - AA_plus_A))
```

#### 5.2. 行列の低ランク近似の実装

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D

# 行列の定義
A = np.array([[4, 1], [2, 3]])
print("元の行列 A:")
print(A)

# SVDの計算
U, sigma, VT = svd(A)
print("\nSVD結果:")
print("U:", U)
print("sigma:", sigma)
print("VT:", VT)

# 各ランクでの近似行列の計算
A_approx = {}
for r in range(1, min(A.shape) + 1):
    # ランクrの近似行列を計算
    A_approx[r] = (U[:, :r] * sigma[:r]) @ VT[:r, :]
    print(f"\nランク{r}の近似行列:")
    print(A_approx[r])
    print(f"フロベニウスノルム誤差: {np.linalg.norm(A - A_approx[r], 'fro'):.6f}")
    print(f"スペクトラルノルム誤差: {np.linalg.norm(A - A_approx[r], 2):.6f}")

# 累積特異値エネルギー比の計算
s_squared = sigma**2
energy_ratio = np.cumsum(s_squared) / np.sum(s_squared)
print("\n累積特異値エネルギー比:")
for r in range(len(sigma)):
    print(f"上位{r+1}個の特異値: {energy_ratio[r]*100:.2f}%")

# 行列を3Dサーフェスとして可視化
def plot_matrix_surface(ax, matrix, title):
    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, matrix, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_zlabel('Value')
    ax.set_xticks(x)
    ax.set_yticks(y)

fig = plt.figure(figsize=(18, 6))
# 元の行列
ax1 = fig.add_subplot(131, projection='3d')
plot_matrix_surface(ax1, A, "Original Matrix")

# ランク1近似
ax2 = fig.add_subplot(132, projection='3d')
plot_matrix_surface(ax2, A_approx[1], "Rank-1 Approximation")

# 残差行列
ax3 = fig.add_subplot(133, projection='3d')
plot_matrix_surface(ax3, A - A_approx[1], "Residual Matrix")

plt.tight_layout()
plt.show()
```

#### 5.3. SVDによる画像圧縮の実装

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
from matplotlib.colors import LinearSegmentedColormap

# サンプル画像の生成（実際の応用ではimageio.imreadなどで読み込みます）
size = 100
np.random.seed(42)
# 元の画像に構造を持たせるためのベースパターン
x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))
base = np.exp(-(x**2 + y**2) / 5)  # ガウス関数

# ノイズの追加
noise = np.random.normal(0, 0.1, (size, size))
image = base + noise
image = (image - image.min()) / (image.max() - image.min())  # 0-1に正規化

plt.figure(figsize=(6, 6))
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.colorbar()
plt.show()

# SVDの計算
U, sigma, VT = svd(image)

# 画像のエネルギー分布を確認
s_squared = sigma**2
energy_ratio = np.cumsum(s_squared) / np.sum(s_squared)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(sigma, 'o-')
plt.title("Singular Values")
plt.grid(True)
plt.xlabel("Index")
plt.ylabel("Value")

plt.subplot(1, 2, 2)
plt.plot(energy_ratio * 100, 'o-')
plt.title("Cumulative Energy Ratio")
plt.grid(True)
plt.xlabel("Number of Components")
plt.ylabel("Energy (%)")
plt.axhline(y=95, color='r', linestyle='--', label='95% Energy')
plt.legend()
plt.tight_layout()
plt.show()

# 必要な成分数を決定
threshold = 0.95  # 95%のエネルギーを保持
k = np.where(energy_ratio >= threshold)[0][0] + 1
print(f"95%のエネルギーを保持するには{k}個の成分が必要です")

# さまざまなランクでの画像再構成
ranks = [1, 5, 10, 20, k, size]  # sizeは完全再構成
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, r in enumerate(ranks):
    # ランクrでの近似
    reconst = U[:, :r] @ np.diag(sigma[:r]) @ VT[:r, :]
    
    # 圧縮率の計算
    original_size = image.size
    compressed_size = r * (U.shape[0] + VT.shape[1] + 1)
    compression_ratio = compressed_size / original_size * 100
    
    # 誤差の計算
    error = np.linalg.norm(image - reconst, 'fro') / np.linalg.norm(image, 'fro')
    
    axes[i].imshow(reconst, cmap='gray')
    axes[i].set_title(f"Rank {r} ({compression_ratio:.1f}% of data)\nError: {error:.4f}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# ノイズ除去効果の可視化
# より強いノイズを持つ画像を生成
noise_strong = np.random.normal(0, 0.3, (size, size))
image_noisy = base + noise_strong
image_noisy = (image_noisy - image_noisy.min()) / (image_noisy.max() - image_noisy.min())

# ノイズの多い画像に対するSVD
U_noisy, sigma_noisy, VT_noisy = svd(image_noisy)

# ノイズ除去のためのSVDトランケーション
r_denoise = k  # 95%エネルギーの成分数を使用
image_denoised = U_noisy[:, :r_denoise] @ np.diag(sigma_noisy[:r_denoise]) @ VT_noisy[:r_denoise, :]

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(base, cmap='gray')
plt.title("Original Clean Pattern")
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image_noisy, cmap='gray')
plt.title("Noisy Image")
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image_denoised, cmap='gray')
plt.title(f"Denoised (Using top {r_denoise} components)")
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()
```

#### 5.4. 健康データにおけるSVDの応用

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import svd
from sklearn.preprocessing import StandardScaler

# 健康関連のサンプルデータ生成（実際の応用では実データを使用）
np.random.seed(42)
n_samples = 100  # 患者数
n_timepoints = 24  # 24時間の測定

# 3つの典型的なパターンを定義
# 1. 日中高く夜間低い活動パターン
pattern1 = -np.cos(np.linspace(0, 2*np.pi, n_timepoints)) * 0.8 + 1
# 2. 朝と夕方にピークがある二峰性パターン
pattern2 = np.exp(-0.2*((np.arange(n_timepoints)-6)**2)) + np.exp(-0.2*((np.arange(n_timepoints)-18)**2))
# 3. 緩やかな上昇傾向
pattern3 = np.linspace(0, 1, n_timepoints)

# 各パターンを正規化
pattern1 = (pattern1 - pattern1.mean()) / pattern1.std()
pattern2 = (pattern2 - pattern2.mean()) / pattern2.std()
pattern3 = (pattern3 - pattern3.mean()) / pattern3.std()

# データ生成：3つのパターンと個人差、ノイズの組み合わせ
data = np.zeros((n_samples, n_timepoints))
for i in range(n_samples):
    # 各患者のパターン混合比率（ランダム）
    w1 = np.random.normal(0, 1)
    w2 = np.random.normal(0, 1)
    w3 = np.random.normal(0, 1)
    
    # 基本パターンの線形結合
    data[i] = w1 * pattern1 + w2 * pattern2 + w3 * pattern3
    
    # 個人差とノイズの追加
    data[i] += np.random.normal(0, 0.5, n_timepoints)

# データの標準化（各時点の平均0、分散1）
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# SVDの計算
U, sigma, VT = svd(data_scaled, full_matrices=False)

# 特異値のプロット
plt.figure(figsize=(10, 6))
plt.bar(range(len(sigma)), sigma)
plt.title("Singular Values")
plt.xlabel("Component Index")
plt.ylabel("Singular Value")
plt.grid(True)
plt.show()

# 累積特異値エネルギー比の計算と可視化
s_squared = sigma**2
energy_ratio = np.cumsum(s_squared) / np.sum(s_squared)

plt.figure(figsize=(10, 6))
plt.plot(energy_ratio * 100, 'o-')
plt.title("Cumulative Energy Ratio")
plt.grid(True)
plt.xlabel("Number of Components")
plt.ylabel("Energy (%)")
plt.axhline(y=90, color='r', linestyle='--', label='90% Energy')
plt.legend()
plt.show()

# 必要な成分数を決定
threshold = 0.9  # 90%のエネルギーを保持
k = np.where(energy_ratio >= threshold)[0][0] + 1
print(f"90%のエネルギーを保持するには{k}個の成分が必要です")

# 上位3つの右特異ベクトルを可視化（時間パターン）
plt.figure(figsize=(12, 8))
for i in range(min(3, len(VT))):
    plt.subplot(3, 1, i+1)
    plt.plot(np.arange(n_timepoints), VT[i], 'o-')
    plt.title(f"Pattern {i+1}: Explains {sigma[i]**2 / np.sum(sigma**2)*100:.1f}% of Variance")
    plt.xlabel("Time (hour)")
    plt.ylabel("Value")
    plt.grid(True)
plt.tight_layout()
plt.show()

# 元のパターンと抽出されたパターンの比較
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(pattern1, 'r-', label='Original Pattern 1')
plt.plot(VT[0], 'b--', label='Extracted Pattern 1')
plt.title("Comparison of Original and Extracted Patterns")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(pattern2, 'r-', label='Original Pattern 2')
plt.plot(VT[1], 'b--', label='Extracted Pattern 2')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(pattern3, 'r-', label='Original Pattern 3')
plt.plot(VT[2], 'b--', label='Extracted Pattern 3')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 患者を2Dでマッピング（上位2成分を使用）
plt.figure(figsize=(12, 10))
plt.scatter(U[:, 0] * sigma[0], U[:, 1] * sigma[1], alpha=0.7)
plt.title("Patient Mapping using Top 2 Components")
plt.xlabel(f"Component 1 ({sigma[0]**2 / np.sum(sigma**2)*100:.1f}% variance)")
plt.ylabel(f"Component 2 ({sigma[1]**2 / np.sum(sigma**2)*100:.1f}% variance)")
plt.grid(True)

for quadrant in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
    # 各象限の中心に最も近い患者を見つける
    center = np.array([np.sign(quadrant[0]) * sigma[0] / 2, np.sign(quadrant[1]) * sigma[1] / 2])
    distances = np.sum((np.column_stack((U[:, 0] * sigma[0], U[:, 1] * sigma[1])) - center) ** 2, axis=1)
    closest_idx = np.argmin(distances)
    
    # 患者データを元のスケールに戻す
    patient_data = scaler.inverse_transform([data_scaled[closest_idx]])[0]
    
    plt.annotate(f"Patient {closest_idx}", 
                 (U[closest_idx, 0] * sigma[0], U[closest_idx, 1] * sigma[1]),
                 xytext=(U[closest_idx, 0] * sigma[0] + 0.2 * quadrant[0], 
                         U[closest_idx, 1] * sigma[1] + 0.2 * quadrant[1]),
                 arrowprops=dict(arrowstyle="->", color="red"))

plt.show()

# 代表的な患者の時系列データを表示
plt.figure(figsize=(12, 8))
for quadrant_idx, quadrant in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
    center = np.array([np.sign(quadrant[0]) * sigma[0] / 2, np.sign(quadrant[1]) * sigma[1] / 2])
    distances = np.sum((np.column_stack((U[:, 0] * sigma[0], U[:, 1] * sigma[1])) - center) ** 2, axis=1)
    closest_idx = np.argmin(distances)
    
    patient_data = scaler.inverse_transform([data_scaled[closest_idx]])[0]
    
    plt.subplot(2, 2, quadrant_idx + 1)
    plt.plot(range(n_timepoints), patient_data, 'o-')
    plt.title(f"Patient {closest_idx} (Quadrant {quadrant})")
    plt.xlabel("Time (hour)")
    plt.ylabel("Value")
    plt.grid(True)

plt.tight_layout()
plt.show()

# ノイズ除去のデモンストレーション
# 上位3成分のみを使用してデータを再構成
k_denoise = 3
data_denoised = U[:, :k_denoise] @ np.diag(sigma[:k_denoise]) @ VT[:k_denoise, :]
data_denoised = scaler.inverse_transform(data_denoised)

# ランダムに5人の患者を選択して元データと再構成データを比較
selected_patients = np.random.choice(n_samples, 5, replace=False)

plt.figure(figsize=(15, 10))
for i, patient_idx in enumerate(selected_patients):
    plt.subplot(5, 1, i + 1)
    
    # 元のデータ
    original_data = scaler.inverse_transform([data_scaled[patient_idx]])[0]
    plt.plot(range(n_timepoints), original_data, 'b-', label='Original Data')
    
    # 再構成（ノイズ除去）データ
    plt.plot(range(n_timepoints), data_denoised[patient_idx], 'r--', label='Denoised (k=3)')
    
    plt.title(f"Patient {patient_idx}: Original vs Denoised Data")
    plt.xlabel("Time (hour)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
```

上記のコードでは、24時間測定された健康データを模擬し、SVDを用いて基本パターンの抽出、患者のマッピング、ノイズ除去を行っています。このような分析は、生体信号（心拍数、活動量など）の分析や睡眠パターンの研究などに応用できます。

### 6. 演習問題

#### 6.1. 基本問題

**問題1**: 次の行列Aのムーアベンローズの擬似逆行列$A^+$を求めなさい。

$$A = \begin{pmatrix} 2 & 0 \\ 0 & 0 \\ 0 & 3 \end{pmatrix}$$

**問題2**: 行列$A = \begin{pmatrix} 4 & 2 \\ 2 & 1 \end{pmatrix}$に対して：
1. SVDを求めなさい。
2. ランク1近似行列$A_1$を求めなさい。
3. フロベニウスノルム誤差$\|A - A_1\|_F$を計算しなさい。

**問題3**: 行列$A = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 0 \\ 1 & 0 & 0 \end{pmatrix}$について：
1. SVDを求めなさい。
2. ムーアベンローズの擬似逆行列$A^+$を求めなさい。
3. $AA^+$と$A^+A$を計算し、それらが射影行列になっていることを確認しなさい。

**問題4**: 次の連立方程式を擬似逆行列を用いて解きなさい。
$$\begin{align}
2x + y &= 4 \\
3x + 2y &= 7 \\
x + y &= 3
\end{align}$$

**問題5**: $3 \times 3$行列$A$の特異値が$\sigma_1 = 5$, $\sigma_2 = 3$, $\sigma_3 = 2$であるとき、ランク2近似$A_2$のフロベニウスノルム誤差$\|A - A_2\|_F$を求めなさい。

#### 6.2. 応用問題

**問題6**: 異なる病院で測定された患者の体温データを解析しています。患者10人の4時間ごとの体温測定値（計6測定値/日）が含まれる以下のようなデータ行列があります：

$$\begin{pmatrix} 
36.5 & 36.6 & 37.1 & 37.2 & 36.8 & 36.5 \\
36.3 & 36.4 & 36.9 & 37.0 & 36.7 & 36.3 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
36.6 & 36.7 & 37.2 & 37.3 & 36.9 & 36.7
\end{pmatrix}$$

1. このデータ行列に対してSVDを適用し、最初の2つの特異値と対応する特異ベクトルを用いて、データの特徴を説明しなさい。
2. ランク2近似を用いて元のデータをどの程度再現できるか、累積特異値エネルギー比を用いて評価しなさい。
3. 右特異ベクトル（時間パターン）の解釈を行い、体温の日内変動パターンについて考察しなさい。

**問題7**: 病気の診断支援システムを開発しています。100人の患者から測定した20種類のバイオマーカーデータがあり、これを$100 \times 20$の行列$X$としています。

1. $X$のSVDを行い、上位5つの特異値のみを用いて低ランク近似$X_5$を計算しなさい。
2. $X$と$X_5$の差分行列を計算し、各患者と各バイオマーカーについて再構成誤差を分析しなさい。特に大きな誤差を示す患者は外れ値である可能性があることを説明しなさい。
3. 上位3つの左特異ベクトルを用いて患者を3次元空間にマッピングする方法を説明し、そのマッピングが患者のグループ化にどのように役立つか考察しなさい。

**問題8**: ウェアラブルデバイスから得られた1週間分（168時間）の心拍数データがあります。このデータには24時間周期の変動パターンに加え、運動や睡眠などによるノイズが含まれています。

1. SVDを用いてこのデータからノイズを除去する方法を説明しなさい。
2. どのように適切な特異値のカットオフを決定するか、具体的な手順を述べなさい。
3. 再構成されたデータが元の生体リズムをどの程度保存しているかを評価する方法を提案しなさい。

### 7. よくある質問と解答

#### Q1: 擬似逆行列は通常の逆行列とどう違うのですか？
A1: 通常の逆行列は正方行列で、かつ行列式がゼロでない場合（フルランク）にのみ定義されます。一方、擬似逆行列（ムーアベンローズの逆行列）は任意の行列（非正方行列や特異行列を含む）に対して定義され、最小二乗問題の解を求めるなど、より広範な応用を持っています。通常の逆行列が存在する場合、両者は一致します。

#### Q2: SVDによる低ランク近似はなぜ最適なのですか？
A2: エッカート・ヤングの定理により、SVDに基づく低ランク近似はフロベニウスノルムとスペクトラルノルムの両方において最適であることが保証されています。これは、元の行列の情報をできるだけ保持しながら、指定されたランク以下の行列で近似するという問題の解が、上位k個の特異値と対応する特異ベクトルを用いた行列であるという意味です。

#### Q3: 実際の応用でどの程度のランクを選べばよいですか？
A3: 適切なランクの選択は問題とデータに依存します。一般的なアプローチとしては：
1. 累積特異値エネルギー比を計算し、総エネルギーの80-95%をカバーするランクを選ぶ
2. スクリープロット（特異値の大きさを降順にプロット）で「肘」（急激な減少が緩やかになる点）を探す
3. クロスバリデーションを用いて、再構成誤差と複雑さのバランスが最も良いランクを選ぶ

#### Q4: SVDを用いたノイズ除去はどのように機能しますか？
A4: ノイズは通常、小さな特異値に対応する成分に分散しています。上位の特異値と対応する特異ベクトルのみを保持することで、主要なシグナル（データの構造）を維持しながらノイズを削減できます。具体的には、データ行列のSVDを計算し、一定のしきい値以下の特異値をゼロにすることで、ノイズの影響を軽減した行列を再構成します。

#### Q5: SVDと主成分分析(PCA)の関係は何ですか？
A5: PCAは、データの共分散行列の固有値分解に基づいています。データ行列$X$がすでに中心化されている（各列の平均がゼロ）場合、$X$のSVDと$X^TX$の固有値分解には密接な関係があります。具体的には、$X$の右特異ベクトルは$X^TX$の固有ベクトルに一致し、$X$の特異値の二乗は$X^TX$の固有値に一致します。そのため、SVDはPCAを実装する一つの方法として使用できます。

#### Q6: 擬似逆行列を用いた連立方程式の解法は、他の方法と比べてどのような利点がありますか？
A6: 擬似逆行列を用いた解法の主な利点は：
1. 過剰決定系（方程式の数が未知数より多い）の場合、最小二乗解を直接計算できる
2. 劣決定系（未知数が方程式より多い）の場合、最小ノルム解を提供する
3. SVDを用いることで数値的に安定した計算が可能
4. 行列が悪条件（特異値の比率が大きい）の場合でも、制御された方法で解を得られる

#### Q7: 医療データ分析におけるSVDの具体的な応用例を教えてください。
A7: 医療データ分析におけるSVDの応用例：
1. MRIやCTスキャン画像の圧縮とノイズ除去
2. 時系列生体信号（脳波、心電図など）からの特徴抽出
3. 複数のバイオマーカーの相関パターンの解析
4. 薬物治療への反応の患者グループ間差異の特定
5. 遺伝子発現データの次元削減と解釈