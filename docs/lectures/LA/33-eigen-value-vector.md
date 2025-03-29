# 線形代数学 I / 基礎 / II - 第33回講義ノート

## 1. 講義情報と予習ガイド

**講義回**: 第33回  
**テーマ**: 固有値・固有ベクトル  
**関連項目**: 線形変換、特性方程式、固有空間、固有多項式  
**予習内容**: 行列の基本演算、行列式の計算、線形変換の概念

## 2. 学習目標

本講義を通じて、以下の目標の達成を目指します：

1. 固有値と固有ベクトルの概念を理解し、その幾何学的意味を説明できる
2. 特性方程式を立て、固有値を求めることができる
3. 固有ベクトルを計算し、その性質を理解する
4. 固有値の総和と行列のトレース、固有値の積と行列式の関係を説明できる
5. データサイエンスにおける固有値・固有ベクトルの応用例を理解する

## 3. 基本概念

### 3.1 固有値と固有ベクトルの定義

行列による線形変換を考えるとき、方向が変わらず大きさだけが変化するベクトルが存在します。このような特別なベクトルを固有ベクトルと呼び、その拡大・縮小率を固有値と呼びます。

> **定義**: $n \times n$行列$A$に対して、ゼロでないベクトル$\mathbf{v} \in \mathbb{R}^n$が存在し、ある定数$\lambda \in \mathbb{R}$（または$\mathbb{C}$）に対して
>
> $A\mathbf{v} = \lambda \mathbf{v}$
>
> を満たすとき、$\lambda$を行列$A$の**固有値**、$\mathbf{v}$を$\lambda$に対応する**固有ベクトル**と呼びます。

この定義は以下のように理解することができます：
- 行列$A$による線形変換を受けても、固有ベクトル$\mathbf{v}$は方向が変わらず
- その大きさが$\lambda$倍になる

### 3.2 固有値・固有ベクトルの幾何的解釈

固有ベクトルと固有値の幾何的意味を考えましょう。行列$A$による線形変換を考えたとき：

1. 固有ベクトルは、変換後も元の方向を保持するベクトル
2. 固有値は、その方向のベクトルがどれだけ引き伸ばされる（または圧縮される）かを表す倍率
3. 固有値が正なら同じ方向に引き伸ばし、負なら反対方向に引き伸ばし

例として、2次元の回転行列を考えると、複素固有値を持ち、実の固有ベクトルは存在しないことがわかります。これは回転によってすべての方向が変わることを意味しています。

### 3.3 固有多項式と特性方程式の詳細解説

固有値と固有ベクトルを求めるプロセスは、線形代数学の重要な部分です。ここでは、固有多項式と特性方程式がなぜ固有値と固有ベクトルの計算に不可欠なのかを詳しく説明します。

#### 固有値・固有ベクトルの基本的な関係

まず、固有値と固有ベクトルの定義を思い出しましょう。$n \times n$行列$A$に対して、ゼロでないベクトル$\mathbf{v}$と数$\lambda$が次の関係を満たすとき：

$$A\mathbf{v} = \lambda \mathbf{v}$$

このとき、$\lambda$を行列$A$の固有値、$\mathbf{v}$を対応する固有ベクトルと呼びます。

この関係式を次のように変形してみましょう：

$$A\mathbf{v} - \lambda \mathbf{v} = \mathbf{0}$$
$$（A - \lambda I）\mathbf{v} = \mathbf{0}$$

ここで$I$は$n \times n$の単位行列です。

#### 同次方程式の非自明解の条件

上記の式 $(A - \lambda I)\mathbf{v} = \mathbf{0}$ は、行列 $(A - \lambda I)$ に関する同次線形方程式です。この方程式において、 $\mathbf{v} \neq \mathbf{0}$ という非自明解（ゼロでない解）が存在するための必要十分条件は何でしょうか？

線形代数学の基本定理によれば、同次方程式$(A - \lambda I)\mathbf{v} = \mathbf{0}$が非自明解を持つための必要十分条件は：

**行列$(A - \lambda I)$が正則でない（逆行列を持たない）こと**

言い換えると：

**行列$(A - \lambda I)$のランクが$n$未満であること**

または：

**行列$(A - \lambda I)$の行列式がゼロであること**

つまり：

$$\det(A - \lambda I) = 0$$

これが特性方程式（固有方程式）です。

#### 固有多項式の導入

$\det(A - \lambda I)$を$\lambda$の関数と見なすと、これは$\lambda$に関する$n$次多項式になります。この多項式を固有多項式（または特性多項式）と呼びます：

$$p_A(\lambda) = \det(A - \lambda I)$$

例えば、$2 \times 2$行列$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$の場合、固有多項式は：

$$p_A(\lambda) = \det\begin{pmatrix} a-\lambda & b \\ c & d-\lambda \end{pmatrix} = (a-\lambda)(d-\lambda) - bc$$

これを展開すると：

$$p_A(\lambda) = \lambda^2 - (a+d)\lambda + (ad-bc) = \lambda^2 - \text{tr}(A)\lambda + \det(A)$$

このように、$2 \times 2$行列の固有多項式は、行列のトレース$\text{tr}(A) = a+d$と行列式$\det(A) = ad-bc$を用いて簡潔に表現できます。

#### 特性方程式を解く理由

特性方程式$\det(A - \lambda I) = 0$を解くことで、行列$A$のすべての固有値を見つけることができます。なぜでしょうか？

1. **固有値の定義に戻る**: $A\mathbf{v} = \lambda\mathbf{v}$が非自明解$\mathbf{v} \neq \mathbf{0}$を持つためには、$\det(A - \lambda I) = 0$が成り立つ必要があります。

2. **代数学の基本定理**: $n$次多項式$p_A(\lambda)$は、重複も含めて正確に$n$個の複素根を持ちます。これらの根が行列$A$の固有値です。

3. **固有値の代数的重複度**: $p_A(\lambda)$における根$\lambda_i$の重複度を、固有値$\lambda_i$の「代数的重複度」と呼びます。

#### 計算の具体例

具体例を通じて理解を深めましょう。次の$2 \times 2$行列を考えます：

$$A = \begin{pmatrix} 3 & 1 \\ 4 & -1 \end{pmatrix}$$

特性多項式を計算します：

$$p_A(\lambda) = \det(A - \lambda I) = \det\begin{pmatrix} 3-\lambda & 1 \\ 4 & -1-\lambda \end{pmatrix}$$
$$= (3-\lambda)(-1-\lambda) - 1 \cdot 4$$
$$= (3-\lambda)(-1-\lambda) - 4$$
$$= -3-3\lambda+\lambda+\lambda^2 - 4$$
$$= \lambda^2 - 2\lambda - 7$$

特性方程式を立てると：
$$\lambda^2 - 2\lambda - 7 = 0$$

この2次方程式を解くと：
$$\lambda = \frac{2 \pm \sqrt{4+28}}{2} = \frac{2 \pm \sqrt{32}}{2} = \frac{2 \pm 4\sqrt{2}}{2} = 1 \pm 2\sqrt{2}$$

したがって、行列$A$の固有値は：
$$\lambda_1 = 1 + 2\sqrt{2} \approx 3.83$$
$$\lambda_2 = 1 - 2\sqrt{2} \approx -1.83$$

#### 固有ベクトルの計算へ

固有値が求まったら、対応する固有ベクトルを求めることができます。固有値$\lambda_i$に対して：

1. 行列$(A - \lambda_i I)$を計算する
2. 同次方程式$(A - \lambda_i I)\mathbf{v} = \mathbf{0}$を解く
3. 非自明解がその固有値に対応する固有ベクトル

例えば、上の例で固有値$\lambda_1 = 1 + 2\sqrt{2}$に対する固有ベクトルを求めるには：

$$A - \lambda_1 I = \begin{pmatrix} 3-(1+2\sqrt{2}) & 1 \\ 4 & -1-(1+2\sqrt{2}) \end{pmatrix} = \begin{pmatrix} 2-2\sqrt{2} & 1 \\ 4 & -2-2\sqrt{2} \end{pmatrix}$$

この行列を用いて同次方程式$(A - \lambda_1 I)\mathbf{v} = \mathbf{0}$を解きます。解として得られる非ゼロベクトル$\mathbf{v}$が、固有値$\lambda_1$に対応する固有ベクトルです。


## 4. 理論と手法

### 4.1 固有値の計算方法

$n \times n$行列$A$の固有値を求める手順：

1. 特性多項式$\det(A - \lambda I)$を計算する
2. 特性方程式$\det(A - \lambda I) = 0$を解く
3. 得られた解$\lambda_1, \lambda_2, \ldots, \lambda_n$が固有値

**例**: 2×2行列の場合、特性多項式は2次式になります：
$\det\begin{pmatrix} a-\lambda & b \\ c & d-\lambda \end{pmatrix} = (a-\lambda)(d-\lambda) - bc = \lambda^2 - (a+d)\lambda + (ad-bc)$

つまり、$\lambda^2 - \text{tr}(A)\lambda + \det(A) = 0$を解くことになります。

### 4.2 固有ベクトルの計算方法

固有値$\lambda$に対応する固有ベクトルを求める手順：

1. 行列$(A - \lambda I)$を計算する
2. 同次方程式$(A - \lambda I)\mathbf{v} = \mathbf{0}$を解く
3. 非自明解（ゼロでない解）がその固有値に対応する固有ベクトル

**注意点**: 
- 固有値が重複している場合、対応する固有ベクトルの数は固有値の重複度以下
- 固有ベクトルはスカラー倍すると依然として固有ベクトル（方向のみが重要）

### 4.3 固有空間

> **定義**: 固有値$\lambda$に対応する**固有空間**$E_\lambda$は、$\lambda$に対応するすべての固有ベクトルと零ベクトルからなる部分空間：
>
> $E_\lambda = \{\mathbf{v} \in \mathbb{R}^n : A\mathbf{v} = \lambda \mathbf{v}\} = \ker(A - \lambda I)$

固有空間の次元は、その固有値の幾何的重複度に等しくなります。

### 4.4 対称行列の固有値・固有ベクトルの特別な性質

実対称行列$A$（$A = A^T$）の場合：

1. すべての固有値は実数
2. 異なる固有値に対応する固有ベクトルは互いに直交
3. $n$個の線形独立な固有ベクトルが存在（直交化可能）

これらの性質は、データサイエンスにおける主成分分析（PCA）などで非常に重要になります。

## 5. Pythonによる実装と可視化

### 5.1 固有値・固有ベクトルの計算

```python
import numpy as np
import matplotlib.pyplot as plt

# 行列の定義
A = np.array([[4, 2], 
              [1, 3]])

# 固有値と固有ベクトルの計算
eigenvalues, eigenvectors = np.linalg.eig(A)

print("行列A:")
print(A)
print("\n固有値:")
print(eigenvalues)
print("\n固有ベクトル (列ベクトルとして):")
print(eigenvectors)

# トレースと固有値の和の確認
print("\nトレースと固有値の和:")
print(f"tr(A) = {np.trace(A)}")
print(f"固有値の和 = {np.sum(eigenvalues)}")

# 行列式と固有値の積の確認
print("\n行列式と固有値の積:")
print(f"det(A) = {np.linalg.det(A)}")
print(f"固有値の積 = {np.prod(eigenvalues)}")
```

### 5.2 固有ベクトルの幾何的解釈の可視化

```python
# 2次元平面での線形変換と固有ベクトルの可視化
def plot_transformation(A, eigenvalues, eigenvectors):
    plt.figure(figsize=(12, 6))
    
    # 単位円上の点群を作成
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    circle_points = np.vstack([x, y])
    
    # 行列Aによる変換後の点群
    transformed_points = A @ circle_points
    
    # 元の単位円と変換後の楕円を表示
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'b-', label='Unit Circle')
    plt.plot(transformed_points[0, :], transformed_points[1, :], 'r-', 
             label='Transformed Circle')
    
    # 固有ベクトルの表示
    for i in range(len(eigenvalues)):
        vec = eigenvectors[:, i] * 2  # 長さを強調
        transformed_vec = A @ vec      # 変換後のベクトル
        
        plt.arrow(0, 0, vec[0], vec[1], head_width=0.1, 
                  head_length=0.1, fc=f'C{i}', ec=f'C{i}', 
                  label=f'Eigenvector {i+1}')
        plt.arrow(0, 0, transformed_vec[0], transformed_vec[1], 
                  head_width=0.1, head_length=0.1, fc=f'C{i}', 
                  ec=f'C{i}', ls='--')
    
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend()
    plt.title("Linear Transformation and Eigenvectors")
    
    # 固有ベクトルの方向と拡大率を表示
    plt.subplot(1, 2, 2)
    plt.quiver([0, 0], [0, 0], 
              [eigenvectors[0, 0], eigenvectors[0, 1]], 
              [eigenvectors[1, 0], eigenvectors[1, 1]], 
              angles='xy', scale_units='xy', scale=1, 
              color=['blue', 'green'])
    plt.text(eigenvectors[0, 0]*1.1, eigenvectors[1, 0]*1.1, 
            f'λ₁={eigenvalues[0]:.2f}')
    plt.text(eigenvectors[0, 1]*1.1, eigenvectors[1, 1]*1.1, 
            f'λ₂={eigenvalues[1]:.2f}')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Eigenvectors and their Eigenvalues")
    
    plt.tight_layout()
    plt.show()

# 可視化関数の呼び出し
plot_transformation(A, eigenvalues, eigenvectors)
```

### 5.3 特性方程式の可視化

```python
# 特性多項式を計算して可視化
def plot_characteristic_polynomial(A):
    # トレースと行列式の計算
    tr_A = np.trace(A)
    det_A = np.linalg.det(A)
    
    # 特性多項式: λ^2 - tr(A)λ + det(A)
    x = np.linspace(-2, 7, 1000)
    y = x**2 - tr_A*x + det_A
    
    # 固有値
    eigenvalues = np.linalg.eigvals(A)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Characteristic Polynomial')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True)
    
    # 特性方程式の解（固有値）をマークする
    for i, ev in enumerate(eigenvalues):
        plt.plot(ev.real, 0, 'ro', markersize=8)
        plt.text(ev.real, 0.5, f'λ{i+1}={ev.real:.2f}')
    
    plt.title(f"Characteristic Polynomial: λ^2 - {tr_A}λ + {det_A}")
    plt.xlabel('λ')
    plt.ylabel('p(λ)')
    plt.legend()
    plt.xlim(min(0, np.min(eigenvalues.real)-1), np.max(eigenvalues.real)+1)
    plt.ylim(-5, 10)
    plt.show()

# 特性多項式の可視化
plot_characteristic_polynomial(A)
```

## 6. 演習問題

### 6.1 基本問題

1. 次の行列の固有値と固有ベクトルを求めなさい。
   $$A = \begin{pmatrix} 3 & 1 \\ 2 & 2 \end{pmatrix}$$

2. 次の行列の固有値と固有ベクトルを求めなさい。
   $$B = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 3 & 4 \\ 0 & 4 & 3 \end{pmatrix}$$

3. 次の行列の固有値がすべて異なることを証明しなさい。
   $$C = \begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{pmatrix}$$

4. 以下の行列の固有値とトレース、行列式の関係を確認しなさい。
   $$D = \begin{pmatrix} 4 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 4 \end{pmatrix}$$

### 6.2 応用問題

5. 対称行列 $E = \begin{pmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{pmatrix}$ の固有値と固有ベクトルを求め、固有ベクトルが互いに直交することを確認しなさい。

6. 行列 $F = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$ の固有値と固有ベクトルを求め、この行列が表す線形変換の幾何学的意味を説明しなさい。

7. ある共分散行列が次のように与えられています。
   $$\Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$$
   この行列の固有値と固有ベクトルを求め、データの主な変動方向を特定しなさい。これは主成分分析（PCA）の基本原理です。

8. 健康データサイエンスの応用問題：患者の体重（kg）と身長（cm）のデータから計算された共分散行列が次のように与えられています。
   $$\Sigma_{健康} = \begin{pmatrix} 25 & 12 \\ 12 & 9 \end{pmatrix}$$
   この行列の固有値と固有ベクトルを求め、それらが表す身体測定値の主要な変動パターンを解釈しなさい。また、このような分析が医療データ解析でどのように役立つか説明しなさい。

## 7. よくある質問と解答

### Q1: 固有値と固有ベクトルは何の役に立つのですか？
**A1**: 固有値と固有ベクトルは多くの応用があります。例えば：
- 行列のべき乗($A^n$)の効率的な計算
- 連立微分方程式の解法
- 主成分分析（PCA）によるデータの次元削減
- 画像処理における特徴抽出
- グラフ理論におけるネットワーク分析
- 量子力学における量子状態の解析

### Q2: 行列のすべての固有値が実数になるのはどのような場合ですか？
**A2**: 行列のすべての固有値が実数になるのは以下の場合です：
- 実対称行列（$A = A^T$）
- エルミート行列（複素数の場合、$A = A^*$）
- 三角行列（この場合、固有値は対角成分に一致）

### Q3: 固有ベクトルが一意に定まらないのはなぜですか？
**A3**: 固有ベクトルはその方向のみが重要で、大きさは任意です。つまり、$\mathbf{v}$が固有ベクトルなら、$c\mathbf{v}$（$c \neq 0$）も同じ固有値に対応する固有ベクトルです。また、固有値の代数的重複度が幾何的重複度より大きい場合、対応する固有空間の基底の選び方に自由度があります。

### Q4: 特性方程式を解くのが難しい高次元行列の場合はどうすればよいですか？
**A4**: 高次元の場合は数値計算手法を用います：
- NumPyの `numpy.linalg.eig()` 関数
- べき乗法（最大固有値を求める反復法）
- QR法（全ての固有値を求める効率的なアルゴリズム）
- ヤコビ法（対称行列向けの数値計算法）

### Q5: 固有値が複素数になる場合、その幾何学的な意味は何ですか？
**A5**: 固有値が複素数になる場合、対応する線形変換は回転を含みます：
- 絶対値（モジュラス）は拡大・縮小率
- 偏角（引数）は回転角度
例えば、2×2回転行列の固有値は$e^{i\theta}$と$e^{-i\theta}$の形（$\theta$は回転角）になります。

## 8. 今回の講義のまとめ

本講義では、固有値と固有ベクトルという線形代数学の重要な概念を学びました。

1. 固有値と固有ベクトルは、行列による線形変換において方向が保存されるベクトルとその拡大率を表す
2. 特性方程式($\det(A - \lambda I) = 0$)を解くことで固有値が求まる
3. 各固有値に対応する固有ベクトルは、同次方程式$(A - \lambda I)\mathbf{v} = \mathbf{0}$を解くことで求められる
4. 行列のトレースは固有値の和、行列式は固有値の積と等しい
5. 実対称行列は特に重要な性質（実固有値、直交固有ベクトル）を持つ

これらの概念は、次回学ぶ行列の対角化や、後の主成分分析などのデータサイエンス手法の基礎となります。固有値と固有ベクトルの幾何学的理解を深めることで、高次元データの分析や複雑な線形変換の理解が容易になります。特に健康データや医療データの分析において、次元削減や重要な変動パターンの発見に役立つ基礎知識となります。