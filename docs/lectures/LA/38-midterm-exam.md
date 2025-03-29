# 線形代数学 第38回講義ノート

## 1. 講義情報と予習ガイド

**講義回**: 第38回
**テーマ**: 固有値と固有ベクトルに関する中間試験
**関連項目**: 固有値、固有ベクトル、対角化、2次形式
**予習すべき内容**: 
- 第32回〜第37回の講義内容（特に固有値・固有ベクトルの計算方法と対角化の手順）
- 行列のべき乗計算と数列の一般項の導出方法

## 2. 学習目標

本講義の中間試験を通じて、以下の能力を確認します：

1. 行列の固有値と固有ベクトルを正確に計算できる
2. 対角化可能な行列に対して、対角化の手順を適切に実行できる
3. 対角化を利用して行列のべき乗（A^n）を効率的に計算できる
4. 行列に関連する数列の一般項を導出できる

## 3. 試験の概要

中間試験は以下の構成で実施されます：

- 試験時間: 60分
- 問題数: 4問
  - 問題1-2: 固有値と固有ベクトルの計算を中心とした問題
  - 問題3-4: 対角化を利用した A^n や数列の一般項の計算問題
- 配点: 各問25点（合計100点）
- 持ち込み: 自筆の講義ノートのみ許可（電子機器、教科書、プリント類は不可）

## 4. 試験対策ポイント

### 4.1 固有値・固有ベクトルの計算

> **定義**: 行列 $A$ に対して、ゼロでないベクトル $v$ と、スカラー $\lambda$ が
> 
> $Av = \lambda v$
> 
> を満たすとき、$\lambda$ を行列 $A$ の**固有値**、$v$ を対応する**固有ベクトル**と呼ぶ。

固有値・固有ベクトルを求める手順：

1. 特性方程式 $det(A - \lambda I) = 0$ を立てる
2. 特性方程式を解いて固有値 $\lambda$ を求める
3. 各固有値 $\lambda$ について、$(A - \lambda I)v = 0$ を解いて対応する固有ベクトル $v$ を求める

#### 例題1: 2×2行列の固有値・固有ベクトル

行列 $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$ の固有値と固有ベクトルを求めよ。

**解答**:

1. 特性方程式を立てる：
   $$det(A - \lambda I) = det\begin{pmatrix} 3-\lambda & 1 \\ 1 & 3-\lambda \end{pmatrix} = (3-\lambda)^2 - 1 = 0$$

2. 特性方程式を解く：
   $(3-\lambda)^2 = 1$より、$\lambda = 2$ または $\lambda = 4$

3. 固有ベクトルを求める：
   
   $\lambda = 2$ のとき：
   $(A - 2I)v = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}v = 0$
   この同次方程式を解くと、$v = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$ （または定数倍）が得られる。
   
   $\lambda = 4$ のとき：
   $(A - 4I)v = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}v = 0$
   この同次方程式を解くと、$v = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ （または定数倍）が得られる。

#### 例題2: 3×3行列の固有値・固有ベクトル

行列 $A = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 3 & 4 \\ 0 & 1 & 2 \end{pmatrix}$ の固有値と固有ベクトルを求めよ。

**解答**:

1. 特性方程式を立てる：
   $$det(A - \lambda I) = det\begin{pmatrix} 2-\lambda & 0 & 0 \\ 0 & 3-\lambda & 4 \\ 0 & 1 & 2-\lambda \end{pmatrix}
   = (2-\lambda)·det\begin{pmatrix} 3-\lambda & 4 \\ 1 & 2-\lambda \end{pmatrix}
   = (2-\lambda)·((3-\lambda)(2-\lambda) - 4)
   = (2-\lambda)·((3-\lambda)(2-\lambda) - 4·1)
   = (2-\lambda)·(6-3\lambda-2\lambda+\lambda^2 - 4)
   = (2-\lambda)·(\lambda^2 - 5\lambda + 2)
   = 0$$

2. 特性方程式を解く：
   $(2-\lambda) = 0 または \lambda^2 - 5\lambda + 2 = 0
   \lambda = 2$ または $(\lambda - 2)(\lambda - 1) = 0$
   よって、$\lambda = 2$（重根）または $\lambda = 1$

3. 固有ベクトルを求める：
   
   $$\lambda = 2 のとき：
   (A - 2I)v = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 4 \\ 0 & 1 & 0 \end{pmatrix}v = 0$$
   
   この同次方程式を解くと、$$v_1 = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} と v_2 = \begin{pmatrix} 0 \\ -4 \\ 1 \end{pmatrix}$$ （またはそれらの線形結合）が得られる。
   
   $\lambda = 1$ のとき：
   $$(A - I)v = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 2 & 4 \\ 0 & 1 & 1 \end{pmatrix}v = 0$$
   
   この同次方程式を解くと、$$v_3 = \begin{pmatrix} 0 \\ -2 \\ 1 \end{pmatrix}$$ （または定数倍）が得られる。

### 4.2 行列の対角化

> **定義**: $n×n$ 行列 $A$ が対角化可能であるとは、正則行列 $P$ と対角行列 $D$ が存在して、
> 
> $P^{-1}AP = D$
> 
> と表せることをいう。このとき、$D$ の対角成分は $A$ の固有値であり、$P$ の列ベクトルは $A$ の対応する固有ベクトルである。

対角化の手順：

1. 行列 A の固有値と固有ベクトルを求める
2. 固有ベクトルを列とする行列 P を作る
3. P^{-1}AP = D となる対角行列 D を求める（D の対角成分は A の固有値）

#### 例題3: 行列の対角化

行列 A = \begin{pmatrix} 4 & -3 \\ 2 & -1 \end{pmatrix} を対角化せよ。

**解答**:

1. 固有値と固有ベクトルを求める：
   
   特性方程式：$det(A - \lambda I) = det\begin{pmatrix} 4-\lambda & -3 \\ 2 & -1-\lambda \end{pmatrix}
   = (4-\lambda)(-1-\lambda) - (-3)(2)
   = -4-4\lambda+\lambda+\lambda^2 - (-6)
   = \lambda^2 - 3\lambda - 2 = 0$
   
   解くと、$\lambda = -1, 2$
   
   $\lambda = -1$ のとき：
   $$(A - (-1)I)v = \begin{pmatrix} 5 & -3 \\ 2 & 0 \end{pmatrix}v = 0
   $$
   この同次方程式を解くと、$v_1 = \begin{pmatrix} 3 \\ 5 \end{pmatrix}$ （または定数倍）が得られる。
   
   $\lambda = 2$ のとき：
   $(A - 2I)v = \begin{pmatrix} 2 & -3 \\ 2 & -3 \end{pmatrix}v = 0$
   
   この同次方程式を解くと、$v_2 = \begin{pmatrix} 3 \\ 2 \end{pmatrix}$ （または定数倍）が得られる。

2. 固有ベクトルを列とする行列 P を作る：
   $P = \begin{pmatrix} 3 & 3 \\ 5 & 2 \end{pmatrix}$

3. 対角行列 D を求める：
   $D = P^{-1}AP = \begin{pmatrix} -1 & 0 \\ 0 & 2 \end{pmatrix}$

### 4.3 行列のべき乗計算

対角化可能な行列 A = PDP^{-1} に対して、A の n 乗は以下のように計算できる：

A^n = (PDP^{-1})^n = PD^nP^{-1}

ここで、D^n は対角行列であるため、各対角成分の n 乗を計算するだけで得られる。

#### 例題4: 行列のべき乗

前問で対角化した行列 $A = \begin{pmatrix} 4 & -3 \\ 2 & -1 \end{pmatrix}$ に対して、$A^{10}$ を求めよ。

**解答**:

$A = PDP^{-1}$ と対角化されているので、
$A^{10} = PD^{10}P^{-1}$

$$D^{10} = \begin{pmatrix} (-1)^{10} & 0 \\ 0 & 2^{10} \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1024 \end{pmatrix}$$

$P = \begin{pmatrix} 3 & 3 \\ 5 & 2 \end{pmatrix}、P^{-1}$ を計算すると、
$P^{-1} = \frac{1}{det(P)} \begin{pmatrix} 2 & -3 \\ -5 & 3 \end{pmatrix} = \frac{1}{-9} \begin{pmatrix} 2 & -3 \\ -5 & 3 \end{pmatrix}$

従って、
$A^{10} = P \begin{pmatrix} 1 & 0 \\ 0 & 1024 \end{pmatrix} P^{-1}
      = \begin{pmatrix} 3 & 3 \\ 5 & 2 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1024 \end{pmatrix} \frac{1}{-9} \begin{pmatrix} 2 & -3 \\ -5 & 3 \end{pmatrix}$

行列の積を計算すると、最終的に $A^{10}$ が得られる。

### 4.4 数列の一般項の計算

行列を用いて表される漸化式から数列の一般項を求める問題もよく出題されます。

#### 例題5: 数列の一般項

漸化式 $a_{n+2} = 5a_{n+1} - 6a_n$ が与えられ、初期値 $a_0 = 2, a_1 = 5$ のとき、一般項$ a_n $を求めよ。

**解答**:

1. 行列形式で表現する：
   $\begin{pmatrix} a_{n+1} \\ a_{n+2} \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ 6 & 5 \end{pmatrix} \begin{pmatrix} a_n \\ a_{n+1} \end{pmatrix}$

   すなわち、$\begin{pmatrix} a_{n+1} \\ a_{n+2} \end{pmatrix} = A \begin{pmatrix} a_n \\ a_{n+1} \end{pmatrix} （A = \begin{pmatrix} 0 & 1 \\ 6 & 5 \end{pmatrix}）$

2. 初期値から、$\begin{pmatrix} a_n \\ a_{n+1} \end{pmatrix} = A^n \begin{pmatrix} a_0 \\ a_1 \end{pmatrix} = A^n \begin{pmatrix} 2 \\ 5 \end{pmatrix}$

3. 行列 $A$ を対角化する：
   特性方程式：
   $$ det(A - \lambda I) = det \begin{pmatrix} -\lambda & 1 \\ 6 & 5 - \lambda \end{pmatrix}
   = (-\lambda)(5-\lambda) - 1·6
   = -5\lambda + \lambda^2 - 6
   = \lambda^2 - 5\lambda - 6 = 0$$
   
   解くと、$\lambda = 6, -1$
   
   固有ベクトルを求め、対角化行列を構成：
   $P = \begin{pmatrix} 1 & 1 \\ 6 & -1 \end{pmatrix}、D = \begin{pmatrix} 6 & 0 \\ 0 & -1 \end{pmatrix}$

4. $A^n$ を計算：
   $A^n = PD^nP^{-1} = P \begin{pmatrix} 6^n & 0 \\ 0 & (-1)^n \end{pmatrix} P^{-1}$

5. 一般項 $a_n$ の導出：
   $\begin{pmatrix} a_n \\ a_{n+1} \end{pmatrix} = A^n \begin{pmatrix} a_0 \\ a_1 \end{pmatrix}$
   
   計算を進めると、$a_n = \frac{1}{7}(2·6^n + 5·(-1)^n)$ が得られる。

## 5. 試験問題サンプル

以下に、中間試験で出題されるような問題のサンプルを示します。

### 問題1: 固有値と固有ベクトルの計算

行列 A = \begin{pmatrix} 1 & 2 & 0 \\ 2 & 1 & 0 \\ 0 & 0 & 3 \end{pmatrix} の固有値と対応する固有ベクトルをすべて求めよ。

### 問題2: 対称行列の対角化

対称行列 B = \begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \end{pmatrix} を直交行列による対角化を行い、B = PDP^T を求めよ。

### 問題3: 行列のべき乗

行列 C = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix} に対して、C^{20} を求めよ。

### 問題4: 数列の一般項

漸化式 x_{n+2} = 6x_{n+1} - 9x_n が与えられ、初期値 x_0 = 1, x_1 = 3 のとき、一般項 x_n を求めよ。また、x_{100} の値を計算せよ。

## 6. Pythonによる実装と可視化

**注意**: 試験ではPythonのコードを書く問題は出題されませんが、理解を深めるために、以下に固有値・固有ベクトルの計算と行列のべき乗計算の実装例を示します。

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 行列の定義
A = np.array([[4, -3], [2, -1]])

# 固有値と固有ベクトルの計算
eigenvalues, eigenvectors = np.linalg.eig(A)

print("固有値:", eigenvalues)
print("固有ベクトル:\n", eigenvectors)

# 対角化の確認
D = np.diag(eigenvalues)
P = eigenvectors
P_inv = np.linalg.inv(P)

# P^{-1}AP = D を確認
result = np.matmul(np.matmul(P_inv, A), P)
print("P^{-1}AP =\n", np.round(result, 10))  # 丸め誤差を考慮

# 行列のべき乗計算
def matrix_power(A, n):
    """行列 A の n 乗を計算する"""
    eigenvalues, P = np.linalg.eig(A)
    P_inv = np.linalg.inv(P)
    D_n = np.diag(eigenvalues ** n)
    return np.matmul(np.matmul(P, D_n), P_inv)

# A^10 の計算
A_10 = matrix_power(A, 10)
print("A^10 =\n", np.round(A_10, 2))

# 直接計算との比較
A_10_direct = np.linalg.matrix_power(A, 10)
print("A^10 (直接計算) =\n", np.round(A_10_direct, 2))

# 固有ベクトルの可視化（2次元の場合）
plt.figure(figsize=(8, 6))
plt.grid(True)

# 元の基底ベクトルを描画
plt.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='e1')
plt.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='e2')

# 固有ベクトルを描画
for i, v in enumerate(eigenvectors.T):
    v_normalized = v / np.linalg.norm(v) * 2  # スケーリング
    plt.arrow(0, 0, v_normalized[0], v_normalized[1], 
              head_width=0.1, head_length=0.1, 
              fc='red', ec='red', 
              label=f'eigenvector {i+1} (\lambda={eigenvalues[i]:.1f})')

# 変換後の基底ベクトルを描画
e1_transformed = A @ np.array([1, 0])
e2_transformed = A @ np.array([0, 1])
plt.arrow(0, 0, e1_transformed[0], e1_transformed[1], 
          head_width=0.1, head_length=0.1, 
          fc='green', ec='green', 
          label='A·e1')
plt.arrow(0, 0, e2_transformed[0], e2_transformed[1], 
          head_width=0.1, head_length=0.1, 
          fc='green', ec='green', 
          label='A·e2')

plt.xlim(-3, 5)
plt.ylim(-3, 5)
plt.legend()
plt.title('Eigenvectors and Transformed Basis')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
```

## 7. 演習問題

### 基本問題

1. 行列 $A = \begin{pmatrix} 3 & 2 \\ 4 & -1 \end{pmatrix}$ の固有値とそれに対応する固有ベクトルを求めよ。

2. 行列 $B = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 2 & 1 \\ 0 & 1 & 2 \end{pmatrix}$ を対角化せよ。

3. 問題2で対角化した行列 $B$ について、$B^5$ を求めよ。

4. 漸化式 $a_{n+2} = 4a_{n+1} - 4a_n$ が与えられ、初期値 $a_0 = 1, a_1 = 2$ のとき、一般項 $a_n$ を求めよ。

### 応用問題

5. 行列 $$C = \begin{pmatrix} 4 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 4 \end{pmatrix}$$ の固有値と固有ベクトルを求め、対角化せよ。

6. 漸化式 $f_{n+2} = f_{n+1} + f_n$ が与えられ、初期値 $f_0 = 0, f_1 = 1$ のとき（フィボナッチ数列）、行列の対角化を用いて一般項 $f_n$ を求めよ。また、$f_{20}$ の値を計算せよ。

7. 行列 $D = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & -3 & 3 \end{pmatrix}$ について、$D^n$ の一般形を求めよ。

8. 健康データ分析に関する応用問題：ある保健所の週ごとの感染症患者数の推移が漸化式 $p_{n+2} = 1.5 p_{n+1} - 0.5 p_n$ で表されると仮定する。第1週と第2週の患者数がそれぞれ 10人、15人であったとき、第n週の患者数 $p_n$ の一般式を求め、第10週の患者数を予測せよ。また、この感染症の拡大傾向について考察せよ。

## 8. よくある質問と解答

### Q1: 固有値が重根の場合の固有ベクトルはどのように求めればよいですか？

A1: 固有値が代数的重複度 k の重根の場合、対応する固有ベクトルの幾何的重複度（線形独立な固有ベクトルの数）は k 以下となります。(A - \lambdaI)v = 0 を解いて、線形独立な解ベクトルをすべて求めます。もし幾何的重複度が代数的重複度より小さい場合、行列は対角化できない可能性があります。このような場合はジョルダン標準形を考える必要があります。

### Q2: 対角化可能の条件は何ですか？

A2: n×n行列 A が対角化可能であるための必要十分条件は、A の固有ベクトルが n 個の線形独立なベクトルを形成することです。これは、各固有値 \lambda について、その幾何的重複度（対応する線形独立な固有ベクトルの数）が代数的重複度（特性方程式の根としての重複度）と等しいことを意味します。

### Q3: 対称行列の対角化について特別な性質はありますか？

A3: 実対称行列 A は常に対角化可能であり、その固有値はすべて実数です。また、異なる固有値に対応する固有ベクトルは互いに直交しています。したがって、A = PDP^T となる直交行列 P（P^T = P^{-1}）が存在します。これをスペクトル分解と呼びます。

### Q4: 行列のべき乗計算において対角化を利用する利点は何ですか？

A4: 対角化を利用すると、A^n = PD^nP^{-1} と表現できるため、D が対角行列であることを利用して D^n を簡単に計算できます。これにより、n が大きくなっても効率的に A^n を求めることができます。直接 A を n 回掛け合わせる方法では計算量が O(n) になりますが、対角化を利用すると O(log n) に削減できます。

### Q5: 数列の漸化式から一般項を求める方法はほかにもありますか？

A5: はい、特性方程式を立てて解く方法もあります。k 階の線形漸化式 a_{n+k} + c_{k-1}a_{n+k-1} + ... + c_0 a_n = 0 に対して、特性方程式 r^k + c_{k-1}r^{k-1} + ... + c_0 = 0 を解き、その解を用いて一般項を表現します。ただし、特性方程式に重根がある場合は注意が必要です。

## 9. 試験に向けたアドバイス

1. **基本概念の理解**: 固有値・固有ベクトルの定義と性質を正確に理解しましょう。

2. **計算練習**: 2×2行列、3×3行列の固有値・固有ベクトルの計算を何度も練習しましょう。

3. **特性方程式の立て方**: det(A - \lambdaI) = 0 の計算を効率的に行う方法を習得しましょう。

4. **対角化の手順**: 対角化の各ステップを確実に実行できるようにしましょう。

5. **チェックポイント**: 対角化の結果が正しいかを P^{-1}AP = D の計算で確認する習慣をつけましょう。

6. **応用問題の対策**: 行列のべき乗計算と数列の一般項導出の問題を解く練習をしましょう。

7. **計算ミスを防ぐ**: 行列式や逆行列の計算では符号のミスに注意しましょう。

8. **時間配分**: 試験中は問題ごとに適切な時間配分を心がけましょう。すべての問題に取り組めるようにしましょう。

最後に、授業で扱った例題や演習問題を再度解いてみることで、理解度を確認し、苦手な部分を特定して重点的に復習することをお勧めします。