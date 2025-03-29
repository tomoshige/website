# 線形代数学 I 第4回講義ノート：行列の定義・行列の和・行列のスカラー倍

## 1. 講義情報と予習ガイド

- **講義回**: 第4回
- **関連項目**: ベクトル演算（第2-3回の内容）
- **予習内容**: ベクトルの和とスカラー倍、ベクトルの内積の復習

## 2. 学習目標

この講義では、以下の能力を身につけることを目標とします：

1. 行列の定義を理解し、適切に表記できる
2. 行列の和を正確に計算できる
3. 行列のスカラー倍を正確に計算できる
4. 行列とベクトルの関係性を理解できる
5. Google Colabを用いて行列計算を実行できる

## 3. 基本概念

### 3.1 行列の定義

> **定義**: 行列（Matrix）とは、数や記号を縦と横に矩形状に配置したものです。$m$行$n$列の行列$A$は次のように表されます：
> 
> $$A = \begin{pmatrix}
> a_{11} & a_{12} & \cdots & a_{1n} \\
> a_{21} & a_{22} & \cdots & a_{2n} \\
> \vdots & \vdots & \ddots & \vdots \\
> a_{m1} & a_{m2} & \cdots & a_{mn}
> \end{pmatrix}$$
> 
> ここで、$a_{ij}$は$i$行$j$列目の要素を表します。

**サイズ**: 行列のサイズは行数×列数で表し、$m \times n$行列などと呼びます。

**例**:

$$A = \begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{pmatrix}$$

この行列$A$は$2 \times 3$行列（2行3列の行列）です。

### 3.2 特殊な形状の行列

1. **正方行列（Square Matrix）**: 行数と列数が等しい行列（$m = n$）
   
   例: $B = \begin{pmatrix}
   1 & 2 \\
   3 & 4
   \end{pmatrix}$ は$2 \times 2$の正方行列

2. **行ベクトル（Row Vector）**: 1行$n$列の行列
   
   例: $r = \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}$ は$1 \times 3$の行ベクトル

3. **列ベクトル（Column Vector）**: $m$行1列の行列
   
   例: $c = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ は$3 \times 1$の列ベクトル

### 3.3 行列の表記法

行列は通常、大文字のアルファベット（$A$, $B$, $C$など）で表します。行列の要素は小文字の添え字付きの文字（$a_{ij}$など）で表します。

- $A$: 行列全体
- $a_{ij}$: 行列$A$の$i$行$j$列目の要素
- $A_{i,j}$: 行列$A$の$i$行$j$列目の要素（別表記）

## 4. 理論と手法

### 4.1 行列の和

> **定義**: 同じサイズの行列$A$と$B$の和$A + B$は、対応する要素同士を足し合わせた行列です：
> 
> $$(A + B)_{ij} = a_{ij} + b_{ij}$$

**注意点**: 異なるサイズの行列同士は足し合わせることができません。

**例**:

$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$$

$$A + B = \begin{pmatrix} 1+5 & 2+6 \\ 3+7 & 4+8 \end{pmatrix} = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix}$$

### 4.2 行列の和の性質

行列の和は以下の性質を持ちます：

1. **交換法則**: $A + B = B + A$
2. **結合法則**: $(A + B) + C = A + (B + C)$
3. **単位元**: 零行列 $O$ について $A + O = A$
4. **逆元**: $-A$ について $A + (-A) = O$

### 4.3 行列のスカラー倍

> **定義**: 行列$A$のスカラー倍$cA$は、$A$の各要素に$c$を掛けた行列です：
> 
> $$(cA)_{ij} = c \cdot a_{ij}$$

**例**:

$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad c = 3$$

$$cA = 3 \cdot \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 3 \cdot 1 & 3 \cdot 2 \\ 3 \cdot 3 & 3 \cdot 4 \end{pmatrix} = \begin{pmatrix} 3 & 6 \\ 9 & 12 \end{pmatrix}$$

### 4.4 行列のスカラー倍の性質

行列のスカラー倍は以下の性質を持ちます：

1. $c(A + B) = cA + cB$
2. $(c + d)A = cA + dA$
3. $c(dA) = (cd)A$
4. $1 \cdot A = A$

### 4.5 行列とベクトルの関係

行列は「ベクトルを列に並べたもの」と見ることができます。例えば、$n$次元の列ベクトル$\vec{v}_1, \vec{v}_2, ..., \vec{v}_m$を考えると、それらを横に並べた行列$A$は：

$$A = \begin{pmatrix} | & | & & | \\ \vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_m \\ | & | & & | \end{pmatrix}$$

同様に、行列を「行ベクトルを縦に積み重ねたもの」と見ることもできます。

**例**:

列ベクトル $\vec{v}_1 = \begin{pmatrix} 1 \\ 3 \end{pmatrix}$, $\vec{v}_2 = \begin{pmatrix} 2 \\ 4 \end{pmatrix}$ を並べると、

$$A = \begin{pmatrix} | & | \\ \vec{v}_1 & \vec{v}_2 \\ | & | \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$$

## 5. Pythonによる実装と可視化

### 5.1 NumPy を用いた行列の操作

Python の NumPy ライブラリを使用して行列の基本操作を実行する方法を見ていきましょう。

```python
import numpy as np
import matplotlib.pyplot as plt

# 行列の定義
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("行列 A:")
print(A)
print("\n行列 B:")
print(B)

# 行列の和
C = A + B
print("\nA + B =")
print(C)

# 行列のスカラー倍
scalar = 3
D = scalar * A
print(f"\n{scalar} × A =")
print(D)

# 行列のサイズ
print(f"\n行列 A のサイズ: {A.shape}")
```

### 5.2 行列の可視化

```python
def plot_matrix(matrix, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label='値')
    plt.title(title)
    
    # 値を表示
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            plt.text(j, i, f'{matrix[i, j]}', 
                     ha='center', va='center', color='white')
    
    plt.tight_layout()
    plt.show()

# 行列の可視化
plot_matrix(A, '行列 A')
plot_matrix(B, '行列 B')
plot_matrix(C, '行列 A + B')
plot_matrix(D, f'行列 {scalar} × A')
```

### 5.3 行列とベクトルの関係の可視化

```python
# ベクトルから行列を構成
v1 = np.array([1, 3])
v2 = np.array([2, 4])

# 列ベクトルとして結合
A_from_columns = np.column_stack((v1, v2))
print("列ベクトルから構成した行列:")
print(A_from_columns)

# 行ベクトルとして結合
row1 = np.array([1, 2])
row2 = np.array([3, 4])
A_from_rows = np.vstack((row1, row2))
print("\n行ベクトルから構成した行列:")
print(A_from_rows)

# 可視化
plt.figure(figsize=(10, 5))

# v1, v2 を別々に描画
plt.subplot(1, 2, 1)
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.grid()
plt.title('ベクトル v1, v2')
plt.legend()

# 行列 A の列ベクトル表現
plt.subplot(1, 2, 2)
plt.quiver(0, 0, A[0, 0], A[1, 0], angles='xy', scale_units='xy', scale=1, color='r', label='A[:,0]')
plt.quiver(0, 0, A[0, 1], A[1, 1], angles='xy', scale_units='xy', scale=1, color='b', label='A[:,1]')
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.grid()
plt.title('行列 A の列ベクトル')
plt.legend()

plt.tight_layout()
plt.show()
```

## 6. 演習問題

### 6.1 基本問題

1. 次の行列のサイズを答えなさい。
   
   (a) $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$
   
   (b) $B = \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}$
   
   (c) $C = \begin{pmatrix} 13 & 14 & 15 & 16 \end{pmatrix}$

2. 次の行列の和を求めなさい。
   
   $A = \begin{pmatrix} 2 & 0 \\ -1 & 3 \end{pmatrix}, \quad B = \begin{pmatrix} 4 & -2 \\ 1 & 5 \end{pmatrix}$

3. 次の行列のスカラー倍を求めなさい。
   
   $A = \begin{pmatrix} 1 & -2 & 3 \\ 0 & 4 & -5 \end{pmatrix}, \quad c = -2$

4. 次の計算をせよ。
   
   $2A - 3B$, ただし $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$

### 6.2 応用問題

1. 列ベクトル $\vec{v}_1 = \begin{pmatrix} 2 \\ 1 \\ 3 \end{pmatrix}$, $\vec{v}_2 = \begin{pmatrix} 0 \\ -1 \\ 4 \end{pmatrix}$, $\vec{v}_3 = \begin{pmatrix} -2 \\ 5 \\ 1 \end{pmatrix}$ から構成される行列 $A$ を書きなさい。また、$2\vec{v}_1 - \vec{v}_2 + 3\vec{v}_3$ を計算し、これを行列 $A$ と適切なベクトル $\vec{x}$ を用いて $A\vec{x}$ と表しなさい。

2. 以下の患者データ行列 $P$ があります：
   
   $$P = \begin{pmatrix} 
   120 & 80 & 90 \\
   130 & 85 & 110 \\
   125 & 75 & 95 \\
   140 & 90 & 120
   \end{pmatrix}$$
   
   各行は患者、各列は異なる健康指標（例：血圧、体重、コレステロール値）を表しています。全ての患者データに対して、標準化のために以下の操作を行います：
   
   - 血圧（1列目）から100を引く
   - 体重（2列目）から70を引く
   - コレステロール値（3列目）から80を引く
   
   この操作を行列の計算として表現し、結果の行列を求めなさい。

3. Google Colabを使って、以下のヘルスデータに関する行列演算を実装しなさい：
   - 患者10人×健康指標5つのランダムな行列データを生成する
   - 各健康指標の平均値を求める
   - すべての値を正規化する（各列の平均が0、標準偏差が1になるように）
   - 結果を可視化する（ヒートマップ）

## 7. よくある質問と解答

**Q1: 行列とベクトルの違いは何ですか？**

A1: ベクトルは行列の特殊な場合と考えることができます。列ベクトルは$n \times 1$行列、行ベクトルは$1 \times m$行列です。行列はベクトルを複数並べたものとも見ることができます。

**Q2: 異なるサイズの行列同士を足し合わせることはできますか？**

A2: できません。行列の加算は、対応する要素同士を足し合わせる操作であるため、行数と列数が一致している必要があります。

**Q3: 行列の和やスカラー倍がデータサイエンスでどのように使われますか？**

A3: 行列の和やスカラー倍は、データの正規化、特徴量のスケーリング、複数のデータセットの結合、時系列データの移動平均の計算など、様々なデータ前処理や分析手法で使用されます。また、機械学習アルゴリズムの内部計算（勾配降下法など）でも重要な役割を果たします。

**Q4: 行列の要素を並べる順序は重要ですか？**

A4: 非常に重要です。行列では要素の位置（行番号と列番号）が情報を持っています。行と列を入れ替えると、全く異なる行列になります。特に、データサイエンスでは行は通常サンプル（観測値）、列は特徴量（変数）を表すことが多いため、その構造を保つことが重要です。

**Q5: なぜPythonのNumPyを使って行列計算をするのですか？**

A5: NumPyは行列計算に最適化された効率的なライブラリです。大規模な行列でも高速に計算でき、また豊富な関数が用意されているため、データサイエンスの作業を効率化できます。また、正確な数値計算が保証されており、行列の様々な操作や分解を簡単に行うことができます。

## 8. まとめ

本講義では、行列の基本概念、表記法、行列の和とスカラー倍の計算方法について学びました。また、行列とベクトルの関係性についても考察しました。これらの概念はデータサイエンスの基礎となる重要な道具です。特に、データの表現や変換、モデリングにおいて、行列演算は中心的な役割を果たします。次回の講義では、行列の積について学習します。これは線形変換やデータの変換において非常に重要な操作です。