# 線形代数学 I - 第21回講義ノート

## 講義情報と予習ガイド

**講義回**: 第21回  
**テーマ**: 行列式の定義と基本性質  
**関連項目**: 行列式、多重線形性、交代性、行列の積と行列式  
**予習内容**: 行列の基本概念、行基本変形の復習

## 1. 学習目標

本講義では以下の目標達成を目指します：

1. 行列式の定義を理解し、その意味を把握する
2. 2次・3次の行列式の計算方法を習得する
3. 行列式の基本性質（多重線形性、交代性）を理解する
4. 行列式と逆行列の関係性を理解する
5. 行列式の応用例を把握する

## 2. 基本概念：行列式とは

### 2.1 行列式の導入

行列式（determinant）は正方行列に対して定義される数値であり、線形代数学において極めて重要な概念です。行列式は以下のような数学的・幾何学的意味を持ちます：

- 行列が表す線形変換による単位体積の変化率
- 逆行列が存在するかどうかの判定基準
- 連立1次方程式の解の存在条件

> **定義**: n次正方行列 $A$ の行列式は $\det(A)$ または $|A|$ と表記され、行列 $A$ から計算される一つのスカラー値です。

### 2.2 2次行列の行列式

2次正方行列 $A = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$ の行列式は次のように定義されます：

> **2次行列式**: $\det(A) = a_{11}a_{22} - a_{12}a_{21}$

これは「主対角線の積から副対角線の積を引く」と覚えることができます。

**例題 2.1**: 行列 $A = \begin{pmatrix} 3 & 5 \\ 2 & 7 \end{pmatrix}$ の行列式を求めよ。

**解答**:
$\det(A) = 3 \times 7 - 5 \times 2 = 21 - 10 = 11$

### 2.3 3次行列の行列式

3次行列の行列式を求める方法としては、サラスの方法（Sarrus's rule）があります：

> **3次行列式（サラスの方法）**: 
> $\det\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix} = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31} - a_{11}a_{23}a_{32} - a_{12}a_{21}a_{33}$

これは次のような図式で覚えることができます：

```
+ | a11 a12 a13 | a11 a12 |
  | a21 a22 a23 | a21 a22 |
  | a31 a32 a33 | a31 a32 |

- | a13 a12 a11 | a13 a12 |
  | a23 a22 a21 | a23 a22 |
  | a33 a32 a31 | a33 a32 |
```

**例題 2.2**: 行列 $B = \begin{pmatrix} 2 & 0 & 1 \\ 3 & 1 & 2 \\ 1 & 2 & 0 \end{pmatrix}$ の行列式を求めよ。

**解答**:
$\det(B) = 2 \times 1 \times 0 + 0 \times 2 \times 1 + 1 \times 3 \times 2 - 1 \times 1 \times 1 - 2 \times 2 \times 1 - 0 \times 3 \times 0$
$= 0 + 0 + 6 - 1 - 4 - 0 = 1$

## 3. 理論と手法：行列式の基本性質

### 3.1 単位行列の行列式

> **性質 1**: 単位行列 $I_n$ の行列式は 1 である。
> $\det(I_n) = 1$

**証明の概略**:
2次単位行列 $I_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ の場合：
$\det(I_2) = 1 \times 1 - 0 \times 0 = 1$

3次単位行列 $I_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$ の場合：
$\det(I_3) = 1 \times 1 \times 1 + 0 \times 0 \times 0 + 0 \times 0 \times 0 - 0 \times 1 \times 0 - 1 \times 0 \times 0 - 0 \times 0 \times 1 = 1$

### 3.2 行列式の多重線形性

行列式は各行（または各列）について線形である性質を持ちます。

> **性質 2（行に関する多重線形性）**:
> 1. ある行のすべての要素に同じ数 $c$ を掛けると、行列式は $c$ 倍になる。
> 2. ある行を2つの行の和として表せる場合、行列式は2つの行列式の和として表せる。

例えば、2次行列で見ると：

$\det\begin{pmatrix} ca_{11} & ca_{12} \\ a_{21} & a_{22} \end{pmatrix} = c \cdot \det\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$

$\det\begin{pmatrix} a_{11}+b_{11} & a_{12}+b_{12} \\ a_{21} & a_{22} \end{pmatrix} = \det\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix} + \det\begin{pmatrix} b_{11} & b_{12} \\ a_{21} & a_{22} \end{pmatrix}$

**例題 3.1**: 行列 $C = \begin{pmatrix} 6 & 10 \\ 2 & 7 \end{pmatrix}$ の行列式を多重線形性を用いて計算せよ。

**解答**:
$C$ の第1行は $A = \begin{pmatrix} 3 & 5 \\ 2 & 7 \end{pmatrix}$ の第1行の2倍であるため、
$\det(C) = 2 \cdot \det(A) = 2 \times 11 = 22$

### 3.3 行列式の交代性

> **性質 3（交代性）**: 行列の2つの行（または列）を入れ替えると、行列式の符号が反転する。

これは、2つの行が同じである場合、行列式は0になることを意味します。

**例題 3.2**: 行列 $D = \begin{pmatrix} 2 & 7 \\ 3 & 5 \end{pmatrix}$ の行列式を求め、行を入れ替えた行列 $E = \begin{pmatrix} 3 & 5 \\ 2 & 7 \end{pmatrix}$ の行列式との関係を確認せよ。

**解答**:
$\det(D) = 2 \times 5 - 7 \times 3 = 10 - 21 = -11$
$\det(E) = 3 \times 7 - 5 \times 2 = 21 - 10 = 11$

確かに $\det(E) = -\det(D)$ が成り立っています。

### 3.4 行列の積と行列式

> **性質 4**: 2つの正方行列 $A$ と $B$ の積の行列式は、それぞれの行列式の積に等しい。
> $\det(AB) = \det(A) \cdot \det(B)$

**例題 3.4**: 行列 $A = \begin{pmatrix} 2 & 1 \\ 3 & 2 \end{pmatrix}$ と $B = \begin{pmatrix} 3 & 4 \\ 1 & 2 \end{pmatrix}$ に対して、$\det(AB) = \det(A) \cdot \det(B)$ を確認せよ。

**解答**:
$\det(A) = 2 \times 2 - 1 \times 3 = 4 - 3 = 1$
$\det(B) = 3 \times 2 - 4 \times 1 = 6 - 4 = 2$
$\det(A) \cdot \det(B) = 1 \times 2 = 2$

一方、
$AB = \begin{pmatrix} 2 & 1 \\ 3 & 2 \end{pmatrix} \begin{pmatrix} 3 & 4 \\ 1 & 2 \end{pmatrix} = \begin{pmatrix} 2 \times 3 + 1 \times 1 & 2 \times 4 + 1 \times 2 \\ 3 \times 3 + 2 \times 1 & 3 \times 4 + 2 \times 2 \end{pmatrix} = \begin{pmatrix} 7 & 10 \\ 11 & 16 \end{pmatrix}$

$\det(AB) = 7 \times 16 - 10 \times 11 = 112 - 110 = 2$

確かに $\det(AB) = \det(A) \cdot \det(B)$ が成り立っています。

### 3.5 逆行列の存在と行列式

> **性質 5**: 正方行列 $A$ が逆行列を持つための必要十分条件は $\det(A) \neq 0$ である。

**例題 3.5**: 行列 $F = \begin{pmatrix} 2 & 4 \\ 1 & 2 \end{pmatrix}$ に逆行列が存在するか判定せよ。

**解答**:
$\det(F) = 2 \times 2 - 4 \times 1 = 4 - 4 = 0$
行列式が0であるため、行列 $F$ は逆行列を持ちません。

## 4. Pythonによる実装と可視化

### 4.1 NumPyを使った行列式の計算

```python
import numpy as np
import matplotlib.pyplot as plt

# 2次行列の行列式
A = np.array([[3, 5], [2, 7]])
det_A = np.linalg.det(A)
print(f"行列A:\n{A}")
print(f"行列Aの行列式: {det_A}")  # 結果: 11.0

# 3次行列の行列式
B = np.array([[2, 0, 1], [3, 1, 2], [1, 2, 0]])
det_B = np.linalg.det(B)
print(f"\n行列B:\n{B}")
print(f"行列Bの行列式: {det_B}")  # 結果: 1.0

# 行列式の多重線形性の検証
C = np.array([[6, 10], [2, 7]])  # Aの第1行を2倍した行列
det_C = np.linalg.det(C)
print(f"\n行列C:\n{C}")
print(f"行列Cの行列式: {det_C}")  # 結果: 22.0
print(f"2×det(A): {2*det_A}")     # 結果: 22.0

# 行列式の交代性の検証
D = np.array([[2, 7], [3, 5]])
E = np.array([[3, 5], [2, 7]])  # Dの行を入れ替えた行列
det_D = np.linalg.det(D)
det_E = np.linalg.det(E)
print(f"\n行列D:\n{D}")
print(f"行列Dの行列式: {det_D}")  # 結果: -11.0
print(f"\n行列E:\n{E}")
print(f"行列Eの行列式: {det_E}")  # 結果: 11.0
print(f"det(E)とdet(D)の関係: {det_E} = {-det_D}")  # det(E) = -det(D) を確認

# 行列の積と行列式の関係
A2 = np.array([[2, 1], [3, 2]])
B2 = np.array([[3, 4], [1, 2]])
AB = np.dot(A2, B2)
det_A2 = np.linalg.det(A2)
det_B2 = np.linalg.det(B2)
det_AB = np.linalg.det(AB)
print(f"\n行列A2:\n{A2}")
print(f"行列A2の行列式: {det_A2}")  # 結果: 1.0
print(f"\n行列B2:\n{B2}")
print(f"行列B2の行列式: {det_B2}")  # 結果: 2.0
print(f"\n行列A2B2:\n{AB}")
print(f"行列A2B2の行列式: {det_AB}")  # 結果: 2.0
print(f"det(A2)×det(B2): {det_A2*det_B2}")  # 結果: 2.0
```

### 4.2 行列式の幾何学的意味の可視化

行列式は、単位正方形（または単位立方体）が線形変換後にどれだけ面積（または体積）が変化するかを表します。

```python
def plot_transformation(A, title="線形変換"):
    # 単位正方形の頂点
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    
    # 変換後の正方形
    transformed_square = np.dot(square, A.T)
    
    # 元の正方形と変換後の正方形をプロット
    plt.figure(figsize=(10, 5))
    
    # 元の正方形
    plt.subplot(1, 2, 1)
    plt.plot(square[:, 0], square[:, 1], 'b-')
    plt.fill(square[:, 0], square[:, 1], 'lightblue', alpha=0.5)
    plt.grid(True)
    plt.axis('equal')
    plt.title("元の単位正方形")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    
    # 変換後の正方形
    plt.subplot(1, 2, 2)
    plt.plot(transformed_square[:, 0], transformed_square[:, 1], 'r-')
    plt.fill(transformed_square[:, 0], transformed_square[:, 1], 'salmon', alpha=0.5)
    plt.grid(True)
    plt.axis('equal')
    plt.title(f"{title}\ndet(A) = {np.linalg.det(A):.2f}")
    
    # 軸の範囲を適当に設定
    max_val = max(np.max(np.abs(transformed_square)), 1.5)
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    
    plt.tight_layout()
    plt.show()

# 例1: 面積が増加する変換
A1 = np.array([[2, 0], [0, 2]])  # 2倍に拡大
plot_transformation(A1, "2倍に拡大する変換")

# 例2: 面積は同じだが形が変わる変換
A2 = np.array([[0, 1], [-1, 0]])  # 90度回転
plot_transformation(A2, "90度回転する変換")

# 例3: 面積が減少する変換
A3 = np.array([[0.5, 0], [0, 0.5]])  # 半分に縮小
plot_transformation(A3, "半分に縮小する変換")

# 例4: 面積が0になる変換（行列式 = 0）
A4 = np.array([[1, 1], [2, 2]])  # ランク落ち
plot_transformation(A4, "1次元に押しつぶす変換")

# 例5: 面積が負になる変換（行列式 < 0）
A5 = np.array([[0, 1], [1, 0]])  # x軸とy軸を入れ替え
plot_transformation(A5, "x軸とy軸を入れ替える変換")
```

## 5. 演習問題

### 5.1 基本問題

1. 次の行列の行列式を求めよ。
   (a) $\begin{pmatrix} 4 & 3 \\ 2 & 5 \end{pmatrix}$
   (b) $\begin{pmatrix} 2 & 0 & 3 \\ 1 & 4 & 2 \\ 0 & 1 & 5 \end{pmatrix}$
   
2. 次の行列の行列式を計算し、逆行列が存在するか判定せよ。
   (a) $\begin{pmatrix} 2 & 6 \\ 1 & 3 \end{pmatrix}$
   (b) $\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$
   
3. 行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ に対して、$\det(A^T) = \det(A)$ を示せ。

4. $\det(A) = 3$ かつ $\det(B) = 4$ のとき、$\det(AB)$ と $\det(2A)$ を求めよ。

### 5.2 応用問題

5. 行列 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ に対して、$\det(A^2) = (\det(A))^2$ を示せ。

6. 行列 $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & k \end{pmatrix}$ において、$\det(A) = 0$ となるような $k$ の値を求めよ。

7. 健康データ分析の応用例：患者の血圧、心拍数、体温の3つの指標について、正常値からの偏差を表す行列 $\begin{pmatrix} x_1 & y_1 & z_1 \\ x_2 & y_2 & z_2 \\ x_3 & y_3 & z_3 \end{pmatrix}$ の行列式が0に近い値をとるとき、これら3つの指標間に強い相関関係があることを意味します。3人の患者データ $\begin{pmatrix} 10 & 5 & 3 \\ 20 & 10 & 6 \\ 30 & 15 & 9 \end{pmatrix}$ の行列式を計算し、これらの指標間の関係性について考察せよ。

## 6. よくある質問と解答

### Q1: 行列式はなぜ重要なのですか？
**A1**: 行列式はデータサイエンスを含む様々な分野で重要な役割を果たします：
- 連立方程式の解の存在条件を調べる
- 行列の逆行列が存在するかを判定する
- 線形変換による面積や体積の変化率を表す
- 固有値問題や主成分分析など、多くの数学的手法の基礎となる

### Q2: 行列式が0であることは何を意味しますか？
**A2**: 行列式が0であることは、その行列が「特異（singular）」であることを意味します。つまり：
- 逆行列が存在しない
- 対応する線形変換が、より低い次元への射影を伴う（情報の損失がある）
- 対応する連立方程式が一意の解を持たない（解なしか無数の解が存在）
- 行列の行（または列）が線形従属である

### Q3: 大きな行列の行列式を効率的に計算する方法はありますか？
**A3**: はい、以下の方法があります：
- 基本変形を利用して上三角行列に変形し、対角成分の積を求める
- 余因子展開（次回の講義で学習します）
- 数値計算ライブラリ（NumPyなど）を使用する

### Q4: 行列式の幾何学的な意味をもう少し詳しく教えてください。
**A4**: 行列式は線形変換による面積や体積の変化率を表します：
- 2×2行列の場合、単位正方形が変換後にどれだけの面積になるかを示す
- 行列式が正なら、変換は向きを保存する
- 行列式が負なら、変換は向きを反転させる
- 行列式の絶対値が大きいほど、変換による拡大率が大きい
- 行列式が0なら、変換後の図形は低次元（線や点）に潰れる

### Q5: 統計学や機械学習における行列式の応用例はありますか？
**A5**: はい、いくつかの重要な応用例があります：
- 多変量正規分布の確率密度関数に行列式が登場する
- 主成分分析での共分散行列の固有値問題に関連する
- 線形回帰の正規方程式の一意解の存在条件の検証
- マハラノビス距離の計算に逆行列（および行列式）が使われる