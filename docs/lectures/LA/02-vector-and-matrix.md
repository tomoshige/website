# 線形代数学 I 第2回 講義ノート

## 1. 講義情報と予習ガイド

- **講義回**: 第2回
- **テーマ**: ベクトルの定義と基本操作①
- **関連項目**: ベクトル空間、ベクトル演算、幾何学的解釈
- **予習内容**: 高校数学の座標平面と空間座標の基礎知識を復習しておくこと

## 2. 学習目標

1. 実ベクトルの定義と表記方法を理解する
2. ベクトルの加法とスカラー倍の演算規則を習得する
3. ベクトル演算の幾何学的意味を理解する
4. Pythonを用いてベクトル演算を実装できるようになる

## 3. 基本概念

### 3.1 実ベクトルの定義

> **定義**: $n$次元実ベクトルとは、$n$個の実数を縦に並べたもので、以下のように表記される：
> 
> $$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$
> 
> ここで $v_1, v_2, \ldots, v_n$ は実数である。

このようなベクトルの集合を $\mathbb{R}^n$ と表し、$n$次元実ベクトル空間と呼びます。特に重要な例として：

- $\mathbb{R}^2$: 2次元実ベクトル空間（平面上のベクトル）
- $\mathbb{R}^3$: 3次元実ベクトル空間（空間上のベクトル）

データサイエンスでは、データの各サンプルを1つのベクトルとして扱うことが一般的です。例えば、ある人の「年齢、身長、体重、血圧」といった4つの健康指標は4次元ベクトルとして表現できます。

### 3.2 実ベクトルの表し方

ベクトルは以下のような様々な方法で表記されます：

1. **成分表示**:
   $$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

2. **太字記号**:
   $\mathbf{v}$ や $\vec{v}$

3. **列ベクトル**（標準的な表記）:
   $$\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$$
   ここで $T$ は転置を表し、列ベクトルであることを明示します。

### 3.3 零ベクトル

> **定義**: 零ベクトル $\mathbf{0}$ とは、すべての成分が0であるベクトルである：
> 
> $$\mathbf{0} = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}$$

零ベクトルは加法演算の単位元としての役割を持ちます。

## 4. 理論と手法

### 4.1 実ベクトルの和

> **定義**: 2つの同じ次元のベクトル $\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$ と $\mathbf{w} = (w_1, w_2, \ldots, w_n)^T$ に対して、その和 $\mathbf{v} + \mathbf{w}$ は次のように定義される：
> 
> $$\mathbf{v} + \mathbf{w} = \begin{pmatrix} v_1 + w_1 \\ v_2 + w_2 \\ \vdots \\ v_n + w_n \end{pmatrix}$$

つまり、ベクトルの加法は「対応する成分同士を足す」操作です。

**例**: $\mathbf{v} = \begin{pmatrix} 1 \\ 3 \\ 5 \end{pmatrix}$ と $\mathbf{w} = \begin{pmatrix} 2 \\ -1 \\ 4 \end{pmatrix}$ のとき

$$\mathbf{v} + \mathbf{w} = \begin{pmatrix} 1 + 2 \\ 3 + (-1) \\ 5 + 4 \end{pmatrix} = \begin{pmatrix} 3 \\ 2 \\ 9 \end{pmatrix}$$

### 4.2 ベクトル加法の性質

ベクトルの加法には以下の性質があります：

1. **結合法則**: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
2. **交換法則**: $\mathbf{v} + \mathbf{w} = \mathbf{w} + \mathbf{v}$
3. **単位元**: 任意のベクトル $\mathbf{v}$ に対して $\mathbf{v} + \mathbf{0} = \mathbf{v}$
4. **逆元**: 任意のベクトル $\mathbf{v}$ に対して $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ となる $-\mathbf{v}$ が存在する

これらの性質により、ベクトルの集合は加法に関して「可換群」となります。

### 4.3 実ベクトルのスカラー倍

> **定義**: ベクトル $\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$ と実数 $\alpha$ に対して、$\mathbf{v}$ のスカラー倍 $\alpha\mathbf{v}$ は次のように定義される：
> 
> $$\alpha\mathbf{v} = \begin{pmatrix} \alpha v_1 \\ \alpha v_2 \\ \vdots \\ \alpha v_n \end{pmatrix}$$

つまり、ベクトルのスカラー倍は「すべての成分に同じ実数をかける」操作です。

**例**: $\mathbf{v} = \begin{pmatrix} 2 \\ 3 \\ -1 \end{pmatrix}$ のとき、$3\mathbf{v}$ は

$$3\mathbf{v} = \begin{pmatrix} 3 \cdot 2 \\ 3 \cdot 3 \\ 3 \cdot (-1) \end{pmatrix} = \begin{pmatrix} 6 \\ 9 \\ -3 \end{pmatrix}$$

### 4.4 スカラー倍の性質

スカラー倍には以下の性質があります：

1. $1\mathbf{v} = \mathbf{v}$
2. $\alpha(\beta\mathbf{v}) = (\alpha\beta)\mathbf{v}$
3. $(\alpha + \beta)\mathbf{v} = \alpha\mathbf{v} + \beta\mathbf{v}$
4. $\alpha(\mathbf{v} + \mathbf{w}) = \alpha\mathbf{v} + \alpha\mathbf{w}$

これらの加法とスカラー倍の性質により、ベクトルの集合は「ベクトル空間」となります。

### 4.5 ベクトルの幾何学的解釈

#### 二次元平面上のベクトル

$\mathbb{R}^2$ のベクトル $\mathbf{v} = (v_1, v_2)^T$ は、原点 $(0,0)$ から点 $(v_1, v_2)$ へ向かう矢印として幾何学的に解釈できます。

#### ベクトルの和の幾何学的意味

2つのベクトル $\mathbf{v}$ と $\mathbf{w}$ の和 $\mathbf{v} + \mathbf{w}$ は、ベクトル $\mathbf{v}$ の終点から $\mathbf{w}$ と平行なベクトルを描いたときの終点へ向かうベクトルです。これは「平行四辺形の法則」とも呼ばれます。

#### スカラー倍の幾何学的意味

ベクトル $\mathbf{v}$ のスカラー倍 $\alpha\mathbf{v}$ は：

- $\alpha > 0$ のとき：$\mathbf{v}$ と同じ方向で、長さが $|\alpha|$ 倍
- $\alpha < 0$ のとき：$\mathbf{v}$ と反対方向で、長さが $|\alpha|$ 倍
- $\alpha = 0$ のとき：零ベクトル

## 5. Pythonによる実装と可視化

### 5.1 NumPyを用いたベクトル演算

```python
import numpy as np
import matplotlib.pyplot as plt

# ベクトルの定義
v = np.array([3, 2])
w = np.array([1, 4])

# ベクトルの和
v_plus_w = v + w
print(f"v + w = {v_plus_w}")

# スカラー倍
alpha = 2.5
alpha_v = alpha * v
print(f"{alpha} * v = {alpha_v}")

# マイナスのスカラー倍
beta = -1.5
beta_w = beta * w
print(f"{beta} * w = {beta_w}")
```

### 5.2 ベクトルの可視化

```python
# ベクトルを描画する関数
def plot_vector(vector, origin=[0, 0], color='b', label=None):
    plt.arrow(origin[0], origin[1], vector[0], vector[1], 
              head_width=0.2, head_length=0.3, fc=color, ec=color, label=label)

# グラフの設定
plt.figure(figsize=(10, 8))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(alpha=0.3)

# ベクトルのプロット
plot_vector(v, color='red', label='v')
plot_vector(w, color='blue', label='w')
plot_vector(v_plus_w, color='green', label='v + w')

# v + wの別解法: vの終点からwを描く
plot_vector(w, origin=v, color='purple', label='w (from v)')

# スカラー倍のプロット
plot_vector(alpha_v, color='orange', label=f'{alpha}v')
plot_vector(beta_w, color='brown', label=f'{beta}w')

# グラフの設定
plt.xlim(-2, 10)
plt.ylim(-2, 10)
plt.title('ベクトルの加法とスカラー倍の幾何学的意味')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')  # アスペクト比を1:1に
plt.show()
```

### 5.3 健康データにおけるベクトル表現の例

```python
# 健康データの例（年齢、身長、体重、血圧、コレステロール値）
patient1 = np.array([35, 170, 70, 120, 200])
patient2 = np.array([42, 165, 80, 135, 220])

# 平均値
average = (patient1 + patient2) / 2
print("平均値:", average)

# 年齢による重み付け（年齢が高いほど重視する場合）
weight1 = patient1[0] / (patient1[0] + patient2[0])  # 35/(35+42) ≈ 0.45
weight2 = patient2[0] / (patient1[0] + patient2[0])  # 42/(35+42) ≈ 0.55

weighted_avg = weight1 * patient1 + weight2 * patient2
print("年齢による重み付け平均:", weighted_avg)
```

## 6. 演習問題

### 基本問題

1. 次のベクトルの計算を行いなさい。
   (a) $\begin{pmatrix} 3 \\ -2 \\ 5 \end{pmatrix} + \begin{pmatrix} -1 \\ 4 \\ 2 \end{pmatrix}$
   (b) $2 \begin{pmatrix} 4 \\ 0 \\ -3 \end{pmatrix} - 3 \begin{pmatrix} 1 \\ -2 \\ 2 \end{pmatrix}$

2. $\mathbf{a} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$, $\mathbf{b} = \begin{pmatrix} -1 \\ 3 \end{pmatrix}$ とするとき、$2\mathbf{a} - \mathbf{b}$ を求めなさい。また、この計算を幾何学的に解釈し、図示しなさい。

3. ベクトル $\mathbf{a} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ と $\mathbf{b} = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$ に対して、$\mathbf{x} = 2\mathbf{a} + \mathbf{b}$ と $\mathbf{y} = \mathbf{a} - 3\mathbf{b}$ を求めなさい。

### 応用問題

4. ある地域の5日間の気温（°C）が $\mathbf{t}_1 = \begin{pmatrix} 25 \\ 27 \\ 24 \\ 26 \\ 28 \end{pmatrix}$ で、別の地域の同じ5日間の気温が $\mathbf{t}_2 = \begin{pmatrix} 22 \\ 23 \\ 21 \\ 24 \\ 25 \end{pmatrix}$ だったとします。
   (a) 二つの地域の気温差 $\mathbf{t}_1 - \mathbf{t}_2$ を求めなさい。
   (b) 両地域の平均気温 $\frac{1}{2}(\mathbf{t}_1 + \mathbf{t}_2)$ を求めなさい。
   (c) 気象予報によると、明日の気温は今日より2°C上昇し、明後日は明日より1°C下降すると予測されています。ベクトル演算を用いて、この予測を表現しなさい。

5. 3人の患者の健康データ（年齢、収縮期血圧、拡張期血圧、血糖値）が以下のベクトルで表されるとします：
   $\mathbf{p}_1 = \begin{pmatrix} 45 \\ 130 \\ 85 \\ 95 \end{pmatrix}$, 
   $\mathbf{p}_2 = \begin{pmatrix} 62 \\ 145 \\ 90 \\ 110 \end{pmatrix}$, 
   $\mathbf{p}_3 = \begin{pmatrix} 38 \\ 120 \\ 80 \\ 90 \end{pmatrix}$

   (a) 3人の平均値 $\frac{1}{3}(\mathbf{p}_1 + \mathbf{p}_2 + \mathbf{p}_3)$ を求めなさい。
   (b) 標準的な健康値を $\mathbf{s} = \begin{pmatrix} - \\ 120 \\ 80 \\ 100 \end{pmatrix}$ とします（年齢は基準としません）。各患者のデータと標準値との差 $\mathbf{p}_i - \mathbf{s}$ を計算し、どの患者が最も標準から離れているかを判定しなさい。

## 7. よくある質問と解答

### Q1: ベクトルの要素（成分）は必ず数値である必要がありますか？
A1: 理論的なベクトル空間では、成分は必ずしも実数である必要はありません。複素数、関数、多項式などをベクトルの要素とする場合もあります。しかし、本講義では主に実数ベクトルを扱います。

### Q2: 異なる次元のベクトル同士は加算できないのはなぜですか？
A2: ベクトルの加法は「対応する成分同士を足す」操作として定義されているため、次元（成分の数）が異なるベクトル同士では対応する成分が定義できず、加算の定義が適用できないからです。データ分析においては、特徴量の数が揃っていることが重要な理由の一つです。

### Q3: 実際のデータ分析では、どのようにしてベクトルを扱いますか？
A3: データ分析では、各データポイント（例: 患者一人のデータ）を1つのベクトルとして表現することが一般的です。例えば、体重、身長、年齢、血圧などの測定値は、その患者を表すベクトルの成分となります。複数の患者のデータは、複数のベクトルの集まり、つまり行列として表現されます。

### Q4: ベクトルのスカラー倍と内積の違いは何ですか？
A4: スカラー倍はベクトルに実数をかける操作で、結果はベクトルになります。一方、内積は2つのベクトル間の演算で、結果はスカラー（実数）になります。内積については次回の講義で詳しく学びます。

### Q5: ベクトルは必ず原点から始まるのですか？
A5: 幾何学的には、ベクトルは「大きさと方向を持つ量」であり、必ずしも原点から始まる必要はありません。ただし、計算上は原点から始まるベクトル（位置ベクトル）として表現することが便利です。任意の2点間のベクトルは、それらの位置ベクトルの差として表現できます。
