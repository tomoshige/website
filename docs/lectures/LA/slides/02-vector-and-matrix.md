---
marp: true
size: 16:9
paginate: true
theme: gaia
backgroundColor: #fff
math: katex
---
<!-- header: 'T. Nakamura | Juntendo Univ.' -->
<!-- footer: '2025/02/08' -->
<style>
section { 
    font-size: 20px; 
}
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
/* ページ番号 */
section::after {
    content: attr(data-marpit-pagination) ' / ' attr(data-marpit-pagination-total);
    fonr-size: 60%;
}
/* 発表会名 */
header {
    width: 100%;
    position: absolute;
    top: unset;
    bottom: 21px;
    left: 0;
    text-align: center;
    font-size: 60%;
}
/* 日付 */
footer {
    text-align: center;
    font-size: 15px;
}
</style>


<!--
_class: lead
_paginate: false
-->

![bg left:50% 80%](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*7c3wpgkwcxbHgvZtaTL7gg.png)

# 線形代数学 I: 第2回講義
## ベクトル - 定義と基本操作
### 中村 知繁





---

## 講義情報

- **講義回**: 第2回
- **テーマ**: ベクトルの定義と基本操作①
- **関連項目**: ベクトル空間、ベクトル演算、幾何学的解釈
- **予習内容**: 高校数学の座標平面と空間座標の基礎知識を復習しておくこと

---

## 学習目標

1. 実ベクトルの定義と表記方法を理解する
2. ベクトルの加法とスカラー倍の演算規則を習得する
3. ベクトル演算の幾何学的意味を理解する
4. Pythonを用いてベクトル演算を実装できるようになる

---

## 基本概念：実ベクトルの定義

> **定義**: $n$次元実ベクトルとは、$n$個の実数を縦に並べたもので、以下のように表記される：
> 
> $$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$
> 
> ここで $v_1, v_2, \ldots, v_n$ は実数である。また、ベクトルの1つ1つの要素を**成分**という。

---

## ベクトル空間

$n$次元実ベクトルの集合を $\mathbb{R}^n$ と表し、$n$次元実ベクトル空間と呼びます。

重要な例：
- $\mathbb{R}^2$: 2次元実ベクトル空間（平面上のベクトル）
- $\mathbb{R}^3$: 3次元実ベクトル空間（空間上のベクトル）

データサイエンスでは、データの各サンプルを1つのベクトルとして扱うことが一般的です。

---

## ベクトルの表し方

ベクトルは以下のような様々な方法で表記されます：

1. **成分表示**:
   $$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

2. **太字記号**:
   $\mathbf{v}$ や $\vec{v}$

3. **列ベクトル**:
   $$\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$$
   ここで $T$ は転置という操作で、縦と横を入れ替える操作です。

---

## 零ベクトル

> **定義**: 零ベクトル $\mathbf{0}$ とは、すべての成分が0であるベクトル
> 
> $$\mathbf{0} = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}$$

零ベクトルは加法演算の単位元としての役割を持ちます。

---

## ベクトルの和

> **定義**: 2つの同じ次元のベクトル $\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$ と $\mathbf{w} = (w_1, w_2, \ldots, w_n)^T$ に対して、その和 $\mathbf{v} + \mathbf{w}$ は次のように定義する：
> 
> $$\mathbf{v} + \mathbf{w} = \begin{pmatrix} v_1 + w_1 \\ v_2 + w_2 \\ \vdots \\ v_n + w_n \end{pmatrix}$$

ベクトルの加法は「**対応する成分同士を足す**」操作です。

---

## ベクトル加法の例

$\mathbf{v} = \begin{pmatrix} 1 \\ 3 \\ 5 \end{pmatrix}$ と $\mathbf{w} = \begin{pmatrix} 2 \\ -1 \\ 4 \end{pmatrix}$ のとき：

$$
\mathbf{v} + \mathbf{w} = \begin{pmatrix} 1 + 2 \\ 3 + (-1) \\ 5 + 4 \end{pmatrix} = \begin{pmatrix} 3 \\ 2 \\ 9 \end{pmatrix}
$$

---

## ベクトル加法の性質

ベクトルの加法は、以下の性質があります：

1. **結合法則**: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
2. **交換法則**: $\mathbf{v} + \mathbf{w} = \mathbf{w} + \mathbf{v}$
3. **単位元**: 任意のベクトル $\mathbf{v}$ に対して $\mathbf{v} + \mathbf{0} = \mathbf{v}$
4. **逆元**: 任意のベクトル $\mathbf{v}$ に対して $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ となる $-\mathbf{v}$ が存在する

---

## ベクトルのスカラー倍

> **定義**: ベクトル $\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$ と実数 $\alpha$ に対して、$\mathbf{v}$ のスカラー倍 $\alpha\mathbf{v}$ は次のように定義される：
> 
> $$\alpha\mathbf{v} = \begin{pmatrix} \alpha v_1 \\ \alpha v_2 \\ \vdots \\ \alpha v_n \end{pmatrix}$$

ベクトルのスカラー倍は「すべての成分に同じ実数をかける」操作です。

---

## スカラー倍の例

$\mathbf{v} = \begin{pmatrix} 2 \\ 3 \\ -1 \end{pmatrix}$ のとき、$3\mathbf{v}$ は：

$$3\mathbf{v} = \begin{pmatrix} 3 \cdot 2 \\ 3 \cdot 3 \\ 3 \cdot (-1) \end{pmatrix} = \begin{pmatrix} 6 \\ 9 \\ -3 \end{pmatrix}$$

---

## スカラー倍の性質

スカラー倍は、次の性質があります：

1. $1\mathbf{v} = \mathbf{v}$
2. $\alpha(\beta\mathbf{v}) = (\alpha\beta)\mathbf{v}$
3. $(\alpha + \beta)\mathbf{v} = \alpha\mathbf{v} + \beta\mathbf{v}$
4. $\alpha(\mathbf{v} + \mathbf{w}) = \alpha\mathbf{v} + \alpha\mathbf{w}$

---

## ベクトルの幾何学的解釈

- $\mathbb{R}^2$ のベクトル $\mathbf{v} = (v_1, v_2)^T$ は、原点 $(0,0)$ から点 $(v_1, v_2)$ へ向かう矢印として解釈できる
- $\mathbb{R}^3$ や高次元でも同様（ただし高次元は視覚化が難しい）

---

## ベクトルの和の幾何学的意味

2つのベクトル $\mathbf{v} + \mathbf{w}$ の和は幾何学的に以下のように解釈できます：

1. $\mathbf{v}$ と $\mathbf{w}$ で形成される平行四辺形の対角線
2. 原点から $\mathbf{v}$ を通り、そこから $\mathbf{w}$ を辿った先の点へ向かうベクトル

これは「平行四辺形の法則」とも呼ばれます。

---

## スカラー倍の幾何学的意味

ベクトル $\mathbf{v}$ とスカラー $\alpha$ について：

- $\alpha > 0$ のとき：$\alpha\mathbf{v}$ は $\mathbf{v}$ と同じ方向で、長さが $|\alpha|$ 倍
- $\alpha < 0$ のとき：$\alpha\mathbf{v}$ は $\mathbf{v}$ と反対方向で、長さが $|\alpha|$ 倍
- $\alpha = 0$ のとき：$\alpha\mathbf{v}$ は零ベクトル

---

## Python実装：ベクトル演算

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

---

## Python実装：ベクトルの可視化

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
plot_vector(w, origin=v, color='purple', label='w (from v)')
```

---

## 例：健康データのベクトル表現

```python
# 健康データ（年齢、身長、体重、血圧、コレステロール値）
patient1 = np.array([35, 170, 70, 120, 200])
patient2 = np.array([42, 165, 80, 135, 220])

# 平均値
average = (patient1 + patient2) / 2
print("平均値:", average)

# 年齢による重み付け
weight1 = patient1[0] / (patient1[0] + patient2[0])  # 35/(35+42) ≈ 0.45
weight2 = patient2[0] / (patient1[0] + patient2[0])  # 42/(35+42) ≈ 0.55

weighted_avg = weight1 * patient1 + weight2 * patient2
print("年齢による重み付け平均:", weighted_avg)
```

---

## 演習問題：基本

1. 次のベクトルの計算を行いなさい。
   
   (a) $\begin{pmatrix} 3 \\ -2 \\ 5 \end{pmatrix} + \begin{pmatrix} -1 \\ 4 \\ 2 \end{pmatrix}$
   
   (b) $2 \begin{pmatrix} 4 \\ 0 \\ -3 \end{pmatrix} - 3 \begin{pmatrix} 1 \\ -2 \\ 2 \end{pmatrix}$

2. $\mathbf{a} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$, $\mathbf{b} = \begin{pmatrix} -1 \\ 3 \end{pmatrix}$ とするとき、$2\mathbf{a} - \mathbf{b}$ を求め、幾何学的に解釈しなさい。

3. ベクトル $\mathbf{a} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ と $\mathbf{b} = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$ に対して、$\mathbf{x} = 2\mathbf{a} + \mathbf{b}$ と $\mathbf{y} = \mathbf{a} - 3\mathbf{b}$ を求めなさい。

---

## 演習問題：応用

4. 2つの地域の5日間の気温（°C）データ：

   $\mathbf{t}_1 = \begin{pmatrix} 25 \\ 27 \\ 24 \\ 26 \\ 28 \end{pmatrix}$ と $\mathbf{t}_2 = \begin{pmatrix} 22 \\ 23 \\ 21 \\ 24 \\ 25 \end{pmatrix}$

   (a) 気温差 $\mathbf{t}_1 - \mathbf{t}_2$ を求めなさい
   
   (b) 平均気温 $\frac{1}{2}(\mathbf{t}_1 + \mathbf{t}_2)$ を求めなさい
   
   (c) 天気予報（明日：+2°C、明後日：-1°C）をベクトル演算で表現しなさい

---

## 演習問題：応用

5. 3人の患者の健康データ（年齢、収縮期血圧、拡張期血圧、血糖値）が以下のベクトルで表されるとします：

   $$\mathbf{p}_1 = \begin{pmatrix} 45 \\ 130 \\ 85 \\ 95 \end{pmatrix}, \quad \mathbf{p}_2 = \begin{pmatrix} 62 \\ 145 \\ 90 \\ 110 \end{pmatrix}, \quad  
   \mathbf{p}_3 = \begin{pmatrix} 38 \\ 120 \\ 80 \\ 90 \end{pmatrix}$$

   (a) 3人の平均値 $\frac{1}{3}(\mathbf{p}_1 + \mathbf{p}_2 + \mathbf{p}_3)$ を求めなさい。

   (b) 標準的な健康値を $\mathbf{s} = \begin{pmatrix} - \\ 120 \\ 80 \\ 100 \end{pmatrix}$ とします（年齢は基準としません）。各患者のデータと標準値との差 $\mathbf{p}_i - \mathbf{s}$ を計算し、どの患者が最も標準から離れていると考えられますか？
