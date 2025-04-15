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

# 線形代数学 I: 第3回講義
## ベクトル - 定義と基本操作
### 中村 知繁

---

## 1. 講義情報と予習ガイド

**講義回**: 第3回  
**関連項目**: ベクトルの内積、ノルム、射影  
**予習すべき内容**: 第2回講義で扱ったベクトルの定義、ベクトルの和、スカラー倍の概念

---

## 2. 学習目標

1. ベクトルのノルム（大きさ）の定義を理解し、計算できる
2. ベクトルの内積の定義と幾何学的意味を理解する
3. 内積を用いてベクトル間の角度を計算できる
4. ベクトルの射影の概念を理解し、計算方法を習得する
5. これらの概念をデータ分析の文脈で応用する方法を学ぶ

---

## 3. 基本概念

### 3.1 ベクトルのノルム（大きさ）

**定義**: $n$次元ベクトル $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$ のノルム（長さ・大きさ）は以下のように定義される：

$$\|\mathbf{x}\| = \sqrt{x_1^2 + x_2^2 + \ldots + x_n^2}$$

ノルムは、原点からベクトルの終点までの距離を表します。

---

### ノルムの例と性質

**例**: ベクトル $\mathbf{v} = (3, 4)^T$ のノルムを計算

$$\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

**ノルムの性質**:

1. 非負性: $\|\mathbf{x}\| \geq 0$ （ゼロベクトルの場合のみ等号成立）
2. スカラー倍: $\|c\mathbf{x}\| = |c|\|\mathbf{x}\|$ （$c$はスカラー）
3. 三角不等式: $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$

---

### 3.2 ベクトルの内積

**定義**: $n$次元ベクトル $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$ と $\mathbf{y} = (y_1, y_2, \ldots, y_n)^T$ の内積は以下のように定義される：

$$\mathbf{x} \cdot \mathbf{y} = x_1y_1 + x_2y_2 + \ldots + x_ny_n = \sum_{i=1}^{n} x_iy_i$$

表記として、$<\mathbf{x},\mathbf{y}>$ を使う場合もある。

---

### 内積の例と性質

**例**: ベクトル $\mathbf{a} = (2, 3, 1)^T$ と $\mathbf{b} = (1, 0, 4)^T$ の内積を計算

$$\mathbf{a} \cdot \mathbf{b} = 2 \times 1 + 3 \times 0 + 1 \times 4 = 2 + 0 + 4 = 6$$

**内積の性質**:

1. 対称性: $\mathbf{x} \cdot \mathbf{y} = \mathbf{y} \cdot \mathbf{x}$
2. 線形性: $(\alpha\mathbf{x} + \beta\mathbf{y}) \cdot \mathbf{z} = \alpha(\mathbf{x} \cdot \mathbf{z}) + \beta(\mathbf{y} \cdot \mathbf{z})$
3. 正定値性: $\mathbf{x} \cdot \mathbf{x} \geq 0$ （ゼロベクトルの場合のみ等号成立）

---

### 3.3 内積とベクトルのノルムの関係

内積とノルムには以下の関係があります：

$$\mathbf{x} \cdot \mathbf{x} = \|\mathbf{x}\|^2$$

つまり、ベクトル自身との内積はそのベクトルのノルムの2乗に等しいです。

---

### 3.4 内積と角度の関係

**定義**: 二つの非ゼロベクトル $\mathbf{x}$ と $\mathbf{y}$ がなす角 $\theta$ は、以下の式で計算できる：

$$\cos \theta = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$$

この関係は、内積の幾何学的解釈として重要です。

---

### 内積と角度の例

**例**: ベクトル $\mathbf{u} = (1, 1)^T$ と $\mathbf{v} = (0, 1)^T$ のなす角を計算

$$\mathbf{u} \cdot \mathbf{v} = 1 \times 0 + 1 \times 1 = 1$$
$$\|\mathbf{u}\| = \sqrt{1^2 + 1^2} = \sqrt{2}$$
$$\|\mathbf{v}\| = \sqrt{0^2 + 1^2} = 1$$

$$\cos \theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{1}{\sqrt{2} \times 1} = \frac{1}{\sqrt{2}} = \frac{\sqrt{2}}{2}$$

したがって、$\theta = 45^\circ$（$\pi/4$ ラジアン）となります。

---

### 3.5 直交ベクトル

**定義**: 二つのベクトル $\mathbf{x}$ と $\mathbf{y}$ が直交するとは、その内積がゼロになることを意味する：

$$\mathbf{x} \cdot \mathbf{y} = 0$$

これは幾何学的には、二つのベクトルが90度（$\pi/2$ ラジアン）の角度をなすことを意味します。

---

### 直交ベクトルの例

**例**: ベクトル $\mathbf{a} = (3, 4)^T$ と $\mathbf{b} = (4, -3)^T$ が直交することを確認

$$\mathbf{a} \cdot \mathbf{b} = 3 \times 4 + 4 \times (-3) = 12 - 12 = 0$$

よって、ベクトル $\mathbf{a}$ と $\mathbf{b}$ は直交しています。

---

## 4. 理論と手法

### 4.1 ベクトルの射影

ベクトルの射影は、あるベクトルを別のベクトルに投影したときの成分を表します。

**定義**: ベクトル $\mathbf{b}$ 上へのベクトル $\mathbf{a}$ の射影ベクトル $\text{proj}_{\mathbf{b}}\mathbf{a}$ は以下のように定義される：

$$\text{proj}_{\mathbf{b}}\mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2} \mathbf{b}$$

射影ベクトルの大きさ（スカラー射影）は以下のように計算されます：

$$\|\text{proj}_{\mathbf{b}}\mathbf{a}\| = \frac{|\mathbf{a} \cdot \mathbf{b}|}{\|\mathbf{b}\|}$$

---

### 射影の例

**例**: ベクトル $\mathbf{a} = (2, 3)^T$ のベクトル $\mathbf{b} = (1, 0)^T$ 上への射影を計算

$$\mathbf{a} \cdot \mathbf{b} = 2 \times 1 + 3 \times 0 = 2$$
$$\|\mathbf{b}\|^2 = 1^2 + 0^2 = 1$$

$$\text{proj}_{\mathbf{b}}\mathbf{a} = \frac{2}{1} (1, 0)^T = (2, 0)^T$$

この結果は、ベクトル $\mathbf{a} = (2, 3)^T$ の$x$軸方向の成分が2であることを示しています。

---

### 4.2 射影の幾何学的解釈

射影は、あるベクトルを別のベクトル方向に分解する操作と考えることができます。

射影の幾何学的解釈として重要なのは：
- 射影ベクトルは元のベクトルの「影」のようなもの
- 射影ベクトルは常に射影先のベクトルと同じ方向（またはその反対方向）
- 射影ベクトルと元のベクトルの差ベクトルは、射影先のベクトルと直交する

---

### 4.3 直交分解

任意のベクトル $\mathbf{a}$ は、あるベクトル $\mathbf{b}$ の方向に平行な成分と垂直な成分に分解できます：

$$\mathbf{a} = \text{proj}_{\mathbf{b}}\mathbf{a} + \mathbf{a_{\perp}}$$

ここで、$\mathbf{a_{\perp}}$ は $\mathbf{b}$ に垂直な成分で、以下のように計算できます：

$$\mathbf{a_{\perp}} = \mathbf{a} - \text{proj}_{\mathbf{b}}\mathbf{a}$$

---

### 直交分解の例

**例**: ベクトル $\mathbf{a} = (2, 3)^T$ をベクトル $\mathbf{b} = (1, 0)^T$ に平行な成分と垂直な成分に分解

平行成分: $\text{proj}_{\mathbf{b}}\mathbf{a} = (2, 0)^T$（上記の計算結果）

垂直成分: $\mathbf{a_{\perp}} = \mathbf{a} - \text{proj}_{\mathbf{b}}\mathbf{a} = (2, 3)^T - (2, 0)^T = (0, 3)^T$

---

### 4.4 コーシー・シュワルツの不等式

**定理**: 任意の二つのベクトル $\mathbf{x}$ と $\mathbf{y}$ に対して、以下の不等式が成り立つ：

$$|\mathbf{x} \cdot \mathbf{y}| \leq \|\mathbf{x}\| \|\mathbf{y}\|$$

等号は、$\mathbf{x}$ と $\mathbf{y}$ が平行（または一方がゼロベクトル）のときに成立する。

この不等式は、内積の絶対値がノルムの積を超えないことを示しています。

---

## 5. Pythonによる実装と可視化

### 5.1 NumPyを使ったベクトル演算

```python
import numpy as np
import matplotlib.pyplot as plt

# ベクトルの定義
a = np.array([2, 3])
b = np.array([1, 0])

# ノルムの計算
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
print(f"ベクトルaのノルム: {norm_a}")
print(f"ベクトルbのノルム: {norm_b}")

# 内積の計算
dot_product = np.dot(a, b)
print(f"ベクトルaとbの内積: {dot_product}")

# 角度の計算（ラジアン）
angle_rad = np.arccos(dot_product / (norm_a * norm_b))
# 角度（度）
angle_deg = np.degrees(angle_rad)
print(f"ベクトルaとbのなす角: {angle_deg:.2f}度")
```

---

### Pythonによる実装（続き）

```python
# 射影の計算
proj_a_on_b = (dot_product / (norm_b**2)) * b
print(f"ベクトルaのbへの射影: {proj_a_on_b}")

# 直交成分の計算
perp_component = a - proj_a_on_b
print(f"ベクトルaのbに垂直な成分: {perp_component}")
```

---

### 5.2 ベクトルの可視化

```python
def plot_vectors_and_projection(a, b, proj_a_on_b, perp_component):
    plt.figure(figsize=(10, 8))
    
    # ベクトルの原点
    origin = np.array([0, 0])
    
    # ベクトルaとbを描画
    plt.arrow(*origin, *a, head_width=0.2, head_length=0.3, 
              fc='blue', ec='blue', label='ベクトルa')
    plt.arrow(*origin, *b, head_width=0.2, head_length=0.3, 
              fc='red', ec='red', label='ベクトルb')
    
    # 射影ベクトルを描画
    plt.arrow(*origin, *proj_a_on_b, head_width=0.2, head_length=0.3, 
              fc='green', ec='green', label='射影ベクトル')
    
    # 垂直成分を描画（射影ベクトルの終点から）
    plt.arrow(*proj_a_on_b, *perp_component, head_width=0.2, head_length=0.3, 
              fc='purple', ec='purple', label='垂直成分')
```

---

### ベクトルの可視化（続き）

```python    
    # 射影線を点線で描画
    plt.plot([a[0], proj_a_on_b[0]], [a[1], proj_a_on_b[1]], 'k--')
    
    # 軸の設定
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # グラフの範囲設定
    margin = 1
    max_val = max(np.max(np.abs(a)), np.max(np.abs(b))) + margin
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    
    plt.gca().set_aspect('equal')
    plt.title('ベクトルの射影と直交分解')
    plt.legend()
    plt.show()

# ベクトルを可視化
plot_vectors_and_projection(a, b, proj_a_on_b, perp_component)
```

---

### 5.3 データ分析における内積の応用例

```python
# サンプルデータ（身長と体重）
heights = np.array([170, 175, 165, 180, 160])  # cm
weights = np.array([65, 70, 60, 75, 55])      # kg

# 平均を引いて中心化
heights_centered = heights - np.mean(heights)
weights_centered = weights - np.mean(weights)

# 相関係数の計算（内積を使用）
correlation = np.dot(heights_centered, weights_centered) / (
    np.linalg.norm(heights_centered) * np.linalg.norm(weights_centered))

print(f"身長と体重の相関係数: {correlation:.4f}")
```

---

## 6. 演習問題

### 基本問題

1. 以下のベクトルのノルムを計算せよ。
   - $\mathbf{a} = (1, 2, 3)^T$
   - $\mathbf{b} = (5, 0, -5)^T$
   - $\mathbf{c} = (2, 2, 2, 2)^T$

2. 以下のベクトルの組の内積を計算せよ。
   - $\mathbf{u} = (3, -1, 2)^T$ と $\mathbf{v} = (2, 4, 1)^T$
   - $\mathbf{p} = (1, 1, 1, 1)^T$ と $\mathbf{q} = (2, -2, 3, -3)^T$

---

### 基本問題（続き）

3. 以下のベクトルの組がなす角度（度）を計算せよ。
   - $\mathbf{a} = (1, 1)^T$ と $\mathbf{b} = (1, -1)^T$
   - $\mathbf{c} = (3, 0, 4)^T$ と $\mathbf{d} = (5, 0, 0)^T$

4. ベクトル $\mathbf{a} = (3, 4, 0)^T$ のベクトル $\mathbf{b} = (1, 1, 1)^T$ 上への射影ベクトルを求めよ。

5. 以下のベクトルの組が直交するかどうかを判定せよ。
   - $\mathbf{u} = (2, -1, 3)^T$ と $\mathbf{v} = (1, 2, 0)^T$
   - $\mathbf{p} = (4, 3)^T$ と $\mathbf{q} = (3, -4)^T$

---

### 応用問題

1. ベクトル $\mathbf{a} = (2, 1, 3)^T$ と $\mathbf{b} = (1, -1, 2)^T$ に対して、$\mathbf{c} = \mathbf{a} - \text{proj}_{\mathbf{b}}\mathbf{a}$ を計算せよ。$\mathbf{c}$ と $\mathbf{b}$ が直交することを確認せよ。

2. コーシー・シュワルツの不等式 $|\mathbf{x} \cdot \mathbf{y}| \leq \|\mathbf{x}\| \|\mathbf{y}\|$ を、ベクトル $\mathbf{x} = (1, 2)^T$ と $\mathbf{y} = (3, 4)^T$ を用いて確認せよ。

---

### 応用問題（続き）

3. 3次元空間内のある平面が、法線ベクトル $\mathbf{n} = (1, 2, 3)^T$ で表されるとする。点 $P(2, 3, 4)$ からこの平面への最短距離を求めよ。(ヒント: 原点から平面までの距離を考え、そこからベクトルの射影を利用する)

4. **健康データ分析の応用**: 5人の患者の血圧（収縮期/拡張期）データが以下のように与えられている。
   - 患者1: (120, 80)
   - 患者2: (130, 85)
   - 患者3: (140, 90)
   - 患者4: (125, 75)
   - 患者5: (135, 88)
   
   このデータを用いて、収縮期血圧と拡張期血圧の相関係数を内積を使って計算せよ。また、各患者のデータを、平均値のベクトルと、それに直交する方向の成分に分解せよ。

---

## 7. よくある質問と解答

### Q1: なぜデータ分析で内積が重要なのですか？

**A1**: 内積は、二つのデータベクトル間の類似性や関連性を測る基本的な道具です。例えば、正規化された二つのデータベクトル間の内積は相関係数になり、データの線形関係を測ることができます。また、内積に基づく射影は、データの次元削減や特徴抽出の基礎となります。

---

### Q2: 内積とベクトルの長さだけから角度が計算できるのはなぜですか？

**A2**: これは余弦定理の応用です。二つのベクトル $\mathbf{a}$ と $\mathbf{b}$ のなす角を $\theta$ とすると、余弦定理から $\|\mathbf{a} - \mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ が成り立ちます。また、$\|\mathbf{a} - \mathbf{b}\|^2 = (\mathbf{a} - \mathbf{b}) \cdot (\mathbf{a} - \mathbf{b}) = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2(\mathbf{a} \cdot \mathbf{b})$ です。これら二つの式を比較すると、$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ が導かれます。