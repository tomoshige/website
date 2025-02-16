# 講義3: ベクトルの基本操作②と内積の導入

## 1. 本日扱う内容について（概要）
- **目的:**  
  - ベクトルの基本操作の続きとして、内積の概念を導入する。  
  - 内積の定義、性質、幾何学的解釈（角度の計算、射影）を学ぶ。  
  - Google Colabを用いたベクトル演算の可視化実習を通じて、理論と実践の両面から理解を深める。

- **講義の流れ:**  
  1. 内積の理論的背景の説明  
  2. 内積の具体的な計算例の提示  
  3. ChatGPTによる解説とGoogle Colabでの実行例の紹介  
  4. 学んだ内容の応用事例の考察  
  5. 演習問題を通じた理解の定着

## 2. 扱う内容の理論の説明（定義・定理・数式を含む）
### 内積の定義
- **内積（ドット積）:**  
  $n$ 次元のベクトル $\mathbf{u} = (u_1, u_2, \dots, u_n)$ と $\mathbf{v} = (v_1, v_2, \dots, v_n)$ に対して、内積は  
  $$
  \langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^{n} u_i v_i
  $$
  と定義される。

### 内積の基本性質
1. **交換法則:**  
   $$
   \langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle
   $$
2. **分配法則:**  
   $$
   \langle \mathbf{u} + \mathbf{v}, \mathbf{w} \rangle = \langle \mathbf{u}, \mathbf{w} \rangle + \langle \mathbf{v}, \mathbf{w} \rangle
   $$
3. **スカラー倍:**  
   $$
   \langle c\mathbf{u}, \mathbf{v} \rangle = c \langle \mathbf{u}, \mathbf{v} \rangle
   $$
4. **正定値性:**  
   $$
   \langle \mathbf{u}, \mathbf{u} \rangle \geq 0,\quad \langle \mathbf{u}, \mathbf{u} \rangle = 0 \iff \mathbf{u} = \mathbf{0}
   $$

### 幾何学的解釈
- **角度の計算:**  
  内積は、ベクトルの大きさとその間の角度 $\theta$ との関係で表される。  
  $$
  \langle \mathbf{u}, \mathbf{v} \rangle = \|\mathbf{u}\|\,\|\mathbf{v}\|\,\cos \theta
  $$
  より、  
  $$
  \theta = \arccos\left(\frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|\,\|\mathbf{v}\|}\right)
  $$

- **射影:**  
  ベクトル $\mathbf{u}$ を $\mathbf{v}$ に射影した結果得られるベクトルは  
  $$
  \text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\langle \mathbf{v}, \mathbf{v} \rangle}\,\mathbf{v}
  $$
  と定義される。

## 3. 扱う内容の例（実例を提示、GPT, Colabなどは利用しない）
### 例1: 二次元ベクトルの内積
- **与えられたベクトル:**  
  $\mathbf{a} = (3, 4)$ と $\mathbf{b} = (2, -1)$
- **内積の計算:**  
  $$
  \langle \mathbf{a}, \mathbf{b} \rangle = 3 \times 2 + 4 \times (-1) = 6 - 4 = 2
  $$
- **角度の計算のための準備:**  
  $\|\mathbf{a}\| = \sqrt{3^2+4^2} = 5$,  
  $\|\mathbf{b}\| = \sqrt{2^2+(-1)^2} = \sqrt{4+1} = \sqrt{5}$

### 例2: 射影の計算
- **与えられたベクトル:**  
  $\mathbf{u} = (4, 3)$ と $\mathbf{v} = (1, 0)$
- **内積の計算:**  
  $$
  \langle \mathbf{u}, \mathbf{v} \rangle = 4 \times 1 + 3 \times 0 = 4
  $$
- **射影の計算:**  
  $\langle \mathbf{v}, \mathbf{v} \rangle = 1^2 + 0^2 = 1$  
  よって、  
  $$
  \text{proj}_{\mathbf{v}} \mathbf{u} = \frac{4}{1}\,(1, 0) = (4, 0)
  $$

## 4. ChatGPTによる解説＋Colabでの実行例
- **ChatGPTによる解説:**  
  内積は、単なる数値計算としての操作を超えて、ベクトル間の相関や類似性、さらには空間内での方向性の理解に直結する重要な概念です。  
  角度や射影といった幾何学的解釈を通して、データ解析、機械学習、物理計算など様々な応用分野で利用されています。

- **Google Colabでの実行例:**

```python
import numpy as np
import matplotlib.pyplot as plt

# 二次元ベクトルの定義
a = np.array([3, 4])
b = np.array([2, -1])

# 内積の計算
inner_product = np.dot(a, b)
print("内積:", inner_product)

# ベクトルの大きさの計算
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
print("aの大きさ:", norm_a)
print("bの大きさ:", norm_b)

# 角度の計算（ラジアン）
theta = np.arccos(inner_product / (norm_a * norm_b))
print("aとbのなす角（ラジアン）:", theta)

# aをbに射影する計算
proj_a_on_b = (np.dot(a, b) / np.dot(b, b)) * b
print("aのbへの射影:", proj_a_on_b)

# ベクトルの可視化
origin = [0, 0]
plt.figure(figsize=(6,6))
plt.quiver(*origin, *a, angles='xy', scale_units='xy', scale=1, color='r', label='a')
plt.quiver(*origin, *b, angles='xy', scale_units='xy', scale=1, color='b', label='b')
plt.quiver(*origin, *proj_a_on_b, angles='xy', scale_units='xy', scale=1, color='g', label='proj a on b')

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title("内積と射影の可視化")
plt.grid(True)
plt.legend()
plt.show()
```

## 5. 学んだ内容がどのように応用されるか
- **機械学習・データ解析:**  
  - **類似度計算:**  
    文章や画像の特徴ベクトル間のコサイン類似度は、内積を正規化することで求められる。  
  - **主成分分析（PCA）:**  
    共分散行列の計算に内積が利用され、データの次元削減に寄与する。

- **物理学:**  
  - **仕事の計算:**  
    力と変位の内積から、実際に物体に対して行われた仕事を求める。

- **コンピュータグラフィックス:**  
  - **ライティングとシェーディング:**  
    表面の法線ベクトルと光源方向の内積を用いて、陰影効果や明暗の計算に応用される。

## 6. 演習問題
1. **演習問題1:**  
   二次元ベクトル $\mathbf{u} = (1, 2)$ と $\mathbf{v} = (3, 4)$ の内積を計算し、  
   その結果から $\mathbf{u}$ と $\mathbf{v}$ のなす角を求めよ。

2. **演習問題2:**  
   三次元ベクトル $\mathbf{p} = (2, -1, 3)$ と $\mathbf{q} = (0, 4, -2)$ の内積を計算し、  
   $\mathbf{p}$ を $\mathbf{q}$ に射影したベクトルを求めよ。

3. **演習問題3:**  
   任意のベクトル $\mathbf{x}$ に対して、なぜ $\langle \mathbf{x}, \mathbf{x} \rangle \geq 0$ となるのか、内積の性質を用いて説明せよ。
