# 講義2: ベクトルの定義と基本操作①

## 1. 本日扱う内容について（概要）
- **目的:**  
  - ベクトルの定義および表現方法を理解する  
  - ベクトルの和とスカラー倍の演算方法とその幾何学的意味を学ぶ  
  - 簡単な例題を通して、基本操作の考え方をディスカッションする

- **講義の流れ:**  
  1. ベクトルの定義・表現方法の説明  
  2. ベクトルの和とスカラー倍の演算（数式を含む）  
  3. 具体的な例題の提示  
  4. ChatGPTによる解説およびGoogle Colabでの実行例  
  5. 演習問題による理解の定着

## 2. 扱う内容の理論の説明（定義・定理・数式を含む）
### ベクトルの定義と表現方法
- **定義:**  
  ベクトルは、大きさ（長さ）と方向を持つ量であり、数値の並びで表現される。  
  一般に、$ n $次元空間のベクトルは  
  $$
  \mathbf{v} = \begin{pmatrix} v_1 \\\ v_2 \\\ \vdots \\\ v_n \end{pmatrix}
  $$
  と表される。

- **表現方法:**  
  高校までは、ベクトルには $\vec{AB}$ のように矢印を書き、$\vec{AB} = (4, 2)$ のように成分は横に表記されていたが、大学以降では縦での表記が標準で、横である場合には横ベクトルを使用していることを明記する場合が多い。
  - **列ベクトル:**  
    
    $$
    \mathbf{v} = \begin{pmatrix} v_1 \\\ v_2 \\\ v_3 \end{pmatrix}
    $$
  
  - **行ベクトル:**  
    $$
    \mathbf{v} = (v_1, v_2, v_3)
    $$

### ベクトルの和
- **定義:**  
  2つのベクトル $\mathbf{u} = \begin{pmatrix} u_1 \\\ u_2 \end{pmatrix}$ と $\mathbf{v} = \begin{pmatrix} v_1 \\\ v_2 \end{pmatrix}$ の和は、各成分ごとに加算する操作である。  
  $$
  \mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\\ u_2 + v_2 \end{pmatrix}
  $$

### スカラー倍
- **定義:**  
  ベクトル $\mathbf{v} = \begin{pmatrix} v_1 \\\ v_2 \end{pmatrix}$ に対して、スカラー $ c $ をかける操作は、各成分に $ c $ を掛けることで行う。  
  $$
  c\mathbf{v} = \begin{pmatrix} c\,v_1 \\\ c\,v_2 \end{pmatrix}
  $$

### 幾何学的意味
- **ベクトルの和:**  
  2つのベクトルを「首尾一貫に」並べる（頭から尾へと連結する）ことで、平行四辺形が形成され、その対角線がベクトルの和を表す。

- **スカラー倍:**  
  ベクトルの長さを伸縮させる操作であり、スカラーが正ならば同じ方向に、負なら逆方向に伸縮する。

## 3. 扱う内容の例（実例を提示、GPT, Colabなどは利用しない）
### 例1: 二次元平面のベクトル
- **与えられたベクトル:**  
  $\mathbf{a} = \begin{pmatrix} 2 \\\ 3 \end{pmatrix}$  
  $\mathbf{b} = \begin{pmatrix} -1 \\\ 4 \end{pmatrix}$

- **和の計算:**  
  $$
  \mathbf{a} + \mathbf{b} = \begin{pmatrix} 2 + (-1) \\\ 3 + 4 \end{pmatrix} = \begin{pmatrix} 1 \\\ 7 \end{pmatrix}
  $$

- **スカラー倍の計算:**  
  $$
  2\mathbf{a} = \begin{pmatrix} 2 \times 2 \\\ 2 \times 3 \end{pmatrix} = \begin{pmatrix} 4 \\\ 6 \end{pmatrix}
  $$

### 例2: 三次元空間のベクトル
- **与えられたベクトル:**  
  $\mathbf{u} = \begin{pmatrix} 1 \\\ 0 \\\ -2 \end{pmatrix}$  
  $\mathbf{v} = \begin{pmatrix} 3 \\\ 5 \\\ 1 \end{pmatrix}$

- **和の計算:**  
  $$
  \mathbf{u} + \mathbf{v} = \begin{pmatrix} 1+3 \\\ 0+5 \\\ -2+1 \end{pmatrix} = \begin{pmatrix} 4 \\\ 5 \\\ -1 \end{pmatrix}
  $$

- **スカラー倍の計算:**  
  $$
  -3\mathbf{v} = \begin{pmatrix} -3 \times 3 \\\ -3 \times 5 \\\ -3 \times 1 \end{pmatrix} = \begin{pmatrix} -9 \\\ -15 \\\ -3 \end{pmatrix}
  $$

## 4. ChatGPTによる解説＋Colabでの実行例
- **ChatGPTによる解説:**  
  - ベクトルは、物理学や工学、コンピュータグラフィックスなど多くの分野で基本となる概念です。  
  - ベクトルの和は、例えば力の合成により全体の力を求めるといった応用があります。  
  - スカラー倍は、ベクトルの大きさの調整や方向の維持に用いられ、正規化などの手法に直結します。

- **Google Colabでの実行例:**
  
```python
import numpy as np
import matplotlib.pyplot as plt

# 二次元ベクトルの定義
a = np.array([2, 3])
b = np.array([-1, 4])

# ベクトルの和
sum_ab = a + b

# スカラー倍
scalar_multiple = 2 * a

# 結果の出力
print("ベクトル a:", a)
print("ベクトル b:", b)
print("a + b:", sum_ab)
print("2 * a:", scalar_multiple)

# ベクトルの可視化
origin = [0, 0]  # 原点

plt.figure(figsize=(6,6))
plt.quiver(*origin, *a, angles='xy', scale_units='xy', scale=1, color='r', label='a')
plt.quiver(*origin, *b, angles='xy', scale_units='xy', scale=1, color='b', label='b')
plt.quiver(*origin, *sum_ab, angles='xy', scale_units='xy', scale=1, color='g', label='a+b')

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.title("二次元ベクトルの和とスカラー倍")
plt.grid()
plt.show()
```

## 5. 演習問題
1. **演習問題1:**  
   次の二次元ベクトル  
   $$
   \mathbf{x} = \begin{pmatrix} 4 \\\ -2 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 1 \\\ 3 \end{pmatrix}
   $$  
   の和と、$\mathbf{x}$ のスカラー倍 $ -3\mathbf{x} $ を計算せよ。

2. **演習問題2:**  
   三次元ベクトル  
   $$
   \mathbf{p} = \begin{pmatrix} 2 \\\ 0 \\\ 5 \end{pmatrix}, \quad \mathbf{q} = \begin{pmatrix} -1 \\\ 4 \\\ 2 \end{pmatrix}
   $$  
   を用いて、$\mathbf{p} + \mathbf{q}$ を求め、さらに $\mathbf{q}$ のスカラー倍 $ 0.5\mathbf{q} $ を計算せよ。

3. **ディスカッション:**  
   ベクトルの和とスカラー倍の演算が、物理学（例：力の合成）やコンピュータグラフィックス（例：座標変換）などの現実の問題にどのように応用されるか、グループで議論しなさい。

---

以上が「講義2: ベクトルの定義と基本操作①」の内容です。講義と演習を通じて、ベクトルの基本的な性質と操作に慣れ、線形代数学の基礎をしっかりと固めましょう。