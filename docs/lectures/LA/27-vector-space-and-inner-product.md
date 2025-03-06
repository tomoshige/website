# 講義タイトル: 講義23: 内積と正規直交基底

## 1. 講義概要
- **講義時間:** 60分　/　**演習時間:** 30分
- **目的:**  
  内積の定義とその性質、内積空間の構造、直交基底および正規直交基底の概念を理解し、グラム・シュミット直交化法を用いて任意の線形独立なベクトル集合から正規直交基底を構成する手法を習得する。
- **今日の目標:**  
  - ベクトルの内積の定義と計算方法、及びその結果からなす角の意味を理解する  
  - 内積空間における直交性の意義と、直交基底・正規直交基底の概念を明確にする  
  - グラム・シュミット直交化法の手順を理論的に整理し、具体例を通じてその計算過程を確認する  
  - Google Colab を利用したシミュレーションにより、内積のイメージや直交化の過程を視覚的に体験する

## 2. 理論的背景と内容の説明
### 2.1 定義・基本概念
- **ベクトルの内積:**  
  実ベクトル $u=(u_1,u_2,\dots,u_n)$ と $v=(v_1,v_2,\dots,v_n)$ の内積は  
  $$
  \langle u, v \rangle = \sum_{i=1}^{n} u_i v_i
  $$
  と定義され、これにより二つのベクトルがどれだけ「似ているか」や「直交しているか」を評価することができる。

- **内積の性質**
  - 線形性: $\langle au + bv, w \rangle = a\langle u, w \rangle + b\langle v, w \rangle$  
  - 対称性: $\langle u, v \rangle = \langle v, u \rangle$  
  - 正定値性: $\langle u, u \rangle \ge 0$ （かつ $\langle u, u \rangle=0$ なら $u=0$）

- **ノルム**
    $\|u\|=\sqrt{\langle u,u\rangle}$）と定義される量を、ベクトルのノルムという。

- **ベクトルのなす角:**
  内積はその定義から、$-\|u\|\|v\| \leq \langle u, v \rangle \leq \|u\|\|v\|$を満たすので、二つのベクトル $u,v$ のなす角 $\theta$ を次のように定義する。  
  $$
  \cos\theta=\frac{\langle u,v\rangle}{\|u\|\|v\|}
  $$

- **内積空間:**  
  内積が定義されたベクトル空間を内積空間と呼ぶ。

- **直交基底と正規直交基底:**  
  ベクトル空間 $V$ の基底 $\{v_1, v_2, \dots, v_n\}$ が各 $i\neq j$ で  
  $$
  \langle v_i, v_j \rangle=0
  $$
  を満たすとき、これを**直交基底**と呼ぶ。さらに、各基底ベクトルが単位ベクトル（$\|v_i\|=1$）であれば**正規直交基底**という。

- **グラム・シュミット直交化法:**  
  任意の線形独立なベクトル集合 $\{v_1, v_2, \dots, v_n\}$ から、以下の手順で直交基底 $\{u_1, u_2, \dots, u_n\}$ を段階的に構成できる。この方法を、**グラム・シュミットの直交化法**という。
  $$
  \begin{aligned}
  u_1 &= v_1, \\
  u_2 &= v_2 - \frac{\langle v_2,u_1\rangle}{\langle u_1,u_1\rangle} u_1\\
  u_3 &= v_3 - \frac{\langle v_3,u_1\rangle}{\langle u_1,u_1\rangle} u_1 - \frac{\langle v_3,u_2\rangle}{\langle u_2,u_2\rangle} u_2\\
  u_k &= v_k - \sum_{j=1}^{k-1} \frac{\langle v_k,u_j\rangle}{\langle u_j,u_j\rangle} u_j \quad (k=2,3,\dots,n).
  \end{aligned}
  $$
  正規直交基底は、各 $u_k$ を $\|u_k\|$ で割って、
  $$
  e_k = \frac{u_k}{\|u_k\|} \quad (k=1,2,\dots,n)
  $$
  とすることで得られる（ノルムが$1$となる）。

### 2.2 定理・命題
- **内積の性質に関する定理:**  
  内積は線形性、対称性、正定値性の三つの性質を満たす。この性質から、コーシー・シュワルツの不等式や三角不等式が導かれる。

- **直交基底の存在定理:**  
  有限次元の内積空間において、任意の線形独立な集合からグラム・シュミット直交化法を用いることで正規直交基底が構成可能である。  
  ※これにより、内積空間の任意の元は正規直交基底の線形結合として一意的に表される。

### 2.3 数式・証明の詳細
- **内積と角度の計算:**  
  例として、$u=(1,2)$ と $v=(3,4)$ の内積は  
  $$
  \langle u, v\rangle = 1\times3 + 2\times4 = 11.
  $$
  また、$\|u\|=\sqrt{1^2+2^2}=\sqrt{5}$、$\|v\|=\sqrt{3^2+4^2}=5$ より、  
  $$
  \cos\theta=\frac{11}{5\sqrt{5}},
  $$
  となる。
- **グラム・シュミット直交化の数式:**  
  与えられた線形独立なベクトル集合 $\{v_1,v_2\}$ に対し、  
  $$
  \begin{aligned}
  u_1 &= v_1,\\[1mm]
  u_2 &= v_2 - \frac{\langle v_2, u_1\rangle}{\langle u_1, u_1\rangle}u_1,
  \end{aligned}
  $$
  として直交基底を求め、正規化して  
  $$
  e_1 = \frac{u_1}{\|u_1\|},\quad e_2 = \frac{u_2}{\|u_2\|}
  $$
  を得る。より高次元の場合も同様の再帰的手順で計算する。

## 3. 扱う内容の実例（GPT, Colab等は使用せず）
- **実例1: 内積の計算となす角の求め方**  
  例として、$\mathbb{R}^2$ のベクトル $u=(3, 4)$ と $v=(4, -3)$ を考える。  
  - 内積は  
    $$
    \langle u, v\rangle = 3\times4 + 4\times(-3)=12-12=0,
    $$
    よって $u$ と $v$ は直交しており、なす角は $90^\circ$ となる。  
  - 一方、内積がゼロでない場合は、上記の式  
    $$
    \cos\theta=\frac{\langle u,v\rangle}{\|u\|\|v\|}
    $$
    を用いて具体的な角度を計算する。

- **実例2: グラム・シュミット直交化の具体例**  
  $\mathbb{R}^3$ で、線形独立なベクトル  
  $$
  v_1=(1,1,0),\quad v_2=(1,0,1),\quad v_3=(0,1,1)
  $$
  を考える。  
  - **Step 1:**  
    $$
    u_1=v_1=(1,1,0)
    $$  
  - **Step 2:**  
    $$
    u_2 = v_2 - \frac{\langle v_2,u_1\rangle}{\langle u_1,u_1\rangle}u_1.
    $$
    計算すると、  
    $$
    \langle v_2,u_1\rangle = 1\times1+0\times1+1\times0=1,\quad \langle u_1,u_1\rangle = 1^2+1^2+0^2=2,
    $$
    よって  
    $$
    u_2 = (1,0,1) - \frac{1}{2}(1,1,0) = \left(1-\frac{1}{2},\;0-\frac{1}{2},\;1-0\right) = \left(\frac{1}{2},\; -\frac{1}{2},\;1\right).
    $$
  - **Step 3:**  
    $$
    u_3 = v_3 - \frac{\langle v_3,u_1\rangle}{\langle u_1,u_1\rangle}u_1 - \frac{\langle v_3,u_2\rangle}{\langle u_2,u_2\rangle}u_2.
    $$
    それぞれ内積とノルムを計算し、$u_3$ を求める。  
  - 最後に、各 $u_i$ を正規化して正規直交基底 $\{e_1, e_2, e_3\}$ を得る。  
  この手順により、任意の線形独立なベクトル集合から、直交性と正規性を兼ね備えた基底が得られることを示す。

## 4. ChatGPTによる解説＋Colabでの実行例
- **ChatGPT解説:**  
  内積は、ベクトル間の類似性や直交性を測る基本的な演算であり、空間内の角度や距離の概念に直結します。  
  直交基底や正規直交基底は、座標変換や分解、信号処理など多岐にわたる応用で利用され、グラム・シュミット直交化法は任意の基底から効率的に正規直交基底を構成するための標準的手法です。  
  理論的な証明だけでなく、具体例やシミュレーションを通してその概念を体験することが理解を深める鍵となります。

- **Colab実行例:**  
  以下は、Python (numpy, matplotlib) を用いて内積の計算、角度の求め方、及びグラム・シュミット直交化をシミュレーションするサンプルコードです。
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  # --- 内積の計算と角度の求め方 ---
  # 2次元ベクトルの例
  u = np.array([3, 4])
  v = np.array([4, -3])
  inner_product = np.dot(u, v)
  norm_u = np.linalg.norm(u)
  norm_v = np.linalg.norm(v)
  theta = np.arccos(inner_product / (norm_u * norm_v))  # ラジアン
  print("内積:", inner_product)
  print("なす角（度）:", np.degrees(theta))

  # --- グラム・シュミット直交化の実装 ---
  def gram_schmidt(V):
      """ V は各行が1つのベクトルとなる行列 """
      U = np.zeros_like(V, dtype=float)
      for i in range(V.shape[0]):
          U[i] = V[i]
          for j in range(i):
              proj = np.dot(V[i], U[j]) / np.dot(U[j], U[j])
              U[i] = U[i] - proj * U[j]
      return U

  # 3次元の例
  V = np.array([[1, 1, 0],
                [1, 0, 1],
                [0, 1, 1]], dtype=float)
  U = gram_schmidt(V)
  # 正規化
  E = np.array([u/np.linalg.norm(u) for u in U])
  print("正規直交基底:")
  print(E)

  # --- 可視化 ---
  # 2次元ベクトルのプロット
  plt.figure(figsize=(6,6))
  origin = [0, 0]
  plt.quiver(*origin, *u, angles='xy', scale_units='xy', scale=1, color='r', label='u')
  plt.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color='b', label='v')
  plt.xlim(-5, 5)
  plt.ylim(-5, 5)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('内積となす角の可視化')
  plt.grid(True)
  plt.legend()
  plt.show()

  # 3次元の正規直交基底の可視化
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, projection='3d')
  colors = ['r', 'b', 'g']
  for i in range(E.shape[0]):
      ax.quiver(0, 0, 0, E[i,0], E[i,1], E[i,2], color=colors[i], label=f'e{i+1}')
  ax.set_xlim([-1, 1])
  ax.set_ylim([-1, 1])
  ax.set_zlim([-1, 1])
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_title('3次元正規直交基底の可視化')
  ax.legend()
  plt.show()
  ```
  このコードにより、内積の計算や角度の求め方、そしてグラム・シュミット直交化の手順が実際にどのように動作するかを確認でき、視覚的にその効果を体験できます。

## 5. 学んだ内容の応用例（optional)
- **応用分野:**  
  ・コンピュータグラフィックス：ライティングやシェーディング、回転変換における直交基底の利用  
  ・機械学習：主成分分析（PCA）やデータの正規化、次元削減において内積空間の概念が重要  
  ・信号処理・統計解析：相関の測定やフィルタ設計に内積が活用される
- **具体例:**  
  内積を用いて類似度を評価する手法は、文書のベクトル表現（例えばTF-IDFやword2vec）などの自然言語処理にも応用され、グラム・シュミット直交化は冗長なデータの削減や数値計算の安定性向上に寄与する。

## 6. 演習問題
- **問題1:**  
  内積の定義とその性質（線形性、対称性、正定値性）を述べ、具体例として $\mathbb{R}^2$ における内積計算を示せ。  
  ※ヒント: ベクトル $u=(a,b)$ と $v=(c,d)$ の内積を計算し、その意味を説明すること。

- **問題2:**  
  実ベクトル $u=(2, -1, 3)$ と $v=(1, 0, -2)$ の内積を計算し、これを用いて二つのベクトルのなす角を求めよ。  
  ※ヒント: 内積とノルムの計算式を利用すること。

- **問題3:**  
  $\mathbb{R}^3$ において、線形独立なベクトル集合  
  $$
  v_1=(1,1,0),\quad v_2=(1,0,1),\quad v_3=(0,1,1)
  $$
  に対して、グラム・シュミット直交化法を用い正規直交基底を構成せよ。  
  ※ヒント: 各ステップでの射影計算と正規化の過程を明確に示すこと。

- **問題4:**  
  Google Colab を利用して、内積の計算とグラム・シュミット直交化のシミュレーションを実装し、その結果から内積空間の直感的なイメージについて考察せよ。  
  ※ヒント: 可視化結果をもとに、ベクトルの直交性や正規直交性の意義を説明すること。