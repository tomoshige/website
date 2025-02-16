# 講義5: 行列積の理論と例題

## 1. 本日扱う内容について（概要）
- **目的:**  
  - 行列積の定義と計算ルールを、より分かりやすく丁寧に学ぶ  
  - 行列積の基本的注意点（サイズの一致、非可換性）を理解する  
  - 行列とベクトルの積が1次変換（線形変換）としてどのように作用するかを確認する  
  - 画像を対象とした行列による拡大・回転・縮小の例を通して、1次変換の効果を視覚的に理解する

- **講義の流れ:**  
  1. 行列積の理論的背景（定義、計算ルール、性質）の詳細な説明  
  2. 行列とベクトルの積による1次変換の紹介  
  3. 手計算による具体的な例題の提示と、非可換性の具体例  
  4. ChatGPTによる解説およびGoogle Colabでの実行例（エラー例、幾何的解釈、回転行列の例、画像の変換例）  
  5. 学んだ内容がどのように応用されるかの考察
  6. 演習問題による理解の確認

## 2. 扱う内容の理論の説明（定義・定理・数式を含む）
### 行列積の定義と計算ルール
- **定義:**  
  行列 $ A $ が $ m \times n $ 行列、行列 $ B $ が $ n \times p $ 行列の場合、行列積 $ AB $ は定義され、結果は $ m \times p $ 行列となる。

- **計算ルール:**  
  行列積の各要素 $(AB)_{ij}$ は以下の式に従って計算される：
  $$
  (AB)_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
  $$
  **計算の流れ：**  
  1. $ A $ の $ i $ 行目の成分 $ A_{i1}, A_{i2}, \dots, A_{in} $ を取り出す。  
  2. $ B $ の $ j $ 列目の成分 $ B_{1j}, B_{2j}, \dots, B_{nj} $ を取り出す。  
  3. 対応する成分同士 $ (A_{ik} $ と $ B_{kj}) $ を掛け合わせ、その結果を全て加算する。

- **重要な注意点:**  
  - **サイズの一致:**  
    行列積を計算するには、行列 $ A $ の列数と行列 $ B $ の行数が一致している必要がある。サイズが一致しない場合、計算は定義されず、プログラムではエラーが発生する。  
  - **非可換性:**  
    一般に、行列積は可換ではなく $ AB \neq BA $ となる。これは、各行と各列の要素の組み合わせが順序に依存するためである。

### 行列とベクトルの積による1次変換
- **定義:**  
  行列 $ A $（サイズが $ m \times n $）と列ベクトル $ \mathbf{v} $（サイズが $ n \times 1 $）の積 $ A\mathbf{v} $ は、$ m \times 1 $ の新たなベクトルとなる。

- **幾何学的意味:**  
  この積は、ベクトル $ \mathbf{v} $ に対して線形変換を作用させるもので、例えば回転、拡大縮小、せん断などの変換を表す。特に、回転行列を用いると、ベクトルの方向がどのように変化するかを視覚的に確認できる。

## 3. 扱う内容の例（実例を提示、GPT, Colabなどは利用しない）
### 例1: 正方行列の積と非可換性の例
- **与えられた行列:**  
  $$
  A = \begin{pmatrix} 1 & 2 \\\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 5 & 6 \\\ 7 & 8 \end{pmatrix}
  $$
- **計算:**  
  $$
  AB = \begin{pmatrix}
  1 \times 5 + 2 \times 7 & 1 \times 6 + 2 \times 8 \\\
  3 \times 5 + 4 \times 7 & 3 \times 6 + 4 \times 8
  \end{pmatrix} = \begin{pmatrix} 19 & 22 \\\ 43 & 50 \end{pmatrix}
  $$
- **非可換性の例:**  
  $$
  BA = \begin{pmatrix}
  5 \times 1 + 6 \times 3 & 5 \times 2 + 6 \times 4 \\\
  7 \times 1 + 8 \times 3 & 7 \times 2 + 8 \times 4
  \end{pmatrix} = \begin{pmatrix} 23 & 34 \\\ 31 & 46 \end{pmatrix}
  $$
  ここで、$ AB \neq BA $ となることが確認できる。

### 例2: 行列とベクトルの積による1次変換
- **回転行列の例:**  
  2次元の回転行列（角度 $ \theta $ の回転）：
  $$
  R(\theta) = \begin{pmatrix} \cos \theta & -\sin \theta \\\ \sin \theta & \cos \theta \end{pmatrix}
  $$
  任意の2次元ベクトル $ \mathbf{v} = \begin{pmatrix} x \\\ y \end{pmatrix} $ に対して、積 $ R(\theta)\mathbf{v} $ は、$ \theta $ だけ回転した新たなベクトルを表す。

## 4. ChatGPTによる解説＋Colab での実行例
### (1) 行列積の基本例と非可換性、サイズ不一致の例

```python
import numpy as np

# 正方行列の例
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 正しいサイズでの積
AB = np.dot(A, B)
BA = np.dot(B, A)

print("A =\n", A)
print("B =\n", B)
print("A * B =\n", AB)
print("B * A =\n", BA)  # AB と BA が異なる（非可換性の例）

# サイズ不一致の例
D = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2行列
try:
    result = np.dot(A, D)
    print("A * D =\n", result)
except ValueError as e:
    print("サイズが一致しないためエラーが発生:", e)
```

### (2) 行列とベクトルの積による1次変換（回転行列）の例
```python
import numpy as np
import matplotlib.pyplot as plt

# 回転角度（ラジアン）: 45度
theta = np.pi / 4

# 2次元回転行列の定義
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# 元のベクトル（例: x軸方向の単位ベクトル）
v = np.array([1, 0])
v_rotated = np.dot(R, v)

print("回転行列 R =\n", R)
print("元のベクトル v =", v)
print("回転後のベクトル R*v =", v_rotated)

# 可視化
plt.figure(figsize=(6,6))
origin = [0, 0]
plt.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color='r', label='v (元)')
plt.quiver(*origin, *v_rotated, angles='xy', scale_units='xy', scale=1, color='b', label='R*v (回転後)')

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(True)
plt.title("回転行列によるベクトルの変換（45度回転）")
plt.legend()
plt.show()
```

### (3) 画像の拡大・回転・縮小による1次変換の例
ここでは、オープンアクセスの画像を用いて、行列による拡大、回転、縮小の変換を示し、1次変換の効果を視覚的に確認します。以下の例では、`skimage` ライブラリの画像を利用しています。

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform

# オープンアクセス画像の読み込み（astronaut画像）
image = data.astronaut()

# 1. 回転変換のみ（45度回転）
theta = np.deg2rad(45)
rotation_tform = transform.AffineTransform(rotation=theta)
image_rotated = transform.warp(image, rotation_tform.inverse)

# 2. 拡大変換のみ（拡大率1.5）
scaling_enlarge_tform = transform.AffineTransform(scale=(1.5, 1.5))
image_enlarged = transform.warp(image, scaling_enlarge_tform.inverse)

# 3. 縮小変換のみ（縮小率0.5）
scaling_shrink_tform = transform.AffineTransform(scale=(0.5, 0.5))
image_shrunk = transform.warp(image, scaling_shrink_tform.inverse)

# 4. 拡大と回転の組み合わせ
combined_matrix = scaling_enlarge_tform.params @ rotation_tform.params
combined_tform = transform.AffineTransform(matrix=combined_matrix)
image_combined = transform.warp(image, combined_tform.inverse)

# 画像の表示
fig, ax = plt.subplots(2, 2, figsize=(12, 12))

ax[0, 0].imshow(image)
ax[0, 0].set_title("Original Image")
ax[0, 0].axis('off')

ax[0, 1].imshow(image_rotated)
ax[0, 1].set_title("Rotated Image (45°)")
ax[0, 1].axis('off')

ax[1, 0].imshow(image_enlarged)
ax[1, 0].set_title("Enlarged Image (scale=1.5)")
ax[1, 0].axis('off')

ax[1, 1].imshow(image_shrunk)
ax[1, 1].set_title("Shrunk Image (scale=0.5)")
ax[1, 1].axis('off')

plt.tight_layout()
plt.show()

# さらに、拡大と回転を組み合わせた変換の例
plt.figure(figsize=(6,6))
plt.imshow(image_combined)
plt.title("Combined Transformation: Rotation + Enlargement")
plt.axis('off')
plt.show()
```

## 5. 学んだ内容がどのように応用されるか
- **線形変換の合成:**  
  複数の行列（各々が回転、拡大縮小、せん断などの変換を表す）を積むことで、連続した変換を一つの行列にまとめ、複雑な変換をシンプルに表現できる。
- **コンピュータグラフィックス:**  
  画像の回転、拡大縮小、平行移動などの操作は、行列による1次変換を用いて効率的に実現され、3Dモデリングや画像処理の基盤技術となっている。
- **機械学習:**  
  特徴抽出やデータ拡張（augmentation）など、画像処理の分野で行列を用いた変換は、モデルの精度向上に寄与する。

## 6. 演習問題
1. **演習問題1:**  
   以下の行列  
   $$
   A = \begin{pmatrix} 1 & 2 \\\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 5 & 6 \\\ 7 & 8 \end{pmatrix}
   $$
   の積 $ AB $ と $ BA $ を計算し、結果が異なること（非可換性）を確認せよ。

2. **演習問題2:**  
   サイズが一致しない行列の積を試み、Python（Google Colab）でどのようなエラーが発生するか確認せよ。  
   例：$ A $（2×2行列）と $ D $（3×2行列）の積を計算してみる。

3. **演習問題3:**  
   2次元回転行列 $ R(\theta) $ を用いて、任意の角度 $ \theta $ での回転変換を実装し、元のベクトルと回転後のベクトルの位置関係を図示して、幾何学的な意味を説明せよ。

4. **演習問題4:**  
   画像に対して、行列積を用いた拡大、回転、縮小の変換を実装せよ。  
   - オープンアクセス画像（例：`skimage.data.astronaut()`）を使用すること。  
   - 各変換（回転のみ、拡大のみ、縮小のみ、及び回転と拡大の組み合わせ）の結果をプロットし、元画像と比較して説明せよ。