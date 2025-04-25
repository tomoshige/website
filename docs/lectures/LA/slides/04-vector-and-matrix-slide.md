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

# 線形代数学 I: 第4回講義
## 行列 - 定義と和の計算
### 中村 知繁

---

## 1. 講義情報と予習ガイド

- **講義回**: 第4回
- **関連項目**: ベクトル演算（第2-3回の内容）
- **予習内容**: ベクトルの和とスカラー倍、ベクトルの内積の復習
- **スライド**: [リンク](...)

---

## 2. 学習目標

1. 行列の定義を理解し、適切に表記できる
2. 行列の和を正確に計算できる
3. 行列のスカラー倍を正確に計算できる
4. 行列とベクトルの関係性を理解できる

---

## 3. 基本概念

### 3.1 行列の定義

**定義**: 行列（Matrix）とは、数や記号を縦と横に矩形状に配置したものです。$m$行$n$列の行列$A$は次のように表されます：

$$A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}$$

ここで、$a_{ij}$は$i$行$j$列目の要素を表します。

---

### 行列のサイズ

**サイズ**: 行列のサイズは行数×列数で表し、$m \times n$行列などと呼びます。

**例**:

$$A = \begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{pmatrix}$$

この行列$A$は$2 \times 3$行列（2行3列の行列）です。

---

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

---

### 3.3 行列の表記法

行列は通常、大文字のアルファベット（$A$, $B$, $C$など）で表します。行列の要素は小文字の添え字付きの文字（$a_{ij}$など）で表します。

- $A$: 行列全体
- $a_{ij}$: 行列$A$の$i$行$j$列目の要素
- $A_{i,j}$: 行列$A$の$i$行$j$列目の要素（別表記）

---

## 4. 計算手法

### 4.1 行列の和

**定義**: 同じサイズの行列$A$と$B$の和$A + B$は、対応する要素同士を足し合わせた行列です：

$$(A + B)_{ij} = a_{ij} + b_{ij}$$

**注意点**: 異なるサイズの行列同士は足し合わせることができません。

---

### 行列の和の例
- $2\times 2$ 行列の例


$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$$

$$A + B = \begin{pmatrix} 1+5 & 2+6 \\ 3+7 & 4+8 \end{pmatrix} = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix}$$

- $3\times 2$ 行列の例

$$A = \begin{pmatrix} 1 & 2 \\ -1 & -3 \\ 1 & -2 \\ \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 1 \\ 1 & 2 \\ 3 & -1  \end{pmatrix}$$

$$A - B = \begin{pmatrix} 1 & 2 \\ -1 & -3 \\ 1 & -2 \\ \end{pmatrix} + \begin{pmatrix} 0 & 1 \\ 1 & 2 \\ 3 & -1  \end{pmatrix} = \begin{pmatrix} 0 & 3 \\ 0 & -1 \\ 4 & -3  \end{pmatrix}$$


---

### 4.2 行列の和の性質

行列の和は以下の性質を持ちます：

1. **交換法則**: $A + B = B + A$
2. **結合法則**: $(A + B) + C = A + (B + C)$
3. **単位元**: 零行列 $O$ について $A + O = A$
4. **逆元**: $-A$ について $A + (-A) = O$

---

### 4.3 行列のスカラー倍

**定義**: 行列$A$のスカラー倍$cA$は、$A$の各要素に$c$を掛けた行列です：

$$(cA)_{ij} = c \cdot a_{ij}$$

---

### 行列のスカラー倍の例

- $2\times 2$ 行列の例
$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad c = 3$$

$$cA = 3 \cdot \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 3 \cdot 1 & 3 \cdot 2 \\ 3 \cdot 3 & 3 \cdot 4 \end{pmatrix} = \begin{pmatrix} 3 & 6 \\ 9 & 12 \end{pmatrix}$$

- $3\times 2$ 行列の例

$$A = \begin{pmatrix} 1 & 2 & 1 \\ 3 & 4 & 2 \end{pmatrix}, \quad c = -2$$

$$cA = -2 \cdot \begin{pmatrix} 1 & 2 & 1 \\ 3 & 4 & 2 \end{pmatrix} =  \begin{pmatrix}(-2) \cdot 1 & (-2) \cdot 2 & (-2) \cdot 1 \\ (-2) \cdot 3 & (-2) \cdot 4 & (-2) \cdot 2 \end{pmatrix} = \begin{pmatrix} -1 & -2 & -1 \\ -3 & -4 & -2 \end{pmatrix}$$

---

### 4.4 行列のスカラー倍の性質

行列のスカラー倍は以下の性質を持ちます：

1. $c(A + B) = cA + cB$
2. $(c + d)A = cA + dA$
3. $c(dA) = (cd)A$
4. $1 \cdot A = A$

---

### 4.5 行列とベクトルの関係

行列は「ベクトルを列に並べたもの」と見ることができます。例えば、$n$次元の列ベクトル$\vec{v}_1, \vec{v}_2, ..., \vec{v}_m$を考えると、それらを横に並べた行列$A$は：

$$A = \begin{pmatrix} | & | & & | \\ \vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_m \\ | & | & & | \end{pmatrix}$$

同様に、行列を「行ベクトルを縦に積み重ねたもの」と見ることもできます。

---

### 例: 列ベクトルから行列を構成

列ベクトル $\vec{v}_1 = \begin{pmatrix} 1 \\ 3 \end{pmatrix}$, $\vec{v}_2 = \begin{pmatrix} 2 \\ 4 \end{pmatrix}$ を並べると、

$$A = \begin{pmatrix} | & | \\ \vec{v}_1 & \vec{v}_2 \\ | & | \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$$

---

### 例: 列ベクトルから行列を構成

列ベクトル $\vec{v}_1 = \begin{pmatrix} 1 \\ 3 \\ 2 \end{pmatrix}$, $\vec{v}_2 = \begin{pmatrix} 2 \\ 4 \\ -3 \end{pmatrix}$, $\vec{v}_3 = \begin{pmatrix} -1 \\ -2 \\ 1 \end{pmatrix}$ を並べると、

$$A = \begin{pmatrix} | & | & | \\ \vec{v}_1 & \vec{v}_2 & \vec{v}_3 \\ | & | & | \end{pmatrix} = \begin{pmatrix} 1 & 2 & -1 \\ 3 & 4 & -2 \\ 2 & -3 & 1 \end{pmatrix}$$

---


## 6. 演習問題

1. 次の行列のサイズを答えなさい。
   
   (a) $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$
   
   (b) $B = \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}$
   
   (c) $C = \begin{pmatrix} 13 & 14 & 15 & 16 \end{pmatrix}$

   (c) $D = \begin{pmatrix} 1 & 2 & 1  \\ -2 & -2 & 1 \\ 1 & 2 & 1  \\ -2 & -2 & 1  \end{pmatrix}$

---

### 演習問題（続き）

2. 次の行列の和を求めなさい。
   
   $A = \begin{pmatrix} 2 & 0 \\ -1 & 3 \end{pmatrix}, \quad B = \begin{pmatrix} 4 & -2 \\ 1 & 5 \end{pmatrix}$

3. 次の行列のスカラー倍を求めなさい。
   
   $A = \begin{pmatrix} 1 & -2 & 3 \\ 0 & 4 & -5 \end{pmatrix}, \quad c = -2$

---

### 演習問題（続き）

4. 次の計算をせよ。
   
   $2A - 3B$, ただし $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$

---

### 演習問題（続き）

5. 以下の患者データ行列 $P$ があります：
   
   $$P = \begin{pmatrix} 
   105 & 60 & 90 \\
   120 & 85 & 110 \\
   125 & 75 & 95 \\
   110 & 80 & 125 \end{pmatrix}$$
   
   各行は患者、各列は異なる健康指標（例：血圧、体重、コレステロール値）を表しています。全ての患者データに対して、標準化のために以下の操作を行います：
   
   - 血圧（1列目）から血圧の平均を引く
   - 体重（2列目）から体重の平均を引く
   - コレステロール値（3列目）からコレステロール値の平均を引く
   
   この操作を行列の計算として表現し、結果の行列を求めなさい。

---

## 7. よくある質問と解答

**Q1: 行列とベクトルの違いは何ですか？**

A1: ベクトルは行列の特殊な場合と考えることができます。列ベクトルは$n \times 1$行列、行ベクトルは$1 \times m$行列です。行列はベクトルを複数並べたものとも見ることができます。

---

### よくある質問（続き）

**Q2: 行列の和やスカラー倍がデータサイエンスでどのように使われますか？**

A2: 行列の和やスカラー倍は、データの正規化、特徴量のスケーリング、複数のデータセットの結合、時系列データの移動平均の計算など、様々なデータ前処理や分析手法で使用されます。また、機械学習アルゴリズムの内部計算（勾配降下法など）でも重要な役割を果たします。

---

### よくある質問（続き）

**Q3: 行列の要素を並べる順序は重要ですか？**

A3: 非常に重要です。行列では要素の位置（行番号と列番号）が情報を持っています。行と列を入れ替えると、全く異なる行列になります。特に、データサイエンスでは行は通常サンプル（観測値）、列は特徴量（変数）を表すことが多いため、その構造を保つことが重要です。
