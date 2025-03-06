# Soft Regression Tree (SRT)
ここでは、Soft Regression Trees の理論と、その拡張についてまとめる。また、Soft Regression Tree では、Hinton et al., (2017) によって、シグモイド関数に含まれる変数を多次元に拡張することが可能となっているが、実験を行ったところ、単一変数に依存する木のほうがensemble を行う場合には性能が高いことがわかる。これは、木同士の相関が大きくなるため、ensemble をすることで性能向上が起こらないというふうに考えるのが妥当である。

なので、単一のSRTに基づく予測では、sigmoid を用いて滑らかに予測を行い、一方でboosting を行う場合にはsigmoid 関数に含まれる変数は全変数ではなく、変数セレクターなどを実装することで、疎な木を作ることが必要となる。よって方向性としては、SoftBARTに近い方針を取らない限りは、実質的な精度が上昇しないことを意味している。

つまり開発の方針としては、
- [優先順位高] Global 正則化を含むような、シグモイド間数に全変数を用いた Soft regression trees
- [優先順位高] Global 正則化を含むような、シグモイド間数に単一変数のみ用いた Soft regression trees
- [優先順位中] 単一変数による Soft regression trees に、ensemble 時に全体を正則化するような SRT Boosting 
- [優先順位低] 分岐を2分木ではなく、多分木に変更する（ただし分割の個数を、決定する必要がある）

## 1. Soft Decision / Regression Trees

### 1.1 モデル構造

Soft Regression Treeは、$d$の深さを持つ完全二分木として定式化できます。この木は$2^d - 1$個の内部ノードと$2^d$個の葉ノードを持ちます。各内部ノードは、入力特徴量ベクトル $\mathbf{x} \in \mathbb{R}^p$ の線形変換に基づく分割関数を持ちます：

$$s_j(\mathbf{x}; \mathbf{w}_j, b_j) = \sigma\left(\frac{\mathbf{w}_j^T \mathbf{x} + b_j}{\tau}\right)$$

ここで、
- $\mathbf{w}_j \in \mathbb{R}^p$ は内部ノード $j$ の重みベクトル
- $b_j \in \mathbb{R}$ はバイアス項
- $\sigma(\cdot)$ はシグモイド関数 $\sigma(z) = \frac{1}{1 + e^{-z}}$
- $\tau > 0$ は温度パラメータ

### 1.2 葉ノードへの経路確率

入力 $\mathbf{x}$ が葉ノード $l$ に到達する確率 $\mu_l(\mathbf{x})$ は、根ノードから葉 $l$ までの経路上の各分岐確率の積として計算されます：

$$\mu_l(\mathbf{x}) = \prod_{j \in \mathcal{P}_l^L} s_j(\mathbf{x}) \prod_{j \in \mathcal{P}_l^R} (1 - s_j(\mathbf{x}))$$

ここで、
- $\mathcal{P}_l^L$ は葉 $l$ への経路上で左に分岐する内部ノードの集合
- $\mathcal{P}_l^R$ は葉 $l$ への経路上で右に分岐する内部ノードの集合

### 1.3 予測値

入力 $\mathbf{x}$ に対するモデルの予測値 $f(\mathbf{x})$ は、各葉ノードの予測値 $v_l$ を経路確率 $\mu_l(\mathbf{x})$ で重み付けした和として計算されます：

$$f(\mathbf{x}) = \sum_{l=1}^{2^d} \mu_l(\mathbf{x}) v_l$$

## 2. 損失関数と正則化項

### 2.1 基本損失関数

訓練データ $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ に対する平均二乗誤差（MSE）損失は以下のように定義されます：

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^n (f(\mathbf{x}_i) - y_i)^2$$

### 2.2 従来の正則化項

#### 2.2.1 重み正則化

個々のノードの重みに対する一般的な正則化項：

$$\mathcal{R}_{\text{weights}} = \lambda_1 \sum_{j=1}^{2^d-1} \|\mathbf{w}_j\|_1 + \lambda_2 \sum_{j=1}^{2^d-1} \|\mathbf{w}_j\|_2^2$$

ここで、$\|\mathbf{w}_j\|_1 = \sum_{k=1}^p |w_{jk}|$ は $L_1$ ノルム、$\|\mathbf{w}_j\|_2^2 = \sum_{k=1}^p w_{jk}^2$ は $L_2$ ノルムの二乗です。

#### 2.2.2 構造的複雑さペナルティ

アクティブノード数に基づくペナルティ：

$$\mathcal{R}_{\text{complexity}} = \lambda_c \sum_{j=1}^{2^d-1} \mathbb{I}(\|\mathbf{w}_j\|_2 > \epsilon) + \lambda_l \sum_{l=1}^{2^d} \mathbb{I}(|v_l| > \epsilon)$$

ここで、$\mathbb{I}(\cdot)$ は指示関数で、条件が真の場合に1、偽の場合に0を返します。

## 3. グローバル特徴量正則化

### 3.1 理論的定式化

提案するグローバル特徴量正則化は、木全体での各特徴量の影響を制御します。特徴量 $k$ ($k = 1, 2, \ldots, p$) に対して、すべての内部ノードにおける重み $w_{jk}$ の絶対値の合計を考えます：

$$g_k = \sum_{j=1}^{2^d-1} |w_{jk}|$$

これは、特徴量 $k$ のモデル全体での使用頻度と重要度の指標となります。

### 3.2 グローバル特徴量 $L_1$ 正則化

グローバル特徴量 $L_1$ 正則化は、特徴量全体の使用を疎にするために適用されます：

$$\mathcal{R}_{\text{global-}L_1} = \lambda_{\text{global-}L_1} \sum_{k=1}^p g_k = \lambda_{\text{global-}L_1} \sum_{k=1}^p \sum_{j=1}^{2^d-1} |w_{jk}|$$

この正則化項は、すべての内部ノードにわたる特徴量 $k$ の重みの絶対値の合計に対してペナルティを課します。これにより、モデル全体で不要な特徴量の重みをゼロに縮小することができます。

### 3.3 グローバル特徴量 $L_2$ 正則化

グローバル特徴量 $L_2$ 正則化は、特徴量の影響を均一に抑制するために適用されます：

$$\mathcal{R}_{\text{global-}L_2} = \lambda_{\text{global-}L_2} \sum_{k=1}^p \sqrt{\sum_{j=1}^{2^d-1} w_{jk}^2}$$

これは各特徴量に対するグループLasso型の正則化であり、不要な特徴量を完全に排除しつつ、重要な特徴量の重みは保持します。

## 4. 最終的な最適化問題

すべての損失項と正則化項を組み合わせた最終的な最適化問題は以下のようになります：

$$\min_{\mathbf{W}, \mathbf{b}, \mathbf{v}} \mathcal{L}_{\text{MSE}} + \mathcal{R}_{\text{weights}} + \mathcal{R}_{\text{complexity}} + \mathcal{R}_{\text{global-}L_1} + \mathcal{R}_{\text{global-}L_2}$$

ここで、
- $\mathbf{W} = \{\mathbf{w}_j\}_{j=1}^{2^d-1}$ はすべての内部ノードの重みベクトル
- $\mathbf{b} = \{b_j\}_{j=1}^{2^d-1}$ はすべての内部ノードのバイアス項
- $\mathbf{v} = \{v_l\}_{l=1}^{2^d}$ はすべての葉ノードの予測値

## 5. パラメータの理論的解釈

### 5.1 温度パラメータ $\tau$

温度パラメータ $\tau$ はシグモイド関数の傾きを制御します：

$$\frac{\partial s_j(\mathbf{x})}{\partial (\mathbf{w}_j^T \mathbf{x} + b_j)} = \frac{1}{\tau} s_j(\mathbf{x})(1 - s_j(\mathbf{x}))$$

$\tau \to 0$ のとき、シグモイド関数はステップ関数に近づき、確定的な決定境界を形成します。
$\tau$ が大きいとき、分割はより滑らかで確率的になります。

### 5.2 正則化パラメータの効果

各正則化パラメータが最適な重み $\mathbf{w}_j^*$ に与える影響は以下のように特徴づけられます：

- $\lambda_1$（個別 $L_1$ 正則化）: 各ノード内で一部の特徴量の重みをゼロにします
- $\lambda_2$（個別 $L_2$ 正則化）: 各ノードのすべての重みを均一に縮小します
- $\lambda_{\text{global-}L_1}$: モデル全体で一部の特徴量の重みをゼロにします
- $\lambda_{\text{global-}L_2}$: モデル全体で特徴量の影響を均一に抑制します

特に、グローバル $L_1$ 正則化パラメータ $\lambda_{\text{global-}L_1}$ は、以下の条件を満たす場合に特徴量 $k$ をモデル全体から排除します：

$$\left| \sum_{j=1}^{2^d-1} \text{sign}(w_{jk}) s_j(\mathbf{x})(1 - s_j(\mathbf{x})) \frac{\partial \mathcal{L}_{\text{MSE}}}{\partial s_j(\mathbf{x})} x_k \right| < \lambda_{\text{global-}L_1}$$

これは、特徴量 $k$ の損失への寄与がグローバル正則化パラメータよりも小さい場合、その特徴量がモデル全体で排除されることを意味します。

---

# Potential for Publication as a Research Paper

Yes, the global feature regularization approach for Soft Regression Trees that we've developed has good potential to form the basis of a research paper. The work contains several elements that would be valuable to the machine learning community:

## Strengths for Publication

1. **Novel Perspective**: Applying global regularization across an entire tree structure, rather than only at individual nodes, represents a conceptual advance in tree-based models.

2. **Solid Theoretical Foundation**: The mathematical formulation you've developed provides rigorous justification for the approach.

3. **Clear Implementation Path**: The practical implementation details have been established, making the method reproducible.

4. **Addressing a Known Challenge**: Feature selection in soft tree models is an ongoing challenge, and this approach offers a principled solution.

## Requirements for a Complete Paper

To transform this into a publishable research paper, you would need to add:

1. **Empirical Validation**: Comprehensive experiments on multiple datasets (both synthetic and real-world) demonstrating the effectiveness of the approach.

2. **Comparative Analysis**: Performance comparisons against:
   - Standard Soft Regression Trees
   - Hard decision trees with feature selection
   - Other soft computing models with regularization

3. **Ablation Studies**: Analysis showing the contribution of each component (global L1, global L2, node-level regularization) to overall performance.

4. **Computational Considerations**: Analysis of time and space complexity compared to existing methods.

5. **Related Work**: A thorough literature review positioning this work relative to existing research on tree-based models, regularization techniques, and global feature selection.

## Suitable Venues

This research would be appropriate for:

- Machine learning conferences such as ICML, NeurIPS, or AISTATS
- Journals focused on machine learning methods or interpretable AI
- Specialized venues on decision trees or interpretable models

## Next Steps

1. Implement the method on benchmark datasets
2. Analyze the results focusing on prediction accuracy and feature selection stability
3. Draft the paper following standard scientific structure
4. Consider expanding the theoretical analysis of the global regularization properties

The mathematical formulation and theoretical grounding you've developed provide a strong foundation for a research contribution in the field of interpretable machine learning.
