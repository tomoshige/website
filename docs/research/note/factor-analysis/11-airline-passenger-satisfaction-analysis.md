## 第11章 順序カテゴリカル変数を対象とした因子分析の実践  
**（Airline Passenger Satisfaction データセットを用いたポリコリック・ポリセリック相関ベースの解析）**

本章では、Airline Passenger Satisfaction に関する実データ（Kaggle提供）を用いて、順序のあるカテゴリカル変数（8列〜24列：満足度に関する項目）に対して、ポリコリック相関（および必要に応じてポリセリック相関）を利用した因子分析を実施します。  
解析の流れは以下の通りです。

1. **データの読み込みと前処理**  
   - 必要なライブラリのインポート  
   - データセットの読み込み  
   - 対象となる順序データ（列8〜24）の抽出と前処理

2. **ポリコリック相関行列の算出**  
   - Python の `semopy` モジュールの `polychoric_corr` 関数を用いて、順序変数同士の相関行列を推定

3. **因子抽出と回転**  
   - 得られたポリコリック相関行列を入力として、因子分析を実施（ここでは最尤法を使用）
   - 直交回転（Varimax）と斜交回転（Promax）の両方の結果を比較
   - 固有値プロット（スクリープロット）による因子数の決定も行う

4. **感度分析（ブートストラップ）**  
   - ブートストラップ法を用いて、推定された因子負荷量の安定性（信頼区間）を評価

5. **結果の可視化と解釈**  
   - ヒートマップや因子間相関行列のプロットなどを用いて、結果を視覚化  
   - 得られた因子構造の解釈および、Varimax と Promax の違いについて議論

以下、各ステップの Python コードと解説を示します。

### 11.1. データの読み込みと前処理

まず、必要なライブラリをインポートし、データセットを読み込みます。  
Airline Passenger Satisfaction データセットは Kaggle からダウンロードしてください。  
満足度に関するデータは 8 列目から 24 列目に該当します（ここでは仮に列名が `Q1` ～ `Q17` として扱います）。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from semopy import polychoric_corr
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler

# データセットの読み込み（ファイル名を適宜調整）
data = pd.read_csv("Airline_Passenger_Satisfaction.csv")

# データの概要を確認
print(data.head())
print(data.columns)

# 満足度に関する列（例: 8列目から24列目）
# 実際のデータセットの列名に合わせて調整してください
ordinal_cols = data.columns[7:24]  # Pythonのインデックスは0始まりのため、7〜23列目
data_ord = data[ordinal_cols].copy()

# 各列のデータ型や分布の確認（順序尺度であることを確認）
print(data_ord.dtypes)
data_ord.describe()
```

*【解説】*  
- データセットから満足度に関する列を抽出します。  
- これらの変数は順序カテゴリカルデータ（例：非常に不満〜非常に満足）として扱われる前提です。

### 11.2. ポリコリック相関行列の算出

`semopy` の `polychoric_corr` 関数を使用して、順序変数間のポリコリック相関行列を推定します。

```python
# semopy の polychoric_corr を用いて、ポリコリック相関行列を算出
# polychoric_corr は、順序データのポリコリック相関行列と閾値の推定も行います。
poly_corr_matrix, thresholds = polychoric_corr(data_ord)

# ポリコリック相関行列の確認
poly_corr_df = pd.DataFrame(poly_corr_matrix, index=ordinal_cols, columns=ordinal_cols)
print("ポリコリック相関行列:")
print(poly_corr_df)

# ヒートマップによる可視化
plt.figure(figsize=(10,8))
sns.heatmap(poly_corr_df, annot=True, cmap='coolwarm')
plt.title("ポリコリック相関行列のヒートマップ")
plt.show()
```

*【解説】*  
- `polychoric_corr` は、順序変数同士の背後にある連続潜在変数の相関を最尤法などで推定し、ポリコリック相関行列を返します。  
- この行列は、因子分析の入力として利用可能です。

### 11.3. 因子抽出と回転

#### 11.3.1 因子数の決定（スクリープロット）

ポリコリック相関行列を用いて、因子数の決定を行います。  
因子分析には、`FactorAnalyzer` を用い、`is_corr_matrix=True` を指定して相関行列を入力します。

```python
# FactorAnalyzer を用いて、ポリコリック相関行列から因子分析を実施するために、
# is_corr_matrix パラメータを True に設定
fa_temp = FactorAnalyzer(rotation=None, is_corr_matrix=True)
fa_temp.fit(poly_corr_matrix)

# 固有値を取得
ev, v = fa_temp.get_eigenvalues()

plt.figure(figsize=(8,4))
plt.scatter(range(1, len(ev)+1), ev)
plt.plot(range(1, len(ev)+1), ev, 'b-')
plt.title('スクリープロット（ポリコリック相関行列）')
plt.xlabel('因子数')
plt.ylabel('固有値')
plt.axhline(1, color='red', linestyle='--')
plt.grid(True)
plt.show()
```

*【解説】*  
- スクリープロットから、固有値が1以上となる因子数、または折れ曲がり点を基に因子数を決定します。  
- この例では仮に 3 因子を採用します（データに応じて調整してください）。

#### 11.3.2 因子分析の実施（Varimax と Promax の比較）

##### 11.3.2.1 Varimax 回転（直交回転）の場合

```python
# Varimax回転を用いて因子分析（3因子、最尤法、相関行列入力）
fa_varimax = FactorAnalyzer(n_factors=3, rotation='varimax', method='ml', is_corr_matrix=True)
fa_varimax.fit(poly_corr_matrix)

# 因子負荷行列の取得
loadings_varimax = pd.DataFrame(fa_varimax.loadings_, index=ordinal_cols,
                                  columns=['Factor1', 'Factor2', 'Factor3'])
print("因子負荷行列（Varimax回転）:")
print(loadings_varimax)

# ヒートマップで可視化
plt.figure(figsize=(10,8))
sns.heatmap(loadings_varimax, annot=True, cmap='coolwarm')
plt.title("因子負荷行列（Varimax回転）")
plt.show()
```

##### 11.3.2.2 Promax 回転（斜交回転）の場合

```python
# Promax回転を用いて因子分析（3因子、最尤法、相関行列入力）
fa_promax = FactorAnalyzer(n_factors=3, rotation='promax', method='ml', is_corr_matrix=True)
fa_promax.fit(poly_corr_matrix)

# 因子負荷行列の取得
loadings_promax = pd.DataFrame(fa_promax.loadings_, index=ordinal_cols,
                                 columns=['Factor1', 'Factor2', 'Factor3'])
print("因子負荷行列（Promax回転）:")
print(loadings_promax)

# 因子間の相関行列（Promax回転の場合）
phi = fa_promax.get_factor_correlation()
phi_df = pd.DataFrame(phi, index=['Factor1', 'Factor2', 'Factor3'], columns=['Factor1', 'Factor2', 'Factor3'])
print("\n因子間相関行列（Promax回転）:")
print(phi_df)

# ヒートマップで可視化
plt.figure(figsize=(10,8))
sns.heatmap(loadings_promax, annot=True, cmap='coolwarm')
plt.title("因子負荷行列（Promax回転）")
plt.show()
```

*【解説】*  
- Varimax 回転は因子間の独立性を保持し、各変数がどの因子に強く寄与しているかが明確です。  
- Promax 回転は因子間の相関を許容するため、より柔軟なモデルが得られ、実際のデータ構造を反映しやすい場合があります。  
- 因子間相関行列（phi_df）を確認し、各因子の関連性を検討します。

---

### 11.4. 感度分析（ブートストラップ法）

ブートストラップ法を用いて、因子負荷量の推定結果の安定性を評価します。ここでは、例として「Q1」（ordinal_cols の先頭の変数）の Factor1 の負荷量に対する 95% 信頼区間を求めます。

```python
n_boot = 500  # ブートストラップ回数
boot_loadings = []

for i in range(n_boot):
    # ブートストラップサンプル（相関行列の再計算は必要な場合があるが、ここではデータの再抽出と再計算の流れを示す）
    boot_indices = np.random.choice(data_ord.index, size=len(data_ord), replace=True)
    data_boot = data_ord.loc[boot_indices]
    
    # ブートストラップサンプルに対してポリコリック相関行列の算出
    boot_poly_corr, _ = polychoric_corr(data_boot)
    
    # 因子分析（Varimax回転）を実施（同じ3因子として）
    fa_boot = FactorAnalyzer(n_factors=3, rotation='varimax', method='ml', is_corr_matrix=True)
    fa_boot.fit(boot_poly_corr)
    
    boot_loadings.append(fa_boot.loadings_)

# 対象変数: ordinal_cols[0]（例: Q1）および Factor1
q1_idx = 0  # ordinal_cols の最初の変数
factor1_values = [boot[q1_idx, 0] for boot in boot_loadings]

lower_bound = np.percentile(factor1_values, 2.5)
upper_bound = np.percentile(factor1_values, 97.5)
print("Q1のFactor1負荷量の95%信頼区間: [{:.3f}, {:.3f}]".format(lower_bound, upper_bound))
```

*【解説】*  
- ブートストラップ法により、因子負荷量の安定性や信頼区間を推定します。  
- 結果が狭い信頼区間に収まれば、推定結果が安定していると判断できます。

### 11.5. 結果の解釈

- **因子負荷行列:**  
  それぞれの回転法（Varimax, Promax）で得られた因子負荷行列を元に、各因子がどの満足度項目に強く寄与しているかを検討します。  
  例えば、もしある因子が「チェックインサービス」や「機内サービス」などの項目に高い負荷を示している場合、その因子は「サービス品質」を反映していると解釈できます。

- **因子間相関:**  
  Promax 回転の場合、因子間の相関が示されるため、関連性の強い因子同士は統合的に解釈する必要があるかもしれません。

- **感度分析:**  
  ブートストラップ法の結果、各変数の因子負荷量の信頼区間を確認することで、推定結果の安定性を評価します。安定した結果が得られていれば、モデルの信頼性が高いと判断できます。

### 11.6. まとめ

本章では、Airline Passenger Satisfaction データセットの順序カテゴリカル変数（満足度に関する8〜24列）に対し、以下のプロセスで因子分析を実施しました。

1. **データ前処理:**  
   対象変数の抽出と前処理（必要に応じた欠損値処理など）

2. **ポリコリック相関行列の算出:**  
   `semopy.polychoric_corr` を用いて、順序データ間の相関行列を推定

3. **因子抽出と回転:**  
   ポリコリック相関行列を入力とし、因子分析を最尤法で実施。Varimax（直交回転）と Promax（斜交回転）の結果を比較

4. **感度分析:**  
   ブートストラップ法を用いて、因子負荷量の安定性（信頼区間）を評価

5. **結果の解釈:**  
   得られた因子構造を理論的背景と照らし合わせ、各因子の意味付けを行い、ヒートマップや因子間相関行列を用いて視覚化

これらの手順を通じて、順序のあるカテゴリカル変数に対する因子分析の手法と、その結果の解釈方法を実践的に学ぶことができました。読者はこの Demo を参考に、自身の順序データに対してもポリコリック・ポリセリック相関を利用した因子分析を適用し、実務に役立てることができるでしょう。