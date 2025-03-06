## 第8章 シミュレーションデータを用いたPythonによる因子分析の実践

この章では、第7章で説明した因子分析のプロセスを、シミュレーションデータを用いてPythonで実際に実行する方法を解説します。具体的には、以下のステップを踏みます。

1. **シミュレーションデータの生成:**  
   潜在因子を生成し、各因子に対する因子負荷量を定めた上で観測変数を構築します。各観測変数は、共通因子の寄与と特有誤差の線形結合として生成されます。

2. **データの前処理:**  
   欠損値処理（シミュレーションデータでは必要ない場合が多いですが）、標準化を行い、変数間のスケールを統一します。

3. **因子分析の実施:**  
   適切な因子抽出法（ここでは最尤法）を用いて、因子負荷行列を推定し、固有値やスクリープロットを作成して因子数を決定します。

4. **回転:**  
   Varimax（直交回転）やPromax（斜交回転）を用いて、解釈しやすい単純構造を実現します。

5. **非正規性・感度分析の考察:**  
   シミュレーションデータでは正規分布に従うデータを生成しますが、コード内でデータ変換やブートストラップを実施する手法の概要も紹介します。

以下に、各ステップの詳細なコード例と解説を示します。

---

### 8.1 シミュレーションデータの生成

まず、3つの潜在因子に基づく12個の観測変数を生成します。各変数は、設定した因子負荷量と独自の誤差を用いて作成されます。

```python
import numpy as np
import pandas as pd

# シードを設定して再現性を確保
np.random.seed(0)
n_samples = 300

# 潜在因子の生成（3因子を仮定）
latent_factors = {
    'F1': np.random.normal(0, 1, n_samples),
    'F2': np.random.normal(0, 1, n_samples),
    'F3': np.random.normal(0, 1, n_samples)
}

# 因子負荷量を辞書で定義（例: 12個の観測変数）
loadings = {
    'Var1': [0.8, 0.1, 0.0],
    'Var2': [0.75, 0.15, 0.05],
    'Var3': [0.7, 0.0, 0.1],
    'Var4': [0.0, 0.8, 0.2],
    'Var5': [0.1, 0.75, 0.0],
    'Var6': [0.05, 0.7, 0.1],
    'Var7': [0.0, 0.2, 0.8],
    'Var8': [0.1, 0.0, 0.75],
    'Var9': [0.05, 0.1, 0.7],
    'Var10': [0.4, 0.4, 0.3],
    'Var11': [0.35, 0.45, 0.25],
    'Var12': [0.3, 0.4, 0.35]
}

# データフレームの作成
data = pd.DataFrame()
for var, load in loadings.items():
    # 各変数を共通因子の線形結合として生成
    common_part = load[0] * latent_factors['F1'] + load[1] * latent_factors['F2'] + load[2] * latent_factors['F3']
    # 特有誤差（平均0, 分散=0.5）
    unique_part = np.random.normal(0, np.sqrt(0.5), n_samples)
    data[var] = common_part + unique_part

# シミュレーションデータの確認
print(data.head())
```

---

### 8.2 データの前処理

次に、各変数のスケールを統一するために標準化を行います。

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_std = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# 標準化後のデータの要約統計量を確認
print(data_std.describe())
```

---

### 8.3 因子抽出の実施

#### 8.3.1 適合性の検定

因子分析を行う前に、バートレットの球面性検定やKMO検定を実施して、データが因子分析に適しているか確認します。

```python
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

chi_square_value, p_value = calculate_bartlett_sphericity(data_std)
kmo_all, kmo_model = calculate_kmo(data_std)

print("Bartlettの球面性検定: chi-square =", chi_square_value, ", p-value =", p_value)
print("KMO検定: KMO =", kmo_model)
```

#### 8.3.2 スクリープロットと因子数の決定

固有値のプロットを用いて、因子数の決定を行います。

```python
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer

fa_temp = FactorAnalyzer(rotation=None)
fa_temp.fit(data_std)
ev, v = fa_temp.get_eigenvalues()

plt.figure(figsize=(8, 4))
plt.scatter(range(1, data_std.shape[1]+1), ev)
plt.plot(range(1, data_std.shape[1]+1), ev, 'b-')
plt.title('スクリープロット')
plt.xlabel('因子数')
plt.ylabel('固有値')
plt.grid(True)
plt.show()
```

このプロットから、固有値が1以上の因子数や折れ曲がり点を参考に因子数を決定します。ここでは例として3因子を採用します。

#### 8.3.3 最尤法による因子抽出と回転

因子抽出を最尤法で行い、Varimax回転（直交回転）を実施します。

```python
# 因子分析の実施：3因子を抽出し、Varimax回転を適用
fa = FactorAnalyzer(n_factors=3, rotation='varimax', method='ml')
fa.fit(data_std)

# 因子負荷行列の表示
loadings_df = pd.DataFrame(fa.loadings_, index=data_std.columns, columns=['Factor1', 'Factor2', 'Factor3'])
print("因子負荷行列（Varimax回転）:")
print(loadings_df)
```

---

### 8.4 非正規性の検討と対処（オプション）

シミュレーションデータは正規分布に基づいて生成されていますが、実際のデータでは非正規性が問題となる場合があります。ここでは、データ変換の例として対数変換の手法を示します。

```python
# 仮に、すべての変数が正の値であると仮定して対数変換を行う場合の例
data_log = data.copy()
for col in data_log.columns:
    # もしデータに0以下の値がある場合は、適当な定数（ここでは1）を足してから変換
    data_log[col] = np.log(data_log[col] + 1)

# 標準化と因子分析の再実施（変数変換後）
data_log_std = pd.DataFrame(scaler.fit_transform(data_log), columns=data.columns)

fa_log = FactorAnalyzer(n_factors=3, rotation='varimax', method='ml')
fa_log.fit(data_log_std)
loadings_log_df = pd.DataFrame(fa_log.loadings_, index=data_log_std.columns, columns=['Factor1', 'Factor2', 'Factor3'])
print("因子負荷行列（対数変換後・Varimax回転）:")
print(loadings_log_df)
```

---

### 8.5 感度分析の実施

#### 8.5.1 ブートストラップ法の実例

ここでは、ブートストラップを用いて因子負荷量の信頼区間を評価する簡単な例を示します。

```python
# 必要なライブラリ
import random

n_boot = 500  # ブートストラップ回数
boot_loadings = []

for i in range(n_boot):
    # 元のデータからブートストラップサンプルを抽出
    boot_indices = np.random.choice(data_std.index, size=len(data_std), replace=True)
    data_boot = data_std.loc[boot_indices]
    
    # 因子分析の実施
    fa_boot = FactorAnalyzer(n_factors=3, rotation='varimax', method='ml')
    fa_boot.fit(data_boot)
    boot_loadings.append(fa_boot.loadings_)

# 例として、最初の変数 (Var1) のFactor1の負荷量の信頼区間を計算
var1_factor1 = [boot[i][0, 0] for boot in boot_loadings]  # boot[i]は配列形式
lower_bound = np.percentile(var1_factor1, 2.5)
upper_bound = np.percentile(var1_factor1, 97.5)
print("Var1のFactor1負荷量の95%信頼区間: [{:.3f}, {:.3f}]".format(lower_bound, upper_bound))
```

#### 8.5.2 サンプル分割法の実例

サンプルを地域や年代などで分割し、各グループで因子分析を行い結果の一貫性を確認します。

```python
# ここでは、シミュレーションのためにランダムに2グループに分割する例を示します
group1 = data_std.sample(frac=0.5, random_state=1)
group2 = data_std.drop(group1.index)

fa_group1 = FactorAnalyzer(n_factors=3, rotation='varimax', method='ml')
fa_group1.fit(group1)
loadings_group1 = pd.DataFrame(fa_group1.loadings_, index=group1.columns, columns=['Factor1', 'Factor2', 'Factor3'])

fa_group2 = FactorAnalyzer(n_factors=3, rotation='varimax', method='ml')
fa_group2.fit(group2)
loadings_group2 = pd.DataFrame(fa_group2.loadings_, index=group2.columns, columns=['Factor1', 'Factor2', 'Factor3'])

print("グループ1の因子負荷行列:")
print(loadings_group1)
print("\nグループ2の因子負荷行列:")
print(loadings_group2)
```

---

### 8.6 結果の解釈とまとめ

シミュレーションデータに基づく因子分析の各ステップを通じて、以下の点が確認されました。

- **前処理:**  
  標準化などの前処理により、変数間のスケールが統一され、因子分析に適した状態となりました。

- **因子抽出と回転:**  
  最尤法とVarimax回転を用いて、解釈しやすい因子負荷行列が得られ、スクリープロットや適合性検定により因子数の妥当性も確認されました。

- **非正規性への対処:**  
  シミュレーションデータは正規性を仮定して生成されましたが、対数変換の例で、実データでの非正規性対策の一端が示されました。

- **感度分析:**  
  ブートストラップ法やサンプル分割法により、因子負荷量の推定結果の安定性を検証する方法が示され、結果のロバスト性を確認する手法として有効であることがわかりました。
