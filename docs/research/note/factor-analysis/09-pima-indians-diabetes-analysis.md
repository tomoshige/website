## 第9章 医療データ（Pima Indians Diabetes Database）を用いた因子分析の実践

この章では、Pima Indians Diabetes Database（医療データ）を用いて、第8章でシミュレーションデータで実施した因子分析の一連の手続き（前処理、因子抽出、回転、非正規性の確認、感度分析）を実際に行います。また、直交回転（Varimax）と斜交回転（Promax）の結果の違いや、ブートストラップによる感度分析も実例を通して解説します。


### 1. ライブラリのインポートとデータの読み込み

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.preprocessing import StandardScaler

# データファイルの読み込み
# KaggleからダウンロードしたCSVファイル名を指定（例："diabetes.csv"）
data = pd.read_csv("diabetes.csv")

# データの最初の数行を確認
print(data.head())
```


### 2. データの前処理

#### 2.1 欠損値の確認と補正
Pima Indians Diabetes Databaseでは、Glucose、BloodPressure、SkinThickness、Insulin、BMI の各変数に「0」が不自然な値として含まれていることが知られています。これらを欠損値とみなし、中央値で補完します。

```python
# 補正対象のカラム
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 各対象カラムで0をNaNに変換
for col in cols_with_zero:
    data[col] = data[col].replace(0, np.nan)

# 各カラムの欠損値数を確認
print("欠損値の数:")
print(data.isnull().sum())

# 各カラムの中央値で補完
for col in cols_with_zero:
    median_val = data[col].median()
    data[col].fillna(median_val, inplace=True)

# 補完後の欠損値確認
print("\n補完後の欠損値数:")
print(data.isnull().sum())
```

#### 2.2 データの分布確認（非正規性の検討）

各変数のヒストグラムとQ-Qプロットを作成し、分布の歪みや裾の重さを確認します。

```python
# ヒストグラムの描画
data.hist(bins=20, figsize=(12, 10))
plt.suptitle("各変数のヒストグラム")
plt.show()

# Q-Qプロットの作成（例としてGlucose変数）
plt.figure(figsize=(6,4))
stats.probplot(data['Glucose'], dist="norm", plot=plt)
plt.title("GlucoseのQ-Qプロット")
plt.show()
```


### 3. データの標準化

因子分析では各変数のスケールが統一されることが望ましいため、標準化を行います。

```python
scaler = StandardScaler()
data_std = pd.DataFrame(scaler.fit_transform(data.drop('Outcome', axis=1)), 
                        columns=data.columns[:-1])
# Outcome変数は解析の対象外（クラスラベル）として扱う
print(data_std.head())
```

---

### 4. 因子抽出の実施

#### 4.1 適合性の検定

因子分析に適しているかを確認するため、バートレットの球面性検定とKMO検定を行います。

```python
chi_square_value, p_value = calculate_bartlett_sphericity(data_std)
kmo_all, kmo_model = calculate_kmo(data_std)

print("Bartlettの球面性検定: chi-square =", chi_square_value, ", p-value =", p_value)
print("KMO検定: KMO =", kmo_model)
```

#### 4.2 スクリープロットと因子数の決定

固有値を確認し、因子数を決定します（この例では、3因子を仮定）。

```python
fa_temp = FactorAnalyzer(rotation=None)
fa_temp.fit(data_std)
ev, v = fa_temp.get_eigenvalues()

plt.figure(figsize=(8,4))
plt.scatter(range(1, data_std.shape[1]+1), ev)
plt.plot(range(1, data_std.shape[1]+1), ev, 'b-')
plt.title('スクリープロット')
plt.xlabel('因子数')
plt.ylabel('固有値')
plt.axhline(1, color='r', linestyle='--')
plt.grid(True)
plt.show()
```

### 5. 因子分析と回転の実施

#### 5.1 Varimax回転（直交回転）

```python
# 3因子で因子分析を実施（最尤法、Varimax回転）
fa_varimax = FactorAnalyzer(n_factors=3, rotation='varimax', method='ml')
fa_varimax.fit(data_std)

# 因子負荷行列の表示
loadings_varimax = pd.DataFrame(fa_varimax.loadings_, index=data_std.columns,
                                  columns=['Factor1', 'Factor2', 'Factor3'])
print("因子負荷行列（Varimax回転）:")
print(loadings_varimax)

# ヒートマップによる可視化
plt.figure(figsize=(8,6))
sns.heatmap(loadings_varimax, annot=True, cmap='coolwarm')
plt.title("因子負荷行列（Varimax回転）")
plt.show()
```

#### 5.2 Promax回転（斜交回転）

```python
# 同じく3因子でPromax回転を実施
fa_promax = FactorAnalyzer(n_factors=3, rotation='promax', method='ml')
fa_promax.fit(data_std)

# 因子負荷行列の表示
loadings_promax = pd.DataFrame(fa_promax.loadings_, index=data_std.columns,
                                 columns=['Factor1', 'Factor2', 'Factor3'])
print("因子負荷行列（Promax回転）:")
print(loadings_promax)

# 因子間の相関（Promaxの場合）
phi = fa_promax.get_factor_correlation()
print("\n因子間相関行列（Promax回転）:")
print(pd.DataFrame(phi, columns=['Factor1', 'Factor2', 'Factor3'], index=['Factor1', 'Factor2', 'Factor3']))

# ヒートマップによる可視化
plt.figure(figsize=(8,6))
sns.heatmap(loadings_promax, annot=True, cmap='coolwarm')
plt.title("因子負荷行列（Promax回転）")
plt.show()
```

**【解説】**  
- Varimax回転では因子間が独立となるため、各変数がどの因子に主に寄与しているかが明確になります。  
- Promax回転では因子間の相関を許容し、実際のデータ構造により柔軟に対応できます。因子間相関行列を確認することで、因子同士の関連性も理解できます。

### 6. 感度分析（ブートストラップ法）

ブートストラップ法を用いて、因子負荷量の推定結果の安定性を評価します。ここでは、例として「Glucose」のFactor1に対する負荷量の95%信頼区間を計算します。

```python
n_boot = 500  # ブートストラップ回数
boot_loadings = []

for i in range(n_boot):
    # ブートストラップサンプルを抽出（置換あり）
    boot_indices = np.random.choice(data_std.index, size=len(data_std), replace=True)
    data_boot = data_std.loc[boot_indices]
    
    # 因子分析（Varimax回転）を実施
    fa_boot = FactorAnalyzer(n_factors=3, rotation='varimax', method='ml')
    fa_boot.fit(data_boot)
    boot_loadings.append(fa_boot.loadings_)

# 例: 'Glucose'（データ_stdの1列目と仮定）のFactor1の負荷量の信頼区間
# 列名が 'Glucose' であれば、そのインデックスを利用
glucose_idx = data_std.columns.get_loc('Glucose')
factor1_loadings = [boot[glucose_idx, 0] for boot in boot_loadings]

lower_bound = np.percentile(factor1_loadings, 2.5)
upper_bound = np.percentile(factor1_loadings, 97.5)
print("GlucoseのFactor1負荷量の95%信頼区間: [{:.3f}, {:.3f}]".format(lower_bound, upper_bound))
```

### 7. 結果の解釈と可視化

#### 7.1 結果の比較

VarimaxとPromaxの因子負荷行列を比較して、各変数がどの因子にどの程度寄与しているかを確認します。場合によっては、解釈上どちらの回転法が理論や目的に合致しているかを検討します。

#### 7.2 可視化

- **スクリープロット:** 既に因子数決定のために作成しました。
- **ヒートマップ:** VarimaxおよびPromaxの因子負荷行列のヒートマップを表示しました。
- **因子間相関:** Promax回転の因子間相関行列も表示し、因子の関連性を確認できます。

#### 7.3 解釈のポイント

- 例えば、もし「Glucose」や「BMI」などが、ある因子に高い負荷を示す場合、その因子は「代謝」や「インスリン感受性」に関連していると解釈できるかもしれません。
- 各変数の理論的背景と、得られた因子構造との整合性を検討し、因子の命名を行います。

---

### 8. まとめ

本章では、Pima Indians Diabetes Databaseを用いて実際の医療データに対する因子分析の一連のプロセスを、以下のステップで実施しました。

1. **データの読み込みと前処理:**  
   欠損値の補完、不要な値の処理、標準化を実施しました。

2. **分布の確認:**  
   ヒストグラムとQ-Qプロットを用いて、各変数の分布（非正規性の可能性）を確認しました。

3. **因子抽出と因子数の決定:**  
   バートレット検定、KMO検定、スクリープロットにより解析の適合性と因子数を検証しました。

4. **回転:**  
   Varimax（直交回転）とPromax（斜交回転）の両手法で因子負荷行列を推定し、その違いを比較しました。

5. **感度分析:**  
   ブートストラップ法を用いて因子負荷量の信頼区間を評価し、推定結果のロバスト性を確認しました。

6. **結果の解釈と可視化:**  
   ヒートマップや因子間相関行列などを用いて、得られた因子構造を視覚的に解釈しました。
