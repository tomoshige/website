# 第8章: ブートストラップと信頼区間

## 8.1 はじめに

この章では、統計的推論の第三部分のブートストラップと信頼区間について学びます。前の章ではサンプリングについて学び、母集団から標本を抽出することで母集団パラメータを推定する方法を見てきました。

第7章では、赤と白のボールが入った大きなボウルからシャベルを使って標本を抽出し、ボウル全体の赤いボールの割合を推定しました。そのとき、複数回のサンプリングを行い、サンプリングによる変動の影響を研究しました。しかし現実では、通常は単一の標本しか取らず、複数回のサンプリングは行いません。

実際には、単一の標本から母集団パラメータの信頼区間を構築する方法が必要になります。それがこの章のテーマです。

オバマ大統領支持率に関する世論調査を例にとると、記事では「調査の誤差の範囲はプラスマイナス2.1パーセントポイントだった」と述べられていました。これは信頼区間の一例で、サンプリング変動による誤差の範囲を示しています。

この章ではブートストラップリサンプリングと信頼区間について学び、これらを用いて単一の標本からどのようにサンプリング変動の影響を評価できるかを見ていきます。

## 8.2 ペニー（硬貨）のサンプリング活動

実践的な活動として、2019年にアメリカで流通しているペニー（1セント硬貨）の平均製造年は何年かを考えてみましょう。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import io

# GitHub上のModernDiveデータを読み込むためのURLを設定
github_url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/"

# ペニーのサンプルデータを読み込む
pennies_sample_url = github_url + "pennies_sample.csv"
pennies_sample = pd.read_csv(pennies_sample_url)

# データの最初の数行を表示
pennies_sample.head()
```

`pennies_sample`データフレームには50枚のペニーに関する情報が含まれています。各行が1枚のペニーに対応し、`ID`と`year`（製造年）の2つの変数があります。

このサンプルから、アメリカで流通しているすべてのペニーについて何が言えるでしょうか？探索的データ分析を行い、サンプルのプロパティを調査してみましょう。

```python
# ペニーの製造年のヒストグラムを作成
plt.figure(figsize=(10, 6))
plt.hist(pennies_sample['year'], bins=range(1960, 2020, 10), edgecolor='white')
plt.title('50枚のペニーの製造年の分布')
plt.xlabel('製造年')
plt.ylabel('頻度')
plt.tight_layout()
plt.show()
```

このヒストグラムから、ほとんどのペニーが1980年から2010年の間に製造されており、1970年以前の古いペニーはごく少数であることがわかります。サンプル内のペニーの平均製造年を計算してみましょう。

```python
# ペニーの平均製造年を計算
x_bar = pennies_sample['year'].mean()
print(f"サンプルのペニーの平均製造年: {x_bar:.2f}")
```

このサンプルから、アメリカで流通しているすべてのペニーの平均製造年の推定値は約1995年です。これは、私たちの調査対象である「母平均」μの点推定値です。

サンプリングに関する統計的概念を整理すると：

- **母集団**: 調査対象となる全体集団（この場合はアメリカで流通しているすべてのペニー）
- **母集団パラメータ**: 知りたい未知の値（この場合は母集団の平均製造年μ）
- **標本統計量/点推定量**: 標本から計算される推定値（この場合は標本平均x̄）

しかし、サンプリングには変動があります。別の50枚のペニーを銀行から入手したら、同じ平均年が得られるでしょうか？おそらく違う値になるでしょう。このサンプリング変動の影響を、単一のサンプルから研究するにはどうすればよいでしょうか？

## 8.3 ブートストラップリサンプリング

**ブートストラップリサンプリング**を行うことで、単一のサンプルからサンプリング変動の影響を研究できます。手順を見ていきましょう。

### 手作業でのリサンプリング手順

1. 50枚のペニーそれぞれを表す紙片を用意します
2. 紙片を帽子に入れます
3. ランダムに1枚の紙片を引き、年を記録します
4. 引いた紙片を**元に戻します**（置換）
5. ステップ3と4を合計50回繰り返し、50個の年の値を得ます

このプロセスは、元のサンプルから**置換を伴うリサンプリング**を行っています。この「ブートストラップサンプル」の平均値を計算すると、元のサンプル平均とは異なる値が得られるでしょう。

```python
# 手作業でのリサンプリングを再現（実際のデータをシミュレーション）
np.random.seed(123)  # 結果を再現できるように乱数シードを設定
pennies_resample = pd.DataFrame({
    'year': np.random.choice(pennies_sample['year'], size=50, replace=True)
})

# リサンプルの平均を計算
resample_mean = pennies_resample['year'].mean()
print(f"リサンプルのペニーの平均製造年: {resample_mean:.2f}")
print(f"元のサンプルの平均製造年: {x_bar:.2f}")
print(f"差: {resample_mean - x_bar:.2f}")
```

元のサンプルを35回リサンプリングしたという仮想のシナリオを考えてみましょう。これは、35人の友人がそれぞれリサンプリングを行ったと想像できます。このデータをシミュレーションしましょう。

```python
# 35回のリサンプリングをシミュレーション
np.random.seed(456)
resampled_means = []
names = [f"friend_{i+1}" for i in range(35)]

# 各友人のリサンプルと平均値を保存
pennies_resamples = pd.DataFrame()
for name in names:
    # 置換を伴うリサンプリング
    resample = np.random.choice(pennies_sample['year'], size=50, replace=True)
    # 各リサンプルのデータを記録
    temp_df = pd.DataFrame({'name': [name] * 50, 'year': resample})
    pennies_resamples = pd.concat([pennies_resamples, temp_df])
    # 平均値を計算
    resampled_means.append({'name': name, 'mean_year': resample.mean()})

# 35の平均値をデータフレームに変換
resampled_means_df = pd.DataFrame(resampled_means)

# 平均値のヒストグラムを作成
plt.figure(figsize=(10, 6))
plt.hist(resampled_means_df['mean_year'], bins=range(1990, 2001, 1), edgecolor='white')
plt.title('35回のリサンプリングから得られた平均年のヒストグラム')
plt.xlabel('平均製造年')
plt.ylabel('頻度')
plt.tight_layout()
plt.show()
```

このヒストグラムは、「ブートストラップ分布」と呼ばれます。1992年から2000年の間に平均値が集中しており、分布はおおよそ1995年を中心としています。これは元のサンプル平均に近い値です。この分布を使って、サンプリング変動が推定値に与える影響を研究できます。

## 8.4 コンピュータによるリサンプリングシミュレーション

Pythonを使用して、リサンプリングのプロセスをより効率的に行い、多数のブートストラップサンプルを生成できます。

```python
# 1回のブートストラップサンプリング
np.random.seed(42)
virtual_resample = np.random.choice(pennies_sample['year'], size=50, replace=True)
virtual_resample_mean = virtual_resample.mean()
print(f"バーチャルリサンプルの平均: {virtual_resample_mean:.2f}")
```

次に、35回のリサンプリングを行い、元の手作業の結果と比較します。

```python
# 35回のブートストラップリサンプリング
np.random.seed(123)
virtual_resamples = np.array([
    np.random.choice(pennies_sample['year'], size=50, replace=True).mean()
    for _ in range(35)
])

# 結果のヒストグラムを作成
plt.figure(figsize=(10, 6))
plt.hist(virtual_resamples, bins=range(1990, 2001, 1), edgecolor='white')
plt.title('35回のバーチャルリサンプリングから得られた平均年のヒストグラム')
plt.xlabel('平均製造年')
plt.ylabel('頻度')
plt.tight_layout()
plt.show()
```

より信頼性の高い結果を得るために、リサンプリングの回数を1000回に増やしましょう。

```python
# 1000回のブートストラップリサンプリング
np.random.seed(789)
virtual_resamples_1000 = np.array([
    np.random.choice(pennies_sample['year'], size=50, replace=True).mean()
    for _ in range(1000)
])

# 結果のヒストグラムを作成
plt.figure(figsize=(10, 6))
plt.hist(virtual_resamples_1000, bins=range(1990, 2001, 1), edgecolor='white')
plt.title('1000回のブートストラップリサンプリングから得られた平均年の分布')
plt.xlabel('平均製造年')
plt.ylabel('頻度')
plt.tight_layout()
plt.show()

# ブートストラップ分布の平均を計算
mean_of_means = virtual_resamples_1000.mean()
print(f"1000個のブートストラップ平均の平均: {mean_of_means:.2f}")
print(f"元のサンプル平均: {x_bar:.2f}")
```

1000回のリサンプリングから得られたブートストラップ分布は、より滑らかで正規分布に近い形状になっています。この分布を使って、母平均μの信頼区間を構築できます。

## 8.5 信頼区間の理解

「点推定値」は単一の値で母集団パラメータを推定しますが、「信頼区間」は妥当な値の範囲を提供します。これは漁師のたとえで言うと、「槍」ではなく「網」を使うようなものです。

信頼区間を構築する方法には、**パーセンタイル法**と**標準誤差法**の2つの主要なアプローチがあります。

### パーセンタイル法

パーセンタイル法では、ブートストラップ分布の中央95%の値を取ります。これは2.5パーセンタイルと97.5パーセンタイルの間の値です。

```python
# パーセンタイル法による95%信頼区間
percentile_ci = np.percentile(virtual_resamples_1000, [2.5, 97.5])
print(f"パーセンタイル法による95%信頼区間: ({percentile_ci[0]:.2f}, {percentile_ci[1]:.2f})")

# 信頼区間をヒストグラム上に表示
plt.figure(figsize=(10, 6))
plt.hist(virtual_resamples_1000, bins=range(1990, 2001, 1), edgecolor='white')
plt.axvline(x=percentile_ci[0], color='black', linestyle='-', linewidth=2)
plt.axvline(x=percentile_ci[1], color='black', linestyle='-', linewidth=2)
plt.title('パーセンタイル法による95%信頼区間')
plt.xlabel('平均製造年')
plt.ylabel('頻度')
plt.tight_layout()
plt.show()
```

### 標準誤差法

標準誤差法は、ブートストラップ分布が正規分布に従う場合に適用できます。これは、元のサンプル平均を中心に、ブートストラップ標準誤差の約1.96倍（95%信頼区間の場合）の範囲を取ります。

```python
# ブートストラップ標準誤差を計算
bootstrap_se = np.std(virtual_resamples_1000)
print(f"ブートストラップ標準誤差: {bootstrap_se:.4f}")

# 標準誤差法による95%信頼区間
se_ci = [x_bar - 1.96 * bootstrap_se, x_bar + 1.96 * bootstrap_se]
print(f"標準誤差法による95%信頼区間: ({se_ci[0]:.2f}, {se_ci[1]:.2f})")

# 両方の信頼区間をヒストグラム上に表示
plt.figure(figsize=(10, 6))
plt.hist(virtual_resamples_1000, bins=range(1990, 2001, 1), edgecolor='white')
plt.axvline(x=percentile_ci[0], color='black', linestyle='-', linewidth=2, label='パーセンタイル法')
plt.axvline(x=percentile_ci[1], color='black', linestyle='-', linewidth=2)
plt.axvline(x=se_ci[0], color='red', linestyle='--', linewidth=2, label='標準誤差法')
plt.axvline(x=se_ci[1], color='red', linestyle='--', linewidth=2)
plt.title('ブートストラップ分布と95%信頼区間の比較')
plt.xlabel('平均製造年')
plt.ylabel('頻度')
plt.legend()
plt.tight_layout()
plt.show()
```

両方の方法による信頼区間が概ね似ていることがわかります。しかし、標準誤差法はブートストラップ分布が正規分布に近い場合にのみ適切です。

## 8.6 Pythonによる信頼区間の構築

Pythonでは、ブートストラップリサンプリングと信頼区間の計算を体系的に行うために、`bootstrap`関数を作成できます。

```python
def bootstrap_ci(data, n_resamples=1000, ci_level=0.95, statistic=np.mean):
    """
    ブートストラップ法による信頼区間を計算する関数
    
    Parameters:
    -----------
    data : array-like
        元のサンプルデータ
    n_resamples : int, default 1000
        ブートストラップサンプルの数
    ci_level : float, default 0.95
        信頼水準（0から1の間）
    statistic : function, default np.mean
        計算する統計量の関数
        
    Returns:
    --------
    tuple : (bootstrap_distribution, lower_ci, upper_ci)
        ブートストラップ分布と信頼区間の下限・上限
    """
    # ブートストラップリサンプリング
    bootstrap_distribution = np.array([
        statistic(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_resamples)
    ])
    
    # パーセンタイル法による信頼区間
    alpha = 1 - ci_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    percentile_ci = np.percentile(bootstrap_distribution, [lower_percentile, upper_percentile])
    
    # 標準誤差法による信頼区間
    point_estimate = statistic(data)
    se = np.std(bootstrap_distribution)
    z_critical = stats.norm.ppf(1 - alpha / 2)  # 95%信頼区間の場合は約1.96
    se_ci = (point_estimate - z_critical * se, point_estimate + z_critical * se)
    
    return {
        'bootstrap_distribution': bootstrap_distribution,
        'percentile_ci': percentile_ci,
        'se_ci': se_ci,
        'point_estimate': point_estimate,
        'standard_error': se
    }
```

この関数を使って、ペニーデータの信頼区間を計算してみましょう。

```python
# ペニーデータに対する95%信頼区間
np.random.seed(42)
bootstrap_results = bootstrap_ci(pennies_sample['year'], n_resamples=1000, ci_level=0.95)

# 結果の表示
print(f"点推定値 (サンプル平均): {bootstrap_results['point_estimate']:.2f}")
print(f"ブートストラップ標準誤差: {bootstrap_results['standard_error']:.4f}")
print(f"パーセンタイル法による95%信頼区間: ({bootstrap_results['percentile_ci'][0]:.2f}, {bootstrap_results['percentile_ci'][1]:.2f})")
print(f"標準誤差法による95%信頼区間: ({bootstrap_results['se_ci'][0]:.2f}, {bootstrap_results['se_ci'][1]:.2f})")

# 結果の可視化
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_results['bootstrap_distribution'], bins=30, edgecolor='white', alpha=0.7)
plt.axvline(x=bootstrap_results['percentile_ci'][0], color='blue', linestyle='-', linewidth=2, label='パーセンタイル法CI')
plt.axvline(x=bootstrap_results['percentile_ci'][1], color='blue', linestyle='-', linewidth=2)
plt.axvline(x=bootstrap_results['se_ci'][0], color='red', linestyle='--', linewidth=2, label='標準誤差法CI')
plt.axvline(x=bootstrap_results['se_ci'][1], color='red', linestyle='--', linewidth=2)
plt.axvline(x=bootstrap_results['point_estimate'], color='green', linestyle='-', linewidth=2, label='サンプル平均')
plt.title('ペニーの平均製造年のブートストラップ分布と95%信頼区間')
plt.xlabel('平均製造年')
plt.ylabel('頻度')
plt.legend()
plt.tight_layout()
plt.show()
```

## 8.7 信頼区間の解釈

信頼区間の正確な統計的解釈は少し長くなります：「サンプリング手順を多数回繰り返した場合、得られる信頼区間の約95%が母集団パラメータの真の値を捕捉する」

これは信頼区間構築手順の信頼性に関する記述です。単一の信頼区間に対しては、「95%の確率で母集団パラメータを含む」という解釈は厳密には正しくありません。なぜなら、特定の信頼区間は母集団パラメータを含むか含まないかのどちらかだからです。

実用的な短い解釈は：「95%信頼区間は、母集団パラメータの真の値を捕捉する95%の『確信』を持つ区間である」

ペニーの例では、「2019年に流通しているすべてのペニーの真の平均製造年は約1991年から2000年の間である」と95%の確信を持って言えます。

### 信頼区間の幅に影響する要因

1. **信頼水準**: 信頼水準が高いほど、信頼区間は広くなります（例：99%信頼区間は95%信頼区間より広い）

```python
# 異なる信頼水準の比較
confidence_levels = [0.80, 0.95, 0.99]
ci_results = []

for level in confidence_levels:
    result = bootstrap_ci(pennies_sample['year'], n_resamples=1000, ci_level=level)
    ci_width = result['percentile_ci'][1] - result['percentile_ci'][0]
    ci_results.append({
        'confidence_level': f"{level*100}%",
        'lower_ci': result['percentile_ci'][0],
        'upper_ci': result['percentile_ci'][1],
        'width': ci_width
    })

# 結果をデータフレームに変換
ci_comparison = pd.DataFrame(ci_results)
print(ci_comparison)

# 可視化
plt.figure(figsize=(10, 6))
for i, row in enumerate(ci_results):
    plt.plot([row['lower_ci'], row['upper_ci']], [i, i], 'o-', linewidth=2, markersize=8, 
             label=f"{row['confidence_level']} CI (幅={row['width']:.2f})")
plt.axvline(x=bootstrap_results['point_estimate'], color='green', linestyle='--', linewidth=2, label='サンプル平均')
plt.yticks(range(len(confidence_levels)), [f"{level*100}%" for level in confidence_levels])
plt.title('異なる信頼水準の信頼区間の比較')
plt.xlabel('平均製造年')
plt.ylabel('信頼水準')
plt.legend(loc='upper right')
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()
```

2. **サンプルサイズ**: サンプルサイズが大きいほど、信頼区間は狭くなります

```python
# サンプルサイズの影響をシミュレーション
np.random.seed(42)
sample_sizes = [25, 50, 100]
sample_results = []

# 仮想的な大きな母集団を作成
population_years = np.random.normal(1995, 15, size=10000).astype(int)

for n in sample_sizes:
    # サンプルを取得
    sample = np.random.choice(population_years, size=n, replace=False)
    result = bootstrap_ci(sample, n_resamples=1000, ci_level=0.95)
    ci_width = result['percentile_ci'][1] - result['percentile_ci'][0]
    sample_results.append({
        'sample_size': f"n = {n}",
        'lower_ci': result['percentile_ci'][0],
        'upper_ci': result['percentile_ci'][1],
        'width': ci_width,
        'point_estimate': result['point_estimate']
    })

# 結果をデータフレームに変換
sample_comparison = pd.DataFrame(sample_results)
print(sample_comparison)

# 可視化
plt.figure(figsize=(10, 6))
for i, row in enumerate(sample_results):
    plt.plot([row['lower_ci'], row['upper_ci']], [i, i], 'o-', linewidth=2, markersize=8,
             label=f"{row['sample_size']} (幅={row['width']:.2f})")
    plt.plot(row['point_estimate'], i, 'rx', markersize=10)
plt.yticks(range(len(sample_sizes)), [f"n = {n}" for n in sample_sizes])
plt.title('異なるサンプルサイズの95%信頼区間の比較')
plt.xlabel('平均製造年')
plt.ylabel('サンプルサイズ')
plt.legend(loc='upper right')
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()
```

## 8.8 ケーススタディ：あくびは伝染するか？

Mythbustersのテレビ番組で行われた実験を分析するケーススタディを見てみましょう。

```python
# Mythbustersのあくびデータを読み込む
mythbusters_url = github_url + "mythbusters_yawn.csv"
mythbusters_yawn = pd.read_csv(mythbusters_url)

# データの概要を確認
print(mythbusters_yawn.head())
print("\nデータの要約:")
print(mythbusters_yawn.groupby(['group', 'yawn']).size().unstack(fill_value=0))
```

このデータセットでは、50人の参加者が「seed」グループ（あくびを見せられる）と「control」グループ（あくびを見せられない）に分けられました。実験の目的は、あくびを見ることであくびが誘発されるかどうかを調べることです。

```python
# 各グループのあくび率を計算
control_yawn_rate = mythbusters_yawn[mythbusters_yawn['group'] == 'control']['yawn'].value_counts(normalize=True).get('yes', 0)
seed_yawn_rate = mythbusters_yawn[mythbusters_yawn['group'] == 'seed']['yawn'].value_counts(normalize=True).get('yes', 0)
diff_proportion = seed_yawn_rate - control_yawn_rate

print(f"Control group yawn rate: {control_yawn_rate:.4f} ({control_yawn_rate*100:.1f}%)")
print(f"Seed group yawn rate: {seed_yawn_rate:.4f} ({seed_yawn_rate*100:.1f}%)")
print(f"Difference (seed - control): {diff_proportion:.4f} ({diff_proportion*100:.1f}%)")
```

二項比率の差に対する信頼区間を計算するためにブートストラップ法を使用します。

```python
def bootstrap_diff_proportions(data, group_col, outcome_col, success_val, group1, group2, n_resamples=1000, ci_level=0.95):
    """
    二つのグループ間の比率の差に対するブートストラップ信頼区間を計算
    
    Parameters:
    -----------
    data : pandas DataFrame
        元のデータセット
    group_col : str
        グループを表す列名
    outcome_col : str
        結果を表す列名
    success_val : any
        成功とみなす値
    group1, group2 : any
        比較する2つのグループの値
    n_resamples : int
        ブートストラップ反復回数
    ci_level : float
        信頼水準
        
    Returns:
    --------
    dict
        ブートストラップ結果を含む辞書
    """
    # 点推定値を計算
    prop1 = data[data[group_col] == group1][outcome_col].eq(success_val).mean()
    prop2 = data[data[group_col] == group2][outcome_col].eq(success_val).mean()
    point_estimate = prop1 - prop2
    
    # ブートストラップリサンプリング
    bootstrap_diffs = []
    for _ in range(n_resamples):
        # 置換を伴うリサンプリング
        resample = data.sample(n=len(data), replace=True)
        # 各グループの比率を計算
        resample_prop1 = resample[resample[group_col] == group1][outcome_col].eq(success_val).mean()
        resample_prop2 = resample[resample[group_col] == group2][outcome_col].eq(success_val).mean()
        # 差を保存
        bootstrap_diffs.append(resample_prop1 - resample_prop2)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # パーセンタイル法による信頼区間
    alpha = 1 - ci_level
    percentile_ci = np.percentile(bootstrap_diffs, [alpha/2*100, (1-alpha/2)*100])
    
    # 標準誤差法による信頼区間
    se = np.std(bootstrap_diffs)
    z_critical = stats.norm.ppf(1 - alpha/2)
    se_ci = (point_estimate - z_critical * se, point_estimate + z_critical * se)
    
    return {
        'bootstrap_distribution': bootstrap_diffs,
        'point_estimate': point_estimate,
        'percentile_ci': percentile_ci,
        'se_ci': se_ci,
        'standard_error': se
    }
```

MythbustersのあくびデータにこのブートストラップLを適用しましょう。

```python
np.random.seed(42)
yawn_bootstrap = bootstrap_diff_proportions(
    mythbusters_yawn, 
    group_col='group', 
    outcome_col='yawn', 
    success_val='yes',
    group1='seed', 
    group2='control', 
    n_resamples=1000, 
    ci_level=0.95
)

# 結果の表示
print(f"点推定値 (seed - control): {yawn_bootstrap['point_estimate']:.4f}")
print(f"パーセンタイル法による95%信頼区間: ({yawn_bootstrap['percentile_ci'][0]:.4f}, {yawn_bootstrap['percentile_ci'][1]:.4f})")
print(f"標準誤差法による95%信頼区間: ({yawn_bootstrap['se_ci'][0]:.4f}, {yawn_bootstrap['se_ci'][1]:.4f})")
print(f"ブートストラップ標準誤差: {yawn_bootstrap['standard_error']:.4f}")

# 結果の可視化
plt.figure(figsize=(10, 6))
plt.hist(yawn_bootstrap['bootstrap_distribution'], bins=30, edgecolor='white')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='差なし (0)')
plt.axvline(x=yawn_bootstrap['point_estimate'], color='green', linestyle='-', linewidth=2, label='観測された差')
plt.axvline(x=yawn_bootstrap['percentile_ci'][0], color='blue', linestyle='-', linewidth=2, label='95%信頼区間')
plt.axvline(x=yawn_bootstrap['percentile_ci'][1], color='blue', linestyle='-', linewidth=2)
plt.title('あくびの伝染: 比率の差のブートストラップ分布 (seed - control)')
plt.xlabel('比率の差 (seed - control)')
plt.ylabel('頻度')
plt.legend()
plt.tight_layout()
plt.show()
```

95%信頼区間は0を含んでいることから、あくびを見せられたグループとそうでないグループの間のあくび率に統計的に有意な差があるという証拠はないことがわかります。つまり、この実験からは「あくびは伝染する」という仮説を支持する強い証拠は得られませんでした。

## 8.9 ブートストラップ分布とサンプリング分布の比較

ブートストラップ分布とサンプリング分布の関係を理解することは重要です。

```python
# ボウルデータを読み込む
bowl_url = github_url + "bowl.csv"
bowl = pd.read_csv(bowl_url)

# 赤いボールの真の割合を計算
p_red = (bowl['color'] == 'red').mean()
print(f"ボウル内の赤いボールの真の割合: {p_red:.4f}")

# サンプリング分布の生成（理想的な状況のシミュレーション）
np.random.seed(76)
sampling_distribution = []
for _ in range(1000):
    sample = bowl.sample(n=50)
    p_hat = (sample['color'] == 'red').mean()
    sampling_distribution.append(p_hat)

# サンプリング分布の標準誤差
sampling_se = np.std(sampling_distribution)
print(f"サンプリング分布の標準誤差: {sampling_se:.4f}")

# 単一サンプルの取得
np.random.seed(42)
bowl_sample = bowl.sample(n=50)
p_hat_sample = (bowl_sample['color'] == 'red').mean()
print(f"サンプルの赤いボールの割合: {p_hat_sample:.4f}")

# ブートストラップ分布の生成（現実的な状況）
bootstrap_distribution = []
for _ in range(1000):
    bootstrap_sample = bowl_sample.sample(n=50, replace=True)
    p_hat_bootstrap = (bootstrap_sample['color'] == 'red').mean()
    bootstrap_distribution.append(p_hat_bootstrap)

# ブートストラップ分布の標準誤差
bootstrap_se = np.std(bootstrap_distribution)
print(f"ブートストラップ分布の標準誤差: {bootstrap_se:.4f}")

# 理論上の標準誤差
theoretical_se = np.sqrt(p_hat_sample * (1 - p_hat_sample) / 50)
print(f"理論上の標準誤差: {theoretical_se:.4f}")

# 両方の分布を比較するプロット
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.hist(sampling_distribution, bins=20, alpha=0.7, color='salmon', edgecolor='white')
plt.axvline(x=p_red, color='red', linestyle='-', linewidth=2, label=f'真の割合 p = {p_red:.4f}')
plt.title('サンプリング分布（多数のサンプルから得られる理想的なケース）')
plt.xlabel('サンプル割合')
plt.ylabel('頻度')
plt.legend()

plt.subplot(2, 1, 2)
plt.hist(bootstrap_distribution, bins=20, alpha=0.7, color='skyblue', edgecolor='white')
plt.axvline(x=p_hat_sample, color='blue', linestyle='--', linewidth=2, 
            label=f'元のサンプル割合 p̂ = {p_hat_sample:.4f}')
plt.title('ブートストラップ分布（単一サンプルから得られる現実的なケース）')
plt.xlabel('ブートストラップサンプル割合')
plt.ylabel('頻度')
plt.legend()

plt.tight_layout()
plt.show()

# 標準誤差の比較
se_comparison = pd.DataFrame({
    'Method': ['Sampling Distribution', 'Bootstrap Distribution', 'Theoretical Formula'],
    'Standard Error': [sampling_se, bootstrap_se, theoretical_se]
})
print("\n標準誤差の比較:")
print(se_comparison)
```

このシミュレーションから以下の重要な点がわかります：

1. **ブートストラップ分布はサンプリング分布と同じ中心を持たない** - ブートストラップ分布は元のサンプル統計量（この場合はp̂）を中心としますが、サンプリング分布は真の母数値（p）を中心とします。

2. **ブートストラップ分布はサンプリング分布と同様の形状と広がりを持つ** - 両方の分布の標準誤差は非常に似ており、理論式による推定値とも近いです。

3. **ブートストラップ法は推定の精度を直接向上させることはできないが、標準誤差の良い推定値を提供する** - これにより、単一サンプルからでも信頼区間を構築できます。

## 8.10 理論に基づく信頼区間

サンプリング分布が正規分布に従う場合、数学的公式を使用して信頼区間を構築することもできます。例えば、比率pの95%信頼区間は以下のように計算できます：

```python
def theoretical_proportion_ci(p_hat, n, ci_level=0.95):
    """
    比率の理論的信頼区間を計算
    
    Parameters:
    -----------
    p_hat : float
        サンプル比率
    n : int
        サンプルサイズ
    ci_level : float
        信頼水準
        
    Returns:
    --------
    tuple
        信頼区間の下限と上限
    """
    alpha = 1 - ci_level
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    # 標準誤差
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    
    # 誤差の範囲
    margin_of_error = z_critical * se
    
    # 信頼区間
    lower_ci = max(0, p_hat - margin_of_error)  # 0未満にならないように
    upper_ci = min(1, p_hat + margin_of_error)  # 1を超えないように
    
    return (lower_ci, upper_ci, margin_of_error)
```

例として、ボウルのサンプルにこの理論式を適用してみましょう。

```python
# 理論に基づく信頼区間
theory_ci, theory_ci_upper, margin_of_error = theoretical_proportion_ci(p_hat_sample, n=50)
print(f"理論に基づく95%信頼区間: ({theory_ci:.4f}, {theory_ci_upper:.4f})")
print(f"誤差の範囲: ±{margin_of_error:.4f}")

# Mythbustersの世論調査の例
print("\nオバマ支持率の世論調査の例:")
obama_p_hat = 0.41
obama_n = 2089
obama_ci_lower, obama_ci_upper, obama_moe = theoretical_proportion_ci(obama_p_hat, obama_n)
print(f"サンプルサイズ: {obama_n}")
print(f"サンプル比率: {obama_p_hat:.2f} ({obama_p_hat*100:.0f}%)")
print(f"誤差の範囲: ±{obama_moe:.4f} (±{obama_moe*100:.1f}%)")
print(f"95%信頼区間: ({obama_ci_lower:.4f}, {obama_ci_upper:.4f}) または ({obama_ci_lower*100:.1f}%, {obama_ci_upper*100:.1f}%)")
```

## 8.11 結論

信頼区間は、母集団パラメータの妥当な値の範囲を提供するための強力なツールです。この章では、以下の重要な概念を学びました：

1. **ブートストラップリサンプリング** - 単一サンプルからサンプリング変動の影響を研究する方法
2. **信頼区間の構築** - パーセンタイル法と標準誤差法による信頼区間の構築方法
3. **信頼区間の解釈** - 信頼区間の正確な統計的解釈と実用的な短い解釈
4. **信頼区間の幅に影響する要因** - 信頼水準とサンプルサイズの影響
5. **ブートストラップ分布とサンプリング分布の関係** - 両者の類似点と相違点

信頼区間についての理解を深めるために、さまざまなデータセットに対して信頼区間を構築し、解釈する練習をしてみることをお勧めします。

以下のPythonコードは、この章の主要な機能をまとめたものです：

```python
def bootstrap_analysis(data, statistic=np.mean, n_resamples=1000, ci_level=0.95):
    """
    完全なブートストラップ分析を実行
    
    Parameters:
    -----------
    data : array-like
        分析するデータ
    statistic : function
        計算する統計量
    n_resamples : int
        ブートストラップ反復回数
    ci_level : float
        信頼水準
        
    Returns:
    --------
    dict
        結果を含む辞書
    """
    # 点推定値
    point_estimate = statistic(data)
    
    # ブートストラップリサンプリング
    bootstrap_stats = []
    for _ in range(n_resamples):
        resample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(resample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # 標準誤差
    se = np.std(bootstrap_stats)
    
    # パーセンタイル法CI
    alpha = 1 - ci_level
    percentile_ci = np.percentile(bootstrap_stats, [alpha/2*100, (1-alpha/2)*100])
    
    # 標準誤差法CI
    z_critical = stats.norm.ppf(1 - alpha/2)
    se_ci = (point_estimate - z_critical * se, point_estimate + z_critical * se)
    
    # 理論に基づく標準誤差（適用可能な場合）
    if statistic == np.mean:
        theoretical_se = np.std(data) / np.sqrt(len(data))
    elif statistic == lambda x: np.mean(x == 1) or statistic == lambda x: np.mean(x == True):
        p_hat = np.mean(data)
        theoretical_se = np.sqrt(p_hat * (1 - p_hat) / len(data))
    else:
        theoretical_se = None
    
    # 結果を可視化
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_stats, bins=30, edgecolor='white', alpha=0.7)
    plt.axvline(x=point_estimate, color='green', linestyle='-', linewidth=2, label='点推定値')
    plt.axvline(x=percentile_ci[0], color='blue', linestyle='-', linewidth=2, label=f'{ci_level*100}%パーセンタイルCI')
    plt.axvline(x=percentile_ci[1], color='blue', linestyle='-', linewidth=2)
    plt.axvline(x=se_ci[0], color='red', linestyle='--', linewidth=2, label=f'{ci_level*100}%標準誤差CI')
    plt.axvline(x=se_ci[1], color='red', linestyle='--', linewidth=2)
    plt.title('ブートストラップ分布と信頼区間')
    plt.xlabel('統計量の値')
    plt.ylabel('頻度')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 結果を返す
    return {
        'point_estimate': point_estimate,
        'bootstrap_distribution': bootstrap_stats,
        'standard_error': se,
        'theoretical_se': theoretical_se,
        'percentile_ci': percentile_ci,
        'se_ci': se_ci,
        'ci_level': ci_level
    }
```

次の章では、信頼区間と密接に関連する統計的推論のもう一つの重要なツールである**仮説検定**について学びます。