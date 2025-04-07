# 2. データの可視化

データサイエンスツールボックスの開発をデータビジュアリゼーションから始めましょう。データを視覚化することによって、生のデータ値を見ただけでは得られない貴重な洞察が得られます。Pythonでのビジュアリゼーションには、Leland Wilkinsonによって開発された「グラフィックの文法」というデータビジュアリゼーション理論に基づく手法を使用します。

基本的に、グラフィックス/プロット/チャート（この本ではこれらの用語を同じ意味で使用します）は、データのパターンを探索するための優れた方法を提供します。例えば、外れ値の存在、個々の変数の分布、変数グループ間の関係などを調べることができます。グラフィックスは、視聴者に理解してほしい発見や洞察を強調するために設計されています。

## 必要なパッケージをインポートする

まずは、この章で必要なPythonライブラリをインポートしましょう。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import requests
from io import StringIO
from matplotlib.ticker import FuncFormatter

# プロットのスタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['figure.figsize'] = 12, 8
```

## グラフィックの文法

データビジュアリゼーションの理論的なフレームワークである「グラフィックの文法」から始めましょう。このフレームワークは、この章で広く使用するデータビジュアリゼーションの基礎となります。英語で文章を構成するように、異なる要素を組み合わせて統計グラフィックスを構築するための一連のルールを定義します。

### 文法の構成要素

簡単に言えば、この文法は以下のことを示しています：

> 統計グラフィックスは、データ変数を幾何学的オブジェクトの審美的属性に「マッピング」したものである。

具体的には、グラフィックを以下の3つの基本要素に分解できます：

1. `data`：対象となるデータセット
2. `geom`：対象となる幾何学的オブジェクト。プロットで観察できるオブジェクトのタイプを指します。例：点、線、棒など
3. `aes`：幾何学的オブジェクトの審美的属性。例：x/y位置、色、形、サイズなど。審美的属性はデータセット内の変数に「マッピング」されます。

### Gapminderデータ

実例を使ってこの文法を理解しましょう。Hans Roslingは「The best stats you've ever seen」というTEDトークで、世界の経済、健康、開発データを紹介しました。2007年の142カ国のデータを見てみましょう。

まず、Gapminderデータをダウンロードします：

```python
# Gapminderデータをダウンロード
!pip install gapminder
from gapminder import gapminder

# 2007年のデータだけをフィルタリング
gapminder_2007 = gapminder[gapminder['year'] == 2007].drop('year', axis=1)
gapminder_2007 = gapminder_2007.rename(columns={
    'country': 'Country',
    'continent': 'Continent',
    'lifeExp': 'Life Expectancy',
    'pop': 'Population',
    'gdpPercap': 'GDP per Capita'
})

# 最初の3カ国を表示
gapminder_2007.head(3)
```

各行はこのテーブルの国を表し、5つの列があります：

1. **Country**：国名
2. **Continent**：国が属する大陸（「Americas」は北米と南米の両方を含む）
3. **Life Expectancy**：平均寿命（年）
4. **Population**：国の人口
5. **GDP per Capita**：一人当たりGDP（米ドル）

このデータをプロットしてみましょう：

```python
# スケーリングのために人口の大きさを調整
size_scaling = gapminder_2007['Population'] / 1000000

# カラーマッピングのための大陸のカテゴリカルデータを準備
continents = gapminder_2007['Continent'].unique()
color_map = dict(zip(continents, sns.color_palette("Set1", len(continents))))
colors = [color_map[c] for c in gapminder_2007['Continent']]

# プロットの作成
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    x=gapminder_2007['GDP per Capita'],
    y=gapminder_2007['Life Expectancy'],
    s=size_scaling,  # サイズを人口に基づいてスケーリング
    c=colors,        # 色を大陸に基づいて設定
    alpha=0.7
)

# 凡例を追加
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             label=continent,
                             markerfacecolor=color_map[continent], 
                             markersize=10)
                  for continent in continents]
plt.legend(handles=legend_elements, title='Continent')

# ラベルとタイトルを設定
plt.xlabel('GDP per capita')
plt.ylabel('Life expectancy')
plt.title('Life expectancy vs GDP per capita (2007)')

# プロットを表示
plt.show()
```

この図で文法に基づいて解釈すると：

1. `data`変数**GDP per Capita**が点の`x`位置にマッピングされます
2. `data`変数**Life Expectancy**が点の`y`位置にマッピングされます
3. `data`変数**Population**が点の`size`（大きさ）にマッピングされます
4. `data`変数**Continent**が点の`color`（色）にマッピングされます

`geom`要素は点（ポイント）です。

## 5つの基本グラフ - 5NG

この本では、5つの基本的なグラフ（5NG）に焦点を当てます：

1. 散布図 (scatterplots)
2. 線グラフ (linegraphs)
3. ヒストグラム (histograms)
4. 箱ひげ図 (boxplots)
5. 棒グラフ (barplots)

これらの基本的なグラフィックを使えば、様々なタイプの変数を視覚化できます。

## 5NG#1: 散布図

まず、最もシンプルな散布図から始めましょう。散布図は2つの数値変数間の関係を視覚化する方法です。NYCフライトデータセットを使用して、出発遅延と到着遅延の関係を調べてみましょう。

```python
# NYCフライトデータをダウンロード
url = "https://raw.githubusercontent.com/tomoshige/website/refs/heads/main/docs/lectures/SIWS/datasets/flights.csv"
response = requests.get(url)
data = StringIO(response.text)
flights = pd.read_csv(data)

# Envoy Air（コード: MQ）のフライトだけをフィルタリング
envoy_flights = flights[flights['carrier'] == 'MQ']

# 散布図を作成
plt.figure(figsize=(10, 6))
plt.scatter(envoy_flights['dep_delay'], envoy_flights['arr_delay'], alpha=0.2)
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Arrival Delay (minutes)')
plt.title('Arrival Delays vs. Departure Delays for Envoy Air Flights (2023)')
plt.grid(True, alpha=0.3)
plt.show()
```

この散布図から、出発遅延と到着遅延の間には正の関係があることがわかります。出発が遅れると、到着も遅れる傾向があります。また、(0,0)付近の点の集積が多いことも観察できます。これは定刻通りに出発・到着したフライトが多いことを示しています。

### オーバープロッティングへの対処

図に見られる(0,0)付近の大量の点は、「オーバープロッティング」と呼ばれる現象を引き起こします。これは、点が互いに重なって表示され、実際のデータポイントの数が見えにくくなる問題です。

オーバープロッティングに対処するには2つの方法があります：

1. 点の透明度を調整する
2. 各点にランダムな「ジッター」（小さなランダム移動）を追加する

**方法1: 透明度の変更**

```python
plt.figure(figsize=(10, 6))
plt.scatter(envoy_flights['dep_delay'], envoy_flights['arr_delay'], alpha=0.2)
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Arrival Delay (minutes)')
plt.title('Arrival Delays vs. Departure Delays (alpha = 0.2)')
plt.grid(True, alpha=0.3)
plt.show()
```

**方法2: ジッターの追加**

```python
# ジッターを加えた散布図
plt.figure(figsize=(10, 6))
# ランダムなジッターを追加
jitter_x = np.random.uniform(-15, 15, size=len(envoy_flights))
jitter_y = np.random.uniform(-15, 15, size=len(envoy_flights))

plt.scatter(
    envoy_flights['dep_delay'] + jitter_x, 
    envoy_flights['arr_delay'] + jitter_y, 
    alpha=0.2
)
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Arrival Delay (minutes)')
plt.title('Arrival Delays vs. Departure Delays (with jitter)')
plt.grid(True, alpha=0.3)
plt.show()
```

この例では、透明度を調整することで重なり具合がよく見えますが、どちらの方法が良いかは状況によって異なります。目的に合わせて両方試してみるのが良いでしょう。

## 5NG#2: 線グラフ

次に線グラフについて見ていきましょう。線グラフは、X軸にある変数（説明変数）が時間のような順序性を持つ場合に特に有効です。

ニューヨーク市の気象データを使って、ニューアーク空港の1月の最初の15日間における風速の時系列を調べてみましょう。

```python
# 気象データをダウンロード
url = "https://raw.githubusercontent.com/tomoshige/website/refs/heads/main/docs/lectures/SIWS/datasets/weather.csv"
response = requests.get(url)
data = StringIO(response.text)
weather = pd.read_csv(data)

# ニューアーク空港(EWR)の1月の最初の15日間のデータをフィルタリング
early_january_2023_weather = weather[
    (weather['origin'] == 'EWR') & 
    (weather['month'] == 1) & 
    (weather['day'] <= 15)
]

# 日時のデータフォーマットを修正
early_january_2023_weather['time_hour'] = pd.to_datetime(early_january_2023_weather['time_hour'], format='%Y-%m-%d %H:%M:%S') 
```

実はこれではエラーが出る。この原因について考えてみよう。ちなみに、解消するためのコードは以下の通り。生成AIなどを用いて、このコードの意味を解き明かしてください。

```python
# prompt: early_january_2023_weather['time_hour'] は、"2023-01-03 20:00:00" のように時刻が記載されている場合と、"2023-01-03" のように時刻が記載されていない場合があります。"2023-01-03" のように時刻が記載されていない場合は、" 00:00:00"を追加し、"2023-01-03 00:00:00"と修正してください

# 時刻データの修正
early_january_2023_weather['time_hour'] = early_january_2023_weather['time_hour'].astype(str)
early_january_2023_weather['time_hour'] = early_january_2023_weather['time_hour'].str.replace(r'(\d{4}-\d{2}-\d{2})$', r'\1 00:00:00', regex=True)
early_january_2023_weather['time_hour'] = pd.to_datetime(early_january_2023_weather['time_hour'])
```

```python
# 線グラフを作成
plt.figure(figsize=(12, 6))
plt.plot(early_january_2023_weather['time_hour'], early_january_2023_weather['wind_speed'])
plt.xlabel('Date and Time')
plt.ylabel('Wind Speed (mph)')
plt.title('Hourly Wind Speed at Newark Airport (Jan 1-15, 2023)')
plt.grid(True, alpha=0.3)
plt.show()
```



線グラフは、時間経過による変数値の変化を視覚的に追跡するのに最適です。この例では、ニューアーク空港の風速が時間とともにどのように変化するかを見ることができます。

## 5NG#3: ヒストグラム

次は、単一の数値変数の分布を視覚化する方法としてヒストグラムを紹介します。例えば、ニューヨーク市の3つの空港における風速の分布を調べてみましょう。

```python
# 風速のヒストグラムを作成
plt.figure(figsize=(10, 6))
plt.hist(weather['wind_speed'].dropna(), bins=30, color='steelblue', edgecolor='white')
plt.xlabel('Wind Speed (mph)')
plt.ylabel('Frequency')
plt.title('Histogram of Hourly Wind Speeds at Three NYC Airports')
plt.grid(True, alpha=0.3)
plt.show()
```

ヒストグラムは、値の範囲をビンに分割し、各ビンに入るデータの数を高さとして表示します。これにより：

1. 最小値と最大値がわかる
2. 「中心」または「最も典型的な」値がわかる
3. 値の広がり方がわかる
4. 頻繁に出現する値と稀な値が識別できる

### ビンの調整

ヒストグラムのビン（区間）の数や幅を調整することで、データの見え方が変わります：

```python
# ビンを調整したヒストグラムを並べて表示
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# ビン数を20に設定
ax[0].hist(weather['wind_speed'].dropna(), bins=20, color='steelblue', edgecolor='white')
ax[0].set_xlabel('Wind Speed (mph)')
ax[0].set_ylabel('Frequency')
ax[0].set_title('With 20 bins')
ax[0].grid(True, alpha=0.3)

# ビン幅を5に設定
ax[1].hist(weather['wind_speed'].dropna(), bins=range(0, 45, 5), color='steelblue', edgecolor='white')
ax[1].set_xlabel('Wind Speed (mph)')
ax[1].set_ylabel('Frequency')
ax[1].set_title('With binwidth = 5 mph')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## ファセット

次に、ファセット（面分割）という概念を紹介します。ファセットは、別の変数の値によって視覚化を分割したい場合に使用します。

例えば、風速のヒストグラムを月ごとに分けて表示してみましょう：

```python
# 月ごとの風速ヒストグラムを作成
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

for i, month in enumerate(range(1, 13)):
    month_data = weather[weather['month'] == month]
    axes[i].hist(month_data['wind_speed'].dropna(), bins=range(0, 45, 5), 
                 color='steelblue', edgecolor='white')
    axes[i].set_title(f'Month {month}')
    axes[i].set_xlim(0, 40)
    # Y軸の範囲を統一
    axes[i].set_ylim(0, 1100)
    
    # x軸ラベルは最下段のグラフだけに表示
    if i >= 9:
        axes[i].set_xlabel('Wind Speed (mph)')
    
    # y軸ラベルは左端のグラフだけに表示
    if i % 3 == 0:
        axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

この分割表示により、月ごとの風速分布の違いを比較できます。すべての月で0～20 mph の間に大半の観測値があり、30 mph を超える観測値は非常に少ないことがわかります。

## 5NG#4: 箱ひげ図

箱ひげ図は、数値変数の分布を別の変数のカテゴリごとに比較する際に役立ちます。箱ひげ図は「五数要約」（最小値、第1四分位数、中央値、第3四分位数、最大値）に基づいて構築されます。

まず、4月の風速データに基づいて箱ひげ図の構造を理解しましょう：

```python
# 4月の風速データをフィルタリング
april_data = weather[(weather['month'] == 4) & (~weather['wind_speed'].isna())]

# 五数要約を計算
min_apr = april_data['wind_speed'].min()
q1 = april_data['wind_speed'].quantile(0.25)
median = april_data['wind_speed'].quantile(0.5)
q3 = april_data['wind_speed'].quantile(0.75)
max_apr = april_data['wind_speed'].max()

print(f"五数要約:")
print(f"最小値: {min_apr}")
print(f"第1四分位数: {q1:.1f}")
print(f"中央値: {median:.1f}")
print(f"第3四分位数: {q3:.1f}")
print(f"最大値: {max_apr}")

# 箱ひげ図の構造を示す図を作成
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# 1. 点と水平線
ax1.scatter(np.zeros(len(april_data)), april_data['wind_speed'], alpha=0.1, jitter=0.2)
for val in [min_apr, q1, median, q3, max_apr]:
    ax1.axhline(y=val, color='black', linestyle='--')
ax1.set_title('Jittered Points with Five-Number Summary')
ax1.set_ylim(0, max_apr + 5)
ax1.set_xticklabels([])

# 2. 箱ひげ図と点、水平線
ax2.boxplot(april_data['wind_speed'], showfliers=False)
ax2.scatter(np.ones(len(april_data)) + np.random.normal(0, 0.05, size=len(april_data)), 
           april_data['wind_speed'], alpha=0.1)
for val in [min_apr, q1, median, q3, max_apr]:
    ax2.axhline(y=val, color='black', linestyle='--')
ax2.set_title('Boxplot with Points and Five-Number Summary')
ax2.set_ylim(0, max_apr + 5)
ax2.set_xticklabels(['April'])

# 3. 箱ひげ図のみ
ax3.boxplot(april_data['wind_speed'])
ax3.set_title('Boxplot Only')
ax3.set_ylim(0, max_apr + 5)
ax3.set_xticklabels(['April'])

plt.show()
```

箱ひげ図は、データの四分位数で分割します：

1. 箱の下端は第1四分位数（25パーセンタイル）
2. 箱の中の実線は中央値（50パーセンタイル）
3. 箱の上端は第3四分位数（75パーセンタイル）
4. ひげは箱から外れたデータの範囲を示し、通常は箱の長さの1.5倍までの範囲を表示
5. それを超える値は外れ値として点で表示

次に、月ごとの風速の分布を比較する箱ひげ図を作成しましょう：

```python
# 月ごとの風速の箱ひげ図
plt.figure(figsize=(14, 8))
# factorプロットのため、月を文字列に変換
weather['month_factor'] = weather['month'].astype(str)
sns.boxplot(x='month_factor', y='wind_speed', data=weather)
plt.xlabel('Month')
plt.ylabel('Wind Speed (mph)')
plt.title('Side-by-side Boxplot of Wind Speed Split by Month')
plt.grid(True, alpha=0.3)
plt.show()
```

この箱ひげ図から、2月と3月の中央値風速が他の月よりも高いことがわかります。また、箱の高さ（四分位範囲）からは、各月の風速の変動性も読み取れます。

## 5NG#5: 棒グラフ

最後に、カテゴリ変数の分布を視覚化するための棒グラフを見ていきましょう。

まず、データが表現されている方法によって2つの異なるアプローチが必要になることを理解しましょう：

```python
# 果物データの2つの表現方法
fruits = pd.DataFrame({
    'fruit': ['apple', 'apple', 'orange', 'apple', 'orange']
})

fruits_counted = pd.DataFrame({
    'fruit': ['apple', 'orange'],
    'number': [3, 2]
})

print("fruits:")
print(fruits)
print("\nfruits_counted:")
print(fruits_counted)
```

カテゴリデータの表現方法に応じて、異なる方法で棒グラフを作成します：

```python
# 1. 集計されていないデータの棒グラフ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

fruits['fruit'].value_counts().plot.bar(ax=ax1)
ax1.set_title('Using value_counts() for non-pre-counted data')
ax1.set_xlabel('Fruit')
ax1.set_ylabel('Count')

# 2. 集計済みデータの棒グラフ
fruits_counted.plot.bar(x='fruit', y='number', ax=ax2)
ax2.set_title('Using pre-counted data')
ax2.set_xlabel('Fruit')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.show()
```

次に、NYCからの航空会社別の出発便数を可視化してみましょう：

```python
# 航空会社別の出発便数の棒グラフ
plt.figure(figsize=(12, 6))
flights['carrier'].value_counts().plot.bar()
plt.xlabel('Carrier')
plt.ylabel('Number of Flights')
plt.title('Number of Flights Departing NYC in 2023 by Airline')
plt.grid(True, alpha=0.3)
plt.show()
```

### 2つのカテゴリ変数の関係

棒グラフは、2つのカテゴリ変数の同時分布を視覚化するのにも役立ちます。例えば、航空会社と出発空港の関係を調べてみましょう：

```python
# 航空会社と出発空港による積み上げ棒グラフ
plt.figure(figsize=(14, 7))
carrier_origin = pd.crosstab(flights['carrier'], flights['origin'])
carrier_origin.plot.bar(stacked=True)
plt.xlabel('Carrier')
plt.ylabel('Number of Flights')
plt.title('Number of Flights by Carrier and Origin (Stacked)')
plt.legend(title='Origin Airport')
plt.grid(True, alpha=0.3)
plt.show()
```

積み上げ棒グラフは単純ですが、特定の比較が難しいこともあります。並べて表示する方法もあります：

```python
# 航空会社と出発空港による並べて表示する棒グラフ
plt.figure(figsize=(14, 7))
pd.crosstab(flights['carrier'], flights['origin']).plot.bar()
plt.xlabel('Carrier')
plt.ylabel('Number of Flights')
plt.title('Number of Flights by Carrier and Origin (Side-by-side)')
plt.legend(title='Origin Airport')
plt.grid(True, alpha=0.3)
plt.show()
```

また、ファセット分割した棒グラフも有効です：

```python
# 出発空港ごとにファセット分割した棒グラフ
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

for i, origin in enumerate(['EWR', 'JFK', 'LGA']):
    origin_data = flights[flights['origin'] == origin]
    origin_data['carrier'].value_counts().plot.bar(ax=axes[i])
    axes[i].set_title(f'Origin: {origin}')
    axes[i].set_xlabel('Carrier')
    axes[i].set_ylabel('Number of Flights')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 円グラフを避ける理由

カテゴリデータの分布を視覚化するために一般的に使用される円グラフには実は問題があります。人間は角度を正確に判断することが難しいからです。角度が90度より大きいと過大評価し、90度より小さいと過小評価します。

棒グラフは単一の水平線で比較できるため、円グラフよりも効果的です。

## 結論

この章では、データビジュアリゼーションの基本的な手法として5つの基本グラフ（5NG）を学びました：

1. 散布図：2つの数値変数の関係を示す
2. 線グラフ：時系列などの順序性のあるデータの変化を示す
3. ヒストグラム：単一の数値変数の分布を示す
4. 箱ひげ図：数値変数の分布をカテゴリごとに比較する
5. 棒グラフ：カテゴリ変数の頻度を示す

これらの基本的なグラフを組み合わせることで、データの様々な側面を視覚化し、洞察を得ることができます。

### 追加リソース

Pythonでのデータビジュアリゼーションについてさらに学ぶには、以下のリソースが役立ちます：

- Matplotlib公式ドキュメント: https://matplotlib.org/
- Seaborn公式ドキュメント: https://seaborn.pydata.org/
- Pandas可視化ガイド: https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html