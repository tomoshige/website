# データラングリング

これまでの旅では、第1章で`glimpse()`や`View()`関数を使ってデータフレームを調べる方法と、第2章で`ggplot2`パッケージを使ったデータ可視化の方法について学びました。特に「5つの基本グラフ」を勉強しました：

1. 散布図（`geom_point()`）
2. 折れ線グラフ（`geom_line()`）
3. 箱ひげ図（`geom_boxplot()`）
4. ヒストグラム（`geom_histogram()`）
5. 棒グラフ（`geom_bar()`または`geom_col()`）

これらの可視化はグラフィックス文法を使って作成しました。データフレーム内の変数を5つの幾何学的オブジェクトのいずれかの審美的属性にマッピングします。Gapminderデータの例（第2章の図）のように、サイズや色などの幾何学的オブジェクトの他の審美的属性も制御できます。

本章では、pandasライブラリのデータラングリング関数を紹介します。これらの関数を使えば、データフレームを変換して目的に合わせることができます。主な関数は：

1. 条件に合う行を選択する
2. 列を要約統計量で集計する
3. 行をグループ化する。グループ化と集計を組み合わせると、グループごとに別々の要約統計量を計算できる
4. 既存の列から新しい列を作成する
5. 行を並べ替える
6. 「キー」変数に基づいて別のデータフレームと結合する

この章では、これらの操作をPythonのpandasライブラリで行う方法を学びます。

pandasを学ぶもう一つの利点は、データベース照会言語の[SQL](https://en.wikipedia.org/wiki/SQL)との類似性です。SQLは大規模なデータベースを迅速かつ効率的に管理するために使用され、多くのデータを持つ組織で広く使われています。pandasを学ぶとSQLも簡単に習得できるでしょう。

### 必要なライブラリ

まずは、この章に必要なライブラリをインポートしましょう。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# プロットの設定
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
```

また、この章で使用するnycflights13データセットをダウンロードします。

```python
# NYCフライトデータの読み込み
flights_url = "https://raw.githubusercontent.com/ismayc/pnwflights14/master/data/flights.csv"
flights = pd.read_csv(flights_url)

# airlinesデータの読み込み
airlines_url = "https://raw.githubusercontent.com/ismayc/pnwflights14/master/data/airlines.csv"
airlines = pd.read_csv(airlines_url)

# airportsデータの読み込み
airports_url = "https://raw.githubusercontent.com/ismayc/pnwflights14/master/data/airports.csv"
airports = pd.read_csv(airports_url)

# weatherデータの読み込み
weather_url = "https://raw.githubusercontent.com/ismayc/pnwflights14/master/data/weather.csv"
weather = pd.read_csv(weather_url)

# planesデータの読み込み
planes_url = "https://raw.githubusercontent.com/ismayc/pnwflights14/master/data/planes.csv"
planes = pd.read_csv(planes_url)

# データの確認
flights.head()
```

## メソッドチェーン

Rの`dplyr`パッケージで使われるパイプ演算子（`%>%`）に相当するのが、Pythonのpandasにおけるメソッドチェーンです。メソッドチェーンを使うと、複数の操作を連続した1つのシーケンスとして組み合わせることができます。

仮想的な例を考えてみましょう。データフレーム`x`に対して、関数`f()`、`g()`、そして`h()`という3つの操作を順番に実行したいとします：

1. `x`を取り、次に
2. `x`を入力として関数`f()`を適用し、次に
3. `f(x)`の出力を入力として関数`g()`を適用し、次に
4. `g(f(x))`の出力を入力として関数`h()`を適用する

Rでこれを実現する1つの方法はネストされた括弧を使うことです：`h(g(f(x)))`

pandasでは、メソッドチェーンを使って同じことができます：

```python
# 仮想的な例
# x.f().g().h()
```

これは以下のように読むことができます：

1. `x`を取り、次に
2. この出力を入力として次の関数`f()`に渡し、次に
3. この出力を入力として次の関数`g()`に渡し、次に
4. この出力を入力として次の関数`h()`に渡す

この章のデータラングリングを通じて：

1. 初期値`x`はデータフレームです。例えば、第1章で探索した`flights`データフレームです。
2. 関数のシーケンス（ここでは`f()`、`g()`、`h()`）は、主に本章の冒頭で紹介した6つのデータラングリング操作のいずれかです。
3. 結果は変換/修正されたデータフレームになります。

例えば、Alaskaの航空会社のフライトだけを抽出するには：

```python
alaska_flights = flights[flights['carrier'] == 'AS']
```

メソッドチェーンを使うと、複数の操作を一連の操作として結合できます。これにより、コードの可読性が向上します。

## 行のフィルタリング

![フィルタリング操作の図](https://d33wubrfki0l68.cloudfront.net/8787a2a48275a9e5a8d21fbc8e77c0f391bdfdac/fef13/images/cheatsheets/filter.png)

pandasのフィルタリング機能は、Microsoft Excelの「フィルター」オプションに似ています。データセット内の変数の値に関する条件を指定し、その条件に一致する行だけをフィルタリングできます。

まず、ニューヨーク市からポートランド、オレゴン州（空港コード「PDX」）への便だけに焦点を当ててみましょう：

```python
portland_flights = flights[flights['dest'] == 'PDX']
portland_flights.head()
```

コードの順序に注目してください。`flights`データフレームを取り、その後`dest`が「PDX」と等しい行だけをフィルタリングしています。等値テストには`==`演算子を使います。

他の演算子も使用できます：

- `>` は「より大きい」
- `<` は「より小さい」
- `>=` は「以上」
- `<=` は「以下」
- `!=` は「等しくない」

さらに、比較演算子を使用して複数の条件を組み合わせることができます：

- `|` は「または」
- `&` は「かつ」

これらの多くを試してみましょう。JFKから出発し、バーリントン（「BTV」）またはシアトル（「SEA」）に向かう、10月、11月、12月に出発した便をフィルタリングします：

```python
btv_sea_flights_fall = flights[
    (flights['origin'] == 'JFK') & 
    ((flights['dest'] == 'BTV') | (flights['dest'] == 'SEA')) & 
    (flights['month'] >= 10)
]
btv_sea_flights_fall.head()
```

括弧の慎重な使用に注目してください。

バーリントン、VTまたはシアトル、WAに向かわない便を選択するために「not」演算子（`~`）を使用することもできます：

```python
not_BTV_SEA = flights[~((flights['dest'] == 'BTV') | (flights['dest'] == 'SEA'))]
not_BTV_SEA.head()
```

多数の空港をフィルタリングしたい場合（例：「SEA」、「SFO」、「PDX」、「BTV」、「BDL」）、`isin()`メソッドを使用すると便利です：

```python
many_airports = flights[flights['dest'].isin(['SEA', 'SFO', 'PDX', 'BTV', 'BDL'])]
many_airports.head()
```

このコードは、`dest`が空港リスト「SEA」、「SFO」、「PDX」、「BTV」、「BDL」にある便をフィルタリングします。`isin()`メソッドは、ある変数の値がリスト内の値と一致するかどうかを確認するのに便利です。

フィルタリングは、最初に適用すべきデータ操作の一つです。これにより、関心のある行だけを含むようにデータセットをクリーンアップできます。

**学習チェック：**

「not」演算子（`~`）を使用して、バーリントン、VTもシアトル、WAにも向かわない行だけをフィルタリングする別の方法は何ですか？前のコードを使用してこれをテストしてみましょう。

## 変数の要約

次によく行われるデータフレームの操作は、要約統計量の計算です。要約統計量は、多くの値を要約する単一の数値です。よく知られている例としては、平均（算術平均とも呼ばれる）や中央値（中間値）などがあります。その他の要約統計量の例としては、合計、最小値、最大値、標準偏差などがあります。

weatherデータフレーム内の温度変数の2つの要約統計量（平均と標準偏差）を計算してみましょう：

![要約関数の図](https://d33wubrfki0l68.cloudfront.net/c5ffd65be2c76de9a9836feee8656712c2696384/c5c95/images/cheatsheets/summary.png)

これらの要約統計量を計算するには、pandasの`mean()`と`std()`メソッドを使用します：

```python
summary_temp = pd.DataFrame({
    'mean': [weather['temp'].mean()],
    'std_dev': [weather['temp'].std()]
})
summary_temp
```

欠損値（NA）がある場合、デフォルトでは要約統計量を計算すると`NaN`が返されます。これを回避するには、`dropna=True`引数を設定します：

```python
summary_temp = pd.DataFrame({
    'mean': [weather['temp'].mean(skipna=True)],
    'std_dev': [weather['temp'].std(skipna=True)]
})
summary_temp
```

ただし、欠損値を無視する際には注意が必要です。これは、欠損値の存在とその潜在的な原因に注意を払う必要があるためです。

他にもいくつかの要約関数があります：

* `mean()`: 平均
* `std()`: 標準偏差（ばらつきの指標）
* `min()`と`max()`: 最小値と最大値
* `quantile([0.25, 0.75])`: 四分位範囲を求めるのに使用
* `sum()`: 複数の数値を加算した合計
* `count()`: 行数のカウント

**学習チェック：**

医師が多数の患者を対象に喫煙と肺癌の関係を研究しています。患者の記録は5年ごとに測定されています。患者の死亡により多くの患者にデータポイントが欠けているため、分析ではこれらの患者を無視することにしました。この医師のアプローチの問題点は何ですか？

次の`summarize()`関数を修正して、`n()`要約関数も使用するようにしてください：`summarize(... , count = n())`。返された値は何に対応していますか？

## 行のグループ化

![グループ化と要約の図](https://d33wubrfki0l68.cloudfront.net/9007dfd0d2ebcfa055d78e4a28cb1aadf7eb4cc0/3c354/images/cheatsheets/group_summary.png)

1年間の単一平均気温ではなく、12ヶ月それぞれに対して12の平均気温が欲しいとします。つまり、月別に分けて平均気温を計算したいです。これは、「月」変数の12の値によって観測を「グループ化」することで実現できます：

```python
summary_monthly_temp = weather.groupby('month').agg({
    'temp': ['mean', 'std']
}).reset_index()

# カラム名の整理
summary_monthly_temp.columns = ['month', 'mean', 'std_dev']
summary_monthly_temp
```

`groupby()`関数はデータフレームを変更しないことに注意することが重要です。代わりに、メタデータ（データに関するデータ）、特にグループ化構造を変更します。`agg()`関数を適用した後にのみデータフレームが変更されます。

例として、ggplot2パッケージに含まれる`diamonds`データフレームを考えてみましょう：

```python
# diamondsデータセットの読み込み
diamonds_url = "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv"
diamonds = pd.read_csv(diamonds_url)
diamonds.head()
```

`cut`でグループ化してみましょう：

```python
diamonds.groupby('cut').size()
```

メタデータは変更されましたが、データ自体は変更されていません。`groupby()`と別のデータラングリング操作（この場合は`agg()`）を組み合わせると、データが実際に変換されます：

```python
diamonds.groupby('cut').agg({'price': 'mean'}).reset_index()
```

次に、`n()`カウント要約関数を再訪してみましょう。3つのニューヨーク市空港からそれぞれ何機の便が出発したかをカウントするとします：

```python
by_origin = flights.groupby('origin').size().reset_index(name='count')
by_origin
```

ニューワーク（「EWR」）が2013年に最も多くの便が出発し、その後に「JFK」と最後にラガーディア（「LGA」）が続いています。

### 複数の変数でのグループ化

一つの変数でのグループ化に限定されません。各月ごとに3つのニューヨーク市空港から出発する便の数を知りたいとします。第2の変数`month`でもグループ化できます：

```python
by_origin_monthly = flights.groupby(['origin', 'month']).size().reset_index(name='count')
by_origin_monthly
```

`by_origin_monthly`には36行あります（3つの空港「EWR」、「JFK」、「LGA」に対して12ヶ月あるため）。

**学習チェック：**

第2章でニューヨーク市の月別気温を見たことを思い出してください。`summary_monthly_temp`データフレームの標準偏差の列は、一年を通じてのニューヨーク市の気温について何を教えてくれますか？

2013年のニューヨーク市の各日の平均気温と標準偏差を得るために必要なコードは何ですか？

`by_monthly_origin`を再作成しますが、`groupby(['origin', 'month'])`ではなく、`groupby(['month', 'origin'])`のように変数を異なる順序でグループ化してください。結果のデータセットで何が異なりますか？

## 既存の変数の変更

![変数の変更の図](https://d33wubrfki0l68.cloudfront.net/3de15079ade7016a277d777a7720d3a490d10dc7/4bc00/images/cheatsheets/mutate.png)

データの一般的な変換は、既存の変数に基づいて新しい変数を作成/計算することです。例えば、温度を華氏（°F）ではなく摂氏（°C）で考える方が慣れているとします。華氏から摂氏への変換式は：

$$
\text{temp in C} = \frac{\text{temp in F} - 32}{1.8}
$$

pandasでは、新しい列を追加することでこの式を`temp`変数に適用できます：

```python
weather['temp_in_C'] = (weather['temp'] - 32) / 1.8

# 華氏と摂氏の両方で月平均気温を計算する
summary_monthly_temp = weather.groupby('month').agg({
    'temp': 'mean',
    'temp_in_C': 'mean'
}).reset_index()

# カラム名の整理
summary_monthly_temp.columns = ['month', 'mean_temp_in_F', 'mean_temp_in_C']
summary_monthly_temp
```

別の例を考えてみましょう。乗客はしばしば便が遅れて出発すると不満を感じますが、パイロットがフライト中に時間を取り戻せば、それほど苛立ちません。これは航空業界では「利得」と呼ばれ、この変数を作成します：

```python
flights['gain'] = flights['dep_delay'] - flights['arr_delay']

# 最初の5行の出発/到着遅延と利得変数を見てみる
flights[['dep_delay', 'arr_delay', 'gain']].head()
```

`gain`変数のいくつかの要約統計量を見てみましょう：

```python
gain_summary = pd.DataFrame({
    'min': [flights['gain'].min()],
    'q1': [flights['gain'].quantile(0.25)],
    'median': [flights['gain'].quantile(0.5)],
    'q3': [flights['gain'].quantile(0.75)],
    'max': [flights['gain'].max()],
    'mean': [flights['gain'].mean()],
    'sd': [flights['gain'].std()],
    'missing': [flights['gain'].isna().sum()]
})
gain_summary
```

平均利得は約5分ですが、最大は109分にもなります！

`gain`は数値変数なので、ヒストグラムを使ってその分布を可視化できます：

```python
plt.figure(figsize=(10, 6))
plt.hist(flights['gain'].dropna(), bins=20, color='steelblue', edgecolor='white')
plt.title('Gain変数のヒストグラム')
plt.xlabel('Gain（分）')
plt.ylabel('頻度')
plt.grid(False)
plt.show()
```

結果のヒストグラムは、先ほど計算した要約統計量とは異なる視点で`gain`変数を提供します。例えば、ほとんどの`gain`値はちょうど0付近にあることに注意してください。

pandasでは、同時に複数の新しい変数を作成することもできます。さらに、新しく作成した変数を参照することもできます：

```python
flights['gain'] = flights['dep_delay'] - flights['arr_delay']
flights['hours'] = flights['air_time'] / 60
flights['gain_per_hour'] = flights['gain'] / flights['hours']
```

**学習チェック：**

`flights`の`gain`変数の正の値は何に対応していますか？負の値はどうですか？ゼロ値は？

`dep_time`から`sched_dep_time`を引き、到着についても同様にして`dep_delay`と`arr_delay`列を作成できますか？コードを試して、結果と実際に`flights`に表示されるものとの違いを説明してください。

## 行の並べ替え

最も一般的に行われるデータラングリングタスクの1つは、データフレームの行を何らかの変数の値に基づいて並べ替えることです。pandasの`sort_values()`メソッドを使用すると、指定した変数の値に従ってデータフレームの行をソート/並べ替えることができます。

2013年にニューヨーク市から出発するすべての国内線の最も頻繁な目的地空港を決定したいとします：

```python
freq_dest = flights.groupby('dest').size().reset_index(name='num_flights')
freq_dest.head()
```

デフォルトでは、結果の`freq_dest`データフレームの行は`dest`アルファベット順に並べられています。代わりに、フライト数（`num_flights`）の最も多いものから最も少ないものへと並べ替えたいとします：

```python
freq_dest.sort_values('num_flights').head()
```

しかし、これは望ましい結果の逆です。行は最も頻度の低い目的地空港が最初に表示されるように並べられています。これは、`sort_values()`がデフォルトで常に行を昇順に並べるためです。順序を「降順」に切り替えるには、`ascending=False`パラメータを使用します：

```python
freq_dest.sort_values('num_flights', ascending=False).head()
```

## データフレームの結合

別の一般的なデータ変換タスクは、2つの異なるデータセットを「結合」または「マージ」することです。例えば、`flights`データフレームでは、変数`carrier`が異なるフライトのキャリアコードを一覧表示しています。「UA」や「AA」に対応する航空会社名は推測しやすいかもしれませんが（ユナイテッド航空とアメリカン航空）、「VX」、「HA」、「B6」のコードはどの航空会社でしょうか？この情報は別のデータフレーム`airlines`に記載されています。

```python
airlines.head()
```

`airlines`では、`carrier`はキャリアコード、`name`は航空会社の正式名称です。この表を使用すると、「VX」、「HA」、「B6」がそれぞれバージンアメリカ、ハワイアン航空、ジェットブルーに対応していることがわかります。しかし、この情報をすべて2つの別々のデータフレームではなく、1つのデータフレームにまとめられたら便利ではないでしょうか？`flights`と`airlines`データフレームを「結合」することでこれを実現できます。

`flights`データフレームの変数`carrier`の値は、`airlines`データフレームの変数`carrier`の値と一致することに注意してください。この場合、変数`carrier`を2つのデータフレームの行を一致させるための「キー変数」として使用できます。

### 一致する「キー」変数名

`flights`と`airlines`の両方のデータフレームで、行を結合/マージ/一致させたいキー変数は同じ名前です：`carrier`。`merge()`メソッドを使用して2つのデータフレームを結合し、変数`carrier`によって行が一致するようにしましょう：

```python
flights_joined = flights.merge(airlines, on='carrier')
flights_joined[['carrier', 'name']].head()
```

`flights`と`flights_joined`データフレームは、`flights_joined`に`name`という追加の変数があることを除いて同一です。`name`の値は、`airlines`データフレームに示されている航空会社名に対応しています。

### 異なる「キー」変数名

2013年にニューヨーク市から出発するすべての国内線の目的地に興味があり、「これらの空港はどの都市にありますか？」、「『ORD』はオーランドですか？」、「『FLL』はどこですか？」などの質問をします。

`airports`データフレームには各空港の空港コードが含まれています：

```python
airports.head()
```

しかし、`airports`と`flights`の両方のデータフレームを見ると、空港コードは異なる名前の変数にあることがわかります。`airports`では空港コードは`faa`にありますが、`flights`では空港コードは`origin`と`dest`にあります。

空港コードでこれら2つのデータフレームを結合するために、`merge`操作では`left_on`と`right_on`パラメータを使用します：

```python
flights_with_airport_names = flights.merge(airports, left_on='dest', right_on='faa')
flights_with_airport_names[['dest', 'faa', 'name']].head()
```

ニューヨーク市から各目的地への便数を計算するパイプラインを構築し、各目的地空港に関する情報も含めましょう：

```python
named_dests = (flights
               .groupby('dest')
               .size()
               .reset_index(name='num_flights')
               .sort_values('num_flights', ascending=False)
               .merge(airports, left_on='dest', right_on='faa')
               .rename(columns={'name': 'airport_name'})
              )
named_dests.head(10)
```

知らなかった場合、「ORD」はシカゴ・オヘア空港の空港コードで、「FLL」はフロリダ州フォートローダーデールの主要空港です。これは`airport_name`変数で確認できます。

### 複数の「キー」変数

*複数のキー変数*で2つのデータフレームを結合したい場合もあります。例えば、`flights`と`weather`データフレームを結合するには、1つ以上のキー変数が必要です：`year`、`month`、`day`、`hour`、`origin`。これは、これら5つの変数の組み合わせが`weather`データフレーム内の各観測単位を一意に識別するためです：3つのNYC空港それぞれの毎時間の天気記録。

キー変数のリストを指定してこれを実現します：

```python
# flights_weather_joined = flights.merge(weather, 
#                                       on=['year', 'month', 'day', 'hour', 'origin'])
# flights_weather_joined.head()
```

**学習チェック：**

`flights`と`weather`を結合する（または各フライトと時間ごとの天気値を一致させる）とき、なぜ`hour`だけではなく、`year`、`month`、`day`、`hour`、`origin`のすべてで結合する必要があるのですか？

2013年のNYCからの上位10の目的地について何が驚きましたか？

### 正規形

`nycflights13`パッケージに含まれるデータフレームは、データの冗長性を最小限に抑える形式になっています。例えば、`flights`データフレームは航空会社の`carrier`コードのみを保存し、航空会社の実際の名前は含まれていません。例えば、`flights`の最初の行には`carrier`が`UA`と等しいですが、航空会社名「United Air Lines Inc.」は含まれていません。

航空会社名は`airlines`データフレームの`name`変数に含まれています。`flights`に航空会社名を含めるには、これら2つのデータフレームを次のように結合できます：

```python
joined_flights = flights.merge(airlines, on='carrier')
joined_flights[['carrier', 'name']].head()
```

これらのデータフレームの間で一方を他方に関連付けるための共通の*キー*があるため、この結合を行うことができます：`flights`と`airlines`の両方のデータフレームの`carrier`変数です。キー変数は多くの場合、前述の*識別変数*です。

これはデータの*正規形*と呼ばれる重要な特性です。情報を失うことなくデータフレームをより冗長性の少ないテーブルに分解するプロセスは*正規化*と呼ばれます。詳細は[Wikipedia](https://en.wikipedia.org/wiki/Database_normalization)で確認できます。

pandasと[SQL](https://en.wikipedia.org/wiki/SQL)はどちらもこのような*正規形*を使用します。これらが共通点を持っているため、どちらかのツールを学ぶと、もう一方も非常に簡単に学ぶことができます。

**学習チェック：**

正規形のデータの利点は何ですか？欠点は何ですか？

## その他の操作

他にも便利なデータラングリング操作があります：

* 変数/列のサブセットのみを選択する
* 変数/列の名前を変更する
* 変数の上位n個の値のみを返す

### 変数の選択

![列の選択の図](https://d33wubrfki0l68.cloudfront.net/e829fb346c6d34d9a19c8a01b636ecb2c075c862/ae23a/images/cheatsheets/select.png)

`nycflights13`パッケージの`flights`データフレームには異なる19の変数が含まれています。これらの19変数の名前は`flights.columns`で確認できます：

```python
flights.columns
```

ただし、これら19の変数のうち、`carrier`と`flight`の2つだけが必要だとします。これらの2つの変数を選択できます：

```python
flights[['carrier', 'flight']].head()
```

この関数を使うと、関心のある変数だけに焦点を絞ることができるため、大きなデータセットの探索が容易になります。

代わりに、特定の変数を削除、つまり選択解除したい場合を考えてみましょう。例えば、`flights`データフレームの変数`year`について考えてみましょう。この変数は常に`2013`なので、実際には「変数」ではありません。この変数をデータフレームから削除したいとします。`drop()`メソッドを使用して`year`を選択解除できます：

```python
flights_no_year = flights.drop('year', axis=1)
flights_no_year.columns
```

列/変数を選択する別の方法は、列の範囲を指定することです：

```python
flight_arr_times = flights.loc[:, 'month':'day'].join(flights.loc[:, 'arr_time':'sched_arr_time'])
flight_arr_times.head()
```

これにより、`month`から`day`までの列と、`arr_time`から`sched_arr_time`までの列がすべて選択され、残りは削除されます。

変数の選択は、列の順序を変更するためにも使用できます。例えば、`year`、`month`、`day`変数の直後に`hour`、`minute`、`time_hour`変数を表示したいとします（残りの変数は破棄しません）：

```python
flights_reorder = flights[['year', 'month', 'day', 'hour', 'minute', 'time_hour'] + 
                          [col for col in flights.columns if col not in ['year', 'month', 'day', 'hour', 'minute', 'time_hour']]]
flights_reorder.columns
```

最後に、`startswith()`、`endswith()`、`contains()`などのメソッドを使用して、それらの条件に一致する変数/列を選択できます：

```python
# "a"で始まる列を選択
flights.loc[:, flights.columns.str.startswith('a')].head()

# "delay"で終わる列を選択
flights.loc[:, flights.columns.str.endswith('delay')].head()

# "time"を含む列を選択
flights.loc[:, flights.columns.str.contains('time')].head()
```

### 変数の名前変更

もう一つの便利な関数は`rename()`で、これは変数の名前を変更します。`dep_time`と`arr_time`のみに焦点を当て、`dep_time`と`arr_time`をそれぞれ`departure_time`と`arrival_time`に変更したいとします：

```python
flights_time_new = flights[['dep_time', 'arr_time']].rename(
    columns={'dep_time': 'departure_time', 'arr_time': 'arrival_time'}
)
flights_time_new.head()
```

### 変数の上位n個の値

変数の上位n個の値も`nlargest()`メソッドを使用して返すことができます。例えば、前の節の例を使用して、上位10の目的地空港のデータフレームを返すことができます：

```python
top_10_dests = named_dests.nlargest(10, 'num_flights')
top_10_dests
```

さらに、これらの結果を`num_flights`の降順に並べ替えることもできます：

```python
top_10_dests = named_dests.nlargest(10, 'num_flights').sort_values('num_flights', ascending=False)
top_10_dests
```

**学習チェック：**

`flights`から`dest`、`air_time`、`distance`の3つの変数すべてを選択する方法は何ですか？これを少なくとも3つの異なる方法で行うコードを示してください。

`startswith()`、`endswith()`、`contains()`を使用して`flights`データフレームから列を選択するにはどうすればよいですか？3つの異なる例を示してください：1つは`startswith()`用、1つは`endswith()`用、1つは`contains()`用。

データフレームに対して`select`関数を使用したい理由は何ですか？

2013年にニューヨーク市から出発した到着遅延が最も大きい上位5つの空港を示す新しいデータフレームを作成してください。

## 結論

### 要約表

データラングリング操作を表にまとめてみましょう：

| 操作 | データラングリング操作 |
|------|----------------------|
| フィルタリング | データフレームの既存の行から条件に基づいてサブセットを選択する |
| 要約 | 1つ以上の列/変数を要約統計量で要約する |
| グループ化 | 行をグループ化する。異なるグループごとに個別の要約統計量を計算するためにグループ化と要約を組み合わせることができる |
| 変数作成 | 既存の列/変数から新しい列/変数を作成する |
| 並べ替え | 特定の変数の値に基づいてデータフレームの行を昇順または降順に並べ替える |
| 結合 | 「キー」変数に沿って一致させることにより、別のデータフレームとデータフレームを結合する |

**学習チェック：**

新たに習得したデータラングリングスキルをテストしてみましょう！

航空業界における旅客航空会社のキャパシティの指標は[提供座席マイル（available seat miles）](https://en.wikipedia.org/wiki/Available_seat_miles)です。これは、利用可能な座席数に飛行するマイル数またはキロメートル数を掛け、すべてのフライトにわたって合計したものです。

例えば、航空機に4つの座席があり、200マイル移動する場合、提供座席マイルは4 × 200 = 800です。

この考えを拡張すると、航空会社が10席の航空機を使用して500マイル飛行する便を2便と、20席の航空機を使用して1000マイル飛行する便を3便持っていた場合、提供座席マイルは2 × 10 × 500 + 3 × 20 × 1000 = 70,000座席マイルとなります。

`nycflights13`パッケージに含まれるデータセットを使用して、各航空会社の提供座席マイルを降順に計算してください。必要なすべてのデータラングリングステップを完了した後、結果のデータフレームには16行（各航空会社に1行）と2列（航空会社名と提供座席マイル）が含まれるはずです。

ヒント：
1. **重要**: 何をしているのかに非常に自信がない限り、すぐにコーディングを始めないことが価値があります。むしろ、最初に必要なすべてのデータラングリングステップを紙に概略的に書き出し、正確なコードではなく、高レベルの*疑似コード*を使用して行うことを明確にします。これにより、何をしようとしているのか（アルゴリズム）と、それをどのように行うか（pandasコードの記述）を混同することがなくなります。
2. `View()`関数を使用してすべてのデータセットをよく見てください：`flights`、`weather`、`planes`、`airports`、`airlines`から提供座席マイルの計算に必要な変数を特定します。
3. 様々なデータセットがどのように結合できるかを示す図も役立ちます。
4. 表のデータラングリング操作をツールボックスとして考えてください！

### その他のリソース

[付録C](https://moderndive.com/C-appendixC.html)では、学生プロジェクトでよく遭遇するデータラングリングの「ヒントとコツ」のページを提供しています。例えば：

* 欠損値の扱い
* 棒グラフの棒の並べ替え
* 軸上のお金の表示
* セル内の値の変更
* 数値変数をカテゴリー変数に変換する
* 比率の計算
* %、カンマ、$の扱い

しかし、可能なすべてのデータラングリングの質問をカバーするヒントとコツのページを提供するのは長すぎて役に立ちません。pandasパッケージのデータラングリング機能をさらに活用したい場合は、pandasの公式ドキュメントや「Python for Data Analysis」（Wes McKinney著）を確認することをお勧めします。

### 次に何が来るか？

これまで、データフレームに保存されたデータの探索、可視化、加工を行ってきました。これらのデータフレームは表計算型の長方形形式で保存されていました：観測に対応する一定数の行と、これらの観測を説明する一定数の列からなる形です。

次の第4章では、表計算型の長方形形式でデータを表現する2つの方法があることがわかります：(1)「ワイド」形式と(2)「トール/ナロー」形式です。トール/ナロー形式は、Rユーザーの間では*「整然データ（tidy）」*形式とも呼ばれています。「整然」と非「整然」形式のデータの区別は微妙ですが、私たちのデータサイエンス作業に大きな影響を与えます。これは、データ可視化のための`seaborn`パッケージやデータラングリングのための`pandas`パッケージを含む、この本で使用されるほぼすべてのパッケージが、すべてのデータフレームが「整然」形式であることを前提としているためです。

さらに、これまでは主にPythonライブラリに保存されているデータの探索、可視化、加工を行ってきました。しかし、Microsoft Excel、Google Sheets、「カンマ区切り値」（CSV）ファイルに保存されているデータを分析したい場合はどうでしょうか？第4章では、pandasを使用してこのデータをPythonにインポートする方法も紹介します。