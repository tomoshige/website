# 第4章 データのインポートと「整然データ」

## はじめに

第2章でプログラミングの基礎概念として、データフレームについて紹介しました：データフレームとは行と列からなる長方形の表形式データ表現で、行は観測値、列は変数を表します。第3章ではnycflights13パッケージに含まれるflightsデータフレームを探索し始めました。第4章では、flightsや他のデータフレーム（weatherなど）を基にした可視化を作成しました。第5章では、既存のデータフレームを変換・修正する方法を学びました。

この「Python + tidyverse」パートの最終章では、「整然データ」（tidy data）と呼ばれるデータ形式の概念を拡張します。整然データという用語は、日常的な意味の「きちんと整理された」という意味以上のものです。ここではデータサイエンティストが使用する形式として明確に定義していきます。

この考え方は、第4章や第5章で必要ではありませんでした。なぜなら、使用したすべてのデータはすでに「整然」形式だったからです。しかし、この章ではこの形式がこれまで扱ってきたツールにとって不可欠であることを学びます。さらに、この後の章で回帰と統計的推測を扱う際にも役立ちます。しかしまず最初に、Pythonでスプレッドシートデータをインポートする方法を紹介します。

## データインポート

この点まで、ほとんどパッケージ内に保存されたデータを使ってきました。しかし、自分のコンピュータやオンラインのどこかに保存された独自のデータを分析したい場合はどうすればよいでしょうか？スプレッドシートデータは、通常、次の3つの形式のいずれかで保存されています：

1. *カンマ区切り値（CSV）* `.csv`ファイル：最も基本的なスプレッドシート形式で、
   * ファイル内の各行はデータの1行/1つの観測値に対応
   * 各行の値はカンマで区切られる
   * 最初の行は通常（必ずではない）列名/変数名を示すヘッダー行

2. Excelの`.xlsx`スプレッドシートファイル：Microsoftの独自形式に基づいています。CSVと比較して、多くのメタデータ（データに関するデータ）を含みます。

3. Google Sheetsファイル：オンラインベースのスプレッドシート作業方法です。CSVまたはExcel形式でダウンロードできます。

Pythonでは、主にpandasライブラリを使用してこれらのファイル形式からデータをインポートします。

### pandasを使ったCSVとExcelファイルのインポート

まず必要なライブラリをインポートしましょう：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

最初に、インターネット上にあるCSVファイルをインポートしてみましょう。`dem_score.csv`は1952年から1992年までの異なる国々の民主主義レベルの評価を含んでいます。pandasの`read_csv()`関数を使って、ウェブからデータを読み込み、`dem_score`というデータフレームに保存します。

```python
# GitHubからデータファイルをダウンロード
dem_score_url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/dem_score.csv"
dem_score = pd.read_csv(dem_score_url)
dem_score.head()
```

このデータフレームでは、最小値の`-10`は高度に独裁的な国を表し、最大値の`10`は高度に民主的な国を表します。

Excelファイルをインポートする場合は、`pd.read_excel()`関数を使います：

```python
# GitHubからExcelファイルをダウンロードする例
# dem_score_excel_url = "https://github.com/moderndive/moderndive_book/raw/master/data/dem_score.xlsx"
# dem_score_excel = pd.read_excel(dem_score_excel_url)
# dem_score_excel.head()
```

## 「整然データ」

次に「整然データ」（tidy data）という概念について学びましょう。例として`fivethirtyeight`データセットの`drinks`データを使用します。Rパッケージとは異なり、Pythonではデータをダウンロードする必要があります：

```python
# fivethirtyeightのdrinksデータをダウンロード
drinks_url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/alcohol-consumption/drinks.csv"
drinks = pd.read_csv(drinks_url)
drinks.head()
```

このデータフレームには、様々な国でのビール、蒸留酒、ワインの平均消費量に関する調査結果が含まれています。第5章で学んだデータラングリング操作をいくつか適用してみましょう：

1. 4カ国（アメリカ、中国、イタリア、サウジアラビア）だけを選択
2. `total_litres_of_pure_alcohol`列以外のすべての列を選択
3. 変数名を`beer_servings`→`beer`、`spirit_servings`→`spirit`、`wine_servings`→`wine`に変更

```python
drinks_smaller = drinks[drinks['country'].isin(['USA', 'China', 'Italy', 'Saudi Arabia'])]
drinks_smaller = drinks_smaller.drop('total_litres_of_pure_alcohol', axis=1)
drinks_smaller = drinks_smaller.rename(columns={
    'beer_servings': 'beer',
    'spirit_servings': 'spirit',
    'wine_servings': 'wine'
})
drinks_smaller
```

この`drinks_smaller`データフレームを使って、以下のような国別・アルコール種類別の消費量を示す棒グラフを作りたいとします：

```python
# データを整然データ形式に変換する必要があります（後で実装）
# 直接この形式のグラフを作成することはできません
```

この可視化を作成するには、以下のグラフィックス文法の要素が必要です：

1. カテゴリー変数`country`（4レベル：China、Italy、Saudi Arabia、USA）を棒のx位置にマッピング
2. 数値変数`servings`を棒のy位置（高さ）にマッピング
3. カテゴリー変数`type`（3レベル：beer、spirit、wine）を棒の色にマッピング

しかし、`drinks_smaller`では`beer`、`spirit`、`wine`という3つの独立した変数があります。棒グラフを作成するためには、*単一の変数*`type`が必要です。これが`beer`、`spirit`、`wine`という3つの値を持ち、この`type`変数をグラフの`color`属性にマッピングできるようにする必要があります。つまり、データフレームは以下のような形式になる必要があります：

```python
# 整然データ形式に変換
drinks_smaller_tidy = drinks_smaller.melt(
    id_vars=['country'],
    var_name='type',
    value_name='servings'
)
drinks_smaller_tidy
```

`drinks_smaller`と`drinks_smaller_tidy`は両方とも長方形の形状で、同じ数値を含んでいますが、形式が異なります。`drinks_smaller`は「幅広」（wide）形式で、`drinks_smaller_tidy`は「長い/狭い」（long/narrow）形式です。

Pythonでのデータサイエンスの文脈では、長い/狭い形式は「整然」形式とも呼ばれます。matplotlib、seaborn、pandasを使ってデータの可視化とラングリングを行うためには、入力データフレームが「整然」形式である必要があります。そのため、「非整然」データはまず「整然」形式に変換する必要があります。

### 「整然データ」の定義

「整然」（tidy）という言葉は日常生活でよく耳にします：
* 「部屋を整然と片付けなさい！」
* 「フィードバックしやすいように宿題を整然と書きなさい」
* など

データが「整然」であるとはどういう意味でしょうか？一般的な英語では「整理整頓された」という明確な意味がありますが、Pythonでのデータサイエンスにおいて「整然」という言葉は、データが標準化された形式に従っていることを意味します。Hadley Wickhamによる*「整然データ」*の定義に従うと：

> *データセット*は、通常、数値（量的な場合）または文字列（質的/カテゴリー的な場合）の値の集まりです。値は2つの方法で整理されます。すべての値は変数と観測値に属します。変数は、単位間で同じ基礎的な属性（身長、温度、期間など）を測定するすべての値を含みます。観測値は、ある単位（人、日、都市など）について属性間で測定されたすべての値を含みます。
> 
> 「整然」データは、データセットの意味をその構造にマッピングする標準的な方法です。データセットが乱雑か整然かは、行、列、テーブルが観測値、変数、タイプとどのように一致するかによって決まります。*整然データ*では：
>
> 1. 各変数が列を形成する。
> 2. 各観測値が行を形成する。
> 3. 各タイプの観測単位がテーブルを形成する。

例えば、次のような株価の表があるとします：

| 日付 | ボーイング株価 | アマゾン株価 | グーグル株価 |
|------|------------|----------|----------|
| 2009-01-01 | $173.55 | $174.90 | $174.34 |
| 2009-01-02 | $172.61 | $171.42 | $170.04 |

このデータは長方形のスプレッドシート形式で整理されていますが、「整然」データの定義には従っていません。3つの変数（日付、株名、株価）が3つの独自の情報に対応していますが、3つの列があるわけではありません。「整然」データ形式では、各変数が独自の列になるべきです：

| 日付 | 株名 | 株価 |
|------|------|------|
| 2009-01-01 | ボーイング | $173.55 |
| 2009-01-01 | アマゾン | $174.90 |
| 2009-01-01 | グーグル | $174.34 |
| 2009-01-02 | ボーイング | $172.61 |
| 2009-01-02 | アマゾン | $171.42 |
| 2009-01-02 | グーグル | $170.04 |

両方のテーブルは同じ情報を提示していますが、形式が異なります。

### 「整然データ」への変換

これまでは既に「整然」形式になっているデータフレームのみを見てきました。そしてこの本の残りの部分でも、ほとんど「整然」形式のデータフレームのみを見ることになります。しかし、世界のすべてのデータセットが常にそうであるとは限りません。

元のデータフレームが幅広（非「整然」）形式でmatplotlibやseabornパッケージを使用したい場合は、まず「整然」形式に変換する必要があります。これを行うには、pandasの`melt()`関数を使用することをお勧めします。

先ほどの`drinks_smaller`データフレームに戻りましょう：

```python
drinks_smaller
```

`melt()`関数を使って「整然」形式に変換します：

```python
drinks_smaller_tidy = drinks_smaller.melt(
    id_vars=['country'],
    var_name='type',
    value_name='servings'
)
drinks_smaller_tidy
```

`melt()`関数の引数は次のように設定します：

1. `id_vars`はグループ化するための変数（「溶かさない」変数）です。ここでは`country`を設定しています。
2. `var_name`は新しい「整然」データフレームの変数名で、元のデータの*列名*を含みます。`var_name='type'`と設定しています。結果の`drinks_smaller_tidy`では、`type`列に3種類のアルコール`beer`、`spirit`、`wine`が含まれています。
3. `value_name`は新しい「整然」データフレームの変数名で、元のデータの*値*を含みます。`value_name='servings'`と設定しています。

これで「整然」形式のデータフレームができたので、棒グラフを作成できます：

```python
# matplotlib/seabornを使用した棒グラフ
plt.figure(figsize=(10, 6))
sns.barplot(x='country', y='servings', hue='type', data=drinks_smaller_tidy)
plt.title('アルコール消費量の国別比較')
plt.xlabel('国')
plt.ylabel('提供数')
plt.show()
```

「幅広」形式のデータを「整然」形式に変換することは、初心者には混乱することがあります。`melt()`関数に慣れる唯一の方法は、さまざまなデータセットを使って練習、練習、そしてもっと練習することです。

逆に「整然」データフレームを「幅広」形式に変換したい場合は、`pivot_table()`または`pivot()`関数を使用する必要があります。

## ケーススタディ：グアテマラの民主主義

もう一つの例として、「整然」形式でないデータフレーム（「幅広」形式）を「整然」形式（「長い/狭い」形式）に変換する方法を見てみましょう。

さらに、matplotlib/seabornを使って、1952年から1992年までの40年間でグアテマラの民主主義スコアがどのように変化したかを示す*時系列プロット*を作成します。

第3章でインポートした`dem_score`データフレームを使用し、グアテマラのデータのみに焦点を当てましょう：

```python
guat_dem = dem_score[dem_score['country'] == 'Guatemala']
guat_dem
```

グラフィックス文法を考えてみましょう。民主主義スコアが年々どのように変化したかを見たいので、次のようにマッピングする必要があります：

* `year`をx位置に
* `democracy_score`をy位置に

しかし、`guat_dem`データフレームは「整然」形式ではないため、このままでは可視化に適用できません。「年」の列の値を新しい「名前」変数の`year`に変換し、データフレーム内部の民主主義スコアの値を新しい「値」変数の`democracy_score`に変換する必要があります。結果のデータフレームは`country`、`year`、`democracy_score`の3つの列を持ちます：

```python
# 整然データへの変換
# 年の列をリスト化（1952から1992までの10年ごと）
year_columns = [str(year) for year in range(1952, 1993, 10)]

guat_dem_tidy = guat_dem.melt(
    id_vars=['country'],
    value_vars=year_columns,
    var_name='year',
    value_name='democracy_score'
)

# yearを文字列から整数に変換
guat_dem_tidy['year'] = guat_dem_tidy['year'].astype(int)
guat_dem_tidy
```

これで時系列プロットを作成して、1952年から1992年までのグアテマラの民主主義スコアの変化を可視化できます：

```python
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='democracy_score', data=guat_dem_tidy)
plt.title('グアテマラの民主主義スコア 1952-1992')
plt.xlabel('年')
plt.ylabel('民主主義スコア')
plt.show()
```

## まとめ

この章では、以下の重要な概念を学びました：

1. pandasを使ってCSVやExcelのデータをPythonに読み込む方法
2. 「整然データ」の概念と、なぜデータを「整然」形式に保つことが重要なのか
3. pandasの`melt()`関数を使って「幅広」形式から「整然」形式にデータを変換する方法
4. 整然データを使って効果的な可視化を作成する方法

データが「整然」形式であることは、以降の章で学ぶモデリングと統計的推測のための基礎となります。次章からは「Python + モデリング」のパートに移り、データフレーム内の異なる変数間の関係をモデル化するために、データの可視化とラングリングのスキルを活用していきます。

まずは5章と6章で線形回帰モデルについて学び、その後7章から9章で統計的推測を扱った後、10章で「回帰のための推測」に戻ります。データサイエンスの旅を続けましょう！