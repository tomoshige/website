# Python でデータを可視化よう

データサイエンスのツールボックスを構築する最初のステップとして、データの可視化を学びます。データを視覚化することで、単なる数値の羅列では見えなかったパターンや傾向を発見できます。Pythonでは、`matplotlib` や `seaborn` を利用してデータを可視化できます。本章では、これらのライブラリを活用し、基本的なプロットを作成する方法を学びます。

グラフ（プロットやチャートとも呼びます）は、データのパターンを探るための強力な手法です。例えば、**外れ値の特定**、**データの分布**、**変数間の関係性**を理解するのに役立ちます。適切な可視化を行うことで、データから得られる洞察を効果的に伝えることができます。ただし、情報を詰め込みすぎると、かえって理解しにくくなることもあるため、適切なバランスを取ることが重要です。

---

## 必要なライブラリ

本章では、gapminderと呼ばれるデータセットを利用します。そのために、まずは以下のパッケージをインストールしてください。

```python
!pip install gapminder
```

次に、必要なライブラリをインポートします。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gapminder import gapminder
```

---

## グラフィックの基礎理論

「グラフィックの文法（Grammar of Graphics）」とは、データの可視化を体系的に整理するための理論です。この概念は Leland Wilkinson によって提唱され、Rの `ggplot2` や Pythonの `plotly` などの可視化ライブラリにも応用されています。

この理論によると、統計グラフは次の3つの要素から構成されます。

1. **データ (`data`)**: 可視化の対象となるデータセット。
2. **幾何オブジェクト (`geom`)**: グラフに描画される基本要素（点、線、棒など）。
3. **美的要素 (`aes`)**: 幾何オブジェクトの見た目を決める要素（位置、色、大きさなど）。

これらを組み合わせることで、データの可視化を構築できます。

---

## Gapminderデータの可視化

Gapminderは、世界の経済・健康・発展状況に関するデータを提供するプロジェクトです。このデータセットには、各国のGDP、寿命、人口などの情報が含まれています。

### データの準備

まず、Gapminderデータを2007年のデータに絞り込んで表示してみましょう。

```python
# 2007年のデータのみを取得
gapminder_2007 = gapminder[gapminder['year'] == 2007]

# 必要な列のみ選択
gapminder_2007 = gapminder_2007[['country', 'continent', 'lifeExp', 'pop', 'gdpPercap']]
gapminder_2007.columns = ['Country', 'Continent', 'Life Expectancy', 'Population', 'GDP per Capita']

# データの先頭を表示
gapminder_2007.head()
```

この表の各行は1つの国を表し、次の情報を含みます。

1. **Country**: 国の名前。
2. **Continent**: 5つの大陸のいずれか（「Americas」は北米と南米を含み、南極は除外）。
3. **Life Expectancy**: 平均寿命（年）。
4. **Population**: 人口（人）。
5. **GDP per Capita**: 1人当たりGDP（米ドル）。

---

### データの可視化

2007年のGDPと平均寿命の関係を可視化してみましょう。

```python
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=gapminder_2007,
    x='GDP per Capita',
    y='Life Expectancy',
    size='Population',
    hue='Continent',
    sizes=(10, 200),
    alpha=0.7
)

plt.xscale('log')  # GDPは対数スケールに変換
plt.xlabel('GDP per Capita (log scale)')
plt.ylabel('Life Expectancy (years)')
plt.title('2007年におけるGDPと平均寿命の関係')
plt.legend(title='Continent', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

このとき、グラフのタイトルをみると、日本語が表示できていないことがわかりますね。これは、非常によく起きる問題です。pythonでは、これを以下のパッケージをインストールすることで解消することができます。

```python
!pip install japanize-matplotlib
```

インストールした後で、最初の `import` 部分を次のように変更しましょう。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from gapminder import gapminder
```

これで日本語が表示されるはずです。次に、このプロットを詳しく見ていきましょう。

- 変数 GDP per Capita（1人当たりGDP） は、x軸にマッピングされています。
- 変数 Life Expectancy（平均寿命） は、ポイントの y軸にマッピングされています。
- 変数 Population（人口） は、ポイントの サイズ（size aesthetic） にマッピングされます。
- 変数 Continent（大陸） は、ポイントの 色（color aesthetic） にマッピングされます。

ここで、`data`（データ）とは pythonの `dataframe`に対応し、データ変数（data variables） がデータフレーム内の特定の列に対応していることが分かります。例えば、。特にこのプロットでは、4つの情報を同時に集約しています。x軸に一人あたりのGDP、y軸に平均寿命、さらに点の大きさで人口、そして、点の色で大陸を表しています。このプロットを通じて、以下の傾向が読み取れます。

- GDPが高い国ほど、平均寿命が長い傾向がある。
- 大陸ごとに分布が異なり、特にアフリカ（Africa）の国々はGDPと平均寿命が低い傾向がある。
- 国ごとの人口（点の大きさ）も視覚的に把握できる。

このように自分が示したいものが何かということを意識して、適切な図を書くことは、データサイエンスの基本となるものです。また、データの種類によっても、最適な図の書き方は異なります。次のステップとして、データサイエンスにおける5つの基本的な図について学びましょう。

---

## Pythonによるデータ可視化入門: 5つの代表的なグラフ (5NG)

このテキストでは、Google Colaboratory上でPython、NumPy、Pandas、matplotlib、seabornを用いて、データ可視化の基本となる5種類のグラフ（5NG）を紹介します。ここでは、seabornに組み込まれている**irisデータセット**を例として使用します。

---

## 5つの代表的なグラフ – 5NG

本書では、以下の5種類のグラフに注目します。これらは一般的に名前が付けられており、以降「5NG」と呼びます：

- **散布図 (scatterplots)**
- **折れ線グラフ (linegraphs)**
- **ヒストグラム (histograms)**
- **箱ひげ図 (boxplots)**
- **棒グラフ (barplots)**

これらの基本的なグラフを覚えておくことで、さまざまな種類の変数を視覚的に表現する際に非常に役立ちます。なお、あるグラフはカテゴリカル変数に適しており、また別のグラフは数値変数に適しています。

---

## 5NG#1: 散布図 (Scatterplots)

散布図は、2つの数値変数間の関係性を視覚化する最も基本的なグラフです。ここでは、irisデータセットの以下の2つの数値変数の関係を散布図で表現します：

- **sepal_length**：x軸
- **petal_length**：y軸

### 基本的な散布図の作成

以下のコードは、irisデータセットを読み込み、`sepal_length` と `petal_length` の関係を散布図で可視化する例です。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Irisデータセットの読み込み
iris = sns.load_dataset("iris")

# 基本的な散布図の作成
sns.scatterplot(data=iris, x="sepal_length", y="petal_length")
plt.title("Irisデータセット: sepal_lengthとpetal_lengthの散布図")
plt.xlabel("sepal_length")
plt.ylabel("petal_length")
plt.show()
```

このコードでは、sns.scatterplot 関数を用いて、irisデータセット内のsepal_length（x軸）とpetal_length（y軸）の各サンプルを点としてプロットしています。図では、各点が1つの花のサンプルを表しており、両変数間の関係性が視覚的に把握できます。

!!! Note
    (LC2.1) IrisデータセットをPandasのDataFrameとして表示し、各変数（sepal_length, sepal_width, petal_length, petal_width, species）の型と内容を確認してみましょう。

### 過剰なプロット（オーバープロッティング）への対処

データ数が非常に多い場合、同じ位置に点が重なってしまい、実際にプロットされている点の数が分かりにくくなる「オーバープロッティング」が発生します。これに対処する方法は主に2つあります。

#### 方法1: 透明度 (alpha) の調整
点の透明度を変更することで、重なった部分が濃く見えるようになり、点の密集度が視覚的に分かりやすくなります。以下は透明度を設定した散布図の例です。

python
コピーする
sns.scatterplot(data=iris, x="sepal_length", y="petal_length", alpha=0.5)
plt.title("透明度を調整した散布図 (alpha=0.5)")
plt.xlabel("sepal_length")
plt.ylabel("petal_length")
plt.show()
ここでは、alpha パラメータを0.5に設定することで、各点の不透明度が50%となり、重なっている部分はより濃く表示されます。

#### 方法2: ジッター (Jitter) の追加
ジッターは、各点に小さなランダムなずれ（ノイズ）を加えて、同じ位置に重なって表示される点を少しずらす手法です。これにより、重なり合っている点が個別に見えるようになります。以下のコードはジッターを加えた散布図の例です。

```python
import numpy as np

# ジッターの強さ（調整可能）
jitter_strength = 0.1

# sepal_lengthとpetal_lengthにランダムなノイズを加える
x_jittered = iris["sepal_length"] + np.random.uniform(-jitter_strength, jitter_strength, size=len(iris))
y_jittered = iris["petal_length"] + np.random.uniform(-jitter_strength, jitter_strength, size=len(iris))

plt.figure()
plt.scatter(x_jittered, y_jittered, alpha=0.7)
plt.title("ジッターを加えた散布図")
plt.xlabel("sepal_length (jittered)")
plt.ylabel("petal_length (jittered)")
plt.show()
```

この例では、np.random.uniform を用いて、各点に -0.1 から 0.1 の範囲のランダムな値を加えています。ジッターはあくまで視覚化のための手法であり、元のデータ自体は変更されません。

Learning Check

(LC2.2) 透明度（alpha）の設定が、重なり合う点の密集度をどのように表現するか考えてみましょう。
(LC2.3) ジッターによって点がわずかにずれると、元のデータのパターンはどのように変化するでしょうか？また、ジッターの強さはどのように調整するのが適切か検討してみてください。

### チェック項目

- (LC2.4) 透明度の調整とジッターの追加、それぞれの手法のメリット・デメリットを比較してみましょう。
- (LC2.5) Irisデータセット内の他の数値変数（例えば、sepal_widthやpetal_width）を用いて、別の散布図を作成し、得られるパターンの違いを観察してみてください。

このように、PythonとGoogle Colaboratoryを利用することで、RやRStudioで行っていたようなグラフィックスの基本操作を手軽に再現し、さまざまな手法でデータの特徴を明らかにすることが可能です。次のセクションでは、折れ線グラフ、ヒストグラム、箱ひげ図、棒グラフの作成方法についても見ていきましょう。

### まとめ
散布図は、2つの数値変数間の関係を直感的に把握するための基本的な可視化手法です。特にデータ量が多い場合、透明度の調整やジッターの追加といった工夫をすることで、重なり合ったデータ点の情報をより明確に伝えることができます。どの手法を用いるかは、データの特性や伝えたい内容に応じて判断する必要があります。


## 5NG#2: 折れ線グラフ (Linegraphs)

折れ線グラフは、x軸に順序性（特に時間などの連続的な情報）がある場合に、2つの数値変数間の関係を視覚化するためのグラフです。隣接するデータ点を線で結ぶことで、時系列データの変化やトレンドを明確に示すことができます。

ここでは、seabornに組み込まれている**flights**データセットを使用して、1949年から1960年までの月ごとの乗客数の推移を折れ線グラフで可視化する方法について説明します。  
flightsデータセットは、以下の変数を含んでいます：

- **year**: 年（1949～1960）
- **month**: 月（"Jan", "Feb", …, "Dec" といった文字列）
- **passengers**: 各月の乗客数

### flightsデータの準備と日時変数の作成

折れ線グラフを作成する際には、x軸に順序性のある変数（ここでは日時）が必要です。  
flightsデータセットでは、`year` と `month` の2つの変数から日時情報を作成する必要があります。以下のコードでは、Pandasを用いてこれらの変数を組み合わせ、`date` という新しい日時型の変数を作成します。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# flightsデータセットの読み込み
flights = sns.load_dataset("flights")
print(flights.head())

# yearとmonthからdatetime型の変数を作成
# flightsデータセットのmonth列は "Jan", "Feb", ... の形式のため、'%Y-%b' を指定
flights['date'] = pd.to_datetime(flights['year'].astype(str) + '-' + flights['month'], format='%Y-%b')
print(flights.head())
```

### 折れ線グラフの作成
作成したdate変数をx軸、passengers変数をy軸として、seabornのlineplot()関数を使い折れ線グラフを描いてみます。以下のコードでは、図のサイズを調整し、タイトルや軸ラベルを設定しています。

```python
plt.figure(figsize=(12,6))
sns.lineplot(data=flights, x="date", y="passengers")
plt.title("1949年～1960年の月ごとの乗客数の推移")
plt.xlabel("日付")
plt.ylabel("乗客数")
plt.show()
```

このコードでは、date（日時型の変数）に沿ってpassengers（乗客数）の各データ点を線で結んでいます。日時の順序が保たれるため、連続したデータの変化を直感的に把握することができます。

### チェック項目
(LC1) flightsデータセットのmonth列はどのような形式になっていますか？また、なぜyearとmonthを組み合わせて日時型の変数を作成する必要があるのでしょうか？
(LC2) 折れ線グラフでデータ点を線で結ぶことは、どのような情報（例：傾向や変化の方向性）を視覚的に伝えるのでしょうか？
(LC3) 他の時系列データ（例：ある都市の月別平均気温など）において、同様の方法で折れ線グラフを作成することは可能でしょうか？その場合、どのような点に注意すべきでしょうか？

### まとめ
折れ線グラフは、時間や順序性のある変数をx軸にとることで、連続したデータの変化を効果的に表現できます。
ここでは、seabornのflightsデータセットを使用し、year と month から作成した日時型の変数を用いて、1949年から1960年までの月ごとの乗客数の推移を線グラフで可視化しました。
この手法は、時系列データの分析やトレンドの把握に非常に有用です。



## 5NG#3: ヒストグラム (Histograms)

ヒストグラムは、1つの数値変数の分布を視覚化するためのグラフです。  ここでは、seaborn に組み込まれている **titanic** データセットを用い、乗客の **age**（年齢）の分布に注目します。  ヒストグラムを使うことで、以下のような点を確認できます。

1. 最小値と最大値はどこか？
2. 中心や「最も典型的な」値はどこか？
3. 値の広がりはどの程度か？
4. 頻出値と稀な値はどこか？

---

### 1. 分布の概要を視覚化する

まず、年齢の各観測値を横一列に並べたプロットを作成して、分布の大まかな様子を確認してみます。  
（このプロットは、後述するヒストグラムと比べるとオーバープロットが発生するため、細かい頻度は読み取りにくいですが、全体の広がりを直感的に把握するのに役立ちます。）

```python
import seaborn as sns
import matplotlib.pyplot as plt

# titanicデータセットの読み込み
titanic = sns.load_dataset("titanic")

# NaNの除外（age列には欠損値が含まれています）
titanic_age = titanic.dropna(subset=["age"])

# 各年齢の値を横一列に並べる（y座標は固定値）
plt.figure(figsize=(10, 1))
sns.stripplot(x="age", data=titanic_age, jitter=False, color="gray")
plt.xlabel("Age")
plt.yticks([])  # y軸の目盛りを非表示に
plt.title("Titanicデータセット：乗客の年齢の散布図（横一列）")
plt.show()
```
このプロットでは、各点が個々の乗客の年齢を示しており、年齢の分布の概形を把握できます。
ただし、多くの点が重なって表示されるため、細かい頻度は見えにくいです。

### 2. ヒストグラムの作成

ヒストグラムでは、x軸（ここでは年齢）をいくつかの「ビン」（区間）に分割し、各区間に含まれる観測値の数（度数）を棒グラフで示します。以下のコードは、デフォルト設定で年齢のヒストグラムを作成する例です。

```python
plt.figure(figsize=(10,6))
sns.histplot(data=titanic_age, x="age")
plt.title("Titanicデータセット：乗客の年齢のヒストグラム (デフォルト設定)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
```

このプロットでは、seaborn が自動的に30個程度のビンに分割してヒストグラムを描画しています。ただし、ビンの数が多すぎる場合、各ビンの幅が狭くなり、分布が「ごちゃごちゃ」して見えることがあります。

### 3. ビンの調整

ヒストグラムの解釈を容易にするために、ビンの数や幅を調整することができます。以下に、ビンの数とビン幅を変更した例を示します。

#### (1) ビンの数を指定する方法

ここでは、bins=20 としてビンの数を20に指定しています。
また、edgecolor="white" を設定することで、各ビンの境界線を白色にし、区切りを明確にしています。

```python
plt.figure(figsize=(10,6))
sns.histplot(data=titanic_age, x="age", bins=20, color="steelblue", edgecolor="white")
plt.title("Titanicデータセット：乗客の年齢のヒストグラム (20ビン)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
```

#### (2) ビン幅を指定する方法

seaborn の histplot() では、binwidth 引数を使って各ビンの幅を直接指定することも可能です。たとえば、ビン幅を5歳に設定する場合は次のようになります。

```python
plt.figure(figsize=(10,6))
sns.histplot(data=titanic_age, x="age", binwidth=5, color="steelblue", edgecolor="white")
plt.title("Titanicデータセット：乗客の年齢のヒストグラム (ビン幅=5)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
```

この例では、年齢が5歳刻みで区切られ、各区間における乗客数が表示されます。

### 4. Faceting（ファセット）による分割表示

ファセットは、ある変数の値ごとに同じ種類のプロットを複数描画し、比較しやすくする手法です。
ここでは、sex（性別）によって乗客の年齢のヒストグラムを分割して表示してみます。

```python
# seaborn の displot() は、ヒストグラムのファセット表示に適しています
sns.displot(data=titanic_age, x="age", bins=20, color="steelblue", edgecolor="white", col="sex", height=4, aspect=1.2)
plt.suptitle("Titanicデータセット：性別ごとの乗客の年齢のヒストグラム", y=1.03)
plt.show()
```

このコードでは、col="sex" により、性別（male と female）ごとに別々のヒストグラムが描画され、各グループ内で年齢の分布を比較できます。

### チェック項目
- (LC1) titanic データセットにおける age 変数の最小値、最大値、中央値はそれぞれどの程度か調べてみましょう。
（例: titanic_age["age"].describe() を用いて確認できます）

- (LC2) ヒストグラムでビンの数や幅を変更すると、分布の解釈にどのような影響があるか考えてみましょう。
（例: ビンの数を増やすと細部が見えやすくなる一方、ノイズが目立つ可能性があります）

- (LC3) 性別（sex）ごとに年齢の分布が異なる理由について、データや歴史的背景から考察してみましょう。

### まとめ
ヒストグラムは、1つの数値変数の分布（最小値、最大値、中心、散らばり、頻出値など）を視覚化するための有効なツールです。ここでは、seaborn の titanic データセットの age 変数を例に、基本のヒストグラムの作成、ビンの調整方法、さらにファセットを用いたグループ別比較の方法を示しました。これらの手法を用いることで、データの特徴や潜在的なパターンを直感的に理解することができます。


## 5NG#4: 箱ひげ図 (Boxplots)

箱ひげ図は、数値変数の**五数要約**（最小値、第1四分位数、中央値、第3四分位数、最大値）に基づいて、データの分布を視覚化する手法です。  
ここでは、seaborn の **tips** データセットを用い、飲食代（`total_bill`）の分布を例に箱ひげ図の作成方法と解釈について説明します。

---

### 1. 特定グループにおける五数要約の確認

まず、tips データセットからディナータイム（`time` が `"Dinner"`）のデータに注目し、その `total_bill` の五数要約を算出します。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# tips データセットの読み込み
tips = sns.load_dataset("tips")

# ディナータイムのデータに絞る
dinner_tips = tips[tips["time"] == "Dinner"]

# ディナータイムの total_bill に対する五数要約の計算
n_dinner = dinner_tips.shape[0]
min_dinner = dinner_tips["total_bill"].min()
max_dinner = dinner_tips["total_bill"].max()
quartiles = dinner_tips["total_bill"].quantile([0.25, 0.5, 0.75])
five_number_summary = pd.DataFrame({
    "total_bill": [min_dinner, quartiles.iloc[0], quartiles.iloc[1], quartiles.iloc[2], max_dinner]
}, index=["Min", "25%", "50%", "75%", "Max"])

print("ディナータイムの total_bill の五数要約:")
print(five_number_summary)
```
このコードを実行すると、ディナータイムにおける total_bill の最小値、第一四分位数、中央値、第三四分位数、最大値が得られます。

### 箱ひげ図作成の段階的な構築
以下では、ディナータイムの total_bill の観測値を対象に、3種類のプロットを段階的に作成していきます。

(a) ジッターを加えたプロットと五数要約の値の表示
まず、各観測値をジッター（わずかな横方向のずれ）を付与してプロットし、さらに五数要約の各値を赤の破線で示します。

```python
# サブプロットの作成（横に3つ並べる）
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# プロットの基本設定：x 軸は "Dinner" の固定値
x_value = ["Dinner"] * n_dinner

# (a) ジッターを加えた点のプロット
sns.stripplot(x=dinner_tips["time"], y=dinner_tips["total_bill"],
              jitter=True, ax=axes[0], color="gray", alpha=0.3)
axes[0].set_title("ジッター付き点と\n五数要約の破線")
axes[0].set_xlabel("")
axes[0].set_ylabel("Total Bill")

# 五数要約の各値を破線で追加
for value in five_number_summary["total_bill"]:
    axes[0].axhline(y=value, color='red', linestyle='--', linewidth=1.0)
    
# (b) 箱ひげ図とジッター付き点、さらに破線による五数要約の表示
sns.boxplot(x="time", y="total_bill", data=dinner_tips, ax=axes[1],
            showcaps=True, boxprops={'facecolor':'None'})
sns.stripplot(x="time", y="total_bill", data=dinner_tips,
              jitter=True, ax=axes[1], color="gray", alpha=0.3)
for value in five_number_summary["total_bill"]:
    axes[1].axhline(y=value, color='red', linestyle='--', linewidth=1.0)
axes[1].set_title("箱ひげ図 + ジッター付き点")
axes[1].set_xlabel("")
axes[1].set_ylabel("")

# (c) 箱ひげ図のみ（余計な点や破線は除去）
sns.boxplot(x="time", y="total_bill", data=dinner_tips, ax=axes[2])
axes[2].set_title("箱ひげ図のみ")
axes[2].set_xlabel("")
axes[2].set_ylabel("")

plt.tight_layout()
plt.show()
```

- 左のプロットでは、ディナータイムの各 total_bill の観測値がジッター付きの点として表示され、赤い破線で五数要約の位置が示されています。
- 中央のプロットでは、箱ひげ図が描画されると同時に、ジッター付きの点と破線も重ねられています。
- 右のプロットは、純粋な箱ひげ図のみで、余計な要素が除去されています。

箱ひげ図は、箱の上下がそれぞれ第一四分位数（25%）と第三四分位数（75%）を示し、箱内の太い線が中央値（50%）を示します。箱の高さは四分位範囲 (IQR) を表し、上下の「ひげ」は通常、箱から 1.5 × IQR の範囲内の最小値・最大値まで伸び、範囲外の観測値は個別の点（アウトライヤー）として表示されます。

### 複数グループにおける箱ひげ図の作成

次に、tips データセット全体を用いて、曜日（day）ごとに total_bill の分布を比較するための サイドバイサイド箱ひげ図 を作成します。
tips データセットの day 変数は既にカテゴリカル変数（"Thur", "Fri", "Sat", "Sun"）となっているため、x 軸に直接割り当てることができます。

```python
plt.figure(figsize=(8, 6))
sns.boxplot(x="day", y="total_bill", data=tips,
            palette="pastel", showcaps=True, 
            boxprops={'edgecolor':'black'},
            whiskerprops={'color':'black'},
            capprops={'color':'black'},
            flierprops={'markerfacecolor':'red', 'marker':'o', 'markersize':5})
plt.title("Tipsデータセット：曜日ごとの Total Bill の箱ひげ図")
plt.xlabel("Day")
plt.ylabel("Total Bill")
plt.show()
```

このプロットでは、各曜日ごとに total_bill の箱ひげ図が描画され、

- 箱は第一四分位数、中央値、第三四分位数を示し、
- ひげは 1.5×IQR の範囲内の値を表し、
- その範囲外の値はアウトライヤーとして赤い点で示されています。

曜日ごとの箱ひげ図を並べることで、各グループ間の分布の違いやばらつき、さらには外れ値の有無を簡単に比較することができます。

### チェック項目

- (LC1) ディナータイムの total_bill の五数要約を確認したとき、各値（最小値、第一四分位数、中央値、第三四分位数、最大値）はどのような数値になっていますか？
- (LC2) 箱ひげ図における箱の高さ（IQR）が大きい場合、どのような情報が得られますか？
- (LC3) アウトライヤー（箱ひげ図上の点）が示す意味は何でしょうか？また、アウトライヤーが多い場合、どのような解釈が可能でしょうか？
- (LC4) 曜日ごとの箱ひげ図を比較することで、どの曜日で飲食代のばらつきが大きいか、また中央値に違いがあるかをどのように読み取れますか？

### まとめ

箱ひげ図は、単一の数値変数の分布を視覚化する強力なツールであり、

- 中央値 や 四分位数 を直感的に把握でき、
- 四分位範囲（IQR） による散らばりの度合いを示し、
- アウトライヤー（外れ値）を容易に特定することができます。

また、カテゴリカル変数（例：曜日）に対してサイドバイサイドで箱ひげ図を作成することで、グループ間の分布の違いやばらつきを比較することが可能です。tips データセットを例に、ディナータイムの total_bill や曜日ごとの total_bill を通じて、箱ひげ図の作成とその解釈の基本を確認しました。


## 5NG#5: 棒グラフ (Barplots)

棒グラフは、カテゴリカル変数の各カテゴリー（レベル）の出現頻度（カウント）を視覚化するための基本的なグラフです。  
数値変数の分布を可視化するヒストグラムや箱ひげ図とは異なり、棒グラフは各カテゴリーの頻度や割合を示すのに適しています。  
また、データが既に「事前集計済み」の場合と、各観測値が個別に記録されている場合で、グラフ作成の手法が異なります。

本節では、[Gapminder](https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv) のデータセットを用いて、カテゴリカル変数（ここでは `continent`：大陸）の出現頻度を棒グラフで可視化する方法を示します。

---

### 1. データの読み込み

まず、Gapminder のデータセットを URL から読み込みます。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Gapminderデータセットの読み込み
url = "https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv"
gapminder = pd.read_csv(url)
gapminder.head()
```

データセットには、`country`（国）、`continent`（大陸）、`year`（年）、`lifeExp`（平均寿命）、`pop`（人口）、`gdpPercap`（1人当たりGDP）などの変数が含まれています。

### 未集計データから棒グラフを作成する

Gapminder のデータセットは各観測値（例えば、各国の各年の記録）が個別に記録されているため、カテゴリカル変数 `continent` の頻度はまだ「集計」されていません。
この場合、Seaborn の `countplot()` を使用すると、自動的に各カテゴリーの出現回数を計算して棒グラフを描画してくれます。

```python
plt.figure(figsize=(8,6))
sns.countplot(data=gapminder, x="continent", palette="pastel", edgecolor="black")
plt.title("Gapminderデータセット：各大陸のレコード数")
plt.xlabel("大陸")
plt.ylabel("カウント")
plt.show()
```

このグラフは、各大陸に該当するレコード数（＝その大陸に属する国の数×観測年数）がどの程度かを示しています。棒同士の間には適度な隙間があり、各カテゴリーの頻度を比較しやすくなっています。

### 事前集計済みデータから棒グラフを作成する
一方、データが「事前集計済み」である場合、つまり各カテゴリーのカウントが既に計算されている場合は、Seaborn の `barplot()` を使用して棒グラフを作成します。

まず、continent ごとのレコード数を集計してみます。

```python
# 大陸ごとのレコード数を計算
continent_counts = gapminder.groupby("continent").size().reset_index(name="count")
print(continent_counts)
```

次に、事前集計済みのデータを用いて棒グラフを描画します。

```python
plt.figure(figsize=(8,6))
sns.barplot(data=continent_counts, x="continent", y="count", palette="pastel", edgecolor="black")
plt.title("Gapminderデータセット：大陸ごとのレコード数（事前集計済み）")
plt.xlabel("大陸")
plt.ylabel("カウント")
plt.show()
```

この方法では、各カテゴリーのカウントが既に `count` 変数に記録されているため、`barplot()` の `y` 軸にその変数を指定します。
結果として、未集計のデータを用いた `countplot()` と同じグラフが得られます。

### 4. 追加例：特定の年における棒グラフ
Gapminder のデータセットは複数年のデータを含むため、例えば特定の年（例：2007年）のデータに絞って、各大陸の「国数」を可視化することも可能です。2007年のデータは各国1レコードで表されるので、各大陸における国数を示す棒グラフになります。

```python
# 2007年のデータに絞る
gapminder_2007 = gapminder[gapminder["year"] == 2007]

plt.figure(figsize=(8,6))
sns.countplot(data=gapminder_2007, x="continent", palette="pastel", edgecolor="black")
plt.title("Gapminderデータセット（2007年）：各大陸の国数")
plt.xlabel("大陸")
plt.ylabel("国数")
plt.show()
```

このグラフでは、各大陸ごとの国数が比較しやすく表示されます。

### チェック項目
- (LC1) ヒストグラムは数値変数の連続的な分布を可視化するのに適していますが、なぜカテゴリカル変数には適していないのでしょうか？
- (LC2) 未集計のデータから棒グラフを作成する場合と、事前集計済みのデータから作成する場合の違いは何ですか？
- (LC3) Gapminder データセットの `continent` 変数において、どの大陸のレコード数（または国数）が多いか、またその理由について考えてみましょう。

### まとめ
棒グラフは、カテゴリカル変数の各カテゴリーの出現頻度を直感的に示すグラフです。

- 未集計データの場合は、Seaborn の countplot() を用いることで自動的に頻度を計算・表示できます。
- 事前集計済みデータの場合は、barplot() を用いて、あらかじめ計算されたカウントをプロットします。

Gapminder のデータセットを用いた例では、continent の各カテゴリーの頻度や、特定の年における国数など、さまざまな角度からカテゴリカル変数の分布を視覚化する方法を確認しました。これにより、データ全体の傾向や、グループ間の比較が容易になります。



## データ可視化のまとめ

本章では、以下の 5 つの代表的なグラフ（5NG）を通じて、さまざまなデータの分布や変数間の関係性を視覚化する方法を学びました。これらのグラフを駆使することで、あらゆるデータセットの特徴を直感的に把握できるようになります。また、さらに多くの変数を各プロットの美的属性（色、形、大きさなど）にマッピングすることで、視覚的表現の可能性は無限に広がります。

| グラフの種類   | 表示内容                                                    | ジオメトリックオブジェクト                                              | 備考                                                                                       |
|----------------|-------------------------------------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| 散布図         | 2つの数値変数間の関係性                                      | `plt.scatter()` または `sns.scatterplot()`                               |                                                                                            |
| 折れ線グラフ   | 2つの数値変数間の関係性                                      | `plt.plot()` または `sns.lineplot()`                                     | x軸に順序がある場合（例：時間）に使用                                                       |
| ヒストグラム   | 1つの数値変数の分布                                          | `plt.hist()` または `sns.histplot()`                                     | ファセットヒストグラムは、別の変数の値で分割した数値変数の分布を表示するのに用いられる         |
| 箱ひげ図       | 1つの数値変数の分布を、別の変数で分割して表示                 | `plt.boxplot()` または `sns.boxplot()`                                   |                                                                                            |
| 棒グラフ       | 1つのカテゴリカル変数の分布                                  | 未集計データの場合：`sns.countplot()`<br>事前集計済みデータの場合：`sns.barplot()` | 積み上げ、並列、ファセット棒グラフを用いると、2つのカテゴリカル変数の同時分布も表現可能       |


### 関数引数の指定

Python の多くの関数は、キーワード引数を使用してパラメータを受け取ります。例えば、Seaborn のプロット関数では、引数の順序に依存せず、以下のように引数名を入れ替えても同じ結果が得られます。

```python
import seaborn as sns

# セグメント 1: data と x を明示的に指定
sns.countplot(data=gapminder, x="continent")

# セグメント 2: 引数の順序を入れ替えても、キーワード引数なので同じ結果に
sns.countplot(x="continent", data=gapminder)
```

上記の例のように、Python ではキーワード引数の順序は任意であり、コードの読みやすさを重視して好みのスタイルで記述できます。

### 追加のリソース

データ可視化の力をさらに引き出すため、以下のリソースを参考にしてください。

- [Seaborn ドキュメント](https://seaborn.pydata.org/)
    Seaborn の使い方や豊富なプロット例が掲載されています。

- [Matplotlib チートシート](https://matplotlib.org/stable/cheatsheets/index.html)
    Matplotlib の主要な関数や設定方法を簡潔にまとめた資料です。

これらの資料は、基本的なグラフ作成の知識を超えて、さらに高度な視覚化技法を習得する際に大いに役立ちます。

### 次の話題

これまで、散布図、折れ線グラフ、ヒストグラム、箱ひげ図、棒グラフという 5 つの基本的なグラフを用いて、データの分布や変数間の関係性を視覚化する方法を学びました。
次の章では、データ前処理・整形（Data Wrangling） に焦点を当て、Pandas を用いたデータのフィルタリングや変換方法を詳しく解説します。
たとえば、以下のようなコードで、データの部分集合を作成し、可視化に活用する方法を学びます。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 例 1: flights データセットから Alaska Airlines のデータのみを抽出
alaska_flights = flights[flights["carrier"] == "AS"]
sns.scatterplot(data=alaska_flights, x="dep_delay", y="arr_delay")
plt.title("Alaska Airlines の出発遅延と到着遅延の関係")
plt.show()

# 例 2: weather データセットから Newark 空港 (EWR) の 1 月上旬のデータのみを抽出
early_january_weather = weather[(weather["origin"] == "EWR") & (weather["month"] == 1) & (weather["day"] <= 15)]
sns.lineplot(data=early_january_weather, x="time_hour", y="temp")
plt.title("Newark 空港 1 月上旬の温度変化")
plt.show()
```

上記の例は、Pandas を用いたデータの抽出と、Seaborn を用いた可視化の基本的な連携例です。次章では、これらのデータ整形技法をより詳細に学び、複雑なデータ分析や可視化のための下地を作っていきます。

---

これで本章のまとめと、今後の展開についての概要を終わります。
次の章では、データの前処理と変換を通じて、より効果的なデータ分析の基盤を構築していきましょう！