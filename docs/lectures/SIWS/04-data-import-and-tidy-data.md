# Data Importing and "Tidy" Data

## はじめに

データサイエンスにおいて、**データフレーム**（表形式データ）は、各行が観測（例：個々の飛行機のフライト、国ごとの統計など）、各列が観測の変数（例：年齢、スコア、日付など）を表すものです。本章では、R 版「Data Science with tidyverse」で扱われた内容をもとに、  
・データのインポート方法  
・「整然とした（tidy）」データの概念とその重要性  
・ワイド形式とロング形式のデータ変換  
・整然なデータを用いた可視化  
などを、Python の主要ライブラリ（numpy, pandas, matplotlib, seaborn）を使って実演します。

「**tidy（整然とした）**データ」とは、以下の3原則に基づくデータ構造のことです：

1. **各変数は 1 列に格納される**  
   例：身長、体重、年齢などはそれぞれ独立した列に配置する。

2. **各観測は 1 行に格納される**  
   例：1 人の被験者の全情報が 1 行にまとまっている。

3. **各観測単位は別のテーブルにまとめる**  
   例：学生データと教師データが混在せず、それぞれ個別のテーブルに保存される。

これらの原則により、データのフィルタリング、変形、集約、結合、可視化などの操作が効率よく行えるようになります。  
また、後続の解析（回帰分析や統計的推論など）においても、この整然としたデータ形式は必須です。

---

## 1. データのインポート

ここでは、実際のデータを自分のコンピュータやオンライン上から読み込む方法について説明します。  
スプレッドシートのデータは、主に以下の形式で保存されます。

- **CSV（Comma Separated Values）ファイル**  
  各行が観測（1行＝1観測）、各値はカンマで区切られ、最初の行がヘッダーの場合が多い。

- **Excel（.xlsx）ファイル**  
  Microsoft Excel の形式で、CSV よりも多くのメタデータ（フォント、セルの色、列幅、数式など）を含む。

- **Google Sheets**  
  クラウド上のスプレッドシート。CSV や Excel 形式でダウンロード可能。

### 1.1 CSV ファイルのインポート

ここでは、インターネット上にある CSV ファイル `dem_score.csv`（<https://moderndive.com/data/dem_score.csv>）を pandas の `read_csv()` 関数を用いて読み込みます。

```python
import pandas as pd

# CSV ファイルをオンラインから読み込む
dem_score = pd.read_csv("https://moderndive.com/data/dem_score.csv")
print("【dem_scoreデータフレーム】")
print(dem_score.head())
```

このデータでは、各国の民主主義度（democracy_score）が測定されており、値が -10 で高度な独裁国家、10 で高度な民主国家を表します。

### 1.2 Excel ファイルのインポート

Excel ファイルの場合は、`read_excel()` 関数を使用します。  
※ Python では RStudio のような GUI を用いたインポートは標準では提供されませんが、ファイルエクスプローラー等でダウンロードした後、以下のように読み込みます。

```python
# 事前に "dem_score.xlsx" をダウンロードして作業ディレクトリに保存しておく
dem_score_excel = pd.read_excel("dem_score.xlsx")
print("【dem_score (Excel) データフレーム】")
print(dem_score_excel.head())
```

---

## 2. "Tidy" Data（整然としたデータ）

ここからは、整然とした（tidy）データの概念と、ワイド形式⇄ロング形式への変換の例を紹介します。

### 2.1 ワイド形式とロング形式の例

#### ワイド形式の例

次の例は、各国ごとに 1999 年と 2000 年の「疾病発生件数（cases）」が記録されたワイド形式のデータです（各年が個別の列になっています）。

```python
# ワイド形式のデータ例
df_wide = pd.DataFrame({
    'country': ['Afghanistan', 'Brazil', 'China'],
    '1999': [745, 37737, 212258],
    '2000': [2666, 80488, 213766]
})
print("【ワイド形式のデータ】")
print(df_wide)
```

#### ワイド形式からロング形式への変換

pandas の `melt()` 関数を用いることで、ワイド形式のデータをロング形式（整然としたデータ）に変換できます。  
ここでは、`country` 列はそのまま残し、残りの各列（年）を 1 つの変数 `year` として、対応する値を `cases` 列にまとめます。

```python
# ワイド形式からロング形式へ変換
df_long = pd.melt(df_wide, id_vars=['country'], var_name='year', value_name='cases')
print("\n【ロング形式に変換したデータ】")
print(df_long)
```

#### ロング形式からワイド形式への再変換

逆に、`pivot()` や `pivot_table()` 関数を使って、ロング形式のデータをワイド形式に戻すことも可能です。

```python
# ロング形式からワイド形式に再変換
df_wide_again = df_long.pivot(index='country', columns='year', values='cases').reset_index()
print("\n【ロング形式から再構築したワイド形式のデータ】")
print(df_wide_again)
```

---

## 3. 整然データの利点を活かした可視化

整然としたデータ形式は、データの可視化や操作を非常に効率よく行うための前提条件となります。  
以下の例では、整形前のデータ `drinks_smaller`（ワイド形式）を、整然なロング形式に変換してから、国ごとに各種アルコールの摂取量をサイドバイサイドの棒グラフで比較しています。

### 3.1 例：drinks データの整形と棒グラフの作成

まず、サンプルデータとして「drinks」データ（アルコール摂取量のデータ）を作成します。  
ここでは、アメリカ、China、Italy、Saudi Arabia の 4 カ国のデータとして、ビール、スピリッツ、ワインの摂取量を用います。

```python
import pandas as pd

# サンプルデータ：drinks_smaller（ワイド形式）
drinks_smaller = pd.DataFrame({
    'country': ['USA', 'China', 'Italy', 'Saudi Arabia'],
    'beer_servings': [245, 89, 212, 15],
    'spirit_servings': [150, 110, 98, 5],
    'wine_servings': [50, 20, 300, 0]
})

# 列名の変更（rename）
drinks_smaller = drinks_smaller.rename(columns={
    'beer_servings': 'beer',
    'spirit_servings': 'spirit',
    'wine_servings': 'wine'
})
print("【drinks_smaller (ワイド形式)】")
print(drinks_smaller)
```

次に、pandas の `melt()` 関数を用いて、ワイド形式のデータをロング形式（整然データ）に変換します。

```python
# ワイド形式からロング形式へ変換
drinks_smaller_tidy = pd.melt(drinks_smaller, id_vars=['country'],
                              var_name='type', value_name='servings')
print("\n【drinks_smaller_tidy (整然なロング形式のデータ)】")
print(drinks_smaller_tidy)
```

この整然なデータを用いて、seaborn の `barplot()` を使いサイドバイサイドの棒グラフを作成します。

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

plt.figure(figsize=(8, 4))
sns.barplot(data=drinks_smaller_tidy, x='country', y='servings', hue='type', dodge=True)
plt.xlabel("国")
plt.ylabel("サービング数")
plt.title("4 カ国におけるアルコール摂取量の比較")
plt.show()
```

---

## 4. "Tidy" Data の定義と例

整然としたデータ（tidy data）とは、以下のルールに従って構造化されたデータです：

1. **各変数は 1 列に**  
2. **各観測は 1 行に**  
3. **各観測単位は 1 テーブルに**

例えば、以下は株価データのワイド形式（非 tidy）の例です。

```python
import pandas as pd

stocks = pd.DataFrame({
    'Date': pd.date_range("2009-01-01", periods=5),
    'Boeing stock price': ["$173.55", "$172.61", "$173.86", "$170.77", "$174.29"],
    'Amazon stock price': ["$174.90", "$171.42", "$171.58", "$173.89", "$170.16"],
    'Google stock price': ["$174.34", "$170.04", "$173.65", "$174.87", "$172.19"]
})
print("【株価データ（非 tidy）】")
print(stocks.head(2))
```

このデータは、各銘柄ごとに列が分かれているため非 tidy です。  
これを、各観測単位（Date）ごとに「Stock Name」と「Stock Price」という 2 列に変換すると tidy になります。

```python
# ワイド形式からロング形式（tidy 形式）へ変換
stocks_tidy = stocks.melt(id_vars=['Date'], 
                          var_name='Stock Name', 
                          value_name='Stock Price')
print("\n【株価データ（tidy 形式）】")
print(stocks_tidy.head(6))
```

なお、同様に他のデータ（例：気象情報や航空安全情報など）も、整然なデータ形式にすることで、pandas や seaborn、matplotlib での解析・可視化が容易になります。

---

## 5. Case Study: Democracy in Guatemala

ここでは、先ほどインポートした `dem_score` データセットを用い、Guatemala（グアテマラ）の民主主義スコアの時系列変化をプロットする例を示します。  
このデータは、1952 年から 1992 年までの各年のスコアがワイド形式になっていますので、まず整然な（ロング形式）データに変換します。

### 5.1 グアテマラのデータ抽出

```python
# dem_score は既に上記で読み込んであるものとする
# Guatemala のみ抽出
guat_dem = dem_score[dem_score['country'] == "Guatemala"]
print("【Guatemala のデータ (ワイド形式)】")
print(guat_dem)
```

### 5.2 ワイド形式からロング形式への変換

ここでは、`pd.melt()` を用いて、列名（各年）を `year` 変数、値を `democracy_score` 変数として抽出します。  
※ pandas では、列名は文字列として扱われるため、必要に応じて型変換します。

```python
# country 列はそのままで、残りの列（年）を melt する
guat_dem_tidy = pd.melt(guat_dem, id_vars=['country'],
                        var_name='year', value_name='democracy_score')
# year 列を整数型に変換
guat_dem_tidy['year'] = guat_dem_tidy['year'].astype(int)
print("\n【Guatemala のデータ (tidy 形式)】")
print(guat_dem_tidy)
```

### 5.3 時系列プロットの作成

seaborn または matplotlib の `plot` 機能を使って、1952 年から 1992 年までの民主主義スコアの変化を線グラフで描きます。

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(guat_dem_tidy['year'], guat_dem_tidy['democracy_score'], marker='o', linestyle='-')
plt.xlabel("Year")
plt.ylabel("Democracy Score")
plt.title("Guatemala における 1952～1992 年の民主主義スコアの変化")
plt.grid(True)
plt.show()
```

※ もし `year` 列の型変換を行わなかった場合、プロット時に正しい順序で並ばないエラーが出る可能性があります。

---

## 6. Python における "Tidy" Data 処理のまとめ

本章では、以下の内容を学びました：

- **データのインポート**  
  - pandas の `read_csv()` や `read_excel()` を用いた CSV／Excel データの読み込み

- **整然とした（tidy）データの概念**  
  - 各変数は 1 列、各観測は 1 行、各観測単位は 1 テーブルにまとめるというルール

- **データの変形操作**  
  - ワイド形式からロング形式への変換：`pd.melt()` を使用  
  - ロング形式からワイド形式への変換：`pivot()` / `pivot_table()` を使用

- **整然データを用いた可視化**  
  - seaborn や matplotlib を用いた棒グラフや線グラフの作成

整然なデータ形式により、Python のデータ操作ライブラリ間で一貫した入力・出力が可能となり、解析・可視化のプロセスが大幅に簡略化されます。

---

## 7. 追加リソースと今後の展開

Python では、pandas、matplotlib、seaborn に加えて、numpy などもデータ解析で頻繁に利用されます。  
R の tidyverse パッケージのように、Python には一連のパッケージが連携して動作するため、データの整形、解析、可視化の各工程がスムーズに連携します。

また、以下のリソースも参考にしてください：

- [pandas ドキュメント](https://pandas.pydata.org/docs/)
- [seaborn ドキュメント](https://seaborn.pydata.org/)
- [matplotlib チートシート](https://matplotlib.org/stable/contents.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

---

## 今後の展開

おめでとうございます！  
これで「Python によるデータのインポートと整然としたデータ」の内容を学習しました。  
次のステップとして、これまでの前処理・可視化のスキルを活かして、回帰分析や統計的推論など、さらに高度なデータモデリングに進んでいきます。

---

<!-- 結果の補足として、参考用のコードファイルへのリンク（必要に応じて追加） -->
```

---
