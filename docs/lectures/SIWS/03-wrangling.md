## データラングリング入門：Python `pandas` を使ったデータ加工

**スライド**[こちら](./Marp/03-visualization-slide.pdf)

これまでの学習で、データフレームの基本的な確認方法や、`matplotlib`や`seaborn`といったライブラリを使ったデータの可視化について学んできました。特に、以下のような基本的なグラフの作成方法に触れたことでしょう。

1.  **散布図**: 2つの量的変数の関係性を見る（`geom_point()`）。
2.  **折れ線グラフ**: 時系列データや順序のあるデータの変化を見る（`geom_line()`）。
3.  **箱ひげ図**: データの分布やばらつき、外れ値を把握する（`geom_boxplot()`）。
4.  **ヒストグラム**: 単一の量的変数の分布形状を把握する（`geom_histogram()`）。
5.  **棒グラフ**: カテゴリごとの量の比較を行う（`geom_bar()`または`geom_col()`）。

これらの可視化は、データフレーム内の情報をグラフ上の要素（点、線、色、形など）に対応付けることで作成されました。

この章では、データ分析の前処理として非常に重要な**データラングリング**（データ加工や整形とも呼ばれます）の基本操作を、Pythonの`pandas`ライブラリを使って学びます。主な操作は以下の通りです。

1.  **行のフィルタリング**: 条件に合う行だけを選ぶ。
2.  **列の集計**: 列データから平均や合計などの要約統計量を計算する。
3.  **グループ化**: 特定の列の値に基づいて行をグループにまとめ、グループごとの集計を行う。
4.  **列の作成・変換**: 既存の列から新しい列を作ったり、列の値を変換したりする。
5.  **並べ替え**: 特定の列の値に基づいて行を並び替える。
6.  **結合**: 複数のデータフレームを特定のキーを使ってつなぎ合わせる。

`pandas`を学ぶことは、データベース操作言語であるSQL（Structured Query Query Language）の考え方を理解する上でも役立ちます。SQLは大量のデータを効率的に扱うために多くの組織で利用されており、`pandas`の操作はSQLのクエリと似ている部分が多くあります。

### 準備：ライブラリのインポートとデータセットの読み込み

まず、この章で使用するライブラリをインポートし、`seaborn`パッケージに含まれる`iris`データセットを読み込みましょう。`iris`データセットは、アヤメという花の品種ごとの特徴（がくの長さ・幅、花びらの長さ・幅）を記録した有名なデータセットです。

```python
import pandas as pd
import seaborn as sns
import numpy as np # numpyも数値計算でよく使われます

# seabornからirisデータセットをロード
iris_df = sns.load_dataset('iris')

# データの最初の5行を確認
print("--- データセットの最初の5行 ---")
print(iris_df.head())
print("\n--- データセットの基本情報 ---")
print(iris_df.info())
print("\n--- データセットの要約統計量 ---")
print(iris_df.describe())
```

---

### メソッドチェーン：処理を繋げる書き方

`pandas`では、複数のデータ操作を連続して行う際に**メソッドチェーン**という書き方がよく使われます。これは、ある操作の結果に対して、続けて次の操作を`.`（ドット）で繋げて記述する方法です。

例えば、データフレーム `df` に対して、操作 `f()`、`g()`、`h()` を順番に適用する場合、
`result = df.f().g().h()`
のように書けます。これは以下のように読み解けます。

1.  `df` を用意する。
2.  `df` に `f()` を適用する。
3.  その結果に `g()` を適用する。
4.  さらにその結果に `h()` を適用する。

この章で学ぶデータラングリング操作の多くは、このメソッドチェーンを使って簡潔に記述できます。

---

### 1. 行のフィルタリング：条件に合うデータを選ぶ (`query`メソッドやブールインデックス)

データセットの中から、特定の条件を満たす行だけを抽出する操作です。

例えば、`iris`データセットから品種 (`species`) が `setosa` のデータだけを選んでみましょう。

```python
# 品種が 'setosa' のデータを抽出
setosa_df = iris_df[iris_df['species'] == 'setosa']
print("--- 'setosa' のデータ ---")
print(setosa_df.head())

# queryメソッドを使った場合 (よりSQLライクな書き方)
setosa_df_query = iris_df.query("species == 'setosa'")
print("\n--- queryメソッドで 'setosa' のデータ ---")
print(setosa_df_query.head())
```

複数の条件を組み合わせることも可能です。

* `&`: AND (かつ)
* `|`: OR (または)
* `>`: より大きい
* `<`: より小さい
* `>=`: 以上
* `<=`: 以下
* `!=`: 等しくない
* `~`: NOT (否定)

例えば、「品種が `versicolor` で、かつ花びらの長さ (`petal_length`) が4.5cmより大きい」データを選んでみましょう。

```python
versicolor_long_petal = iris_df[
    (iris_df['species'] == 'versicolor') & (iris_df['petal_length'] > 4.5)
]
print("--- 'versicolor' で花びらが4.5cmより大きいデータ ---")
print(versicolor_long_petal)

# queryメソッドを使った場合
versicolor_long_petal_query = iris_df.query("species == 'versicolor' and petal_length > 4.5")
print("\n--- queryメソッドで 'versicolor' で花びらが4.5cmより大きいデータ ---")
print(versicolor_long_petal_query)
```

リストに含まれる値でフィルタリングするには `isin()` メソッドが便利です。
例えば、品種が `setosa` または `virginica` のデータを選んでみましょう。

```python
setosa_virginica_df = iris_df[iris_df['species'].isin(['setosa', 'virginica'])]
print("--- 'setosa' または 'virginica' のデータ (最初の数行) ---")
print(setosa_virginica_df.head())
print("\n--- 'setosa' または 'virginica' のデータ (最後の数行) ---")
print(setosa_virginica_df.tail())
```

---

### 2. 列の集計：データの特徴を掴む (`agg`メソッドなど)

データフレームの列に対して、平均、合計、最大値、最小値などの**要約統計量**を計算する操作です。

例えば、`iris`データセットの `sepal_length`（がくの長さ）の平均と標準偏差を計算してみましょう。

```python
mean_sepal_length = iris_df['sepal_length'].mean()
std_sepal_length = iris_df['sepal_length'].std()

print(f"がくの長さの平均: {mean_sepal_length:.2f} cm")
print(f"がくの長さの標準偏差: {std_sepal_length:.2f} cm")

# 複数の統計量を一度に計算 (aggメソッド)
summary_stats = iris_df['sepal_length'].agg(['mean', 'std', 'min', 'max'])
print("\n--- sepal_lengthの要約統計量 ---")
print(summary_stats)
```

データに欠損値 (`NaN`) が含まれる場合、多くの集計関数はデフォルトでこれらを無視して計算します（例: `mean(skipna=True)` がデフォルト）。

主な集計関数：
* `mean()`: 平均値
* `std()`: 標準偏差
* `min()`, `max()`: 最小値、最大値
* `median()`: 中央値
* `sum()`: 合計
* `count()`: 件数 (非欠損値の数)
* `quantile()`: 分位数 (例: `quantile(0.25)`で第一四分位数)

---

### 3. グループ化：グループごとの特徴を見る (`groupby`メソッド)

特定の列の値に基づいてデータをグループに分け、それぞれのグループに対して集計処理を行う操作です。

例えば、`iris`データセットで、品種 (`species`) ごとに各特徴量の平均値を計算してみましょう。

```python
# 品種ごとにグループ化し、各特徴量の平均値を計算
species_mean_df = iris_df.groupby('species').mean()
print("--- 品種ごとの平均値 ---")
print(species_mean_df)

# 特定の列だけ集計することも可能
petal_length_mean_by_species = iris_df.groupby('species')['petal_length'].mean()
print("\n--- 品種ごとの花びらの長さの平均 ---")
print(petal_length_mean_by_species)

# 複数の集計を行うことも可能
petal_summary_by_species = iris_df.groupby('species')['petal_length'].agg(['mean', 'std', 'count'])
print("\n--- 品種ごとの花びらの長さの集計 (平均, 標準偏差, 個数) ---")
print(petal_summary_by_species)
```
`groupby()` を実行しただけではデータフレームは直接変更されず、グループ化されたオブジェクトが生成されます。その後に集計関数（`mean()`, `sum()`, `agg()`など）を適用することで、実際の計算が行われます。

---

### 4. 列の作成・変換：新しい情報を加える (`assign`メソッドや直接代入)

既存の列の値を使って新しい列を作成したり、列の値を変換したりする操作です。

例えば、`iris`データセットに、花びらの面積 (`petal_area`) を計算して新しい列として追加してみましょう（簡単のため、`petal_length * petal_width` で計算します）。

```python
# 方法1: 直接代入
iris_df_mutated = iris_df.copy() # 元のデータフレームを変更しないためにコピー
iris_df_mutated['petal_area'] = iris_df_mutated['petal_length'] * iris_df_mutated['petal_width']
print("--- 'petal_area' 列を追加 (直接代入) ---")
print(iris_df_mutated.head())

# 方法2: assignメソッド (メソッドチェーンに適している)
iris_df_assigned = iris_df.assign(
    petal_area = iris_df['petal_length'] * iris_df['petal_width'],
    sepal_area = iris_df['sepal_length'] * iris_df['sepal_width'] # 複数の列を一度に作成可能
)
print("\n--- 'petal_area' と 'sepal_area' 列を追加 (assign) ---")
print(iris_df_assigned.head())
```
新しく作成した列をすぐに次の計算で使うこともできます。

---

### 5. 並べ替え：データを順序付ける (`sort_values`メソッド)

特定の列の値に基づいて、データフレームの行を昇順または降順に並べ替える操作です。

例えば、`iris`データセットを `petal_length`（花びらの長さ）が大きい順（降順）に並べ替えてみましょう。

```python
# 花びらの長さで降順にソート
sorted_iris_df = iris_df.sort_values(by='petal_length', ascending=False)
print("--- 花びらの長さで降順にソート (最初の5行) ---")
print(sorted_iris_df.head())

# 複数のキーでソートも可能 (例: speciesで昇順、その中でpetal_lengthで降順)
sorted_multi_key_iris_df = iris_df.sort_values(
    by=['species', 'petal_length'],
    ascending=[True, False] # speciesは昇順, petal_lengthは降順
)
print("\n--- 品種(昇順)、花びらの長さ(降順)でソート (一部抜粋) ---")
print(sorted_multi_key_iris_df.head(10)) # 先頭10件表示
```
`ascending=True` が昇順（デフォルト）、`ascending=False` が降順です。

---

### 6. 結合：複数のデータを繋ぎ合わせる (`merge`メソッド)

2つ以上のデータフレームを、共通のキー（列）をもとにして横に繋ぎ合わせる操作です。

例として、`iris`データセットの品種名 (`species`) に対して、日本語の品種名情報を持つ別のデータフレームを作成し、結合してみましょう。

```python
# 日本語の品種名情報を持つデータフレームを作成
species_jpn_df = pd.DataFrame({
    'species': ['setosa', 'versicolor', 'virginica'],
    'species_jpn': ['ヒオウギアヤメ', 'ブルーフラッグ', 'バージニカ']
})
print("--- 日本語品種名データフレーム ---")
print(species_jpn_df)

# 'species' 列をキーとして iris_df と species_jpn_df を結合
merged_iris_df = pd.merge(iris_df, species_jpn_df, on='species', how='left')
# how='left' は左側のiris_dfを基準に結合し、対応するspecies_jpnがない場合はNaNが入る（今回は全て対応あり）
# 他にも 'right', 'inner'(共通のキーのみ), 'outer'(全てのキー) がある

print("\n--- 日本語品種名と結合したデータ (最初の5行) ---")
print(merged_iris_df.head())
print("\n--- 日本語品種名と結合したデータ (ランダムな5行) ---")
print(merged_iris_df.sample(5)) # ランダムに5行表示
```

結合する際に、キーとなる列名が左右のデータフレームで異なる場合は `left_on` と `right_on` で指定します。

データフレームの結合は、異なる情報源からのデータを統合して分析する際に非常に強力な機能です。

---

### その他の便利な操作

上記以外にも、データラングリングで役立つ操作がいくつかあります。

* **列の選択 (`[]` や `loc`, `iloc`)**: 特定の列だけを選び出す。
    ```python
    selected_columns_df = iris_df[['species', 'petal_length', 'petal_width']]
    print("\n--- 特定の列を選択 ---")
    print(selected_columns_df.head())
    ```

* **列名の変更 (`rename`メソッド)**: 列の名前を変更する。
    ```python
    renamed_df = iris_df.rename(columns={
        'sepal_length': 'gaku_nagasa',
        'sepal_width': 'gaku_haba'
    })
    print("\n--- 列名を変更 ---")
    print(renamed_df.head())
    ```

* **上位/下位N件の取得 (`head`, `tail`, `nlargest`, `nsmallest`)**:
    ```python
    # petal_area列を作成 (既出のassignを使用)
    iris_with_area = iris_df.assign(petal_area = iris_df['petal_length'] * iris_df['petal_width'])

    # petal_areaが大きい上位5件
    top_5_petal_area = iris_with_area.nlargest(5, 'petal_area')
    print("\n--- 花びらの面積が大きい上位5件 ---")
    print(top_5_petal_area)
    ```

---

### まとめ

この章では、`pandas`を使った基本的なデータラングリング操作を学びました。

| 操作             | `pandas`での主な実現方法                     | 説明                                                                 |
| ---------------- | -------------------------------------------- | -------------------------------------------------------------------- |
| 行のフィルタリング   | `[]` (ブールインデックス), `query()`         | 条件に合う行を選択する                                                 |
| 列の集計         | `mean()`, `sum()`, `agg()` など             | 列の値を要約統計量でまとめる                                           |
| グループ化       | `groupby()` と集計関数                        | グループごとに集計する                                                 |
| 列の作成・変換   | 直接代入, `assign()`                        | 既存の列から新しい列を作成したり、値を変換したりする                     |
| 並べ替え         | `sort_values()`                              | 特定の列の値に基づいて行を並び替える                                     |
| 結合             | `merge()`                                    | 共通のキーを使って複数のデータフレームを繋ぎ合わせる                   |
| 列の選択         | `[]`, `loc[]`, `iloc[]`                      | 特定の列を選び出す                                                   |
| 列名の変更       | `rename()`                                   | 列の名前を変更する                                                   |
| 上位/下位N件取得 | `head()`, `tail()`, `nlargest()`, `nsmallest()` | データの一部や極端な値を持つデータを抽出する                             |

これらの操作を組み合わせることで、分析に適した形にデータを整形できます。データ分析プロジェクトでは、このデータラングリングの工程が多くの時間を占めることもありますが、非常に重要なスキルです。色々なデータセットで練習してみてください。


---

## 演習課題：レストランのチップデータの分析

`seaborn`ライブラリには、レストランでの食事におけるチップの支払いに関する情報が含まれた`tips`データセットがあります。このデータセットを使って、以下の課題に取り組んでみましょう。

**`tips`データセットの主な列:**

* `total_bill`: 会計総額 (ドル)
* `tip`: チップ額 (ドル)
* `sex`: 会計者の性別 (Male, Female)
* `smoker`: 喫煙席かどうか (Yes, No)
* `day`: 曜日 (Thur, Fri, Sat, Sun)
* `time`: 時間帯 (Lunch, Dinner)
* `size`: 同席人数

まず、必要なライブラリをインポートし、データセットを読み込みましょう。

```python
import pandas as pd
import seaborn as sns

# tipsデータセットをロード
tips_df = sns.load_dataset('tips')

# データの最初の数行と情報を確認
print("--- tipsデータセットの最初の5行 ---")
print(tips_df.head())
print("\n--- tipsデータセットの基本情報 ---")
tips_df.info()
```

---

### 課題

週末（土曜日 'Sat' と日曜日 'Sun'）のディナータイム ('Dinner') において、以下の分析を行ってください。

1.    各会計における**チップの割合**（チップ額 ÷ 会計総額）を計算し、`tip_rate` という名前の新しい列としてデータフレームに追加してください。
2.    喫煙者 (`smoker` が 'Yes') と非喫煙者 (`smoker` が 'No') のグループ別に、以下の値を集計してください。
    * 平均チップ率 (`tip_rate` の平均値)
    * 平均会計総額 (`total_bill` の平均値)
    * 来店組数 (該当するグループのレコード数)
3.    上記の集計結果を、**平均チップ率が高い順**に並べ替えて表示してください。
4.    最終的に表示するデータフレームの列名は、それぞれ `smoker`, `avg_tip_rate`, `avg_total_bill`, `num_parties` としてください。

**ヒント：**

* **フィルタリング**:
    * まず、`day` 列が 'Sat' または 'Sun' である行を抽出します。`isin()` メソッドが便利です。
    * 次に、`time` 列が 'Dinner' である行を抽出します。
    * これらの条件を `&` (AND) で組み合わせます。
* **新しい列の作成**:
    * `tip` 列と `total_bill` 列を使って `tip_rate` 列を作成します。
* **グループ化と集計**:
    * `smoker` 列でグループ化します (`groupby()` メソッド)。
    * `agg()` メソッドを使って、各グループの `tip_rate` の平均、`total_bill` の平均、そして組数 (例えば `tip` 列や任意の列の `count` または `size` で取得できます) を計算します。
* **列名の変更**:
    * 集計後のデータフレームの列名を `rename()` メソッドで変更します。
* **並べ替え**:
    * `sort_values()` メソッドを使って、`avg_tip_rate` 列で降順 (`ascending=False`) に並べ替えます。

**期待される出力のイメージ (実際の数値は計算結果に基づきます):**

| smoker   |   avg_tip_rate |   avg_total_bill |   num_parties |
|:---------|---------------:|-----------------:|--------------:|
| (どちらか) |       0.xxxxxx |         yy.yyyyy |            zz |
| (もう一方) |       0.aaaaaa |         bb.bbbbb |            cc |

###

```python
# prompt: tips_df のデータから、day が 'Sat' または 'Sun' かつ、time が 'Dinner'のデータに絞り込んだものを作りたいです。

tips_df_weekend_dinner = tips_df[
    ((tips_df['day']=='Sat') | (tips_df['day']=='Sun')) & (tips_df['time']=='Dinner')
].copy()

# チップ率の計算
tips_df_weekend_dinner['tip_rate'] = tips_df_weekend_dinner['tip'] / tips_df_weekend_dinner['total_bill']

# 喫煙者/非喫煙者でグループ化し集計
smoker_summary = tips_df_weekend_dinner.groupby('smoker').agg(
    avg_tip_rate=('tip_rate', 'mean'),
    avg_total_bill=('total_bill', 'mean'),
    num_parties=('smoker', 'count')
).reset_index() # smokerを列に戻す

# 平均チップ率で降順にソート
smoker_summary_sorted = smoker_summary.sort_values(by='avg_tip_rate', ascending=False)

# 表示する列名を指定
smoker_summary_sorted = smoker_summary_sorted[['smoker', 'avg_tip_rate', 'avg_total_bill', 'num_parties']]

print("\n--- 週末ディナータイムの喫煙者/非喫煙者別集計 ---")
smoker_summary_sorted
```
