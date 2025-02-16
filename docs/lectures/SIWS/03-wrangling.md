# Python による データ整形（Data Wrangling）

## はじめに

本章では、データサイエンスにおいて最も重要な前処理・変換作業である **データ整形（データ・ワリングリング）** の基本的な操作方法を学びます。  
具体的には、以下の内容を取り扱います。

- **データの読み込みと確認**  
  実データ（Students Performance in Exams）を読み込み、データの内容や構造を把握する方法。

- **行のフィルタリングと列の選択**  
  必要なデータのみを抽出するための条件指定や、解析に必要な列だけを取り出す方法。

- **新しい変数の作成**  
  既存の変数から新たな指標（例：総得点、平均点）を計算し、データに追加する方法。

- **データの並べ替え**  
  特定の基準（例：平均点の降順）に基づいてデータをソートする方法。

- **グループ化と集約**  
  カテゴリごとにデータをまとめ、平均値などの統計量を算出する方法。

- **データの結合と再構築（ピボット操作）**  
  複数のデータセットを結合したり、データの形式を変換（ワイド形式⇄ロング形式）する方法。

- **データの可視化**  
  データの傾向や分布を視覚的に把握するためのグラフ作成手法。

これらの操作は、データのクレンジング、特徴量エンジニアリング、分析の前処理など、実際のデータサイエンスプロジェクトにおいて不可欠なステップです。  
正確なデータ整形は、後続のモデル構築や分析の精度向上に直結するため、本章で学ぶ内容は非常に重要です。

---

## 1. 必要なライブラリのインポート

まず、数値計算、データ操作、グラフ描画のために必要なライブラリをインポートします。

### 説明
- **numpy**: 数値計算ライブラリ（今回は補助的に利用）。
- **pandas**: データ操作や解析のためのライブラリで、DataFrame を用いて表形式データを扱います。
- **matplotlib** と **seaborn**: データの可視化ライブラリです。`seaborn` は `matplotlib` をベースにしており、見やすいグラフを簡単に作成できます。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# グラフのスタイル設定（見た目を整えるためのオプション）
sns.set(style="whitegrid")
```

---

## 2. 実データの読み込みと確認

今回は、Kaggle などで公開されている「Students Performance in Exams」データセットを使用します。  
データには、以下のような列が含まれています：  
- `gender`  
- `race/ethnicity`  
- `parental level of education`  
- `lunch`  
- `test preparation course`  
- `math score`  
- `reading score`  
- `writing score`

データは、GitHub 上に公開されている CSV ファイルから読み込みます。

```python
# データセットの URL（※実際の利用時はデータの出所・ライセンスに注意してください）
data_url = "https://raw.githubusercontent.com/selva86/datasets/master/StudentsPerformance.csv"

# CSV ファイルからデータを読み込む
df = pd.read_csv(data_url)

# データの先頭部分を確認
print("=== オリジナルデータ ===")
print(df.head())
```

---

## 3. データのフィルタリング

特定の条件に合致する行のみを抽出する操作です。  
ここでは例として、「数学のスコアが 70 点以上」の生徒のみを抽出します。  
pandas では、ブールインデックスを用いて条件を指定します。

```python
# 数学のスコアが70以上の生徒のみ抽出
df_filtered = df[df['math score'] >= 70]
print("\n=== 数学のスコアが70以上のデータ ===")
print(df_filtered.head())
```

---

## 4. 必要な列の選択

解析に必要な列だけを抽出します。  
ここでは、`gender`、`math score`、`reading score`、`writing score` の4列のみを選択します。

```python
# 必要な列だけを抽出
df_selected = df[['gender', 'math score', 'reading score', 'writing score']]
print("\n=== 選択した列 (gender, math score, reading score, writing score) ===")
print(df_selected.head())
```

---

## 5. 新しい変数の作成（mutate に相当）

R の `mutate()` と同様に、新たな列を追加して計算結果を保存します。  
ここでは、各生徒の総得点 (`total_score`) と平均点 (`mean_score`) を計算して追加します。

```python
# 新しい列の作成：総得点と平均点を追加
df = df.assign(
    total_score = df['math score'] + df['reading score'] + df['writing score'],
    mean_score = df[['math score', 'reading score', 'writing score']].mean(axis=1)
)
print("\n=== 新しい変数 (total_score, mean_score) を追加したデータ ===")
print(df.head())
```

### 詳細説明
- `df.assign(...)` は、元の DataFrame に対して新しい列を追加し、結果の DataFrame を返します。  
- `mean(axis=1)` は、各行（axis=1）ごとに平均を計算する指定です。

---

## 6. 行の並べ替え（arrange に相当）

データを特定の列に基づいて並べ替えます。  
ここでは、`mean_score`（平均点）の降順に並べ替えを行います。

```python
# 平均点の降順で並べ替え
df_sorted = df.sort_values(by='mean_score', ascending=False)
print("\n=== 平均点で降順ソートしたデータ ===")
print(df_sorted.head())
```

---

## 7. グループ化と要約（group_by & summarise に相当）

### (a) 性別ごとの数学の平均スコア

性別ごとにグループ化し、各グループ内の「math score」の平均を算出します。

```python
# 性別ごとの数学の平均スコア
mean_math_by_gender = df.groupby('gender')['math score'].mean().reset_index()
print("\n=== 性別ごとの数学の平均スコア ===")
print(mean_math_by_gender)
```

### (b) テスト準備コース別に数学、読解、作文の平均スコアを集計

生徒がテスト準備コースを受講したかどうかで、各科目の平均スコアを集計します。

```python
# テスト準備コースごとに数学、読解、作文の平均スコアを集計
mean_scores_by_prep = df.groupby('test preparation course').agg({
    'math score': 'mean',
    'reading score': 'mean',
    'writing score': 'mean'
}).reset_index()
print("\n=== テスト準備コースごとの各科目の平均スコア ===")
print(mean_scores_by_prep)
```

### 詳細説明
- `groupby()` により、指定した列（ここでは `gender` や `test preparation course`）でデータをグループ化します。  
- `agg()` を使用して、各グループに対して複数の集約関数（例：`mean`）を適用できます。  
- `reset_index()` により、グループ化後のインデックスを通常の列に戻します。

---

## 8. データの結合（Join）

複数の DataFrame を共通のキーで結合する操作です。  
ここでは、元のデータに「親の学歴」を数値化した情報を付加する例を示します。

### (a) 追加情報の DataFrame 作成

親の学歴は文字列ですが、ここでは便宜上、各レベルに数値の「ランク」を割り当てた DataFrame を作成します。  
例として、以下のランク付けを行います：
- `some high school`: 1  
- `high school`: 2  
- `some college`: 3  
- `associate's degree`: 4  
- `bachelor's degree`: 5  
- `master's degree`: 6  

※実際のデータでは、値の種類や表記が異なる場合があるため、適宜調整してください。

```python
# 親の学歴とそのランクのマッピング情報
parent_ed_levels = pd.DataFrame({
    'parental level of education': [
        'some high school', 'high school', 'some college', 
        "associate's degree", "bachelor's degree", "master's degree"
    ],
    'edu_rank': [1, 2, 3, 4, 5, 6]
})

# 作成したマッピング情報を、元のデータと「親の学歴」をキーに左結合（left join）
df_joined = pd.merge(df, parent_ed_levels, on='parental level of education', how='left')
print("\n=== 親の学歴のランク情報を結合したデータ ===")
print(df_joined[['parental level of education', 'edu_rank']].drop_duplicates())
```

### 詳細説明
- `pd.merge()` は、2 つの DataFrame を共通のキー（ここでは `parental level of education`）で結合するための関数です。  
- `how='left'` とすることで、左側（元のデータ）の全行を保持し、右側の DataFrame から対応する情報を追加します。

---

## 9. データの再構築（ピボット操作：wide ⇄ long）

### (a) ワイド形式からロング形式への変換

3 科目（数学、読解、作文）のスコアが個別の列として存在するデータを、  
`pd.melt()` を用いて「subject」と「score」の 2 列からなるロング形式に変換します。

```python
# 数学、読解、作文のスコアをロング形式に変換
df_long = pd.melt(df,
                  id_vars=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'],
                  value_vars=['math score', 'reading score', 'writing score'],
                  var_name='subject',
                  value_name='score')
print("\n=== ロング形式のデータ ===")
print(df_long.head())
```

### (b) ロング形式からワイド形式への再変換

先ほどのロング形式のデータを、`pivot_table()` を用いて元のワイド形式に戻します。

```python
# ロング形式からワイド形式に変換
df_wide = df_long.pivot_table(index=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'],
                              columns='subject',
                              values='score').reset_index()
print("\n=== ワイド形式に戻したデータ ===")
print(df_wide.head())
```

### 詳細説明
- `pd.melt()` は、複数の列を 1 つの「値」列にまとめ、対応する「変数名」を示す列を作成します。  
- `pivot_table()` を用いると、指定したインデックスと列名を元にデータを再構築し、ワイド形式に戻すことができます。

---

## 10. データの可視化

ここでは、整形したデータを基に可視化を行い、データの傾向や分布を把握します。

### (a) 散布図：数学スコアと読解スコアの関係

性別ごとに色分けして、数学スコアと読解スコアの関係を散布図で表示します。

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='math score', y='reading score', hue='gender', s=100)
plt.title('数学スコアと読解スコアの関係')
plt.xlabel('数学スコア')
plt.ylabel('読解スコア')
plt.show()
```

### (b) ボックスプロット：性別ごとの数学スコアの分布

各性別における数学スコアの分布（中央値、四分位範囲、外れ値など）をボックスプロットで可視化します。

```python
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='gender', y='math score')
plt.title('性別ごとの数学スコアの分布')
plt.xlabel('性別')
plt.ylabel('数学スコア')
plt.show()
```

### 詳細説明
- `sns.scatterplot()` は、散布図を作成するための関数です。`hue` パラメータにより性別ごとに色分けを行っています。  
- `sns.boxplot()` は、カテゴリ別のデータ分布（中央値、四分位範囲、外れ値など）を視覚的に把握するために使用します。

---

## まとめ

本ドキュメントでは、実際の「Students Performance in Exams」データセットを用い、Python によるデータ整形と操作の基本的な手法を以下の流れで解説しました。

- **データの読み込み・確認**  
  → CSV ファイルからデータを読み込み、`head()` で先頭部分を表示。

- **行のフィルタリング**  
  → 例として、数学スコアが 70 点以上の生徒を抽出。

- **列の選択**  
  → 解析に必要な列（例：`gender`, `math score`, `reading score`, `writing score`）を選択。

- **新しい変数の作成**  
  → `assign()` を用い、総得点 (`total_score`) や平均点 (`mean_score`) を計算して追加。

- **並べ替え**  
  → `sort_values()` により、平均点の降順で並べ替え。

- **グループ化と要約**  
  → `groupby()` と `agg()` を使い、性別やテスト準備コースごとに各科目の平均スコアを算出。

- **データの結合**  
  → 別途作成した「親の学歴ランク」情報の DataFrame と結合し、付加情報を追加。

- **データの再構築（ピボット操作）**  
  → `melt()` でワイド形式のスコア列をロング形式に変換し、`pivot_table()` でワイド形式に戻す操作を実施。

- **データの可視化**  
  → seaborn と matplotlib を用いて、散布図やボックスプロットでデータの傾向や分布を視覚化。

これらの基本操作は、データクレンジング、特徴量エンジニアリング、そして正確な分析結果を得るための前処理として、データサイエンスの現場で必須のスキルです。  
本章で学んだ手法を身につけることで、実際のプロジェクトにおけるデータ解析やモデリングの精度向上につながります。


---

## 演習問題（データ整形）

このドキュメントでは、これまで学んだ内容を復習するための演習問題です。ただし、これらのコードを覚える必要はありません。
このコードは「こういう意味か」と理解できれば問題はないです。 まずは、実際にコードを動かして、出力結果やグラフを確認してください。
その後で、それぞれのプログラムがどのような意味なのかを学び直しましょう。

---

### 演習 1: データの読み込みと基本確認

#### 問題
1. データセットの URL  
   `https://raw.githubusercontent.com/selva86/datasets/master/StudentsPerformance.csv`  
   からデータを読み込み、DataFrame を作成してください。  
2. 読み込んだデータの先頭 10 行を表示してください。

#### 解答例
```python
import pandas as pd

# データセットの URL
data_url = "https://raw.githubusercontent.com/selva86/datasets/master/StudentsPerformance.csv"

# CSV ファイルからデータを読み込む
df = pd.read_csv(data_url)

# データの先頭 10 行を表示
print("データの先頭10行:")
print(df.head(10))
```

---

### 演習 2: データのフィルタリングと列選択

#### 問題
1. 「math score」が 80 点以上の生徒のみを抽出してください。  
2. 抽出したデータから、`gender`、`math score`、`reading score`、`writing score` の 4 列だけを選択してください。

#### 解答例
```python
# math scoreが80点以上の生徒を抽出
df_filtered = df[df['math score'] >= 80]
print("math scoreが80以上のデータ:")
print(df_filtered.head())

# 必要な列のみを選択
df_selected = df_filtered[['gender', 'math score', 'reading score', 'writing score']]
print("選択した列:")
print(df_selected.head())
```

---

### 演習 3: 新しい変数の作成

#### 問題
1. 「reading-writing average」という列を追加し、`reading score` と `writing score` の平均を計算してください。  
2. 各生徒の総得点 (`total_score`: `math score` + `reading score` + `writing score`) を計算し、列として追加してください。

#### 解答例
```python
# 新しい列の作成：reading-writing average と total_score
df['reading-writing average'] = df[['reading score', 'writing score']].mean(axis=1)
df['total_score'] = df['math score'] + df['reading score'] + df['writing score']

print("新しい列を追加したデータ:")
print(df[['reading-writing average', 'total_score']].head())
```

---

### 演習 4: 並べ替え

#### 問題
総得点 (`total_score`) に基づいてデータを降順に並べ替え、上位 5 件のデータを表示してください。

#### 解答例
```python
# total_score に基づいて降順に並べ替え
df_sorted = df.sort_values(by='total_score', ascending=False)
print("総得点で降順に並べ替えた上位5件:")
print(df_sorted.head(5))
```

---

### 演習 5: グループ化と要約

#### 問題
1. 性別ごとにグループ化し、`math score` の平均値を計算して表示してください。  
2. `test preparation course` 別に、`math score`、`reading score`、`writing score` の各平均値を集計してください。

#### 解答例
```python
# 性別ごとの math score の平均を計算
mean_math_by_gender = df.groupby('gender')['math score'].mean().reset_index()
print("性別ごとの math score の平均:")
print(mean_math_by_gender)

# test preparation course 別に各科目の平均値を集計
mean_scores_by_prep = df.groupby('test preparation course').agg({
    'math score': 'mean',
    'reading score': 'mean',
    'writing score': 'mean'
}).reset_index()
print("test preparation course 別の各科目の平均:")
print(mean_scores_by_prep)
```

---

### 演習 6: データの結合（Join）

#### 問題
1. 「parental level of education」に対して、以下のルールで数値のランクを付与する DataFrame を作成してください。  
   - `some high school`: 1  
   - `high school`: 2  
   - `some college`: 3  
   - `associate's degree`: 4  
   - `bachelor's degree`: 5  
   - `master's degree`: 6  
2. 作成した DataFrame を、元のデータと `parental level of education` をキーに左結合し、`parental level of education` と `edu_rank` のユニークな組み合わせを表示してください。

#### 解答例
```python
# 親の学歴ランクの DataFrame を作成
parent_ed_levels = pd.DataFrame({
    'parental level of education': [
        'some high school',
        'high school',
        'some college',
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ],
    'edu_rank': [1, 2, 3, 4, 5, 6]
})

# 元のデータと左結合
df_joined = pd.merge(df, parent_ed_levels, on='parental level of education', how='left')
print("親の学歴とedu_rankのユニークな組み合わせ:")
print(df_joined[['parental level of education', 'edu_rank']].drop_duplicates())
```

---

### 演習 7: ピボット操作（ワイド⇄ロング変換）

#### 問題
1. `pd.melt()` を用いて、`math score`、`reading score`、`writing score` の各列を「subject」と「score」の 2 列に変換してください。  
   ※ `id_vars` として `gender`、`race/ethnicity`、`parental level of education`、`lunch`、`test preparation course` を指定してください。  
2. 先ほどのロング形式データを `pivot_table()` を使って、元のワイド形式（各科目が別の列）に戻してください。

#### 解答例
<details>
```python
# ワイド形式からロング形式に変換
df_long = pd.melt(df,
                  id_vars=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'],
                  value_vars=['math score', 'reading score', 'writing score'],
                  var_name='subject',
                  value_name='score')
print("ロング形式のデータ:")
print(df_long.head())

# ロング形式からワイド形式に再変換
df_wide = df_long.pivot_table(index=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'],
                              columns='subject',
                              values='score').reset_index()
print("ワイド形式に戻したデータ:")
print(df_wide.head())
```
</details>

---

### 演習 8: データの可視化

#### 問題
1. 性別ごとに色分けした散布図を作成し、「math score」と「reading score」の関係を可視化してください。  
2. `lunch` 別に「writing score」の分布を示すボックスプロットを作成してください。

#### 解答例
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 散布図の作成: math score vs reading score (性別で色分け)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='math score', y='reading score', hue='gender', s=100)
plt.title('Math Score vs Reading Score by Gender')
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.show()

# ボックスプロットの作成: lunch 別の writing score 分布
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='lunch', y='writing score')
plt.title('Writing Score Distribution by Lunch')
plt.xlabel('Lunch')
plt.ylabel('Writing Score')
plt.show()
```


---

### 演習 9: 応用問題

#### 問題 9-1: 複合的なグループ化と可視化
ロング形式に変換したデータを用いて、各科目ごとに「parental level of education」別の平均スコアを算出し、棒グラフで可視化してください。

#### 解答例 9-1
```python
# ロング形式 (df_long) を用いて、parental level of education と subject 別の平均スコアを計算
grouped = df_long.groupby(['parental level of education', 'subject'])['score'].mean().reset_index()
print("parental level of education と subject 別の平均スコア:")
print(grouped)

# 棒グラフの作成
plt.figure(figsize=(10, 6))
sns.barplot(data=grouped, x='parental level of education', y='score', hue='subject')
plt.title('Average Score by Parental Level of Education and Subject')
plt.xlabel('Parental Level of Education')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.show()
```

#### 問題 9-2: 仮説検証のための前処理
ここでは、**test preparation course** が **total_score** に与える影響を検証する例を示します。  
- `test preparation course` 別に **total_score** の分布をボックスプロットで可視化し、  
- 各グループの平均 **total_score** を算出してください。

#### 解答例 9-2
```python
# ボックスプロットによる可視化: test preparation course 別の total_score 分布
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='test preparation course', y='total_score')
plt.title('Total Score by Test Preparation Course')
plt.xlabel('Test Preparation Course')
plt.ylabel('Total Score')
plt.show()

# test preparation course 別の平均 total_score を計算
mean_total_by_prep = df.groupby('test preparation course')['total_score'].mean().reset_index()
print("Test Preparation Course 別の平均 Total Score:")
print(mean_total_by_prep)
```



## What's to come?

So far in this book, we've explored, visualized, and wrangled data saved in data frames. These data frames were saved in a spreadsheet-like format: in a rectangular shape with a certain number of rows corresponding to observations and a certain number of columns corresponding to variables describing these observations. 

We'll see in the upcoming Chapter \@ref(tidy) that there are actually two ways to represent data in spreadsheet-type rectangular format: (1) "wide" format and (2) "tall/narrow" format. The tall/narrow format is also known as *"tidy"* format in R user circles. While the distinction between "tidy" and non-"tidy" formatted data is subtle, it has immense implications for our data science work. This is because almost all the packages used in this book, including the `ggplot2` package for data visualization and the `dplyr` package for data wrangling, all assume that all data frames are in "tidy" format. 

Furthermore, up until now we've only explored, visualized, and wrangled data saved within R packages. But what if you want to analyze data that you have saved in a Microsoft Excel, a Google Sheets, or a "Comma-Separated Values" (CSV) file? In Section \@ref(csv), we'll show you how to import this data into R using the `readr` package. 
