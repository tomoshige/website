---
marp: true
size: 16:9
paginate: true
theme: gaia
backgroundColor: #fff
math: katex
---
<!-- header: 'T. Nakamura | Juntendo Univ.' -->
<!-- footer: '2025/02/08' -->
<style>
section { 
    font-size: 20px; 
}
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
/* ページ番号 */
section::after {
    content: attr(data-marpit-pagination) ' / ' attr(data-marpit-pagination-total);
    fonr-size: 60%;
}
/* 発表会名 */
header {
    width: 100%;
    position: absolute;
    top: unset;
    bottom: 21px;
    left: 0;
    text-align: center;
    font-size: 60%;
}
/* 日付 */
footer {
    text-align: center;
    font-size: 15px;
}
</style>

<style scoped>section { font-size: 20px; }h1 {margin-bottom: 60px;}h3 {margin-bottom: 60px;}</style>

<!--
_class: lead
_paginate: false
-->

# Python によるデータサイエンス入門

## Google Colabを使った実践的データ分析
### 健康データサイエンス学部 
---

# 講師紹介

- **専門分野**: 統計学習理論、機械学習
- **研究テーマ**: 予測モデリング、データ分析パイプライン
- **使用言語**: Python, R, Julia
- **連絡先**: [example@email.com](mailto:example@email.com)

---

# 本日の内容

1. Python と Google Colab の基本
2. プログラミングの基本概念
3. Python ライブラリの活用
4. データの読み込みと探索
5. データフレームの操作
6. データの可視化
7. 実践演習: NYCフライトデータ分析

---

# Pythonとデータサイエンスの関係

![width:900px](https://miro.medium.com/v2/resize:fit:1400/1*kWKEUwQGG4ZwnSiHZ-qDTw.png)

- プログラミング言語としてのPython
- データサイエンスにおける強み
  - 豊富なライブラリ
  - 読みやすい構文
  - 幅広いコミュニティサポート

---

# Python と Google Colab とは？

![bg right:40% 80%](https://colab.research.google.com/img/colab_favicon_256px.png)

- **Python**: 汎用プログラミング言語
  - データ分析、AI/ML、Web開発などに使用
  - 読みやすく学びやすい構文

- **Google Colab**: ブラウザベースの開発環境
  - Jupyter Notebookベース
  - クラウド上で実行（ローカルインストール不要）
  - GPU/TPUサポート
  - コードと文章を組み合わせた「ノートブック」形式

---

# Google Colabの基本的な使い方

![width:900px](https://raw.githubusercontent.com/InflationX/ViewPump/master/art/google-colab-usage.png)

- コードセル: Pythonコードを書く場所
- テキストセル: マークダウン形式で説明を書く場所
- 実行ボタン: コードを実行
- 出力: 実行結果の表示

---

# プログラミングの基本概念

- **変数**: データを格納する場所
  ```python
  x = 10
  name = "データサイエンス"
  ```

- **データ型**: 整数、浮動小数点、文字列、ブール値など
  ```python
  age = 25               # 整数
  height = 175.5         # 浮動小数点
  is_student = True      # ブール値
  course = "Python入門"  # 文字列
  ```

---

# プログラミングの基本概念（続き）

- **リスト**: 複数の値を格納する配列
  ```python
  numbers = [1, 2, 3, 4, 5]
  names = ["Alice", "Bob", "Charlie"]
  ```

- **関数**: 特定のタスクを実行するコードブロック
  ```python
  def greet(name):
      return f"こんにちは、{name}さん！"
  
  message = greet("太郎")  # "こんにちは、太郎さん！"
  ```

---

# 条件分岐とループ

- **条件分岐**: 条件に基づいて異なる処理を実行
  ```python
  if score >= 80:
      grade = "A"
  elif score >= 70:
      grade = "B"
  else:
      grade = "C"
  ```

- **ループ**: 繰り返し処理
  ```python
  # for ループ
  for i in range(5):
      print(i)  # 0, 1, 2, 3, 4 を順に表示
      
  # while ループ
  count = 0
  while count < 5:
      print(count)
      count += 1
  ```

---

# エラーと例外処理

- **エラー**: コードに問題があるときに発生
  - 構文エラー: コードの書き方が間違っている
  - 実行時エラー: コード実行中に発生

- **例外処理**: エラーを適切に処理する仕組み
  ```python
  try:
      result = 10 / 0  # ゼロ除算エラー
  except ZeroDivisionError:
      print("ゼロで割ることはできません")
  finally:
      print("処理を終了します")
  ```

---

# Python ライブラリとは？

![bg right:40% 80%](https://opensource.com/sites/default/files/lead-images/python_programming_question.png)

- 特定の機能を提供するコードの集まり
- 「車輪の再発明」を避ける
- インポートして使用

```python
# ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

# データサイエンスで重要なライブラリ

1. **NumPy**: 数値計算の基盤
   - 多次元配列、行列演算

2. **pandas**: データ分析の中核
   - DataFrameオブジェクト、データ操作

3. **Matplotlib/Seaborn**: データ可視化
   - グラフ、チャート、ヒートマップなど

4. **scikit-learn**: 機械学習
   - アルゴリズム、前処理、評価指標

---

# ライブラリのインストールとインポート

- **インストール**: `pip`コマンドを使用
  ```python
  !pip install numpy pandas matplotlib seaborn
  ```

- **インポート**: `import`文を使用
  ```python
  # 標準的なインポート方法
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # グラフをノートブック内に表示する設定
  %matplotlib inline
  ```

---

# データ読み込みと探索

- **CSVファイルの読み込み**:
  ```python
  # URLからCSVファイルをダウンロード
  !wget https://raw.githubusercontent.com/hadley/nycflights13/master/data-raw/flights.csv
  
  # CSVファイルを読み込む
  flights = pd.read_csv('flights.csv')
  ```

- **データの確認**:
  ```python
  # 最初の5行を表示
  flights.head()
  
  # データフレームの情報
  flights.info()
  ```

---

# データフレームの基本

![width:400px](https://pandas.pydata.org/docs/_images/01_table_dataframe.svg)

- 行と列からなる2次元のテーブル構造
- 各列は異なるデータ型を持つことが可能
- SQLのテーブルやExcelのシートに似ている
- インデックス（行ラベル）と列名を持つ

---

# データフレームの基本操作

- **基本情報の取得**:
  ```python
  flights.shape        # (行数, 列数)
  flights.columns      # 列名の一覧
  flights.dtypes       # 各列のデータ型
  flights.describe()   # 数値列の要約統計量
  ```

- **データの選択**:
  ```python
  # 列の選択
  flights['month']     # 単一列の選択
  flights[['month', 'day', 'dep_time']]  # 複数列の選択
  
  # 行の選択
  flights.iloc[0:5]    # 位置インデックスによる選択
  flights.loc[flights['month'] == 1]  # 条件による選択
  ```

---

# データの絞り込みとフィルタリング

- **条件に基づくフィルタリング**:
  ```python
  # 1月のフライト
  january_flights = flights[flights['month'] == 1]
  
  # 遅延したフライト（15分以上）
  delayed_flights = flights[flights['dep_delay'] > 15]
  
  # 複合条件
  jfk_to_lax = flights[(flights['origin'] == 'JFK') & 
                       (flights['dest'] == 'LAX')]
  ```

---

# グループ化と集計

- **`groupby`を使用したデータ集計**:
  ```python
  # 月ごとのフライト数
  monthly_flights = flights.groupby('month').size()
  
  # 航空会社（キャリア）ごとの平均遅延時間
  carrier_delays = flights.groupby('carrier')['dep_delay'].mean()
  
  # 複数の集計関数を適用
  origin_stats = flights.groupby('origin').agg({
      'flight': 'count',
      'dep_delay': ['mean', 'median'],
      'arr_delay': ['mean', 'median']
  })
  ```

---

# データ可視化の基本

![bg right:40% 80%](https://matplotlib.org/stable/_images/sphx_glr_logos2_003.png)

- データを視覚的に表現して理解を深める
- パターン、傾向、外れ値を見つける
- Pythonの主要可視化ライブラリ:
  - **Matplotlib**: 基本的なグラフ作成
  - **Seaborn**: 統計データの可視化
  - **Plotly**: インタラクティブな可視化

---

# Matplotlibの基本

```python
# 基本的な折れ線グラフ
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], [10, 20, 25, 30], marker='o')
plt.title('シンプルな折れ線グラフ')
plt.xlabel('X軸')
plt.ylabel('Y軸')
plt.grid(True)
plt.show()
```

![width:300px](https://matplotlib.org/stable/_images/sphx_glr_plot_001.png)

---

# データ可視化の種類

1. **分布を見る**:
   - ヒストグラム、密度プロット、箱ひげ図

2. **関係を見る**:
   - 散布図、ヒートマップ、ペアプロット

3. **比較する**:
   - 棒グラフ、折れ線グラフ、レーダーチャート

4. **構成を見る**:
   - 円グラフ、積み上げ棒グラフ、ツリーマップ

---

# NYCフライトデータ分析

![bg right:40% 80%](https://www.ny.com/transportation/airports/images/jfk-airport.jpg)

- 2013年ニューヨーク市の空港から出発した国内線フライトデータ
- 主要データフレーム:
  - `flights`: 個々のフライト情報
  - `airlines`: 航空会社の情報
  - `airports`: 空港情報
- 分析の目的:
  - フライト遅延のパターンを理解する
  - 遅延に影響する要因を特定する

---

# NYCフライトデータの読み込み

```python
# データのダウンロードと読み込み
!wget https://raw.githubusercontent.com/hadley/nycflights13/master/data-raw/flights.csv
!wget https://raw.githubusercontent.com/hadley/nycflights13/master/data-raw/airlines.csv
!wget https://raw.githubusercontent.com/hadley/nycflights13/master/data-raw/airports.csv

# データフレームとして読み込む
flights = pd.read_csv('flights.csv')
airlines = pd.read_csv('airlines.csv')
airports = pd.read_csv('airports.csv')

# データの最初の数行を確認
flights.head()
```

---

# データの基本的な探索

```python
# flights データフレームの基本情報
print(f"行数: {flights.shape[0]}, 列数: {flights.shape[1]}")
flights.info()

# 基本統計量
flights.describe()

# 欠損値の確認
missing_values = flights.isnull().sum()
print(missing_values[missing_values > 0])
```

---

# 月ごとのフライト数と遅延パターン

```python
# 月ごとのフライト数
monthly_flights = flights.groupby('month').size()

# 月ごとの平均遅延時間
monthly_delays = flights.groupby('month')['dep_delay'].mean()

# 可視化
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
monthly_flights.plot(kind='bar')
plt.title('月ごとのフライト数')
plt.xlabel('月')
plt.ylabel('フライト数')

plt.subplot(1, 2, 2)
monthly_delays.plot(kind='bar', color='orange')
plt.title('月ごとの平均出発遅延時間')
plt.xlabel('月')
plt.ylabel('平均遅延（分）')
plt.tight_layout()
plt.show()
```

---

# 航空会社ごとの遅延パフォーマンス

```python
# 航空会社ごとの平均遅延時間を計算
carrier_delays = flights.groupby('carrier')['dep_delay'].mean().sort_values()

# 航空会社名を追加
carrier_full_names = pd.merge(
    pd.DataFrame(carrier_delays).reset_index(),
    airlines,
    on='carrier'
)

# 可視化
plt.figure(figsize=(12, 6))
sns.barplot(x='carrier', y='dep_delay', data=carrier_full_names)
plt.title('航空会社ごとの平均出発遅延時間')
plt.xlabel('航空会社コード')
plt.ylabel('平均遅延時間（分）')
plt.xticks(rotation=45)
plt.show()
```

---

# 空港別・時間帯別の遅延パターン

```python
# 出発時刻から時間を抽出
flights['hour'] = flights['dep_time'] // 100

# 空港と時間帯ごとの平均遅延時間
hourly_delays = flights.groupby(['origin', 'hour'])['dep_delay'].mean().unstack()

# ヒートマップで可視化
plt.figure(figsize=(14, 8))
sns.heatmap(hourly_delays, cmap='YlOrRd', annot=True, fmt=".1f")
plt.title('空港・時間帯別の平均出発遅延時間（分）')
plt.xlabel('時間帯（24時間制）')
plt.ylabel('出発空港')
plt.show()
```

---

# 実践演習: 遅延分析プロジェクト

![bg right:40% 80%](https://www.incimages.com/uploaded_files/image/1920x1080/getty_883231284_200013331818843182490_335833.jpg)

1. **データを準備する**
   - 必要なデータの読み込み
   - 欠損値・異常値の処理
   - 特徴量エンジニアリング

2. **探索的データ分析**
   - 各変数の分布と関係の調査
   - 時間的・空間的パターンの特定

3. **仮説検証**
   - 遅延に影響する要因の統計的分析
   - 可視化による検証

---

# 課題: フライト遅延の詳細分析

以下の分析タスクに取り組んでください:

1. どの月が最も遅延が多いですか？そして最も少ないですか？

2. 曜日（day_of_week列を作成）と遅延時間の関係はありますか？

3. 航空会社ごとの遅延パフォーマンスを分析し、最も信頼性の高い航空会社とそうでない航空会社を特定してください。

4. 距離と遅延時間の関係を分析してください。長距離フライトは短距離フライトよりも遅延が多いですか？

---

# ボーナス課題: 遅延予測モデル

scikit-learnを使って簡単な機械学習モデルを構築:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 遅延を2値分類問題として定義（15分以上遅延=1, それ以外=0）
flights['delayed'] = (flights['dep_delay'] > 15).astype(int)

# 特徴量とターゲットを準備
X = flights[['month', 'day', 'hour', 'origin', 'distance']]
y = flights['delayed']

# カテゴリカル変数をエンコーディング
X = pd.get_dummies(X, columns=['origin'])

# モデルの構築と評価
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

# まとめ

- **Python**はデータサイエンスに最適なプログラミング言語
- **Google Colab**は簡単に始められる開発環境
- **pandas**はデータ分析の中核ライブラリ
- データ探索と可視化はインサイト発見の鍵
- 実践的なプロジェクトで学習を深めることが重要

---

# 次のステップ

- **より高度なデータ操作**: pandas, NumPyの深掘り
- **データ可視化の応用**: インタラクティブ可視化（Plotly）
- **統計分析**: 仮説検定、相関分析
- **機械学習**: scikit-learnによるモデル構築
- **ディープラーニング**: TensorFlow, PyTorch
- **実際のプロジェクト**: Kaggleコンペティションへの参加

---

# 質問・ディスカッション

![bg right:40% 80%](https://cdn.pixabay.com/photo/2018/05/08/08/44/artificial-intelligence-3382507_1280.jpg)

- ここまでの内容で質問はありますか？
- データサイエンスプロジェクトのアイデア
- Pythonの学習リソース

---

# 参考資料・リソース

- **書籍**:
  - "Python for Data Analysis" by Wes McKinney
  - "Python Data Science Handbook" by Jake VanderPlas

- **オンラインコース**:
  - Coursera: "Applied Data Science with Python"
  - DataCamp: "Data Scientist with Python"

- **ウェブサイト**:
  - [Python.org](https://www.python.org)
  - [pandas documentation](https://pandas.pydata.org/docs/)
  - [Kaggle Learn](https://www.kaggle.com/learn)

---

# ご清聴ありがとうございました

![bg right:40% 90%](https://cdn.pixabay.com/photo/2016/11/30/20/58/programming-1873854_1280.png)

**連絡先**:
- Email: [example@email.com](mailto:example@email.com)
- GitHub: [github.com/username](https://github.com/username)
- LinkedIn: [linkedin.com/in/username](https://linkedin.com/in/username)

**講義資料は以下からダウンロード可能です**:
[github.com/username/python-datascience-intro](https://github.com/username/python-datascience-intro)