# 第1章 Pythonでのデータ活用入門

## 第1回の目標
- Python を Google Colab 上で動かす
- データを読み込んで、確認する
- 簡単なコードを実行する
- コード: <https://colab.research.google.com/drive/1lP0DkYCDpp1XrH2s5fHsiyQGbJ4xyw9I?usp=sharing>

## 1.1 はじめに

皆さん、こんにちは！この章では、Pythonを使ってデータ分析の世界への第一歩を踏み出します。特にプログラミングが初めての方でも理解できるよう、基本的な概念から丁寧に解説していきます。

データ分析を始める前に、いくつか知っておくべき重要な考え方があります。それは、私たちが使う「道具」であるPythonという言語と、その「作業場」となるGoogle Colaboratory（Colab）のような環境についてです。また、どのようにPythonコードを書き、データを扱うための便利なツール（ライブラリ）を使うのかも学びます。

本章を終える頃には、皆さんは簡単なPythonコードを書き、実際のデータセット（今回はニューヨーク市のフライトデータ）を読み込んで、その中身を覗き見ることができるようになります。さあ、一緒にデータの世界を探検しましょう！

## 1.2 Pythonと開発環境 (Google Colab)

### 1.2.1 Python言語とIDE（Colab）の関係性

まず、「Python」と「Google Colab」は何が違うのでしょうか？

*   **Python**: プログラミング言語そのものです。計算したり、データを処理したりする命令のルールブックのようなもの。車の「エンジン」に例えられます。
*   **Google Colab (またはIDE)**: Pythonコードを書いたり、実行したり、結果を見たりするための便利なツールが揃った「作業環境」です。Webブラウザ上で使え、環境設定の手間が少ないのが特徴です。車の「ダッシュボード」に例えられます。スピードメーターやナビがあると運転しやすいように、Colabを使うとPythonでの作業がずっと楽になります。

他にもVS CodeやJupyter Notebookなど様々な開発環境がありますが、この講義では手軽に始められるColabを主に使用します。

### 1.2.2 環境準備：Google Colabとローカル環境

*   **Google Colab**:
    *   WebブラウザとGoogleアカウントがあれば、<https://colab.research.google.com/> からすぐに利用開始できます。
    *   ソフトウェアのインストールは不要です。
    *   多くのデータ分析用ライブラリが最初から使える状態になっています。
*   **ローカル環境 (自分のPC)**:
    *   Python本体をインストールする必要があります (<https://www.python.org/>)。
    *   必要に応じてIDE（VS Codeなど）もインストールします。
    *   使うライブラリは自分で `pip` コマンドを使ってインストールします。
    *   本書では主にColabを使う前提で進めますが、ローカル環境を構築することも可能です。

### 1.2.3 Google ColabでのPythonの使用方法

Colabを開くと、ノートブック形式の画面が表示されます。

*   **セル**: ノートブックは「セル」の集まりです。
    *   **テキストセル**: この文章のように、説明やメモを書くためのセルです。
    *   **コードセル**: Pythonコードを書き込み、実行するためのセルです。
*   **実行**: コードセルを選択し、左側にある「再生ボタン」▶️ をクリックするか、`Shift + Enter` キーを押すと、そのセルの中のコードが実行されます。
*   **実行結果**: コードの実行結果は、そのコードセルのすぐ下に表示されます。

## 1.3 Pythonコーディングの基礎

### 1.3.1 基本的なプログラミング概念と用語

プログラミングには特有の言葉やルールがあります。いくつか見ていきましょう。

*   **変数**: データ（数値や文字など）を入れておくための「箱」のようなものです。箱に名前（変数名）を付けて、後で中身を使えるようにします。
    *   例: `my_variable = 10` （`my_variable` という名前の箱に `10` を入れる）
    *   値を入れることを**代入**といい、 `=` 記号を使います。
*   **データ型**: 変数に入れるデータの種類です。Pythonは自動で判断してくれますが、主なものを覚えておきましょう。
    *   `int`: 整数 (例: `10`, `-5`, `0`)
    *   `float`: 浮動小数点数（小数を含む数） (例: `3.14`, `-0.5`)
    *   `bool`: ブール値（真偽値） (`True` または `False`)
    *   `str`: 文字列 (テキスト) (例: `"Hello"`, `'Python'`, `"123"`) ※文字は `"` か `'` で囲みます。
*   **データ構造**: 複数の値をまとめて扱うための仕組みです。
    *   **リスト `list`**: `[]` で囲み、カンマ `,` で区切って値を並べたもの。様々なデータ型を混在できます。(例: `[1, 2, 3]`, `['apple', 'banana']`)
    *   **NumPy配列 `ndarray`**: 数値計算に特化した配列。同じデータ型の要素を効率的に扱えます。（後の章で詳しく学びます）
    *   **Pandas DataFrame**: 2次元の表形式データ（スプレッドシートのような形式）を扱うための構造。行と列を持ちます。（後ほど詳しく学びます）
*   **条件分岐**: 特定の条件が正しいかどうか (`True`/`False`) で処理を変えることです。
    *   比較演算子:
        *   `==`: 等しい (例: `5 == 5` は `True`) ※ `=` 1つは代入なので注意！
        *   `!=`: 等しくない (例: `5 != 3` は `True`)
        *   `<`, `>`, `<=`, `>=`: 大小比較
    *   論理演算子:
        *   `and`: 両方の条件が `True` なら `True`
        *   `or`: どちらか一方の条件が `True` なら `True`
        *   `not`: `True` と `False` を反転
*   **関数**: 特定の処理をまとめたものです。名前を呼び出すことで、その処理を実行できます。
    *   関数には **引数 (ひきすう)** という入力値を渡すことができます。
    *   例: `print("Hello")` は `"Hello"` という引数を `print` 関数に渡し、画面に出力します。
    *   引数には **デフォルト値** が設定されている場合があり、省略するとその値が使われます。

### 1.3.2 エラー、警告、メッセージの理解

コードを実行すると、赤文字などでメッセージが出ることがあります。慌てずに内容を確認しましょう。

*   **エラー (Exceptions)**:
    *   コードに文法的な間違いや実行不可能な指示がある場合に発生します。
    *   `NameError`, `TypeError`, `ValueError` など、エラーの種類と原因を示すメッセージが表示されます。
    *   エラーが発生すると、通常コードの実行はその場で停止します。
    *   **<span style="color:red">赤信号</span>**: 立ち止まって原因を修正する必要があります。
*   **警告 (Warnings)**:
    *   コードは実行されるものの、潜在的な問題や注意すべき点を知らせてくれます。
    *   `Warning:` という文字で始まることが多いです。
    *   実行結果に影響がないか確認し、必要なら対処します。
    *   **<span style="color:gold">黄信号</span>**: 進行は可能ですが、注意が必要です。
*   **メッセージ (Prints)**:
    *   `print()` 関数による出力や、ライブラリが情報提供のために表示するメッセージです。
    *   エラーや警告ではありません。
    *   **<span style="color:green">青信号</span>**: 問題なく進行できます。

### 1.3.3 コーディング学習のヒント

*   **正確さが重要**: コンピュータは指示されたことしかできません。曖昧さや間違いは許されません。
*   **コピー、ペースト、調整**: 最初は、動くコードをコピーし、少しずつ変更して試すのが効果的です。補助輪のようなものだと考えましょう。
*   **実践あるのみ**: 実際にコードを書き、動かし、エラーと格闘する中で上達します。
*   **目標を持つ**: 「このデータを分析したい」といった具体的な目標があると、学習がスムーズに進みます。

## 1.4 Pythonライブラリ (パッケージ)

### 1.4.1 ライブラリとは？

Python本体にも多くの機能がありますが、専門的な作業（データ分析、グラフ作成など）を行うためには、追加のツールが必要です。それが**ライブラリ**（または**パッケージ**）です。

*   スマートフォンの「アプリ」のようなものだと考えてください。基本的な電話機能に加えて、地図アプリやSNSアプリをインストールして機能を追加しますよね？ それと同じです。
*   Pythonには世界中の開発者が作った便利なライブラリがたくさんあります。
*   データサイエンスでよく使うライブラリ：
    *   `pandas`: データフレーム（表形式データ）の操作・分析
    *   `numpy`: 数値計算（配列操作など）
    *   `matplotlib`, `seaborn`: グラフ作成・データ可視化
    *   `scipy`, `statsmodels`: 統計解析

ライブラリを使うには、通常2つのステップが必要です。

1.  **インストール**: アプリをスマホにダウンロードするように、ライブラリをPython環境に追加します。（Colabでは多くが最初から入っています）
2.  **インポート**: アプリを使う前にアイコンをタップして起動するように、コードの中でライブラリを読み込んで使える状態にします。

### 1.4.2 ライブラリのインストール (`pip`)

*   Colabに最初から入っていないライブラリを使いたい場合や、ローカル環境では、`pip` というコマンドを使ってインストールします。
*   Colabのコードセルでは、先頭に `!` を付けて実行します。
    ```python
    !pip install ライブラリ名
    ```
*   例: `seaborn` をインストールする場合
    ```python
    !pip install seaborn
    ```
*   インストールは通常、環境ごとに一度行えばOKです。

---
**学習チェック (LC1.1)**

`pandas` と `numpy` ライブラリがインストールされていないと仮定して、Colabでそれらをインストールするコマンドを書いてみましょう。

```python
# 回答欄
# !pip install pandas numpy
```
---

### 1.4.3 ライブラリのインポート (`import`)

*   インストールしたライブラリを使うためには、Pythonコードの最初の方で `import` 文を使って読み込む必要があります。
    ```python
    import ライブラリ名
    ```
*   **別名 (alias)** を付けるのが一般的です。特に `pandas` は `pd`、`numpy` は `np` という別名を付ける慣習があります。これにより、コード内でライブラリの機能を使うときに短い名前で呼び出せます。
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    ```
*   **注意**: インポートは、ColabノートブックやPythonスクリプトを実行するたびに**毎回**行う必要があります。
*   インポート時に `ModuleNotFoundError` というエラーが出たら、ライブラリがインストールされていない可能性があります。

---
**学習チェック (LC1.2)**

データ可視化のためによく使われる `matplotlib.pyplot` を `plt` という別名で、`seaborn` を `sns` という別名でインポートするコードを書いてみましょう。

```python
# 回答欄
# import matplotlib.pyplot as plt
# import seaborn as sns
```
---

### 1.4.4 ライブラリの使用時の注意点 (`NameError`)

*   ライブラリを `import` するのを忘れて、そのライブラリの機能を使おうとすると `NameError` が発生します。
    *   例: `import pandas as pd` をせずに `pd.DataFrame(...)` を使おうとすると、「`pd` って何？」とPythonに怒られます。
*   これは非常によくある間違いなので、エラーが出たらまず `import` を確認しましょう。

## 1.5 最初のデータセット探索: NYCフライトデータ

さあ、実際にデータを扱ってみましょう！ここでは2023年にニューヨーク市の主要3空港から出発した国内線のフライトに関するデータを使います。

### 1.5.1 準備：ライブラリのインポートとデータ読み込み

まず、必要なライブラリをインポートし、データをインターネットから読み込みます。データはCSV (Comma Separated Values) という形式で公開されているものを `pandas` を使って読み込み、データフレームという形式にします。

```python
# 必要なライブラリをインポート
import pandas as pd
import numpy as np
import requests # Webからデータを取得するため
from io import StringIO # テキストデータをファイルのように扱うため

# --- データ読み込み ---
# flights データ
url_flights = "https://raw.githubusercontent.com/tomoshige/website/refs/heads/main/docs/lectures/SIWS/datasets/flights.csv"
response_flights = requests.get(url_flights)
data_flights = StringIO(response_flights.text)
flights = pd.read_csv(data_flights)
print("flights データ読み込み完了")

# airlines データ
url_airlines = "https://raw.githubusercontent.com/tomoshige/website/refs/heads/main/docs/lectures/SIWS/datasets/airlines.csv"
response_airlines = requests.get(url_airlines)
data_airlines = StringIO(response_airlines.text)
airlines = pd.read_csv(data_airlines)
print("airlines データ読み込み完了")

# airports データ
url_airports = "https://raw.githubusercontent.com/tomoshige/website/refs/heads/main/docs/lectures/SIWS/datasets/airports.csv"
response_airports = requests.get(url_airports)
data_airports = StringIO(response_airports.text)
airports = pd.read_csv(data_airports)
print("airports データ読み込み完了")

# planes データ
url_planes = "https://raw.githubusercontent.com/tomoshige/website/refs/heads/main/docs/lectures/SIWS/datasets/planes.csv"
response_planes = requests.get(url_planes)
data_planes = StringIO(response_planes.text)
planes = pd.read_csv(data_planes)
print("planes データ読み込み完了")

# weather データ
url_weather = "https://raw.githubusercontent.com/tomoshige/website/refs/heads/main/docs/lectures/SIWS/datasets/weather.csv"
response_weather = requests.get(url_weather)
data_weather = StringIO(response_weather.text)
weather = pd.read_csv(data_weather)
print("weather データ読み込み完了")

```

これで、`flights`, `airlines`, `airports`, `planes`, `weather` という5つのデータフレームが使える状態になりました。

### 1.5.2 `nycflights` データセットについて

これらのデータセットには以下の情報が含まれています。

*   `flights`: 全てのフライト情報（出発/到着時間、遅延、航空会社、距離など）。これがメインのデータになります。
*   `airlines`: 航空会社のコードと正式名称の対応表。
*   `airports`: 空港のコード、名称、位置情報（緯度経度など）。
*   `planes`: 各航空機の製造情報など。
*   `weather`: NYCの3空港の1時間ごとの気象データ。

### 1.5.3 `flights` データフレームの概要確認

データがどんな形をしているか見てみましょう。`pandas` データフレームには便利な機能があります。

*   `.head()`: 最初の数行を表示します（デフォルトは5行）。データの中身をちらっと見るのに便利です。
    ```python
    print(flights.head())
    ```
*   `.info()`: データフレームの全体像（行数、列数、各列のデータ型、欠損値の有無など）を表示します。
    ```python
    flights.info()
    ```

`.info()` の出力を見ると、何行 (entries/rows) × 何列 (columns) のデータか、各列の名前 (Column)、欠損していない値の数 (Non-Null Count)、データの種類 (Dtype) が分かります。`int64` は整数、`float64` は小数、`object` は主に文字列です。

### 1.5.4 データフレームの探索方法

データフレームの中身をもっと詳しく見る方法をいくつか紹介します。

1.  **`display()` / Colabのデータビューア**: Colabのセルでデータフレーム名（例: `flights`）だけを書いて実行すると、インタラクティブな表形式で表示されます。ソートしたりフィルタしたりできる場合もあります。
    ```python
    # セルにこれだけ書いて実行
    flights
    ```
    または
    ```python
    from IPython.display import display
    display(flights)
    ```

2.  **`.info()`**: 先ほど見たように、データフレームの構造（行数、列数、データ型）を確認します。

3.  **`.head()`, `.tail()`, `.sample()`**:
    *   `.head(n)`: 最初の `n` 行を表示。
    *   `.tail(n)`: 最後の `n` 行を表示。
    *   `.sample(n)`: ランダムに `n` 行を抽出して表示。データ全体から偏りなく様子を見たい場合に便利です。
    ```python
    print("--- 最初の3行 ---")
    print(flights.head(3))
    print("\n--- 最後の2行 ---")
    print(flights.tail(2))
    print("\n--- ランダムな4行 ---")
    print(flights.sample(4))
    ```

4.  **特定の列へのアクセス (`[]` または `.` )**: データフレームから特定の列だけを取り出して見ることができます。
    *   `データフレーム名['列名']`: 列名を文字列で指定します。こちらの方が確実です。
    *   `データフレーム名.列名`: 列名がPythonの変数名として使える形式（スペースや特殊文字を含まないなど）の場合に使えます。
    ```python
    # airlines データフレームの 'name' 列を表示
    print(airlines['name'])

    # flights データフレームの 'dep_delay' (出発遅延) 列を表示
    print(flights.dep_delay) # ドット表記の例

    # 取り出した列は Pandas Series という一次元のデータ構造になります
    print(type(airlines['name']))
    ```

---
**学習チェック (LC1.3)**

`flights` データセットの**行**は何を表していますか？

*   A. 航空会社のデータ
*   B. 1つのフライトのデータ
*   C. 空港のデータ
*   D. 複数のフライトのデータ

*回答: B*

---
**学習チェック (LC1.4)**

`flights` データセットにある変数（列）の中で、*カテゴリカル変数*（種類や区分を表す変数）の例を挙げてください。また、それらが*量的変数*（数値を表す変数）とどう違うか説明してください。

*回答例:*
*   カテゴリカル変数: `carrier` (航空会社コード), `origin` (出発空港), `dest` (到着空港)。これらはグループ分けであり、数値計算（足し算など）に意味がありません。
*   量的変数: `dep_delay` (出発遅延時間), `distance` (飛行距離)。これらは数値であり、平均や合計などの計算が可能です。

---

### 1.5.5 識別変数と測定変数

データフレームの列（変数）には、少し性質の違うものがあります。

*   **識別変数 (Identification Variable)**: 各行（観測単位）を一意に特定するための変数。
    *   例: `airports` データフレームの `faa` (空港コード) や `name` (空港名)。これがあれば、どの空港のデータか特定できます。
*   **測定変数 (Measurement Variable)**: 各行（観測単位）の特性や測定値を表す変数。
    *   例: `airports` データフレームの `lat` (緯度), `lon` (経度), `alt` (高度)。これらは空港の具体的な特徴を示します。

データによっては、複数の識別変数を組み合わせないと一意にならない場合もあります（例: `flights` データでは、`year`, `month`, `day`, `carrier`, `flight` を組み合わせるなど）。

---
**学習チェック (LC1.5)**

`airports` データフレームの変数 `lat`, `lon`, `alt`, `tz`, `dst`, `tzone` は、各空港のどのような特性を記述していると思いますか？ <https://github.com/moderndive/nycflights23/tree/main/R> を参考にして、推測してください。

*回答例:*
*   `lat`: 緯度
*   `lon`: 経度
*   `alt`: 高度
*   `tz`: タイムゾーン (UTCからの時差)
*   `dst`: 夏時間ルール
*   `tzone`: タイムゾーン名

---
**学習チェック (LC1.6)**

あなたが考える（または知っている）データフレームで、少なくとも3つの変数があり、そのうち1つが識別変数、他の2つが測定変数であるような例を挙げてください。（例: 学生名簿、商品リストなど）

*回答例:*
学生名簿データフレーム:
*   識別変数: `student_id` (学生番号)
*   測定変数: `score` (試験の点数), `attendance_rate` (出席率)

---

### 1.5.6 ヘルプ機能

Pythonやライブラリの関数、オブジェクトについて詳しく知りたいときは、ヘルプ機能を使いましょう。

*   オブジェクトや関数の後に `?` を付けて実行する (Colab/Jupyter環境)。
*   `help()` 関数を使う。

```python
# pandas の read_csv 関数のヘルプを表示
?pd.read_csv

# または
# help(pd.read_csv)

# flights データフレーム（pandas.DataFrame）のヘルプを表示
# ?flights
# help(flights)
```

`flights` のようなデータフレーム変数自体に `?` を付けても、`pandas.DataFrame` というクラスの一般的なヘルプが表示されます。個々の変数（列）の意味を知るには、データセットのドキュメントや説明を参照する必要があります。

---
**学習チェック (LC1.7)**

Web検索なども利用して、`airports` データフレームの変数 `lat`, `lon`, `alt`, `tz`, `dst`, `tzone` が具体的に何を意味するか、LC1.5での推測を確認・修正してください。

*回答例（より詳細）:*
*   `lat`: 緯度 (北緯が正)
*   `lon`: 経度 (東経が正、西経が負)
*   `alt`: 高度 (フィート単位)
*   `tz`: UTCからの時差 (時間単位、例: -5 は EST)
*   `dst`: 夏時間 (Daylight Saving Time) の適用ルール (A=米国標準, U=不明, N=なし など)
*   `tzone`: IANA (Olson) タイムゾーン名 (例: 'America/New_York')

---

## 1.6 まとめ

この章では、PythonとGoogle Colabを使ったデータ分析の第一歩として、以下のことを学びました。

*   Python（エンジン）とColab（ダッシュボード）の関係
*   基本的なコーディング（変数、データ型、関数、条件分岐）
*   エラーや警告の読み方
*   ライブラリの重要性、インストール (`!pip`) とインポート (`import`)
*   Pandas DataFrameを使ったデータの読み込みと基本的な探索 (`.head()`, `.info()`, `.sample()`, 列選択)
*   識別変数と測定変数の違い
*   ヘルプ機能の使い方

これらのツールは、これからデータ分析を進める上での基礎となります。重要なのは、実際にコードを書いて動かしてみることです。エラーを恐れずに、どんどん試してみてください。