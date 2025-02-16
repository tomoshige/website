# Pythonでのデータ分析を始めよう

Pythonを使ってデータを探索する前に、まず理解しておくべき重要な概念があります。

1. PythonとGoogle Colaboratoryとは何か？
2. Pythonのコードの書き方
3. Pythonのモジュールとは？

これらの概念について、次のセクションで詳しく説明します（セクションaaa）。すでにある程度の知識がある場合は、セクションaaa に進んで、最初のデータセットを紹介する部分から読み始めても構いません。本書では、2013年にニューヨーク市（NYC）の主要3空港から出発したすべての国内線フライトのデータ を扱い、以降の章で詳しく分析していきます。

## Python と Google Colaboratoryとは？

本書では、PythonをGoogle Colaboratory（以下、Google Colab） を通じて使用することを前提とします。初めての方の中には、PythonとGoogle Colabの違いがわかりにくいと感じることがあるかもしれません。これをわかりやすく説明すると、Pythonは車のエンジン、Google Colabは車のダッシュボード のような関係にあります（図を参照）。

!!! Note
    ここにイメージ図を挿入

もう少し詳しく説明すると、Pythonは計算を実行するプログラミング言語 であり、一方でGoogle Colabは、Google社が提供する機械学習向けの統合開発環境（IDE） です。Google Colabには、Pythonを便利に使うためのさまざまな機能やツールが追加されています。

これは、運転するときにスピードメーターやバックミラー、ナビゲーションシステムがあることで安全かつスムーズに運転できるのと同じように、Google Colabを使うことでPythonをより直感的かつ効率的に操作できる ということです。

### Google Colaboratory で Python を始める

**Google Colaboratory（Google Colab）** は、Google が提供する**無料のクラウド環境**で、Python をブラウザ上で実行できるプラットフォームです。インストール不要で使えるため、初心者でも簡単に Python を始められます。

1. Google Colab を使う準備

Google Colab を使用するには、Google アカウントが必要です。以下の手順に従ってセットアップを行いましょう。

2. Google Colab にアクセス

  1. [Google Colaboratory](https://colab.research.google.com/) にアクセスします。
  2. Google アカウントでログインします。

3. 新しいノートブックを作成

  1. Google Colab のホーム画面で **「新しいノートブック」** をクリックします。
  2. 新しい Python の Jupyter Notebook が開きます。

4. 簡単な Python コードを実行

  1. ノートブックのセルに以下のコードを入力します：
  ```python
  print("Hello, Google Colab!")
  ```
  2. Shift + Enter を押すか、セルの左側にある再生ボタン ▶ をクリックして実行します。
  3. 出力として `Hello, Google Colab!` と表示されれば成功です。


### Google Colaboratory で Python を使う

以前の車のアナロジーを思い出してください。私たちはエンジンを直接操作するのではなく、ダッシュボード上の要素を使って車を運転します。同様に、Python を直接操作するのではなく、**Google Colaboratory（Google Colab）** を使用して Python を実行します。

Google Colab は、Google が提供するクラウドベースの Jupyter Notebook 環境です。これを利用することで、Python の環境構築を行うことなく、ブラウザ上でコードを書き、実行することができます。

Google Colab を開くと、以下のようなインターフェースが表示されます。

!!! Note  
    ここに Google Colab のインターフェースの画像を挿入

Google Colab の画面は、大きく分けて 3 つの部分に分かれています。

1. **コードセル**: ここに Python コードを記述し、実行します。
2. **出力エリア**: コードの実行結果が表示される場所です。
3. **ファイル管理エリア**: Google ドライブやローカルファイルを管理できます。

---

## Python の基本的な書き方

Python を使い始めると、「Python はどうやって使うの？」という疑問が浮かぶでしょう。Python は **インタプリタ型のプログラミング言語** であり、Excel や SPSS のような **ポイント & クリック** 操作ではなく、**コードを入力して実行する** ことで動作します。

Python を使うには、基本的なプログラミングの概念を理解する必要があります。本書はプログラミングに特化した書籍ではありませんが、データを探索・分析するために必要な最低限のプログラミング知識を学んでいきます。

---

### プログラミングの基礎と概念

基本的なプログラミングの概念と用語について説明します。すべてを暗記する必要はなく、「実際にやりながら学ぶ」ことを目指します。このガイドでは、通常の文章と `computer_code` を区別するために異なるフォントを使用します。

学習を進める上で、Python と Google Colaboratory を活用しながら、繰り返し練習することが重要です。

#### 基本概念

* *コードセル (Code Cell)*: Google Colaboratory でコードを入力し、実行する場所。
* *コードの実行*: Python に命令を与え、実際に処理を行わせること。
* *変数 (Variables)*: 値を保存するためのオブジェクト。変数に値を *代入* し、その内容を表示する方法を学びます。
* *データ型 (Data Types)*: `int` (整数), `float` (浮動小数点数), `bool` (論理型), `str` (文字列) など。
  - 整数 (`int`): `-1, 0, 2, 4092` など。
  - 浮動小数点数 (`float`): `-24.932, 0.8` など。
  - 論理型 (`bool`): `True` または `False`。
  - 文字列 (`str`): `"cabbage"`, `"Hamilton"`, `"This ramen is delicious."` など。

#### リスト (List)

複数の値をまとめたデータ構造で、`[]` を使って作成します。

```python
numbers = [6, 11, 13, 31, 90, 92]
```

#### データフレーム (DataFrame)

表形式のデータ構造であり、`pandas` ライブラリを使用して作成します。

```python
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)
print(df)
```

#### 条件分岐 (Conditionals)

* `==` を使って等価性を比較 (`=` は代入のために使用)。
  ```python
  print(2 + 1 == 3)  # True
  ```
* ブール演算: `True` / `False` の評価。
  ```python
  print(4 + 2 >= 3)  # True
  print(3 + 5 <= 1)  # False
  ```
* 論理演算子: `and` (かつ), `or` (または)。
  ```python
  print((2 + 1 == 3) and (2 + 1 == 4))  # False
  print((2 + 1 == 3) or (2 + 1 == 4))   # True
  ```

#### 関数 (Functions)

関数は特定のタスクを実行するためのものです。Python では `def` キーワードを使用して関数を定義できます。

```python
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
```

また、組み込み関数の `range()` を使用すると、数値のシーケンスを作成できます。

```python
list(range(2, 6))  # [2, 3, 4, 5]
```

---

### エラー、警告、メッセージ {#messages}

Google Colaboratory で Python を使う際、エラー、警告、メッセージが表示されることがあります。これらは赤字で表示されるため、最初は戸惑うかもしれませんが、冷静に対応すれば問題ありません。

Python で表示される赤字のメッセージには以下の種類があります。

#### **エラー (Errors)**

プログラムが正常に動作しない場合に発生し、コードの実行が停止します。

```python
print(1 / 0)  # ZeroDivisionError: division by zero
```

この場合、`0` での割り算が許可されていないためエラーが発生します。

#### **警告 (Warnings)**

コードの実行は継続できるものの、注意すべきことがある場合に表示されます。

```python
import warnings
warnings.warn("This is a warning message!")
```

#### **メッセージ (Messages)**

通常の診断情報や処理結果が表示される場合。

```python
print("Data loaded successfully!")
```

エラーや警告が表示されても焦らず、

* **エラー**: コードが実行できないので修正が必要 → <span style="color:red">赤信号: 停止して問題を確認</span>
* **警告**: 実行できるが注意が必要 → <span style="color:gold">黄色信号: 注意しながら進行</span>
* **メッセージ**: 単なる情報提供 → <span style="color:green">緑信号: 問題なし</span>

と考えるとよいでしょう。

Python と Google Colaboratory を活用しながら、繰り返し練習していきましょう！

### プログラミングを学ぶ上での心得

プログラミングを学ぶことは、外国語を学ぶことに似ています。最初は難しく、挫折することもあるかもしれません。しかし、間違いを恐れずに努力を続ければ、誰でも学び、上達することができます。プログラミングを学ぶ際に役立ついくつかのポイントを紹介します。

* **コンピュータは実はそれほど賢くない**: コンピュータやスマートフォンは「賢い」と思われがちですが、それは人間が多くの時間とエネルギーを費やして設計した結果です。実際には、コンピュータにはすべての指示を明確かつ正確に伝える必要があります。曖昧な指示やミスがあると、正しく動作しません。

* **「コピー、ペースト、修正」アプローチを活用する**: 最初にプログラミング言語を学ぶときや、特に複雑なコードを理解する必要があるときは、既存の動作するコードをコピーし、自分の目的に合わせて修正するのが効率的です。これを *「コピー、ペースト、修正」* アプローチと呼びます。最初は暗記してコードを書くのではなく、提供されたサンプルコードをコピーし、それを修正しながら学ぶことを推奨します。これは自転車の補助輪のようなもので、慣れてくれば徐々に補助なしでコードを書けるようになります。

* **実践を通じて学ぶのが最良の方法**: プログラミングのスキルを向上させる最も効果的な方法は、実際に手を動かしてコードを書くことです。特に、自分が興味のあるデータを分析するなど、具体的なプロジェクトを持つと、学習がスムーズに進みます。

* **練習が鍵**: 外国語を上達させる唯一の方法が繰り返し話すことであるように、プログラミングを上達させる唯一の方法も多くの練習を積むことです。心配しないでください！ 私たちは十分な練習の機会を提供します。

Python と Google Colaboratory を活用しながら、繰り返し練習していきましょう！



## Pythonのライブラリとは？

Python初心者が最初に戸惑う概念のひとつに、Pythonのライブラリ（またはパッケージ）があります。Pythonライブラリは、追加の関数、データ、そしてドキュメントを提供することで、Pythonの機能を拡張します。これらのライブラリは、世界中のPythonユーザーによって作られており、通常はPyPI（Python Package Index）などから無料でダウンロードすることができます。例えば、本書で使用するライブラリの中には、以下のものがあります：

- matplotlib: データの可視化のために使用します。
- Pandas: データの整形や操作のために使用します。
- NumPy: 数値計算や配列操作のために使用します。

Pythonライブラリは、スマートフォンにダウンロードできるアプリに例えることができます。つまり、Python自体は新しいスマートフォンのようなものです。最初は基本的な機能は備わっていますが、すべての機能が揃っているわけではありません。Pythonライブラリは、スマートフォンにApp StoreやGoogle Playからアプリをダウンロードするのと同じように、必要に応じて追加することで、あなたの作業環境を拡張してくれます。

このアナロジーを続けるために、写真の編集や共有に使うInstagramアプリを例に考えてみましょう。たとえば、新しいスマートフォンを購入して、撮った写真を友人や家族とInstagramで共有したいとします。その場合、以下のステップが必要です：

1. アプリのインストール:新しいスマートフォンにはInstagramアプリがプリインストールされていないため、App StoreやGoogle Playからアプリをダウンロードする必要があります。一度インストールすれば、その後はアップデートがあるまで再インストールする必要はありません。

2. アプリの起動:インストールが完了したら、Instagramアプリを起動します。

Instagramアプリを起動すれば、写真を友人や家族と共有できるようになります。これと同様に、Pythonライブラリを利用する場合も次の2つのステップを踏みます：

1. ライブラリのインストール:これは、スマートフォンにアプリをインストールするのと同じです。多くのライブラリは、PythonやGoogle Colaboratoryの初期状態には含まれていないため、初めて使用する際にはpipなどを使ってインストールする必要があります。一度インストールすれば、通常は更新が必要になるまで再インストールすることはありません。

2. ライブラリのインポート:これは、インストールしたアプリを起動するのと同じです。Pythonでは、使用するライブラリをプログラム内で毎回import文を用いて読み込む必要があります。

では、データの可視化のためのmatplotlibライブラリを例に、この2つのステップを実際に行ってみましょう。

```python
# matplotlibのインストール
!pip install matplotlib

# matplotlibのインポート
import matplotlib.pyplot as plt

# 簡単な折れ線グラフの描画
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]
plt.plot(x, y)
plt.xlabel("X軸")
plt.ylabel("Y軸")
plt.title("Matplotlibのサンプルグラフ")
plt.show()
```

### Pythonライブラリのインストール

> **Google Colaboratoryについての注意**: 
> Google Colaboratoryでは、多くのライブラリが事前にインストールされていますが、追加で必要なライブラリは自分でインストールする必要があります。Google Colaboratoryを使わずにローカル環境でPythonを実行する場合も、ライブラリのインストール手順を知っておくことが重要です（今回はローカル環境は使用しませんが、今後利用する際の参考に記載しておきます）。

Pythonのライブラリをインストールする方法は2種類あります。

1. **簡単な方法**（Google ColaboratoryまたはJupyter Notebookで実行）:
    `!pip install` コマンドを用いてライブラリをインストールします。

```python
!pip install seaborn pandas numpy matplotlib
```

2. **ローカル環境でのインストール**（ターミナルまたはコマンドプロンプトで実行：今回は利用しない）:

```sh
pip install seaborn pandas numpy matplotlib
```

ローカル環境においては、スマートフォンのアプリと同様、一度インストールすれば再度インストールする必要はありませんが、Google Colaboratoryの環境においては、時間が経過するとリセットされるため、再度インストールが必要になる場合があります。イメージとしては、貸し出しスマホです。一度店に返却すると中身のデータが消されていて、再度アプリを入れ直す必要があるのと同じです。

### ライブラリの読み込み

ライブラリをインストールした後、それをPythonプログラム内で使用するためには「読み込む」必要があります。以下のコードを実行することで、ライブラリを読み込むことができます。

```python
# libraryの読み込み
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

もし以下のようなエラーが表示された場合:

```
ModuleNotFoundError: No module named 'seaborn'
```

これはライブラリがインストールされていないことを示しています。その場合は、前述の`pip install`コマンドを実行してください。

!!! Note
    `pip` を用いて、`seaborn` をインストールしてみてください。

### ライブラリの利用

Pythonでデータ分析を行う際、適切にライブラリをインストールし、読み込むことが重要です。例えば、`seaborn` を使用してデータの可視化を行う場合、以下のコードを実行するとシンプルな散布図を描画できます。

```python
# サンプルデータの作成
np.random.seed(10)
data = pd.DataFrame({
    "x": np.random.rand(50),
    "y": np.random.rand(50)
})

# Seabornを使った散布図の描画
sns.scatterplot(x="x", y="y", data=data)
plt.title("Seabornを用いた散布図の例")
plt.show()
```

このように、ライブラリを正しくインストールし、読み込むことで、Pythonでのデータ分析や可視化が簡単に行えるようになります。

## 初めてのデータセットを探索してみよう

これまで学んだことを活用して、実際のデータを探索してみましょう。データは画像、テキスト、数値などさまざまな形式で存在しますが、本書では主に「スプレッドシート」形式のデータセットに焦点を当てます。これは多くの分野でデータが収集・保存される最も一般的な方法です。Pythonでは、これらの「スプレッドシート」形式のデータセットを**データフレーム**と呼びます。以降、本書ではデータフレームとして保存されたデータの操作に注目していきます。

まず、必要なパッケージを読み込みます。以下のコードを実行してください。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### データの読み込み

今回は、広告費用と売上の関係を示したデータセットを使用します。このデータセットには、TV、ラジオ、新聞に投じた予算と売上の情報が含まれています。データを読み込むには、以下のコードを実行してください。

```python
url = 'https://raw.githubusercontent.com/a-martyn/ISL-python/ee156568a8f7307be71dad5390bae12b51dcd93f/Notebooks/data/Advertising.csv'
data = pd.read_csv(url, index_col=0)
```

### データフレームの構造を確認する

データフレームの構造を理解するために、以下のコードを実行してみましょう。

```python
data.info()
```

この出力から、データフレームには200行と4列があり、各列のデータ型と欠損値の有無が確認できます。

次に、データの最初の数行を表示して、データの内容を確認します。

```python
data.head()
```

この出力から、各列の名前と最初の5行のデータが表示され、データの概要を把握できます。

### データフレームの探索

データフレームを探索する方法はいくつかあります。ここでは、3つの方法を紹介します。

1. **`head()`メソッド**: データフレームの最初の数行を表示します。

    ```python
    data.head()
    ```

2. **`describe()`メソッド**: 数値データの基本的な統計量を表示します。

    ```python
    data.describe()
    ```

3. **`columns`属性**: データフレームの列名を表示します。

    ```python
    data.columns
    ```

これらの方法を組み合わせて、データの概要を把握しましょう。

### データの可視化

データの関係性を視覚的に理解するために、散布図を作成してみましょう。例えば、TV広告費用と売上の関係を確認するには、以下のコードを実行します。

```python
plt.scatter(data['TV'], data['Sales'])
plt.xlabel('TV Advertising Budget (in thousands of dollars)')
plt.ylabel('Sales (in thousands of units)')
plt.title('TV Advertising vs Sales')
plt.show()
```

この散布図から、TV広告費用と売上の間に正の相関があることが視覚的に確認できます。

同様に、ラジオや新聞の広告費用と売上の関係も散布図で確認してみましょう。

```python
# ラジオ広告費用と売上の関係
plt.scatter(data['Radio'], data['Sales'])
plt.xlabel('Radio Advertising Budget (in thousands of dollars)')
plt.ylabel('Sales (in thousands of units)')
plt.title('Radio Advertising vs Sales')
plt.show()

# 新聞広告費用と売上の関係
plt.scatter(data['Newspaper'], data['Sales'])
plt.xlabel('Newspaper Advertising Budget (in thousands of dollars)')
plt.ylabel('Sales (in thousands of units)')
plt.title('Newspaper Advertising vs Sales')
plt.show()
```

これらの散布図を通じて、各広告媒体の予算と売上の関係性を視覚的に理解することができます。

### データの基本統計量

データの基本的な統計量を確認することで、データの分布や中心傾向を理解できます。以下のコードを実行してみましょう。

```python
data.describe()
```

この出力から、各変数の平均値、標準偏差、最小値、最大値、四分位数などの情報が得られます。以上の手順で、PythonとPandasを使用してデータセットを読み込み、その構造を理解し、基本的な統計量を確認し、データの可視化を行いました。これらの方法を活用して、さまざまなデータセットを探索し、分析の基礎を築いていきましょう。

## まとめ

本章では、Pythonを使用してデータを探索するための基本的なツールセットを紹介しました。この章にすべての知識が含まれているわけではありません。すべてを盛り込むと膨大な量になり、実用的ではなくなってしまうからです！最も重要なのは、Google Colaboratory上で実際にコードを実行し、試行錯誤を繰り返しながら学ぶことです。

### 追加リソース

もし、PythonやGoogle Colaboratory、データ分析に不慣れで、より詳細な入門書を求めている場合は、以下のリソースを参照することをおすすめします。

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)  
  NumPy、Pandas、Matplotlib、Scikit-Learnなど、Pythonを使ったデータサイエンスの基礎を学ぶための包括的なガイドです。

- [Google Colaboratoryの公式ドキュメント](https://colab.research.google.com/notebooks/welcome.ipynb)  
  Google Colabの基本的な使い方を学ぶことができます。クラウド環境でPythonを実行できるため、環境構築なしにすぐに始められます。

- [Pandasの公式ドキュメント](https://pandas.pydata.org/docs/)  
  データフレーム操作の詳細なリファレンスです。

- [Matplotlibの公式ドキュメント](https://matplotlib.org/stable/contents.html)  
  データの可視化に関する詳細な情報が得られます。

### 今後の展開

次の章では、「データサイエンスにおける最も重要なツールの1つ」とも言えるデータの可視化について学びます。PandasとMatplotlibを活用し、視覚的にデータを探索する方法を探求していきます。

データ可視化は、`head()` や `describe()` のような関数では捉えきれないパターンや傾向を明らかにする強力なツールです。以下のような基本的なプロットを学び、データの背後にあるストーリーを読み解く力を養いましょう。

```python
import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
url = 'https://raw.githubusercontent.com/a-martyn/ISL-python/ee156568a8f7307be71dad5390bae12b51dcd93f/Notebooks/data/Advertising.csv'
data = pd.read_csv(url, index_col=0)

# 散布図の作成（TV広告費と売上の関係）
plt.scatter(data['TV'], data['Sales'])
plt.xlabel('TV Advertising Budget (in thousands of dollars)')
plt.ylabel('Sales (in thousands of units)')
plt.title('TV Advertising vs Sales')
plt.show()
```

このように、次の章ではデータのパターンをより直感的に理解する方法を学びます。データサイエンスの旅を続けていきましょう！


次の章では、「データサイエンスにおける最も重要なツールの1つ」とも言えるデータの可視化について学びます。PandasとMatplotlibを活用し、視覚的にデータを探索する方法を探求していきます。

データ可視化は、`head()` や `describe()` のような関数では捉えきれないパターンや傾向を明らかにする強力なツールです。以下のような基本的なプロットを学び、データの背後にあるストーリーを読み解く力を養いましょう。

```python
import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
url = 'https://raw.githubusercontent.com/a-martyn/ISL-python/ee156568a8f7307be71dad5390bae12b51dcd93f/Notebooks/data/Advertising.csv'
data = pd.read_csv(url, index_col=0)

# 散布図の作成（TV広告費と売上の関係）
plt.scatter(data['TV'], data['Sales'])
plt.xlabel('TV Advertising Budget (in thousands of dollars)')
plt.ylabel('Sales (in thousands of units)')
plt.title('TV Advertising vs Sales')
plt.show()
```

## 理解度チェック課題

以下の問題を解いて、Python、Google Colaboratory、Pandas、Matplotlibを用いたデータ探索の理解を深めましょう。

### 問題 1: データの読み込み

以下のコードを実行すると、どのようなデータが表示されますか？  
データの概要を説明してください。

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/a-martyn/ISL-python/ee156568a8f7307be71dad5390bae12b51dcd93f/Notebooks/data/Advertising.csv'
data = pd.read_csv(url, index_col=0)
print(data.head())
```

### 問題 2: データの統計情報

以下のコードを実行すると、どのような情報が得られますか？  
得られた情報から、どの広告媒体が売上に最も影響を与えていると考えられますか？

```python
print(data.describe())
```

### 問題 3: データの可視化

以下のコードを実行して、TV広告費と売上の関係をプロットしてください。  
グラフの傾向を説明し、売上に対するTV広告の影響について考察してください。

```python
import matplotlib.pyplot as plt

plt.scatter(data['TV'], data['Sales'])
plt.xlabel('TV Advertising Budget (in thousands of dollars)')
plt.ylabel('Sales (in thousands of units)')
plt.title('TV Advertising vs Sales')
plt.show()
```

### 問題 4: 他の広告媒体との関係

ラジオ (`Radio`) や新聞 (`Newspaper`) の広告費と売上の関係を調べるために、  
上記のコードを修正して、それぞれの散布図を作成してください。  
結果から、どの広告媒体が売上に対して最も大きな影響を持つか考察してください。

### 問題 5: データの応用

企業が広告費を最適に配分するためには、どのような分析を行うべきでしょうか？  
また、PandasやMatplotlibを活用して、どのような追加の視覚化や統計分析ができるか提案してください。

---

以上の問題を解くことで、Pythonを用いたデータの探索と可視化の基礎を確認できます。
実際にコードを実行し、得られた結果をもとに考察を深めてみてください。
