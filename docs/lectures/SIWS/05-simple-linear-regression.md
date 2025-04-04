# 第5章 基本的回帰分析

## はじめに

これまでに習得したデータ可視化のスキル（第2章）、データ操作のスキル（第3章）、データのインポートと「整然データ」の概念（第4章）を踏まえて、データモデリングに進みましょう。データモデリングの基本的な前提は、以下の関係を明示することです：

* *目的変数* $y$（*従属変数*または応答変数とも呼ばれる）と
* *説明変数*/*予測変数* $x$（*独立変数*または共変量とも呼ばれる）

数学的な用語では、目的変数 $y$ を説明変数 $x$ の「関数として」モデル化します。ここで「関数」とは、Rの`ggplot()`のような関数ではなく、数学的な関数を指します。説明変数と予測変数という2つの異なる用語がありますが、これらは互換的に使用されることが多いです。大まかに言えば、データモデリングは次の2つの目的のいずれかを果たします：

1. **説明のためのモデリング**：目的変数 $y$ と説明変数 $x$ の間の関係を明示的に記述し定量化し、その関係の有意性を判断し、これらの関係を要約する指標を持ち、場合によっては変数間の*因果関係*を特定することを目的とします。
2. **予測のためのモデリング**：予測変数 $x$ に含まれる情報に基づいて目的変数 $y$ を予測することを目的とします。説明のためのモデリングとは異なり、すべての変数がどのように関連し相互作用するかを理解することよりも、$x$ の情報を使って $y$ について良い予測ができるかどうかに重点を置きます。

例えば、患者が肺がんを発症するかどうかという目的変数 $y$ と、喫煙習慣、年齢、社会経済的地位などのリスク要因に関する情報 $x$ に興味があるとします。説明のためのモデリングでは、異なるリスク要因の影響を記述し定量化することに関心があります。これは、特定の年齢層の喫煙者を禁煙プログラムの広告でターゲットにするなど、集団の肺がん発生率を減らすための介入を設計したいからかもしれません。一方、予測のためのモデリングでは、個々のリスク要因がどのように肺がんに寄与するかを理解することよりも、どの人が肺がんにかかるかをうまく予測できるかどうかだけが重要です。

本書では、説明のためのモデリングに焦点を当て、$x$ を*説明変数*と呼びます。予測のためのモデリングに興味がある場合は、『Pythonによる統計的学習入門』などの書籍を参照することをお勧めします。また、決定木モデルやニューラルネットワークなど、多くのモデリング技術がありますが、本書では特定の技術に焦点を当てます：*線形回帰*です。線形回帰は、最も一般的かつ理解しやすいモデリングアプローチの一つです。

線形回帰には*数値的*な目的変数 $y$ と、*数値的*または*カテゴリ的*な説明変数 $x$ が含まれます。さらに、$y$ と $x$ の関係は線形、つまり直線であると仮定されます。ただし、「直線」が何を構成するかは、説明変数 $x$ の性質によって異なります。

第5章では、1つの説明変数 $x$ を持つモデルのみを考えます。5.1節では、説明変数が数値的です。このシナリオは*単純線形回帰*として知られています。5.2節では、説明変数がカテゴリ的です。

第6章では、基本的な回帰分析の考え方を拡張し、2つの説明変数 $x_1$ と $x_2$ を持つモデルを考えます。6.1節では、2つの数値的説明変数を持ちます。6.2節では、1つの数値的説明変数と1つのカテゴリ的説明変数を持ちます。

第7章では、回帰モデルを再検討し、第8章、第9章、第10章で学ぶ*統計的推論*のツールを使用して結果を分析します。

それでは、基本的な回帰分析から始めましょう。これは、1つの説明変数 $x$ を持つ線形回帰モデルを指します。また、*相関係数*、「相関は必ずしも因果関係ではない」、「最適な当てはめ直線」とは何かなどの重要な統計的概念についても説明します。

### 必要なパッケージ

この章で必要なパッケージをすべてインポートしましょう。このチュートリアルでは、Pythonを使用するため、R言語のパッケージに相当するPythonライブラリをインポートします。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW
import scipy.stats as stats
from IPython.display import display, HTML
```

Google Colabで実行する場合は、以下のコードでデータをダウンロードしましょう：

```python
# データのダウンロード
!mkdir -p data
!wget -O data/evals.csv https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/evals.csv
!wget -O data/gapminder2007.csv https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/gapminder2007.csv
```

## 5.1 1つの数値的説明変数

大学や大学の教授や講師の中には、学生から高い教育評価を受ける人とそうでない人がいるのはなぜでしょうか？異なる人口統計グループの教員間で教育評価に違いはありますか？学生のバイアスが影響している可能性はありますか？これらはすべて、大学/大学の管理者が関心を持つ質問です。教育評価は、どの講師や教授が昇進するかを判断する際に考慮される多くの基準の一つだからです。

テキサス州オースティンのテキサス大学（UT Austin）の研究者たちは、次の研究課題に取り組みました：教育の評価スコアの違いを説明する要因は何か？そのために、463のコースに関する教員とコース情報を収集しました。

この節では、教師の「美しさ」スコア（これがどのように決定されたかは後で説明します）という1つの数値変数に基づいて教師の教育スコアの違いを説明してみましょう。「美しさ」スコアが高い教師は教育評価も高いのでしょうか？あるいは「美しさ」スコアが高い教師は教育評価が低い傾向があるのでしょうか？または「美しさ」スコアと教育評価の間に関係はないのでしょうか？これらの疑問に答えるために、*単純線形回帰*を使用して教育スコアと「美しさ」スコアの関係をモデル化します：

1. 数値的な目的変数 $y$（教師の教育スコア）と
2. 1つの数値的な説明変数 $x$（教師の「美しさ」スコア）を持ちます。

### 探索的データ分析

UT Austinの463コースのデータを読み込みましょう：

```python
# データの読み込み
evals = pd.read_csv('data/evals.csv')

# この章で使用する変数のみを選択
evals_ch5 = evals[['ID', 'score', 'bty_avg', 'age']]
```

分析やモデリングを行う前の重要なステップは、*探索的データ分析*（EDA）を実行することです。EDAによって、データ内の個々の変数の分布、変数間に存在する可能性のある関係、外れ値や欠損値の有無、そして最も重要なことにモデルの構築方法についての理解が得られます。EDAでは主に以下の3つのステップを行います：

1. 生のデータ値を見る
2. 平均値、中央値、四分位範囲などの要約統計量を計算する
3. データの可視化を作成する

まず最初のステップとして、生のデータ値を見てみましょう：

```python
# データの最初の数行を表示
evals_ch5.head()

# データの構造を確認
print(evals_ch5.info())
```

変数の詳細は以下の通りです：

1. `ID`：1から463までのコースを区別するための識別変数。
2. `score`：コース講師の平均教育スコア。1が最低、5が最高。これが興味のある目的変数 $y$ です。
3. `bty_avg`：コース講師の平均「美しさ」スコア。1が最低、10が最高。これが興味のある説明変数 $x$ です。
4. `age`：コース講師の年齢。これは後で学習確認で使用する別の説明変数 $x$ です。

次に、要約統計量を計算しましょう：

```python
# 要約統計量の計算
evals_ch5.describe()

# 相関係数の計算
correlation = evals_ch5[['score', 'bty_avg']].corr().iloc[0, 1]
print(f'相関係数: {correlation:.3f}')
```

相関係数の値は約0.187であり、これは教育スコアと「美しさ」スコアの間に「弱い正の」関係があることを示しています。相関係数は-1から1の範囲で、以下を示します：

* -1は完全な*負の関係*を示します：一方の変数が増加すると、もう一方の変数は直線的に減少する傾向があります。
* 0は関係がないことを示します：両方の変数の値は互いに独立して上下します。
* +1は完全な*正の関係*を示します：一方の変数が増加すると、もう一方の変数も線形的に増加する傾向があります。

次に、データの可視化を行いましょう。両方の変数が数値的なので、散布図が適切です：

```python
# 散布図の作成
plt.figure(figsize=(10, 6))
plt.scatter(evals_ch5['bty_avg'], evals_ch5['score'], alpha=0.5)
plt.xlabel('美しさスコア')
plt.ylabel('教育スコア')
plt.title('教育スコアと美しさスコアの関係')

# 回帰直線の追加
x = evals_ch5['bty_avg']
y = evals_ch5['score']
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='blue')

plt.grid(True, alpha=0.3)
plt.show()

print(f'回帰直線の式: score = {b:.3f} + {m:.3f} * bty_avg')
```

この図から、ほとんどの「美しさ」スコアは2から8の間にあり、ほとんどの教育スコアは3から5の間にあることがわかります。また、教育スコアと「美しさ」スコアの間の関係は「弱い正の」関係であるように見えます。これは、先ほど計算した相関係数0.187と一致しています。

さらに、オーバープロットの問題を調べるために、ジッタリングを適用してみましょう：

```python
# ジッタリングを適用した散布図
plt.figure(figsize=(10, 6))
plt.scatter(evals_ch5['bty_avg'] + np.random.normal(0, 0.05, len(evals_ch5)), 
            evals_ch5['score'] + np.random.normal(0, 0.05, len(evals_ch5)), 
            alpha=0.5)
plt.xlabel('美しさスコア (ジッタリング適用)')
plt.ylabel('教育スコア (ジッタリング適用)')
plt.title('教育スコアと美しさスコアの関係 (ジッタリング適用)')
plt.plot(x, m*x + b, color='blue')
plt.grid(True, alpha=0.3)
plt.show()
```

ジッタリングを適用すると、点が重なっている場所がより明確になります。これにより、データの分布をよりよく理解できます。

### 単純線形回帰

中学校や高校の代数学から、直線の方程式は $y = a + b\cdot x$ であることを思い出すかもしれません。この式は2つの係数 $a$ と $b$ で定義されます。切片係数 $a$ は $x = 0$ のときの $y$ の値です。傾き係数 $b$ は $x$ が1増加するごとの $y$ の増加量です。これは「上昇/実行」とも呼ばれます。

しかし、回帰直線を定義する際は、少し異なる表記を使用します：回帰直線の方程式は $\widehat{y} = b_0 + b_1 \cdot x$ です。切片係数は $b_0$ なので、$b_0$ は $x = 0$ のときの $\widehat{y}$ の値です。$x$ の傾き係数は $b_1$ で、$x$ が1増加するごとの $\widehat{y}$ の増加量です。$y$ の上に「ハット」をつけるのはなぜでしょうか？これは、回帰において「当てはめ値」、つまり与えられた $x$ 値に対する回帰直線上の $y$ の値を示すために一般的に使用される表記法です。

Pythonで線形回帰モデルを適合させましょう：

```python
# statsmodelsを使用した線形回帰
model = smf.ols('score ~ bty_avg', data=evals_ch5).fit()
print(model.summary())

# 回帰表の出力（簡略化バージョン）
coef = model.params
std_err = model.bse
t_vals = model.tvalues
p_vals = model.pvalues
conf_int = model.conf_int()

regression_table = pd.DataFrame({
    'term': ['intercept', 'bty_avg'],
    'estimate': coef,
    'std_error': std_err,
    't_statistic': t_vals,
    'p_value': p_vals,
    'lower_ci': conf_int[0],
    'upper_ci': conf_int[1]
})

print(regression_table)
```

回帰表の`estimate`列から、切片 $b_0$ ≈ 3.88と傾き $b_1$ ≈ 0.067があります。したがって、回帰直線の方程式は以下のようになります：

$$
\begin{aligned}
\widehat{y} &= b_0 + b_1 \cdot x\\
\widehat{\text{score}} &= b_0 + b_{\text{bty}\_\text{avg}} \cdot\text{bty}\_\text{avg}\\
&= 3.88 + 0.067\cdot\text{bty}\_\text{avg}
\end{aligned}
$$

切片 $b_0$ = 3.88は、「美しさ」スコア`bty_avg`が0のコースの平均教育スコア $\widehat{y}$ = $\widehat{\text{score}}$ です。あるいはグラフで言えば、$x$ = 0のとき、直線が$y$軸と交わる点です。ただし、回帰直線の切片には数学的な解釈がありますが、ここでは*実用的な*解釈はありません。`bty_avg`が0の値を観察することは不可能だからです。これは6人のパネリストの「美しさ」スコア（1から10の範囲）の平均です。さらに、回帰直線を含む散布図を見ると、「美しさ」スコアが0に近い教師はいません。

より興味深いのは、`bty_avg`の傾き $b_1$ = $b_{\text{bty}\_\text{avg}}$ ≈ 0.067です。これは教育スコアと「美しさ」スコアの変数間の関係を要約しています。符号が正であることから、これらの2つの変数間に正の関係があることが示唆されています。つまり、「美しさ」スコアが高い教師は教育スコアも高い傾向があります。先ほど相関係数が0.187であることを思い出してください。両方とも同じ正の符号を持っていますが、値は異なります。相関の解釈は「線形関連の強さ」です。傾きの解釈は少し異なります：

> `bty_avg`が1単位増加するごとに、*平均して*、`score`は0.067単位*関連して*増加します。

我々は*関連した*増加のみを述べており、必ずしも*因果的な*増加ではないことに注意してください。例えば、高い「美しさ」スコアが直接的に高い教育スコアを引き起こすわけではないかもしれません。代わりに、次のようなことが当てはまるかもしれません：裕福な背景を持つ個人は教育背景が強いため、教育スコアが高くなる傾向があり、同時にこれらの裕福な個人は「美しさ」スコアも高い傾向があります。言い換えれば、2つの変数が強く関連しているからといって、一方が他方を引き起こすとは限りません。これは「相関は必ずしも因果関係ではない」とよく言われるフレーズに要約されています。

さらに、この関連した増加は*平均して*0.067単位の教育`score`であると言います。なぜなら、`bty_avg`スコアが1単位異なる2人の教師がいても、教育スコアの差が正確に0.067になるとは限らないからです。傾き0.067が言っているのは、すべての可能なコースにわたって、「美しさ」スコアが1異なる2人の教師の教育スコアの*平均*差が0.067であるということです。

### 観測値/予測値と残差

回帰分析では、以下の3つの重要な概念があります：

1. 観測値 $y$：目的変数の実際の観測値
2. 予測値 $\widehat{y}$：与えられた $x$ 値に対する回帰直線上の値
3. 残差 $y - \widehat{y}$：観測値と予測値の間の誤差

例えば、「美しさ」スコアが $x$ = 7.333で、実際の教育スコアが $y$ = 4.9のある講師を考えてみましょう。このケースでは：

- 予測値 $\widehat{y} = b_0 + b_1 \cdot x = 3.88 + 0.067 \cdot 7.333 = 4.37$ です。
- 残差は $y - \widehat{y} = 4.9 - 4.37 = 0.53$ です。

Pythonではこれらの値を以下のように計算できます：

```python
# 予測値と残差の計算
evals_ch5['score_hat'] = model.predict(evals_ch5)
evals_ch5['residual'] = evals_ch5['score'] - evals_ch5['score_hat']

# 例示用に特定の講師のデータを表示
example_instructor = evals_ch5[evals_ch5['bty_avg'] == 7.333].iloc[0]
print(f"「美しさ」スコア: {example_instructor['bty_avg']}")
print(f"実際の教育スコア (y): {example_instructor['score']}")
print(f"予測された教育スコア (ŷ): {example_instructor['score_hat']:.3f}")
print(f"残差 (y - ŷ): {example_instructor['residual']:.3f}")

# 残差と予測値の関係を可視化
plt.figure(figsize=(10, 6))
plt.scatter(evals_ch5['score_hat'], evals_ch5['residual'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('予測値 (ŷ)')
plt.ylabel('残差 (y - ŷ)')
plt.title('残差プロット')
plt.grid(True, alpha=0.3)
plt.show()

# ヒストグラムで残差の分布を確認
plt.figure(figsize=(10, 6))
plt.hist(evals_ch5['residual'], bins=20, alpha=0.7, color='skyblue')
plt.xlabel('残差 (y - ŷ)')
plt.ylabel('頻度')
plt.title('残差のヒストグラム')
plt.grid(True, alpha=0.3)
plt.show()
```

### 最適な当てはめ直線

回帰直線は「最適な当てはめ」直線とも呼ばれます。しかし、「最適」とは何を意味するのでしょうか？回帰で「最適」を決定するために使用される基準を詳しく見てみましょう。

回帰分析では、すべての可能な直線の中から、「残差平方和」を最小にする直線を選びます。残差平方和は以下のように計算されます：

$$
\sum_{i=1}^{n}(y_i - \widehat{y}_i)^2
$$

これは、モデルの「適合度の欠如」の尺度です。残差平方和の値が大きいほど、適合度の欠如が大きくなります。これは、より悪い当てはめのモデルに対応します。

回帰直線がすべての点に完全に当てはまる場合、残差平方和は0になります。これは、回帰直線がすべての点に完全に当てはまる場合、予測値 $\widehat{y}$ はすべてのケースで観測値 $y$ に等しく、したがって残差 $y-\widehat{y}$ はすべてのケースで0になるためです。そして、多数の0の合計はまだ0です。

さらに、463点の雲を通して描くことができるすべての可能な直線のうち、回帰直線はこの値を最小化します。言い換えれば、回帰とそれに対応する予測値 $\widehat{y}$ は残差平方和を最小化します。

Pythonで残差平方和を計算しましょう：

```python
# 残差平方和の計算
sum_squared_residuals = np.sum(evals_ch5['residual']**2)
print(f"残差平方和: {sum_squared_residuals:.3f}")
```

他の直線を散布図に描いて、残差平方和がどのように変化するかを見ることもできます：

```python
# オリジナルの回帰直線と任意の2つの直線を比較
plt.figure(figsize=(10, 6))
plt.scatter(evals_ch5['bty_avg'], evals_ch5['score'], alpha=0.5)

# オリジナルの回帰直線
x_range = np.linspace(min(evals_ch5['bty_avg']), max(evals_ch5['bty_avg']), 100)
plt.plot(x_range, b + m*x_range, color='blue', label=f'回帰直線: y = {b:.3f} + {m:.3f}x')

# 任意の直線1
m1, b1 = 0.04, 4.0  # 任意の傾きと切片
plt.plot(x_range, b1 + m1*x_range, color='red', linestyle='--', 
         label=f'任意の直線1: y = {b1:.3f} + {m1:.3f}x')

# 任意の直線2
m2, b2 = 0.1, 3.5  # 別の任意の傾きと切片
plt.plot(x_range, b2 + m2*x_range, color='green', linestyle='-.', 
         label=f'任意の直線2: y = {b2:.3f} + {m2:.3f}x')

plt.xlabel('美しさスコア')
plt.ylabel('教育スコア')
plt.title('オリジナルの回帰直線と任意の直線の比較')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# それぞれの直線に対する残差平方和を計算
def calculate_sse(slope, intercept):
    predicted = intercept + slope * evals_ch5['bty_avg']
    residuals = evals_ch5['score'] - predicted
    return np.sum(residuals**2)

sse_regression = calculate_sse(m, b)
sse_line1 = calculate_sse(m1, b1)
sse_line2 = calculate_sse(m2, b2)

print(f"回帰直線の残差平方和: {sse_regression:.3f}")
print(f"任意の直線1の残差平方和: {sse_line1:.3f}")
print(f"任意の直線2の残差平方和: {sse_line2:.3f}")
```

## 5.2 1つのカテゴリ的説明変数

世界中のすべての国で平均寿命は同じではないという不幸な真実があります。国際開発機関はこの問題に取り組むためにリソースをどこに配分すべきかを特定するために、平均寿命の違いを研究することに関心を持っています。この節では、平均寿命の違いを2つの方法で探ります：

1. 大陸間の違い：アフリカ、南北アメリカ、アジア、ヨーロッパ、オセアニアという世界の5つの人口のある大陸間で平均寿命に有意な違いはありますか？
2. 大陸内の違い：世界の5つの大陸内で平均寿命はどのように変動しますか？例えば、アフリカの国々の平均寿命のばらつきは、アジアの国々の平均寿命のばらつきよりも大きいですか？

このような疑問に答えるために、Gapminderデータセットを使用します。このデータセットには、1952年から2007年の5年間隔で142カ国の平均寿命、一人当たりGDP、人口などの国際開発統計が含まれています。

2007年のデータを読み込み、分析に必要な変数のみを選択します：

```python
# 2007年のGapminderデータを読み込む
gapminder2007 = pd.read_csv('data/gapminder2007.csv')

# データの最初の数行を表示
gapminder2007.head()
```

### 探索的データ分析

まず、データの要約統計量を確認しましょう：

```python
# 数値変数の要約統計量
gapminder2007.describe()

# continentごとの国の数を確認
continent_counts = gapminder2007['continent'].value_counts().reset_index()
continent_counts.columns = ['continent', 'count']
print(continent_counts)

# 各大陸の平均寿命を計算
lifeExp_by_continent = gapminder2007.groupby('continent')['lifeExp'].agg(['mean', 'median', 'std']).reset_index()
print(lifeExp_by_continent)
```

データの可視化をいくつか作成しましょう：

```python
# 平均寿命のヒストグラム
plt.figure(figsize=(10, 6))
plt.hist(gapminder2007['lifeExp'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('平均寿命')
plt.ylabel('国の数')
plt.title('全世界の平均寿命分布（2007年）')
plt.grid(True, alpha=0.3)
plt.show()

# 大陸ごとのヒストグラム
plt.figure(figsize=(12, 8))
for i, continent in enumerate(gapminder2007['continent'].unique(), 1):
    plt.subplot(2, 3, i)
    subset = gapminder2007[gapminder2007['continent'] == continent]
    plt.hist(subset['lifeExp'], bins=10, color='skyblue', edgecolor='black')
    plt.title(f'{continent}')
    plt.xlabel('平均寿命')
    plt.ylabel('国の数')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 箱ひげ図
plt.figure(figsize=(12, 6))
sns.boxplot(x='continent', y='lifeExp', data=gapminder2007)
plt.xlabel('大陸')
plt.ylabel('平均寿命')
plt.title('大陸別の平均寿命（2007年）')
plt.grid(True, alpha=0.3)
plt.show()
```

残念ながら、アフリカの平均寿命分布は他の大陸よりもかなり低く、ヨーロッパでは平均寿命が高い傾向があり、さらにその変動も少ないことがわかります。一方、アジアとアフリカは平均寿命の変動が最も大きいです。オセアニアの変動は最も少ないですが、オセアニアにはオーストラリアとニュージーランドの2カ国しかないことに注意してください。

### 線形回帰

カテゴリ的な説明変数を持つ線形回帰モデルを適合させましょう：

```python
# カテゴリ変数を使った線形回帰
model_categorical = smf.ols('lifeExp ~ C(continent)', data=gapminder2007).fit()
print(model_categorical.summary())

# 回帰表の作成（簡略化バージョン）
coef_cat = model_categorical.params
std_err_cat = model_categorical.bse
t_vals_cat = model_categorical.tvalues
p_vals_cat = model_categorical.pvalues
conf_int_cat = model_categorical.conf_int()

# 回帰表のデータフレームを作成
terms = ['intercept'] + [f"continent: {c}" for c in model_categorical.params.index[1:]]
regression_table_cat = pd.DataFrame({
    'term': terms,
    'estimate': coef_cat,
    'std_error': std_err_cat,
    't_statistic': t_vals_cat,
    'p_value': p_vals_cat,
    'lower_ci': conf_int_cat[0],
    'upper_ci': conf_int_cat[1]
})

print(regression_table_cat)
```

この回帰表を解釈してみましょう。`term`列と`estimate`列の値に注目します：

1. `intercept`はアフリカ諸国の平均寿命の平均に対応し、約54.8年です。アフリカが「比較のためのベースライン」として選ばれたのは、5つの大陸の中でアルファベット順で最初にくるからです。
2. `continent: Americas`は南北アメリカの国々に対応し、値+18.8はアフリカに対する平均寿命の差です。つまり、南北アメリカの国々の平均寿命の平均は$54.8 + 18.8 = 73.6$年です。
3. `continent: Asia`はアジアの国々に対応し、値+15.9はアフリカに対する平均寿命の差です。つまり、アジアの国々の平均寿命の平均は$54.8 + 15.9 = 70.7$年です。
4. `continent: Europe`はヨーロッパの国々に対応し、値+22.8はアフリカに対する平均寿命の差です。つまり、ヨーロッパの国々の平均寿命の平均は$54.8 + 22.8 = 77.6$年です。
5. `continent: Oceania`はオセアニアの国々に対応し、値+25.9はアフリカに対する平均寿命の差です。つまり、オセアニアの国々の平均寿命の平均は$54.8 + 25.9 = 80.7$年です。

カテゴリ変数を使用した線形回帰モデルでは、$k$個の可能なカテゴリを持つカテゴリ的説明変数 $x$ を使用すると、回帰表は切片と $k - 1$ 個の「オフセット」を返します。この場合、$k = 5$大陸があるので、回帰モデルは比較のためのベースライングループであるアフリカに対応する切片と、南北アメリカ、アジア、ヨーロッパ、オセアニアに対応する $k - 1 = 4$ 個のオフセットを返します。

### 観測値/予測値と残差

カテゴリ変数を用いた回帰モデルの観測値、予測値、残差を計算しましょう：

```python
# 予測値と残差の計算
gapminder2007['lifeExp_hat'] = model_categorical.predict(gapminder2007)
gapminder2007['residual'] = gapminder2007['lifeExp'] - gapminder2007['lifeExp_hat']

# 国名を含めて最初の10行を表示
result_df = gapminder2007[['country', 'continent', 'lifeExp', 'lifeExp_hat', 'residual']].head(10)
print(result_df)

# 残差の箱ひげ図
plt.figure(figsize=(12, 6))
sns.boxplot(x='continent', y='residual', data=gapminder2007)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('大陸')
plt.ylabel('残差 (平均寿命 - 予測平均寿命)')
plt.title('大陸別の残差分布')
plt.grid(True, alpha=0.3)
plt.show()

# 残差が最も小さい5カ国
worst_5 = gapminder2007.sort_values('residual').head(5)[['country', 'continent', 'lifeExp', 'lifeExp_hat', 'residual']]
print("残差が最も小さい（最も負の）5カ国:")
print(worst_5)

# 残差が最も大きい5カ国
best_5 = gapminder2007.sort_values('residual', ascending=False).head(5)[['country', 'continent', 'lifeExp', 'lifeExp_hat', 'residual']]
print("\n残差が最も大きい（最も正の）5カ国:")
print(best_5)
```

この残差は、国の平均寿命とその国が属する大陸の平均平均寿命からの偏差として解釈できます。例えば、アフガニスタンの残差 $y - \widehat{y}$ の大きな負の値は、アフガニスタンの平均寿命がアジア諸国の平均平均寿命よりもかなり低いことを示しています。これは部分的に、その国が長年にわたって経験した多くの戦争によって説明できます。

## 5.3 関連トピック

### 相関は必ずしも因果関係ではない

この章を通して、回帰の傾き係数を解釈する際に慎重でした。常に説明変数 $x$ の目的変数 $y$ への「関連する」効果について議論しました。例えば、「`bty_avg`が1単位増加するごとに、*平均して*、`score`は0.067単位*関連して*増加します」という表現をしました。我々は「関連する」という用語を含めることで、*因果的な*主張をしていないことを特に注意しています。したがって、「美しさ」スコア`bty_avg`は教育`score`と正の相関がありますが、この研究がどのように実施されたかについてのさらなる情報がなければ、「美しさ」スコアの教育スコアへの直接的な因果効果についての主張はできません。

例えば、ある医師が靴を履いて寝た患者は、頭痛で目覚める傾向があることを医療記録から発見したとします。そこで医師は「靴を履いて寝ることは頭痛の原因である！」と宣言します。

しかし、誰かが靴を履いて寝ている場合、それはアルコールに酔っているからである可能性が高いです。さらに、飲酒量が多いとより多くの二日酔いが生じ、したがってより多くの頭痛につながります。ここでの飲酒量は、*交絡/潜在*変数として知られています。それは背景に「潜んで」おり、「靴を履いて寝ること」と「頭痛で目覚めること」の因果関係（もしあれば）を「交絡」させます。

以下のような*因果グラフ*でこれを要約できます：

- Y は*応答*変数です。ここでは「頭痛で目覚めること」です。
- X は*処置*変数で、我々が因果効果に関心を持っているものです。ここでは「靴を履いて寝ること」です。
- Z は*交絡*変数で、X と Y の両方に影響を与え、それによってそれらの関係を「交絡」させます。ここでは、交絡変数はアルコールです。

アルコールは、人々が靴を履いて寝る可能性を高めると同時に、頭痛で目覚める可能性も高めます。したがって、X と Y の関係に関する回帰モデルには、Z も説明変数として含める必要があります。言い換えれば、医師は前夜に誰が飲んでいたかを考慮する必要があります。

因果関係の確立は難しい問題であり、しばしば慎重に設計された実験や交絡変数の効果を制御するための方法が必要です。これらのアプローチはどちらも、可能な限りすべての交絡変数を考慮するか、その影響を無効にしようとします。これにより、研究者は関心のある関係のみに焦点を当てることができます：目的変数 Y と処置変数 X の関係です。

ニュース記事を読むときは、相関が必ずしも因果関係を意味するわけではないという罠に陥らないように注意してください。

### 最適な当てはめ直線

回帰直線は「最適な当てはめ」直線とも呼ばれます。しかし、「最適」とは何を意味するのでしょうか？回帰で「最適」を決定するために使用される基準を詳しく見てみましょう。

図5.4を思い出すと、「美しさ」スコアが $x = 7.333$ の教師について、*観測値* $y$ を円で、*予測値* $\widehat{y}$ を四角で、*残差* $y - \widehat{y}$ を矢印でマークしました。

回帰直線の「最適」を決定する基準は、*残差平方和*の最小化です。残差平方和は以下のように計算されます：

$$
\sum_{i=1}^{n}(y_i - \widehat{y}_i)^2
$$

回帰直線がすべての点に完全に当てはまる場合、残差平方和は0になります。これは、予測値 $\widehat{y}$ がすべてのケースで観測値 $y$ に等しく、したがって残差 $y-\widehat{y}$ がすべてのケースで0になるためです。

さらに、463点の雲を通して引くことができるすべての可能な直線のうち、回帰直線はこの値を最小化します。これは、微積分と線形代数を使って証明できる数学的に保証された事実です。そのため、線形回帰直線は*最適な当てはめ直線*や*最小二乗直線*とも呼ばれます。

## 結論

この章では、基本的な回帰分析の概念を紹介しました。単純線形回帰について学び、数値的な説明変数とカテゴリ的な説明変数の両方を扱う方法を見てきました。また、重要な統計的概念である相関係数についても学び、相関と因果関係の違いについて議論しました。

Pythonを使用して回帰分析を行う方法と、回帰モデルの結果を解釈する方法を示しました。また、残差分析や最小二乗法の考え方についても説明しました。

次の章では、複数の説明変数を持つ回帰モデルについて学びます。これにより、より洗練された強力なモデルを構築し、目的変数 $y$ をより適切に説明することができるようになります。

これで第5章「基本的回帰分析」の翻訳と、R言語のコードをPythonに変換する作業を完了しました。

この章では、データモデリングの基本概念と、単純線形回帰分析のアプローチについて学びました。特に以下の重要なポイントを説明しました：

1. **説明のためのモデリング**と**予測のためのモデリング**の違い
2. 数値的説明変数を持つ単純線形回帰の実装と解釈
3. カテゴリ的説明変数を持つ回帰モデルの実装と解釈
4. 回帰分析における相関と因果関係の違い
5. 「最適な当てはめ直線」の意味と最小二乗法の原理

また、観測値、予測値、残差などの重要な概念や、モデルの適合度を評価する方法についても説明しました。

Pythonコードは`statsmodels`、`pandas`、`matplotlib`、`seaborn`などのライブラリを使用して実装しており、Google Colabで実行可能です。データはGitHubから直接ダウンロードするようにしています。

作成したコードは、人々の教育評価スコアとその「美しさ」スコアの関係、そして異なる大陸間での平均寿命の違いという2つの異なる事例を分析することで、回帰分析の適用方法と解釈を具体的に示しています。

コードには詳細なコメントを付け、視覚化も多く含めることで、回帰分析の概念をより直感的に理解できるようにしました。

```python
# 必要なライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW
import scipy.stats as stats
from IPython.display import display, HTML

# Google Colabで実行する場合のデータダウンロード
!mkdir -p data
!wget -O data/evals.csv https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/evals.csv
!wget -O data/gapminder2007.csv https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/gapminder2007.csv

# データの読み込み
evals = pd.read_csv('data/evals.csv')

# この章で使用する変数のみを選択
evals_ch5 = evals[['ID', 'score', 'bty_avg', 'age']]

# ----- 1つの数値的説明変数 -----
# データの最初の数行を表示
print("UT Austin講師評価データの最初の数行:")
display(evals_ch5.head())

# データの構造を確認
print("\nデータの構造:")
print(evals_ch5.info())

# 要約統計量の計算
print("\n要約統計量:")
display(evals_ch5.describe())

# 相関係数の計算
correlation = evals_ch5[['score', 'bty_avg']].corr().iloc[0, 1]
print(f'\n相関係数: {correlation:.3f}')

# 散布図の作成
plt.figure(figsize=(10, 6))
plt.scatter(evals_ch5['bty_avg'], evals_ch5['score'], alpha=0.5)
plt.xlabel('美しさスコア')
plt.ylabel('教育スコア')
plt.title('教育スコアと美しさスコアの関係')

# 回帰直線の追加
x = evals_ch5['bty_avg']
y = evals_ch5['score']
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='blue')

plt.grid(True, alpha=0.3)
plt.show()

print(f'回帰直線の式: score = {b:.3f} + {m:.3f} * bty_avg')

# ジッタリングを適用した散布図
plt.figure(figsize=(10, 6))
plt.scatter(evals_ch5['bty_avg'] + np.random.normal(0, 0.05, len(evals_ch5)), 
            evals_ch5['score'] + np.random.normal(0, 0.05, len(evals_ch5)), 
            alpha=0.5)
plt.xlabel('美しさスコア (ジッタリング適用)')
plt.ylabel('教育スコア (ジッタリング適用)')
plt.title('教育スコアと美しさスコアの関係 (ジッタリング適用)')
plt.plot(x, m*x + b, color='blue')
plt.grid(True, alpha=0.3)
plt.show()

# statsmodelsを使用した線形回帰
model = smf.ols('score ~ bty_avg', data=evals_ch5).fit()
print("\n回帰モデルの要約:")
print(model.summary())

# 回帰表の出力（簡略化バージョン）
coef = model.params
std_err = model.bse
t_vals = model.tvalues
p_vals = model.pvalues
conf_int = model.conf_int()

regression_table = pd.DataFrame({
    'term': ['intercept', 'bty_avg'],
    'estimate': coef,
    'std_error': std_err,
    't_statistic': t_vals,
    'p_value': p_vals,
    'lower_ci': conf_int[0],
    'upper_ci': conf_int[1]
})

print("\n回帰表:")
display(regression_table)

# 予測値と残差の計算
evals_ch5['score_hat'] = model.predict(evals_ch5)
evals_ch5['residual'] = evals_ch5['score'] - evals_ch5['score_hat']

# 例示用に特定の講師のデータを表示
example_instructor = evals_ch5[evals_ch5['bty_avg'] == 7.333].iloc[0]
print(f"\n特定の講師の例:")
print(f"「美しさ」スコア: {example_instructor['bty_avg']}")
print(f"実際の教育スコア (y): {example_instructor['score']}")
print(f"予測された教育スコア (ŷ): {example_instructor['score_hat']:.3f}")
print(f"残差 (y - ŷ): {example_instructor['residual']:.3f}")

# 残差と予測値の関係を可視化
plt.figure(figsize=(10, 6))
plt.scatter(evals_ch5['score_hat'], evals_ch5['residual'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('予測値 (ŷ)')
plt.ylabel('残差 (y - ŷ)')
plt.title('残差プロット')
plt.grid(True, alpha=0.3)
plt.show()

# ヒストグラムで残差の分布を確認
plt.figure(figsize=(10, 6))
plt.hist(evals_ch5['residual'], bins=20, alpha=0.7, color='skyblue')
plt.xlabel('残差 (y - ŷ)')
plt.ylabel('頻度')
plt.title('残差のヒストグラム')
plt.grid(True, alpha=0.3)
plt.show()

# 残差平方和の計算
sum_squared_residuals = np.sum(evals_ch5['residual']**2)
print(f"\n残差平方和: {sum_squared_residuals:.3f}")

# オリジナルの回帰直線と任意の2つの直線を比較
plt.figure(figsize=(10, 6))
plt.scatter(evals_ch5['bty_avg'], evals_ch5['score'], alpha=0.5)

# オリジナルの回帰直線
x_range = np.linspace(min(evals_ch5['bty_avg']), max(evals_ch5['bty_avg']), 100)
plt.plot(x_range, b + m*x_range, color='blue', label=f'回帰直線: y = {b:.3f} + {m:.3f}x')

# 任意の直線1
m1, b1 = 0.04, 4.0  # 任意の傾きと切片
plt.plot(x_range, b1 + m1*x_range, color='red', linestyle='--', 
         label=f'任意の直線1: y = {b1:.3f} + {m1:.3f}x')

# 任意の直線2
m2, b2 = 0.1, 3.5  # 別の任意の傾きと切片
plt.plot(x_range, b2 + m2*x_range, color='green', linestyle='-.', 
         label=f'任意の直線2: y = {b2:.3f} + {m2:.3f}x')

plt.xlabel('美しさスコア')
plt.ylabel('教育スコア')
plt.title('オリジナルの回帰直線と任意の直線の比較')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# それぞれの直線に対する残差平方和を計算
def calculate_sse(slope, intercept):
    predicted = intercept + slope * evals_ch5['bty_avg']
    residuals = evals_ch5['score'] - predicted
    return np.sum(residuals**2)

sse_regression = calculate_sse(m, b)
sse_line1 = calculate_sse(m1, b1)
sse_line2 = calculate_sse(m2, b2)

print(f"\n回帰直線の残差平方和: {sse_regression:.3f}")
print(f"任意の直線1の残差平方和: {sse_line1:.3f}")
print(f"任意の直線2の残差平方和: {sse_line2:.3f}")

# ----- 1つのカテゴリ的説明変数 -----
# 2007年のGapminderデータを読み込む
print("\n\n=== カテゴリ変数を用いた回帰分析 ===\n")
gapminder2007 = pd.read_csv('data/gapminder2007.csv')

# データの最初の数行を表示
print("Gapminder 2007データの最初の数行:")
display(gapminder2007.head())

# 数値変数の要約統計量
print("\n要約統計量:")
display(gapminder2007.describe())

# continentごとの国の数を確認
continent_counts = gapminder2007['continent'].value_counts().reset_index()
continent_counts.columns = ['continent', 'count']
print("\n大陸ごとの国の数:")
display(continent_counts)

# 各大陸の平均寿命を計算
lifeExp_by_continent = gapminder2007.groupby('continent')['lifeExp'].agg(['mean', 'median', 'std']).reset_index()
print("\n大陸ごとの平均寿命統計:")
display(lifeExp_by_continent)

# 平均寿命のヒストグラム
plt.figure(figsize=(10, 6))
plt.hist(gapminder2007['lifeExp'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('平均寿命')
plt.ylabel('国の数')
plt.title('全世界の平均寿命分布（2007年）')
plt.grid(True, alpha=0.3)
plt.show()

# 大陸ごとのヒストグラム
plt.figure(figsize=(12, 8))
for i, continent in enumerate(gapminder2007['continent'].unique(), 1):
    plt.subplot(2, 3, i)
    subset = gapminder2007[gapminder2007['continent'] == continent]
    plt.hist(subset['lifeExp'], bins=10, color='skyblue', edgecolor='black')
    plt.title(f'{continent}')
    plt.xlabel('平均寿命')
    plt.ylabel('国の数')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 箱ひげ図
plt.figure(figsize=(12, 6))
sns.boxplot(x='continent', y='lifeExp', data=gapminder2007)
plt.xlabel('大陸')
plt.ylabel('平均寿命')
plt.title('大陸別の平均寿命（2007年）')
plt.grid(True, alpha=0.3)
plt.show()

# カテゴリ変数を使った線形回帰
model_categorical = smf.ols('lifeExp ~ C(continent)', data=gapminder2007).fit()
print("\n大陸を説明変数とした回帰モデルの要約:")
print(model_categorical.summary())

# 回帰表の作成
coef_cat = model_categorical.params
std_err_cat = model_categorical.bse
t_vals_cat = model_categorical.tvalues
p_vals_cat = model_categorical.pvalues
conf_int_cat = model_categorical.conf_int()

# インデックス名を取得
index_names = coef_cat.index.tolist()
# 最初の要素（切片）とそれ以外を分ける
terms = ['intercept']
for i in range(1, len(index_names)):
    # C(continent)[T.XXX]形式の文字列からXXXを抽出
    continent_name = index_names[i].split('[T.')[1].rstrip(']')
    terms.append(f"continent: {continent_name}")

regression_table_cat = pd.DataFrame({
    'term': terms,
    'estimate': coef_cat.values,
    'std_error': std_err_cat.values,
    't_statistic': t_vals_cat.values,
    'p_value': p_vals_cat.values,
    'lower_ci': conf_int_cat[0].values,
    'upper_ci': conf_int_cat[1].values
})

print("\n回帰表:")
display(regression_table_cat)

# 予測値と残差の計算
gapminder2007['lifeExp_hat'] = model_categorical.predict(gapminder2007)
gapminder2007['residual'] = gapminder2007['lifeExp'] - gapminder2007['lifeExp_hat']

# 国名を含めて最初の10行を表示
result_df = gapminder2007[['country', 'continent', 'lifeExp', 'lifeExp_hat', 'residual']].head(10)
print("\n観測値、予測値、残差の最初の10行:")
display(result_df)

# 残差の箱ひげ図
plt.figure(figsize=(12, 6))
sns.boxplot(x='continent', y='residual', data=gapminder2007)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('大陸')
plt.ylabel('残差 (平均寿命 - 予測平均寿命)')
plt.title('大陸別の残差分布')
plt.grid(True, alpha=0.3)
plt.show()

# 残差が最も小さい5カ国
worst_5 = gapminder2007.sort_values('residual').head(5)[['country', 'continent', 'lifeExp', 'lifeExp_hat', 'residual']]
print("\n残差が最も小さい（最も負の）5カ国:")
display(worst_5)

# 残差が最も大きい5カ国
best_5 = gapminder2007.sort_values('residual', ascending=False).head(5)[['country', 'continent', 'lifeExp', 'lifeExp_hat', 'residual']]
print("\n残差が最も大きい（最も正の）5カ国:")
display(best_5)
```