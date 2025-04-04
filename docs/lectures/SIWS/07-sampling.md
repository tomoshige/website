# 第7章: サンプリング

この章では、*サンプリング*について学ぶことで、統計的推論に関する本書の第三部を始めます。サンプリングの概念は、第8章と第9章で扱う信頼区間と仮説検定の基礎となります。データサイエンスの部分で学んだツール、特にデータの可視化とデータの整形も、理解を深める上で重要な役割を果たすことがわかるでしょう。前述のように、本書の概念はすべて、「データで物語を語る」ことができるように集約されています。

## 必要なパッケージ

まずは、この章で必要なパッケージをインポートしましょう。

```python
# Google Colabで実行するためのセットアップ
!pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats

# プロットの見た目を設定
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12
```

## サンプリングボウルの活動

実践的な活動から始めましょう。

### このボウルの球の何割が赤いですか？

赤と白の球が入ったボウルを考えます。球はすべて同じサイズです。この本のオリジナル画像では、赤と白の球が入ったボウルが示されていて、事前に混ぜられているため、赤と白の球の分布にはパターンがありません。

それでは、このボウルの球のうち、赤い球の割合はどれくらいでしょうか？

この質問に答える一つの方法は、すべての球を個別に取り出して、赤い球と白い球の数を数え、赤い球の数を球の総数で割ることです。しかし、これは長く退屈な作業になるでしょう。

### シャベルを一度使う

すべて数える代わりに、シャベルをボウルに入れて50個の球を取り出してみましょう。

50個の球のうち17個が赤であれば、取り出した球の34%が赤ということになります。これを、ボウル全体の赤い球の割合の推定値と考えることができます。すべての球を数えるよりもはるかに少ない時間と労力で推測できました。

しかし、もしこの活動を最初からやり直したらどうでしょうか？つまり、50個の球をボウルに戻して、もう一度始めるとします。また正確に17個の赤い球が出てくるでしょうか？言い換えると、赤い球の割合の推測値は再び正確に34%になるでしょうか？もしかしたら？

この活動を何度か繰り返すとどうなるでしょうか？毎回正確に17個の赤い球が出てくるでしょうか？つまり、ボウルの赤い球の割合の推測値は毎回正確に34%になるでしょうか？おそらくそうはならないでしょう。33組の友人グループに手伝ってもらって、値がどのように変わるかを理解してみましょう。

### シャベルを33回使う

33組の友人グループには、それぞれ次のことをしてもらいます：

- シャベルを使って50個の球を取り出す
- 赤い球の数を数え、赤い球の割合を計算する
- 球をボウルに戻す
- 前のグループの結果が次のグループに影響しないように、ボウルの中身を少し混ぜる

各グループがサンプルから得た赤い球の割合を記録し、ヒストグラムを作成して視覚化します。

Pythonでこのデータを再現しましょう。まず、ModernDiveパッケージの`tactile_prop_red`データセットを再現します：

```python
# GitHubからデータをダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/tactile_prop_red.csv"
tactile_prop_red = pd.read_csv(url)

# データの確認
tactile_prop_red.head(10)
```

このデータを視覚化して、33グループの赤い球の割合の分布を見てみましょう：

```python
plt.figure(figsize=(10, 6))
plt.hist(tactile_prop_red['prop_red'], bins=np.arange(0.2, 0.55, 0.05), 
         edgecolor='white', color='skyblue')
plt.xlabel('50個の球のうち赤い球の割合')
plt.ylabel('頻度')
plt.title('33の割合の分布')
plt.xticks(np.arange(0.2, 0.55, 0.05))
plt.tight_layout()
plt.show()
```

### 私たちは何をしたのか？

この活動で実証したのは、**サンプリング**という統計概念です。ボウルの球のうち赤い球の割合を知りたいと思っていました。ボウルには大量の球があるため、赤と白の球を全て数えるのは時間がかかります。そこで、シャベルを使って50個の球の**サンプル**を抽出し、**推定**を行いました。このサンプルの50個の球から、ボウル全体の赤い球の割合を34%と推定しました。

さらに、球を取り出す前にボウルを混ぜたので、サンプルはランダムに抽出されました。サンプルがランダムに抽出されたため、サンプルはそれぞれ異なっていました。サンプルがそれぞれ異なっていたため、図に示されるような赤い球の割合のばらつきが生じました。これは**サンプリングのばらつき**の概念として知られています。

このサンプリング活動の目的は、サンプリングに関連する2つの重要な概念について理解を深めることでした：

1. サンプリングの変動の影響を理解すること
2. サンプリングの変動にサンプルサイズが与える影響を理解すること

## 仮想サンプリング

前のセクションでは、物理的なボウルと物理的なシャベルを使って、手作業でサンプリング活動を行いました。このセクションでは、コンピュータを使用して、この実際のサンプリング活動を仮想的に模倣します。

### 仮想シャベルを一度使う

まず、実際のボウルの仮想版を作成する必要があります。ModernDiveパッケージの`bowl`データフレームを再現しましょう：

```python
# ボウルデータの作成
# オリジナルのボウルには2400個の球が含まれていて、そのうち900個が赤(37.5%)
np.random.seed(76)
bowl = pd.DataFrame({
    'ball_ID': range(1, 2401),
    'color': np.random.choice(['red', 'white'], size=2400, p=[0.375, 0.625])
})

# データの確認
bowl.head()
```

次に、シャベルを使って50個の球をサンプリングする仮想バージョンを作成します：

```python
# 仮想シャベルの作成 - 50個の球をランダムに抽出
np.random.seed(76)
virtual_shovel = bowl.sample(n=50, replace=False).reset_index(drop=True)

# レプリケートを追加（後で便利）
virtual_shovel['replicate'] = 1

# 結果を確認
virtual_shovel.head()
```

サンプル内の赤い球の割合を計算しましょう：

```python
# 赤い球かどうかを示す新しい列を作成
virtual_shovel['is_red'] = (virtual_shovel['color'] == 'red')

# 赤い球の数を集計
num_red = virtual_shovel['is_red'].sum()

# 赤い球の割合を計算
prop_red = num_red / 50

print(f"50個の球のうち、{num_red}個が赤です。")
print(f"赤い球の割合: {prop_red:.3f} ({prop_red*100:.1f}%)")
```

より簡潔なコードで同じ計算を行うこともできます：

```python
# より簡潔なコード
summary = pd.DataFrame({
    'num_red': [virtual_shovel['color'].eq('red').sum()],
    'prop_red': [virtual_shovel['color'].eq('red').sum() / 50]
})
summary
```

### 仮想シャベルを33回使う

実際のサンプリング活動では、33グループの生徒にそれぞれシャベルを使ってもらい、33のサンプル（各サンプルは50個の球）を得ました。仮想的にこの複数回のサンプリングを実行してみましょう：

```python
# 33回のレプリケートでサンプリングを実行
np.random.seed(76)
virtual_samples = pd.DataFrame()

for i in range(1, 34):
    sample = bowl.sample(n=50, replace=False).copy()
    sample['replicate'] = i
    virtual_samples = pd.concat([virtual_samples, sample])

# 最初の行を確認
virtual_samples.head()
```

次に、33のサンプルの各々について赤い球の割合を計算します：

```python
# 各レプリケートの赤い球の割合を計算
virtual_prop_red = virtual_samples.groupby('replicate').apply(
    lambda x: pd.Series({
        'red': (x['color'] == 'red').sum(),
        'prop_red': (x['color'] == 'red').sum() / 50
    })
).reset_index()

# 結果の最初の10行を表示
virtual_prop_red.head(10)
```

実際のサンプルと同様に、仮想サンプルにも結果のばらつきがあります。これを視覚化してみましょう：

```python
plt.figure(figsize=(10, 6))
plt.hist(virtual_prop_red['prop_red'], bins=np.arange(0.2, 0.55, 0.05), 
         edgecolor='white', color='skyblue')
plt.xlabel('50個の球のうち赤い球の割合')
plt.ylabel('頻度')
plt.title('33の仮想サンプルにおける赤い球の割合の分布')
plt.xticks(np.arange(0.2, 0.55, 0.05))
plt.tight_layout()
plt.show()
```

実際のサンプルと仮想サンプルを比較してみましょう：

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 仮想サンプリング
axes[0].hist(virtual_prop_red['prop_red'], bins=np.arange(0.2, 0.55, 0.05), 
             edgecolor='white', color='skyblue')
axes[0].set_title('仮想サンプリング')
axes[0].set_xlabel('50個の球のうち赤い球の割合')
axes[0].set_ylabel('頻度')
axes[0].set_xticks(np.arange(0.2, 0.55, 0.05))

# 実際のサンプリング
axes[1].hist(tactile_prop_red['prop_red'], bins=np.arange(0.2, 0.55, 0.05), 
             edgecolor='white', color='skyblue')
axes[1].set_title('実際のサンプリング')
axes[1].set_xlabel('50個の球のうち赤い球の割合')
axes[1].set_ylabel('頻度')
axes[1].set_xticks(np.arange(0.2, 0.55, 0.05))

plt.tight_layout()
plt.show()
```

両方のヒストグラムは、中心と変動が似ていますが、完全に同じではありません。これらのわずかな違いも、ランダムなサンプリングの変動によるものです。また、どちらの分布もある程度ベル型であることに注目してください。

### 仮想シャベルを1000回使う

サンプリングの変動の効果をさらに研究するために、33回ではなく、もっと多くのサンプル、例えば1000回のサンプリングを考えてみましょう。手作業で1000個のサンプルを取るのは大変なので、コンピュータを使用して仮想サンプリングを行います：

```python
# 1000回のレプリケートでサンプリングを実行
np.random.seed(76)
virtual_samples = pd.DataFrame()

for i in range(1, 1001):
    sample = bowl.sample(n=50, replace=False).copy()
    sample['replicate'] = i
    virtual_samples = pd.concat([virtual_samples, sample])

# 各レプリケートの赤い球の割合を計算
virtual_prop_red = virtual_samples.groupby('replicate').apply(
    lambda x: pd.Series({
        'red': (x['color'] == 'red').sum(),
        'prop_red': (x['color'] == 'red').sum() / 50
    })
).reset_index()

# 1000の割合の分布を視覚化
plt.figure(figsize=(10, 6))
plt.hist(virtual_prop_red['prop_red'], bins=np.arange(0.15, 0.65, 0.05), 
         edgecolor='white', color='skyblue')
plt.xlabel('50個の球のうち赤い球の割合')
plt.ylabel('頻度')
plt.title('1000の割合の分布')
plt.xticks(np.arange(0.15, 0.65, 0.05))
plt.tight_layout()
plt.show()
```

再び、35%から40%の間の赤い球の割合が最も頻繁に発生していることがわかります。時折、20%から25%の間や55%から60%の間の割合も得られますが、それらはまれです。さらに、より対称的で滑らかなベル型の分布が得られました。この分布は実際、正規分布によく近似されます。

### 異なるシャベルの使用

異なるサイズのシャベル（25、50、100個のスロットを持つシャベル）を使って、サンプルサイズがサンプリングの変動にどのような影響を与えるかを見てみましょう：

```python
# 3つの異なるサンプルサイズ（25、50、100）で1000回ずつサンプリング
sample_sizes = [25, 50, 100]
results = {}

for n in sample_sizes:
    np.random.seed(76)
    virtual_props = []
    
    # 1000回のサンプリングを実行
    for i in range(1000):
        sample = bowl.sample(n=n, replace=False)
        prop = (sample['color'] == 'red').mean()
        virtual_props.append(prop)
    
    results[n] = virtual_props

# 結果を視覚化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, n in enumerate(sample_sizes):
    axes[i].hist(results[n], bins=np.arange(0.15, 0.65, 0.05), 
                 edgecolor='white', color='skyblue')
    axes[i].set_title(f'n = {n}')
    axes[i].set_xlabel('シャベルの球のうち赤い球の割合')
    axes[i].set_xlim(0.15, 0.65)
    axes[i].set_xticks(np.arange(0.2, 0.6, 0.1))

plt.suptitle('3つの異なるシャベルサイズにおける赤い球の割合の分布の比較')
plt.tight_layout()
plt.show()
```

サンプルサイズが大きくなるにつれて、赤い球の割合の1000のレプリケートの変動が小さくなることがわかります。つまり、サンプルサイズが25から50、100に増えるにつれて、サンプリングの変動による違いが少なくなり、同じ値の周りにより密集します。目視では、3つのヒストグラムはいずれも約40%を中心に分布しているようです。

3つの1000の値の集合のばらつきを、標準偏差を使って数値的に明示することができます：

```python
# 3つのサンプルサイズの標準偏差を計算
std_devs = {n: np.std(results[n], ddof=1) for n in sample_sizes}

# 表形式で表示
std_dev_df = pd.DataFrame({
    'シャベルのスロット数': sample_sizes,
    '赤い球の割合の標準偏差': [std_devs[n] for n in sample_sizes]
})

std_dev_df
```

図に示されているように、サンプルサイズが大きくなるほど、変動が小さくなります。つまり、1000の赤い球の割合の値のばらつきが少なくなります。サンプルサイズが大きくなるほど、ボウルの赤い球の真の割合の推測がより正確になります。

## サンプリングの枠組み

触覚的および仮想的サンプリング活動の両方で、推定の目的でサンプリングを使用しました。ボウルの赤い球の割合を*推定*するためにサンプルを抽出しました。すべての球を数える時間のかかる方法の代わりに、サンプリングを使用しました。

### 用語と表記

サンプリングに関連するさまざまな概念に言葉とラベルを付けるために、いくつかの用語と数学的な表記を紹介します。

**母集団**に関連する最初の用語と表記は次のとおりです：

1. **母集団**は、関心のある個体または観測値の集合です。これは一般的に**研究母集団**とも呼ばれます。母集団のサイズは大文字の$N$で数学的に表します。
2. **母集団パラメータ**は、母集団に関する数値的な要約で、未知ですが知りたいと思うものです。例えば、カナダ人全体の平均身長のような平均値である場合、関心のある母集団パラメータは*母平均*です。
3. **全数調査**は、母集団内のすべての$N$個体を徹底的に列挙または計数することです。これは、母集団パラメータの値を*正確に*計算するために行います。注目すべきは、母集団の個体数$N$が増えるにつれて、全数調査がより高価になる（時間、労力、お金の面で）ことです。

私たちのサンプリング活動では、**母集団**は$N$ = 2400個の同じサイズの赤と白の球のボウルのコレクションです。

**母集団パラメータ**は、ボウルの球のうち赤い球の割合です。母集団の値の割合に関心がある場合、母集団パラメータには特定の名前があります：*母集団比率*。母集団比率を文字$p$で表します。

この母集団比率$p$を正確に計算するためには、まず$N$ = 2400個すべてに対して**全数調査**を行い、赤い数を数える必要があります。その後、この数を2400で割って赤い球の割合を求めます。

実際に「仮想」全数調査を行ってみましょう：

```python
# 仮想全数調査 - ボウル全体の赤い球の数を数える
red_count = (bowl['color'] == 'red').sum()
total_count = len(bowl)
population_proportion = red_count / total_count

print(f"ボウル全体には{red_count}個の赤い球があります。")
print(f"ボウル全体の球の数は{total_count}個です。")
print(f"母集団比率 p = {population_proportion:.3f} = {population_proportion*100:.1f}%")
```

この全数調査から、母集団パラメータの値がわかりました：この場合、母集団比率$p$は0.375です。

**サンプル**に関連する2番目の用語と表記セットは次のとおりです：

1. **サンプリング**は、母集団からサンプルを収集する行為であり、一般に全数調査を実行できない場合にのみ行います。サンプルサイズは小文字の$n$で数学的に表し、母集団のサイズを表す大文字の$N$とは対照的です。通常、サンプルサイズ$n$は母集団サイズ$N$よりもはるかに小さいです。したがって、サンプリングは全数調査を実行するよりもはるかに安価な代替手段です。
2. **点推定値**（**標本統計量**とも呼ばれる）は、サンプルから計算される要約統計量であり、未知の母集団パラメータを*推定*するものです。

以前、50個のスロットを持つシャベルを使用して$n$ = 50のサンプルを抽出する**サンプリング**を行いました。

サンプルの赤い球の割合（サンプル比率）$\hat{p}$は、ボウルの赤い球の母集団比率$p$の**点推定値**です。一般的に、推定値には「ハット」記号を使用することが統計学の慣例です。

**サンプリング方法論**に関連する3番目の用語セットは、サンプルを収集するために使用される方法です。ここと本書の残りの部分を通じて、サンプルを収集する*方法*がその品質に直接影響することがわかるでしょう。

1. サンプルが母集団を「大まかに反映している」場合、そのサンプルは**代表的**であると言われます。つまり、サンプルの特性が母集団の特性を「よく」表現している場合です。
2. サンプルに基づく結果を母集団に一般化できる場合、そのサンプルは**一般化可能**であると言います。言い換えると、サンプルを使用して母集団について「良い」推測ができる場合です。
3. 母集団内の特定の個体が他の個体よりもサンプルに含まれる確率が高い場合、サンプリング手順は**偏っている**と言います。母集団内のすべての個体がサンプルに含まれる確率が等しい場合、サンプリング手順は**偏りがない**と言います。

シャベルを使用して抽出した$n$個の球のサンプルが、その内容がボウルの内容を「大まかに類似している」場合、母集団を**代表している**と言います。その場合、シャベルの球のうち赤い球の割合は、ボウルの$N$ = 2400個の球のうち赤い球の割合に**一般化**できます。または、別の言い方をすれば、$\hat{p}$は$p$の「良い推測」です。 

サンプリングの目標に関連する4番目と最後の用語および表記セットは次のとおりです：

1. サンプルが偏りがなく母集団を代表していることを確実にする1つの方法は、**ランダムサンプリング**を使用することです。
2. **推論**は、何か未知のものについて「推測する」行為です。**統計的推論**は、サンプルを使用して母集団について推測する行為です。

4つの用語および表記セットをすべてまとめてみましょう：

* $n$ = 50個の球をランダムに抽出したので、シャベルを使用する前にすべての同じサイズの球を混ぜたので、
* シャベルの内容はボウルの内容を*偏りなく*代表しているので、
* シャベルに基づく結果はボウルに*一般化*できるので、
* シャベルの$n$ = 50個の球のうち赤い球の標本比率$\hat{p}$はボウルの$N$ = 2400個の球のうち赤い球の母集団比率$p$の「良い推測」であるので、
* ボウルの2400個の球の*全数調査*を行う代わりに、シャベルからのサンプルを使用してボウルについて**推論**できます。

あなたが行ってきたのは**統計的推論**です。これは統計学の中で最も重要な概念の1つです。それほど重要なので、本書のタイトルにもこの用語を含めました：「データサイエンスを通じた統計的推論」。より一般的に言えば、

* サイズ$n$のサンプルのサンプリングが*ランダム*に行われている場合、
* サンプルはサイズ$N$の母集団に対して*偏りがなく*代表的であるので、
* このサンプルに基づく結果は母集団に*一般化*できるので、
* 点推定値は未知の母集団パラメータの「良い推測」であるので、
* 全数調査を行う代わりに、サンプリングを使用して母集団について*推論*できます。

### 統計的定義

サンプルサイズ$n$ = 25、$n$ = 50、$n$ = 100のそれぞれで1000回反復した仮想サンプルの結果を考え直してみましょう：

```python
# データをまとめてデータフレームに変換
all_results = pd.DataFrame({
    'n': np.repeat([25, 50, 100], 1000),
    'prop_red': np.concatenate([results[25], results[50], results[100]])
})

# 3つのサンプルサイズの分布を並べて表示
plt.figure(figsize=(15, 6))
sns.histplot(data=all_results, x='prop_red', hue='n', multiple='dodge', 
             bins=np.arange(0.15, 0.65, 0.05), edgecolor='white')
plt.xlabel('標本比率 $\\hat{p}$')
plt.ylabel('頻度')
plt.title('標本比率 $\\hat{p}$ の3つのサンプリング分布')
plt.xticks(np.arange(0.2, 0.6, 0.1))
plt.tight_layout()
plt.show()
```

これらのタイプの分布には特別な名前があります：**点推定値のサンプリング分布**です。その視覚化は、任意の点推定値（この場合、標本比率$\hat{p}$）の分布に対するサンプリング変動の影響を表示します。これらのサンプリング分布を使用すると、特定のサンプルサイズ$n$に対して、どのような値が一般的に期待できるかについて記述することができます。

すべての3つのサンプリング分布の中心を観察すると、すべてがおよそ0.4 = 40%を中心にしています。また、25個のスロットを持つシャベルを使用する場合、赤い球の標本比率が0.2 = 20%になる可能性がある程度ありますが、100個のスロットを持つシャベルを使用する場合、20%の割合はほとんど観察されないことがわかります。また、サンプリング変動に対するサンプルサイズの影響も観察できます。サンプルサイズ$n$が25から50、100に増えるにつれて、サンプリング分布の変動が減少し、値が同じ中心（約40%）の周りにより密集します。

標準偏差を使用して、サンプルサイズごとの分布の変動を比較しましょう：

```python
# 3つのサンプルサイズの標準偏差を計算
std_devs = all_results.groupby('n')['prop_red'].std().reset_index()
std_devs.columns = ['サンプルサイズ (n)', '標本比率 $\\hat{p}$ の標準誤差']
std_devs
```

サンプルサイズが大きくなるにつれて、標本比率の標準偏差が小さくなります。この種の標準偏差には別の特別な名前があります：**点推定値の標準誤差**です。標準誤差は、推定値に対するサンプリング変動の影響を定量化します。つまり、赤い球のシャベルの割合が1つのサンプルから別のサンプル、さらに別のサンプルへとどの程度*変動する*かを定量化します。一般的な規則として、サンプルサイズが大きくなると、標準誤差は小さくなります。

物語の教訓をまとめましょう。サンプルがランダムに生成された場合、結果として得られる点推定値は、真の未知の母集団パラメータの「良い推測」になります。私たちのサンプリング活動では、シャベルで球を取り出す前にボウルを確実に混ぜたので、赤だったシャベルの球の結果として得られる標本比率$\hat{p}$は、赤いボウルの球の母集団比率$p$の「良い推測」でした。

しかし、点推定値が「良い推測」であるとはどういう意味でしょうか？時には、母集団パラメータの真の値よりも小さい推定値を得ることがあり、また他の場合には、より大きい推定値を得ることがあります。これはサンプリング変動によるものです。しかし、このサンプリング変動にもかかわらず、推定値は「平均して」正確であり、したがって真の値を中心に分布します。これは、サンプリングがランダムに行われ、したがって偏りのない方法で行われたためです。

全数調査から、ボウルの$N$ = 2400個の球の母集団比率$p$の値は赤/2400 = 0.375 = 37.5%であることがわかりました。3つのサンプリング分布を再表示し、真の母集団比率$p$ = 37.5%を垂直線で示してみましょう：

```python
# 3つのサンプリング分布と真の母集団比率
plt.figure(figsize=(15, 6))
sns.histplot(data=all_results, x='prop_red', hue='n', multiple='dodge', 
             bins=np.arange(0.15, 0.65, 0.05), edgecolor='white')
plt.axvline(x=0.375, color='red', linestyle='--', linewidth=2)
plt.xlabel('標本比率 $\\hat{p}$')
plt.ylabel('頻度')
plt.title('標本比率 $\\hat{p}$ の3つのサンプリング分布（真の母集団比率$p$を垂直線で表示）')
plt.xticks(np.arange(0.2, 0.6, 0.1))
plt.tight_layout()
plt.show()
```

3つのサンプリング分布すべてにおいて、標本比率$\hat{p}$にある程度のエラーがありますが、平均して$\hat{p}$は真の母集団比率$p$を中心にしていることがわかります。

このセクションでは、サンプルサイズ$n$が大きくなるにつれて、点推定値の変動が少なくなり、真の母集団パラメータの周りにより集中することも見ました。この変動は減少する*標準誤差*によって定量化されます。つまり、点推定値の典型的なエラーは減少します。サンプリング活動では、サンプルサイズが大きくなるにつれて、標本比率$\hat{p}$の変動が小さくなりました。

ランダムサンプリングは点推定値の*正確さ*を確保し、一方、大きなサンプルサイズは点推定値の*精度*を確保します。「正確さ」と「精度」という用語は同じことを意味するように聞こえるかもしれませんが、微妙な違いがあります。正確さは推定値が「的を射ている」程度を表し、精度は推定値の「一貫性」を表します。

```python
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# 的の画像を作成（単純な円でシミュレート）
def create_target_image(accuracy, precision):
    fig, ax = plt.subplots(figsize=(3, 3))
    # 的を描画
    circle1 = plt.Circle((0.5, 0.5), 0.4, color='lightgray')
    circle2 = plt.Circle((0.5, 0.5), 0.3, color='gray')
    circle3 = plt.Circle((0.5, 0.5), 0.2, color='darkgray')
    circle4 = plt.Circle((0.5, 0.5), 0.1, color='black')
    
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    
    # 弾痕をシミュレート
    np.random.seed(42)
    if accuracy == 'high' and precision == 'high':
        # 中心近くで密集
        points_x = np.random.normal(0.5, 0.05, 10)
        points_y = np.random.normal(0.5, 0.05, 10)
    elif accuracy == 'high' and precision == 'low':
        # 中心近くでばらつき
        points_x = np.random.normal(0.5, 0.15, 10)
        points_y = np.random.normal(0.5, 0.15, 10)
    elif accuracy == 'low' and precision == 'high':
        # 中心から外れて密集
        points_x = np.random.normal(0.7, 0.05, 10)
        points_y = np.random.normal(0.7, 0.05, 10)
    else:  # low accuracy, low precision
        # 中心から外れてばらつき
        points_x = np.random.normal(0.7, 0.15, 10)
        points_y = np.random.normal(0.7, 0.15, 10)
    
    ax.scatter(points_x, points_y, color='red', s=50)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 画像として保存
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

# 2x2グリッドでの正確さと精度の比較
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

titles = [
    ['高い正確さ, 高い精度', '高い正確さ, 低い精度'],
    ['低い正確さ, 高い精度', '低い正確さ, 低い精度']
]

accuracy_values = ['high', 'high', 'low', 'low']
precision_values = ['high', 'low', 'high', 'low']

for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        img = create_target_image(accuracy_values[idx], precision_values[idx])
        axes[i, j].imshow(img)
        axes[i, j].set_title(titles[i][j])
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()
```

実際の状況では、1000の異なるサンプルを取るのではなく、1つだけ取ります。では、1000の異なるサンプルを取る練習の目的は何だったのでしょうか？これらのシミュレーションを使用して、次のことを研究しました：

1. 推定値に対する*サンプリング変動*の影響
2. *サンプリング変動*に対するサンプルサイズの影響

## ケーススタディ：世論調査

ボウル活動よりも現実的なサンプリングシナリオに切り替えましょう：世論調査です。実際には、調査者は前のサンプリング活動のように1000回の繰り返しサンプルを取るのではなく、できるだけ大きな*単一のサンプル*を取ります。

2013年12月4日、米国のNational Public Radio（NPR）は、「世論調査：若いアメリカ人の間でのオバマ支持が侵食されている」というタイトルの記事で、18〜29歳の若いアメリカ人の間でのオバマ大統領の支持率に関する世論調査について報じました。この調査はハーバード大学のケネディスクールの政治研究所によって実施されました。記事からの引用：

> 2008年と2012年に大多数で彼に投票した後、若いアメリカ人はオバマ大統領に対して冷めています。
> 
> ハーバード大学政治研究所の新しい世論調査によると、ミレニアル世代（18〜29歳の成人）のわずか41%がオバマの職務遂行を支持しており、これはこのグループの中で過去最低であり、4月から11ポイント下落しています。

この記事の実際の世論調査の要素と、これまでに学んだ用語、表記、定義を使用して、セクション7.1と7.2の「触覚的」および「仮想的」ボウル活動と結びつけてみましょう。ボウルでのサンプリング活動は、調査者が実生活で行おうとしていることの理想化されたバージョンであることがわかります。

まず、関心のある$N$個体または観測値の **（研究）母集団** は何ですか？

* ボウル：$N$ = 2400個の同じサイズの赤と白の球
* オバマ調査：$N$ = ? 18〜29歳の若いアメリカ人

第二に、**母集団パラメータ**は何ですか？

* ボウル：ボウルの*すべての*球のうち赤い球の母集団比率$p$
* オバマ調査：オバマの職務遂行を支持する*すべての*若いアメリカ人の母集団比率$p$

第三に、**全数調査**はどのようなものですか？

* ボウル：$N$ = 2400個すべての球を手作業で調べ、赤い球の母集団比率$p$を正確に計算します
* オバマ調査：すべての$N$若いアメリカ人を見つけ、彼らすべてにオバマの職務遂行を支持するかどうかを尋ねます。この場合、母集団サイズ$N$が何であるかさえわかりません！

第四に、サイズ$n$のサンプルを得るために**サンプリング**をどのように実行しますか？

* ボウル：$n$個のスロットを持つシャベルを使用します
* オバマ調査：一つの方法は、すべての若いアメリカ人の電話番号のリストを取得し、$n$個の電話番号を選択することです。この調査の場合、このサンプルサイズは$n = 2089$人の若いアメリカ人でした。

第五に、未知の母集団パラメータの**点推定値**または**標本統計量**は何ですか？

* ボウル：シャベルの中の球のうち赤い球の標本比率$\hat{p}$
* オバマ調査：サンプル内の若いアメリカ人のうちオバマの職務遂行を支持する標本比率$\hat{p}$。この調査の場合、$\hat{p} = 0.41 = 41\%$で、記事の2段落目に引用されているパーセンテージです。

第六に、サンプリング手順は**代表的**ですか？

* ボウル：シャベルの内容はボウルの内容を代表していますか？サンプリング前にボウルを混ぜたので、代表的であると確信できます。
* オバマ調査：$n = 2089$人の若いアメリカ人のサンプルは、18〜29歳の*すべての*若いアメリカ人を代表していますか？これはサンプリングがランダムかどうかによります。

第七に、サンプルはより大きな母集団に**一般化可能**ですか？

* ボウル：シャベルの球のうち赤い球の標本比率$\hat{p}$は、ボウルの球のうち赤い球の母集団比率$p$の「良い推測」ですか？サンプルが代表的であることを考えると、答えはイエスです。
* オバマ調査：若いアメリカ人のサンプルのうちオバマを支持した標本比率$\hat{p}$ = 0.41は、2013年のこの時点ですべての若いアメリカ人のうちオバマを支持した母集団比率$p$の「良い推測」ですか？つまり、2013年の調査時点で*すべての*若いアメリカ人の約41%がオバマを支持していたと自信を持って言えますか？繰り返しになりますが、これはサンプリングがランダムかどうかによります。

第八に、サンプリング手順は**偏りがない**ですか？言い換えると、すべての観測値がサンプルに含まれる確率は等しいですか？

* ボウル：各球が同じサイズで、シャベルを使用する前にボウルを混ぜたので、各球がサンプルに含まれる確率は等しく、したがってサンプリングには偏りがありませんでした。
* オバマ調査：すべての若いアメリカ人がこの調査に代表される平等なチャンスを持っていましたか？繰り返しになりますが、これはサンプリングがランダムかどうかによります。

第九に最後に、サンプリングは**ランダム**に行われましたか？

* ボウル：サンプリング前にボウルを十分に混ぜた限り、サンプルはランダムでした。
* オバマ調査：サンプルはランダムに実施されましたか？ハーバード大学のケネディスクールの政治研究所が使用した*サンプリング方法論*を知らなければ、この質問に答えることはできません。

つまり、ハーバード大学のケネディスクールの政治研究所による調査は、シャベルを使用してボウルから球をサンプリングする*一例*と考えることができます。さらに、別の調査会社がほぼ同じ時期に若いアメリカ人を対象に同様の調査を実施した場合、41%とは異なる推定値を得る可能性があります。これは*サンプリング変動*によるものです。

実際の世論調査のサンプリングが本当にランダムであることは、*すべての*若いアメリカ人のオバマに対する意見を推論するために不可欠でした。彼らのサンプルは本当にランダムでしたか？彼らが使用した*サンプリング方法論*を知らなければ、そのような質問に答えるのは難しいです。たとえば、この調査が携帯電話番号のみを使用して実施された場合、携帯電話を持たない人々は除外され、したがってサンプルに代表されません。ハーバード大学のケネディスクールの政治研究所がインターネットニュースサイトでこの調査を実施した場合はどうでしょうか？その場合、この特定のインターネットニュースサイトを読まない人々は除外されます。サンプルがランダムであることを確保するのは、ボウルのサンプリング練習では簡単でしたが、オバマ調査のような実生活の状況では、これはより難しいことです。

## 中心極限定理

この章では、球のボウル（私たちの母集団）に（仮想的に）アクセスし、赤い球の割合を把握したいという願望から始まりました。この母集団にアクセスできるにもかかわらず、実際には、母集団が大きすぎたり、常に変化したり、全数調査を行うには高すぎたりするため、ほとんどの場合、**母集団にアクセスすることはできません**。この現実を受け入れるということは、_統計的推論_を使用する必要があることを受け入れることを意味します。

前のセクションでは、「統計的推論は、サンプルを使用して母集団について推測する行為である」と述べました。しかし、どのようにこの推論を_行う_のでしょうか？前のセクションでは、_サンプリングの枠組み_を定義し、実際には多くのサンプルを取るのではなく、1つの大きなサンプルを取ると述べました。

実際には、*1つだけ*のサンプルを取り、その*1つ*のサンプルを使用して母集団パラメータについての記述を行います。母集団について記述する能力は、有名な定理、つまり数学的に証明された真実である*中心極限定理*によって許されます。あなたが図で視覚化し、表にまとめたものはこの定理のデモンストレーションでした。簡単に言えば、サンプルサイズが大きくなるにつれて、これらのサンプル平均のサンプリング分布がより正規形状に近づき、より狭くなるということです。

言い換えると、サンプルサイズが大きくなると、(1)（標本比率のような）点推定値のサンプリング分布は*正規分布*に従うようになり、(2)これらのサンプリング分布の変動は小さくなり、標準誤差によって定量化されます。

中心極限定理の驚くべき点は、基礎となる母集団分布の形状に関係なく、平均（バニーの重さの標本平均やドラゴンの翼幅の標本平均など）や比率（シャベルの赤い球の標本比率など）のサンプリング分布は**正規**になるということです。正規分布はその中心と広さによって定義され、中心極限定理は両方を提供します：

1. 点推定値のサンプリング分布は真の母集団パラメータを中心にしています。
2. 点推定値のサンプリング分布の幅がどの程度かについての推定値があり、それは標準誤差によって与えられます。

中心極限定理が私たちのために作り出すのは、*単一の*サンプルと母集団の間のはしごです。中心極限定理により、(1)サンプルの点推定値は真の母集団パラメータを中心とする正規分布から引き出され、(2)その正規分布の幅は点推定値の標準誤差によって決まると言えます。これをボウルに関連付けると、1つのサンプルを引き出して赤い球の標本比率$\hat{p}$を得ると、この$\hat{p}$の値は、真の母集団比率の赤い球$p$を中心とし、計算された標準誤差を持つ正規曲線から引き出されるのです。

## 結論

### サンプリングシナリオ

この章では、未知の比率を推測するために触覚的および仮想的なサンプリング演習を行いました。また、実生活でのサンプリングのケーススタディとして世論調査を紹介しました。各場合において、母集団比率$p$を推定するために標本比率$\hat{p}$を使用しました。しかし、比率に関連するシナリオだけに限定されるわけではありません。つまり、他の点推定値を使用して他の母集団パラメータを推定するためにサンプリングを使用することもできます。

1. **比率のシナリオ**：母集団での特定の特性を持つ個体の割合について質問する場合（例：赤い球の割合）、母集団パラメータは$p$、標本推定値は$\hat{p}$。
2. **平均のシナリオ**：母集団内の数値的変数の平均について質問する場合（例：流通しているすべてのペニーの平均年齢）、母集団パラメータは$\mu$、標本推定値は$\bar{x}$。
3. **二つの比率の差のシナリオ**：二つの異なる母集団からの比率の差について質問する場合（例：あくびを見て最初にあくびをする人の割合と、他人のあくびを見ずにあくびをする人の割合の差）、母集団パラメータは$p_1 - p_2$、標本推定値は$\hat{p}_1 - \hat{p}_2$。
4. **二つの平均の差のシナリオ**：二つの異なる母集団からの平均の差について質問する場合（例：アクション映画とロマンス映画のIMDbの平均評価の差）、母集団パラメータは$\mu_1 - \mu_2$、標本推定値は$\bar{x}_1 - \bar{x}_2$。
5. **回帰のシナリオ**：説明変数（年齢や学歴など）と興味のある結果変数（教員評価スコアなど）の間の関係をモデル化したい場合、母集団パラメータは回帰線の傾き$\beta_1$、標本推定値は$b_1$。

### 理論に基づく標準誤差

多くの場合、標準誤差を近似する数式が存在します！ボウルの赤い球の割合を推定するために標本比率赤$\hat{p}$を使用した場合、標本比率$\hat{p}$の標準誤差を近似する数式は次のとおりです：

$$\text{SE}_{\widehat{p}} \approx \sqrt{\frac{\widehat{p}(1-\widehat{p})}{n}}$$

例えば、$n = 50$個の球をサンプリングし、21個の赤い球を観察したとします。これは標本比率$\hat{p}$が21/50 = 0.42であることを意味します。したがって、この数式を使用すると、$\hat{p}$の標準誤差の近似値は次のようになります：

$$\text{SE}_{\widehat{p}} \approx \sqrt{\frac{0.42(1-0.42)}{50}} = \sqrt{0.004872} = 0.0698 \approx 0.070$$

代わりに$n = 100$個の球をサンプリングし、42個の赤い球を観察したとします。これは再び標本比率$\hat{p}$が42/100 = 0.42であることを意味します。しかし、この数式を使用すると、$\hat{p}$の標準誤差の近似値は次のようになります：

$$\text{SE}_{\widehat{p}} \approx \sqrt{\frac{0.42(1-0.42)}{100}} = \sqrt{0.002436} = 0.0494$$

標準誤差が0.0698から0.0494に減少したことがわかります。つまり、$n$ = 100を使用した推定値の「典型的な」エラーは$n$ = 50に比べて減少し、したがってより*精密*になります。

この数式で観察すべき重要なポイントは、分母に$n$があることです。サンプルサイズ$n$が大きくなると、標準誤差は小さくなります。これは中心極限定理の主要なメッセージの1つです：サンプルサイズ$n$が大きくなると、平均の分布は狭くなり、これは標本平均のサンプリング分布の標準偏差によって定量化されます。これはさらに特別な名前を持ちます：標本平均の標準誤差です。

この数式が正しい理由は何ですか？残念ながら、この時点でこれを証明するツールはありません。より高度な確率と統計のコースを受ける必要があります。（これはベルヌーイ分布および二項分布の概念に関連しています。興味があれば、その導出について[こちら](http://onlinestatbook.com/2/sampling_distributions/samp_dist_p.html)で詳しく読むことができます。）

サンプリングは統計的推論の基盤となる重要な概念です。この章では、サンプリングの基本原理を学び、サンプリング変動とサンプルサイズがどのように推定精度に影響するかを理解することができました。

次の章では、信頼区間について学びます。信頼区間は、今回学んだサンプリングの概念を基にして、母集団パラメータの値がどの範囲に含まれるかを推定する方法です。例えば、ハーバード大学の調査では「誤差の範囲はプラスマイナス2.1%」と報告されていましたが、これは信頼区間の一例です。

このPythonコードを使って、自分でもサンプリングのシミュレーションを行い、サンプリング分布がどのように形成されるか、またサンプルサイズが推定の精度にどう影響するかを確認してみてください。中心極限定理の面白い性質として、サンプルサイズが大きくなるにつれて、サンプリング分布が正規分布に近づくことも観察できるでしょう。

```python
# Google Colabで実行するためのセットアップ
!pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats

# プロットの見た目を設定
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# ----------------------------------------
# 7.2 仮想サンプリング
# ----------------------------------------

# ボウルデータの作成
# オリジナルのボウルには2400個の球が含まれていて、そのうち900個が赤(37.5%)
np.random.seed(76)
bowl = pd.DataFrame({
    'ball_ID': range(1, 2401),
    'color': np.random.choice(['red', 'white'], size=2400, p=[0.375, 0.625])
})

# タクタイルサンプリングのデータをGitHubからダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/tactile_prop_red.csv"
tactile_prop_red = pd.read_csv(url)

# 仮想シャベルを一度使う
np.random.seed(76)
virtual_shovel = bowl.sample(n=50, replace=False).reset_index(drop=True)
virtual_shovel['replicate'] = 1

# 赤い球の割合を計算
num_red = (virtual_shovel['color'] == 'red').sum()
prop_red = num_red / 50

print(f"50個の球のうち、{num_red}個が赤です。")
print(f"赤い球の割合: {prop_red:.3f} ({prop_red*100:.1f}%)")

# 仮想シャベルを33回使う
np.random.seed(76)
virtual_samples = pd.DataFrame()

for i in range(1, 34):
    sample = bowl.sample(n=50, replace=False).copy()
    sample['replicate'] = i
    virtual_samples = pd.concat([virtual_samples, sample])

# 各レプリケートの赤い球の割合を計算
virtual_prop_red_33 = virtual_samples.groupby('replicate').apply(
    lambda x: pd.Series({
        'red': (x['color'] == 'red').sum(),
        'prop_red': (x['color'] == 'red').sum() / 50
    })
).reset_index()

# 33の割合の分布を視覚化
plt.figure(figsize=(10, 6))
plt.hist(virtual_prop_red_33['prop_red'], bins=np.arange(0.2, 0.55, 0.05), 
         edgecolor='white', color='skyblue')
plt.xlabel('50個の球のうち赤い球の割合')
plt.ylabel('頻度')
plt.title('33の仮想サンプルにおける赤い球の割合の分布')
plt.xticks(np.arange(0.2, 0.55, 0.05))
plt.tight_layout()
plt.show()

# 仮想シャベルを1000回使う
np.random.seed(76)
virtual_samples_1000 = pd.DataFrame()

for i in range(1, 1001):
    sample = bowl.sample(n=50, replace=False).copy()
    sample['replicate'] = i
    virtual_samples_1000 = pd.concat([virtual_samples_1000, sample])

# 各レプリケートの赤い球の割合を計算
virtual_prop_red_1000 = virtual_samples_1000.groupby('replicate').apply(
    lambda x: pd.Series({
        'red': (x['color'] == 'red').sum(),
        'prop_red': (x['color'] == 'red').sum() / 50
    })
).reset_index()

# 1000の割合の分布を視覚化
plt.figure(figsize=(10, 6))
plt.hist(virtual_prop_red_1000['prop_red'], bins=np.arange(0.15, 0.65, 0.05), 
         edgecolor='white', color='skyblue')
plt.xlabel('50個の球のうち赤い球の割合')
plt.ylabel('頻度')
plt.title('1000の割合の分布')
plt.xticks(np.arange(0.15, 0.65, 0.05))
plt.tight_layout()
plt.show()

# 異なるシャベルの使用
# 3つの異なるサンプルサイズ（25、50、100）で1000回ずつサンプリング
sample_sizes = [25, 50, 100]
results = {}

for n in sample_sizes:
    np.random.seed(76)
    virtual_props = []
    
    # 1000回のサンプリングを実行
    for i in range(1000):
        sample = bowl.sample(n=n, replace=False)
        prop = (sample['color'] == 'red').mean()
        virtual_props.append(prop)
    
    results[n] = virtual_props

# 3つのサンプルサイズの分布を並べて表示
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, n in enumerate(sample_sizes):
    axes[i].hist(results[n], bins=np.arange(0.15, 0.65, 0.05), 
                 edgecolor='white', color='skyblue')
    axes[i].set_title(f'n = {n}')
    axes[i].set_xlabel('シャベルの球のうち赤い球の割合')
    axes[i].set_xlim(0.15, 0.65)
    axes[i].set_xticks(np.arange(0.2, 0.6, 0.1))

plt.suptitle('3つの異なるシャベルサイズにおける赤い球の割合の分布の比較')
plt.tight_layout()
plt.show()

# 3つのサンプルサイズの標準偏差を計算
std_devs = {n: np.std(results[n], ddof=1) for n in sample_sizes}

# 表形式で表示
std_dev_df = pd.DataFrame({
    'シャベルのスロット数': sample_sizes,
    '赤い球の割合の標準偏差': [std_devs[n] for n in sample_sizes]
})

print(std_dev_df)

# ----------------------------------------
# 7.3 サンプリングの枠組み
# ----------------------------------------

# 仮想全数調査 - ボウル全体の赤い球の数を数える
red_count = (bowl['color'] == 'red').sum()
total_count = len(bowl)
population_proportion = red_count / total_count

print(f"ボウル全体には{red_count}個の赤い球があります。")
print(f"ボウル全体の球の数は{total_count}個です。")
print(f"母集団比率 p = {population_proportion:.3f} = {population_proportion*100:.1f}%")

# データをまとめてデータフレームに変換して可視化
all_results = pd.DataFrame({
    'n': np.repeat([25, 50, 100], 1000),
    'prop_red': np.concatenate([results[25], results[50], results[100]])
})

# 3つのサンプルサイズの分布を並べて表示
plt.figure(figsize=(15, 6))
sns.histplot(data=all_results, x='prop_red', hue='n', multiple='dodge', 
             bins=np.arange(0.15, 0.65, 0.05), edgecolor='white')
plt.xlabel('標本比率 $\\hat{p}$')
plt.ylabel('頻度')
plt.title('標本比率 $\\hat{p}$ の3つのサンプリング分布')
plt.xticks(np.arange(0.2, 0.6, 0.1))
plt.tight_layout()
plt.show()

# 3つのサンプルサイズの標準偏差を計算
std_devs = all_results.groupby('n')['prop_red'].std().reset_index()
std_devs.columns = ['サンプルサイズ (n)', '標本比率 $\\hat{p}$ の標準誤差']
print(std_devs)

# 3つのサンプリング分布と真の母集団比率
plt.figure(figsize=(15, 6))
sns.histplot(data=all_results, x='prop_red', hue='n', multiple='dodge', 
             bins=np.arange(0.15, 0.65, 0.05), edgecolor='white')
plt.axvline(x=population_proportion, color='red', linestyle='--', linewidth=2)
plt.xlabel('標本比率 $\\hat{p}$')
plt.ylabel('頻度')
plt.title('標本比率 $\\hat{p}$ の3つのサンプリング分布（真の母集団比率$p$を垂直線で表示）')
plt.xticks(np.arange(0.2, 0.6, 0.1))
plt.tight_layout()
plt.show()

# 正確さと精度の違いを示す図
def create_target_plot(accuracy, precision):
    fig, ax = plt.subplots(figsize=(5, 5))
    # 的を描画
    circle1 = plt.Circle((0.5, 0.5), 0.4, color='lightgray')
    circle2 = plt.Circle((0.5, 0.5), 0.3, color='gray')
    circle3 = plt.Circle((0.5, 0.5), 0.2, color='darkgray')
    circle4 = plt.Circle((0.5, 0.5), 0.1, color='black')
    
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    
    # 弾痕をシミュレート
    np.random.seed(42)
    if accuracy == 'high' and precision == 'high':
        # 中心近くで密集
        points_x = np.random.normal(0.5, 0.05, 10)
        points_y = np.random.normal(0.5, 0.05, 10)
    elif accuracy == 'high' and precision == 'low':
        # 中心近くでばらつき
        points_x = np.random.normal(0.5, 0.15, 10)
        points_y = np.random.normal(0.5, 0.15, 10)
    elif accuracy == 'low' and precision == 'high':
        # 中心から外れて密集
        points_x = np.random.normal(0.7, 0.05, 10)
        points_y = np.random.normal(0.7, 0.05, 10)
    else:  # low accuracy, low precision
        # 中心から外れてばらつき
        points_x = np.random.normal(0.7, 0.15, 10)
        points_y = np.random.normal(0.7, 0.15, 10)
    
    ax.scatter(points_x, points_y, color='red', s=50)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{"高い" if accuracy == "high" else "低い"}正確さ, {"高い" if precision == "high" else "低い"}精度')
    
    return fig

# 2x2グリッドでの正確さと精度の比較
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

accuracy_values = ['high', 'high', 'low', 'low']
precision_values = ['high', 'low', 'high', 'low']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for acc, prec, (i, j) in zip(accuracy_values, precision_values, positions):
    # 的を描画
    circle1 = plt.Circle((0.5, 0.5), 0.4, color='lightgray', transform=axes[i, j].transAxes)
    circle2 = plt.Circle((0.5, 0.5), 0.3, color='gray', transform=axes[i, j].transAxes)
    circle3 = plt.Circle((0.5, 0.5), 0.2, color='darkgray', transform=axes[i, j].transAxes)
    circle4 = plt.Circle((0.5, 0.5), 0.1, color='black', transform=axes[i, j].transAxes)
    
    axes[i, j].add_artist(circle1)
    axes[i, j].add_artist(circle2)
    axes[i, j].add_artist(circle3)
    axes[i, j].add_artist(circle4)
    
    # 弾痕をシミュレート
    np.random.seed(42 + i*10 + j)
    if acc == 'high' and prec == 'high':
        # 中心近くで密集
        points_x = np.random.normal(0.5, 0.05, 10)
        points_y = np.random.normal(0.5, 0.05, 10)
    elif acc == 'high' and prec == 'low':
        # 中心近くでばらつき
        points_x = np.random.normal(0.5, 0.15, 10)
        points_y = np.random.normal(0.5, 0.15, 10)
    elif acc == 'low' and prec == 'high':
        # 中心から外れて密集
        points_x = np.random.normal(0.7, 0.05, 10)
        points_y = np.random.normal(0.7, 0.05, 10)
    else:  # low accuracy, low precision
        # 中心から外れてばらつき
        points_x = np.random.normal(0.7, 0.15, 10)
        points_y = np.random.normal(0.7, 0.15, 10)
    
    axes[i, j].scatter(points_x, points_y, color='red', s=50, transform=axes[i, j].transAxes)
    axes[i, j].set_xlim(0, 1)
    axes[i, j].set_ylim(0, 1)
    axes[i, j].set_aspect('equal')
    axes[i, j].axis('off')
    axes[i, j].set_title(f'{"高い" if acc == "high" else "低い"}正確さ, {"高い" if prec == "high" else "低い"}精度')

plt.tight_layout()
plt.show()

# 標準誤差の理論に基づく近似値の計算例
p_hat = 0.42
n_values = [50, 100]

for n in n_values:
    se = np.sqrt((p_hat * (1 - p_hat)) / n)
    print(f"n = {n}の場合の標準誤差: {se:.4f}")
```