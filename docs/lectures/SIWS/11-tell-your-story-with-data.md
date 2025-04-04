# データであなたの物語を語る

## はじめに

本書を通じてこれまで、さまざまな方法でデータを扱う方法を学んできました。変数の種類に応じて最適なプロットの選択など、データを効果的に可視化する戦略を身につけました。また、スプレッドシート形式でデータを要約し、さまざまな変数の要約統計量を計算する方法を学びました。さらに、サンプリングを使用して母集団に関する結論を導き出すプロセスとしての統計的推測の価値も理解しました。最後に、線形回帰モデルの適合方法と、信頼区間や仮説検定の解釈が有効になるために必要な条件の確認の重要性も探究しました。これらすべてを通じて、多くの計算技術を学び、再現可能なPythonコードの作成に焦点を当ててきました。

ここでは、世界中のデータジャーナリストによる「効果的なデータストーリーテリング」に関する事例研究をさらに紹介します。優れたデータストーリーは読者を誤解させるのではなく、ストーリーテリングを通じてデータが私たちの生活で果たす重要性を理解させるものです。

## レビュー

これまでの内容を振り返ってみましょう。最初にデータの基礎を学び、Pythonのコーディングを始め、最初のパッケージをインストールして読み込み、2023年のニューヨーク市の主要空港からの国内出発便のデータを探索しました。その後、次の3つの分野（パート2と4は1つにまとめられています）を扱いました：

1. データサイエンス：Pythonのパッケージを使用してデータサイエンスツールボックスを組み立てました。特に、
   + 第3章：Matplotlib, seabornを使用してデータを可視化
   + 第4章：Pandasを使用してデータを整理
   + 第5章：「整然データ（tidy data）」の概念を学び、標準化されたデータフレームの入出力形式としての理解、さらにPandasを使用してスプレッドシートファイルをPythonにインポートする方法

2. 統計/データモデリング：これらのデータサイエンスツールを使用して、最初のデータモデルを適合させました。特に、
   + 第6章：説明変数が1つだけの基本的な回帰モデルを発見
   + 第7章：説明変数が複数ある重回帰モデルを検討

3. 統計的推測：新たに獲得したデータサイエンスツールを使用して、統計的推測を解き明かしました。特に、
   + 第8章：サンプリング変動が統計的推測で果たす役割と、サンプルサイズがこのサンプリング変動に与える影響を学習
   + 第9章：ブートストラップを使用して信頼区間を構築し、理論に基づくアプローチについても学習
   + 第10章：順列法を使用して仮説検定を実施

4. 統計/データモデリングの再検討：統計的推測の理解を踏まえて、第6章と第7章で構築したモデルを再検討しました。特に、
   + 第10章：回帰の設定における信頼区間と仮説検定の解釈を理論ベースとシミュレーションベースの両方のアプローチで行いました

私たちは「データで考える（thinking with data）」という、もともとDr. Diane Lambertが作った表現の基礎となる哲学を経験してきました。この哲学は「実践的なデータサイエンス統計学」の中でよく要約されています：

> 日常的な分析作業の多くの側面は、従来の統計文献やカリキュラムにほとんど存在しません。しかし、これらの活動はデータアナリストや応用統計家の時間と労力のかなりの部分を占めています。このコレクションの目標は、現代のデータ分析ワークフローの可視性と採用を高めることです。私たちは、産業界と学界、ソフトウェアエンジニアリングと統計学およびコンピュータサイエンス、そして異なるドメイン間でのツールとフレームワークの移転を促進することを目指しています。

つまり、21世紀以降に「データで考える」準備を整えるためには、アナリストは「データ/サイエンスパイプライン」を全体的に経験する必要があります。長い間、統計教育はこのパイプラインの一部分だけに焦点を当ててきたと私たちは考えています。

この本を締めくくるにあたり、データを扱ういくつかの追加的な事例研究を紹介します。まず、「データ/サイエンスパイプライン」の完全な通過を体験して、アメリカ合衆国ワシントン州シアトルの住宅販売価格を分析します。次に、データジャーナリズムのウェブサイトFiveThirtyEight.comから効果的なデータストーリーテリングの例をいくつか紹介します。これらの事例研究を通じて、「データで考える」だけでなく、「データであなたの物語を語る」方法も学べるでしょう。

## 必要なライブラリ

まずは、この章で必要なパッケージをすべて読み込みましょう。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import permutation_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
import datetime as dt
import requests
from io import StringIO

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
```

## 事例研究：シアトル住宅価格 {#seattle-house-prices}

Kaggle.comは機械学習と予測モデリングのコンペティションサイトで、企業、政府機関、その他の個人がアップロードしたデータセットをホストしています。そのうちの1つは「King County, USAの住宅販売」で、2014年5月から2015年5月までの間にアメリカ合衆国ワシントン州キング郡（シアトル大都市圏を含む）で販売された住宅の販売価格に関する情報が含まれています。

このデータセットをGitHubから直接ダウンロードしましょう：

```python
# GitHubからhouse_pricesデータセットを直接ダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive/master/data-raw/house_prices.csv"
house_prices = pd.read_csv(url)

# データの最初の数行を表示
house_prices.head()
```

このデータセットには約21,000軒の住宅と21の変数が含まれています。これらの変数は、住宅の特徴を説明しています。この事例研究では、以下の変数を使った重回帰モデルを作成します：

- 目的変数 y: 住宅の販売`price`（価格）
- 2つの説明変数：
  1. 数値的説明変数 x₁: 住宅サイズ`sqft_living`（リビングスペースの平方フィート）。1平方フィートは約0.09平方メートル。
  2. カテゴリカル説明変数 x₂: 住宅の`condition`（状態）、5段階のカテゴリカル変数で、`1`は「劣悪」、`5`は「優良」を示す。

### 探索的データ分析：パート1 {#house-prices-EDA-I}

データが提示されたとき、まず探索的データ分析（EDA）を実行することが重要です。探索的データ分析は、データの感覚をつかみ、データの問題を特定し、外れ値を明らかにし、モデル構築に役立ちます。EDの一般的な3つのステップを思い出しましょう：

1. 生データ値を見る
2. 要約統計量を計算する
3. データの可視化を作成する

まず、生データを見るために`info()`と`describe()`関数を使いましょう：

```python
# データの構造を見る
house_prices.info()

# データの基本的な統計量
house_prices.describe()
```

この段階でのEDAでいくつか質問できることがあります：どの変数が数値的ですか？どれがカテゴリカルですか？カテゴリカル変数の場合、そのレベルは何ですか？回帰モデルで使用する変数以外に、住宅価格の予測に役立つと思われる変数は何ですか？

例えば、`condition`変数には値`1`から`5`がありますが、これらは数値値としてではなく、カテゴリとして扱われるべきです。

次に、要約統計量を計算する、EDの2番目のステップを実行しましょう：

```python
# 関心のある変数の要約統計量
house_prices[['price', 'sqft_living', 'condition']].describe().T
```

`price`の平均値は中央値よりも大きいことに注目してください。これは、非常に高価な住宅が少数存在し、平均を押し上げているためです。言い換えれば、データセットには「外れ値」の住宅価格があります（これは可視化を作成するときにさらに明らかになります）。

しかし、中央値はこのような外れ値の住宅価格に対してそれほど敏感ではありません。これが、不動産市場のニュースが一般的に平均ではなく、中央値の住宅価格を報告する理由です。中央値は平均よりも「外れ値に対して頑健」であると言えます。同様に、標準偏差と四分位範囲（IQR）はどちらも広がりと変動性の尺度ですが、IQRは四分位数に基づいて`Q3 - Q1`として計算されるため、より「外れ値に対して頑健」です。

次に、探索的データ分析の最後のステップ、データの可視化を行いましょう。まず、単変量の可視化を作成します。これらは一度に1つの変数に焦点を当てたプロットです。`price`と`sqft_living`は数値変数なので、ヒストグラムを使って分布を可視化できます。一方、`condition`はカテゴリカルなので、棒グラフを使って分布を可視化できます。

```python
# サブプロットの設定
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# 住宅価格のヒストグラム
axs[0].hist(house_prices['price'], bins=50, color='skyblue', edgecolor='white')
axs[0].set_xlabel('価格 (USD)')
axs[0].set_title('住宅価格')

# リビングスペースのヒストグラム
axs[1].hist(house_prices['sqft_living'], bins=50, color='skyblue', edgecolor='white')
axs[1].set_xlabel('リビングスペース（平方フィート）')
axs[1].set_title('住宅サイズ')

# 住宅状態の棒グラフ
condition_counts = house_prices['condition'].value_counts().sort_index()
axs[2].bar(condition_counts.index.astype(str), condition_counts.values, color='skyblue')
axs[2].set_xlabel('状態')
axs[2].set_ylabel('頻度')
axs[2].set_title('住宅状態')

plt.tight_layout()
plt.show()
```

まず、下のプロットから、ほとんどの住宅は状態`3`であり、状態`4`と`5`がそれに続き、状態`1`または`2`はほとんどないことがわかります。

次に、`price`のヒストグラム（上のプロット）では、ほとんどの住宅は200万ドル未満であることがわかります。また、x軸が800万ドルまで伸びていますが、その価格に近い住宅はあまりないようです。これは、以前の要約統計量で述べたように、非常に少数の住宅が800万ドルに近い価格を持っているためです。これらが前述の外れ値の住宅価格です。変数`price`は右に長い尾を持つ「右に歪んでいる（right-skewed）」と言います。

さらに、中央のプロットの`sqft_living`のヒストグラムを見ると、ほとんどの住宅は5000平方フィート未満のリビングスペースを持っていることがわかります。比較のために、アメリカンフットボールのフィールドは約57,600平方フィート、標準的なサッカーフィールドは約64,000平方フィートです。この変数も右に歪んでいますが、`price`変数ほど極端ではありません。

`price`と`sqft_living`の両方の変数について、右への歪みがあるため、x軸の下端にある住宅を区別するのが難しくなっています。これは、非常に高価で巨大な住宅の少数によって、x軸のスケールが圧縮されているためです。

この歪みに対処するには何ができるでしょうか？対数変換（log10変換）を適用しましょう。対数変換により、加法的変化ではなく乗法的変化に焦点を当てた変数のスケールを変更できます。言い換えれば、絶対的な変化ではなく相対的な変化に視点が移ります。このような乗法的/相対的変化は、「桁数の変化」とも呼ばれます。

右に歪んだ変数`price`と`sqft_living`に対して、新しく対数変換されたバージョンを作成しましょう。後者には`log10_size`という名前を付けます：

```python
# 対数変換を適用
house_prices['log10_price'] = np.log10(house_prices['price'])
house_prices['log10_size'] = np.log10(house_prices['sqft_living'])

# 変換前後を表示（最初の10行）
house_prices[['price', 'log10_price', 'sqft_living', 'log10_size']].head(10)
```

特に6行目と3行目の住宅に注目してください。6行目の住宅の`price`は100万ドルをわずかに超えています。10^6は100万なので、その`log10_price`は約6です。これに対して、`log10_price`が6未満の他のすべての住宅は、`price`が100万ドル未満です。3行目の住宅は、`sqft_living`が1000未満の唯一の住宅です。1000 = 10^3なので、`log10_size`が3未満の唯一の住宅です。

次に、`price`に対するこの変換の前後の効果を可視化しましょう：

```python
# 価格の対数変換前後の可視化
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# 変換前
axs[0].hist(house_prices['price'], bins=50, color='skyblue', edgecolor='white')
axs[0].set_xlabel('価格 (USD)')
axs[0].set_title('住宅価格：変換前')

# 変換後
axs[1].hist(house_prices['log10_price'], bins=50, color='skyblue', edgecolor='white')
axs[1].set_xlabel('log10価格 (USD)')
axs[1].set_title('住宅価格：変換後')

plt.tight_layout()
plt.show()
```

変換後、分布は歪みが少なくなり、この場合、より対称的でより釣鐘型になっていることがわかります。これで、低価格の住宅をより簡単に区別できるようになりました。

同様に、住宅サイズについても可視化してみましょう。`sqft_living`が`log10_size`に対数変換されています：

```python
# サイズの対数変換前後の可視化
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# 変換前
axs[0].hist(house_prices['sqft_living'], bins=50, color='skyblue', edgecolor='white')
axs[0].set_xlabel('リビングスペース（平方フィート）')
axs[0].set_title('住宅サイズ：変換前')

# 変換後
axs[1].hist(house_prices['log10_size'], bins=50, color='skyblue', edgecolor='white')
axs[1].set_xlabel('log10リビングスペース（平方フィート）')
axs[1].set_title('住宅サイズ：変換後')

plt.tight_layout()
plt.show()
```

対数変換が変数の歪みを取り除く同様の効果を持っていることがわかります。これら2つの場合では、結果の分布がより対称的で釣鐘型になっていますが、これは必ずしも常にそうであるとは限りません。

`log10_price`と`log10_size`の対称性を考慮して、重回帰モデルを新しい変数を使用するように修正します：

1. 目的変数 y: 住宅の販売`log10_price`
2. 2つの説明変数：
   1. 数値的説明変数 x₁: 住宅サイズ`log10_size`（対数変換したリビングスペースの平方フィート）
   2. カテゴリカル説明変数 x₂: 住宅の`condition`（5段階のカテゴリカル変数）

### 探索的データ分析：パート2 {#house-prices-EDA-II}

次に、多変量の可視化を作成してEDAを続けましょう。これらの可視化は、複数の変数間の関係を示します。これは、モデリングの目標が変数間の関係を探ることなので、実行すべき重要なEDAのステップです。

数値的な目的変数、数値的な説明変数、およびカテゴリカルな説明変数を含むモデルなので、2つの選択肢があります：
1. インタラクションモデル：各`condition`レベルの回帰線が異なる傾きと異なる切片を持つ
2. 平行線モデル：各`condition`レベルの回帰線が同じ傾きだが異なる切片を持つ

まず、インタラクションモデルを可視化しましょう：

```python
# conditionをカテゴリとして扱う
house_prices['condition'] = house_prices['condition'].astype('category')

# インタラクションモデルの可視化
plt.figure(figsize=(12, 8))
sns.lmplot(
    data=house_prices, 
    x='log10_size', 
    y='log10_price', 
    hue='condition',
    scatter_kws={'alpha': 0.05},
    height=8,
    aspect=1.5
)
plt.title('シアトルの住宅価格（インタラクションモデル）')
plt.xlabel('log10サイズ')
plt.ylabel('log10価格')
plt.tight_layout()
plt.show()
```

次に、各条件レベルごとにファセット（分割）表示を作成しましょう：

```python
# ファセット表示のインタラクションモデル
g = sns.FacetGrid(
    house_prices, 
    col='condition', 
    col_wrap=3,
    height=4,
    aspect=1.2
)
g.map_dataframe(sns.scatterplot, x='log10_size', y='log10_price', alpha=0.4)
g.map_dataframe(sns.regplot, x='log10_size', y='log10_price', scatter=False)
g.set_titles('状態: {col_name}')
g.set_axis_labels('log10サイズ', 'log10価格')
g.fig.suptitle('シアトルの住宅価格', y=1.05, fontsize=16)
plt.tight_layout()
plt.show()
```

また、各条件レベルの住宅数を確認しましょう：

```python
# 各条件レベルの住宅数
condition_counts = house_prices['condition'].value_counts().sort_index()
print(condition_counts)
```

どちらの場合も、住宅価格とサイズの間には正の関係があり、住宅がより大きくなるにつれて価格が高くなる傾向があることがわかります。さらに、両方のプロットで、状態5の住宅はほとんどのサイズで最も高価である傾向があり、次に状態4と3が続きます。状態1と2については、このパターンはそれほど明確ではありません。単変量の棒グラフで見たように、状態1または2の住宅はほんの少数しかありません。

### 回帰モデリング {#house-prices-regression}

インタラクションモデル（各条件レベルに異なる傾きと切片）と平行線モデル（各条件レベルに同じ傾きと異なる切片）のどちらが「より良い」でしょうか？

モデル選択においては、追加の複雑さが保証される場合にのみ、より複雑なモデルを選択すべきです。この場合、より複雑なモデルはインタラクションモデルで、5つの切片と5つの傾きの合計を考慮します。これに対して、平行線モデルは5つの切片と1つの共通傾きだけを考慮します。

インタラクションモデルの追加の複雑さは保証されるでしょうか？プロットを見ると、一部の線にわずかなX型（交差）パターンがあるため、インタラクションモデルが適切だと考えられます。したがって、この分析の残りの部分ではインタラクションモデルに焦点を当てます。

インタラクションモデルの5つの異なる傾きと5つの異なる切片は何でしょうか？回帰テーブルからこれらの値を取得できます：

```python
# 条件をカテゴリ型に確実に変換し、最初のレベルをベースラインとして設定
house_prices['condition'] = pd.Categorical(house_prices['condition'])

# インタラクションモデルの作成（statsmodelsを使用）
formula = "log10_price ~ log10_size * C(condition)"
price_interaction = smf.ols(formula=formula, data=house_prices).fit()

# 結果の表示
print(price_interaction.summary())
```

この結果を解釈しましょう。カテゴリカル変数`condition`の「比較のベースライン」グループは状態1の住宅です。したがって、`Intercept`（切片）と`log10_size`の値は、このベースライングループの切片と傾きです。次に、`C(condition)[T.2]`から`C(condition)[T.5]`の項は、条件1の切片に対する切片の「オフセット」です。最後に、`log10_size:C(condition)[T.2]`から`log10_size:C(condition)[T.5]`は、条件1の`log10_size`の傾きに対する傾きの「オフセット」です。

これを簡略化するために、これらの値を使用して5つの回帰線の方程式を書き出しましょう：

```python
# 係数を取得
coef = price_interaction.params

# 各条件レベルの回帰線の方程式を表示
print("条件1の回帰線: log10(price) = {:.3f} + {:.3f} * log10(size)".format(
    coef['Intercept'], coef['log10_size']))

print("条件2の回帰線: log10(price) = {:.3f} + {:.3f} * log10(size)".format(
    coef['Intercept'] + coef['C(condition)[T.2]'], 
    coef['log10_size'] + coef['log10_size:C(condition)[T.2]']))

print("条件3の回帰線: log10(price) = {:.3f} + {:.3f} * log10(size)".format(
    coef['Intercept'] + coef['C(condition)[T.3]'], 
    coef['log10_size'] + coef['log10_size:C(condition)[T.3]']))

print("条件4の回帰線: log10(price) = {:.3f} + {:.3f} * log10(size)".format(
    coef['Intercept'] + coef['C(condition)[T.4]'], 
    coef['log10_size'] + coef['log10_size:C(condition)[T.4]']))

print("条件5の回帰線: log10(price) = {:.3f} + {:.3f} * log10(size)".format(
    coef['Intercept'] + coef['C(condition)[T.5]'], 
    coef['log10_size'] + coef['log10_size:C(condition)[T.5]']))
```

これらの方程式は、前述のインタラクションモデルのプロットの回帰線に対応しています。5つのすべての条件タイプの住宅について、住宅のサイズが大きくなるにつれて価格が上昇します。これは多くの人が予想する通りです。しかし、サイズに対する価格の上昇率は、条件3、4、5の住宅で最も速くなっています。これらは5つの中で最も大きな3つの傾きです。

### 予測の作成 {#house-prices-making-predictions}

不動産業者で、誰かから自分の家がいくらで売れるかと電話で聞かれたとします。彼らは、それが状態5で、サイズが1900平方フィートだと言っています。あなたは何と答えますか？適合させたインタラクションモデルを使用して予測を作成しましょう！

まず、新しい住宅データを作成します：

```python
# 新しい住宅のデータを作成
new_house = pd.DataFrame({
    'log10_size': [np.log10(1900)],
    'condition': ['5']
})

# インタラクションモデルを使用して予測
# 新しい家のlog10_priceを予測
log10_price_pred = price_interaction.predict(new_house)[0]
print(f"予測されたlog10(price): {log10_price_pred:.3f}")

# 実際の価格に変換
price_pred = 10**log10_price_pred
print(f"予測された価格: ${price_pred:,.2f}")
```

予測を視覚化しましょう：

```python
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=house_prices, 
    x='log10_size', 
    y='log10_price', 
    hue='condition',
    alpha=0.05
)

# 各条件レベルの回帰線を追加
for condition in house_prices['condition'].unique():
    subset = house_prices[house_prices['condition'] == condition]
    sns.regplot(
        x='log10_size', 
        y='log10_price', 
        data=subset, 
        scatter=False,
        label=f'条件 {condition}'
    )

# 予測点を追加
plt.axvline(x=np.log10(1900), linestyle='dashed', color='black')
plt.scatter(np.log10(1900), log10_price_pred, color='black', s=100)

plt.title('シアトルの住宅価格（予測付き）')
plt.xlabel('log10サイズ')
plt.ylabel('log10価格')
plt.legend(title='条件')
plt.tight_layout()
plt.show()
```

### 重回帰の統計的推測 {#house-prices-inference-for-regression}

次に、理論ベースとシミュレーションベースの両方の手法を使用して、住宅価格に対する重回帰の結果を確認しましょう。

#### 部分傾きに対する理論ベースの仮説検定 {-}

先ほど解釈した回帰結果を使用して、部分傾きに対する仮説検定を行うことができます。例えば、`log10_size`の部分傾きがゼロであるという帰無仮説を検定できます。`log10_size`に対応する行のp値を見ると、値は0に近く、統計量も大きいことがわかります。有意水準α = 0.05では、`log10_size`は理論ベースのモデルでは`log10_price`と統計的に有意な関係を持つ唯一の説明変数です。

#### 部分傾きに対するシミュレーションベースの仮説検定 {-}

シミュレーションベースの手法を使用して、部分傾きに対する仮説検定を行うこともできます。Pythonでは、これを行うためにpermutation testを実装できます：

```python
# シミュレーションベースの推論のために、条件を数値としてエンコード
house_prices_copy = house_prices.copy()
house_prices_copy['condition'] = house_prices_copy['condition'].astype(int)

def permutation_test(df, n_permutations=1000):
    # 実際のモデルの係数を計算
    formula = "log10_price ~ log10_size * condition"
    real_model = smf.ols(formula=formula, data=df).fit()
    real_params = real_model.params
    
    # 結果を格納するための配列
    permuted_params = np.zeros((n_permutations, len(real_params)))
    
    for i in range(n_permutations):
        # log10_priceをシャッフル
        df_perm = df.copy()
        df_perm['log10_price'] = np.random.permutation(df_perm['log10_price'])
        
        # 同じモデルをシャッフルしたデータに適合
        perm_model = smf.ols(formula=formula, data=df_perm).fit()
        permuted_params[i, :] = perm_model.params
    
    # 各係数のp値を計算
    p_values = []
    for j in range(len(real_params)):
        # 両側検定のp値を計算
        p_value = np.mean(np.abs(permuted_params[:, j]) >= np.abs(real_params[j]))
        p_values.append(p_value)
    
    return dict(zip(real_params.index, p_values))

# 計算に時間がかかるため、サンプルを使用（実際にはより多くのpermutationを使用すべき）
sample_df = house_prices_copy.sample(n=1000, random_state=42)
p_values = permutation_test(sample_df, n_permutations=100)

print("Permutation test p-values:")
for param, p_value in p_values.items():
    print(f"{param}: {p_value:.4f}")
```

この結果は、理論ベースの分析から得た結論と一致しています。`log10_size`だけが統計的に有意な説明変数であり、他の変数はモデル内で有意ではありません。

ここで重要なことは、分布の想定が満たされているかどうかによって、理論ベースの結果とシミュレーションベースの結果が常に一致するとは限らないということです。

## 事例研究：効果的なデータストーリーテリング {#data-journalism}

### ハリウッドのジェンダー表現に関するベクデルテスト

Walt Hickeyによる記事「ハリウッドの女性排除に対するドルと金銭の事例」を読んで分析することをお勧めします。この記事の中で、ウォルトは数十年にわたって、アリソン・ベクデルによって作成された映画におけるジェンダー表現の非公式テストであるベクデルテストに合格する映画がどれだけあるかを研究しました。

記事を読む際には、Walt Hickeyがデータ、グラフィック、分析をどのように使用して読者にストーリーを伝えているかを注意深く考えてください。再現性の精神に基づいて、FiveThirtyEightはこの記事に使用したデータとPythonコードも共有しています。

FiveThirtyEightのデータセットに直接アクセスするために、そのデータを読み込んでみましょう：

```python
# FiveThirtyEightのベクデルテストデータを読み込む
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/bechdel/movies.csv"
bechdel_data = pd.read_csv(url)

# データの最初の数行を確認
bechdel_data.head()
```

### 1999年の米国出生数

FiveThirtyEightの`US_births_1994_2003`データフレームには、1994年から2003年の間の米国における日々の出生数に関する情報が含まれています。このデータフレームに直接アクセスしましょう：

```python
# 米国出生データを読み込む
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births_1994_2003.csv"
US_births_1994_2003 = pd.read_csv(url)

# データの最初の数行を確認
US_births_1994_2003.head()
```

データを探索する前に、`date`列を適切な日付形式に変換しましょう：

```python
# date列をdatetime形式に変換
US_births_1994_2003['date'] = pd.to_datetime(US_births_1994_2003[['year', 'month', 'date']].astype(str).agg('-'.join, axis=1))

# 1999年のデータだけをフィルタリング
US_births_1999 = US_births_1994_2003[US_births_1994_2003['year'] == 1999].copy()

# 最初の数行を確認
US_births_1999.head()
```

`date`は時間の概念であり、順序付けられているため、散布図よりも折れ線グラフの方が適切な可視化です。このようなプロットは「時系列」プロットと呼ばれます：

```python
plt.figure(figsize=(14, 6))
plt.plot(US_births_1999['date'], US_births_1999['births'])
plt.title('1999年の米国出生数')
plt.xlabel('日付')
plt.ylabel('出生数')
plt.grid(True)
plt.tight_layout()
plt.show()
```

図では、2000年1月1日の直前に大きな落ち込みがあることがわかります。これはおそらく休暇シーズンのためでしょう。しかし、1999年10月1日の直前に14,000以上の出生数の大きなスパイクはどうでしょうか？その異常に高いスパイクの理由は何でしょうか？

`US_births_1999`の行を出生数の降順でソートしてみましょう：

```python
# 出生数の降順でソート
US_births_1999.sort_values('births', ascending=False).head(10)
```

最も出生数が多い日（14,540）は実際には1999-09-09です。この日付を月/日/年形式（米国の標準形式）で書くと、出生数が最も多い日は9/9/99です！すべて9です！親が意図的にこの日に出産を誘発した可能性があるのでしょうか？おそらく？原因が何であれ、これは面白い事実です！

## まとめ

これで、「データであなたの物語を語る」ことについて検討しました。シアトルの住宅価格の分析を通じて「データ/サイエンスパイプライン」の完全な通過を体験し、FiveThirtyEightからのデータジャーナリズムの例も見てきました。

「データで考える」だけでなく、「データであなたの物語を語る」ことができるようになりました。これらのスキルが、将来にわたってデータを使って素晴らしいストーリーを語るのに役立つことを願っています。Pythonとデータ分析の世界へのこの旅に参加してくれてありがとう！