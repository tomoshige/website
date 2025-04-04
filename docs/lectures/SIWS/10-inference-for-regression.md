# 回帰分析における統計的推論

この章では、第5章と第6章で学んだ回帰モデルを再検討します。第7章と第8章で導入した統計的推論の方法を考慮しながら、これを行います。サンプルデータに以前導入した線形回帰法を適用することで、応答変数と説明変数の間の関係について全体の母集団の洞察を得ることができることを示します。

## 必要なライブラリ

```python
# 必要なライブラリをインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
import pingouin as pg
import scipy.stats as stats
import random
import warnings
warnings.filterwarnings('ignore')

# プロットのスタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
```

## 単純線形回帰モデル

### 国連加盟国の再検討

第5章で扱った国連加盟国の例を簡単に復習しましょう。2024年現在の国連加盟国のデータは、GitHubから直接ダウンロードします。

```python
# データをダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/un_member_states_2024.csv"
un_member_states_2024 = pd.read_csv(url)

# 必要な変数を選択し、欠損値を含む行を削除
UN_data_ch10 = un_member_states_2024.rename(
    columns={'life_expectancy_2022': 'life_exp', 'fertility_rate_2022': 'fert_rate'}
)[['country', 'life_exp', 'fert_rate']].dropna()

# データの最初の数行を表示
UN_data_ch10.head()
```

データの数値変数の要約統計量を表示します。

```python
# 数値変数の要約統計量
UN_data_ch10[['life_exp', 'fert_rate']].describe()
```

欠損値のない観測値が数個あります。応答変数である出生率（`fert_rate`）と説明変数である平均寿命（`life_exp`）の間の単純線形回帰を使用すると、回帰直線は次のようになります：

$$
\widehat{y}_i = b_0 + b_1 \cdot x_i
$$

ここで、添え字$i$は国連データセット内の$i$番目の観測値（国）を表し、$i = 1, \dots, n$です。この国連データでは$n$はデータセットの行数です。値$x_i$は$i$番目の加盟国の平均寿命値を表し、$\widehat{y}_i$は$i$番目の加盟国の予測出生率です。

予測出生率は回帰直線の結果であり、通常、観測された応答$y_i$とは異なります。残差は$y_i - \widehat{y}_i$として与えられます。

第5章で説明したように、切片（$b_0$）と傾き（$b_1$）は最小二乗基準に基づいて「最適な」直線である回帰係数です。言い換えると、最小二乗係数（$b_0$と$b_1$）を使用して計算された予測値$\widehat{y}$は、*残差の二乗和*を最小化します：

$$
\sum_{i=1}^{n}(y_i - \widehat{y}_i)^2
$$

第5章と同様に、線形回帰モデルを当てはめます。「当てはめる」とは、残差の二乗和を最小化する回帰係数$b_0$と$b_1$を計算することを意味します。Pythonでは、`statsmodels`ライブラリを使用してこれを行います：

```python
# 線形回帰モデルを当てはめる
model = smf.ols('fert_rate ~ life_exp', data=UN_data_ch10).fit()

# 係数を表示
print("切片 (b0):", round(model.params[0], 2))
print("傾き (b1):", round(model.params[1], 2))
```

回帰直線は$\widehat{y}_i = b_0 + b_1 \cdot x_i$です。ここで$x_i$は$i$番目の国の平均寿命であり、$\widehat{y}_i$はそれに対応する予測出生率です。
$b_0$係数は切片であり、説明変数$x_i$の値の範囲がゼロを含む場合にのみ意味を持ちます。平均寿命は常に正の値であるため、この例では切片に解釈を与えません。
$b_1$係数は回帰直線の傾きです。どの国にとっても、平均寿命が約1年増加すると、出生率が約0.137単位減少すると予想されます。

出生率と平均寿命の関係を散布図とともに可視化し、最小二乗基準を使用した回帰直線も含めます：

```python
# 出生率と平均寿命の関係をプロット
plt.figure(figsize=(10, 6))
sns.scatterplot(x='life_exp', y='fert_rate', data=UN_data_ch10)
sns.regplot(x='life_exp', y='fert_rate', data=UN_data_ch10, 
            scatter=False, ci=None, line_kws={'color': 'red', 'linewidth': 0.5})
plt.xlabel('平均寿命 (x)')
plt.ylabel('出生率 (y)')
plt.title('出生率と平均寿命の関係')
plt.tight_layout()
plt.show()
```

最後に、データセット内の観測値の予測値と残差を求める方法を復習します。
例えばフランスは国連加盟国の一つです。線形回帰に基づいてフランスの予測出生率を求めたいとします。

```python
# フランスのデータを取得
france_data = UN_data_ch10[UN_data_ch10['country'] == 'France']
print(france_data)

# フランスの実際の値、予測値、残差を計算
france_life_exp = france_data['life_exp'].values[0]
france_fert_rate = france_data['fert_rate'].values[0]
france_predicted = model.params[0] + model.params[1] * france_life_exp
france_residual = france_fert_rate - france_predicted

print(f"フランスの平均寿命: {france_life_exp}")
print(f"フランスの実際の出生率: {france_fert_rate}")
print(f"回帰直線による予測出生率: {france_predicted:.4f}")
print(f"残差: {france_residual:.4f}")
```

Pythonでは、回帰モデルを使って各国連加盟国の予測値と残差を直接計算できます：

```python
# 予測値と残差を計算
UN_data_ch10['predicted'] = model.predict(UN_data_ch10)
UN_data_ch10['residual'] = UN_data_ch10['fert_rate'] - UN_data_ch10['predicted']

# 最初の数行を表示
UN_data_ch10[['country', 'life_exp', 'fert_rate', 'predicted', 'residual']].head()
```

これで第5章の復習は完了です。この情報を統計的推論にどのように使用するかを説明します。

### モデル

第7章（信頼区間）と第8章（仮説検定）で行ったように、この問題を母集団とそれに関連する関心のあるパラメータの文脈で提示します。
この母集団からランダムサンプルを取り、これらのパラメータを推定します。

この母集団には応答変数（$Y$）と説明変数（$X$）があり、これらの変数の間には線形モデルで表される*統計的線形関係*があると仮定します：

$$Y = \beta_0 + \beta_1 \cdot X + \epsilon,$$ 

ここで$\beta_0$は母集団の切片、$\beta_1$は母集団の傾きです。これらは直線の方程式を生成する説明変数（$X$）と共にモデルのパラメータです。
この関係の統計的部分は、*誤差項*と呼ばれる確率変数$\epsilon$によって与えられます。
誤差項は、$Y$のうち直線によって説明されない部分を表します。

誤差項$\epsilon$の分布に関して追加の仮定をします。
誤差項の期待値はゼロであり、標準偏差は$\sigma$と呼ばれる正の定数に等しいと仮定します。つまり：$E(\epsilon) = 0$ および $SD(\epsilon) = \sigma$。

これらの量の意味を復習しましょう。
この母集団から多くの観測値を取ると、誤差項は時にはゼロより大きく、時にはゼロより小さくなりますが、平均するとゼロになると予想されます。
同様に、一部の誤差項はゼロに非常に近く、他はゼロから非常に離れていますが、平均すると、それらはおおよそゼロから$\sigma$単位離れていることが予想されます。

標準偏差の二乗は分散と呼ばれ、$Var(\epsilon) = \sigma^2$です。
$X$の値に関係なく、誤差項の分散は$\sigma^2$に等しくなります。
この特性は*等分散性*または分散の一定性と呼ばれます。
これは後の分析で役立ちます。

### 推論のためのサンプルの使用

第7章と第8章で行ったように、母集団のパラメータを推定するためにサンプルを使用します。
ここでは、米国ワイオミング州イエローストーン国立公園にあるオールド・フェイスフル間欠泉から収集されたデータを使用します。
このデータセットには、間欠泉噴火の`duration`（秒単位）と次の噴火までの`waiting`時間（分単位）が含まれています。
現在の噴火の持続時間は、次の噴火までの待ち時間をかなり正確に予測するのに役立ちます。
このサンプルとして、2024年6月1日から8月19日までにボランティアによって収集され、[geysertimes.org](https://geysertimes.org/)のウェブサイトに保存されたデータを使用します。

```python
# オールド・フェイスフル間欠泉のデータをダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/old_faithful_2024.csv"
old_faithful_2024 = pd.read_csv(url)

# データの最初の10行を表示
old_faithful_2024.head(10)
```

最初の行を見ると、例えば、2024年8月19日の午前5:38に発生した噴火が235秒続き、次の噴火までの待ち時間は180分であったことがわかります。
これらの2つの変数の要約を表示します：

```python
# 持続時間と待ち時間の要約統計量
old_faithful_2024[['duration', 'waiting']].describe()
```

サンプルには114の噴火があり、99秒から300秒まで続き、次の噴火までの待ち時間は102分から201分までです。
各観測値は、説明変数（$X$）の値と応答（$Y$）の値のペアであることに注目してください。サンプルは次の形式をとります：

$$\begin{array}{c}
(x_1,y_1)\\
(x_2, y_2)\\
\vdots\\
(x_n, y_n)\\
\end{array}$$

ここで、例えば、$(x_2, y_2)$はサンプル内の2番目の観測値の説明変数と応答値のペアです。
より一般的には、$i$番目のペアを$(x_i, y_i)$と表記します。ここで$x_i$は説明変数$X$の観測値であり、$y_i$は応答変数$Y$の観測値です。
サンプルには$n$個の観測値があるので、$i=1, \dots, n$とします。

この例では$n = 114$で、$(x_2, y_2)$は(2番目の噴火の持続時間, 2番目の噴火に対する次の噴火までの待ち時間)のペアです。

噴火の持続時間と待ち時間の関係を散布図で可視化します：

```python
# 持続時間と待ち時間の関係をプロット
plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration', y='waiting', data=old_faithful_2024, alpha=0.3)
plt.xlabel('持続時間 (秒)')
plt.ylabel('待ち時間 (分)')
plt.title('噴火の持続時間と次の噴火までの待ち時間の関係')
plt.tight_layout()
plt.show()
```

この関係は正の相関を持ち、ある程度線形であるように見えます。

### 最小二乗法

これらの変数の関連が線形または近似的に線形であれば、サンプル内の各観測値に対して記述したモデルを適用できます：

$$\begin{aligned}
y_1 &= \beta_0 + \beta_1 \cdot x_1 + \epsilon_1\\
y_2 &= \beta_0 + \beta_1 \cdot x_2 + \epsilon_2\\
\vdots & \phantom{= \beta_0 + \beta_1 \cdot + \epsilon_2 +}\vdots \\
y_n &= \beta_0 + \beta_1 \cdot x_n + \epsilon_n
\end{aligned}$$

このモデルを使用して説明変数と応答の関係を記述したいと考えていますが、パラメータ$\beta_0$と$\beta_1$は未知です。
第5章で紹介した*最小二乗法*を適用してランダムサンプルを使用してこれらのパラメータを推定します。
*残差の二乗和*を最小化する切片（$\beta_0$）と傾き（$\beta_1$）の推定量を計算します：

$$\sum_{i=1}^n \left[y_i - (\beta_0 + \beta_1 \cdot x_i)\right]^2.$$

これは最適化問題であり、解析的に解くには微積分が必要ですが、これはこの本の範囲を超えています。
第5章で最初に導入した回帰係数が解となります：$b_0$は$\beta_0$の推定量、$b_1$は$\beta_1$の推定量です。
これらは*最小二乗推定量*と呼ばれ、その数学的表現は：

$$b_1 =  \frac{\sum_{i=1}^n(x_i - \bar x)(y_i - \bar y)}{\sum_{i=1}^n(x_i - \bar x)^2} \text{ かつ } b_0 = \bar y - b_1 \cdot \bar x.$$

さらに、$\epsilon_i$の標準偏差の*推定量*は次のように与えられます：

$$s = \sqrt{\frac{\sum_{i=1}^n \left[y_i - (b_0 + b_1 \cdot x_i)\right]^2}{n-2}} = \sqrt{\frac{\sum_{i=1}^n \left(y_i - \widehat{y}_i\right)^2}{n-2}}.$$

これらまたは同等の計算は、`statsmodels`ライブラリの`OLS`関数を使用する際にPythonで行われます。
オールド・フェイスフル間欠泉のデータを使うと：

```python
# 回帰モデルを当てはめる
model_1 = smf.ols('waiting ~ duration', data=old_faithful_2024).fit()

# 係数と標準偏差を表示
print("切片 (b0):", round(model_1.params[0], 3))
print("傾き (b1):", round(model_1.params[1], 3))
print("標準偏差 (s):", round(model_1.mse_resid**0.5, 3))  # 平方残差平均の平方根
```

このデータに基づき、線形モデルが適切であると仮定すると、噴火が1秒長く続くごとに、次の噴火までの待ち時間は平均して0.37分増加すると言えます。
どの噴火もゼロ秒より長く続くため、この例では切片に意味のある解釈はありません。
最後に、次の噴火までの待ち時間は、回帰直線の値から平均して約20.37分離れていることを概ね予想できます。

### 最小二乗推定量の特性

最小二乗法は、残差二乗和を可能な限り小さくする最小二乗推定量$b_0$と$b_1$を選択することにより、*最適な*直線を生成します。
しかし、$b_0$と$b_1$の選択は観測されたサンプルに依存します。
データから取られる各ランダムサンプルに対して、$b_0$と$b_1$の異なる値が決定されます。
その意味で、最小二乗推定量$b_0$と$b_1$は確率変数であり、次のような非常に有用な特性を持っています：

- $b_0$と$b_1$は$\beta_0$と$\beta_1$の不偏推定量です。つまり、数学的表記では：$E(b_0) = \beta_0$および$E(b_1) = \beta_1$。
  これは、一部のランダムサンプルでは$b_1$が$\beta_1$より大きく、他のサンプルでは$\beta_1$より小さくなることを意味します。
  平均すると、$b_1$は$\beta_1$に等しくなります。
- $b_0$と$b_1$は観測された応答$y_1$、$y_2$、$\dots$、$y_n$の線形結合です。
  例えば$b_1$の場合、$b_1 = \sum_{i=1}^n c_iy_i$となるような既知の定数$c_1$、$c_2$、$\dots$、$c_n$が存在します。
- $s^2$は分散$\sigma^2$の不偏推定量です。

これらの特性は、回帰の理論に基づく推論を行う次のサブセクションで役立ちます。

### 基本的な回帰と他の方法の関連

このセクションを締めくくるために、回帰が二つの異なる統計手法とどのように関連するかを調査します。一つはすでにこの本で説明したサンプル平均の差であり、もう一つはANOVAという新しい手法ですが関連しています。どちらも回帰の枠組みで表現できることを見ていきます。

#### 二標本平均差

二標本平均差は、第8章で見たように二つのグループの平均を比較するために使用される一般的な統計手法です。治療群と対照群のような二つのグループ間の応答の平均に有意な差があるかどうかを判断するためによく使用されます。二標本平均差は、二つのグループを表すダミー変数を使用して回帰の枠組みで表現できます。

`moderndive`パッケージの`movies_sample`データフレームを再度考えてみましょう。「アクション」と「ロマンス」のジャンルの平均評価を比較します。ジャンルのダミー変数を使用して線形モデルを当てはめるために`statsmodels`を使用できます：

```python
# 映画データをダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/movies_sample.csv"
movies_sample = pd.read_csv(url)

# アクションとロマンスのジャンルだけをフィルタリング
movies_filtered = movies_sample[movies_sample['genre'].isin(['Action', 'Romance'])]

# 回帰モデルを当てはめる
mod_diff_means = smf.ols('rating ~ genre', data=movies_filtered).fit()

# 回帰テーブルを表示
print(mod_diff_means.summary().tables[1])
```

`genre: Romance`行の`p-value`は、仮説検定の$p$値です：

$$
H_0: \text{アクションとロマンスの平均評価は同じ}
$$
$$
H_A: \text{アクションとロマンスの平均評価は異なる}
$$ 

この$p$値は第8章で見つけたものと密接に一致していますが、ここでは線形モデルを使った理論に基づくアプローチを使用しています。`genre: Romance`行の`coef`は、第8章でも見たアクションとロマンスのジャンル間の平均の観測された差ですが、「アクション」ジャンルが基準レベルであるため符号が逆になっています。

#### ANOVA

ANOVAまたは分散分析は、複数のグループの平均間に統計的に有意な差があるかどうかを確認することによって、3つ以上のグループの平均を比較するための統計手法です。ANOVAはグループを表すダミー変数を使用して回帰の枠組みで表現できます。`moderndive`パッケージの`spotify_by_genre`データフレームの`country`、`hip-hop`、`rock`のジャンル間で`popularity`（数値）の値を比較するとします。

```python
# Spotifyデータをダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/spotify_by_genre.csv"
spotify_by_genre = pd.read_csv(url)

# 必要なジャンルのみを選択
spotify_for_anova = spotify_by_genre[
    spotify_by_genre['track_genre'].isin(['country', 'hip-hop', 'rock'])
][['artists', 'track_name', 'popularity', 'track_genre']]

# ランダムにいくつかの行を表示
np.random.seed(6)
print(spotify_for_anova.sample(5))
```

線形モデルを当てはめる前に、`track_genre`と`popularity`のボックスプロットを見て、3つのジャンルの分布に違いがあるかどうかを確認しましょう：

```python
# ジャンル別のポピュラリティのボックスプロット
plt.figure(figsize=(10, 6))
sns.boxplot(x='track_genre', y='popularity', data=spotify_for_anova)
plt.xlabel('ジャンル')
plt.ylabel('ポピュラリティ')
plt.title('ジャンル別のポピュラリティ分布')
plt.tight_layout()
plt.show()
```

`track_genre`でグループ化して`popularity`の平均を計算することもできます：

```python
# ジャンル別の平均ポピュラリティを計算
mean_popularities_by_genre = spotify_for_anova.groupby('track_genre')['popularity'].mean().reset_index()
mean_popularities_by_genre.columns = ['track_genre', 'mean_popularity']
print(mean_popularities_by_genre)
```

ジャンルのダミー変数を使用して線形モデルを当てはめるために`statsmodels`を使用します：

```python
# 回帰モデルを当てはめる
mod_anova = smf.ols('popularity ~ track_genre', data=spotify_for_anova).fit()

# 回帰テーブルを表示
print(mod_anova.summary().tables[1])
```

`track_genre[T.hip-hop]`と`track_genre[T.rock]`行の`coef`は、「hip-hop」と「country」のジャンル間、および「rock」と「country」のジャンル間の平均の差です。「country」ジャンルが基準レベルです。これらの値は（いくつかの丸め誤差があるものの）`mean_popularities_by_genre`に示されているものと一致します。

`p-value`列は、`hip-hop`が`country`と比較して統計的に高い平均`popularity`を持つことを示しており、値はほぼ0（0と報告）です。また、`rock`は0.153という統計的に有意でない$p$値を示しており、`rock`は`country`と比較して有意に高いポピュラリティを持たないと言えます。

伝統的なANOVAはこのレベルの詳細を提供しません。`scipy.stats`モジュールの`f_oneway`関数を使用して実行できます：

```python
# 各ジャンルのポピュラリティ値を取得
country_pop = spotify_for_anova[spotify_for_anova['track_genre'] == 'country']['popularity']
hiphop_pop = spotify_for_anova[spotify_for_anova['track_genre'] == 'hip-hop']['popularity']
rock_pop = spotify_for_anova[spotify_for_anova['track_genre'] == 'rock']['popularity']

# ANOVAを実行
f_stat, p_value = stats.f_oneway(country_pop, hiphop_pop, rock_pop)
print(f"F統計量: {f_stat:.3f}")
print(f"p値: {p_value:.6f}")
```

ここでの小さな$p$値（ほぼ0）は、3つのジャンル間の平均ポピュラリティが等しいという帰無仮説を棄却することにつながります。これは線形モデルを使用して見つけた結果と一致しています。ただし、伝統的なANOVAの結果は、どの平均が互いに異なるかを教えてくれませんが、線形モデルはそれを教えてくれます。ANOVAはグループの平均に差が存在することだけを教えてくれます。

## 単純線形回帰の理論に基づく推論

### 概念的枠組み

線形モデルの仮定を復習することから始めましょう。オールド・フェイスフル間欠泉のデータを使っていくつかの枠組みを説明します。$n = 114$個の観測値のランダムサンプルがあることを思い出してください。

噴火の`duration`と次の噴火までの`waiting`時間の間に線形関係があると仮定しているので、$i$番目の観測値の線形関係を$y_i = \beta_0 + \beta_1 \cdot x_i + \epsilon_i$（$i=1,\dots,n$）と表現できます。$x_i$がサンプル内の$i$番目の噴火の`duration`、$y_i$が次の噴火までの`waiting`時間、そして$\beta_0$と$\beta_1$が定数と考えられる母集団パラメータであることに注意してください。
誤差項$\epsilon_i$は、観測された応答$y_i$が期待される応答$\beta_0 + \beta_1 \cdot x_i$とどれだけ異なるかを表す確率変数です。

`old_faithful_2024`データセットから2つの観測値を使用して、誤差項の役割を説明できます。
ここでは、線形モデルが適切であり、`duration`と`waiting`時間の関係を真に表していると仮定します。
49番目と51番目の観測値を選択します：

```python
# 49番目と51番目の観測値を選択
print(old_faithful_2024.iloc[[48, 50]])  # Pythonはゼロベースのインデックスを使用
```

両方の観測値で`duration`時間は同じですが、応答である`waiting`時間は異なります。
線形モデルが適切であると仮定すると、両方の応答は次のように表現できます：

$$\begin{aligned}
y_{49} &= \beta_0 + \beta_1 \cdot 236 + \epsilon_{49}\\
y_{51} &= \beta_0 + \beta_1 \cdot 236 + \epsilon_{51}
\end{aligned}$$

しかし、$y_{49} = 139$で$y_{51} = 176$です。
応答の違いは誤差項によるもので、線形モデルでは説明されない応答の変動を考慮しています。

線形モデルでは、誤差項$\epsilon_i$の期待値は$E(\epsilon_i) = 0$で、標準偏差は$SD(\epsilon_i) = \sigma$です。
ランダムサンプルを取ると仮定しているので、任意の2つの異なる噴火$i$と$j$に対する任意の2つの誤差項$\epsilon_i$と$\epsilon_j$は独立していると仮定します。

理論に基づく推論を行うために、もう一つの仮定が必要です。
誤差項が平均ゼロ、標準偏差$\sigma$の正規分布に従うと仮定します：

$$\epsilon_i \sim Normal(0, \sigma).$$ 

母集団パラメータ$\beta_0$と$\beta_1$は定数です。
同様に、$i$番目の噴火の`duration`、$x_i$は既知であり、これも定数です。
したがって、表現$\beta_0 + \beta_1 \cdot x_i$は定数です。対照的に、$\epsilon_i$は正規分布に従う確率変数です。

応答$y_i$（$i$番目の噴火から次の噴火までの`waiting`時間）は、定数$\beta_0 + \beta_1 \cdot x_i$と正規分布に従う確率変数$\epsilon_i$の和です。
確率変数と正規分布の特性に基づいて、$y_i$も平均$\beta_0 + \beta_1 \cdot x_i$、標準偏差$\sigma$の正規分布に従う確率変数であると言えます：

$$y_i \sim Normal(\beta_0 + \beta_1 x_i\,,\, \sigma)$$ 

$i=1,\dots,n$に対して。
$\epsilon_i$と$\epsilon_j$は独立しているので、任意の$i \ne j$に対して$y_i$と$y_j$も独立しています。

さらに、最小二乗推定量$b_1$は確率変数$y_1, \dots, y_n$の線形結合であると述べました。
したがって、$b_1$も確率変数です！
これはどういう意味でしょうか？
傾きの係数は、$n$ペアの`duration`と`waiting`時間の*特定のサンプル*から得られます。
$n$ペアの異なるサンプルを収集した場合、*サンプリング変動*により傾きの係数は異なる可能性が高いです。

仮説的に`duration`と`waiting`時間のペアの多くのランダムサンプルを収集し、最小二乗法を使用して各サンプルの傾き$b_1$を計算するとします。
これらの傾きは$b_1$のサンプリング分布を形成します。これは第6章でサンプル比率の文脈で議論しました。
$y_1, \dots, y_n$が正規分布に従い、$b_1$がこれらの確率変数の線形結合であるため、$b_1$も正規分布に従うことがわかります。
この本の範囲を超える計算の後、応答$y_1, \dots, y_n$の期待値と標準偏差の特性を考慮すると、次のことが示されます：

$$b_1 \sim Normal \left(\beta_1\,,\, \frac{\sigma}{\sqrt{\sum_{i=1}^n(x_i - \bar x)^2}}\right)$$

つまり、$b_1$は期待値$\beta_1$と上記の表現（括弧内のコンマの後）に等しい標準偏差を持つ正規分布に従います。
同様に、$b_0$は$y_1, \dots, y_n$の線形結合であり、応答の期待値と標準偏差の特性を使用すると、次のようになります：

$$b_0 \sim Normal \left(\beta_0\,,\, \sigma\sqrt{\frac1n + \frac{\bar x^2}{\sum_{i=1}^n(x_i - \bar x)^2}}\right)$$ 

最小二乗推定量を標準化して、

$$z_0 = \frac{b_0 - \beta_0}{\left(\sigma\sqrt{\frac1n + \frac{\bar x^2}{\sum_{i=1}^n(x_i - \bar x)^2}}\right)}\qquad\text{ そして }\qquad z_1 = \frac{b_1 - \beta_1}{\left(\frac{\sigma}{\sqrt{\sum_{i=1}^n(x_i - \bar x)^2}}\right)}$$ 

が対応する標準正規分布になります。

### 最小二乗推定量の標準誤差

第6章とサブセクション「CLT-mean」で議論したように、中心極限定理により、サンプル平均$\overline{X}$の分布はパラメータ$\mu$を平均とし、標準偏差$\sigma/\sqrt{n}$の近似的な正規分布になります。
その後、$\overline{X}$の推定標準誤差を使用して信頼区間と仮説検定を構築しました。

類似の処理が$b_0$と$b_1$の信頼区間と仮説検定を構築するために使用されます。
上記の方程式で$b_0$と$b_1$の標準偏差がサンプルサイズ$n$、説明変数の値、その平均、そして$y_i$の標準偏差（$\sigma$）を使用して構築されることに注目してください。
これらの値の多くは私たちに知られていますが、$\sigma$は通常知られていません。

代わりに、サブセクション「最小二乗法」で導入した標準偏差の推定量$s$を使用して$\sigma$を推定します。
$b_1$の推定標準偏差は$b_1$の*標準誤差*と呼ばれ、次のように与えられます：

$$SE(b_1) = \frac{s}{\sqrt{\sum_{i=1}^n(x_i - \bar x)^2}}.$$ 

*標準誤差*はサンプルから計算された任意の点推定の標準偏差であることを思い出してください。
$b_1$の*標準誤差*は、傾きの推定量$b_1$が異なるランダムサンプルに対してどれだけの変動を持つ可能性があるかを定量化します。
標準誤差が大きいほど、推定された傾き$b_1$にはより多くの変動が予想されます。
同様に、$b_0$の*標準誤差*は：

$$SE(b_0) = s\sqrt{\frac1n + \frac{\bar x^2}{\sum_{i=1}^n(x_i - \bar x)^2}}$$

第7章で説明したように、パラメータ$\sigma$の代わりに推定量$s$を使用する場合、計算に追加の不確実性が導入されます。
例えば、$b_1$を

$$t = \frac{b_1 - \beta_1}{SE(b_1)}$$ 

を使用して標準化できます。$SE(b_1)$を計算するために$s$を使用しているため、標準誤差の値はサンプルごとに変化し、この追加の不確実性により検定統計量$t$の分布はもはや正規分布ではありません。
代わりに、自由度$n-2$の$t$分布に従います。
2つの自由度の損失は、線形モデルで2つのパラメータ$\beta_0$と$\beta_1$を推定しようとしている事実に関連しています。
これで最小二乗推定量$b_0$と$b_1$の推論を行う準備ができました。

### 最小二乗推定量の信頼区間

$\beta_1$の95%信頼区間は、`duration`と`waiting`時間の間の線形関係の母集団傾き$\beta_1$のもっともらしい値の範囲と考えることができます。
一般的に、推定量のサンプリング分布が正規または近似的に正規である場合、関連するパラメータの信頼区間は

$$
\text{点推定} \pm \text{誤差の余裕} = \text{点推定} \pm (\text{臨界値} \cdot \text{標準誤差})
$$

$\beta_1$の95%信頼区間の公式は$b_1 \pm q \cdot SE(b_1)$で与えられます。ここで臨界値$q$は必要な信頼水準、使用されるサンプルサイズ、および$t$分布に必要な対応する自由度によって決定されます。
オールド・フェイスフル間欠泉の例で$\beta_1$の95%信頼区間を手動で見つける方法を説明しますが、後でPythonで`statsmodels`を使用して直接行う方法を示します。
まず、$n = 114$なので、自由度は$n-2 = 112$です。自由度112の$t$分布に基づく95%信頼区間の臨界値は$q = 1.981$です。次に、先ほど求めた推定量$b_0$、$b_1$、および$s$を確認します：

```python
# 回帰係数と標準偏差
print("b0 (切片):", round(model_1.params[0], 3))
print("b1 (傾き):", round(model_1.params[1], 3))
print("s (残差標準偏差):", round(model_1.mse_resid**0.5, 3))
```

次に、以前に提示した式を使用して$b_1$の標準誤差を計算します：

```python
# b1の標準誤差を計算
x = old_faithful_2024['duration']
s = model_1.mse_resid**0.5
denom_se_b1 = np.sqrt(np.sum((x - x.mean())**2))
se_b1 = s / denom_se_b1
print("b1の標準誤差:", round(se_b1, 3))
```

最後に、$\beta_1$の95%信頼区間は次のように与えられます：

```python
# 自由度と臨界値
df = len(old_faithful_2024) - 2
q = stats.t.ppf(0.975, df)  # 95%信頼区間のt分布臨界値

# b1の95%信頼区間を計算
b1 = model_1.params[1]
lb1 = b1 - q * se_b1
ub1 = b1 + q * se_b1
print(f"b1の95%信頼区間: ({lb1:.3f}, {ub1:.3f})")
```

母集団傾き$\beta_1$の値がこの区間内にある95%の確信があります。

$\beta_0$の95%信頼区間の構築も、$b_0$、$SE(b_0)$、および$t$分布の自由度が同じ$n-2$であるため、同じ臨界値$q$を使用して同じステップに従います：

```python
# b0の標準誤差を計算
se_b0 = s * np.sqrt(1/len(old_faithful_2024) + x.mean()**2/np.sum((x - x.mean())**2))
print("b0の標準誤差:", round(se_b0, 3))

# b0の95%信頼区間を計算
b0 = model_1.params[0]
lb0 = b0 - q * se_b0
ub0 = b0 + q * se_b0
print(f"b0の95%信頼区間: ({lb0:.3f}, {ub0:.3f})")
```

信頼区間の結果は、線形モデルの仮定が満たされている場合にのみ有効です。
これらの仮定については「モデル適合度」セクションで説明します。

### 母集団傾きの仮説検定

$\beta_1$の仮説検定を行うために、両側検定の一般的な定式化は

$$\begin{aligned}
H_0: \beta_1 = B\\
H_A: \beta_1 \ne B
\end{aligned}$$

ここで$B$は$\beta_1$の仮説値です。第8章で導入した仮説検定に関連する用語、表記、定義を思い出してください。
*仮説検定*は2つの競合する仮説間のテストで構成されています：(1) *帰無仮説*$H_0$対(2) *対立仮説*$H_A$。

#### 検定統計量

*検定統計量*は、仮説検定に使用される点推定量です。ここでは、*t検定統計量*は

$$t = \frac{b_1 - B}{SE(b_1)}$$ 

この検定統計量は、帰無仮説の下で、自由度$n-2$の$t$分布に従います。
特に有用な検定は、説明変数と応答の間に線形関連があるかどうかを検定することであり、これは次の検定と同等です：

$$\begin{aligned}
H_0: \beta_1 = 0\\
H_A: \beta_1 \ne 0
\end{aligned}$$

例えば、オールド・フェイスフル間欠泉の噴火の持続時間（`duration`）と次の噴火までの待ち時間（`waiting`）の間に線形関係があるかどうかを判断するためにこの検定を使用できます。
*帰無仮説*$H_0$は、母集団傾き$\beta_1$が0であると仮定します。
これが真であれば、`duration`と`waiting`時間の間に*線形関係がない*ことになります。
仮説検定を行う際、帰無仮説$H_0: \beta_1 = 0$が真であると仮定し、観測されたデータに基づいてそれに対する証拠を見つけようとします。

一方、*対立仮説*$H_A$は、母集団傾き$\beta_1$がゼロではないことを述べており、噴火の`duration`が長いと次の噴火までの`waiting`時間が長くなるか短くなる可能性があることを意味します。
これは、説明変数と応答の間に正または負の線形関係があることを示唆しています。
帰無仮説に対する証拠がこの文脈でどちらの方向でも発生する可能性があるため、これを*両側*検定と呼びます。
この問題の*t検定*統計量は次のように与えられます：

```python
# t検定統計量を計算
t_stat = b1 / se_b1
print("t検定統計量:", round(t_stat, 3))
```

#### p値

第8章で導入した仮説検定に関連する用語、表記、定義を思い出してください。
$p$値の定義は、*帰無仮説*$H_0$が真であると*仮定して*、観測されたものと同様またはより極端な検定統計量を得る確率です。
直感的に$p$値は、`duration`と`waiting`時間の間に関係がないと仮定した場合、推定された傾き（$b_1$）がどれだけ「極端」であるかを定量化すると考えることができます。

両側検定の場合、例えば検定統計量が$t = 2$であれば、$p$値は$t$曲線の$-2$の左側と$2$の右側の面積として計算されます：

```python
# p値を計算
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
print("p値:", p_value)
```

オールド・フェイスフル間欠泉の噴火の例では、$H_0: \beta_1 = 0$の検定に対する検定統計量は$t = 19.765$でした。
$p$値は非常に小さいため、Pythonは単純にゼロに等しいと表示します。

#### 解釈

第8章で概説した仮説検定手順に従うと、$p$値は実際上0であるため、どのような有意水準$\alpha$を選択しても、$H_0$を$H_A$の有利に棄却することになります。
言い換えると、`duration`と`waiting`時間の間に線形関連がないと仮定すると、ランダムサンプルを使用して得られたものと同じくらい極端な傾きを観測する確率は実際上ゼロでした。
結論として、`duration`と`waiting`時間の間に線形関係がないという帰無仮説を棄却します。
これらの変数間に線形関係があると結論づけるのに十分な統計的証拠があります。

### Pythonでの回帰テーブル

第10章で議論した最小二乗推定量、標準誤差、検定統計量、$p$値、および信頼区間の境界はすべて、`statsmodels`ライブラリを使用してPythonで一度に計算できます。
`model_1`に対して、出力は次のようになります：

```python
# 回帰テーブルを表示
print(model_1.summary().tables[1])

# 95%信頼区間を表示
print("\n95%信頼区間:")
print(model_1.conf_int())
```

テーブルの最初の行は切片$\beta_0$に関する推論を扱い、2行目は傾き$\beta_1$に関する推論を扱っていることに注意してください。
テーブルのヘッダーは推論のための情報を提示しています：

- `coef`列には最小二乗推定量、$b_0$（1行目）と$b_1$（2行目）が含まれています。
- `std err`には$SE(b_0)$と$SE(b_1)$（$b_0$と$b_1$の標準誤差）がそれぞれ含まれています。
- `t`列には$b_0$（1行目）と$b_1$（2行目）の$t$検定統計量が含まれています。
  $b_1$に焦点を当てると、$t$検定統計量は方程式$t = \frac{b_1 - 0}{SE(b_1)} = 19.765$を使用して構築されました。
  これは仮説$H_0: \beta_1 = 0$対$H_A: \beta_1 \ne 0$に対応しています。
- `P>|t|`は、帰無仮説が真であると仮定して、観測されたものと同様またはより極端な検定統計量を得る確率です。
  この仮説検定では、$t$検定統計量は19.765に等しく、したがって$p$値はほぼゼロであり、対立仮説の有利に帰無仮説を棄却することを示唆しています。
- `[0.025 0.975]`の値は$\beta_1$の95%信頼区間の下限と上限です。

これらの量の概念的枠組みと詳細な説明については、前のサブセクションを参照してください。

### モデル適合度とモデルの仮定

線形モデルとその要素に関する多くの仮定とともにモデルを導入し、これが応答と説明変数の間の関係を適切に表現していると仮定してきました。
実際のアプリケーションでは、関係が線形モデルによって適切に記述されているかどうか、または導入したすべての仮定が満たされているかどうかは不確かです。

もちろん、この章で説明した線形モデル、または他の任意のモデルが自然界で提示される現象の完全な表現であることは期待していません。
モデルは現実の単純化であり、問題の関係を正確に表現することを意図していませんが、この関係の理解を向上させるのに役立つ有用な近似を提供することを意図しています。
さらに、私たちは可能な限り単純でありながら、研究している自然現象の関連する特徴を捉えるモデルを望んでいます。
このアプローチは*倹約の原則*または*オッカムの剃刀*として知られています。

しかし、線形モデルのような単純なモデルでも、それがデータの関係を正確に表現しているかどうかを知りたいと思います。
これは*モデル適合度*と呼ばれます。
さらに、モデルの仮定が満たされているかどうかを判断したいと思います。

線形モデルでは4つの要素をチェックしたいと思います。
頭字語は、それぞれの部分から特定の文字を取って単語や言葉を形成する作文です。
これら4つの要素を覚えるために、次の頭字語を使用できます：

1. 変数間の関係の**L**inearity（線形性）
   - $i = 1, \dots, n$のそれぞれに対して、$y_i$と$x_i$の関係は本当に線形ですか？言い換えると、線形モデル$y_i = \beta_0 + \beta_1 \cdot x_i + \epsilon_i$は適切ですか？
2. 各応答値$y_i$の**I**ndependence（独立性）
   - 任意の$i \ne j$に対して$y_i$と$y_j$は独立ですか？
3. 誤差項の**N**ormality（正規性）
   - 誤差項の分布は少なくともおおよそ正規分布ですか？
4. $y_i$（および誤差項$\epsilon_i$）の分散の**E**quality（等質性）または定常性
   - 応答$y_i$の分散、または同等に標準偏差は、予測値（$\widehat{y}_i$）または説明変数値（$x_i$）に関係なく常に同じですか？

この場合、私たちの頭字語は**LINE**という単語に従います。
これは線形回帰を使用する際にチェックすべき項目の覚えやすい手がかりとなります。
**L**inearity、**N**ormality、**E**qual or constant varianceをチェックするために、次のサブセクションで説明するように*残差診断*を通じて線形回帰の残差を使用します。
**I**ndependenceをチェックするために、データが時系列または他の種類の配列を使用して収集された場合、残差を使用できます。
そうでなければ、ランダムサンプルを取ることで独立性が達成される可能性があり、これにより順序依存性のタイプが排除されます。

残差の計算方法を復習することから始め、可視化を通じて残差診断を導入し、オールド・フェイスフル間欠泉の例を使用して4つの**LINE**要素がそれぞれ満たされているかどうかを判断し、その意味を議論します。

#### 残差

$n$ペアの$(x_1, y_1), \dots, (x_n,y_n)$のランダムサンプルが与えられたことを思い出してください。線形回帰は次のように与えられました：

$$\widehat{y}_i = b_0 + b_1 \cdot x_i$$ 

すべての観測値$i = 1, \dots, n$に対して。
第5章で定義した残差は、*観測された応答*から*予測値*を引いたものです。
残差を文字$e$で表すと次のようになります：

$$e_i = y_i - \widehat{y}_i$$ 

$i = 1, \dots, n$に対して。
これら2つの式を組み合わせると、

$$y_i = \underline{\widehat{y}_i} + e_i = \underline{b_0 + b_1 \cdot x_i} + e_i$$

結果の式は線形モデルと非常に似ています：

$$y_i = \beta_0 + \beta_1 \cdot x_i + \epsilon_i$$

この文脈では、残差は誤差項の粗い推定と考えることができます。
線形モデルの多くの仮定は誤差項に関連しているため、残差を研究することでこれらの仮定をチェックできます。

オールド・フェイスフル間欠泉の噴火の特定の残差を視覚化します：

```python
# 特定の点を選ぶ
index = old_faithful_2024[(old_faithful_2024['duration'] == 211) & (old_faithful_2024['waiting'] == 178)].index[0]

# 選んだ点の実測値、予測値、残差を取得
x = old_faithful_2024.loc[index, 'duration']
y = old_faithful_2024.loc[index, 'waiting']
y_hat = model_1.predict(old_faithful_2024)[index]
resid = y - y_hat

# 残差をプロット
plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration', y='waiting', data=old_faithful_2024, alpha=0.3)
plt.plot(old_faithful_2024['duration'], model_1.predict(old_faithful_2024), 
         color='blue', linewidth=0.5)
plt.scatter(x, y, color='red', s=100, zorder=5)  # 観測値
plt.scatter(x, y_hat, color='red', marker='s', s=100, zorder=5)  # 予測値
plt.arrow(x, y, 0, y_hat-y, color='blue', width=0.5, head_width=3, 
          head_length=2, length_includes_head=True, zorder=5)
plt.xlabel('持続時間')
plt.ylabel('待ち時間')
plt.title('持続時間と待ち時間の関係')
plt.tight_layout()
plt.show()
```

`model_1`回帰モデルに`predict`メソッドを適用して、すべての$n = 114$個の残差を計算できます。
結果の`residual`値が`waiting - waiting_hat`とほぼ等しいことに注意してください（四捨五入誤差による若干の違いがある可能性があります）。

```python
# 予測値と残差を計算
old_faithful_2024['waiting_hat'] = model_1.predict(old_faithful_2024)
old_faithful_2024['residual'] = old_faithful_2024['waiting'] - old_faithful_2024['waiting_hat']

# 最初の数行を表示
old_faithful_2024[['duration', 'waiting', 'waiting_hat', 'residual']].head()
```

#### 残差診断

*残差診断*は条件**L**、**N**、**E**を検証するために使用されます。
より洗練された統計的アプローチが使用できますが、ここではデータの可視化に焦点を当てます。最も有用なプロットの1つは*残差プロット*で、これは残差と予測値の散布図です。
計算した`waiting_hat`と`residual`を使用して散布図を描きます：

```python
# 残差プロット
plt.figure(figsize=(10, 6))
sns.scatterplot(x='waiting_hat', y='residual', data=old_faithful_2024, alpha=0.6)
plt.axhline(y=0, color='blue', linestyle='-')
plt.xlabel('予測値 (waiting_hat)')
plt.ylabel('残差')
plt.title('予測値と残差の散布図')
plt.tight_layout()
plt.show()
```

元のデータの散布図と残差プロットを並べて表示します：

```python
# 元のデータと残差プロットを並べて表示
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 散布図
sns.scatterplot(x='duration', y='waiting', data=old_faithful_2024, 
                alpha=0.6, ax=axes[0])
sns.regplot(x='duration', y='waiting', data=old_faithful_2024, 
            scatter=False, ci=None, line_kws={'color': 'blue', 'linewidth': 0.5},
            ax=axes[0])
axes[0].set_xlabel('持続時間')
axes[0].set_ylabel('待ち時間')

# 残差プロット
sns.scatterplot(x='waiting_hat', y='residual', data=old_faithful_2024, 
                alpha=0.6, ax=axes[1])
axes[1].axhline(y=0, color='blue', linestyle='-')
axes[1].set_xlabel('予測値 (waiting_hat)')
axes[1].set_ylabel('残差')

plt.tight_layout()
plt.show()
```

左の図では、残差は観測された応答と線形回帰の間の垂直距離によって決定されることに注意してください。
右の図（残差）では、線形回帰の効果を取り除き、各点からゼロ線（y軸）までの垂直距離として残差の効果が見られます。
この残差プロットを使用すると、次に説明するようにモデルの仮定に反するパターンやトレンドをより簡単に発見できます。

##### 関係の線形性

応答$y_i$と説明変数$x_i$の関連が**L**inearであるかどうかをチェックしたいと思います。
モデルの誤差項により、予測値と残差の散布図にいくつかのランダムな変動が示されることが予想されますが、変動は任意の方向に系統的であってはならず、トレンドは非線形パターンを示してはなりません。

残差のばらつき（y軸）が任意の予測値（x軸）に対してほぼ同じで、点がゼロ線の上下にほぼ同じくらい位置しているように見える、残差と予測値の散布図でパターンが示されていないものは*ヌル*プロットと呼ばれます。
予測値または説明変数に対する残差のプロットが*ヌル*プロットである場合、モデルの仮定に対する証拠は示されません。
言い換えると、線形モデルが適切であることを望むなら、残差と予測値をプロットするときに*ヌル*プロットが見られることを期待します。

これはオールド・フェイスフル間欠泉の例では大部分当てはまります。予測値（`waiting_hat`）に対する残差を示す図の右側を見ると、完全にランダムな配置ではなく点のいくつかのクラスターが存在するように見えますが、はっきりとした系統的なトレンドや非線形関係の出現はありません。
したがって、このプロットに基づいて、データは線形性の仮定に違反していないと考えられます。

対照的に、`waiting`と`duration`の散布図とそれに関連する残差プロットが次のようであると仮定してみましょう（シミュレーションデータを使用）：

```python
# 非線形関係のシミュレーション
np.random.seed(76)
x = old_faithful_2024['duration'].values
y_nonlinear = 150 + ((x/2 - np.min(old_faithful_2024['duration'])) * 
                      (x/2 - np.max(old_faithful_2024['duration']))) / \
              (np.max(old_faithful_2024['duration']) - np.min(old_faithful_2024['duration'])) * \
              (-1/2) + np.random.normal(0, 1.5, len(x))

# シミュレーションデータを含むデータフレーム
data_aux = pd.DataFrame({
    'x': x,
    'y': y_nonlinear
})

# 線形モデルを当てはめる
model_nonlinear = smf.ols('y ~ x', data=data_aux).fit()
data_aux['y_hat'] = model_nonlinear.predict(data_aux)
data_aux['residual'] = data_aux['y'] - data_aux['y_hat']

# 散布図と残差プロットを並べて表示
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 散布図
sns.scatterplot(x='x', y='y', data=data_aux, alpha=0.6, ax=axes[0])
sns.regplot(x='x', y='y', data=data_aux, scatter=False, ci=None, 
            line_kws={'color': 'blue', 'linewidth': 0.5}, ax=axes[0])
axes[0].set_xlabel('持続時間')
axes[0].set_ylabel('待ち時間')

# 残差プロット
sns.scatterplot(x='y_hat', y='residual', data=data_aux, alpha=0.6, ax=axes[1])
axes[1].axhline(y=0, color='blue', linestyle='-')
axes[1].set_xlabel('予測値 (waiting_hat)')
axes[1].set_ylabel('残差')

plt.tight_layout()
plt.show()
```

散布図と回帰直線（左の図）を一見すると、回帰直線が関係の適切な要約であると考えがちですが、よく見ると、`duration`の低い値に対する残差は主に回帰直線の下にあり、`duration`の中間範囲の値に対する残差は主に回帰直線の上にあり、`duration`の大きな値に対する残差は再び回帰直線の下にあることに気づくかもしれません。

これが、回帰の効果を取り除いて残差に完全に焦点を当てることができる予測値に対する残差のプロット（右の図）を使用する理由です。
点は明らかに直線を形成しておらず、むしろU字型の多項式曲線を形成しています。
これが実際に観測されたデータである場合、これらのデータで線形回帰を使用すると、有効または適切でない結果が生じます。

##### 誤差項と応答の独立性

チェックしたいもう一つの仮定は、応答値の**I**ndependenceです。
それらが独立していない場合、観測されたデータに依存性のパターンが現れる可能性があります。

残差も、誤差項の粗い近似であるため、この目的に使用できます。
データが時系列または他の種類の配列で収集された場合、残差を時間に対してプロットすることで独立性の欠如を判断するのに役立つ場合もあります。
オールド・フェイスフル間欠泉の噴火の例では、使用できる時間コンポーネントがあります：`old_faithful_2024`データセットには`date`変数が含まれています。
`residuals`と`date`（時間）のプロットを表示します：

```python
# 日付と残差のプロット
plt.figure(figsize=(10, 6))
sns.scatterplot(x='date', y='residual', data=old_faithful_2024)
plt.xlabel('日付')
plt.ylabel('残差')
plt.title('日付と残差の関係')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

時間（`date`）に対する残差のプロットはヌルプロットのように見えます。
このプロットに基づいて、残差は依存性の証拠を示していないと言えます。

このデータセットの観測値は、この期間中に発生するすべてのオールド・フェイスフル間欠泉の噴火のサブセットに過ぎず、そのほとんどまたはすべては次々と連続して発生する噴火ではありません。
このデータセットの各観測値は、オールド・フェイスフルの一意の噴火を表し、`waiting`時間と`duration`がそれぞれの事象に対して別々に記録されています。
これらの噴火は互いに独立して発生するため、`waiting`対`duration`の回帰から導かれる残差も独立していることが期待されます。
「サンプル回帰推論」で説明したように、これをランダムサンプルと考えることができます。

この場合、独立性の仮定は許容できると思われます。
`old_faithful_2024`データには依存性の問題につながる可能性のある反復測定やグループ化された観測値が含まれていないことに注意してください。
したがって、誤差項が互いに系統的に関連していないと信じて回帰分析を進めることができます。
特に時系列または他の配列測定が含まれていない場合、独立性の欠如を判断することは簡単ではない場合がありますが、ランダムサンプルを取ることがゴールデンスタンダードです。

##### 誤差項の正規性

チェックしたい3番目の仮定は、誤差項が期待値がゼロの**N**ormal分布に従うかどうかです。
誤差項の値の粗い推定値として残差を使用すると、残差が時にはプラスで時にはマイナスであることがわかります。
*平均的に*、誤差がゼロに等しく、その分布の形がベル型曲線に近似するかどうかを確認したいと思います。

残差の分布を可視化するためにヒストグラムを使用できます：

```python
# 残差のヒストグラム
plt.figure(figsize=(10, 6))
sns.histplot(old_faithful_2024['residual'], bins=20, kde=True)
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('残差')
plt.ylabel('頻度')
plt.title('残差のヒストグラム')
plt.tight_layout()
plt.show()
```

また、*量的量的*プロットまたは*QQプロット*を使用できます。
QQプロットは、残差の分位数（またはパーセンタイル）を正規分布の分位数に対する散布図を作成します。
残差がおおよそ正規分布に従う場合、散布図は45度の直線になります。
オールド・フェイスフル間欠泉の例のQQプロットを描くには：

```python
# QQプロット
plt.figure(figsize=(10, 6))
sm.qqplot(old_faithful_2024['residual'], line='45', fit=True)
plt.title('残差のQQプロット')
plt.tight_layout()
plt.show()
```

残差のヒストグラムと正規分布曲線を含むヒストグラム（左の図）とQQプロット（右の図）の両方を含めます：

```python
# 残差のヒストグラムとQQプロットを並べて表示
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ヒストグラム
sns.histplot(old_faithful_2024['residual'], bins=20, kde=True, ax=axes[0])
axes[0].axvline(x=0, color='red', linestyle='--')
axes[0].set_xlabel('残差')
axes[0].set_ylabel('頻度')
axes[0].set_title('残差のヒストグラム')
axes[0].set_xlim(-50, 50)

# QQプロット
sm.qqplot(old_faithful_2024['residual'], line='45', fit=True, ax=axes[1])
axes[1].set_title('残差のQQプロット')

plt.tight_layout()
plt.show()
```

残差のヒストグラムは完全には正規分布ではなく、中心のすぐ右側に最も高いビン値が現れるなどのいくつかの偏差があります。
しかし、ヒストグラムは正規性からあまりにも離れているようには見えません。
QQプロット（右の図）はこの結論を支持しています。
散布図は45度の線上に正確にはありませんが、それからあまり逸脱していません。

これらの結果を、明らかに正規性に従わないシミュレーションで見つかった残差と比較します：

```python
# 非正規分布の残差をシミュレーション
np.random.seed(3)
non_normal_residuals = (np.random.normal(0, s, size=len(old_faithful_2024))**2/40 - 
                        np.random.normal(0, s, size=len(old_faithful_2024)).mean() - 10)

# 残差のヒストグラムとQQプロットを並べて表示
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ヒストグラム
sns.histplot(non_normal_residuals, bins=20, kde=True, ax=axes[0])
axes[0].axvline(x=0, color='red', linestyle='--')
axes[0].set_xlabel('残差')
axes[0].set_ylabel('頻度')
axes[0].set_title('非正規分布の残差のヒストグラム')
axes[0].set_xlim(-50, 50)

# QQプロット
sm.qqplot(non_normal_residuals, line='45', fit=True, ax=axes[1])
axes[1].set_title('非正規分布の残差のQQプロット')

plt.tight_layout()
plt.show()
```

右側の明らかに非正規の残差を生じるモデルの場合、回帰の推論の結果は有効ではないでしょう。

##### 誤差に対する分散の等質性

最後にチェックすべき仮定は、すべての予測値または説明変数値にわたる誤差項の分散の**E**qualityまたは定常性です。
分散の定常性は*等分散性*としても知られています。残差を再び誤差項の粗い推定値として使用して、残差の分散が任意の予測値$\widehat{y}_i$または説明変数$x_i$に対して同じであるかどうかをチェックします。
先ほど予測値に対する残差の散布図を示しました。
説明変数`duration`に対する残差の散布図も作成できます：

```python
# 説明変数に対する残差プロット
plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration', y='residual', data=old_faithful_2024, alpha=0.6)
plt.axhline(y=0, color='blue', linestyle='-')
plt.xlabel('持続時間')
plt.ylabel('残差')
plt.title('持続時間と残差の関係')
plt.tight_layout()
plt.show()
```

x軸のスケールの変更を除いて、予測値（$\widehat{y}_i$）または説明変数値（$x_i$）のいずれかに対して残差（$e_i$）のプロットを作成することは（可視化の目的で）同等です。
これは、予測値が説明変数値の線形変換、$\widehat{y}_i = b_0 + b_1\cdot x_i$であるために発生します。

異なる`duration`値に対する残差の垂直分散または広がりを観察してください：

- `duration`が100〜150秒の値の場合、残差値はおよそ-25〜40の間にあり、約65単位の広がりがあります。
- `duration`が150〜200秒の値の場合、観測値はわずかしかなく、広がりが何であるかは明確ではありません。
- `duration`が200〜250秒の値の場合、残差値はおよそ-37〜32の間にあり、約69単位の広がりがあります。
- `duration`が250〜300秒の値の場合、残差値はおよそ-42〜27の間にあり、約69単位の広がりがあります。

広がりは`duration`のすべての値に対して完全に一定ではありません。
`duration`の値が大きいほどわずかに大きくなるように見えますが、`duration`の値が高いほど観測数も多いように見えます。
また、2つまたは3つの点のクラスターがあり、残差の分散は完全に均一ではないことにも注意してください。
残差プロットが完全にヌルプロットではありませんが、等分散性の仮定に対する明確な証拠はありません。

実際のデータを扱う際にこのようなプロットを見ることは驚くべきことではありません。
残差プロットが完全に*ヌル*プロットではない可能性があります。これは、モデルを改善できる情報が不足している可能性があるためです。
例えば、モデルに別の説明変数を含めることができます。
`duration`と`waiting`時間の関係を近似するために線形モデルを使用していることを忘れないでください。このモデルがこの関係を完全に記述することは期待していません。
これらのプロットを見るとき、データが使用されている仮定を満たしていないという明確な証拠を見つけようとしています。
この例では、一定分散の仮定に違反しているとは思われません。

シミュレーションデータを使用した非定常分散の例を示します：

```python
# 非等分散性をシミュレーション
np.random.seed(76)
heteroscedastic_residuals = np.random.normal(0, 0.075 * x**2) * 0.4

# 説明変数と非等分散残差の散布図
plt.figure(figsize=(10, 6))
sns.scatterplot(x=x, y=heteroscedastic_residuals)
plt.axhline(y=0, color='blue', linestyle='-', linewidth=0.5)
plt.xlabel('持続時間')
plt.ylabel('残差')
plt.title('説明変数と非等分散残差の関係')
plt.tight_layout()
plt.show()
```

説明変数の値が増加するにつれて残差の広がりが増加することに注目してください。
一定分散の欠如は*異分散性*としても知られています。
異分散性が存在する場合、最小二乗推定量の標準誤差、信頼区間、または関連する仮説検定の結論などの結果の一部は有効ではありません。

##### 結論は何か？

モデルの仮定のいずれに対しても決定的な証拠は見つかりませんでした：

1. 変数間の関係の**L**inearity（線形性）
2. 誤差項の**I**ndependence（独立性）
3. 誤差項の**N**ormality（正規性）
4. 分散の**E**quality（等質性）または定常性

これは、私たちのモデルが完全に適切であったことを意味するものではありません。
例えば、残差プロットはヌルプロットではなく、モデルでは説明できない点のいくつかのクラスターがありました。
しかし全体的に、仮定の明確な違反と見なされるトレンドはなく、このモデルから得られる結論は有効である可能性があります。

**モデルの仮定が満たされていない場合はどうすればよいですか？**

モデルの仮定に明確な違反がある場合、見つかったすべての結果は疑わしいかもしれません。
さらに、モデルを改善するためにいくつかの対策を取ることができます。
これらの対策はこの本の範囲を超えているため、ここで詳細に扱うことはありませんが、将来の参考のために潜在的な解決策について簡単に説明します。

変数間の関係の**L**inearityが満たされていない場合、説明変数、応答、またはその両方の単純な変換が問題を解決する可能性があります。そうでない場合、*スプライン回帰*、*一般化線形モデル*、または*非線形モデル*などの代替手法を使用してこれらの状況に対処できます。追加の説明変数が利用可能な場合、*多重線形回帰*のように他の説明変数を含めるとより良い結果が得られる場合があります。

**I**ndependenceの仮定が満たされていないが、依存性が手元のデータ内の変数によって確立されている場合、*線形混合効果モデル*も使用できます。これらのモデルは*階層*または*マルチレベルモデル*とも呼ばれることがあります。

誤差項の**N**ormalityの仮定からの小さな逸脱はあまり心配する必要はなく、信頼区間や仮説検定に関連するものを含むほとんどの結果は引き続き有効である可能性があります。一方、正規性の仮定に対する違反が多い場合、多くの結果は有効でなくなる可能性があります。ここで提案した高度な方法を使用すると、これらの問題も修正できる場合があります。

分散の**E**qualityまたは定常性が満たされていない場合、それらの重みを既知にする関連情報が利用可能であれば、個々の観測値に重みを追加して分散を調整することが可能かもしれません。この方法は*重み付き線形回帰*または*重み付き最小二乗法*と呼ばれ、私たちが研究したモデルの直接の拡張です。重みに関する情報が利用できない場合、モデル内の分散の内部構造の推定量を提供するいくつかの方法を使用できます。これらの方法の中で最も一般的なものの1つは*サンドイッチ推定量*と呼ばれています。

モデルの仮定が満たされているかどうかをチェックすることは、回帰分析の重要なコンポーネントです。信頼区間の構築と解釈、仮説検定の実施、仮説検定の結果からの結論の提供は、仮定が満たされているかどうかによって直接影響を受けます。同時に、回帰分析では、プロットを可視化して解釈する際に主観性のレベルが存在することがよくあり、時には難しい統計的決定に直面することもあります。

では、どうすればよいでしょうか？私たちは透明性と結果を伝える際の明確さを提案します。関連する仮定からの逸脱を示唆する重要な要素を強調し、適切な結論を提供することが重要です。このようにして、分析の利害関係者はモデルの欠点を認識し、彼らに提示された結論に同意するかどうかを決定できます。

## 単純線形回帰のシミュレーションに基づく推論

このセクションでは、第7章と第8章で前に学んだシミュレーションベースの方法を使用して、回帰テーブルの値を再作成します。
特に、以下のためにPythonで`pingouin`パッケージと`scipy`パッケージを使用します：

- ブートストラップリサンプリングによる母集団傾き$\beta_1$の95%信頼区間の構築。これは、第7章でアーモンドデータと`mythbusters_yawn`データで行いました。
- 並べ替えテストを使用した$H_0: \beta_1 = 0$対$H_A: \beta_1 \neq 0$の仮説検定の実施。これは、第8章で`spotify_sample`データと`movies_sample`IMDbデータで行いました。

### `pingouin`を使用した母集団傾きの信頼区間

単一のサンプルの噴火データを使用して$\beta_1$の95%信頼区間を構築します。これを行うために、以下のステップを実行します：

1. オールド・フェイスフル間欠泉のデータを使用して、ブートストラップ法により傾き$b_1$のブートストラップ分布を作成します。
2. このブートストラップ分布を使用して、パーセンタイル法と（適切であれば）標準誤差法を用いて95%信頼区間を構築します。

重要なのは、ブートストラップリサンプリングが行単位で行われることです。つまり、元の`waiting`と`duration`の値のペアは常に一緒に保持されますが、異なるペアが複数回リサンプリングされる可能性があります。得られる信頼区間は、オールド・フェイスフル噴火の待ち時間と持続時間の関係を定量化する未知の母集団傾き$\beta_1$の妥当な値の範囲を示します。

まず、Pythonで傾き$b_1$のブートストラップ分布を構築しましょう：

```python
# ブートストラップ分布を作成するための関数
def bootstrap_slope(data, n_resamples=1000):
    # 結果を格納するための配列
    bootstrap_slopes = np.empty(n_resamples)
    
    # データの行数
    n = len(data)
    
    # ブートストラップリサンプリング
    for i in range(n_resamples):
        # 行単位でリサンプリング
        indices = np.random.choice(n, n, replace=True)
        resample = data.iloc[indices]
        
        # リサンプルに対して回帰を実行
        model = smf.ols('waiting ~ duration', data=resample).fit()
        
        # 傾きを保存
        bootstrap_slopes[i] = model.params[1]
    
    return bootstrap_slopes

# シードを設定して再現性を確保
np.random.seed(76)

# ブートストラップ分布を作成
bootstrap_distn_slope = bootstrap_slope(old_faithful_2024, n_resamples=1000)

# ブートストラップ分布の最初の数値を表示
print("ブートストラップ分布の最初の10値:")
print(bootstrap_distn_slope[:10])
```

ブートストラップで得られた1000個の傾き$b_1$の値を可視化しましょう：

```python
# ブートストラップ分布を可視化
plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_distn_slope, kde=True)
plt.axvline(x=model_1.params[1], color='red', linestyle='--', 
            label=f'観測された傾き: {model_1.params[1]:.3f}')
plt.xlabel('傾き (b1)')
plt.ylabel('頻度')
plt.title('b1のブートストラップ分布')
plt.legend()
plt.tight_layout()
plt.show()
```

ブートストラップ分布が概ね釣鐘型であることに注目してください。第7章で説明したように、$b_1$のブートストラップ分布の形状は$b_1$のサンプリング分布の形状に近似します。

#### パーセンタイル法

まず、パーセンタイル法を使用して$\beta_1$の95%信頼区間を計算しましょう。中央の95%の値を含む2.5パーセンタイルと97.5パーセンタイルを特定します。この方法はブートストラップ分布が正規形である必要がないことを思い出してください。

```python
# パーセンタイル法によるb1の95%信頼区間
percentile_ci = np.percentile(bootstrap_distn_slope, [2.5, 97.5])
print(f"パーセンタイル法によるb1の95%信頼区間: ({percentile_ci[0]:.3f}, {percentile_ci[1]:.3f})")
```

これが$\beta_1$のパーセンタイルベースの95%信頼区間です。

#### 標準誤差法

ブートストラップ分布が概ね釣鐘型であるため、標準誤差法を使用して$\beta_1$の95%信頼区間を構築することもできます。

これを行うには、最初に観測された傾き$b_1$を計算する必要があります。これは標準誤差ベースの信頼区間の中心として機能します。回帰テーブルでこれが$b_1$ = 0.37であることがわかりましたが、計算して確認しましょう：

```python
# 観測された傾きb1
observed_slope = model_1.params[1]
print(f"観測された傾き b1: {observed_slope:.3f}")

# 標準誤差法によるb1の95%信頼区間
se = np.std(bootstrap_distn_slope, ddof=1)
se_ci_lower = observed_slope - 1.96 * se
se_ci_upper = observed_slope + 1.96 * se
print(f"標準誤差法によるb1の95%信頼区間: ({se_ci_lower:.3f}, {se_ci_upper:.3f})")
```

標準誤差法に基づく$\beta_1$の95%信頼区間はパーセンタイルベースの信頼区間と若干異なります。どちらの信頼区間も0を含まず、完全に0より上にあることに注目してください。これはオールド・フェイスフルの噴火の待ち時間と持続時間の間に意味のある正の関係があることを示唆しています。

### `pingouin`を使用した母集団傾きの仮説検定

次に、$H_0: \beta_1 = 0$対$H_A: \beta_1 \neq 0$の仮説検定を行いましょう。第8章の「一つのテストしかない」ダイアグラムに従う仮説検定のパラダイムを使用します。

まず、帰無仮説$H_0$で仮定されている通り$\beta_1$がゼロであることの意味を考えてみましょう。$\beta_1 = 0$であれば、待ち時間と持続時間の間に関係がないと言っていることになります。この特定の帰無仮説$H_0$を仮定することは、私たちの「仮説の世界」では`waiting`と`duration`の間に関係がないことを意味します。したがって、`waiting`変数をシャッフル/並べ替えても問題はありません。

並べ替えテストを実行して、傾きの検定統計量$b_1$の帰無分布を構築します。第8章の仮説検定に関連する用語、表記、定義から、*帰無分布*は帰無仮説$H_0$が真であると仮定した場合の検定統計量$b_1$のサンプリング分布であることを思い出してください。

```python
# 並べ替えテストの関数
def permutation_test_slope(data, n_permutations=1000):
    # 結果を格納するための配列
    null_slopes = np.empty(n_permutations)
    
    # 元のX変数とY変数
    X = data['duration'].values
    Y = data['waiting'].values
    
    # 観測されたデータに対する傾きを計算
    observed_model = smf.ols('waiting ~ duration', data=data).fit()
    observed_slope = observed_model.params[1]
    
    # 並べ替えテスト
    for i in range(n_permutations):
        # Y値をシャッフル
        Y_permuted = np.random.permutation(Y)
        
        # 並べ替えたデータで回帰を実行
        permuted_data = pd.DataFrame({'duration': X, 'waiting': Y_permuted})
        model = smf.ols('waiting ~ duration', data=permuted_data).fit()
        
        # 傾きを保存
        null_slopes[i] = model.params[1]
    
    return null_slopes, observed_slope

# シードを設定して再現性を確保
np.random.seed(76)

# 並べ替えテストを実行
null_distn_slope, observed_slope = permutation_test_slope(old_faithful_2024, n_permutations=1000)

# 帰無分布の最初の数値を表示
print("帰無分布の最初の10値:")
print(null_distn_slope[:10])
print(f"観測された傾き: {observed_slope:.3f}")
```

傾き$b_1$の帰無分布を可視化しましょう：

```python
# 帰無分布を可視化
plt.figure(figsize=(10, 6))
sns.histplot(null_distn_slope, kde=True)
plt.axvline(x=observed_slope, color='red', linestyle='--', 
            label=f'観測された傾き: {observed_slope:.3f}')
plt.xlabel('傾き (b1)')
plt.ylabel('頻度')
plt.title('帰無分布と観測された傾き')
plt.legend()
plt.tight_layout()
plt.show()
```

帰無分布が$b_1$ = 0を中心にしていることに注目してください。これは私たちの仮説の世界では、`waiting`と`duration`の間に関係がないため$\beta_1 = 0$だからです。したがって、シミュレーション全体で観察される最も典型的な傾き$b_1$は0です。さらに、この中心値の0の周りに変動があることにも注目してください。

観測された検定統計量$b_1$ = 0.37を帰無分布と比較して$p$値を可視化しましょう：

```python
# 両側p値を計算
p_value = np.mean(np.abs(null_distn_slope) >= np.abs(observed_slope))

# 帰無分布とp値を可視化
plt.figure(figsize=(10, 6))
sns.histplot(null_distn_slope, kde=True)
plt.axvline(x=observed_slope, color='red', linestyle='--', 
            label=f'観測された傾き: {observed_slope:.3f}')
# 観測値以上または-観測値以下の領域に色をつける
extreme_values = null_distn_slope[np.abs(null_distn_slope) >= np.abs(observed_slope)]
if len(extreme_values) > 0:
    sns.histplot(extreme_values, color='red', alpha=0.5)
plt.xlabel('傾き (b1)')
plt.ylabel('頻度')
plt.title(f'帰無分布とp値 (p = {p_value:.3f})')
plt.legend()
plt.tight_layout()
plt.show()

print(f"p値: {p_value}")
```

観測された傾き0.37がこの帰無分布から遠く離れて右側にあり、色付き領域が重ならないため、$p$値は0になります。数値的な$p$値も計算しましょう：

```python
# 両側p値を計算
p_value = np.mean(np.abs(null_distn_slope) >= np.abs(observed_slope))
print(f"p値: {p_value}")
```

これは回帰テーブルの$p$値0と一致します。したがって、帰無仮説$H_0: \beta_1 = 0$を対立仮説$H_A: \beta_1 \neq 0$の有利に棄却します。これによりオールド・フェイスフルの噴火の待ち時間と持続時間の値の間に有意な関係があることを示唆する証拠があります。

回帰に対する推論の条件が満たされ、帰無分布が釣鐘型である場合、先ほど示したシミュレーションベースの結果と回帰テーブルに示された理論ベースの結果の間で類似した結果が見られる可能性が高いです。

## 多重線形回帰モデル

### モデル

単純回帰から多重回帰モデルへの拡張について次に説明します。母集団には応答変数（$Y$）と2つ以上の説明変数（$X_1, X_2, \dots, X_p$）（$p \ge 2$）があると仮定します。
これらの変数間の*統計的線形関係*は次のように与えられます：

$$Y = \beta_0 + \beta_1 \cdot X_1 + \dots + \beta_p X_p + \epsilon$$ 

ここで$\beta_0$は母集団の切片、$\beta_j$は説明変数$X_j$に関連する母集団の偏回帰係数です。
誤差項$\epsilon$は直線によって説明されない$Y$の部分を考慮します。
単純な場合と同様に、期待値は$E(\epsilon) = 0$、標準偏差は$SD(\epsilon) = \sigma$、分散は$Var(\epsilon) = \sigma^2$であると仮定します。
分散と標準偏差は$X_1, X_2, \dots, X_p$の値に関係なく一定です。
この母集団から多くの観測値を取ると、誤差項は時にはゼロより大きく、時にはゼロより小さくなりますが、平均するとゼロになり、おおよそ$\sigma$単位だけゼロから離れています。

### 例：コーヒー品質評価スコア

単純線形回帰の場合と同様に、母集団のパラメータを推定するためにランダムサンプルを使用します。
これらの方法を説明するために、`moderndive`パッケージから`coffee_quality`データフレームを使用します。
このデータセットはCoffee Quality Instituteからのもので、10の異なる属性に基づくコーヒー評価スコアに関する情報が含まれています：`aroma`、`flavor`、`aftertaste`、`acidity`、`body`、`balance`、`uniformity`、`clean_cup`、`sweetness`、`overall`。
さらに、データフレームには`moisture_percentage`やコーヒーの`country`、`continent_of_origin`などの他の情報も含まれています。
これはランダムサンプルであると仮定できます。

```python
# コーヒー品質データをダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/coffee_quality.csv"
coffee_quality = pd.read_csv(url)

# 必要な変数を選択し、continent_of_originをfactor型に変換
coffee_data = coffee_quality[['aroma', 'flavor', 'moisture_percentage', 'continent_of_origin', 'total_cup_points']]

# continent_of_originをカテゴリ変数として扱う
coffee_data['continent_of_origin'] = pd.Categorical(coffee_data['continent_of_origin'])

# データの最初の10行を表示
coffee_data.head(10)
```

`total_cup_points`（応答変数）を数値説明変数`aroma`、`flavor`、`moisture_percentage`と4つのカテゴリを持つカテゴリ説明変数`continent_of_origin`（`Africa`、`Asia`、`North America`、`South America`）に回帰することを計画しています。

```python
# 数値変数の要約統計量
coffee_data.describe()
```

以下のコードでデータの可視化を作成します。多重回帰を行う際には、すべての変数ペアの組み合わせの散布図を含む散布図行列を構築すると便利です。Pythonでは、`seaborn`の`pairplot`関数を使用して散布図行列といくつかの有用な追加情報を生成できます：

```python
# 散布図行列を作成
plt.figure(figsize=(12, 12))
sns.pairplot(coffee_data, hue='continent_of_origin')
plt.suptitle('コーヒー変数の散布図行列', y=1.02)
plt.tight_layout()
plt.show()
```

まず、縦軸に応答（`total_cup_points`）がある図について説明します。図の最後の行にあるプロットです。
`total_cup_points`と`aroma`（最下行、一番左のプロット）または`total_cup_points`と`flavor`（最下行、左から2番目のプロット）をプロットすると、強い正の線形関係が観察されます。
`total_cup_points`と`moisture_percentage`のプロット（最下行、左から3番目のプロット）はあまり情報を提供せず、これらの変数は何らかの方法で関連していないように見えますが、ゼロ付近に`moisture_percentage`の外れ値が観察されます。
`total_cup_points`と`continent_of_origin`のプロット（最下行、左から4番目のプロット）は、それぞれのファクターレベルに対して4つのヒストグラムを示しています。4番目のグループは観測数が少ないように見えますが、分散が大きいようです。

これら2つの変数を接続する関連するボックスプロット（4行目、一番右のプロット）は、因子の最初のレベル（`Africa`）が他の3つよりも平均カップポイントが高いことを示唆しています。
`aroma`と`flavor`の散布図では強い正の線形関連が示唆されるなど、数値説明変数間の線形関連を見つけることも有用です。
対照的に、`moisture_percentage`と`aroma`または`flavor`のいずれかを観察しても、ほとんど関係は見られません。

この例での数値変数間の相関係数を確認しましょう：

```python
# 数値変数間の相関
correlations = coffee_data.select_dtypes(include=[np.number]).corr()
print(correlations)
```

相関係数は散布図行列を使用して得られた知見を裏付けています。
`total_cup_points`と`aroma`および`total_cup_points`と`flavor`の相関は正で1に近く、強い正の線形関連を示唆しています。
相関係数は関連が近似的に線形である場合に関連することを思い出してください。
`moisture_percentage`と他の変数の相関はゼロに近く、`moisture_percentage`が他の変数（応答または説明変数）と線形に関連している可能性が低いことを示唆しています。
また、`aroma`と`flavor`の相関は強い正の関連性という私たちの結論を支持します。

### 多重回帰の最小二乗法

3つの数値説明変数と4つのレベル（`Africa`、`Asia`、`North America`、`South America`）を持つ1つの因子（`continent_of_origin`）があることに注目してください。
線形モデルに因子レベルを導入するために、第6章で説明したようにダミー変数を使用して因子レベルを表現します。
必要なダミー変数は次のとおりです：

$$\begin{aligned}
D_1 &= \left\{
\begin{array}{ll}
1 & \text{原産地の大陸がアフリカの場合} \phantom{afdasfd} \\
0 & \text{それ以外の場合}\end{array}
\right.\\
D_2 &= \left\{
\begin{array}{ll}
1 & \text{原産地の大陸がアジアの場合}\phantom{asdfasdfa} \\
0 & \text{それ以外の場合}\end{array}
\right.\\
D_3 &= \left\{
\begin{array}{ll}
1 & \text{原産地の大陸が北アメリカの場合}\phantom{} \\
0 & \text{それ以外の場合}\end{array}
\right.\\
D_4 &= \left\{
\begin{array}{ll}
1 & \text{原産地の大陸が南アメリカの場合} \phantom{}\\
0 & \text{それ以外の場合}\end{array}
\right.\\
\end{aligned}$$

また、最初のレベルは切片でカウントされるため、最初のレベルを削除することを思い出してください。
単純線形の場合と同様に、応答と説明変数の間の線形性を仮定し、サンプル内の各観測に対して「モデル」サブセクションで説明した線形モデルを適用します。
モデルを$i$番目の観測に関して表現すると、次のようになります：

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3} + \beta_{02}D_{i2} + \beta_{03}D_{i3} + \beta_{04}D_{i4} + \epsilon_i
$$

ここで、サンプルの$i$番目の観測に対して、$x_{i1}$は`aroma`スコア、$x_{i2}$は`flavor`スコア、$x_{i3}$は`moisture_percentage`、$D_{i2}$は`Asia`のダミー変数、$D_{i3}$は`North America`のダミー変数、$D_{i4}$は`South America`のダミー変数を表します。
$i$はサンプル内の任意の観測を表す添え字であることを思い出してください。
あるいは、すべての観測に対してモデルを提示することもできます：

$$\begin{aligned}
y_1 
&= 
\beta_0 + \beta_1 x_{11} + \beta_2 x_{12} + \beta_3 x_{13}+ \beta_{02}D_{12} + \beta_{03}D_{13} + \beta_{04}D_{14} + \epsilon_1\\
y_2 
&= 
\beta_0 + \beta_1 x_{21} + \beta_2 x_{22} + \beta_3 x_{23}+ \beta_{02}D_{22} + \beta_{03}D_{23} + \beta_{04}D_{24} + \epsilon_2\\
& \phantom{  a}\vdots \\
y_n 
&= 
\beta_0 + \beta_1 x_{n1} + \beta_2 x_{n2} + \beta_3 x_{n3}+ \beta_{02}D_{n2} + \beta_{03}D_{n3} + \beta_{04}D_{n4} + \epsilon_n
\end{aligned}$$

多重回帰に適用される最小二乗法の拡張が続きます。
*残差の二乗和*を最小化する係数推定量を取得したいと考えています：

$$\sum_{i=1}^n \left[y_i - (\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3}+ \beta_{02}D_{i2} + \beta_{03}D_{i3} + \beta_{04}D_{i4} )\right]^2.$$

この最適化問題は単純線形の場合と同様で、微積分を使用して解かれます。
現在、より多くの方程式に対処する必要があります。コーヒーの例では7つの方程式で、それぞれが推定する必要のある係数推定量に接続されています。
この問題の解決策は行列と行列計算を使用して達成され、第6章および「モデル4インタラクションテーブル」サブセクションで導入された回帰係数です。
これらは*最小二乗推定量*と呼ばれます：$b_0$は$\beta_0$の最小二乗推定量、$b_1$は$\beta_1$の最小二乗推定量などです。

予測値、残差、分散（$s^2$）および標準偏差（$s$）の推定量は、単純線形の場合の直接的な拡張です。一般的な場合では、$p$個の説明変数に対して、予測値は

$$\widehat{y}_i = b_0 + b_1 x_{i1} + b_2 x_{i2} + \dots + b_p x_{ip},$$

残差は$e_i = y_i - \widehat{y}_i$、モデル分散推定量は

$$
s^2 = \frac{\sum_{i=1}^n \left(y_i - \widehat{y}_i\right)^2}{n-p}
$$

ここで$p$は係数の数です。これらの式をコーヒースコアの例に適用すると、予測値は

$$\widehat{y}_i = b_0 + b_1 x_{i1} + b_2 x_{i2} + b_3 x_{i3}+ b_{02}D_{i2} + b_{03}D_{i3} + b_{04}D_{i4}\,\,,$$

分散推定量は

$$\begin{aligned}
s^2 &= \frac{\sum_{i=1}^n \left[y_i - (b_0 + b_1 x_{i1} + b_2 x_{i2} + b_3 x_{i3}+ b_{02}D_{i2} + b_{03}D_{i3} + b_{04}D_{i4} )\right]^2}{n-7}\\
&= \frac{\sum_{i=1}^n \left(y_i - \widehat{y}_i\right)^2}{n-7},
\end{aligned}$$

標準偏差推定量は

$$
s = \sqrt{\frac{\sum_{i=1}^n \left(y_i - \widehat{y}_i\right)^2}{n-7}}.
$$

取られたランダムサンプルに依存する確率変数として最小二乗推定量を考える場合、これらの推定量の特性は単純線形の場合に提示された特性の直接的な拡張です：

- 最小二乗推定量はモデルのパラメータの不偏推定量です。例えば、$\beta_1$（`aroma`の偏回帰係数の推定量）を選択すると、期待値はパラメータに等しくなります、$E(b_1) = \beta_1$。これは、一部のランダムサンプルでは推定値$b_1$が$\beta_1$より大きく、他のサンプルでは$\beta_1$より小さくなることを意味します。しかし、平均すると、$b_1$は$\beta_1$に等しくなります。
- 最小二乗推定量は観測された応答$y_1$、$y_2$、$\dots$、$y_n$の線形結合です。例えば$b_3$の場合、$b_1 = \sum_{i=1}^n c_iy_i$となるような既知の定数$c_1$、$c_2$、$\dots$、$c_n$が存在します。

`statsmodels`の`OLS`関数を使用すると、PythonですべてのPython必要な計算が行われます。コーヒーの例では、次のようになります：

```python
# 回帰モデルを当てはめる
mod_mult = smf.ols('total_cup_points ~ aroma + flavor + moisture_percentage + continent_of_origin', 
                   data=coffee_data).fit()

# モデルの係数と標準偏差を表示
print("切片 (b0):", round(mod_mult.params[0], 3))
print("aroma (b1):", round(mod_mult.params[1], 3))
print("flavor (b2):", round(mod_mult.params[2], 3))
print("moisture_percentage (b3):", round(mod_mult.params[3], 3))
print("continent_of_origin[T.Asia] (b02):", round(mod_mult.params[4], 3))
print("continent_of_origin[T.North America] (b03):", round(mod_mult.params[5], 3))
print("continent_of_origin[T.South America] (b04):", round(mod_mult.params[6], 3))
print("標準偏差 (s):", round(np.sqrt(mod_mult.mse_resid), 3))
```

線形モデルが適切である場合、これらの係数をサブセクション「モデル4インタラクションテーブル」および「モデル3テーブル」で行ったように解釈できます。
数値説明変数の係数（$b_1$、$b_2$、および$b_3$）は偏回帰係数であり、それぞれが他のすべての説明変数を一定のレベルに保ちながら対応する説明変数を1単位増加させる追加効果（または効果）を表していることを思い出してください。
例えば、`mod_mult`を使用すると、ある観測に対して他のすべての説明変数を一定のレベルに保ちながら`flavor`スコアを1単位増加させると、`total_cup_points`は平均して一定の単位増加します。
この解釈は`mod_mult`線形回帰モデルに対してのみ有効です。使用する説明変数を変更して、追加または削除すると、モデルが変更され、`flavor`の偏回帰係数は大きさと意味の両方で異なります。偏回帰係数は、その特定のモデルに対してすべての他の説明変数を含むモデルに追加された場合の`flavor`の追加貢献であることを忘れないでください。

さらに、`mod_mult`はサブセクション「モデル4テーブル」で説明したようなインタラクションのないモデルであることに注目してください。`continent_of_origin`の因子レベルの係数（$b_{02}$、$b_{03}$、および$b_{04}$）は、問題の観測のカテゴリに基づいてモデルの切片にのみ影響します。
因子レベルの順序は`Africa`、`Asia`、`North America`、および`South America`であることを思い出してください。
例えば、5番目の観測の原産地大陸が南アメリカの場合（$D_{04} = 1$および$D_{02} = D_{03} = 0$）、回帰式は次のようになります：

$$\begin{aligned}
\widehat{y}_5 &= b_0 + b_1 x_{51} + b_2 x_{52} + b_3 x_{53}+ b_{02}D_{52} + b_{03}D_{53} + b_{04}D_{54}\\
&= b_0 + b_1 x_{51} + b_2 x_{52} + b_3 x_{53}+ b_{02}\cdot 0 + b_{03}\cdot 0 + b_{04}\cdot 1\\
&= (b_0 + b_{04}) + b_1 x_{51} + b_2 x_{52} + b_3 x_{53}\\
\end{aligned}$$

そしてこの観測に対する回帰切片は次のように推定されます：

$$b_0 + b_{04} = 58.543 + (-0.149) = 58.394.$$

すべての説明変数のスコアがゼロであることを通常期待していませんが、常に説明変数が取る値の範囲を確認できます。
`mod_mult`の場合、説明変数の範囲を`describe()`を使用して抽出できます：

```python
# 数値説明変数の範囲
coffee_data[['aroma', 'flavor', 'moisture_percentage']].describe()[['min', 'max']]
```

見ての通り、`moisture_percentage`だけがその範囲にゼロを含んでおり、切片が問題の文脈で特別な意味を持つためには、すべての数値説明変数が範囲にゼロを含む必要があります。

ここでは、説明変数のサブセットだけを使用し、インタラクションなしのモデルを構築することを決定したことに注意してください。
サブセクション「モデル適合度-mult」で、多くの説明変数が利用可能な場合に使用する最適な説明変数のサブセットを決定する方法について説明します。
これで多重線形回帰の推論について議論する準備ができました。

## 多重線形回帰の理論に基づく推論

このセクションでは、多重線形回帰の推論を理解するために必要ないくつかの概念的枠組みを紹介します。
コーヒーの例と「回帰テーブル」サブセクションで導入した`statsmodels`ライブラリを使用してこの枠組みを説明します。

多重線形回帰の推論は、単純線形回帰の推論の自然な拡張です。線形モデルが$i$番目の観測に対して次のように与えられることを思い出してください：

$$y_i = \beta_0 + \beta_1 \cdot x_{i1} + \dots + \beta_p x_{ip} + \epsilon_i\,.$$

再び、誤差項が期待値（平均）がゼロで標準偏差が$\sigma$の正規分布に従うと仮定します：

$$\epsilon_i \sim Normal(0, \sigma).$$

誤差項が線形モデル内の唯一のランダム要素であるため、応答$y_i$は定数

$$\beta_0 + \beta_1 \cdot x_{i1} + \dots + \beta_p x_{ip}$$

と確率変数：誤差項$\epsilon_i$の和から生じます。
正規分布、期待値、および分散（および標準偏差）の特性を使用すると、$y_i$も平均$\beta_0 + \beta_1 \cdot x_{i1} + \dots + \beta_p x_{ip}$、標準偏差$\sigma$の正規分布に従うことを示すことができます：

$$y_i \sim Normal(\beta_0 + \beta_1 \cdot x_{i1} + \dots + \beta_p x_{ip}\,,\, \sigma)$$ 

$i=1,\dots,n$に対して。
また、$\epsilon_i$と$\epsilon_j$は独立であると仮定するので、任意の$i \ne j$に対して$y_i$と$y_j$も独立です。
さらに、（最小二乗）推定量（$b_0, b_1, \dots, b_p$）は上で示したように正規分布に従う確率変数$y_1, \dots, y_n$の線形結合です。
再び、正規分布、期待値、および分散の特性に従うと、以下のことを示すことができます：

- （最小二乗）推定量は正規分布に従います。
- 推定量は不偏であり、これは各推定量の期待値が推定しているパラメータであることを意味します。例えば、$E(b_1) = \beta_1$、または一般に$j = 0,1,\dots, p$に対して$E(b_j) = \beta_j$です。
- 各推定量（$b_j$）の分散と標準偏差は$\sigma$および説明変数の観測データ（サンプル内の説明変数の値）の関数です。
  簡潔にするために、推定量$b_j$の標準偏差は$SD(b_j)$で表されますが、これが応答（$\sigma$）の標準偏差の関数であることを忘れないでください。
- 上記の情報を使用すると、（最小二乗）推定量$b_j$の分布は次のように与えられます：

$$b_j \sim Normal(\beta_j, SD(b_j))$$

$j = 1, \dots, p$に対して。また、$\sigma$は通常未知であり、$\sigma$の代わりに推定標準偏差$s$を使用して推定されることに注意してください。$b_j$の推定標準偏差は$b_j$の標準誤差と呼ばれ、$SE(b_j)$と書かれます。再び、標準誤差が$s$の関数であることを忘れないでください。

標準誤差は回帰モデルに`summary()`メソッドを適用すると表示されます。`mod_mult`モデルの出力は以下の通りです：

```python
# 回帰サマリーを表示
print(mod_mult.summary())
```

数値説明変数の標準誤差は$SE(b_1) = 0.131$、$SE(b_2) = 0.099$、$SE(b_3) = 0.097$であることがわかります。

### 推定量のモデル依存性

多重線形回帰の多くの推論方法および結果は、単純線形回帰で議論した方法および結果の直接的な拡張です。
最も重要な違いは、最小二乗推定量が偏回帰係数を表し、その値がモデル内の他の説明変数に依存するという事実です。
モデルで使用される説明変数のセットを変更すると、最小二乗推定値とその標準誤差も変わる可能性が高いです。
そしてこれらの変更により、信頼区間の限界、仮説検定の検定統計量、およびそれらの説明変数に関する潜在的に異なる結論が導かれます。

`coffee_data`の例を使用して、次のモデルを考えてみましょう：

$$y_i = \beta_0 + \beta_1 \cdot x_{i1} + \beta_2 \cdot x_{i2} + \beta_3 x_{i3} + \epsilon_i\,$$

ここで$\beta_1$、$\beta_2$、および$\beta_3$は、それぞれ`aroma`、`flavor`、および`moisture_precentage`の偏回帰係数のパラメータです。
パラメータ$\beta_1$、$\beta_2$、および$\beta_3$は定数であり、我々には未知ですが定数であると仮定していることを思い出してください。
多重線形回帰を使用して、それぞれ最小二乗推定値$b_1$、$b_2$、および$b_3$を見つけます。`coffee_data`を使用した結果は計算され、オブジェクト`mod_mult_1`に格納されます：

```python
# 最初の回帰モデル（3つの数値説明変数）
mod_mult_1 = smf.ols('total_cup_points ~ aroma + flavor + moisture_percentage', 
                    data=coffee_data).fit()

# 係数と標準偏差を表示
print("切片 (b0):", round(mod_mult_1.params[0], 3))
print("aroma (b1):", round(mod_mult_1.params[1], 3))
print("flavor (b2):", round(mod_mult_1.params[2], 3))
print("moisture_percentage (b3):", round(mod_mult_1.params[3], 3))
print("標準偏差 (s):", round(np.sqrt(mod_mult_1.mse_resid), 3))
```

今度は、説明変数`flavor`なしでモデルを構築することにしたと仮定します：

$$y_i = \beta_0 + \beta_1 \cdot x_{i1} + \beta_3 \cdot x_{i3} + \epsilon_i\,.$$

多重回帰を使用して、それぞれ最小二乗推定値$b'_1$および$b'_3$を計算します。ここで$'$はこの異なるモデルでの潜在的に異なる係数値を表します。`coffee_data`を使用した結果は計算され、オブジェクト`mod_mult_2`に格納されます：

```python
# 2番目の回帰モデル（2つの数値説明変数）
mod_mult_2 = smf.ols('total_cup_points ~ aroma + moisture_percentage', 
                    data=coffee_data).fit()

# 係数と標準偏差を表示
print("切片 (b'0):", round(mod_mult_2.params[0], 3))
print("aroma (b'1):", round(mod_mult_2.params[1], 3))
print("moisture_percentage (b'3):", round(mod_mult_2.params[2], 3))
print("標準偏差 (s'):", round(np.sqrt(mod_mult_2.mse_resid), 3))
```

両方のモデルを使用して`aroma`の偏回帰係数に焦点を当てます。
モデル`mod_mult_1`では、`aroma`の偏回帰係数は$b_1 = 0.525$です。
モデル`mod_mult_2`では、`aroma`の偏回帰係数は$b'_1 = 1.195$です。
結果は本当に異なります。これは各場合で使用されるモデルが異なるためです。
同様に、見つかった他の係数も異なり、標準偏差推定値も異なり、信頼区間や仮説検定の推論結果も異なる可能性があります！

説明変数に関する結果、結論、解釈は、使用されたモデルに対してのみ有効です。
例えば、`aroma`とその`total_cup_points`への効果または影響に関する解釈または結論は、`mod_mult_1`、`mod_mult_2`、または別のモデルを使用したかどうかによって完全に依存します。
あるモデルを使用した結論が別のモデルに変換できると仮定しないでください。

さらに、どのモデルが最も適切であるかを判断することが重要です。
明らかに、`mod_mult_1`と`mod_mult_2`の両方が正しいわけではなく、最も適切なものを使用したいと考えています。
`mod_mult_1`も`mod_mult_2`も適切でなく、代わりに別のモデルを使用すべき場合はどうでしょうか？
これらの質問に対処する統計的推論には2つの領域があります。
最初の領域は、コーヒーの例のように、一方が他方から説明変数のサブセットを使用する2つのモデル間の比較を扱います。ここで`mod_mult_1`は説明変数`aroma`、`flavor`、および`moisture_percentage`を使用し、`mod_mult_2`はそれらの説明変数のうち2つだけ：`aroma`と`moisture_percentage`を使用しました。
この比較に対処する方法については「hypo-test-mult-lm」サブセクションで説明します。
2番目の領域は*モデル選択*または*変数選択*と呼ばれ、利用可能な可能なモデルの中から最も適切なモデルを決定するための代替方法を使用します。

### 信頼区間

多重線形回帰の任意の係数に対する95%信頼区間は、単純線形回帰に対して行ったのと全く同じ方法で構築されますが、常にそれらが得られたモデルに依存するものとして解釈する必要があります。

例えば、モデル`mod_mult`の`aroma`の偏回帰係数$b_1$の95%信頼区間を構築してみましょう：

```python
# b1の95%信頼区間
confidence_interval = mod_mult.conf_int(alpha=0.05)
print("aroma (b1)の95%信頼区間:", confidence_interval.iloc[1])
```

この区間の解釈は通常通りです：「モデル`mod_mult`における`aroma`の母集団偏回帰係数（$\beta_1$）が特定の範囲内の数値であることを95%確信しています。」

`statsmodels`のサマリーテーブルは`mod_mult`モデルでこれらの値を見つけることができます。

```python
# 98%信頼区間を計算
confidence_interval_98 = mod_mult.conf_int(alpha=0.02)
print("98%信頼区間:")
print(confidence_interval_98)
```

例えば、`flavor`の係数の解釈は次のとおりです：「$\beta_2$（`flavor`の母集団偏回帰係数）の値が特定の範囲にあることを98%確信しています。」

### 単一係数の仮説検定

モデルの1つの係数、例えば$\beta_1$の仮説検定は、単純線形回帰の場合と同様です。両側検定の一般的な定式化は次のとおりです：

$$\begin{aligned}
H_0: \beta_1 = B\quad \text{ただし }\beta_0, \beta_2, \dots, \beta_p \text{ は与えられ任意。}\\
H_A: \beta_1 \ne B\quad \text{ただし }\beta_0, \beta_2, \dots, \beta_p \text{ は与えられ任意。}
\end{aligned}$$

ここで$B$は$\beta_1$の仮説値です。
テストが適切なモデルの文脈でのみ問題になることを認識するために、$\beta_0, \beta_2, \dots, \beta_p$が与えられているが任意であることを強調しています。
また、$\beta_1$だけでなく他のどのパラメータに対しても検定を実行できることに注意してください。

単純線形回帰と同様に、最も一般的に使用されるテストは$j=0,1,\dots,p$に対して$\beta_j = 0$をチェックするものです。
$\beta_1$の場合、両側検定は次のようになります：

$$\begin{aligned}
H_0: \beta_1 = 0\quad \text{ただし }\beta_0, \beta_2, \dots, \beta_p \text{ は与えられ任意}\\
H_A: \beta_1 \ne 0\quad \text{ただし }\beta_0, \beta_2, \dots, \beta_p \text{ は与えられ任意}
\end{aligned}$$

単純線形回帰では、$\beta_1 = 0$をテストすることは応答と唯一の説明変数の間に線形関係があるかどうかをテストすることでした。
現在、$\beta_1 = 0$をテストすることは、対応する説明変数が他のすべての説明変数を既に含むモデルの一部であるべきかどうかをテストすることです。

このテストは偏回帰係数パラメータのいずれに対しても実行できます。
例えば、コーヒーの例とモデル`mod_mult_1`（3つの数値説明変数だけを持つモデル）を使用して、$\beta_2$（説明変数`flavor`の母集団偏回帰係数）のテストを実行します。仮説は次のとおりです：

$$\begin{aligned}
H_0: \beta_2 = 0\quad \text{ただし }\beta_0, \beta_1, \beta_3, \beta_{02}, \beta_{03}, \beta_{04} \text{ は与えられ任意。}\\
H_A: \beta_2 \ne 0\quad \text{ただし }\beta_0, \beta_1, \beta_3, \beta_{02}, \beta_{03}, \beta_{04} \text{ は与えられ任意。}
\end{aligned}$$

関連するコードは次のとおりです：

```python
# モデルの要約を表示して仮説検定を確認
print(mod_mult_1.summary().tables[1])
```

t検定統計量は

$$t = \frac{b_2 - 0}{SE(b_2)} = \frac{1.334 - 0}{0.104} = 12.827$$

この検定統計量を使用すると、関連するp値はほぼゼロでPythonは出力を単にゼロとして表示します。
$\beta_2 = 0$という帰無仮説を棄却するのに十分な証拠があります。
帰無仮説を棄却する場合、結果が*統計的に有意*であり、対立仮説（$\beta_2 \ne 0$）を結論づけるのに十分な証拠があると言います。
これは、`flavor`スコアの変化が既に`aroma`と`moisture_percentage`を含むモデルに`flavor`が追加されたときに`total_cup_points`に関する情報を提供することを意味します。

表は全ての係数に関する情報を提供しています。特に、$\beta_1$（`aroma`）のテストも統計的に有意ですが、$\beta_3$（`moisture_percentage`）のテストはそうではない（p値 = 0.214）ことに注目してください。後者の場合、結論はこの偏回帰係数がゼロであるという帰無仮説を棄却する統計的証拠がないということです。言い換えれば、`aroma`と`flavor`を既に含むモデルに`moisture_percentage`を追加することが、応答`total_cup_points`の変化を説明するのに役立つという証拠は見つかりませんでした。`moisture_percentage`説明変数をモデルから削除できます。

### モデル比較の仮説検定

多重線形回帰に対して実行できる別の仮説検定があります。それは、与えられた説明変数のセットを持つ*完全モデル*と、それらの説明変数のサブセットだけを持つ*縮小モデル*の2つのモデルを比較するテストです。

`coffee_data`の例を使用して、完全モデルは次のようであると仮定します：

$$Y = \beta_0 + \beta_1 \cdot X_1 + \beta_2 \cdot X_2 + \beta_3 \cdot X_3 + \epsilon$$ 

ここで$\beta_1$、$\beta_2$、および$\beta_3$はそれぞれ`aroma`、`flavor`、および`moisture_precentage`の偏回帰係数のパラメータです。`coffee_data`データセットを使用したこのモデルに対する多重線形回帰の結果はオブジェクト`mod_mult_1`に格納されています。縮小モデルは説明変数`flavor`を含まず、次のように与えられます：

$$Y = \beta_0 + \beta_1 \cdot X_1  + \beta_3 \cdot X_3 + \epsilon.$$ 

このモデルを使用した多重線形回帰の出力はオブジェクト`mod_mult_2`に格納されています。完全モデルと縮小モデルを比較するための仮説検定は次のように書くことができます：

$$\begin{aligned}
H_0:\quad &Y = \beta_0 + \beta_1 \cdot X_1 + \beta_3 \cdot X_3 + \epsilon\\
H_A:\quad &Y = \beta_0 + \beta_1 \cdot X_1 + \beta_2 \cdot X_2 + \beta_3 \cdot X_3 + \epsilon
\end{aligned}$$

あるいは言葉で言うと：

$$\begin{aligned}
H_0&:\quad \text{縮小モデルが適切である}\\
H_A&:\quad \text{完全モデルが必要である}
\end{aligned}$$

このテストはANOVAテストまたは$F$テストと呼ばれ、テスト統計量の分布が$F$分布に従うためです。
それがどのように機能するかというと、テストは完全モデルと縮小モデルの両方の残差の二乗和を比較し、これらのモデル間の違いが完全モデルが必要であることを示唆するのに十分な大きさであったかどうかを決定します。

PythonでこのテストANOVAの結果を取得するために、`statsmodels`の`anova_lm`関数を使用し、縮小モデルの後に完全モデルを入力します：

```python
import statsmodels.stats.anova as anova

# ANOVAテストを実行
anova_table = anova.anova_lm(mod_mult_2, mod_mult_1)
print(anova_table)
```

テスト統計量は`F`列の2行目に与えられています。
テスト統計量は$F = 164.522$であり、関連するp値はほぼゼロです。
結論として、帰無仮説を棄却し、完全モデルが必要であると結論付けます。

ANOVAテストは一度に複数の説明変数をテストするために使用できます。
これは、モデルに因子（カテゴリ変数）がある場合に役立ちます。すべての因子レベルを同時にテストする必要があるためです。
もう一度例を使用します。今回は完全モデルを3つの数値説明変数に加えて因子`continent_of_origin`を持つモデルとし、縮小モデルを`continent_of_origin`なしのモデルとします。
これらのモデルはすでに`mod_mult`と`mod_mult_1`でそれぞれ計算されています。
ANOVAテストは次のように実行されます：

```python
# 2つ目のANOVAテストを実行
anova_table2 = anova.anova_lm(mod_mult_1, mod_mult)
print(anova_table2)
```

この出力では、自由度が3であることに注意してください。これは一度に3つの係数$\beta_{02}$、$\beta_{03}$、および$\beta_{04}$をテストしているためです。
モデルに因子を含めるためのテストを行う場合、常にすべての因子レベルを一度にテストする必要があります。
出力に基づくと、テスト統計量は$F = 1.584$であり、関連するp値は$0.196$です。
帰無仮説を棄却しないことを選択し、完全モデルは必要ではないと結論付けるかもしれません。あるいは、既に`aroma`、`flavor`、および`moisture_percentage`を持つモデルに因子`continent_of_origin`を追加することは適切ではないと結論付けることもできます。

### モデル適合度と診断

単純線形回帰と同様に、残差を使用してモデルの適合度を判断し、いくつかの仮定が満たされていないかどうかを判断します。
特に、モデル違反がない場合、予測値に対する残差のプロットを引き続き使用し、プロットはヌルプロットに近くなるはずです。

単純線形回帰に提示された処理と解釈のほとんどは同様ですが、予測値に対する残差のプロットがヌルプロットでない場合、モデルの仮定の1つ以上に何らかの違反があることがわかります。
もはや何が理由かは明確ではありませんが、少なくとも1つの仮定が満たされていません。
一方、残差と予測値のプロットがヌルプロットに近く見える場合、どの仮定も破られておらず、このモデルの使用を続行できます。

コーヒーの例を使用しましょう。`aroma`、`flavor`、および`continent_of_origin`の説明変数が統計的に有意であり、`moisture_percentage`はそうでないことを示しました。
したがって、関連する説明変数だけを持つモデルを`mod_mult_final`と呼ばれるものを作成し、残差を決定し、予測値に対する残差のプロットとQQプロットを作成します：

```python
# 最終モデルを当てはめる
mod_mult_final = smf.ols('total_cup_points ~ aroma + flavor + continent_of_origin', 
                         data=coffee_data).fit()

# 予測値と残差を計算
coffee_data['predicted'] = mod_mult_final.predict(coffee_data)
coffee_data['residual'] = coffee_data['total_cup_points'] - coffee_data['predicted']

# 残差vs予測値プロットとQQプロットを並べて表示
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 残差vs予測値プロット
sns.scatterplot(x='predicted', y='residual', data=coffee_data, ax=axes[0])
axes[0].axhline(y=0, color='blue', linestyle='-')
axes[0].set_xlabel('予測値 (total_cup_points)')
axes[0].set_ylabel('残差')
axes[0].set_title('予測値に対する残差')

# QQプロット
sm.qqplot(coffee_data['residual'], line='45', fit=True, ax=axes[1])
axes[1].set_title('残差のQQプロット')

plt.tight_layout()
plt.show()
```

予測値に対する残差のプロット（左）はヌルプロットに近く見えます。この結果は望ましいものです。なぜなら、このプロットにパターンが見られないため、**L**inearity条件をサポートするからです。予測値に対する残差の垂直分散がかなり均一に見えるため、**E**qual or constant varianceも成り立ちます。収集されたデータがランダムであると仮定し、考慮すべき時系列または他の配列がないため、**I**ndependence（独立性）の仮定も許容できると思われます。最後に、QQプロットは（1つまたは2つの観測を除いて）残差がほぼ正規分布に従うことを示唆しています。このモデルはモデルの仮定を保持するのに十分に良いと結論付けることができます。

この例で理論に基づく推論の処理を終えます。次に、シミュレーションに基づく推論について学びます。

## 多重線形回帰のシミュレーションに基づく推論

### `pingouin`を使用した偏回帰係数の信頼区間

前に第7章と第8章で学んだシミュレーションベースの方法を使用して、多重線形回帰による偏回帰係数の妥当な値の範囲を計算しましょう。シミュレーションベースの方法は正規性や大きなサンプルサイズの仮定に依存しないという点で理論ベースの方法の代替となることを思い出してください。先ほど単純線形回帰と同様のパイソンでブートストラップ法を使用します。

#### 観測された予測モデルの取得

因子`continent_of_origin`と3つの数値説明変数`aroma`、`flavor`、および`moisture_percentage`を含む`coffee_data`の完全モデルを再検討します。第8章と先ほどのセクションで行ったように、観測された統計量を取得できます。この場合、観測されたモデル係数を取得します：

```python
# 観測された係数を取得
observed_fit = mod_mult.params
print("観測された係数:")
print(observed_fit)
```

予想通り、これらの値は`mod_mult`テーブルの最初の2列に与えられた値と一致します。

`observed_fit`の値は信頼区間の偏回帰係数の点推定値になります。

#### 偏回帰係数のブートストラップ分布

次に、`pingouin`ワークフローを使用して偏回帰係数のブートストラップ分布を見つけます。単純線形回帰でのやり方と同じように、ブートストラップ分布を構築するために値の行全体を再サンプリングします：

```python
# シードを設定して再現性を確保
np.random.seed(76)

# 多重回帰のためのブートストラップ関数
def bootstrap_mlr(data, formula, n_resamples=1000):
    # 結果を格納するためのリスト
    bootstrap_params = []
    
    # データの行数
    n = len(data)
    
    # ブートストラップリサンプリング
    for i in range(n_resamples):
        # 行単位でリサンプリング
        indices = np.random.choice(n, n, replace=True)
        resample = data.iloc[indices]
        
        # リサンプルに対して回帰を実行
        model = smf.ols(formula, data=resample).fit()
        
        # パラメータを保存
        bootstrap_params.append(model.params)
    
    # 結果をデータフレームに変換
    return pd.DataFrame(bootstrap_params)

# 多重回帰のブートストラップ分布を作成
formula = 'total_cup_points ~ continent_of_origin + aroma + flavor + moisture_percentage'
boot_distribution_mlr = bootstrap_mlr(coffee_data, formula, n_resamples=1000)

# ブートストラップ分布の先頭を表示
print(boot_distribution_mlr.head())
```

コーヒーモデルには切片、`continent_of_origin`の3つのレベル、`aroma`、`flavor`、および`moisture_percentage`に対応する7つの係数があるため、各`replicate`に対して7行あります。ブートストラップ分布を可視化しましょう：

```python
# 偏回帰係数のブートストラップ分布をプロット
fig, axes = plt.subplots(4, 2, figsize=(12, 16))
axes = axes.flatten()

# 各係数のヒストグラム
for i, col in enumerate(boot_distribution_mlr.columns):
    if i < 7:  # 7つの係数のみをプロット
        sns.histplot(boot_distribution_mlr[col], kde=True, ax=axes[i])
        axes[i].axvline(x=observed_fit[i], color='red', linestyle='--')
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel('係数値')
        axes[i].set_ylabel('頻度')

# 不要なサブプロットを削除
axes[7].axis('off')

plt.tight_layout()
plt.suptitle('偏回帰係数のブートストラップ分布', y=1.02, fontsize=16)
plt.show()
```

#### 偏回帰係数の信頼区間

単純線形回帰で行ったように、偏回帰係数の95%信頼区間を構築できます。ここでは、パーセンタイルベースの方法を使用してこれらの区間を構築することに焦点を当てます：

```python
# パーセンタイルベースの95%信頼区間を計算
confidence_intervals_mlr = pd.DataFrame({
    'lower': boot_distribution_mlr.quantile(0.025),
    'upper': boot_distribution_mlr.quantile(0.975)
})

# 観測された係数値を追加
confidence_intervals_mlr['observed'] = observed_fit

# 信頼区間を表示
print("95%信頼区間（パーセンタイル法）:")
print(confidence_intervals_mlr)
```

信頼区間を見直すと、`aroma`と`flavor`の信頼区間は0を含まないことに注目します。これはそれらが統計的に有意であることを示唆しています。これは、他の予測変数の存在下でも応答変数に意味のある信頼性の高い影響を与えていることを示しています。

また、`moisture_percentage`の信頼区間に0が含まれていることにも注目します。これはそれがこの多重線形回帰モデルにおいて有用な説明変数ではない可能性があるという証拠を再び提供しています。これは、他の予測変数を考慮した後の応答変数の変動性の説明に`moisture_percentage`が大きく寄与していないことを意味します。

これらの信頼区間も視覚化できます：

```python
# 信頼区間を視覚化
fig, ax = plt.subplots(figsize=(10, 8))

# 信頼区間の範囲をプロット
for i, var in enumerate(confidence_intervals_mlr.index):
    lower = confidence_intervals_mlr.loc[var, 'lower']
    upper = confidence_intervals_mlr.loc[var, 'upper']
    observed = confidence_intervals_mlr.loc[var, 'observed']
    
    # 水平の線で区間を表示
    ax.plot([lower, upper], [i, i], 'b-', linewidth=2)
    
    # 観測値をポイントで表示
    ax.plot(observed, i, 'ro', markersize=8)
    
    # 0の位置に垂直線を追加
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

# y軸のラベルを設定
ax.set_yticks(range(len(confidence_intervals_mlr.index)))
ax.set_yticklabels(confidence_intervals_mlr.index)

# グラフのラベルを設定
ax.set_xlabel('係数値')
ax.set_title('偏回帰係数の95%信頼区間')

plt.tight_layout()
plt.show()
```

### `pingouin`を使用した偏回帰係数の仮説検定

多重線形回帰における偏回帰係数の仮説検定も実施できます。帰無仮説$H_0: \beta_i = 0$対対立仮説$H_A: \beta_i \neq 0$を各偏回帰係数に対してテストするために並べ替えテストを使用します。以前と同様に、帰無仮説の下での偏回帰係数の帰無分布を構築します。

#### 偏回帰係数の帰無分布

`coffee_data`データセットの説明変数`continent_of_origin`、`aroma`、`flavor`、および`moisture_percentage`の値に対して応答変数`total_cup_points`の値をシャッフルします。これは説明変数と応答の間の独立性の仮定の下で行われます。構文はブートストラップ分布の構築と似ていますが、`replace=False`を使用し、`permute`タイプで`hypothesize`を`null = "independence"`に設定します：

```python
# 多重回帰のための並べ替えテスト関数
def permutation_test_mlr(data, formula, n_permutations=1000):
    # データを準備
    X_vars = formula.split('~')[1].strip().split(' + ')
    y_var = formula.split('~')[0].strip()
    
    # 観測された係数を取得
    observed_model = smf.ols(formula, data=data).fit()
    observed_params = observed_model.params
    
    # 結果を格納するためのリスト
    null_params_list = []
    
    # 並べ替えテスト
    for i in range(n_permutations):
        # 応答変数のコピーを作成
        permuted_data = data.copy()
        
        # 応答変数をシャッフル
        permuted_data[y_var] = np.random.permutation(permuted_data[y_var].values)
        
        # 並べ替えたデータで回帰を実行
        model = smf.ols(formula, data=permuted_data).fit()
        
        # パラメータを保存
        null_params_list.append(model.params)
    
    # 結果をデータフレームに変換
    null_params = pd.DataFrame(null_params_list)
    
    return null_params, observed_params

# シードを設定して再現性を確保
np.random.seed(2024)

# 並べ替えテストを実行
null_distribution_mlr, observed_params = permutation_test_mlr(
    coffee_data, 
    'total_cup_points ~ continent_of_origin + aroma + flavor + moisture_percentage', 
    n_permutations=1000
)

# 帰無分布の先頭を表示
print(null_distribution_mlr.head())
```

#### 偏回帰係数の仮説検定

多重線形回帰における偏回帰係数の仮説検定を行います。帰無仮説$H_0: \beta_i = 0$対対立仮説$H_A: \beta_i \neq 0$を各偏回帰係数に対してテストするために並べ替えテストを使用します。有意水準$\alpha = 0.05$を使用しましょう。観測された検定統計量と比較することで帰無分布のp値を可視化できます：

```python
# 各係数のp値を計算してプロット
fig, axes = plt.subplots(4, 2, figsize=(12, 16))
axes = axes.flatten()

# 各係数のp値をプロット
for i, col in enumerate(null_distribution_mlr.columns):
    if i < 7:  # 7つの係数のみをプロット
        # ヒストグラム
        sns.histplot(null_distribution_mlr[col], kde=True, ax=axes[i])
        
        # 観測値を垂直線で表示
        obs_value = observed_params[i]
        axes[i].axvline(x=obs_value, color='red', linestyle='--')
        
        # 極値領域を着色（両側p値のため）
        extreme_values = null_distribution_mlr[col][
            np.abs(null_distribution_mlr[col]) >= np.abs(obs_value)
        ]
        if len(extreme_values) > 0:
            sns.histplot(extreme_values, color='red', alpha=0.5, ax=axes[i])
        
        # p値を計算
        p_value = np.mean(np.abs(null_distribution_mlr[col]) >= np.abs(obs_value))
        
        axes[i].set_title(f'{col} (p = {p_value:.3f})')
        axes[i].set_xlabel('係数値')
        axes[i].set_ylabel('頻度')

# 不要なサブプロットを削除
axes[7].axis('off')

plt.tight_layout()
plt.suptitle('偏回帰係数のp値', y=1.02, fontsize=16)
plt.show()

# 数値でp値を表示
p_values = {}
for col in null_distribution_mlr.columns:
    obs_value = observed_params[col]
    p_value = np.mean(np.abs(null_distribution_mlr[col]) >= np.abs(obs_value))
    p_values[col] = p_value

print("p値:")
for col, p_value in p_values.items():
    print(f"{col}: {p_value:.4f}")
```

これらの可視化から、観測された検定統計量が帰無分布の遠く右側に落ちるため、`aroma`と`flavor`が統計的に有意であることを推測できます。一方、観測された検定統計量が帰無分布内に落ちるため、`moisture_percentage`は統計的に有意ではありません。

これらの結果は、p値の帰無分布の可視化と回帰テーブルでの結論と一致します。`aroma`と`flavor`については帰無仮説$H_0: \beta_i = 0$を棄却しますが、`moisture_percentage`については棄却しません。

## 結論

### 統計的推論のまとめ

これで"ModernDiveでのR"（第2版）の"回帰分析における統計的推論"の章が完了しました。

第5章と第6章で学んだ回帰モデリング手法、第6章のサンプリングによる推論の理解、そして第7章と第8章の信頼区間と仮説検定のような統計的推論ツールを身につけたことで、さまざまなデータにおける変数間の関係の有意性を研究する準備が整いました。ここで提示された多くのアイデアは、多重回帰やその他のより高度なモデリング手法に拡張できます。