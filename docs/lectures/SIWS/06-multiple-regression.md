# 第6章 重回帰分析

第5章では説明のためのモデリングの概念を紹介しました。その目的は、ある結果変数 $y$ と説明変数 $x$ との関係を明示的に表現することでした。モデリングにはさまざまなアプローチがありますが、特に一般的で理解しやすい手法である*線形回帰*に焦点を当てました。また、前章では単純化のため、1つの説明変数 $x$ のみを持つモデルを考えました。この説明変数は、第5.1節では数値的、第5.2節では分類的なものでした。

本章では、複数の説明変数 $x$ を含むモデルを検討し始めます。教員評価スコア（第5.1節）や平均寿命（第5.2節）などの結果変数をモデル化する際、複数の説明変数の情報を含めると有用であることは容易に想像できます。

複数の説明変数を持つ回帰モデルでは、任意の1つの説明変数の関連効果はモデルに含まれる他の説明変数と併せて解釈する必要があります。それでは始めましょう！

### 必要なパッケージ {-}

Pythonでの作業に必要なライブラリをインポートしましょう：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
```

## 6.1 1つの数値的説明変数と1つの分類的説明変数

テキサス大学オースティン校の教員評価データを再び見てみましょう。第5.1節では、学生による教育評価スコアと「美しさ」スコアの関係を調査しました。変数「score」は結果変数 $y$ で、変数「bty_avg」（平均「美しさ」スコア）は説明変数 $x$ でした。

このセクションでは、異なるモデルを検討します。結果変数は引き続き教育スコアですが、今回は2つの異なる説明変数を含めます：年齢と（二値の）性別です。年齢が高い教員は学生からより良い評価を受けるのでしょうか？それとも若い教員の方が良い評価を受けるのでしょうか？異なる性別の教員で評価に違いはあるのでしょうか？これらの問題に対して*重回帰*を使用して変数間の関係をモデル化します：

1. 数値的結果変数 $y$：教員の教育スコア
2. 2つの説明変数：
   1. 数値的説明変数 $x_1$：教員の年齢
   2. 分類的説明変数 $x_2$：教員の（二値の）性別

ただし、この研究が行われた当時、性別に関する一般的な考え方により、この変数は二値変数として記録されていました。このように性別を単純化したモデルの結果は不完全かもしれませんが、その結果は今日でも関連性があると考えられます。

### 探索的データ分析 {#model4EDA}

データをダウンロードして準備しましょう：

```python
# GitHubからデータをダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/evals.csv"
evals = pd.read_csv(url)

# 分析に必要な変数のみを選択
evals_ch6 = evals[['ID', 'score', 'age', 'gender']]
```

探索的データ分析の一般的な3つのステップを思い出しましょう：

1. 生データ値を確認する
2. 要約統計量を計算する
3. データの可視化を作成する

まず、生データ値を見てみましょう：

```python
# データの構造を確認
print(evals_ch6.info())

# 最初の数行を表示
print(evals_ch6.head())

# ランダムに5行を選んで表示
print(evals_ch6.sample(5))
```

次に、要約統計量を計算します：

```python
# 基本的な要約統計量
print(evals_ch6.describe())

# 分類変数の集計
print(evals_ch6['gender'].value_counts())

# 相関係数の計算（数値変数のみ）
print(evals_ch6[['score', 'age']].corr())
```

最後に、データの可視化を作成します。結果変数「score」と説明変数「age」はどちらも数値的なので、散布図を使用しますが、性別情報を色で表示します：

```python
# 散布図：色で性別を示す
plt.figure(figsize=(10, 6))
# 性別でデータを分割
males = evals_ch6[evals_ch6['gender'] == 'male']
females = evals_ch6[evals_ch6['gender'] == 'female']

# 散布図をプロット
plt.scatter(males['age'], males['score'], color='blue', alpha=0.6, label='男性')
plt.scatter(females['age'], females['score'], color='red', alpha=0.6, label='女性')

# 各グループの回帰直線
male_fit = np.polyfit(males['age'], males['score'], 1)
female_fit = np.polyfit(females['age'], females['score'], 1)
male_line = np.poly1d(male_fit)
female_line = np.poly1d(female_fit)

# 回帰直線のプロット
ages = np.array([min(evals_ch6['age']), max(evals_ch6['age'])])
plt.plot(ages, male_line(ages), color='blue', linewidth=2)
plt.plot(ages, female_line(ages), color='red', linewidth=2)

plt.xlabel('年齢')
plt.ylabel('教育スコア')
plt.title('教育スコアと年齢の関係（性別で色分け）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

この図から、いくつかの興味深い傾向が見えてきます。まず、60歳以上の女性教員はほとんどいないことがわかります（赤い点が60以上にほとんどない）。第二に、両方の回帰直線は年齢とともに負の傾きを持っています（つまり、年齢が高い教員ほど評価が低い傾向がある）が、女性教員の場合、年齢の傾きはより顕著です。言い換えれば、女性教員は男性教員よりも年齢で厳しいペナルティを受けているように見えます。

### 交互作用モデル

図中の2つの回帰直線の方程式を回帰表の値を使って書き出してみましょう。まず、分類的説明変数を持つ回帰についておさらいします。

整理のために、モデルを実行して係数を取得しましょう：

```python
# ダミー変数を作成（性別用）
evals_ch6_encoded = pd.get_dummies(evals_ch6, columns=['gender'], drop_first=True)
evals_ch6_encoded.rename(columns={'gender_male': 'is_male'}, inplace=True)

# 交互作用項を作成
evals_ch6_encoded['age_male_interaction'] = evals_ch6_encoded['age'] * evals_ch6_encoded['is_male']

# 交互作用モデルの実行
X = evals_ch6_encoded[['age', 'is_male', 'age_male_interaction']]
y = evals_ch6_encoded['score']

interaction_model = LinearRegression()
interaction_model.fit(X, y)

# 結果の表示
intercept = interaction_model.intercept_
coefficients = interaction_model.coef_

print("交互作用モデル回帰表：")
print(f"切片: {intercept:.3f}")
print(f"年齢の係数: {coefficients[0]:.3f}")
print(f"性別（男性）の係数: {coefficients[1]:.3f}")
print(f"交互作用項（年齢*男性）の係数: {coefficients[2]:.3f}")

# 各性別の回帰直線の係数を計算
female_intercept = intercept
female_slope = coefficients[0]
male_intercept = intercept + coefficients[1]
male_slope = coefficients[0] + coefficients[2]

print("\n各性別の回帰直線：")
print(f"女性: score = {female_intercept:.3f} + ({female_slope:.3f}) * age")
print(f"男性: score = {male_intercept:.3f} + ({male_slope:.3f}) * age")
```

アルファベット順に「female」が「male」より先に来るため、女性教員が「比較のベースライン」グループとなります。そのため、「intercept」は女性教員のみの切片となります。

同様に、「age」の係数は女性教員のみの年齢の傾きです。つまり、図中の赤い回帰直線の切片は女性の切片値で、年齢の傾きは女性の年齢傾き値です。この場合、切片には数学的な解釈はありますが、教員の年齢がゼロになることはないため、実際的な解釈はありません。

男性教員（図中の青い線）の切片と年齢の傾きはどうでしょうか？ここで「オフセット」の概念が再び登場します。

「gender: male」の値は男性教員の切片ではなく、女性教員に対する男性教員の切片の*オフセット*です。男性教員の切片は「intercept + gender: male」です。

同様に、「age:gendermale」は男性教員の年齢の傾きではなく、男性教員の傾きの*オフセット*です。したがって、男性教員の年齢の傾きは「age + age:gendermale」です。

両方のモデルの傾きに注目すると：

女性教員の年齢傾きは負値であり、年齢が1歳上がるごとに、教育スコアは平均的に約一定値低下することを意味します。男性教員の場合、対応する関連する低下は平均的にそれより小さい値です。両方の年齢傾きは負ですが、女性教員の年齢傾きはより負です。これは図中の観察と一致しており、このモデルでは年齢が女性教員の教育スコアに与える影響は男性教員よりも大きいことを示唆しています。

回帰直線の方程式を書き下すと：

$$
\begin{aligned}
\widehat{y} = \widehat{\text{score}} &= b_0 + b_{\text{age}} \cdot \text{age} + b_{\text{male}} \cdot \mathbb{1}_{\text{is male}}(x) + b_{\text{age,male}} \cdot \text{age} \cdot \mathbb{1}_{\text{is male}}(x)
\end{aligned}
$$

ここで、$\mathbb{1}_{\text{is male}}(x)$は指示関数で、次のように定義されます：

$$
\mathbb{1}_{\text{is male}}(x) = \left\{
\begin{array}{ll}
1 & \text{（講師 } x \text{ が男性の場合）} \\
0 & \text{（それ以外の場合）}\end{array}
\right.
$$

これを使って女性教員の予測値を計算すると：

$$
\begin{aligned}
\widehat{y} = \widehat{\text{score}} &= b_0 + b_{\text{age}} \cdot \text{age} + b_{\text{male}} \cdot 0 + b_{\text{age,male}} \cdot \text{age} \cdot 0\\
&= b_0 + b_{\text{age}} \cdot \text{age}
\end{aligned}
$$

男性教員の場合：

$$
\begin{aligned}
\widehat{y} = \widehat{\text{score}} &= b_0 + b_{\text{age}} \cdot \text{age} + b_{\text{male}} \cdot 1 + b_{\text{age,male}} \cdot \text{age} \cdot 1\\
&= (b_0 + b_{\text{male}}) + (b_{\text{age}} + b_{\text{age,male}}) \cdot \text{age}
\end{aligned}
$$

この種のモデルを「交互作用モデル」と呼ぶ理由を説明しましょう。方程式中の$b_{\text{age,male}}$項は「交互作用効果」と呼ばれます。ある変数の関連効果が別の変数の値に依存する場合、交互作用効果があるといいます。つまり、2つの変数が「相互作用」しています。ここでは、年齢変数の関連効果が性別変数の値に依存しています。

教育スコアへの交互作用効果を別の見方で考えると、ある教員について、年齢*自体*の関連効果、性別*自体*の関連効果がある可能性がありますが、年齢と性別が*一緒に*考慮されると、2つの個々の効果を超える*追加的な効果*がある可能性があります。

### 平行傾斜モデル

1つの数値的説明変数と1つの分類的説明変数を持つ回帰モデルを作成する際、交互作用モデルに限定されるわけではありません。もう一つのモデルとして*平行傾斜モデル*があります。交互作用モデルでは回帰直線が異なる切片と異なる傾きを持つのに対し、平行傾斜モデルでは異なる切片を許容しますが、すべての直線に同じ傾きを*強制*します。結果として、回帰直線は平行になります。

平行傾斜モデルをPythonで実装し、可視化しましょう：

```python
# 平行傾斜モデルの実行
X_parallel = evals_ch6_encoded[['age', 'is_male']]
y = evals_ch6_encoded['score']

parallel_model = LinearRegression()
parallel_model.fit(X_parallel, y)

# 結果の表示
intercept_parallel = parallel_model.intercept_
coefficients_parallel = parallel_model.coef_

print("平行傾斜モデル回帰表：")
print(f"切片: {intercept_parallel:.3f}")
print(f"年齢の係数: {coefficients_parallel[0]:.3f}")
print(f"性別（男性）の係数: {coefficients_parallel[1]:.3f}")

# 各性別の回帰直線の係数を計算
female_intercept_parallel = intercept_parallel
female_slope_parallel = coefficients_parallel[0]
male_intercept_parallel = intercept_parallel + coefficients_parallel[1]
male_slope_parallel = coefficients_parallel[0]  # 同じ傾き

print("\n各性別の回帰直線（平行）：")
print(f"女性: score = {female_intercept_parallel:.3f} + ({female_slope_parallel:.3f}) * age")
print(f"男性: score = {male_intercept_parallel:.3f} + ({male_slope_parallel:.3f}) * age")

# 平行傾斜モデルの可視化
plt.figure(figsize=(10, 6))
# 散布図をプロット
plt.scatter(males['age'], males['score'], color='blue', alpha=0.6, label='男性')
plt.scatter(females['age'], females['score'], color='red', alpha=0.6, label='女性')

# 平行傾斜直線のプロット
ages = np.array([min(evals_ch6['age']), max(evals_ch6['age'])])
plt.plot(ages, female_intercept_parallel + female_slope_parallel * ages, color='red', linewidth=2)
plt.plot(ages, male_intercept_parallel + male_slope_parallel * ages, color='blue', linewidth=2)

plt.xlabel('年齢')
plt.ylabel('教育スコア')
plt.title('教育スコアと年齢の関係（平行傾斜モデル）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

平行傾斜モデルでは、女性と男性教員に対応する直線が平行になり、同じ負の傾きを持つことが分かります。これは、年齢が高い教員は若い教員よりも教育スコアが低くなる傾向があることを示しています。また、直線が平行であるため、年齢による影響は男女の教員で同じと仮定しています。

ただし、図では女性教員（赤い線）と男性教員（青い線）で切片が異なることもわかります。男性教員の線が女性教員の線より上にあることから、年齢に関係なく、女性教員は男性教員より低い教育スコアを受ける傾向があることが示されています。

交互作用モデルと平行傾斜モデルを視覚的に比較してみましょう：

```python
# 交互作用モデルと平行傾斜モデルを並べて比較
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 交互作用モデル
axes[0].scatter(males['age'], males['score'], color='blue', alpha=0.6, label='男性')
axes[0].scatter(females['age'], females['score'], color='red', alpha=0.6, label='女性')
ages = np.array([min(evals_ch6['age']), max(evals_ch6['age'])])
axes[0].plot(ages, female_intercept + female_slope * ages, color='red', linewidth=2)
axes[0].plot(ages, male_intercept + male_slope * ages, color='blue', linewidth=2)
axes[0].set_xlabel('年齢')
axes[0].set_ylabel('教育スコア')
axes[0].set_title('交互作用モデル')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 平行傾斜モデル
axes[1].scatter(males['age'], males['score'], color='blue', alpha=0.6, label='男性')
axes[1].scatter(females['age'], females['score'], color='red', alpha=0.6, label='女性')
axes[1].plot(ages, female_intercept_parallel + female_slope_parallel * ages, color='red', linewidth=2)
axes[1].plot(ages, male_intercept_parallel + male_slope_parallel * ages, color='blue', linewidth=2)
axes[1].set_xlabel('年齢')
axes[1].set_ylabel('教育スコア')
axes[1].set_title('平行傾斜モデル')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

この時点で「なぜ平行傾斜モデルを使うのか」と疑問に思うかもしれません。左側のプロットでは、2つの直線は明らかに平行ではないので、なぜそれらを平行に*強制*するのでしょうか？このデータに関しては、交互作用モデルの方が適切であると主張できます。ただし、セクション6.3.1のモデル選択では、平行傾斜モデルの方が適切かもしれない例を示します。

### 観測値、予測値、残差

簡潔にするために、交互作用モデルの観測値、予測値、残差のみを計算します：

```python
# 交互作用モデルの予測値と残差を計算
X = evals_ch6_encoded[['age', 'is_male', 'age_male_interaction']]
predictions = interaction_model.predict(X)
residuals = y - predictions

# 結果をデータフレームにまとめる
regression_points = evals_ch6.copy()
regression_points['score_hat'] = predictions
regression_points['residual'] = residuals

# 結果の表示
print(regression_points.head(10))

# 女性教員（36歳）と男性教員（59歳）の予測値を計算
female_example_age = 36
male_example_age = 59

female_prediction = female_intercept + female_slope * female_example_age
male_prediction = male_intercept + male_slope * male_example_age

print(f"\n36歳の女性教員の予測スコア: {female_prediction:.2f}")
print(f"59歳の男性教員の予測スコア: {male_prediction:.2f}")

# 予測値の可視化
plt.figure(figsize=(10, 6))
plt.scatter(males['age'], males['score'], color='blue', alpha=0.4, label='男性')
plt.scatter(females['age'], females['score'], color='red', alpha=0.4, label='女性')

# 回帰直線
plt.plot(ages, female_intercept + female_slope * ages, color='red', linewidth=2)
plt.plot(ages, male_intercept + male_slope * ages, color='blue', linewidth=2)

# 予測例の表示
plt.scatter([female_example_age], [female_prediction], color='red', s=100, 
            edgecolor='black', linewidth=2, label='36歳女性教員の予測値')
plt.scatter([male_example_age], [male_prediction], color='blue', s=100, 
            edgecolor='black', linewidth=2, label='59歳男性教員の予測値')
plt.axvline(x=female_example_age, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=male_example_age, color='gray', linestyle='--', alpha=0.5)

plt.xlabel('年齢')
plt.ylabel('教育スコア')
plt.title('交互作用モデルの予測値')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 6.2 2つの数値的説明変数

次に、2つの数値的説明変数を持つ重回帰モデルを考えてみましょう。このセクションでは、[*An Introduction to Statistical Learning with Applications in R (ISLR)*](https://www.statlearning.com/)という中級レベルの統計学習とマシンラーニングの教科書からのデータを使用します。

ISLRの`Credit`データセットには、400人のクレジットカード保有者の情報が含まれています。結果変数はクレジットカード負債で、説明変数には収入、信用限度額、信用格付け、年齢などが含まれています。このデータは実際の個人の金融情報に基づいたものではなく、教育目的で作成されたシミュレーションデータセットです。

このセクションでは、以下の回帰モデルを適合させます：

1. 数値的結果変数 $y$：カード保有者のクレジットカード負債
2. 2つの説明変数：
   1. 数値的説明変数 $x_1$：カード保有者の信用限度額
   2. 数値的説明変数 $x_2$：カード保有者の収入（千ドル単位）

### 探索的データ分析

Creditデータセットをダウンロードして準備しましょう：

```python
# ISLRのCreditデータをダウンロード（GitHub上の代替リポジトリから）
url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Credit.csv"
credit = pd.read_csv(url)

# 必要な変数のみを選択して名前を変更
credit_ch6 = credit[['ID', 'Balance', 'Limit', 'Income', 'Rating', 'Age']].copy()
credit_ch6.rename(columns={
    'Balance': 'debt',
    'Limit': 'credit_limit',
    'Income': 'income',
    'Rating': 'credit_rating',
    'Age': 'age'
}, inplace=True)
```

生データ値を確認しましょう：

```python
# データの構造を確認
print(credit_ch6.info())

# 最初の行を表示
print(credit_ch6.head())

# ランダムに5行を選んで表示
print(credit_ch6.sample(5))
```

要約統計量を計算します：

```python
# 基本的な要約統計量
print(credit_ch6.describe())

# 相関行列
correlation_matrix = credit_ch6[['debt', 'credit_limit', 'income']].corr()
print("相関行列:")
print(correlation_matrix)

# より明確な相関の表示
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('相関行列のヒートマップ')
plt.show()
```

変数間の関係を可視化します：

```python
# 結果変数と各説明変数の散布図
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 負債と信用限度額の散布図
axes[0].scatter(credit_ch6['credit_limit'], credit_ch6['debt'])
axes[0].set_xlabel('信用限度額（ドル）')
axes[0].set_ylabel('クレジットカード負債（ドル）')
axes[0].set_title('負債と信用限度額の関係')
# 回帰直線を追加
m1, b1 = np.polyfit(credit_ch6['credit_limit'], credit_ch6['debt'], 1)
x1 = np.array([min(credit_ch6['credit_limit']), max(credit_ch6['credit_limit'])])
axes[0].plot(x1, m1 * x1 + b1, 'r-', linewidth=2)
axes[0].grid(True, alpha=0.3)

# 負債と収入の散布図
axes[1].scatter(credit_ch6['income'], credit_ch6['debt'])
axes[1].set_xlabel('収入（千ドル）')
axes[1].set_ylabel('クレジットカード負債（ドル）')
axes[1].set_title('負債と収入の関係')
# 回帰直線を追加
m2, b2 = np.polyfit(credit_ch6['income'], credit_ch6['debt'], 1)
x2 = np.array([min(credit_ch6['income']), max(credit_ch6['income'])])
axes[1].plot(x2, m2 * x2 + b2, 'r-', linewidth=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

信用限度額とクレジットカード負債の間には正の関係があることがわかります。信用限度額が増加するにつれて、クレジットカード負債も増加します。これは、先ほど計算した0.862という強い正の相関係数と一致しています。収入の場合は、正の関係が見られますが、信用限度額と負債の関係ほど強くありません（相関係数は0.464）。

信用限度額と収入の間にも高い相関（0.792）があることに注目すべきです。これは*共線性*と呼ばれる現象で、ある人の信用限度額を知れば、その人の収入についてもかなり良い推測ができることを意味します。したがって、これら2つの変数はある程度冗長な情報を提供しています。

### 回帰平面

3次元的な関係を含む回帰モデルを適合させましょう：

```python
# 重回帰モデルを構築
X = credit_ch6[['credit_limit', 'income']]
y = credit_ch6['debt']

model = LinearRegression()
model.fit(X, y)

# 結果を表示
intercept = model.intercept_
coefficients = model.coef_

print("回帰係数：")
print(f"切片: {intercept:.3f}")
print(f"信用限度額の係数: {coefficients[0]:.3f}")
print(f"収入の係数: {coefficients[1]:.3f}")

# 回帰平面の方程式
print("\n回帰平面の方程式:")
print(f"債務 = {intercept:.3f} + {coefficients[0]:.3f} * 信用限度額 + {coefficients[1]:.3f} * 収入")

# 回帰平面の3D可視化を試みる
from mpl_toolkits.mplot3d import Axes3D

# グリッドデータを作成
credit_limit_range = np.linspace(min(credit_ch6['credit_limit']), max(credit_ch6['credit_limit']), 30)
income_range = np.linspace(min(credit_ch6['income']), max(credit_ch6['income']), 30)
credit_limit_grid, income_grid = np.meshgrid(credit_limit_range, income_range)
z_grid = intercept + coefficients[0] * credit_limit_grid + coefficients[1] * income_grid

# 3Dプロット
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 散布図
ax.scatter(credit_ch6['credit_limit'], credit_ch6['income'], credit_ch6['debt'], 
           color='blue', alpha=0.6, label='実際のデータ')

# 回帰平面
ax.plot_surface(credit_limit_grid, income_grid, z_grid, alpha=0.3, color='red')

ax.set_xlabel('信用限度額（ドル）')
ax.set_ylabel('収入（千ドル）')
ax.set_zlabel('クレジットカード負債（ドル）')
ax.set_title('回帰平面：信用限度額と収入に基づくクレジットカード負債の予測')
plt.legend()
plt.show()
```

「estimate」列の3つの値を解釈しましょう。まず、「intercept」値は-385.179ドルです。この切片は、信用限度額が0ドルで収入も0ドルの個人のクレジットカード負債を表します。しかし、データには信用限度額や収入が0の個人はいないため、切片には実用的な解釈はありません。むしろ、切片は3D空間に回帰平面を配置するために使用されます。

次に、「credit_limit」値は0.264ドルです。モデル内の他のすべての説明変数を考慮すると、信用限度額が1ドル増加するごとに、クレジットカード負債は平均して0.26ドル増加します。第5.1節でしたように、因果関係を示唆しないよう注意します。これは単に関連する増加を示しているだけです。

さらに、解釈の前に「モデル内の他のすべての説明変数を考慮すると」という表現を用います。ここでは、他のすべての説明変数とは「income」を意味します。これにより、同じモデル内で複数の説明変数の関連効果を同時に解釈していることを強調しています。

第三に、「income」は-7.66ドルです。モデル内の他のすべての説明変数を考慮すると、「income」が1単位（実際の収入で1,000ドル）増加するごとに、クレジットカード負債は平均して7.66ドル減少します。

これらの結果をまとめると、予測値 $\widehat{y} = \widehat{\text{debt}}$ を与える回帰平面の方程式は：

$$
\begin{aligned}
\widehat{y} &= b_0 + b_1 \cdot x_1 +  b_2 \cdot x_2\\
\widehat{\text{debt}} &= b_0 + b_{\text{limit}} \cdot \text{limit} + b_{\text{income}} \cdot \text{income}\\
&= -385.179 + 0.263 \cdot\text{limit} - 7.663 \cdot\text{income}
\end{aligned}
$$

興味深いことに、図6.1で「debt」と「income」の関係を単独でプロットしたとき、正の関係があるように見えましたが、「debt」、「credit_limit」、「income」の関係を共同でモデル化すると、「debt」と「income」の関係は負になっています（「income」の傾きが-7.663ドルである）。これらの矛盾する結果は、*シンプソンのパラドックス*と呼ばれる現象によるものです。これについては、セクション6.3.3で詳しく説明します。

### 観測値、予測値、残差

回帰モデルのすべての予測値と残差を計算しましょう：

```python
# 予測値と残差を計算
predictions = model.predict(X)
residuals = y - predictions

# 結果をデータフレームにまとめる
regression_points = credit_ch6.copy()
regression_points['debt_hat'] = predictions
regression_points['residual'] = residuals

# 最初の10行を表示
print(regression_points.head(10))

# 予測の評価
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f"\n平均二乗誤差: {mse:.2f}")
print(f"R² 決定係数: {r2:.4f}")
```

## 6.3 関連トピック

### 可視化を用いたモデル選択

交互作用モデルと平行傾斜モデルのどちらを使用すべきでしょうか？セクション6.1.3と6.1.4では、結果変数 $y$（教育スコア）に対して、数値的説明変数 $x_1$（年齢）と分類的説明変数 $x_2$（二値の性別）を使った両方のモデルを適合させました。これらのモデルを比較しましょう。

多くの人が「明らかに異なる傾きを持つ線を（右側のプロットのように）平行にする必要があるのはなぜか」と疑問に思うかもしれません。

答えは「オッカムのかみそり」という哲学的原則にあります。これは「他のすべての条件が等しい場合、単純な解決策は複雑な解決策よりも正しい可能性が高い」という原則です。モデリングの枠組みで見ると、「他のすべての条件が等しい場合、単純なモデルは複雑なモデルよりも望ましい」と言い換えることができます。つまり、追加の複雑さが*正当化される*場合にのみ、より複雑なモデルを選択すべきです。

交互作用モデルと平行傾斜モデルの回帰直線の方程式を再検討しましょう：

$$
\begin{aligned}
\text{交互作用} &: \widehat{y} = \widehat{\text{score}} = b_0 + b_{\text{age}} \cdot \text{age} + b_{\text{male}} \cdot \mathbb{1}_{\text{is male}}(x) + \\
& \qquad b_{\text{age,male}} \cdot \text{age} \cdot \mathbb{1}_{\text{is male}}\\
\text{平行傾斜} &: \widehat{y} = \widehat{\text{score}} = b_0 + b_{\text{age}} \cdot \text{age} + b_{\text{male}} \cdot \mathbb{1}_{\text{is male}}(x)
\end{aligned}
$$

交互作用モデルは、平行傾斜モデルには存在しない追加の$b_{\text{age,male}} \cdot \text{age} \cdot \mathbb{1}_{\text{is male}}$交互作用項があるため、「より複雑」です。あるいは、交互作用モデルの回帰表には*4つ*の行があるのに対し、平行傾斜モデルの回帰表は*3つ*の行があります。問題は「この追加の複雑さは正当化されるか？」です。この場合、左側のプロットに見られる2つの回帰直線の明確なX字型のパターンから、この追加の複雑さは正当化されると言えるでしょう。

しかし、追加の複雑さが*正当化されない*例を考えてみましょう。`moderndive`パッケージに含まれる`MA_schools`データを使用して、マサチューセッツ州の公立高校の2017年のデータを分析します：

```python
# マサチューセッツ州の学校データをダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/MA_schools.csv"
MA_schools = pd.read_csv(url)

# データの前処理：学校規模（登録数）のカテゴリを作成
MA_schools['size'] = pd.cut(
    MA_schools['total_enrollment'], 
    bins=[0, 341, 541, 5000],  # 13-341, 342-541, 542-4264
    labels=['small', 'medium', 'large']
)

print(MA_schools.head())
print(MA_schools['size'].value_counts())

# 交互作用モデルと平行傾斜モデルを適合させる
# ダミー変数を作成
MA_schools_encoded = pd.get_dummies(MA_schools, columns=['size'], drop_first=True)

# 必要な変数を選択
X_columns = ['perc_disadvan', 'size_medium', 'size_large', 
             'perc_disadvan:size_medium', 'perc_disadvan:size_large']

# 交互作用項を作成
MA_schools_encoded['perc_disadvan:size_medium'] = MA_schools_encoded['perc_disadvan'] * MA_schools_encoded['size_medium']
MA_schools_encoded['perc_disadvan:size_large'] = MA_schools_encoded['perc_disadvan'] * MA_schools_encoded['size_large']

# 交互作用モデル
X_interaction = MA_schools_encoded[['perc_disadvan', 'size_medium', 'size_large', 
                                    'perc_disadvan:size_medium', 'perc_disadvan:size_large']]
y = MA_schools_encoded['average_sat_math']

interaction_model = LinearRegression()
interaction_model.fit(X_interaction, y)

# 交互作用モデルの係数
print("\n交互作用モデルの係数:")
print(f"切片: {interaction_model.intercept_:.3f}")
print(f"経済的不利の割合: {interaction_model.coef_[0]:.3f}")
print(f"中規模校: {interaction_model.coef_[1]:.3f}")
print(f"大規模校: {interaction_model.coef_[2]:.3f}")
print(f"経済的不利 × 中規模校: {interaction_model.coef_[3]:.3f}")
print(f"経済的不利 × 大規模校: {interaction_model.coef_[4]:.3f}")

# 平行傾斜モデル
X_parallel = MA_schools_encoded[['perc_disadvan', 'size_medium', 'size_large']]
parallel_model = LinearRegression()
parallel_model.fit(X_parallel, y)

# 平行傾斜モデルの係数
print("\n平行傾斜モデルの係数:")
print(f"切片: {parallel_model.intercept_:.3f}")
print(f"経済的不利の割合: {parallel_model.coef_[0]:.3f}")
print(f"中規模校: {parallel_model.coef_[1]:.3f}")
print(f"大規模校: {parallel_model.coef_[2]:.3f}")

# モデルの可視化
plt.figure(figsize=(15, 7))

# 散布図のプロット（サイズによって色分け）
sizes = MA_schools['size'].unique()
colors = ['red', 'green', 'blue']
size_map = {size: color for size, color in zip(sizes, colors)}

# サイズごとに散布図をプロット
for size in sizes:
    subset = MA_schools[MA_schools['size'] == size]
    plt.scatter(subset['perc_disadvan'], subset['average_sat_math'], 
               color=size_map[size], alpha=0.25, label=f'{size}規模校')

# 交互作用モデルと平行傾斜モデルの予測線を追加
x_range = np.linspace(0, 100, 100)

# 小規模校（ベースライン）
y_small_int = interaction_model.intercept_ + interaction_model.coef_[0] * x_range
y_medium_int = (interaction_model.intercept_ + interaction_model.coef_[1]) + \
              (interaction_model.coef_[0] + interaction_model.coef_[3]) * x_range
y_large_int = (interaction_model.intercept_ + interaction_model.coef_[2]) + \
             (interaction_model.coef_[0] + interaction_model.coef_[4]) * x_range

# 平行傾斜モデルの予測
y_small_par = parallel_model.intercept_ + parallel_model.coef_[0] * x_range
y_medium_par = (parallel_model.intercept_ + parallel_model.coef_[1]) + \
              parallel_model.coef_[0] * x_range
y_large_par = (parallel_model.intercept_ + parallel_model.coef_[2]) + \
             parallel_model.coef_[0] * x_range

# 左側のプロット（交互作用モデル）
plt.subplot(1, 2, 1)
for size in sizes:
    subset = MA_schools[MA_schools['size'] == size]
    plt.scatter(subset['perc_disadvan'], subset['average_sat_math'], 
               color=size_map[size], alpha=0.25)

plt.plot(x_range, y_small_int, color='red', linestyle='-', linewidth=2, label='小規模校')
plt.plot(x_range, y_medium_int, color='green', linestyle='-', linewidth=2, label='中規模校')
plt.plot(x_range, y_large_int, color='blue', linestyle='-', linewidth=2, label='大規模校')

plt.xlabel('経済的不利の割合')
plt.ylabel('数学SATスコア')
plt.title('交互作用モデル')
plt.legend()
plt.grid(True, alpha=0.3)

# 右側のプロット（平行傾斜モデル）
plt.subplot(1, 2, 2)
for size in sizes:
    subset = MA_schools[MA_schools['size'] == size]
    plt.scatter(subset['perc_disadvan'], subset['average_sat_math'], 
               color=size_map[size], alpha=0.25)

plt.plot(x_range, y_small_par, color='red', linestyle='-', linewidth=2, label='小規模校')
plt.plot(x_range, y_medium_par, color='green', linestyle='-', linewidth=2, label='中規模校')
plt.plot(x_range, y_large_par, color='blue', linestyle='-', linewidth=2, label='大規模校')

plt.xlabel('経済的不利の割合')
plt.ylabel('数学SATスコア')
plt.title('平行傾斜モデル')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

結果を見ると、左側のプロット（交互作用モデル）では、傾きは確かに異なりますが、その差はわずかでほぼ同じです。左側のプロットと平行傾斜モデルに対応する右側のプロットを比較すると、2つのモデルはそれほど違いがないことがわかります。この場合、交互作用モデルの追加の複雑さは*正当化されない*と主張できるでしょう。したがって、オッカムのかみそりに従って、「より単純な」平行傾斜モデルを選択すべきです。

交互作用モデルの回帰表には6行あるのに対し、平行傾斜モデルの回帰表には4行しかありません。これが交互作用モデルの「複雑さ」を反映しています。

さらに、交互作用モデルでは*傾きのオフセット*「perc_disadvan:sizemedium」が0.146で「perc_disadvan:sizelarge」が0.189と、小規模校の*ベースライングループの傾き*-2.932に比べて小さいことに注目してください。言い換えると、3つの傾きは同様に負です：小規模校では-2.932、中規模校では-2.786（=-2.932 + 0.146）、大規模校では-2.743（=-2.932 + 0.189）。これらの結果は、学校規模に関係なく、数学SATスコアの平均と経済的に不利な生徒の割合との関係は類似していて、かなり負であることを示唆しています。

あなたが今行ったのは基本的な*モデル選択*です：複数の候補モデルの中からデータに最もよく合うモデルを選ぶプロセスです。今回行ったモデル選択は「目視テスト」を使用しました：視覚化を定性的に見てモデルを選択します。次のセクションでは、$R^2$（「R-二乗」と発音）値を使用した数値的アプローチでモデル選択を行います。

### R二乗を用いたモデル選択

前のセクションでは、マサチューセッツ州の高校の平均数学SATスコアを説明しようとする交互作用モデルと平行傾斜モデルを比較しました。交互作用モデルは「より複雑」であることがわかりました。また、3つの線（小規模、中規模、大規模高校に対応）があまり違わないことを観察し、「より単純な」平行傾斜モデルを選ぶべきだと主張しました。

このセクションでは、定性的な「目視テスト」ではなく、$R^2$要約統計量（「R-二乗」）を使用した数値的・定量的アプローチでモデル選択を行います。そのためには、数値変数の*分散*という新しい概念を導入する必要があります。

数値変数の*広がり*（または*変動*）の要約統計量としては、すでに標準偏差と四分位範囲（IQR）を学びました。ここでは、広がりの3つ目の要約統計量である*分散*を紹介します。分散は単に標準偏差の二乗です。

モデルから1)観測値 $y$、2)予測値 $\widehat{y}$、3)残差 $y - \widehat{y}$ を取得し、これらの値の分散を計算しましょう：

```python
# マサチューセッツ州学校データでの交互作用モデルと平行傾斜モデルの比較

# 交互作用モデルからの予測と残差
y_hat_interaction = interaction_model.predict(X_interaction)
residuals_interaction = y - y_hat_interaction

# 平行傾斜モデルからの予測と残差
y_hat_parallel = parallel_model.predict(X_parallel)
residuals_parallel = y - y_hat_parallel

# 分散計算（要約統計量を生成）
var_y = np.var(y)
var_y_hat_interaction = np.var(y_hat_interaction)
var_residual_interaction = np.var(residuals_interaction)
var_y_hat_parallel = np.var(y_hat_parallel)
var_residual_parallel = np.var(residuals_parallel)

# R²の計算
r2_interaction = var_y_hat_interaction / var_y
r2_parallel = var_y_hat_parallel / var_y

print("分散比較：")
print(f"Y(SATスコア)の分散: {var_y:.3f}")
print("\n交互作用モデル:")
print(f"予測値の分散: {var_y_hat_interaction:.3f}")
print(f"残差の分散: {var_residual_interaction:.3f}")
print(f"R²: {r2_interaction:.3f}")
print("\n平行傾斜モデル:")
print(f"予測値の分散: {var_y_hat_parallel:.3f}")
print(f"残差の分散: {var_residual_parallel:.3f}")
print(f"R²: {r2_parallel:.3f}")
print(f"\n差分: {r2_interaction - r2_parallel:.5f}")

# scikit-learnのR²スコアでも確認
from sklearn.metrics import r2_score
print("\nscikit-learnのr2_score()による計算：")
print(f"交互作用モデルR²: {r2_score(y, y_hat_interaction):.3f}")
print(f"平行傾斜モデルR²: {r2_score(y, y_hat_parallel):.3f}")

# 次に、UT Austin教員データで同様の分析を行う
# 交互作用モデルと平行傾斜モデルの比較
print("\n\nUT Austin教員データでのモデル比較：")

# 交互作用モデル
X_interaction = evals_ch6_encoded[['age', 'is_male', 'age_male_interaction']]
y = evals_ch6_encoded['score']
interaction_model_evals = LinearRegression()
interaction_model_evals.fit(X_interaction, y)
y_hat_interaction = interaction_model_evals.predict(X_interaction)
residuals_interaction = y - y_hat_interaction

# 平行傾斜モデル
X_parallel = evals_ch6_encoded[['age', 'is_male']]
parallel_model_evals = LinearRegression()
parallel_model_evals.fit(X_parallel, y)
y_hat_parallel = parallel_model_evals.predict(X_parallel)
residuals_parallel = y - y_hat_parallel

# 分散計算
var_y = np.var(y)
var_y_hat_interaction = np.var(y_hat_interaction)
var_residual_interaction = np.var(residuals_interaction)
var_y_hat_parallel = np.var(y_hat_parallel)
var_residual_parallel = np.var(residuals_parallel)

# R²計算
r2_interaction = var_y_hat_interaction / var_y
r2_parallel = var_y_hat_parallel / var_y

print("分散比較：")
print(f"Y(教育スコア)の分散: {var_y:.3f}")
print("\n交互作用モデル:")
print(f"予測値の分散: {var_y_hat_interaction:.3f}")
print(f"残差の分散: {var_residual_interaction:.3f}")
print(f"R²: {r2_interaction:.3f}")
print("\n平行傾斜モデル:")
print(f"予測値の分散: {var_y_hat_parallel:.3f}")
print(f"残差の分散: {var_residual_parallel:.3f}")
print(f"R²: {r2_parallel:.3f}")
print(f"\n差分: {r2_interaction - r2_parallel:.3f}")
print(f"増加率: {(r2_interaction/r2_parallel - 1)*100:.1f}%")
```

$y$ の分散は、予測値 $\widehat{y}$ の分散と残差の分散の和に等しいことがわかります。これらの3つの項は個別に何を示しているのでしょうか？

まず、$y$ の分散（$var(y)$として表記）は、マサチューセッツ州の高校の平均数学SATスコアがどれだけ異なるかを示しています。回帰モデリングの目的は、この変動を*説明する*モデルを適合させることです。つまり、特定の学校がなぜ高いSATスコアを持ち、他の学校が低いスコアを持つのかを理解したいのです。これはモデルとは独立しています。つまり、交互作用モデルでも平行傾斜モデルでも、$var(y)$ は同じままです。

次に、$\widehat{y}$ の分散（$var(\widehat{y})$として表記）は、交互作用モデルからの予測値がどれだけ変動するかを示しています。つまり、(1)学生の社会経済的に不利な割合と(2)学校規模を交互作用モデルで考慮した後、平均数学SATスコアに関するモデルの説明がどの程度変動するかということです。

第三に、残差の分散はモデルからの「残り物」がどれだけ変動するかを示しています。図の点が3つの線の周りにどのように散らばっているかを観察してください。もし全ての点が3つの線のいずれかに*正確に*乗っていたとしたら、全ての残差はゼロになり、残差の分散もゼロになります。

これで $R^2$ を紹介する準備ができました：

$$
R^2 = \frac{var(\widehat{y})}{var(y)}
$$

これは*結果変数 $y$ の広がり/変動のうち、モデルによって説明される割合*です。ここでモデルの説明力は予測値 $\widehat{y}$ に組み込まれています。さらに、数学的に $0 \leq var(\widehat{y}) \leq var(y)$ であることが証明できるため、次のことが保証されます：

$$
0 \leq R^2 \leq 1
$$

$R^2$ の解釈は次のようになります：

1. $R^2$ 値が0の場合、モデルは $y$ の変動の0%を説明することを示しています。マサチューセッツ州の高校データにモデルを適合させて $R^2 = 0$ を得たとします。これは、使用した説明変数 $x$ の組み合わせと選択したモデル形式（交互作用または平行傾斜）が平均数学SATスコアについて*何も*教えてくれないことを意味します。モデルの適合度が低いです。
2. $R^2$ 値が1の場合、モデルは $y$ の変動の100%を説明することを示しています。マサチューセッツ州の高校データにモデルを適合させて $R^2 = 1$ を得たとします。これは、使用した説明変数 $x$ の組み合わせと選択したモデル形式が平均数学SATスコアについて*知るべきすべて*を教えてくれることを意味します。

ただし実際には、$R^2$ 値が1になることはほとんどありません。マサチューセッツ州の高校のコンテキストで考えてみてください。特定の高校がSATで平均的に良い成績を収め、他の高校が良い成績を収められない理由に影響を与える要因は無数にあります。人間が設計した統計モデルがマサチューセッツ州のすべての高校生の異質性をすべて捉えることができるという考えは傲慢に近いでしょう。しかし、たとえそのようなモデルが完璧でなくても、教育政策の決定に役立つ可能性はあります。モデリングの一般原則として、統計学者のジョージ・ボックスの有名な引用を心に留めておくべきです：「すべてのモデルは間違っているが、いくつかは有用である」

$R^2$ の値が約0.829で、交互作用モデルと平行傾斜モデルでほぼ同じであることがわかります。つまり、交互作用モデルの*追加の複雑さ*は $R^2$ 値をほとんど改善しません。したがって、「より単純な」平行傾斜モデルを選択する傾向があります。

一方、UT Austin教員のスコアモデルでは、交互作用モデルの $R^2$ は平行傾斜モデルよりも大幅に高く、交互作用モデルの追加の複雑さが正当化されることを示しています。

### 相関係数

クレジットカードの「debt」と「income」（千ドル単位）の相関係数は0.464でした。これを「income」がドル単位（千ドル単位ではなく）で測定された場合はどうなるでしょうか？これは「income」に1000を掛けることで確認できます：

```python
# ドル単位での相関係数と千ドル単位での相関係数の比較
credit_ch6_modified = credit_ch6.copy()
credit_ch6_modified['income_dollars'] = credit_ch6_modified['income'] * 1000

corr_original = credit_ch6[['debt', 'income']].corr().iloc[0, 1]
corr_modified = credit_ch6_modified[['debt', 'income_dollars']].corr().iloc[0, 1]

print("相関係数の比較：")
print(f"収入（千ドル単位）と負債の相関: {corr_original:.3f}")
print(f"収入（ドル単位）と負債の相関: {corr_modified:.3f}")
```

相関係数は同じままです！これは相関係数が*線形変換に対して不変*であることを示しています。$x$ と $y$ の相関は、任意の数値 $a$ と $b$ について $a\cdot x + b$ と $y$ の相関と同じになります。

### シンプソンのパラドックス

セクション6.2では、クレジットカードの「debt」と「income」の関係を研究する際に、2つの矛盾する結果を見ました。一方では、「debt」と「income」の関係は*正*であるように見えました。しかし、多重回帰結果では、「debt」と「income」の関係は*負*であることが示されました。これらの矛盾する結果はどのように説明できるでしょうか？

これは*シンプソンのパラドックス*と呼ばれる現象によるものです。これは、集計されたデータに存在するトレンドが、データをグループに分解した場合に消えたり逆転したりする現象です。

シンプソンのパラドックスが「credit_ch6」データにどのように現れるかを示しましょう。まず、数値的説明変数「credit_limit」の分布をヒストグラムで可視化します：

```python
# credit_limitの分布をヒストグラムで可視化
plt.figure(figsize=(10, 6))
plt.hist(credit_ch6['credit_limit'], bins=30, color='skyblue', edgecolor='black')
# 四分位数を垂直線で示す
quartiles = np.percentile(credit_ch6['credit_limit'], [25, 50, 75])
for q in quartiles:
    plt.axvline(x=q, color='red', linestyle='--', linewidth=2)
plt.xlabel('信用限度額')
plt.ylabel('頻度')
plt.title('信用限度額の分布と4つの信用限度額区分')
plt.grid(True, alpha=0.3)
plt.show()

print("信用限度額の四分位数：")
print(f"25%: ${quartiles[0]:.0f}")
print(f"50%（中央値）: ${quartiles[1]:.0f}")
print(f"75%: ${quartiles[2]:.0f}")

# credit_limitを4つの等しいグループに分ける
credit_ch6['limit_bracket'] = pd.qcut(credit_ch6['credit_limit'], 4, 
                                     labels=['low', 'med-low', 'med-high', 'high'])

# シンプソンのパラドックスを示す図
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 左側のプロット：全体の関係
axes[0].scatter(credit_ch6['income'], credit_ch6['debt'])
m, b = np.polyfit(credit_ch6['income'], credit_ch6['debt'], 1)
x_range = np.array([min(credit_ch6['income']), max(credit_ch6['income'])])
axes[0].plot(x_range, m * x_range + b, 'r-', linewidth=2)
axes[0].set_xlabel('収入（千ドル）')
axes[0].set_ylabel('クレジットカード負債（ドル）')
axes[0].set_title('負債と収入の全体的な関係')
axes[0].grid(True, alpha=0.3)

# 右側のプロット：信用限度額区分で色分けした関係
colors = ['blue', 'green', 'orange', 'red']
brackets = credit_ch6['limit_bracket'].unique()

for bracket, color in zip(brackets, colors):
    subset = credit_ch6[credit_ch6['limit_bracket'] == bracket]
    axes[1].scatter(subset['income'], subset['debt'], color=color, alpha=0.6, label=f'{bracket}信用限度額')
    if len(subset) > 1:  # 少なくとも2点がないと回帰線が引けない
        m, b = np.polyfit(subset['income'], subset['debt'], 1)
        x_range = np.array([min(subset['income']), max(subset['income'])])
        axes[1].plot(x_range, m * x_range + b, color=color, linewidth=2)

axes[1].set_xlabel('収入（千ドル）')
axes[1].set_ylabel('クレジットカード負債（ドル）')
axes[1].set_title('負債と収入の関係（信用限度額区分で色分け）')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

# 各信用限度額区分での相関係数を計算
print("\n各信用限度額区分での相関係数：")
for bracket in brackets:
    subset = credit_ch6[credit_ch6['limit_bracket'] == bracket]
    corr = subset[['income', 'debt']].corr().iloc[0, 1]
    print(f"{bracket}信用限度額区分: {corr:.3f}")

# 全体の相関係数
print(f"\n全体の相関係数: {credit_ch6[['income', 'debt']].corr().iloc[0, 1]:.3f}")
```

垂直の破線は変数「credit_limit」を4つの等しいグループに分ける*四分位数*です。これらの四分位数を数値変数「credit_limit」を4つのレベルを持つ分類変数「信用限度額区分」に変換するものと考えましょう。つまり：

1. 信用限度額の25%は$0から$3088の間でした。これらの100人を「低」「credit_limit」区分に割り当てましょう。
2. 信用限度額の25%は$3088から$4622の間でした。これらの100人を「中低」「credit_limit」区分に割り当てましょう。
3. 信用限度額の25%は$4622から$5873の間でした。これらの100人を「中高」「credit_limit」区分に割り当てましょう。
4. 信用限度額の25%は$5873を超えていました。これらの100人を「高」「credit_limit」区分に割り当てましょう。

図の左側は「debt」と「income」の関係を*全体的に*示しており、全体的に正の関係があることを示唆しています。しかし、右側の図は「debt」と「income」の関係を*「credit_limit」区分ごとに分けて*示しています。つまり、「debt」と「income」の4つの*個別*の関係：「低」「credit_limit」区分の関係、「中低」「credit_limit」区分の関係などを示しています。

右側のプロットでは、「中低」と「中高」の「credit_limit」区分では「debt」と「income」の関係が明らかに負であり、「低」の「credit_limit」区分ではやや平坦です。「debt」と「income」の関係が正のままなのは「高」の「credit_limit」区分のみです。しかし、この関係も左側のプロットの回帰直線の傾きより緩やかで、全体の関係よりも弱い正の関係です。

シンプソンのパラドックスのこの例では、「credit_limit」は「debt」と「income」の関係の*交絡変数*です。したがって、「debt」と「income」の関係を適切にモデル化するには、「credit_limit」を考慮する必要があります。

## 6.4 結論

重回帰分析についての章を終えるにあたり、*モデル選択*の概念を紹介しました：候補となるモデルセットの中からデータに最もよく適合するモデルを選択するプロセスです。視覚化を使った「目視テスト」と$R^2$値を使った数値的アプローチの両方を使用してモデル選択を行いました。

さらに、シンプソンのパラドックスという興味深い統計現象も紹介しました：集計データに存在するトレンドが、データをグループに分解した場合に消えたり逆転したりする現象です。

おめでとうございます！これで「moderndiveによるデータモデリング」の部分が完了しました。次は「inferによる統計的推測」のパートIIIに進みます。統計的推測はサンプリングを使用して未知の量について推測する科学です。

最も有名なサンプリングの例として*世論調査*があります。人口全体に意見を尋ねることは長く困難な作業であるため、調査員は通常、母集団を代表するより小さなサンプルを取ります。このサンプルの結果に基づいて、調査員は母集団全体について主張することを望みます。

第7章（サンプリング）、第8章（信頼区間）、第9章（仮説検定）を学んだ後、第10章（回帰の推測）で第5章と第6章で学んだ回帰モデルを再訪します。これまでのところ、回帰表の「estimate」列のみを研究してきましたが、次の4つの章では残りの列の意味に焦点を当てます：標準誤差（「std_error」）、検定統計量（「statistic」）、p値（「p_value」）、信頼区間の下限と上限（「lower_ci」と「upper_ci」）。

さらに第10章では、残差 $y - \widehat{y}$ の概念を再訪し、回帰モデルの結果を解釈する際のその重要性について説明します。すべての「get_regression_points()」出力の「residual」変数の*残差分析*を行います。残差分析により、*回帰の推測のための条件*として知られるものを検証できます。