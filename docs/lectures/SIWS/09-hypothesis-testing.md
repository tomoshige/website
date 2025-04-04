# 第9章 仮説検定

## はじめに

第8章の信頼区間について学んだので、今度は統計的推論のもう一つのよく使われる方法である仮説検定について学びましょう。仮説検定により、母集団からのサンプルデータを用いて、競合する仮説の妥当性について推論することができます。例えば、これから見る「昇進」の例（9.2節）では、1970年代の心理学研究から収集されたデータを使用して、当時の銀行業界に性別に基づく昇進差別が存在したかどうかを調査します。

第7章と第8章ですでに多くの必要な概念を学んできました。これらの考え方をさらに拡張し、仮説検定の一般的なフレームワークを提供します。このフレームワークを理解すれば、さまざまなシナリオに適用できるようになります。

信頼区間と同様に、仮説検定にも一般的なフレームワークがあります。`infer`パッケージはこのフレームワークに基づいて設計されています。特定の種類の仮説検定のための理論ベースのアプローチに焦点を当てるよりも、長期的な学習のためには一般的なフレームワークを学ぶ方が良いと考えています。

必要なライブラリをインポートしましょう：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random

# Google Colabで実行している場合、matplotlibの設定を変更して日本語表示を可能にする
import matplotlib as mpl
mpl.rcParams['font.family'] = 'IPAexGothic'

# データの表示形式を設定
pd.set_option('display.max_columns', None)

# 乱数シードを設定
np.random.seed(76)
random.seed(76)

# スタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
```

## 9.1 昇進活動

まず、銀行における昇進に対する性別の影響を調査する活動から始めましょう。

### 銀行での昇進に性別は影響するか？

1970年代に銀行で働いていて、昇進の申請のために履歴書を提出するとします。あなたの性別は昇進の機会に影響するでしょうか？この質問に答えるために、1974年に『Journal of Applied Psychology』で発表された研究のデータに焦点を当てます。

この研究では、48人の銀行の監督者が「銀行の支店長」という架空の役割を担うように求められました。それぞれの監督者に履歴書が与えられ、その候補者が新しいポジションへの昇進に適しているかどうかを尋ねられました。

しかし、これらの48通の履歴書は、履歴書の一番上にある応募者の名前を除いて、すべての点で同一でした。監督者のうち24人にはステレオタイプ的に「男性」の名前の履歴書が無作為に与えられ、残りの24人には「女性」の名前の履歴書が無作為に与えられました。名前の性別のみが履歴書ごとに異なっていたため、研究者はこの変数が昇進率に与える影響を分離することができました。

今日の視点（私たち著者を含む）では、このような性別の二元的な見方に同意しない人も多いですが、この研究が行われた時代には、より微妙な性別の見方が一般的ではなかったことを覚えておくことが重要です。

まず、GitHubからデータをダウンロードして調査しましょう：

```python
# GitHubからデータをダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/promotions.csv"
promotions = pd.read_csv(url)

# ランダムにいくつかの行を選択して表示
promotions.sample(6).sort_values('id')
```

`id`変数は48行すべての識別変数として機能し、`decision`変数は応募者が昇進に選ばれたかどうかを示し、`gender`変数は履歴書に使用された名前の性別を示します。このデータは実際の24人の男性と24人の女性に関するものではなく、24個は「男性」の名前が、24個は「女性」の名前が割り当てられた同一の48通の履歴書に関するものであることを思い出してください。

2つのカテゴリ変数`decision`と`gender`の関係を探索的に分析してみましょう。積み上げ棒グラフを使用してこの関係を視覚化できます：

```python
# 積み上げ棒グラフを作成
plt.figure(figsize=(8, 4))
data = pd.crosstab(promotions['gender'], promotions['decision'])
data.plot(kind='bar', stacked=True)
plt.xlabel('履歴書の名前の性別')
plt.ylabel('人数')
plt.title('性別と昇進決定の関係')
plt.legend(title='決定')
plt.tight_layout()
plt.show()
```

図を見ると、女性の名前の履歴書は昇進に選ばれる可能性がはるかに低いように見えます。これらの昇進率を数値で表現しましょう：

```python
# 性別と決定によるクロス集計表を作成
cross_tab = pd.crosstab(promotions['gender'], promotions['decision'], margins=True)
print(cross_tab)

# 昇進率を計算
promotion_rates = pd.crosstab(promotions['gender'], promotions['decision'], normalize='index')
print("\n昇進率（行ごとの割合）:")
print(promotion_rates['promoted'])
```

男性の名前の履歴書24通のうち、21通が昇進に選ばれました（割合は21/24 = 0.875 = 87.5%）。一方、女性の名前の履歴書24通のうち、14通が昇進に選ばれました（割合は14/24 = 0.583 = 58.3%）。これらの昇進率を比較すると、男性の名前の履歴書は女性の名前の履歴書よりも0.875 - 0.583 = 0.292 = 29.2%高い割合で昇進に選ばれました。これは男性の名前の履歴書に有利な傾向を示唆しています。

しかし、これは銀行での昇進における性別差別が存在するという*決定的な*証拠を提供するでしょうか？性別差別が存在しない仮想的な世界でも、29.2%の昇進率の差が偶然によって生じる可能性はあるでしょうか？言い換えれば、この仮想的な世界では*標本変動*がどのような役割を果たすでしょうか？この質問に答えるために、再び*シミュレーション*を実行するためにコンピュータを使用します。

### 一度のシャッフル

まず、性差別が存在しない仮想的な宇宙を想像してみてください。このような仮想的な宇宙では、応募者の性別は昇進の可能性に影響を与えません。私たちの`promotions`データフレームに戻ると、`gender`変数は関係のないラベルになります。これらの`gender`ラベルが関係ないのであれば、それらをランダムに再割り当て（「シャッフル」）しても結果に影響しないはずです！

この考え方を説明するために、48通の履歴書のうち任意に選んだ6通に焦点を当ててみましょう。以下の表は、`decision`列は3つの履歴書が昇進につながり、3つがつながらなかったことを示しています。`gender`列は、元の履歴書の名前の性別を示しています。

しかし、性差別がないという仮想的な宇宙では、性別は無関係であり、`gender`値をランダムに「シャッフル」しても問題ありません。`shuffled_gender`列は、可能な1つのランダムなシャッフルを示しています。第4列では、男性と女性の名前の数がそれぞれ3つのままですが、異なる順序で表示されていることに注目してください。

このような性別ラベルのランダムなシャッフルは、性差別が存在しない仮想的な宇宙でのみ意味を持ちます。この性別変数のシャッフルを48通すべての履歴書に手動で拡張するにはどうすればよいでしょうか？52枚の標準的なトランプを使用する方法があります。

カードの半分が赤（ダイヤとハート）でもう半分が黒（スペードとクラブ）なので、赤のカード2枚と黒のカード2枚を取り除けば、24枚の赤のカードと24枚の黒のカードが残ります。これらの48枚のカードをシャッフルした後、1枚ずつめくり、赤のカードごとに「男性」、黒のカードごとに「女性」を割り当てることができます。

このようなシャッフルの1例をPythonで実行してみましょう：

```python
# シャッフルされた性別を作成
np.random.seed(42)  # 再現性のために乱数シードを設定
shuffled_gender = promotions['gender'].sample(frac=1).reset_index(drop=True)

# 元のデータと同じ意思決定を保持しながら、シャッフルされた性別を持つ新しいデータフレームを作成
promotions_shuffled = promotions.copy()
promotions_shuffled['gender'] = shuffled_gender

# 元のデータとシャッフルされたデータの棒グラフを作成して比較
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 元のデータ
pd.crosstab(promotions['gender'], promotions['decision']).plot(
    kind='bar', stacked=True, ax=axes[0], title='元のデータ')
axes[0].set_xlabel('履歴書の名前の性別')
axes[0].set_ylabel('人数')

# シャッフルされたデータ
pd.crosstab(promotions_shuffled['gender'], promotions_shuffled['decision']).plot(
    kind='bar', stacked=True, ax=axes[1], title='シャッフルされたデータ')
axes[1].set_xlabel('履歴書の名前の性別')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
```

元のデータの左側のバープロットと比較して、右側の新しい「シャッフルされた」データでは「男性」と「女性」の昇進率が似通っています。

各グループの昇進率も計算してみましょう：

```python
# シャッフルされたデータの昇進率を計算
shuffled_rates = pd.crosstab(
    promotions_shuffled['gender'], 
    promotions_shuffled['decision'], 
    normalize='index'
)

print("シャッフルされたデータの昇進率:")
print(shuffled_rates['promoted'])

# 差を計算
print(f"\n男性と女性の昇進率の差: {shuffled_rates.loc['male', 'promoted'] - shuffled_rates.loc['female', 'promoted']:.3f}")
```

性差別がない仮想的な世界では、「男性」履歴書の70.8%が昇進に選ばれました。一方、「女性」履歴書の75.0%が昇進に選ばれました。これらの2つの値を比較すると、「男性」履歴書と「女性」履歴書の選ばれた割合に-4.2%の差があることがわかります。

この差は、元々観察された差である29.2%とは異なることに注目してください。これは再び*標本変動*によるものです。この標本変動の効果をよりよく理解するにはどうすればよいでしょうか？このシャッフルを何度も繰り返すことです！

### 16回のシャッフル

16人の友人グループがこのシャッフル演習を繰り返すように依頼しました。彼らはこれらの値を共有スプレッドシートに記録しました。

各シャッフルに対して、昇進率の差を計算し、ヒストグラムでそれらの分布を表示してみましょう。また、実際に観測された昇進率の差29.2%を縦線で示します。

```python
# 16回のシャッフルをシミュレーション
np.random.seed(123)
n_shuffles = 16
results = []

for i in range(n_shuffles):
    # 性別をシャッフル
    shuffled_gender = promotions['gender'].sample(frac=1).reset_index(drop=True)
    
    # 新しいデータフレームを作成
    promotions_shuffled = promotions.copy()
    promotions_shuffled['gender'] = shuffled_gender
    
    # 男性と女性の昇進率を計算
    rates = pd.crosstab(
        promotions_shuffled['gender'], 
        promotions_shuffled['decision'], 
        normalize='index'
    )
    
    # 差を計算して保存
    diff = rates.loc['male', 'promoted'] - rates.loc['female', 'promoted']
    results.append(diff)

# 結果をプロット
plt.figure(figsize=(8, 5))
plt.hist(results, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=0.292, color='red', linestyle='-', linewidth=2, 
           label='観測された差 (29.2%)')
plt.xlabel('昇進率の差 (男性 - 女性)')
plt.ylabel('頻度')
plt.title('シャッフルされた差の分布')
plt.legend()
plt.tight_layout()
plt.show()
```

このヒストグラムについて議論する前に、最も重要なことを強調しておきます：このヒストグラムは、性差別がない*仮説上の宇宙*で観察されるであろう昇進率の差を表しています。

まず、ヒストグラムがほぼ0を中心にしていることがわかります。昇進率の差が0であるということは、両方の性別が同じ昇進率を持っていることを意味します。言い換えれば、これらの16個の値の中心は、性差別がない仮説上の宇宙で予想されることと一致しています。

しかし、値は0を中心にしていますが、0の周りには変動があります。これは、性差別がない仮説上の宇宙でも、偶然の*標本変動*によって昇進率に小さな差が依然として観察される可能性があるためです。

実生活で観察されたものに目を向けると：29.2%の差は縦の濃い線でマークされています。あなた自身に尋ねてみてください：性差別のない仮説上の世界で、このような差が観察される可能性はどれくらいでしょうか？意見が分かれるかもしれませんが、私たちの意見では、そうそうあることではありません！次に自問してください：これらの結果は、性差別のない仮説上の宇宙について何を物語っていますか？

### 今何をしたのか？

今示した活動は、*置換検定*を使用した*仮説検定*という統計的手順です。「置換」という用語は「シャッフル」の数学的用語です：一連の値をランダムに並べ替えることです。

実際、置換は第8章でブートストラップ法とともに実行した*再サンプリング*の別の形式です。ブートストラップ法は*置換*による再サンプリングを含む一方、置換法は*置換なし*の再サンプリングを含みます。

第8章の紙片を使ったペニーと帽子の演習を思い出してください：ペニーをサンプリングした後、それを帽子に戻しました。今度はトランプのデッキについて考えてみましょう。カードを引いた後、それを目の前に置き、色を記録し、*デッキに戻しませんでした*。

前の例では、性差別のない仮説上の宇宙の妥当性をテストしました。48通の履歴書のサンプルにある証拠は、この仮説上の宇宙と矛盾しているようでした。したがって、この仮説上の宇宙を*棄却*し、性差別が存在すると宣言する傾向があります。

第8章のあくびは伝染するかどうかのケーススタディを思い出してください。前の例も母集団の比率の差についての推論を含んでいます。今回は、$p_{m} - p_{f}$となります。ここで、$p_{m}$は男性の名前の履歴書が昇進を推薦される母集団の比率、$p_{f}$は女性の名前の履歴書に対する同等の比率です。

## 9.2 仮説検定の理解

サンプリングに関する用語、表記法、定義について第7章で学んだように、仮説検定に関連する多くの用語、表記法、定義もあります。これらを学ぶことは最初は非常に困難な作業のように思えるかもしれません。しかし、練習を重ねることで、誰でもそれらをマスターすることができます。

まず、**仮説**は未知の母集団パラメータの値に関する声明です。履歴書の活動では、私たちの関心のある母集団パラメータは母集団比率の差$p_{m} - p_{f}$でした。

次に、**仮説検定**は2つの競合する仮説間の検定で構成されます：(1) **帰無仮説** $H_0$（「Hノート」と発音）対(2) **対立仮説** $H_A$（$H_1$とも表記）。

一般的に帰無仮説は「効果なし」または「関心のある差なし」という主張です。多くの場合、帰無仮説は現状または面白いことが何も起こっていない状況を表します。さらに、一般的に対立仮説は実験者または研究者が確立したいまたは支持する証拠を見つけたい主張です。これは帰無仮説$H_0$に対する「挑戦者」仮説と見なされます。履歴書の活動では、適切な仮説検定は次のようになります：

$$
\begin{aligned}
H_0 &: \text{男性と女性は同じ昇進率である}\\
\text{vs } H_A &: \text{男性の昇進率は女性よりも高い}
\end{aligned}
$$

私たちが行った選択のいくつかに注目してください。まず、帰無仮説$H_0$を昇進率に差がないことに設定し、「挑戦者」対立仮説$H_A$を差があることに設定しました。原則的には二つを逆にしても間違いではありませんが、統計的推論では、帰無仮説を「何も起こっていない」状況を反映するように設定することが慣例です。先ほど述べたように、この場合、$H_0$は昇進率に差がないことに対応します。さらに、$H_A$を男性が*より高い*昇進率を持つように設定しました。これは男性に有利な差別があるという事前の疑いを反映した主観的な選択です。このような対立仮説を*片側対立仮説*と呼びます。しかし、そのような疑いを共有せず、より高いかより低いかにかかわらず差があるかどうかだけを調査したい場合、*両側対立仮説*として知られるものを設定します。

母集団パラメータ$p_{m} - p_{f}$に対する数学的表記を使用して、仮説検定の定式化を次のように再表現できます：

$$
\begin{aligned}
H_0 &: p_{m} - p_{f} = 0\\
\text{vs } H_A&: p_{m} - p_{f} > 0
\end{aligned}
$$

対立仮説$H_A$が片側であり、$p_{m} - p_{f} > 0$であることに注目してください。両側対立仮説を選択していた場合、$p_{m} - p_{f} \neq 0$と設定していたでしょう。今のところ簡単にするために、より単純な片側対立仮説を使用しましょう。9.5節で両側対立仮説の例を紹介します。

第三に、**検定統計量**は仮説検定に使用される*点推定値/標本統計量*の式です。標本統計量は単にサンプル観測値に基づく要約統計量であることに注意してください。第5章で見たように、要約統計量は多くの値を取り、一つだけを返します。ここでのサンプルは、男性名の履歴書$n_m$ = 24と女性名の履歴書$n_f$ = 24になります。したがって、関心のある点推定値は標本比率の差$\hat{p}_{m} - \hat{p}_{f}$です。

第四に、**観測検定統計量**は実生活で観測された検定統計量の値です。私たちの場合、この値は`promotions`データフレームに保存されたデータを使用して計算しました。それは男性名の履歴書に有利な$\hat{p}_{m} -\hat{p}_{f} = 0.875 - 0.583 = 0.292 = 29.2\%$の観測された差でした。

第五に、**帰無分布**は*帰無仮説$H_0$が真であると仮定した場合の*検定統計量のサンプリング分布です。これは長い説明です！少しずつ解きほぐしましょう。帰無分布を理解するための鍵は、帰無仮説$H_0$が真であると*仮定*されていることです。この時点で$H_0$が真であるとは言っていません。仮説検定の目的のためにのみ真であると仮定しています。私たちの場合、これは昇進率に性差別がない仮説上の宇宙に対応します。帰無仮説$H_0$を仮定すると（「$H_0$の下で」とも言います）、検定統計量はサンプリング変動によってどのように変化するでしょうか？私たちの場合、標本比率の差$\hat{p}_{m} - \hat{p}_{f}$は$H_0$の下でのサンプリングによってどのように変化するでしょうか？第7章から、点推定値がサンプリング変動によってどのように変化するかを示す分布は*サンプリング分布*と呼ばれることを思い出してください。帰無分布について覚えておくべき唯一の追加の点は、それらが*帰無仮説$H_0$が真であると仮定した*サンプリング分布であることです。

私たちの場合、以前に帰無分布を可視化しました。それは友人たちが計算した16の標本比率の差の分布で、性差別のない仮説上の宇宙を*仮定*しています。また、観測検定統計量の値である0.292を縦線でマークしました。

第六に、**$p$値**は*帰無仮説$H_0$が真であると仮定した場合に*、観測検定統計量と同じくらい極端またはより極端な検定統計量を得る確率です。これもゆっくり解きほぐしましょう。$p$値を「驚き」の量的表現と考えることができます：$H_0$が真であると仮定すると、私たちが観察したことにどれくらい驚くでしょうか？あるいは、私たちの場合、性差別のない仮説上の宇宙では、$H_0$が真であると仮定して、サンプルから0.292の昇進率の差を観察することにどれくらい驚くでしょうか？非常に驚くでしょうか？やや驚くでしょうか？

$p$値はこの確率を数値化します。あるいは、図の16の標本比率の差の場合、どれくらいの割合がより「極端な」結果を持っていたでしょうか？ここで、極端とは「男性」応募者が「女性」応募者よりも高い昇進率を持つという対立仮説$H_A$の観点から定義されます。言い換えれば、男性に有利な差別が$0.875 - 0.583 = 0.292 = 29.2\%$よりも_さらに_顕著だったのはどれくらいの頻度でしょうか？

この場合、16回中0回、観測された差0.292以上の比率の差を得ました。非常にまれな（実際には発生しない）結果です！仮説上の性差別のない宇宙でこのような顕著な昇進率の差がまれであることを考えると、私たちは仮説上の宇宙を*棄却*する傾向があります。代わりに、「男性」応募者に有利な差別が存在するという仮説を支持します。言い換えれば、$H_0$を$H_A$のために棄却します。

第七に最後に、多くの仮説検定手順では、あらかじめ検定の**有意水準**を設定することが一般的に推奨されています。ギリシャ文字$\alpha$（「アルファ」と発音）で表されます。この値は$p$値のカットオフとして機能し、$p$値が$\alpha$を下回る場合、「帰無仮説$H_0$を棄却する」ことになります。

あるいは、$p$値が$\alpha$を下回らない場合、「$H_0$を棄却しない」ことになります。後者の声明は「$H_0$を受け入れる」というのとは少し異なることに注意してください。この区別はかなり微妙であり、すぐには明らかではありません。そのため、9.4節で再訪します。

異なる分野では異なる$\alpha$の値が使用される傾向がありますが、0.1、0.01、0.05などのよく使われる値があります。0.05は多くの人があまり考えずに選択することが多い値です。9.4節で$\alpha$有意水準についてさらに詳しく説明しますが、まず、`infer`パッケージを使用して昇進活動に対応する仮説検定を完全に実施しましょう。

## 9.3 仮説検定の実施

Rの`infer`パッケージと同様の機能をPythonで実装して、仮説検定のフレームワークを実施しましょう。以下では、`infer`のワークフローに似た関数を作成します：

```python
def specify(data, formula=None, response=None, explanatory=None, success=None):
    """
    式または応答・説明変数を指定してデータを準備します
    """
    result = data.copy()
    
    if formula is not None:
        # formula = 'response ~ explanatory' の形式を解析
        parts = formula.split('~')
        response = parts[0].strip()
        explanatory = parts[1].strip() if len(parts) > 1 else None
    
    # データに関するメタ情報を設定
    result.attrs['response'] = response
    result.attrs['explanatory'] = explanatory
    result.attrs['success'] = success
    
    return result

def hypothesize(data, null='independence'):
    """
    帰無仮説を指定します
    """
    result = data.copy()
    result.attrs['null'] = null
    result.attrs.update(data.attrs)
    return result

def generate(data, reps=1, type='permute'):
    """
    シミュレーションを生成します
    """
    result_list = []
    
    # データの属性を取得
    response = data.attrs.get('response')
    explanatory = data.attrs.get('explanatory')
    success = data.attrs.get('success')
    null = data.attrs.get('null')
    
    if type == 'permute':
        for i in range(reps):
            if null == 'independence':
                # 説明変数をシャッフル
                shuffled_data = data.copy()
                shuffled_data[explanatory] = np.random.permutation(data[explanatory].values)
                shuffled_data['replicate'] = i + 1
                result_list.append(shuffled_data)
    
    result = pd.concat(result_list, ignore_index=True)
    result.attrs.update(data.attrs)
    return result

def calculate(data, stat='diff in props', order=None):
    """
    要約統計量を計算します
    """
    response = data.attrs.get('response')
    explanatory = data.attrs.get('explanatory')
    success = data.attrs.get('success')
    
    if stat == 'diff in props':
        if 'replicate' in data.columns:
            # 複数の複製に対して計算
            result = []
            for rep, group in data.groupby('replicate'):
                prop_table = pd.crosstab(
                    group[explanatory], 
                    group[response] == success,
                    normalize='index'
                )[True]
                
                if order is not None:
                    diff = prop_table[order[0]] - prop_table[order[1]]
                else:
                    # デフォルトの順序
                    diff = prop_table.iloc[0] - prop_table.iloc[1]
                    
                result.append({'replicate': rep, 'stat': diff})
            
            return pd.DataFrame(result)
        else:
            # 単一のデータセットに対して計算
            prop_table = pd.crosstab(
                data[explanatory], 
                data[response] == success,
                normalize='index'
            )[True]
            
            if order is not None:
                diff = prop_table[order[0]] - prop_table[order[1]]
            else:
                # デフォルトの順序
                diff = prop_table.iloc[0] - prop_table.iloc[1]
                
            return pd.DataFrame({'stat': [diff]})
    
    elif stat == 't':
        # t統計量の計算
        if 'replicate' in data.columns:
            # 複数の複製に対して計算
            result = []
            for rep, group in data.groupby('replicate'):
                grouped = group.groupby(explanatory)[response]
                group1 = grouped.get_group(order[0])
                group2 = grouped.get_group(order[1])
                
                n1, n2 = len(group1), len(group2)
                mean1, mean2 = group1.mean(), group2.mean()
                var1, var2 = group1.var(), group2.var()
                
                # 2標本のt統計量
                t_stat = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)
                result.append({'replicate': rep, 'stat': t_stat})
                
            return pd.DataFrame(result)
        else:
            # 単一のデータセットに対して計算
            grouped = data.groupby(explanatory)[response]
            group1 = grouped.get_group(order[0])
            group2 = grouped.get_group(order[1])
            
            n1, n2 = len(group1), len(group2)
            mean1, mean2 = group1.mean(), group2.mean()
            var1, var2 = group1.var(), group2.var()
            
            # 2標本のt統計量
            t_stat = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)
            return pd.DataFrame({'stat': [t_stat]})
    
    return None

def get_p_value(null_distribution, obs_stat, direction='greater'):
    """
    p値を計算します
    """
    observed = obs_stat['stat'].iloc[0]
    
    if direction == 'greater':
        p_value = (null_distribution['stat'] >= observed).mean()
    elif direction == 'less':
        p_value = (null_distribution['stat'] <= observed).mean()
    elif direction == 'two-sided' or direction == 'both':
        p_value = (abs(null_distribution['stat']) >= abs(observed)).mean()
    
    return pd.DataFrame({'p_value': [p_value]})

def visualize(null_distribution, bins=10, obs_stat=None, direction=None):
    """
    帰無分布を可視化します
    """
    plt.figure(figsize=(10, 6))
    
    # ヒストグラムの作成
    plt.hist(null_distribution['stat'], bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 観測された統計量と方向が提供されている場合、p値を影付けする
    if obs_stat is not None and direction is not None:
        observed = obs_stat['stat'].iloc[0]
        plt.axvline(x=observed, color='red', linestyle='-', linewidth=2, 
                   label=f'観測値: {observed:.3f}')
        
        if direction == 'greater':
            # 右側を強調表示
            x = np.linspace(observed, null_distribution['stat'].max(), 100)
            plt.fill_between(x, 0, 1, alpha=0.3, color='red', transform=plt.gca().get_xaxis_transform())
            plt.title(f'帰無分布 (右側 p値)')
        elif direction == 'less':
            # 左側を強調表示
            x = np.linspace(null_distribution['stat'].min(), observed, 100)
            plt.fill_between(x, 0, 1, alpha=0.3, color='red', transform=plt.gca().get_xaxis_transform())
            plt.title(f'帰無分布 (左側 p値)')
        elif direction == 'two-sided' or direction == 'both':
            # 両側を強調表示
            abs_observed = abs(observed)
            x_left = np.linspace(null_distribution['stat'].min(), -abs_observed, 100)
            x_right = np.linspace(abs_observed, null_distribution['stat'].max(), 100)
            plt.fill_between(x_left, 0, 1, alpha=0.3, color='red', transform=plt.gca().get_xaxis_transform())
            plt.fill_between(x_right, 0, 1, alpha=0.3, color='red', transform=plt.gca().get_xaxis_transform())
            plt.title(f'帰無分布 (両側 p値)')
    else:
        plt.title('帰無分布')
    
    plt.xlabel('統計量')
    plt.ylabel('度数')
    
    if obs_stat is not None:
        plt.legend()
    
    plt.tight_layout()
    return plt
```

これで、`infer`パッケージと同様のワークフローで仮説検定を実行できるようになりました。以下のステップに従って検定を行います。

### 1. 変数を指定する

まず、`promotions`データフレームの興味のある変数を指定します。`decision`を応答変数、`gender`を説明変数として、また「昇進した」という成功イベントに興味があるため、`success='promoted'`と設定します。

```python
# 変数を指定
promotions_specified = specify(
    promotions, 
    formula='decision ~ gender', 
    success='promoted'
)
```

### 2. 帰無仮説を指定する

帰無仮説を明示的に指定します。この場合、2つの独立したサンプルがあるため、`null='independence'`を設定します。

```python
# 帰無仮説を指定
promotions_hypothesized = hypothesize(
    promotions_specified, 
    null='independence'
)
```

### 3. 複製を生成する

帰無仮説が真であると仮定してシミュレーションを1000回実行します。

```python
# 帰無仮説の下でのシミュレーションを生成
np.random.seed(42)
promotions_generated = generate(
    promotions_hypothesized, 
    reps=1000, 
    type='permute'
)
```

### 4. 統計量を計算する

各シミュレーションに対して適切な統計量を計算します。この場合、男性と女性の間の比率の差に関心があります。

```python
# 帰無分布を計算
null_distribution = calculate(
    promotions_generated, 
    stat='diff in props', 
    order=['male', 'female']
)

# 観測された統計量を計算
obs_diff_prop = calculate(
    promotions_specified, 
    stat='diff in props', 
    order=['male', 'female']
)

print(f"観測された差: {obs_diff_prop['stat'].iloc[0]:.3f}")
```

### 5. p値を可視化する

最後に、帰無分布と観測値を可視化し、p値を計算します。

```python
# 帰無分布を可視化して右側p値を強調表示
visualize(null_distribution, bins=15, obs_stat=obs_diff_prop, direction='greater')
plt.show()

# p値を計算
p_value = get_p_value(null_distribution, obs_stat=obs_diff_prop, direction='greater')
print(f"p値: {p_value['p_value'].iloc[0]:.3f}")
```

p値の定義を念頭に置くと、帰無分布内で観測値0.292以上のサンプリング変動だけによる昇進率の差を観察する確率は非常に小さいことがわかります。このp値はあらかじめ指定した有意水準α = 0.05よりも小さいため、帰無仮説$H_0: p_{m} - p_{f} = 0$を棄却する傾向があります。統計的でない言葉で言えば、結論は次のようになります：このサンプルデータには、性別に基づく差別が存在するという仮説を棄却すべきではないという十分な証拠があります。男性の名前の履歴書に有利な性別差別が原因である可能性が高いです。

### 信頼区間との比較

`infer`パッケージの優れた点の一つは、最小限の変更で仮説検定と信頼区間の構築を切り替えられることです。帰無分布を作成するためのコードを思い出しましょう：

```python
# 帰無分布を作成するコード
promotions_specified = specify(promotions, formula='decision ~ gender', success='promoted')
promotions_hypothesized = hypothesize(promotions_specified, null='independence')
promotions_generated = generate(promotions_hypothesized, reps=1000, type='permute')
null_distribution = calculate(promotions_generated, stat='diff in props', order=['male', 'female'])
```

$p_{m} - p_{f}$の95%信頼区間を構築するために必要なブートストラップ分布を作成するには、2つの変更だけが必要です。まず、帰無仮説を想定していないため、`hypothesize()`ステップを削除します。次に、`generate()`ステップの再サンプリングの`type`を`'permute'`から`'bootstrap'`に変更します。

Pythonでブートストラップを実装しましょう：

```python
def bootstrap_generate(data, reps=1, size=None):
    """
    ブートストラップサンプルを生成します
    """
    result_list = []
    
    # データの属性を取得
    response = data.attrs.get('response')
    explanatory = data.attrs.get('explanatory')
    
    if size is None:
        size = len(data)
    
    for i in range(reps):
        # データから置換ありでリサンプリング
        bootstrap_sample = data.sample(n=size, replace=True)
        bootstrap_sample['replicate'] = i + 1
        result_list.append(bootstrap_sample)
    
    result = pd.concat(result_list, ignore_index=True)
    result.attrs.update(data.attrs)
    return result

def get_confidence_interval(bootstrap_distribution, level=0.95, type='percentile', point_estimate=None):
    """
    ブートストラップ分布から信頼区間を計算します
    """
    alpha = 1 - level
    
    if type == 'percentile':
        lower = bootstrap_distribution['stat'].quantile(alpha/2)
        upper = bootstrap_distribution['stat'].quantile(1 - alpha/2)
        return pd.DataFrame({'lower_ci': [lower], 'upper_ci': [upper]})
    
    elif type == 'se' and point_estimate is not None:
        # 標準誤差法
        se = bootstrap_distribution['stat'].std()
        point_est = point_estimate['stat'].iloc[0]
        z = stats.norm.ppf(1 - alpha/2)  # 通常、95%信頼区間では1.96
        
        lower = point_est - z * se
        upper = point_est + z * se
        return pd.DataFrame({'lower_ci': [lower], 'upper_ci': [upper]})
    
    return None
```

これらの関数を使用して、95%信頼区間を構築しましょう：

```python
# ブートストラップ分布を作成
promotions_specified = specify(promotions, formula='decision ~ gender', success='promoted')
# hypothesize()ステップを削除
bootstrap_samples = bootstrap_generate(promotions_specified, reps=1000)
bootstrap_distribution = calculate(bootstrap_samples, stat='diff in props', order=['male', 'female'])

# パーセンタイル法で95%信頼区間を構築
percentile_ci = get_confidence_interval(bootstrap_distribution, level=0.95, type='percentile')
print("パーセンタイル法の95%信頼区間:")
print(percentile_ci)

# 標準誤差法で95%信頼区間を構築
se_ci = get_confidence_interval(bootstrap_distribution, level=0.95, type='se', point_estimate=obs_diff_prop)
print("\n標準誤差法の95%信頼区間:")
print(se_ci)

# ブートストラップ分布とパーセンタイル法の信頼区間を可視化
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_distribution['stat'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=percentile_ci['lower_ci'].iloc[0], color='black', linestyle='-', linewidth=2)
plt.axvline(x=percentile_ci['upper_ci'].iloc[0], color='black', linestyle='-', linewidth=2)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='0の値')
plt.xlabel('比率の差 (男性 - 女性)')
plt.ylabel('頻度')
plt.title('ブートストラップ分布とパーセンタイル法の95%信頼区間')
plt.legend()
plt.tight_layout()
plt.show()
```

95%信頼区間に含まれていない重要な値があることに注目してください：0の値です。つまり、$p_{m}$と$p_{f}$の差が0という値は私たちの「ネット」に含まれておらず、$p_{m}$と$p_{f}$が真に異なることを示唆しています！さらに、$p_{m} - p_{f}$の95%信頼区間全体が0より上にあることを観察してください。これは男性に有利な差であることを示唆しています。

## 9.4 仮説検定の解釈

仮説検定の結果を解釈することは、この統計的推論の方法の中でより難しい側面の一つです。このセクションでは、プロセスの解読を助ける方法と、いくつかの一般的な誤解について説明します。

### 2つの可能な結果

9.2節で述べたように、あらかじめ指定された有意水準$\alpha$が与えられた場合、仮説検定には2つの可能な結果があります：

* $p$値が$\alpha$より小さい場合、私たちは帰無仮説$H_0$を対立仮説$H_A$のために*棄却します*。
* $p$値が$\alpha$以上の場合、帰無仮説$H_0$を*棄却しません*。

残念ながら、後者の結果はしばしば「帰無仮説$H_0$を受け入れる」と誤解されます。一見すると、「$H_0$を棄却しない」という声明と「$H_0$を受け入れる」という声明は同等のように思えるかもしれませんが、実際には微妙な違いがあります。「帰無仮説$H_0$を受け入れる」と言うことは、「帰無仮説$H_0$が真であると考える」と述べることと同等です。しかし、「帰無仮説$H_0$を棄却しない」と言うことは別のことを意味します：「$H_0$がまだ偽である可能性があるが、そう言うのに十分な証拠がない」。つまり、十分な証拠の欠如です。しかし、証拠の欠如は存在しないという証拠ではありません。

この区別をさらに明らかにするために、米国の刑事司法制度を類似のものとして使用しましょう。米国の刑事裁判は、被告人について矛盾する2つの主張の間で選択をしなければならない仮説検定と同様の状況です：

1. 被告人は実際に「無罪」または「有罪」のいずれかです。
2. 被告人は「有罪が証明されるまで無罪」と推定されます。
3. 被告人は、有罪であるという*強力な証拠*がある場合にのみ有罪とされます。「合理的な疑いを超えて」という表現は、被告人を有罪とするのに十分な証拠があるかどうかを判断するためのガイドラインとしてよく使われます。
4. 被告人は最終的な評決で「無罪」または「有罪」のいずれかとされます。

言い換えれば、_無罪_の評決は被告人が_無実_であることを示唆しているのではなく、「被告人が実際にはまだ有罪である可能性があるが、この事実を証明するのに十分な証拠がなかった」ということです。では、仮説検定との関連性を見てみましょう：

1. 帰無仮説$H_0$または対立仮説$H_A$のいずれかが真です。
2. 仮説検定は帰無仮説$H_0$が真であると仮定して実施されます。
3. サンプルで見つかった証拠が$H_A$が真であることを示唆する場合にのみ、帰無仮説$H_0$を$H_A$のために棄却します。有意水準$\alpha$は、必要とする証拠の強さの閾値を設定するためのガイドラインとして使用されます。
4. 最終的に「$H_0$を棄却しない」または「$H_0$を棄却する」と決定します。

したがって、本能的には「$H_0$を棄却しない」と「$H_0$を受け入れる」が同等の声明であると思われるかもしれませんが、そうではありません。「$H_0$を受け入れる」ことは被告人を無実とすることと同等です。しかし、裁判所は被告人を「無実」とは認定せず、むしろ「無罪」と認定します。違った言い方をすれば、弁護士は依頼人が無実であることを証明する必要はなく、依頼人が「合理的な疑いを超えて有罪」ではないことを証明するだけで十分です。

9.3節の昇進活動に戻ると、私たちの仮説検定は$H_0: p_{m} - p_{f} = 0$対$H_A: p_{m} - p_{f} > 0$であり、あらかじめ指定した有意水準$\alpha$ = 0.05を使用したことを思い出してください。$p$値は0.001以下でした。$p$値が$\alpha$ = 0.05よりも小さかったため、$H_0$を棄却しました。言い換えれば、$\alpha$ = 0.05の有意水準で$H_0$が偽であるために、この特定のサンプルで必要なレベルの証拠を見つけました。また、非統計的な言語でこの結論を述べると次のようになります：このデータには、性別差別が存在したことを示唆するのに十分な証拠がありました。

### 誤りの種類

残念ながら、陪審員や裁判官が刑事裁判で間違った評決に達することで、不正確な決定をする可能性があります。例えば、本当に無実の被告人を「有罪」とすること。あるいは反対に、本当に有罪の被告人を「無罪」とすること。これはしばしば、検察官がすべての関連証拠にアクセスできず、警察が見つけることができる証拠だけに限られているという事実に起因します。

同じことが仮説検定にも当てはまります。母集団パラメータについて間違った決定を下す可能性があります。なぜなら、母集団からのサンプルデータしか持っておらず、サンプリング変動が間違った結論につながる可能性があるからです。

刑事裁判には2つの可能な誤った結論があります：(1)本当に無実の人が有罪とされるか、(2)本当に有罪の人が無罪とされるか。同様に、仮説検定には2つの可能な誤りがあります：(1)実際は$H_0$が真である場合に$H_0$を棄却すること（**第一種の誤り**）、または(2)実際は$H_0$が偽である場合に$H_0$を棄却しないこと（**第二種の誤り**）。「第一種の誤り」の別の用語は「偽陽性」であり、「第二種の誤り」の別の用語は「偽陰性」です。

このエラーのリスクは、母集団全体の全数調査を実施する代わりにサンプルに基づいて推論するために研究者が支払う代償です。しかし、これまでの多くの例や活動で見てきたように、全数調査はしばしば非常に高価であり、他の場合には不可能であるため、研究者にはサンプルを使用する以外の選択肢がありません。したがって、サンプルに基づく仮説検定では、第一種の誤りが発生する可能性と第二種の誤りが発生する可能性の両方を容認する必要があります。

第一種の誤りと第二種の誤りの概念を理解するために、これらの用語を刑事司法のアナロジーに適用してみましょう：

| | 実際に無罪 | 実際に有罪 |
|--------|----------|----------|
| **無罪判決** | 正しい | 第二種の誤り |
| **有罪判決** | 第一種の誤り | 正しい |

したがって、第一種の誤りは本当に無実の人を誤って刑務所に入れることに対応し、第二種の誤りは本当に有罪の人を自由にすることに対応します。同様に、仮説検定の対応する表は次のようになります：

| | $H_0$が真 | $H_A$が真 |
|--------|----------|----------|
| **$H_0$を棄却しない** | 正しい | 第二種の誤り |
| **$H_0$を棄却する** | 第一種の誤り | 正しい |

### αをどのように選ぶか？

サンプルを使用して母集団について推論する場合、誤りのリスクがあります。信頼区間の場合、対応する「誤り」は、母集団パラメータの真の値を含まない信頼区間を構築することです。仮説検定の場合、これは第一種または第二種の誤りを犯すことになります。明らかに、どちらのエラーの確率も最小限に抑えたいと考えています。間違った結論を出す確率を小さくしたいです：

- 第一種の誤りが発生する確率は$\alpha$で表されます。$\alpha$の値は仮説検定の*有意水準*と呼ばれ、9.2節で定義しました。
- 第二種の誤りの確率は$\beta$で表されます。$1-\beta$の値は仮説検定の*検出力*として知られています。

言い換えれば、$\alpha$は実際に$H_0$が真である場合に$H_0$を誤って棄却する確率に対応します。一方、$\beta$は実際に$H_0$が偽である場合に$H_0$を誤って棄却しない確率に対応します。

理想的には、$\alpha = 0$かつ$\beta = 0$を望みます。つまり、どちらのエラーを犯す可能性も0です。しかし、これは推論のためにサンプリングを行う状況では決して実現できません。サンプルデータを使用する場合、常にどちらかのエラーを犯す可能性があります。さらに、これら2つのエラー確率は逆の関係にあります。第一種の誤りの確率が下がると、第二種の誤りの確率が上がります。

実際に行われるのは通常、あらかじめ有意水準$\alpha$を指定することによって第一種の誤りの確率を固定し、$\beta$を最小化しようとすることです。言い換えれば、帰無仮説$H_0$の不正確な棄却の一定の割合を許容し、$H_0$の不正確な非棄却の割合を最小化しようとします。

たとえば$\alpha$ = 0.01を使用した場合、長期的には帰無仮説$H_0$を1%の時間で誤って棄却する仮説検定手順を使用することになります。これは信頼区間の信頼水準を設定するのと類似しています。

では、$\alpha$にはどのような値を使用すべきでしょうか？異なる分野には異なる慣例がありますが、0.10、0.05、0.01、0.001などのよく使用される値があります。ただし、比較的小さな$\alpha$の値を使用する場合、他の条件が同じであれば、$p$値が$\alpha$より小さくなるのは難しくなることに留意することが重要です。したがって、帰無仮説をあまり棄却しなくなります。言い換えれば、*非常に強い*証拠がある場合にのみ帰無仮説$H_0$を棄却することになります。これは「保守的な」検定として知られています。

一方、比較的大きな$\alpha$の値を使用した場合、他の条件が同じであれば、$p$値が$\alpha$より小さくなるのは容易になります。したがって、帰無仮説をより頻繁に棄却することになります。言い換えれば、*軽度の*証拠しかなくても帰無仮説$H_0$を棄却することになります。これは「リベラルな」検定として知られています。

## 9.5 ケーススタディ：アクション映画とロマンス映画はどちらの評価が高いか？

仮説検定に関する知識を適用して、「IMDbではアクション映画とロマンス映画のどちらの評価が高いか？」という質問に答えてみましょう。IMDbは、映画やテレビ番組のキャスト、あらすじ、トリビア、評価に関する情報を提供するインターネット上のデータベースです。アクション映画とロマンス映画がIMDbで平均的にどちらが高い評価を得ているかを調査します。

### IMDb評価データ

まず、GitHubからデータをダウンロードしましょう：

```python
# GitHubからサンプルデータをダウンロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/movies_sample.csv"
movies_sample = pd.read_csv(url)

# データを確認
movies_sample.head()
```

変数には映画の`title`と撮影された`year`が含まれています。さらに、10星満点のIMDb評価である数値変数`rating`と、映画が`Action`か`Romance`かを示す二項カテゴリ変数`genre`があります。私たちは、`Action`と`Romance`のどちらの映画が平均して高い`rating`を得たかに興味があります。

このデータの探索的データ分析を行いましょう。数値変数とカテゴリ変数の関係を示すために箱ひげ図を使用できます。また、分割したヒストグラムを使用することもできますが、簡潔にするために箱ひげ図のみを表示します：

```python
# ジャンルと評価の関係を箱ひげ図で可視化
plt.figure(figsize=(10, 6))
sns.boxplot(x='genre', y='rating', data=movies_sample)
plt.xlabel('ジャンル')
plt.ylabel('IMDb評価')
plt.title('ジャンル別のIMDb評価')
plt.tight_layout()
plt.show()
```

図を見ると、ロマンス映画の方が中央値の評価が高いことがわかります。しかし、アクション映画とロマンス映画の間で平均`rating`に*有意な*差があると信じる理由はあるでしょうか？このプロットだけでは判断するのは難しいです。箱ひげ図はロマンス映画の方がサンプルの中央値評価が高いことを示しています。

しかし、箱の間には大きな重なりがあります。また、分布が歪んでいるかどうかによって、中央値は必ずしも平均値と同じではないことも思い出してください。

二項カテゴリ変数`genre`によって分割した要約統計量を計算しましょう：映画の数、平均評価、標準偏差です。

```python
# ジャンル別の要約統計量
summary_stats = movies_sample.groupby('genre').agg({
    'title': 'count',  # 映画の数
    'rating': ['mean', 'std']  # 平均と標準偏差
}).reset_index()

# カラム名を整形
summary_stats.columns = ['genre', 'n', 'mean_rating', 'std_dev']
print(summary_stats)
```

Romance映画は36本あり、平均評価は6.32星です。Action映画は32本あり、平均評価は5.28星です。これらの平均評価の差は6.32 - 5.28 = 1.04星でロマンス映画に有利です。しかし、この1.04の差は*すべての*ロマンス映画とアクション映画の真の差を示しているのでしょうか？それともこの差を偶然と*サンプリング変動*のせいにできるでしょうか？この質問に答えるために、仮説検定を使用します。

### サンプリングシナリオ

この研究を第7章で学んだサンプリングに関する用語と表記法の観点から再考してみましょう。*研究対象の母集団*はIMDbデータベース内のアクションまたはロマンス（両方ではない）映画のすべてです。この母集団から抽出された*サンプル*は、`movies_sample`データセットに含まれる68本の映画です。

このサンプルは母集団`movies`からランダムに抽出されたものなので、IMDb上のすべてのロマンス映画とアクション映画を代表しています。したがって、`movies_sample`に基づく分析と結果は母集団全体に一般化できます。関連する*母集団パラメータ*と*点推定値*は何でしょうか？

私たちは以下の仮説検定を行います：

$$
\begin{aligned}
H_0 &: \mu_a - \mu_r = 0\\
\text{vs } H_A&: \mu_a - \mu_r \neq 0
\end{aligned}
$$

つまり、帰無仮説$H_0$は、ロマンス映画とアクション映画は同じ平均評価を持つことを示唆しています。これは私たちが*仮定*する「仮説上の宇宙」です。一方、対立仮説$H_A$は違いがあることを示唆しています。昇進の演習で使用した片側対立仮説$H_A: p_m - p_f > 0$とは異なり、今回は両側対立仮説$H_A: \mu_a - \mu_r \neq 0$を検討しています。

さらに、$\alpha$ = 0.001という低い有意水準をあらかじめ指定します。この値を低く設定することで、他の条件が同じであれば、$p$値が$\alpha$より小さくなる可能性が低くなります。したがって、帰無仮説$H_0$を対立仮説$H_A$のために棄却する可能性が低くなります。言い換えれば、すべてのアクション映画とロマンス映画の平均評価に差がないという仮説は、かなり強力な証拠がある場合にのみ棄却します。これは「保守的な」仮説検定手順として知られています。

### 仮説検定の実施

前に定義した関数を使用して、仮説検定を実施しましょう：

```python
# 変数を指定
movies_specified = specify(movies_sample, formula='rating ~ genre')

# 帰無仮説を仮定
movies_hypothesized = hypothesize(movies_specified, null='independence')

# 帰無仮説の下で1000回のシミュレーションを生成
np.random.seed(42)
movies_generated = generate(movies_hypothesized, reps=1000, type='permute')

# 帰無分布を計算
null_distribution_movies = calculate(
    movies_generated, 
    stat='diff in means',  # 今回は平均の差を使用
    order=['Action', 'Romance']
)

# 観測された統計量を計算
obs_diff_means = calculate(
    movies_specified, 
    stat='diff in means', 
    order=['Action', 'Romance']
)

print(f"観測された評価の差（アクション - ロマンス）: {obs_diff_means['stat'].iloc[0]:.3f}")

# 帰無分布をプロット
visualize(null_distribution_movies, bins=10, obs_stat=obs_diff_means, direction='two-sided')
plt.show()

# p値を計算
p_value_movies = get_p_value(null_distribution_movies, obs_stat=obs_diff_means, direction='two-sided')
print(f"p値: {p_value_movies['p_value'].iloc[0]:.5f}")
```

観測された評価の差は-1.04（アクション - ロマンス）で、これはロマンス映画の方が平均的に1.04星高い評価を受けていることを示しています。p値は非常に小さく、帰無仮説の下でこのような極端な差が偶然によって発生する確率は非常に低いことを示しています。

しかし、このp値はさらに小さいあらかじめ指定された$\alpha$有意水準0.001よりも大きいです。したがって、帰無仮説$H_0: \mu_a - \mu_r = 0$を棄却しない傾向があります。非統計的な言語では、結論は次のようになります：このサンプルデータでは、ロマンス映画とアクション映画の間のIMDb評価に差がないという仮説を棄却するのに必要な証拠がありません。したがって、すべてのIMDb映画について、平均してロマンス映画とアクション映画の評価に差があるとは言えません。

## 9.6 結論

### 理論ベースの仮説検定

第7章と第8章で数学的公式を使用して標準誤差を計算し、信頼区間を構築する理論ベースの方法を示したように、今度は仮説検定を実施するための伝統的な理論ベースの方法の例を紹介します。この方法は、帰無分布を構成するために確率モデル、確率分布、いくつかの仮定に依存しています。これは、コンピュータシミュレーションを使用して帰無分布を構築するというこの本全体で採用してきたアプローチとは対照的です。

これらの伝統的な理論ベースの方法は、研究者が数千の計算を迅速かつ効率的に実行できるコンピュータにアクセスできなかったため、何十年も使用されてきました。現在、計算能力ははるかに安価でアクセスしやすくなっているため、シミュレーションベースの方法がはるかに実行可能になっています。しかし、多くの分野の研究者は引き続き理論ベースの方法を使用しています。そのため、ここで例を含めることが重要だと考えています。

このセクションで示すように、理論ベースの方法は最終的にシミュレーションベースの方法の近似です。ここで焦点を当てる理論ベースの方法は、平均の差をテストするための*二標本t検定*として知られています。ただし、使用する検定統計量は平均の差$\bar{x}_1 - \bar{x}_2$ではなく、関連する*二標本t統計量*になります。使用するデータは、9.5節のアクション映画とロマンス映画の`movies_sample`データです。

#### 二標本t統計量

統計学における一般的なタスクは、「変数の標準化」のプロセスです。異なる変数を標準化することで、それらをより比較しやすくします。例えば、アメリカのオレゴン州ポートランドからの気温記録とカナダのケベック州モントリオールからの気温記録の分布を比較したいとします。アメリカの気温は一般に華氏で記録され、カナダの気温は一般に摂氏で記録されているため、どのように比較できるでしょうか？一つのアプローチは、華氏を摂氏に変換する、またはその逆を行うことです。もう一つのアプローチは、ケルビン単位の温度のような共通の「標準化された」スケールに両方を変換することです。

確率と統計理論から変数を標準化する一般的な方法は、*z*スコアを計算することです：

$$z = \frac{x - \mu}{\sigma}$$

ここで、$x$は変数の1つの値、$\mu$はその変数の平均、$\sigma$はその変数の標準偏差を表します。まず各$x$の値から平均$\mu$を引き、次に$x - \mu$を標準偏差$\sigma$で割ります。これらの操作は、変数を0を中心に*再中心化*し、変数$x$を「標準単位」と呼ばれるものに*再スケール*する効果があります。したがって、変数が取りうるすべての値について、その値が平均$\mu$からどれだけの標準単位離れているかを示す対応する$z$スコアがあります。$z$スコアは平均0、標準偏差1の正規分布に従います。この曲線は「$z$分布」または「標準正規」曲線と呼ばれ、付録で議論されている一般的なベル型のパターンを持っています。

アクション映画とロマンス映画の評価の平均差$\bar{x}_a - \bar{x}_r$に戻って、この変数をどのように標準化すればよいでしょうか？再び、その平均を引き、標準偏差で割ることによってです。第7章から二つの事実を思い出してください。まず、サンプリングが代表的な方法で行われた場合、$\bar{x}_a - \bar{x}_r$のサンプリング分布は真の母集団パラメータ$\mu_a - \mu_r$を中心にします。第二に、$\bar{x}_a - \bar{x}_r$のような点推定値の標準偏差には特別な名前があります：標準誤差です。

これらの考え方を適用して、*二標本t統計量*を紹介します：

$$t = \dfrac{ (\bar{x}_a - \bar{x}_r) - (\mu_a - \mu_r)}{ \text{SE}_{\bar{x}_a - \bar{x}_r} } = \dfrac{ (\bar{x}_a - \bar{x}_r) - (\mu_a - \mu_r)}{ \sqrt{\dfrac{{s_a}^2}{n_a} + \dfrac{{s_r}^2}{n_r}}  }$$

ここで解きほぐしましょう。分子では、$\bar{x}_a-\bar{x}_r$はサンプル平均の差、$\mu_a - \mu_r$は母集団平均の差です。分母では、$s_a$と$s_r$はサンプル`movies_sample`のアクション映画とロマンス映画の*サンプル標準偏差*です。最後に、$n_a$と$n_r$はアクション映画とロマンス映画のサンプルサイズです。これらを平方根の下にまとめると、標準誤差$\text{SE}_{\bar{x}_a - \bar{x}_r}$が得られます。

Pythonで二標本t検定を実装しましょう：

```python
from scipy import stats

# アクション映画とロマンス映画のデータを分離
action_ratings = movies_sample[movies_sample['genre'] == 'Action']['rating']
romance_ratings = movies_sample[movies_sample['genre'] == 'Romance']['rating']

# t検定を実行
t_stat, p_value = stats.ttest_ind(action_ratings, romance_ratings, equal_var=False)

print(f"t統計量: {t_stat:.3f}")
print(f"p値: {p_value:.5f}")

# 自由度を計算
n_a = len(action_ratings)
n_r = len(romance_ratings)
s_a = action_ratings.std()
s_r = romance_ratings.std()

# Welchのt検定の自由度（近似）
df = ((s_a**2/n_a + s_r**2/n_r)**2) / ((s_a**2/n_a)**2/(n_a-1) + (s_r**2/n_r)**2/(n_r-1))
print(f"自由度: {df:.1f}")
```

理論ベースのt検定の結果は、私たちのシミュレーションベースの方法と同様に、ロマンス映画とアクション映画の平均評価には統計的に有意な差があることを示しています。

### 推論が必要でない場合

これまで、`infer`パッケージのワークフローを複数の例で示してきました：信頼区間の構築と仮説検定の実施です。各例で、最初に探索的データ分析（EDA）を行うことを重視しました。具体的には、生のデータ値を見て、`matplotlib`と`seaborn`によるデータ可視化、そして`pandas`によるデータ操作です。常にこれを行うことを*強く*お勧めします。統計の初心者として、EDAは信頼区間や仮説検定などの統計的手法が教えてくれることについての直感を養うのに役立ちます。統計のベテラン実践者としても、EDAは統計的調査の指針となります。特に、統計的推論が本当に必要なのかどうかを考えさせてくれます。

例を考えてみましょう。ニューヨーク市の空港から出発するすべてのフライトのうち、ハワイアン航空のフライトはアラスカ航空のフライトよりも長い時間空中にいるかどうかについて興味があるとします。さらに、2013年のフライトがすべてのフライトの代表的なサンプルであると仮定します。

このような質問に答えるために使用できる2つの可能な統計的推論方法があります。まず、母集団平均の差$\mu_{HA} - \mu_{AS}$の95%信頼区間を構築することができます。ここで、$\mu_{HA}$はすべてのハワイアン航空フライトの平均空中時間、$\mu_{AS}$はすべてのアラスカ航空フライトの平均空中時間です。次に、区間全体が0より大きいかどうかを確認し、$\mu_{HA} - \mu_{AS} > 0$、つまり$\mu_{HA} > \mu_{AS}$であることを示唆するかどうかを確認できます。もう一つの方法は、帰無仮説$H_0: \mu_{HA} - \mu_{AS} = 0$対対立仮説$H_A: \mu_{HA} - \mu_{AS} > 0$の仮説検定を実施することです。

しかし、まず先に述べたように探索的な可視化を構築しましょう。`air_time`は数値で`carrier`はカテゴリなので、箱ひげ図はこれら2つの変数の関係を表示できます：

```python
# GitHubからフライトデータをロード
url = "https://raw.githubusercontent.com/moderndive/moderndive_book/master/data/flights_sample.csv"
flights_sample = pd.read_csv(url)

# ハワイアン航空(HA)とアラスカ航空(AS)のフライトを選択
ha_as_flights = flights_sample[flights_sample['carrier'].isin(['HA', 'AS'])]

# 箱ひげ図を作成
plt.figure(figsize=(10, 6))
sns.boxplot(x='carrier', y='air_time', data=ha_as_flights)
plt.xlabel('航空会社')
plt.ylabel('飛行時間')
plt.title('2013年ニューヨーク発のハワイアン航空とアラスカ航空の飛行時間')
plt.tight_layout()
plt.show()
```

これは「統計学の博士号は必要ない」瞬間と呼ぶものです。アラスカ航空とハワイアン航空の飛行時間に*有意な*違いがあることを知るために専門家である必要はありません。2つの箱ひげ図はまったく重なっていません！信頼区間を構築したり仮説検定を実施したりすることは、率直に言って箱ひげ図以上の洞察をほとんど提供しないでしょう。

なぜこのような明確な違いがこれら2つの航空会社の間で観察されるのか調査しましょう。データを整理して、`flights_sample`の行を`carrier`だけでなく目的地`dest`でもグループ化しましょう。その後、観測数と平均飛行時間の2つの要約統計量を計算します：

```python
# キャリアと目的地でグループ化して要約統計量を計算
summary = ha_as_flights.groupby(['carrier', 'dest']).agg({
    'air_time': ['count', 'mean']
}).reset_index()

# カラム名を整形
summary.columns = ['carrier', 'dest', 'n', 'mean_time']
print(summary)
```

2013年のニューヨークからの飛行機では、アラスカ航空は`SEA`（シアトル）にのみ飛んでおり、ハワイアン航空は`HNL`（ホノルル）にのみ飛んでいることがわかります。ニューヨークからシアトルまでとニューヨークからホノルルまでの距離の明らかな違いを考えると、飛行時間に非常に異なる（_統計的に有意に異なる_、実際には）観察されることは驚くべきことではありません。

これは、データ可視化と記述統計を使用した単純な探索的データ分析以上のことを行わなくても、適切な結論を得ることができる明確な例です。そのため、信頼区間や仮説検定などの統計的推論方法を実行する前に、サンプルデータの探索的データ分析を実施することを強くお勧めします。

### p値の問題点

仮説検定とp値についての多くの一般的な誤解に加えて、p値と仮説検定の使用拡大のもう一つの不幸な結果は、「p-hacking」と呼ばれる現象です。p-hackingは、科学的なアイデアを犠牲にしてでも、「統計的に有意」な結果だけを「選び出す」行為です。p値についての誤解と問題点については、最近多くの記事が書かれています。いくつかをチェックすることをお勧めします：

1. [p値の誤解](https://en.wikipedia.org/wiki/Misunderstandings_of_p-values)
2. [p値についてのオタク的な議論が科学について示すこと - そしてそれを修正する方法](https://www.vox.com/science-and-health/2017/7/31/16021654/p-values-statistical-significance-redefine-0005)
3. [統計学者がp値の誤用に警告を発する](https://www.nature.com/articles/nature.2016.19503)
4. [栄養学について読むことは信頼できない](https://fivethirtyeight.com/features/you-cant-trust-what-you-read-about-nutrition/)
5. [p値の問題の連祷](http://www.fharrell.com/post/pval-litany/)

このような問題は非常に深刻になってきたため、アメリカ統計学会（ASA）は2016年に「統計的有意性とP値に関するASA声明」というタイトルの声明を出しました。この声明では、p値の適切な使用と解釈の基礎となる6つの原則が示されています。ASAはこのp値に関するガイダンスを、定量的科学の実施と解釈を改善し、科学研究の再現性の重要性が高まっていることを知らせるために発表しました。

著者として、私たちは統計的推論には信頼区間の使用を好みます。なぜなら、信頼区間は大きな誤解を受けにくいと考えているからです。しかし、多くの分野ではまだ統計的推論にp値を排他的に使用しており、これがこのテキストにp値を含めた理由の一つです。「p-hacking」とそれが科学に与える影響についてもっと学ぶことをお勧めします。

## まとめ

仮説検定フレームワークを図で示しました。この章では、仮説検定の基本的な考え方と実施方法について学びました。

第8章の信頼区間と今章の仮説検定の理解を身につけたところで、次の第10章では回帰分析のための推論について学びます。

第5章の基本的な回帰分析と第6章の重回帰分析で学んだ回帰モデルを再検討します。例えば、教員の教育スコアをその「美しさ」スコアの関数としての回帰モデルを思い出してください。

次の章では、`std_error`（標準誤差）、`statistic`（観測された*標準化*検定統計量で、`p_value`を計算するために使用される）、そして`lower_ci`と`upper_ci`で与えられる95%信頼区間など、残りの列を解説します。