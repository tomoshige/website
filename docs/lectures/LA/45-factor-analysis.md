# 線形代数学 I / 基礎 / II
## 第45回講義：因子分析の推定法

### 1. 講義情報と予習ガイド

**講義回**: 第45回  
**関連項目**: 因子分析、最尤推定、因子回転  
**予習すべき内容**: 第44回「因子分析の基礎」、共分散行列の性質、固有値分解

本講義を十分に理解するためには、前回学習した因子分析の基本モデル、潜在変数と観測変数の関係、および共分散行列の性質について復習しておくことが重要です。また、行列の固有値分解と対角化についての知識も必要となります。

### 2. 学習目標

本日の講義を通じて、以下のことが理解できるようになります：

1. 因子分析における因子の推定方法を理解し、数学的背景を説明できるようになる
2. 最尤推定法の原理と因子分析への適用方法を習得し、その数式表現を理解する
3. 因子回転の目的と主要な回転法（バリマックス法、プロマックス法）の数学的原理と解釈方法を習得する
4. 因子構造の解釈と評価のための定量的・定性的方法を理解する
5. Pythonによる因子分析の実装方法を習得し、実データへの応用ができるようになる

### 3. 基本概念

#### 3.1 因子分析モデルの復習

因子分析は、観測された変数（観測変数）の背後にある潜在的な要因（潜在変数または因子）を特定するための統計的手法です。第44回の講義で学んだ内容を振り返りながら、より詳細に理解を深めていきましょう。

> **因子分析モデル**：  
> $\mathbf{X} = \mathbf{\Lambda F} + \mathbf{\varepsilon}$
> 
> ここで：
> - $\mathbf{X}$ は $p$ 次元の観測変数ベクトル
> - $\mathbf{\Lambda}$ は $p \times m$ の因子負荷量行列
> - $\mathbf{F}$ は $m$ 次元の共通因子ベクトル
> - $\mathbf{\varepsilon}$ は $p$ 次元の特殊因子（独自因子）ベクトル

このモデルにおいて、共通因子 $\mathbf{F}$ と特殊因子 $\mathbf{\varepsilon}$ に関して、以下の重要な仮定を置いています：

1. $E[\mathbf{F}] = \mathbf{0}$ （共通因子の期待値はゼロベクトル）
2. $\mathrm{Cov}[\mathbf{F}] = \mathbf{I}$ （共通因子間の相関はゼロ：直交因子の場合）
3. $E[\mathbf{\varepsilon}] = \mathbf{0}$ （特殊因子の期待値はゼロベクトル）
4. $\mathrm{Cov}[\mathbf{\varepsilon}] = \mathbf{\Psi}$ （特殊因子の共分散行列は対角行列）
5. $\mathrm{Cov}[\mathbf{F}, \mathbf{\varepsilon}] = \mathbf{0}$ （共通因子と特殊因子は無相関）

この仮定の下で、観測変数 $\mathbf{X}$ の共分散行列 $\mathbf{\Sigma}$ は以下のように表されます：

> **共分散構造**：  
> $\mathbf{\Sigma} = \mathbf{\Lambda \Lambda^T} + \mathbf{\Psi}$
>
> ここで：
> - $\mathbf{\Sigma}$ は観測変数の共分散行列
> - $\mathbf{\Lambda \Lambda^T}$ は共通因子による共分散
> - $\mathbf{\Psi}$ は特殊因子による共分散（対角行列）

この分解は、観測変数の共分散が共通因子による部分と特殊因子による部分に分けられることを示しています。因子分析の目的は、観測データの共分散行列 $\mathbf{\Sigma}$ に最もフィットする因子負荷量行列 $\mathbf{\Lambda}$ と特殊因子の共分散行列 $\mathbf{\Psi}$ を推定することです。

各観測変数 $X_i$ の分散は以下のように分解されます：

> **分散の分解**：  
> $\mathrm{Var}(X_i) = \sum_{j=1}^{m} \lambda_{ij}^2 + \psi_i$
>
> ここで：
> - $\sum_{j=1}^{m} \lambda_{ij}^2$ は変数 $X_i$ の共通性（communality）
> - $\psi_i$ は変数 $X_i$ の独自性（uniqueness）

共通性は共通因子によって説明される分散の割合を表し、独自性は特殊因子に帰属する分散の割合を表します。

#### 3.2 因子分析における不定性

因子分析モデルには**回転の不定性**（rotational indeterminacy）が存在します。これは、因子負荷量行列 $\mathbf{\Lambda}$ に直交行列 $\mathbf{T}$ を掛けた $\mathbf{\Lambda T}$ も同様に妥当な解となる性質を指します。

> **回転の不定性の数学的表現**：  
> $\mathbf{\Lambda \Lambda^T} = (\mathbf{\Lambda T})(\mathbf{\Lambda T})^T = \mathbf{\Lambda T T^T \Lambda^T} = \mathbf{\Lambda \Lambda^T}$
>
> ここで $\mathbf{T}$ は直交行列（$\mathbf{T T^T} = \mathbf{I}$）です。

この性質は、同じデータに対して異なる因子負荷量パターンが得られる可能性を意味します。つまり、数学的には等価でも、解釈が異なる複数の解が存在します。

この不定性があるため、因子分析の結果を解釈する際には注意が必要です。しかし、この性質を利用して、より解釈しやすい因子構造を求めるための**因子回転**という手法が開発されました。回転によって共通因子空間における座標軸を変更することで、より明確な構造を得ることが可能になります。

### 4. 理論と手法

#### 4.1 最尤推定による因子の推定

因子分析モデルのパラメータ（因子負荷量行列 $\mathbf{\Lambda}$ と特殊因子の共分散行列 $\mathbf{\Psi}$）を推定する代表的な方法として、最尤推定法（Maximum Likelihood Estimation, MLE）があります。

> **最尤推定法の基本原理**：  
> 観測データが得られる確率（尤度）を最大にするようなパラメータの値を求める

因子分析モデルでは、観測変数 $\mathbf{X}$ が多変量正規分布に従うと仮定します：

$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$

ここで、平均ベクトル $\boldsymbol{\mu}$ と共分散行列 $\mathbf{\Sigma} = \mathbf{\Lambda \Lambda^T} + \mathbf{\Psi}$ です。多くの場合、データはあらかじめ標準化されるため、$\boldsymbol{\mu} = \mathbf{0}$ と仮定できます。

多変量正規分布の確率密度関数は以下のように表されます：

> **多変量正規分布の確率密度関数**：  
> $f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$

$n$ 個の独立な観測値 $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$ に対する対数尤度関数は次のようになります：

> **対数尤度関数**：  
> $\ln L = -\frac{n}{2} \ln(2\pi) - \frac{n}{2} \ln |\mathbf{\Sigma}| - \frac{1}{2} \sum_{i=1}^{n} (\mathbf{x}_i - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})$

パラメータ $\mathbf{\Lambda}$ と $\mathbf{\Psi}$ は、この対数尤度関数を最大化するように推定されます。実際の計算では、観測データの標本共分散行列 $\mathbf{S}$ を用いて、以下の関数を最小化します：

> **最小化する関数（Fitting Function）**：  
> $F_{ML} = \ln |\mathbf{\Sigma}| + \mathrm{tr}(\mathbf{S} \mathbf{\Sigma}^{-1}) - \ln |\mathbf{S}| - p$
>
> ここで $p$ は変数の数、$\mathrm{tr}$ はトレース（対角成分の和）を表します。

この最小化問題は解析的に解くことが困難なため、通常は以下のような反復アルゴリズムを用いて数値的に解きます：

1. 初期値の設定：因子負荷量行列 $\mathbf{\Lambda}^{(0)}$ と特殊因子の共分散行列 $\mathbf{\Psi}^{(0)}$ の初期値を設定
2. 反復計算：
   a. 現在の推定値から共分散行列 $\mathbf{\Sigma}^{(t)} = \mathbf{\Lambda}^{(t)} (\mathbf{\Lambda}^{(t)})^T + \mathbf{\Psi}^{(t)}$ を計算
   b. フィッティング関数 $F_{ML}$ の値を計算
   c. フィッティング関数が減少するように $\mathbf{\Lambda}^{(t)}$ と $\mathbf{\Psi}^{(t)}$ を更新
3. 収束判定：パラメータの変化が十分小さくなるか、フィッティング関数の値の変化が閾値以下になるまで反復

この反復過程は、Newton-Raphson法やEM（期待値最大化）アルゴリズムなどの最適化手法を用いて実装されることが一般的です。

最尤推定法の利点は、以下の点にあります：

1. 漸近的に不偏で効率的な推定量が得られる
2. 統計的検定（モデルの適合度や因子数の検定など）が可能
3. 様々なモデル間の比較が可能（例：異なる因子数のモデル比較）

一方、最尤推定法の課題としては、以下の点が挙げられます：

1. 多変量正規性の仮定が必要
2. サンプルサイズが小さい場合、推定が不安定になることがある
3. 反復計算が収束しない場合がある（特にHeywood caseと呼ばれる、特殊因子の分散が負になるなどの問題）

#### 4.2 因子数の決定

適切な因子数を決定することは、因子分析において非常に重要です。因子数が少なすぎると重要な構造を見逃し、多すぎると解釈が困難になります。以下に、因子数を決定するための主な方法を詳しく説明します。

##### 4.2.1 スクリープロット（Scree Plot）

スクリープロットは、固有値を大きい順に並べてプロットしたグラフです。固有値の減少パターンを視覚的に確認し、急激な減少（「肘」または「屈曲点」）が見られるポイントで因子数を決定します。

> **スクリープロットの手順**：
> 1. 相関行列または共分散行列の固有値を計算
> 2. 固有値を大きい順に並べる
> 3. 固有値と因子番号（順序）をプロット
> 4. グラフの「肘」の位置を特定し、その位置までの因子数を採用

この方法の利点は視覚的に理解しやすく、データの構造を直感的に把握できることです。一方、「肘」の位置が明確でない場合もあり、主観的な判断が入る可能性があります。

##### 4.2.2 カイザー基準（Kaiser Criterion）

カイザー基準は、固有値が1.0以上の因子のみを採用する方法です。標準化されたデータ（各変数の分散が1）の場合、固有値が1未満の因子は単一の変数よりも情報量が少ないと解釈されます。

> **カイザー基準の適用方法**：
> 1. 相関行列の固有値を計算
> 2. 固有値が1.0以上の因子のみを採用

この方法は客観的で適用が簡単ですが、場合によっては因子数を過大評価または過小評価する傾向があります。特に変数の数が多い場合や変数間の相関が低い場合、過大評価する傾向があります。

##### 4.2.3 累積寄与率（Cumulative Percentage of Variance）

累積寄与率は、因子が説明する分散の割合の累積値が一定の閾値（通常70%〜80%）を超えるまで因子を追加する方法です。

> **累積寄与率の計算手順**：
> 1. 各固有値 $\lambda_i$ の寄与率を計算：$\lambda_i / \sum_{j=1}^{p} \lambda_j$
> 2. 寄与率を累積していき、閾値（例：80%）を超える最小の因子数を採用

この方法は、データの説明力に直接関連した基準を提供しますが、適切な閾値の選択が分野や研究目的によって異なる場合があります。

##### 4.2.4 平行分析（Parallel Analysis）

平行分析は、実データから得られる固有値と、同じサイズのランダムデータから得られる固有値を比較する方法です。実データの固有値がランダムデータの固有値を上回る場合のみ、その因子を採用します。

> **平行分析の手順**：
> 1. 実データの相関行列または共分散行列の固有値を計算
> 2. 同じサイズ（変数数×サンプル数）のランダムデータを複数回生成し、各ランダムデータの固有値を計算
> 3. ランダムデータから得られた固有値の平均または95パーセンタイルを計算
> 4. 実データの固有値がランダムデータの固有値（平均または95パーセンタイル）を上回る因子のみを採用

平行分析は、偶然の構造と実際の構造を区別する点で優れていますが、計算が複雑で特殊なソフトウェアが必要な場合があります。

##### 4.2.5 尤度比検定（Likelihood Ratio Test）

最尤推定法を用いる場合、異なる因子数のモデル間でカイ二乗検定に基づく尤度比検定を行うことができます。

> **尤度比検定の手順**：
> 1. 因子数 $m$ のモデルと因子数 $m-1$ のモデルをそれぞれ最尤推定法で推定
> 2. 尤度比統計量を計算：$\chi^2 = (n-1) \times (F_{ML}^{(m-1)} - F_{ML}^{(m)})$
> 3. 自由度 $df = p - m + 1 - (p - m)$ でカイ二乗検定を実施
> 4. 有意でない結果が得られるまで因子数を増やし、最初に有意でなくなった因子数を採用

この方法は統計的に厳密ですが、多変量正規性の仮定が必要であり、サンプルサイズの影響を受けやすい（大規模サンプルでは小さな違いでも統計的に有意になる）という欠点があります。

実際の因子分析では、これらの方法を組み合わせて用いることが推奨されます。また、理論的な解釈可能性や先行研究との整合性も考慮して因子数を決定することが重要です。

#### 4.3 因子回転

因子回転は、推定された因子負荷量行列をより解釈しやすい形に変換するための手法です。回転の主な目的は**単純構造**（simple structure）の達成です。

> **単純構造の原則**（Thurstoneによる）：  
> 1. 各変数は少数の因子に高い負荷量を持ち、他の因子に対する負荷量はゼロまたは非常に小さい
> 2. 各因子は変数の一部のみに高い負荷量を持ち、他の変数に対する負荷量は小さい
> 3. 異なる因子は変数の異なる集合に高い負荷量を持つ
> 4. 高い負荷量を持つ変数が多い因子と少ない因子が存在する
> 5. 因子のペアごとに、両方の因子に高い負荷量を持つ変数の数が少ない

単純構造を持つ因子負荷量行列は、各変数がどの因子に関連しているかが明確になり、各因子の解釈が容易になります。

回転法は大きく分けて以下の2種類があります：

1. **直交回転**（Orthogonal Rotation）：因子間の相関がゼロという制約を保持
2. **斜交回転**（Oblique Rotation）：因子間の相関を許容

##### 4.3.1 バリマックス回転（直交回転）

バリマックス（Varimax）回転は最も一般的な直交回転法です。各因子の負荷量の二乗値の分散を最大化することで、各因子に対して一部の変数が高い負荷量を、他の変数が低い負荷量を持つようにします。

> **バリマックスの目的関数**：  
> $V = \sum_{j=1}^{m} \left[ \frac{1}{p} \sum_{i=1}^{p} \left( \frac{\lambda_{ij}^2}{h_i^2} \right)^2 - \left( \frac{1}{p} \sum_{i=1}^{p} \frac{\lambda_{ij}^2}{h_i^2} \right)^2 \right]$
>
> ここで：
> - $\lambda_{ij}$ は変数 $i$ の因子 $j$ に対する回転後の負荷量
> - $h_i^2$ は変数 $i$ の共通性（$h_i^2 = \sum_{j=1}^{m} \lambda_{ij}^2$）
> - $p$ は変数の数
> - $m$ は因子の数

この目的関数は、各因子の負荷量パターンの「単純さ」を測るものと解釈できます。バリマックス回転は、この目的関数を最大化する直交回転行列 $\mathbf{T}$ を求めることで実行されます。

バリマックス回転の具体的なアルゴリズムは以下の通りです：

1. 2つの因子（$j$ と $k$）を選択
2. これらの因子に関して、回転角 $\phi$ を変化させながら目的関数 $V$ を最大化する角度を探索
3. 最適な角度 $\phi$ で回転行列を計算し、因子負荷量行列を更新
4. すべての因子ペアに対してこのプロセスを繰り返す
5. 収束するまで（目的関数の変化が十分小さくなるまで）1-4を反復

バリマックス回転の利点は、各因子が互いに独立（無相関）であるため解釈が簡単なことです。欠点は、実際の現象では因子間に相関がある場合が多く、その構造を正確に反映できない可能性があることです。

##### 4.3.2 プロマックス回転（斜交回転）

プロマックス（Promax）回転は、バリマックス回転の結果をさらに「シャープ化」することで、より単純な構造を得る斜交回転法です。因子間の相関を許容するため、より現実的なモデルが得られることがあります。

プロマックス回転の手順は以下の通りです：

1. バリマックス回転を実行して初期の直交解を得る
2. バリマックス回転後の負荷量をべき乗変換してターゲット行列を作成
3. ターゲット行列に近づくように因子負荷量行列を回転（斜交回転）

> **プロマックスのターゲット行列**：  
> $T_{ij} = \lambda_{ij}^{\mathrm{varimax}} \cdot |\lambda_{ij}^{\mathrm{varimax}}|^{k-1}$
>
> ここで：
> - $\lambda_{ij}^{\mathrm{varimax}}$ はバリマックス回転後の因子負荷量
> - $k$ はべき乗パラメータ（通常 $k = 2,3,4$）
> - $|\lambda_{ij}^{\mathrm{varimax}}|$ は絶対値を表す

このべき乗変換により、大きな負荷量はより大きく、小さな負荷量はより小さくなります。つまり、負荷量のパターンがより「シャープ」になります。

次に、このターゲット行列に近づくような斜交回転を行います。この過程で、因子間の相関が許容されます。最終的に得られる因子構造は、以下の2つの行列で表されます：

1. **パターン行列**（Pattern Matrix）：各変数と各因子間の直接的な関係を表す
2. **構造行列**（Structure Matrix）：因子間の相関を考慮した、各変数と各因子間の相関を表す

> **パターン行列と構造行列の関係**：  
> 構造行列 = パターン行列 × 因子間相関行列

プロマックス回転の利点は、因子間の相関を許容することで、より現実的なモデルが得られる可能性が高いことです。特に心理学、社会科学、健康科学などの分野では、完全に独立した因子よりも相関のある因子を仮定する方が自然な場合が多いです。欠点は、直交回転に比べて解釈がやや複雑になることです。

#### 4.4 回転結果の解釈

因子回転後、以下の点に注目して結果を解釈します：

##### 4.4.1 因子負荷量パターンの解釈

因子負荷量は、各観測変数と各因子の関連の強さを表します。絶対値が大きいほど関連が強いことを示します。

> **因子負荷量の解釈の基準**（一般的な目安）：
> - |負荷量| ≥ 0.7：非常に強い関連
> - 0.5 ≤ |負荷量| < 0.7：強い関連
> - 0.3 ≤ |負荷量| < 0.5：中程度の関連
> - |負荷量| < 0.3：弱い関連（通常は解釈に含めない）

各因子は、高い負荷量を持つ変数群の共通の特性を表すと解釈します。例えば、「不安」「抑うつ」「緊張」といった変数が高い負荷量を持つ因子は、「心理的ストレス」などと解釈できるかもしれません。

##### 4.4.2 因子間相関の解釈

斜交回転（プロマックスなど）の場合、因子間相関行列が得られます。この相関が高い場合（例：|相関| > 0.3）、対応する因子間には関連があると解釈できます。

因子間相関が非常に高い場合（例：|相関| > 0.7）、これらの因子は分離せず、より高次の因子として統合することを検討する必要があるかもしれません。

##### 4.4.3 共通性の解釈

各変数の共通性（communality）は、すべての因子によって説明される分散の割合を表します。

> **共通性の計算**：  
> $h_i^2 = \sum_{j=1}^{m} \lambda_{ij}^2$
>
> ここで $\lambda_{ij}$ は変数 $i$ の因子 $j$ に対する負荷量です。

共通性が低い変数（例：$h_i^2 < 0.3$）は、現在のモデルではうまく説明されていないことを示します。そのような変数は、モデルから除外するか、別の分析を検討する必要があるかもしれません。

##### 4.4.4 因子スコアの解釈

因子スコアは、各観測対象（個人など）の各因子における「位置」を表します。因子スコアの分布や極端な値を持つケースを分析することで、データにおける特徴的なパターンを発見できる場合があります。

因子スコアは標準化されていることが多く、平均0、標準偏差1の分布を持ちます。したがって、因子スコアが±2を超えるケースは、その因子において特に高い（または低い）特性を持つと解釈できます。

### 5. Pythonによる実装と可視化

以下に、Pythonのscikit-learnとfactoranalyzerライブラリを用いた因子分析の詳細な実装例を示します。各ステップについて丁寧に解説します。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import seaborn as sns
from sklearn.datasets import load_iris

# サンプルデータとしてIrisデータセットを使用
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
print("データの最初の5行:")
print(X.head())
print("\nデータの基本統計量:")
print(X.describe())

# データの相関行列を可視化
plt.figure(figsize=(10, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('アヤメデータの相関行列')
plt.tight_layout()
plt.show()

# Bartlettの球面性検定
# この検定は、相関行列が単位行列（すべての変数が無相関）である帰無仮説を検定
chi_square_value, p_value = calculate_bartlett_sphericity(X)
print(f"Bartlettの球面性検定: chi²={chi_square_value:.4f}, p={p_value:.10f}")
print("p値が非常に小さいため、変数間に有意な相関があり、因子分析が適切であることが示されました。")

# KMO（Kaiser-Meyer-Olkin）測度
# サンプリングの妥当性を評価する指標。0.6以上が望ましい
kmo_all, kmo_model = calculate_kmo(X)
print(f"KMO: {kmo_model:.4f}")
if kmo_model > 0.8:
    print("KMO値が0.8以上であり、因子分析に非常に適したデータであることを示しています。")
elif kmo_model > 0.6:
    print("KMO値が0.6以上であり、因子分析に適したデータであることを示しています。")
else:
    print("KMO値が0.6未満であり、因子分析の適用には注意が必要です。")

# スクリープロットを作成して因子数を決定
fa = FactorAnalyzer(rotation=None)
fa.fit(X)

# 固有値
ev, _ = fa.get_eigenvalues()
plt.figure(figsize=(10, 6))
plt.scatter(range(1, X.shape[1] + 1), ev)
plt.plot(range(1, X.shape[1] + 1), ev)
plt.title('スクリープロット')
plt.xlabel('因子数')
plt.ylabel('固有値')
plt.grid(True)
plt.axhline(y=1, linestyle='--', color='red', label='Kaiser基準 (固有値=1)')
plt.legend()
plt.show()

print("\n固有値:")
for i, eigenvalue in enumerate(ev, 1):
    print(f"因子 {i}: {eigenvalue:.4f}")

# 累積寄与率を計算
total_variance = sum(ev)
cumulative_variance_ratio = np.cumsum(ev) / total_variance
print("\n累積寄与率:")
for i, ratio in enumerate(cumulative_variance_ratio, 1):
    print(f"因子 {i}まで: {ratio:.4f} ({ratio*100:.2f}%)")

# スクリープロットと累積寄与率の分析結果から因子数を決定
# 今回は例として2因子を採用する
n_factors = 2
print(f"\nスクリープロット、Kaiser基準、累積寄与率を総合的に判断し、{n_factors}因子を採用します。")

# 因子数を2と仮定して因子分析を実行（回転なし）
fa_no_rotation = FactorAnalyzer(n_factors=n_factors, rotation=None, method='ml')
fa_no_rotation.fit(X)
loadings_no_rotation = fa_no_rotation.loadings_

# バリマックス回転による因子分析
fa_varimax = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='ml')
fa_varimax.fit(X)
loadings_varimax = fa_varimax.loadings_

# プロマックス回転による因子分析
fa_promax = FactorAnalyzer(n_factors=n_factors, rotation='promax', method='ml')
fa_promax.fit(X)
loadings_promax = fa_promax.loadings_
factor_corr = fa_promax.phi_  # 因子間相関

# 結果の表示
loadings_df_no_rotation = pd.DataFrame(loadings_no_rotation, 
                                      index=X.columns, 
                                      columns=[f'因子 {i+1}' for i in range(n_factors)])
loadings_df_varimax = pd.DataFrame(loadings_varimax, 
                                  index=X.columns, 
                                  columns=[f'因子 {i+1}' for i in range(n_factors)])
loadings_df_promax = pd.DataFrame(loadings_promax, 
                                 index=X.columns, 
                                 columns=[f'因子 {i+1}' for i in range(n_factors)])

print("\n回転なしの因子負荷量:")
print(loadings_df_no_rotation)

print("\nバリマックス回転後の因子負荷量:")
print(loadings_df_varimax)

print("\nプロマックス回転後の因子負荷量:")
print(loadings_df_promax)

print("\nプロマックス回転後の因子間相関:")
print(pd.DataFrame(factor_corr, 
                  index=[f'因子 {i+1}' for i in range(n_factors)], 
                  columns=[f'因子 {i+1}' for i in range(n_factors)]))

# 共通性
communalities = fa_varimax.get_communalities()
print("\n共通性:")
print(pd.DataFrame({'共通性': communalities}, index=X.columns))

# 因子負荷量のヒートマップ可視化
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.heatmap(loadings_df_no_rotation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
plt.title('回転なし')

plt.subplot(1, 3, 2)
sns.heatmap(loadings_df_varimax, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
plt.title('バリマックス回転')

plt.subplot(1, 3, 3)
sns.heatmap(loadings_df_promax, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
plt.title('プロマックス回転')

plt.tight_layout()
plt.show()

# 因子スコアの計算
factor_scores = fa_varimax.transform(X)
factor_scores_df = pd.DataFrame(factor_scores, 
                               columns=[f'因子 {i+1}' for i in range(n_factors)])

# 因子スコアの散布図
plt.figure(figsize=(10, 8))
scatter = plt.scatter(factor_scores_df['因子 1'], 
                     factor_scores_df['因子 2'], 
                     c=iris.target, 
                     cmap='viridis', 
                     alpha=0.7)
plt.title('因子スコアの散布図')
plt.xlabel('因子 1')
plt.ylabel('因子 2')
plt.grid(True)
legend = plt.colorbar(scatter)
legend.set_label('アヤメの種類')

# クラスごとの中心を追加
for i, species in enumerate(['setosa', 'versicolor', 'virginica']):
    idx = np.where(iris.target == i)
    centroid_x = np.mean(factor_scores[idx, 0])
    centroid_y = np.mean(factor_scores[idx, 1])
    plt.scatter(centroid_x, centroid_y, marker='X', s=200, 
                edgecolor='black', facecolor='none')
    plt.annotate(species, (centroid_x, centroid_y), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold')

plt.show()

# 各回転方法の解釈
print("\n因子負荷量の解釈（バリマックス回転後）:")
for var_name in X.columns:
    max_loading_idx = np.argmax(np.abs(loadings_df_varimax.loc[var_name].values))
    max_loading = loadings_df_varimax.loc[var_name].values[max_loading_idx]
    factor_name = f"因子 {max_loading_idx+1}"
    loading_strength = ""
    if abs(max_loading) >= 0.7:
        loading_strength = "非常に強い"
    elif abs(max_loading) >= 0.5:
        loading_strength = "強い"
    elif abs(max_loading) >= 0.3:
        loading_strength = "中程度の"
    else:
        loading_strength = "弱い"
    
    print(f"{var_name}は{factor_name}に{loading_strength}負荷（{max_loading:.3f}）を持っています。")

# 因子の命名
print("\n因子の解釈:")
print("因子1: 花と関連する特徴を表す「花の形態因子」")
print("因子2: 葉と関連する特徴を表す「葉の形態因子」")
```

このコードの実行結果を解説します：

1. **データの前処理と適合性の確認**：
   - Bartlettの球面性検定により、変数間に有意な相関があることを確認（因子分析の適用条件）
   - KMO測度により、サンプリングの妥当性を評価（値が0.6以上であれば良好）

2. **因子数の決定**：
   - スクリープロットにより固有値の減少パターンを可視化
   - Kaiser基準（固有値＞1）を適用
   - 累積寄与率を計算（全分散のどれだけの割合を説明できるか）
   - これらの方法を総合的に判断して因子数を決定

3. **様々な回転方法による因子分析**：
   - 回転なし、バリマックス回転（直交）、プロマックス回転（斜交）の3種類の方法で因子分析を実行
   - 各方法の因子負荷量行列を比較

4. **結果の解釈と可視化**：
   - 因子負荷量のヒートマップによる視覚化
   - 共通性の計算と解釈
   - 因子スコアの散布図による視覚化
   - 各変数の因子への負荷量の強さと方向に基づく解釈
   - 因子の命名と全体構造の解釈

#### 5.1 健康データへの応用例

以下は、健康関連データに対する因子分析の応用例です。ここでは人工的に生成した健康データを用いて、因子分析の実際の応用方法を示します。

```python
# 健康関連データの因子分析（例示用の人工データ）
np.random.seed(42)
n_samples = 200

# 2つの潜在因子を作成（身体的健康と精神的健康）
physical_health = np.random.normal(0, 1, n_samples)
mental_health = np.random.normal(0, 1, n_samples)

# 測定変数を生成（ノイズを含む）
weight = 0.8 * physical_health + 0.1 * mental_health + np.random.normal(0, 0.5, n_samples)
bmi = 0.7 * physical_health + 0.0 * mental_health + np.random.normal(0, 0.6, n_samples)
blood_pressure = 0.6 * physical_health + 0.2 * mental_health + np.random.normal(0, 0.7, n_samples)
cholesterol = 0.5 * physical_health + 0.1 * mental_health + np.random.normal(0, 0.7, n_samples)

stress = 0.2 * physical_health + 0.8 * mental_health + np.random.normal(0, 0.5, n_samples)
anxiety = 0.1 * physical_health + 0.7 * mental_health + np.random.normal(0, 0.6, n_samples)
depression = 0.2 * physical_health + 0.6 * mental_health + np.random.normal(0, 0.7, n_samples)
sleep_quality = 0.3 * physical_health + 0.5 * mental_health + np.random.normal(0, 0.7, n_samples)

# データフレーム作成
health_data = pd.DataFrame({
    '体重': weight,
    'BMI': bmi,
    '血圧': blood_pressure,
    'コレステロール': cholesterol,
    'ストレス': stress,
    '不安': anxiety,
    'うつ症状': depression,
    '睡眠の質': sleep_quality
})

# データの相関行列を可視化
plt.figure(figsize=(10, 8))
correlation_matrix = health_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('健康データの相関行列')
plt.tight_layout()
plt.show()

# Bartlettの球面性検定
chi_square_value, p_value = calculate_bartlett_sphericity(health_data)
print(f"Bartlettの球面性検定: chi²={chi_square_value:.4f}, p={p_value:.10f}")

# KMO測度
kmo_all, kmo_model = calculate_kmo(health_data)
print(f"KMO: {kmo_model:.4f}")

# スクリープロットを作成して因子数を決定
fa_health = FactorAnalyzer(rotation=None)
fa_health.fit(health_data)
ev, _ = fa_health.get_eigenvalues()

plt.figure(figsize=(10, 6))
plt.scatter(range(1, health_data.shape[1] + 1), ev)
plt.plot(range(1, health_data.shape[1] + 1), ev)
plt.title('健康データのスクリープロット')
plt.xlabel('因子数')
plt.ylabel('固有値')
plt.grid(True)
plt.axhline(y=1, linestyle='--', color='red', label='Kaiser基準 (固有値=1)')
plt.legend()
plt.show()

# 累積寄与率を計算
total_variance = sum(ev)
cumulative_variance_ratio = np.cumsum(ev) / total_variance
print("\n累積寄与率:")
for i, ratio in enumerate(cumulative_variance_ratio, 1):
    print(f"因子 {i}まで: {ratio:.4f} ({ratio*100:.2f}%)")

# 因子分析（プロマックス回転）
n_factors = 2  # スクリープロットから2因子が適切と判断
fa_health = FactorAnalyzer(n_factors=n_factors, rotation='promax', method='ml')
fa_health.fit(health_data)
loadings_health = fa_health.loadings_
factor_corr_health = fa_health.phi_  # 因子間相関

# 結果表示
loadings_df_health = pd.DataFrame(loadings_health, 
                                 index=health_data.columns, 
                                 columns=['身体的健康因子', '精神的健康因子'])
print("\n健康データの因子負荷量（プロマックス回転後）:")
print(loadings_df_health)

# 共通性
communalities_health = fa_health.get_communalities()
print("\n共通性:")
print(pd.DataFrame({'共通性': communalities_health}, index=health_data.columns))

# 因子間相関
print("\n健康データの因子間相関:")
print(pd.DataFrame(factor_corr_health, 
                  index=['身体的健康因子', '精神的健康因子'], 
                  columns=['身体的健康因子', '精神的健康因子']))

# 因子負荷量のヒートマップ
plt.figure(figsize=(10, 8))
sns.heatmap(loadings_df_health, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
plt.title('健康データの因子構造（プロマックス回転後）')
plt.tight_layout()
plt.show()

# 因子スコアの計算
factor_scores_health = fa_health.transform(health_data)
factor_scores_df_health = pd.DataFrame(factor_scores_health, 
                                     columns=['身体的健康因子', '精神的健康因子'])

# 因子スコアの散布図
plt.figure(figsize=(10, 8))
plt.scatter(factor_scores_df_health['身体的健康因子'], 
           factor_scores_df_health['精神的健康因子'], 
           alpha=0.7)
plt.title('健康データの因子スコア散布図')
plt.xlabel('身体的健康因子')
plt.ylabel('精神的健康因子')
plt.grid(True)
plt.axhline(y=0, color='gray', linestyle='--')
plt.axvline(x=0, color='gray', linestyle='--')

# 象限に名前を付ける
plt.text(2, 2, '身体的・精神的に健康', fontsize=12, ha='right')
plt.text(-2, 2, '精神的に健康\n身体的に不健康', fontsize=12, ha='left')
plt.text(2, -2, '身体的に健康\n精神的に不健康', fontsize=12, ha='right')
plt.text(-2, -2, '身体的・精神的に不健康', fontsize=12, ha='left')

plt.show()

# 変数の解釈
print("\n因子負荷量の解釈:")
for var_name in health_data.columns:
    max_loading_idx = np.argmax(np.abs(loadings_df_health.loc[var_name].values))
    max_loading = loadings_df_health.loc[var_name].values[max_loading_idx]
    factor_name = loadings_df_health.columns[max_loading_idx]
    loading_strength = ""
    if abs(max_loading) >= 0.7:
        loading_strength = "非常に強い"
    elif abs(max_loading) >= 0.5:
        loading_strength = "強い"
    elif abs(max_loading) >= 0.3:
        loading_strength = "中程度の"
    else:
        loading_strength = "弱い"
    
    print(f"{var_name}は{factor_name}に{loading_strength}負荷（{max_loading:.3f}）を持っています。")

# 健康リスク評価（例）
# 因子スコアに基づいてリスクグループを分類
def classify_risk(phys_score, mental_score):
    if phys_score < -1 and mental_score < -1:
        return "高リスク（身体的・精神的要因）"
    elif phys_score < -1:
        return "中リスク（主に身体的要因）"
    elif mental_score < -1:
        return "中リスク（主に精神的要因）"
    else:
        return "低リスク"

# リスク分類を追加
factor_scores_df_health['リスク分類'] = factor_scores_df_health.apply(
    lambda row: classify_risk(row['身体的健康因子'], row['精神的健康因子']), axis=1)

# リスクグループごとの人数をカウント
risk_counts = factor_scores_df_health['リスク分類'].value_counts()
print("\n健康リスク分類:")
print(risk_counts)

# リスクグループの可視化
plt.figure(figsize=(10, 6))
risk_counts.plot(kind='bar', color='skyblue')
plt.title('健康リスク分類の分布')
plt.xlabel('リスク分類')
plt.ylabel('人数')
plt.tight_layout()
plt.show()
```

この健康データの応用例では、以下のことを行っています：

1. **人工的な健康データの作成**：
   - 身体的健康と精神的健康の2つの潜在因子を基に、8つの健康指標（体重、BMI、血圧、コレステロール、ストレス、不安、うつ症状、睡眠の質）を生成

2. **因子分析の実施**：
   - 相関行列の可視化、適合性の確認（Bartlettの検定、KMO測度）
   - スクリープロットと累積寄与率による因子数の決定
   - プロマックス回転による因子分析の実行

3. **結果の解釈と応用**：
   - 因子負荷量のヒートマップ可視化
   - 因子スコアの散布図とその解釈
   - 健康リスク評価への応用（因子スコアに基づくリスク分類）

この例では、因子分析が健康データから潜在的な健康状態（身体的健康と精神的健康）を抽出し、それに基づいて個人の健康リスクを評価する方法を示しています。実際の健康データサイエンスでは、このようなアプローチが健康状態の評価、リスク予測、介入計画の策定などに活用されています。

### 6. 演習問題

#### 6.1 基本問題

1. 以下の相関行列に対して因子分析を適用します。スクリープロットから適切な因子数を判断し、その理由を説明してください。

   ```
   相関行列 R:
   [1.00, 0.70, 0.60, 0.30, 0.25]
   [0.70, 1.00, 0.65, 0.35, 0.20]
   [0.60, 0.65, 1.00, 0.40, 0.30]
   [0.30, 0.35, 0.40, 1.00, 0.75]
   [0.25, 0.20, 0.30, 0.75, 1.00]
   ```

   **解答例**：
   この相関行列から固有値を計算すると、[2.68, 1.47, 0.42, 0.29, 0.14]のような値が得られます。スクリープロットでは、固有値が1.0を超える因子は2つ（第1因子と第2因子）であり、第2因子と第3因子の間で固有値の急激な減少（「肘」）が見られます。また、最初の2因子で全分散の約83%（(2.68+1.47)/5≈0.83）を説明しています。したがって、Kaiser基準、スクリープロットの「肘」、累積寄与率のいずれの観点からも、2因子モデルが適切だと判断できます。

2. 2因子モデルを仮定し、以下の因子負荷量行列が得られたとします。

   ```
   Λ（回転前）：
   [0.80, 0.40]
   [0.75, 0.35]
   [0.70, 0.45]
   [0.50, 0.70]
   [0.45, 0.75]
   ```

   この行列に対してバリマックス回転を適用すると、どのような結果が期待されますか？回転後の因子構造はどのように解釈できるでしょうか？

   **解答例**：
   バリマックス回転を適用すると、各変数が一方の因子に高い負荷量を、もう一方の因子に低い負荷量を持つように回転されます。回転後の因子負荷量行列は以下のようになる可能性があります：

   ```
   Λ（バリマックス回転後）：
   [0.89, 0.10]
   [0.82, 0.07]
   [0.81, 0.18]
   [0.22, 0.83]
   [0.15, 0.87]
   ```

   この回転後の構造では、最初の3つの変数は第1因子に高い負荷量を持ち、残りの2つの変数は第2因子に高い負荷量を持ちます。この結果から、2つの明確な潜在因子が存在することがわかります。例えば、もしこれが心理学的尺度の分析であれば、第1因子は「認知的側面」、第2因子は「情緒的側面」のような解釈が可能かもしれません。または健康データであれば、第1因子は「身体的健康」、第2因子は「精神的健康」のような解釈が考えられます。

3. プロマックス回転とバリマックス回転の主な違いを説明し、どのような状況でプロマックス回転が望ましいかを説明してください。

   **解答例**：
   プロマックス回転とバリマックス回転の主な違いは以下の通りです：

   1. バリマックス回転は直交回転であり、因子間の相関をゼロに保ちます。一方、プロマックス回転は斜交回転であり、因子間の相関を許容します。
   
   2. バリマックス回転は各因子の負荷量の二乗値の分散を最大化することで単純構造を求めます。プロマックス回転はまずバリマックス回転を行い、さらにその結果をべき乗変換（「シャープ化」）することでより単純な構造を求めます。
   
   3. バリマックス回転では、解釈が単純になりますが、潜在因子間の相関が実際に存在する場合、それを無視することになります。プロマックス回転では、より現実的なモデルが得られることがありますが、因子間の相関があるため解釈がやや複雑になります。

   プロマックス回転が望ましい状況：
   
   1. 潜在因子間に相関があると理論的に予測される場合（例：心理学的特性、健康状態の様々な側面など）
   
   2. より単純な因子構造を求める場合（各変数がより明確に一つの因子に関連づけられる）
   
   3. 現実のデータ生成過程をより忠実に反映したモデルを求める場合
   
   特に、心理学、社会科学、健康科学などの多くの分野では、完全に独立した潜在因子を仮定するのは非現実的であることが多く、プロマックス回転などの斜交回転が適切な選択となることが多いです。

4. 共通性が低い変数が因子分析に含まれている場合、どのような問題が生じる可能性がありますか？そのような変数の取り扱いについて説明してください。

   **解答例**：
   共通性が低い変数（例：共通性 < 0.3）が因子分析に含まれている場合、以下のような問題が生じる可能性があります：

   1. その変数は現在の因子構造ではうまく説明されていないことを意味し、因子解釈の信頼性が低下する
   
   2. 低い共通性の変数が多いと、モデル全体の説明力（累積寄与率）が低下する
   
   3. 低い共通性は、その変数が現在の因子とは別の潜在因子に関連している可能性や、測定誤差が大きい可能性を示唆する

   共通性が低い変数の取り扱いについては、以下のような方法があります：

   1. **変数の除外**：共通性が非常に低い変数（例：< 0.2）は、分析から除外することを検討する
   
   2. **因子数の増加**：低い共通性は因子数が不足している可能性を示唆するため、因子数を増やすことを検討する
   
   3. **変数の再検討**：変数の測定方法や定義の見直しを行う
   
   4. **別の分析手法の検討**：その変数が他の変数とは異なる構造を持つ場合、別の分析手法（例：クラスター分析、多次元尺度法など）の併用を検討する

   なお、変数を除外する前に、その変数が理論的に重要であるかどうかを考慮することが重要です。理論的に重要な変数の場合、低い共通性を持つ場合でも、分析に含めることが正当化される場合があります。
