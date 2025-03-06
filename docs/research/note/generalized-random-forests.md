一般化ランダムフォレストについて、包括的なレビュー論文を作成します。
以下の構成で作成しまます。

## 1. ランダムフォレストの概観
### 1.1 ランダムフォレストの背景
### 1.2 ランダムフォレストの拡張と応用
### 1.3 Breiman 2001 から 2012年までのランダムフォレストの理論解析の歩み
[Breiman (2001)](https://link.springer.com/content/pdf/10.1023/a:1010933404324.pdf)年に提案したランダムフォレスト（Random Forest）は、複数の決定木をランダムに生成し、それらの予測を平均する強力なアンサンブル学習法です​。実践的には高次元データへの適応力や汎用性の高さから非常に成功を収めました​。しかし、その卓越した性能にもかかわらず、提案当初は理論的性質の理解が大きく遅れていました。木の構造がデータに強く依存し、さらにランダム化要素も含むため、その数学的解析は困難であり、「広く使われているにもかかわらず理論的性質はほとんど知られていない」と指摘されていました​。このように、2000年代前半までは理論と実践のギャップが大きかった。
この他、[Breiman (2004)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2a42f39add8332a7139d44a6e77496c0571e4f24) でも理論的な部分について解析を試みているが、ランダムフォレストの理論解析には至らず、直感的な理解と$$を結びつけたものにとどまっている。2010年以降の理論解析に大きな貢献を果たしているのは、[Lin and Jean (2006)](https://www.tandfonline.com/doi/abs/10.1198/016214505000001230?casa_token=tUe3DH9TNEEAAAAA:sQZepLDXzrVjTx9kd14uU0BhkQKRTBrQLncg2fs-mm63ya_auCXQvNYscDwWd4Swep1SBDE5tWB0qA)の研究で、この研究ではk-nearest neighborhood estimator と ランダムフォレストによる推定量を結びつけたという意味で非常に大きな貢献をしている。この Lin and Jean (2006) の議論を拡張させたのが、[Biau and Devroye (2010)](https://www.sciencedirect.com/science/article/pii/S0047259X10001387)である。これらの結果は、2014年以降に続くランダムフォレストの理論的な結果の大きな礎となっている。また、[Meinshausen (2006)](https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf) では、ランダムフォレストによる条件付き分位点関数の推定法である quantile regression forest においてもランダムフォレストの一致性に言及されている。

### 2012年以降のランダムフォレストの理論解析の歩み：一致性と、漸近正規性の証明
Biau (2012)​ はBreimanの提案したアルゴリズムに近いランダムフォレストモデル（例えば各木でランダムに特徴次元を選ぶ方法など簡略化した設定）について初めて厳密な解析を行いました​。この研究では、「ランダムフォレストは一貫して（consistent）動作し、またスパース性（不要なノイズ特徴が多数存在する状況）に適応する」ことが示されています​。スパース性への適応とは、真に有効な特徴変数がごく一部であっても、その収束レート（予測誤差の減少速度）が有効特徴の次元数にのみ依存し、無関係な特徴の数には依存しないことを意味します​。これは、ランダムフォレストが高次元でも不要な変数を無視し、本質的な変数に集中できることを直感的に裏付ける結果です。

#### Scornet et al. 2015の貢献

続いて、Erwan Scornet と Gérard Biau らによるさらなる発展があります。[Scornet・Biau・Vert (2015)](https://projecteuclid.org/journals/annals-of-statistics/volume-43/issue-4/Consistency-of-random-forests/10.1214/15-AOS1321.full)​ では、Breiman (2001) のオリジナルのランダムフォレストアルゴリズムにおける（非ブートストラップサンプリングと、$m_{try}$ パラメータによるランダムな特徴選択による分割）ランダムフォレスト推定量の一致性が、推定対象となる真の関数が特徴量に対して加法的である場合に、一致性が成り立つことを示しました。

- 加法性の仮定
$$
    Y = \sum_{j=1}^{p}m_{j}(X^{(j)})+\varepsilon \quad \mathrm{where} \quad X = (X^{(1)},...,X^{(p)}) \sim U([0,1]^{p}),\quad \varepsilon \sim N(0,\sigma^2) 
$$
ここで、[Scornet・Biau・Vert (2015)](https://projecteuclid.org/journals/annals-of-statistics/volume-43/issue-4/Consistency-of-random-forests/10.1214/15-AOS1321.full)​の一致性の証明に限らず、一致性や漸近正規性の証明では、ランダムフォレストを構成する木のパラメータをどのように制御するかが本質的であり、損失関数をどのように設定しているかは一致性や漸近正規性に対して影響を与えていません。この点には注意が必要です。

[Scornet・Biau・Vert (2015)](https://projecteuclid.org/journals/annals-of-statistics/volume-43/issue-4/Consistency-of-random-forests/10.1214/15-AOS1321.full)においては、木に対して次の仮定が置かれます。まず、木を構成する際のサブサンプルサイズ$a_n$は、観測されたサンプル$n$ に対して、無限大に発散します$a_n \rightarrow \infty$. また、木を構成する葉の数$t_n$も、$t_n \rightarrow \infty$を満たしますが、この発散速度には制約があり、$t_n (\log a_n)^9/a_n \rightarrow 0$ となることが条件として与えられます。つまり、サンプルの発散速度に対して、木の葉を増やす速度はそれよりも遅い速度が要求されます。

また、$m_{try} = p$の状況下において、真の関数が、観測された変数$p$個のうち、モデルに含まれる変数が$s$個である状況を考えます。
$$
    Y = \sum_{j=1}^{s} m_j (X^{(j)}) + \varepsilon
$$
このとき、任意の$m_j$ が 任意の$Y$の区間$[a,b]$において定数関数でないならば、十分高い確率で木の分割変数は$\{1,2,...,s\}$から選ばれることを示しました。これは、ランダムフォレストが高次元の観測のもとで有効に動作することを示す根拠の1つと捉えることができます。

#### Wager et al., 2014-2018の貢献
次に、Stefan Wager が Stanfordの研究グループで取り組んだランダムフォレストの一致性と漸近正規性、および漸近分散の導出、さらにオリジナルのcausal forestの提案までの流れを説明します。まず、ランダムフォレストの一致性について[Wager and Walther (2014)](https://arxiv.org/abs/1503.06388) では、Scornet et al.,(2015)とは異なる仮定をおいて一致性を示しています。まず、木の葉の数$k_n$に対して、
$$
    \lim_{n\rightarrow \infty}\frac{\log(n)\max \left(\log(d), \log\log(n)\right)}{k} = 0
$$
の仮定および Sparse signal の仮定をおく。すなわち、$p$次元の特徴量$X$に対して、有効な次元の集合$\mathcal{Q} =\{1,2,...,s\}$次元で、$(Y,X_{\mathcal{Q}}) \perp \!\!\! \perp (X_{-\mathcal{Q}})$ が成り立つ。
さらに、Monotone signal の仮定をおく。この仮定は、$j \in \mathcal{Q}$（有効な次元）に対して、$x_{(-j)} \in [0,1]^{d-1}$ を固定した時に、以下の式を満たすような最小の効果$\beta > 0$が存在することである。
```math
    \left|\mathbb{E} \left[ Y_i \mid (X_i)_{-j} = x_{-j}, (X_i)_{j} > \frac{1}{2} \right] - \mathbb{E}\left[ Y_i \mid (X_i)_{-j} = x_{-j}, (X_i)_{j} \leq \frac{1}{2} \right] \right| \geq \beta
```
さらに、$E[Y|X=x]$ に対するLipchitz連続性を仮定したもとで、以下で定義されるGuess-and-Check forestは、一様一致性を持つ。Guess-and-check forest は次のように定義される。

---
### Guess-and-Check Forest

### Input
- $ n $ 個の訓練データ $(X_i, Y_i)$
- 最小葉ノードサイズ $ k $
- バランスパラメータ $ 0 < \alpha < 1/2 $

### アルゴリズム
Guess-and-Check Tree は、以下の分割手順を再帰的に適用し、分割が不可能になるまで処理を行う。すなわち、全ての終端ノードの訓練データ数が $ 2k $ 未満になるか、そもそも (12) の条件を満たす分割が存在しない場合まで繰り返す。

1. ノード $ \nu $ を選択し、そこに少なくとも $ 2k $ 個の訓練データが含まれることを確認する。
2. **候補となる分割変数** $ j \in \{1, \dots, d\} $ を一様ランダムに選択する。
3. **最小二乗誤差の分割点** $ \hat{\theta} $ を選択する。より具体的には、
    $$
    \hat{\theta} = \arg\max_{\theta} \ell (\theta)
    $$
    ここで、目的関数 $ \ell(\theta) $ は次のように定義される。
    $$
    \ell (\theta) := \frac{4 N^-(\theta) N^+(\theta)}{(N^-(\theta) + N^+(\theta))^2} \Delta^2(\theta)
    $$
    分割点 $ \theta $ は、ノード $ \nu $ に属するサンプル $ X_i $ の成分 $ (X_i)_j $ のいずれかに対応する値とする。
    $$
    \alpha \times | \{ i : X_i \in \nu \} |,  k \, \leq\, N^-(\theta), \quad N^+(\theta)
    $$
    また、以下の定義を用いる。
    $$
    \Delta(\theta) = \frac{1}{N^+}\sum_{\{ i : X_i \in \nu(x), (X_i)_j > \theta \}} Y_i - \frac{1}{N^-}\sum_{\{ i : X_i \in \nu(x), (X_i)_j \leq \theta \}} Y_i
    $$
    ただし、
    $$
    N^-(\theta) = | \{ i : X_i \in \nu, (X_i)_j \leq \theta \} |, \, N^+(\theta) = | \{ i : X_i \in \nu, (X_i)_j > \theta \} |
    $$
4. **分割条件の判定:**
    - 変数 $ j $ に対して、すでに成功した分割が存在する場合。
    - または、以下の条件が満たされる場合。
    $$
    \ell (\hat{\theta}) \geq \left( 2 \times 9M \sqrt{\frac{\log(n) \log(d)}{k \log((1 - \alpha)^{-1})}} \right)^2
    $$
    この場合、$ j $ 番目の変数に対してノード $ \nu $ を $ \hat{\theta} $ で分割する。そうでない場合は、ノード $ \nu $ はこの時点では分割されない。

### **Guess-and-Check Forest**
Guess-and-Check Forest は、上記のプロシージャに従って **独立に生成された \( B \) 本の Guess-and-Check 木の平均** を取ることで構築される。

---
次は、Random forestの漸近分散の推定法に関する結果である。まず、漸近正規性の結果に先立って、[Wager, Hastie and Efron (2014)](https://jmlr.org/papers/volume15/wager14a/wager14a.pdf) では、[Efron (2014)](https://www.tandfonline.com/doi/abs/10.1080/01621459.2013.823775) においてモデル選択によるデータ駆動型のアプローチがもたらす予測誤差の過小評価を補正する方法として提案した Infinitesimal Jacknifeによる方法を、ランダムフォレストに拡張し、ランダムフォレストによる推定量のばらつきを評価する方法を提案した。
$b=1,2,...,B$番目の回帰木による推定値を$\hat{f}_{b}(x)$とし、 $N_{ib} \in \{0,1\}$ が $b$番目の回帰木（Double Sample Treeの場合には、いずれか一方のグループに含まれる場合に$N_{bi}=1$と定義する）に含まれるかどうかを表す指示関数とする。
$$
    \hat{V}_{IJ}(x) = \frac{n-1}{n} \left(\frac{n}{n-s}\right)^2 \sum_{i=1}^{n}\mathrm{Cov}^{*}\left[\hat{f}_{b}^{*}(x), N_{ib}^{*}\right]^2
$$
これに対して、有限標本化での推定量は
$$
    \hat{V}_{IJ}^{B}(x) = \frac{n-1}{n} \left(\frac{n}{n-s}\right)^2 \sum_{i=1}^{n}\frac{1}{B}\sum_{b=1}^{B}\left(N_{bi}^{*}-\frac{s}{n}\right)\left(t_{b}^{*}(x)-\bar{t}^{*}(x)\right)
$$
となる。ただし、この結果はEmpiricalな評価にとどまっており、実際に漸近分散と一致することを示したのは、このあとの[Wager and Athey (2019)](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1319839)である。漸近正規性の議論に移る前にもう1つの重要な論文は、[Athey and Imbens (2016)](https://www.pnas.org/doi/10.1073/pnas.1510489113) において指摘された木構造モデルの学習におけるバイアスの問題である。木構造モデルを学習する際には、テストデータへの当てはまりがよくなるように学習を行う必要がある。しかし、一般的なCARTのような学習では訓練データを用いて木構造の学習と木による予測値の両方を学習するため、実はテストデータに対しての当てはまりを最適化しているのではないということを示している。また、この論文では、ランダム化比較試験に対する介入効果の異質性を発見する回帰木としてCausal Treeを提案している。従来のCART的なアプローチでは、共変量$X$の分割と分割によって生成された葉の値には相関があるため、相関に依存した因果効果が検出されてしまう。そこで、**Honest（誠実性）** と呼ばれる概念を導入することで、この問題を解消する。誠実性とは、データを2つに分け、一方で木構造を学習させ、もう一方で予測値の計算と検定を行えば、分割と予測値は独立になるため、効果の検証ができるということである。ここで、導入された**誠実性**が次に示す漸近正規性を示すための1つの鍵となる。

[Wager and Athey (2019)](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1319839) では、次の性質を満たすbagging treeに対して、有限標本化でのバイアスを評価し、漸近正規性を示した。
- サブサンプルを2つに分割し、サブサンプルの片方を木構造の学習に、もう一方を葉の推定値の計算に用いる Double Sample Tree を 弱学習器として採用する
- 任意の特徴量$X^{(j)}, \, j=1,2,...,d$が、ノードの分割で選択される確率が$0$ではない。
- subsample size $s_n$ は、サンプルサイズ$n$、次元$d$、および分割を制御するパラメータ$\alpha \leq 0.2$ に対して、次の関係式を満たす。
    $$
        s_n \approx n^{\beta}, \qquad 1-\left(1+\frac{d}{\pi}\frac{\log(\alpha^{-1})}{\log\left((1-\alpha)^{-1}\right)}\right)^{-1} < \beta < 1
    $$
- $\mu(x) := E[Y|X]$ が リプシッツ連続 である。

また、この論文では上記に述べた$\hat{V}_{IJ}$がランダムフォレストの漸近分散に対する一致推定量であることが示されており、ランダムフォレストによる推定量の分散を与えた点で画期的な論文であった。

これに加えて[Wager and Athey (2019)](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1319839)では、因果効果を推定する方法として、causal forestを提案している。このcausal forestは現在はほとんど採用されないが、概念だけ紹介しておく。causal forest は、[Athey and Imbens (2016)](https://www.pnas.org/doi/10.1073/pnas.1510489113) で提案された causal tree を弱学習器として用いるランダムフォレストである。causal tree は、回帰におけるノードの分割基準を、処置効果推定の文脈へと拡張したものである。ただし、単純に拡張したものではなく、テストデータへの当てはまりを最適化するような基準を用いている。
$$
\begin{align*}
    -\widehat{\text{EMSE}}_{r} \left( \mathcal{S}^{\text{tr}}, N^{\text{est}}, \Pi \right) 
    &\equiv \frac{1}{N^{\text{tr}}} \sum_{i \in \mathcal{S}^{\text{tr}}} \hat{\tau}^2 \left( X_i, \mathcal{S}^{\text{tr}}, \Pi \right) \\
    &\quad - \left( \frac{1}{N^{\text{tr}}} + \frac{1}{N^{\text{est}}} \right) \cdot \sum_{\ell \in \Pi} 
    \left( \frac{S^2_{\mathcal{S}^{\text{tr}}_{\text{treat}}} (\ell)}{p} 
    + \frac{S^2_{\mathcal{S}^{\text{tr}}_{\text{control}}} (\ell)}{1 - p} \right).
\end{align*}
$$






## 一般化ランダムフォレストの一致性・漸近正規性
### 一般化ランダムフォレストと局所推定方程式
### 一般化ランダムフォレストの一致性
### 一般化ランダムフォレストの漸近正規性
### 一般化ランダムフォレストの漸近分散推定

## 一般化ランダムフォレストの応用
### quantile regression forest と、一般化ランダムフォレストによる分位点推定
### 一般化ランダムフォレストによる条件付き因果効果の推定
### 一般化ランダムフォレストによるInstrumental variableを用いた因果効果の推定
### survival forest と 一般化ランダムフォレストによる生存関数推定
### 一般化ランダムフォレストによる 条件付き causal survival effectの推定

## 一般化ランダムフォレストの拡張
### Locally Linear Forest
### boosted regression forest 
### Sufficient dimension reduction forest

## ランダムフォレストのハイパーパラメータ選択

## ランダムフォレスト予測の解釈
### 変数重要度とは
### 変数重要度の理論 (A General Framework for Inference on Algorithm-Agnostic Variable Importance)
### SIRUS (Interpretable Random Forests via Rule Extraction)
### Sobol-MDA (MDA for random forests: inconsistency, and a practical solution via the Sobol-MDA)
### SHAFF (SHAFF: Fast and consistent SHApley eFfect estimates via random Forests)
### Variable importance for causal forests

## 一般化ランダムフォレストの応用事例
### 医療分野
- Suicide Risk Among Hospitalized Versus Discharged Deliberate Self-Harm Patients: Generalized Random Forest Analysis Using a Large Claims Data Set
- Assessing the properties of patient-specific treatment effect estimates from causal forest algorithms under essential heterogeneity
### 経済分野
- Understanding Heterogeneous Impact of Medicaid Expansion Using Generalized Random Forest

### 金融分野
- Predicting Value at Risk for Cryptocurrencies With Generalized Random Forests

### 画像処理分野
- Meta-forests: Domain generalization on random forests with meta-learning

## Some important topics 
### X-outlier and overlap assumption
### Y-outlier and Huberized Loss
### Conformal Predictions

