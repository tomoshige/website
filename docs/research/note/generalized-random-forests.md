一般化ランダムフォレストについて、包括的なレビュー論文を作成します。
以下の構成で作成しまます。

## 1. ランダムフォレストの概観
### 1.1 ランダムフォレストの背景
### 1.2 ランダムフォレストの拡張と応用
### 1.3 Breiman 2001 から 2010年までのランダムフォレストの理論解析の歩み
[Breiman (2001)](https://link.springer.com/content/pdf/10.1023/a:1010933404324.pdf)年に提案したランダムフォレスト（Random Forest）は、複数の決定木をランダムに生成し、それらの予測を平均する強力なアンサンブル学習法です​。実践的には高次元データへの適応力や汎用性の高さから非常に成功を収めました​。しかし、その卓越した性能にもかかわらず、提案当初は理論的性質の理解が大きく遅れていました。木の構造がデータに強く依存し、さらにランダム化要素も含むため、その数学的解析は困難であり、「広く使われているにもかかわらず理論的性質はほとんど知られていない」と指摘されていました​。このように、2000年代前半までは理論と実践のギャップが大きかった。
この他、[Breiman (2004)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2a42f39add8332a7139d44a6e77496c0571e4f24) でも理論的な部分について解析を試みているが、ランダムフォレストの理論解析には至らず、直感的な理解と$$を結びつけたものにとどまっている。2010年以降の理論解析に大きな貢献を果たしているのは、[Lin and Jean (2006)](https://www.tandfonline.com/doi/abs/10.1198/016214505000001230?casa_token=tUe3DH9TNEEAAAAA:sQZepLDXzrVjTx9kd14uU0BhkQKRTBrQLncg2fs-mm63ya_auCXQvNYscDwWd4Swep1SBDE5tWB0qA)の研究で、この研究ではk-nearest neighborhood estimator と ランダムフォレストによる推定量を結びつけたという意味で非常に大きな貢献をしている。この Lin and Jean (2006) の議論を拡張させたのが、[Biau and Devroye (2010)](https://www.sciencedirect.com/science/article/pii/S0047259X10001387)である。これらの結果は、2014年以降に続くランダムフォレストの理論的な結果の大きな礎となっている。また、[Meinshausen (2006)](https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf) では、ランダムフォレストによる条件付き分位点関数の推定法である quantile regression forest においてもランダムフォレストの一致性に言及されている。

### 2010年以降のランダムフォレストの理論解析の歩み：一致性と、漸近正規性の証明



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

