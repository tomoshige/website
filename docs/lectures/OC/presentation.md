---
marp: true
theme: default
paginate: true
header: "生成AIによるパラダイムシフト"
footer: "© 2025"
style: |
  section {
    font-family: 'Helvetica', 'Arial', sans-serif;
    background-color: #fff;
    color: #333;
    font-size: 1.5em;
  }
  h1 {
    color: #1a237e;
    font-size: 2em;
  }
  h2 {
    color: #283593;
    margin-top: 0.7em;
  }
  h3 {
    color: #3949ab;
  }
  img {
    display: block;
    margin: 0 auto;
  }
  code {
    background: #f0f0f0;
    border-radius: 4px;
    padding: 0.2em 0.4em;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1em;
  }
  li {
    margin-bottom: 0.5em;
  }
  .transition {
    background-color: #3949ab;
    color: white;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
  }
  .transition h1, .transition h2 {
    color: white;
  }
  .highlight {
    color: #d81b60;
    font-weight: bold;
  }
  .definition {
    font-size: 0.8em;
    background-color: #f5f5f5;
    border-left: 4px solid #3949ab;
    padding: 0.5em 1em;
    margin: 0.5em 0;
  }
  .question {
    background-color: #e3f2fd;
    padding: 0.8em;
    border-radius: 4px;
    margin-top: 1em;
  }
---

# 生成AIによるパラダイムシフト
### 生成AI時代に必要な能力

---

## 目次

1. イントロダクション：生成AIの進化と転換点
2. AIの仕組みと技術的基盤
3. 根本的なパラダイムシフト
4. 問題定義・解決能力の質が問われる時代
5. まとめ：生成AI時代の学びの方向性

---

## 1. イントロダクション：生成AIの進化

- **2022**: ChatGPTの登場 - 会話型AIの誕生と言語処理の革新
- **2023末**: 日本語能力の向上、代筆レベルに到達
- **2024初**: 医師国家試験や薬剤師試験に合格するレベルに到達
- **2024.9**: OpenAI o1-preview - 「長考」機能によるSTEM領域での革新
- **2025現在**: Claude 3.7 Sonnet、GPT-4.5、o3-mini-high - 人間上位1%の知的能力

![width:800px](image.png)

---

## 転換点：従来のAIと現在の能力差

<div class="columns">
<div>

**従来の弱点**:
- STEM領域の問題解決力の欠如
- 限定的なプログラミング能力
- 短期的思考に限定

**現在の能力**:
- 長考による問題解決能力
- 博士課程レベルの質問への対応
- 人間エンジニア上位1%レベルのコード生成

</div>
<div>

![width:100%](image-1.png)

</div>
</div>

<div class="question">
💡 **考察**: このような能力変化は、あなたの専門分野にどのような影響を与えるでしょうか？
</div>

---

<!-- 技術セクションへの移行スライド -->
<div class="transition">

# 2. AIの仕組みと技術的基盤
## 優れた問題解決者を理解する

</div>

---

## 2.1 Transformerと自己注意機構

![bg right:40% width:100%](image-4.png)

- **Transformerアーキテクチャ**: 「Attention is all you need」(2017)
- テーマの記憶と文脈を踏まえた次単語予測
- **Multi-head Attention**: 単語間の複雑な関係性を同時に学習

<div class="definition">
**自己注意機構 (Self-Attention)**: 文章内の各単語が他のすべての単語にどの程度「注意」を払うべきかを計算し、文脈に応じた意味を獲得する機構
</div>

---

## 2.2 強化学習の統合

- 生成した文章の質を評価するフィードバックループ
- **RLHF** (Reinforcement Learning from Human Feedback): 人間の選好に基づく報酬モデル
- 自己改善メカニズム（自己批評・修正）により品質向上

<div class="columns">
<div>

![width:100%](image-2.png)

</div>
<div>

![width:100%](image-3.png)

</div>
</div>

<div class="definition">
**RLHF**: 人間の評価者が生成されたテキストの質を判断し、その評価をもとにAIモデルを訓練する手法
</div>

---

## 2.3 専門知識試験における高性能化 (1/2)

### 医師国家試験 (2025)
- o3-mini-highによる成績:
  - 必修問題: <span class="highlight">上位10%</span>相当
  - 一般臨床問題: <span class="highlight">全受験者中第3位</span>相当
- 専門知識領域においてトップレベルの成績

### 薬剤師国家試験 (2024)
- o1-previewによる <span class="highlight">正答率100%</span>

---

## 2.3 専門知識試験における高性能化 (2/2)

### 東大・京大数学入試 (2025)
- 東大数学: 大問6問中<span class="highlight">5.5問正解</span>
- 京大数学: <span class="highlight">全問正解</span>（上位1%レベル）

<div class="definition">
**長考**: 生成AIが自身の解答を内部で検証・修正しながら、複雑な問題に対して段階的に解を導出するプロセス
</div>

<div class="question">
💡 **討論**: 大学入試において、AIが人間より高い成績を収めることの社会的影響は？
</div>

---

## 2.4 コーディング能力の飛躍的向上

### SWE-bench評価
- 実際のGitHubの課題解決能力を定量的に測定
- バグ修正、機能実装、コード生成の総合評価
- **Claude 3.7 Sonnetの能力**: 人間エンジニアの<span class="highlight">上位1%以上</span>

![width:700px](image-5.png)

---

## 実例：姿勢推定アプリケーション開発

```python
# Claude 3.7 Sonnetで生成した姿勢推定コード例
import mediapipe as mp
import cv2
import numpy as np

# MediaPipe Poseモデルの初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True
)
mp_drawing = mp.solutions.drawing_utils

def analyze_posture(image_path):
    # 画像読み込み
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 姿勢推定実行
    results = pose.process(image_rgb)
    
    # 結果の解析と姿勢評価...
```

---

## 発展的応用：姿勢推定技術の活用領域

<div class="columns">
<div>

**姿勢推定技術基盤**:
- コンピュータビジョンアーキテクチャ
- 2D/3Dポーズ推定方法
- リアルタイム処理最適化

**スポーツ科学応用**:
- アスリートパフォーマンス分析
- 傷害リスク評価
- トレーニング最適化

</div>
<div>

**健康モニタリング**:
- 神経学的症状の早期発見
- リハビリテーション進行の定量化
- 高齢者ケアのリモートモニタリング
- 電子健康記録との統合

</div>
</div>

---

## 2.5 高度なリサーチ能力：Deep Research

- 先行事例調査の<span class="highlight">自動化と効率化</span>
- <span class="highlight">2〜3時間の調査作業を10分程度</span>に短縮
- 調査結果の根拠（URL、文献）の明示
- 情報収集から加工までの一貫処理
- 文献調査における<span class="highlight">ハルシネーション問題の解決</span>

<div class="definition">
**Deep Research**: GPT-4.5/o3-mini-highに搭載された機能で、調査した情報の信頼性を担保するために出典を明示するWebブラウジング能力
</div>

---

## 2.6 企業/研究機関向け進化：MCPの登場

<div class="columns">
<div>

- **検索拡張生成（RAG）**: 
  - 外部データソースとの連携
  - 情報鮮度の維持

- **Model Context Protocol (MCP)**:
  - ローカルデータの安全な活用
  - セキュリティを担保した知識拡張
  - プライベートデータの統合

</div>
<div>

![width:100%](image-6.png)

</div>
</div>

<div class="definition">
**MCP**: Anthropicが開発した技術で、社内のプライベートデータを安全にAIに参照させるためのプロトコル
</div>

---

<!-- パラダイムシフトセクションへの移行スライド -->
<div class="transition">

# 3. 根本的なパラダイムシフト
## 変わる人間の役割

</div>

---

## AIがどれだけ進歩しても変わらない根本原則

<div class="columns">
<div>

### 1. 人間の責任原則
- AI出力に対する最終的な責任は人間にある
- 正しさの検証は常に必須
- 法的・倫理的責任の所在は変わらない

</div>
<div>

### 2. 問題定義の主体性
- 問題は人間の活動から生まれる
- 何が問題かはAIではなく人間が決める
- 価値判断は人間の領域

</div>
</div>

<div class="question">
💡 **考察**: あなたの研究領域では、どのような問題がAIでは特定できないでしょうか？
</div>

---

## 人間の役割の本質的シフト

![bg right:40% 80%](https://via.placeholder.com/400x300/e3f2fd/3949ab?text=Human+Role+Shift)

### 旧パラダイム
- 問題解決者
- 知識の保持者
- 計算実行者

### 新パラダイム
- <span class="highlight">問題定義者</span>: 範囲と制約の確立
- <span class="highlight">問題翻訳者</span>: AIが解ける形への変換
- <span class="highlight">出力検証者</span>: 解決策に対する責任

---

## 人間-AI協働の新しい方程式

$$\text{問題解決} = \underbrace{\text{問題定義}}_{\text{人間}} + \underbrace{\text{問題翻訳}}_{\text{人間}} + \underbrace{\text{解答生成}}_{\text{AI}} + \underbrace{\text{出力検証}}_{\text{人間}}$$

<div class="columns">
<div>

### 従来型学習の焦点
- 解法の暗記と適用
- 計算スキル
- 記憶力

</div>
<div>

### 新時代の学習焦点
- 問題の本質理解
- AI対話・指示能力
- 批判的検証能力

</div>
</div>

---

## 例：数学教育パラダイムの変化 (1/2)

### 従来の問題アプローチ:
次の不定積分を求めなさい。  
$$\int \frac{x}{(1+x^2)^2} \, dx.$$

<div class="columns">
<div>

**評価される能力**:
- 置換積分の知識
- 計算の正確さ
- 解法の記憶

</div>
<div>

**教育の焦点**:
- 公式の暗記
- 解法パターンの習得
- 計算トレーニング

</div>
</div>

---

## 例：数学教育パラダイムの変化 (2/2)

### 新時代の問題アプローチ:
以下の積分解答例の誤りを指摘し、最終答えへの影響を説明せよ。

1. 置換 $u = 1+x^2$ とおくと、$du = 2x\,dx$ となる。
2. $x\,dx = du$ と置き換え、$\int \frac{x}{(1+x^2)^2} \, dx = \int \frac{1}{u^2}\,du$
3. $\int \frac{1}{u^2}\,du = -\frac{1}{u}+C = -\frac{1}{1+x^2}+C$

<div class="columns">
<div>

**評価される能力**:
- 誤りの論理的検出
- 数学的批判的思考
- 影響の評価能力

</div>
<div>

**教育の焦点**:
- 理解の深さ
- 批判的思考
- 検証スキル

</div>
</div>

---

## 例：社会科学のレポート課題の変化

<div class="columns">
<div>

### 従来の問題:
年収103万円の壁とは何か説明し、この年収103万円の壁による問題点を述べなさい。

**評価される能力**:
- 知識の記憶と再現
- 定型的な記述力
- 基本的分析能力

</div>
<div>

### 新時代の問題:
年収103万円の壁について調査したレポートを読み、この内容の適切性を検証した上で、あなたの立場を明確にして意見を述べなさい。

**評価される能力**:
- 情報の妥当性評価
- メタ分析能力
- 独自視点の構築力

</div>
</div>

---

<!-- 問題解決セクションへの移行スライド -->
<div class="transition">

# 4. 問題定義・解決能力の質が問われる時代
## 循環型問題解決プロセスの時代へ

</div>

---

## 課題解決プロセスの本質的変化

<div class="columns">
<div>

### 従来の課題解決プロセス
1. 定義された問題を理解
2. 解法を適用
3. 解答を提出
4. フィードバックを待つ
5. *(数日〜数週間後)*
6. 次の問題へ

**特徴**: 一方向的・低頻度サイクル

</div>
<div>

### 生成AI時代の課題解決プロセス
1. 問題の本質を抽出
2. AIが解ける形に問題を翻訳
3. AI解答を即時取得
4. 解答を検証・評価
5. 問題定義を洗練
6. *(数分〜数時間で)*
7. 繰り返し改善

**特徴**: <span class="highlight">循環的・高頻度サイクル</span>

</div>
</div>

---

## 新時代に重要となるスキルセット

<div class="columns">
<div>

### 生成AIとの対話能力
- プロンプト設計力
- 効果的なフィードバック
- 問題の適切な分解能力

### 解答の精査能力
- 批判的思考
- ドメイン知識に基づく検証
- エッジケースの特定

</div>
<div>

### 問題の言い換え・翻訳能力
- 曖昧さの除去
- 数学的・論理的定式化
- 解像度の高い問題設定

### メタ認知能力
- 自己の知識境界の認識
- 問題アプローチの戦略選択
- 検証手法の最適化

</div>
</div>

---

## 問題解像度を高める例：曖昧な問題の翻訳 (1/2)

### 曖昧な問題提起:
既存の画像を水増しするAという方法を作り、その水増しデータで別データBと比較できるか？

<div class="columns">
<div>

**問題点**:
- 「水増し」の定義が不明確
- 比較の目的が不明
- 統計的枠組みの欠如
- 評価基準の不在

</div>
<div>

**AIへの指示として不適切な理由**:
- 多義的な解釈が可能
- 数理的定式化がない
- 解くべき問題の境界が不明確

</div>
</div>

---

## 問題解像度を高める例：曖昧な問題の翻訳 (2/2)

### 高解像度の問題定義:

画像の背後にある生成過程を $f(X)$ とする。この $f(X)$ を、画像データセット D を用いてニューラルネットワーク（VAE, GAN, U-net, Diffusionモデル）で学習したものを $\hat{f}(z|D)$ とする。このとき、$\hat{f}(z|D)$ を用いて生成した画像を $D'$ として、新たなデータ $B$ が与えられた時、$D'$ と $B$ を統計的仮説検定で差の検定を行うと、有効サンプルサイズ、Type I Error, Type II Error の観点から妥当な検定となりうるか？


<div class="question">
💡 **実践**: あなたの研究分野で曖昧な問題を1つ挙げ、高解像度に翻訳してみましょう
</div>

---

## 英語力の必要性の再定義

<div class="columns">
<div>

### 相対的重要度が低下する面
- 単純な英語の文章作成（Writing）
- 一般的な会話（Speaking）
- 基本的なリスニング

### より重要になる面
- 英文法（Grammar）の理解
- 英語読解力（Reading）
- 専門領域における英語表現
- AI出力英語の検証能力

</div>
<div>

![bg 80%](https://via.placeholder.com/400x300/e8eaf6/3949ab?text=English+Skills+Shift)

</div>
</div>

$$
\text{必要な英語力} = \text{専門用語理解} + \text{構文把握} + \text{AI出力検証}
$$

---

<!-- まとめセクションへの移行スライド -->
<div class="transition">

# 5. 生成AI時代に大学で学ぶ意義
## 学びの本質的価値の再定義

</div>

---

## 大学教育の本質的価値の変化

<div class="columns">
<div>

### 問題の解像度を高めるツール獲得
- 専門領域の体系的知識
- ドメイン固有の思考フレームワーク
- 複雑性を扱う概念的ツール
- 問題分解の方法論

</div>
<div>

### 解答の精査能力を高めるツール獲得
- 批判的思考法の訓練
- 検証のためのメソドロジー
- エラー検出パターンの習得
- 品質評価の判断基準

</div>
</div>

<div class="question">
💡 **討論**: 大学教育は生成AI時代にどのように変革すべきでしょうか？
</div>

---

## まとめ：生成AI時代の学びの方向性

<div class="columns">
<div>

### 知識の価値の再定義
- 単なる暗記から批判的評価へ
- 知識は検証ツールとして重要性増大
  
### 学びの目的の変化
- 解法の習得から問題定義能力へ
- 答えの再現から思考プロセスの習得へ

</div>
<div>

### 人-AI協働の高速ループ習得
- 問題発見→AI解決→検証→問題精緻化
- この循環の質を高める基盤知識
- 試行錯誤の効率化

### メタスキルの重要性
- 学び方を学ぶ力
- AIとの協働方法の最適化
- 批判的思考の体系的習得

</div>
</div>

---

## 最終メッセージ：生成AIと共に成長する

![bg right:40% 80%](https://via.placeholder.com/400x300/e8eaf6/3949ab?text=Future+of+Learning)

### 変わらない本質
- 人間の洞察と判断の重要性
- 問題を見つける創造性の価値
- 検証と責任の所在

### これからの時代の成功要因
- AIを使いこなす思考法の習得
- 高速フィードバックループの構築
- 批判的思考による質の担保

---

# ご清聴ありがとうございました

## 参考文献・リソース
- "Attention is all you need" (Vaswani et al., 2017)
- "解像度を高める" (https://speakerdeck.com/tumada/jie-xiang-du-wogao-meru)
- "SWE-bench: 生成AIコーディング能力評価" (https://qiita.com/tosenbo/items/57ed6ded19da2b24d900)
- "Reinforcement Learning from Human Feedback: A Review" (Casper et al., 2023)
- "The AI Revolution in Scientific Discovery" (Nature, 2024)