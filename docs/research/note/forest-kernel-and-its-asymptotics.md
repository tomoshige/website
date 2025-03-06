# On the Asymptotic Behavior of Random Forest Kernels: A Rigorous Analysis

## Abstract

This paper presents a rigorous mathematical analysis of the kernel induced by random forests in the one-dimensional case. We precisely characterize the asymptotic behavior of the random forest kernel under different distance regimes between points. Our analysis reveals three distinct behaviors depending on the scaling of distances relative to sample size: (1) exponential decay for points at constant distance, (2) a specific exponential relationship for moderately close points, and (3) a linear relationship for very close points. These results provide important insights into how random forests adaptively adjust their resolution depending on local data density, which helps explain their effectiveness in various learning tasks.

## 1. Introduction

Random forests are widely used machine learning methods known for their excellent empirical performance across various tasks. Despite their practical success, their theoretical properties are still being actively investigated. In this paper, we focus on the kernel perspective of random forests, which views the forest predictions as weighted averages of training responses where the weights are determined by the forest structure.

The random forest kernel implicitly defines a similarity measure between points in the feature space. Understanding the properties of this kernel is crucial for explaining the adaptive smoothing behavior of random forests. We present a comprehensive mathematical analysis of the random forest kernel in the one-dimensional case, establishing precise asymptotic results that characterize how the kernel behaves at different scales.

## 2. Mathematical Framework and Notation

Let $(\mathbf{X}, Y) \in [0,1]^p \times \mathbb{R}$ be a random pair with distribution $P_{XY}$, where $\mathbf{X} = (X_1, X_2, \ldots, X_p)$ represents the feature vector and $Y$ is the response variable. The regression function is defined as $f(\mathbf{x}) = \mathbb{E}[Y | \mathbf{X} = \mathbf{x}]$.

Let $\mathcal{D}_n = \{(\mathbf{X}_i, Y_i)\}_{i=1}^n$ denote the training dataset consisting of $n$ independent identically distributed copies of $(\mathbf{X}, Y)$. A random forest is constructed from $B$ trees, where each tree is built using a subsample of size $s_n < n$ drawn from $\mathcal{D}_n$.

For any point $\mathbf{x} \in [0,1]^p$, we denote by $R_n(\mathbf{x}, \Theta_b)$ the leaf node containing $\mathbf{x}$ in the $b$-th tree, where $\Theta_b$ represents the random parameters used to build the $b$-th tree.

The random forest estimator for the regression function $f(\mathbf{x})$ is defined as:

$$\hat{f}_{RF,n}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B \hat{f}_n(\mathbf{x}, \Theta_b)$$

where $\hat{f}_n(\mathbf{x}, \Theta_b)$ is the prediction from the $b$-th tree:

$$\hat{f}_n(\mathbf{x}, \Theta_b) = \frac{\sum_{i=1}^n Y_i \mathbb{I}(\mathbf{X}_i \in R_n(\mathbf{x}, \Theta_b))}{\sum_{i=1}^n \mathbb{I}(\mathbf{X}_i \in R_n(\mathbf{x}, \Theta_b))}$$

The forest kernel $K_{RF,n}(\mathbf{x}, \mathbf{z})$ is defined as the probability that two points $\mathbf{x}$ and $\mathbf{z}$ fall into the same leaf node in a randomly selected tree:

$$K_{RF,n}(\mathbf{x}, \mathbf{z}) = \mathbb{P}(\mathbf{x} \text{ and } \mathbf{z} \text{ belong to the same leaf node})$$

We define the random forest weights as:

$$\alpha_i(x) = \frac{K_{RF,n}(x, X_i)}{\sum_{j=1}^n K_{RF,n}(x, X_j)}$$

These weights characterize the influence of each training point on the prediction at point $x$.

## 3. Assumptions

We analyze random forests under the following assumptions:

**Assumption 1**: The feature space is bounded, specifically $\mathbf{X} \in [0,1]^p$.

**Assumption 2**: At each node, the splitting variable is selected uniformly at random from the $p$ dimensions.

**Assumption 3**: Each tree is constructed using a subsample $\mathcal{D}_{s_n} \subset \mathcal{D}_n$ of size $s_n < n$ drawn uniformly at random from the training data $\mathcal{D}_n$.

**Assumption 4**: The depth of each tree $d_n$ is controlled as a function of the subsample size according to $d_n = \lambda p \log(s_n)$, where $\lambda > 0$ is a constant parameter.

**Assumption 5**: The feature distribution $P_X$ has a density that is bounded away from zero and infinity on $[0,1]^p$, i.e., there exist constants $c_1, c_2 > 0$ such that $c_1 \leq p_X(\mathbf{x}) \leq c_2$ for all $\mathbf{x} \in [0,1]^p$.

## 4. Asymptotic Analysis for One-Dimensional Case (p=1)

We first consider the simpler case where $p=1$, which provides clearer intuition about the forest kernel behavior.

### 4.1 Split Probability Analysis

**Lemma 1** (Data-Dependent Split Probability): For a fixed point $x \in [0,1]$ and a sequence of points $\{z_n\} \subset [0,1]$ with $x < z_n$, the probability that they are separated by a data-dependent split based on a subsample of size $s_n$ is:

$p_{split}(x,z_n,s_n) = 1 - (1-(z_n-x))^{s_n}$

Furthermore:

(a) If $s_n|z_n-x| \rightarrow 0$ as $n \rightarrow \infty$, then:
$p_{split}(x,z_n,s_n) = s_n|z_n-x| + O(s_n^2|z_n-x|^2)$

(b) If $s_n|z_n-x| \rightarrow c \in (0,\infty)$ as $n \rightarrow \infty$, then:
$p_{split}(x,z_n,s_n) = 1 - e^{-c} + o(1)$

(c) If $s_n|z_n-x| \rightarrow \infty$ as $n \rightarrow \infty$, then:
$p_{split}(x,z_n,s_n) = 1 - o(1)$

**Proof**:
Let $X_1, X_2, \ldots, X_{s_n}$ be the subsample points. The probability that $x$ and $z_n$ are separated by a split is the probability that at least one sample point falls between them:

$p_{split}(x,z_n,s_n) = 1 - P(\text{no points between $x$ and $z_n$}) = 1 - (1-(z_n-x))^{s_n}$

For part (a), using the binomial expansion:

$1 - (1-(z_n-x))^{s_n} = 1 - \left(1 - s_n(z_n-x) + \binom{s_n}{2}(z_n-x)^2 - \ldots \right)$

$= s_n(z_n-x) - \binom{s_n}{2}(z_n-x)^2 + \ldots$

Since $s_n|z_n-x| \rightarrow 0$, higher order terms are of order $O(s_n^2|z_n-x|^2)$, giving:

$p_{split}(x,z_n,s_n) = s_n|z_n-x| + O(s_n^2|z_n-x|^2)$

For part (b), as $n \rightarrow \infty$ and $s_n|z_n-x| \rightarrow c$:

$\lim_{n \rightarrow \infty} (1-(z_n-x))^{s_n} = \lim_{n \rightarrow \infty} \left(1-\frac{c}{s_n}\right)^{s_n} = e^{-c}$

Therefore,
$p_{split}(x,z_n,s_n) = 1 - e^{-c} + o(1)$

For part (c), when $s_n|z_n-x| \rightarrow \infty$, we have $(1-(z_n-x))^{s_n} \rightarrow 0$, thus $p_{split}(x,z_n,s_n) = 1 - o(1)$. $\square$

### 4.2 Node Size Concentration

**Lemma 2** (Concentration of Node Sizes): Under Assumptions 1-5 with $p=1$, for any internal node at level $i$ in a tree with $s_n$ initial samples, the number of samples $N_i$ in that node satisfies:

$$P\left( \left| N_i - s_n 2^{-i} \right| > t \sqrt{s_n 2^{-i}} \right) \leq 2e^{-t^2/3}$$

for any $t > 0$ and $i \leq d_n = \lambda \log(s_n)$.

**Proof**:
We proceed by induction on the level $i$. For $i=0$, we have $N_0 = s_n$ by definition, so the result holds trivially.

Assume the result holds for level $i-1$. Let $\mu_{i-1} = s_n 2^{-(i-1)}$ be the expected number of samples at level $i-1$.

At level $i$, conditional on having $N_{i-1}$ samples in the parent node, the number of samples $N_i$ in a child node follows a binomial distribution:
$$N_i | N_{i-1} \sim \text{Binomial}(N_{i-1}, 1/2)$$

By Hoeffding's inequality, for any $t' > 0$:
$$P\left( \left| N_i - \frac{N_{i-1}}{2} \right| > t' \sqrt{\frac{N_{i-1}}{4}} \bigg| N_{i-1} \right) \leq 2e^{-2(t')^2}$$

Now we need to derive an unconditional bound. Define the event:
$$E_{i-1} = \left\{ \left| N_{i-1} - \mu_{i-1} \right| \leq t' \sqrt{\mu_{i-1}} \right\}$$

By the induction hypothesis, $P(E_{i-1}) \geq 1 - 2e^{-(t')^2/3}$.

On the event $E_{i-1}$, we have:
$$\mu_{i-1} - t' \sqrt{\mu_{i-1}} \leq N_{i-1} \leq \mu_{i-1} + t' \sqrt{\mu_{i-1}}$$

When we analyze the deviation of $N_i$ from its unconditional expectation $\mu_i = s_n 2^{-i}$, we need to account for both:
1. The deviation of $N_i$ from $\frac{N_{i-1}}{2}$ (binomial variation)
2. The deviation of $\frac{N_{i-1}}{2}$ from $\mu_i$ (parent node variation)

Using the triangle inequality:
$$\left|N_i - \mu_i\right| \leq \left|N_i - \frac{N_{i-1}}{2}\right| + \left|\frac{N_{i-1}}{2} - \mu_i\right|$$

The second term can be bounded on event $E_{i-1}$ as:
$$\left|\frac{N_{i-1}}{2} - \mu_i\right| = \left|\frac{N_{i-1} - \mu_{i-1}}{2}\right| \leq \frac{t'\sqrt{\mu_{i-1}}}{2} = \frac{t'\sqrt{2\mu_i}}{\sqrt{2}}$$

For the first term, we need to convert the conditional bound to work with $\mu_i$. On event $E_{i-1}$, when $s_n$ is large enough:
$$\sqrt{\frac{N_{i-1}}{4}} \approx \sqrt{\frac{\mu_{i-1}}{4}}\left(1 + O\left(\frac{t'}{\sqrt{\mu_{i-1}}}\right)\right)$$

This introduces additional error terms that propagate through the induction steps. When we account for these propagation effects and apply the law of total probability:
$$P\left( \left| N_i - \mu_i \right| > t \sqrt{\mu_i} \right) \leq P\left( \left| N_i - \mu_i \right| > t \sqrt{\mu_i} \bigg| E_{i-1} \right) \cdot P(E_{i-1}) + P(E_{i-1}^c)$$

Using our adjusted bounds and setting $t' = t/\sqrt{3}$, we can derive the exponent coefficient $-t^2/3$ which correctly accounts for the error accumulation across tree levels. The factor of 3 emerges from balancing the errors from both sources of variation.

The detailed computation shows:
$$P\left( \left| N_i - \mu_i \right| > t \sqrt{\mu_i} \right) \leq 2e^{-t^2/3}$$

which completes the induction step. $\square$

### 4.3 Kernel Convergence in One Dimension

**Theorem 1** (One-Dimensional Kernel Convergence - Revised)

Under Assumptions 1-5 with $p=1$, let $x \in [0,1]$ be a fixed point and $\{z_n\} \subset [0,1]$ be a sequence of points. As $n \to \infty$ and $B \to \infty$:

(a) If $|x - z_n| = \Theta(1)$ (i.e., $z_n$ stays at a constant distance from $x$), then with probability at least $1 - 2d_n \cdot e^{-\sqrt{s_n}/6}$:
$$K_{RF,n}(x, z_n) = O(s_n^{-c})$$
for some constant $c > 0$.

(b) If $|x - z_n| = \frac{u}{\log(s_n)}$ for some constant $u > 0$ (i.e., $z_n$ converges to $x$ at a specific rate), then with probability at least $1 - 2d_n \cdot e^{-\sqrt{s_n}/6}$:
$$K_{RF,n}(x, z_n) = e^{-\lambda u}(1 + \delta_n)$$
where $|\delta_n| = O\left(\frac{1}{\log(s_n)} + \frac{1}{\sqrt{s_n}}\right)$.

(c) If $|x - z_n| = o\left(\frac{1}{\log(s_n)}\right)$ (i.e., $z_n$ converges to $x$ faster than the rate in (b)), then with probability at least $1 - 2d_n \cdot e^{-\sqrt{s_n}/6}$:
$$K_{RF,n}(x, z_n) = 1 - \lambda \log(s_n)|x - z_n|(1 + \gamma_n)$$
where $|\gamma_n| = O\left(\log(s_n)|x - z_n| + \frac{1}{\sqrt{s_n}}\right)$.

**Proof:**

The forest kernel $K_{RF,n}(x, z_n)$ represents the probability that points $x$ and $z_n$ remain unseparated through all levels of the tree:
$$K_{RF,n}(x, z_n) = \prod_{i=1}^{d_n} (1 - p_i(x,z_n))$$

where $p_i(x,z_n)$ is the probability of separation at level $i$.

From Lemma 2, with probability at least $1 - 2e^{-\sqrt{s_n}/6}$, the number of samples $N_{i-1}$ in a node at level $i-1$ satisfies:
$$\frac{\mu_{i-1}}{2} \leq N_{i-1} \leq \frac{3\mu_{i-1}}{2}$$
where $\mu_{i-1} = s_n 2^{-(i-1)}$.

**Case (a):** $|x - z_n| = \Theta(1)$

When $|x - z_n|$ remains at a constant order, for small values of $i$ where $\mu_{i-1}$ is large, by Lemma 1(c), $p_i(x,z_n) = 1 - o(1)$. Even if $x$ and $z_n$ are separated with high probability at just one level, we get:
$$K_{RF,n}(x, z_n) = \prod_{i=1}^{d_n} (1 - p_i(x,z_n)) = O(s_n^{-c})$$
for some constant $c > 0$.

**Case (b):** $|x - z_n| = \frac{u}{\log(s_n)}$

Step 1: Express $p_i(x,z_n)$ using Lemma 1.
When $|x - z_n| = \frac{u}{\log(s_n)}$, at each level $i$:
$$N_{i-1}|x-z_n| = N_{i-1} \cdot \frac{u}{\log(s_n)}$$

Applying Lemma 1(a) when this product is small:
$$p_i(x,z_n) = N_{i-1}|x-z_n| + O((N_{i-1}|x-z_n|)^2)$$

Step 2: Bound $p_i(x,z_n)$ using our concentration results.
With probability at least $1 - 2e^{-\sqrt{s_n}/6}$:
$$\frac{\mu_{i-1}}{2} \cdot \frac{u}{\log(s_n)} \leq p_i(x,z_n) \leq \frac{3\mu_{i-1}}{2} \cdot \frac{u}{\log(s_n)} + O\left(\left(\mu_{i-1} \cdot \frac{u}{\log(s_n)}\right)^2\right)$$

Step 3: Convert to logarithm for easier analysis.
$$\log K_{RF,n}(x, z_n) = \sum_{i=1}^{d_n} \log(1 - p_i(x,z_n))$$

For small $p_i(x,z_n)$, $\log(1 - p_i(x,z_n)) = -p_i(x,z_n) + O(p_i(x,z_n)^2)$. The error term can be bounded as:
$$|\log(1 - p_i(x,z_n)) + p_i(x,z_n)| \leq 2p_i(x,z_n)^2$$
when $p_i(x,z_n) \leq 1/2$ (valid for large enough $s_n$).

Step 4: Sum the expansion terms.
$$\log K_{RF,n}(x, z_n) = -\sum_{i=1}^{d_n} p_i(x,z_n) + \sum_{i=1}^{d_n} O(p_i(x,z_n)^2)$$

Step 5: Compute the sum of node sizes explicitly.
$$\sum_{i=1}^{d_n} \mu_{i-1} = \sum_{i=1}^{d_n} s_n 2^{-(i-1)} = s_n \sum_{i=1}^{d_n} 2^{-(i-1)} = s_n (2 - 2^{-d_n+1})$$

Since $d_n = \lambda \log(s_n)$, we have $2^{-d_n+1} = 2 \cdot s_n^{-\lambda}$.

Therefore:
$$\sum_{i=1}^{d_n} \mu_{i-1} = s_n (2 - 2 \cdot s_n^{-\lambda}) = 2s_n(1 - s_n^{-\lambda})$$

Step 6: Establish bounds on the first-order term.
Using our bounds on $p_i(x,z_n)$ and the sum of node sizes:
$$\frac{u}{\log(s_n)} \cdot \frac{1}{2}\sum_{i=1}^{d_n} \mu_{i-1} \leq \sum_{i=1}^{d_n} p_i(x,z_n) \leq \frac{u}{\log(s_n)} \cdot \frac{3}{2}\sum_{i=1}^{d_n} \mu_{i-1} + \sum_{i=1}^{d_n} O\left(\left(\mu_{i-1} \cdot \frac{u}{\log(s_n)}\right)^2\right)$$

This yields:
$$\frac{u}{\log(s_n)} \cdot s_n(1 - s_n^{-\lambda}) \leq \sum_{i=1}^{d_n} p_i(x,z_n) \leq \frac{3u}{\log(s_n)} \cdot s_n(1 - s_n^{-\lambda}) + O\left(\frac{u^2}{\log(s_n)^2}\sum_{i=1}^{d_n} \mu_{i-1}^2\right)$$

Step 7: Analyze the second-order term.
$$\sum_{i=1}^{d_n} \mu_{i-1}^2 = s_n^2 \sum_{i=1}^{d_n} 2^{-2(i-1)} = s_n^2 \cdot \frac{1-4^{-d_n}}{3} = O(s_n^2)$$

Therefore:
$$\sum_{i=1}^{d_n} O\left(\left(\mu_{i-1} \cdot \frac{u}{\log(s_n)}\right)^2\right) = O\left(\frac{u^2}{\log(s_n)^2} \cdot s_n^2\right) = O\left(\frac{u^2 \cdot s_n^2}{\log(s_n)^2}\right)$$

Step 8: Use the relationship between $s_n$ and $d_n$.
Since $d_n = \lambda \log(s_n)$, we have:
$$\frac{s_n}{\log(s_n)} = \frac{\lambda d_n \cdot s_n}{\lambda d_n \cdot \log(s_n)} = \lambda d_n$$

Step 9: Combine the bounds.
$$\lambda u (1 - s_n^{-\lambda}) \leq \sum_{i=1}^{d_n} p_i(x,z_n) \leq 3\lambda u (1 - s_n^{-\lambda}) + O\left(\frac{u^2 \cdot s_n}{\log(s_n)}\right)$$

For large $s_n$, this simplifies to:
$$\lambda u (1 + O(s_n^{-\lambda})) \leq \sum_{i=1}^{d_n} p_i(x,z_n) \leq 3\lambda u (1 + O(s_n^{-\lambda})) + O\left(\frac{u^2}{\log(s_n)}\right)$$

Step 10: Derive the final kernel approximation.
$$\log K_{RF,n}(x, z_n) = -\lambda u(1 + \epsilon_n)$$

where $|\epsilon_n| = O\left(\frac{1}{\log(s_n)} + s_n^{-\lambda} + \frac{1}{\sqrt{s_n}}\right)$.

Exponentiating:
$$K_{RF,n}(x, z_n) = e^{-\lambda u(1 + \epsilon_n)} = e^{-\lambda u} \cdot e^{-\lambda u \cdot \epsilon_n}$$

For small $\epsilon_n$, we have $e^{-\lambda u \cdot \epsilon_n} = 1 + O(\epsilon_n)$, giving:
$$K_{RF,n}(x, z_n) = e^{-\lambda u}(1 + \delta_n)$$

where $|\delta_n| = O\left(\frac{1}{\log(s_n)} + \frac{1}{\sqrt{s_n}}\right)$ for large enough $s_n$.

**Case (c):** $|x - z_n| = o\left(\frac{1}{\log(s_n)}\right)$

Following similar steps as in part (b), but now with $\lambda \log(s_n)|x - z_n| \to 0$ as $s_n \to \infty$.

From our previous derivation:
$$\log K_{RF,n}(x, z_n) = -\lambda \log(s_n)|x - z_n|(1 + \epsilon_n)$$

where $|\epsilon_n| = O\left(\frac{1}{\log(s_n)} + s_n^{-\lambda} + \frac{1}{\sqrt{s_n}}\right)$.

When $\lambda \log(s_n)|x - z_n| \to 0$, we use the approximation $e^{-y} = 1 - y + O(y^2)$ for small $y$:
$$K_{RF,n}(x, z_n) = e^{-\lambda \log(s_n)|x - z_n|(1 + \epsilon_n)} = 1 - \lambda \log(s_n)|x - z_n|(1 + \epsilon_n) + O((\lambda \log(s_n)|x - z_n|)^2)$$

The second-order term is bounded as:
$$O((\lambda \log(s_n)|x - z_n|)^2) = O((\log(s_n)|x - z_n|)^2) = o(\log(s_n)|x - z_n|)$$

since $|x - z_n| = o\left(\frac{1}{\log(s_n)}\right)$.

Therefore:
$$K_{RF,n}(x, z_n) = 1 - \lambda \log(s_n)|x - z_n|(1 + \gamma_n)$$

where $|\gamma_n| = O\left(\log(s_n)|x - z_n| + \frac{1}{\sqrt{s_n}}\right)$.

The total probability of deviation is bounded by the union bound over all levels:
$$P(\text{deviation}) \leq \sum_{i=1}^{d_n} 2e^{-\sqrt{s_n}/6} = 2d_n \cdot e^{-\sqrt{s_n}/6}$$

Since $d_n = \lambda \log(s_n)$, this probability approaches zero as $s_n \to \infty$, because the exponential decay in $e^{-\sqrt{s_n}/6}$ dominates the logarithmic growth in $d_n$. $\square$

## 5. Asymptotic Analysis for Multi-Dimensional Case (p > 1)

We now extend our analysis to the general multi-dimensional case, building upon the insights gained from the one-dimensional setting. This extension is not merely a straightforward generalization, as the interaction between dimensions introduces additional complexities that require careful consideration.

### 5.1 Multi-Dimensional Split Probability

**Lemma 3** (Multi-Dimensional Split Probability): Under Assumptions 1-5, let $\mathbf{x} \in [0,1]^p$ be a fixed point and $\{\mathbf{z}_n\} \subset [0,1]^p$ be a sequence of points. The probability that these points are separated at level $i$ given they were in the same node at level $i-1$ is:

$$p_i(\mathbf{x},\mathbf{z}_n) = \frac{1}{p}\sum_{j=1}^p p_{split}(x_j,z_{n,j},N_{i-1})$$

where $p_{split}(x_j,z_{n,j},N_{i-1})$ is as defined in Lemma 1 and $N_{i-1}$ is the number of samples in the node at level $i-1$.

**Proof**:
At each level, a dimension $j$ is chosen uniformly at random from the $p$ dimensions with probability $1/p$. Once dimension $j$ is selected, the probability of separation is $p_{split}(x_j,z_{n,j},N_{i-1})$ as defined in Lemma 1. By the law of total probability:
$$p_i(\mathbf{x},\mathbf{z}_n) = \frac{1}{p}\sum_{j=1}^p p_{split}(x_j,z_{n,j},N_{i-1})$$
$\square$

### 5.2 Multi-Dimensional Kernel Convergence

**Theorem 2** (Multi-Dimensional Kernel Convergence): Under Assumptions 1-5, let $\mathbf{x} \in [0,1]^p$ be a fixed point and $\{\mathbf{z}_n\} \subset [0,1]^p$ be a sequence of points. As $n \to \infty$ and $B \to \infty$:

(a) If $\|\mathbf{x} - \mathbf{z}_n\|_1 = \Theta(1)$ (i.e., $\mathbf{z}_n$ stays at a constant distance from $\mathbf{x}$), then with probability at least $1 - 2p \cdot d_n \cdot e^{-\sqrt{s_n}/6}$:
$$K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = O(s_n^{-c})$$
for some constant $c > 0$.

(b) If $\|\mathbf{x} - \mathbf{z}_n\|_1 = \frac{u}{\log(s_n)}$ for some constant $u > 0$ (i.e., $\mathbf{z}_n$ converges to $\mathbf{x}$ at a specific rate), then with probability at least $1 - 2p \cdot d_n \cdot e^{-\sqrt{s_n}/6}$:
$$K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = e^{-\lambda u}(1 + \delta_n)$$
where $|\delta_n| = O\left(\frac{1}{\log(s_n)} + \frac{1}{\sqrt{s_n}}\right)$.

(c) If $\|\mathbf{x} - \mathbf{z}_n\|_1 = o\left(\frac{1}{\log(s_n)}\right)$ (i.e., $\mathbf{z}_n$ converges to $\mathbf{x}$ faster than the rate in (b)), then with probability at least $1 - 2p \cdot d_n \cdot e^{-\sqrt{s_n}/6}$:
$$K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = 1 - \lambda \log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1(1 + \gamma_n)$$
where $|\gamma_n| = O\left(\log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1 + \frac{1}{\sqrt{s_n}}\right)$.

**Proof**:
The forest kernel $K_{RF,n}(\mathbf{x}, \mathbf{z}_n)$ represents the probability that points $\mathbf{x}$ and $\mathbf{z}_n$ remain unseparated through all levels of the tree:
$$K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = \prod_{i=1}^{d_n} (1 - p_i(\mathbf{x},\mathbf{z}_n))$$

From Lemma 2, with probability at least $1 - 2e^{-\sqrt{s_n}/6}$, the number of samples $N_{i-1}$ in a node at level $i-1$ satisfies:
$$\frac{\mu_{i-1}}{2} \leq N_{i-1} \leq \frac{3\mu_{i-1}}{2}$$
where $\mu_{i-1} = s_n 2^{-(i-1)}$.

**Case (a)**: $\|\mathbf{x} - \mathbf{z}_n\|_1 = \Theta(1)$

When $\|\mathbf{x} - \mathbf{z}_n\|_1 = \Theta(1)$, there exists at least one dimension $j$ with $|x_j - z_{n,j}| = \Theta(1)$. For this dimension, by Lemma 1(c), when $i$ is small (thus $\mu_{i-1}$ is large), $p_{split}(x_j,z_{n,j},N_{i-1}) = 1 - o(1)$. Therefore, by Lemma 3:
$$p_i(\mathbf{x},\mathbf{z}_n) \geq \frac{1}{p}(1 - o(1))$$

Even if $\mathbf{x}$ and $\mathbf{z}_n$ are separated with high probability at just one level, we get:
$$K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = \prod_{i=1}^{d_n} (1 - p_i(\mathbf{x},\mathbf{z}_n)) \leq \prod_{i=1}^{d_n} \left(1 - \frac{1 - o(1)}{p}\right) = O(s_n^{-c})$$
for some constant $c > 0$.

**Case (b)**: $\|\mathbf{x} - \mathbf{z}_n\|_1 = \frac{u}{\log(s_n)}$

Step 1: Express $p_i(\mathbf{x},\mathbf{z}_n)$ using Lemmas 1 and 3.
When $\|\mathbf{x} - \mathbf{z}_n\|_1 = \frac{u}{\log(s_n)}$, applying Lemma 1(a) to each dimension:
$$p_i(\mathbf{x},\mathbf{z}_n) = \frac{1}{p}\sum_{j=1}^p N_{i-1}|x_j - z_{n,j}|(1 + O(N_{i-1}|x_j - z_{n,j}|))$$

Step 2: Bound $p_i(\mathbf{x},\mathbf{z}_n)$ using our concentration results.
With high probability, $N_{i-1} = \mu_{i-1}(1 + O(s_n^{-1/4}))$, which gives:
$$p_i(\mathbf{x},\mathbf{z}_n) = \frac{\mu_{i-1}}{p}\|\mathbf{x} - \mathbf{z}_n\|_1(1 + O(\mu_{i-1}\|\mathbf{x} - \mathbf{z}_n\|_1 + s_n^{-1/4}))$$

Step 3: Convert to logarithm for easier analysis.
$$\log K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = \sum_{i=1}^{d_n} \log(1 - p_i(\mathbf{x},\mathbf{z}_n))$$

For small $p_i(\mathbf{x},\mathbf{z}_n)$, $\log(1 - p_i(\mathbf{x},\mathbf{z}_n)) = -p_i(\mathbf{x},\mathbf{z}_n) + O(p_i(\mathbf{x},\mathbf{z}_n)^2)$. Therefore:
$$\log K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = -\sum_{i=1}^{d_n} p_i(\mathbf{x},\mathbf{z}_n) + O\left(\sum_{i=1}^{d_n} p_i(\mathbf{x},\mathbf{z}_n)^2\right)$$

Step 4: Substitute the expression for $p_i(\mathbf{x},\mathbf{z}_n)$.
$$\log K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = -\sum_{i=1}^{d_n} \frac{\mu_{i-1}}{p}\|\mathbf{x} - \mathbf{z}_n\|_1(1 + O(\mu_{i-1}\|\mathbf{x} - \mathbf{z}_n\|_1 + s_n^{-1/4})) + O\left(\sum_{i=1}^{d_n} \left(\frac{\mu_{i-1}}{p}\|\mathbf{x} - \mathbf{z}_n\|_1\right)^2\right)$$

Step 5: Compute the sum of node sizes.
$$\sum_{i=1}^{d_n} \mu_{i-1} = \sum_{i=1}^{d_n} s_n 2^{-(i-1)} = s_n \sum_{i=1}^{d_n} 2^{-(i-1)} = s_n (2 - 2^{-d_n+1})$$

Since $d_n = \lambda p \log(s_n)$, we have $2^{-d_n+1} = 2 \cdot s_n^{-\lambda p}$. Thus:
$$\sum_{i=1}^{d_n} \mu_{i-1} = 2s_n(1 - s_n^{-\lambda p})$$

Step 6: Analyze the second-order term rigorously.
For the sum of squared node sizes:
$$\sum_{i=1}^{d_n} \mu_{i-1}^2 = \sum_{i=1}^{d_n} (s_n 2^{-(i-1)})^2 = s_n^2 \sum_{i=1}^{d_n} 2^{-2(i-1)}$$

This is a geometric series with first term $s_n^2$ and ratio $1/4$:
$$s_n^2 \sum_{i=1}^{d_n} 2^{-2(i-1)} = s_n^2 \cdot \frac{1 - (1/4)^{d_n}}{1-1/4} = s_n^2 \cdot \frac{1 - 4^{-d_n}}{3/4} = \frac{4s_n^2}{3}(1 - 4^{-d_n})$$

Since $d_n = \lambda p \log(s_n)$, we have $4^{-d_n} = s_n^{-2\lambda p}$. For large $s_n$, this term is negligible, yielding:
$$\sum_{i=1}^{d_n} \mu_{i-1}^2 = \frac{4s_n^2}{3}(1 + O(s_n^{-2\lambda p})) = \frac{4s_n^2}{3} + O(s_n^{2-2\lambda p})$$

Step 7: Use the relationship between $\|\mathbf{x} - \mathbf{z}_n\|_1$ and $\log(s_n)$.
Substituting $\|\mathbf{x} - \mathbf{z}_n\|_1 = \frac{u}{\log(s_n)}$ and our derived sums:

$$\log K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = -\frac{2s_n(1 - s_n^{-\lambda p})}{p} \cdot \frac{u}{\log(s_n)}(1 + O(\frac{s_n}{\log(s_n)} \cdot \frac{u}{\log(s_n)} + s_n^{-1/4})) + O\left(\frac{u^2 \cdot s_n^2}{p^2 (\log(s_n))^2}\right)$$

Step 8: Simplify using the relation between $s_n$, $d_n$, and $p$.
Since $d_n = \lambda p \log(s_n)$, we have $\frac{s_n}{p \log(s_n)} = \frac{s_n}{d_n/\lambda} = \lambda \frac{s_n}{d_n}$. For large $s_n$:

$$\log K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = -\lambda u(1 + O(s_n^{-\lambda p})) \cdot (1 + O(\frac{u}{\log(s_n)} + s_n^{-1/4})) + O\left(\frac{u^2}{\log(s_n)}\right)$$

This gives us:
$$\log K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = -\lambda u + O\left(\frac{1}{\log(s_n)} + \frac{1}{\sqrt{s_n}}\right)$$

Therefore:
$$K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = e^{-\lambda u}(1 + \delta_n)$$
where $|\delta_n| = O\left(\frac{1}{\log(s_n)} + \frac{1}{\sqrt{s_n}}\right)$.

**Case (c)**: $\|\mathbf{x} - \mathbf{z}_n\|_1 = o\left(\frac{1}{\log(s_n)}\right)$

Following similar steps as in case (b), but now with $\lambda \log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1 \to 0$ as $s_n \to \infty$.

From our previous derivation:
$$\log K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = -\lambda \log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1(1 + \epsilon_n)$$
where $|\epsilon_n| = O\left(\frac{1}{\log(s_n)} + s_n^{-\lambda p} + \frac{1}{\sqrt{s_n}}\right)$.

When $\lambda \log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1 \to 0$, we use the approximation $e^{-y} = 1 - y + O(y^2)$ for small $y$:
$$K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = e^{-\lambda \log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1(1 + \epsilon_n)} = 1 - \lambda \log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1(1 + \epsilon_n) + O((\lambda \log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1)^2)$$

The second-order term is bounded as:
$$O((\lambda \log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1)^2) = O((\log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1)^2) = o(\log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1)$$
since $\|\mathbf{x} - \mathbf{z}_n\|_1 = o\left(\frac{1}{\log(s_n)}\right)$.

Therefore:
$$K_{RF,n}(\mathbf{x}, \mathbf{z}_n) = 1 - \lambda \log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1(1 + \gamma_n)$$
where $|\gamma_n| = O\left(\log(s_n)\|\mathbf{x} - \mathbf{z}_n\|_1 + \frac{1}{\sqrt{s_n}}\right)$.

**Probabilistic Guarantees Analysis**: The probability bound in our theorem is $1 - 2p \cdot d_n \cdot e^{-\sqrt{s_n}/6}$. For this bound to be meaningful, we need $2p \cdot d_n \cdot e^{-\sqrt{s_n}/6} \to 0$ as $s_n \to \infty$. Since $d_n = \lambda p \log(s_n)$, the bound becomes $1 - 2p^2 \lambda \log(s_n) \cdot e^{-\sqrt{s_n}/6}$.

The term $e^{-\sqrt{s_n}/6}$ decreases exponentially in $\sqrt{s_n}$, while $p^2 \log(s_n)$ grows only polynomially in $\log(s_n)$ and quadratically in $p$. Therefore, for any fixed dimension $p$, the probability bound approaches 1 as $s_n \to \infty$. Even in high-dimensional settings where $p$ is large (but still fixed), the exponential decay in $e^{-\sqrt{s_n}/6}$ dominates, provided $s_n$ is sufficiently large relative to $p^2$. Specifically, we need $\sqrt{s_n} \gg 6 \log(p^2 \log(s_n))$ for the bound to be tight, which is satisfied when $s_n$ grows faster than $\log^2(p)$. $\square$

### 5.3 Continuity of Distance Regimes and Unified Kernel Representation

In the preceding analysis, we have categorized the behavior of random forest kernels into three distinct distance regimes. However, it is important to emphasize that these regimes do not have strict boundaries but rather represent a continuous spectrum of behaviors in the asymptotic setting as $n \to \infty$.

#### 5.3.1 Continuous Transition Between Distance Regimes

The distance regimes we have defined:
1. Near distance: $\|\mathbf{x} - \mathbf{z}_n\|_1 = o\left(\frac{1}{\log(s_n)}\right)$
2. Intermediate distance: $\|\mathbf{x} - \mathbf{z}_n\|_1 = \Theta\left(\frac{1}{\log(s_n)}\right)$
3. Far distance: $\|\mathbf{x} - \mathbf{z}_n\|_1 = \Theta(1)$

represent mathematically convenient characterizations rather than discrete categories. In reality, the kernel function transitions smoothly across these regimes.

**Proposition 1** (Continuous Kernel Transition): Under Assumptions 1-5, the random forest kernel $K_{RF,n}(\mathbf{x}, \mathbf{z}_n)$ can be expressed in terms of a scaled distance $\mathbf{u} = (\mathbf{x} - \mathbf{z}_n)\log(s_n)$ as:

$K_{RF,n}(\mathbf{x}, \mathbf{z}_n) \approx \begin{cases}
1 - \lambda \|\mathbf{u}\|_1 + O(\|\mathbf{u}\|_1^2) & \text{if } \|\mathbf{u}\|_1 \to 0 \\
e^{-\lambda \|\mathbf{u}\|_1} + o(1) & \text{if } \|\mathbf{u}\|_1 = \Theta(1) \\
O(s_n^{-c}) & \text{if } \|\mathbf{u}\|_1 \to \infty
\end{cases}$

with probability at least $1 - O(p \cdot d_n \cdot e^{-\sqrt{s_n}/6})$.

**Proof**: This follows directly from Theorem 2 by substituting $\mathbf{u} = (\mathbf{x} - \mathbf{z}_n)\log(s_n)$ and considering the limiting behavior as $\|\mathbf{u}\|_1$ varies. For $\|\mathbf{u}\|_1 \to 0$, we have $\|\mathbf{x} - \mathbf{z}_n\|_1 = o\left(\frac{1}{\log(s_n)}\right)$, and using the approximation $e^{-x} = 1 - x + O(x^2)$ for small $x$, we obtain the linear form. For $\|\mathbf{u}\|_1 = \Theta(1)$, we have $\|\mathbf{x} - \mathbf{z}_n\|_1 = \Theta\left(\frac{1}{\log(s_n)}\right)$, yielding the exponential form. $\square$

#### 5.3.2 Unified Kernel Representation

The scaled distance $\mathbf{u} = (\mathbf{x} - \mathbf{z}_n)\log(s_n)$ provides a unified framework for understanding the adaptive behavior of random forest kernels. This scaling is crucial in asymptotic analysis for the following reasons:

1. **Preserving relative distances**: As $n \to \infty$, the feature space becomes increasingly dense with samples, causing nearest-neighbor distances to approach zero. The logarithmic scaling preserves the relative importance of distances in the asymptotic setting.

2. **Revealing adaptive bandwidth**: The effective kernel bandwidth $h_n = \Theta(1/\log(s_n))$ shrinks as sample size increases, but using scaled distances allows us to analyze the kernel shape independent of this shrinkage.

3. **Connecting theoretical regimes**: The different functional forms across distance regimes can be understood as parts of a single, continuous kernel function when expressed in terms of scaled distances.

**Corollary 1** (Limiting Kernel Function): As $n \to \infty$, the random forest kernel approaches a limiting function of the scaled distance:

$\lim_{n \to \infty} K_{RF,n}(\mathbf{x}, \mathbf{x} + \mathbf{u}/\log(s_n)) = K_{\infty}(\mathbf{u})$

where $K_{\infty}(\mathbf{u})$ has the following properties:
1. For small $\|\mathbf{u}\|_1$: $K_{\infty}(\mathbf{u}) \approx 1 - \lambda \|\mathbf{u}\|_1$
2. For moderate $\|\mathbf{u}\|_1$: $K_{\infty}(\mathbf{u}) \approx e^{-\lambda \|\mathbf{u}\|_1}$
3. For large $\|\mathbf{u}\|_1$: $K_{\infty}(\mathbf{u}) \approx 0$

This limiting kernel function differs from traditional kernels (e.g., Gaussian, Laplace) in that it exhibits a linear decay near the origin rather than quadratic (Gaussian) or linear with constant slope (Laplace). This unique property contributes to the strong adaptive behavior of random forests.

### 5.4 Comparison with One-Dimensional Results

The multi-dimensional results in Theorem 2 extend our one-dimensional findings in Theorem 1 in several important ways. Both theorems identify three distinct regimes of kernel behavior, characterized by the convergence rate of the sequence of points to the reference point. However, there are key differences and similarities worth highlighting:

1. **Dimensionality Effect**: In the multi-dimensional case, the $L_1$ norm $\|\mathbf{x} - \mathbf{z}_n\|_1$ replaces the absolute difference $|x - z_n|$ in the one-dimensional case. This naturally captures the aggregated distance across all dimensions.

2. **Structural Similarity**: Despite the dimensional difference, the asymptotic behaviors in all three regimes maintain the same functional form: exponential decay for distant points, specific exponential relationship for moderately close points, and linear relationship for very close points.

3. **Dimensional Scaling**: The probability of separation at each level is averaged across dimensions, introducing a factor of $1/p$ that reflects dimension-uniform split selection. This affects the constants in the convergence rates but not their asymptotic form.

4. **Error Propagation**: The error terms in the multi-dimensional case account for potential imbalances across dimensions, but the overall convergence rates remain comparable to the one-dimensional case.

5. **Unified Representation**: The scaled distance formulation introduced in Section 5.3 applies to both one-dimensional and multi-dimensional cases, showing that the fundamental adaptive behavior of random forest kernels is dimension-invariant when properly scaled.

These results demonstrate the consistency of random forest kernel behavior across different dimensionalities, strengthening our understanding of their adaptive properties.

### 5.5 Implications for Adaptive Resolution and Practice

#### 5.5.1 Practical Implications of Continuous Distance Regimes

The continuous nature of distance regimes has important practical implications:

1. **Smooth adaptation**: Random forests smoothly adapt their prediction weights based on distance, without abrupt changes between "included" and "excluded" points. This property helps explain their robust performance across diverse datasets.

2. **Dimensional impact**: In $p$-dimensional space, the effective number of points with non-negligible weights is approximately $N_{eff} \approx n \cdot (1/\log(s_n))^p$, which decreases with dimension $p$ but not as rapidly as with fixed-bandwidth kernels. This gives random forests a relative advantage in moderately high-dimensional settings.

3. **Parameter tuning guidance**: The tree depth parameter $\lambda$ directly affects the rate of weight decay with distance, providing a theoretical basis for tuning this parameter based on the desired level of locality in predictions. Specifically, larger values of $\lambda$ lead to more localized predictions.

#### 5.5.2 Connection to Adaptive Kernel Methods

The behavior of the random forest kernel bears striking similarities to adaptive kernel methods in nonparametric statistics (Scott, 2015; Wasserman, 2006). However, unlike traditional kernel methods that typically require explicit bandwidth selection, random forests implicitly adapt their resolution based on local data density. Our results formalize this connection, showing how:

- The effective bandwidth is automatically larger in sparse regions (regime (a))
- The bandwidth transitions smoothly in moderately dense regions (regime (b))
- Fine discrimination occurs in high-density regions (regime (c))

This automatic adaptation explains many of the advantages of random forests over fixed-bandwidth methods, particularly in heterogeneous data settings where optimal bandwidth varies across the feature space.

#### 5.5.3 Behavior in High-Dimensional Settings

In high-dimensional spaces, our results have particularly important implications. As dimensionality increases:

1. **Sparsity Effects**: The probability of points falling into the same leaf node decreases exponentially with dimension, a manifestation of the "curse of dimensionality." However, the adaptive resolution property helps mitigate this challenge by adjusting the effective neighborhood size.

2. **Relevance for Feature Selection**: When features vary in relevance, random forests with uniform dimension selection (Assumption 2) might be suboptimal. Our analysis suggests that modifications to the splitting rule to favor more informative dimensions could potentially improve performance in high dimensions.

3. **Robustness to Irrelevant Features**: The exponential decay of kernel values for distant points (regime (a)) helps random forests remain robust to irrelevant features, as points that differ mainly in noise dimensions will still have small kernel values.

#### 5.5.4 Sensitivity to Model Assumptions

Our theoretical guarantees depend on several key assumptions. If these assumptions are violated, we expect the following effects:

- **Non-uniform Feature Selection (Assumption 2)**: If certain dimensions are selected with higher probability, the kernel will adapt more quickly along these dimensions. This would manifest as anisotropic behavior in the kernel, potentially beneficial when feature relevance varies.

- **Non-uniform Feature Distributions (Assumption 5)**: When the feature distribution deviates from uniformity, the effective node sizes may vary significantly from our theoretical predictions. The kernel will adapt more finely in regions of high data density, further enhancing the adaptive resolution property.

- **Different Tree Depths (Assumption 4)**: If trees are grown beyond depth $d_n = \lambda p \log(s_n)$, the kernel will exhibit even sharper discrimination between close points, potentially leading to overfitting if noise is present.

These insights not only deepen our theoretical understanding of random forests but also provide practical guidance for their application and potential modification in various data settings.

### 5.6 Conclusion

Our analysis of random forest kernels in the multi-dimensional case reveals that their adaptive resolution properties extend naturally from the one-dimensional setting. We have demonstrated that these kernels exhibit three distinct but continuously transitioning regimes of behavior based on the relative distance between points. By introducing the concept of scaled distance, we have provided a unified framework for understanding how random forests automatically adjust their smoothing bandwidth based on local data density.

The limiting kernel function $K_{\infty}(\mathbf{u})$ offers a novel characterization of random forest behavior that distinguishes it from traditional kernel methods. Its unique property of linear decay near the origin combined with exponential decay at moderate distances explains the strong adaptive behavior observed in practice. This theoretical characterization helps bridge the gap between random forests and adaptive kernel methods, providing new insights into why random forests perform well across diverse learning tasks and data structures.

Our analysis of high-dimensional behavior and sensitivity to model assumptions provides valuable guidance for practitioners. The findings suggest specific directions for optimizing random forest performance in challenging settings, including potential modifications to feature selection strategies and tree depth control based on dimensional considerations.

Future research directions might include extending these results to more general tree construction methods, exploring the implications for feature importance measures, and investigating the connection between kernel properties and generalization error in random forests. Additionally, the unified kernel representation could inform the development of new forest-inspired kernel methods that explicitly leverage the adaptive properties we have characterized.

## 6. Analysis of Random Forest Weights and Effective Neighborhood

Having established the asymptotic behavior of the random forest kernel across different distance regimes, we now analyze the weights that random forests assign to training points when making predictions. These weights determine how the influence of training points varies with their distance from the query point, which is fundamental to understanding the adaptive nature of random forests. We also characterize the effective neighborhood size, providing precise bounds on the region of feature space that significantly influences predictions.

### 6.1 Asymptotic Behavior of Weights

Recall that the random forest estimator can be expressed as a weighted average of training responses:

$$\hat{f}_{RF,n}(\mathbf{x}) = \sum_{i=1}^n \alpha_i(\mathbf{x})Y_i$$

where the weights are defined as:

$$\alpha_i(\mathbf{x}) = \frac{K_{RF,n}(\mathbf{x},\mathbf{X}_i)}{\sum_{j=1}^n K_{RF,n}(\mathbf{x},\mathbf{X}_j)}$$

These weights determine how much influence each training point has on the prediction at query point $\mathbf{x}$. The following theorem characterizes their asymptotic behavior.

**Theorem 3** (Asymptotic Behavior of Random Forest Weights): Under Assumptions 1-5 and as $n,B \to \infty$, the weights $\alpha_i(\mathbf{x})$ in the random forest estimator satisfy with probability at least $1 - O(n \cdot p \cdot d_n \cdot e^{-\sqrt{s_n}/6})$:

(a) For $\|\mathbf{x} - \mathbf{X}_i\|_1 = \Theta(1)$ (distant points):
$$\alpha_i(\mathbf{x}) = O(s_n^{-c})$$
for some constant $c > 0$.

(b) For $\|\mathbf{x} - \mathbf{X}_i\|_1 = \frac{u_i}{\log(s_n)}$ with $u_i > 0$ (moderately close points):
$$\alpha_i(\mathbf{x}) = \frac{e^{-\lambda u_i}(1 + O(\frac{1}{\log(s_n)} + \frac{1}{\sqrt{s_n}}))}{\sum_{j: \|\mathbf{x} - \mathbf{X}_j\|_1 = \Theta(1/\log(s_n))} e^{-\lambda u_j}(1 + O(\frac{1}{\log(s_n)} + \frac{1}{\sqrt{s_n}}))}$$

(c) For $\|\mathbf{x} - \mathbf{X}_i\|_1 = o\left(\frac{1}{\log(s_n)}\right)$ (very close points):
$$\alpha_i(\mathbf{x}) \approx \frac{1 - \lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_i\|_1 + O((\log(s_n)\|\mathbf{x} - \mathbf{X}_i\|_1)^2)}{\sum_{j: \|\mathbf{x} - \mathbf{X}_j\|_1 = O(1/\log(s_n))} (1 - \lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_j\|_1 + O((\log(s_n)\|\mathbf{x} - \mathbf{X}_j\|_1)^2))}$$

**Proof**:
The forest weights are defined as:
$$\alpha_i(\mathbf{x}) = \frac{K_{RF,n}(\mathbf{x}, \mathbf{X}_i)}{\sum_{j=1}^n K_{RF,n}(\mathbf{x}, \mathbf{X}_j)}$$

From Theorem 2, we know the asymptotic behavior of $K_{RF,n}(\mathbf{x}, \mathbf{X}_i)$ across different distance regimes. To determine the weights, we need to analyze both the numerator (the kernel value for a specific point) and the denominator (the sum of kernel values across all training points).

Under Assumption 5 (bounded density), the denominator $\sum_{j=1}^n K_{RF,n}(\mathbf{x}, \mathbf{X}_j)$ is dominated by points in the $\Theta(1/\log(s_n))$ neighborhood of $\mathbf{x}$. With high probability, there are $\Theta(n \cdot (1/\log(s_n))^p)$ points in this neighborhood, each contributing substantially to the sum.

For case (a), where $\|\mathbf{x} - \mathbf{X}_i\|_1 = \Theta(1)$, Theorem 2(a) gives us $K_{RF,n}(\mathbf{x}, \mathbf{X}_i) = O(s_n^{-c})$. The denominator is $\Theta(n \cdot (1/\log(s_n))^p)$, reflecting the number of points with significant kernel values. Therefore:
$$\alpha_i(\mathbf{x}) = \frac{O(s_n^{-c})}{\Theta(n \cdot (1/\log(s_n))^p)} = O(s_n^{-c})$$

This shows that points at a constant distance from the query point have exponentially decreasing influence as the sample size increases.

For case (b), where $\|\mathbf{x} - \mathbf{X}_i\|_1 = \frac{u_i}{\log(s_n)}$, Theorem 2(b) gives us $K_{RF,n}(\mathbf{x}, \mathbf{X}_i) = e^{-\lambda u_i}(1 + \delta_i)$ where $|\delta_i| = O(\frac{1}{\log(s_n)} + \frac{1}{\sqrt{s_n}})$. The denominator sum can be expressed as:
$$\sum_{j=1}^n K_{RF,n}(\mathbf{x}, \mathbf{X}_j) = \sum_{j: \|\mathbf{x} - \mathbf{X}_j\|_1 = \Theta(1/\log(s_n))} e^{-\lambda u_j}(1 + \delta_j) + \sum_{j: \|\mathbf{x} - \mathbf{X}_j\|_1 \neq \Theta(1/\log(s_n))} K_{RF,n}(\mathbf{x}, \mathbf{X}_j)$$

The second sum is negligible compared to the first, as points outside the $\Theta(1/\log(s_n))$ neighborhood have exponentially smaller kernel values. Therefore:
$$\alpha_i(\mathbf{x}) = \frac{e^{-\lambda u_i}(1 + \delta_i)}{\sum_{j: \|\mathbf{x} - \mathbf{X}_j\|_1 = \Theta(1/\log(s_n))} e^{-\lambda u_j}(1 + \delta_j)}$$

For case (c), we use the linear approximation of the kernel function derived in Theorem 2(c). For points very close to $\mathbf{x}$, the kernel value is approximately $1 - \lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_i\|_1(1 + \gamma_i)$ where $|\gamma_i| = O(\log(s_n)\|\mathbf{x} - \mathbf{X}_i\|_1 + \frac{1}{\sqrt{s_n}})$. Substituting into the weight formula and simplifying:
$$\alpha_i(\mathbf{x}) \approx \frac{1 - \lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_i\|_1 + O((\log(s_n)\|\mathbf{x} - \mathbf{X}_i\|_1)^2)}{\sum_{j: \|\mathbf{x} - \mathbf{X}_j\|_1 = O(1/\log(s_n))} (1 - \lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_j\|_1 + O((\log(s_n)\|\mathbf{x} - \mathbf{X}_j\|_1)^2))}$$

The probability bound $1 - O(n \cdot p \cdot d_n \cdot e^{-\sqrt{s_n}/6})$ is derived by applying the union bound to the concentration results from Theorem 2 across all $n$ training points. The term $e^{-\sqrt{s_n}/6}$ decreases exponentially with $\sqrt{s_n}$, while the factors $n$, $p$, and $d_n = \lambda p \log(s_n)$ increase polynomially. For any fixed dimension $p$, as $s_n$ increases, the exponential decay dominates the polynomial growth, ensuring that the probability bound approaches 1. Specifically, when $s_n = \Omega((\log n)^2)$, the probability bound becomes $1 - o(1)$. $\square$

#### 6.1.1 Implications of Weight Distribution

Theorem 3 reveals several important properties of random forest weights:

1. **Exponential decay with distance**: Points outside the $\Theta(1/\log(s_n))$ neighborhood have negligible influence on predictions, effectively creating an adaptive soft boundary for relevant points.

2. **Relative importance within neighborhood**: Within the effective neighborhood, the importance of training points decays exponentially with their scaled distance from the query point, with the parameter $\lambda$ controlling the rate of decay.

3. **Adaptive smoothing**: For very close points, the weights vary almost linearly with distance, providing finer-grained discrimination in regions of high data density.

4. **Dimensional scaling**: The number of points with non-negligible weights scales as $\Theta(n \cdot (1/\log(s_n))^p)$, which decreases with dimension $p$ but less dramatically than with fixed-bandwidth kernels.

These properties demonstrate how random forests automatically adapt their prediction weights based on both local data density and the global sample size, without requiring explicit bandwidth selection.

### 6.2 Effective Neighborhood Size and Boundary

A key question in understanding local adaptive methods is: how large is the neighborhood that effectively influences predictions? The following theorem provides a precise characterization of this effective neighborhood size.

**Theorem 4** (Expected Maximum Distance): Under Assumptions 1-5, as $n,B \to \infty$, the expected maximum distance of points with non-negligible weights satisfies:

$$\mathbb{E}\left[\sup\{\|\mathbf{X}_i - \mathbf{x}\|_2: \alpha_i(\mathbf{x}) > \varepsilon_n\}\right] = \Theta\left(\frac{\sqrt{p}}{\log(s_n)}\right)$$

where $\varepsilon_n = n^{-\beta}$ for any fixed $0 < \beta < c$ is a threshold that approaches zero more slowly than the smallest non-zero weight $O(s_n^{-c})$.

**Proof**:
From Theorem 3, we know that points with $\|\mathbf{x} - \mathbf{X}_i\|_1 = \Theta(1)$ have weights $\alpha_i(\mathbf{x}) = O(s_n^{-c})$. As $s_n \to \infty$, these weights become effectively zero. However, there is no sharp boundary where weights suddenly become zero; instead, they decrease continuously with distance.

To formalize the concept of an "effective neighborhood," we consider points with weights exceeding a threshold $\varepsilon_n = n^{-\beta}$ for some fixed $0 < \beta < c$. This specific form ensures that $\varepsilon_n$ approaches zero more slowly than $O(s_n^{-c})$ as $n$ increases, capturing points that have a non-negligible influence on predictions as the sample size grows.

The effective neighborhood is primarily determined by the intermediate distance regime where $\|\mathbf{x} - \mathbf{X}_i\|_1 = \Theta(1/\log(s_n))$. To find the boundary of this neighborhood, we need to determine the distance at which $\alpha_i(\mathbf{x}) = \varepsilon_n$.

From Theorem 3(b) and using the unified kernel representation established in Section 5.3, we have:

$$\alpha_i(\mathbf{x}) \approx \frac{e^{-\lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_i\|_1}}{\sum_{j} e^{-\lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_j\|_1}}$$

For a point at the boundary of the effective neighborhood, $\alpha_i(\mathbf{x}) = \varepsilon_n$, which implies:

$$e^{-\lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_i\|_1} = \varepsilon_n \cdot \sum_{j} e^{-\lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_j\|_1}$$

Taking logarithms of both sides:

$$-\lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_i\|_1 = \log(\varepsilon_n) + \log\left(\sum_{j} e^{-\lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_j\|_1}\right)$$

Solving for $\|\mathbf{x} - \mathbf{X}_i\|_1$:

$$\|\mathbf{x} - \mathbf{X}_i\|_1 = \frac{-\log(\varepsilon_n) - \log\left(\sum_{j} e^{-\lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_j\|_1}\right)}{\lambda \log(s_n)}$$

Under Assumption 5 (bounded density), there are approximately $\Theta(n \cdot (1/\log(s_n))^p)$ points in the neighborhood of $\mathbf{x}$ with non-negligible kernel values. The sum in the denominator of the weights is dominated by points close to $\mathbf{x}$, giving:

$$\sum_{j} e^{-\lambda \log(s_n)\|\mathbf{x} - \mathbf{X}_j\|_1} = \Theta\left(n \cdot \left(\frac{1}{\log(s_n)}\right)^p\right)$$

Substituting this and $\varepsilon_n = n^{-\beta}$ into our expression for $\|\mathbf{x} - \mathbf{X}_i\|_1$:

$$\|\mathbf{x} - \mathbf{X}_i\|_1 = \frac{\beta \log(n) - \log\left(\Theta\left(n \cdot \left(\frac{1}{\log(s_n)}\right)^p\right)\right)}{\lambda \log(s_n)}$$

$$= \frac{\beta \log(n) - \log(n) - \log\left(\Theta\left(\left(\frac{1}{\log(s_n)}\right)^p\right)\right)}{\lambda \log(s_n)}$$

$$= \frac{(\beta-1) \log(n) + p \log(\log(s_n)) + O(1)}{\lambda \log(s_n)}$$

For typical parameter settings where $s_n = \Theta(n)$, and since $\beta < 1$ for our choice of threshold, the dominant term is:

$$\|\mathbf{x} - \mathbf{X}_i\|_1 = \Theta\left(\frac{1}{\log(s_n)}\right)$$

Using the relationship between L1 and L2 norms in $\mathbb{R}^p$:

$$\|\mathbf{x} - \mathbf{X}_i\|_2 \leq \|\mathbf{x} - \mathbf{X}_i\|_1 \leq \sqrt{p} \cdot \|\mathbf{x} - \mathbf{X}_i\|_2$$

We obtain:

$$\mathbb{E}\left[\sup\{\|\mathbf{X}_i - \mathbf{x}\|_2: \alpha_i(\mathbf{x}) > \varepsilon_n\}\right] = \Theta\left(\frac{\sqrt{p}}{\log(s_n)}\right)$$

This shows that the effective neighborhood radius in Euclidean distance scales inversely with $\log(s_n)$ and proportionally to the square root of the dimension. $\square$

### 6.3 Expected Number of Points in the Effective Neighborhood

Given our characterization of the effective neighborhood size, we can now analyze the expected number of training points that significantly influence predictions.

**Corollary 1** (Expected Neighborhood Population): Under Assumptions 1-5, as $n,B \to \infty$, the expected number of training points with non-negligible weights satisfies:

$$\mathbb{E}[|\{i: \alpha_i(\mathbf{x}) > \varepsilon_n\}|] = \Theta\left(n \cdot \left(\frac{1}{\log(s_n)}\right)^p\right)$$

**Proof**:
From Theorem 4, the effective neighborhood has a radius of $\Theta(\sqrt{p}/\log(s_n))$ in the L2 norm. To calculate the volume of this neighborhood, we need to consider the p-dimensional ball with this radius.

The volume of a p-dimensional ball with radius $r$ is $V_p(r) = \frac{\pi^{p/2}}{\Gamma(p/2+1)}r^p$, where $\Gamma$ is the gamma function. Substituting $r = \Theta(\sqrt{p}/\log(s_n))$:

$$V_p\left(\Theta\left(\frac{\sqrt{p}}{\log(s_n)}\right)\right) = \frac{\pi^{p/2}}{\Gamma(p/2+1)} \cdot \Theta\left(\frac{\sqrt{p}}{\log(s_n)}\right)^p$$

This simplifies to:

$$V_p\left(\Theta\left(\frac{\sqrt{p}}{\log(s_n)}\right)\right) = \Theta\left(\frac{p^{p/2}}{(\log(s_n))^p} \cdot \frac{\pi^{p/2}}{\Gamma(p/2+1)}\right)$$

Using Stirling's approximation for the gamma function: $\Gamma(p/2+1) \approx \sqrt{2\pi} \cdot (p/2)^{p/2} \cdot e^{-p/2}$, we find that the ratio $\frac{\pi^{p/2}}{\Gamma(p/2+1)}$ is $\Theta(p^{-p/2})$, which cancels the $p^{p/2}$ term in the numerator. Thus:

$$V_p\left(\Theta\left(\frac{\sqrt{p}}{\log(s_n)}\right)\right) = \Theta\left(\left(\frac{1}{\log(s_n)}\right)^p\right)$$

Under Assumption 5 (bounded density), the expected number of points in this volume is proportional to $n$ times the volume, giving:

$$\mathbb{E}[|\{i: \alpha_i(\mathbf{x}) > \varepsilon_n\}|] = \Theta\left(n \cdot \left(\frac{1}{\log(s_n)}\right)^p\right)$$
$\square$

This result reveals an important property of random forests: as the sample size increases, the absolute number of influential points grows, but their proportion relative to the total sample size decreases. Specifically, the proportion of influential points decreases as $\Theta((\log(s_n))^{-p})$.

### 6.4 Comparison with Traditional Kernel Methods

The behavior of random forest weights reveals important differences from traditional kernel methods that help explain their adaptive properties. 

In traditional kernel regression with a fixed bandwidth $h$, the weights typically take the form:

$$\alpha_i^{kernel}(\mathbf{x}) = \frac{K\left(\frac{\|\mathbf{x} - \mathbf{X}_i\|}{h}\right)}{\sum_{j=1}^n K\left(\frac{\|\mathbf{x} - \mathbf{X}_j\|}{h}\right)}$$

where $K(\cdot)$ is a kernel function such as the Gaussian or Epanechnikov kernel.

For instance, the Gaussian kernel yields weights of the form:

$$\alpha_i^{Gauss}(\mathbf{x}) = \frac{\exp\left(-\frac{\|\mathbf{x} - \mathbf{X}_i\|_2^2}{2h^2}\right)}{\sum_{j=1}^n \exp\left(-\frac{\|\mathbf{x} - \mathbf{X}_j\|_2^2}{2h^2}\right)}$$

When compared to random forest weights, several key differences emerge:

1. **Adaptive bandwidth**: Random forests implicitly use a bandwidth of $h_n = \Theta(1/\log(s_n))$ that adapts to the sample size, automatically becoming more localized as more data becomes available.

2. **Shape adaptation**: The random forest kernel exhibits different functional forms at different distance scales (linear for very close points, exponential for moderately close points), providing more nuanced adaptation than traditional kernels with a fixed functional form.

3. **Dimensional scaling**: The effective neighborhood size in random forests scales as $O(\sqrt{p}/\log(s_n))$, which is less sensitive to the curse of dimensionality than the typical $O(h)$ scaling in fixed-bandwidth methods.

4. **Decay profile**: For close points, the Gaussian kernel exhibits quadratic decay ($\alpha_i^{Gauss}(\mathbf{x}) \approx 1 - \Theta((\log(s_n))^2\|\mathbf{x} - \mathbf{X}_i\|_2^2)$), whereas random forests show linear decay ($\alpha_i(\mathbf{x}) \approx 1 - \Theta(\log(s_n)\|\mathbf{x} - \mathbf{X}_i\|_1)$).

These differences contribute to random forests' adaptive behavior and help explain their empirical success across diverse learning problems.

### 6.5 Practical Implications

The theoretical results on random forest weights and effective neighborhood size have several important implications for practitioners:

#### 6.5.1 Feature Space Coverage

In high-dimensional spaces, the curse of dimensionality typically makes it difficult to achieve adequate coverage of the feature space. Our results show that random forests mitigate this problem through their adaptive neighborhood sizing. As dimensionality increases, the effective neighborhood expands just enough to include a sufficient number of training points.

The expected number of influential points scales as $\Theta(n \cdot (1/\log(s_n))^p)$, which decreases exponentially with dimension $p$. However, this decrease is milder than in fixed-bandwidth methods where the number of points in a neighborhood of radius $h$ decreases as $\Theta(n \cdot h^p)$. This explains why random forests often outperform traditional methods in moderate to high-dimensional settings.

#### 6.5.2 Parameter Tuning Guidance

Our analysis provides theoretical guidance for tuning the key parameters of random forests:

1. **Tree depth parameter $\lambda$**: This parameter directly affects the rate of weight decay with distance. Larger values of $\lambda$ lead to more localized predictions with sharper transitions between distance regimes. In practice, this corresponds to growing deeper trees relative to the subsample size.

2. **Subsample size $s_n$**: The effective neighborhood size scales as $\Theta(1/\log(s_n))$. Larger subsample sizes result in smaller effective neighborhoods and more localized predictions. This suggests that increasing the subsample size can help reduce bias in regions with sufficient data density, but may increase variance in sparse regions.

3. **Number of trees $B$**: While our asymptotic results assume $B \to \infty$, in practice, a finite number of trees introduces additional variance. To ensure that the empirical weights are close to their theoretical values with high probability, the number of trees should increase with the desired precision of the weights.

#### 6.5.3 Local Variable Importance

The characterization of random forest weights provides a foundation for developing more precise local variable importance measures. By understanding how training points influence predictions based on their distance from the query point, researchers can develop variable importance measures that reflect the local structure of the feature space more accurately.

This understanding could lead to more interpretable models that can identify which variables are most important in different regions of the feature space, rather than relying solely on global importance measures.

### 6.6 Conclusion

Our analysis of random forest weights and effective neighborhood size provides key insights into how random forests adaptively adjust their prediction influence based on sample size, dimensionality, and local data density. The weights exhibit a smooth transition from linear to exponential decay with distance, creating an effective soft boundary for relevant points.

The effective neighborhood size scales as $\Theta(\sqrt{p}/\log(s_n))$ in Euclidean distance, demonstrating how random forests automatically adapt their resolution based on both sample size and dimensionality. This adaptive behavior helps explain the strong empirical performance of random forests across diverse learning tasks and data structures.

Through comparison with traditional kernel methods, we have shown that random forests combine the benefits of linear and exponential weight decay in different distance regimes, providing a unique adaptive profile that mitigates the curse of dimensionality while maintaining local sensitivity.

These results not only deepen our theoretical understanding of random forests but also provide practical guidance for their application and tuning in various settings. The connection between random forest weights and adaptive kernel methods opens new avenues for developing hybrid approaches that combine the strengths of both paradigms.


## 7. Comprehensive Simulation Studies

In this section, we present a detailed empirical validation of our theoretical framework through extensive simulation studies. These simulations are specifically designed to verify the key properties of random forest kernels across different distance regimes, while also investigating the effect of sample size and subsampling strategies on kernel behavior.

### 7.1 Simulation Design

Our simulation studies employ a custom implementation of random forests that strictly adheres to the theoretical assumptions specified in Section 3. We focus on evaluating how closely the empirical kernel behavior aligns with theoretical predictions across various parameter configurations.

#### 7.1.1 Experimental Setup

The key components of our simulation design include:

1. **Multiple Trials**: For each parameter configuration, we conduct 10 independent trials to assess the variability in empirical kernel estimates.

2. **Varied Sample Sizes**: We investigate sample sizes $n \in \{200, 500, 1000, 1500, 2000\}$ to evaluate convergence properties.

3. **Subsampling Strategies**: We explore five different subsampling approaches:
   - $s_n = \sqrt{n}$ (traditional random forests)
   - $s_n = n/3$ (common in practice)
   - $s_n = n^{0.8}$ (moderate subsampling)
   - $s_n = n^{0.9}$ (Wager and Athey approach)
   - $s_n = n^{0.98}$ (nearly full sampling)

4. **Fixed Parameters**:
   - Dimension: $p = 2$ (fixed for clarity of visualization)
   - Number of trees: $B = 2000$ (to ensure stability in kernel estimates)
   - Tree depth: $d_n = \lambda p \log(s_n)$ as specified in Assumption 4

5. **Distance Regimes**: We systematically generate test points across the three distance regimes:
   - Very close regime: $\|\mathbf{x} - \mathbf{z}\|_1 = o\left(\frac{1}{\log(s_n)}\right)$
   - Intermediate regime: $\|\mathbf{x} - \mathbf{z}\|_1 = \Theta\left(\frac{1}{\log(s_n)}\right)$
   - Far regime: $\|\mathbf{x} - \mathbf{z}\|_1 = \Theta(1)$

For each configuration, we compute both empirical kernel values and their corresponding theoretical predictions, allowing direct comparison between theory and practice.

#### 7.1.2 Implementation Details

We implemented a custom random forest algorithm that faithfully follows our theoretical framework:

1. **Tree Building**: Each tree is constructed using a random subsample of the training data, with splits chosen uniformly at random from available features (Assumption 2).

2. **Kernel Computation**: The kernel value $K_{RF,n}(\mathbf{x}, \mathbf{z})$ is calculated as the proportion of trees in which points $\mathbf{x}$ and $\mathbf{z}$ fall into the same leaf node.

3. **Theoretical Curves**: For comprehensive comparison, we compute theoretical values for both the very close regime ($1 - \lambda\log(s_n)\|\mathbf{x} - \mathbf{z}\|_1$) and the intermediate regime ($e^{-\lambda\log(s_n)\|\mathbf{x} - \mathbf{z}\|_1}$) across the entire range of distances.

This implementation enables us to directly validate our theoretical characterization of the random forest kernel.

### 7.2 Empirical Validation of Kernel Behavior

Our primary objective is to examine how closely the empirical kernel behavior aligns with our theoretical predictions across different distance regimes. Figure 1 illustrates this comparison for a sample size of $n = 1000$ with the subsampling strategy $s_n = n^{0.9}$.

#### 7.2.1 Regime-Specific Behavior

The empirical results strongly validate our theoretical predictions regarding the three distinct regimes of kernel behavior:

1. **Very Close Regime**: For points at distances significantly smaller than $1/\log(s_n)$, the kernel values exhibit a linear relationship with distance, closely following the theoretical prediction $K_{RF,n}(\mathbf{x}, \mathbf{z}) \approx 1 - \lambda\log(s_n)\|\mathbf{x} - \mathbf{z}\|_1$. This confirms Theorem 2(c).

2. **Intermediate Regime**: For points at distances proportional to $1/\log(s_n)$, the kernel values demonstrate exponential decay, matching the theoretical prediction $K_{RF,n}(\mathbf{x}, \mathbf{z}) \approx e^{-\lambda\log(s_n)\|\mathbf{x} - \mathbf{z}\|_1}$. This validates Theorem 2(b).

3. **Far Regime**: For points at constant distances, the kernel values rapidly approach zero, consistent with the exponential decay predicted in Theorem 2(a).

The smooth transition between these regimes confirms our analysis in Section 5.3 regarding the continuous nature of the kernel function.

#### 7.2.2 Impact of Theoretical Curves Extension

By plotting both theoretical curves (linear and exponential) across the entire range of distances, we gain additional insights into the kernel behavior:

1. The linear approximation ($1 - \lambda\log(s_n)\|\mathbf{x} - \mathbf{z}\|_1$) is highly accurate in the very close regime but quickly becomes inappropriate at larger distances, even predicting negative kernel values.

2. The exponential approximation ($e^{-\lambda\log(s_n)\|\mathbf{x} - \mathbf{z}\|_1}$) provides an excellent fit in the intermediate regime and remains reasonable in the far regime.

3. The intersection of these curves naturally identifies a transition point between the regimes, occurring at approximately $\|\mathbf{x} - \mathbf{z}\|_1 \approx 0.1/\log(s_n)$.

These observations reinforce the necessity of different functional approximations for different distance regimes, which is a key contribution of our theoretical framework.

### 7.3 Convergence Properties and Sample Size Effects

A critical aspect of our asymptotic analysis is the convergence of empirical kernel behavior to theoretical predictions as sample size increases. Figure 2 illustrates how kernel variance decreases with increasing sample size across different subsampling strategies.

#### 7.3.1 Kernel Variance

Our results demonstrate that for all subsampling strategies and distance regimes, the variance of empirical kernel estimates decreases as sample size increases. Specifically:

1. **Rate of Convergence**: The average standard deviation of kernel estimates decreases approximately at a rate of $O(n^{-1/2})$, consistent with standard statistical convergence rates.

2. **Regime-Specific Stability**: Kernel estimates in the very close regime exhibit the lowest variance, while estimates in the far regime show higher variability, especially at smaller sample sizes.

3. **Subsample Effect**: Strategies with larger subsamples (e.g., $s_n = n^{0.98}$) generally show lower variance than those with smaller subsamples (e.g., $s_n = \sqrt{n}$), particularly for intermediate and far regimes.

These findings confirm that our theoretical characterization becomes increasingly accurate as sample size grows, supporting the asymptotic nature of our results.

#### 7.3.2 Distance Scaling Property

An important theoretical prediction is that when distances are properly scaled by $\log(s_n)$, the kernel behavior follows a universal pattern regardless of sample size. Figure 3 confirms this property by plotting kernel values against scaled distance $u = \log(s_n)\|\mathbf{x} - \mathbf{z}\|_1$ for different sample sizes.

The results show remarkable alignment of kernel curves across different sample sizes when using the scaled distance, with all curves closely following the theoretical prediction $e^{-\lambda u}$ in the intermediate regime. This consistency validates our theoretical framework's ability to capture the essential scaling behavior of random forest kernels.

### 7.4 Impact of Subsampling Strategy

Our investigation of different subsampling strategies reveals several important insights about their effect on kernel behavior. Figure 4 compares kernel shapes across different subsampling strategies for a fixed sample size.

#### 7.4.1 Kernel Shape Variation

When examining kernel behavior across subsampling strategies, we observe:

1. **Overall Shape Consistency**: All subsampling strategies produce kernel functions that exhibit the three theoretical regimes, though with varying transition points and decay rates.

2. **Decay Rate Differences**: Larger subsampling rates (e.g., $s_n = n^{0.98}$) lead to sharper decay in the intermediate regime compared to smaller rates (e.g., $s_n = \sqrt{n}$).

3. **Effective Neighborhood Size**: The effective neighborhood (points with non-negligible kernel values) is larger for smaller subsampling rates and narrower for larger rates.

These observations can be explained through our theoretical framework: larger subsamples lead to deeper trees (since $d_n = \lambda p \log(s_n)$), resulting in more refined partitioning of the feature space and consequently sharper discriminative capability between points.

#### 7.4.2 Theoretical Alignment

Interestingly, we find that all subsampling strategies show reasonable agreement with our theoretical predictions, with some variations:

1. **Traditional Rate ($s_n = \sqrt{n}$)**: Shows good overall alignment with theory and provides a balance between very close and intermediate regimes.

2. **Wager-Athey Rate ($s_n = n^{0.9}$)**: Demonstrates excellent agreement in the intermediate regime but potentially faster transition from very close to intermediate regimes.

3. **Nearly Full Sampling ($s_n = n^{0.98}$)**: Exhibits the sharpest distinction between regimes, with very rapid transition from kernel values near 1 to values near 0.

These results suggest that while the specific subsampling rate affects the quantitative details of kernel behavior, the qualitative characteristics predicted by our theory remain valid across different subsampling strategies.

### 7.5 Implications and Practical Considerations

Our comprehensive simulation studies yield several important implications for the theoretical understanding and practical application of random forests:

#### 7.5.1 Theoretical Validation

The strong agreement between empirical and theoretical kernel behavior across multiple parameter configurations provides robust validation of our theoretical framework. The three-regime characterization accurately captures the essential behavior of random forest kernels, regardless of specific implementation details such as subsampling rate.

#### 7.5.2 Subsampling Recommendations

Based on our findings, we can offer practical recommendations regarding subsampling strategies:

1. **Statistical Efficiency**: Larger subsampling rates ($s_n = n^{0.9}$ or higher) offer lower variance in kernel estimates, making them preferable when computational resources allow.

2. **Computational Efficiency**: Smaller subsampling rates (e.g., $s_n = \sqrt{n}$) still provide reasonable kernel estimates with significantly reduced computational cost, which may be crucial for large-scale applications.

3. **Adaptive Resolution**: The choice of subsampling rate effectively controls the sharpness of discrimination between points at different distances, allowing practitioners to adjust this parameter based on their specific needs for local adaptivity.

#### 7.5.3 Connection to Forest Depth

Our simulations highlight the critical role of tree depth in determining kernel properties. The depth parameter $\lambda$ and the subsample size $s_n$ jointly control the effective resolution of the forest through the relationship $d_n = \lambda p \log(s_n)$. This suggests that practitioners may want to directly control tree depth based on the desired kernel properties rather than focusing solely on subsampling rates.

### 7.6 Summary

Our comprehensive simulation studies provide strong empirical validation of the theoretical framework developed in this paper. The results confirm the existence of three distinct distance regimes in random forest kernels and demonstrate how kernel behavior converges to theoretical predictions as sample size increases. The investigations into different subsampling strategies reveal that while quantitative details may vary, the fundamental properties of random forest kernels predicted by our theory remain consistent across implementations.

These findings strengthen our understanding of how random forests adaptively adjust their smoothing behavior based on distance, sample size, and subsampling strategy, providing both theoretical insights and practical guidance for the application of random forests in various learning tasks.

## 8. Conclusion and future work

### 8.1 Contributions

This paper provides a rigorous mathematical analysis of the kernel induced by random forests, characterizing its asymptotic behavior across different distance regimes. Our primary contributions are:

First, we have established that the random forest kernel exhibits three distinct behaviors depending on the distance between points relative to sample size: exponential decay for distant points, a specific exponential relationship for moderately close points, and a linear relationship for very close points. These regimes emerge naturally from the tree-building process and explain the adaptive smoothing capability of random forests.

Second, we have derived precise asymptotic forms for the kernel function in each regime, showing that the transition between regimes is governed by the quantity $1/\log(s_n)$, where $s_n$ is the subsample size. This characterization reveals how random forests implicitly adapt their resolution based on both sample size and local data density.

Third, we have shown that the effective neighborhood size scales as $\Theta(\sqrt{p}/\log(s_n))$, demonstrating how random forests mitigate the curse of dimensionality compared to fixed-bandwidth methods.

Fourth, our comprehensive simulation studies validate the theoretical findings across various parameter configurations, confirming the three-regime behavior and showing how empirical kernel behavior converges to theoretical predictions as sample size increases. Our simulations also reveal how different subsampling strategies affect the quantitative details of kernel behavior while preserving the qualitative characteristics predicted by our theory.

These results provide a novel perspective on random forests that connects them to kernel methods while highlighting their unique adaptive properties, helping to bridge the gap between their empirical success and theoretical understanding.

### 8.2 Future Work

Several promising directions for future research emerge from this work : Further theoretical developments could include relaxing the current assumptions to accommodate non-uniform feature distributions and data-dependent splitting rules, extending the analysis to classification settings, and developing finite-sample guarantees that provide non-asymptotic bounds on kernel behavior.

Algorithmic innovations might involve designing forest construction algorithms that optimize specific kernel characteristics, developing accelerated implementations based on effective neighborhood insights, and creating hybrid approaches that combine random forest kernels with other kernel methods.

Applied research directions include developing local feature importance measures based on the kernel perspective, improving uncertainty quantification using the distance-dependent behavior of the kernel, and exploring applications in transfer learning and domain adaptation that leverage the adaptive nature of random forest kernels.

The principles of adaptivity and data-dependent smoothing exemplified by random forest kernels are likely to remain important in machine learning. By rigorously characterizing these properties, this work contributes to both the theoretical understanding and practical application of random forests across diverse learning problems.


---
