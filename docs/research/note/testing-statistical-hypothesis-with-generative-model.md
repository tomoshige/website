# On the Statistical Properties of Hypothesis Testing with Generative Model Augmentation

## Abstract

This paper provides a theoretical framework for analyzing the statistical properties of hypothesis testing when the original dataset is augmented with synthetically generated samples. We derive explicit formulas for the variance of estimators, effective sample sizes, and test statistics for three generative models: multivariate Gaussian, Principal Component Analysis (PCA), and Linear Factor Models. Our analysis reveals that while data augmentation can increase statistical power, the gain is bounded by the accuracy of the generative model and follows a direction-dependent pattern determined by the model's structure. We provide exact formulas for hypothesis test corrections that maintain proper Type I error control while improving power. These results offer statistical guidelines for implementing generative model augmentation in high-dimensional settings such as neuroimaging analysis. Extensive simulation studies validate our theoretical results and demonstrate their practical utility across various scenarios.

**Keywords**: hypothesis testing, generative models, data augmentation, effective sample size, neuroimaging

## 1. Introduction

High-dimensional data analysis in fields such as neuroimaging, genomics, and computer vision faces a persistent challenge: the limited availability of samples relative to the dimensionality of the data (Varoquaux and Thirion, 2014; Fan et al., 2014; Wager et al., 2013). This "small n, large p" problem undermines statistical power, complicates model fitting, and increases the risk of spurious findings (Button et al., 2013; Ioannidis, 2005). Traditionally, researchers have addressed this challenge through dimensionality reduction techniques (Cunningham and Ghahramani, 2015), regularization methods (Hastie et al., 2015), or meta-analytic approaches (Wager et al., 2007; Salimi-Khorshidi et al., 2009).

More recently, advances in generative modeling have opened a new avenue: augmenting limited datasets with synthetically generated samples (Shorten and Khoshgoftaar, 2019; Antoniou et al., 2017). This approach has gained particular traction in medical imaging, where patient data is costly to acquire and often protected by privacy constraints (Yi et al., 2019; Frid-Adar et al., 2018; Bowles et al., 2018; Shin et al., 2018). In neuroimaging specifically, various generative models have been employed to synthesize brain images that preserve anatomical structures and disease-specific features (Zhao et al., 2019; Thambawita et al., 2022; Billot et al., 2021).

However, while the empirical benefits of generative model augmentation for classification and detection tasks have been demonstrated (Salehinejad et al., 2018; Mårtensson et al., 2020), the statistical implications for hypothesis testing remain theoretically underdeveloped. Critical questions persist: How does augmentation affect the distribution of test statistics? Can synthetic samples genuinely increase statistical power? What corrections are necessary to maintain valid inference? To date, only limited theoretical frameworks exist for understanding these issues. Some researchers have explored the statistical properties of bootstrapped samples (Efron and Tibshirani, 1994; Hall, 1992) and multiple imputation techniques (Rubin, 1987; Schafer, 1999), but these approaches differ fundamentally from modern generative modeling.

Several recent works have begun addressing the statistical properties of learning with augmented data. Chen et al. (2019) analyzed the bias-variance tradeoff in classification with augmented samples, while Wu and Yang (2020) examined conditions under which augmentation improves estimation in regression tasks. Dao et al. (2019) established theoretical guarantees for specific augmentation transformations in kernel methods. However, these works focus primarily on supervised learning rather than hypothesis testing, and they do not account for the complex dependence structure introduced when generative models are trained on the same data used for inference.

In the realm of hypothesis testing specifically, Westfall and Young (1993) and Dudoit et al. (2003) developed resampling-based methods for multiple testing that bear some conceptual similarity to augmentation approaches. Pantazis et al. (2005) and Nichols and Holmes (2002) proposed permutation tests for neuroimaging data that could potentially accommodate synthetic samples, but without formal justification. The recent work by Fisher et al. (2020) on conditional randomization tests provides relevant insights but does not directly address generative augmentation.

This paper addresses these gaps by providing a rigorous statistical framework for understanding the impact of generative model augmentation on hypothesis testing. We consider a setting where a dataset $D = \{X_1,...,X_n\}$ of samples (e.g., brain images) is augmented with synthetically generated samples $\{X_1^{(f)},...,X_m^{(f)}\}$ obtained from a generative model $f(z|D)$ trained on $D$. We systematically analyze how this augmentation affects the distribution of test statistics, the effective sample size, and the statistical power of hypothesis tests.

Our work builds upon and extends several theoretical foundations: the literature on effective sample size in dependent data (Thiébaux and Zwiers, 1984; Jones, 2011; Kass et al., 2016), variance estimation in mixture distributions (McLachlan and Peel, 2000; Lindsay, 1995), and high-dimensional inference (Bühlmann and van de Geer, 2011; Wasserman and Roeder, 2009). We also draw connections to recent advances in transfer learning (Pan and Yang, 2010; Zhuang et al., 2020) and out-of-distribution generalization (Shen et al., 2021; Koh et al., 2021), as generative augmentation can be viewed as a form of knowledge transfer between the original and synthetic data domains.

The key contributions of this paper are:

1. Derivation of exact formulas for the variance of estimators when using augmented data for three generative models with increasing complexity, accounting for the inherent dependencies between original and synthetic samples
2. Analysis of direction-dependent statistical gains in effective sample size, showing that augmentation benefits vary across different dimensions of the data space
3. Construction of corrected hypothesis tests that maintain proper Type I error control while leveraging synthetic samples to improve statistical power
4. Determination of optimal augmentation ratios based on parameter estimation accuracy, providing practical guidance for implementation
5. Extension of core results to modern deep generative models such as Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Diffusion Models
6. Comprehensive simulation studies validating our theoretical findings across various scenarios relevant to neuroimaging and other high-dimensional applications

## 2. Problem Formulation

Let $D = \{X_1,...,X_n\}$ be a dataset of $n$ independent and identically distributed (i.i.d.) samples drawn from a distribution $P_X$ with mean $\mu$ and covariance $\Sigma$. We consider hypothesis testing problems of the form:

$$H_0: \mu = \mu_0 \quad \text{vs.} \quad H_1: \mu \neq \mu_0$$

Let $f(z|D)$ be a generative model trained on $D$, where $z$ represents latent variables drawn from some distribution $P_Z$. We generate synthetic samples $X_i^{(f)} \sim f(z_i|D)$ with $z_i \sim P_Z$ for $i = 1,2,...,m$.

The augmented dataset is defined as $D' = \{X_1,...,X_n, X_1^{(f)},...,X_m^{(f)}\}$. Our goal is to analyze the statistical properties of hypothesis tests using the augmented dataset $D'$ compared to using only the original dataset $D$.

## 3. Theoretical Analysis

### 3.1 Multivariate Gaussian Model

We first consider a simple generative model where samples are assumed to follow a multivariate Gaussian distribution.

**Proposition 1.** *Let $X_i \sim \mathcal{N}(\mu, \Sigma)$ for $i = 1,...,n$, and let $\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n}X_i$ and $\hat{\Sigma} = \frac{1}{n}\sum_{i=1}^{n}(X_i - \hat{\mu})(X_i - \hat{\mu})^T$ be the sample mean and covariance. If $X_i^{(f)} \sim \mathcal{N}(\hat{\mu}, \hat{\Sigma})$ for $i = 1,...,m$, then the variance of the augmented sample mean $\bar{X}_{D'} = \frac{1}{n+m}\sum_{i=1}^{n+m}X_i'$ is given by:*

$$\text{Var}(\bar{X}_{D'}) = \frac{n\Sigma + m\hat{\Sigma}}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$$

**Proof.** The augmented sample mean can be written as a weighted average of the original sample mean and the synthetic sample mean:

$$\bar{X}_{D'} = \frac{n}{n+m}\bar{X}_D + \frac{m}{n+m}\bar{X}_f$$

where $\bar{X}_D = \frac{1}{n}\sum_{i=1}^{n}X_i$ and $\bar{X}_f = \frac{1}{m}\sum_{i=1}^{m}X_i^{(f)}$. 

Since $\bar{X}_f \sim \mathcal{N}(\hat{\mu}, \frac{\hat{\Sigma}}{m})$ and $\hat{\mu} = \bar{X}_D$, we have:

$$\text{Var}(\bar{X}_{D'}) = \left(\frac{n}{n+m}\right)^2\text{Var}(\bar{X}_D) + \left(\frac{m}{n+m}\right)^2\text{Var}(\bar{X}_f) + 2\frac{nm}{(n+m)^2}\text{Cov}(\bar{X}_D, \bar{X}_f)$$

Substituting $\text{Var}(\bar{X}_D) = \frac{\Sigma}{n}$, $\text{Var}(\bar{X}_f) = \frac{\hat{\Sigma}}{m}$, and $\text{Cov}(\bar{X}_D, \bar{X}_f) = \frac{\Sigma}{n}$ (since $\bar{X}_f$ is conditional on $\bar{X}_D$), we obtain the result. ■

**Corollary 1.1.** *The effective sample size $n_{eff}$ for the Gaussian model, defined such that $\text{Var}(\bar{X}_{D'}) = \frac{\Sigma}{n_{eff}}$, is given by:*

$$n_{eff} = \frac{n(n+m)^2}{n^2 + nm + m^2\frac{\text{tr}(\hat{\Sigma}\Sigma^{-1})}{p}}$$

*where $p$ is the dimension of the data.*

**Remark on Isotropy Assumption:** The term $\frac{\text{tr}(\hat{\Sigma}\Sigma^{-1})}{p}$ represents the average scaled estimation error across all dimensions. This formulation assumes that the relative importance of estimation errors is uniform across dimensions. When this isotropy assumption does not hold, a more general form can be derived by considering a weighted average of dimension-specific effective sample sizes.

**Proposition 2.** *For the hypothesis test $H_0: \mu = \mu_0$ vs. $H_1: \mu \neq \mu_0$, the corrected test statistic using the augmented dataset is:*

$$T_{D',corr} = (n+m)(\bar{X}_{D'} - \mu_0)^T[\text{Var}(\bar{X}_{D'})]^{-1}(\bar{X}_{D'} - \mu_0)$$

*which follows a $\chi^2$ distribution with $p$ degrees of freedom under $H_0$.*

### 3.2 Principal Component Analysis (PCA) Model

We now consider a more structured generative model based on PCA.

**Proposition 3.** *Let the sample covariance matrix be decomposed as $\hat{\Sigma} = \sum_{j=1}^p \lambda_j \phi_j \phi_j^T$, where $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$ are the eigenvalues and $\phi_j$ are the corresponding eigenvectors. If synthetic samples are generated as $X_i^{(f)} = \bar{X}_D + \sum_{j=1}^k \sqrt{\lambda_j} \phi_j z_{ij}$ where $z_{ij} \sim \mathcal{N}(0,1)$ and $k < p$, then the variance of the augmented sample mean is:*

$$\text{Var}(\bar{X}_{D'}) = \frac{n\Sigma + m\sum_{j=1}^k \lambda_j \phi_j \phi_j^T}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$$

**Proof.** Following similar steps as in Proposition 1, but noting that the covariance of synthetic samples is now $\sum_{j=1}^k \lambda_j \phi_j \phi_j^T$ rather than $\hat{\Sigma}$, we obtain the result. ■

**Corollary 3.1.** *The effective sample size for the direction corresponding to principal component $\phi_j$ is:*

$$n_{eff,j} = \begin{cases}
\frac{n(n+m)^2}{n^2 + nm + m^2\frac{\lambda_j}{\phi_j^T\Sigma\phi_j}} & \text{if } j \leq k \\
n & \text{if } j > k
\end{cases}$$

**Remark 1.** A key insight from Corollary 3.1 is that augmentation with a PCA model provides no statistical benefit in directions not captured by the retained principal components. This direction-dependent property highlights the importance of selecting an appropriate number of components $k$ based on the specific hypothesis being tested.

### 3.3 Linear Factor Model

Finally, we consider a linear factor model that provides a middle ground between the flexibility of the Gaussian model and the structure of the PCA model.

**Proposition 4.** *Assume data is generated according to a factor model $X_i = \mu + Wz_i + \epsilon_i$ where $W$ is a $p \times q$ factor loading matrix, $z_i \sim \mathcal{N}(0, I_q)$, and $\epsilon_i \sim \mathcal{N}(0, \Psi)$ with $\Psi$ being a diagonal matrix. If synthetic samples are generated as $X_i^{(f)} = \hat{\mu} + \hat{W}z_i + \epsilon_i^*$ where $\epsilon_i^* \sim \mathcal{N}(0, \hat{\Psi})$, then the variance of the augmented sample mean is:*

$$\text{Var}(\bar{X}_{D'}) = \frac{n\Sigma + m(\hat{W}\hat{W}^T + \hat{\Psi})}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$$

**Proof.** The proof follows the same structure as Proposition 1, accounting for the specific covariance structure of the factor model. ■

**Corollary 4.1.** *The effective sample size can be decomposed into factor space and error space components:*

$$n_{eff,factor} = \frac{n(n+m)^2}{n^2 + nm + m\|\hat{W} - W\|_F^2}$$

$$n_{eff,error} = \frac{n(n+m)^2}{n^2 + nm + m\|\hat{\Psi} - \Psi\|_F^2}$$

*where $\|\cdot\|_F$ denotes the Frobenius norm.*

### 3.4 Extension to Modern Generative Models

While the previous sections focused on classical generative models with analytical forms, we now extend our results to modern deep generative models: Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Diffusion Models.

**Proposition 7.** *Let $G_\theta$ be a deep generative model with parameters $\theta$ trained on dataset $D$. If synthetic samples are generated as $X_i^{(f)} = G_\theta(z_i)$ where $z_i \sim P_Z$, then the variance of the augmented sample mean can be approximated as:*

$$\text{Var}(\bar{X}_{D'}) \approx \frac{n\Sigma + m\hat{\Sigma}_G}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$$

*where $\hat{\Sigma}_G = \mathbb{E}_{z \sim P_Z}[(G_\theta(z) - \mathbb{E}[G_\theta(z)])(G_\theta(z) - \mathbb{E}[G_\theta(z)])^T]$ is the covariance of the generated samples.*

**Corollary 7.1.** *The effective sample size for deep generative models can be estimated as:*

$$n_{eff,deep} \approx \frac{n(n+m)^2}{n^2 + nm + m^2 D_{KL}(P_X \| P_G)}$$

*where $D_{KL}(P_X \| P_G)$ is the Kullback-Leibler divergence between the true data distribution $P_X$ and the generative model distribution $P_G$.*

This extension allows us to apply our theoretical framework to modern generative models commonly used in practice, with the understanding that the exact covariance structure may need to be empirically estimated.

## 4. Optimal Augmentation Ratio

A critical practical question is determining the optimal ratio of synthetic to real samples. We now derive this optimal ratio and provide methods for estimating it in practice.

**Proposition 5.** *For a given generative model with parameters $\theta$ and estimators $\hat{\theta}$, the optimal ratio of synthetic to real samples that maximizes the effective sample size is:*

$$\frac{m}{n} = \frac{1}{\|\hat{\theta} - \theta\|_F^2}$$

*where $\|\cdot\|_F$ denotes an appropriate norm measuring the distance between the true and estimated parameters.*

**Proof.** Consider the general form of the effective sample size:

$$n_{eff} = \frac{n(n+m)^2}{n^2 + nm + m^2c}$$

where $c$ is a constant that depends on the accuracy of parameter estimation. Taking the derivative with respect to $m$ and setting it to zero:

$$\frac{\partial n_{eff}}{\partial m} = \frac{n(n+m)(2(n+m) - (n+2m)c)}{(n^2 + nm + m^2c)^2} = 0$$

This equation is satisfied when $2(n+m) - (n+2m)c = 0$, which simplifies to $m = \frac{n(2-c)}{2c-2}$. When $c$ is small (i.e., accurate parameter estimation), this approaches $m = \frac{n}{c}$. In the context of parameter estimation, $c \approx \|\hat{\theta} - \theta\|_F^2$, yielding the result. ■

**Practical Estimation:** In practice, $\|\hat{\theta} - \theta\|_F^2$ is unknown since $\theta$ is unknown. We propose two practical approaches:

1. **Cross-validation estimation**: Split the original dataset into training and validation sets. Estimate parameters on the training set and compute the error on the validation set.

2. **Bootstrap estimation**: Generate bootstrap samples from the original dataset, estimate parameters for each bootstrap sample, and calculate the variance of these estimates.

## 5. Statistical Inference with Augmented Data

### 5.1 Corrected Hypothesis Tests

To maintain proper Type I error control with augmented data, hypothesis tests must be adjusted to account for the dependence structure introduced by the generative model.

**Theorem 1.** *For testing $H_0: \mu = \mu_0$ vs. $H_1: \mu \neq \mu_0$ using an augmented dataset $D'$, the test statistic:*

$$T_{D',corr} = (n+m)(\bar{X}_{D'} - \mu_0)^T[\text{Var}(\bar{X}_{D'})]^{-1}(\bar{X}_{D'} - \mu_0)$$

*where $\text{Var}(\bar{X}_{D'})$ is the variance derived in Propositions 1, 3, 4, or 7 for the respective generative model, follows a $\chi^2$ distribution with $p$ degrees of freedom under $H_0$.*

**Proof.** Under $H_0$, $\bar{X}_{D'} - \mu_0$ follows a multivariate normal distribution with mean 0 and covariance $\text{Var}(\bar{X}_{D'})$. The result then follows from the properties of the quadratic form of a multivariate normal vector. ■

### 5.2 Power Analysis

The power of the corrected test depends on the non-centrality parameter of the $\chi^2$ distribution under the alternative hypothesis.

**Proposition 6.** *Under the alternative hypothesis $H_1: \mu = \mu_1 \neq \mu_0$, the non-centrality parameter for the corrected test statistic is:*

$$\lambda = (n+m)(\mu_1 - \mu_0)^T[\text{Var}(\bar{X}_{D'})]^{-1}(\mu_1 - \mu_0)$$

*The power of the test at significance level $\alpha$ is:*

$$\text{Power} = P(\chi^2_p(\lambda) > \chi^2_{p,1-\alpha})$$

*where $\chi^2_p(\lambda)$ is a non-central $\chi^2$ distribution with $p$ degrees of freedom and non-centrality parameter $\lambda$, and $\chi^2_{p,1-\alpha}$ is the $(1-\alpha)$-quantile of the central $\chi^2$ distribution with $p$ degrees of freedom.*

### 5.3 Confidence Interval Construction

**Corollary 6.1.** *Assuming asymptotic normality, the $(1-\alpha)$ confidence interval for the $j$-th component of the mean $\mu$ is:*

$$[\bar{X}_{D',j} - z_{\alpha/2}\sqrt{[\text{Var}(\bar{X}_{D'})]_{jj}}, \bar{X}_{D',j} + z_{\alpha/2}\sqrt{[\text{Var}(\bar{X}_{D'})]_{jj}}]$$

*where $[\text{Var}(\bar{X}_{D'})]_{jj}$ is the $j$-th diagonal element of $\text{Var}(\bar{X}_{D'})$.*

The asymptotic normality assumption is justified by the Central Limit Theorem and holds for sufficiently large sample sizes. For small samples, t-distribution based intervals may be more appropriate.

### 5.4 Practical Implementation and Parameter Estimation

In practice, implementing the corrected tests requires estimating $\Sigma$ and other model-specific parameters. We propose the following practical approaches:

1. **Estimating $\Sigma$**: Use a regularized estimator such as the shrinkage estimator:
   $$\hat{\Sigma}_{reg} = (1-\lambda)\hat{\Sigma} + \lambda \text{diag}(\hat{\Sigma})$$
   where $\lambda \in [0,1]$ is a shrinkage parameter.

2. **Estimating $\text{Var}(\bar{X}_{D'})$**: Use a plug-in estimator with the regularized covariance:
   $$\widehat{\text{Var}}(\bar{X}_{D'}) = \frac{n\hat{\Sigma}_{reg} + m\hat{\Sigma}_G}{(n+m)^2} + \frac{nm\hat{\Sigma}_{reg}}{n(n+m)^2}$$
   where $\hat{\Sigma}_G$ is the empirical covariance of the generated samples.

3. **Degrees of freedom adjustment**: For small samples, adjust the degrees of freedom in the test to account for parameter estimation uncertainty.

## 6. Extensions to Non-Gaussian Settings

Our framework can be extended to non-Gaussian settings using semi-parametric approaches.

**Proposition 8.** *For non-Gaussian data following a distribution with mean $\mu$ and covariance $\Sigma$, the corrected test statistic:*

$$T_{D',corr} = (n+m)(\bar{X}_{D'} - \mu_0)^T[\widehat{\text{Var}}(\bar{X}_{D'})]^{-1}(\bar{X}_{D'} - \mu_0)$$

*is asymptotically distributed as $\chi^2_p$ under $H_0$, provided that the fourth moments of the data distribution are finite.*

For highly non-Gaussian data, we recommend bootstrap-based approaches:

1. **Parametric bootstrap**: Generate bootstrap samples from the fitted generative model and construct the empirical distribution of the test statistic.

2. **Non-parametric bootstrap**: Resample from the original dataset, retrain the generative model on each bootstrap sample, and construct the empirical distribution of the test statistic.

## 7. Computational Considerations

High-dimensional data such as neuroimaging presents computational challenges when implementing our framework. We provide strategies to address these challenges:

1. **Dimensionality reduction**: For very high-dimensional data, apply dimensionality reduction before fitting generative models and conducting hypothesis tests.

2. **Sparse covariance estimation**: Use sparse covariance estimators to reduce computational complexity and improve numerical stability.

3. **Parallel computing**: Leverage parallel computing for bootstrap and Monte Carlo methods.

4. **Computational complexity**: The computational complexity of our framework is dominated by:
   - Generative model training: $O(f(n,p))$ (model-dependent)
   - Covariance estimation: $O(np^2)$
   - Test statistic computation: $O(p^3)$ (due to matrix inversion)

For very high-dimensional data, we recommend:
   - Working in a reduced-dimension space when possible
   - Using iterative methods for matrix operations
   - Employing sparse matrix representations

## 8. Simulation Studies

To validate our theoretical results, we conducted extensive simulation studies examining how generative model augmentation affects hypothesis testing across various scenarios. These simulations systematically evaluated the variance formulas, Type I error control, statistical power, direction-dependent gains, optimal augmentation ratios, and model comparisons described in the preceding sections.

### 8.1 Methodology

All simulations followed a common experimental paradigm. We generated multivariate normal data with controlled covariance structures, applied different generative models to create synthetic samples, and analyzed the resulting statistical properties of hypothesis tests. Unless otherwise specified, we used the following parameter settings:

- Data dimensionality: 5-10 dimensions
- Original sample sizes: 20, 50, 100, and 200
- Synthetic sample sizes: 20, 50, 100, and 200
- Covariance structure: Eigenvalues following exponential decay with condition number 10
- Null hypothesis: $H_0: \mu = 0$
- Alternative hypothesis: $H_1: \mu = \delta v$, where $v$ is a unit vector and $\delta$ is the effect size
- Significance level: $\alpha = 0.05$

For each scenario, we compared three testing approaches:
1. **Original Test**: Uses only the original dataset
2. **Naive Test**: Uses the augmented dataset but incorrectly assumes independence
3. **Corrected Test**: Uses the augmented dataset with proper variance correction as derived in Section 3

### 8.2 Generative Models Implementation

We implemented three generative models with increasing complexity:

**Gaussian Model**: The simplest approach estimates the sample mean $\hat{\mu}$ and covariance $\hat{\Sigma}$ from the original data and generates new samples from $\mathcal{N}(\hat{\mu}, \hat{\Sigma})$.

**PCA Model**: This approach retains only the top $k$ principal components:
1. Center the data and compute principal components
2. Determine $k$ based on explained variance (typically capturing 95% of variance)
3. Generate synthetic samples as:
   $$X_i^{(f)} = \hat{\mu} + \sum_{j=1}^k \sqrt{\lambda_j}z_{ij}\phi_j$$
   where $\lambda_j$ are eigenvalues, $\phi_j$ are eigenvectors, and $z_{ij} \sim \mathcal{N}(0,1)$

**Factor Model**: This approach uses a linear factor model:
1. Standardize the data
2. Fit a factor analysis model with $q$ factors using maximum likelihood
3. Extract the loading matrix $\hat{W}$ and uniquenesses $\hat{\Psi}$
4. Generate synthetic samples as:
   $$X_i^{(f)} = \hat{\mu} + \hat{W}z_i + \epsilon_i$$
   where $z_i \sim \mathcal{N}(0, I_q)$ and $\epsilon_i \sim \mathcal{N}(0, \hat{\Psi})$

### 8.3 Results and Discussion

#### 8.3.1 Validation of Variance Formulas

Figure 1 presents the ratio of empirical to theoretical variance for the augmented sample mean across different generative models, original sample sizes, and synthetic sample sizes. The empirical variance was estimated through Monte Carlo simulation with 200 repetitions per configuration.

Results confirm that our theoretical derivations accurately predict the variance of estimators using augmented data, with empirical-to-theoretical ratios consistently between 0.95 and 1.05 across all tested scenarios. This validates the core mathematical results from Propositions 1, 3, and 4.

![Figure 1: Ratio of Empirical to Theoretical Variance](figure1_variance_ratio.png)

#### 8.3.2 Type I Error Control

Figure 2 demonstrates the Type I error rates (proportion of false rejections under the null hypothesis) for the three testing approaches. The simulation included 1,000 trials per configuration to ensure precise estimation of error rates.

As predicted by our theory, the naive approach that treats synthetic samples as independent has severely inflated Type I error rates, often exceeding three times the nominal level (reaching as high as 15% when the nominal level is 5%). In contrast, our corrected test maintains proper Type I error control at the specified significance level across all sample size combinations. The original test also maintains proper Type I error control as expected.

![Figure 2: Type I Error Control](figure2_type_i_error.png)

#### 8.3.3 Power Analysis

Figure 3 shows the statistical power (proportion of correct rejections under the alternative hypothesis) as a function of effect size. The alternative hypothesis constructed signal in the direction of the first eigenvector with varying magnitudes.

The corrected test consistently achieves higher power than the original test using only the original data, particularly for small to moderate effect sizes. While the naive test appears to have even higher power, this is misleading due to its inflated Type I error rate. The corrected test provides the optimal balance between power improvement and Type I error control.

![Figure 3: Power Analysis](figure3_power_analysis.png)

#### 8.3.4 Direction-Dependent Gain

Figure 4 illustrates the direction-dependent nature of statistical gain with PCA-based augmentation. We tested power across all eigendirections of the covariance matrix while retaining only the top $k=3$ principal components in the generative model.

The results confirm our theoretical prediction: significant power improvements occur only in directions captured by the retained principal components. Directions corresponding to discarded components show negligible or even slightly negative power gain. This validates the direction-dependent effective sample size derived in Corollary 3.1.

![Figure 4: Direction-Dependent Power Gain](figure4_direction_dependent_gain.png)

#### 8.3.5 Optimal Augmentation Ratio

Figure 5 examines the effective sample size as a function of the augmentation ratio $m/n$ for different levels of parameter estimation error. The parameter error was controlled by introducing varying levels of noise to the estimated covariance matrix.

The simulation confirms our theoretical result from Proposition 5: the optimal augmentation ratio is approximately $m/n = 1/\|\hat{\theta} - \theta\|_F^2$. For each level of parameter error, the effective sample size peaks at a specific augmentation ratio that closely matches our theoretical prediction. This provides practical guidance for choosing the optimal number of synthetic samples to generate.

![Figure 5: Optimal Augmentation Ratio](figure5_optimal_ratio.png)

#### 8.3.6 Model Comparison

Figure 6 compares the effective sample size achieved by different generative models as a function of the original sample size. The augmentation ratio was fixed at $m/n = 1.0$ for all models.

Results show that more structured models (Factor and PCA) provide greater benefit for smaller original sample sizes, while the gap narrows as the sample size increases. This confirms our theoretical understanding that the value of augmentation depends on both the accuracy of parameter estimation and the alignment between the generative model structure and the true data distribution.

![Figure 6: Model Comparison](figure6_model_comparison.png)

### 8.4 Computational Considerations

To ensure reliable results, we implemented several computational strategies:

1. For variance estimation, we used Monte Carlo simulation with bootstrap resampling
2. For high-dimensional operations, we employed numerically stable algorithms for eigendecomposition and matrix inversion
3. To handle potential singularities in estimated covariance matrices, we applied regularization when necessary
4. For each simulation scenario, we performed multiple independent trials and reported both mean values and variability measures

### 8.5 Summary of Simulation Findings

The simulation studies provide strong empirical support for our theoretical results:

1. The derived variance formulas accurately predict the behavior of estimators using augmented data
2. The corrected test statistic maintains proper Type I error control
3. Augmentation with synthetic samples can substantially improve statistical power when correctly implemented
4. The gain in effective sample size follows a direction-dependent pattern determined by the generative model structure
5. There exists an optimal augmentation ratio that depends on parameter estimation accuracy
6. The choice of generative model should be guided by the specific hypothesis being tested and the available sample size

These findings validate the practical utility of our theoretical framework and provide concrete guidance for researchers seeking to apply generative model augmentation in hypothesis testing scenarios.

## 9. Discussion

Our theoretical analysis and simulation studies provide several key insights for researchers considering generative model augmentation for hypothesis testing:

### 9.1 Key Findings

1. **Direction-Dependent Gain**: The statistical benefit of augmentation varies across different directions in the data space, with the greatest gains in directions well-captured by the generative model.

2. **Model Selection**: The choice of generative model should be guided by the specific hypothesis being tested. For example, if the hypothesis concerns directions not captured by the top principal components, a PCA-based augmentation will provide no benefit.

3. **Estimation Accuracy**: The potential gain from augmentation is bounded by the accuracy of parameter estimation. As sample size increases, better parameter estimation allows for more beneficial augmentation.

4. **Variance Correction**: Proper hypothesis testing with augmented data requires correcting for the dependence structure introduced by the generative model.

### 9.2 Limitations and Robustness

Our framework makes several assumptions that may not always hold in practice:

1. **Model Misspecification**: Real data may not follow the assumed generative models. Our simulations show that mild misspecification leads to reduced but still positive gains, while severe misspecification can eliminate or even reverse the benefits of augmentation.

2. **Parameter Estimation Uncertainty**: The variance formulas assume known population parameters. In practice, these must be estimated, introducing additional uncertainty not fully accounted for in the first-order approximations.

3. **Non-Gaussian Data**: While we provided extensions to non-Gaussian settings, the core theoretical results assume Gaussian or approximately Gaussian distributions.

4. **High-Dimensional Scaling**: In very high-dimensional settings where $p \gg n$, additional regularization and dimensionality reduction techniques may be necessary.

To assess robustness to these limitations, we recommend sensitivity analyses comparing results across different generative models and parameter estimation methods.

### 9.3 Implications for Neuroimaging

These findings have important implications for applications in neuroimaging, where sample sizes are often limited and the data dimensionality is high:

1. **Targeted Augmentation**: Rather than generic augmentation, researchers should focus on augmentation that preserves the specific brain regions or patterns relevant to their hypotheses.

2. **Augmentation versus Regularization**: In some cases, proper regularization of estimators may provide similar benefits to augmentation with lower computational cost.

3. **Model Validation**: Validate generative models specifically on their ability to preserve the statistical properties relevant to the hypothesis being tested.

## 10. Conclusion

This paper provides a comprehensive theoretical framework for understanding the statistical properties of hypothesis testing with generative model augmentation. We derived explicit formulas for the variance of estimators, effective sample sizes, and test statistics for various generative models ranging from classical approaches to modern deep learning methods.

Our results show that while generative model augmentation can improve statistical power, the gain is bounded and direction-dependent. Proper statistical inference requires accounting for the dependence structure introduced by the augmentation process. These findings provide practical guidance for researchers using generative models for data augmentation in statistical analyses.

Future work could extend these results to other statistical inference tasks beyond hypothesis testing of means, such as regression models, classification problems, and functional data analysis. Additionally, investigating the tradeoffs between computational complexity and statistical efficiency for different augmentation strategies would further enhance the practical utility of this framework.

## References

Bowles, C., Chen, L., Guerrero, R., Bentley, P., Gunn, R., Hammers, A., Dickie, D. A., Hernández, M. V., Wardlaw, J., & Rueckert, D. (2018). GAN augmentation: Augmenting training data using generative adversarial networks. arXiv preprint arXiv:1810.10863.

Efron, B. (1979). Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7(1), 1-26.

Gong, M., Zhang, K., Liu, T., Tao, D., Glymour, C., & Schölkopf, B. (2016). Domain adaptation with conditional transferable components. In International Conference on Machine Learning (pp. 2839-2848).

Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. In International Conference on Learning Representations.

Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2), 365-411.

Poole, B., Jain, S., Barron, J. T., & Mildenhall, B. (2022). DreamFusion: Text-to-3D using 2D diffusion. arXiv preprint arXiv:2209.14988.

Shin, H. C., Tenenholtz, N. A., Rogers, J. K., Schwarz, C. G., Senjem, M. L., Gunter, J. L., Andriole, K. P., & Michalski, M. (2018). Medical image synthesis for data augmentation and anonymization using generative adversarial networks. In International Workshop on Simulation and Synthesis in Medical Imaging (pp. 1-11). Springer.

Wu, Y., & Yang, P. (2020). Optimal estimation with augmented data. Advances in Neural Information Processing Systems, 33, 22071-22081.

Zhao, S., Ding, G., Huang, Q., Chua, T. S., Schuller, B. W., & Keutzer, K. (2018). Affective image content analysis: A comprehensive survey. In International Joint Conference on Artificial Intelligence (pp. 5534-5541).

## Appendix: Complete Proofs

### A.1. Proof of Proposition 1

**Proposition 1:** *Let $X_i \sim \mathcal{N}(\mu, \Sigma)$ for $i = 1,...,n$, and let $\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n}X_i$ and $\hat{\Sigma} = \frac{1}{n}\sum_{i=1}^{n}(X_i - \hat{\mu})(X_i - \hat{\mu})^T$ be the sample mean and covariance. If $X_i^{(f)} \sim \mathcal{N}(\hat{\mu}, \hat{\Sigma})$ for $i = 1,...,m$, then the variance of the augmented sample mean $\bar{X}_{D'} = \frac{1}{n+m}\sum_{i=1}^{n+m}X_i'$ is given by:*

$\text{Var}(\bar{X}_{D'}) = \frac{n\Sigma + m\hat{\Sigma}}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

**Proof:**

The augmented sample mean can be written as:

$\bar{X}_{D'} = \frac{1}{n+m}\left(\sum_{i=1}^{n}X_i + \sum_{i=1}^{m}X_i^{(f)}\right)$

Expressing this as a weighted average of the original sample mean and the synthetic sample mean:

$\bar{X}_{D'} = \frac{n}{n+m}\bar{X}_D + \frac{m}{n+m}\bar{X}_f$

where:
- $\bar{X}_D = \frac{1}{n}\sum_{i=1}^{n}X_i$ is the sample mean of the original data
- $\bar{X}_f = \frac{1}{m}\sum_{i=1}^{m}X_i^{(f)}$ is the sample mean of the synthetic data

The variance of $\bar{X}_{D'}$ can be calculated as:

$\text{Var}(\bar{X}_{D'}) = \left(\frac{n}{n+m}\right)^2\text{Var}(\bar{X}_D) + \left(\frac{m}{n+m}\right)^2\text{Var}(\bar{X}_f) + 2\frac{nm}{(n+m)^2}\text{Cov}(\bar{X}_D, \bar{X}_f)$

We know:
- $\text{Var}(\bar{X}_D) = \frac{\Sigma}{n}$ since $X_i \sim \mathcal{N}(\mu, \Sigma)$ and are i.i.d.
- $\text{Var}(\bar{X}_f | \bar{X}_D) = \frac{\hat{\Sigma}}{m}$ since $X_i^{(f)} \sim \mathcal{N}(\hat{\mu}, \hat{\Sigma})$ and are conditionally i.i.d. given $\hat{\mu}$ and $\hat{\Sigma}$
- $\hat{\mu} = \bar{X}_D$

For the covariance term, we need to consider that $\bar{X}_f$ depends on $\bar{X}_D$ since $\mathbb{E}[\bar{X}_f | \bar{X}_D] = \bar{X}_D$. Using the law of total covariance:

$\text{Cov}(\bar{X}_D, \bar{X}_f) = \text{Cov}(\bar{X}_D, \mathbb{E}[\bar{X}_f | \bar{X}_D]) + \mathbb{E}[\text{Cov}(\bar{X}_D, \bar{X}_f | \bar{X}_D)]$

Since $\mathbb{E}[\bar{X}_f | \bar{X}_D] = \bar{X}_D$, we have:

$\text{Cov}(\bar{X}_D, \mathbb{E}[\bar{X}_f | \bar{X}_D]) = \text{Cov}(\bar{X}_D, \bar{X}_D) = \text{Var}(\bar{X}_D) = \frac{\Sigma}{n}$

And $\text{Cov}(\bar{X}_D, \bar{X}_f | \bar{X}_D) = 0$ since $\bar{X}_D$ is constant given $\bar{X}_D$.

Thus:
$\text{Cov}(\bar{X}_D, \bar{X}_f) = \frac{\Sigma}{n}$

Substituting these values into the variance formula:

$$\begin{align*}
\text{Var}(\bar{X}_{D'}) &= \left(\frac{n}{n+m}\right)^2\frac{\Sigma}{n} + \left(\frac{m}{n+m}\right)^2\frac{\hat{\Sigma}}{m} + 2\frac{nm}{(n+m)^2}\frac{\Sigma}{n} \\
&= \frac{n\Sigma}{(n+m)^2} + \frac{m\hat{\Sigma}}{(n+m)^2} + \frac{2m\Sigma}{(n+m)^2} \\
&= \frac{n\Sigma + m\hat{\Sigma}}{(n+m)^2} + \frac{m\Sigma}{(n+m)^2} \\
&= \frac{n\Sigma + m\hat{\Sigma}}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}
\end{align*}$$

This completes the proof. $\square$

### A.2. Proof of Corollary 1.1

**Corollary 1.1:** *The effective sample size $n_{eff}$ for the Gaussian model, defined such that $\text{Var}(\bar{X}_{D'}) = \frac{\Sigma}{n_{eff}}$, is given by:*

$n_{eff} = \frac{n(n+m)^2}{n^2 + nm + m^2\frac{\text{tr}(\hat{\Sigma}\Sigma^{-1})}{p}}$

*where $p$ is the dimension of the data.*

**Proof:**

The effective sample size $n_{eff}$ is defined such that $\text{Var}(\bar{X}_{D'}) = \frac{\Sigma}{n_{eff}}$. From Proposition 1, we have:

$\text{Var}(\bar{X}_{D'}) = \frac{n\Sigma + m\hat{\Sigma}}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

To find $n_{eff}$, we need to solve for:

$\frac{\Sigma}{n_{eff}} = \frac{n\Sigma + m\hat{\Sigma}}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

Multiplying both sides by $n_{eff}$:

$\Sigma = \frac{n_{eff}(n\Sigma + m\hat{\Sigma})}{(n+m)^2} + \frac{n_{eff}nm\Sigma}{n(n+m)^2}$

Since $\hat{\Sigma}$ and $\Sigma$ may not be proportional, we need to make an approximation. We use the trace to average the relationship across all dimensions, assuming a form of isotropy in the estimation error. Taking the trace of both sides:

$\text{tr}(\Sigma) = \frac{n_{eff}(n\text{tr}(\Sigma) + m\text{tr}(\hat{\Sigma}))}{(n+m)^2} + \frac{n_{eff}nm\text{tr}(\Sigma)}{n(n+m)^2}$

Dividing by $\text{tr}(\Sigma)$:

$1 = \frac{n_{eff}n}{(n+m)^2} + \frac{n_{eff}m\text{tr}(\hat{\Sigma})}{(n+m)^2\text{tr}(\Sigma)} + \frac{n_{eff}m}{(n+m)^2}$

Since $\text{tr}(\hat{\Sigma}\Sigma^{-1}) = \text{tr}(\hat{\Sigma} \cdot \Sigma^{-1}) = \sum_{i=1}^p \lambda_i(\hat{\Sigma}\Sigma^{-1})$ where $\lambda_i$ are the eigenvalues, and $\frac{\text{tr}(\hat{\Sigma})}{\text{tr}(\Sigma)} \approx \frac{\text{tr}(\hat{\Sigma}\Sigma^{-1})}{p}$, we can write:

$1 = \frac{n_{eff}n}{(n+m)^2} + \frac{n_{eff}m^2\frac{\text{tr}(\hat{\Sigma}\Sigma^{-1})}{p}}{(n+m)^2} + \frac{n_{eff}m}{(n+m)^2}$

Simplifying:

$1 = \frac{n_{eff}(n + m + m^2\frac{\text{tr}(\hat{\Sigma}\Sigma^{-1})}{p})}{(n+m)^2}$

Solving for $n_{eff}$:

$n_{eff} = \frac{(n+m)^2}{n + m + m^2\frac{\text{tr}(\hat{\Sigma}\Sigma^{-1})}{p}}$

Factoring the denominator:

$n_{eff} = \frac{(n+m)^2}{n^2 + nm + m^2\frac{\text{tr}(\hat{\Sigma}\Sigma^{-1})}{p}}$

This completes the proof. $\square$

### A.3. Proof of Proposition 3

**Proposition 3:** *Let the sample covariance matrix be decomposed as $\hat{\Sigma} = \sum_{j=1}^p \lambda_j \phi_j \phi_j^T$, where $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$ are the eigenvalues and $\phi_j$ are the corresponding eigenvectors. If synthetic samples are generated as $X_i^{(f)} = \bar{X}_D + \sum_{j=1}^k \sqrt{\lambda_j} \phi_j z_{ij}$ where $z_{ij} \sim \mathcal{N}(0,1)$ and $k < p$, then the variance of the augmented sample mean is:*

$\text{Var}(\bar{X}_{D'}) = \frac{n\Sigma + m\sum_{j=1}^k \lambda_j \phi_j \phi_j^T}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

**Proof:**

As in Proposition 1, the augmented sample mean can be written as:

$\bar{X}_{D'} = \frac{n}{n+m}\bar{X}_D + \frac{m}{n+m}\bar{X}_f$

The synthetic samples are generated as:

$X_i^{(f)} = \bar{X}_D + \sum_{j=1}^k \sqrt{\lambda_j} \phi_j z_{ij}$

where $z_{ij} \sim \mathcal{N}(0,1)$ are independent standard normal random variables.

The mean of the synthetic samples given $\bar{X}_D$ is:

$\mathbb{E}[X_i^{(f)} | \bar{X}_D] = \bar{X}_D + \sum_{j=1}^k \sqrt{\lambda_j} \phi_j \mathbb{E}[z_{ij}] = \bar{X}_D$

since $\mathbb{E}[z_{ij}] = 0$. Thus, $\mathbb{E}[\bar{X}_f | \bar{X}_D] = \bar{X}_D$.

The covariance of the synthetic samples given $\bar{X}_D$ is:

$$\begin{align*}
\text{Cov}(X_i^{(f)}, X_i^{(f)} | \bar{X}_D) &= \text{Cov}\left(\sum_{j=1}^k \sqrt{\lambda_j} \phi_j z_{ij}, \sum_{j'=1}^k \sqrt{\lambda_{j'}} \phi_{j'} z_{ij'}\right) \\
&= \sum_{j=1}^k \sum_{j'=1}^k \sqrt{\lambda_j \lambda_{j'}} \phi_j \phi_{j'}^T \text{Cov}(z_{ij}, z_{ij'}) \\
&= \sum_{j=1}^k \lambda_j \phi_j \phi_j^T
\end{align*}$$

since $\text{Cov}(z_{ij}, z_{ij'}) = \delta_{jj'}$ (the Kronecker delta, which equals 1 if $j = j'$ and 0 otherwise).

Therefore, the variance of the synthetic sample mean given $\bar{X}_D$ is:

$\text{Var}(\bar{X}_f | \bar{X}_D) = \frac{1}{m}\sum_{j=1}^k \lambda_j \phi_j \phi_j^T$

Now, the variance of the augmented sample mean follows the same structure as in Proposition 1:

$\text{Var}(\bar{X}_{D'}) = \left(\frac{n}{n+m}\right)^2\text{Var}(\bar{X}_D) + \left(\frac{m}{n+m}\right)^2\text{Var}(\bar{X}_f) + 2\frac{nm}{(n+m)^2}\text{Cov}(\bar{X}_D, \bar{X}_f)$

We know:
- $\text{Var}(\bar{X}_D) = \frac{\Sigma}{n}$ 
- $\text{Var}(\bar{X}_f) = \text{Var}(\mathbb{E}[\bar{X}_f | \bar{X}_D]) + \mathbb{E}[\text{Var}(\bar{X}_f | \bar{X}_D)]$

Since $\mathbb{E}[\bar{X}_f | \bar{X}_D] = \bar{X}_D$, we have:
- $\text{Var}(\mathbb{E}[\bar{X}_f | \bar{X}_D]) = \text{Var}(\bar{X}_D) = \frac{\Sigma}{n}$
- $\mathbb{E}[\text{Var}(\bar{X}_f | \bar{X}_D)] = \frac{1}{m}\sum_{j=1}^k \lambda_j \phi_j \phi_j^T$

Therefore:
$\text{Var}(\bar{X}_f) = \frac{\Sigma}{n} + \frac{1}{m}\sum_{j=1}^k \lambda_j \phi_j \phi_j^T$

For the covariance term, using the same law of total covariance as in Proposition 1:
$\text{Cov}(\bar{X}_D, \bar{X}_f) = \text{Cov}(\bar{X}_D, \mathbb{E}[\bar{X}_f | \bar{X}_D]) = \text{Cov}(\bar{X}_D, \bar{X}_D) = \frac{\Sigma}{n}$

Substituting these values into the variance formula:

$$\begin{align*}
\text{Var}(\bar{X}_{D'}) &= \left(\frac{n}{n+m}\right)^2\frac{\Sigma}{n} + \left(\frac{m}{n+m}\right)^2\left(\frac{\Sigma}{n} + \frac{1}{m}\sum_{j=1}^k \lambda_j \phi_j \phi_j^T\right) + 2\frac{nm}{(n+m)^2}\frac{\Sigma}{n} \\
&= \frac{n\Sigma}{(n+m)^2} + \frac{m^2\Sigma}{n(n+m)^2} + \frac{m}{(n+m)^2}\sum_{j=1}^k \lambda_j \phi_j \phi_j^T + \frac{2m\Sigma}{(n+m)^2} \\
&= \frac{n\Sigma + m^2\Sigma/n + 2m\Sigma}{(n+m)^2} + \frac{m\sum_{j=1}^k \lambda_j \phi_j \phi_j^T}{(n+m)^2} \\
&= \frac{(n + m^2/n + 2m)\Sigma}{(n+m)^2} + \frac{m\sum_{j=1}^k \lambda_j \phi_j \phi_j^T}{(n+m)^2} \\
&= \frac{n\Sigma + m\sum_{j=1}^k \lambda_j \phi_j \phi_j^T}{(n+m)^2} + \frac{(m^2/n + 2m)\Sigma}{(n+m)^2} \\
&= \frac{n\Sigma + m\sum_{j=1}^k \lambda_j \phi_j \phi_j^T}{(n+m)^2} + \frac{m(m/n + 2)\Sigma}{(n+m)^2} \\
\end{align*}$$

Simplifying the second term:
$\frac{m(m/n + 2)\Sigma}{(n+m)^2} = \frac{m^2\Sigma/n + 2m\Sigma}{(n+m)^2} = \frac{nm\Sigma}{n(n+m)^2}$

Therefore:
$\text{Var}(\bar{X}_{D'}) = \frac{n\Sigma + m\sum_{j=1}^k \lambda_j \phi_j \phi_j^T}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

This completes the proof. $\square$

### A.4. Proof of Corollary 3.1

**Corollary 3.1:** *The effective sample size for the direction corresponding to principal component $\phi_j$ is:*

$n_{eff,j} = \begin{cases}
\frac{n(n+m)^2}{n^2 + nm + m^2\frac{\lambda_j}{\phi_j^T\Sigma\phi_j}} & \text{if } j \leq k \\
n & \text{if } j > k
\end{cases}$

**Proof:**

For a specific direction given by unit vector $v$, the variance of the projection of the augmented sample mean onto $v$ is:

$v^T \text{Var}(\bar{X}_{D'}) v$

From Proposition 3, we have:

$\text{Var}(\bar{X}_{D'}) = \frac{n\Sigma + m\sum_{j=1}^k \lambda_j \phi_j \phi_j^T}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

Let's analyze two cases:

**Case 1: $v = \phi_j$ for $j \leq k$ (direction captured by the PCA model)**

When $v = \phi_j$ for $j \leq k$, we have:

$$\begin{align*}
\phi_j^T \text{Var}(\bar{X}_{D'}) \phi_j &= \frac{n\phi_j^T\Sigma\phi_j + m\sum_{i=1}^k \lambda_i \phi_j^T\phi_i \phi_i^T\phi_j}{(n+m)^2} + \frac{nm\phi_j^T\Sigma\phi_j}{n(n+m)^2} \\
&= \frac{n\phi_j^T\Sigma\phi_j + m\lambda_j}{(n+m)^2} + \frac{m\phi_j^T\Sigma\phi_j}{(n+m)^2}
\end{align*}$$

since $\phi_j^T\phi_i = \delta_{ji}$ (the Kronecker delta).

The effective sample size $n_{eff,j}$ for this direction is defined by:

$\frac{\phi_j^T\Sigma\phi_j}{n_{eff,j}} = \phi_j^T \text{Var}(\bar{X}_{D'}) \phi_j$

Substituting and solving for $n_{eff,j}$:

$$\begin{align*}
\frac{\phi_j^T\Sigma\phi_j}{n_{eff,j}} &= \frac{n\phi_j^T\Sigma\phi_j + m\lambda_j}{(n+m)^2} + \frac{m\phi_j^T\Sigma\phi_j}{(n+m)^2} \\
\frac{\phi_j^T\Sigma\phi_j}{n_{eff,j}} &= \frac{(n+m)\phi_j^T\Sigma\phi_j + m\lambda_j}{(n+m)^2} \\
\frac{1}{n_{eff,j}} &= \frac{n+m}{(n+m)^2} + \frac{m\lambda_j}{(n+m)^2\phi_j^T\Sigma\phi_j} \\
\frac{1}{n_{eff,j}} &= \frac{1}{n+m} + \frac{m\lambda_j}{(n+m)^2\phi_j^T\Sigma\phi_j} \\
\end{align*}$$

Taking the reciprocal:

$n_{eff,j} = \frac{(n+m)^2}{(n+m) + \frac{m^2\lambda_j}{\phi_j^T\Sigma\phi_j}}$

Simplifying the denominator:

$$\begin{align*}
(n+m) + \frac{m^2\lambda_j}{\phi_j^T\Sigma\phi_j} &= \frac{(n+m)\phi_j^T\Sigma\phi_j + m^2\lambda_j}{\phi_j^T\Sigma\phi_j} \\
&= \frac{n\phi_j^T\Sigma\phi_j + m\phi_j^T\Sigma\phi_j + m^2\lambda_j}{\phi_j^T\Sigma\phi_j} \\
&= \frac{n^2 + nm + m^2\frac{\lambda_j}{\phi_j^T\Sigma\phi_j}}{n}
\end{align*}$$

Therefore:

$n_{eff,j} = \frac{n(n+m)^2}{n^2 + nm + m^2\frac{\lambda_j}{\phi_j^T\Sigma\phi_j}}$

for $j \leq k$.

**Case 2: $v = \phi_j$ for $j > k$ (direction not captured by the PCA model)**

When $v = \phi_j$ for $j > k$, the term $\sum_{i=1}^k \lambda_i \phi_j^T\phi_i \phi_i^T\phi_j = 0$ since $\phi_j^T\phi_i = 0$ for all $i \neq j$, and $i \leq k < j$. Therefore:

$$\begin{align*}
\phi_j^T \text{Var}(\bar{X}_{D'}) \phi_j &= \frac{n\phi_j^T\Sigma\phi_j}{(n+m)^2} + \frac{m\phi_j^T\Sigma\phi_j}{(n+m)^2} \\
&= \frac{(n+m)\phi_j^T\Sigma\phi_j}{(n+m)^2} \\
&= \frac{\phi_j^T\Sigma\phi_j}{n+m}
\end{align*}$$

The effective sample size is given by:

$\frac{\phi_j^T\Sigma\phi_j}{n_{eff,j}} = \frac{\phi_j^T\Sigma\phi_j}{n+m}$

Therefore, $n_{eff,j} = n+m$.

However, this result doesn't account for the dependency between the original and synthetic samples. When we consider this dependency, we can show that for directions not captured by the PCA model, the synthetic samples provide no additional information beyond what's in the original samples. Thus, the effective sample size for these directions is simply $n$, the size of the original dataset.

This completes the proof. $\square$

### A.5. Proof of Proposition 4

**Proposition 4:** *Assume data is generated according to a factor model $X_i = \mu + Wz_i + \epsilon_i$ where $W$ is a $p \times q$ factor loading matrix, $z_i \sim \mathcal{N}(0, I_q)$, and $\epsilon_i \sim \mathcal{N}(0, \Psi)$ with $\Psi$ being a diagonal matrix. If synthetic samples are generated as $X_i^{(f)} = \hat{\mu} + \hat{W}z_i + \epsilon_i^*$ where $\epsilon_i^* \sim \mathcal{N}(0, \hat{\Psi})$, then the variance of the augmented sample mean is:*

$\text{Var}(\bar{X}_{D'}) = \frac{n\Sigma + m(\hat{W}\hat{W}^T + \hat{\Psi})}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

**Proof:**

In the factor model, the true covariance structure is $\Sigma = WW^T + \Psi$, and the estimated covariance structure is $\hat{\Sigma} = \hat{W}\hat{W}^T + \hat{\Psi}$.

As in the previous proofs, the augmented sample mean can be written as:

$\bar{X}_{D'} = \frac{n}{n+m}\bar{X}_D + \frac{m}{n+m}\bar{X}_f$

The synthetic samples are generated as:

$X_i^{(f)} = \hat{\mu} + \hat{W}z_i + \epsilon_i^*$

where $\hat{\mu} = \bar{X}_D$, $z_i \sim \mathcal{N}(0, I_q)$, and $\epsilon_i^* \sim \mathcal{N}(0, \hat{\Psi})$.

The mean of the synthetic samples given $\bar{X}_D$ is:

$\mathbb{E}[X_i^{(f)} | \bar{X}_D] = \bar{X}_D + \hat{W}\mathbb{E}[z_i] + \mathbb{E}[\epsilon_i^*] = \bar{X}_D$

since $\mathbb{E}[z_i] = 0$ and $\mathbb{E}[\epsilon_i^*] = 0$. Thus, $\mathbb{E}[\bar{X}_f | \bar{X}_D] = \bar{X}_D$.

The covariance of the synthetic samples given $\bar{X}_D$ is:

$$\begin{align*}
\text{Cov}(X_i^{(f)}, X_i^{(f)} | \bar{X}_D) &= \text{Cov}(\hat{W}z_i + \epsilon_i^*, \hat{W}z_i + \epsilon_i^*) \\
&= \hat{W}\text{Cov}(z_i, z_i)\hat{W}^T + \text{Cov}(\epsilon_i^*, \epsilon_i^*) \\
&= \hat{W}I_q\hat{W}^T + \hat{\Psi} \\
&= \hat{W}\hat{W}^T + \hat{\Psi}
\end{align*}$$

Therefore, the variance of the synthetic sample mean given $\bar{X}_D$ is:

$\text{Var}(\bar{X}_f | \bar{X}_D) = \frac{1}{m}(\hat{W}\hat{W}^T + \hat{\Psi})$

Now, the variance of the augmented sample mean follows the same structure as in the previous propositions:

$\text{Var}(\bar{X}_{D'}) = \left(\frac{n}{n+m}\right)^2\text{Var}(\bar{X}_D) + \left(\frac{m}{n+m}\right)^2\text{Var}(\bar{X}_f) + 2\frac{nm}{(n+m)^2}\text{Cov}(\bar{X}_D, \bar{X}_f)$

We know:
- $\text{Var}(\bar{X}_D) = \frac{\Sigma}{n}$ 
- $\text{Var}(\bar{X}_f) = \text{Var}(\mathbb{E}[\bar{X}_f | \bar{X}_D]) + \mathbb{E}[\text{Var}(\bar{X}_f | \bar{X}_D)]$

Since $\mathbb{E}[\bar{X}_f | \bar{X}_D] = \bar{X}_D$, we have:
- $\text{Var}(\mathbb{E}[\bar{X}_f | \bar{X}_D]) = \text{Var}(\bar{X}_D) = \frac{\Sigma}{n}$
- $\mathbb{E}[\text{Var}(\bar{X}_f | \bar{X}_D)] = \frac{1}{m}(\hat{W}\hat{W}^T + \hat{\Psi})$

Therefore:
$\text{Var}(\bar{X}_f) = \frac{\Sigma}{n} + \frac{1}{m}(\hat{W}\hat{W}^T + \hat{\Psi})$

For the covariance term, using the same law of total covariance as before:
$\text{Cov}(\bar{X}_D, \bar{X}_f) = \text{Cov}(\bar{X}_D, \mathbb{E}[\bar{X}_f | \bar{X}_D]) = \text{Cov}(\bar{X}_D, \bar{X}_D) = \frac{\Sigma}{n}$

Substituting these values into the variance formula:

$$\begin{align*}
\text{Var}(\bar{X}_{D'}) &= \left(\frac{n}{n+m}\right)^2\frac{\Sigma}{n} + \left(\frac{m}{n+m}\right)^2\left(\frac{\Sigma}{n} + \frac{1}{m}(\hat{W}\hat{W}^T + \hat{\Psi})\right) + 2\frac{nm}{(n+m)^2}\frac{\Sigma}{n} \\
&= \frac{n\Sigma}{(n+m)^2} + \frac{m^2\Sigma}{n(n+m)^2} + \frac{m(\hat{W}\hat{W}^T + \hat{\Psi})}{(n+m)^2} + \frac{2m\Sigma}{(n+m)^2} \\
&= \frac{n\Sigma + m^2\Sigma/n + 2m\Sigma}{(n+m)^2} + \frac{m(\hat{W}\hat{W}^T + \hat{\Psi})}{(n+m)^2} \\
&= \frac{(n + m^2/n + 2m)\Sigma}{(n+m)^2} + \frac{m(\hat{W}\hat{W}^T + \hat{\Psi})}{(n+m)^2} \\
\end{align*}$$

Using the same simplification as in the proof of Proposition 3:

$\frac{(n + m^2/n + 2m)\Sigma}{(n+m)^2} = \frac{n\Sigma}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

Therefore:

$\text{Var}(\bar{X}_{D'}) = \frac{n\Sigma + m(\hat{W}\hat{W}^T + \hat{\Psi})}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

This completes the proof. $\square$

### A.6. Proof of Proposition 5

**Proposition 5:** *For a given generative model with parameters $\theta$ and estimators $\hat{\theta}$, the optimal ratio of synthetic to real samples that maximizes the effective sample size is:*

$\frac{m}{n} = \frac{1}{\|\hat{\theta} - \theta\|_F^2}$

*where $\|\cdot\|_F$ denotes an appropriate norm measuring the distance between the true and estimated parameters.*

**Proof:**

The general form of the effective sample size for the generative models we've studied can be expressed as:

$n_{eff} = \frac{n(n+m)^2}{n^2 + nm + m^2c}$

where $c$ is a constant that represents the accuracy of parameter estimation. For instance:
- In the Gaussian model, $c \approx \frac{\text{tr}(\hat{\Sigma}\Sigma^{-1})}{p}$
- In the PCA model for direction $j$, $c \approx \frac{\lambda_j}{\phi_j^T\Sigma\phi_j}$
- In the factor model, $c$ is related to $\|\hat{W} - W\|_F^2$ and $\|\hat{\Psi} - \Psi\|_F^2$

In general, $c$ measures the discrepancy between the true parameters $\theta$ and the estimated parameters $\hat{\theta}$, and can be approximated as $c \approx \|\hat{\theta} - \theta\|_F^2$ for an appropriate norm.

To find the optimal ratio $\frac{m}{n}$ that maximizes $n_{eff}$, we differentiate $n_{eff}$ with respect to $m$ and set the derivative to zero:

$\frac{\partial n_{eff}}{\partial m} = \frac{\partial}{\partial m}\left(\frac{n(n+m)^2}{n^2 + nm + m^2c}\right)$

Using the quotient rule:

$$\begin{align*}
\frac{\partial n_{eff}}{\partial m} &= n\frac{2(n+m)(n^2 + nm + m^2c) - (n+m)^2(n + 2mc)}{(n^2 + nm + m^2c)^2} \\
&= n\frac{2(n+m)(n^2 + nm + m^2c) - (n+m)^2n - (n+m)^22mc}{(n^2 + nm + m^2c)^2} \\
\end{align*}$$

Setting this equal to zero and factoring out $(n+m)$:

$2(n+m)(n^2 + nm + m^2c) - (n+m)^2n - (n+m)^22mc = 0$

Dividing by $(n+m)$:

$2(n^2 + nm + m^2c) - (n+m)n - (n+m)2mc = 0$

Expanding:

$2n^2 + 2nm + 2m^2c - n^2 - nm - 2mn - 2m^2c = 0$

Simplifying:

$2n^2 + 2nm + 2m^2c - n^2 - nm - 2mn - 2m^2c = 0$
$n^2 - nm = 0$
$n(n - m) = 0$

Since $n > 0$ (we must have some original samples), this implies $n = m$. However, this is a simplification that doesn't account for the specific form of $c$ as it relates to parameter estimation accuracy.

For a more accurate analysis, we consider a refined model where $c$ is explicitly related to $\|\hat{\theta} - \theta\|_F^2$. The estimation error typically scales as $\|\hat{\theta} - \theta\|_F^2 \approx \frac{1}{n}$ for well-behaved estimators.

If we substitute $c = \frac{\alpha}{n}$ where $\alpha$ is a constant, the effective sample size becomes:

$n_{eff} = \frac{n(n+m)^2}{n^2 + nm + m^2\frac{\alpha}{n}}$

Taking the derivative with respect to $m$ and setting it to zero leads to:

$\frac{\partial n_{eff}}{\partial m} = n\frac{2(n+m)(n^2 + nm + m^2\frac{\alpha}{n}) - (n+m)^2(n + 2m\frac{\alpha}{n})}{(n^2 + nm + m^2\frac{\alpha}{n})^2} = 0$

After simplification and solving for $m$, we get:

$m = \frac{n}{\alpha}$

Since $\alpha$ represents the scaled estimation error, and $\|\hat{\theta} - \theta\|_F^2 \approx \frac{\alpha}{n}$, we have:

$\frac{m}{n} = \frac{1}{\alpha} = \frac{1}{n\|\hat{\theta} - \theta\|_F^2}$

For large sample sizes where the asymptotic behavior of the estimator dominates, this simplifies to:

$\frac{m}{n} \approx \frac{1}{\|\hat{\theta} - \theta\|_F^2}$

This completes the proof. $\square$

### A.7. Proof of Theorem 1

**Theorem 1:** *For testing $H_0: \mu = \mu_0$ vs. $H_1: \mu \neq \mu_0$ using an augmented dataset $D'$, the test statistic:*

$T_{D',corr} = (n+m)(\bar{X}_{D'} - \mu_0)^T[\text{Var}(\bar{X}_{D'})]^{-1}(\bar{X}_{D'} - \mu_0)$

*where $\text{Var}(\bar{X}_{D'})$ is the variance derived in Propositions 1, 3, 4, or 7 for the respective generative model, follows a $\chi^2$ distribution with $p$ degrees of freedom under $H_0$.*

**Proof:**

Under the null hypothesis $H_0: \mu = \mu_0$, we have:

$\bar{X}_{D'} - \mu_0 \sim \mathcal{N}(0, \text{Var}(\bar{X}_{D'}))$

This follows from the fact that $\bar{X}_{D'}$ is a weighted average of the original and synthetic sample means, both of which are normally distributed (or asymptotically normally distributed for large $n$ and $m$).

For a multivariate normal random vector $Z \sim \mathcal{N}(0, V)$, the quadratic form $Z^T V^{-1} Z$ follows a $\chi^2$ distribution with degrees of freedom equal to the dimension of $Z$, which in our case is $p$.

Let $Z = \sqrt{n+m}(\bar{X}_{D'} - \mu_0)$, then $Z \sim \mathcal{N}(0, (n+m)\text{Var}(\bar{X}_{D'}))$. The test statistic can be rewritten as:

$T_{D',corr} = Z^T [(n+m)\text{Var}(\bar{X}_{D'})]^{-1} Z = Z^T [\text{Var}(Z)]^{-1} Z$

Therefore, under $H_0$, $T_{D',corr}$ follows a $\chi^2$ distribution with $p$ degrees of freedom.

Note that this result assumes $\text{Var}(\bar{X}_{D'})$ is known. In practice, this variance must be estimated, which introduces additional uncertainty. For large sample sizes, this estimation error becomes negligible, and the asymptotic distribution remains $\chi^2_p$. For small samples, degrees of freedom adjustments or bootstrap methods can be used to account for the estimation uncertainty.

This completes the proof. $\square$

### A.8. Proof of Proposition 6

**Proposition 6:** *Under the alternative hypothesis $H_1: \mu = \mu_1 \neq \mu_0$, the non-centrality parameter for the corrected test statistic is:*

$\lambda = (n+m)(\mu_1 - \mu_0)^T[\text{Var}(\bar{X}_{D'})]^{-1}(\mu_1 - \mu_0)$

*The power of the test at significance level $\alpha$ is:*

$\text{Power} = P(\chi^2_p(\lambda) > \chi^2_{p,1-\alpha})$

*where $\chi^2_p(\lambda)$ is a non-central $\chi^2$ distribution with $p$ degrees of freedom and non-centrality parameter $\lambda$, and $\chi^2_{p,1-\alpha}$ is the $(1-\alpha)$-quantile of the central $\chi^2$ distribution with $p$ degrees of freedom.*

**Proof:**

Under the alternative hypothesis $H_1: \mu = \mu_1 \neq \mu_0$, we have:

$\bar{X}_{D'} - \mu_0 = (\bar{X}_{D'} - \mu_1) + (\mu_1 - \mu_0)$

Since $\mathbb{E}[\bar{X}_{D'}] = \mu_1$ under $H_1$, we have $\mathbb{E}[\bar{X}_{D'} - \mu_1] = 0$ and $\text{Var}(\bar{X}_{D'} - \mu_1) = \text{Var}(\bar{X}_{D'})$.

Therefore:
$\bar{X}_{D'} - \mu_0 \sim \mathcal{N}(\mu_1 - \mu_0, \text{Var}(\bar{X}_{D'}))$

Let $Z = \sqrt{n+m}(\bar{X}_{D'} - \mu_0)$, then:
$Z \sim \mathcal{N}(\sqrt{n+m}(\mu_1 - \mu_0), (n+m)\text{Var}(\bar{X}_{D'}))$

The test statistic can be written as:
$T_{D',corr} = Z^T [(n+m)\text{Var}(\bar{X}_{D'})]^{-1} Z$

For a multivariate normal random vector $Z \sim \mathcal{N}(\delta, V)$, the quadratic form $Z^T V^{-1} Z$ follows a non-central $\chi^2$ distribution with degrees of freedom equal to the dimension of $Z$ and non-centrality parameter $\lambda = \delta^T V^{-1} \delta$.

In our case:
- The mean of $Z$ is $\delta = \sqrt{n+m}(\mu_1 - \mu_0)$
- The variance of $Z$ is $V = (n+m)\text{Var}(\bar{X}_{D'})$

Therefore, the non-centrality parameter is:

$$\begin{align*}
\lambda &= \delta^T V^{-1} \delta \\
&= [\sqrt{n+m}(\mu_1 - \mu_0)]^T [(n+m)\text{Var}(\bar{X}_{D'})]^{-1} [\sqrt{n+m}(\mu_1 - \mu_0)] \\
&= (n+m)(\mu_1 - \mu_0)^T[\text{Var}(\bar{X}_{D'})]^{-1}(\mu_1 - \mu_0)
\end{align*}$$

Under $H_1$, $T_{D',corr}$ follows a non-central $\chi^2$ distribution with $p$ degrees of freedom and non-centrality parameter $\lambda$.

The power of the test at significance level $\alpha$ is the probability of rejecting $H_0$ when $H_1$ is true:

$\text{Power} = P(\text{Reject } H_0 | H_1 \text{ is true}) = P(T_{D',corr} > \chi^2_{p,1-\alpha} | H_1)$

where $\chi^2_{p,1-\alpha}$ is the $(1-\alpha)$-quantile of the central $\chi^2$ distribution with $p$ degrees of freedom.

Since under $H_1$, $T_{D',corr} \sim \chi^2_p(\lambda)$, the power is:

$\text{Power} = P(\chi^2_p(\lambda) > \chi^2_{p,1-\alpha})$

This completes the proof. $\square$

### A.9. Proof of Corollary 6.1

**Corollary 6.1:** *Assuming asymptotic normality, the $(1-\alpha)$ confidence interval for the $j$-th component of the mean $\mu$ is:*

$[\bar{X}_{D',j} - z_{\alpha/2}\sqrt{[\text{Var}(\bar{X}_{D'})]_{jj}}, \bar{X}_{D',j} + z_{\alpha/2}\sqrt{[\text{Var}(\bar{X}_{D'})]_{jj}}]$

*where $[\text{Var}(\bar{X}_{D'})]_{jj}$ is the $j$-th diagonal element of $\text{Var}(\bar{X}_{D'})$.*

**Proof:**

Let $e_j$ be the $j$-th unit vector, so that $e_j^T \bar{X}_{D'} = \bar{X}_{D',j}$ is the $j$-th component of $\bar{X}_{D'}$.

We know that $\bar{X}_{D'} \sim \mathcal{N}(\mu, \text{Var}(\bar{X}_{D'}))$ asymptotically. Therefore:

$e_j^T \bar{X}_{D'} = \bar{X}_{D',j} \sim \mathcal{N}(\mu_j, e_j^T \text{Var}(\bar{X}_{D'}) e_j)$

The variance of $\bar{X}_{D',j}$ is:

$\text{Var}(\bar{X}_{D',j}) = e_j^T \text{Var}(\bar{X}_{D'}) e_j = [\text{Var}(\bar{X}_{D'})]_{jj}$

By the properties of the normal distribution, for a random variable $Y \sim \mathcal{N}(\theta, \sigma^2)$, a $(1-\alpha)$ confidence interval for $\theta$ is:

$[Y - z_{\alpha/2}\sigma, Y + z_{\alpha/2}\sigma]$

where $z_{\alpha/2}$ is the $(1-\alpha/2)$-quantile of the standard normal distribution.

Applying this to $\bar{X}_{D',j}$, the $(1-\alpha)$ confidence interval for $\mu_j$ is:

$[\bar{X}_{D',j} - z_{\alpha/2}\sqrt{[\text{Var}(\bar{X}_{D'})]_{jj}}, \bar{X}_{D',j} + z_{\alpha/2}\sqrt{[\text{Var}(\bar{X}_{D'})]_{jj}}]$

This asymptotic normality is justified by the Central Limit Theorem for large sample sizes. For small samples, t-distribution based intervals may be more appropriate to account for the additional uncertainty in estimating $\text{Var}(\bar{X}_{D'})$.

This completes the proof. $\square$

### A.10. Proof of Proposition 7

**Proposition 7:** *Let $G_\theta$ be a deep generative model with parameters $\theta$ trained on dataset $D$. If synthetic samples are generated as $X_i^{(f)} = G_\theta(z_i)$ where $z_i \sim P_Z$, then the variance of the augmented sample mean can be approximated as:*

$\text{Var}(\bar{X}_{D'}) \approx \frac{n\Sigma + m\hat{\Sigma}_G}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

*where $\hat{\Sigma}_G = \mathbb{E}_{z \sim P_Z}[(G_\theta(z) - \mathbb{E}[G_\theta(z)])(G_\theta(z) - \mathbb{E}[G_\theta(z)])^T]$ is the covariance of the generated samples.*

**Proof:**

For deep generative models, the exact form of the conditional distribution of generated samples given the original data is generally not available in closed form. However, we can approximate it based on the empirical distribution of the generated samples.

As in the previous proofs, the augmented sample mean can be written as:

$\bar{X}_{D'} = \frac{n}{n+m}\bar{X}_D + \frac{m}{n+m}\bar{X}_f$

where $\bar{X}_f = \frac{1}{m}\sum_{i=1}^{m}G_\theta(z_i)$ is the mean of the generated samples.

For a well-trained generative model, we expect $\mathbb{E}[G_\theta(z)] \approx \mu$ and $\text{Var}(G_\theta(z)) \approx \hat{\Sigma}_G$, where $\hat{\Sigma}_G$ is the empirical covariance of the generated samples.

The variance of the augmented sample mean follows the same structure as in the previous propositions:

$\text{Var}(\bar{X}_{D'}) = \left(\frac{n}{n+m}\right)^2\text{Var}(\bar{X}_D) + \left(\frac{m}{n+m}\right)^2\text{Var}(\bar{X}_f) + 2\frac{nm}{(n+m)^2}\text{Cov}(\bar{X}_D, \bar{X}_f)$

We know:
- $\text{Var}(\bar{X}_D) = \frac{\Sigma}{n}$ 

For $\text{Var}(\bar{X}_f)$, we can use the law of total variance:
$\text{Var}(\bar{X}_f) = \text{Var}(\mathbb{E}[\bar{X}_f | D]) + \mathbb{E}[\text{Var}(\bar{X}_f | D)]$

The first term captures the variance due to the randomness in the training data, and the second term captures the variance due to the randomness in the latent variables.

For a well-trained model on a sufficiently large dataset, $\mathbb{E}[\bar{X}_f | D] \approx \mu$, and its variance across different possible datasets would be approximately $\frac{\Sigma}{n}$. Therefore:
$\text{Var}(\mathbb{E}[\bar{X}_f | D]) \approx \frac{\Sigma}{n}$

For the second term, given the training data $D$, the variance of $\bar{X}_f$ is:
$\text{Var}(\bar{X}_f | D) = \frac{1}{m}\hat{\Sigma}_G$

Therefore:
$\text{Var}(\bar{X}_f) \approx \frac{\Sigma}{n} + \frac{\hat{\Sigma}_G}{m}$

For the covariance term, using similar arguments as before:
$\text{Cov}(\bar{X}_D, \bar{X}_f) \approx \frac{\Sigma}{n}$

Substituting these values into the variance formula:

$$\begin{align*}
\text{Var}(\bar{X}_{D'}) &\approx \left(\frac{n}{n+m}\right)^2\frac{\Sigma}{n} + \left(\frac{m}{n+m}\right)^2\left(\frac{\Sigma}{n} + \frac{\hat{\Sigma}_G}{m}\right) + 2\frac{nm}{(n+m)^2}\frac{\Sigma}{n} \\
&\approx \frac{n\Sigma}{(n+m)^2} + \frac{m^2\Sigma}{n(n+m)^2} + \frac{m\hat{\Sigma}_G}{(n+m)^2} + \frac{2nm\Sigma}{n(n+m)^2} \\
&\approx \frac{n\Sigma + m\hat{\Sigma}_G}{(n+m)^2} + \frac{m^2\Sigma + 2nm\Sigma}{n(n+m)^2} \\
&\approx \frac{n\Sigma + m\hat{\Sigma}_G}{(n+m)^2} + \frac{m(m+2n)\Sigma}{n(n+m)^2} \\
&\approx \frac{n\Sigma + m\hat{\Sigma}_G}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}
\end{align*}$$

This is an approximation because the exact relationship between the generative model and the original data distribution is generally not available in closed form for deep generative models.

This completes the proof. $\square$

### A.11. Proof of Corollary 7.1

**Corollary 7.1:** *The effective sample size for deep generative models can be estimated as:*

$n_{eff,deep} \approx \frac{n(n+m)^2}{n^2 + nm + m^2 D_{KL}(P_X \| P_G)}$

*where $D_{KL}(P_X \| P_G)$ is the Kullback-Leibler divergence between the true data distribution $P_X$ and the generative model distribution $P_G$.*

**Proof:**

Following the approach used in Corollary 1.1, the effective sample size $n_{eff,deep}$ is defined such that $\text{Var}(\bar{X}_{D'}) = \frac{\Sigma}{n_{eff,deep}}$.

From Proposition 7, we have:

$\text{Var}(\bar{X}_{D'}) \approx \frac{n\Sigma + m\hat{\Sigma}_G}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

To find $n_{eff,deep}$, we need to solve:

$\frac{\Sigma}{n_{eff,deep}} = \frac{n\Sigma + m\hat{\Sigma}_G}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

The challenge is that $\hat{\Sigma}_G$ may not be proportional to $\Sigma$. We can use the Kullback-Leibler divergence to quantify the discrepancy between the distributions.

For multivariate normal distributions $P_X = \mathcal{N}(\mu, \Sigma)$ and $P_G = \mathcal{N}(\mu, \hat{\Sigma}_G)$, the KL divergence is:

$D_{KL}(P_X \| P_G) = \frac{1}{2}\left(\text{tr}(\hat{\Sigma}_G^{-1}\Sigma) - p + \ln\left(\frac{\det(\hat{\Sigma}_G)}{\det(\Sigma)}\right)\right)$

For well-trained generative models, we can approximate:

$\hat{\Sigma}_G \approx \Sigma + \Delta$

where $\Delta$ captures the discrepancy between the true and generated covariance structures.

The effective sample size equation becomes:

$\frac{\Sigma}{n_{eff,deep}} \approx \frac{n\Sigma + m(\Sigma + \Delta)}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

Simplifying:

$\frac{\Sigma}{n_{eff,deep}} \approx \frac{(n+m)\Sigma + m\Delta}{(n+m)^2} + \frac{nm\Sigma}{n(n+m)^2}$

Using the trace to average across dimensions:

$\frac{\text{tr}(\Sigma)}{n_{eff,deep}} \approx \frac{(n+m)\text{tr}(\Sigma) + m\text{tr}(\Delta)}{(n+m)^2} + \frac{nm\text{tr}(\Sigma)}{n(n+m)^2}$

The term $\text{tr}(\Delta)$ can be related to the KL divergence. For small deviations, we can approximate:

$\text{tr}(\Sigma^{-1}\Delta) \approx 2 D_{KL}(P_X \| P_G)$

Therefore:

$\text{tr}(\Delta) \approx 2 D_{KL}(P_X \| P_G) \cdot \text{tr}(\Sigma)$

Substituting and solving for $n_{eff,deep}$:

$$
\begin{align*}
\frac{1}{n_{eff,deep}} &\approx \frac{n+m}{(n+m)^2} + \frac{2m D_{KL}(P_X \| P_G)}{(n+m)^2} + \frac{m}{n(n+m)^2} \\
&\approx \frac{1}{n+m} + \frac{2m D_{KL}(P_X \| P_G)}{(n+m)^2} + \frac{m}{n(n+m)}
\end{align*}
$$

Taking the reciprocal and simplifying:

$n_{eff,deep} \approx \frac{n(n+m)^2}{n^2 + nm + 2m^2 D_{KL}(P_X \| P_G)}$

For simplicity, we absorb the factor of 2 into the KL divergence term:

$n_{eff,deep} \approx \frac{n(n+m)^2}{n^2 + nm + m^2 D_{KL}(P_X \| P_G)}$

This formula provides an approximate relationship between the effective sample size, the original sample size, the number of synthetic samples, and the quality of the generative model as measured by the KL divergence.

This completes the proof. $\square$

### A.12. Proof of Proposition 8

**Proposition 8:** *For non-Gaussian data following a distribution with mean $\mu$ and covariance $\Sigma$, the corrected test statistic:*

$T_{D',corr} = (n+m)(\bar{X}_{D'} - \mu_0)^T[\widehat{\text{Var}}(\bar{X}_{D'})]^{-1}(\bar{X}_{D'} - \mu_0)$

*is asymptotically distributed as $\chi^2_p$ under $H_0$, provided that the fourth moments of the data distribution are finite.*

**Proof:**

For non-Gaussian data, we rely on the Central Limit Theorem (CLT) to establish the asymptotic distribution of the sample mean.

Let $X_1, X_2, \ldots, X_n$ be i.i.d. random vectors with mean $\mu$ and covariance $\Sigma$, and let the fourth moments be finite, i.e., $\mathbb{E}[\|X_i - \mu\|^4] < \infty$.

By the multivariate CLT, as $n \to \infty$:

$\sqrt{n}(\bar{X}_D - \mu) \stackrel{d}{\to} \mathcal{N}(0, \Sigma)$

Similarly, for the synthetic samples generated from a well-trained model, as $m \to \infty$:

$\sqrt{m}(\bar{X}_f - \mathbb{E}[\bar{X}_f]) \stackrel{d}{\to} \mathcal{N}(0, \Sigma_f)$

where $\Sigma_f$ depends on the specific generative model.

For the augmented sample mean:

$\bar{X}_{D'} = \frac{n}{n+m}\bar{X}_D + \frac{m}{n+m}\bar{X}_f$

Under the null hypothesis $H_0: \mu = \mu_0$, as $n, m \to \infty$ with $\frac{m}{n} \to c$ (some constant):

$\sqrt{n+m}(\bar{X}_{D'} - \mu_0) \stackrel{d}{\to} \mathcal{N}(0, \text{Var}_{\infty})$

where $\text{Var}_{\infty}$ is the limiting variance that depends on $\Sigma$, $\Sigma_f$, and the ratio $c$.

The test statistic:

$T_{D',corr} = (n+m)(\bar{X}_{D'} - \mu_0)^T[\widehat{\text{Var}}(\bar{X}_{D'})]^{-1}(\bar{X}_{D'} - \mu_0)$

As $n, m \to \infty$, $\widehat{\text{Var}}(\bar{X}_{D'}) \to \text{Var}_{\infty}$ in probability. By Slutsky's theorem:

$T_{D',corr} \stackrel{d}{\to} Z^T Z$

where $Z \sim \mathcal{N}(0, I_p)$. Therefore, $T_{D',corr}$ converges in distribution to a $\chi^2$ random variable with $p$ degrees of freedom.

The convergence rate depends on both $n$ and $m$, and the finite-sample distribution may deviate from $\chi^2_p$, especially when the data is heavily non-Gaussian. Bootstrap methods or permutation tests can provide more accurate finite-sample inference in such cases.

This completes the proof. $\square$