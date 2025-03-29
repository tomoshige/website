# What is this project
このプロジェクトでは、統計的因果推論で用いられるデータセットについて、その内容、前処理、論文での使用のされ方についてまとめています。現状、raw データを提供しているけど、その後の論文に基づいた加工方法が一緒に載っていないサイトなどが多く、結局使用方法がわからないなどの問題が多く見受けられます。このページは、その問題に対処し、さまざまな人がBenchmarkデータを利用可能なように整備することを1つの目的としています。

### Right heart catheterization (RHC)
Right heart catheterization (RHC) is a diagnostic procedure for directly measuring cardiac function in
critically ill patients. In an influential study, Connors et al. (1996) studied the effectiveness of RHC with an
observational study design. The study collected data on 5735 hospitalized adult patients; 2184 of them are
assigned to the treatment (Z = 1), receipt of RHC within 24 hours of admission, and the remaining 3551
assigned to the control condition (Z = 0). The outcome was survival at 30 days after admission. The goal is
to assess the causal effect of RHC on the binary outcome, death at 30 days after admission.
[Link](https://www2.stat.duke.edu/~fl35/teaching/640/labs/lab-2-PS-binary.pdf)


### Natural GAS Compressor
This is the code to conduct the simulations and data analysis in the paper "Estimating Population Average Causal Effects in the Presence of Non-Overlap: The Effect of Natural Gas Compressor Station Exposure on Cancer Mortality". The folder simulations_main provides code to conduct the simulations in Section 3 of the main manuscript. The folder simulation_supp provides code to conduct the simulations in Section 2 of the Supplementary Materials. The folder natgas_analysis contains code and data to reproduce the data analysis in Section 4 of the main manuscript. Each simulation requires the user-specified command line arguments shown at the head of the file.

- Github : [Link](https://github.com/rachelnethery/overlap)
- Paper : [ESTIMATING POPULATION AVERAGE CAUSAL EFFECTS IN THE PRESENCE OF NON-OVERLAP: THE EFFECT OF NATURAL GAS COMPRESSOR STATION EXPOSURE ON CANCER MORTALITY](https://pmc.ncbi.nlm.nih.gov/articles/PMC6658123/)

### Extreme Overlap
- [Addressing Extreme Propensity Scores via the Overlap Weights](https://public.econ.duke.edu/~vjh3/working_papers/overlap.pdf)
- [Dealing with limited overlap in estimation of average treatment effects](https://public.econ.duke.edu/~vjh3/working_papers/overlap.pdf)
- [Doing Great at Estimating CATE? On the Neglected Assumptions in Benchmark Comparisons of Treatment Effect Estimators](https://arxiv.org/pdf/2107.13346)
- [Addressing Extreme Propensity Scores in Estimating Counterfactual Survival Functions via the Overlap Weights](https://arxiv.org/pdf/2108.04394)

### Isolation Forest
- [Link](https://qiita.com/tchih11/items/d76a106e742eb8d92fb4)

### Awesome Causal Dataset
- [Gihub](https://github.com/rguo12/awesome-causality-data?tab=readme-ov-file)

### RealCalse
- [ArXiv](https://arxiv.org/pdf/2011.15007)
- [Github](https://github.com/bradyneal/realcause)

### Paper with code
- [Website](https://paperswithcode.com/datasets?task=causal-inference)