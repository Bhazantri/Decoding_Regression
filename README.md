# Decoding_Regression
PyTorch-based implementation of the decoding-based regression framework introduced by Song and Bahri (Google DeepMind, 2025).
Decoding_Regression is a comprehensive PyTorch-based implementation of the decoding-based regression framework introduced by Song and Bahri (Google DeepMind, 2025). This repository encapsulates a novel approach to regression tasks where numeric outputs are tokenized into discrete sequences and modeled autoregressively using transformer decoders, contrasting with traditional pointwise and histogram-based regression heads. The implementation rigorously adheres to the paper's specifications, including tokenization schemes, model architectures, theoretical underpinnings, and empirical validations across synthetic and real-world datasets. It targets researchers and practitioners in machine learning seeking to explore flexible, distribution-agnostic regression methods leveraging language model-inspired techniques.

# Technical Foundation 

The core innovation lies in representing continuous numeric targets 𝑦∈𝑅y∈R as sequences of discrete tokens (𝑡1,𝑡2,…,𝑡𝐾)∈𝑉𝐾(t 1​ ,t 2 ,…,t K​ )∈V K , where 𝑉V is afinite vocabulary (e.g., {0,1…,𝐵−1}{0,1,…,B−1} for base-𝐵B). This enables autoregressive modeling via 𝑝𝜃(𝑦∣𝑥)=∏𝑘=1𝐾𝑝𝜃(𝑡𝑘∣𝜙(𝑥),𝑡1,…,𝑡𝑘−1)p θ​ (y∣x)=∏ k=1K​ p θ​ (t k​ ∣ϕ(x),t 1 ,…,t k−1 ), where 𝜙(𝑥) ϕ(x) is a feature representation from an encoder, typically a multi-layer perceptron (MLP). The approach contrasts with conventional regression heads:

1. Pointwise Head: Outputs a scalar 𝑦^=𝑓(𝜙(𝑥))y​ =f(ϕ(x)) optimized via mean squared error (MSE), ℓ=(𝑦−𝑦^)2ℓ=(y− ^​ ) 2 , assuming a unimodal distribution.

2. Histogram (Riemann) Head: Models 𝑝(𝑦∣𝑥)p(y∣x) as a piecewise-constant distribution over 𝑛n bins, parameterized by softmax over 𝜙(𝑥)𝑇𝑊ϕ(x) T W, with 𝑝(𝑦𝑖∣𝑥)=Softmax(𝑖)(𝜙(𝑥)𝑇𝑊)p(y i ∣x)=Softmax (i) (ϕ(x)  W).

3. Decoding-based Head: Employs a transformer decoder to predict token sequences, trained with cross-entropy loss, 𝐻(𝑦,𝑝𝜃)=−∑𝑘=1∑𝑡𝑘∈𝑉𝐼(𝑡^𝑘=𝑡𝑘)log𝑝𝜃(𝑡^𝑘∣𝑡<𝑘,𝜙(𝑥))H(y,p ​
 )=−∑ k=1K​ ∑ t k​ ∈V​ I( t^  k​ =t k​ )logp θ​ ( t^  k ∣t <,ϕ(x)), enabling flexible density estimation over 𝑅R.


The paper’s theoretical contribution (Theorem 1) is validated empirically, asserting that for a 
𝐾K-bit universal model, the risk 𝑅(𝑓,𝑓^𝑁𝑘)=𝐸∫01(𝑓(𝑦)−𝑓^𝑁𝑘(𝑦))2𝑑𝑦R(f, f^​  Nk​ )=E∫ 01​ (f(y)− f^​  Nk​ (y)) 2 dy scales as 2−2𝑘12∫01𝑓′(𝑦)2𝑑𝑦+2𝑘𝑁122 −2k ​ ∫ 01​ f ′ (y) 2dy+ N2k, balancing bias and variance trade-offs as 𝐾K (sequence length) and 𝑁N (sample size) vary.


# Tokenization Schemes
The repository implements three tokenization strategies:

Normalized Tokenization: Maps 𝑦∈[y⁡,𝑦max⁡]y∈[y min​ ,y max​ ] to [0,1][0,1] via 𝑦′=(y-ymin)/𝑦max⁡−𝑦min)y ′=(y−y min​ )/(y max​ −y min​ ), then represents 𝑦′y ′  as abase𝐵B expansion up to length 𝐾K. For example, 𝑦′=0.123y ′ =0.123 with 𝐵=10,𝐾=4B=10,K=4 yields ⟨1⟩⟨2⟩⟨3⟩⟨0⟩⟨1⟩⟨2⟩⟨3⟩⟨0⟩. Detokenization reconstructs 𝑦=∑𝑖=1𝐾𝑡𝑖𝐵−𝑖⋅(𝑦max⁡−𝑦min⁡)+𝑦min⁡y=∑ i=1K​ t i B −i ⋅(y max −y min)+y min​ .

Unnormalized Tokenization: Generalizes IEEE-754 floating-point representation to base-𝐵B, encoding 𝑦=𝑠⋅𝐵𝑒⋅𝑚y=s⋅B e ⋅m where 𝑠∈{−1,+1}s∈{−1,+1}, 𝑒∈𝑍e∈Z, and 𝑚∈[0,𝐵)m∈[0,B). Tokens are ⟨𝑠⟩⟨𝑠𝑒⟩⟨𝑒1⟩…⟨𝑒𝐸⟩⟨𝑚1⟩…⟨𝑚𝑀⟩⟨s⟩⟨s e ⟩⟨e 1 ⟩…⟨e E ⟩⟨m 1 ⟩…⟨m M ⟩, with 𝐸E and 𝑀M as exponent and mantissa lengths (e.g., 10−222⋅1.23410 −222⋅1.234 as ⟨+⟩⟨−⟩⟨2⟩⟨2⟩⟨2⟩⟨1⟩⟨2⟩⟨3⟩⟨4⟩⟨+⟩⟨−⟩⟨2⟩⟨2⟩⟨2⟩⟨1⟩⟨2⟩⟨3⟩⟨4⟩ for 𝐵=10,𝐸=3,𝑀=4B=10,E=3,M=4).

Hamming Distance-based Tokenization (Appendix A.3): Encodes 𝑦∈[0,1]y∈[0,1] into binary sequences with bounded distortion under bitwise edits, e.g., {0,1,…,7}/7{0,1,…,7}/7 mapped to {(000),(001),(010),(100),(111),(101),(011),(110)} {(000),(001),(010),(100),(111),(101),(011),(110)}, though less performant in practice.


# Model Architectures

1. Encoder: An MLP with configurable layers (𝐿∈[2,5]L∈[2,5]) and hidden units (𝐻∈[256,2048]H∈[256,2048]), applying ReLU activations, transforming input 𝑥∈𝑅𝐷x∈R D𝜙
(𝑥)∈𝑅𝐻ϕ(x)∈R H .Pointwise Head: Linear projection 𝜙(𝑥)→𝑅ϕ(x)→R with sigmoid activation to enforce [0,1][0,1] outputs.Histogram Head: Projects 𝜙(𝑥)→𝑅𝑛ϕ(x)→R n ( 𝑛∈[16,16384]n∈[16,16384] bins), computes softmax probabilities, and returns the expected value over bin centers.

2.Decoder Head: A transformer decoder with 𝐿∈[1,5]L∈[1,5] layers, 𝐻∈[32,256]H∈[32,256] hidden units, and 𝑁ℎ∈[1,8]N h​ ∈[1,8] attention heads, predicting token sequences autoregressively. Supports constrained decoding for valid numeric representations.

3. Mixture Density Network (MDN) Head: Outputs mixture weights 𝜋∈Δ𝑀π∈Δ M , means 𝜇∈𝑅𝑀μ∈R M , and variances 𝜎2∈𝑅+𝑀σ 2 ∈R +M​  (via ELU+1 activation), modeling 𝑝(𝑦∣𝑥)=∑𝑚=1𝑀𝜋𝑚𝑁(𝑦;𝜇𝑚,𝜎𝑚2)p(y∣x)=∑ m=1M​ π m​ N(y;μ m ,σ m2 ).
  

 Training and Optimization
 
 1. Loss Functions: MSE for pointwise/histogram, cross-entropy for decoder, and negative log-likelihood for MDN.

 2. Optimizer: Adam with learning rates 𝜂∈[10−4,5⋅10−4]η∈[10 −4 ,5⋅10 −4 ], weight decay 𝜆∈[0,1]λ∈[0,1].Early Stopping: Patience of 5 epochs on validation loss (10% 
   split).

3. Normalization: 𝑥x-coordinates scaled via (𝑥−𝜇𝑥)/𝜎𝑥(x−μ x​ )/σ x​ ; 𝑦y-values via min-max scaling to [0,1][0,1] or shifted to [−0.5,0.5][−0.5,0.5] for 
  pointwise/MDN heads.

# Experiments

The repository replicates all experiments from Sections 4.1–4.5 and Appendices A.1–A.5:

Synthetic Curve Fitting (Section 4.1):
1. Functions: 𝑦=sin⁡(𝑥)y=sin(x), 𝑦=𝑒−𝑥2y=e −x 2  , 𝑦=𝑥2y=x 2  with Gaussian noise 𝑁(0,0.1)N(0,0.1).
   Evaluates MSE and visual fit, showcasing decoder’s ability to handle high Lipschitz constants and wide ranges.


Real-World Regression (Section 4.2):
1, Datasets: UCI (e.g., Airfoil, Housing) and OpenML-CTR23/AMLB subsets.
2. Metrics: MSE and Kendall-Tau scores, comparing data efficiency across regimes (𝑁∈[10,04]N∈[10,10 4 ]).


Density Estimation (Section 4.3):

Visualizes 𝑝(𝑦∣𝑥) p(y∣x) fits (e.g., bimodal distributions) using vanilla temperature sampling (𝑇≈1.0T≈1.0).
Computes negative log-likelihood (NLL) on UCI datasets, benchmarking against MDN and Riemann heads.

Ablation: Decoder Size (Section 4.4):Sweeps 𝐿∈[1,5]L∈[1,5],𝑁ℎ∈[1,8]N h​ ∈[1,8], 𝐻∈[32,256]H∈[32,256], assessing overfitting and NLL trade-offs.

Ablation: Error Correction (Section 4.5):Implements repetition-based error correction (e.g., repeat 𝑟∈[1,5r∈[1,5] times, majority voting), reducing outlier sensitivity in unnormalized decoding.

# Implementation Details
Dependencies: PyTorch 2.0+, NumPy, Matplotlib, Scikit-learn, Pandas, Requests.
Structure: Modular design with src/ for core logic, experiments/ for scripts, and data/ for datasets.
Hyperparameters: Fully configurable per Appendix D, e.g., 𝐵∈[2,10]B∈[2,10], 𝐾∈[4,8]K∈[4,8], 𝐸∈[1,4]
E∈[1,4], 𝑀∈[2,8]M∈[2,8], 𝑀MDN∈[1,1000]M MDN∈[1,1000].

Outputs: Results saved in results/ with plots and logs for analysis.

# Install dependencies
pip install -r requirements.txt

# Download UCI datasets
python data/uci_datasets.py

# Run experiments
python experiments/synthetic.py
python experiments/real_world.py
python experiments/density_est.py
python experiments/ablation_size.py
python experiments/ablation_error.py

Validation and Future Work
The implementation validates the paper’s claims of competitiveness (e.g., decoder MSE often below pointwise/histogram) and density estimation flexibility (NLL < 0.7 on UCI tasks). Future extensions could explore alternative tokenizations (e.g., Stern-Brocot trees), multi-target regression, or convolutional encoders for vision tasks, as suggested in Section 5.

This repository serves as a technical cornerstone for advancing autoregressive regression research, bridging language modeling and numeric prediction with a robust, reproducible framework.
