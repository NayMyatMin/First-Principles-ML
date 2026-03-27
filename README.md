# First-Principles-ML

A comprehensive, from-scratch machine learning repository for deep understanding and coding interview preparation. Every algorithm is implemented from first principles using NumPy, with PyTorch equivalents where applicable.

## Philosophy

- **Implement, don't import.** Understand what happens inside `sklearn.fit()`.
- **Intuition first.** Every notebook starts with *why* the algorithm works, not just *how*.
- **Interview-ready.** Code is clean enough to write on a whiteboard (or shared screen).
- **Depth over breadth.** Fewer algorithms, fully understood, beats a shallow survey.

## Repository Structure

### [`01-mathematical-foundations/`](01-mathematical-foundations/)
The computational building blocks that underpin all of ML.

| Notebook | Topics |
|----------|--------|
| `01_autograd_engine.ipynb` | Computational graphs, reverse-mode autodiff, building a minimal autograd |
| `02_optimization_algorithms.ipynb` | SGD, Momentum, Nesterov, RMSProp, Adam — implement and visualize each |
| `03_probability_and_estimation.ipynb` | MLE, MAP, sampling methods, conjugate priors |
| `04_information_theory.ipynb` | Entropy, cross-entropy, KL divergence, mutual information |

### [`02-classical-ml-from-scratch/`](02-classical-ml-from-scratch/)
The core section. Deep NumPy implementations with strong intuition and interview context.

| Notebook | Topics |
|----------|--------|
| `01_linear_regression.ipynb` | Normal equation, gradient descent, regularization (Ridge, Lasso, ElasticNet) |
| `02_logistic_regression.ipynb` | Binary & multinomial, decision boundaries, Newton's method |
| `03_softmax_regression.ipynb` | Multi-class extension, numerical stability, connection to logistic |
| `04_knn.ipynb` | Brute force, KD-trees, distance metrics, curse of dimensionality |
| `05_naive_bayes.ipynb` | Gaussian, Multinomial, Bernoulli — when & why it works despite the "naive" assumption |
| `06_decision_trees.ipynb` | ID3, CART, Gini vs entropy, pruning strategies |
| `07_random_forest.ipynb` | Bagging, feature randomness, OOB error, feature importance |
| `08_gradient_boosting.ipynb` | Additive modeling, shrinkage, mini-XGBoost from scratch |
| `09_svm.ipynb` | Hard/soft margin, kernel trick, SMO algorithm, dual formulation |
| `10_pca_svd.ipynb` | Eigendecomposition, SVD, variance explained, kernel PCA |
| `11_kmeans.ipynb` | Lloyd's algorithm, K-means++, elbow method, limitations |
| `12_gmm_em.ipynb` | Expectation-Maximization, latent variables, comparison with K-means |
| `13_dbscan.ipynb` | Density-based clustering, core/border/noise points, epsilon-neighborhood |
| `14_hidden_markov_models.ipynb` | Forward-backward, Viterbi, Baum-Welch |

### [`03-deep-learning-from-scratch/`](03-deep-learning-from-scratch/)
Build a neural network library from the ground up in NumPy.

| Notebook | Topics |
|----------|--------|
| `01_computational_graph_autograd.ipynb` | Extend the autograd engine for full neural net support |
| `02_neural_network_from_scratch.ipynb` | Dense layers, forward/backward pass, mini-batch training |
| `03_activations_and_losses.ipynb` | ReLU, GELU, Swish, Sigmoid + CE, BCE, MSE, Focal loss |
| `04_backpropagation.ipynb` | Chain rule, Jacobians, manual gradient derivations |
| `05_normalization.ipynb` | BatchNorm, LayerNorm, GroupNorm, RMSNorm — forward & backward |
| `06_regularization.ipynb` | Dropout, weight decay, gradient clipping, label smoothing |
| `07_cnn_from_scratch.ipynb` | Conv2D (im2col), pooling, full forward/backward pass |
| `08_rnn_lstm_gru.ipynb` | Vanilla RNN, LSTM, GRU — unrolled backprop through time |
| `09_attention_mechanism.ipynb` | Scaled dot-product, multi-head, self-attention from scratch |

### [`04-transformer-and-llm/`](04-transformer-and-llm/)
Transformer architecture and modern LLM components in PyTorch.

| Notebook | Topics |
|----------|--------|
| `01_transformer_from_scratch.ipynb` | Full encoder-decoder, masked attention, residual connections |
| `02_positional_encodings.ipynb` | Sinusoidal, RoPE, ALiBi — implement and visualize each |
| `03_kv_cache.ipynb` | Inference optimization, incremental decoding |
| `04_decoding_strategies.ipynb` | Greedy, beam search, top-k, nucleus (top-p), temperature |
| `05_bpe_tokenizer.ipynb` | Byte-pair encoding from scratch, vocabulary building |
| `06_lora_qlora.ipynb` | Low-rank adaptation, quantization-aware fine-tuning |
| `07_flash_attention.ipynb` | Memory-efficient attention, tiling strategy |
| `08_mqa_gqa.ipynb` | Multi-query attention, grouped-query attention |
| `09_mixture_of_experts.ipynb` | Sparse MoE layer, top-k routing, load balancing |

### [`05-training-infrastructure/`](05-training-infrastructure/)
Production-grade training patterns in PyTorch.

| Notebook | Topics |
|----------|--------|
| `01_custom_dataset_dataloader.ipynb` | Dataset, DataLoader, samplers, collate functions |
| `02_mixed_precision_training.ipynb` | FP16/BF16, AMP, loss scaling |
| `03_distributed_training.ipynb` | DDP, model parallelism, FSDP overview |
| `04_gradient_accumulation.ipynb` | Effective batch size, memory-compute tradeoff |
| `05_lr_schedulers.ipynb` | Step, cosine, warmup, cyclic, OneCycleLR |
| `06_checkpointing_early_stopping.ipynb` | Model saving, resumption, patience-based stopping |

### [`06-evaluation-and-metrics/`](06-evaluation-and-metrics/)
Every metric you might need to implement or explain in an interview.

| Notebook | Topics |
|----------|--------|
| `01_classification_metrics.ipynb` | Precision, recall, F1, AUC-ROC, PR curves, confusion matrix |
| `02_regression_metrics.ipynb` | MSE, MAE, RMSE, R², adjusted R², MAPE |
| `03_ranking_metrics.ipynb` | NDCG, MRR, MAP, precision@k, recall@k |
| `04_nlp_metrics.ipynb` | BLEU, ROUGE, perplexity, BERTScore |
| `05_calibration.ipynb` | Reliability diagrams, ECE, temperature scaling |
| `06_statistical_testing.ipynb` | Bootstrap CI, paired t-test, McNemar's test |

### [`07-feature-engineering-and-data/`](07-feature-engineering-and-data/)
Data handling patterns for tabular ML.

| Notebook | Topics |
|----------|--------|
| `01_missing_data.ipynb` | Imputation strategies, MCAR/MAR/MNAR |
| `02_categorical_encoding.ipynb` | One-hot, ordinal, target encoding, embedding layers |
| `03_feature_scaling_selection.ipynb` | StandardScaler, MinMax, RobustScaler, mutual info, L1 selection |
| `04_imbalanced_data.ipynb` | SMOTE, class weights, focal loss, under/oversampling |
| `05_data_augmentation.ipynb` | Image, text, tabular augmentation strategies |
| `06_cross_validation.ipynb` | K-fold, stratified, time-series split, nested CV |

### [`08-ml-system-design/`](08-ml-system-design/)
Frameworks and templates for ML system design interviews.

| Notebook | Topics |
|----------|--------|
| `01_design_template.ipynb` | Reusable framework: problem → metrics → data → model → serving → monitoring |
| `02_recommendation_system.ipynb` | Collaborative filtering, content-based, two-tower, retrieval + ranking |
| `03_search_ranking.ipynb` | Query understanding, candidate retrieval, learning-to-rank |
| `04_fraud_detection.ipynb` | Real-time scoring, feature stores, concept drift |
| `05_ad_click_prediction.ipynb` | CTR prediction, feature crossing, calibration |

### [`09-coding-patterns/`](09-coding-patterns/)
Algorithmic patterns that recur in ML interview coding rounds.

| Notebook | Topics |
|----------|--------|
| `01_similarity_search.ipynb` | LSH, approximate nearest neighbors, FAISS patterns |
| `02_streaming_algorithms.ipynb` | Online mean/variance, reservoir sampling, count-min sketch |
| `03_time_series_patterns.ipynb` | Sliding window, exponential smoothing, lag features |
| `04_dynamic_programming_ml.ipynb` | Viterbi, CTC decoding, sequence alignment |

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/First-Principles-ML.git
cd First-Principles-ML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Study Guide

**Recommended order for interview prep:**

1. **Week 1-2:** `02-classical-ml-from-scratch/` — this is the highest-ROI section
2. **Week 3:** `01-mathematical-foundations/` — solidify optimization and probability
3. **Week 4:** `03-deep-learning-from-scratch/` — backprop and layers
4. **Week 5:** `04-transformer-and-llm/` — implement the architecture you use daily
5. **Week 6:** `06-evaluation-and-metrics/` + `07-feature-engineering-and-data/`
6. **Week 7:** `05-training-infrastructure/` + `09-coding-patterns/`
7. **Week 8:** `08-ml-system-design/` — tie it all together

## License

MIT
