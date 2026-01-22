# Phishing Detection: Fine-Tuned Small LLM vs GPT-4

**Author**: Guy Lissak, MSc Data Science, HIT — Cyber Security for Data Scientists Course

Comparing fine-tuned Qwen 2.5 0.5B against GPT-4 zero-shot for phishing email classification.

## Research Context

This project replicates and extends **Bucher & Martini (2024)**: *"Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification"* ([arXiv:2406.08660](https://arxiv.org/abs/2406.08660)).

**Original paper finding**: Fine-tuned RoBERTa (0.92 F1) significantly outperformed GPT-4 zero-shot (0.87 F1) on sentiment analysis of US economy news.

**Our adaptation**: Applied the same methodology to **phishing email detection** using Qwen 2.5 0.5B-Instruct with LoRA fine-tuning.

## Hypotheses

| # | Hypothesis | Experiment |
|---|------------|------------|
| **H1** | Fine-tuning significantly improves small LLM performance compared to zero-shot and few-shot prompting | Experiment 1 |
| **H2** | A fine-tuned small LLM (0.5B params) can match or exceed GPT-4 zero-shot on domain-specific classification | Experiment 2 |

## Experiment Flow

1. **EDA**: Analyze class distribution, text lengths, token counts
2. **Data Preparation**: Filter outliers, balance classes, create train/val/test splits
3. **Experiment 1**: Compare Qwen zero-shot vs few-shot vs fine-tuned (LoRA)
4. **Experiment 2**: Compare fine-tuned Qwen vs GPT-4 zero-shot

## Results

### Experiment 1: Prompt Engineering vs Fine-Tuning (Qwen 0.5B)

| Method | Accuracy | F1 Macro |
|--------|----------|----------|
| Zero-Shot | 52.8% | 0.452 |
| Few-Shot (7 examples/class) | 60.6% | 0.582 |
| **Fine-Tuned (LoRA)** | **96.5%** | **0.965** |

**H1 Confirmed**: Fine-tuning improves accuracy from 52.8% to 96.5% (+83% relative improvement). Prompt engineering alone fails for small models on specialized tasks.

### Experiment 2: Fine-Tuned Qwen vs GPT-4 (Paper Replication)

| Model | Parameters | Accuracy | F1 Macro |
|-------|------------|----------|----------|
| **Fine-Tuned Qwen** | 0.5B | **96.5%** | **0.965** |
| GPT-4 Zero-Shot | ~1.7T | 95.6% | 0.956 |

**H2 Confirmed**: Fine-tuned small model slightly outperforms GPT-4, though the gap is narrower than the original paper (~1% vs ~5%).

## Conclusions

1. **Fine-tuning is essential** for small models — zero-shot and few-shot prompting fail on specialized tasks
2. **Small fine-tuned models can match GPT-4** — achieving equivalent accuracy with 3400x fewer parameters
3. **Efficiency is the key advantage** — free inference, full data privacy, offline capable

## Project Structure

```
├── experiment_notebook.ipynb   # Main experiment notebook (run in Colab)
├── visualization.py            # Plotting functions
├── data_processing.py          # Data loading and preprocessing
├── model_builder.py            # Model training and evaluation
├── combined_phishing_dataset.csv  # Balanced dataset
└── README.md
```
