# Multi-Class News Classification with BERT

Fine-tuning `bert-base-uncased` (PyTorch + Hugging Face) to classify news articles from [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) into four categories: **World, Sports, Business, Sci/Tech**.

The BERT model reaches **94.54% test accuracy**, a **+2.38 point** gain over a TF-IDF + Logistic Regression baseline (92.16%).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<bhumannagariarchana>/bert-ag-news-classification/blob/main/notebooks/BERT_AG_News_PyTorch_Colab.ipynb)

## Highlights

- **Proper data splits:** 90/10 stratified train/validation split of the original training set (108,000 / 12,000). The original 7,600-article test set is used only once, at the end.
- **Baseline for comparison:** TF-IDF (1 to 2 grams) + Logistic Regression.
- **Max length chosen from data:** the 99th percentile is 125 tokens, so `MAX_LEN = 128` instead of BERT's 512, combined with dynamic padding for faster training.
- **Training setup:** AdamW with weight-decay exclusions for bias and LayerNorm, linear warmup/decay (10% warmup), mixed precision (fp16), gradient clipping at 1.0, and best-checkpoint saving by validation accuracy.
- **Analysis:** classification report, confusion matrix, error analysis, and an inference function.

## Results

| Model | Test accuracy |
|-------|---------------|
| TF-IDF + Logistic Regression | 92.16% |
| **BERT (bert-base-uncased)** | **94.54%** |

Per-class results on the test set (1,900 articles per class):

| Class    | Precision | Recall | F1     |
|----------|-----------|--------|--------|
| World    | 0.9588    | 0.9547 | 0.9568 |
| Sports   | 0.9853    | 0.9868 | 0.9861 |
| Business | 0.9226    | 0.9095 | 0.9160 |
| Sci/Tech | 0.9151    | 0.9305 | 0.9228 |
| **Macro avg** | **0.9454** | **0.9454** | **0.9454** |

Confusion matrix (rows = true, columns = predicted):

|              | World | Sports | Business | Sci/Tech |
|--------------|------:|-------:|---------:|---------:|
| **World**    | 1814  | 9      | 40       | 37       |
| **Sports**   | 10    | 1875   | 7        | 8        |
| **Business** | 44    | 9      | 1728     | 119      |
| **Sci/Tech** | 24    | 10     | 98       | 1768     |

415 of 7,600 test articles were misclassified.

### Training log

| Epoch | Train loss | Val loss | Val accuracy |
|------:|-----------:|---------:|-------------:|
| 1 | 0.2950 | 0.1784 | 0.9397 |
| 2 | 0.1360 | 0.1720 | 0.9467 |
| 3 | 0.0864 | 0.1926 | **0.9479** |

Each epoch took about 640 seconds on a Colab T4 GPU. Validation loss rises slightly in epoch 3 while accuracy still improves, a mild sign of overfitting, so 3 epochs is about the right stopping point.

### Observations

- **Sports is easy** (F1 0.986). **Business vs. Sci/Tech is the hardest pair**: 217 of the 415 errors are confusions between those two classes, which makes sense for articles about tech companies, deals, and earnings.
- The raw AG News text contains HTML artifacts (e.g. `&lt;p&gt;`, `#39;s`) that are not cleaned here. Cleaning them is a possible small improvement.

### Example predictions

| Headline | Prediction (confidence) |
|----------|-------------------------|
| Manchester City beat Arsenal 2-1 in a thrilling Premier League clash | Sports (0.961) |
| Apple unveils new chip promising faster AI performance on laptops | Sci/Tech (0.987) |
| Stocks tumble as central bank signals more interest rate hikes | Business (0.957) |
| UN leaders meet to discuss ceasefire talks in the region | World (0.999) |

## Run it

The easiest way is Google Colab:

1. Open the notebook with the badge above.
2. Set `Runtime > Change runtime type > T4 GPU`.
3. Run `Runtime > Run all`. Full training takes about 30 minutes. Set `USE_SUBSET = True` in the config cell for a quick pipeline test.

To run locally (a GPU is strongly recommended):

```bash
git clone https://github.com/<your-username>/bert-ag-news-classification.git
cd bert-ag-news-classification
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/BERT_AG_News_PyTorch_Colab.ipynb
```

## Configuration

| Parameter     | Value |
|---------------|-------|
| Model         | `bert-base-uncased` (109.5M parameters) |
| Max length    | 128 (chosen from data) |
| Batch size    | 32 |
| Epochs        | 3 |
| Learning rate | 2e-5 |
| Warmup        | 10% |
| Weight decay  | 0.01 |
| Seed          | 42 |

## Project Structure

```
bert-ag-news-classification/
├── notebooks/
│   └── BERT_AG_News_PyTorch_Colab.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## Possible Next Steps

- Clean HTML artifacts from the text
- Compare DistilBERT / RoBERTa / DeBERTa
- Add a Gradio or FastAPI demo
- Push the trained model to the Hugging Face Hub

## License

MIT
