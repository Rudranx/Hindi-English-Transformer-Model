# Transformer from Scratch

This project implements the **original Transformer architecture** from scratch using **PyTorch**, with an application to **English-to-Hindi translation**.

The goal of this project is not to achieve state-of-the-art translation accuracy, but rather to deeply understand and manually build each component of the Transformer model as introduced in the paper *"Attention is All You Need" (Vaswani et al., 2017)*.

---

## 🚀 Features

* Custom implementation of:

  * Multi-Head Self Attention
  * Positional Encoding
  * Transformer Encoder and Decoder blocks
  * Masked Self Attention and Cross Attention
  * Greedy Decoding
* Training loop from scratch
* BLEU score evaluation
* Minimal dependencies (no HuggingFace, no `nn.Transformer`)

---

## 🧪 Example

```
Input: How are you?
Output: कैसे हैं आप?
```

---

## 📈 Results

Trained on a small subset of an English-Hindi parallel corpus (preprocessed into start/end-tokenized pairs).

| Epoch | Train Loss | BLEU Score (100 samples) |
| ----- | ---------- | ------------------------ |
| 1     | 6.45       | 0.004                    |
| 10    | 3.89       | 0.059                    |
| 50    | 0.90       | 0.135                    |

> 🔍 Note: This BLEU score is **not optimized**, and the model is trained with minimal preprocessing for learning purposes only.

---

## 🧠 Key Learnings

* Inner workings of multi-head self attention
* Positional information without recurrence
* Attention masking for sequence generation
* Token-level translation via greedy decoding

---

## 📚 References

* Vaswani et al., 2017 — ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
* BLEU Score — Papineni et al., 2002

---


## 👋 Author

Built with ❤️ by \[Manas Tiwari].
