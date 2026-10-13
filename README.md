# 📖🖋️ C-Caption Transformer: Learns C Programming Language and English Captions

A small transformer that learns from scratch to write and explain simple C programs.

It is a character-level, decoder-only transformer (multi-head self-attention, about 11M parameters) trained in Google Colab
on a T4 GPU, and it learns three things from one text format:

```
/* Caption: Write a program to add 2 numbers. */   <- prompt: caption -> plan + code + explanation
#include <stdio.h>

/* Plan:                                           <- the model writes its steps first
   1. Read a and b.
   2. Store a + b in sum.
   3. Print the calculation. */
...
/* Explanation: The program reads a and b ... */   <- prompt with code -> explanation
// END

/* Question: What does malloc do? */               <- prompt: question -> answer
/* Answer: malloc() asks the heap for ... */
// END
```

> 🌱 NOTE: This project explores the workings of a transformer by building and training a small one end to end. For real-world applications, the usual approach is to start from a large pretrained model and adapt it through fine-tuning, prompting or RAG.

> ☁️ Google Colab provides the GPU for training; Google Drive stores the dataset and checkpoints.

> 📘 New to transformers? [DETAILS.md](DETAILS.md) explains the notebook and code step by step: the dataset, the model, training, generation and a glossary.

## 📁 Files

| File | What it is |
|---|---|
| 📓&nbsp;`c_caption_transformer.ipynb` | The Colab notebook: data loading, model, training, generation, checks |
| 📄&nbsp;`c_caption_dataset.txt` | The dataset (for the `c_captions_v3` run): 10,000 programs from 36 task families, each with a step-by-step plan and an English explanation, plus 876 Q&A documents on 73 C topics |
| 🛠️&nbsp;`make_c_dataset.py` | The generator. `python make_c_dataset.py` recreates `c_caption_dataset.txt` exactly |
| 📘&nbsp;`DETAILS.md` | A step-by-step explanation of the notebook and code |

📌 The programs include functions with 3–4 parameters, results returned through pointer parameters or in a struct,
and different declaration styles (`int a = 1, b = 2;`, `int *p, *q;`, declare first and assign later).

## 💭 A simple form of reasoning

- 📝 **Plan first:** each training program has a numbered `/* Plan: ... */` before its code, so the model learns to
  write its steps and then code from them (the idea behind "chain of thought").
- 🔁 **Check and retry:** the notebook's last cell writes up to `TRIES` programs and keeps the first one gcc accepts.

⚠️ Neither gives real understanding at this size. Twist captions such as "Reverse an array but skip the first
element." show the limit: the plan may mention the twist while the code still follows the familiar pattern.

## 🎓 Teach it with your feedback

👍 At the end of the notebook, the 📝 cell lets you mark the last generated program **👍 Correct** or **👎 Wrong** and
paste the correct code. The code is checked with gcc and saved to `MyDrive/c_caption_transformer/feedback.jsonl`.

🔧 Ticking `LEARN_NOW` in the 🎓 cell then fine-tunes a **copy** of the model (about 300 steps, a few minutes) on your
feedback mixed with normal training data, saves it to `<RUN_NAME>_feedback/best.pt`, and shows the answer before and
after. Each correction teaches that caption and close rewordings; a few corrections won't teach the model to solve
other new tasks.

🛡️ With their default settings both cells do nothing, so *Run all* never saves feedback or starts learning by accident.

## 🚀 Run it

1. ☁️ Upload `c_caption_dataset.txt` to Google Drive into `MyDrive/c_caption_transformer/`.
2. 📓 Open the notebook in Colab (*File → Upload notebook*) and choose *Runtime → Change runtime type → T4 GPU*.
3. ▶️ Tick `QUICK_TEST` and *Run all* for a one-minute smoke test, then untick it and *Run all* for real training
   (8,000 steps, roughly 40–50 minutes or more on a free T4; the training cell prints the time left).

💾 Checkpoints are written to `MyDrive/c_caption_transformer/<RUN_NAME>/` every 500 steps, and training resumes from
`latest.pt` after a disconnect. The model uses `BLOCK_SIZE = 1024`: a typical program with its plan is about 870
characters, and about three quarters of them fit in the model's view completely.

## 📊 Results

📉 Validation loss fell steadily to about 0.06, with training and validation loss close together. In tests:

| Test | Result |
|---|---|
| Familiar task ("Write a program to add 2 numbers.") | ✅ Correct plan, code and explanation |
| New wording of a familiar task ("…uppercase a string") | ✅ Correct |
| Familiar task + a twist ("Reverse the digits … and check whether it is the same number.") | ⚠️ Only the familiar half |
| New task ("Armstrong number", "ASCII value of a") | ❌ Writes the nearest familiar program (prime check, GCD) |

## 📝 Notes

- 🔍 The model imitates patterns from its training data. It writes valid C for familiar tasks, but for new tasks it
  usually reuses the nearest pattern, so read what it produces. Explanations and answers can be confidently wrong in
  the same way, especially for questions outside the 73 Q&A topics. "Compiles" does not mean "correct", and numbers
  can disagree between the plan, the code and the explanation.
- 🧪 Six test captions are deliberately absent from the dataset, so they measure generalisation:
  "Find the second largest number in an array.", "Count the uppercase letters in a string.",
  "Print the sum of the even numbers from 1 to 100." and three twist captions: "Reverse an array but skip the
  first element.", "Count the vowels in a sentence, but not the letter e." and "Print the numbers from 1 to n,
  but only the multiples of 3."

## ⚖️ License

MIT: see [LICENSE](LICENSE). The dataset is generated by `make_c_dataset.py` in this repository; there are no
third-party files. At runtime the notebook uses PyTorch, matplotlib and gcc (preinstalled in Colab) and, only if
`USE_SHAKESPEARE` is on, downloads [Tiny Shakespeare](https://github.com/karpathy/char-rnn) (public domain text).
These are not included here and keep their own licenses.
