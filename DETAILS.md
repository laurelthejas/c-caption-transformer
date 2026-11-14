# 📖🖋️ C-Caption Transformer: the notebook and code explained

*How a small transformer learns from scratch to write and explain simple C programs, and how each part of the notebook works.*

## 🎯 1. What does it do?

The C-Caption Transformer is a small language model (about 11 million parameters) trained from scratch in Google Colab. It learned three things from one text format:

- **Write a program:** give it a caption, it writes a step-by-step plan, the C code and an English explanation.
- **Explain code:** give it a C program, it writes an explanation.
- **Answer questions:** ask about a C topic (such as malloc or pointers), it writes a short answer.

*One training example: caption, plan, code, explanation, then // END*

```c
/* Caption: Write a program to add 2 numbers. */
#include <stdio.h>

/* Plan:
   1. Read first and second.
   2. Store first + second in sum.
   3. Print the calculation. */
int main(void) {
    ...
}
/* Explanation: The program reads first and second from the keyboard with scanf ... */
// END
```

> **The key idea:** the model never sees rules about C. It only reads thousands of examples like the one above and learns to continue text in the same style, one character at a time.

> This project explores the workings of a transformer by building and training a small one end to end. For real-world applications, the usual approach is to start from a large pretrained model and adapt it through fine-tuning, prompting or RAG.

## 💡 2. Five ideas to understand first

| Idea | In plain words |
|---|---|
| **Characters become numbers** | A computer can only calculate with numbers, so every character (`a`, `b`, `{`, `;`, a space…) gets an ID. This model knows 92 different characters. |
| **Guess the next character** | Training is a guessing game: read some text, guess the next character, check the real one, adjust a little. Repeat millions of times. |
| **Loss = how wrong the guesses are** | A number that goes down as the model gets better. 0 would mean perfect guesses. |
| **Attention = looking back** | To guess well, each position looks back at the earlier characters and decides which ones matter (for example, the matching opening brace). |
| **Learning vs memorising** | Some programs are kept aside (validation). If the model does well on those too, it learned patterns instead of memorising the training text. |

## 📚 3. The dataset (make_c_dataset.py)

Nobody typed 10,000 programs by hand. The script `make_c_dataset.py` generates them from templates and randomly varies the names, types, values, loop styles and comment wording, so the model sees thousands of different programs rather than one program repeated.

- 10,000 programs from 36 task families (sums, leap years, GCD, binary search, linked lists, files…).
- Each program has a caption, a numbered plan written before the code, the code, and an explanation.
- 876 question-and-answer documents about 73 C topics are added at the end.
- Running `python make_c_dataset.py` recreates exactly the same file, because the random generator always starts from the same seed.

> **A fair test:** six captions are deliberately never in the dataset, for example “Find the second largest number in an array.” They show whether the model can handle something new.

## 🗺️ 4. A tour of the notebook

The notebook has 19 cells in four parts. Settings are form fields (the `#@param` markers), so you can change values in Colab without editing code.

| Cell | What it does |
|---|---|
| **⚙️ Settings** | Every setting in one place: run name, data file, steps, model size. QUICK_TEST runs a tiny version in about a minute to catch mistakes. |
| **🔧 Setup** | Finds the GPU, connects Google Drive and creates the checkpoint folder. |
| **📚 Load the dataset** | Reads the dataset file, splits it into documents at // END, and keeps some aside for validation. |
| **✅ Check with gcc** | Asks the C compiler whether the training programs are valid C. |
| **🔤 Tokenizer + batches** | Turns characters into IDs and cuts random training windows out of the text. |
| **🏗️ Model** | Defines the transformer: embeddings, attention, blocks. |
| **🚀 Train** | The training loop. Saves checkpoints and resumes automatically after a disconnect. |
| **📉 Loss curve** | Draws training and validation loss so you can see learning (and spot memorising). |
| **✍️ Use the model** | Loads best.pt and offers write_program, explain_code and ask. |
| **🔍 Valid and new?** | For each generated program: does gcc accept it, and is it new or a copy? |
| **💬 Try the model** | Type a caption or question; it writes up to TRIES programs and keeps the first that compiles. |
| **📝 + 🎓 Feedback** | Mark an answer right or wrong, paste the correct code, and optionally fine-tune a copy of the model on your corrections. |

## 🔤 5. Turning text into numbers (the tokenizer)

The notebook makes a list of every character in the data and gives each one an ID. `stoi` means “string to integer” and `itos` is the reverse:

*From the 🔤 cell*

```python
chars = sorted(set(train_text + val_text))
stoi = {ch: i for i, ch in enumerate(chars)}     # e.g. {'a': 52, ...}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):            # text -> list of ids
    return [stoi[c] for c in s if c in stoi]

def decode(ids):          # list of ids -> text
    return "".join(itos[i] for i in ids)
```

To train, `get_batch()` cuts 16 random windows of 1,024 characters out of the text. The target is the same window shifted one character to the right, so at every position the answer is just the next character:

| Input (x) | i | n | t | ␣ | m | a | i | n |
|---|---|---|---|---|---|---|---|---|
| **Target (y)** | n | t | ␣ | m | a | i | n | ( |

> **In plain words:** every window gives the model 1,024 small quizzes at once: “after these characters, what comes next?”

## 🏗️ 6. The model: a decoder-only transformer

Each character goes through these stages. The numbers are the real settings of the trained model:

**Character IDs** → **Embeddings** (384 numbers each) → **6 transformer blocks** (attention + thinking) → **Scores for all** (92 characters)

### 6.1 Embeddings: meaning plus position

*From the 🏗️ cell*

```python
self.tok_emb = nn.Embedding(vocab_size, N_EMBD)   # learned vector for each character ("what")
self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)   # learned vector for each position ("where")
...
x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))
```

Each character becomes a list of 384 numbers (its “ID card”), and each position 0 to 1,023 gets its own list too. Adding them tells the model both **what** the character is and **where** it is.

Why 384? It is a design choice, `N_EMBD = 384` in the ⚙️ Settings cell, not something calculated: large enough to hold useful patterns, small enough to train on a single free Colab GPU, and it splits evenly across the 6 attention heads (6 × 64 = 384).

### 6.2 Self-attention: looking back

*From the 🏗️ cell (the heart of attention)*

```python
q, k, v = self.qkv(x).split(C, dim=2)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

- **Query, key, value:** each position asks a question (query), offers a label (key) and carries information (value). Matching queries to keys decides who listens to whom.
- **is_causal=True:** a position may only look at earlier characters, never ahead. That is what “decoder-only” means: like writing a story one word at a time.
- **6 heads:** attention runs 6 times in parallel (64 numbers each), so different heads can focus on different things, such as brackets, variable names or the plan.

### 6.3 A block, and the stack

*One transformer block*

```python
def forward(self, x):
    x = x + self.attn(self.ln1(x))     # look at earlier characters
    return x + self.mlp(self.ln2(x))   # think about what was found
```

LayerNorm keeps the numbers in a stable range, and the “x +” parts (residual connections) let information flow straight through. Six blocks are stacked. At the end, a final layer turns the 384 numbers into one score for each of the 92 characters.

| Setting | Value | Meaning |
|---|---|---|
| **BLOCK_SIZE** | 1,024 | Characters the model can see at once (a whole typical program) |
| **N_EMBD** | 384 | Numbers describing each position |
| **N_HEAD** | 6 | Attention heads per block (64 numbers each) |
| **N_LAYER** | 6 | Transformer blocks stacked |
| **DROPOUT** | 0.2 | Switches off 20% of signals during training, to reduce overfitting (memorising) |
| **Parameters** | ≈ 11.1 million | All the learned numbers together |

### 6.4 Softmax: from scores to probabilities

The final layer outputs a raw score (a logit) for each of the 92 characters, and these scores can be any number. Softmax converts them into probabilities that are positive and add up to 1: raise e to the power of each score, then divide each result by their total.

| | Character A | Character B | Character C |
|---|---|---|---|
| **Raw score** | 2.0 | 1.0 | 0.1 |
| **After softmax** | 0.66 | 0.24 | 0.10 |
| **After softmax, temperature 0.7** | 0.77 | 0.18 | 0.05 |

For example, after `int mai`, suppose the next characters `n`, `l` and `d` get the scores above. Softmax gives `n` a 66% chance of being picked, or 77% at the notebook's temperature of 0.7, so the model will most likely write `main`.

Softmax is used in three places:

- **Attention:** `scaled_dot_product_attention` applies it to the query-key match scores, so each position's attention weights add up to 1.
- **Training:** `F.cross_entropy` applies it internally as log-softmax, which is why the model outputs raw scores and has no softmax layer of its own.
- **Generation:** temperature divides the scores before softmax, and top-k sets the excluded scores to minus infinity, which gives them a probability of 0.

*From the ✍️ cell: choosing the next character*

```python
nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
```

### 6.5 LayerNorm and batch normalization

This model uses LayerNorm, not batch normalization (BatchNorm). BatchNorm rescales each feature using its mean and variance across the examples in a batch, and stores running averages to use at inference. It is common in image models, but its behaviour depends on the batch size and differs between training and inference.

LayerNorm computes the mean and variance from the 384 numbers at each position, independently of the rest of the batch. It therefore behaves the same way during training (16 windows) and generation (one sequence).

## 🚀 7. Training: the guessing game

Every training step does four things:

**1. Take a batch** (16 random windows) → **2. Guess** (forward pass + loss) → **3. Find the blame** (backward pass) → **4. Adjust** (optimizer step)

- **AdamW** decides how much to change each of the 11 million numbers, using the gradient (the direction that lowers the loss).
- **Learning rate schedule:** a short warm-up, then big steps early and smaller steps later (cosine decay), like sanding with coarse then fine sandpaper.
- **Evaluation every 250 steps:** the loss is measured on training text and on validation text the model never trains on.
- **Checkpoints:** best.pt (lowest validation loss so far), latest.pt (to resume after Colab disconnects) and a numbered copy every 500 steps, all saved to Google Drive.

*From the 🚀 cell: the learning rate schedule*

```python
def lr_at(step):
    if step < WARMUP:
        return LEARNING_RATE * (step + 1) / WARMUP          # warm-up
    progress = min(1.0, (step - WARMUP) / max(1, MAX_ITERS - WARMUP))
    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + math.cos(math.pi * progress))
```

The real run trained for 8,000 steps on a T4 GPU. The best checkpoint's validation loss was 0.053, with training and validation loss close together, so the model did not memorise the training programs. The validation programs come from the same templates, though, so this shows the model handles new variations of familiar tasks, not new tasks (see section 11). The 📉 cell draws both curves; if they drift apart, the model is memorising.

## ✍️ 8. Using the model: writing one character at a time

Generation repeats one small step: show the model the text so far, get scores for the next character, pick one, add it, repeat, until the model writes `// END`.

*From the ✍️ cell: the three prompt formats*

```python
def write_program(caption):
    return generate(f"/* Caption: {caption} */\n")

def explain_code(code):
    return generate(code + "\n/* Explanation:")

def ask(question):
    return generate(f"/* Question: {question} */\n/* Answer:")
```

- **Temperature (0.7):** lower means safer, more predictable choices; higher means more variety and more mistakes.
- **Top-k (20):** only the 20 most likely next characters are allowed, which avoids silly picks.
- **Plan first:** because every training example has a plan before the code, the model writes its steps first and then codes from them, a simple form of “chain of thought”.
- **Check and retry:** the 💬 cell writes up to 5 programs and keeps the first one that gcc accepts.

> **Honest checks:** the 🔍 cell asks two questions for every program: does gcc accept it, and is it an exact copy of a training program? A good model compiles AND writes new combinations of what it learned. And “compiles” does not mean “correct”: always read the code.

## 🎓 9. Teaching it with feedback

After trying the model, you can mark its last program 👍 Correct or 👎 Wrong and paste the correct code (the 📝 cell). The code is checked with gcc and saved to Google Drive. Ticking LEARN_NOW in the 🎓 cell fine-tunes a **copy** of the model for about 300 steps on your corrections mixed with normal training data, and shows the answer before and after. A few corrections teach that caption and close rewordings, not brand-new skills.

## 💻 10. Running it on a CPU: the app

Training needs a GPU, but the trained checkpoint (`best.pt`, saved by your training run) is small enough to run on a normal laptop CPU. The Streamlit app, `app.py`, does this: the README shows how to start it. One trick makes it fast:

> **The KV cache:** the notebook's generate() re-reads the whole text for every new character, which is fine on a GPU but takes minutes per program on a CPU. Remembering the attention keys and values that were already computed turns each new character into one small step. `c_caption_model.py` does this, and a test confirmed that with the same seed it writes exactly the same text as the notebook's method, including after the text grows past the model's 1,024-character view.

The app's three modes use the notebook's three prompt formats, and its settings are the notebook's: temperature, top-k and **Tries**, which retries until gcc accepts a program. The model classes in `c_caption_model.py` are exact copies of the notebook's, so a checkpoint from any training run loads in the app.

## 📊 11. Results and limits

| Test | Result |
|---|---|
| **Familiar task (“Write a program to add 2 numbers.”)** | ✅ Correct plan, code and explanation |
| **New wording of a familiar task (“…uppercase a string”)** | ✅ Correct |
| **Familiar task with a twist (“Reverse an array but skip the first element.”)** | ⚠️ Only the familiar half: it reverses the whole array |
| **New task (“Find the second largest number”, “Armstrong number”)** | ❌ Writes the nearest familiar program instead (find the largest, prime check) |

- The model imitates patterns. For familiar tasks it writes valid, sensible C; for new tasks it reuses the nearest pattern.
- Why it answers this way: the model learned from 10,000 programs built from 36 task templates, so for a request outside those patterns, such as adding *and* multiplying in one program, it writes the closest program it has seen. Models such as GPT-2 use the same kind of transformer, but learn from millions of varied documents written by people, with over ten times as many parameters, which is what lets them handle new wordings far better.
- Explanations can be confidently wrong, and numbers can disagree between the plan, the code and the explanation (for example a binary search on an array that is not sorted).
- The key takeaway: a model this small learns patterns, not understanding. Bigger models and more varied data push the limits further, but reading what a model writes always matters.

## 🧪 12. Try it yourself

1. **Turn the temperature up and down** with TEMPERATURE in the ✍️ cell (0, 0.7, 1.2). When does the code break?
2. **Try a twist caption:** change a familiar caption slightly and see whether the plan and the code follow your change.
3. **Explain your own code:** paste a short program in a different style. Is the explanation right?
4. **Train a tiny model:** tick QUICK_TEST in Colab and Run all. It trains in about a minute; compare its output with the real model's.
5. **Change the data:** run make_c_dataset.py with fewer programs, train a new run, and compare the loss curves.

## 📘 13. Glossary

| Word | Meaning |
|---|---|
| **Transformer** | A neural network that uses attention to decide which earlier parts of the text matter. |
| **Decoder-only** | Each position may only look back, never ahead. Used for writing text. |
| **Token** | One unit of text the model reads. Here a token is a single character. |
| **Embedding** | A learned list of numbers that represents a token or a position. |
| **Attention head** | One “reader” inside attention; 6 heads read in parallel with different focus. |
| **Parameters** | The learned numbers inside the model (about 11.1 million here). |
| **Loss** | How wrong the model's guesses are; training pushes it down. |
| **Validation set** | Data kept aside to check whether the model learned or memorised. |
| **Checkpoint** | A saved copy of the model's numbers (best.pt, latest.pt). |
| **Temperature / top-k** | Settings that control how adventurous the model's choices are. |
| **Fine-tuning** | Training an already trained model a little more on new examples. |

## 📜 14. Credits

The model follows published work, with thanks to its authors:

- **Transformer:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin. [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS 2017. The attention mechanism and the transformer block used here.
- **Decoder-only language models:** Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). OpenAI, 2018. The GPT design of training a transformer to predict the next token.
- **Tools:** PyTorch for the model and training, GCC for checking the programs, and Streamlit for the app.
