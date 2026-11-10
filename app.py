"""Streamlit app for the C-Caption Transformer: write a program from a caption, explain code, or answer a question."""
import os
import threading
from pathlib import Path

import streamlit as st
import torch

from c_caption_model import CCaptionModel, clean, compiles, find_compiler, split_program

ROOT = Path(__file__).parent
CHECKPOINT = Path(os.environ.get("CHECKPOINT", ROOT / "weights" / "best.pt"))
HAS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if HAS_GPU else "cpu"
COMPILER = find_compiler()

WRITE, EXPLAIN, ASK = "Write a program", "Explain code", "Ask a question"
WRITE_OWN = "Write my own..."
# The notebook's test captions: three like the training data, and six that are deliberately not in it.
SEEN = ["Reverse a string in place.", "Write a program to add 2 numbers.",
        "Find the minimum and maximum of an array with one function."]
UNSEEN = ["Find the second largest number in an array.", "Count the uppercase letters in a string.",
          "Print the sum of the even numbers from 1 to 100.", "Reverse an array but skip the first element.",
          "Count the vowels in a sentence, but not the letter e.",
          "Print the numbers from 1 to n, but only the multiples of 3."]
QUESTIONS = ["What does malloc do?", "How can a function return two values?", "What is a union?"]
NOT_IN_QA = {"What is a union?"}
SAMPLE_CODE = """#include <stdio.h>

int triple(int x) {
    return 3 * x;
}

int main(void) {
    int a = 4, b = 9;
    printf("%d %d\\n", triple(a), triple(b));
    return 0;
}"""


@st.cache_resource(show_spinner="Loading the model...")
def load_model(path):
    return CCaptionModel(path, DEVICE)


@st.cache_resource(show_spinner=False)
def generation_lock():
    # Tabs share one model, so one answer is written at a time.
    return threading.Lock()


def choose(label, options, key, note=None):
    # A sample, or the user's own text. note(choice) can return a caption to show under the choice.
    choice = st.selectbox(label, options + [WRITE_OWN], key=f"{key}_choice")
    if choice == WRITE_OWN:
        return st.text_input("Your text", key=f"{key}_own").strip()
    if note and note(choice):
        st.caption(note(choice))
    return choice


def run(make_text):
    # Runs one generation, showing the text as it grows, because a CPU can take a while.
    live = st.empty()

    def on_text(text):
        live.code(text[-1500:], language="c")

    with st.spinner("Writing..."), generation_lock():
        text = make_text(on_text)
    live.empty()
    return text


st.set_page_config(page_title="C-Caption Transformer", page_icon="📖")
st.title("📖🖋️ C-Caption Transformer")
st.caption("A small character-level transformer (about 11M parameters) trained from scratch to write and explain "
           "simple C programs. It imitates patterns from its training data, so read what it writes.")

has_model = CHECKPOINT.exists()
model = load_model(str(CHECKPOINT)) if has_model else None

with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.05,
                            help="Below 1 is safer and more predictable. Above 1 is more adventurous, with more mistakes.")
    top_k = st.slider("Top-k", 1, 60, 20, help="Picks each next character from only this many of the most likely ones.")
    seed = st.number_input("Seed", min_value=0, value=1337, step=1,
                           help="The same seed and settings repeat an answer. A new seed gives a new one.")
    tries = st.slider("Tries", 1, 10, 5, disabled=COMPILER is None,
                      help="Writes up to this many programs and keeps the first one gcc accepts. Needs gcc.")
    st.divider()
    st.caption(f"Device: {torch.cuda.get_device_name(0) if HAS_GPU else 'CPU'}")
    if model:
        st.caption(f"Model: loaded (training step {model.step}, validation loss {model.best_val:.3f})"
                   if model.best_val is not None else "Model: loaded")
    else:
        st.caption("Model: not found")
    st.caption(f"gcc check: {'available' if COMPILER else 'not available'}")

if not has_model:
    rel = CHECKPOINT.relative_to(ROOT) if CHECKPOINT.is_relative_to(ROOT) else CHECKPOINT
    st.warning(f"Trained model not found at `{rel}`. See the README for how to add it.")
if COMPILER is None:
    st.info("gcc was not found, so programs are not checked for valid C and Tries is off.")

settings = dict(temperature=temperature, top_k=top_k, seed=int(seed))
mode = st.radio("What would you like to do?", [WRITE, EXPLAIN, ASK], horizontal=True)
st.divider()

if mode == WRITE:
    caption = choose("Caption: choose a sample or write your own", SEEN + UNSEEN, "write",
                     note=lambda c: "This caption is not in the training data." if c in UNSEEN else None)
    if st.button("Write program", type="primary", disabled=not caption or not has_model):
        def make(on_text):
            for attempt in range(1, (tries if COMPILER else 1) + 1):
                out = model.write_program(caption, on_text=on_text, **{**settings, "seed": settings["seed"] + attempt - 1})
                ok, error = compiles(out, COMPILER) if COMPILER else (None, "")
                if ok or ok is None:
                    break
            return out, ok, error, attempt
        st.session_state["write_result"] = run(make)
    if "write_result" in st.session_state:
        out, ok, error, attempt = st.session_state["write_result"]
        head, plan, body, explanation = split_program(out)
        st.subheader("Plan")
        # The model indents every step but the first, so the lines are lined up before showing them.
        st.text("\n".join(line.strip() for line in plan.splitlines()) or "(no plan)")
        st.subheader("Program")
        program = (head + "\n\n" + body).strip()
        st.code(program, language="c")
        st.subheader("Explanation")
        st.write(explanation or "(no explanation)")
        if ok is True:
            st.success(f"gcc accepts this program (attempt {attempt} of {tries}).")
        elif ok is False:
            st.error(f"None of the {tries} attempts compiled. First error: {error}")
        st.download_button("Download program.c", program + "\n", file_name="program.c", mime="text/x-c")
        with st.expander("Everything the model wrote"):
            st.code(out, language="c")

elif mode == EXPLAIN:
    code = st.text_area("C code to explain", value=SAMPLE_CODE, height=260,
                        help="The sample program is not in the training data.")
    if st.button("Explain code", type="primary", disabled=not code.strip() or not has_model):
        st.session_state["explain_result"] = run(lambda on_text: model.explain_code(code, on_text=on_text, **settings))
    if "explain_result" in st.session_state:
        st.subheader("Explanation")
        st.write(clean(st.session_state["explain_result"]) or "(no explanation)")

else:
    question = choose("Question: choose a sample or write your own", QUESTIONS, "ask",
                      note=lambda q: "This topic is not in the training data, so expect a guess." if q in NOT_IN_QA else None)
    if st.button("Answer", type="primary", disabled=not question or not has_model):
        st.session_state["ask_result"] = run(lambda on_text: model.ask(question, on_text=on_text, **settings))
    if "ask_result" in st.session_state:
        st.subheader("Answer")
        st.write(clean(st.session_state["ask_result"].partition("/* Answer:")[2]) or "(no answer)")
