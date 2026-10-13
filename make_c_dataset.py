"""
make_c_dataset.py - writes the dataset file for the C-Caption Transformer notebook.

Every program starts with  /* Caption: ... */, then a numbered  /* Plan: ... */  (the steps, written
before the code), the code, an English  /* Explanation: ... */  and finally  // END.
Question-and-answer documents about C concepts are added at the end.
Usage:  python make_c_dataset.py [number_of_programs] [output_file] [qa_repeats_per_topic]
        python make_c_dataset.py        -> 10000 programs + Q&A, c_caption_dataset.txt
"""
# Every sample is a small, complete C program that starts with a caption:
#   /* Caption: <what the program does> */
#   ...code with step-by-step comments...
#   // END
# Names, types, values, loop styles and comment wording are randomised, so the
# model sees thousands of DIFFERENT programs instead of one block repeated.
import random
import textwrap
from string import Template

END = "// END"
R = random.Random(0)


def pick(*options):
    return R.choice(options)


def fill(template, /, **kw):
    """Dedent a C template, then fill ${name} slots (C never uses '$')."""
    return Template(textwrap.dedent(template).strip("\n")).substitute(**kw)


def numtype():
    return pick(("int", "%d"), ("int", "%d"), ("long", "%ld"), ("double", "%.2f"))


def values(typ, k, lo=1, hi=99):
    if typ == "double":
        return [f"{R.uniform(lo, hi):.1f}" for _ in range(k)]
    suffix = "L" if typ == "long" else ""
    return [f"{R.randint(lo, hi)}{suffix}" for _ in range(k)]


def braces(vals):
    return "{" + ", ".join(vals) + "}"


def arr_name():
    return pick("values", "numbers", "data", "items", "scores", "marks", "nums", "readings", "samples", "prices", "ages")


def len_name():
    return pick("n", "count", "size", "len", "length")


def idx_name():
    return pick("i", "i", "j", "k", "idx")


def acc_name():
    return pick("sum", "total", "acc", "result")


def length_decl(a, n, k):
    return pick(f"int {n} = sizeof({a}) / sizeof({a}[0]);", f"int {n} = {k};", f"const int {n} = {k};")


def loop(i, n, body, style=None):
    """A counting loop over 0..n-1, as a for or a while loop (4-space indented)."""
    style = style or pick("for", "for", "while")
    body = textwrap.indent(body, "        ")
    if style == "for":
        return f"    for (int {i} = 0; {i} < {n}; {i}++) {{\n{body}\n    }}"
    return f"    int {i} = 0;\n    while ({i} < {n}) {{\n{body}\n        {i}++;\n    }}"


def names_list(names):
    """['a', 'b', 'c'] -> 'a, b and c'"""
    names = [str(x) for x in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def inputs_desc(code, names):
    """How a program gets its numbers, for the explanation."""
    if "scanf" in code:
        return f"reads {names_list(names)} from the keyboard with scanf and stops with a message if the input is not valid"
    return f"uses fixed values for {names_list(names)}"


def rs(code):
    """First word of a plan step: the program reads its input, or sets fixed values."""
    return "Read" if "scanf" in code else "Set"


def loop_word(code):
    return "while loop" if "while (" in code else "for loop"


def print_array(typ, fmt, a, n, i):
    return fill(r"""
        for (int ${i} = 0; ${i} < ${n}; ${i}++) {
            printf("${fmt} ", ${a}[${i}]);
        }
        printf("\n");""", i=i, n=n, fmt=fmt, a=a)


WORDS = ["hello", "planet", "racecar", "level", "program", "compiler", "pointer", "memory", "stack", "kernel",
         "garden", "window", "silver", "rotor", "madam", "banana", "keyboard", "noon", "refer", "orange", "civic",
         "rocket", "binary", "integer", "library", "stats", "deified", "matrix", "network", "bridge"]
SENTENCES = ["the quick brown fox jumps over the lazy dog", "c gives you direct control over memory",
             "pointers store the address of another variable", "every program starts in main",
             "arrays in c start at index zero", "a function can call itself recursively",
             "strings end with a null character", "learning c teaches how computers work",
             "loops repeat a block of code", "structs group related data together",
             "the compiler turns source code into machine code", "always free the memory you allocate",
             "good comments explain why the code exists", "keep functions short and focused"]


# ----------------------------------------------------------------------------- arrays
def t_sum():
    typ, fmt = numtype(); a, n, i, s = arr_name(), len_name(), idx_name(), acc_name()
    vals = values(typ, R.randint(4, 10)); fn = pick("sum_array", "array_sum", "add_all", "sum_values", "total_of")
    cap = pick(f"Add up all the numbers in an array of {typ} values.", "Compute the sum of the elements of an array.",
               "Find the total of all elements in an array using a loop.", f"Sum a {typ} array with a helper function.")
    body = loop(i, n, f"{s} += {a}[{i}];")
    code = fill(r"""
        /* ${doc} */
        ${typ} ${fn}(const ${typ} ${a}[], int ${n}) {
            ${typ} ${s} = 0;
        ${body}
            return ${s};
        }

        int main(void) {
            ${typ} ${a}[] = ${init};
            ${decl}
            printf("${label}${fmt}\n", ${fn}(${a}, ${n}));
            return 0;
        }""", doc=pick("Return the sum of the first n elements.", "Add every element to a running total.",
                       "Walk through the array and accumulate each value."),
        typ=typ, fn=fn, a=a, n=n, s=s, body=body, init=braces(vals), decl=length_decl(a, n, len(vals)),
        label=pick("Sum: ", "Total = ", "The sum is "), fmt=fmt)
    return cap, ["stdio.h"], code, f"{fn}() receives the array {a} and its length {n}. It starts {s} at 0, adds every element to it in a {loop_word(code)} and returns the total. main() fills {a} with {len(vals)} {typ} values and prints their sum.", [f"Write {fn}() that takes the array {a} and its length {n}.", f"Start {s} at 0 and add every element in a {loop_word(code)}.", f"Return {s}.", f"In main(), fill {a} with {len(vals)} values and print the sum."]


def t_average():
    typ, fmt = numtype(); a, n, i, s = arr_name(), len_name(), idx_name(), acc_name()
    vals = values(typ, R.randint(3, 9)); fn = pick("average", "mean", "array_average", "compute_mean")
    cap = pick("Calculate the average of the values in an array.", "Compute the mean of a list of numbers.",
               f"Find the arithmetic mean of {len(vals)} {typ} values.")
    code = fill(r"""
        /* ${doc} */
        double ${fn}(const ${typ} ${a}[], int ${n}) {
            double ${s} = 0.0;
        ${body}
            return ${s} / ${n};
        }

        int main(void) {
            ${typ} ${a}[] = ${init};
            ${decl}
            printf("${label}%.2f\n", ${fn}(${a}, ${n}));
            return 0;
        }""", doc=pick("Sum the values, then divide by how many there are.", "The mean is the total divided by the count."),
        typ=typ, fn=fn, a=a, n=n, s=s, body=loop(i, n, f"{s} += {a}[{i}];"), init=braces(vals),
        decl=length_decl(a, n, len(vals)), label=pick("Average: ", "Mean = ", "The average is "))
    return cap, ["stdio.h"], code, f"{fn}() adds the {len(vals)} values of {a} into the double {s} and divides by {n}, so the mean keeps its decimal part even when the values are whole numbers. main() prints it with two decimals.", [f"Write {fn}() that takes {a} and {n}.", f"Add the values into the double {s}.", f"Return {s} / {n}.", "Print the average with two decimals."]


def t_extreme():
    typ, fmt = numtype(); a, n, i = arr_name(), len_name(), idx_name()
    vals = values(typ, R.randint(4, 10)); big = R.random() < 0.5
    word, op = ("largest", ">") if big else ("smallest", "<")
    fn = pick(f"find_{'max' if big else 'min'}", f"{'max' if big else 'min'}_of", f"{word}_value")
    best = pick("best", "m", word, "current")
    cap = pick(f"Find the {word} number in an array.", f"Return the {word} element of an array of {typ} values.",
               f"Search an array for its {word} value.")
    code = fill(r"""
        /* Start with the first element and keep the ${word} value seen so far. */
        ${typ} ${fn}(const ${typ} ${a}[], int ${n}) {
            ${typ} ${best} = ${a}[0];
            for (int ${i} = 1; ${i} < ${n}; ${i}++) {
                if (${a}[${i}] ${op} ${best}) {
                    ${best} = ${a}[${i}];
                }
            }
            return ${best};
        }

        int main(void) {
            ${typ} ${a}[] = ${init};
            ${decl}
            printf("${label}${fmt}\n", ${fn}(${a}, ${n}));
            return 0;
        }""", word=word, typ=typ, fn=fn, a=a, n=n, i=i, op=op, best=best, init=braces(vals),
        decl=length_decl(a, n, len(vals)), label=pick(f"The {word} value is ", f"{word.capitalize()}: "), fmt=fmt)
    return cap, ["stdio.h"], code, f"{fn}() takes the first element as the {word} value so far and stores it in {best}. The loop compares every other element with {best} using {op} and keeps the {word} one. main() prints the {word} of the {len(vals)} values.", [f"Write {fn}() that takes {a} and {n}.", f"Store {a}[0] in {best}.", f"Loop over the other elements and keep the {word} value in {best}.", f"Return {best} and print it."]


def t_count_parity():
    a, n, i, c = arr_name(), len_name(), idx_name(), pick("count", "found", "matches", "tally")
    if n == c: c = "found"
    vals = values("int", R.randint(5, 12)); even = R.random() < 0.5
    word, test = ("even", "== 0") if even else ("odd", "!= 0")
    cap = pick(f"Count how many {word} numbers are in an array.", f"Count the {word} values in a list of integers.")
    code = fill(r"""
        /* A number is ${word} when the remainder after dividing by 2 is ${rem}. */
        int count_${word}(const int ${a}[], int ${n}) {
            int ${c} = 0;
        ${body}
            return ${c};
        }

        int main(void) {
            int ${a}[] = ${init};
            ${decl}
            printf("${label}%d\n", count_${word}(${a}, ${n}));
            return 0;
        }""", word=word, rem="zero" if even else "one", a=a, n=n, c=c,
        body=loop(i, n, f"if ({a}[{i}] % 2 {test}) {{\n    {c}++;\n}}"), init=braces(vals),
        decl=length_decl(a, n, len(vals)), label=pick(f"{word.capitalize()} numbers: ", f"Count of {word} values = "))
    return cap, ["stdio.h"], code, f"count_{word}() tests each element of {a} with the % operator: a remainder of {0 if even else 1} after dividing by 2 means the number is {word}, so {c} goes up by one. main() prints how many of the {len(vals)} values are {word}.", [f"Write count_{word}() that takes {a} and {n}.", f"For each element, test {a}[{i}] % 2 {test}.", f"Add one to {c} for every {word} number.", f"Return {c} and print it."]


def t_linear_search():
    a, n, i = arr_name(), len_name(), idx_name()
    vals = values("int", R.randint(5, 10)); target = R.choice(vals) if R.random() < 0.7 else str(R.randint(100, 200))
    key = pick("target", "key", "wanted", "needle")
    cap = pick("Search an array for a value and print its position.", "Find the index of a number using linear search.",
               "Check every element of an array to find a target value.")
    code = fill(r"""
        /* Return the index of ${key}, or -1 when it is not in the array. */
        int linear_search(const int ${a}[], int ${n}, int ${key}) {
            for (int ${i} = 0; ${i} < ${n}; ${i}++) {
                if (${a}[${i}] == ${key}) {
                    return ${i};
                }
            }
            return -1;
        }

        int main(void) {
            int ${a}[] = ${init};
            ${decl}
            int ${key} = ${target};
            int pos = linear_search(${a}, ${n}, ${key});
            if (pos >= 0) {
                printf("Found %d at index %d\n", ${key}, pos);
            } else {
                printf("%d is not in the array\n", ${key});
            }
            return 0;
        }""", key=key, a=a, n=n, i=i, init=braces(vals), decl=length_decl(a, n, len(vals)), target=target)
    return cap, ["stdio.h"], code, f"linear_search() checks the elements of {a} one by one and returns the index of the first one equal to {key}, or -1 when it reaches the end without a match. main() searches for {target} and prints where it is, or that it is missing.", [f"Write linear_search() with the array, its length and {key}.", f"Compare each element with {key}.", "Return the index of the first match, or -1.", f"In main(), search for {target} and print the result."]


def t_binary_search():
    a, n = arr_name(), len_name()
    vals = sorted(R.sample(range(1, 200), R.randint(6, 12))); target = R.choice(vals) if R.random() < 0.7 else 201
    lo, hi, mid = pick(("low", "high", "mid"), ("lo", "hi", "mid"), ("left", "right", "middle"))
    cap = pick("Find a number in a sorted array with binary search.", "Use binary search to locate a value in a sorted list.",
               "Search a sorted array by halving the search range each step.")
    code = fill(r"""
        /* The array must be sorted. Each step halves the range that can still hold the key. */
        int binary_search(const int ${a}[], int ${n}, int key) {
            int ${lo} = 0;
            int ${hi} = ${n} - 1;
            while (${lo} <= ${hi}) {
                int ${mid} = ${lo} + (${hi} - ${lo}) / 2;
                if (${a}[${mid}] == key) {
                    return ${mid};
                } else if (${a}[${mid}] < key) {
                    ${lo} = ${mid} + 1;   /* key is in the right half */
                } else {
                    ${hi} = ${mid} - 1;   /* key is in the left half */
                }
            }
            return -1;
        }

        int main(void) {
            int ${a}[] = ${init};
            ${decl}
            int index = binary_search(${a}, ${n}, ${target});
            printf("Index of ${target}: %d\n", index);
            return 0;
        }""", a=a, n=n, lo=lo, hi=hi, mid=mid, init=braces(map(str, vals)), decl=length_decl(a, n, len(vals)),
        target=target)
    return cap, ["stdio.h"], code, f"binary_search() needs a sorted array. It keeps the range from {lo} to {hi}, compares the middle element {mid} with the key and throws away the half that cannot contain it, so it needs only about log2(n) steps. main() looks for {target} and prints its index, or -1 if it is not there.", ["Make sure the array is sorted.", f"Set {lo} to 0 and {hi} to {n} - 1.", f"While {lo} <= {hi}, compare the middle element {mid} with the key.", "Return the index if it matches; otherwise keep only the half that can hold the key.", f"Return -1 if the range becomes empty, and print the index of {target}."]


def t_sort():
    typ, fmt = numtype(); a, n = arr_name(), len_name(); vals = values(typ, R.randint(5, 10))
    kind = pick("bubble", "selection", "insertion"); desc = R.random() < 0.25
    order = "descending" if desc else "ascending"; cmp = "<" if desc else ">"
    cap = pick(f"Sort an array in {order} order using {kind} sort.", f"Implement {kind} sort for an array of {typ} values.",
               f"Use {kind} sort to put numbers in {order} order.")
    if kind == "bubble":
        algo = fill(r"""
            /* Repeatedly swap neighbours that are in the wrong order. */
            void bubble_sort(${typ} ${a}[], int ${n}) {
                for (int i = 0; i < ${n} - 1; i++) {
                    for (int j = 0; j < ${n} - 1 - i; j++) {
                        if (${a}[j] ${cmp} ${a}[j + 1]) {
                            ${typ} tmp = ${a}[j];
                            ${a}[j] = ${a}[j + 1];
                            ${a}[j + 1] = tmp;
                        }
                    }
                }
            }""", typ=typ, a=a, n=n, cmp=cmp)
    elif kind == "selection":
        algo = fill(r"""
            /* Select the right element for each position and swap it into place. */
            void selection_sort(${typ} ${a}[], int ${n}) {
                for (int i = 0; i < ${n} - 1; i++) {
                    int chosen = i;
                    for (int j = i + 1; j < ${n}; j++) {
                        if (${a}[chosen] ${cmp} ${a}[j]) {
                            chosen = j;
                        }
                    }
                    ${typ} tmp = ${a}[i];
                    ${a}[i] = ${a}[chosen];
                    ${a}[chosen] = tmp;
                }
            }""", typ=typ, a=a, n=n, cmp=cmp)
    else:
        algo = fill(r"""
            /* Insert each element into the already sorted part on its left. */
            void insertion_sort(${typ} ${a}[], int ${n}) {
                for (int i = 1; i < ${n}; i++) {
                    ${typ} current = ${a}[i];
                    int j = i - 1;
                    while (j >= 0 && ${a}[j] ${cmp} current) {
                        ${a}[j + 1] = ${a}[j];   /* shift bigger values right */
                        j--;
                    }
                    ${a}[j + 1] = current;
                }
            }""", typ=typ, a=a, n=n, cmp=cmp)
    code = algo + "\n\n" + fill(r"""
        int main(void) {
            ${typ} ${a}[] = ${init};
            ${decl}
            ${kind}_sort(${a}, ${n});
        ${show}
            return 0;
        }""", typ=typ, a=a, init=braces(vals), decl=length_decl(a, n, len(vals)), kind=kind, n=n,
        show=textwrap.indent(print_array(typ, fmt, a, n, "i"), "    "))
    return cap, ["stdio.h"], code, (f"bubble_sort() walks through {a} again and again and swaps neighbours that are in the wrong order, so the {'smallest' if desc else 'largest'} values move to the end first." if kind == "bubble" else f"selection_sort() finds the {'largest' if desc else 'smallest'} remaining element for each position and swaps it into place." if kind == "selection" else f"insertion_sort() takes each element in turn and shifts the {'smaller' if desc else 'larger'} values of the sorted part one place right until the element fits.") + f" main() then prints the {len(vals)} values in {order} order.", {"bubble": [f"Write bubble_sort() for the {typ} array {a}.", "Compare each pair of neighbours.", f"Swap them when they are in the wrong order for {order} order.", "Repeat the passes until the array is sorted.", "Print the sorted values."], "selection": [f"Write selection_sort() for the {typ} array {a}.", f"For each position, find the {'largest' if desc else 'smallest'} remaining element.", "Swap it into that position.", "Print the sorted values."], "insertion": [f"Write insertion_sort() for the {typ} array {a}.", "Take each element in turn as current.", f"Shift the {'smaller' if desc else 'larger'} values of the sorted part one place right.", "Put current into the gap.", "Print the sorted values."]}[kind]


def t_reverse_array():
    typ, fmt = numtype(); a, n = arr_name(), len_name(); vals = values(typ, R.randint(4, 9))
    l, r = pick(("left", "right"), ("start", "end"), ("i", "j"))
    cap = pick("Reverse the elements of an array in place.", "Reverse an array by swapping from both ends.")
    code = fill(r"""
        /* Swap the outer pair, then move both ends toward the middle. */
        void reverse_array(${typ} ${a}[], int ${n}) {
            int ${l} = 0;
            int ${r} = ${n} - 1;
            while (${l} < ${r}) {
                ${typ} tmp = ${a}[${l}];
                ${a}[${l}] = ${a}[${r}];
                ${a}[${r}] = tmp;
                ${l}++;
                ${r}--;
            }
        }

        int main(void) {
            ${typ} ${a}[] = ${init};
            ${decl}
            reverse_array(${a}, ${n});
        ${show}
            return 0;
        }""", typ=typ, a=a, n=n, l=l, r=r, init=braces(vals), decl=length_decl(a, n, len(vals)),
        show=textwrap.indent(print_array(typ, fmt, a, n, "i"), "    "))
    return cap, ["stdio.h"], code, f"reverse_array() swaps the first and last elements, then moves {l} forward and {r} backward until they meet, so the array is reversed in place without a second array. main() prints the reversed values.", [f"Write reverse_array() for {a}.", f"Set {l} to the first index and {r} to the last.", f"Swap {a}[{l}] and {a}[{r}], then move both inwards.", f"Stop when {l} meets {r} and print the array."]


# ----------------------------------------------------------------------------- math
def t_factorial():
    x = R.randint(3, 15); v = pick("n", "num", "x", "value"); rec = R.random() < 0.5
    cap = pick(f"Compute the factorial of a number {'recursively' if rec else 'with a loop'}.",
               f"Calculate {x}! using {'recursion' if rec else 'iteration'}.", "Print the factorial of an integer.")
    if rec:
        fn = fill(r"""
            /* n! = n * (n - 1)!, and 0! = 1 stops the recursion. */
            unsigned long long factorial(int ${v}) {
                if (${v} <= 1) {
                    return 1;
                }
                return ${v} * factorial(${v} - 1);
            }""", v=v)
    else:
        fn = fill(r"""
            /* Multiply 1 * 2 * ... * n. */
            unsigned long long factorial(int ${v}) {
                unsigned long long result = 1;
                for (int i = 2; i <= ${v}; i++) {
                    result *= i;
                }
                return result;
            }""", v=v)
    code = fn + "\n\n" + fill(r"""
        int main(void) {
            int ${v} = ${x};
            printf("%d! = %llu\n", ${v}, factorial(${v}));
            return 0;
        }""", v=v, x=x)
    return cap, ["stdio.h"], code, (f"factorial() calls itself: n! is n * (n - 1)!, and the recursion stops when {v} is 1 or less." if rec else f"factorial() multiplies result by every number from 2 up to {v}.") + f" main() prints {x}! as an unsigned long long, because factorials grow very fast.", ([f"Write factorial({v}) that calls itself.", f"Return 1 when {v} is 1 or less.", f"Otherwise return {v} * factorial({v} - 1)."] if rec else [f"Write factorial({v}) with result = 1.", f"Multiply result by every number from 2 to {v}.", "Return result."]) + [f"Print {x}! as an unsigned long long."]


def t_fibonacci():
    k = R.randint(8, 25); rec = R.random() < 0.35
    cap = pick(f"Print the first {k} Fibonacci numbers.", "Generate the Fibonacci sequence.",
               f"Show the Fibonacci series {'using recursion' if rec else 'with a loop'}.")
    if rec:
        code = fill(r"""
            /* Each Fibonacci number is the sum of the two before it. */
            long fib(int n) {
                if (n < 2) {
                    return n;
                }
                return fib(n - 1) + fib(n - 2);
            }

            int main(void) {
                for (int i = 0; i < ${k}; i++) {
                    printf("%ld ", fib(i));
                }
                printf("\n");
                return 0;
            }""", k=k)
    else:
        a, b = pick(("a", "b"), ("prev", "curr"), ("first", "second"))
        code = fill(r"""
            int main(void) {
                long ${a} = 0, ${b} = 1;
                /* Print a term, then slide the pair one step along the sequence. */
                for (int i = 0; i < ${k}; i++) {
                    printf("%ld ", ${a});
                    long next = ${a} + ${b};
                    ${a} = ${b};
                    ${b} = next;
                }
                printf("\n");
                return 0;
            }""", a=a, b=b, k=k)
    return cap, ["stdio.h"], code, (f"fib() returns n for 0 and 1 and otherwise adds the two previous Fibonacci numbers by calling itself twice, which is simple but slow for large n. main() prints the first {k} numbers." if rec else f"The loop keeps the last two numbers in {a} and {b}. Each round it prints {a}, then moves the pair forward: {a} becomes {b} and {b} becomes their sum. It prints the first {k} Fibonacci numbers."), (["Write fib(n) that returns n when n < 2.", "Otherwise return fib(n - 1) + fib(n - 2).", f"Print fib(i) for i from 0 to {k - 1}."] if rec else [f"Start with {a} = 0 and {b} = 1.", f"Repeat {k} times and print {a} each time.", f"Compute next = {a} + {b}, then set {a} = {b} and {b} = next."])


def t_gcd_lcm():
    x, y = R.randint(6, 300), R.randint(6, 300); lcm = R.random() < 0.4
    cap = (pick("Find the least common multiple of two numbers.", "Compute the LCM of two integers using the GCD.")
           if lcm else pick("Find the greatest common divisor of two numbers.", "Use Euclid's algorithm to compute the GCD."))
    code = fill(r"""
        /* Euclid: gcd(a, b) = gcd(b, a % b) until b becomes zero. */
        int gcd(int a, int b) {
            while (b != 0) {
                int r = a % b;
                a = b;
                b = r;
            }
            return a;
        }""")
    if lcm:
        code += "\n\n" + fill(r"""
            /* lcm(a, b) = a / gcd(a, b) * b (divide first to avoid overflow). */
            long lcm(int a, int b) {
                return (long)a / gcd(a, b) * b;
            }

            int main(void) {
                printf("lcm(${x}, ${y}) = %ld\n", lcm(${x}, ${y}));
                return 0;
            }""", x=x, y=y)
    else:
        code += "\n\n" + fill(r"""
            int main(void) {
                int x = ${x}, y = ${y};
                printf("gcd(%d, %d) = %d\n", x, y, gcd(x, y));
                return 0;
            }""", x=x, y=y)
    return cap, ["stdio.h"], code, "gcd() uses Euclid's algorithm: it keeps replacing (a, b) with (b, a % b) until b is 0. " + (f"lcm() divides a by the gcd before multiplying by b, which avoids overflow. main() prints the LCM of {x} and {y}." if lcm else f"The last non-zero value is the greatest common divisor, and main() prints the GCD of {x} and {y}."), (["Write gcd() with Euclid's algorithm: replace (a, b) by (b, a % b) until b is 0.", "Write lcm() as a / gcd(a, b) * b.", f"Print the LCM of {x} and {y}."] if lcm else ["Write gcd() with Euclid's algorithm.", "Replace (a, b) by (b, a % b) until b is 0.", f"Return a and print the GCD of {x} and {y}."])


def t_prime():
    many = R.random() < 0.5; limit = R.randint(20, 120); x = R.randint(2, 500)
    cap = (pick(f"Print all prime numbers up to {limit}.", "List the prime numbers below a limit.")
           if many else pick("Check whether a number is prime.", f"Test if {x} is a prime number."))
    head = fill(r"""
        /* A prime has no divisor between 2 and its square root. */
        bool is_prime(int n) {
            if (n < 2) {
                return false;
            }
            for (int d = 2; d * d <= n; d++) {
                if (n % d == 0) {
                    return false;
                }
            }
            return true;
        }""")
    if many:
        main = fill(r"""
            int main(void) {
                for (int n = 2; n <= ${limit}; n++) {
                    if (is_prime(n)) {
                        printf("%d ", n);
                    }
                }
                printf("\n");
                return 0;
            }""", limit=limit)
    else:
        main = fill(r"""
            int main(void) {
                int number = ${x};
                printf("%d %s prime\n", number, is_prime(number) ? "is" : "is not");
                return 0;
            }""", x=x)
    return cap, ["stdio.h", "stdbool.h"], head + "\n\n" + main, "is_prime() tries every divisor d from 2 while d * d <= n; if one divides n evenly, n is not prime. " + (f"main() tests every number from 2 to {limit} and prints the primes." if many else f"main() checks {x} and prints whether it is prime."), ["Write is_prime(n); numbers below 2 are not prime.", "Try every divisor d while d * d <= n.", "Return false if one divides n, otherwise true."] + ([f"Print every prime from 2 to {limit}."] if many else [f"Check {x} and print the result."])


def t_power():
    b, e = R.randint(2, 9), R.randint(2, 12)
    cap = pick("Raise a number to an integer power with a loop.", f"Compute {b} to the power of {e} without pow().")
    code = fill(r"""
        /* Multiply the base by itself exp times. */
        long long power(int base, int exp) {
            long long result = 1;
            for (int i = 0; i < exp; i++) {
                result *= base;
            }
            return result;
        }

        int main(void) {
            printf("${b}^${e} = %lld\n", power(${b}, ${e}));
            return 0;
        }""", b=b, e=e)
    return cap, ["stdio.h"], code, f"power() starts result at 1 and multiplies it by base exp times. It uses long long because powers grow quickly. main() prints {b} to the power {e}.", ["Write power(base, exp) with result = 1.", "Multiply result by base exp times.", f"Print {b} to the power {e}."]


def t_digits():
    x = R.randint(100, 999999); kind = pick("sum", "reverse", "count")
    cap = {"sum": pick("Add up the digits of a number.", "Compute the sum of the digits of an integer."),
           "reverse": pick("Reverse the digits of a number.", "Print an integer with its digits reversed."),
           "count": pick("Count how many digits a number has.", "Find the number of digits in an integer.")}[kind]
    step = {"sum": "result += n % 10;", "reverse": "result = result * 10 + n % 10;", "count": "result++;"}[kind]
    code = fill(r"""
        /* Take the last digit with % 10, then drop it with / 10. */
        int ${kind}_digits(int n) {
            int result = 0;
            while (n > 0) {
                ${step}
                n /= 10;
            }
            return result;
        }

        int main(void) {
            int number = ${x};
            printf("${kind} of digits of %d: %d\n", number, ${kind}_digits(number));
            return 0;
        }""", kind=kind, step=step, x=x)
    return cap, ["stdio.h"], code, {"sum": "sum_digits() adds the last digit (n % 10) to result and then removes it (n / 10) until no digits are left.", "reverse": "reverse_digits() shifts result one place left (result * 10) and adds the last digit of n, which reverses the order.", "count": "count_digits() removes the last digit (n / 10) and adds one to result each time, until n reaches 0."}[kind] + f" main() applies it to {x}.", {"sum": ["Start result at 0.", "Add n % 10 to result, then divide n by 10."], "reverse": ["Start result at 0.", "Multiply result by 10, add n % 10, then divide n by 10."], "count": ["Start result at 0.", "Divide n by 10 and add one to result."]}[kind] + ["Repeat until n is 0.", f"Apply it to {x} and print the result."]


def t_leap_year():
    y = R.randint(1890, 2100)
    cap = pick("Check whether a year is a leap year.", f"Decide if {y} is a leap year.")
    code = fill(r"""
        /* Leap years divide by 4, except centuries, which must divide by 400. */
        int is_leap(int year) {
            return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        }

        int main(void) {
            int year = ${y};
            if (is_leap(year)) {
                printf("%d is a leap year\n", year);
            } else {
                printf("%d is not a leap year\n", year);
            }
            return 0;
        }""", y=y)
    return cap, ["stdio.h"], code, f"is_leap() returns true for years divisible by 4, except century years, which must also be divisible by 400. main() checks {y} and prints whether it is a leap year.", ["Write is_leap(year).", "A year divisible by 4 is a leap year, unless it is divisible by 100.", "Years divisible by 400 are leap years again.", f"Check {y} and print the result."]


def t_table():
    kind = pick("mult", "temp", "squares")
    if kind == "mult":
        x, k = R.randint(2, 15), R.randint(5, 12)
        cap = pick(f"Print the multiplication table of {x}.", "Display a multiplication table with a loop.")
        code = fill(r"""
            int main(void) {
                int number = ${x};
                /* One line per multiplier: number x i = product. */
                for (int i = 1; i <= ${k}; i++) {
                    printf("%d x %d = %d\n", number, i, number * i);
                }
                return 0;
            }""", x=x, k=k)
    elif kind == "temp":
        lo, hi, step = R.choice([0, -20, -10]), R.choice([40, 60, 100]), R.choice([5, 10, 20])
        cap = pick("Print a Celsius to Fahrenheit conversion table.", "Convert temperatures from Celsius to Fahrenheit.")
        code = fill(r"""
            /* F = C * 9 / 5 + 32 */
            double to_fahrenheit(double celsius) {
                return celsius * 9.0 / 5.0 + 32.0;
            }

            int main(void) {
                printf("Celsius  Fahrenheit\n");
                for (int c = ${lo}; c <= ${hi}; c += ${step}) {
                    printf("%7d  %10.1f\n", c, to_fahrenheit(c));
                }
                return 0;
            }""", lo=lo, hi=hi, step=step)
    else:
        k = R.randint(5, 15)
        cap = pick(f"Print the squares and cubes of 1 to {k}.", "Print a table of numbers with their squares and cubes.")
        code = fill(r"""
            int main(void) {
                printf("n  square  cube\n");
                for (int n = 1; n <= ${k}; n++) {
                    printf("%d  %d  %d\n", n, n * n, n * n * n);
                }
                return 0;
            }""", k=k)
    return cap, ["stdio.h"], code, (f"The for loop runs i from 1 to {k} and prints {x} times i on each line." if kind == "mult" else f"to_fahrenheit() applies F = C * 9 / 5 + 32. main() prints a table from {lo} to {hi} degrees Celsius in steps of {step}." if kind == "temp" else f"The loop goes from 1 to {k} and prints each number with its square (n * n) and its cube (n * n * n)."), ([f"Loop i from 1 to {k}.", f"Print {x} x i = {x} * i on each line."] if kind == "mult" else ["Write to_fahrenheit() with F = C * 9 / 5 + 32.", f"Loop c from {lo} to {hi} in steps of {step}.", "Print each Celsius value with its Fahrenheit value."] if kind == "temp" else [f"Loop n from 1 to {k}.", "Print n, n * n and n * n * n on each line."])


def t_binary():
    x = R.randint(1, 1023)
    cap = pick("Print the binary representation of an integer.", f"Convert {x} to binary.")
    code = fill(r"""
        /* Check each bit from the highest to the lowest. */
        void print_binary(unsigned int n, int bits) {
            for (int b = bits - 1; b >= 0; b--) {
                putchar((n >> b) & 1 ? '1' : '0');
            }
            putchar('\n');
        }

        int main(void) {
            unsigned int number = ${x};
            printf("%u in binary: ", number);
            print_binary(number, ${bits});
            return 0;
        }""", x=x, bits=pick(8, 10, 12, 16))
    return cap, ["stdio.h"], code, f"print_binary() checks one bit at a time, from the highest to the lowest: (n >> b) & 1 is 1 when bit b is set, so it prints the character 1 or 0. main() prints {x} in binary.", ["Write print_binary(n, bits).", "Go through the bits from the highest to the lowest.", "Print 1 when (n >> b) & 1 is set, otherwise 0.", f"Print {x} in binary."]


# ----------------------------------------------------------------------------- strings
def t_string():
    w = R.choice(WORDS); s = pick("text", "word", "str", "buffer", "line")
    kind = pick("length", "reverse", "palindrome", "upper", "copy")
    if kind == "length":
        cap = pick("Compute the length of a string without strlen.", "Count the characters in a string.")
        code = fill(r"""
            /* Count characters until the terminating '\0'. */
            int string_length(const char *${s}) {
                int len = 0;
                while (${s}[len] != '\0') {
                    len++;
                }
                return len;
            }

            int main(void) {
                const char *${s} = "${w}";
                printf("Length of \"%s\" is %d\n", ${s}, string_length(${s}));
                return 0;
            }""", s=s, w=w)
        return cap, ["stdio.h"], code, f"string_length() counts characters until it reaches the null character that ends every C string. main() prints the length of the word {w}.", ["Write string_length() with a counter len = 0.", "Move through the string until the null character.", f"Return len and print the length of {w}."]
    if kind == "reverse":
        cap = pick("Reverse a string in place.", "Reverse the characters of a word.")
        code = fill(r"""
            /* Swap characters from both ends until they meet in the middle. */
            void reverse_string(char *${s}) {
                int i = 0;
                int j = strlen(${s}) - 1;
                while (i < j) {
                    char tmp = ${s}[i];
                    ${s}[i] = ${s}[j];
                    ${s}[j] = tmp;
                    i++;
                    j--;
                }
            }

            int main(void) {
                char ${s}[] = "${w}";
                reverse_string(${s});
                printf("%s\n", ${s});
                return 0;
            }""", s=s, w=w)
        return cap, ["stdio.h", "string.h"], code, f"reverse_string() swaps characters from both ends of {s}, moving inwards until the two indexes meet. main() reverses the word {w} in place and prints it.", [f"Write reverse_string() for {s}.", "Set i to the first index and j to the last.", "Swap the two characters and move i and j inwards until they meet.", f"Reverse {w} and print it."]
    if kind == "palindrome":
        cap = pick("Check whether a word is a palindrome.", f"Test if \"{w}\" reads the same backwards.")
        code = fill(r"""
            /* Compare characters from the front and back, moving inwards. */
            bool is_palindrome(const char *${s}) {
                int i = 0;
                int j = strlen(${s}) - 1;
                while (i < j) {
                    if (${s}[i] != ${s}[j]) {
                        return false;
                    }
                    i++;
                    j--;
                }
                return true;
            }

            int main(void) {
                const char *${s} = "${w}";
                printf("%s %s a palindrome\n", ${s}, is_palindrome(${s}) ? "is" : "is not");
                return 0;
            }""", s=s, w=w)
        return cap, ["stdio.h", "string.h", "stdbool.h"], code, f"is_palindrome() compares the first and last characters and moves inwards; any mismatch means the word is not a palindrome. main() checks the word {w} and prints the result.", [f"Write is_palindrome() for {s}.", "Compare the first and last characters, moving inwards.", "Return false at the first mismatch, otherwise true.", f"Check {w} and print the result."]
    if kind == "upper":
        line = R.choice(SENTENCES)
        cap = pick("Convert a string to uppercase.", "Change every letter in a sentence to a capital letter.")
        code = fill(r"""
            /* toupper changes lowercase letters and leaves other characters alone. */
            void to_upper(char *${s}) {
                for (int i = 0; ${s}[i] != '\0'; i++) {
                    ${s}[i] = toupper((unsigned char)${s}[i]);
                }
            }

            int main(void) {
                char ${s}[] = "${line}";
                to_upper(${s});
                printf("%s\n", ${s});
                return 0;
            }""", s=s, line=line)
        return cap, ["stdio.h", "ctype.h"], code, f"to_upper() walks through {s} until the null character and replaces each character with toupper(), which only changes lowercase letters. main() prints the sentence in capitals.", [f"Write to_upper() for {s}.", "Replace every character with toupper() until the null character.", "Print the sentence."]
    cap = pick("Copy one string into another by hand.", "Write your own version of strcpy.")
    code = fill(r"""
        /* Copy each character, including the final '\0'. */
        void copy_string(char *dest, const char *src) {
            int i = 0;
            while ((dest[i] = src[i]) != '\0') {
                i++;
            }
        }

        int main(void) {
            char source[] = "${w}";
            char ${s}[${size}];
            copy_string(${s}, source);
            printf("Copied: %s\n", ${s});
            return 0;
        }""", w=w, s=s, size=pick(32, 64, 100))
    return cap, ["stdio.h"], code, f"copy_string() copies src into dest one character at a time, including the final null character, which is what strcpy does. The array {s} must be big enough to hold the copy. main() copies the word {w} and prints it.", ["Write copy_string(dest, src).", "Copy src into dest one character at a time.", "Stop after copying the null character.", f"Copy {w} into {s} and print it."]


def t_text_count():
    line = R.choice(SENTENCES); kind = pick("vowels", "words", "char")
    if kind == "vowels":
        cap = pick("Count the vowels in a sentence.", "Count how many vowels a string contains.")
        code = fill(r"""
            /* strchr finds the letter in the list of vowels, or returns NULL. */
            int count_vowels(const char *text) {
                int count = 0;
                for (int i = 0; text[i] != '\0'; i++) {
                    if (strchr("aeiouAEIOU", text[i]) != NULL) {
                        count++;
                    }
                }
                return count;
            }

            int main(void) {
                const char *sentence = "${line}";
                printf("Vowels: %d\n", count_vowels(sentence));
                return 0;
            }""", line=line)
        return cap, ["stdio.h", "string.h"], code, "count_vowels() uses strchr to test whether each character is one of aeiouAEIOU and counts the matches. main() prints the number of vowels in the sentence.", ["Write count_vowels() with count = 0.", "For each character, check whether strchr finds it in aeiouAEIOU.", "Return count and print it."]
    if kind == "words":
        cap = pick("Count the words in a sentence.", "Count how many words are separated by spaces in a string.")
        code = fill(r"""
            /* A new word starts when a non-space follows a space (or the start). */
            int count_words(const char *text) {
                int words = 0;
                int in_word = 0;
                for (int i = 0; text[i] != '\0'; i++) {
                    if (isspace((unsigned char)text[i])) {
                        in_word = 0;
                    } else if (!in_word) {
                        in_word = 1;
                        words++;
                    }
                }
                return words;
            }

            int main(void) {
                const char *sentence = "${line}";
                printf("Words: %d\n", count_words(sentence));
                return 0;
            }""", line=line)
        return cap, ["stdio.h", "ctype.h"], code, "count_words() uses the flag in_word: a new word starts when a non-space character follows a space or the start of the text. main() prints the number of words in the sentence.", ["Write count_words() with words = 0 and an in_word flag.", "A space ends a word; a non-space after a space starts a new word and adds one to words.", "Return words and print it."]
    ch = R.choice(sorted(set(line.replace(" ", ""))))
    cap = pick(f"Count how many times '{ch}' appears in a string.", "Count the occurrences of a character in a sentence.")
    code = fill(r"""
        int count_char(const char *text, char target) {
            int count = 0;
            /* Compare every character with the target. */
            for (int i = 0; text[i] != '\0'; i++) {
                if (text[i] == target) {
                    count++;
                }
            }
            return count;
        }

        int main(void) {
            const char *sentence = "${line}";
            printf("'${ch}' appears %d times\n", count_char(sentence, '${ch}'));
            return 0;
        }""", line=line, ch=ch)
    return cap, ["stdio.h"], code, f"count_char() compares every character of the text with the target character and counts the matches. main() prints how many times {ch} appears in the sentence.", ["Write count_char() with count = 0.", f"Compare each character with the target {ch}.", "Return count and print it."]


# ----------------------------------------------------------------------------- pointers, structs, memory
def t_swap():
    typ, fmt = numtype(); x, y = values(typ, 2); p, q = pick(("a", "b"), ("x", "y"), ("first", "second"))
    cap = pick("Swap two variables using pointers.", f"Write a function that swaps two {typ} values through pointers.")
    code = fill(r"""
        /* Pass addresses so the function can change the caller's variables. */
        void swap(${typ} *${p}, ${typ} *${q}) {
            ${typ} tmp = *${p};
            *${p} = *${q};
            *${q} = tmp;
        }

        int main(void) {
            ${typ} ${p} = ${x}, ${q} = ${y};
            printf("Before: ${fmt} ${fmt}\n", ${p}, ${q});
            swap(&${p}, &${q});
            printf("After:  ${fmt} ${fmt}\n", ${p}, ${q});
            return 0;
        }""", typ=typ, fmt=fmt, p=p, q=q, x=x, y=y)
    return cap, ["stdio.h"], code, f"swap() receives the addresses of {p} and {q}, so writing through the pointers *{p} and *{q} changes the caller's variables. tmp holds one value during the exchange. main() declares both variables on one line and prints them before and after the swap.", [f"Write swap() that takes pointers to two {typ} values.", f"Save *{p} in tmp, copy *{q} into *{p}, then tmp into *{q}.", f"In main(), print {p} and {q}, call swap(&{p}, &{q}) and print them again."]


NAMES = ["Asha", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gabe", "Hana", "Ivan", "Jia", "Kofi", "Lena", "Maya", "Noor"]


def t_struct():
    kind = pick("student", "point", "rect")
    if kind == "student":
        k = R.randint(2, 5); people = R.sample(NAMES, k)
        rows = ",\n".join(f'        {{"{nm}", {R.randint(40, 100)}}}' for nm in people)
        cap = pick("Store students in an array of structs and find the class average.",
                   "Use a struct to hold names and marks, then print the average mark.")
        code = fill(r"""
            /* A struct groups a student's name and mark together. */
            struct Student {
                char name[20];
                int mark;
            };

            int main(void) {
                struct Student students[] = {
            ${rows}
                };
                int count = sizeof(students) / sizeof(students[0]);
                double total = 0;
                for (int i = 0; i < count; i++) {
                    printf("%s: %d\n", students[i].name, students[i].mark);
                    total += students[i].mark;
                }
                printf("Average mark: %.1f\n", total / count);
                return 0;
            }""", rows=rows)
        return cap, ["stdio.h"], code, f"struct Student groups a name and a mark. main() stores {k} students in an array of structs, prints each one, adds the marks to total and prints the average.", ["Define struct Student with a name and a mark.", f"Store {k} students in an array.", "Print each student and add the marks to total.", "Print total divided by the number of students."]
    if kind == "point":
        x1, y1, x2, y2 = (R.randint(-10, 10) for _ in range(4))
        cap = pick("Compute the distance between two points using a struct.", "Define a Point struct and measure the distance between two points.")
        code = fill(r"""
            typedef struct {
                double x;
                double y;
            } Point;

            /* Pythagoras: sqrt(dx * dx + dy * dy). */
            double distance(Point a, Point b) {
                double dx = b.x - a.x;
                double dy = b.y - a.y;
                return sqrt(dx * dx + dy * dy);
            }

            int main(void) {
                Point p = {${x1}, ${y1}};
                Point q = {${x2}, ${y2}};
                printf("Distance: %.3f\n", distance(p, q));
                return 0;
            }""", x1=x1, y1=y1, x2=x2, y2=y2)
        return cap, ["stdio.h", "math.h"], code, "Point is a typedef struct with x and y. distance() uses Pythagoras, sqrt(dx * dx + dy * dy), to measure the gap between the two points, and main() prints it.", ["Define a Point struct with x and y.", "Write distance() using sqrt(dx * dx + dy * dy).", "Print the distance between p and q."]
    w, h = R.randint(1, 30), R.randint(1, 30)
    cap = pick("Use a struct to store a rectangle and print its area and perimeter.", "Calculate the area and perimeter of a rectangle struct.")
    code = fill(r"""
        typedef struct {
            int width;
            int height;
        } Rectangle;

        int area(Rectangle r) {
            return r.width * r.height;
        }

        int perimeter(Rectangle r) {
            return 2 * (r.width + r.height);
        }

        int main(void) {
            Rectangle box = {${w}, ${h}};
            printf("Area: %d, Perimeter: %d\n", area(box), perimeter(box));
            return 0;
        }""", w=w, h=h)
    return cap, ["stdio.h"], code, f"Rectangle stores a width and a height. area() returns width * height and perimeter() returns 2 * (width + height). main() makes a {w} by {h} rectangle and prints both results.", ["Define a Rectangle struct with width and height.", "Write area() and perimeter().", f"Make a {w} by {h} rectangle and print both results."]


def t_linked_list():
    vals = values("int", R.randint(3, 6)); node = pick("Node", "ListNode", "Item")
    cap = pick("Build a linked list, print it, and free it.", "Create a singly linked list by inserting at the head.",
               "Implement a simple linked list with malloc and free.")
    pushes = "\n".join(f"    head = push_front(head, {v});" for v in vals)
    code = fill(r"""
        typedef struct ${node} {
            int value;
            struct ${node} *next;
        } ${node};

        /* Allocate a new node and put it in front of the current head. */
        ${node} *push_front(${node} *head, int value) {
            ${node} *node = malloc(sizeof(${node}));
            if (node == NULL) {
                fprintf(stderr, "Out of memory\n");
                exit(EXIT_FAILURE);
            }
            node->value = value;
            node->next = head;
            return node;
        }

        void print_list(const ${node} *head) {
            for (const ${node} *cur = head; cur != NULL; cur = cur->next) {
                printf("%d -> ", cur->value);
            }
            printf("NULL\n");
        }

        /* Free every node; save next before freeing the current one. */
        void free_list(${node} *head) {
            while (head != NULL) {
                ${node} *next = head->next;
                free(head);
                head = next;
            }
        }

        int main(void) {
            ${node} *head = NULL;
        ${pushes}
            print_list(head);
            free_list(head);
            return 0;
        }""", node=node, pushes=pushes)
    return cap, ["stdio.h", "stdlib.h"], code, f"Each {node} holds a value and a pointer to the next node. push_front() allocates a node with malloc, puts it in front of head and returns it; print_list() follows the next pointers until NULL; free_list() frees every node. main() pushes {len(vals)} values, so they print in reverse order.", [f"Define {node} with a value and a next pointer.", "Write push_front() that mallocs a node and links it in front of head.", "Write print_list() that follows next until NULL.", "Write free_list() that frees every node.", f"Push {len(vals)} values, print the list and free it."]


def t_stack_queue():
    vals = values("int", R.randint(3, 6)); cap_n = R.choice([8, 10, 16, 32])
    if R.random() < 0.5:
        cap = pick("Implement a stack with an array (push and pop).", "Build a fixed-size integer stack in C.")
        pushes = "\n".join(f"    push(&s, {v});" for v in vals)
        code = fill(r"""
            #define CAPACITY ${cap_n}

            typedef struct {
                int items[CAPACITY];
                int top;   /* number of items on the stack */
            } Stack;

            bool push(Stack *s, int value) {
                if (s->top == CAPACITY) {
                    return false;   /* stack is full */
                }
                s->items[s->top++] = value;
                return true;
            }

            bool pop(Stack *s, int *value) {
                if (s->top == 0) {
                    return false;   /* stack is empty */
                }
                *value = s->items[--s->top];
                return true;
            }

            int main(void) {
                Stack s = {.top = 0};
            ${pushes}
                int value;
                /* Items come back out in reverse order (last in, first out). */
                while (pop(&s, &value)) {
                    printf("%d ", value);
                }
                printf("\n");
                return 0;
            }""", cap_n=cap_n, pushes=pushes)
        return cap, ["stdio.h", "stdbool.h"], code, f"The Stack keeps up to {cap_n} items in an array, and top counts how many are stored. push() adds at the top and pop() removes from the top; both return false when they cannot. main() pushes {len(vals)} values and pops them all, so they come out in reverse order (last in, first out).", [f"Define a Stack with room for {cap_n} items and a top counter.", "Write push() and pop(), returning false when the stack is full or empty.", f"Push {len(vals)} values.", "Pop and print until the stack is empty."]
    cap = pick("Implement a circular queue with an array.", "Build a FIFO queue using a ring buffer.")
    adds = "\n".join(f"    enqueue(&q, {v});" for v in vals)
    code = fill(r"""
        #define CAPACITY ${cap_n}

        typedef struct {
            int items[CAPACITY];
            int head;
            int count;
        } Queue;

        bool enqueue(Queue *q, int value) {
            if (q->count == CAPACITY) {
                return false;
            }
            /* The tail wraps around to the start of the array. */
            q->items[(q->head + q->count) % CAPACITY] = value;
            q->count++;
            return true;
        }

        bool dequeue(Queue *q, int *value) {
            if (q->count == 0) {
                return false;
            }
            *value = q->items[q->head];
            q->head = (q->head + 1) % CAPACITY;
            q->count--;
            return true;
        }

        int main(void) {
            Queue q = {.head = 0, .count = 0};
        ${adds}
            int value;
            /* Items come out in the order they went in (first in, first out). */
            while (dequeue(&q, &value)) {
                printf("%d ", value);
            }
            printf("\n");
            return 0;
        }""", cap_n=cap_n, adds=adds)
    return cap, ["stdio.h", "stdbool.h"], code, f"The Queue stores up to {cap_n} items in a ring buffer: head is the oldest item and % CAPACITY wraps positions back to the start of the array. enqueue() adds at the back and dequeue() removes from the front, so main() gets its {len(vals)} values back in the same order (first in, first out).", [f"Define a Queue with room for {cap_n} items, a head and a count.", "Write enqueue() that stores at (head + count) % CAPACITY.", "Write dequeue() that takes the item at head and moves head on.", f"Add {len(vals)} values, then remove and print them in order."]


def t_matrix():
    rows, cols = R.randint(2, 4), R.randint(2, 4)
    def mat(r, c):
        return "{" + ", ".join("{" + ", ".join(str(R.randint(0, 9)) for _ in range(c)) + "}" for _ in range(r)) + "}"
    kind = pick("add", "transpose", "multiply")
    if kind == "add":
        cap = pick(f"Add two {rows}x{cols} matrices.", "Add two matrices element by element.")
        code = fill(r"""
            #define ROWS ${rows}
            #define COLS ${cols}

            int main(void) {
                int a[ROWS][COLS] = ${ma};
                int b[ROWS][COLS] = ${mb};
                int sum[ROWS][COLS];
                /* Each entry of the result is the sum of the matching entries. */
                for (int r = 0; r < ROWS; r++) {
                    for (int c = 0; c < COLS; c++) {
                        sum[r][c] = a[r][c] + b[r][c];
                        printf("%3d", sum[r][c]);
                    }
                    printf("\n");
                }
                return 0;
            }""", rows=rows, cols=cols, ma=mat(rows, cols), mb=mat(rows, cols))
    elif kind == "transpose":
        cap = pick("Transpose a matrix.", f"Turn the rows of a {rows}x{cols} matrix into columns.")
        code = fill(r"""
            #define ROWS ${rows}
            #define COLS ${cols}

            int main(void) {
                int m[ROWS][COLS] = ${ma};
                int t[COLS][ROWS];
                /* Element (r, c) moves to (c, r). */
                for (int r = 0; r < ROWS; r++) {
                    for (int c = 0; c < COLS; c++) {
                        t[c][r] = m[r][c];
                    }
                }
                for (int r = 0; r < COLS; r++) {
                    for (int c = 0; c < ROWS; c++) {
                        printf("%3d", t[r][c]);
                    }
                    printf("\n");
                }
                return 0;
            }""", rows=rows, cols=cols, ma=mat(rows, cols))
    else:
        n = R.randint(2, 3)
        cap = pick(f"Multiply two {n}x{n} matrices.", "Compute the product of two square matrices.")
        code = fill(r"""
            #define N ${n}

            int main(void) {
                int a[N][N] = ${ma};
                int b[N][N] = ${mb};
                int product[N][N] = {0};
                /* product[i][j] is row i of a times column j of b. */
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        for (int k = 0; k < N; k++) {
                            product[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        printf("%4d", product[i][j]);
                    }
                    printf("\n");
                }
                return 0;
            }""", n=n, ma=mat(n, n), mb=mat(n, n))
    return cap, ["stdio.h"], code, (f"Both matrices are {rows}x{cols} two-dimensional arrays. Two nested loops visit every row r and column c, store a[r][c] + b[r][c] in sum and print each row." if kind == "add" else f"The nested loops copy m[r][c] into t[c][r], so the {rows}x{cols} matrix becomes {cols}x{rows}. A second pair of loops prints the result." if kind == "transpose" else f"For each entry product[i][j] the innermost loop adds a[i][k] * b[k][j] over k, which is row i of a times column j of b. main() multiplies two {n}x{n} matrices and prints the product."), ([f"Define ROWS as {rows} and COLS as {cols}.", "Loop over every row r and column c.", "Store a[r][c] + b[r][c] in sum and print it."] if kind == "add" else [f"Define ROWS as {rows} and COLS as {cols}.", "Copy m[r][c] into t[c][r] for every element.", "Print t row by row."] if kind == "transpose" else [f"Define N as {n}.", "For every i and j, add a[i][k] * b[k][j] over k into product[i][j].", "Print product row by row."])


def t_dynamic_array():
    k = R.randint(5, 15); kind = pick("squares", "cubes", "evens", "multiples")
    expr = {"squares": "i * i", "cubes": "i * i * i", "evens": "2 * i", "multiples": f"{R.randint(3, 9)} * i"}[kind]
    cap = pick(f"Allocate an array with malloc and fill it with {kind}.", f"Use dynamic memory to store the first {k} {kind}.")
    p = pick("arr", "buffer", "table", "list")
    code = fill(r"""
        int main(void) {
            int n = ${k};
            /* Ask the heap for room for n ints; malloc returns NULL on failure. */
            int *${p} = malloc(n * sizeof(int));
            if (${p} == NULL) {
                fprintf(stderr, "malloc failed\n");
                return 1;
            }
            for (int i = 0; i < n; i++) {
                ${p}[i] = ${expr};
            }
            for (int i = 0; i < n; i++) {
                printf("%d ", ${p}[i]);
            }
            printf("\n");
            free(${p});   /* give the memory back */
            return 0;
        }""", k=k, p=p, expr=expr)
    return cap, ["stdio.h", "stdlib.h"], code, f"malloc() reserves room for {k} ints on the heap and returns NULL if it fails, which the program checks. The first loop fills {p} with {kind}, the second prints them, and free() gives the memory back at the end.", [f"Use malloc to get room for {k} ints and check for NULL.", f"Fill {p} with {kind}.", "Print the values.", f"Call free({p}) at the end."]


def t_file_io():
    fname = pick("notes.txt", "output.txt", "log.txt", "data.txt")
    lines = R.sample(SENTENCES, R.randint(2, 4))
    writes = "\n".join(f'    fprintf(file, "{l}\\n");' for l in lines)
    cap = pick("Write lines to a text file and read them back.", "Save text to a file with fprintf and print it with fgets.")
    code = fill(r"""
        int main(void) {
            FILE *file = fopen("${fname}", "w");
            if (file == NULL) {
                perror("fopen");
                return 1;
            }
        ${writes}
            fclose(file);

            /* Reopen the file for reading and print it line by line. */
            file = fopen("${fname}", "r");
            if (file == NULL) {
                perror("fopen");
                return 1;
            }
            char line[128];
            while (fgets(line, sizeof(line), file) != NULL) {
                printf("%s", line);
            }
            fclose(file);
            return 0;
        }""", fname=fname, writes=writes)
    return cap, ["stdio.h"], code, f"fopen() in write mode creates {fname}, fprintf() writes {len(lines)} lines into it and fclose() saves it. The file is then opened again in read mode, and fgets() reads it back line by line into the buffer line, which is printed. Every fopen() result is checked for NULL.", [f"Open {fname} in write mode and check for NULL.", f"Write {len(lines)} lines with fprintf and close the file.", "Open it again in read mode.", "Read and print each line with fgets, then close the file."]


# ----------------------------------------------------------------------------- beginner programs
SCAN = {"int": "%d", "long": "%ld", "double": "%lf"}


def block(template, /, **kw):
    """fill(), then indent 4 spaces: a snippet that goes inside main()."""
    return textwrap.indent(fill(template, **kw), "    ")


def read_or_set(typ, names, label):
    """Declare variables and either read them with scanf or give them fixed values (4-space indented)."""
    if R.random() < 0.5:
        fmts = " ".join(SCAN[typ] for _ in names)
        addrs = ", ".join(f"&{n}" for n in names)
        return block(r"""
                ${typ} ${decl};
                printf("${label}");
                if (scanf("${fmts}", ${addrs}) != ${count}) {
                    printf("Invalid input\n");
                    return 1;
                }""", typ=typ, decl=", ".join(names), label=label, fmts=fmts, addrs=addrs, count=len(names))
    vals = values(typ, len(names), 1, 99)
    return f"    {typ} " + ", ".join(f"{n} = {v}" for n, v in zip(names, vals)) + ";"


def t_two_numbers():
    typ, fmt = numtype(); two = pick("two", "2")
    kind = "decimal numbers" if typ == "double" else pick("numbers", "integers")
    op = pick("add", "add", "subtract", "multiply", "divide")
    sym, res = {"add": ("+", "sum"), "subtract": ("-", "difference"), "multiply": ("*", "product"),
                "divide": ("/", "quotient")}[op]
    a, b = pick(("a", "b"), ("x", "y"), ("num1", "num2"), ("first", "second"))
    out = pick(res, "result")
    cap = {"add": pick(f"Add {two} {kind}.", f"Find the sum of {two} {kind}.", f"Add {two} {kind} and print the sum."),
           "subtract": pick(f"Subtract {two} {kind}.", f"Find the difference of {two} {kind}."),
           "multiply": pick(f"Multiply {two} {kind}.", f"Find the product of {two} {kind}."),
           "divide": pick(f"Divide {two} {kind}.", f"Find the quotient of {two} {kind}.")}[op]
    head = fill(r"""
        int main(void) {
        ${inputs}""", inputs=read_or_set(typ, [a, b], f"Enter {two} numbers: "))
    if op == "divide":
        body = block(r"""
                /* Dividing by zero is not allowed, so check first. */
                if (${b} == 0) {
                    printf("Cannot divide by zero\n");
                    return 1;
                }""", b=b) + "\n"
        if typ == "double":
            body += f"    double {out} = {a} / {b};\n    printf(\"{res.capitalize()}: %.2f\\n\", {out});"
        else:
            body += (f"    /* Integer division drops the fraction; % gives the remainder. */\n"
                     f"    printf(\"{res.capitalize()}: {fmt}, remainder: {fmt}\\n\", {a} / {b}, {a} % {b});")
    elif R.random() < 0.3:
        helper = fill(r"""
            /* Return the ${res} of two numbers. */
            ${typ} ${op}_numbers(${typ} p, ${typ} q) {
                return p ${sym} q;
            }""", res=res, typ=typ, op=op, sym=sym)
        head = helper + "\n\n" + head
        body = f"    printf(\"{res.capitalize()}: {fmt}\\n\", {op}_numbers({a}, {b}));"
    else:
        body = (f"    {typ} {out} = {a} {sym} {b};   /* the {res} */\n"
                f"    printf(\"{fmt} {sym} {fmt} = {fmt}\\n\", {a}, {b}, {out});")
    return cap, ["stdio.h"], head + "\n" + body + "\n    return 0;\n}", f"The program {inputs_desc(head, [a, b])}. " + ((f"It checks that {b} is not zero, divides {a} by {b} and prints the quotient with two decimals." if typ == "double" else f"It checks that {b} is not zero, then prints {a} / {b}, where integer division drops the fraction, and the remainder {a} % {b}.") if op == "divide" else f"The helper {op}_numbers() takes two {typ} parameters and returns p {sym} q, and main() prints the {res}." if f"{op}_numbers(" in head else f"It stores {a} {sym} {b} in {out} and prints the whole calculation."), [f"{rs(head)} {a} and {b}."] + ([f"If {b} is 0, print an error and stop.", (f"Print {a} / {b} with two decimals." if typ == "double" else f"Print {a} / {b} and the remainder {a} % {b}.")] if op == "divide" else [f"Write {op}_numbers() that returns p {sym} q.", f"Print the {res} it returns."] if f"{op}_numbers(" in head else [f"Store {a} {sym} {b} in {out}.", "Print the calculation."])


def t_even_odd():
    v = pick("n", "num", "number", "x", "value")
    cap = pick("Check whether a number is even or odd.", "Tell if an integer is even or odd.",
               "Decide if a number is odd or even.")
    code = fill(r"""
        int main(void) {
        ${inputs}
            /* Even numbers leave no remainder when divided by 2. */
            if (${v} % 2 == 0) {
                printf("%d is even\n", ${v});
            } else {
                printf("%d is odd\n", ${v});
            }
            return 0;
        }""", inputs=read_or_set("int", [v], "Enter a number: "), v=v)
    return cap, ["stdio.h"], code, f"The program {inputs_desc(code, [v])}. {v} % 2 is the remainder after dividing by 2: 0 means even, anything else means odd, and the if-else prints which one it is.", [f"{rs(code)} the number {v}.", f"Check whether {v} % 2 is 0.", "Print even or odd."]


def t_biggest():
    typ, fmt = numtype(); three = R.random() < 0.5
    names = list(pick(("a", "b", "c"), ("x", "y", "z"), ("num1", "num2", "num3")))[: 3 if three else 2]
    word = pick("largest", "biggest", "maximum") if three else pick("larger", "bigger", "maximum")
    cnt = pick("three", "3") if three else pick("two", "2")
    cap = pick(f"Find the {word} of {cnt} numbers.", f"Print the {word} of {cnt} {typ} values.",
               f"Compare {cnt} numbers and print the {word}.")
    a, b = names[0], names[1]
    if three:
        c = names[2]
        logic = block(r"""
                ${typ} best = ${a};
                /* Compare with each of the others and keep the bigger one. */
                if (${b} > best) {
                    best = ${b};
                }
                if (${c} > best) {
                    best = ${c};
                }""", typ=typ, a=a, b=b, c=c)
    else:
        logic = pick(f"    {typ} best = ({a} > {b}) ? {a} : {b};   /* the ?: operator picks one of two values */",
                     f"    {typ} best = {b};\n    if ({a} > {b}) {{\n        best = {a};\n    }}")
    code = fill(r"""
        int main(void) {
        ${inputs}
        ${logic}
            printf("The ${word} is ${fmt}\n", best);
            return 0;
        }""", inputs=read_or_set(typ, names, f"Enter {cnt} numbers: "), logic=logic, word=word, fmt=fmt)
    return cap, ["stdio.h"], code, f"The program {inputs_desc(code, names)}. " + (f"It assumes {a} is the {word}, compares it with the other two and keeps the bigger value in best, which it prints." if three else f"The conditional operator ({a} > {b}) ? {a} : {b} picks the bigger value, which is printed." if "?" in code else f"It starts with best = {b} and replaces it with {a} when {a} is bigger, then prints best."), ([f"{rs(code)} {names_list(names)}.", f"Assume {a} is the {word}.", "Compare it with the other two and keep the bigger value.", "Print the result."] if three else [f"{rs(code)} {a} and {b}.", f"Pick the bigger of {a} and {b}.", "Print it."])


def t_formula():
    kind = pick("circle", "triangle", "interest", "temperature", "distance", "average3")
    if kind == "circle":
        cap = pick("Calculate the area of a circle.", "Find the area and circumference of a circle from its radius.")
        code = fill(r"""
            #define PI 3.14159265

            int main(void) {
            ${inputs}
                /* area = PI * r * r, circumference = 2 * PI * r */
                printf("Area: %.2f\n", PI * radius * radius);
                printf("Circumference: %.2f\n", 2 * PI * radius);
                return 0;
            }""", inputs=read_or_set("double", ["radius"], "Enter the radius: "))
    elif kind == "triangle":
        cap = pick("Calculate the area of a triangle.", "Find the area of a triangle from its base and height.")
        code = fill(r"""
            int main(void) {
            ${inputs}
                /* area = base * height / 2 */
                printf("Area: %.2f\n", base * height / 2.0);
                return 0;
            }""", inputs=read_or_set("double", ["base", "height"], "Enter base and height: "))
    elif kind == "interest":
        cap = pick("Calculate simple interest.", "Compute the simple interest on a loan.")
        code = fill(r"""
            int main(void) {
            ${inputs}
                /* simple interest = principal * rate * years / 100 */
                double interest = principal * rate * years / 100.0;
                printf("Interest: %.2f\n", interest);
                printf("Total: %.2f\n", principal + interest);
                return 0;
            }""", inputs=read_or_set("double", ["principal", "rate", "years"], "Enter principal, rate and years: "))
    elif kind == "temperature":
        to_f = R.random() < 0.5
        cap = (pick("Convert a temperature from Celsius to Fahrenheit.", "Change degrees Celsius into Fahrenheit.") if to_f
               else pick("Convert a temperature from Fahrenheit to Celsius.", "Change degrees Fahrenheit into Celsius."))
        src, dst = ("celsius", "fahrenheit") if to_f else ("fahrenheit", "celsius")
        formula = "celsius * 9.0 / 5.0 + 32.0" if to_f else "(fahrenheit - 32.0) * 5.0 / 9.0"
        code = fill(r"""
            int main(void) {
            ${inputs}
                double ${dst} = ${formula};
                printf("%.1f ${src} = %.1f ${dst}\n", ${src}, ${dst});
                return 0;
            }""", inputs=read_or_set("double", [src], "Enter the temperature: "), src=src, dst=dst, formula=formula)
    elif kind == "distance":
        cap = pick("Convert kilometres to miles.", "Change a distance in kilometres into miles.")
        code = fill(r"""
            int main(void) {
            ${inputs}
                /* 1 kilometre is about 0.621371 miles */
                double miles = km * 0.621371;
                printf("%.2f km = %.2f miles\n", km, miles);
                return 0;
            }""", inputs=read_or_set("double", ["km"], "Enter a distance in km: "))
    else:
        cnt = pick("three", "3")
        cap = pick(f"Find the average of {cnt} numbers.", f"Calculate the mean of {cnt} numbers.")
        code = fill(r"""
            int main(void) {
            ${inputs}
                /* divide by 3.0 (not 3) so the result keeps its decimals */
                double average = (a + b + c) / 3.0;
                printf("Average: %.2f\n", average);
                return 0;
            }""", inputs=read_or_set(pick("int", "double"), ["a", "b", "c"], f"Enter {cnt} numbers: "))
    return cap, ["stdio.h"], code, {"circle": lambda: f"The program {inputs_desc(code, ['radius'])}. It uses PI, defined with #define, to print the area (PI * r * r) and the circumference (2 * PI * r).", "triangle": lambda: f"The program {inputs_desc(code, ['base', 'height'])} and prints the area of the triangle, base * height / 2.", "interest": lambda: f"The program {inputs_desc(code, ['principal', 'rate', 'years'])}. Simple interest is principal * rate * years / 100; it prints the interest and the total.", "temperature": lambda: f"The program {inputs_desc(code, [src])} and converts it with {formula}, printing both temperatures with one decimal.", "distance": lambda: f"The program {inputs_desc(code, ['km'])}, multiplies by 0.621371 (miles per kilometre) and prints both distances.", "average3": lambda: f"The program {inputs_desc(code, ['a', 'b', 'c'])}. It divides the sum by 3.0 rather than 3, so the average keeps its decimals, and prints it."}[kind](), {"circle": lambda: [f"{rs(code)} the radius.", "Compute the area PI * r * r.", "Compute the circumference 2 * PI * r.", "Print both."], "triangle": lambda: [f"{rs(code)} the base and the height.", "Compute base * height / 2.", "Print the area."], "interest": lambda: [f"{rs(code)} the principal, the rate and the years.", "Compute principal * rate * years / 100.", "Print the interest and the total."], "temperature": lambda: [f"{rs(code)} the temperature in {src}.", f"Convert it with {formula}.", "Print both temperatures."], "distance": lambda: [f"{rs(code)} the distance in km.", "Multiply by 0.621371 to get miles.", "Print both distances."], "average3": lambda: [f"{rs(code)} a, b and c.", "Divide their sum by 3.0.", "Print the average."]}[kind]()


def t_count_loop():
    n = R.randint(5, 100); kind = pick("print", "sum", "sum_formula", "countdown")
    if kind == "print":
        cap = pick(f"Print the numbers from 1 to {n}.", "Print the numbers from 1 to n with a loop.")
        body = loop("i", "n", 'printf("%d\\n", i + 1);')
        code = fill(r"""
            int main(void) {
                int n = ${n};
            ${body}
                return 0;
            }""", n=n, body=body)
    elif kind == "countdown":
        cap = pick(f"Print a countdown from {n} to 1.", "Count down from n to 1.")
        code = fill(r"""
            int main(void) {
                /* Start high and subtract one each time round the loop. */
                for (int i = ${n}; i >= 1; i--) {
                    printf("%d ", i);
                }
                printf("\n");
                return 0;
            }""", n=n)
    else:
        cap = pick(f"Find the sum of the numbers from 1 to {n}.", "Add up all the numbers from 1 to n.",
                   "Calculate the sum of the first n natural numbers.")
        if kind == "sum":
            calc = block(r"""
                    int sum = 0;
                    for (int i = 1; i <= n; i++) {
                        sum += i;
                    }""")
        else:
            calc = "    /* Gauss's formula: 1 + 2 + ... + n = n * (n + 1) / 2 */\n    int sum = n * (n + 1) / 2;"
        code = fill(r"""
            int main(void) {
                int n = ${n};
            ${calc}
                printf("Sum of 1 to %d = %d\n", n, sum);
                return 0;
            }""", n=n, calc=calc)
    return cap, ["stdio.h"], code, {"print": lambda: f"The loop runs {n} times and prints i + 1, so the output is the numbers 1 to {n}, one per line.", "countdown": lambda: f"The for loop starts at {n} and subtracts one each round (i--) until it reaches 1.", "sum": lambda: f"The for loop adds every number from 1 to {n} to sum and prints the result.", "sum_formula": lambda: f"Instead of a loop it uses Gauss's formula n * (n + 1) / 2, which gives the sum of 1 to {n} in one step."}[kind](), {"print": lambda: [f"Set n = {n}.", "Loop i from 0 to n - 1 and print i + 1."], "countdown": lambda: [f"Start i at {n}.", "Print i and subtract one until i reaches 1."], "sum": lambda: [f"Set n = {n} and sum = 0.", "Add every number from 1 to n to sum.", "Print sum."], "sum_formula": lambda: [f"Set n = {n}.", "Compute n * (n + 1) / 2.", "Print the result."]}[kind]()


def t_hello():
    kind = pick("hello", "hello", "vars")
    if kind == "hello":
        msg = pick("Hello, World!", "Hello, C!", "Hello from C", "Welcome to C programming")
        cap = pick("Write a hello world program.", "Print Hello, World! to the screen.", "Print a greeting message.")
        code = fill(r"""
            int main(void) {
                /* printf prints text; \n moves to a new line */
                printf("${msg}\n");
                return 0;
            }""", msg=msg)
    else:
        name, age, height = R.choice(NAMES), R.randint(8, 80), f"{R.uniform(1.2, 2.0):.2f}"
        cap = pick("Store a name, age and height in variables and print them.", "Print variables of different types with printf.")
        code = fill(r"""
            int main(void) {
                char name[] = "${name}";
                int age = ${age};
                double height = ${height};
                /* %s = string, %d = int, %.2f = double with 2 decimals */
                printf("Name: %s\n", name);
                printf("Age: %d\n", age);
                printf("Height: %.2f m\n", height);
                return 0;
            }""", name=name, age=age, height=height)
    return cap, ["stdio.h"], code, (f"printf() prints the text {msg} and the \\n escape moves the cursor to a new line. Returning 0 from main() tells the system the program succeeded." if kind == "hello" else f"The program stores a name ({name}), an age ({age}) and a height ({height}) in variables of different types and prints them with matching placeholders: %s for the string, %d for the int and %.2f for the double."), (["Include stdio.h for printf.", f"Print {msg} followed by a newline.", "Return 0."] if kind == "hello" else ["Store a name, an age and a height in variables.", "Print each one with the matching placeholder: %s, %d and %.2f."])


# ----------------------------------------------------------------------------- functions with several parameters
def num(lo=0.5, hi=20.0):
    return f"{R.uniform(lo, hi):.1f}"


def t_multi_param():
    kind = pick("clamp", "volume", "in_range", "average3", "max4", "distance", "line")
    if kind == "clamp":
        lo, hi = sorted(R.sample(range(0, 100), 2)); tests = [R.randint(-20, 120) for _ in range(3)]
        cap = pick("Write a function that clamps a value between a minimum and a maximum.",
                   "Limit a number to a range with a three-parameter function.",
                   "Keep a value inside the limits low and high.")
        calls = "\n".join(f'    printf("clamp(%d) = %d\\n", {t}, clamp({t}, low, high));' for t in tests)
        code = fill(r"""
            /* Return value, but never less than low or more than high. */
            int clamp(int value, int low, int high) {
                if (value < low) {
                    return low;
                }
                if (value > high) {
                    return high;
                }
                return value;
            }

            int main(void) {
                int low = ${lo}, high = ${hi};
            ${calls}
                return 0;
            }""", lo=lo, hi=hi, calls=calls)
        expl = (f"clamp() takes three parameters: the value and the two limits. It returns low when the value is too "
                f"small, high when it is too big and the value itself otherwise. main() declares low = {lo} and "
                f"high = {hi} on one line and clamps {names_list(tests)}.")
        return cap, ["stdio.h"], code, expl, ["Write clamp(value, low, high).", "Return low if the value is too small and high if it is too big.", "Otherwise return the value.", f"Set low = {lo} and high = {hi} and print clamp() for {names_list(tests)}."]
    if kind == "volume":
        l, w, h = num(), num(), num()
        cap = pick("Write a function with three parameters that returns the volume of a box.",
                   "Compute the volume of a box from its length, width and height.")
        code = fill(r"""
            /* volume = length * width * height */
            double box_volume(double length, double width, double height) {
                return length * width * height;
            }

            int main(void) {
                double length = ${l}, width = ${w}, height = ${h};
                printf("Volume: %.2f\n", box_volume(length, width, height));
                return 0;
            }""", l=l, w=w, h=h)
        expl = (f"box_volume() has three double parameters and returns their product. main() declares length, width "
                f"and height on one line ({l}, {w} and {h}) and prints the volume with two decimals.")
        return cap, ["stdio.h"], code, expl, ["Write box_volume() with three double parameters.", "Return length * width * height.", "Declare the three sizes on one line and print the volume."]
    if kind == "in_range":
        lo, hi = sorted(R.sample(range(0, 100), 2)); x = R.randint(-10, 110)
        cap = pick("Check whether a number lies between two limits.",
                   "Write a function that tests if a value is inside a range.")
        code = fill(r"""
            /* true when low <= x <= high */
            bool in_range(int x, int low, int high) {
                return x >= low && x <= high;
            }

            int main(void) {
                int value = ${x}, low = ${lo}, high = ${hi};
                printf("%d is %s the range %d to %d\n", value,
                       in_range(value, low, high) ? "inside" : "outside", low, high);
                return 0;
            }""", x=x, lo=lo, hi=hi)
        expl = (f"in_range() takes the value and both limits and returns true only when the value is at least low and "
                f"at most high. main() declares value, low and high on one line and prints whether {x} is inside "
                f"{lo} to {hi}.")
        return cap, ["stdio.h", "stdbool.h"], code, expl, ["Write in_range(x, low, high).", "Return true when x >= low and x <= high.", f"Print whether {x} is inside {lo} to {hi}."]
    if kind == "average3":
        a, b, c = num(1, 100), num(1, 100), num(1, 100)
        cap = pick("Write a function that returns the average of three numbers.",
                   "Pass three numbers to a function and return their mean.")
        code = fill(r"""
            double average_of_three(double a, double b, double c) {
                return (a + b + c) / 3.0;
            }

            int main(void) {
                double a = ${a}, b = ${b}, c = ${c};
                printf("Average: %.2f\n", average_of_three(a, b, c));
                return 0;
            }""", a=a, b=b, c=c)
        expl = (f"average_of_three() receives three doubles, adds them and divides by 3.0. main() declares a, b and c "
                f"on one line ({a}, {b} and {c}) and prints the result.")
        return cap, ["stdio.h"], code, expl, ["Write average_of_three(a, b, c).", "Return (a + b + c) / 3.0.", "Declare a, b and c on one line and print the average."]
    if kind == "max4":
        vals = values("int", 4)
        cap = pick("Find the largest of four numbers with a function.",
                   "Write a function with four parameters that returns the maximum.")
        code = fill(r"""
            /* Return the larger of two values. */
            int max2(int a, int b) {
                return a > b ? a : b;
            }

            /* Reuse max2 to compare four values. */
            int max_of_four(int a, int b, int c, int d) {
                return max2(max2(a, b), max2(c, d));
            }

            int main(void) {
                int a = ${v0}, b = ${v1}, c = ${v2}, d = ${v3};
                printf("Largest: %d\n", max_of_four(a, b, c, d));
                return 0;
            }""", v0=vals[0], v1=vals[1], v2=vals[2], v3=vals[3])
        expl = (f"max_of_four() takes four parameters and reuses the helper max2(), which returns the larger of two "
                f"values: it compares a with b, c with d, and then the two winners. main() prints the largest of "
                f"{names_list(vals)}.")
        return cap, ["stdio.h"], code, expl, ["Write max2() that returns the larger of two values.", "Write max_of_four() as max2(max2(a, b), max2(c, d)).", "Print the largest of the four values."]
    if kind == "distance":
        x1, y1, x2, y2 = (R.randint(-10, 10) for _ in range(4))
        cap = pick("Write a function that returns the distance between two points given x1, y1, x2 and y2.",
                   "Compute the distance between two points with a four-parameter function.")
        code = fill(r"""
            /* Pythagoras: the distance is sqrt(dx * dx + dy * dy). */
            double distance(double x1, double y1, double x2, double y2) {
                double dx = x2 - x1, dy = y2 - y1;
                return sqrt(dx * dx + dy * dy);
            }

            int main(void) {
                printf("Distance: %.3f\n", distance(${x1}, ${y1}, ${x2}, ${y2}));
                return 0;
            }""", x1=x1, y1=y1, x2=x2, y2=y2)
        expl = (f"distance() takes the coordinates of both points as four parameters. It declares dx and dy on one "
                f"line and uses Pythagoras with sqrt() from math.h. main() prints the distance between ({x1}, {y1}) "
                f"and ({x2}, {y2}).")
        return cap, ["stdio.h", "math.h"], code, expl, ["Write distance(x1, y1, x2, y2).", "Compute dx and dy, then return sqrt(dx * dx + dy * dy).", f"Print the distance between ({x1}, {y1}) and ({x2}, {y2})."]
    m, c, k = num(-5, 5), num(-10, 10), R.randint(3, 8)
    cap = pick("Evaluate the straight line y = m * x + c with a function.",
               "Write a function with three parameters that computes y = m * x + c.")
    code = fill(r"""
        /* A straight line: slope m, starting height c. */
        double line_y(double m, double x, double c) {
            return m * x + c;
        }

        int main(void) {
            double m = ${m}, c = ${c};
            for (int x = 0; x <= ${k}; x++) {
                printf("x = %d, y = %.1f\n", x, line_y(m, x, c));
            }
            return 0;
        }""", m=m, c=c, k=k)
    expl = (f"line_y() takes the slope m, the position x and the offset c and returns m * x + c. main() declares "
            f"m = {m} and c = {c} on one line and prints y for every x from 0 to {k}.")
    return cap, ["stdio.h"], code, expl, ["Write line_y(m, x, c) that returns m * x + c.", "Declare m and c on one line.", f"Print y for every x from 0 to {k}."]


# ----------------------------------------------------------------------------- returning results through pointers
def t_out_params():
    kind = pick("min_max", "divide", "sum_avg", "split_time", "circle")
    two_values = "Return two values from a function using pointer parameters."
    if kind == "min_max":
        a, n = arr_name(), len_name(); vals = values("int", R.randint(4, 9))
        lo, hi = pick(("smallest", "largest"), ("min", "max"), ("low", "high"), ("lowest", "highest"))
        cap = pick("Find the minimum and maximum of an array with one function.",
                   "Return both the smallest and the largest value of an array through pointers.", two_values)
        code = fill(r"""
            /* One pass finds both results; they are stored through the pointers. */
            void min_max(const int ${a}[], int ${n}, int *min, int *max) {
                *min = ${a}[0];
                *max = ${a}[0];
                for (int i = 1; i < ${n}; i++) {
                    if (${a}[i] < *min) {
                        *min = ${a}[i];
                    }
                    if (${a}[i] > *max) {
                        *max = ${a}[i];
                    }
                }
            }

            int main(void) {
                int ${a}[] = ${init};
                ${decl}
                int ${lo}, ${hi};
                min_max(${a}, ${n}, &${lo}, &${hi});
                printf("Min: %d, Max: %d\n", ${lo}, ${hi});
                return 0;
            }""", a=a, n=n, lo=lo, hi=hi, init=braces(vals), decl=length_decl(a, n, len(vals)))
        expl = (f"A C function can return only one value, so min_max() returns void and writes its two results "
                f"through the pointer parameters min and max. main() declares {lo} and {hi} on one line and passes "
                f"their addresses with &, then prints both results for the {len(vals)} values.")
        return cap, ["stdio.h"], code, expl, ["Write min_max() with the array, its length and the pointers min and max.", "Start both at the first element.", "Update *min and *max while looping over the rest.", f"Declare {lo} and {hi}, pass their addresses and print them."]
    if kind == "divide":
        x = R.randint(10, 500); y = 0 if R.random() < 0.15 else R.randint(2, 30)
        cap = pick("Divide two integers and get both the quotient and the remainder.",
                   "Write a function that returns the quotient and remainder through pointers.", two_values)
        code = fill(r"""
            /* Returns false for division by zero; otherwise stores both results. */
            bool divide(int a, int b, int *quotient, int *remainder) {
                if (b == 0) {
                    return false;
                }
                *quotient = a / b;
                *remainder = a % b;
                return true;
            }

            int main(void) {
                int a = ${x}, b = ${y};
                int q, r;
                if (divide(a, b, &q, &r)) {
                    printf("%d / %d = %d remainder %d\n", a, b, q, r);
                } else {
                    printf("Cannot divide by zero\n");
                }
                return 0;
            }""", x=x, y=y)
        expl = (f"divide() has four parameters: two inputs and two pointers for the results. Its bool return value "
                f"reports success, while the quotient and remainder are written through the pointers. main() divides "
                f"{x} by {y}" + (", which fails because the divisor is zero." if y == 0 else
                                 " and prints both results."))
        return cap, ["stdio.h", "stdbool.h"], code, expl, ["Write divide(a, b, &quotient, &remainder) that returns a bool.", "Return false if b is 0.", "Store a / b and a % b through the pointers and return true.", f"Divide {x} by {y} and print the results or the error."]
    if kind == "sum_avg":
        a = arr_name(); vals = values("double", R.randint(3, 8))
        cap = pick("Compute the sum and the average of an array in one function.",
                   "Return a sum and an average together using pointer parameters.", two_values)
        code = fill(r"""
            void sum_and_average(const double v[], int n, double *sum, double *average) {
                *sum = 0.0;
                for (int i = 0; i < n; i++) {
                    *sum += v[i];
                }
                *average = *sum / n;
            }

            int main(void) {
                double ${a}[] = ${init};
                int n = sizeof(${a}) / sizeof(${a}[0]);
                double total, mean;
                sum_and_average(${a}, n, &total, &mean);
                printf("Sum: %.2f, Average: %.2f\n", total, mean);
                return 0;
            }""", a=a, init=braces(vals))
        expl = (f"sum_and_average() takes the array, its length and two pointers. It adds up the {len(vals)} values "
                f"through *sum and then stores *sum / n in *average. main() declares total and mean on one line and "
                f"passes their addresses.")
        return cap, ["stdio.h"], code, expl, ["Write sum_and_average() with the pointers sum and average.", "Add the values into *sum.", "Store *sum / n in *average.", "Declare total and mean, pass their addresses and print them."]
    if kind == "split_time":
        t = R.randint(100, 90000)
        cap = pick("Convert seconds into hours, minutes and seconds.",
                   "Split a number of seconds into hours, minutes and seconds with pointer parameters.")
        code = fill(r"""
            void split_seconds(int total, int *hours, int *minutes, int *seconds) {
                *hours = total / 3600;
                *minutes = (total % 3600) / 60;
                *seconds = total % 60;
            }

            int main(void) {
                int total = ${t};
                int h, m, s;
                split_seconds(total, &h, &m, &s);
                printf("%d seconds = %d h %d min %d s\n", total, h, m, s);
                return 0;
            }""", t=t)
        expl = (f"split_seconds() fills three results through pointers: hours is total / 3600, minutes is what is "
                f"left divided by 60, and seconds is total % 60. main() declares h, m and s on one line and converts "
                f"{t} seconds.")
        return cap, ["stdio.h"], code, expl, ["Write split_seconds() with the pointers hours, minutes and seconds.", "Store total / 3600, (total % 3600) / 60 and total % 60 through them.", f"Convert {t} seconds and print the parts."]
    r = num(0.5, 30)
    cap = pick("Compute the area and circumference of a circle with one function.",
               "Return a circle's area and circumference through pointers.", two_values)
    code = fill(r"""
        #define PI 3.14159265

        void circle_measures(double radius, double *area, double *circumference) {
            *area = PI * radius * radius;
            *circumference = 2 * PI * radius;
        }

        int main(void) {
            double radius = ${r};
            double area, circumference;
            circle_measures(radius, &area, &circumference);
            printf("Area: %.2f, Circumference: %.2f\n", area, circumference);
            return 0;
        }""", r=r)
    expl = (f"circle_measures() computes two results from one radius and stores them through the pointers area and "
            f"circumference. main() declares both variables on one line, passes their addresses and prints the "
            f"results for a radius of {r}.")
    return cap, ["stdio.h"], code, expl, ["Write circle_measures() with the pointers area and circumference.", "Store PI * r * r and 2 * PI * r through them.", f"Print both for a radius of {r}."]


# ----------------------------------------------------------------------------- functions that return a struct
def t_struct_return():
    kind = pick("midpoint", "stats", "duration", "fraction")
    if kind == "midpoint":
        x1, y1, x2, y2 = (R.randint(-20, 20) for _ in range(4))
        cap = pick("Write a function that returns the midpoint of two points as a struct.",
                   "Return a Point struct from a function.")
        code = fill(r"""
            typedef struct {
                double x;
                double y;
            } Point;

            /* A function can return a whole struct. */
            Point midpoint(Point a, Point b) {
                Point m = {(a.x + b.x) / 2, (a.y + b.y) / 2};
                return m;
            }

            int main(void) {
                Point p = {${x1}, ${y1}}, q = {${x2}, ${y2}};
                Point mid = midpoint(p, q);
                printf("Midpoint: (%.1f, %.1f)\n", mid.x, mid.y);
                return 0;
            }""", x1=x1, y1=y1, x2=x2, y2=y2)
        expl = (f"midpoint() takes two Point structs and returns a new Point whose x and y are the averages of theirs. "
                f"main() declares p and q on one line, ({x1}, {y1}) and ({x2}, {y2}), and prints the midpoint "
                f"using mid.x and mid.y.")
        return cap, ["stdio.h"], code, expl, ["Define a Point struct with x and y.", "Write midpoint() that returns a Point with the averages of x and y.", "Print mid.x and mid.y."]
    if kind == "stats":
        a = arr_name(); vals = values("int", R.randint(4, 9))
        cap = pick("Return the minimum, maximum and average of an array in a struct.",
                   "Use a struct to return several results from one function.")
        code = fill(r"""
            typedef struct {
                int min;
                int max;
                double average;
            } Stats;

            /* Pack several results into one struct and return it. */
            Stats array_stats(const int v[], int n) {
                Stats s = {v[0], v[0], 0.0};
                double total = 0;
                for (int i = 0; i < n; i++) {
                    if (v[i] < s.min) {
                        s.min = v[i];
                    }
                    if (v[i] > s.max) {
                        s.max = v[i];
                    }
                    total += v[i];
                }
                s.average = total / n;
                return s;
            }

            int main(void) {
                int ${a}[] = ${init};
                int n = sizeof(${a}) / sizeof(${a}[0]);
                Stats st = array_stats(${a}, n);
                printf("Min %d, Max %d, Average %.2f\n", st.min, st.max, st.average);
                return 0;
            }""", a=a, init=braces(vals))
        expl = (f"Stats groups three results. array_stats() fills one Stats while it loops over the {len(vals)} "
                f"values and returns the whole struct, which is a clean way to return several results at once. main() "
                f"reads them as st.min, st.max and st.average.")
        return cap, ["stdio.h"], code, expl, ["Define a Stats struct with min, max and average.", "Write array_stats() that fills one Stats in a loop.", "Return the struct.", "Print st.min, st.max and st.average."]
    if kind == "duration":
        t = R.randint(100, 90000)
        cap = pick("Convert seconds into a struct with hours, minutes and seconds.",
                   "Return a Duration struct built from a number of seconds.")
        code = fill(r"""
            typedef struct {
                int hours;
                int minutes;
                int seconds;
            } Duration;

            Duration from_seconds(int total) {
                Duration d;
                d.hours = total / 3600;
                d.minutes = (total % 3600) / 60;
                d.seconds = total % 60;
                return d;
            }

            int main(void) {
                Duration d = from_seconds(${t});
                printf("%d h %d min %d s\n", d.hours, d.minutes, d.seconds);
                return 0;
            }""", t=t)
        expl = (f"from_seconds() fills the three members of a Duration struct using / and %, then returns the struct. "
                f"main() converts {t} seconds and prints the members.")
        return cap, ["stdio.h"], code, expl, ["Define a Duration struct with hours, minutes and seconds.", "Write from_seconds() that fills it using / and %.", f"Convert {t} seconds and print the members."]
    n1, d1, n2, d2 = R.randint(1, 9), R.randint(2, 9), R.randint(1, 9), R.randint(2, 9)
    cap = pick("Multiply two fractions using a struct.", "Write a function that returns a Fraction struct.")
    code = fill(r"""
        typedef struct {
            int num;
            int den;
        } Fraction;

        /* Multiply the numerators and the denominators. */
        Fraction multiply(Fraction a, Fraction b) {
            Fraction r = {a.num * b.num, a.den * b.den};
            return r;
        }

        int main(void) {
            Fraction x = {${n1}, ${d1}}, y = {${n2}, ${d2}};
            Fraction p = multiply(x, y);
            printf("%d/%d * %d/%d = %d/%d\n", x.num, x.den, y.num, y.den, p.num, p.den);
            return 0;
        }""", n1=n1, d1=d1, n2=n2, d2=d2)
    expl = (f"Fraction stores a numerator and a denominator. multiply() takes two Fractions and returns a new one "
            f"with num = a.num * b.num and den = a.den * b.den. main() multiplies {n1}/{d1} by {n2}/{d2} "
            f"(without reducing the result).")
    return cap, ["stdio.h"], code, expl, ["Define a Fraction struct with num and den.", "Write multiply() that returns num = a.num * b.num and den = a.den * b.den.", f"Print {n1}/{d1} times {n2}/{d2}."]


# ----------------------------------------------------------------------------- declaration styles
def t_declarations():
    kind = pick("same_line", "pointers", "declare_then_assign")
    if kind == "same_line":
        a, b, c = values("int", 3); x, y = values("double", 2); f, l = sorted(R.sample("ABCDEFGHIKLMNOPRSTUVWXYZ", 2))
        cap = pick("Declare several variables on one line and print them.",
                   "Show how to declare variables of different types on one line each.")
        code = fill(r"""
            int main(void) {
                /* Variables of the same type can share one declaration, separated by commas. */
                int a = ${a}, b = ${b}, c = ${c};
                double x = ${x}, y = ${y};
                char first = '${f}', last = '${l}';
                printf("a = %d, b = %d, c = %d\n", a, b, c);
                printf("x = %.2f, y = %.2f\n", x, y);
                printf("first = %c, last = %c\n", first, last);
                printf("a + b + c = %d\n", a + b + c);
                return 0;
            }""", a=a, b=b, c=c, x=x, y=y, f=f, l=l)
        expl = ("Each line declares several variables of one type, separated by commas, and gives each its own "
                "starting value: three ints, two doubles and two chars. printf() prints them with %d, %.2f and %c, "
                "and the last line prints the sum of a, b and c.")
        return cap, ["stdio.h"], code, expl, ["Declare three ints on one line.", "Declare two doubles and two chars the same way.", "Print each group and the sum of the ints."]
    if kind == "pointers":
        x, y, k = R.randint(1, 50), R.randint(1, 50), R.randint(2, 20)
        cap = pick("Declare two pointers in one statement and use them.",
                   "Show the right way to declare several pointers on one line.")
        code = fill(r"""
            int main(void) {
                int x = ${x}, y = ${y};
                /* Each pointer needs its own '*': in "int *p, q;" only p is a pointer. */
                int *px = &x, *py = &y;
                *px += ${k};
                printf("x = %d, y = %d\n", x, y);
                printf("*px = %d, *py = %d\n", *px, *py);
                return 0;
            }""", x=x, y=y, k=k)
        expl = (f"int *px = &x, *py = &y; declares two pointers in one statement; each name needs its own *. "
                f"*px += {k} changes x through the pointer, so x becomes {x + k} while y stays {y}.")
        return cap, ["stdio.h"], code, expl, ["Declare x and y on one line.", "Declare the pointers px and py on one line, each with its own *.", f"Add {k} to x through px and print the values."]
    n = R.randint(3, 15)
    cap = pick("Declare variables first and give them values later.",
               "Declare several variables on one line without initial values.")
    code = fill(r"""
        int main(void) {
            int count, total, i;   /* declared now, given values below */
            count = ${n};
            total = 0;
            for (i = 1; i <= count; i++) {
                total += i * i;
            }
            printf("Sum of the first %d squares: %d\n", count, total);
            return 0;
        }""", n=n)
    expl = (f"count, total and i are declared together on one line without values, then assigned before they are "
            f"used, because an uninitialised local variable holds garbage. The loop adds the squares of 1 to {n} "
            f"to total.")
    return cap, ["stdio.h"], code, expl, ["Declare count, total and i on one line without values.", f"Set count = {n} and total = 0.", "Add the squares of 1 to count to total and print it."]


TASKS = [t_sum, t_average, t_extreme, t_count_parity, t_linear_search, t_binary_search, t_sort, t_sort,
         t_reverse_array, t_factorial, t_fibonacci, t_gcd_lcm, t_prime, t_power, t_digits, t_leap_year, t_table,
         t_binary, t_string, t_string, t_text_count, t_swap, t_struct, t_linked_list, t_stack_queue, t_matrix,
         t_dynamic_array, t_file_io,
         t_two_numbers, t_two_numbers, t_even_odd, t_biggest, t_formula, t_count_loop, t_hello,
         t_multi_param, t_multi_param, t_out_params, t_out_params, t_struct_return, t_declarations]
HEADER_ORDER = ["stdio.h", "stdlib.h", "string.h", "ctype.h", "math.h", "stdbool.h"]


def rephrase(cap):
    """Ask for the same program in different ways, like people do."""
    if R.random() < 0.5:
        return cap
    body = cap[0].lower() + cap[1:].rstrip(".")
    if R.random() < 0.2:
        return f"How do I {body} in C?"
    return pick("Write a program to ", "Write a C program to ", "C program to ", "Program to ") + body + "."


def comment_block(label, text):
    """'/* Label: text */', wrapped at about 90 characters."""
    lines = textwrap.wrap(f"{label}: {text}", 88)
    return "/* " + "\n   ".join(lines) + " */"


def plan_block(steps):
    """'/* Plan:' + numbered steps (wrapped) + '*/': the model learns to write these steps before the code."""
    lines = ["   " + "\n      ".join(textwrap.wrap(f"{i}. {step}", 84)) for i, step in enumerate(steps, 1)]
    return "/* Plan:\n" + "\n".join(lines) + " */"


def make_program(task=None):
    cap, headers, code, expl, plan = (task or R.choice(TASKS))()
    cap = rephrase(cap)
    includes = "\n".join(f"#include <{h}>" for h in sorted(set(headers), key=HEADER_ORDER.index))
    return (f"/* Caption: {cap} */\n{includes}\n\n{plan_block(plan)}\n{code}\n"
            f"{comment_block('Explanation', expl)}\n{END}\n")


# ----------------------------------------------------------------------------- questions and answers about C
QA = [
    (["What does printf do?", "How do I print text in C?", "What is printf used for?"],
     "printf() writes formatted text to the screen. Placeholders such as %d, %f and %s in the format string are "
     "replaced by the values that follow it."),
    (["What does scanf do?", "How do I read a number from the keyboard?", "How can I get user input?"],
     "scanf() reads formatted input from the keyboard into variables, so you pass their addresses, for example "
     "scanf(\"%d\", &n). It returns how many values it read, which you should check."),
    (["What does %d mean in printf?", "Which printf placeholder should I use for each type?"],
     "%d is the placeholder for an int. Use %ld for long, %lld for long long, %f for double, %c for char, "
     "%s for a string and %u for unsigned int."),
    (["What is the main function?", "Where does a C program start?"],
     "main() is where every C program starts running. Returning 0 from main tells the operating system that the "
     "program finished successfully."),
    (["What does #include do?", "Why do I need #include <stdio.h>?"],
     "#include copies a header file into your program before it is compiled, so you can use the functions it "
     "declares, such as printf from stdio.h."),
    (["What is a variable?", "How do I create a variable?"],
     "A variable is a named piece of memory that holds a value of a fixed type, such as int or double. You must "
     "declare it before you use it, for example int count = 0;."),
    (["Can I declare several variables on one line?", "How do I declare multiple variables at once?"],
     "Yes. Separate them with commas, for example int a = 1, b = 2, c;. They all get the same type, and each one "
     "can have its own starting value."),
    (["Why is only the first variable a pointer in int *p, q;?", "How do I declare two pointers on one line?"],
     "The * belongs to each variable, not to the type. In int *p, q; only p is a pointer and q is a plain int. "
     "Write int *p, *q; to get two pointers."),
    (["What is the difference between int and double?", "When should I use double instead of int?"],
     "int stores whole numbers. double stores numbers with a fractional part, such as 3.75, with about 15 "
     "significant digits."),
    (["What is a char?", "How do I store a single character?"],
     "A char holds one character, such as 'A', stored as a small integer code. Character constants use single "
     "quotes, while strings use double quotes."),
    (["What is a string in C?", "How are strings stored in C?"],
     "A string is an array of char that ends with the null character '\\0'. Functions such as strlen and printf "
     "use that final '\\0' to know where the string stops."),
    (["What is an array?", "How do arrays work in C?"],
     "An array stores several values of the same type next to each other in memory. Elements are numbered from "
     "0, so an array of 5 ints uses the indexes 0 to 4."),
    (["Why do arrays start at index 0?", "Why is the first element arr[0]?"],
     "The index is the distance from the start of the array. The first element is 0 steps from the start, so its "
     "index is 0."),
    (["How do I find the length of an array?", "How many elements does my array have?"],
     "Divide the size of the whole array by the size of one element: sizeof(arr) / sizeof(arr[0]). This only "
     "works where the array is declared, not on a pointer passed to a function."),
    (["What is a function?", "Why should I use functions?"],
     "A function is a named block of code that can take parameters and return a value. Splitting a program into "
     "small functions makes it easier to read, test and reuse."),
    (["How do I pass several parameters to a function?", "Can a function have more than one parameter?"],
     "Yes. List each parameter with its type, separated by commas, for example "
     "double box_volume(double length, double width, double height). The caller passes the values in the same order."),
    (["What does return do?", "How does a function give back a result?"],
     "return ends the function and hands a value back to the caller. A void function returns nothing and can use "
     "return; on its own to stop early."),
    (["What does void mean?", "What is a void function?"],
     "void means nothing. A void function returns no value, and int main(void) means that main takes no parameters."),
    (["How can a function return two values?", "How do I return several results from a function?"],
     "A C function returns only one value. To get more results, pass pointers and write the results through them, "
     "for example min_max(arr, n, &min, &max), or return a struct that holds all of them."),
    (["Why does changing a parameter not change my variable?", "What does pass by value mean?"],
     "C passes arguments by value, so the function gets a copy. To let a function change the caller's variable, "
     "pass its address with & and use a pointer parameter."),
    (["What is a pointer?", "How do pointers work?"],
     "A pointer is a variable that holds a memory address. &x gives the address of x, and *p reads or writes the "
     "value stored at the address in p."),
    (["What does the & operator do?", "Why do I write & in scanf?"],
     "Before a variable, & gives its memory address. That is how scanf, or any function that must change a "
     "variable, knows where to store the result."),
    (["What does * mean in front of a pointer?", "What is dereferencing?"],
     "In an expression, *p dereferences the pointer: it means the value stored at the address in p. In a "
     "declaration, int *p; says that p is a pointer to an int."),
    (["What is NULL?", "Why should I check for NULL?"],
     "NULL is a pointer value that points to nothing. Functions such as malloc and fopen return NULL when they "
     "fail, so check for it before you use the pointer."),
    (["What does malloc do?", "How do I allocate memory at run time?"],
     "malloc() asks the heap for a block of the given number of bytes and returns a pointer to it, or NULL if "
     "there is not enough memory. The new memory is not initialised."),
    (["What does free do?", "When should I call free?"],
     "free() gives memory from malloc back to the system. Call it once for each block when you are done, and do "
     "not use the pointer afterwards."),
    (["What is a memory leak?", "What happens if I forget to call free?"],
     "A memory leak happens when a program allocates memory with malloc but never frees it. The memory stays in "
     "use until the program ends."),
    (["What is the difference between the stack and the heap?", "Where are local variables stored?"],
     "Local variables live on the stack and disappear when their function returns. Memory from malloc lives on "
     "the heap and stays until you free it."),
    (["What is a struct?", "How do I group related data together?"],
     "A struct groups related variables, possibly of different types, under one name, for example a Point with x "
     "and y. You reach a member with a dot, as in p.x."),
    (["What does typedef do?", "Why do people write typedef struct?"],
     "typedef gives a type a new name. typedef struct { double x; double y; } Point; lets you write Point instead "
     "of the whole struct definition."),
    (["What is the difference between . and ->?", "When do I use the arrow operator?"],
     "Use a dot with a struct variable (p.x) and an arrow with a pointer to a struct (ptr->x). ptr->x is short for "
     "(*ptr).x."),
    (["Can a function return a struct?", "How do I return a struct from a function?"],
     "Yes. A function can return a struct by value, for example Point midpoint(Point a, Point b). It is a clean "
     "way to return several related results at once."),
    (["How does a for loop work?", "What are the three parts of a for loop?"],
     "for (start; condition; step) runs start once, then repeats the body while the condition is true, running "
     "step after each round. for (int i = 0; i < n; i++) visits 0 to n - 1."),
    (["What is a while loop?", "When should I use a while loop?"],
     "A while loop repeats its body as long as its condition is true. The condition is checked before each "
     "round, so the body may not run at all."),
    (["What is the difference between while and do-while?", "What is a do-while loop?"],
     "A while loop checks its condition before each round. A do-while loop checks it afterwards, so its body "
     "always runs at least once."),
    (["What do break and continue do?", "How do I leave a loop early?"],
     "break leaves the loop immediately. continue skips the rest of the current round and goes on with the next one."),
    (["How does if-else work?", "How do I make a decision in C?"],
     "if runs a block when its condition is true, and the optional else block runs when it is false. Chain more "
     "conditions with else if."),
    (["When should I use a switch statement?", "How does switch work?"],
     "switch chooses between fixed values of one integer or char, such as menu options. End each case with "
     "break, or execution falls through into the next case."),
    (["What is the difference between = and ==?", "Why does my if statement always run?"],
     "= assigns a value, while == compares two values and gives 1 when they are equal. Writing = inside an if "
     "by mistake is a classic bug."),
    (["What does the % operator do?", "How do I get the remainder of a division?"],
     "% gives the remainder of an integer division: 17 % 5 is 2. It is often used to test for even numbers "
     "(n % 2 == 0) or to wrap around in a ring buffer."),
    (["Why does 7 / 2 give 3 in C?", "What is integer division?"],
     "When both numbers are integers, / drops the fractional part. Write 7 / 2.0, or cast one side to double, to "
     "get 3.5."),
    (["What is type casting?", "How do I convert an int to a double?"],
     "A cast converts a value to another type by writing the type in brackets, for example (double)total / count. "
     "It is often used to avoid integer division."),
    (["What is the ?: operator?", "What is the ternary operator?"],
     "condition ? a : b is a short if-else that produces a value: a when the condition is true and b otherwise, "
     "for example int max = (x > y) ? x : y;."),
    (["What does i++ mean?", "How do I add one to a variable?"],
     "i++ adds one to i. In the same way i-- subtracts one, and i += 5 adds five."),
    (["What does const mean?", "Why write const in a parameter list?"],
     "const marks a variable or parameter as read-only, so the compiler stops code from changing it. A const int "
     "array parameter promises that the function will not modify the array."),
    (["What does #define do?", "How do I make a constant?"],
     "#define creates a constant or macro that the preprocessor replaces before compiling, for example "
     "#define PI 3.14159265. By convention these names are written in capitals."),
    (["What is recursion?", "Can a function call itself?"],
     "Recursion is when a function calls itself on a smaller version of the problem. It needs a base case that "
     "stops the calls, like factorial(1) returning 1."),
    (["What is the difference between local and global variables?", "What is variable scope?"],
     "A local variable exists only inside the block or function where it is declared. A global variable is "
     "declared outside all functions and can be used anywhere in the file."),
    (["How do I use true and false in C?", "Does C have a bool type?"],
     "Include stdbool.h to get the bool type and the values true and false. Without it, C treats 0 as false and "
     "any other number as true."),
    (["What does strlen do?", "How do I get the length of a string?"],
     "strlen() from string.h returns the number of characters in a string, not counting the final '\\0'."),
    (["How do I copy a string?", "What does strcpy do?"],
     "strcpy(dest, src) from string.h copies src into dest, including the final '\\0'. The destination array must "
     "be big enough for the whole string."),
    (["How do I compare two strings?", "Why does == not work for strings?"],
     "Use strcmp(a, b) from string.h: it returns 0 when the strings are equal. == only compares the addresses of "
     "the two arrays, not their characters."),
    (["How do I open a file in C?", "What does fopen do?"],
     "fopen(name, mode) opens a file with mode \"r\" to read, \"w\" to write or \"a\" to append. It returns a FILE "
     "pointer, or NULL on failure, and fclose() closes the file."),
    (["How do I read a line of text?", "What does fgets do?"],
     "fgets(buffer, size, stream) reads one line, at most size - 1 characters, and ends it with '\\0', so it "
     "cannot overflow the buffer."),
    (["How do I compile a C program?", "How do I run my C program?"],
     "With gcc, run gcc program.c -o program to build it and ./program to run it. Add -Wall to get helpful warnings."),
    (["What does a compiler do?", "Why does C need a compiler?"],
     "A compiler translates C source code into machine code that the processor can run. It also reports syntax "
     "errors and type mistakes before the program runs."),
    (["What is a segmentation fault?", "Why does my program crash with a segfault?"],
     "A segmentation fault means the program touched memory it may not use, often through a NULL pointer, an "
     "uninitialised pointer or an array index out of range."),
    (["What happens if I go past the end of an array?", "Does C check array bounds?"],
     "C does not check array bounds. Writing past the end overwrites other memory, which can crash the program or "
     "cause hard-to-find bugs."),
    (["What value does an uninitialised variable have?", "Why do I get random numbers from my variable?"],
     "A local variable that is not initialised holds whatever was in that memory before, so its value is "
     "unpredictable. Always give variables a starting value."),
    (["What is a header file?", "What goes in a .h file?"],
     "A header file declares functions, types and constants so that other files can use them. You include it "
     "with #include."),
    (["How do I write comments in C?", "What kinds of comments does C have?"],
     "A block comment starts with slash-star and ends with star-slash and may span several lines; // starts a "
     "comment that runs to the end of the line. The compiler ignores both."),
    (["What is a 2D array?", "How do I make a table of numbers?"],
     "A 2D array is an array of arrays, like a table. int m[3][4] has 3 rows and 4 columns, and m[r][c] is the "
     "element in row r and column c."),
    (["What is a linked list?", "How is a linked list different from an array?"],
     "A linked list is a chain of nodes; each node holds a value and a pointer to the next node, and the last one "
     "points to NULL. Adding at the front is fast, but finding an element means walking the chain."),
    (["What is a stack data structure?", "What does last in, first out mean?"],
     "A stack is a last in, first out collection: push adds an item on top and pop removes the top item."),
    (["What is a queue?", "What does first in, first out mean?"],
     "A queue is a first in, first out collection: items are added at the back and removed from the front, like "
     "people waiting in line."),
    (["When can I use binary search?", "Why is binary search fast?"],
     "Only on sorted data. Binary search compares with the middle element and throws away half of the range each "
     "step, so it needs about log2(n) comparisons instead of n."),
    (["Which simple sorting algorithms exist?", "How does bubble sort compare to insertion sort?"],
     "Bubble sort swaps neighbours that are out of order, selection sort picks the smallest remaining element for "
     "each position, and insertion sort inserts each element into the sorted part. All three take about n * n steps."),
    (["Why does main return 0?", "What is an exit status?"],
     "The value returned from main is the program's exit status. 0 means success, and any other value tells the "
     "system that something went wrong."),
    (["What does unsigned mean?", "How do I store only positive numbers?"],
     "An unsigned integer cannot be negative, so it can hold larger positive values in the same space. Print an "
     "unsigned int with %u."),
    (["When should I use long long?", "How do I store very big integers?"],
     "Use long long when values can go beyond the range of int, roughly 2 billion, for example factorials or large "
     "powers. Print it with %lld."),
    (["What does putchar do?", "How do I print a single character?"],
     "putchar() prints one character, for example putchar('A') or putchar('\\n')."),
    (["What does toupper do?", "How do I turn a letter into a capital?"],
     "toupper() from ctype.h turns a lowercase letter into its capital and returns every other character unchanged."),
    (["How do I calculate a square root?", "How do I use sqrt in C?"],
     "Include math.h and call sqrt(x), which returns a double. With gcc you may need to add -lm when linking."),
]


def make_qa(per_topic):
    """Each topic per_topic times, with varied wording of the question and the same answer."""
    docs = []
    for questions, answer in QA:
        for _ in range(per_topic):
            q = R.choice(questions)
            if R.random() < 0.3:
                q = "In C, " + q[0].lower() + q[1:]
            docs.append(f"/* Question: {q} */\n{comment_block('Answer', answer)}\n{END}\n")
    return docs


def make_corpus(n_programs, seed=0, qa_per_topic=0):
    """n_programs distinct captioned C programs (duplicates are skipped), then the Q&A documents."""
    R.seed(seed)
    programs, seen, tries = [], set(), 0
    while len(programs) < n_programs and tries < n_programs * 20:
        tries += 1
        p = make_program()
        if p not in seen:
            seen.add(p)
            programs.append(p)
    return programs + make_qa(qa_per_topic)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    out = sys.argv[2] if len(sys.argv) > 2 else "c_caption_dataset.txt"
    qa = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    docs = make_corpus(n, seed=1337, qa_per_topic=qa)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(docs))
    print(f"wrote {n} programs and {len(docs) - n} Q&A documents to {out}")
