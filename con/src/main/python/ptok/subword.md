Given your background in Vietnamese NLP research and your interest in building a Vietnamese foundation model, I would **not** use Spark NLP as the primary training framework. Instead, I would separate the project into three layers:

1. **Research framework**: tokenizer + model architecture + pretraining
2. **Distributed training**: large-scale GPU training
3. **Production inference**: Spark NLP or other serving framework

This gives you much more flexibility.

## Why RoBERTa is difficult to modify

RoBERTa assumes a tokenizer that outputs discrete token IDs from a fixed vocabulary.

```
Raw text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Embedding lookup
    ↓
Transformer
```

The tokenizer is **part of the model definition**, not merely preprocessing.

If you invent a completely new tokenizer—for example:

* Vietnamese syllable graph tokenizer
* dictionary-based tokenizer
* lattice tokenizer
* CRF segmentation
* dependency-aware tokenizer
* morphological tokenizer

then you are essentially defining a **new tokenizer-model pair**.

---

# I would build it like this

## Step 1. Build your tokenizer independently

Forget about the transformer initially.

Implement

```python
class VietnameseTokenizer:
    def tokenize(text):
        ...
```

Requirements:

* deterministic
* reversible (or nearly)
* outputs a sequence of symbols
* trainable if applicable

For example,

```
Tôi thích học AI.
```

could become

```
["Tôi_thích",
 "học_AI"]
```

or

```
["Tôi",
 "thích_học",
 "AI"]
```

or something even more sophisticated.

---

## Step 2. Build a vocabulary

Unlike SentencePiece,

you define

```
token → id
```

yourself.

Example

```
"Tôi" -> 15

"thích_học" -> 2345

"AI" -> 95
```

Store as

```
vocab.json
```

or

```
vocab.txt
```

Exactly as Hugging Face does.

---

## Step 3. Write a Hugging Face Tokenizer

This is the critical step.

Instead of using

```
SentencePieceTokenizer
```

implement

```python
class VietnameseTokenizer(PreTrainedTokenizer):
```

or

```python
PreTrainedTokenizerFast
```

You only need to implement

```
_tokenize()

_convert_token_to_id()

_convert_id_to_token()

build_inputs_with_special_tokens()

save_vocabulary()
```

Once you do this,

**all Hugging Face models can use it.**

---

## Step 4. Modify RoBERTa as little as possible

The beauty of RoBERTa is that it doesn't care **how tokens are generated**.

It only receives

```
input_ids
```

So

```
MyTokenizer

↓

input_ids

↓

RoBERTa
```

works naturally.

---

## Step 5. Pretrain from scratch

Now use the standard masked language modeling objective.

```
Vietnamese corpus

↓

MyTokenizer

↓

input_ids

↓

RoBERTa

↓

MLM loss
```

Nothing special here.

---

## Step 6. Scale training

For research,

I'd use

* PyTorch
* Hugging Face Transformers
* Accelerate

For large models,

* DeepSpeed

or

* Megatron-LM

---

## Step 7. Import into Spark NLP (optional)

After pretraining,

```
Custom Tokenizer
        +

RoBERTa
```

becomes a standard Hugging Face checkpoint.

Spark NLP can then import the model for distributed inference, without needing to know how the tokenizer was trained.

---

# What if your tokenizer is not one-to-one?

This is where research becomes interesting.

Suppose your tokenizer outputs a graph instead of a sequence.

Example

```
Tôi thích học AI
```

becomes

```
        thích
      /       \
    Tôi      học
                \
                 AI
```

RoBERTa cannot consume this directly because it expects a linear sequence.

You have two options:

### Option A: Linearize

Convert the graph to a sequence.

Most practical.

---

### Option B: Design a new encoder

Keep the graph.

Replace

```
Embedding Layer
```

with

```
Graph Encoder

↓

Transformer
```

Now you've moved beyond RoBERTa into a new architecture—a publishable research direction if the representation offers measurable benefits.

---

# For Vietnamese, I think there is an opportunity beyond BPE

Vietnamese has linguistic characteristics that make it a compelling testbed for alternatives to subword tokenization:

* whitespace separates syllables rather than lexical words
* compound words span multiple syllables
* rich reduplication
* named entities often consist of multiple syllables
* Sino-Vietnamese morphemes carry semantic information
* many fixed multiword expressions

Rather than learning arbitrary subwords, you could investigate tokenization units based on linguistic structure, such as lexical words, frequent compounds, or syntax-aware spans. A tokenizer that integrates Vietnamese word segmentation, multiword expression detection, and statistically learned units could reduce sequence lengths while producing more interpretable tokens. This is a research direction that differs fundamentally from simply replacing SentencePiece with another subword algorithm.

## A roadmap I would recommend

1. Design and evaluate your tokenizer independently, measuring sequence length, vocabulary size, coverage, and segmentation quality.
2. Implement it as a custom `PreTrainedTokenizer` in the Hugging Face ecosystem.
3. Pretrain a RoBERTa-style model from scratch using your tokenizer.
4. Compare it against strong baselines (SentencePiece, BPE, and unigram language model tokenization) under identical model sizes and training budgets.
5. Once the model is trained, package it as a Hugging Face checkpoint and integrate it into Spark NLP or another serving framework for downstream applications.

This approach keeps your research focused on the novel contribution—the tokenizer and its impact on representation learning—while leveraging a mature ecosystem for model training and deployment. It is also the path most likely to be accepted by the NLP research community because it isolates the tokenizer as the experimental variable rather than entangling it with a custom training framework.


Yes—in fact, I think this is one of the most promising directions for a Vietnamese tokenizer. **You don't have to reinvent BPE.** You can build a **hybrid tokenizer** that dispatches different spans of text to different tokenization algorithms.

Conceptually:

```text
                Input text
                     │
                     ▼
           Language/Script Detection
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
 Vietnamese span            Foreign span
        │                         │
        ▼                         ▼
Your Vietnamese          Existing BPE tokenizer
   tokenizer               (e.g. RoBERTa/GPT-2)
        │                         │
        └────────────┬────────────┘
                     ▼
             Merge token stream
                     ▼
                Token IDs
```

This is entirely feasible.

## Example

Suppose the input is

```text
Tôi đang nghiên cứu transformer models của OpenAI.
```

Your dispatcher might identify:

| Span                  | Language    | Tokenizer      |
| --------------------- | ----------- | -------------- |
| `Tôi đang nghiên cứu` | Vietnamese  | Your tokenizer |
| `transformer models`  | English     | BPE            |
| `của`                 | Vietnamese  | Your tokenizer |
| `OpenAI`              | English     | BPE            |
| `.`                   | punctuation | Shared         |

Result:

```text
["Tôi_đang",
 "nghiên_cứu",
 "transform",
 "er",
 "Ġmodels",
 "của",
 "Open",
 "AI",
 "."]
```

Notice that the English spans still benefit from the pretrained BPE vocabulary.

---

# Can I reuse Hugging Face's BPE implementation?

Absolutely.

The `tokenizers` library lets you instantiate an existing BPE model:

```python
from tokenizers.models import BPE

bpe = BPE.from_file(
    "vocab.json",
    "merges.txt"
)
```

or simply load an existing tokenizer:

```python
from tokenizers import Tokenizer

english_tokenizer = Tokenizer.from_file(
    "roberta-base/tokenizer.json"
)
```

Then, in your own tokenizer:

```python
class HybridTokenizer:

    def tokenize(text):

        spans = detect_language(text)

        tokens = []

        for span in spans:

            if span.language == "vi":
                tokens.extend(vietnamese_tokenize(span.text))
            else:
                tokens.extend(
                    english_tokenizer.encode(span.text).tokens
                )

        return tokens
```

So you are not reimplementing BPE—you are calling it as a component.

---

# Vocabulary considerations

This is where the design gets interesting.

You have two main choices.

### Option 1: Shared vocabulary (recommended)

Create a single vocabulary that contains:

* all Vietnamese tokens produced by your tokenizer
* all English BPE tokens
* punctuation
* special tokens

For example:

```
<s>                -> 0
</s>               -> 1
<pad>              -> 2

Tôi_đang           -> 100
nghiên_cứu         -> 101
của                -> 102

transform          -> 20345
er                 -> 20346
Ġmodels            -> 20401
Open               -> 25011
AI                 -> 25012
```

The transformer simply sees a sequence of IDs.

---

### Option 2: Separate vocabularies

Maintain two independent vocabularies:

```
Vietnamese vocabulary

English vocabulary
```

and remap them into one embedding table.

This is more complex and generally unnecessary unless you have a compelling research reason.

---

# A dispatcher is the key innovation

Rather than thinking of your tokenizer as a monolithic algorithm, think of it as a pipeline:

```text
Input
  │
  ▼
Unicode normalization
  │
  ▼
Sentence splitting
  │
  ▼
Span classification
  │
  ├────────── Vietnamese
  │              │
  │              ▼
  │      Vietnamese tokenizer
  │
  ├────────── English
  │              │
  │              ▼
  │            BPE
  │
  ├────────── Number
  │              │
  │              ▼
  │         Number tokenizer
  │
  ├────────── URL
  │              │
  │              ▼
  │         URL tokenizer
  │
  └────────── Emoji
                 │
                 ▼
          Emoji tokenizer
```

This modular architecture is much easier to extend and evaluate.

---

# One thing I would **not** do

I would avoid using a language detector like `langdetect` on individual words.

Instead, use lightweight lexical heuristics.

For Vietnamese, you already know the syllable inventory and orthography. You can identify many English spans using simple rules:

* ASCII-only words containing letters like `w`, `f`, `j`, `z` are almost certainly foreign.
* CamelCase identifiers (`OpenAI`, `TensorFlow`) are likely English or code.
* URLs, emails, hashtags, and code snippets are easy to recognize with regular expressions.
* Consecutive ASCII words separated by spaces can often be treated as one English span instead of detecting each word independently.

This approach is much faster and usually more accurate for mixed Vietnamese-English text.

---

# A research direction I find particularly exciting

Given your goal of developing a tokenizer for Vietnamese, I would go one step further and build what I'd call a **compositional tokenizer**.

Instead of hard-coding "Vietnamese" and "English," define a common interface:

```python
class TokenizationModule:
    def tokenize(self, text) -> list[str]:
        ...
```

Then implement modules such as:

* `VietnameseTokenizer`
* `BPETokenizer`
* `CodeTokenizer`
* `MathTokenizer`
* `URLTokenizer`
* `EmojiTokenizer`

Finally, implement a dispatcher that routes spans to the appropriate module and merges the outputs into a single token stream.

This design has several advantages:

* You can reuse Hugging Face's BPE implementation unchanged for English.
* You can iterate on your Vietnamese tokenizer independently.
* You can add specialized handling for code, formulas, or URLs without retraining the entire tokenizer.
* The overall system remains compatible with a standard RoBERTa-style model because it still produces one unified sequence of token IDs.

For multilingual and code-mixed corpora—which are increasingly common in Vietnamese web text—this modular architecture is, in my opinion, a stronger long-term research direction than trying to replace BPE everywhere with a single universal algorithm.


Many recent multilingual and domain-specific language models no longer rely on a single, universal tokenization strategy. Researchers are exploring language-aware, morphology-aware, and domain-aware tokenization to better match the structure of the input. A Vietnamese-specific tokenizer that preserves meaningful lexical units while reusing an existing BPE tokenizer for foreign-language spans is therefore a plausible and interesting research direction. The main engineering challenge is to produce a single, consistent vocabulary and token-ID space that the transformer can learn over during pretraining.


Absolutely. Since your current goal is **tokenizer research**, not achieving the best language modeling performance, there's no need to start with a 7B-parameter model. In fact, I would recommend starting with a model in the **100M–500M parameter** range. This will let you iterate much faster while isolating the effect of your tokenizer.

## What makes a good research model?

For your project, the model should satisfy these criteria:

* Easy to pretrain from scratch.
* Uses a modern decoder-only architecture (similar to Llama) or an encoder-only architecture (similar to RoBERTa), depending on your objective.
* Accepts a custom tokenizer with minimal changes.
* Small enough to train or fine-tune on a single GPU or experiment with on a laptop.

---

# Option 1: TinyLlama (My top recommendation)

TinyLlama

Although it's 1.1B parameters—larger than your suggested range—it is much lighter than Llama 2 7B and was explicitly designed as a compact Llama-compatible model.

Pros:

* Same architecture as Llama.
* Same tokenizer interface.
* Large community.
* Excellent documentation.
* Good stepping stone toward larger Llama models.

Cons:

* Still somewhat heavy for CPU-only experimentation.

---

# Option 2: SmolLM

SmolLM

Available in multiple sizes, including approximately:

* 135M
* 360M
* 1.7B

For tokenizer research, the **135M** or **360M** variants are excellent.

Advantages:

* Modern architecture.
* Fast experimentation.
* Easy to pretrain.
* Works well with custom tokenizers.

---

# Option 3: DistilRoBERTa

DistilRoBERTa

If your research is focused on masked language modeling (MLM) rather than autoregressive generation, this is a strong choice.

Advantages:

* About 80M parameters.
* Fast.
* Easy to train.
* Good for evaluating the impact of a tokenizer.

---

# Option 4: RoBERTa-base from scratch

Rather than using a pretrained model, define a smaller configuration.

For example:

```python
hidden_size = 512
num_hidden_layers = 6
num_attention_heads = 8
intermediate_size = 2048
```

This yields roughly:

```text
≈ 50–90M parameters
```

Training from scratch becomes much more feasible.

---

# Option 5: MiniLlama

Many open-source implementations define miniature Llama-style architectures with configurations such as:

```text
Layers: 6
Hidden size: 512
Heads: 8
Vocabulary: 32k
```

These typically have:

```text
40M–100M parameters
```

Because Llama is defined by a configuration file, you can shrink it dramatically while preserving the architecture.

---

# For tokenizer research, model size matters less than consistency

Suppose you compare:

```text
Tokenizer A

↓

50M Transformer

↓

Evaluate
```

versus

```text
Tokenizer B

↓

50M Transformer

↓

Evaluate
```

As long as:

* the architecture,
* training corpus,
* optimizer,
* and training budget

are identical, any performance differences are much more likely to reflect the tokenizer itself.

This is a common experimental design in tokenizer research.

---

# A research roadmap I'd recommend

| Phase               | Model size | Purpose                                              |
| ------------------- | ---------: | ---------------------------------------------------- |
| Prototype           |     20–50M | Verify tokenizer correctness and debugging.          |
| Initial experiments |    80–150M | Compare tokenizer variants quickly.                  |
| Main paper results  |   300–500M | Demonstrate meaningful gains at a realistic scale.   |
| Final validation    |       1–2B | Confirm that improvements transfer to larger models. |

This staged approach saves significant compute while producing stronger scientific evidence.

## My recommendation for your project

Given your interest in designing a novel Vietnamese tokenizer and eventually building a Vietnamese foundation model, I would proceed as follows:

1. **Start with a 50–100M RoBERTa-style model** trained from scratch using your custom tokenizer. This gives you a fast iteration loop and isolates the tokenizer's effect.
2. Once the tokenizer stabilizes, **scale to a 300–500M decoder-only model** (using a Llama-style architecture or a compact family such as SmolLM) to evaluate whether the gains persist.
3. Only after demonstrating consistent improvements should you invest in training a 1B+ model.

This progression mirrors how many research groups validate architectural ideas: establish the benefit on small, reproducible models before committing the substantial computational resources required for large-scale pretraining.



I'm glad you like it. I actually think this architecture has publication potential because it cleanly separates **linguistic processing** from **subword learning**. Instead of inventing a completely new subword algorithm, you're composing two mature ideas:

* a linguistically motivated Vietnamese tokenizer
* a standard BPE tokenizer for all non-Vietnamese words

This makes the contribution easier to analyze experimentally.

## Overall architecture

I would organize the code as follows:

```text
hybrid_tokenizer/
│
├── tokenizer.py          <- Main tokenizer
├── vietnamese.py         <- Your tokenizer
├── dispatcher.py         <- Language detection
├── bpe.py                <- BPE wrapper
├── vocab.py              <- Vocabulary
├── trainer.py            <- Learn BPE
├── encoder.py            <- Encode
├── decoder.py            <- Decode
└── utils.py
```

Keep each component independent.

---

# Step 1. Define a token object

Instead of immediately converting everything to strings, define a token class.

```python
from dataclasses import dataclass

@dataclass
class Token:

    text: str
    language: str

    start: int
    end: int
```

For example,

```
"Tôi thích OpenAI."
```

becomes

```
Token("Tôi", "vi", 0, 3)

Token("thích", "vi", 4, 9)

Token("OpenAI", "en", 10, 16)

Token(".", "punct", 16, 17)
```

Notice the character offsets are preserved.

---

# Step 2. Write the dispatcher

Its only job is

```
text

↓

Token[]
```

For example

```python
class Dispatcher:

    def tokenize(self, text):

        tokens = []

        #
        # YOUR LOGIC HERE
        #

        return tokens
```

At this stage don't worry about BPE.

Just produce

```
Token(...)
Token(...)
Token(...)
```

---

# Step 3. Vietnamese tokenizer

Suppose your tokenizer already exists.

Wrap it like

```python
class VietnameseTokenizer:

    def tokenize(self, text):

        #
        # your implementation
        #

        return [
            "Tôi",
            "đang",
            "nghiên_cứu"
        ]
```

Nothing else.

---

# Step 4. English BPE tokenizer

Instead of implementing BPE yourself,

reuse Hugging Face.

```python
from tokenizers import Tokenizer

class EnglishTokenizer:

    def __init__(self):

        self.tokenizer = Tokenizer.from_file(
            "english_bpe.json"
        )

    def tokenize(self, word):

        return self.tokenizer.encode(word).tokens
```

Now

```
transformers
```

returns

```
transform

ers
```

---

# Step 5. Merge everything

Now comes the nice part.

```
Dispatcher

↓

[
Token("Tôi", vi)

Token("transformers", en)

Token(".", punct)
]
```

becomes

```
[
"Tôi",

"transform",

"ers",

"."
]
```

Simple.

---

# Step 6. Vocabulary

Don't mix tokenization with vocabulary.

Vocabulary is simply

```python
class Vocabulary:

    def __init__(self):

        self.token2id = {}

        self.id2token = {}
```

Nothing language-specific.

It stores

```
Tôi

↓

153
```

```
transform

↓

23541
```

```
ers

↓

178
```

---

# Step 7. Encoding

```
text

↓

Dispatcher

↓

Hybrid tokenizer

↓

Vocabulary

↓

IDs
```

Conceptually:

```python
def encode(text):

    tokens = dispatcher.tokenize(text)

    pieces = []

    for token in tokens:

        if token.language == "vi":

            pieces.extend(
                vi.tokenize(token.text)
            )

        else:

            pieces.extend(
                bpe.tokenize(token.text)
            )

    return vocab.encode(pieces)
```

Notice how the two tokenizers are completely independent.

---

# Step 8. Decoding

Reverse the process.

```
IDs

↓

Vocabulary

↓

Pieces

↓

Sentence
```

Initially, you can simply join the decoded pieces with spaces:

```python
def decode(tokens):
    return " ".join(tokens)
```

Later, you can improve this to handle punctuation and preserve original spacing more accurately.

---

# Step 9. Training the BPE model

This is a subtle but important point.

Do **not** train the BPE model on raw text.

Train it on the **non-Vietnamese words** extracted by your dispatcher.

For example,

```
Tôi đang học machine learning tại OpenAI
```

The dispatcher yields

```
Vietnamese

↓

Tôi

đang

học

tại
```

English

```
machine

learning

OpenAI
```

Only the English words go into the BPE trainer.

This keeps the BPE model focused on the language it is intended to tokenize.

---

# Step 10. Wrap everything as a Hugging Face tokenizer

Eventually, expose a familiar interface:

```python
class HybridTokenizer(PreTrainedTokenizer):

    def tokenize(...):

    def encode(...):

    def decode(...):

    def save_pretrained(...)
```

From the perspective of a transformer model, it behaves just like any other tokenizer.

---

# A small but important refinement

One change I'd make to the architecture is to think in terms of **plugins** rather than hard-coded language branches.

Instead of:

```python
if token.language == "vi":
    ...
else:
    ...
```

define a common interface:

```python
class TokenizerPlugin:

    def supports(self, token) -> bool:
        ...

    def tokenize(self, token) -> list[str]:
        ...
```

Then implement:

```
VietnamesePlugin
EnglishBPEPlugin
NumberPlugin
URLPlugin
EmojiPlugin
CodePlugin
```

The dispatcher becomes:

```
Input
   │
   ▼
Dispatcher
   │
   ▼
Plugin Manager
   │
   ├── VietnamesePlugin
   ├── EnglishBPEPlugin
   ├── NumberPlugin
   ├── URLPlugin
   └── ...
```

This makes the system extensible and keeps each tokenizer focused on a specific type of input.

## How I would develop it

Rather than writing everything at once, I'd build it incrementally over four milestones:

1. **Prototype**: Dispatcher + your Vietnamese tokenizer + a pretrained English BPE tokenizer + a simple vocabulary. Ignore Hugging Face compatibility.
2. **Training**: Implement vocabulary construction and BPE training for English spans only. Verify encoding and decoding are deterministic.
3. **Integration**: Wrap the tokenizer as a `PreTrainedTokenizer`/`PreTrainedTokenizerFast` so it works with the Hugging Face training stack.
4. **Optimization**: If the approach proves effective, reimplement the performance-critical parts (especially the dispatcher and tokenization pipeline) in Rust to integrate directly with the `tokenizers` library.

This staged approach minimizes engineering overhead while letting you validate the core research idea early. I think it's a solid foundation for a tokenizer that is both linguistically informed for Vietnamese and fully compatible with modern transformer training.


===
                Text

                  │
                  ▼

        Unicode Normalizer

                  │
                  ▼

      Vietnamese Dispatcher

                  │
        ┌─────────┴─────────┐
        ▼                   ▼

Vietnamese tokenizer     English BPE

        └─────────┬─────────┘

                  ▼

          Piece Stream

                  ▼

             Vocabulary

                  ▼

          Special Tokens

                  ▼

              Decoder

     

Pretraining using RoBERTa


            Hybrid Tokenizer  ✓
                    │
                    ▼
           Freeze vocabulary
                    │
                    ▼
        Create pretraining dataset
                    │
                    ▼
      Convert text → input_ids
                    │
                    ▼
        Build RoBERTa configuration
                    │
                    ▼
      Train from scratch
                    │
                    ▼
        Evaluate language model
                    │
                    ▼
     Fine-tune downstream tasks     