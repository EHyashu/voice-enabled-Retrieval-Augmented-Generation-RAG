# Chunking Strategies Rationale and Analysis

To design an optimal Retrieval-Augmented Generation (RAG) system for low-latency voice queries, we implement and compare four sophisticated chunking strategies. Each strategy offers distinct advantages for search relevancy and response generation.

---

## 1. Semantic Chunking

### Description
Semantic chunking divides documents based on semantic shifts or punctuation marks representing complete thoughts. Instead of arbitrary length constraints, it splits text at sentence boundaries.

### Implementation
We use NLTK's sentence boundary detection, enhanced with regex rules to handle Indic-specific sentence boundaries:
- **Hindi punctuation**: The Purna Viram (`।`) and double Purna Viram (`॥`).
- **English punctuation**: Periods (`.`), question marks (`?`), and exclamation points (`!`).

### Rationale
- **High Coherence**: Ensures that each chunk constitutes a grammatically complete and logically coherent statement.
- **Low Noise**: Prevents sentences from being clipped mid-word or mid-thought, which can disrupt embeddings and decrease matching precision.

---

## 2. Hierarchical / Recursive Chunking

### Description
Recursive chunking splits text using a sequence of separators (typically `\n\n`, `\n`, `" "`, and `""`) in a top-down hierarchy. It strives to keep chunks close to a target size while maintaining a predefined overlap.

### Implementation
- **Chunk Size**: ~400 words (or character equivalents).
- **Chunk Overlap**: 50% overlap (~200 words).
- Sliding window mechanism that steps forward by half the size of the generated chunk.

### Rationale
- **Context Preservation**: The 50% overlap ensures that any information split across two chunks is represented fully within the neighborhood of at least one chunk.
- **Dynamic Adaptability**: Adapts layout splits (paragraphs vs. sentences) dynamically, matching the structural layout of the source text.

---

## 3. Token-aware Chunking

### Description
Token-aware chunking counts individual text tokens (using sub-word byte-pair encoding or a tokenizer) and splits the text when a strict token limit is reached.

### Implementation
- **Encoding**: OpenAI's `cl100k_base` (via `tiktoken`) or GPT-2 byte-pair encoding.
- **Chunk Limit**: Exactly 150 tokens.

### Rationale
- **Deterministic Resource Constraints**: LLM context windows and vector embedding model input constraints are defined in tokens. Token-aware chunking guarantees that no chunk will exceed model input limits.
- **Uniformity**: Creates uniform chunks, helping control downstream latency in both embedding calculation and LLM context loading.

---

## 4. Context-aware Chunking

### Description
Context-aware chunking infuses metadata or structural context directly into the text body of the chunk. In MSMARCO (a Q&A dataset), this means associating a passage chunk with its corresponding query.

### Implementation
We split the passage recursively, and then format the text as:
`Query Context: {query_text}\nPassage Detail [{index}]: {passage_chunk}`

### Rationale
- **Asymmetric Retrieval Alignment**: Voice queries are questions. Adding the original question context directly to the passage text creates a strong semantic link between similar questions and the passage content.
- **LLM Grounding**: When the LLM reads retrieved chunks, it immediately sees the question context, helping it output highly focused answers without losing track of the user's original query.
