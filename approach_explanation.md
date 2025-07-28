# Persona-Driven Document Intelligence: Our Approach

Our system is designed to function as an intelligent document analyst, carefully going through a collection of documents to find the exact information a specific user needs to accomplish their task. The core of our methodology is a multi-stage funnel that tries to walk the fine line between efficiency and deep semantic understanding/accuracy, ensuring we deliver highly relevant, quality-checked results within the strict performance constraints (such as CPU only execution and a sub-60-second processing time).

---

## The Analysis Funnel: From Broad to Specific

Our approach progressively refines the search for relevant content, starting broad and narrowing down to the most granular, useful insights.

### Stage 1: High-Speed Keyword Filtering with BM25

First, we parse the input PDFs, breaking them down into the logical sections based on their headings according to the code from Section 1-A. Given that we could be dealing with thousands of potential sections, running complex models on all of them would be inefficient.

To solve this, we employ **BM25**, a powerful and highly efficient keyword-based scoring algorithm. We create a search query by extracting key terms from the provided `persona` and `job-to-be-done`. BM25 then rapidly scores all sections against this query, allowing us to quickly discard a vast majority of irrelevant content. This  initial pass acts as a crucial optimization, ensuring that only the most promising candidates proceed to the next, more computationally intensive stage.

### Stage 2: Deep Semantic Understanding with Transformers

The sections that pass the BM25 filter are then filtered according to a deeper semantic analysis. For this, we use a lightweight `sentence-transformer` model (`all-MiniLM-L6-v2`). This model was chosen specifically for its excellent performance-to-size ratio, which allows us to meet the <1GB model size constraint without sacrificing the ability to understand the nuances and context of the language.

Unlike keyword matching, this model comprehends the *true meaning* behind the text. To handle long sections effectively, we developed a **weighted embedding strategy**. The strategy is as follows:
```
A section is broken into smaller chunks, and the final embedding is a weighted average, giving more importance to chunks that are semantically closer to the section's title. This focuses the analysis on the most pertinent parts of the text.
```

### Stage 3: Quality-Aware Reranking and Subsection Extraction

The final ranking is more than just a combination of keyword and semantic scores. We introduce two final factors:

1.  **Named Entity Density:** Using `spaCy`, we measure the density of named entities (like people, organizations, or locations). A higher density often indicates more fact-rich content, which is typically more relevant and information dense(as a result, more likely to be selected).
2.  **Content Quality Score:** This score penalizes sections that are too short, lack sentence structure, or are simply a title repeated in the body. This critical step ensures that the final output is not just relevant, but also actually informative.


The final section score is a weighted combination of the BM25 score, semantic score, and entity density, all modulated by the content quality score. We then select top-k (currently set as 5) sections and return them as output. Here, we set a constraint of a maximum of 2 sections from a document to ensure document diversity in the final output.


Finally, to provide the most granular insights as required, we perform a **subsection analysis**. We take the top-ranked documents, create sliding window chunks of sentences, and re-run a lightweight version of our ranking pipeline on them. This extracts the most relevant 3-4 sentence snippets, delivering precise, actionable information to the user. This multi-layered process guarantees a final output that is not only accurate and relevant to the user's persona but also diverse and of the highest quality.