# AI_Trends_RAG
A comprehensive RAG project for AI Trends implementing and comparing multiple retrieval strategies — dense, sparse, graph-based, and hybrid. It contains custom evaluation functions and integrated advanced techniques like reranking to optimize model accuracy and relevance.

The notebook is in tutorial format, you can follow.

## Features:

- **Dense, Sparse, Graph, and Hybrid Retrievals:** Compare multiple retrieval paradigms within a unified framework.

- **Graph Generation:** Automatically constructs a knowledge graph from data using entity and triples to model semantic and relational structures.

- **BERTopic** for question generation related to top 50 topics in the dataset.

- **Classification of generated questions** into Conceptual, Comparative or Analytical using facebook/bart-large-mnli model.

- Use an **API** to fetch AI trends graph.

- Different **chunking strategies** such as recursive and semantic chunking.

- **Reranking Mechanism:** Improves the final response faithfulness in hybrid retriever using reranking.

- **Custom Evaluation Functions:** Measure retrieval quality and model performance using cosine similarity and a bag-of-words overlap with parallel processing.

- **Modular Design:** Each component can be tested, extended, or replaced independently.

- A **literature review** for recent research in RAG systems.


## Tech Stack

**Language:** Python

**Frameworks:** PyTorch, LangChain

**Libraries:** HuggingFace Transformers, ChromaDB, NetworkX, Scikit-learn

**LLM Model:** google-flan-t5-large

**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2


## Evaluation

| **Retriever** | **Faithfulness** | **Answer Relevance** | **Context Precision** | **Context Recall** |
|:---------------|:----------------:|:--------------------:|:---------------------:|:------------------:|
| **Hybrid Rerank** | **0.48701** | 0.263702 | 0.025667 | **0.103332** |
| Dense Only     | 0.00000 | 0.344296 | 0.031895 | 0.013853 |
| Graph API      | 0.00000 | **0.387426** | **0.052338** | 0.006304 |
| Graph NER      | 0.00000 | 0.282118 | 0.029462 | 0.013188 |
| Hybrid         | 0.00000 | 0.321491 | 0.031340 | 0.013853 |



## Architecture Diagram

  
<img width="120" height="480" alt="rag drawio (1)" src="https://github.com/user-attachments/assets/f3686f17-c6fe-4702-a2c7-7d93b434fabd" />



## Future Work

- Integrate graph embeddings for richer node representations

- Add multimodal retrieval (text + image)

- Evaluate with a larger human generated Q&A dataset

- Deploy a web-based demo for interactive testing

- Experiment with retrieval compression for scalability

