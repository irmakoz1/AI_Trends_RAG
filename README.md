# AI_Trends_RAG
A comprehensive RAG project for AI Trends implementing and comparing multiple retrieval strategies — dense, sparse, graph-based, and hybrid. It contains custom evaluation functions and integrated advanced techniques like reranking to optimize model accuracy and relevance.

The notebook is in tutorial format, you can follow.

<img width="120" height="480" alt="rag drawio (1)" src="https://github.com/user-attachments/assets/f3686f17-c6fe-4702-a2c7-7d93b434fabd" />


## Features

| Feature                                   | Description                                                                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Multiple Retrieval Paradigms**          | Supports dense, sparse, graph-based, and hybrid retrievals within a single unified framework.                                        |
| **Automated Graph Generation**            | Builds knowledge graphs from data using entities and triples with semantic and relational modeling, visualized via **Pyvis**.      |
| **Topic Modeling with BERTopic**          | Identifies the top 50 most relevant topics in the dataset.                                                                          |
| **Q&A Generation**                        | Generates questions from documents related to the most relevant topics using LLMs, supporting comparison across models.              |
| **Question Classification**               | Categorizes generated questions into **Conceptual, Comparative, or Analytical** using the `facebook/bart-large-mnli` model.          |
| **AI Trends API**                         | Fetches knowledge graphs of AI trends for enriched data insights.                                                                    |
| **Flexible Chunking Strategies**          | Supports recursive and semantic chunking for efficient document processing.                                                          |
| **Embedding & Vector Store Construction** | Batch processing with **LangChain**, **Sentence Transformers**, and **ChromaDB** for fast retrieval and evaluation.                  |
| **Reranking Mechanism**                   | Enhances response faithfulness in hybrid retrievals using **CrossEncoder** from Sentence Transformers.                               |
| **Custom Evaluation Metrics**             | Measures retrieval quality and model performance using cosine similarity and bag-of-words overlap, with parallel processing support. |
| **Modular, Object-Oriented Design**       | Components are abstracted and modular, making them easy to test, extend, or replace independently.                                   |
| **Literature Review**                     | Includes a review of recent research on Retrieval-Augmented Generation (RAG) systems.                                                |


## Tech Stack


| Component           | Details                                                                           |
| ------------------- | --------------------------------------------------------------------------------- |
| **Language**        | Python                                                                            |
| **Frameworks**      | PyTorch, LangChain                                                                |
| **Libraries**       | HuggingFace Sentence Transformers, NetworkX, Scikit-learn, nltk, BERTopic, pandas |
| **Vector Store**    | ChromaDB                                                                          |
| **LLM Model**       | google-flan-t5-large                                                              |
| **Reranking Model** | ms-marco-MiniLM-L-6-v2                                                            |
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2                                            |



## Project Structure

RAG-Graph-Implementation/


├── 📘 README.md

│    Project overview, usage instructions, results, and documentation.


├── 📓 RAG.ipynb

│    Main Jupyter notebook implementing and comparing RAG methods (Dense, Sparse, Graph-based, Hybrid) with evaluation pipeline.

├── 🗂️ files/

    ├──  📓ai_semantic_graph.json

    │    Generated knowledge graph capturing entity and relationship links from retrieved documents and answers.

    ├──  📓config.json

    │    Configuration file for BERT-based topic modeling (bertopic).

    ├── 📓topics.json

    │    Output from bertopic, storing discovered topics, keywords, and representative documents.

    ├── 📓qa_results.csv

    │    Generated answers and evaluation results for each retriever type (faithfulness, answer relevance, context precision/recall).

    └── (Large model / vector files not pushed due to Git size limits)

    │    Example: ChromaDB index, sentence embeddings.



## Evaluation

| **Retriever** | **Faithfulness** | **Answer Relevance** | **Context Precision** | **Context Recall** |
|:---------------|:----------------:|:--------------------:|:---------------------:|:------------------:|
| **Hybrid Rerank** | **0.48701** | 0.263702 | 0.025667 | **0.103332** |
| Dense Only     | 0.00000 | 0.344296 | 0.031895 | 0.013853 |
| Graph API      | 0.00000 | **0.387426** | **0.052338** | 0.006304 |
| Graph NER      | 0.00000 | 0.282118 | 0.029462 | 0.013188 |
| Hybrid         | 0.00000 | 0.321491 | 0.031340 | 0.013853 |




## Future Work

- Integrate graph embeddings for richer node representations

- Add multimodal retrieval (text + image)

- Evaluate with a larger human generated Q&A dataset

- Deploy a web-based demo for interactive testing

- Experiment with retrieval compression for scalability

