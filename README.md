With the development of Natural Language Processing (NLP) technology, Retrieval-Augmented Generation (RAG) models have achieved remarkable progress in many fields, especially in tasks such as question answering systems, text generation, and information retrieval. RAG models have gained widespread attention due to their powerful performance and flexibility. By incorporating external documents, RAG models enhance the generated content, improving accuracy and relevance. However, effectively splitting documents and combining appropriate retrieval strategies remains a key factor in enhancing the performance of RAG models. This blog will explore different RAG document splitting strategies and analyze their applications, effects, and pros and cons.

What is RAG (Retrieval-Augmented Generation)?
The core idea of the RAG model is to retrieve relevant documents from an external knowledge base and combine them with generative models (such as GPT, BERT, etc.) to generate high-quality text. Unlike traditional generative models, RAG models enhance the generated content's accuracy and richness by retrieving external documents, making the generated text more aligned with the query's intent. The primary advantage of RAG is its ability to dynamically integrate external knowledge, avoiding the "knowledge blind spots" of generative models, thereby improving the quality and relevance of the generated text.

The Importance of Document Splitting Strategies in RAG
The core of the RAG model depends on retrieving and generating content from external documents. However, how to effectively split a long document into smaller chunks for better retrieval and generation directly affects the performance of the RAG model. Document splitting should not only maintain semantic consistency but also ensure that each chunk independently provides valuable information. To achieve this, researchers have proposed various splitting strategies, each with its unique advantages, disadvantages, and application scenarios.
1. Simple RAG
Strategy Introduction: Treats the original document as a whole, directly extracting key information from the document for generation.

Example:
If the user asks, “What are the basic concepts of machine learning?” the RAG model takes the entire machine learning introductory document as input, directly extracting key information such as definitions and algorithms to generate the answer.

Advantages:

Simple and fast, suitable for short documents or scenarios with relatively simple information.

Disadvantages:

For long, complex documents, it may overlook details or context, leading to generic content generation.
2. Semantic Chunking
Strategy Introduction: Splits the document into semantically related chunks, where each chunk can be processed independently to generate responses.

Example:
If the user asks, “What are the applications of deep learning?” By semantic chunking, the model splits the document into chunks, each corresponding to a deep learning application area, such as “image processing,” “natural language processing,” etc. The model only extracts relevant content from the “image processing” chunk to generate the answer.

Advantages:

Improves the relevance and accuracy of the generated content, particularly for complex documents.

Disadvantages:

When splitting the document, semantic integrity needs to be considered, or critical information may be lost.

3. Context Enriched Retrieval
Strategy Introduction: Enhances the document chunks with contextual information so that each chunk can better understand and match related content.

Example:
When querying “What are the challenges in speech recognition technology?” The RAG model not only retrieves relevant content from documents about speech recognition but also enriches the retrieval chunks with contextual information based on previous user queries (e.g., the user previously queried “natural language processing”), making the retrieval results more targeted.

Advantages:

Enhances the contextual understanding of document chunks, improving the accuracy of retrieval and generation.

Disadvantages:

Increases computational complexity and resource consumption, as more context information must be processed, potentially affecting performance.

4. Contextual Chunk Headers
Strategy Introduction: Adds contextual headers to each document chunk to help the retrieval model better understand the theme and content of each chunk.

Example:
If the user asks, “What are the effects of greenhouse gases?” the document is split into multiple chunks, each with a header like “Types of Greenhouse Gases,” “Effects of Greenhouse Gases,” etc. The RAG model identifies the header “Effects of Greenhouse Gases” and directly retrieves the relevant chunk to generate the answer.

Advantages:

Improves the efficiency of document chunk identification, reducing irrelevant content interference.

Disadvantages:

Requires structural processing of the document, and header generation can be complex.

5. Document Augmentation
Strategy Introduction: Enhances the document content to increase the diversity and richness of the information, improving the quality of the chunks.

Example:
If the user asks, “What is the latest research on cancer immunotherapy?” The RAG model not only extracts information from existing documents but also integrates external materials such as research data and reports from related fields to provide a more comprehensive answer.

Advantages:

Provides more information richness, enhancing the generative model’s expressiveness.

Disadvantages:

Increases computational load, requiring additional data sources and processing steps.

6. Query Transformation
Strategy Introduction: Transforms the query to better match the content in document chunks.

Example:
If the user asks, “What are the main algorithms of deep learning?” the query is transformed into a standardized query, such as “Core components of deep learning algorithms” for retrieval, and the relevant chunk is extracted from the document.

Advantages:

Makes the query more closely aligned with the document chunks, improving retrieval performance.

Disadvantages:

Requires designing an effective query transformation mechanism, increasing complexity.

7. Reranker
Strategy Introduction: Reorders the retrieval results to select the most relevant document chunks for generation.

Example:
If the user asks, “What are the latest advancements in cancer treatment?” the RAG model retrieves multiple document chunks about immunotherapy, targeted therapy, etc. By reordering, the model prioritizes chunks containing the latest cancer treatments (e.g., immunotherapy) to ensure the generated answer best matches the user's needs.

Advantages:

Improves the accuracy and relevance of the generated content by ensuring the most relevant information is presented first.

Disadvantages:

Requires additional sorting processes, potentially increasing computational cost and processing time.

8. RSE (Re-ranking with Semantic Expansion)
Strategy Introduction: Reorders the retrieval results based on semantic expansion, ensuring the retrieved document chunks are more diverse and relevant.

Example:
When querying “Latest advancements in cancer drugs,” semantic expansion automatically includes related methods such as “immunotherapy,” “targeted therapy,” and related treatments, ensuring the returned document chunks are more diverse and relevant, providing a more comprehensive answer.

Advantages:

Enhances the diversity and relevance of retrieval results, especially for diverse queries.

Disadvantages:

Requires more computational resources and semantic expansion models.

9. Adaptive RAG
Strategy Introduction: Dynamically adjusts the RAG model’s behavior based on different queries and document content to improve generation quality.

Example:
When the user queries “Latest technological advancements in deep learning,” the model dynamically adjusts its retrieval and generation strategy, such as prioritizing information related to “Generative Adversarial Networks (GANs)” or “Reinforcement Learning” to meet the specific user needs.

Advantages:

Enhances model adaptability, dynamically adjusting based on different queries.

Disadvantages:

Algorithm design is complex, adding difficulty to adjustments and optimization.

10. Self RAG
Strategy Introduction: Optimizes the RAG model through a self-enhancement mechanism, improving performance through feedback loops within the model itself.

Example:
During the generation process, the model optimizes itself based on previous responses and user feedback. If a user points out an error, the model corrects itself based on the error and generates more accurate answers.

Advantages:

Enhances generation quality and robustness through self-optimization.

Disadvantages:

More complex to implement, requiring additional training and feedback mechanisms.

11. Knowledge Graph
Strategy Introduction: Uses knowledge graphs to enhance documents, providing more precise and structured information.

Example:
When processing a query about “artificial intelligence,” the RAG model uses a knowledge graph to connect relationships between different technologies (e.g., machine learning, deep learning, natural language processing) and generate a more informative response.

Advantages:

Provides structured background information, improving the quality of the generated content.

Disadvantages:

Requires building and maintaining a knowledge graph, adding system complexity.

12. Hierarchical Indices
Strategy Introduction: Organizes documents using hierarchical indices to make the retrieval process more efficient.

Example:
When retrieving documents with multiple sections, the model uses hierarchical indices to divide the document into different parts, such as “Technical Section,” “Application Section,” “Conclusion Section,” improving retrieval and generation efficiency.

Advantages:

Improves retrieval efficiency, especially for long documents.

Disadvantages:

Requires complex indexing mechanisms, increasing implementation difficulty.

13. HyDE (Hypothetical Document Embedding)
Strategy Introduction: Uses hypothetical document embeddings to map documents to higher-dimensional spaces, improving generation quality.

Example:
If the user asks, “What is the latest research in autonomous driving technology?” the model maps the relevant documents into higher-dimensional space, enhancing its understanding of various aspects (e.g., sensors, algorithms) and generating a more comprehensive answer.

Advantages:

Enhances document representation, improving generation quality.

Disadvantages:

Increases computational resource demands, requiring additional embedding processes.

14. Fusion
Strategy Introduction: Fuses multiple documents or outputs from various models to enhance generation quality.

Example:
If the user asks, “What are the use cases of IoT,” the model fuses information from multiple documents (e.g., “smart home applications,” “industrial automation applications”) to generate a comprehensive answer covering different perspectives.

Advantages:

Improves diversity and accuracy of generated content, especially for complex queries.

Disadvantages:

Requires coordination and fusion of multiple data sources, increasing system complexity.

15. CRAG (Corrective RAG)
Strategy Introduction: Enhances RAG performance by correcting errors in generated content, improving the quality of the generated text.

Example:
If the user queries, “How do intelligent robots change the manufacturing industry?” the initial generated answer may have inaccuracies. The model corrects these errors through a correction mechanism, producing a more accurate response.

Advantages:

Improves accuracy of generated content, reducing errors.

Disadvantages:

Increases processing time and depends on additional error correction steps.

16. Feedback Loop
Strategy Introduction: Uses the model’s output and user feedback to continuously adjust the generation strategy, making the generated content more accurate.

Example:
If the user queries, “What is the application of artificial intelligence in healthcare?” and provides feedback like “focus more on cancer detection technology,” the model adjusts its response based on the feedback, generating content that better aligns with the user's expectations.

Advantages:

Increases model adaptability and generation quality, dynamically optimizing based on feedback.

Disadvantages:

Requires continuous feedback and adjustment mechanisms, which could increase computational load.

Conclusion
This article introduces 16 RAG document splitting strategies, ranging from simple text splitting to complex contextual enhancements. Each strategy has its unique advantages and suitable application scenarios. Choosing the appropriate strategy is crucial for enhancing RAG model performance. By flexibly selecting and combining these strategies, we can optimize RAG models' effectiveness in various fields, especially in complex generation tasks.

In the author's practical experience, the Feedback Loop, Fusion, and Reranker strategies are recommended due to the following reasons:

Feedback Loop: This strategy allows the model to dynamically adjust based on user feedback, continuously optimizing generated content and improving model adaptability and accuracy, particularly suited for highly customized tasks.

Fusion: By combining multiple information sources, Fusion improves the comprehensiveness and accuracy of answers, especially when dealing with complex queries, as it integrates different perspectives and data.

Reranker: During the retrieval phase, Reranker reorders the retrieval results, ensuring that the most relevant document chunks are presented first. This is crucial for improving the precision of retrieval and the accuracy of generated answers, especially for diversified and complex queries.

1. Feedback Loop + RAG
In this combined strategy, we first use RAG to perform document retrieval, and then adjust the generated content based on user feedback.
```
text ="Deep learning is a subset of machine learning. It uses neural networks to model and understand complex patterns in large amounts of data. One popular deep learning technique is Convolutional Neural Networks (CNNs), which are commonly used for image processing tasks such as object recognition and classification. Another significant area in deep learning is Reinforcement Learning, which focuses on decision-making by training agents to take actions in an environment to maximize cumulative rewards."

from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load RAG model and retriever
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
rag_retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

# Load GPT-2 model and tokenizer (for comparison)
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# User query
user_query = "What is deep learning?"

# Assume the user feedback modifies the query to focus more on practical applications like CNNs and Reinforcement Learning
user_feedback = "Focus more on practical applications like CNNs and Reinforcement Learning."

# Adjust the query based on feedback
adjusted_query = f"Deep learning and its applications, focusing on CNNs and Reinforcement Learning."

# Use RAG to retrieve relevant documents
inputs = rag_tokenizer(adjusted_query, return_tensors="pt")
retrieved_docs = rag_retriever.retrieve(adjusted_query)

# Generate the answer using RAG
outputs = rag_model.generate(input_ids=inputs['input_ids'], context_input_ids=retrieved_docs['input_ids'], num_return_sequences=1)
rag_answer = rag_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate answer using GPT-2 (for comparison)
inputs_gpt2 = gpt2_tokenizer.encode(user_query, return_tensors='pt')
outputs_gpt2 = gpt2_model.generate(inputs_gpt2, max_length=100, num_return_sequences=1)
gpt2_answer = gpt2_tokenizer.decode(outputs_gpt2[0], skip_special_tokens=True)

# Display results
print(f"Feedback Loop + RAG Answer: {rag_answer}")
print(f"GPT-2 Model Answer: {gpt2_answer}")
``` 
2. Fusion + RAG
In this strategy, we combine multiple information sources and then use the RAG model to generate a more comprehensive answer. Assume we retrieve information from multiple document chunks and combine them to generate an answer.
```
# Assume we retrieved the following document chunks
document_chunks = [
    "Deep learning is a subset of machine learning, involving neural networks.",
    "Convolutional Neural Networks (CNNs) are widely used for image recognition tasks.",
    "Reinforcement Learning is a technique used to train agents in decision-making tasks."
]

# Simulate fusion of multiple chunks of information
fused_content = " ".join(document_chunks)

# Use RAG to retrieve relevant documents
inputs = rag_tokenizer(fused_content, return_tensors="pt")
retrieved_docs = rag_retriever.retrieve(fused_content)

# Generate the answer using RAG
outputs = rag_model.generate(input_ids=inputs['input_ids'], context_input_ids=retrieved_docs['input_ids'], num_return_sequences=1)
rag_answer = rag_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display results
print(f"Fusion + RAG Answer: {rag_answer}")
```
3. Reranker + RAG
In this strategy, we first use the RAG model for document retrieval and then re-order the retrieved results, ensuring that the most relevant document chunks are prioritized to generate the answer.
```
# Assume we retrieved the following document chunks
document_chunks = [
    "Deep learning is a subset of machine learning.",
    "Convolutional Neural Networks (CNNs) are used for image processing.",
    "Reinforcement learning is focused on decision-making tasks."
]

# Simulate relevance scores (assumed from retrieval)
relevance_scores = [0.9, 0.8, 0.85]  # Example relevance scores

# Re-order document chunks based on relevance
sorted_chunks = [x for _, x in sorted(zip(relevance_scores, document_chunks), reverse=True)]

# Combine the most relevant chunks into the final content
final_content = " ".join(sorted_chunks)

# Use RAG to retrieve relevant documents
inputs = rag_tokenizer(final_content, return_tensors="pt")
retrieved_docs = rag_retriever.retrieve(final_content)

# Generate the answer using RAG
outputs = rag_model.generate(input_ids=inputs['input_ids'], context_input_ids=retrieved_docs['input_ids'], num_return_sequences=1)
rag_answer = rag_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display results
print(f"Reranker + RAG Answer: {rag_answer}")
```

Summary and Comparison of Results:
Feedback Loop + RAG: Based on user feedback, the query is adjusted to focus on specific applications, such as CNNs and Reinforcement Learning, and then RAG generates a more personalized, precise answer.

Example Generated Content: "Deep learning and its applications, focusing on CNNs and Reinforcement Learning."

Fusion + RAG: Multiple information sources (such as different document chunks) are fused together, and then RAG generates a more comprehensive answer by integrating various aspects of the information.

Example Generated Content: "Deep learning is a subset of machine learning, involving neural networks. CNNs are widely used for image recognition, and Reinforcement Learning is a technique used for training agents in decision-making tasks."

Reranker + RAG: The retrieved document chunks are reordered based on relevance scores, ensuring the most relevant chunks are used to generate the answer. This helps prioritize the best matching information.

Example Generated Content: "Deep learning is a subset of machine learning. CNNs are used for image processing. Reinforcement learning is focused on decision-making tasks."

Final Result Comparison:
![Example Image](https://github.com/selene009/selene.github.io/blob/main/1753680005233.png)


Conclusion:
Feedback Loop + RAG: Suitable for highly personalized tasks, where the query needs to be dynamically adjusted based on user feedback.

Fusion + RAG: Suitable for handling complex or multi-faceted queries, where combining information from multiple sources provides a richer and more accurate answer.

Reranker + RAG: Ensures that the most relevant information is prioritized, making it ideal for tasks that require precise retrieval, such as legal document search or academic paper retrieval.

By combining these strategies with RAG, you can significantly improve the quality of generated content, especially in complex scenarios where personalization, comprehensive information, and precise retrieval are crucial.
