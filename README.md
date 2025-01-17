Problem Statement #1: Building a Simple Q&A Chatbot with RAG and LangChain

Create a question-answering system that can understand, and answer questions based on PDF documents (like course materials or company documentation). The system should use LangChain and RAG (Retrieval - Augmented Generation) to provide accurate answers based on the document content.

Project Goals

1. Build a working prototype that can:
a. Read and process PDF documents
b. Answer questions based on the document content
c. Provide relevant responses using RAG architecture

2. Learn key concepts:
a. Document processing with LangChain
b. Vector embeddings
c. RAG architecture
d. LLM integration


Technical Requirements


Core Components

1. Document Processing
a. Process 1-3 PDF documents
b. Convert documents into text chunks using LangChain
c. Store document contents properly

2. Vector Store Setup
a. Create embeddings using LangChain
b. Set up Qdrant (cloud or on-premises)
c. Implement basic vector search

3. Question-Answering System
a. Set up document retrieval
b. Connect with chosen LLM
c. Generate answers using retrieved context

4. Web Interface
a. Create a chat interface using any web technology
b. Handle user input and display responses


Technical Stack
• Python 3.9+
• LangChain (for document processing, embeddings, and LLM integration)
• Qdrant (vector store)
• Choice of LLM API:
 OpenAI
 Google Gemini
 Anthropic Claude
 Or any other preferred LLM
• Choice of web technology for interface


Expected Deliverables

1. Daily GitHub commit updates (for non-code commits, commit bullet-point updates to a UPDATES.md)

2. Working Python application

3. Web-based chat interface

4. Sample documents for testing

5. Basic documentation

6. Project presentation
Learning Outcomes
• Understanding of RAG architecture
• Experience with LangChain
• Practical knowledge of vector databases
• Basic prompt engineering skills
