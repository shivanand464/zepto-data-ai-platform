# Support Assistant

## 1. Overview

The Support Assistant is a policy-grounded question-answering API for Zepto customer-support policies.

The module:

* Stores Zepto policy documents as a local knowledge base.
* Generates embeddings using `all-MiniLM-L6-v2`.
* Stores and retrieves policy documents using ChromaDB.
* Uses LangGraph to classify the user query and route it through the appropriate workflow.
* Exposes a FastAPI `/ask` endpoint.
* Uses a mock LLM-style response in the current implementation.
* Returns structured responses containing an answer, source documents, and confidence.
* Can be packaged and run using Docker.

## 2. Project Structure

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── chroma_db/
├── ingest.py
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 3. Policy Knowledge Base

The `docs/` directory contains 8 policy documents covering customer-support topics such as:

* Delivery
* Returns
* Refunds
* Membership
* Order tracking
* Order cancellation
* Gift cards
* Customer support

The documents are loaded during the ingestion process and stored in ChromaDB with their source filenames as metadata.

## 4. Embedding and Retrieval

The embedding model used is:

```text
all-MiniLM-L6-v2
```

The ingestion workflow:

1. Loads all `doc_*.txt` files.
2. Generates normalized sentence embeddings.
3. Creates a persistent ChromaDB collection named `zepto_policies`.
4. Stores each document with its source filename.
5. Uses cosine distance for similarity retrieval.
6. Retrieves the top 3 documents for a user query.

### Run ingestion

From the `support_assistant` directory:

```powershell
python ingest.py
```

Expected output includes:

```text
Loaded 8 policy documents.
ChromaDB ingestion complete.
Collection: zepto_policies
Stored documents: 8
```

The script also performs a retrieval test using:

```text
How long does Zepto delivery take?
```

## 5. LangGraph Workflow

The assistant uses three LangGraph nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The workflow is:

```text
START
  |
  v
classify_intent
  |
  +---- policy_question ----> retrieve_and_answer ----> END
  |
  +---- general_question ---> direct_answer ----------> END
```

### Intent classification

The current implementation uses a lightweight keyword-based classifier.

Policy-related keywords include terms such as:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

Policy questions are routed to the retrieval workflow.

Non-policy questions are routed to `direct_answer`, which returns:

```text
I can only answer questions about Zepto policies right now.
```

## 6. Structured Prompt

The application contains a structured prompt template with the following sections:

* ROLE
* CONTEXT
* TASK
* FORMAT
* LENGTH
* NEGATIVE CONSTRAINT
* FEW-SHOT EXAMPLE

The prompt instructs the assistant to answer using only the retrieved Zepto policy context and not invent unsupported information.

The mock response currently uses the retrieved policy context directly.

## 7. FastAPI API

The application exposes:

```text
POST /ask
```

### Request schema

```json
{
  "query": "How long does delivery take?"
}
```

The request is validated using Pydantic.

### Response schema

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "doc_01.txt",
    "doc_02.txt",
    "doc_04.txt"
  ],
  "confidence": 1.0
}
```

The response contains:

* `answer`: Generated/mock answer.
* `sources`: Retrieved policy document filenames.
* `confidence`: Confidence value between 0 and 1.

## 8. Run Locally

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the API:

```powershell
uvicorn main:app --reload
```

The API is available at:

```text
http://localhost:8000
```

### Policy question test

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method Post -ContentType "application/json" -Body '{"query":"How long does delivery take?"}'
```

The tested response successfully retrieved policy context beginning with:

```text
Zepto delivers grocery and household essentials...
```

### Non-policy question test

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method Post -ContentType "application/json" -Body '{"query":"What is the capital of France?"}'
```

Expected behavior:

```text
I can only answer questions about Zepto policies right now.
```

## 9. Docker

The module includes a Dockerfile based on Python 3.12.

### Build the image

From the `support_assistant` directory:

```powershell
docker build -t zepto-support-assistant .
```

### Run the container

```powershell
docker run -d --name zepto-support-assistant-container -p 8000:8000 zepto-support-assistant
```

### Check the running container

```powershell
docker ps
```

The container exposes:

```text
0.0.0.0:8000 -> 8000/tcp
```

### Test the Dockerized API

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method Post -ContentType "application/json" -Body '{"query":"How long does delivery take?"}'
```

The Dockerized API was tested successfully with a policy question and returned retrieved Zepto policy context.

The non-policy route was also tested successfully:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method Post -ContentType "application/json" -Body '{"query":"What is the capital of France?"}'
```

Result:

```text
I can only answer questions about Zepto policies right now.
```

## 10. Docker Verification

The Docker image was successfully built as:

```text
zepto-support-assistant:latest
```

The image was verified using:

```powershell
docker images zepto-support-assistant
```

The running container was verified using:

```powershell
docker ps
```

The final Dockerized API tests confirmed:

1. Policy questions reach the retrieval workflow.
2. Retrieved policy context is returned in the response.
3. Source document names are included.
4. Non-policy questions reach the direct-answer workflow.
5. The FastAPI endpoint works from inside the Docker container.

## 11. Configuration

The application supports the following environment setting:

```text
MOCK_LLM
```

The current default is:

```text
MOCK_LLM=1
```

This keeps the application self-contained and uses the retrieved policy context to produce the answer without requiring an external LLM API.

## 12. Dependencies

Main Python dependencies:

```text
fastapi
uvicorn
chromadb
sentence-transformers
langgraph
pydantic
```

Install them with:

```powershell
pip install -r requirements.txt
```

## 13. Limitations

The current implementation intentionally uses mock LLM behavior.

The retrieval pipeline and LangGraph workflow are implemented, but no external LLM provider is required for the current version.

The assistant is restricted to the Zepto policy knowledge base and does not answer unrelated general-knowledge questions.

## 14. End-to-End Architecture

```text
                 User Query
                     |
                     v
              FastAPI /ask
                     |
                     v
             LangGraph Workflow
                     |
              classify_intent
                /         \
               /           \
              v             v
     policy_question   general_question
           |                 |
           v                 v
   ChromaDB Retrieval   direct_answer
           |
           v
   Retrieved Policy Docs
           |
           v
     Mock LLM Response
           |
           v
     Structured JSON
   answer / sources /
       confidence
```

## 15. Module Completion

The Support Assistant module includes:

* [x] Policy corpus
* [x] Embedding model
* [x] ChromaDB vector database
* [x] Document ingestion
* [x] Similarity retrieval
* [x] LangGraph workflow
* [x] Three workflow nodes
* [x] Conditional routing
* [x] Structured prompt
* [x] Mock LLM behavior
* [x] Pydantic request/response schemas
* [x] FastAPI `/ask` endpoint
* [x] Dockerfile
* [x] Docker image
* [x] Docker container
* [x] End-to-end API testing
