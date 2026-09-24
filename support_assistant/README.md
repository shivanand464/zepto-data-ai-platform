# Support Assistant

## 1. Overview

The Support Assistant is a policy-grounded question-answering API for Zepto customer-support policies.

The module:

* Stores Zepto policy documents as a local knowledge base.
* Generates embeddings using `all-MiniLM-L6-v2`.
* Stores and retrieves policy documents using ChromaDB.
* Uses LangGraph to classify the user query and route it through the appropriate workflow.
* Uses a structured prompt containing role, context, task, format, length, negative constraint, and few-shot example sections.
* Exposes a FastAPI `/ask` endpoint.
* Uses a mock LLM-style response by default.
* Validates structured responses with Pydantic.
* Includes validation retry logic for the optional real-LLM path.
* Returns structured responses containing an answer, source documents, and confidence.
* Can be packaged and run using Docker.

The default implementation is self-contained and does not require an external LLM API key.

---

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

---

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

All 8 documents are embedded and queryable through the `zepto_policies` ChromaDB collection.

---

## 4. RAG Architecture

The Support Assistant follows a Retrieval-Augmented Generation (RAG) architecture.

```text
Policy Documents
      |
      v
   Ingestion
      |
      v
Sentence Embeddings
(all-MiniLM-L6-v2)
      |
      v
   ChromaDB
 Vector Database
      |
      |
      |        User Query
      |             |
      |             v
      |        Query Embedding
      |             |
      |             v
      +------> Similarity Retrieval
                    |
                    v
             Top-3 Policy Chunks
                    |
                    v
             Structured Prompt
                    |
                    v
          Response Generation
             (Mock / Optional LLM)
                    |
                    v
          Pydantic Validation
                    |
                    v
       answer / sources / confidence
```

### RAG flow

1. The 8 policy documents are loaded from `docs/`.
2. Each document is converted into an embedding using `all-MiniLM-L6-v2`.
3. The embeddings and source metadata are stored in ChromaDB.
4. A user's query is converted into an embedding using the same model.
5. ChromaDB retrieves the top 3 most relevant policy documents using cosine similarity.
6. The retrieved policy context is inserted into the structured prompt.
7. The response is generated using the default mock behavior or the optional real-LLM extension.
8. The response is validated against the Pydantic response schema.
9. The API returns the answer, source filenames, and confidence.

---

## 5. Embedding and Retrieval

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

---

## 6. LangGraph Workflow

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

---

## 7. Structured Prompt

The application contains a structured prompt template with the following sections:

* ROLE
* CONTEXT
* TASK
* FORMAT
* LENGTH
* NEGATIVE CONSTRAINT
* FEW-SHOT EXAMPLE

The prompt instructs the assistant to answer using only the retrieved Zepto policy context and not invent unsupported information.

The negative constraint helps keep generated responses grounded in the retrieved policy documents.

The few-shot example demonstrates the expected answer format and behavior.

The mock response currently uses the retrieved policy context directly.

---

## 8. Response Validation and Retry Logic

The API response is validated using the following Pydantic structure:

```text
answer: str
sources: list[str]
confidence: float
```

The confidence value must be between `0.0` and `1.0`.

The default mock path returns a valid structured response directly.

The optional real-LLM path includes response validation with:

* 1 initial generation attempt.
* Up to 2 additional attempts if validation fails.
* A total maximum of 3 attempts.
* A validation error if all attempts fail.

This keeps the response contract enforced even when an external LLM provider is configured.

---

## 9. FastAPI API

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

---

## 10. FastAPI Example Calls and Raw JSON Responses

### Example 1: Policy question

Request:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method Post -ContentType "application/json" -Body '{"query":"How long does delivery take?"}'
```

Raw JSON response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard de",
  "sources": [
    "doc_01.txt",
    "doc_02.txt",
    "doc_04.txt"
  ],
  "confidence": 1.0
}
```

This demonstrates that a policy-related query is classified as a policy question, routed to the retrieval workflow, and answered using retrieved policy context.

### Example 2: Non-policy question

Request:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method Post -ContentType "application/json" -Body '{"query":"What is the capital of France?"}'
```

Raw JSON response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This demonstrates that a non-policy question is routed to the direct-answer node rather than the policy retrieval workflow.

---

## 11. Run Locally

Install dependencies:

```powershell
pip install -r requirements.txt
```

If the ChromaDB collection has not been created yet, run:

```powershell
python ingest.py
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

---

## 12. Docker

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

---

## 13. Docker Verification

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

---

## 14. Configuration

The application supports the following environment setting:

```text
MOCK_LLM
```

The current default is:

```text
MOCK_LLM=1
```

This keeps the application self-contained and uses the retrieved policy context to produce the answer without requiring an external LLM API.

### Mock mode

```text
MOCK_LLM=1
```

Mock mode is the default and is suitable for local evaluation without an API key.

### Optional real-LLM mode

```text
MOCK_LLM=0
```

The code contains the optional real-LLM extension point and response-validation retry structure. An external LLM provider would need to be configured before using a real model.

No external LLM API key is required for the default graded mock workflow.

---

## 15. Dependencies

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

---

## 16. Limitations

The current implementation intentionally uses mock LLM behavior by default.

The retrieval pipeline and LangGraph workflow are implemented, but no external LLM provider is required for the current version.

The optional real-LLM extension point is present, including structured response validation and retry handling, but a provider must be configured before real LLM generation can be used.

The assistant is restricted to the Zepto policy knowledge base and does not answer unrelated general-knowledge questions.

---

## 17. End-to-End Architecture

```text
                    Policy Documents
                           |
                           v
                      Ingestion
                           |
                           v
                Sentence Embeddings
                  all-MiniLM-L6-v2
                           |
                           v
                       ChromaDB
                           |
                           |
                           |             User Query
                           |                 |
                           |                 v
                           |          FastAPI /ask
                           |                 |
                           |                 v
                           |        LangGraph Workflow
                           |                 |
                           |          classify_intent
                           |             /        \
                           |            /          \
                           |           v            v
                           |   policy_question   general_question
                           |          |                 |
                           |          v                 v
                           |   Query Embedding    direct_answer
                           |          |
                           |          v
                           +-- Similarity Retrieval
                                      |
                                      v
                               Top-3 Documents
                                      |
                                      v
                              Structured Prompt
                                      |
                                      v
                           Mock / Optional LLM
                                      |
                                      v
                              Pydantic Validation
                                      |
                                      v
                              Structured JSON
                         answer / sources / confidence
```

---

## 18. Module Completion

The Support Assistant module includes:

* [x] 8-document policy corpus
* [x] `all-MiniLM-L6-v2` embedding model
* [x] ChromaDB vector database
* [x] Document ingestion
* [x] Similarity retrieval
* [x] LangGraph `StateGraph`
* [x] `classify_intent` node
* [x] `retrieve_and_answer` node
* [x] `direct_answer` node
* [x] Conditional routing
* [x] Structured prompt
* [x] ROLE / CONTEXT / TASK / FORMAT / LENGTH components
* [x] Negative constraint
* [x] Few-shot example
* [x] Mock LLM behavior
* [x] Pydantic request/response schemas
* [x] Response validation
* [x] Optional real-LLM validation retries
* [x] FastAPI `/ask` endpoint
* [x] Two raw JSON API response examples
* [x] Dockerfile
* [x] Docker image
* [x] Docker container
* [x] End-to-end API testing
* [x] RAG architecture documentation

---

## 19. Summary

The Support Assistant provides a local, policy-grounded RAG workflow for Zepto support questions.

The complete flow is:

```text
8 Policy Documents
        ↓
Embedding
        ↓
ChromaDB
        ↓
User Query
        ↓
LangGraph Intent Classification
        ↓
Top-3 Policy Retrieval
        ↓
Structured Prompt
        ↓
Mock / Optional LLM Generation
        ↓
Pydantic Validation
        ↓
FastAPI JSON Response
```

The default configuration runs without an external LLM API key and provides reproducible policy-grounded responses for evaluation.
