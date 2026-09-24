# Zepto Data & AI Platform

End-to-end data engineering, analytics, machine learning, and RAG-based support assistant capstone project.

The project contains three independent modules:

1. **Data Pipeline** — scrapes and cleans book data, stores it in normalized SQLite tables, and performs SQL/Pandas analysis.
2. **Analytics** — performs Titanic EDA, statistical analysis, classification, imbalance handling, hyperparameter tuning, and regression.
3. **Support Assistant** — provides a policy-grounded RAG API using embeddings, ChromaDB, LangGraph, FastAPI, and Docker.

---

## 1. Project Structure

```text
zepto-capstone/
│
├── data_pipeline/
│   ├── scrape_and_load.py
│   ├── queries.py
│   ├── validate_pandas.py
│   ├── check_database.py
│   ├── books.db
│   ├── query_results.txt
│   └── README.md
│
├── analytics/
│   ├── analysis.py
│   ├── titanic.csv
│   ├── best_classification_pipeline.joblib
│   ├── plots/
│   └── README.md
│
├── support_assistant/
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   ├── chroma_db/
│   ├── ingest.py
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── .gitignore
└── README.md
```

---

# 2. Setup

## Requirements

The project was developed using:

* Python 3.12
* Git
* GitHub
* VS Code
* Docker for the Support Assistant container workflow

Create and activate a virtual environment from the project root:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies required by the modules.

For the Data Pipeline:

```powershell
pip install requests beautifulsoup4 pandas
```

For Analytics:

```powershell
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn joblib
```

For the Support Assistant:

```powershell
pip install -r support_assistant/requirements.txt
```

---

# 3. Module 1 — Data Pipeline

## Objective

The Data Pipeline module scrapes book information from:

```text
https://books.toscrape.com/
```

The pipeline collects books from multiple categories, cleans the scraped values, converts prices from GBP to INR, stores the results in normalized SQLite tables, and performs SQL and Pandas analysis.

## Pipeline flow

```text
books.toscrape.com
        |
        v
Requests + BeautifulSoup
        |
        v
Raw book information
        |
        v
Cleaning and normalization
        |
        v
GBP → INR conversion
(1 GBP = 105.50 INR)
        |
        v
SQLite
(categories + books)
        |
        +----------------+
        |                |
        v                v
      SQL            Pandas
     queries          validation
```

## Run the scraper and database pipeline

From the project root:

```powershell
python data_pipeline/scrape_and_load.py
```

This creates/populates:

```text
data_pipeline/books.db
```

The completed dataset contains books from at least three categories, including:

* Travel
* Mystery
* Historical Fiction

The completed run contains 69 books across 3 categories.

## Run SQL analysis

```powershell
python data_pipeline/queries.py
```

The SQL analysis includes queries demonstrating:

* `SELECT`
* `WHERE`
* `ORDER BY`
* `LIMIT`
* `DISTINCT`
* `IN`
* `BETWEEN`
* `JOIN`

Query strings and outputs are saved in:

```text
data_pipeline/query_results.txt
```

## Validate SQL results with Pandas

```powershell
python data_pipeline/validate_pandas.py
```

The Pandas validation uses `pd.read_sql()` for SQL queries and independently reproduces the JOIN result using `pd.merge()`.

## Database validation

```powershell
python data_pipeline/check_database.py
```

The database uses normalized:

```text
categories
books
```

tables with primary-key and foreign-key relationships.

Additional details are documented in:

```text
data_pipeline/README.md
```

---

# 4. Module 2 — Analytics

## Objective

The Analytics module performs exploratory data analysis and predictive modeling using the Titanic dataset.

The analysis covers:

* Dataset profiling
* Missing-value analysis
* Missing-value handling
* Univariate analysis
* IQR-based outlier analysis
* Fare statistics
* Survival analysis
* Correlation analysis
* Multivariate visualization
* Feature standardization
* Stratified train/test splitting
* Training-only preprocessing
* Logistic Regression
* Decision Tree
* Random Forest
* Imbalance handling
* GridSearchCV
* Random Forest OOB evaluation
* Confusion matrices
* ROC/AUC evaluation
* Regression diagnostics

## Dataset

The analysis loads the Titanic dataset using:

```python
sns.load_dataset("titanic")
```

A committed fallback copy is also stored as:

```text
analytics/titanic.csv
```

## Run the analysis

From the project root:

```powershell
python analytics/analysis.py
```

The script generates the required analysis outputs and plots under the analytics output directories.

## Classification models

The classification workflow evaluates:

* Logistic Regression
* Decision Tree
* Random Forest

The evaluation includes:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* Confusion matrices
* ROC curves

The analysis also compares:

* Baseline training
* Class-weighted training
* SMOTE training

SMOTE is applied only to the training data to avoid test-set leakage.

## Hyperparameter tuning

Random Forest hyperparameters are tuned with GridSearchCV using:

* `n_estimators`
* `max_depth`
* `max_features`

The tuned model also reports its out-of-bag score.

The fitted best classification pipeline is saved as:

```text
analytics/best_classification_pipeline.joblib
```

## Regression

The regression analysis reports:

* MAE
* RMSE
* R²
* Adjusted R²

Residual analysis is also performed to examine whether residual spread changes across fitted values.

Additional details, findings, model results, and plots are documented in:

```text
analytics/README.md
```

---

# 5. Module 3 — Support Assistant

## Objective

The Support Assistant is a policy-grounded RAG application for Zepto customer-support questions.

It uses:

* 8 policy documents
* `all-MiniLM-L6-v2`
* Text chunking with overlap
* ChromaDB
* LangGraph
* FastAPI
* Pydantic
* Docker

The default mode does not require an external LLM API key.

## RAG architecture

```text
8 Policy Documents
        |
        v
Document Ingestion
        |
        v
Text Chunking
500-character chunks
50-character overlap
        |
        v
all-MiniLM-L6-v2
        |
        v
ChromaDB
        |
        |
        |       User Query
        |           |
        |           v
        |      Query Embedding
        |           |
        +----> Similarity Search
                    |
                    v
              Top-3 Chunks
                    |
                    v
             Structured Prompt
                    |
                    v
             Response Generation
                    |
                    v
             Pydantic Validation
                    |
                    v
       answer / sources / confidence
```

## Build the knowledge base

From the project root:

```powershell
python support_assistant/ingest.py
```

The ingestion process:

1. Loads all 8 policy documents.
2. Splits the documents into overlapping text chunks.
3. Generates normalized embeddings using `all-MiniLM-L6-v2`.
4. Creates the persistent ChromaDB collection `zepto_policies`.
5. Stores source filenames and chunk metadata.
6. Runs a retrieval test.

The current corpus produces 11 chunks.

## LangGraph workflow

The application uses three nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

Workflow:

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

Policy-related keywords include:

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

## Structured prompt

The Support Assistant prompt contains:

* ROLE
* CONTEXT
* TASK
* FORMAT
* LENGTH
* NEGATIVE CONSTRAINT
* FEW-SHOT EXAMPLE

The negative constraint keeps the answer grounded in the retrieved policy context.

## Response schema

Responses are validated using Pydantic with:

```text
answer: str
sources: list[str]
confidence: float
```

The confidence value is constrained to the range:

```text
0.0 – 1.0
```

The optional real-LLM path includes one initial validation attempt plus up to two additional retries if validation fails.

## Run the API

From the project root:

```powershell
uvicorn support_assistant.main:app --reload
```

The API runs at:

```text
http://localhost:8000
```

### Policy question

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"How long does delivery take?"}'
```

The response contains the retrieved policy answer and source documents.

### Non-policy question

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What is the capital of France?"}'
```

The assistant returns a direct response explaining that it only answers Zepto policy questions.

Additional Support Assistant documentation and raw JSON examples are available in:

```text
support_assistant/README.md
```

---

# 6. Docker — Support Assistant

The Support Assistant includes a Python 3.12 Dockerfile.

From the `support_assistant` directory:

```powershell
cd support_assistant
```

Build the image:

```powershell
docker build -t zepto-support-assistant .
```

Run the container:

```powershell
docker run -d `
  --name zepto-support-assistant-container `
  -p 8000:8000 `
  zepto-support-assistant
```

Check the container:

```powershell
docker ps
```

Test the API:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"How long does delivery take?"}'
```

Stop the container when finished:

```powershell
docker stop zepto-support-assistant-container
```

Remove it if required:

```powershell
docker rm zepto-support-assistant-container
```

Return to the project root:

```powershell
cd ..
```

---

# 7. Design Decisions

## Data Pipeline

### Requests + BeautifulSoup

The scraper uses Requests and BeautifulSoup to retrieve and parse the website without requiring an API key.

### Fixed currency conversion

The required baseline conversion is used:

```text
1 GBP = 105.50 INR
```

This provides a deterministic conversion for evaluation.

### Normalized SQLite schema

Book and category information are stored separately using:

```text
categories
books
```

with a foreign-key relationship.

This avoids repeatedly storing category information for every book.

### SQL + Pandas validation

SQL is used for database-side analysis while Pandas independently validates selected results.

The JOIN result is reproduced using:

```python
pd.merge()
```

to demonstrate that the SQL and DataFrame representations agree.

---

## Analytics

### Training-only preprocessing

Preprocessing is fitted only on the training data and then applied to the test data.

This avoids information leakage from the test set.

### Stratified splitting

The classification split preserves the class distribution between training and test sets.

### Multiple classifiers

Logistic Regression, Decision Tree, and Random Forest are evaluated to compare different model structures.

### Imbalance handling

Baseline, class-weighted, and SMOTE approaches are compared rather than assuming a single imbalance strategy is universally appropriate.

SMOTE is applied only to training data.

### Reproducibility

Random states are used where appropriate so that the analysis and model evaluation can be reproduced.

---

## Support Assistant

### Local embeddings

`all-MiniLM-L6-v2` provides a lightweight local embedding model without requiring an external API key.

### Chunking with overlap

Documents are split into 500-character chunks with 50-character overlap.

The overlap helps preserve context between neighboring chunks.

### ChromaDB

ChromaDB provides persistent local vector storage and cosine-similarity retrieval.

### LangGraph routing

LangGraph separates intent classification from retrieval-based and direct-answer workflows.

### Mock-first architecture

The default `MOCK_LLM=1` mode makes the application runnable without an external LLM provider or API key.

### Structured validation

Pydantic enforces the expected response shape and confidence range.

The optional real-LLM path includes retry handling for validation failures.

### FastAPI

FastAPI provides a simple HTTP interface around the LangGraph workflow.

---

# 8. Environment Configuration

The Support Assistant supports:

```text
MOCK_LLM
```

Default:

```text
MOCK_LLM=1
```

Mock mode is self-contained and does not require an external LLM API key.

Optional real-LLM mode can be selected with:

```text
MOCK_LLM=0
```

An external provider must be configured separately before real LLM generation can be used.

No API key is required for the default graded workflow.

---

# 9. Verification Completed

## Data Pipeline

The completed pipeline contains:

* 69 books
* 3 categories
* Normalized SQLite schema
* Required SQL query patterns
* SQL query outputs
* Pandas SQL validation
* Independent JOIN reproduction with `pd.merge()`

## Analytics

The completed analysis contains:

* Titanic profiling
* Missing-value analysis
* Univariate analysis
* IQR outlier analysis
* Fare statistics
* Survival analysis
* Correlation matrix and heatmap
* Multivariate charts
* Standardization sanity check
* Stratified classification split
* Training-only preprocessing
* Logistic Regression
* Decision Tree
* Random Forest
* Imbalance comparison
* GridSearchCV
* OOB evaluation
* Confusion matrices
* ROC/AUC analysis
* Regression metrics
* Regression residual analysis
* Saved fitted classification pipeline

## Support Assistant

The completed Support Assistant contains:

* 8 policy documents
* 11 retrieval chunks
* `all-MiniLM-L6-v2` embeddings
* Persistent ChromaDB
* Top-3 similarity retrieval
* LangGraph `StateGraph`
* Three workflow nodes
* Conditional routing
* Structured prompt
* Mock LLM behavior
* Pydantic validation
* Optional validation retries
* FastAPI `/ask`
* Dockerfile
* Docker configuration
* Policy and non-policy API tests
* RAG architecture documentation

---

# 10. Git Workflow

Development was organized using feature branches.

The project includes separate feature branches for:

```text
feature/data-pipeline
feature/analytics
feature/analytics-improvements
feature/support-assistant
```

The completed work was merged into:

```text
main
```

The final `main` branch was pushed to GitHub and verified with a clean working tree.

---

# 11. Running the Complete Project

## Data Pipeline

```powershell
python data_pipeline/scrape_and_load.py
python data_pipeline/queries.py
python data_pipeline/validate_pandas.py
python data_pipeline/check_database.py
```

## Analytics

```powershell
python analytics/analysis.py
```

## Support Assistant

Build the knowledge base:

```powershell
python support_assistant/ingest.py
```

Start the API:

```powershell
uvicorn support_assistant.main:app --reload
```

Then send requests to:

```text
POST http://localhost:8000/ask
```

---

# 12. Repository

GitHub repository:

```text
https://github.com/shivanand464/zepto-data-ai-platform
```

The repository contains all three completed modules and their supporting documentation.

---

# 13. Final Project Status

```text
Data Pipeline       COMPLETE
Analytics           COMPLETE
Support Assistant   COMPLETE
Documentation       COMPLETE
Git integration     COMPLETE
```

The project is organized as one public GitHub repository with the three required modules and module-specific documentation.
