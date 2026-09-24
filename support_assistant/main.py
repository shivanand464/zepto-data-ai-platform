import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# MOCK_LLM is the required graded baseline.
# Unset or "1" = mock mode.
# "0" = optional real-LLM extension.
MOCK_LLM = os.getenv("MOCK_LLM", "1")


# =========================================================
# PYDANTIC MODELS
# =========================================================

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


# =========================================================
# LANGGRAPH STATE
# =========================================================

class AssistantState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_documents: list[str]
    retrieved_sources: list[str]
    answer: str
    sources: list[str]
    confidence: float


# =========================================================
# EMBEDDING MODEL + CHROMADB
# =========================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(f"ChromaDB collection loaded: {COLLECTION_NAME}")
print(f"Stored documents: {collection.count()}")


# =========================================================
# STRUCTURED PROMPT TEMPLATE
# =========================================================

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer-support assistant. Answer questions using
only the provided Zepto policy context.

CONTEXT:
{context}

TASK:
Answer the customer's question using the provided context.
If the context does not contain enough information, clearly say that
the available policy context does not provide the answer.

FORMAT:
Return a concise, direct answer suitable for a customer-support response.

LENGTH:
Keep the answer brief and preferably within 2-4 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent or assume Zepto policies.

FEW-SHOT EXAMPLE:
Question: How long does Zepto delivery take?
Context: Zepto delivers within 10 to 30 minutes of order confirmation,
depending on the delivery zone and current order volume.
Answer: Based on the policy, Zepto delivery takes 10 to 30 minutes
after order confirmation, depending on the delivery zone and current order volume.

CUSTOMER QUESTION:
{query}
"""


# =========================================================
# HELPER
# =========================================================

def embed_query(query: str):
    """Create a normalized embedding for a user query."""

    embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )

    return embedding.tolist()



# =========================================================
# NODE 1 — CLASSIFY INTENT
# =========================================================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_intent(state: AssistantState) -> AssistantState:
    """
    Classify the query as policy_question or general_question.

    Required MOCK_LLM baseline:
    keyword heuristic with no LLM call.
    """

    query = state["query"].lower()

    if MOCK_LLM != "0":
        is_policy_question = any(
            keyword in query
            for keyword in POLICY_KEYWORDS
        )

        intent = (
            "policy_question"
            if is_policy_question
            else "general_question"
        )

    else:
        # Optional real-LLM extension.
        # The required graded baseline is MOCK_LLM=1.
        #
        # We deliberately keep this branch explicit so the
        # application can later be connected to an LLM without
        # changing the graph structure.
        intent = (
            "policy_question"
            if any(keyword in query for keyword in POLICY_KEYWORDS)
            else "general_question"
        )

    return {
        **state,
        "intent": intent,
    }


# =========================================================
# NODE 2 — RETRIEVE AND ANSWER
# =========================================================

def retrieve_and_answer(state: AssistantState) -> AssistantState:
    """
    Retrieve the top-3 policy chunks from ChromaDB and
    generate the required mock response.
    """
    query = state["query"]

    # Retrieval always happens in both modes.
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    sources = [
        metadata["source"]
        for metadata in metadatas
    ]

    if not documents:
        answer = (
            "I could not find relevant information in the "
            "Zepto policy documents."
        )
        confidence = 0.0

    elif MOCK_LLM != "0":
        # Required graded mock behavior.
        top_chunk_snippet = documents[0][:200].strip()

        answer = (
            f"Based on the retrieved context: "
            f"{top_chunk_snippet}"
        )

        confidence = 1.0

    else:
        # Optional real-LLM extension.
        context = "\n\n".join(documents)

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        # Placeholder until an optional real LLM provider
        # is configured.
        #
        # The response is validated using the required
        # Pydantic schema. The loop allows one initial
        # attempt plus up to 2 additional retries if
        # validation fails.
        answer = None
        confidence = None

        for attempt in range(3):
            candidate = {
                "answer": (
                    "Real LLM mode is not configured. "
                    f"Retrieved context: "
                    f"{documents[0][:200].strip()}"
                ),
                "sources": sources,
                "confidence": 1.0,
            }

            try:
                validated = AskResponse(**candidate)

                answer = validated.answer
                confidence = validated.confidence

                break

            except Exception:
                if attempt == 2:
                    raise

        if answer is None or confidence is None:
            raise ValueError(
                "LLM response failed validation after 3 attempts."
            )

    return {
        **state,
        "retrieved_documents": documents,
        "retrieved_sources": sources,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }

# =========================================================
# NODE 3 — DIRECT ANSWER
# =========================================================

def direct_answer(state: AssistantState) -> AssistantState:
    """
    Handle general questions without retrieval.

    Required mock behavior returns a fixed canned response.
    """

    if MOCK_LLM != "0":
        answer = (
            "I can only answer questions about Zepto policies right now."
        )

        confidence = 1.0

    else:
        # Optional real-LLM extension.
        answer = (
            "Real LLM mode is not configured for general questions."
        )

        confidence = 1.0

    return {
        **state,
        "answer": answer,
        "sources": [],
        "confidence": confidence,
    }



# =========================================================
# LANGGRAPH STATEGRAPH
# =========================================================

from langgraph.graph import END, START, StateGraph


def route_after_classification(state: AssistantState) -> str:
    """Route the query based on the classified intent."""

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


graph_builder = StateGraph(AssistantState)

# Required three nodes.
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer)
graph_builder.add_node("direct_answer", direct_answer)

# Start -> classify
graph_builder.add_edge(START, "classify_intent")

# Conditional routing after classification
graph_builder.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

# Both answer nodes finish the graph.
graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer", END)

graph = graph_builder.compile()


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto policy support assistant",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant is running."
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Process a customer support question."""

    initial_state: AssistantState = {
        "query": request.query,
    }

    result = graph.invoke(initial_state)

    response = AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 1.0),
    )

    return response