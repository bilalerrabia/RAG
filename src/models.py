"""Pydantic models for type-safe data handling in the RAG pipeline."""
from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """Represents a minimal source of information."""
    file_path: str
    first_character_index: int
    last_character_index: int


class ChunkData(MinimalSource):
    """Represents a chunk of data with its text content."""
    text: str


class UnansweredQuestion(BaseModel):
    """Represents an unanswered question."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_str: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents an answered question with sources."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset of RAG questions."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represents the search results for a single question."""
    question_id: str
    question_str: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents an answer to a question."""
    answer: str


class StudentSearchResults(BaseModel):
    """Represents the search results for a dataset of questions."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Represents the search results and answers for a dataset of questions."""
    search_results: List[MinimalSearchResults]
