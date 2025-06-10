import logging
import uuid
from typing import Any, Dict, List, Literal, Optional

from autogen_core import CancellationToken, Component, Image
from autogen_core.memory import (
    Memory,
    MemoryContent,
    MemoryMimeType,
    MemoryQueryResult,
    UpdateContextResult,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine

logger = logging.getLogger(__name__)


try:
    from llama_index.core import VectorStoreIndex
except ImportError as e:
    raise ImportError(
        "To use the VectorStoreIndexMemory the llama-index extra must be installed. Run `pip install autogen-ext[llama-index]`"
    ) from e


class VectorStoreIndexMemory(Memory):

    def __init__(
        self,
        vector_store: "BasePydanticVectorStore",
        embed_model: Optional["EmbedType"] = None,
        **kwargs: Any,
    ):
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, embed_model=self.embed_model, **kwargs
        )
        self._engine = self._index.as_query_engine()

    @property
    def vector_store(self) -> BasePydanticVectorStore:
        return self._vector_store

    @property
    def embed_model(self) -> Optional["EmbedType"]:
        return self._embed_model

    @property
    def index(self) -> VectorStoreIndex:
        return self._index

    @property
    def retriever(self) -> BaseQueryEngine:
        return self._engine

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        # Extract query from last message
        last_message = messages[-1]
        query_text = (
            last_message.content
            if isinstance(last_message.content, str)
            else str(last_message)
        )

        # Query memory and get results
        query_results = await self.query(query_text)

        if query_results.results:
            # Format results for context
            memory_strings = [
                f"{i}. {str(memory.content)}"
                for i, memory in enumerate(query_results.results, 1)
            ]
            memory_context = "\nRelevant memory content:\n" + "\n".join(memory_strings)

            # Add to context
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=query_results)

    def _extract_text(self, content_item: str | MemoryContent) -> str:
        """Extract searchable text from content."""
        if isinstance(content_item, str):
            return content_item

        content = content_item.content
        mime_type = content_item.mime_type

        if mime_type in [MemoryMimeType.TEXT, MemoryMimeType.MARKDOWN]:
            return str(content)
        elif mime_type == MemoryMimeType.JSON:
            if isinstance(content, dict):
                # Store original JSON string representation
                return str(content).lower()
            raise ValueError("JSON content must be a dict")
        elif isinstance(content, Image):
            raise ValueError("Image content cannot be converted to text")
        else:
            raise ValueError(f"Unsupported content type: {mime_type}")

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Query memory content based on vector similarity."""

        try:
            # Extract text for query
            query_text = self._extract_text(query)

            # Query ChromaDB
            results = self._engine.query(
                query=query_text,
            )

            # Convert results to MemoryContent list
            memory_results: List[MemoryContent] = []

            if (
                not results
                or not results.get("documents")
                or not results.get("metadatas")
                or not results.get("distances")
            ):
                return MemoryQueryResult(results=memory_results)

            documents: List[Document] = (
                results["documents"][0] if results["documents"] else []
            )
            metadatas: List[Metadata] = (
                results["metadatas"][0] if results["metadatas"] else []
            )
            distances: List[float] = (
                results["distances"][0] if results["distances"] else []
            )
            ids: List[str] = results["ids"][0] if results["ids"] else []

            for doc, metadata_dict, distance, doc_id in zip(
                documents, metadatas, distances, ids, strict=False
            ):
                # Calculate score
                score = self._calculate_score(distance)
                metadata = dict(metadata_dict)
                metadata["score"] = score
                metadata["id"] = doc_id
                if (
                    self._config.score_threshold is not None
                    and score < self._config.score_threshold
                ):
                    continue

                # Extract mime_type from metadata
                mime_type = str(
                    metadata_dict.get("mime_type", MemoryMimeType.TEXT.value)
                )

                # Create MemoryContent
                content = MemoryContent(
                    content=doc,
                    mime_type=mime_type,
                    metadata=metadata,
                )
                memory_results.append(content)

            return MemoryQueryResult(results=memory_results)

        except Exception as e:
            logger.error(f"Failed to query ChromaDB: {e}")
            raise

    @abstractmethod
    async def add(
        self,
        content: MemoryContent,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """
        Add a new content to memory.

        Args:
            content: The memory content to add
            cancellation_token: Optional token to cancel operation
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all entries from memory."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up any resources used by the memory implementation."""
        ...
