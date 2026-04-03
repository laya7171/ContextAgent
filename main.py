import argparse
import json
from pathlib import Path
from typing import List, TypedDict

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_tavily import TavilySearch as TavilySearchTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph as GraphBuilder


class State(TypedDict):
	query: str
	answer: str
	response: str
	tool_calls: bool
	suggestion: List[str]


class AppContext:
	def __init__(self) -> None:
		load_dotenv()

		self.embedding_model = OllamaEmbeddings(model="all-minilm:33m")
		self.answer_llm = ChatOllama(model="phi4-mini:latest")
		self.tavily_search_tool = TavilySearchTool(max_results=1)

		self.client = chromadb.PersistentClient(path="./chroma_db")
		self.collection = self.client.get_or_create_collection(name="my_chunks")

	def ensure_pdf_indexed(self, pdf_path: str) -> None:
		if self.collection.count() > 0:
			return

		loader = PyPDFLoader(pdf_path)
		documents = loader.load()
		splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=5)
		chunks = splitter.split_documents(documents)

		docs_for_chroma = [chunk.page_content for chunk in chunks]
		metas_for_chroma = [chunk.metadata for chunk in chunks]
		ids = [f"chunk-{i}" for i in range(len(chunks))]
		embeddings = self.embedding_model.embed_documents(docs_for_chroma)

		self.collection.upsert(
			ids=ids,
			embeddings=embeddings,
			documents=docs_for_chroma,
			metadatas=metas_for_chroma,
		)


def build_graph(ctx: AppContext):
	def chat_node(state: State):
		query = state.get("query", "").strip()
		if not query:
			return {
				"query": "",
				"answer": "Please provide a query.",
				"tool_calls": True,
				"suggestion": ["Ask a clear question so I can retrieve context."],
			}

		query_vector = ctx.embedding_model.embed_query(query)
		result = ctx.collection.query(query_embeddings=[query_vector], n_results=3)
		retrieved_docs = result.get("documents", [[]])[0] if result.get("documents") else []
		usable_docs = [doc for doc in retrieved_docs if isinstance(doc, str) and doc.strip()]

		if not usable_docs:
			return {
				"query": query,
				"answer": "",
				"tool_calls": True,
				"suggestion": ["No relevant context found in vector DB. Trigger the tool node."],
			}

		retrieved_text = "\n\n".join(usable_docs)
		prompt = PromptTemplate(
			template=(
				"You are a helpful RAG assistant. Use only the retrieved context to answer. "
				"If context is insufficient, say that clearly.\n\n"
				"Retrieved documents:\n{retrieved_docs}\n\n"
				"User query: {user_query}"
			),
			input_variables=["retrieved_docs", "user_query"],
		)
		formatted_prompt = prompt.format(retrieved_docs=retrieved_text, user_query=query)

		answer_msg = ctx.answer_llm.invoke(formatted_prompt)
		answer_text = answer_msg.content if hasattr(answer_msg, "content") else str(answer_msg)

		return {
			"query": query,
			"answer": answer_text,
			"tool_calls": False,
			"suggestion": [],
		}

	def tools_decider(state: State):
		return "tool_node" if state.get("tool_calls", False) else "answer_node"

	def tool_node(state: State):
		query = state["query"]
		try:
			search_response = ctx.tavily_search_tool.invoke(query)
			results = search_response.get("results", [])
			if not results:
				return {"answer": "No web results found."}
			return {"answer": results[0].get("content", "No content found.")}
		except Exception:
			return {"answer": "Web tool failed. Please check Tavily API key/configuration."}

	def answer_node(state: State):
		query = state["query"]
		answer = state["answer"]
		prompt = PromptTemplate(
			template=(
				"You are a helpful assistant. Answer the user query based on the provided context.\n"
				"Query: {query}\n"
				"Context: {answer}"
			),
			input_variables=["query", "answer"],
		)
		formatted_prompt = prompt.format(query=query, answer=answer)
		response_msg = ctx.answer_llm.invoke(formatted_prompt)
		response_text = response_msg.content if hasattr(response_msg, "content") else str(response_msg)
		return {"response": response_text}

	def suggestion_node(state: State):
		response = state["response"]
		prompt = PromptTemplate(
			template=(
				"You are a helpful assistant. Generate exactly 3 follow-up questions based on the query and response.\n"
				"Query: {query}\n"
				"Response: {response}\n"
				"Return each question on a new line without numbering."
			),
			input_variables=["query", "response"],
		)
		formatted_prompt = prompt.format(query=state["query"], response=response)
		suggestion_msg = ctx.answer_llm.invoke(formatted_prompt)
		suggestion_text = suggestion_msg.content if hasattr(suggestion_msg, "content") else str(suggestion_msg)
		suggestions = [line.strip(" -0123456789.") for line in suggestion_text.splitlines() if line.strip()]
		return {"suggestion": suggestions[:3]}

	graph_builder = GraphBuilder(State)
	graph_builder.add_node("chat_node", chat_node)
	graph_builder.add_node("tool_node", tool_node)
	graph_builder.add_node("answer_node", answer_node)
	graph_builder.add_node("suggestion_node", suggestion_node)

	graph_builder.add_edge(START, "chat_node")
	graph_builder.add_conditional_edges(
		"chat_node",
		tools_decider,
		{
			"tool_node": "tool_node",
			"answer_node": "answer_node",
		},
	)
	graph_builder.add_edge("tool_node", "answer_node")
	graph_builder.add_edge("answer_node", "suggestion_node")
	graph_builder.add_edge("suggestion_node", END)
	return graph_builder.compile()


def main() -> None:
	parser = argparse.ArgumentParser(description="RAG graph runner")
	parser.add_argument(
		"--query",
		default="What is the roadmap for mastering DSA?",
		help="User query to run through the graph",
	)
	parser.add_argument(
		"--pdf",
		default=str(Path("data") / "COMPLETE_DSA_MASTERY_ROADMAP_FORMATTED.pdf"),
		help="Path to PDF file used for indexing",
	)
	args = parser.parse_args()

	ctx = AppContext()
	ctx.ensure_pdf_indexed(args.pdf)
	graph = build_graph(ctx)

	initial_state: State = {
		"query": args.query,
		"answer": "",
		"response": "",
		"tool_calls": False,
		"suggestion": [],
	}
	output = graph.invoke(initial_state)

	final_output = {
		"response": output.get("response", ""),
		"suggestion": output.get("suggestion", []),
	}
	print(json.dumps(final_output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
	main()
