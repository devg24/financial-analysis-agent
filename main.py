import os

from dotenv import load_dotenv
import langchain
from langchain_core.messages import HumanMessage

from config import Settings
from graph_builder import build_financial_graph
from runner import create_llm, run_financial_query

load_dotenv()
langchain.debug = os.getenv("LANGCHAIN_DEBUG", "").lower() in ("1", "true", "yes")


def main():
    settings = Settings()
    llm = create_llm(settings)

    print("--- Multi-Agent System + Summarizer Initialized ---")
    print("Type 'exit' to quit.\n")

    compiled = build_financial_graph(llm)

    while True:
        user_query = input("\nAsk about a stock: ")
        if user_query.lower() == "exit":
            break

        print("\n--- Agent Workflow Started ---")
        result = run_financial_query(compiled, user_query)
        for step in result["steps"]:
            print(f"\n[{step['node']}]: {step['content']}")
        if result.get("memo") is not None:
            print("\n--- Investment Memo ---")
            print(result["memo"])
        print("\n--- Workflow Complete ---")


if __name__ == "__main__":
    main()
