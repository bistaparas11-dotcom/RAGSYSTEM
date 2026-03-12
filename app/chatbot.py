"""
Interactive RAG chatbot over SEC 10-K filings.

Usage:
    python -m app.chatbot

Commands:
    /reset           — clear conversation history
    /mode <m>        — switch retrieval mode: local | global | hybrid
    /filing <id>     — restrict to a specific filing_id
    /compare         — guided company comparison
    /risks <id>      — risk summary for a filing
    /financials <id> — financial metrics for a filing
    /filings         — list all available filings
    /help            — show this help
    /quit            — exit
"""

import sys
from app.llm.fin_rag_engine import FinRAGEngine
from app.data.retrieval.graph_retriever import GraphRetriever


BANNER = """
|==========================================================|
|          FinRAG: SEC 10-K Financial RAG Chat             |
|      LightRAG implementation using Neo4j + Weaviate      |
|                      Model: Groq                         |
|==========================================================|
Type /help for commands, /quit to exit.
"""


def print_response(resp):
    print(f"\n{'─'*60}")
    print(resp.answer)
    print(f"\n Sources: {', '.join(resp.source_filings) or 'N/A'}")
    print(f" [Mode]: {resp.retrieval_mode}")
    print(f"{'─'*60}\n")


def list_filings(graph: GraphRetriever):
    docs = graph.list_documents()
    if not docs:
        print("No filings found in the knowledge graph.")
        return
    print(f"\n{'─'*60}")
    print(f"{'Company':<20} {'Filing ID':<30} {'Year':<6}")
    print(f"{'─'*20} {'─'*30} {'─'*6}")
    for d in docs:
        print(f"{d.get('company_name','?'):<20} {d.get('filing_id','?'):<30} {d.get('fiscal_year','?'):<6}")
    print(f"{'─'*60}\n")


def main():
    print(BANNER)

    engine = FinRAGEngine()
    graph = GraphRetriever()

    mode = "hybrid"
    filing_id = None

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd == "/quit":
                    print("Goodbye!")
                    break

                elif cmd == "/help":
                    print(__doc__)

                elif cmd == "/reset":
                    engine.reset_history()
                    print("Conversation history cleared.")

                elif cmd == "/mode":
                    if arg in ("local", "global", "hybrid"):
                        mode = arg
                        print(f"Retrieval mode set to: {mode}")
                    else:
                        print("[!] Valid modes: local | global | hybrid")

                elif cmd == "/filing":
                    filing_id = arg if arg else None
                    print(f"Filing filter: {filing_id or 'None (all filings)'}")

                elif cmd == "/filings":
                    list_filings(graph)

                elif cmd == "/compare":
                    companies = input("Companies (comma-separated): ").strip().split(",")
                    topic     = input("Topic: ").strip()
                    companies = [c.strip() for c in companies if c.strip()]
                    if companies and topic:
                        print("\n Comparing...")
                        resp = engine.compare_companies(companies, topic)
                        print_response(resp)
                    else:
                        print("[ERROR] Please provide at least one company and a topic.")

                elif cmd == "/risks":
                    fid = arg or filing_id
                    if not fid:
                        fid = input("Filing ID: ").strip()
                    print(f"\n Retrieving risks for {fid}...")
                    resp = engine.summarise_risks(fid)
                    print_response(resp)

                elif cmd == "/financials":
                    fid = arg or filing_id
                    if not fid:
                        fid = input("Filing ID: ").strip()
                    print(f"\n Extracting financials for {fid}...")
                    resp = engine.extract_financials(fid)
                    print_response(resp)

                else:
                    print(f"[ERROR] Unknown command: {cmd}. Type /help for options.")

                continue

            # Normal question
            print("\n Thinking...\n")
            resp = engine.ask(
                question=user_input,
                mode=mode,
                filing_id=filing_id,
            )
            print_response(resp)

    finally:
        engine.close()
        graph.close()


if __name__ == "__main__":
    main()