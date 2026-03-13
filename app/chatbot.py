"""
Interactive RAG chatbot over SEC 10-K filings.

Usage:
    python -m app.chatbot

Commands:
    /reset                — clear conversation history
    /mode <m>             — switch retrieval mode: local | global | hybrid
    /filing <id/name>     — restrict to a specific filing_id
    /compare              — guided company comparison
    /risks <id/name>      — risk summary for a filing
    /financials <id/name> — financial metrics for a filing
    /filings              — list all available filings
    /help                 — show this help
    /quit                 — exit
"""

from app.llm.fin_rag_engine import FinRAGEngine


BANNER = """
|==========================================================|
|          FinRAG: SEC 10-K Financial RAG Chat             |
|      Lightrag -> Neo4j + Weaviate + BGE-M3 + Groq        |
|==========================================================|
Type /help for commands, /quit to exit.
"""


def print_response(resp):
    print(f"\n{'─'*60}")
    print(resp.answer)
    print(f"\n Sources: {', '.join(resp.source_filings) or 'N/A'}")
    print(f" [Mode]: {resp.retrieval_mode}")
    print(f"{'─'*60}\n")


def list_filings(engine: FinRAGEngine):
    docs = engine.resolver.list_all()
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
    mode = "hybrid"
    filing_id = None

    try:
        while True:
            try:
                raw = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not raw:
                continue

            # Commands
            if raw.startswith("/"):
                parts = raw.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd == "/quit":
                    print("Goodbye!")
                    break

                elif cmd == "/help":
                    print(__doc__)

                elif cmd == "/reset":
                    engine.reset_history()
                    print("[OK] Conversation history cleared.")

                elif cmd == "/mode":
                    if arg in ("local", "global", "hybrid"):
                        mode = arg
                        print(f"[OK] Retrieval mode set to: {mode}")
                    else:
                        print("[!] Valid modes: local | global | hybrid")

                elif cmd == "/filing":
                    if arg:
                        filing_ref = arg
                        resolved = engine.resolver.resolve(arg)
                        if resolved:
                            print(f"[OK] Filing filter: {resolved}")
                        else:
                            print(f"[!] {arg} not found in knowledge graph.")
                            filing_ref = None

                elif cmd == "/filings":
                    list_filings(engine)
                    
                elif cmd == "/compare":
                    companies_raw = input("Companies (comma-separated): ").strip()
                    topic = input("Topic: ").strip()
                    companies = [c.strip() for c in companies_raw.split(",") if c.strip()]
                    if companies and topic:
                        print("\n Comparing...")
                        resp = engine.compare_companies(companies, topic)
                        print_response(resp)
                    else:
                        print("[ERROR] Please provide at least one company and a topic.")

                elif cmd == "/risks":
                    ref = arg or filing_id
                    if not ref:
                        ref = input("Filing ID: ").strip()
                    print(f"\n Retrieving risks for {ref}...")
                    resp = engine.summarise_risks(ref)
                    print_response(resp)

                elif cmd == "/financials":
                    ref = arg or filing_id
                    if not ref:
                        ref = input("Filing ID: ").strip()
                    print(f"\n Extracting financials for {ref}...")
                    resp = engine.extract_financials(ref)
                    print_response(resp)

                else:
                    print(f"[ERROR] Unknown command: {cmd}. Type /help for options.")

                continue

            # Normal question
            print("\n Thinking...\n")
            resp = engine.ask(
                question=raw,
                mode=mode,
                filing_id=filing_id,
            )
            print_response(resp)

    finally:
        engine.close()


if __name__ == "__main__":
    main()