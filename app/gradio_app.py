"""
FinSight Gradio UI — web interface for the FinRAG chatbot.

Run:
    python -m app.gradio_app
    # or
    gradio app/gradio_app.py

Features:
  - Chat interface with conversation history
  - Retrieval mode selector (local / global / hybrid)
  - Filing filter with live dropdown from Neo4j
  - Source citations shown per response
  - Sidebar commands: /risks, /financials, /compare
  - Reset conversation button
"""

import gradio as gr
from typing import List, Tuple, Optional

from app.llm.fin_rag_engine import FinRAGEngine
from app.data.retrieval.filing_resolver import FilingResolver


# ── Global engine (initialised once) ────────────────────────────────── #

engine   = FinRAGEngine()
resolver = FilingResolver()


def get_filing_choices() -> List[str]:
    """Load all available filings for the dropdown."""
    docs = resolver.list_all()
    choices = ["All filings"]
    for d in docs:
        label = f"{d.get('company_name', '?')}  —  {d.get('filing_id', '?')}  ({d.get('fiscal_year', '?')})"
        choices.append(label)
    return choices


def extract_filing_id(choice: str) -> Optional[str]:
    """Parse filing_id out of the dropdown label."""
    if not choice or choice == "All filings":
        return None
    # label format: "NVIDIA  —  NVDA10K2024.md  (2024)"
    parts = choice.split("—")
    if len(parts) >= 2:
        return parts[1].strip().split(" ")[0].strip()
    return None


# ── Core chat function ───────────────────────────────────────────────── #

def chat(
    message:      str,
    history:      List[dict],
    mode:         str,
    filing_choice: str,
    top_k:        int,
) -> Tuple[List[dict], str]:
    """
    Main chat handler called by Gradio.
    Returns (updated_history, sources_text).
    """
    if not message.strip():
        return history, ""

    filing_id = extract_filing_id(filing_choice)

    # Handle slash commands
    if message.strip().startswith("/"):
        parts = message.strip().split(maxsplit=1)
        cmd   = parts[0].lower()
        arg   = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/reset":
            engine.reset_history()
            history = []
            return history, "[OK] Conversation reset."

        elif cmd == "/risks":
            ref = arg or filing_id
            if not ref:
                reply = "[X] Please specify a company: `/risks NVIDIA` or select a filing from the dropdown."
                history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
                return history, ""
            resp = engine.summarise_risks(ref)
            sources = ", ".join(resp.source_filings) or "N/A"
            history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": resp.answer}])
            return history, f" {sources}"

        elif cmd == "/financials":
            ref = arg or filing_id
            if not ref:
                reply = "[X] Please specify a company: `/financials MSFT` or select a filing."
                history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
                return history, ""
            resp = engine.extract_financials(ref)
            sources = ", ".join(resp.source_filings) or "N/A"
            history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": resp.answer}])
            return history, f" {sources}"

        elif cmd == "/compare":
            if not arg:
                reply = "[X] Usage: `/compare NVIDIA, Microsoft | AI revenue`"
                history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
                return history, ""
            # Parse: "NVIDIA, Microsoft | topic"
            if "|" in arg:
                companies_raw, topic = arg.split("|", 1)
            else:
                companies_raw = arg
                topic = "overall business performance"
            companies = [c.strip() for c in companies_raw.split(",") if c.strip()]
            resp = engine.compare_companies(companies, topic.strip())
            sources = ", ".join(resp.source_filings) or "N/A"
            history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": resp.answer}])
            return history, f" {sources}"

        elif cmd == "/help":
            help_text = (
                "**Available commands:**\n\n"
                "- `/reset` — clear conversation history\n"
                "- `/risks <company>` — risk summary (e.g. `/risks NVIDIA`)\n"
                "- `/financials <company>` — key financials (e.g. `/financials MSFT`)\n"
                "- `/compare <co1>, <co2> | <topic>` — compare companies\n\n"
                "Or just ask naturally: *What are Google's main revenue segments?*"
            )
            history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": help_text}])
            return history, ""

    # Normal question
    resp = engine.ask(
        question=message,
        mode=mode.lower(),
        filing_id=filing_id,
        top_k=top_k,
        rerank_top_k=5,
    )

    sources = ", ".join(resp.source_filings) if resp.source_filings else "N/A"
    history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": resp.answer}])
    return history, f" Sources: {sources}  |  🔍 Mode: {resp.retrieval_mode}"


def reset_chat() -> Tuple[List, str, str]:
    engine.reset_history()
    return [], "", "[OK] Conversation cleared."


# ── Gradio UI ────────────────────────────────────────────────────────── #

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap');

body, .gradio-container {
    font-family: 'DM Mono', monospace !important;
    background: #0d0f12 !important;
}

/* Header */
.fin-header {
    background: linear-gradient(135deg, #0d0f12 0%, #111419 100%);
    border-bottom: 1px solid #1e2530;
    padding: 20px 28px 16px;
    margin-bottom: 0;
}
.fin-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 28px !important;
    color: #e8dfc8 !important;
    letter-spacing: -0.5px;
    margin: 0 0 4px 0;
}
.fin-subtitle {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: #4a5568 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Chatbot */
.chatbot-wrap .wrap {
    background: #0d0f12 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 8px !important;
}
.chatbot-wrap .message.user {
    background: #141820 !important;
    border: 1px solid #1e2530 !important;
    color: #c8d0dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    border-radius: 6px 6px 2px 6px !important;
}
.chatbot-wrap .message.bot {
    background: #0f1318 !important;
    border: 1px solid #1a2235 !important;
    color: #d4c9a8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    border-radius: 6px 6px 6px 2px !important;
}

/* Input */
.input-row textarea {
    background: #111419 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 6px !important;
    color: #c8d0dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    resize: none !important;
}
.input-row textarea:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.1) !important;
}

/* Buttons */
.send-btn {
    background: #c9a84c !important;
    color: #0d0f12 !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 6px !important;
    min-width: 80px !important;
}
.send-btn:hover {
    background: #dbb855 !important;
}
.reset-btn {
    background: transparent !important;
    color: #4a5568 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    border: 1px solid #1e2530 !important;
    border-radius: 6px !important;
}
.reset-btn:hover {
    border-color: #4a5568 !important;
    color: #6b7a8d !important;
}

/* Sidebar controls */
.sidebar {
    background: #0f1318 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
.sidebar label {
    color: #4a5568 !important;
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}
.sidebar .wrap {
    background: #111419 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 4px !important;
    color: #c8d0dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}
.sidebar input[type=range] {
    accent-color: #c9a84c !important;
}

/* Sources bar */
.sources-bar {
    background: #0f1318 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 4px !important;
    color: #4a6080 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    padding: 8px 12px !important;
}

/* Section labels */
.section-label {
    color: #2a3545 !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    margin-bottom: 8px !important;
    font-family: 'DM Mono', monospace !important;
}

/* Command pills */
.cmd-pill {
    display: inline-block;
    background: #141820;
    border: 1px solid #1e2530;
    color: #c9a84c;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 3px;
    margin: 2px;
    font-family: 'DM Mono', monospace;
}

/* Hide Gradio footer */
footer { display: none !important; }
.built-with { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f12; }
::-webkit-scrollbar-thumb { background: #1e2530; border-radius: 2px; }
"""

COMMANDS_MD = """
<div style="font-family:'DM Mono',monospace;font-size:11px;color:#4a5568;line-height:2">
<span style="color:#2a3545;letter-spacing:1.5px;text-transform:uppercase;font-size:10px">Commands</span><br>
<span style="color:#c9a84c">/risks</span> &lt;company&gt;<br>
<span style="color:#c9a84c">/financials</span> &lt;company&gt;<br>
<span style="color:#c9a84c">/compare</span> A, B | topic<br>
<span style="color:#c9a84c">/reset</span><br>
<span style="color:#c9a84c">/help</span>
</div>
"""


def build_ui():
    filing_choices = get_filing_choices()

    with gr.Blocks(
        title="FinSight",
    ) as demo:

        # ── Header ──────────────────────────────────────────────────── #
        gr.HTML("""
        <div class="fin-header">
            <div class="fin-title">FinSight</div>
            <div class="fin-subtitle">SEC 10-K · LightRAG · Neo4j · Weaviate · DeepSeek-R1 · Groq</div>
        </div>
        """)

        with gr.Row(equal_height=True):

            # ── Main chat panel ──────────────────────────────────────── #
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="",
                    height=520,
                    show_label=False,
                    elem_classes=["chatbot-wrap"],
                    render_markdown=True,
                    avatar_images=(None, None),
                    placeholder=(
                        "<div style='text-align:center;color:#1e2530;"
                        "font-family:DM Mono,monospace;font-size:12px;"
                        "padding:60px 20px'>"
                        "Ask about any SEC 10-K filing<br>"
                        "<span style='font-size:10px;letter-spacing:1px;"
                        "text-transform:uppercase;color:#151c28'>"
                        "or type /help for commands</span>"
                        "</div>"
                    ),
                )

                sources_display = gr.Textbox(
                    label="",
                    placeholder="Sources will appear here after each response",
                    interactive=False,
                    show_label=False,
                    elem_classes=["sources-bar"],
                    lines=1,
                )

                with gr.Row(elem_classes=["input-row"]):
                    msg_input = gr.Textbox(
                        placeholder="Ask a question or type /help for commands...",
                        show_label=False,
                        scale=5,
                        lines=1,
                        max_lines=4,
                        autofocus=True,
                    )
                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1,
                        elem_classes=["send-btn"],
                    )

            # ── Sidebar ──────────────────────────────────────────────── #
            with gr.Column(scale=1, elem_classes=["sidebar"]):

                gr.HTML("<div class='section-label'>Retrieval mode</div>")
                mode_radio = gr.Radio(
                    choices=["Hybrid", "Local", "Global"],
                    value="Hybrid",
                    show_label=False,
                )

                gr.HTML("<div class='section-label' style='margin-top:16px'>Filing filter</div>")
                filing_dropdown = gr.Dropdown(
                    choices=filing_choices,
                    value="All filings",
                    show_label=False,
                    allow_custom_value=False,
                )

                gr.HTML("<div class='section-label' style='margin-top:16px'>Chunks retrieved</div>")
                top_k_slider = gr.Slider(
                    minimum=4,
                    maximum=20,
                    value=10,
                    step=2,
                    show_label=False,
                )

                gr.HTML("<div class='section-label' style='margin-top:16px'>Quick actions</div>")

                risks_btn = gr.Button("Risk summary", size="sm", variant="secondary")
                financials_btn = gr.Button("Key financials", size="sm", variant="secondary")

                gr.HTML("<div style='margin-top:16px'>" + COMMANDS_MD + "</div>")

                reset_btn = gr.Button(
                    "↺  Reset conversation",
                    size="sm",
                    elem_classes=["reset-btn"],
                )

        # ── Event handlers ───────────────────────────────────────────── #

        def submit(message, history, mode, filing, top_k):
            new_history, sources = chat(message, history, mode, filing, top_k)
            return new_history, sources, ""   # clear input after send

        # Send on button click
        send_btn.click(
            fn=submit,
            inputs=[msg_input, chatbot, mode_radio, filing_dropdown, top_k_slider],
            outputs=[chatbot, sources_display, msg_input],
        )

        # Send on Enter (Shift+Enter for newline)
        msg_input.submit(
            fn=submit,
            inputs=[msg_input, chatbot, mode_radio, filing_dropdown, top_k_slider],
            outputs=[chatbot, sources_display, msg_input],
        )

        # Quick action: risks button
        def quick_risks(history, filing):
            fid = extract_filing_id(filing)
            if not fid:
                msg = "Please select a filing from the dropdown first."
                return history + [{"role": "user", "content": "/risks"}, {"role": "assistant", "content": msg}], ""
            new_h, src = chat(f"/risks {fid}", history, "Local", filing, 10)
            return new_h, src

        risks_btn.click(
            fn=quick_risks,
            inputs=[chatbot, filing_dropdown],
            outputs=[chatbot, sources_display],
        )

        # Quick action: financials button
        def quick_financials(history, filing):
            fid = extract_filing_id(filing)
            if not fid:
                msg = "Please select a filing from the dropdown first."
                return history + [{"role": "user", "content": "/financials"}, {"role": "assistant", "content": msg}], ""
            new_h, src = chat(f"/financials {fid}", history, "Local", filing, 10)
            return new_h, src

        financials_btn.click(
            fn=quick_financials,
            inputs=[chatbot, filing_dropdown],
            outputs=[chatbot, sources_display],
        )

        # Reset
        reset_btn.click(
            fn=reset_chat,
            outputs=[chatbot, sources_display, sources_display],
        )

    return demo


# ── Launch ───────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,        # set True to get a public gradio.live URL
        show_error=True,
        theme=gr.themes.Base(
            primary_hue="yellow",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("DM Mono"),
        ),
        css=CSS,
    )