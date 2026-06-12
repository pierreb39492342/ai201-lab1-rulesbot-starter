"""
app.py
======

Generation + interface layer for "The Unofficial Guide" RAG project.

This script wires together the end-to-end RAG loop:

    Query -> retrieve() [store.py] -> grounded prompt -> Groq LLM -> Gradio UI

It:
    1. Initializes a Groq client (model: llama-3.3-70b-versatile) using
       GROQ_API_KEY from the environment.
    2. Defines `generate_answer(query, retrieved_chunks)`, which builds a
       strict, context-only system prompt from the retrieved chunks and
       calls the LLM.
    3. Collects the unique source files used for the answer, for citation.
    4. Launches a Gradio web UI with a question textbox, an "Ask" button,
       and two output fields: "Answer" and "Retrieved From".
"""

import os

from groq import Groq
import gradio as gr

from store import retrieve


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4  # Number of chunks to retrieve from the vector store per query

# Initialize the Groq client. The API key is read from the environment so
# it never needs to be hardcoded into the script.
_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(query, retrieved_chunks):
    """
    Build a grounded prompt from `retrieved_chunks` and ask the Groq LLM
    to answer `query` using ONLY that context.

    Prompt structure:
        - The SYSTEM message contains the strict grounding instructions
          (no outside knowledge, exact fallback phrase if the context is
          insufficient).
        - The USER message contains the retrieved context, formatted as a
          series of clearly delimited "Context Block" sections (each
          tagged with its source file and chunk index so the model can see
          where each piece of information came from), followed by the
          actual question.

    This separation -- instructions in the system message, data + question
    in the user message -- helps the model treat the context blocks as
    *evidence to consult* rather than as instructions to follow.

    Returns the LLM's raw text response (a string).
    """

    # --- Build the strict, grounded system prompt ----------------------
    system_prompt = (
        "Answer the user's question using ONLY the facts provided in the "
        "context blocks below. If the context does not contain enough "
        "information to conclusively answer the question, respond exactly "
        "with: 'I don't have enough information on that.' Do not rely on "
        "external knowledge or assume details."
    )

    # --- Format each retrieved chunk as a labeled "Context Block" -------
    # Each block is tagged with its source file and chunk index so the
    # model (and a human reading the prompt) can trace any fact back to
    # where it came from. Blocks are separated by blank lines and a
    # "---" divider for clear visual separation.
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        block = (
            f"Context Block {i} "
            f"(source_file: {chunk['source_file']}, "
            f"chunk_index: {chunk['chunk_index']}):\n"
            f"{chunk['text']}"
        )
        context_blocks.append(block)

    context_section = "\n\n---\n\n".join(context_blocks) if context_blocks else (
        "(No context blocks were retrieved.)"
    )

    # --- Assemble the user message: context first, then the question ----
    user_message = (
        f"CONTEXT:\n\n{context_section}\n\n"
        f"---\n\n"
        f"QUESTION: {query}"
    )

    # --- Call the Groq LLM ------------------------------------------------
    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# End-to-End RAG Pipeline (Query -> Search -> Prompt -> Answer)
# ---------------------------------------------------------------------------

def answer_question(query):
    """
    Full pipeline used by the Gradio UI:
        1. Retrieve the top-k most relevant chunks for `query` from the
           vector store (store.py's `retrieve()`).
        2. Generate a grounded answer from those chunks via the LLM.
        3. Collect the unique source files among the retrieved chunks for
           display as citations.

    Returns a tuple: (answer_text, sources_text)
    """

    if not query or not query.strip():
        return "Please enter a question.", ""

    retrieved_chunks = retrieve(query, k=TOP_K)

    if not retrieved_chunks:
        return "I don't have enough information on that.", "(no sources found)"

    answer = generate_answer(query, retrieved_chunks)

    # --- Collect unique source_file names, preserving first-seen order ---
    # We use a dict-as-ordered-set trick (dict.fromkeys) so the source list
    # has no duplicates but still reflects the order chunks were retrieved.
    unique_sources = list(dict.fromkeys(
        chunk["source_file"] for chunk in retrieved_chunks
    ))
    sources_text = "\n".join(f"- {source}" for source in unique_sources)

    return answer, sources_text


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="The Unofficial Guide") as demo:
    gr.Markdown(
        "# The Unofficial Guide\n"
        "Ask a question about off-campus housing near the Dania Beach "
        "campus (e.g. The Enclave). Answers are grounded strictly in "
        "retrieved student reviews, Discord logs, and forum threads."
    )

    query_input = gr.Textbox(
        label="Your Question",
        placeholder="e.g. What do students say about the commute or traffic "
                    "timings from campus to the Enclave?",
        lines=2,
    )

    ask_button = gr.Button("Ask", variant="primary")

    answer_output = gr.Textbox(label="Answer", lines=8, interactive=False)
    sources_output = gr.Textbox(label="Retrieved From", lines=4, interactive=False)

    ask_button.click(
        fn=answer_question,
        inputs=query_input,
        outputs=[answer_output, sources_output],
    )

    # Also allow pressing Enter in the textbox to trigger the same action.
    query_input.submit(
        fn=answer_question,
        inputs=query_input,
        outputs=[answer_output, sources_output],
    )


if __name__ == "__main__":
    demo.launch()
