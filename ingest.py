"""
ingest.py
=========

Document processing pipeline for "The Unofficial Guide" RAG project.

Domain: Off-Campus Housing Experiences (student reviews, Discord logs, and
forum threads about local apartments such as The Enclave near the Dania
Beach campus).

This script:
    1. Reads all .txt files from a local `data/` directory.
    2. Cleans each document (strips HTML entities, forum/navigation
       boilerplate, cookie banners, etc.) while preserving the
       substantive review content (ratings, timestamps, apartment names).
    3. Splits each cleaned document into overlapping fixed-size chunks
       using a sliding window.
    4. Validates the result by printing the total chunk count and a
       handful of random sample chunks.

Only standard library modules are used (os, re, random).
"""

import os
import re
import random


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "data"        # Directory containing the raw .txt source files
CHUNK_SIZE = 500          # Number of characters per chunk
CHUNK_OVERLAP = 100       # Number of characters shared between consecutive chunks
NUM_SAMPLES = 3           # Number of random sample chunks to display for inspection


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(raw_text):
    """
    Clean raw scraped/exported text from forum threads, Discord logs, or
    review pages.

    The goal is to strip out web/forum "noise" while keeping anything that
    looks like substantive review content -- including apartment names,
    star ratings (e.g. "4/5", "★★★★"), and timestamps
    (e.g. "[2024-01-05 14:32]" or "Posted on 03/14/2023").

    Steps performed:
        1. Decode the most common HTML entities (&amp;, &nbsp;, &quot;, etc.)
           that are often left over when raw HTML is scraped/exported as text.
        2. Remove any leftover raw HTML tags (e.g. <div>, <br/>).
        3. Remove lines that are clearly site chrome / navigation / cookie
           banners rather than user-generated content (e.g. lines that are
           just "Home | Forums | Login", or contain "Accept Cookies",
           "Privacy Policy", "Subscribe to our newsletter", etc.).
        4. Collapse excessive whitespace/blank lines so chunk boundaries
           land on meaningful text rather than runs of empty lines.

    Returns the cleaned text as a single string.
    """

    text = raw_text

    # --- 1. Decode common HTML entities -------------------------------
    # A small manual map covers the entities most commonly seen in scraped
    # forum/Discord export text. We avoid importing html.unescape so the
    # logic stays explicit and easy to extend for this project.
    html_entities = {
        "&amp;": "&",
        "&nbsp;": " ",
        "&quot;": '"',
        "&#39;": "'",
        "&apos;": "'",
        "&lt;": "<",
        "&gt;": ">",
        "&mdash;": "-",
        "&ndash;": "-",
        "&rsquo;": "'",
        "&lsquo;": "'",
        "&ldquo;": '"',
        "&rdquo;": '"',
        "&hellip;": "...",
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    # --- 2. Strip any remaining raw HTML tags --------------------------
    # Matches things like <div class="post">, </span>, <br/>, etc.
    text = re.sub(r"<[^>]+>", " ", text)

    # --- 3. Remove boilerplate / navigation / cookie-banner lines ------
    # We work line-by-line so we can selectively drop "noise" lines while
    # keeping substantive review/discussion lines intact.
    boilerplate_patterns = [
        r"^\s*(home|forums?|login|sign\s*up|register|search)\s*(\||>|/|$)",  # nav bars
        r"cookie",                       # cookie consent banners
        r"privacy policy",
        r"terms (of|and) (service|use)",
        r"all rights reserved",
        r"subscribe (to )?(our )?newsletter",
        r"^\s*(advertisement|sponsored)\s*$",
        r"click here to (continue|read more|learn more)",
        r"^\s*(share|tweet|like|follow us)\s*[:\-]?\s*$",
        r"^\s*[-=_]{3,}\s*$",             # decorative separator lines, e.g. "-----"
        r"^\s*(page \d+ of \d+)\s*$",     # pagination footers
    ]
    boilerplate_regex = re.compile(
        "|".join(boilerplate_patterns), re.IGNORECASE
    )

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()

        # Drop empty lines here; we'll re-join with controlled spacing below.
        if not stripped:
            cleaned_lines.append("")
            continue

        # Skip lines that match any boilerplate pattern.
        if boilerplate_regex.search(stripped):
            continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # --- 4. Collapse excessive whitespace -------------------------------
    # Collapse runs of 3+ blank lines into a single blank line, and collapse
    # runs of horizontal whitespace (spaces/tabs) into a single space.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text, source_file, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split `text` into overlapping fixed-size chunks using a sliding window.

    How the sliding window works:
        - We start at position `start = 0` and take a slice of `chunk_size`
          characters: text[start : start + chunk_size].
        - For the NEXT chunk, instead of starting where the previous chunk
          ended, we step backwards by `overlap` characters. This means the
          new window starts at:

              next_start = start + (chunk_size - overlap)

          Equivalently, the "step" (or "stride") of the window is
          (chunk_size - overlap). A smaller step means more overlap, which
          means more chunks but better preservation of context that would
          otherwise be split awkwardly across a chunk boundary.

        - We repeat this until `start` reaches (or exceeds) the end of the
          text.

    Why overlap matters for RAG:
        Without overlap, a sentence or review that happens to fall exactly
        on a chunk boundary gets cut in half, and neither resulting chunk
        contains the full thought. By re-including the last `overlap`
        characters of the previous chunk at the start of the next chunk,
        we increase the chance that any given piece of context appears
        whole in at least one chunk.

    Returns a list of dictionaries, each with:
        - "text":        the chunk's text content
        - "source_file": the originating file's name
        - "chunk_index": the integer position of this chunk within the
                          document (0-based, in document order)
    """

    chunks = []

    text_length = len(text)
    step = chunk_size - overlap  # how far the window slides each iteration

    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap to make progress")

    start = 0
    chunk_index = 0

    while start < text_length:
        end = start + chunk_size
        chunk_str = text[start:end].strip()

        # Avoid emitting empty/whitespace-only chunks (can happen if the
        # tail of the document is mostly whitespace after cleaning).
        if chunk_str:
            chunks.append({
                "text": chunk_str,
                "source_file": source_file,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        # Slide the window forward by `step` characters.
        start += step

        # If the remaining text is shorter than the overlap itself, the
        # next iteration would just re-produce a near-duplicate tail chunk
        # forever -- the `start < text_length` condition combined with the
        # positive `step` guarantees forward progress and termination.

    return chunks


# ---------------------------------------------------------------------------
# Ingestion Pipeline
# ---------------------------------------------------------------------------

def ingest_directory(data_dir=DATA_DIR):
    """
    Walk through `data_dir`, read every .txt file, clean its contents, and
    chunk it. Returns a flat list of chunk dictionaries across all files.
    """

    all_chunks = []

    if not os.path.isdir(data_dir):
        print(f"[WARNING] Data directory '{data_dir}' does not exist. "
              f"Create it and add .txt files to ingest.")
        return all_chunks

    for filename in sorted(os.listdir(data_dir)):
        if not filename.lower().endswith(".txt"):
            continue  # Skip non-text files for now.

        file_path = os.path.join(data_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        except OSError as exc:
            print(f"[WARNING] Could not read '{file_path}': {exc}")
            continue

        cleaned = clean_text(raw_text)

        if not cleaned:
            print(f"[INFO] '{filename}' produced no content after cleaning; skipping.")
            continue

        file_chunks = chunk_text(cleaned, source_file=filename)
        all_chunks.extend(file_chunks)

        print(f"[INFO] Processed '{filename}': "
              f"{len(raw_text)} raw chars -> {len(cleaned)} cleaned chars "
              f"-> {len(file_chunks)} chunks")

    return all_chunks


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_chunks(chunks, num_samples=NUM_SAMPLES):
    """
    Print summary statistics and a few random sample chunks so the chunks
    can be visually inspected for cleanliness and self-containment.
    """

    total = len(chunks)
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total chunks generated across dataset: {total}")

    if total == 0:
        print("No chunks to sample.")
        return

    sample_size = min(num_samples, total)
    samples = random.sample(chunks, sample_size)

    print(f"\nDisplaying {sample_size} random sample chunk(s) for inspection:\n")

    for i, chunk in enumerate(samples, start=1):
        print("-" * 70)
        print(f"SAMPLE {i}")
        print(f"  Source file : {chunk['source_file']}")
        print(f"  Chunk index : {chunk['chunk_index']}")
        print(f"  Char length : {len(chunk['text'])}")
        print("  Content:")
        print("  " + "-" * 40)
        # Indent the chunk text for readability in the terminal.
        for line in chunk["text"].splitlines():
            print(f"  {line}")
        print("  " + "-" * 40)

    print("-" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Starting ingestion pipeline (data dir: '{DATA_DIR}', "
          f"chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")

    chunks = ingest_directory(DATA_DIR)
    validate_chunks(chunks)


if __name__ == "__main__":
    main()
