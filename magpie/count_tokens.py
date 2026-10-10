"""Liczy tokeny w pliku sft.jsonl tokenizerem trenowanego modelu (domyślnie APT4 z Poziomki).

Liczymy samą treść — pytanie, odpowiedź, rozumowanie i ich sumę — bez narzutu szablonu czatu
(BOS, znaczniki ról). Wymaga extra „tokens”: uv sync --extra tokens
"""
import argparse
import os
import json
import statistics as st

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
from transformers import AutoTokenizer  # noqa: E402


def lengths(tok, texts: list[str], chunk: int = 10000) -> list[int]:
    # Szybki tokenizer (Rust) koduje paczkę równolegle na wszystkich rdzeniach; paczkami, żeby nie trzymać
    # w pamięci ID tokenów całego zbioru naraz.
    out = []
    for i in range(0, len(texts), chunk):
        out += [len(x) for x in tok(texts[i:i + chunk], add_special_tokens=False)["input_ids"]]
    return out


def show(name: str, v: list[int]) -> None:
    q = st.quantiles(v, n=100)
    print(f"{name:26} suma {sum(v):>12,}  śr. {st.mean(v):7.0f}  mediana {st.median(v):6.0f}  "
          f"p95 {q[94]:6.0f}  p99 {q[98]:6.0f}  max {max(v):6}".replace(",", " "))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="plik sft.jsonl")
    p.add_argument("--tokenizer", default="cpral/poziomka-instruct-2026-10-01")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.input, encoding="utf-8") as f:
        convs = [json.loads(line)["messages"] for line in f]
    role = lambda c, r, key="content": next((m.get(key) or "" for m in c if m["role"] == r), "")

    print(f"rozmów: {len(convs)}   tokenizer: {args.tokenizer} (słownik {len(tok)})\n")
    prompt = lengths(tok, [role(c, "user") for c in convs])
    answer = lengths(tok, [role(c, "assistant") for c in convs])
    show("pytanie", prompt)
    show("odpowiedź", answer)
    reasoning = [role(c, "assistant", "reasoning_content") for c in convs]
    thought = lengths(tok, reasoning) if any(reasoning) else [0] * len(convs)
    if any(reasoning):
        show("rozumowanie", thought)
    full = [a + b + c for a, b, c in zip(prompt, answer, thought)]
    show("razem", full)
    for lim in (2048, 4096, 8192):
        print(f"  rozmów > {lim} tokenów: {sum(x > lim for x in full)}")


if __name__ == "__main__":
    main()
