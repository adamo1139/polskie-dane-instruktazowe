# polskie-dane-instruktazowe

Generacja polskich pytań użytkownika metodą Magpie na Muse Glimmer (vLLM, `/v1/completions`).
Model dostaje tylko prompt systemowy i otwartą turę użytkownika, sam dopisuje wiadomość
i zamyka turę tokenem `<|eom|>`/`<|eot|>`.

## Struktura

```
magpie/
  topics.py    16 tematów × ~40–58 podtematów (miejscownik)
  prompt.py    prompt systemowy (glimmer8), persony Polak/Polka, budowa zadań z --master-seed
  filters.py   czyszczenie (</user, „Pytanie:”) i klasyfikacja (echo, przecieki, reguły jakości)
  generate.py  asynchroniczny generator (aiohttp + aiolimiter), zapis co 100, wznawianie, finalize
```

## Uruchomienie

```bash
uv sync
ulimit -n 10000
.venv/bin/magpie-generate --total 50000 --concurrency 1500 --out data/magpie_glimmer8
```

- Wyniki: `raw.jsonl` (wszystko, dopisywane co `--flush-every`), po zakończeniu `accepted.jsonl`
  (OK + deduplikacja; samoodpowiedzi tylko oznaczone `self_answer_suspect`).
- Każdy nowy `--out` dostaje losowy `--master-seed` (zapisany w `<out>/meta.json`), więc różne runy
  mają różne zadania. Wznowienie: to samo polecenie — seed jest czytany z `meta.json`; niezgodny
  `--master-seed` lub `--subs-min/--subs-max` przerywa run. Nie zmieniaj `topics.py` w trakcie runu.
- Statystyki bez generowania: `--finalize-only`.
- Timeouty: `--ttft-timeout 300`, `--chunk-timeout 120`, `--total-timeout 600` (s).
- Dobieranie do większej liczby: podnieś `--total` z tym samym `--out` — gotowe id są pomijane.

## Prompt (glimmer8, 15 podtematów)

> Rozmowa o historii Polski, na przykład o Wielkiej Emigracji, …, Jadwidze Andegaweńskiej albo
> powstaniu styczniowym. Polka pisze teraz jedno krótkie, naturalne pytanie po polsku na jeden
> z tych tematów. Jej wiadomość kończy się zaraz po tym pytaniu.

Test n=200: ~82% przyjętych, echo ~12%. Historia eksperymentów (warianty 1–10, sampling, bad_words):
`../findings.md`.
