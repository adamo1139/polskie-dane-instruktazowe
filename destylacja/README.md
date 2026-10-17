# Destylacja off-policy: top-k logitów nauczyciela

Off-policy KD Poziomki (ucznia) od instruktora (Bielik-PL-Minitron-7B po nauka-lotnictwa-2):
nauczyciel raz czyta gotowe rozmowy z cache SFT i zapisuje swoje k najbardziej prawdopodobnych
tokenów na każdej pozycji; uczeń trenuje potem na tych samych tokenach z lossem do tego rozkładu.
Nauczyciel nie musi być w pamięci podczas treningu ucznia.

Tu (polskie-dane-instruktazowe/destylacja) jest generowanie logitów. Trening ucznia jest w osobnym
repo **adamo1139/poziomka-destylacja** (kopia Ling-V2, żeby nie ruszać pretreningów i SFT).

```
sft.jsonl ──build cache──▶ cache SFT (poziomka-sft-v2) ──teacher_topk.py──▶ top-k nauczyciela
                                   │                                               │
                                   └──────────── train_poziomka_kd.py (uczeń) ◀────┘
```

## Pliki

| plik | co robi |
|---|---|
| `teacher_topk.py` | top-k nauczyciela dla każdego tokenu cache; backend `hf` (transformers) albo `sglang` |
| `run_teacher_topk.sh` | 8 procesów (1 na GPU), postęp i ETA w jednej linii, na końcu `--finalize` |
| `subset_cache.py` | pod-cache z części shardów (symlinki, bez kopiowania), np. losowe 25% |

## 1. Cache SFT

Logity liczy się na **tym samym cache**, na którym potem trenuje uczeń — wiersze wyniku są
wyrównane z tokenami cache, więc nie ma ponownej tokenizacji. Format `poziomka-sft-v2`
(`prepare_poziomka_sft.py` z Ling-V2), szablon v13, tokenizer APT4.

- z promptem systemowym (np. destylowanie-1): `nauczyciel_lotnictwa_1/megatron/build_nauka_lotnictwa_1_cache.sh`
  (`DATASET=… NAME=… RECORDS=… SEQ_LENGTH=…`, `BIELIK_HF=…/Ling-V2/resource/tokenizer/apt4`);
- bez systemu (polskie-sprawy-vN): `Ling-V2/examples/sft/megatron/build_polskie_sprawy_v3_cache.sh vN`
  (`WORK_DIR=/media/nvme_2tb/...`, bo na `/` riggu brakuje miejsca).

Długość: nauczyciel ma `max_position_embeddings` 32768 (rope_theta 1e6), więc cache 16384 jest OK.

Część danych: shardy train z `prepare_polskie_sprawy_v3.py` dostają rozmowy po kolei z potasowanego
sft.jsonl, więc k z 16 shardów to losowe k/16 zbioru:

```bash
python3 destylacja/subset_cache.py --cache <cache> --out <cache>-25pct --train-shards 4
```

## 2. Top-k nauczyciela

### Który backend

Zmierzone na 3090 Ti, 1000 rekordów polskie-sprawy-v8 (k=10, jedna karta):

| backend / wagi | tok/s | NLL | top-1 | masa top-10 |
|---|---|---|---|---|
| `hf`, bf16 | ~4 400 | — | — | — |
| `sglang`, bf16 | 4 280 | 0,5133 | 0,8600 | 0,9811 |
| `sglang`, **W8A8 INT8** | **7 355** | 0,5177 | 0,8589 | 0,9810 |

- Przy samym prefillu silnik nie ma znaczenia (SGLang bf16 ≈ HF bf16); całe przyspieszenie 1,7× dają
  kernele INT8 (W8A8), które ma tylko SGLang/vLLM. Koszt jakości: NLL +0,9%.
- `--chunked-prefill` 8192/16384 nie pomaga (7 206 / 7 151 tok/s) — zostaje 4096.
- Na `destylowanie-1` (k=40, HF bf16, 8 GPU): 35,5k tok/s łącznie, NLL 0,516, top-1 0,853, masa top-40 0,9845.

`hf` potrzebuje flash-attention 2 (packing przez `position_ids`); `sglang` — venv SGLang
(`~/projects/sglang`), bez flash_attn. Checkpoint W8A8 w `hf` jest rozpakowywany do bf16
(`CompressedTensorsConfig(run_compressed=False)`) — działa, ale bez zysku z INT8.

### Benchmark (1 GPU, bez zapisu)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
python3 destylacja/teacher_topk.py --backend sglang \
  --cache <cache> --model <instruktor W8A8> --out /tmp/x --k 10 --benchmark 1000
```

Czas pełnego przebiegu ≈ pozycje / (8 × tok/s z benchmarku). Miejsce: (4k + 2) B na pozycję
(k=10: 42 B, k=40: 162 B) — dla v8 (~1,31 mld pozycji) k=10 to ~55 GB.

### Pełny przebieg

```bash
CACHE=<cache> TEACHER_HF=<model> OUT=/media/nvme_2tb/maked/bielik_train/teacher-topk-<nazwa>-k10 \
K=10 BACKEND=sglang PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash destylacja/run_teacher_topk.sh
```

- Wszystkie 8 kart muszą być wolne (wyłącz serwer SGLang).
- **OUT na nvme**, nie na `/`.
- Wznowienie: ta sama komenda — kawałki z plikiem `.done` są pomijane. Rank, który padnie,
  zatrzymuje cały przebieg i pokazuje końcówkę swojego logu (`OUT/logs/rank*.log`).
- Na końcu `--finalize` sprawdza komplet (liczba pozycji = tokeny cache − rekordy, ten sam cache,
  k i model we wszystkich kawałkach) i pisze `manifest.json` ze statystykami.
- Ręcznie: `python3 destylacja/teacher_topk.py --cache … --out … --k 10 --world 8 --finalize`;
  pojedynczy rank: `--rank R --world 8` z `CUDA_VISIBLE_DEVICES=R`.

### Format wyniku (`teacher-topk-v1`)

Obok każdego shardu cache, te same offsety; wiersz j = rozkład nauczyciela dla tokenu j
(p(token j | wcześniejsze tokeny rekordu)); wiersz BOS = zera:

```
{prefix}.topk_ids.bin     <u2 [tokens, k]   ID top-k, malejąco
{prefix}.topk_logp.bin    <f2 [tokens, k]   log p (T=1, pełny słownik, bez renormalizacji)
{prefix}.target_logp.bin  <f2 [tokens]      log p tokenu, który faktycznie stoi w danych
manifest.json             k, model, cache_manifest_sha256, totals, shards
```

Masa ogona = 1 − Σ exp(topk_logp). W treningu `labels = ids[s+1:e]`, więc labelom odpowiadają
wiersze `s+1 … e−1`.

Na HF wrzucamy tylko `manifest.json` i `*.bin` — `*.done` i `logs/` to pliki robocze
(np. cpral/nauczyciel-topk-destylowanie-1-k40).

### Kontrole, które zatrzymują przebieg

- `hf`: test pakowania na pierwszej paczce — paczka vs osobne forwardy, porównana ze wzorcem przecieku
  (ciągłe `position_ids`). Szum bf16 to |Δ| ~0,02; przeciek daje wielokrotnie więcej.
- `sglang`: dla każdego rekordu liczba pozycji, pełne top-k i **ID tokenów w logprobach = tokeny cache**.
- oba: vocab 32000 i tokenizer modelu = tokenizer cache.

## 3. Trening ucznia (poziomka-destylacja)

`examples/sft/megatron/` w adamo1139/poziomka-destylacja:

- `poziomka_kd_loss.py` — reverse KL(uczeń ‖ nauczyciel) na top-k + jeden kubełek ogona; gradient
  liczony analitycznie kawałkami (bez pełnych logitów fp32). Kubełkowy KL ≤ prawdziwy KL.
- `poziomka_kd_data.py` — `MMapKDDataset`: dataset SFT + wiersze nauczyciela dla każdego okna
  (packing bez zmian); odrzuca top-k policzone dla innego cache (sha256 manifestu).
- `train_poziomka_kd.py` — packed forward bez labeli (ostatni stage dostaje logity), loss KD;
  na starcie sprawdza, że ≥ 90% labeli jest w top-k nauczyciela (~98,6% na destylowanie-1).
- `run_poziomka_kd_run14.sh` — wzór runu: start z checkpointu ucznia (run13), PP8, GBS 64.
- testy CPU: `test_poziomka_kd_loss.py`, `test_poziomka_kd_data.py`.

Metryki w logu: `kd loss` (reverse KL), `student ce` (zwykłe CE, porównywalne z `lm loss` SFT),
`student tail` (masa ucznia poza top-k nauczyciela — nie powinna spadać dużo poniżej masy ogona
nauczyciela). Przy reverse KL `student ce` może lekko rosnąć: uczeń skupia się na głównych trybach
nauczyciela (run14: kd loss 0,45 → 0,33, student ce 0,845 → 0,89, tail 4,5% → 2,7%).

## Pułapki, na które już trafiliśmy

- **Brak miejsca na `/`** (rigg): wyniki, cache i `WANDB_DIR` trzymaj na `/media/nvme_2tb`.
- **SGLang raportuje ostatni token słownika (APT4 31999 = `°`) jako 0** — błąd o jeden w
  `logprob_result_processor.py` (`x < vocab_size - 1`); dotyczy tylko raportu, `teacher_topk.py` to uwzględnia.
- **SGLang OOM w logits_processor**: przy `logprob_start_len=0` logity całego kawałka prefillu są
  w pamięci. `mem_fraction_static` to wagi + KV — logity mieszczą się w *reszcie*, więc przy OOM
  zmniejsz `--mem-fraction` (domyślnie 0,5) albo `--chunked-prefill` (4096), nie zwiększaj.
- **`sgl.Engine` z `python3 - <<EOF` nie działa** (procesy `spawn` muszą zaimportować plik).
- **flash-attn-4 w venvie SGLang nie działa na 3090 Ti** i siedzi pod tym samym modułem
  `flash_attn` co FA2 — FA2 instaluj w osobnym venvie.
