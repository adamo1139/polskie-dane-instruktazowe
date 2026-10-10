# Nauczyciel Lotnictwa 1 — plan

Szkic do dyskusji. **[decyzja]** = czeka na wybór, **[sprawdzić]** = czeka na pomiar.

## Cel

Dotrenować Bielika tak, żeby jeden model umiał:

1. **pisać wiadomość użytkownika** (Magpie): po otwartej turze usera generuje polskie pytanie
   i **poprawnie zamyka turę** `<|im_end|>` — dziś Bielik często tego nie robi;
2. **rozumować po polsku** przed odpowiedzią (tryb z myśleniem);
3. **odpowiadać** po polsku, także bez rozumowania.

## Dane

[`cpral/polskie-sprawy-v4`](https://huggingface.co/datasets/cpral/polskie-sprawy-v4) — `sft.jsonl`,
jednoturowe rozmowy, czyszczenie (dawny etap 0) już zrobione.

Sprawdzone 2026-10-10 (kopia lokalna: `data/hf_polskie_sprawy_v4/sft.jsonl`, SHA256 zgodny z HF):

- **538 079 rozmów**, wszystkie `user, assistant`. Bez rozumowania 429 309 (Bielik 399 266, `private` off
  30 043), z rozumowaniem 108 770 (`private` 59 598, Mistral Small 4 31 688, Qwen-27B 13 511, GLM 3973).
- To `glimmer8_all_qwen_nothink_2026_10_09_merged` minus 478 rozmów (424 Mistrala z `on` bez rozumowania
  + 54 z tagami `<think>`/`[THINK]` w tekście) i z usuniętym rozumowaniem w 1 rozmowie `off`.
- README v4 podaje 508 514 rozmów — nieaktualne (to liczba sprzed runu Qwena bez myślenia); tokeny
  (359,6 mln APT4) są z pełnego zbioru, więc po usunięciu 478 rozmów nieznacznie mniej.
- **Wersja do treningu: `data/polskie_sprawy_v4_sys/sft.jsonl`** = v4 + oryginalny prompt systemowy Magpie
  (glimmer8) jako pierwsza wiadomość, z pola `system` w `data/magpie_glimmer8*/accepted.jsonl`, dopasowany po
  SHA256 pytania (`add_system_prompts.py`). Dopasowane 538 079/538 079; 1286 pytań wystąpiło z kilkoma
  promptami (wszystkie prawdziwe) — wzięty pierwszy w kolejności posortowanych plików.
- `prepare_polskie_sprawy_v3.py --off-prefix none` (na v4 i na wersji z systemem) przechodzi bez błędu: train 532 733
  (ON 107 709 / OFF 425 024), validation 5346 (ON 1061 / OFF 4285).

## Format (jak Poziomka v13)

Przygotowanie: [`prepare_polskie_sprawy_v3.py`](https://github.com/adamo1139/Ling-V2/blob/main/examples/sft/megatron/prepare_polskie_sprawy_v3.py)
z `--off-prefix none`, renderowanie szablonem
[`poziomka_v13_chat_template.jinja`](https://github.com/adamo1139/Ling-V2/blob/main/examples/sft/megatron/poziomka_v13_chat_template.jinja).

```
<bos><|im_start|>system\n{prompt glimmer8}<|im_end|>\n<|im_start|>user\n{pytanie}<|im_end|>\n
<|im_start|>assistant\n<think>\n{rozumowanie}\n</think>\n{odpowiedź}<|im_end|>\n     # ON
<|im_start|>assistant\n<think></think>\n{odpowiedź}<|im_end|>\n                       # OFF
```

- Tryb rozróżnia token po `<think>`: `\n` (ON) albo `<` (OFF); prefill przy inferencji: `<think>\n` / `<think></think>\n`.
- Walidacja: 1% rozmów wybranych po SHA256 id (deterministycznie), trening w 16 shardach.
- **Loss na każdym tokenie** (wszystkie role) — w szablonie `loss_roles` = `system`, `user`, `assistant`.
  Model uczy się więc trzech rzeczy generacyjnych:
  - **prompt systemowy** (temat + podtematy + persona) — z samego `<bos><|im_start|>system\n` model sam
    losuje zadanie do generacji danych;
  - **pytanie** sterowane promptem — Magpie = prompt glimmer8 (własny albo wygenerowany) + prefill
    `<|im_start|>user\n`, stop na `<|im_end|>`;
  - **rozumowanie i odpowiedź**.
- Odpowiedzi powstały bez promptu Magpie (model odpowiadający widział tylko pytanie) — w treningu asystent
  odpowiada „pod” promptem o pisaniu pytania; **[sprawdzić]** w ewaluacji odpowiedzi z innym / bez promptu.

Tokenizer: wszystkie Bieliki v3 „PL” i małe v3 (1.5B, 4.5B, PL-Minitron-7B, PL-11B) mają **ten sam
słownik co Poziomka** (APT4, 32 000; `<s>`=1, `</s>`=2, `<|im_start|>`=3, `<|im_end|>`=4 — zgodne z
`SPECIAL_IDS` w `poziomka_data.py`). `<think>` to 4 zwykłe tokeny, jak w Poziomce (parser SGLang działa).

## Pilotaż: Bielik-1.5B-v3.0-Instruct, pełny fine-tuning, Megatron, PP=8

Dane: [`cpral/nauka-lotnictwa-1`](https://huggingface.co/datasets/cpral/nauka-lotnictwa-1) (= `polskie_sprawy_v4_sys`).

### Model (z `config.json`)

| HF | Megatron |
|---|---|
| `LlamaForCausalLM`, 32 warstwy, hidden 1536 | `--num-layers 32 --hidden-size 1536` |
| 12 głów, 2 KV, head_dim 128 | `--num-attention-heads 12 --num-query-groups 2 --group-query-attention --kv-channels 128` |
| intermediate 8960, SiLU (gate/up/down) | `--ffn-hidden-size 8960 --swiglu` (fc1 = [gate; up]) |
| **biasy we wszystkich liniowych** (`attention_bias`, `mlp_bias`: q/k/v/o, gate/up/down) | **bez** `--disable-bias-linear` (domyślne biasy Megatrona = qkv, proj, fc1, fc2) |
| RMSNorm eps 1e-6 | `--normalization RMSNorm --norm-epsilon 1e-6` |
| RoPE θ 1e6, pełny head_dim, brak skalowania | `--position-embedding-type rope --rotary-base 1000000 --rotary-percent 1.0` |
| `max_position_embeddings` 8192 | `--max-position-embeddings 8192`, `SEQ_LENGTH=8192` |
| embeddingi nie współdzielone, vocab 32 000 | `--untie-embeddings-and-output-weights --vocab-size 32000` |
| bez qk-norm, bez MoE | bez `--qk-layernorm`, bez argumentów MoE |

Wymiary jak Qwen2.5-1.5B (poza 32 warstwami zamiast 28 i dodatkowymi biasami o/MLP).

### Równoległość

- `--pipeline-model-parallel-size 8 --tensor-model-parallel-size 1`, DP 1 → **4 warstwy na etap**;
  embedding na etapie 0, `lm_head` na etapie 7 (vocab 32 000, więc etapy prawie równe).
- Bańka pipeline'u: przy micro-batch 1 i GBS 16 → 16 mikrobatchy, (PP−1)/(m+PP−1) = 7/23 ≈ 30% (jak run 10).
  Przy 1.5B każdy etap liczy bardzo mało, więc narzut komunikacji będzie względnie większy niż przy Poziomce.
- Pamięć: pełny FT ~16 B/parametr ≈ 24 GB łącznie, przy PP 8 ~3 GB/kartę + aktywacje — duży zapas.

### Co trzeba przygotować

1. **Konwerter HF → DCP** — gotowy w `megatron/` (na bazie `tools/convert_hf_to_dcp.sh` +
   `load_hf_save_dcp.py` z Ling-V2): `convert_bielik_hf_to_dcp.sh` + `load_hf_llama_save_dcp.py`.
   q/k/v (wagi i biasy) przeplatane per grupa zapytań, fc1 = [gate; up] (wagi i biasy), sprawdzanie
   `config.json` (m.in. `attention_bias`/`mlp_bias`, `head_dim`, `rope_theta`), a na każdym etapie PP —
   że **każdy parametr został wczytany i każdy tensor HF użyty**. **[sprawdzić]** na riggu: uruchomienie
   + parity logitów HF vs Megatron.
2. **`megatron/bielik_1p5b_model_args.sh`** — gotowy (argumenty z tabeli). Zakładamy, że `model_provider`
   z Ling-patched `pretrain_gpt.py` buduje gęsty model z biasami; konwerter i tak przerwie, jeśli
   któregoś parametru (np. biasu) nie będzie albo kształt się nie zgodzi.
3. **Cache danych:** `prepare_poziomka_sft.py` na `nauka-lotnictwa-1` po `prepare_polskie_sprawy_v3.py
   --off-prefix none`, tokenizer Bielika 1.5B, szablon v13, `LOSS_ROLES = ("all",)` (domyślne),
   **[decyzja]** długość: `config.json` Bielika 1.5B deklaruje `max_position_embeddings` 8192 (karta modelu
   o kontekście milczy; 11B v3 ma 32768). θ 1e6 przekracza próg 6,4e5 dla 32k (Xu i in.), więc 16k/32k
   jest możliwe, ale model raczej nie widział dłuższych sekwencji niż 8192. Danych to prawie nie dotyczy:
   tylko ~120 rozmów (+ prompt systemowy) przekracza 8192 tokenów — za długie tracą rozumowanie
   (`remove_reasoning_to_fit`).
4. **Skrypt uruchomienia** na wzór `run_poziomka_sft_run10.sh`: pakowanie (thd, micro-batch 1),
   GBS 16, warmup 20, zapis synchroniczny, `TRAIN_ITERS` z liczby kubełków po spakowaniu.
   **LR 5e-5, stały (constant scheduler)**, warmup jak w run 10.
5. **Eksport DCP → HF** z `chat_template.jinja` = v13 i test w SGLang (parser rozumowania).

## Ewaluacja

| co | jak | cel |
|---|---|---|
| zamykanie tury usera | odsetek generacji z prefillu `<|im_start|>user\n` zakończonych `<|im_end|>` | ≫ bazowy Bielik **[sprawdzić]** wynik bazowy |
| jakość pytań | `magpie.verify` (basal, próg 0.7) + deduplikacja | ≥ ~90% ok (jak Glimmer) |
| język rozumowania | OpenLID, ≥ 90% polskiego | > 95% rozmów |
| pętle / ucięcia | odsetek `length`, heurystyka pętli | < 5% |
| przełączanie trybu | prefill ON → jest rozumowanie; OFF → brak | ~100% |
| jakość odpowiedzi | porównanie z bazowym Bielikiem (sędzia LLM / ręczna próbka) | nie gorzej |

## Otwarte pytania

1. Eksport DCP → HF dla Llamy z biasami (odpowiednik `convert_dcp_to_safetensors_apt4.py`).
2. Długość sekwencji pilotażu (8192 z configu czy 16k/32k) i liczba kroków (cała epoka czy część).
3. Wieloturowość — na razie brak w danych.
