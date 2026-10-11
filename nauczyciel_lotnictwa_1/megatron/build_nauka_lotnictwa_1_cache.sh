#!/usr/bin/env bash
set -euo pipefail

# Buduje cache SFT cpral/nauka-lotnictwa-1 (polskie-sprawy-v4-nc + prompt systemowy Magpie)
# w formacie v13 dla Bielika 1.5B. Na wzór Ling-V2 build_polskie_sprawy_v3_cache.sh, z różnicami:
# - dane mają turę system (prompt glimmer8) przed user; loss na wszystkich tokenach (oprócz BOS),
#   czyli także na prompcie systemowym (domyślne --loss-roles all),
# - 8192 zamiast 16384 (max_position_embeddings Bielika 1.5B); za długie rozmowy (141 z 538,079) są
#   odrzucane (--long-policy drop). remove-reasoning w prepare_poziomka_sft.py wpisuje do treści
#   stary prefiks v12 '<think>\n</think>\n', który w v13 wygląda jak start rozumowania (po <think>
#   jest \n) — dokładnie dwuznaczność, którą v13 usuwa; truncate zostawia rozmowy bez <|im_end|>.
# - tokenizer z katalogu modelu Bielika (ten sam słownik APT4 i ID specjalne co Poziomka).
# Liczby ON/OFF i kubełków nie są z góry znane: skrypt je wypisuje zamiast porównywać.
#
# Inne dane / model / długość (domyślne = pilotaż 1.5B, cpral/nauka-lotnictwa-1, 8192):
#   DATASET=<hf id> albo RAW_JSONL=<lokalny sft.jsonl>, NAME=<nazwa w katalogach>,
#   SEQ_LENGTH=..., RECORDS=<oczekiwana liczba rekordów, 0 = nie sprawdzaj>, CACHE_PREFIX=...
#   Rozmowy mogą być wieloturowe: system, potem user/assistant na przemian, na końcu assistant.
#   UPSAMPLE_MULTITURN=k: rozmowy z ≥ 2 turami użytkownika wchodzą k razy (kopie mają id
#   "<id>#<n>", więc trafiają do tego samego podziału train/validation co oryginał — podział
#   bierze pierwsze 8 znaków id — a prepare_polskie_sprawy_v3.py nie odrzuca ich jako duplikatów).
#
# Uruchomienie na riggu:
#   LING_DIR=~/projects/pretrain/Ling-V2 \
#   BIELIK_HF=/media/nvme_2tb/maked/models/Bielik-1.5B-v3.0-Instruct \
#   bash build_nauka_lotnictwa_1_cache.sh
# Wyniki w ${WORK_DIR} (domyślnie katalog nad LING_DIR, na riggu ~/projects/pretrain).
# Odmawia nadpisania istniejących katalogów.

: "${LING_DIR:?Set LING_DIR to the Ling-V2 checkout}"
: "${BIELIK_HF:?Set BIELIK_HF to the HF Bielik-1.5B-v3.0-Instruct directory}"
SFT_DIR="${LING_DIR}/examples/sft/megatron"
WORK_DIR="${WORK_DIR:-$(cd -- "${LING_DIR}/.." && pwd)}"

DATASET="${DATASET:-cpral/nauka-lotnictwa-1}"
NAME="${NAME:-nauka-lotnictwa-1}"
RECORDS="${RECORDS:-538079}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
CACHE_PREFIX="${CACHE_PREFIX:-bielik-sft-cache}"
RAW_DIR="${WORK_DIR}/${NAME}"
SPLIT_DIR="${WORK_DIR}/${NAME}-sft-v13"
CACHE_DIR="${WORK_DIR}/${CACHE_PREFIX}-${NAME}-v13-${SEQ_LENGTH}"
TEMPLATE="${SFT_DIR}/poziomka_v13_chat_template.jinja"
WORKERS="${WORKERS:-16}"
UPSAMPLE_MULTITURN="${UPSAMPLE_MULTITURN:-1}"

for d in "${SPLIT_DIR}" "${CACHE_DIR}"; do
    [[ ! -e "${d}" ]] || { echo "Refusing existing output: ${d}" >&2; exit 1; }
done
[[ -f "${BIELIK_HF}/tokenizer.json" ]] || { echo "No tokenizer.json in ${BIELIK_HF}" >&2; exit 1; }

if [[ -n "${RAW_JSONL:-}" ]]; then
    mkdir -p "${RAW_DIR}"
    [[ -f "${RAW_DIR}/sft.jsonl" ]] || cp "${RAW_JSONL}" "${RAW_DIR}/sft.jsonl"
elif [[ ! -f "${RAW_DIR}/sft.jsonl" ]]; then
    hf download "${DATASET}" --repo-type dataset --include sft.jsonl --local-dir "${RAW_DIR}"
fi
records=$(wc -l < "${RAW_DIR}/sft.jsonl")
[[ "${RECORDS}" == 0 || "${records}" == "${RECORDS}" ]] || { echo "Expected ${RECORDS} records in ${RAW_DIR}/sft.jsonl, found ${records}" >&2; exit 1; }
# Every record: Magpie system prompt, then user/assistant alternating, ending with assistant.
python3 - "${RAW_DIR}/sft.jsonl" <<'PY'
import collections, json, sys
bad, turns = 0, collections.Counter()
for line in open(sys.argv[1], encoding="utf-8"):
    roles = [m["role"] for m in json.loads(line)["messages"]]
    n = (len(roles) - 1) // 2
    if roles != ["system"] + ["user", "assistant"] * n or n < 1:
        bad += 1
    turns[n] += 1
if bad:
    sys.exit(f"{bad} records are not system + (user, assistant) x n")
print(f"All records: system + (user, assistant) x n; turns: {dict(sorted(turns.items()))}")
PY

SOURCE="${RAW_DIR}/sft.jsonl"
if [[ "${UPSAMPLE_MULTITURN}" != 1 ]]; then
    SOURCE="${RAW_DIR}/sft_multiturn_x${UPSAMPLE_MULTITURN}.jsonl"
    [[ -f "${SOURCE}" ]] || python3 - "${RAW_DIR}/sft.jsonl" "${SOURCE}" "${UPSAMPLE_MULTITURN}" <<'PY'
import json, sys
src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
n = extra = 0
with open(src, encoding="utf-8") as fin, open(dst + ".tmp", "x", encoding="utf-8") as fout:
    for line in fin:
        r = json.loads(line)
        fout.write(line)
        n += 1
        if sum(m["role"] == "user" for m in r["messages"]) >= 2:
            for copy in range(1, k):
                fout.write(json.dumps({**r, "id": f"{r['id']}#{copy}"}, ensure_ascii=False) + "\n")
                extra += 1
import os; os.rename(dst + ".tmp", dst)
print(f"Upsampled multi-turn x{k}: {n:,} records + {extra:,} copies = {n + extra:,}")
PY
fi

python3 "${SFT_DIR}/prepare_polskie_sprawy_v3.py" \
    --input "${SOURCE}" --output "${SPLIT_DIR}" --off-prefix none

python3 "${SFT_DIR}/prepare_poziomka_sft.py" \
    --input "${SPLIT_DIR}" --tokenizer "${BIELIK_HF}" --chat-template "${TEMPLATE}" \
    --output "${CACHE_DIR}" --seq-length "${SEQ_LENGTH}" --long-policy drop \
    --loss-roles all --workers "${WORKERS}"

PYTHONPATH="${SFT_DIR}" python3 - "${CACHE_DIR}" <<'PY'
import collections, glob, json, math, sys
import numpy as np
from transformers import PreTrainedTokenizerFast
from poziomka_data import MMapSFTDataset

cache = sys.argv[1]
tok = PreTrainedTokenizerFast.from_pretrained(f"{cache}/tokenizer")
lt, th, ink, gt = tok.convert_tokens_to_ids(["<", "th", "ink", ">"])
newline = tok.convert_tokens_to_ids("<0x0A>")
after = collections.Counter()
for path in sorted(glob.glob(f"{cache}/train/*.tokens.bin")):
    tokens = np.fromfile(path, dtype=np.uint16)
    offsets = np.fromfile(path.replace(".tokens.bin", ".offsets.bin"), dtype=np.uint64)
    for i in range(len(offsets) - 1):
        s = tokens[int(offsets[i]):int(offsets[i + 1])]
        hit = np.where((s[:-4] == lt) & (s[1:-3] == th) & (s[2:-2] == ink) & (s[3:-1] == gt))[0]
        nxt = int(s[hit[0] + 4]) if len(hit) else None
        after["ON (\\n)" if nxt == newline else "OFF (<)" if nxt == lt else f"other {nxt}"] += 1
print("Token after the first <think>:", dict(after))
print("(przed cache: ON 107,709 / OFF 425,024; różnica = rozmowy odrzucone jako za długie)")

totals = json.load(open(f"{cache}/manifest.json"))["totals"]
print("Manifest totals:", json.dumps(totals, indent=1))

train = MMapSFTDataset(f"{cache}/manifest.json", "train", pack=True)
bins = len(train.bins)
print(f"Packed train sequences: {bins:,} (efficiency {train.packing_efficiency():.3f})")
print(f"TRAIN_ITERS at GBS 16: {math.ceil(bins / 16)}, at GBS 32: {math.ceil(bins / 32)}, at GBS 64: {math.ceil(bins / 64)}")
PY

echo "Ready: ${CACHE_DIR}"
