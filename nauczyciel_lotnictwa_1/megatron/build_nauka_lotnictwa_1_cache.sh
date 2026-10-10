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

DATASET="cpral/nauka-lotnictwa-1"
RECORDS=538079
SEQ_LENGTH=8192
RAW_DIR="${WORK_DIR}/nauka-lotnictwa-1"
SPLIT_DIR="${WORK_DIR}/nauka-lotnictwa-1-sft-v13"
CACHE_DIR="${WORK_DIR}/bielik-sft-cache-nauka-lotnictwa-1-v13-${SEQ_LENGTH}"
TEMPLATE="${SFT_DIR}/poziomka_v13_chat_template.jinja"
WORKERS="${WORKERS:-16}"

for d in "${SPLIT_DIR}" "${CACHE_DIR}"; do
    [[ ! -e "${d}" ]] || { echo "Refusing existing output: ${d}" >&2; exit 1; }
done
[[ -f "${BIELIK_HF}/tokenizer.json" ]] || { echo "No tokenizer.json in ${BIELIK_HF}" >&2; exit 1; }

if [[ ! -f "${RAW_DIR}/sft.jsonl" ]]; then
    hf download "${DATASET}" --repo-type dataset --include sft.jsonl --local-dir "${RAW_DIR}"
fi
records=$(wc -l < "${RAW_DIR}/sft.jsonl")
[[ "${records}" == "${RECORDS}" ]] || { echo "Expected ${RECORDS} records in ${RAW_DIR}/sft.jsonl, found ${records}" >&2; exit 1; }
# Every record must start with the Magpie system prompt.
python3 - "${RAW_DIR}/sft.jsonl" <<'PY'
import json, sys
bad = sum(1 for line in open(sys.argv[1], encoding="utf-8")
          if [m["role"] for m in json.loads(line)["messages"]] != ["system", "user", "assistant"])
if bad:
    sys.exit(f"{bad} records are not system, user, assistant")
print("All records: system, user, assistant")
PY

python3 "${SFT_DIR}/prepare_polskie_sprawy_v3.py" \
    --input "${RAW_DIR}/sft.jsonl" --output "${SPLIT_DIR}" --off-prefix none

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
