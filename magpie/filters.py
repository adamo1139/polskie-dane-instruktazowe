"""Czyszczenie i klasyfikacja wygenerowanej wiadomości użytkownika.

classify() zwraca None (OK) albo powód odrzucenia. Reguły to tani filtr regexowy — łapią echo
promptu, przecieki roli/reasoningu i wiadomości nienadające się na pierwszą turę rozmowy.
Nie łapią rzeczy wymagających zrozumienia treści (brak kontekstu, off-topic, bełkot).
"""
import re

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")

# Przedrostki do odcięcia — za nimi zwykle stoi dobre pytanie („Pytanie: Czy …?”).
SPEAKER_PREFIX = re.compile(r"^(Polak|Polka|Użytkownik|Użytkowniczka|Pytanie( użytkownika)?|Krótko|"
                            r"Wiadomość (Polki|Polaka|użytkownika))\s*:\s*", re.IGNORECASE)

# Model opisuje scenę zamiast pisać jako użytkownik (także parafrazą promptu).
SCENE_START = re.compile(r"^\W*(Rozmow\w*|Rozmow\b|Temat\b|Kontekst|Scena\b|Opis\b|Pewn[ay] Pol|"
                         r"Pol(ka|ak) (pisze|pyta|zadaje)|Użytkownik|Użytkowniczka)", re.IGNORECASE)
# Cyrylica/greka/dewanagari itp. w polskim tekście = homoglify albo bełkot.
NON_LATIN = re.compile(r"[Ͱ-ϿЀ-ӿऀ-ॿ؀-ۿ一-鿿]")
ROLE_LEAK = re.compile(r"(^|\n)\s*(assistant|asystent|user|użytkownik)\b[^\n]{0,30}[:>]|assistant to="
                       r"|</?(user|assistant)", re.IGNORECASE)
# Angielski reasoning Glimmera, który przeciekł do tury użytkownika.
EN_LEAK = re.compile(r"\b(We need to|The user|Answer in|User response|presumably)\b")

# --- Wiadomość nie nadaje się na pierwszą turę użytkownika ---
CONTINUATION = re.compile(r"^\W*(Dziękuj|Dzięki|Bardzo dziękuj|Jasne[,.!]|Oczywiście[,.!]|Rozumiem[,.!]|"
                          r"Zrozumiał|To teraz|A teraz|A mogę|Wracając do|Jak już mówił|Tak jak pisał|"
                          r"Czas już najwyższy|Masz rację|No dobra, to)", re.IGNORECASE)
TRUNCATED = re.compile(r"(\.\.\.|…)\s*\??\s*$")
META_PROMPT = re.compile(r"naturalne pytanie|krótkie,? naturalne|napisz (jedno |krótkie )?pytanie|"
                         r"zapytam o jeden|jeden z tych tematów|instrukcj|jailbreak|zignoruj poprzednie|"
                         r"system prompt|prompt systemow|jako model językowy", re.IGNORECASE)
MARKDOWN_DOC = re.compile(r"(^|\n)\s*#{1,4}\s|```")
SPEAKER_LABEL = re.compile(r"^[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,15}:\s")
ASSISTANT_RULES = re.compile(r"^\W*(Twoja odpowiedź|Twoje odpowiedzi|Na twojej odpowiedzi|Odpowiadasz|"
                             r"Odpowiadaj|Każda (twoja )?odpowiedź|Jesteś (asystentem|ekspertem|specjalistą)|"
                             r"Wciel się|Zachowuj się)", re.IGNORECASE)
QUALITY_RULES = [
    (CONTINUATION.match, "kontynuacja rozmowy"),
    (TRUNCATED.search, "urwane"),
    (META_PROMPT.search, "meta"),
    (MARKDOWN_DOC.search, "dokument/markdown"),
    (SPEAKER_LABEL.match, "etykieta mówiącego"),
    (ASSISTANT_RULES.match, "reguły dla asystenta"),
]

# Pytanie, pusta linia i dalszy tekst — możliwa samoodpowiedź (tylko flaga, nie odrzucenie).
SELF_ANSWER = re.compile(r"\?\s*\n\s*\n\s*\S")


def _set_raw(out: dict, raw: str) -> dict:
    return {**out, "raw_orig": out.get("raw_orig", out["raw"]), "raw": raw,
            "first": raw.split("\n")[0].strip() if raw else ""}


def clean(out: dict) -> dict:
    # Tekstowe „</user…” (bez tokenu specjalnego) zamyka turę — dalej pisze już asystent.
    cut = out["raw"].find("</user")
    if cut >= 0:
        out = _set_raw(out, out["raw"][:cut].rstrip())
    if SPEAKER_PREFIX.match(out["raw"]):
        out = _set_raw(out, SPEAKER_PREFIX.sub("", out["raw"], count=1))
    return out


def classify(out: dict, system: str) -> str | None:
    raw = out["raw"]
    if out["finish"] != "stop":
        return "brak EOS"
    if not out["first"]:
        return "puste"
    if any(s in raw for s in (x.strip() for x in SENT_SPLIT.split(system)) if len(s) > 15):
        return "echo systemu"
    if SCENE_START.match(raw):
        return "echo (parafraza)"
    if NON_LATIN.search(raw):
        return "obce pismo"
    if ROLE_LEAK.search(raw):
        return "przeciek roli"
    if len(raw.split()) < 3:
        return "za krótkie"
    for test, reason in QUALITY_RULES:
        if test(raw):
            return reason
    if EN_LEAK.search(raw):
        return "przeciek EN"
    return None
