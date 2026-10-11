"""Prompt systemowy (wariant glimmer8) i budowa zadań dla Muse Glimmer.

Magpie: dajemy modelowi tylko prompt systemowy i otwartą turę użytkownika — model sam dopisuje
wiadomość użytkownika i zamyka turę tokenem <|eom|> albo <|eot|>.
"""
import random

from .topics import SUBTOPICS

# Lista 10–15 konkretnych podtematów ogranicza echo promptu (model chwyta się konkretu zamiast
# opisywać scenę); „albo” przed ostatnim, bo podtematy same zawierają „i” („przedszkolu i szkole”).
SYSTEM_TEMPLATE = (
    "Rozmowa o {topic}, na przykład o {subs}. {person} pisze teraz jedno krótkie, naturalne "
    "pytanie po polsku na jeden z tych tematów. {poss} wiadomość kończy się zaraz po tym pytaniu."
)
PERSONAS = [("Polak", "Jego"), ("Polka", "Jej")]

# Szablony tury systemowej + otwartej tury użytkownika. BOS dokleja serwer w /v1/completions
# (sprawdzone dla vLLM z Glimmerem i SGLang z Bielikiem), więc go tu nie ma.
TEMPLATES = {
    # Harmony-podobny Glimmera; <|eom|> nie jest EOS w vLLM, trzeba go podać.
    "glimmer": {"prompt": "<|start|>system<|message|>{system}<|eot|><|start|>user<|message|>",
                "stop": ["<|eot|>", "<|start|>"], "stop_token_ids": [200007, 200008]},
    # ChatML Poziomki v13 (APT4): Bielik po SFT na nauka-lotnictwa-1; <|im_end|> = 4.
    "v13": {"prompt": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n",
            "stop": ["<|im_end|>", "<|im_start|>"], "stop_token_ids": [4]},
}
# Formaty pełnych tur (historia rozmowy dla magpie.multiturn). Tura asystenta Glimmera sprawdzona przez
# /tokenize na szablonie modelu: <|start|>assistant to=user<|message|>…<|eot|>.
TURNS = {
    "glimmer": {"system": "<|start|>system<|message|>{content}<|eot|>",
                "user": "<|start|>user<|message|>{content}<|eot|>",
                "assistant": "<|start|>assistant to=user<|message|>{content}<|eot|>",
                "open_user": "<|start|>user<|message|>"},
    # Tura asystenta bez rozumowania jak w szablonie v13: <think></think>\n przed treścią.
    "v13": {"system": "<|im_start|>system\n{content}<|im_end|>\n",
            "user": "<|im_start|>user\n{content}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n<think></think>\n{content}<|im_end|>\n",
            "open_user": "<|im_start|>user\n"},
}
STOP_STRINGS = TEMPLATES["glimmer"]["stop"]
STOP_TOKEN_IDS = TEMPLATES["glimmer"]["stop_token_ids"]


def join_subtopics(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " albo " + items[-1]


def build_prompt(system: str, template: str = "glimmer") -> str:
    return TEMPLATES[template]["prompt"].format(system=system)


def build_jobs(total: int, master_seed: int, subs_min: int, subs_max: int) -> list[dict]:
    """Deterministyczna lista zadań: ten sam master_seed i ta sama topics.py → te same zadania."""
    rng = random.Random(master_seed)
    topics = list(SUBTOPICS)
    jobs = []
    for i in range(total):
        person, poss = rng.choice(PERSONAS)
        topic = topics[i % len(topics)]
        subs = rng.sample(SUBTOPICS[topic], rng.randint(subs_min, subs_max))
        jobs.append({
            "id": i,
            "system": SYSTEM_TEMPLATE.format(topic=topic, subs=join_subtopics(subs),
                                             person=person, poss=poss),
            "seed": rng.randint(1, 10**9),
            "persona": person, "topic": topic, "subtopics": subs,
        })
    return jobs


def build_history_prompt(messages: list[dict], template: str = "glimmer") -> str:
    """Historia (system, user, assistant, …) + otwarta tura użytkownika. Rozumowanie z historii
    jest pomijane — użytkownik widzi tylko odpowiedź."""
    fmt = TURNS[template]
    return "".join(fmt[m["role"]].format(content=m["content"]) for m in messages) + fmt["open_user"]
