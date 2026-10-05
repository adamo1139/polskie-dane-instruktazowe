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

# Szablon Harmony-podobny Glimmera; BOS dokleja serwer w /v1/completions, więc go tu nie ma.
STOP_STRINGS = ["<|eot|>", "<|start|>"]
STOP_TOKEN_IDS = [200007, 200008]  # <|eom|>, <|eot|> — <|eom|> nie jest EOS w vLLM, trzeba podać


def join_subtopics(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " albo " + items[-1]


def build_prompt(system: str) -> str:
    return f"<|start|>system<|message|>{system}<|eot|><|start|>user<|message|>"


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
