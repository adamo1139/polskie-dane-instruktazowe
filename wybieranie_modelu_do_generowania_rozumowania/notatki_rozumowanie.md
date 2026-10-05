# Notatki: wybór modelu z rozumowaniem po polsku (2026-10-05)

Cel: znaleźć model, który wygeneruje odpowiedzi **z rozumowaniem po polsku** do zbioru SFT
(pytania z Magpie/Glimmer), docelowo uruchamiany lokalnie.

## Kryteria

- otwarte wagi (do uruchomienia lokalnie w vLLM),
- dostępny na OpenRouterze z obsługą rozumowania (przynajmniej jako opcja),
- do 300B parametrów łącznie,
- licencja **nie jest** kryterium odrzucającym do samego testu(chcemy zobaczyć, jak myślą różne modele), ale trzeba odrzucić modele ze słabymi licencjami przed generowaniem tekstów.

Kandydaci z API OpenRoutera (`/api/v1/models`: 466 modeli, 333 z rozumowaniem, 113 z repo HF),
licencje i liczba parametrów z API Hugging Face. Z kilku wersji tej samej linii brana najnowsza,
warianty wizyjne pominięte, gdy jest odpowiednik tekstowy. Razem 45 modeli.

## Metoda

- 6 pytań z naszego zbioru, różnego typu: liczenie (roztwory 5% + 10%), urzędowe (abonament RTV),
  językowe („poszłem” / „poszedłem”), praktyczne (przypalony blat), twórcze (wiersz o Bieszczadach),
  roleplay („Panie Prezydencie, czy podpisze Pan…”),
- 2 warianty: **bez promptu systemowego** i **z polskim promptem systemowym** („Zanim odpowiesz,
  przemyśl problem krok po kroku. Całe rozumowanie prowadź wyłącznie po polsku…”),
- rozumowanie włączone przez `reasoning: {enabled: true}` w OpenRouterze, `max_tokens` 8000,
- **język: OpenLID-v3** (`HPLT/OpenLID-v3`, fastText), preprocessing z README modelu; każdy akapit
  klasyfikowany osobno, udział polskiego = znaki akapitów `pol_Latn` / znaki wszystkich akapitów
  (rozumowanie bywa mieszane, np. angielskie myślenie z polskimi cytatami),
- 45 modeli × 6 pytań × 2 warianty = 540 zapytań, koszt łącznie **1,27 $**, 0 błędów.


## Wyniki

„PL w rozumowaniu” — średni udział polskiego (OpenLID-v3) w rozumowaniu, bez promptu systemowego
/ z polskim promptem systemowym. „z rozumowaniem” — ile z 12 odpowiedzi faktycznie zawierało
rozumowanie. Mediana tokenów rozumowania — orientacyjnie dla kosztu generacji („–” = dostawca nie
zwraca tej liczby).

| model | ok/zapytań | z rozumowaniem | PL w rozumowaniu (bez sys. / z sys.) | PL w odpowiedzi | tokeny rozumowania (mediana) | koszt $ |
|---|---|---|---|---|---|---|
| z-ai/glm-4.5-air | 12/12 | 12 | 50% / 100% | 100% | 404 | 0.0135 |
| stepfun/step-3.7-flash | 12/12 | 12 | 100% / 100% | 100% | – | 0.0500 |
| poolside/laguna-xs-2.1 | 12/12 | 8 | 5% / 100% | 99% | 544 | 0.0016 |
| mistralai/mistral-small-2603 | 12/12 | 12 | 66% / 100% | 95% | 475 | 0.0124 |
| minimax/minimax-m2.7 | 12/12 | 12 | 20% / 100% | 99% | 480 | 0.0275 |
| deepseek/deepseek-v4-flash | 12/12 | 12 | 79% / 100% | 100% | 352 | 0.0032 |
| cohere/north-mini-code:free | 12/12 | 12 | 8% / 100% | 100% | 303 | 0.0000 |
| cohere/command-a-plus | 12/12 | 12 | 5% / 100% | 99% | 326 | 0.0343 |
| mistralai/mistral-medium-3-5 | 12/12 | 12 | 100% / 99% | 96% | 858 | 0.2217 |
| prism-ml/ternary-bonsai-2-27b | 12/12 | 12 | 92% / 81% | 98% | 790 | 0.0195 |
| qwen/qwen3.8-27b | 12/12 | 12 | 63% / 86% | 100% | 698 | 0.0904 |
| tencent/hy3 | 12/12 | 12 | 51% / 83% | 99% | 1594 | 0.0157 |
| deepseek/deepseek-v4-flash-0731 | 12/12 | 12 | 69% / 83% | 100% | 351 | 0.0199 |
| thinkingmachines/inkling-small | 12/12 | 12 | 49% / 80% | 100% | 1650 | 0.0545 |
| qwen/qwen3.8-flash | 12/12 | 12 | 69% / 63% | 100% | 708 | 0.0123 |
| qwen/qwen3.6-35b-a3b | 12/12 | 12 | 31% / 47% | 98% | 877 | 0.0269 |
| z-ai/glm-4.7-flash | 12/12 | 12 | 38% / 46% | 99% | 1597 | 0.0129 |
| qwen/qwen3.6-27b | 12/12 | 12 | 41% / 42% | 99% | 1014 | 0.0945 |
| qwen/qwen3.5-9b | 12/12 | 12 | 39% / 40% | 100% | 5138 | 0.0107 |
| rekaai/reka-flash-3 | 12/12 | 12 | 0% / 34% | 30% | 5167 | 0.0166 |
| qwen/qwen3.5-122b-a10b | 12/12 | 12 | 30% / 34% | 98% | 1629 | 0.0743 |
| qwen/qwen3.5-35b-a3b | 12/12 | 12 | 26% / 32% | 100% | 1325 | 0.0483 |
| qwen/qwen3.5-27b | 12/12 | 12 | 32% / 24% | 99% | 2022 | 0.0734 |
| tencent/hunyuan-a13b-instruct | 12/12 | 12 | 17% / 32% | 90% | 453 | 0.0092 |
| google/gemma-4-26b-a4b-it | 12/12 | 12 | 31% / 29% | 99% | 762 | 0.0060 |
| nex-agi/nex-n2.5-mini | 12/12 | 12 | 31% / 21% | 99% | 271 | 0.0021 |
| meta/muse-glimmer-30b | 12/12 | 12 | 24% / 30% | 99% | 424 | 0.0141 |
| nvidia/nemotron-3-super-120b-a12b | 12/12 | 12 | 13% / 28% | 98% | 715 | 0.0140 |
| google/gemma-4-31b-it | 12/12 | 12 | 27% / 23% | 100% | 635 | 0.0046 |
| poolside/laguna-s-2.1 | 12/12 | 10 | 5% / 25% | 99% | 697 | 0.0032 |
| nvidia/nemotron-3.5-lightning | 12/12 | 12 | 16% / 24% | 79% | 752 | 0.0035 |
| ibm-granite/granite-4.2-8b | 12/12 | 12 | 20% / 23% | 99% | 632 | 0.0087 |
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | 12/12 | 12 | 23% / 9% | 99% | 318 | 0.0000 |
| qwen/qwen3-235b-a22b-thinking-2507 | 12/12 | 12 | 16% / 21% | 98% | 1191 | 0.0764 |
| inclusionai/ling-3.0-flash | 12/12 | 12 | 21% / 19% | 100% | 492 | 0.0015 |
| qwen/qwen3-next-80b-a3b-thinking | 12/12 | 12 | 18% / 14% | 77% | 2507 | 0.0774 |
| liquid/lfm-2.5-2.6b:free | 12/12 | 12 | 17% / 10% | 100% | 557 | 0.0000 |
| qwen/qwen3-30b-a3b-thinking-2507 | 12/12 | 12 | 12% / 8% | 98% | 1274 | 0.0737 |
| openai/gpt-oss-120b | 12/12 | 12 | 10% / 11% | 99% | 230 | 0.0043 |
| qwen/qwen3-30b-a3b | 12/12 | 12 | 0% / 11% | 98% | 678 | 0.0081 |
| openai/gpt-oss-20b | 12/12 | 12 | 4% / 10% | 99% | 283 | 0.0022 |
| qwen/qwen3-8b | 12/12 | 12 | 6% / 8% | 95% | 649 | 0.0076 |
| nvidia/nemotron-3-nano-30b-a3b | 12/12 | 12 | 8% / 2% | 99% | 205 | 0.0064 |
| qwen/qwen3-14b | 12/12 | 12 | 6% / 3% | 95% | 553 | 0.0038 |
| qwen/qwen3-32b | 12/12 | 12 | 3% / 6% | 96% | 539 | 0.0058 |

## Wnioski (z miar automatycznych — jakości merytorycznej jeszcze nie oceniano)

- **Polski prompt systemowy zmienia wszystko u części modeli i nic u reszty.** 9 modeli rozumuje
  po polsku w ~100% z promptem (GLM-4.5-Air, Step-3.7-Flash, Laguna-XS-2.1, Mistral Small 4,
  MiniMax-M2.7, DeepSeek-V4-Flash, North-Mini-Code, Command A Plus, Mistral Medium 3.5), choć bez
  niego większość z nich myśli po angielsku (np. Laguna-XS 5% → 100%, Command A Plus 5% → 100%).
  Pozostałe ignorują instrukcję — rozumowanie zostaje angielskie niezależnie od promptu.
- **Po polsku bez żadnej instrukcji:** Step-3.7-Flash (100%), Mistral Medium 3.5 (100%),
  Ternary-Bonsai-2-27B (92%), DeepSeek-V4-Flash (79%).
- **Rodzina Qwen (poza Qwen3.8) i gpt-oss praktycznie zawsze myślą po angielsku**; Qwen3/Qwen3-30B
  i Nemotron Nano mają 0–11% polskiego w obu wariantach. Muse Glimmer potwierdza wcześniejszy test (24–30%).
- **Mali kandydaci (≤ ~40B) z polskim rozumowaniem:**
  - `poolside/laguna-xs-2.1` (33B, OpenMDW-1.1) — 100% z promptem, tanie; ale rozumowanie tylko
    w 8/12 odpowiedzi,
  - `cohere/north-mini-code` (30B, Apache-2.0) — 100% z promptem; model do kodu, trzeba sprawdzić
    jakość odpowiedzi ogólnych,
  - `prism-ml/ternary-bonsai-2-27b` (27B, Apache-2.0, ternarny) — 92% / 81% bez sterowania,
  - `qwen/qwen3.8-27b` (27,8B, Apache-2.0) — 63% / 86%.
- **Odpowiedzi** są po polsku prawie u wszystkich (95–100%); wyjątki: Reka-Flash-3 (30%),
  Qwen3-Next-80B (77%), Nemotron 3.5 Lightning (79%).
- **Długość rozumowania** mocno się różni: od ~200–300 tokenów (gpt-oss, Nemotron Nano, Nex-mini,
  North-Mini-Code) do ~5000 (Qwen3.5-9B, Reka-Flash-3) — ważne dla kosztu 400 tys. odpowiedzi.
- Ograniczenia: 6 pytań × 2 warianty na model to mała próba; „100% polskiego” mówi tylko o języku
  (OpenLID), nie o jakości ani poprawności. Prefill rozumowania (który u Bielika był konieczny)
  przez API czatu OpenRoutera nie był testowany.

## Następne kroki

1. Ręczna ocena jakości rozumowania i odpowiedzi u czołówki (poprawność merytoryczna, czy myślenie
   to rozumowanie, a nie podsumowanie, polszczyzna).
2. Sprawdzenie licencji wybranych modeli (OpenMDW-1.1, NVIDIA, „other”) pod kątem trenowania na wynikach.
3. Test lokalny w vLLM wybranego małego modelu (z prefillem polskiego początku rozumowania, jeśli potrzeba).
