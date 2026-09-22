# Launch kit — open-dream-rsi (wrzesień 2026)

Timing: HN tematy Dream-RSI żyją (49726955, 49725509 — 1-6 dni). Jechać dziś-jutro.

## 1. GitHub repo metadata (po loginie — mogę wykonać przez przeglądarkę)

Description:
```
Open-source Dream-RSI: self-improvement loop around a frozen LLM. LLM-written exploration policies with replay-gated promotion + a curated knowledge base. 54% fewer API calls. Zero deps, MIT.
```
Topics: `dream-rsi`, `recursive-self-improvement`, `llm`, `exploration`, `agents`, `self-improvement`, `llm-agents`, `ai-agents`
Homepage: (pusty lub link do README#benchmark)

## 2. Show HN (news.ycombinator.com/submit — tytuł + body)

Title (do 80 zn.):
```
Show HN: Open Dream-RSI – self-improvement loop around a frozen LLM (zero deps)
```
URL: https://github.com/patrykorwat/open-dream-rsi

Text body (komentarz do posta):
```
I implemented the Dream-RSI paper (arXiv:2609.14858) as a zero-dependency
Python library and added two things the paper leaves implicit:

1. LLM-written exploration policies with replay-gated promotion. The loop asks
   your model to write choose_action(frontier, step) as Python source,
   validates it via AST, runs it in a scrubbed subprocess, scores it by
   counterfactual rollout on the recorded discovery tree, and promotes it only
   on evidence. The gate, not trust, decides.

2. A knowledge curator. Verifier failures get distilled into short,
   schema-validated lesson records that are retrieved into future proposals and
   pruned by uses/wins counters — the loop gets smarter between runs, not just
   cheaper.

Two deterministic, key-free benchmarks (scripted solver, so the model is a
constant and every delta is attributable to the machinery):
- bench: 54% fewer API calls at equal solve quality
- bench-policy (decoy-trap suite that isolates exploration where solve-rate
  graphs saturate): greedy solves 0/120, epsilon 24%, replay-gated policies
  72% (100% by cycle 5) at FEWER calls than epsilon wastes on luck; the
  knowledge base alone — zero policy calls — matches 72%.

Honest limits (also in the preprint): headline numbers come from scripted
worlds — they validate the machinery (sandboxes, gates, promotion, retrieval),
not live-model lesson quality. Live-model eval is next.

pip-installable, OpenAI-compatible endpoint (OpenAI/vLLM/Ollama/LM Studio),
live dashboard, MIT. Preprint: paper/main.pdf in repo. Happy to answer
questions about the replay gate specifically — it's the part I'd argue about.
```

## 3. X thread (3 posty, @patrykorwat)

1/ Dream-RSI (DeepMind, arXiv 2609.14858) says: improve the *process around*
a frozen LLM. I shipped it as a zero-dep Python library + two things the paper
doesn't spell out: replay-gated LLM-written exploration policies and a
knowledge curator. Show HN today. → github.com/patrykorwat/open-dream-rsi

2/ On the decoy-trap benchmark (isolates exploration where solve-rates
saturate): score-greedy solves 0/120. ε-greedy 24%. Replay-gated policies 72%
— at FEWER total calls than ε burns on luck. The knowledge base alone (zero
policy calls) also hits 72%: memory replaces luck. Numbers reproducible with
one command, no API key.

3/ Every artefact the model writes (code AND lessons) passes a structural gate
before touching the loop: AST validation, scrubbed subprocess, replay-rollout
promotion, uses/wins eviction. The gate, not trust, decides.
Preprint + tests + dashboard in repo. RTs appreciated 🙏

## 4. Reddit

- r/MachineLearning self-promo rules zakazują — NIE wrzucać directly.
  Zamiast: komentarz-link w żywych treściach Dream-RSI (sprawdzić
  "Dream-RSI self-improvement" w r/MachineLearning) — organic, zgodnie z rules.
- r/LocalLLaMA: post OK ("I built a self-improvement loop that works with any
  OpenAI-compatible local endpoint — vLLM/Ollama; 54% fewer calls, bench included").
- r/singularity: NIE (spam-filtr przy RSI-hype).

## 5. Inne kanały (kolejność wg ROI)

1. HN comment pod istniejącym threadem 49726955 (nie self-promo link —
   merytoryczny komentarz o replay-gacie + link na końcu "I built a library
   impl, link in profile" — ostrożnie, HN karza za astroturfing).
2. PR: dopisać repo do "Related implementations" w README
   github.com/zhengkid/Dream-RSI (official) — highest-ROI single action.
   TheAstrayDev/dream-rsi-sdk README też przyjmuje peer links.
3. alphaXiv comment pod 2609.14858.
4. Lobsters (login), Hacker News daily.
5. arXiv: sam submission wymaga konta + ew. endorsementu — przygotowane
   (paper/main.tex, cs.AI/cs.SE); nie jest kanalem launchu, ale cytowanie
   "preprint" bez ID jest OK przez pierwsze dni.
6. Bluesky: brak API kredy w srodowisku — manualnie, ten sam copy co X thread.
7. dev.to blog crosspost (longer-form: jak dziala counterfactual replay gate).

## 6. Checklist autentykacji (do odblokowania przez usera)

- [ ] GitHub login w przegladarze (vault pusty — po zalogowaniu zapisze sie do vault)
- [ ] gh CLI: `gh auth login` lub GH_TOKEN (do PR-ow do zhengkid/Dream-RSI)
- [ ] HN: konto + haslo (login przez /login, passkey lub vault)
- [ ] X: xurl auth (instrukcja w skillu xurl) lub manualny post
- [ ] Bluesky: login (app password) — manualny post OK
- [ ] Reddit: konto
