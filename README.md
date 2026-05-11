# commit-msg-llm

> Fine-tuned LLM that writes git commit messages from diffs.
> Full MLOps lifecycle: train → register → serve → monitor → auto-redeploy.

![CI](https://github.com/muhammadh01/commit-msg-llm/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## What it does

```bash
curl -X POST https://api.durak.dev/generate \
  -H "Content-Type: application/json" \
  -d '{"diff":"# MODIFY src/auth.py\n+if user.is_banned: return None"}'
```

```json
{"message": "add ban check to login flow", "cached": false}
```

## Architecture

```
                       ┌─────────────────┐
                       │   durak.dev     │
                       │  Nginx + TLS    │
                       └────────┬────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        ┌───────▼────────┐              ┌──────▼───────┐
        │  FastAPI app   │◄────────────►│    Redis     │
        │   (Qwen 7B +   │              │    cache     │
        │   QLoRA)       │              └──────────────┘
        └───────┬────────┘
                │
        ┌───────▼────────┐              ┌──────────────┐
        │   Prometheus   │◄─────────────│   Grafana    │
        │    metrics     │              │  dashboards  │
        └────────────────┘              └──────────────┘

Training pipeline (GitHub Actions):
CommitChronicle → Kaggle GPU → QLoRA → MLflow → HF Hub → K8s blue-green
```

## Tech stack

| Layer | Tools |
|---|---|
| Model | Qwen2.5-7B + QLoRA (4-bit) |
| Training | PyTorch, HuggingFace `peft`, `trl`, Kaggle GPU |
| Tracking | MLflow (params, metrics, artifacts) |
| Registry | HuggingFace Hub |
| Serving | FastAPI + vLLM, Redis cache |
| Infra | Docker, Kubernetes (DigitalOcean), Nginx + Let's Encrypt |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions (test, build, blue-green deploy) |
| Eval | BLEU + ROUGE + GPT-4o-mini as judge |

## Quick start

```bash
git clone https://github.com/muhammadh01/commit-msg-llm
cd commit-msg-llm
make docker-up
curl http://localhost:8000/health
```

## Commands

```bash
make install        # install python deps
make train          # local LoRA sanity training
make serve          # run FastAPI locally
make test           # run pytest
make docker-up      # start full stack (api + redis)
make docker-down    # stop stack
make docker-logs    # tail api logs
```

## Roadmap

- [x] **Week 1** — Local training + dockerized API + tests + MLflow
- [ ] **Week 2** — Real training on Kaggle (10k examples, Qwen 7B QLoRA), GPT-4 eval
- [ ] **Week 3** — Deploy to DigitalOcean K8s, public at `api.durak.dev`
- [ ] **Week 4** — Prometheus/Grafana monitoring, auto-redeploy, React frontend

## Author

Built by [@muhammadh01](https://github.com/muhammadh01) — B.Sc. AI student at JKU Linz, transitioning from DevOps to MLOps.

## License

MIT
