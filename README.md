# commit-msg-llm

> AI-powered commit message generator. Fine-tuned Qwen2.5-0.5B, served from a full MLOps stack.

**🌍 [Live Demo →](https://commit-msg.durak.dev)**

```
🎨 https://commit-msg.durak.dev        Frontend (Next.js)
🔌 https://commit.durak.dev            API (FastAPI)
📚 https://commit.durak.dev/docs       OpenAPI docs
📊 https://commit-grafana.durak.dev    Grafana dashboard
🤗 huggingface.co/biighunter/commit-msg-llm-adapter
```

---

## What it does

Paste a `git diff` → get a clean, conventional commit message.

Powered by Qwen2.5-0.5B fine-tuned with LoRA on 6,676 real commits from the [CommitChronicle](https://huggingface.co/datasets/JetBrains-Research/commit-chronicle) dataset.

## Stack

| Layer | Tech |
|---|---|
| **Training** | Qwen2.5-0.5B · LoRA · PEFT · Kaggle GPU |
| **Tracking** | MLflow |
| **Registry** | HuggingFace Hub · ghcr.io |
| **Serving** | FastAPI · Redis cache · PyTorch |
| **Infra** | Kubernetes (DigitalOcean) · Helm |
| **Ingress** | Nginx · cert-manager · Let's Encrypt |
| **Monitoring** | Prometheus · Grafana · custom metrics |
| **CI/CD** | GitHub Actions · release-please |
| **Eval** | BLEU · ROUGE · GPT-4o-mini LLM-as-judge |
| **Frontend** | Next.js 15 · TypeScript · Tailwind · shadcn/ui · Framer Motion |

## Architecture

```
                                ┌──────────────────┐
                                │  GitHub Actions  │
                                │   (test→build→   │
                                │   push→deploy)   │
                                └─────────┬────────┘
                                          │
                                          ▼
┌─────────┐      HTTPS       ┌──────────────────────────┐
│ Browser │ ─────────────────│  K8s cluster (DO fra1)   │
└─────────┘                  │                          │
                             │  ┌────────┐  ┌─────────┐ │
                             │  │ web    │──│  api    │ │
                             │  │ Next.js│  │ FastAPI │ │
                             │  └────────┘  └────┬────┘ │
                             │                   │      │
                             │              ┌────▼────┐ │
                             │              │  redis  │ │
                             │              └─────────┘ │
                             │                          │
                             │   Prometheus → Grafana   │
                             └──────────────────────────┘
```

## Run locally

```bash
# Backend
make install
make serve  # http://localhost:8000

# Frontend
cd web && npm install && npm run dev  # http://localhost:3000
```

## Training

```bash
make data    # download CommitChronicle (~10k samples)
make train   # LoRA fine-tune (GPU recommended)
make eval    # BLEU + ROUGE + LLM judge
```

## Deployment

Push to `main` → GitHub Actions:
1. Runs tests + lint
2. Builds Docker image
3. Pushes to ghcr.io
4. Rolls out to K8s
5. Smoke-tests health endpoint

## Metrics

Custom Prometheus metrics exposed at `/metrics`:
- `generate_total{status}` — total generations by status
- `generate_latency_seconds` — inference latency histogram
- `cache_events_total{event}` — Redis cache hits/misses

Visualize at [commit-grafana.durak.dev](https://commit-grafana.durak.dev).

## License

MIT
