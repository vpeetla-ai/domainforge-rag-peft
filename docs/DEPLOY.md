# Deploy — DomainForge

## Live URLs (target)

| Layer | URL |
|-------|-----|
| UI | https://domainforge-rag-peft.vercel.app |
| API | https://domainforge-api.onrender.com |

## Vercel (UI)

```bash
cd ui
npm install
npm run build
npx vercel --prod --yes
npx vercel alias set <deployment>.vercel.app domainforge-rag-peft.vercel.app
```

Set `NEXT_PUBLIC_API_URL=https://domainforge-api.onrender.com` in Vercel project env (or `ui/.env.production`).

## Render (API) — one-time Blueprint

1. Open [Deploy to Render](https://render.com/deploy?repo=https://github.com/vpeetla-ai/domainforge-rag-peft)
2. Confirm `render.yaml` — service `domainforge-api`, plan `free`
3. After deploy: `curl https://domainforge-api.onrender.com/health`

Or: Render Dashboard → **New** → **Blueprint** → connect `vpeetla-ai/domainforge-rag-peft`.

### Env vars (auto from render.yaml)

| Key | Value |
|-----|-------|
| `APP_ENV` | production |
| `MOCK_LLM` | true |
| `RETRIEVER_MODE` | memory |
| `CORS_ORIGINS` | https://domainforge-rag-peft.vercel.app,https://venkat-ai.com |
| `DOMAINFORGE_API_KEY` | generated |

## Verify

```bash
curl https://domainforge-rag-peft.vercel.app
curl https://domainforge-api.onrender.com/health
curl https://domainforge-api.onrender.com/v1/metrics
```

## Free tier notes

- Render cold start ~30–60s after idle — first `/v1/query` may be slow
- UI is static export on Vercel (always warm)
- Render cannot train Mistral or host Ollama — use [GPU_OLLAMA_PIPELINE.md](GPU_OLLAMA_PIPELINE.md)

## Real inference (Ollama on GPU host)

1. Train on CUDA: `bash scripts/gpu_pipeline.sh`
2. On GPU machine: `ollama serve` + models `domainforge-triage` / `domainforge-triage-dpo`
3. Render env: `MOCK_LLM=false`, `OLLAMA_BASE_URL=http://<gpu-host>:11434`
4. Query with `solution=s3_peft_hybrid` or `s4_dpo_peft` — response `inference_backend: ollama`
