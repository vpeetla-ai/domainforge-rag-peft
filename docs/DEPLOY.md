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
