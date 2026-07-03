# Crop Disease WebApp

Deployment notes for Vercel:

- Set the following Environment Variables in the Vercel project settings:
  - `MODEL_URL` — public or signed URL where the model file can be downloaded (e.g. S3 presigned URL).
  - `WEATHER_API_KEY` — (optional) your weather API key.
  - `SECRET_KEY` — set a secure secret for Flask sessions (overrides default in code).

- The repository excludes `model/` and `venv/` from the deployment bundle via `.vercelignore`.
- The app will download the model at runtime from `MODEL_URL` when first started.

To deploy locally using Vercel CLI:

```bash
npm i -g vercel
vercel login
vercel --prod
```

If the model is large, prefer hosting on S3/Google Cloud and using a signed `MODEL_URL`.
