# MatchaQ

FastAPI web app for matching portfolios against a job description.

## Run locally

```bash
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- `http://localhost:8000`
- `http://<your-local-ip>:8000` for other devices on the same Wi-Fi/network

## Share with other people

For people outside your local network, deploy this folder to any Python web host that supports:

- `requirements.txt`
- `Procfile`
- `render.yaml` for Render Blueprint deploys

The app exposes:

- `/` main web page
- `/health` health check

## Notes

- Port is controlled by `PORT`
- Host is controlled by `HOST`
- Set `RELOAD=true` only for development
- Render is pinned to Python `3.11.9` for better library compatibility

## Deploy to Render

1. Push this project to GitHub.
2. In Render, choose `New +` > `Blueprint`.
3. Connect the GitHub repo that contains this project.
4. Render will detect `render.yaml` and create the web service.
5. Wait for the build to finish, then open the generated `onrender.com` URL.
