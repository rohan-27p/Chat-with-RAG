# PDF Chat

Upload a PDF and ask questions grounded only in that document. The project now uses a single backend stack:

- Frontend: React + TypeScript + Vite
- Backend: Python + FastAPI
- LLM: Groq `llama-3.1-8b-instant`
- Retrieval: local embeddings + FAISS

## Requirements

- Python 3.11+
- Node.js 18+
- A `GROQ_API_KEY`

## Setup

### Backend

```bash
cd backend
python -m venv .venv
```

Activate the venv:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `backend/.env`:

```env
GROQ_API_KEY=gsk_...
PORT=3001
FRONTEND_URL=http://localhost:5173
```

Start the backend:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 3001 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Project Structure

```text
frontend/        React app
backend/app/     FastAPI routes and RAG pipeline
backend/tests/   Evaluation tooling
backend/config.py
backend/requirements.txt
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'fitz'`
  Install Python deps in the active venv with `pip install -r requirements.txt`.
- `Groq API key not found`
  Add `GROQ_API_KEY` to `backend/.env`.
- CORS errors
  Make sure `FRONTEND_URL` matches your frontend origin.

## Tests

```bash
cd backend
python -m pip install -r tests/requirements-test.txt
python -m tests.run_evaluation
```

See [backend/TESTING.md](</c:/Users/lostdecimal/Downloads/pdf-chat-RAG-main/pdf-chat-RAG-main/backend/TESTING.md>) for more detail.
