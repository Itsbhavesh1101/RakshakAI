# Rakshak Frontend

React + TypeScript dashboard for the Rakshak AI Surveillance Platform.

## Install

```bash
npm install
```

## Run

```bash
npm run dev
```

Open `http://localhost:5173`. The Vite dev server automatically connects to the backend at `http://localhost:8000`.

To override the API base URL:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

On Windows PowerShell:

```powershell
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

## Checks

```bash
npm run lint
npm run build
```
