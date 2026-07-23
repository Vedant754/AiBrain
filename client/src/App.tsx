import { useEffect, useState } from 'react'



interface HealthResponse {
  status: string
  app_name: string
  environment: string
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then(setHealth)
      .catch(() => setError('Could not reach backend. Is uvicorn running on :8000?'))
  }, [])

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-100">
      <div className="text-center space-y-2">
        <h1 className="text-2xl font-semibold">Personal AI Document Reader</h1>
        {health && (
          <p className="text-emerald-400">
            Backend status: {health.status} ({health.app_name}, {health.environment})
          </p>
        )}
        {error && <p className="text-red-400">{error}</p>}
        {!health && !error && <p className="text-slate-400">Checking backend...</p>}
      </div>
    </div>
  )
}

export default App
