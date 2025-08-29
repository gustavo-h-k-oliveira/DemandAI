import React, { useEffect, useState } from "react";
import { ChartContainer } from "./components/ChartContainer.jsx";
import { MagnifyingGlass } from "phosphor-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [status, setStatus] = useState("carregando...");
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.json())
      .then(d => setStatus(d.status || JSON.stringify(d)))
      .catch(() => setStatus("erro"));
  }, []);
  return (
    <div className="min-h-screen p-6">
      <header className="flex items-center gap-3 mb-6">
        <h1 className="text-2xl font-bold">DemandAI</h1>
        <span className="text-sm text-muted-foreground">API_URL: {API_URL}</span>
      </header>

      <section className="grid md:grid-cols-2 gap-6">
        <div className="rounded-lg border p-4">
          <h2 className="font-semibold mb-2 flex items-center gap-2">
            <MagnifyingGlass size={18} /> Backend health
          </h2>
          <p className="text-sm">{status}</p>
        </div>

        <div className="rounded-lg border p-4">
          <h2 className="font-semibold mb-2">Previsão (exemplo)</h2>
          <ChartContainer />
        </div>
      </section>
    </div>
  );
}