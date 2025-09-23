import { useState, useMemo } from "react";

export function Prediction() {
  const API_BASE = useMemo(() => import.meta.env.VITE_API_URL || "http://localhost:8000", []);

  const [year, setYear] = useState(new Date().getFullYear());
  const [month, setMonth] = useState(new Date().getMonth() + 1);
  const [campaign, setCampaign] = useState(0);
  const [seasonality, setSeasonality] = useState(0);

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setPrediction(null);

    const y = Number(year);
    const m = Number(month);
    const c = Number(campaign);
    const s = Number(seasonality);

    if (!Number.isFinite(y) || y < 2000 || y > 2100) {
      setError("Ano inválido.");
      return;
    }
    if (!Number.isFinite(m) || m < 1 || m > 12) {
      setError("Mês deve ser entre 1 e 12.");
      return;
    }
    if (![0, 1].includes(c)) {
      setError("Campanha deve ser 0 ou 1.");
      return;
    }
    if (![0, 1].includes(s)) {
      setError("Sazonalidade deve ser 0 ou 1.");
      return;
    }

    const payload = {
      // Ordem das features enviada ao backend
      features: [y, m, c, s],
    };

    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data?.detail || `Erro ${res.status}`);
      }

      const data = await res.json();
      setPrediction(data.prediction);
    } catch (err) {
      setError(err.message || "Erro ao chamar a API.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 520, margin: "2rem auto", padding: "1rem" }}>
      <h1>Predição</h1>
      <p>Informe as features principais para prever a quantidade.</p>

      <form onSubmit={onSubmit} style={{ display: "grid", gap: "0.75rem", marginTop: "1rem" }}>
        <label>
          Ano
          <input
            type="number"
            value={year}
            onChange={(e) => setYear(e.target.value)}
            min={2000}
            max={2100}
            required
            style={{ width: "100%", padding: "0.5rem" }}
          />
        </label>

        <label>
          Mês (1-12)
          <input
            type="number"
            value={month}
            onChange={(e) => setMonth(e.target.value)}
            min={1}
            max={12}
            required
            style={{ width: "100%", padding: "0.5rem" }}
          />
        </label>

        <label>
          Campanha
          <select
            value={campaign}
            onChange={(e) => setCampaign(Number(e.target.value))}
            style={{ width: "100%", padding: "0.5rem" }}
          >
            <option value={0}>0 - Não</option>
            <option value={1}>1 - Sim</option>
          </select>
        </label>

        <label>
          Sazonalidade
          <select
            value={seasonality}
            onChange={(e) => setSeasonality(Number(e.target.value))}
            style={{ width: "100%", padding: "0.5rem" }}
          >
            <option value={0}>0 - Baixa/Normal</option>
            <option value={1}>1 - Alta</option>
          </select>
        </label>

        <button type="submit" disabled={loading} style={{ padding: "0.75rem", cursor: "pointer" }}>
          {loading ? "Calculando..." : "Prever"}
        </button>
      </form>

      {error && (
        <div style={{ color: "white", background: "#d9534f", padding: "0.5rem", marginTop: "1rem" }}>
          {error}
        </div>
      )}

      {prediction !== null && !error && (
        <div style={{ color: "#155724", background: "#d4edda", padding: "0.5rem", marginTop: "1rem" }}>
          Predição: <strong>{prediction}</strong>
        </div>
      )}
    </div>
  );
}