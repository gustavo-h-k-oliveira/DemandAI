import React, { useEffect, useState } from "react";

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
    <div style={{ fontFamily: "system-ui", padding: 24 }}>
      <h1>React + Vite</h1>
      <p>Backend health: {status}</p>
      <p>API_URL: {API_URL}</p>
    </div>
  );
}