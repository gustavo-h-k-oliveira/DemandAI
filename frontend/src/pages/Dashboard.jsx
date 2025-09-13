import React, { useEffect, useState } from 'react';
import { ChartContainer } from "../components/ChartContainer.jsx";
import { MagnifyingGlass } from "phosphor-react";

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function Dashboard() {
    const [status, setStatus] = useState("carregando...");
    const [summary, setSummary] = useState(null);

    useEffect(() => {
        fetch(`${API_URL}/health`)
            .then(r => r.json())
            .then(d => setStatus(d.status || JSON.stringify(d)))
            .catch(() => setStatus("erro"));

        fetch(`${API_URL}/api/dashboard/summary`)
            .then((response) => response.json())
            .then((data) => {
                setSummary(data);
                setStatus("carregado");
            })
            .catch((error) => {
                console.error("Erro ao buscar dados:", error);
                setStatus("erro");
            });
    }, []);

    return (
        <>
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
    
            <div className="p-6">
                <h1 className="text-3xl font-bold mb-6">Dashboard</h1>
                {summary && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="p-4 bg-white rounded-lg shadow">
                            <h3 className="font-semibold text-gray-600">
                                Total de Predições
                            </h3>
                            <p className="text-2xl font-bold">
                                {summary.total_predictions}
                            </p>
                        </div>
                        <div className="p-4 bg-white rounded-lg shadow">
                            <h3 className="font-semibold text-gray-600">Usuário</h3>
                            <p className="text-lg">{summary.user_email}</p>
                        </div>
                    </div>
                )}
            </div>
        </>
    );
}
