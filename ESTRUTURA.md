# Estrutura do Projeto DemandAI

## 📁 Organização de Pastas

```
DemandAI/
├── 📂 src/                     # Código fonte principal
│   ├── __init__.py
│   └── app.py                  # Aplicação FastAPI principal
├── 📂 scripts/                 # Scripts de treinamento e análise
│   ├── bombom_lasso_model.py
│   ├── bombom_moranguete_rf_model.py
│   ├── teta_bel_rf_model.py
│   ├── topbel_lasso_model.py
│   ├── topbel_leite_condensado_rf_model.py
│   ├── topbel_ridge_model.py
│   ├── topbel_tradicional_rf_model.py
│   ├── train_all_rf_models.py
│   ├── rf_models_summary.py
│   ├── optimize_rf_models.py
│   ├── advanced_model_optimization.py
│   └── analise_overfitting_completa.py
├── 📂 data/                    # Datasets
│   ├── dataset.csv
│   └── dataset_with_features.csv
├── 📂 models/                  # Modelos treinados (.pkl)
│   ├── bombom_lasso_model.pkl
│   ├── bombom_moranguete_rf_model.pkl
│   ├── teta_bel_rf_model.pkl
│   ├── topbel_lasso_model.pkl
│   ├── topbel_leite_condensado_rf_model.pkl
│   ├── topbel_ridge_conservative_model.pkl
│   └── topbel_tradicional_rf_model.pkl
├── 📂 templates/               # Templates HTML
│   └── form.html
├── 📂 images/                  # Gráficos e visualizações
│   ├── analise_comparativa_modelos.png
│   ├── bombom_moranguete_13g_160un_rf_optimized_analysis.png
│   ├── teta_bel_tradicional_50un_rf_optimized_analysis.png
│   ├── topbel_leite_condensado_50un_rf_optimized_analysis.png
│   └── topbel_tradicional_50un_rf_optimized_analysis.png
├── 📂 analysis/                # Análises e relatórios
│   ├── modelos_comparacao_original_vs_otimizado.csv
│   └── rf_models_optimized_summary.csv
├── 📂 frontend/                # Arquivos frontend (se houver)
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── 📋 requirements.txt
└── 📖 README.md
```

## 🚀 Como Executar

### Com Docker
```bash
docker-compose up --build
```

### Desenvolvimento Local
```bash
# A partir da raiz do projeto
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

## 📝 Mudanças na Estrutura

- **src/**: Código fonte principal da aplicação
- **scripts/**: Scripts de treinamento e análise de modelos
- **data/**: Datasets organizados
- **images/**: Visualizações e gráficos gerados
- **analysis/**: Relatórios e análises comparativas
- **Caminhos atualizados**: Todos os caminhos relativos foram ajustados para a nova estrutura