# 🌳 Modelos Random Forest - DemandAI

Este documento descreve os modelos Random Forest criados para previsão de demanda dos 4 produtos da empresa.

## 📊 Resumo dos Modelos

Foram criados 4 modelos Random Forest individuais, um para cada produto:

1. **BOMBOM MORANGUETE 13G 160UN**
2. **TETA BEL TRADICIONAL 50UN**  
3. **TOPBEL LEITE CONDENSADO 50UN**
4. **TOPBEL TRADICIONAL 50UN**

## 🎯 Configuração do Treinamento

- **Período de Treino**: 2021-2023 (36 registros por produto)
- **Período de Teste**: 2024-2025 (20 registros por produto)
- **Total de Features**: 76 features por modelo
- **Algoritmo**: Random Forest Regressor com Grid Search
- **Validação Cruzada**: 5-fold CV

## 🏆 Desempenho dos Modelos

### Ranking por R² no Conjunto de Teste

| Posição | Produto | R² Teste | RMSE Teste | MAE Teste |
|---------|---------|----------|------------|-----------|
| 1º | TOPBEL TRADICIONAL | 0.7767 | 3088.92 | 2412.88 |
| 2º | TOPBEL LEITE CONDENSADO | 0.7499 | 656.47 | 543.53 |
| 3º | BOMBOM MORANGUETE | 0.7183 | 2252.47 | 1815.22 |
| 4º | TETA BEL TRADICIONAL | 0.6238 | 3094.86 | 2662.03 |

### Estatísticas Gerais

- **R² Médio**: 0.7172
- **R² Mediano**: 0.7341
- **Todos os modelos**: R² > 0.6 (desempenho aceitável)
- **3 de 4 modelos**: R² > 0.7 (bom desempenho)

## 🔧 Hiperparâmetros Otimizados

| Produto | N_Estimators | Max_Depth | Min_Samples_Split | Min_Samples_Leaf |
|---------|--------------|-----------|-------------------|------------------|
| BOMBOM MORANGUETE | 200 | 10 | 2 | 1 |
| TETA BEL TRADICIONAL | 100 | 10 | 2 | 2 |
| TOPBEL LEITE CONDENSADO | 200 | 10 | 5 | 1 |
| TOPBEL TRADICIONAL | 100 | 10 | 2 | 2 |

## 🎯 Features Mais Importantes

| Produto | Feature Principal | Importância |
|---------|-------------------|-------------|
| BOMBOM MORANGUETE | quantity_ewm3 | 0.4732 |
| TETA BEL TRADICIONAL | quantity_ewm3 | 0.4530 |
| TOPBEL LEITE CONDENSADO | quantity_max3 | 0.3584 |
| TOPBEL TRADICIONAL | quantity_ewm3 | 0.3681 |

### Padrão Observado
- **quantity_ewm3** (média móvel exponencial de 3 períodos) é a feature mais importante em 3 dos 4 modelos
- Features relacionadas a **médias móveis** e **lags** são cruciais para predição
- **Sazonalidade** e **tendências** também têm papel importante

## 📁 Arquivos Gerados

Para cada produto, foram gerados os seguintes arquivos na pasta `models/`:

### Modelos Treinados
- `{produto}_rf_model.pkl` - Modelo serializado
- `{produto}_rf_model_info.json` - Metadados e métricas
- `{produto}_rf_analysis.png` - Visualizações individuais

### Arquivos Comparativos
- `rf_models_comparison.png` - Comparação visual entre modelos
- `rf_models_summary.csv` - Resumo em CSV

## 🚀 Como Usar os Modelos

### 1. Carregar um Modelo

```python
import joblib
import pandas as pd

# Carregar modelo
modelo = joblib.load('models/bombom_moranguete_rf_model.pkl')

# Carregar dados de entrada (com as mesmas 76 features do treinamento)
X_new = pd.read_csv('novos_dados.csv')

# Fazer predição
previsao = modelo.predict(X_new)
```

### 2. Executar Todos os Modelos

```python
# Executar script principal que treina todos os modelos
python train_all_rf_models.py

# Gerar resumo comparativo
python rf_models_summary.py
```

### 3. Treinar Modelo Individual

```python
# Treinar apenas um modelo específico
python bombom_moranguete_rf_model.py
python teta_bel_rf_model.py
python topbel_leite_condensado_rf_model.py  
python topbel_tradicional_rf_model.py
```

## 📋 Features Utilizadas

Os modelos utilizam 76 features, incluindo:

### Temporais
- `year`, `month`, `quarter`
- `month_sin`, `month_cos`, `quarter_sin`, `quarter_cos`

### Lags e Médias Móveis
- `quantity_lag1`, `quantity_lag2`, ..., `quantity_lag12`
- `quantity_ma3`, `quantity_ma6`, `quantity_ma9`, `quantity_ma12`
- `quantity_ewm3`, `quantity_ewm6`, `quantity_ewm12`

### Estatísticas Descritivas
- `quantity_std3`, `quantity_std6`, `quantity_std9`, `quantity_std12`
- `quantity_min3`, `quantity_max3`, `quantity_volatility3`

### Sazonalidade e Campanhas
- `campaign`, `seasonality`
- `is_high_season`, `is_low_season`, `is_holiday_season`
- `is_summer`, `is_winter`

### Tendências e Variações
- `trend`, `trend_norm`, `trend_squared`, `trend_cubed`
- `quantity_pct_change`, `quantity_diff`
- `yoy_growth`, `yoy_diff`

## 💡 Conclusões e Recomendações

### ✅ Pontos Positivos
- **Todos os modelos** apresentam desempenho aceitável (R² > 0.6)
- **75% dos modelos** têm bom desempenho (R² > 0.7)
- **Hiperparâmetros otimizados** via Grid Search
- **Features engineered** bem construídas
- **Modelos prontos para produção**

### 📈 Oportunidades de Melhoria
- **TETA BEL** apresenta menor desempenho (R² = 0.6238)
- Considerar **ensemble methods** combinando múltiplos algoritmos
- Adicionar **features externas** (feriados, eventos, economia)
- Implementar **cross-validation temporal** específica para séries temporais

### 🎯 Próximos Passos
1. **Deploy em produção** dos modelos
2. **Monitoramento contínuo** do desempenho
3. **Retreinamento periódico** com novos dados
4. **A/B testing** com predições em produção
5. **Análise de drift** nas features e target

## 📊 Tempo de Execução

- **Tempo total**: ~7 minutos (427 segundos)
- **Por modelo**: ~1.7 minutos em média
- **Grid Search**: 540 combinações testadas por modelo

## 🔄 Versionamento

- **Versão**: 1.0
- **Data**: 2025-09-24
- **Python**: 3.12
- **Scikit-learn**: 1.7.0

---

*Os modelos foram desenvolvidos usando as melhores práticas de ML e estão prontos para uso em produção. Para dúvidas ou sugestões, consulte a documentação técnica ou entre em contato com a equipe de Data Science.*