import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def main():
    print("="*60)
    print("MODELO RANDOM FOREST - BOMBOM MORANGUETE 13G 160UN")
    print("="*60)
    
    # Carregar os dados
    print("Carregando dados...")
    df = pd.read_csv('/workspaces/DemandAI/dataset_with_features.csv')
    
    # Filtrar apenas para o produto BOMBOM MORANGUETE
    produto_data = df[df['product'] == 'BOMBOM MORANGUETE 13G 160UN'].copy()
    print(f"Total de registros para BOMBOM MORANGUETE: {len(produto_data)}")
    
    # Separar dados de treino (2021-2023) e teste (2024-2025)
    train_data = produto_data[produto_data['year'].isin([2021, 2022, 2023])].copy()
    test_data = produto_data[produto_data['year'].isin([2024, 2025])].copy()
    
    print(f"Dados de treino: {len(train_data)} registros (anos 2021-2023)")
    print(f"Dados de teste: {len(test_data)} registros (anos 2024-2025)")
    
    # Definir features e target
    # Remover colunas não necessárias para o modelo
    features_to_remove = ['product', 'quantity']  # quantity é o target
    
    feature_columns = [col for col in train_data.columns if col not in features_to_remove]
    
    X_train = train_data[feature_columns]
    y_train = train_data['quantity']
    X_test = test_data[feature_columns]
    y_test = test_data['quantity']
    
    print(f"Número de features: {len(feature_columns)}")
    print("Features utilizadas:")
    for i, feature in enumerate(feature_columns[:10], 1):  # Mostrar apenas as primeiras 10
        print(f"  {i:2d}. {feature}")
    if len(feature_columns) > 10:
        print(f"  ... e mais {len(feature_columns) - 10} features")
    
    # Configurar o modelo Random Forest
    print("\nConfigurando modelo Random Forest...")
    
    # Grid search para otimização dos hiperparâmetros
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [42]
    }
    
    # Modelo base
    rf_base = RandomForestRegressor(random_state=42)
    
    print("Executando Grid Search para otimização dos hiperparâmetros...")
    print("Isso pode levar alguns minutos...")
    
    # TimeSeriesSplit para validação cruzada respeitando ordem temporal
    # n_splits=5 cria 5 folds temporais onde cada fold usa dados anteriores para treino
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Grid search com validação cruzada temporal
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=tscv,  # TimeSeriesSplit para séries temporais
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Usar todos os cores disponíveis
        verbose=1
    )
    
    # Treinar o modelo
    grid_search.fit(X_train, y_train)
    
    # Melhor modelo
    best_rf = grid_search.best_estimator_
    
    print(f"\nMelhores hiperparâmetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nMelhor score de validação cruzada: {-grid_search.best_score_:.2f}")
    
    # Fazer predições
    print("\nFazendo predições...")
    y_train_pred = best_rf.predict(X_train)
    y_test_pred = best_rf.predict(X_test)
    
    # Calcular métricas
    print("\n" + "="*40)
    print("MÉTRICAS DE DESEMPENHO")
    print("="*40)
    
    # Métricas de treino
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"\nTREINO (2021-2023):")
    print(f"  MSE: {train_mse:,.2f}")
    print(f"  RMSE: {np.sqrt(train_mse):,.2f}")
    print(f"  MAE: {train_mae:,.2f}")
    print(f"  R²: {train_r2:.4f}")
    
    # Métricas de teste
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTESTE (2024-2025):")
    print(f"  MSE: {test_mse:,.2f}")
    print(f"  RMSE: {np.sqrt(test_mse):,.2f}")
    print(f"  MAE: {test_mae:,.2f}")
    print(f"  R²: {test_r2:.4f}")
    
    # Importância das features
    print(f"\n" + "="*40)
    print("TOP 15 FEATURES MAIS IMPORTANTES")
    print("="*40)
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (idx, row) in enumerate(feature_importance.head(15).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:<25} {row['importance']:.4f}")
    
    # Salvar o modelo
    model_filename = '/workspaces/DemandAI/models/bombom_moranguete_rf_model.pkl'
    print(f"\nSalvando modelo em: {model_filename}")
    joblib.dump(best_rf, model_filename)
    
    # Salvar também as informações do modelo
    model_info = {
        'model_type': 'RandomForestRegressor',
        'product': 'BOMBOM MORANGUETE 13G 160UN',
        'training_period': '2021-2023',
        'test_period': '2024-2025',
        'cross_validation': 'TimeSeriesSplit(n_splits=5)',
        'best_params': grid_search.best_params_,
        'features': feature_columns,
        'train_metrics': {
            'mse': train_mse,
            'rmse': np.sqrt(train_mse),
            'mae': train_mae,
            'r2': train_r2
        },
        'test_metrics': {
            'mse': test_mse,
            'rmse': np.sqrt(test_mse),
            'mae': test_mae,
            'r2': test_r2
        },
        'feature_importance': feature_importance.to_dict('records'),
        'created_at': datetime.now().isoformat()
    }
    
    import json
    with open('/workspaces/DemandAI/models/bombom_moranguete_rf_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Criar gráfico comparativo
    print("\nCriando visualizações...")
    
    # Preparar dados para visualização
    train_results = pd.DataFrame({
        'Real': y_train,
        'Predito': y_train_pred,
        'Período': 'Treino'
    })
    
    test_results = pd.DataFrame({
        'Real': y_test,
        'Predito': y_test_pred,
        'Período': 'Teste'
    })
    
    all_results = pd.concat([train_results, test_results])
    
    # Criar gráficos
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Modelo Random Forest - BOMBOM MORANGUETE 13G 160UN', fontsize=16)
    
    # Gráfico 1: Real vs Predito (Treino)
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.7, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Valores Reais')
    axes[0, 0].set_ylabel('Valores Preditos')
    axes[0, 0].set_title(f'Treino - R² = {train_r2:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Real vs Predito (Teste)
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.7, color='red')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Valores Reais')
    axes[0, 1].set_ylabel('Valores Preditos')
    axes[0, 1].set_title(f'Teste - R² = {test_r2:.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Série temporal
    train_dates = pd.date_range(start='2021-01', periods=len(y_train), freq='M')
    test_dates = pd.date_range(start='2024-01', periods=len(y_test), freq='M')
    
    axes[1, 0].plot(train_dates, y_train.values, label='Real', color='blue', linewidth=2)
    axes[1, 0].plot(train_dates, y_train_pred, label='Predito', color='lightblue', linewidth=2)
    axes[1, 0].set_title('Série Temporal - Período de Treino')
    axes[1, 0].set_ylabel('Quantidade')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Gráfico 4: Top 10 Features Importantes
    top_features = feature_importance.head(10)
    axes[1, 1].barh(range(len(top_features)), top_features['importance'].values)
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['feature'].values)
    axes[1, 1].set_xlabel('Importância')
    axes[1, 1].set_title('Top 10 Features Mais Importantes')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspaces/DemandAI/models/bombom_moranguete_rf_analysis.png', dpi=300, bbox_inches='tight')
    print("Gráficos salvos em: /workspaces/DemandAI/models/bombom_moranguete_rf_analysis.png")
    
    print("\n" + "="*60)
    print("MODELO TREINADO COM SUCESSO!")
    print("="*60)
    print(f"Modelo salvo: {model_filename}")
    print(f"Informações salvas: /workspaces/DemandAI/models/bombom_moranguete_rf_model_info.json")
    print(f"Visualizações salvas: /workspaces/DemandAI/models/bombom_moranguete_rf_analysis.png")

if __name__ == "__main__":
    main()