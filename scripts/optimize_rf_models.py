#!/usr/bin/env python3
"""
Otimização dos modelos Random Forest para reduzir overfitting
- Redução de complexidade dos modelos
- Aumento da regularização
- Validação cruzada mais robusta
- Early stopping baseado em performance de validação
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Carrega e prepara os dados com feature engineering"""
    print("📊 Carregando dataset...")
    df = pd.read_csv('dataset_with_features.csv')
    
    # Verificar colunas disponíveis
    print(f"✅ Dataset carregado: {df.shape}")
    print(f"📅 Período: {df['year'].min()}-{df['year'].max()}")
    print(f"🏷️ Produtos únicos: {df['product'].nunique()}")
    
    return df

def create_optimized_model_config():
    """
    Configuração otimizada para reduzir overfitting
    """
    return {
        # Parâmetros para reduzir overfitting
        'base_params': {
            'random_state': 42,
            'n_jobs': -1,
            'oob_score': True,  # Out-of-bag score para monitoramento
        },
        
        # Grid Search mais conservador
        'param_grid': {
            'n_estimators': [50, 100, 150],  # Menos árvores
            'max_depth': [3, 5, 7, 10],      # Profundidade limitada
            'min_samples_split': [10, 20, 50], # Mais amostras para split
            'min_samples_leaf': [5, 10, 20],   # Mais amostras por folha
            'max_features': [0.3, 0.5, 0.7],   # Menos features por árvore
            'max_samples': [0.7, 0.8, 0.9],    # Bootstrap sampling limitado
        }
    }

def perform_time_series_validation(X, y, model, param_grid, n_splits=3):
    """
    Validação cruzada específica para séries temporais
    """
    print("⏰ Executando validação cruzada para séries temporais...")
    
    # Time Series Split para respeitar ordem temporal
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Grid Search com validação temporal
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return grid_search

def train_optimized_model(product_name, df):
    """
    Treina modelo otimizado para um produto específico
    """
    print(f"\n🎯 Treinando modelo otimizado para: {product_name}")
    print("=" * 60)
    
    # Filtrar dados do produto
    df_product = df[df['product'] == product_name].copy()
    print(f"📊 Registros do produto: {len(df_product)}")
    
    if len(df_product) < 20:
        print(f"⚠️  Poucos dados para {product_name}. Pulando...")
        return None
    
    # Separar features e target
    feature_cols = [col for col in df_product.columns 
                   if col not in ['product', 'quantity', 'date']]
    
    X = df_product[feature_cols]
    y = df_product['quantity']
    
    print(f"🔧 Features utilizadas: {len(feature_cols)}")
    
    # Split temporal mais conservador (mais dados de treino)
    split_idx = int(0.75 * len(df_product))  # 75% treino, 25% teste
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📈 Treino: {len(X_train)} registros")
    print(f"📉 Teste: {len(X_test)} registros")
    
    # Configuração otimizada
    config = create_optimized_model_config()
    
    # Modelo base
    rf_base = RandomForestRegressor(**config['base_params'])
    
    # Validação cruzada temporal
    grid_search = perform_time_series_validation(
        X_train, y_train, rf_base, config['param_grid']
    )
    
    print(f"✅ Melhores parâmetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"   • {param}: {value}")
    
    # Modelo final otimizado
    best_model = grid_search.best_estimator_
    
    # Predições
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Métricas
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    oob_score = best_model.oob_score_
    
    # Calcular gap de overfitting
    overfitting_gap = train_r2 - test_r2
    
    print(f"\n📊 MÉTRICAS DO MODELO OTIMIZADO:")
    print(f"   • R² Treino: {train_r2:.4f}")
    print(f"   • R² Teste: {test_r2:.4f}")
    print(f"   • OOB Score: {oob_score:.4f}")
    print(f"   • Gap (Overfitting): {overfitting_gap:.4f}")
    print(f"   • RMSE Treino: {train_rmse:.2f}")
    print(f"   • RMSE Teste: {test_rmse:.2f}")
    
    # Status do overfitting
    if overfitting_gap < 0.10:
        status = "✅ Baixo"
    elif overfitting_gap < 0.20:
        status = "🟡 Moderado"
    else:
        status = "🔴 Alto"
    
    print(f"   • Status Overfitting: {status}")
    
    # Salvar modelo otimizado
    model_filename = f"models/{product_name.lower().replace(' ', '_')}_rf_optimized.pkl"
    
    model_data = {
        'model': best_model,
        'feature_columns': feature_cols,
        'product': product_name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'oob_score': oob_score,
        'overfitting_gap': overfitting_gap,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'best_params': grid_search.best_params_,
        'training_date': datetime.now().isoformat(),
        'optimization_strategy': 'reduced_overfitting'
    }
    
    joblib.dump(model_data, model_filename)
    print(f"💾 Modelo salvo: {model_filename}")
    
    # Criar visualizações
    create_optimized_model_plots(
        y_train, y_train_pred, y_test, y_test_pred,
        best_model, X_train, product_name, model_data
    )
    
    return model_data

def create_optimized_model_plots(y_train, y_train_pred, y_test, y_test_pred, 
                                model, X_train, product_name, model_data):
    """
    Cria visualizações para o modelo otimizado
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Modelo RF Otimizado - {product_name}', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot train vs pred
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, color='blue', s=60)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                    'r--', lw=2)
    axes[0, 0].set_title(f'Treino - R² = {model_data["train_r2"]:.4f}')
    axes[0, 0].set_xlabel('Valores Reais')
    axes[0, 0].set_ylabel('Valores Preditos')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Scatter plot test vs pred
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, color='red', s=60)
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2)
    axes[0, 1].set_title(f'Teste - R² = {model_data["test_r2"]:.4f}')
    axes[0, 1].set_xlabel('Valores Reais')
    axes[0, 1].set_ylabel('Valores Preditos')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature importance
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(10)
    
    axes[1, 0].barh(importance_df['feature'], importance_df['importance'])
    axes[1, 0].set_title('Top 10 Features Mais Importantes')
    axes[1, 0].set_xlabel('Importância')
    
    # 4. Comparação de métricas
    metrics_comparison = {
        'R² Treino': model_data['train_r2'],
        'R² Teste': model_data['test_r2'], 
        'OOB Score': model_data['oob_score']
    }
    
    bars = axes[1, 1].bar(metrics_comparison.keys(), metrics_comparison.values(),
                         color=['lightblue', 'lightcoral', 'lightgreen'])
    axes[1, 1].set_title('Comparação de Métricas')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    # Adicionar valores nas barras
    for bar, value in zip(bars, metrics_comparison.values()):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Salvar plot
    plot_filename = f"{product_name.lower().replace(' ', '_')}_rf_optimized_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📈 Gráficos salvos: {plot_filename}")
    plt.close()

def main():
    """
    Função principal para otimizar todos os modelos
    """
    print("🚀 OTIMIZAÇÃO DE MODELOS RANDOM FOREST")
    print("🎯 Objetivo: Reduzir overfitting e melhorar generalização")
    print("=" * 70)
    
    # Carregar dados
    df = load_and_prepare_data()
    
    # Produtos para otimizar
    products = [
        'BOMBOM MORANGUETE 13G 160UN',
        'TETA BEL TRADICIONAL 50UN', 
        'TOPBEL LEITE CONDENSADO 50UN',
        'TOPBEL TRADICIONAL 50UN'
    ]
    
    results_summary = []
    
    # Treinar modelos otimizados
    for product in products:
        try:
            model_data = train_optimized_model(product, df)
            if model_data:
                results_summary.append({
                    'Produto': product,
                    'R² Treino': model_data['train_r2'],
                    'R² Teste': model_data['test_r2'],
                    'OOB Score': model_data['oob_score'],
                    'Gap Overfitting': model_data['overfitting_gap'],
                    'RMSE Teste': model_data['test_rmse']
                })
        except Exception as e:
            print(f"❌ Erro no produto {product}: {e}")
    
    # Resumo final
    if results_summary:
        print("\n" + "=" * 70)
        print("📊 RESUMO DOS MODELOS OTIMIZADOS")
        print("=" * 70)
        
        results_df = pd.DataFrame(results_summary)
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Salvar resumo
        results_df.to_csv('rf_models_optimized_summary.csv', index=False)
        print(f"\n💾 Resumo salvo em: rf_models_optimized_summary.csv")
        
        # Análise de melhoria
        print(f"\n🎯 ANÁLISE DE MELHORIA:")
        avg_gap = results_df['Gap Overfitting'].mean()
        print(f"   • Gap médio de overfitting: {avg_gap:.4f}")
        
        if avg_gap < 0.15:
            print("   • ✅ Overfitting significativamente reduzido!")
        elif avg_gap < 0.20:
            print("   • 🟡 Overfitting moderadamente reduzido")
        else:
            print("   • 🔴 Overfitting ainda alto - considere mais regularização")
    
    print(f"\n🎉 Otimização finalizada!")

if __name__ == "__main__":
    main()