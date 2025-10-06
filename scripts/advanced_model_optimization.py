#!/usr/bin/env python3
"""
Implementação de estratégias avançadas contra overfitting:
- Feature Selection com Recursive Feature Elimination
- XGBoost com regularização e early stopping
- Ensemble de múltiplos modelos
- Otimização bayesiana de hiperparâmetros com Optuna
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelOptimizer:
    """
    Classe para otimização avançada de modelos contra overfitting
    """
    
    def __init__(self, n_features_select=30, random_state=42):
        self.n_features_select = n_features_select
        self.random_state = random_state
        self.selected_features = None
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        """Carrega e prepara dados"""
        print("📊 Carregando dataset para otimização avançada...")
        df = pd.read_csv('dataset_with_features.csv')
        print(f"✅ Dataset: {df.shape}")
        return df
    
    def feature_selection_analysis(self, X, y, product_name):
        """
        Análise e seleção de features usando múltiplas técnicas
        """
        print(f"🔍 Realizando feature selection para {product_name}...")
        
        # 1. Recursive Feature Elimination com Random Forest
        rf_selector = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
        rfe = RFE(estimator=rf_selector, n_features_to_select=self.n_features_select)
        rfe.fit(X, y)
        rfe_features = X.columns[rfe.support_].tolist()
        
        # 2. SelectKBest com f_regression
        selector = SelectKBest(score_func=f_regression, k=self.n_features_select)
        selector.fit(X, y)
        kbest_features = X.columns[selector.get_support()].tolist()
        
        # 3. Random Forest Feature Importance
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        rf_temp.fit(X, y)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        rf_features = feature_importance.head(self.n_features_select)['feature'].tolist()
        
        # 4. Combinar e ranquear features
        feature_votes = {}
        for feature in X.columns:
            votes = 0
            if feature in rfe_features: votes += 1
            if feature in kbest_features: votes += 1  
            if feature in rf_features: votes += 1
            feature_votes[feature] = votes
        
        # Selecionar features com mais votos
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [f[0] for f in sorted_features[:self.n_features_select]]
        
        print(f"✅ Features selecionadas: {len(self.selected_features)}")
        print(f"   • RFE: {len(rfe_features)} features")
        print(f"   • KBest: {len(kbest_features)} features") 
        print(f"   • RF Importance: {len(rf_features)} features")
        
        return self.selected_features
    
    def optimize_xgboost_optuna(self, X_train, y_train, X_val, y_val):
        """
        Otimização de hiperparâmetros XGBoost com Optuna
        """
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'random_state': self.random_state,
                
                # Hiperparâmetros para otimizar
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                
                # Regularização
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)
        
        print("🔧 Otimizando XGBoost com Optuna...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        
        print(f"✅ Melhor score: {study.best_value:.4f}")
        return study.best_params
    
    def create_ensemble_model(self, X_train, y_train, X_val, y_val, best_xgb_params):
        """
        Cria ensemble de modelos otimizados
        """
        print("🤖 Criando ensemble de modelos...")
        
        # 1. Random Forest otimizado
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features=0.5,
            random_state=self.random_state,
            oob_score=True
        )
        
        # 2. XGBoost otimizado
        xgb_model = xgb.XGBRegressor(**best_xgb_params, random_state=self.random_state)
        
        # 3. Treinar modelos
        rf_model.fit(X_train, y_train)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        return {'rf': rf_model, 'xgb': xgb_model}
    
    def evaluate_ensemble(self, models, X_train, y_train, X_test, y_test):
        """
        Avalia performance do ensemble
        """
        results = {}
        
        for name, model in models.items():
            # Predições
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Métricas
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'overfitting_gap': train_r2 - test_r2,
                'model': model
            }
        
        # Ensemble médio
        ensemble_train_pred = np.mean([model.predict(X_train) for model in models.values()], axis=0)
        ensemble_test_pred = np.mean([model.predict(X_test) for model in models.values()], axis=0)
        
        ensemble_train_r2 = r2_score(y_train, ensemble_train_pred)
        ensemble_test_r2 = r2_score(y_test, ensemble_test_pred)
        
        results['ensemble'] = {
            'train_r2': ensemble_train_r2,
            'test_r2': ensemble_test_r2,
            'train_rmse': np.sqrt(mean_squared_error(y_train, ensemble_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, ensemble_test_pred)),
            'overfitting_gap': ensemble_train_r2 - ensemble_test_r2,
            'predictions': {'train': ensemble_train_pred, 'test': ensemble_test_pred}
        }
        
        return results
    
    def train_advanced_model(self, product_name, df):
        """
        Pipeline completo de treinamento avançado
        """
        print(f"\n🚀 TREINAMENTO AVANÇADO: {product_name}")
        print("=" * 60)
        
        # Filtrar produto
        df_product = df[df['product'] == product_name].copy()
        
        if len(df_product) < 20:
            print(f"⚠️  Poucos dados para {product_name}")
            return None
        
        # Preparar features
        feature_cols = [col for col in df_product.columns 
                       if col not in ['product', 'quantity', 'date']]
        
        X = df_product[feature_cols]
        y = df_product['quantity']
        
        print(f"📊 Dados: {len(df_product)} registros, {len(feature_cols)} features originais")
        
        # Feature Selection
        selected_features = self.feature_selection_analysis(X, y, product_name)
        X_selected = X[selected_features]
        
        # Split temporal: 60% train, 20% validation, 20% test
        n_train = int(0.6 * len(df_product))
        n_val = int(0.2 * len(df_product))
        
        X_train = X_selected.iloc[:n_train]
        X_val = X_selected.iloc[n_train:n_train+n_val]  
        X_test = X_selected.iloc[n_train+n_val:]
        
        y_train = y.iloc[:n_train]
        y_val = y.iloc[n_train:n_train+n_val]
        y_test = y.iloc[n_train+n_val:]
        
        print(f"📈 Split: {len(X_train)} treino, {len(X_val)} validação, {len(X_test)} teste")
        
        # Otimização XGBoost
        best_xgb_params = self.optimize_xgboost_optuna(X_train, y_train, X_val, y_val)
        
        # Criar ensemble
        models = self.create_ensemble_model(X_train, y_train, X_val, y_val, best_xgb_params)
        
        # Avaliar ensemble
        results = self.evaluate_ensemble(models, X_train, y_train, X_test, y_test)
        
        # Exibir resultados
        print(f"\n📊 RESULTADOS COMPARATIVOS:")
        print("-" * 50)
        for name, metrics in results.items():
            if name != 'ensemble':
                print(f"{name.upper()}:")
                print(f"   • R² Treino: {metrics['train_r2']:.4f}")
                print(f"   • R² Teste: {metrics['test_r2']:.4f}")
                print(f"   • Gap: {metrics['overfitting_gap']:.4f}")
                print(f"   • RMSE Teste: {metrics['test_rmse']:.2f}")
        
        print(f"\nENSEMBLE (Média):")
        ensemble_metrics = results['ensemble']
        print(f"   • R² Treino: {ensemble_metrics['train_r2']:.4f}")
        print(f"   • R² Teste: {ensemble_metrics['test_r2']:.4f}")  
        print(f"   • Gap: {ensemble_metrics['overfitting_gap']:.4f}")
        print(f"   • RMSE Teste: {ensemble_metrics['test_rmse']:.2f}")
        
        # Determinar melhor modelo
        best_model_name = min(results.keys(), 
                             key=lambda x: results[x]['overfitting_gap'] if x != 'ensemble' else results[x]['overfitting_gap'])
        
        print(f"\n🏆 Melhor modelo: {best_model_name.upper()}")
        
        # Salvar modelo avançado
        model_data = {
            'ensemble_models': models,
            'selected_features': selected_features,
            'product': product_name,
            'results': results,
            'best_model': best_model_name,
            'best_xgb_params': best_xgb_params
        }
        
        filename = f"models/{product_name.lower().replace(' ', '_')}_advanced_model.pkl"
        joblib.dump(model_data, filename)
        print(f"💾 Modelo avançado salvo: {filename}")
        
        # Criar visualizações
        self.create_advanced_plots(results, product_name)
        
        return model_data
    
    def create_advanced_plots(self, results, product_name):
        """
        Cria visualizações avançadas
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análise Avançada - {product_name}', fontsize=16, fontweight='bold')
        
        # 1. Comparação de R² Score
        models = [k for k in results.keys() if k != 'ensemble']
        train_r2 = [results[k]['train_r2'] for k in models]
        test_r2 = [results[k]['test_r2'] for k in models]
        
        x_pos = np.arange(len(models))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x_pos - width/2, train_r2, width, label='Treino', alpha=0.8)
        bars2 = axes[0, 0].bar(x_pos + width/2, test_r2, width, label='Teste', alpha=0.8)
        
        axes[0, 0].set_title('R² Score: Treino vs Teste')
        axes[0, 0].set_xlabel('Modelos')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([m.upper() for m in models])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Gap de Overfitting
        gaps = [results[k]['overfitting_gap'] for k in models]
        colors = ['green' if gap < 0.15 else 'orange' if gap < 0.25 else 'red' for gap in gaps]
        
        bars = axes[0, 1].bar(models, gaps, color=colors, alpha=0.7)
        axes[0, 1].set_title('Gap de Overfitting por Modelo')
        axes[0, 1].set_ylabel('Gap (R² Treino - R² Teste)')
        axes[0, 1].axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='Limite Moderado')
        axes[0, 1].axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='Limite Alto')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Adicionar valores
        for bar, gap in zip(bars, gaps):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{gap:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. RMSE Comparison
        test_rmse = [results[k]['test_rmse'] for k in models]
        axes[1, 0].bar(models, test_rmse, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('RMSE de Teste')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Ensemble Performance
        ensemble_data = {
            'Métrica': ['R² Treino', 'R² Teste', 'Gap Overfitting'],
            'Valor': [
                results['ensemble']['train_r2'],
                results['ensemble']['test_r2'],
                results['ensemble']['overfitting_gap']
            ]
        }
        
        colors_ensemble = ['lightblue', 'lightgreen', 'salmon']
        bars = axes[1, 1].bar(ensemble_data['Métrica'], ensemble_data['Valor'], 
                             color=colors_ensemble, alpha=0.8)
        axes[1, 1].set_title('Performance do Ensemble')
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Adicionar valores
        for bar, value in zip(bars, ensemble_data['Valor']):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        filename = f"{product_name.lower().replace(' ', '_')}_advanced_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 Visualizações salvas: {filename}")
        plt.close()

def main():
    """
    Executa otimização avançada para todos os produtos
    """
    print("🚀 OTIMIZAÇÃO AVANÇADA CONTRA OVERFITTING")
    print("🎯 Features: Selection + Ensemble + XGBoost + Optuna")
    print("=" * 70)
    
    optimizer = AdvancedModelOptimizer(n_features_select=25)
    df = optimizer.load_and_prepare_data()
    
    products = [
        'BOMBOM MORANGUETE 13G 160UN',
        'TETA BEL TRADICIONAL 50UN',
        'TOPBEL LEITE CONDENSADO 50UN', 
        'TOPBEL TRADICIONAL 50UN'
    ]
    
    summary_results = []
    
    for product in products:
        try:
            result = optimizer.train_advanced_model(product, df)
            if result:
                best_model = result['best_model']
                best_metrics = result['results'][best_model]
                
                summary_results.append({
                    'Produto': product,
                    'Melhor Modelo': best_model.upper(),
                    'R² Treino': best_metrics['train_r2'],
                    'R² Teste': best_metrics['test_r2'],
                    'Gap Overfitting': best_metrics['overfitting_gap'],
                    'RMSE Teste': best_metrics['test_rmse'],
                    'Status': '✅ Baixo' if best_metrics['overfitting_gap'] < 0.15 
                             else '🟡 Moderado' if best_metrics['overfitting_gap'] < 0.25 
                             else '🔴 Alto'
                })
                
        except Exception as e:
            print(f"❌ Erro no produto {product}: {e}")
    
    # Resumo final
    if summary_results:
        print(f"\n" + "=" * 80)
        print("📊 RESUMO FINAL - MODELOS AVANÇADOS OTIMIZADOS")
        print("=" * 80)
        
        df_summary = pd.DataFrame(summary_results)
        print(df_summary.to_string(index=False, float_format='%.4f'))
        
        df_summary.to_csv('modelos_avancados_resumo.csv', index=False)
        print(f"\n💾 Resumo salvo: modelos_avancados_resumo.csv")
        
        # Estatísticas finais
        avg_gap = df_summary['Gap Overfitting'].mean()
        models_low_overfitting = sum(df_summary['Gap Overfitting'] < 0.15)
        
        print(f"\n🎯 ESTATÍSTICAS FINAIS:")
        print(f"   • Gap médio de overfitting: {avg_gap:.4f}")
        print(f"   • Modelos com baixo overfitting: {models_low_overfitting}/4")
        print(f"   • Melhoria esperada: Feature selection + Ensemble + XGBoost")

if __name__ == "__main__":
    main()