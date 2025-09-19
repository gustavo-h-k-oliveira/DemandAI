import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    df = pd.read_csv('dataset.csv', sep=',', encoding="latin1")
    df = df.dropna()
    df = df.drop('PESOL', axis=1)
    df.columns = ['product', 'year', 'month', 'campaign', 'seasonality', 'quantity']
    return df

def create_anti_overfitting_features(df):
    """Create features designed to prevent overfitting"""
    df_sorted = df.sort_values(['product', 'year', 'month']).reset_index(drop=True)
    all_data = []
    
    for product in df_sorted['product'].unique():
        product_data = df_sorted[df_sorted['product'] == product].copy().reset_index(drop=True)
        
        # ONLY ESSENTIAL FEATURES (reduce complexity)
        # Simple lag features (only 1 period to avoid data leakage)
        product_data['quantity_lag1'] = product_data['quantity'].shift(1)
        
        # Conservative rolling average (longer window)
        product_data['quantity_ma6'] = product_data['quantity'].rolling(window=6, min_periods=3).mean()
        
        # Basic seasonal features
        product_data['month_sin'] = np.sin(2 * np.pi * product_data['month'] / 12)
        product_data['month_cos'] = np.cos(2 * np.pi * product_data['month'] / 12)
        
        # Simple trend (normalized)
        product_data['time_trend'] = (np.arange(len(product_data)) / len(product_data))
        
        all_data.append(product_data)
    
    enhanced_df = pd.concat(all_data, ignore_index=True)
    enhanced_df = enhanced_df.fillna(method='bfill').fillna(0)
    return enhanced_df

def train_regularized_models():
    """Train models with strong regularization parameters"""
    
    print("="*80)
    print("ANTI-OVERFITTING MODEL TRAINING")
    print("="*80)
    
    df = load_and_preprocess_data()
    enhanced_df = create_anti_overfitting_features(df)
    
    # REDUCED FEATURE SET (only 7 features instead of 16)
    feature_columns = [
        'month', 'campaign', 'seasonality',
        'quantity_lag1', 'quantity_ma6',
        'month_sin', 'month_cos', 'time_trend'
    ]
    
    products = enhanced_df['product'].unique()
    results = {}
    
    for product in products:
        print(f"\n{'='*50}")
        print(f"TRAINING: {product}")
        print(f"{'='*50}")
        
        product_data = enhanced_df[enhanced_df['product'] == product].copy()
        
        if len(product_data) < 10:
            continue
            
        X = product_data[feature_columns]
        y = product_data['quantity']
        
        # CONSERVATIVE TRAIN/TEST SPLIT (70/30 instead of 80/20)
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            # STRONG REGULARIZATION
            'Ridge_Strong': Ridge(alpha=10.0, random_state=42),  # 1000x stronger than before
            'Lasso_Strong': Lasso(alpha=5.0, random_state=42),   # Strong feature selection
            'RF_Conservative': RandomForestRegressor(
                n_estimators=50,      # Fewer trees
                max_depth=3,          # Very shallow
                min_samples_split=5,  # Higher minimum
                min_samples_leaf=3,   # Higher minimum
                max_features=0.5,     # Use only half features
                random_state=42
            )
        }
        
        product_results = {}
        
        for model_name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # OVERFITTING DETECTION
            r2_gap = train_r2 - test_r2
            overfitting_status = "HIGH" if r2_gap > 0.2 else "MEDIUM" if r2_gap > 0.1 else "LOW"
            
            print(f"\n{model_name}:")
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²:  {test_r2:.4f}")
            print(f"  R² Gap:   {r2_gap:.4f}")
            print(f"  Overfitting Risk: {overfitting_status}")
            print(f"  Test MAE: {test_mae:.2f}")
            print(f"  Test RMSE: {test_rmse:.2f}")
            
            product_results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'r2_gap': r2_gap,
                'overfitting_risk': overfitting_status,
                'test_mae': test_mae,
                'test_rmse': test_rmse
            }
        
        results[product] = product_results
    
    return results

def time_series_validation():
    """Perform time series cross-validation"""
    
    print(f"\n{'='*80}")
    print("TIME SERIES CROSS-VALIDATION")
    print(f"{'='*80}")
    
    df = load_and_preprocess_data()
    enhanced_df = create_anti_overfitting_features(df)
    
    feature_columns = [
        'month', 'campaign', 'seasonality',
        'quantity_lag1', 'quantity_ma6',
        'month_sin', 'month_cos', 'time_trend'
    ]
    
    products = enhanced_df['product'].unique()
    
    for product in products:
        print(f"\n{'-'*50}")
        print(f"CV ANALYSIS: {product}")
        print(f"{'-'*50}")
        
        product_data = enhanced_df[enhanced_df['product'] == product].copy()
        
        if len(product_data) < 15:
            print("Insufficient data for CV")
            continue
            
        X = product_data[feature_columns]
        y = product_data['quantity']
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        model = Ridge(alpha=10.0)
        scaler = StandardScaler()
        
        cv_train_scores = []
        cv_test_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_cv = scaler.fit_transform(X.iloc[train_idx])
            X_test_cv = scaler.transform(X.iloc[test_idx])
            y_train_cv = y.iloc[train_idx]
            y_test_cv = y.iloc[test_idx]
            
            model.fit(X_train_cv, y_train_cv)
            
            train_pred = model.predict(X_train_cv)
            test_pred = model.predict(X_test_cv)
            
            train_r2 = r2_score(y_train_cv, train_pred)
            test_r2 = r2_score(y_test_cv, test_pred)
            
            cv_train_scores.append(train_r2)
            cv_test_scores.append(test_r2)
            
            print(f"  Fold {fold+1}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}, Gap={train_r2-test_r2:.3f}")
        
        print(f"  CV Mean Train R²: {np.mean(cv_train_scores):.3f} ± {np.std(cv_train_scores):.3f}")
        print(f"  CV Mean Test R²:  {np.mean(cv_test_scores):.3f} ± {np.std(cv_test_scores):.3f}")
        print(f"  CV Mean Gap:      {np.mean(cv_train_scores) - np.mean(cv_test_scores):.3f}")

if __name__ == "__main__":
    print("🛡️  IMPLEMENTING ANTI-OVERFITTING MEASURES...")
    
    # Train regularized models
    results = train_regularized_models()
    
    # Perform cross-validation
    time_series_validation()
    
    print(f"\n{'='*80}")
    print("🎯 ANTI-OVERFITTING SUMMARY")
    print(f"{'='*80}")
    print("✅ Applied strong regularization (alpha=10.0)")
    print("✅ Reduced features from 16 to 8")
    print("✅ Conservative train/test split (70/30)")
    print("✅ Shallow trees (max_depth=3)")
    print("✅ Time series cross-validation")
    print("✅ Overfitting detection implemented")