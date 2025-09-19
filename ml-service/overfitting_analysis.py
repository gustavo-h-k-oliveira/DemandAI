import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, validation_curve, learning_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv('dataset.csv', sep=',', encoding="latin1")
    df = df.dropna()
    df = df.drop('PESOL', axis=1)
    df.columns = ['product', 'year', 'month', 'campaign', 'seasonality', 'quantity']
    return df

def create_enhanced_features(df):
    """Create enhanced features for better model performance"""
    df_sorted = df.sort_values(['product', 'year', 'month']).reset_index(drop=True)
    all_data = []
    
    for product in df_sorted['product'].unique():
        product_data = df_sorted[df_sorted['product'] == product].copy().reset_index(drop=True)
        
        # Lag features
        product_data['quantity_lag1'] = product_data['quantity'].shift(1)
        product_data['quantity_lag2'] = product_data['quantity'].shift(2)
        product_data['quantity_lag3'] = product_data['quantity'].shift(3)
        
        # Rolling averages
        product_data['quantity_ma3'] = product_data['quantity'].rolling(window=3, min_periods=1).mean()
        product_data['quantity_ma6'] = product_data['quantity'].rolling(window=6, min_periods=1).mean()
        
        # Seasonal features
        product_data['month_sin'] = np.sin(2 * np.pi * product_data['month'] / 12)
        product_data['month_cos'] = np.cos(2 * np.pi * product_data['month'] / 12)
        
        # Interaction features
        product_data['campaign_season_interaction'] = product_data['campaign'] * product_data['seasonality']
        product_data['month_campaign_interaction'] = product_data['month'] * product_data['campaign']
        
        # Trend feature
        product_data['time_trend'] = range(len(product_data))
        product_data['quantity_pct_change'] = product_data['quantity'].pct_change(periods=1)
        
        # YoY growth
        monthly_growth = []
        for i, row in product_data.iterrows():
            current_year, current_month = row['year'], row['month']
            prev_year_data = product_data[
                (product_data['year'] == current_year - 1) & 
                (product_data['month'] == current_month)
            ]
            
            if len(prev_year_data) > 0:
                prev_quantity = prev_year_data['quantity'].iloc[0]
                growth = (row['quantity'] - prev_quantity) / prev_quantity if prev_quantity > 0 else 0
            else:
                growth = 0
            monthly_growth.append(growth)
        
        product_data['yoy_growth'] = monthly_growth
        all_data.append(product_data)
    
    enhanced_df = pd.concat(all_data, ignore_index=True)
    enhanced_df = enhanced_df.fillna(method='bfill').fillna(0)
    return enhanced_df

def time_series_cross_validation(X, y, model, cv_splits=5):
    """Perform time series cross validation"""
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    train_scores = []
    val_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale if needed
        if hasattr(model, 'alpha'):  # Ridge/Lasso
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_cv)
            X_val_scaled = scaler.transform(X_val_cv)
            
            model.fit(X_train_scaled, y_train_cv)
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
        else:
            model.fit(X_train_cv, y_train_cv)
            train_pred = model.predict(X_train_cv)
            val_pred = model.predict(X_val_cv)
        
        train_r2 = r2_score(y_train_cv, train_pred)
        val_r2 = r2_score(y_val_cv, val_pred)
        
        train_scores.append(train_r2)
        val_scores.append(val_r2)
    
    return train_scores, val_scores

def detect_overfitting_comprehensive():
    """Comprehensive overfitting detection and validation"""
    
    print("="*80)
    print("OVERFITTING DETECTION AND VALIDATION")
    print("="*80)
    
    df = load_and_preprocess_data()
    enhanced_df = create_enhanced_features(df)
    
    feature_columns = [
        'year', 'month', 'campaign', 'seasonality',
        'quantity_lag1', 'quantity_lag2', 'quantity_lag3',
        'quantity_ma3', 'quantity_ma6',
        'month_sin', 'month_cos',
        'campaign_season_interaction', 'month_campaign_interaction',
        'time_trend', 'quantity_pct_change', 'yoy_growth'
    ]
    
    products = enhanced_df['product'].unique()
    overfitting_results = {}
    
    for product in products:
        print(f"\n{'='*60}")
        print(f"OVERFITTING ANALYSIS FOR: {product}")
        print(f"{'='*60}")
        
        product_data = enhanced_df[enhanced_df['product'] == product].copy()
        
        if len(product_data) < 15:
            print(f"Insufficient data: {len(product_data)} samples")
            continue
        
        X = product_data[feature_columns]
        y = product_data['quantity']
        
        # 1. MULTIPLE TRAIN/VALIDATION SPLITS
        print("\n1. MULTIPLE TRAIN/VALIDATION SPLITS ANALYSIS")
        print("-" * 50)
        
        split_results = {}
        split_ratios = [0.6, 0.7, 0.8, 0.9]
        
        for ratio in split_ratios:
            split_idx = int(len(product_data) * ratio)
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            if len(X_test) < 3:
                continue
            
            # Test different models
            models = {
                'Ridge': Ridge(alpha=0.1),
                'Lasso': Lasso(alpha=0.1, max_iter=1000),
                'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            }
            
            for model_name, model in models.items():
                try:
                    if model_name in ['Ridge', 'Lasso']:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        model.fit(X_train_scaled, y_train)
                        train_pred = model.predict(X_train_scaled)
                        test_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        train_pred = model.predict(X_train)
                        test_pred = model.predict(X_test)
                    
                    train_r2 = r2_score(y_train, train_pred)
                    test_r2 = r2_score(y_test, test_pred)
                    gap = train_r2 - test_r2
                    
                    if model_name not in split_results:
                        split_results[model_name] = []
                    
                    split_results[model_name].append({
                        'ratio': ratio,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'gap': gap,
                        'train_samples': len(X_train),
                        'test_samples': len(X_test)
                    })
                    
                except Exception as e:
                    continue
        
        # Display split analysis results
        for model_name, results in split_results.items():
            print(f"\n{model_name}:")
            print("Split | Train R² | Test R²  | Gap     | Train/Test")
            print("------|----------|----------|---------|----------")
            for result in results:
                print(f"{result['ratio']:.1f}   | {result['train_r2']:8.4f} | {result['test_r2']:8.4f} | {result['gap']:7.4f} | {result['train_samples']}/{result['test_samples']}")
        
        # 2. TIME SERIES CROSS VALIDATION
        print(f"\n2. TIME SERIES CROSS VALIDATION")
        print("-" * 50)
        
        cv_results = {}
        models_cv = {
            'Ridge': Ridge(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        }
        
        for model_name, model in models_cv.items():
            try:
                train_scores, val_scores = time_series_cross_validation(X, y, model, cv_splits=3)
                
                cv_results[model_name] = {
                    'train_mean': np.mean(train_scores),
                    'train_std': np.std(train_scores),
                    'val_mean': np.mean(val_scores),
                    'val_std': np.std(val_scores),
                    'gap_mean': np.mean(train_scores) - np.mean(val_scores),
                    'individual_gaps': [t - v for t, v in zip(train_scores, val_scores)]
                }
                
                print(f"\n{model_name}:")
                print(f"  Train R²: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
                print(f"  Val R²:   {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
                print(f"  Gap:      {np.mean(train_scores) - np.mean(val_scores):.4f}")
                print(f"  Fold gaps: {[f'{gap:.4f}' for gap in cv_results[model_name]['individual_gaps']]}")
                
            except Exception as e:
                print(f"  Error in CV for {model_name}: {str(e)}")
        
        # 3. LEARNING CURVES ANALYSIS
        print(f"\n3. LEARNING CURVES ANALYSIS")
        print("-" * 50)
        
        learning_curve_results = {}
        
        for model_name, model in models_cv.items():
            try:
                # Calculate learning curve with different training set sizes
                train_sizes = np.linspace(0.3, 1.0, 5)
                
                if hasattr(model, 'alpha'):  # Ridge/Lasso
                    # Create pipeline with scaler for Ridge
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                    train_sizes_abs, train_scores_lc, val_scores_lc = learning_curve(
                        pipeline, X, y, train_sizes=train_sizes, 
                        cv=TimeSeriesSplit(n_splits=3), scoring='r2'
                    )
                else:
                    train_sizes_abs, train_scores_lc, val_scores_lc = learning_curve(
                        model, X, y, train_sizes=train_sizes,
                        cv=TimeSeriesSplit(n_splits=3), scoring='r2'
                    )
                
                learning_curve_results[model_name] = {
                    'train_sizes': train_sizes_abs,
                    'train_scores_mean': np.mean(train_scores_lc, axis=1),
                    'train_scores_std': np.std(train_scores_lc, axis=1),
                    'val_scores_mean': np.mean(val_scores_lc, axis=1),
                    'val_scores_std': np.std(val_scores_lc, axis=1)
                }
                
                print(f"\n{model_name} Learning Curve:")
                print("Samples | Train R²        | Val R²          | Gap")
                print("--------|-----------------|-----------------|--------")
                
                for i, size in enumerate(train_sizes_abs):
                    train_mean = learning_curve_results[model_name]['train_scores_mean'][i]
                    train_std = learning_curve_results[model_name]['train_scores_std'][i]
                    val_mean = learning_curve_results[model_name]['val_scores_mean'][i]
                    val_std = learning_curve_results[model_name]['val_scores_std'][i]
                    gap = train_mean - val_mean
                    
                    print(f"{size:7.0f} | {train_mean:.4f} ± {train_std:.4f} | {val_mean:.4f} ± {val_std:.4f} | {gap:6.4f}")
            
            except Exception as e:
                print(f"  Error in learning curve for {model_name}: {str(e)}")
        
        # 4. OVERFITTING DIAGNOSIS
        print(f"\n4. OVERFITTING DIAGNOSIS")
        print("-" * 50)
        
        diagnosis = {}
        
        for model_name in cv_results.keys():
            gap = cv_results[model_name]['gap_mean']
            val_std = cv_results[model_name]['val_std']
            
            # Overfitting criteria
            high_gap = gap > 0.1  # Train-Val gap > 10%
            high_variance = val_std > 0.2  # High validation variance
            perfect_train = cv_results[model_name]['train_mean'] > 0.99  # Near-perfect training score
            
            diagnosis[model_name] = {
                'overfitting_risk': 'HIGH' if (high_gap and perfect_train) else 'MEDIUM' if high_gap else 'LOW',
                'variance_risk': 'HIGH' if high_variance else 'LOW',
                'gap': gap,
                'recommendations': []
            }
            
            # Generate recommendations
            if high_gap:
                diagnosis[model_name]['recommendations'].append("Increase regularization")
                diagnosis[model_name]['recommendations'].append("Reduce model complexity")
            
            if high_variance:
                diagnosis[model_name]['recommendations'].append("Collect more data")
                diagnosis[model_name]['recommendations'].append("Use ensemble methods")
            
            if perfect_train:
                diagnosis[model_name]['recommendations'].append("Check for data leakage")
                diagnosis[model_name]['recommendations'].append("Validate feature engineering")
            
            print(f"\n{model_name}:")
            print(f"  Overfitting Risk: {diagnosis[model_name]['overfitting_risk']}")
            print(f"  Variance Risk: {diagnosis[model_name]['variance_risk']}")
            print(f"  Train-Val Gap: {gap:.4f}")
            
            if diagnosis[model_name]['recommendations']:
                print(f"  Recommendations:")
                for rec in diagnosis[model_name]['recommendations']:
                    print(f"    - {rec}")
        
        overfitting_results[product] = {
            'split_results': split_results,
            'cv_results': cv_results,
            'learning_curve_results': learning_curve_results,
            'diagnosis': diagnosis
        }
    
    return overfitting_results

def save_overfitting_analysis(results, filename="overfitting_analysis.txt"):
    """Save overfitting analysis results"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("OVERFITTING DETECTION AND VALIDATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERFITTING DETECTION METHODS USED:\n")
        f.write("1. Multiple Train/Validation Split Analysis\n")
        f.write("2. Time Series Cross Validation\n")
        f.write("3. Learning Curves Analysis\n")
        f.write("4. Statistical Diagnosis\n\n")
        
        # Overall summary
        f.write("OVERALL OVERFITTING RISK SUMMARY\n")
        f.write("="*50 + "\n")
        
        risk_summary = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for product, product_results in results.items():
            f.write(f"\n{product}:\n")
            for model_name, diagnosis in product_results['diagnosis'].items():
                risk = diagnosis['overfitting_risk']
                risk_summary[risk] += 1
                f.write(f"  {model_name:<15}: {risk:6} risk (gap: {diagnosis['gap']:.4f})\n")
        
        f.write(f"\nRisk Distribution:\n")
        f.write(f"  HIGH risk models:   {risk_summary['HIGH']}\n")
        f.write(f"  MEDIUM risk models: {risk_summary['MEDIUM']}\n")
        f.write(f"  LOW risk models:    {risk_summary['LOW']}\n")
        
        # Detailed results for each product
        f.write(f"\n\nDETAILED ANALYSIS BY PRODUCT\n")
        f.write("="*80 + "\n")
        
        for product, product_results in results.items():
            f.write(f"\nPRODUCT: {product}\n")
            f.write("-" * len(f"PRODUCT: {product}") + "\n")
            
            # Cross validation results
            f.write("\nTime Series Cross Validation Results:\n")
            for model_name, cv_result in product_results['cv_results'].items():
                f.write(f"{model_name}:\n")
                f.write(f"  Train R²: {cv_result['train_mean']:.4f} ± {cv_result['train_std']:.4f}\n")
                f.write(f"  Val R²:   {cv_result['val_mean']:.4f} ± {cv_result['val_std']:.4f}\n")
                f.write(f"  Gap:      {cv_result['gap_mean']:.4f}\n\n")
            
            # Diagnosis and recommendations
            f.write("Overfitting Diagnosis:\n")
            for model_name, diagnosis in product_results['diagnosis'].items():
                f.write(f"{model_name}:\n")
                f.write(f"  Risk Level: {diagnosis['overfitting_risk']}\n")
                f.write(f"  Variance Risk: {diagnosis['variance_risk']}\n")
                if diagnosis['recommendations']:
                    f.write(f"  Recommendations:\n")
                    for rec in diagnosis['recommendations']:
                        f.write(f"    - {rec}\n")
                f.write("\n")
        
        # General recommendations
        f.write("\nGENERAL RECOMMENDATIONS TO PREVENT OVERFITTING\n")
        f.write("="*60 + "\n")
        f.write("1. DATA STRATEGIES:\n")
        f.write("   - Collect more historical data\n")
        f.write("   - Use proper train/validation/test splits\n")
        f.write("   - Implement time series cross validation\n\n")
        
        f.write("2. MODEL STRATEGIES:\n")
        f.write("   - Increase regularization (higher alpha values)\n")
        f.write("   - Reduce model complexity (lower max_depth, fewer features)\n")
        f.write("   - Use ensemble methods\n")
        f.write("   - Early stopping for iterative algorithms\n\n")
        
        f.write("3. VALIDATION STRATEGIES:\n")
        f.write("   - Always use out-of-sample validation\n")
        f.write("   - Monitor train vs validation performance\n")
        f.write("   - Test on completely unseen data\n")
        f.write("   - Use multiple evaluation metrics\n\n")
        
        f.write("4. FEATURE ENGINEERING:\n")
        f.write("   - Avoid look-ahead bias in features\n")
        f.write("   - Check for data leakage\n")
        f.write("   - Validate feature importance\n")
        f.write("   - Consider feature selection techniques\n\n")
        
        f.write("END OF OVERFITTING ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Overfitting analysis saved to: {filename}")

if __name__ == "__main__":
    print("🔍 Starting Comprehensive Overfitting Detection...")
    
    # Run overfitting analysis
    results = detect_overfitting_comprehensive()
    
    # Save results
    save_overfitting_analysis(results)
    
    print(f"\n{'='*80}")
    print("🎉 OVERFITTING ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print("📋 Key Metrics Analyzed:")
    print("   - Train vs Validation R² gaps")
    print("   - Cross-validation stability")
    print("   - Learning curve patterns")
    print("   - Model variance across splits")
    print("\n📁 Report saved to: overfitting_analysis.txt")