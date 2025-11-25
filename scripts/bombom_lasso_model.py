import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

class BombomLassoModel:
    """Specialized Lasso model for BOMBOM MORANGUETE with feature selection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.selected_features = []
        self.feature_importance = {}
        self.training_results = {}
        self.product_name = "BOMBOM MORANGUETE 13G 160UN"
    
    def load_and_prepare_data(self):
        """Load and prepare data specifically for BOMBOM MORANGUETE"""
        print("="*70)
        print(f"LOADING DATA FOR: {self.product_name}")
        print("="*70)
        
        df = pd.read_csv('dataset.csv', sep=',', encoding="latin1")
        df = df.dropna()
        df = df.drop('PESOL', axis=1)
        df.columns = ['product', 'year', 'month', 'campaign', 'seasonality', 'quantity']
        
        # Filter for BOMBOM MORANGUETE only
        bombom_data = df[df['product'] == self.product_name].copy()
        
        if len(bombom_data) == 0:
            raise ValueError(f"No data found for product: {self.product_name}")
        
        # Sort by date
        bombom_data = bombom_data.sort_values(['year', 'month']).reset_index(drop=True)
        
        print(f"Total data points: {len(bombom_data)}")
        print(f"Date range: {bombom_data['year'].min()}/{bombom_data['month'].min()} to {bombom_data['year'].max()}/{bombom_data['month'].max()}")
        print(f"Quantity range: {bombom_data['quantity'].min():.0f} - {bombom_data['quantity'].max():.0f}")
        print(f"Average quantity: {bombom_data['quantity'].mean():.0f}")
        
        # Create comprehensive features for feature selection
        print("\n📊 Creating features for Lasso feature selection...")
        
        # 1. Basic temporal features
        bombom_data['month_sin'] = np.sin(2 * np.pi * bombom_data['month'] / 12)
        bombom_data['month_cos'] = np.cos(2 * np.pi * bombom_data['month'] / 12)
        bombom_data['quarter'] = pd.to_datetime(bombom_data[['year', 'month']].assign(day=1)).dt.quarter
        
        # 2. Lag features (multiple periods)
        for lag in [1, 2, 3, 6]:
            bombom_data[f'quantity_lag{lag}'] = bombom_data['quantity'].shift(lag)
        
        # 3. Rolling statistics (multiple windows) - past-only
        past_q = bombom_data['quantity'].shift(1)
        for window in [3, 6, 12]:
            bombom_data[f'quantity_ma{window}'] = past_q.rolling(window=window, min_periods=1).mean()
            bombom_data[f'quantity_std{window}'] = past_q.rolling(window=window, min_periods=1).std()
        
        # 4. Trend and growth features
        bombom_data['trend'] = np.arange(len(bombom_data))
        bombom_data['trend_squared'] = bombom_data['trend'] ** 2
        # Past-only growth/diffs
        bombom_data['quantity_pct_change'] = past_q.pct_change()
        bombom_data['quantity_diff'] = past_q.diff()
        
        # 5. Seasonal indicators
        bombom_data['is_high_season'] = ((bombom_data['month'] >= 7) & (bombom_data['month'] <= 10)).astype(int)
        bombom_data['is_low_season'] = ((bombom_data['month'] >= 1) & (bombom_data['month'] <= 3)).astype(int)
        bombom_data['is_holiday_season'] = ((bombom_data['month'] == 12) | (bombom_data['month'] == 1)).astype(int)
        
        # 6. Month dummies (one-hot encoding)
        for month in range(1, 13):
            bombom_data[f'month_{month}'] = (bombom_data['month'] == month).astype(int)
        
        # 7. Quarter dummies
        for quarter in range(1, 5):
            bombom_data[f'quarter_{quarter}'] = (bombom_data['quarter'] == quarter).astype(int)
        
        # 8. Interaction features
        bombom_data['campaign_season_interaction'] = bombom_data['campaign'] * bombom_data['seasonality']
        bombom_data['campaign_month'] = bombom_data['campaign'] * bombom_data['month']
        bombom_data['seasonality_month'] = bombom_data['seasonality'] * bombom_data['month']
        
        # 9. Year-over-year features (if sufficient data)
        if len(bombom_data) >= 24:
            bombom_data['quantity_yoy'] = bombom_data['quantity'].shift(12)
            prev_q = bombom_data['quantity'].shift(1)
            prev_yoy = bombom_data['quantity'].shift(13)
            bombom_data['yoy_growth'] = (prev_q - prev_yoy) / prev_yoy.replace(0, np.nan)
        
        # Define all potential features for Lasso selection
        self.feature_columns = [
            'year', 'month', 'campaign', 'seasonality', 'quarter',
            'month_sin', 'month_cos', 'trend', 'trend_squared',
            'quantity_lag1', 'quantity_lag2', 'quantity_lag3', 'quantity_lag6',
            'quantity_ma3', 'quantity_ma6', 'quantity_ma12',
            'quantity_std3', 'quantity_std6', 'quantity_std12',
            'quantity_pct_change', 'quantity_diff',
            'is_high_season', 'is_low_season', 'is_holiday_season',
            'campaign_season_interaction', 'campaign_month', 'seasonality_month'
        ]
        
        # Add month dummies
        self.feature_columns.extend([f'month_{i}' for i in range(1, 13)])
        
        # Add quarter dummies
        self.feature_columns.extend([f'quarter_{i}' for i in range(1, 5)])
        
        # Add YoY features if they exist
        if 'yoy_growth' in bombom_data.columns:
            self.feature_columns.extend(['quantity_yoy', 'yoy_growth'])
        
        # Fill NaN values
        numeric_cols = bombom_data.select_dtypes(include=[np.number]).columns
        bombom_data[numeric_cols] = bombom_data[numeric_cols].fillna(method='bfill')
        bombom_data[numeric_cols] = bombom_data[numeric_cols].fillna(bombom_data[numeric_cols].mean())
        
        print(f"Total features created: {len(self.feature_columns)}")
        
        return bombom_data
    
    def perform_lasso_feature_selection(self, X, y):
        """Perform feature selection using Lasso with cross-validation"""
        print("\n🎯 PERFORMING LASSO FEATURE SELECTION")
        print("="*50)
        
        # Test different alpha values
        alphas = np.logspace(-4, 2, 50)  # From 0.0001 to 100
        
        # Use LassoCV for automatic alpha selection
        lasso_cv = LassoCV(
            alphas=alphas,
            cv=TimeSeriesSplit(n_splits=3),  # Time series cross-validation
            random_state=42,
            max_iter=2000
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit LassoCV
        lasso_cv.fit(X_scaled, y)
        
        print(f"Optimal alpha: {lasso_cv.alpha_:.6f}")
        print(f"CV Score (R²): {lasso_cv.score(X_scaled, y):.4f}")
        
        # Get feature coefficients
        coefficients = lasso_cv.coef_
        
        # Identify selected features (non-zero coefficients)
        selected_mask = np.abs(coefficients) > 1e-6
        self.selected_features = [self.feature_columns[i] for i in range(len(self.feature_columns)) if selected_mask[i]]
        
        # Store feature importance
        self.feature_importance = {
            feature: coef for feature, coef in zip(self.feature_columns, coefficients)
            if abs(coef) > 1e-6
        }
        
        print(f"\nFeatures selected: {len(self.selected_features)} out of {len(self.feature_columns)}")
        print(f"Feature reduction: {(1 - len(self.selected_features)/len(self.feature_columns))*100:.1f}%")
        
        # Show top selected features
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\nTop 10 Selected Features:")
        print("-" * 50)
        for i, (feature, coef) in enumerate(sorted_features[:10]):
            print(f"{i+1:2d}. {feature:<25} | Coefficient: {coef:8.4f}")
        
        # Create final model with optimal alpha
        self.model = Lasso(alpha=lasso_cv.alpha_, random_state=42, max_iter=2000)
        
        return lasso_cv.alpha_
    
    def train_final_model(self, df):
        """Train the final Lasso model with selected features"""
        print(f"\n🚀 TRAINING FINAL LASSO MODEL")
        print("="*50)
        
        X = df[self.feature_columns]
        y = df['quantity']
        
        # Perform feature selection
        optimal_alpha = self.perform_lasso_feature_selection(X, y)
        
        # Train/test split (use last 20% for testing)
        split_idx = int(len(df) * 0.8)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing:  {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train final model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Ensure non-negative predictions
        train_pred = np.maximum(train_pred, 0)
        test_pred = np.maximum(test_pred, 0)
        
        # Calculate comprehensive metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # MAPE calculation
        train_mape = np.mean([abs((actual - pred) / actual) for actual, pred in zip(y_train, train_pred) if actual > 0]) * 100
        test_mape = np.mean([abs((actual - pred) / actual) for actual, pred in zip(y_test, test_pred) if actual > 0]) * 100
        
        # Store results
        self.training_results = {
            'optimal_alpha': optimal_alpha,
            'features_selected': len(self.selected_features),
            'feature_reduction_pct': (1 - len(self.selected_features)/len(self.feature_columns))*100,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'r2_gap': train_r2 - test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'overfitting_risk': 'HIGH' if (train_r2 - test_r2) > 0.3 else 'MEDIUM' if (train_r2 - test_r2) > 0.1 else 'LOW'
        }
        
        # Display results
        print(f"\n📊 TRAINING RESULTS:")
        print("="*50)
        print(f"Optimal Alpha: {optimal_alpha:.6f}")
        print(f"Features Selected: {len(self.selected_features)}/{len(self.feature_columns)} ({self.training_results['feature_reduction_pct']:.1f}% reduction)")
        print(f"\nPerformance Metrics:")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  R² Gap:   {train_r2 - test_r2:.4f}")
        print(f"  Test MAE: {test_mae:.1f}")
        print(f"  Test MAPE: {test_mape:.1f}%")
        print(f"  Test RMSE: {test_rmse:.1f}")
        print(f"  Overfitting Risk: {self.training_results['overfitting_risk']}")
        
        # Show actual vs predicted
        print(f"\n🎯 PREDICTIONS vs ACTUAL (Test Set):")
        print("-" * 50)
        print("Sample | Actual  | Predicted | Error   | Error%")
        print("-------|---------|-----------|---------|-------")
        
        test_dates = df.iloc[split_idx:][['year', 'month']].values
        for i, ((year, month), actual, pred) in enumerate(zip(test_dates, y_test.values, test_pred)):
            error = pred - actual
            error_pct = (error / actual * 100) if actual > 0 else 0
            print(f"{i+1:6d} | {actual:7.0f} | {pred:9.1f} | {error:7.0f} | {error_pct:6.1f}%")
        
        return self.training_results
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_final_model() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.maximum(predictions, 0)  # Ensure non-negative
    
    def save_model(self, filename='bombom_lasso_model.pkl'):
        """Save the trained model"""
        import os
        os.makedirs('models', exist_ok=True)
        out_path = filename if filename.startswith('models/') else os.path.join('models', filename)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'training_results': self.training_results,
            'product_name': self.product_name
        }
        joblib.dump(model_data, out_path)
        print(f"\n✅ Model saved to: {out_path}")
    
    @classmethod
    def load_model(cls, filename='bombom_lasso_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filename)
        
        bombom_model = cls()
        bombom_model.model = model_data['model']
        bombom_model.scaler = model_data['scaler']
        bombom_model.feature_columns = model_data['feature_columns']
        bombom_model.selected_features = model_data['selected_features']
        bombom_model.feature_importance = model_data['feature_importance']
        bombom_model.training_results = model_data['training_results']
        bombom_model.product_name = model_data['product_name']
        
        return bombom_model
    
    def generate_report(self):
        """Generate comprehensive model report"""
        print(f"\n{'='*80}")
        print(f"BOMBOM MORANGUETE LASSO MODEL - COMPREHENSIVE REPORT")
        print(f"{'='*80}")
        
        print(f"Product: {self.product_name}")
        print(f"Model Type: Lasso Regression with Feature Selection")
        print(f"Optimal Alpha: {self.training_results['optimal_alpha']:.6f}")
        
        print(f"\nFEATURE SELECTION RESULTS:")
        print(f"  Original Features: {len(self.feature_columns)}")
        print(f"  Selected Features: {self.training_results['features_selected']}")
        print(f"  Reduction: {self.training_results['feature_reduction_pct']:.1f}%")
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"  Test R²: {self.training_results['test_r2']:.4f}")
        print(f"  Test MAE: {self.training_results['test_mae']:.1f}")
        print(f"  Test MAPE: {self.training_results['test_mape']:.1f}%")
        print(f"  Overfitting Risk: {self.training_results['overfitting_risk']}")
        
        print(f"\nTOP 5 MOST IMPORTANT FEATURES:")
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for i, (feature, coef) in enumerate(sorted_features[:5]):
            print(f"  {i+1}. {feature:<25} | Coefficient: {coef:8.4f}")

def main():
    print("🍫 BOMBOM MORANGUETE LASSO FEATURE SELECTION MODEL")
    print("="*80)
    
    # Initialize model
    bombom_model = BombomLassoModel()
    
    # Load and prepare data
    try:
        df = bombom_model.load_and_prepare_data()
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Train model with feature selection
    results = bombom_model.train_final_model(df)
    
    # Generate comprehensive report
    bombom_model.generate_report()
    
    # Save model
    bombom_model.save_model('models/bombom_lasso_model.pkl')
    
    print(f"\n{'='*80}")
    print("🎉 BOMBOM MORANGUETE LASSO MODEL TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"✅ Feature selection completed: {results['features_selected']} features selected")
    print(f"✅ Model trained and evaluated")
    print(f"✅ Model saved for production use")

if __name__ == "__main__":
    main()
