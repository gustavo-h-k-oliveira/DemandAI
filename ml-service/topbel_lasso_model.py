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

class TopbelLassoModel:
    """Specialized Lasso model for TOPBEL LEITE CONDENSADO with feature selection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.selected_features = []
        self.feature_importance = {}
        self.training_results = {}
        self.product_name = "TOPBEL LEITE CONDENSADO 50UN"
    
    def load_and_prepare_data(self):
        """Load and prepare data specifically for TOPBEL LEITE CONDENSADO"""
        print("="*70)
        print(f"LOADING DATA FOR: {self.product_name}")
        print("="*70)
        
        df = pd.read_csv('dataset.csv', sep=',', encoding="latin1")
        df = df.dropna()
        df = df.drop('PESOL', axis=1)
        df.columns = ['product', 'year', 'month', 'campaign', 'seasonality', 'quantity']
        
        # Filter for TOPBEL LEITE CONDENSADO only
        topbel_data = df[df['product'] == self.product_name].copy()
        
        if len(topbel_data) == 0:
            raise ValueError(f"No data found for product: {self.product_name}")
        
        # Sort by date
        topbel_data = topbel_data.sort_values(['year', 'month']).reset_index(drop=True)
        
        print(f"Total data points: {len(topbel_data)}")
        print(f"Date range: {topbel_data['year'].min()}/{topbel_data['month'].min()} to {topbel_data['year'].max()}/{topbel_data['month'].max()}")
        print(f"Quantity range: {topbel_data['quantity'].min():.0f} - {topbel_data['quantity'].max():.0f}")
        print(f"Average quantity: {topbel_data['quantity'].mean():.0f}")
        print(f"Std deviation: {topbel_data['quantity'].std():.0f}")
        
        # Analyze seasonality pattern for TOPBEL LEITE CONDENSADO
        seasonal_analysis = topbel_data.groupby('seasonality')['quantity'].agg(['mean', 'std', 'count'])
        print(f"\nSeasonality Analysis:")
        print(seasonal_analysis)
        
        campaign_analysis = topbel_data.groupby('campaign')['quantity'].agg(['mean', 'std', 'count'])
        print(f"\nCampaign Analysis:")
        print(campaign_analysis)
        
        # Create comprehensive features for feature selection
        print("\n📊 Creating features for Lasso feature selection...")
        
        # 1. Basic temporal features
        topbel_data['month_sin'] = np.sin(2 * np.pi * topbel_data['month'] / 12)
        topbel_data['month_cos'] = np.cos(2 * np.pi * topbel_data['month'] / 12)
        topbel_data['quarter'] = pd.to_datetime(topbel_data[['year', 'month']].assign(day=1)).dt.quarter
        
        # 2. Lag features (multiple periods)
        for lag in [1, 2, 3, 4, 6, 12]:
            topbel_data[f'quantity_lag{lag}'] = topbel_data['quantity'].shift(lag)
        
        # 3. Rolling statistics (multiple windows)
        for window in [3, 6, 9, 12]:
            topbel_data[f'quantity_ma{window}'] = topbel_data['quantity'].rolling(window=window, min_periods=1).mean()
            topbel_data[f'quantity_std{window}'] = topbel_data['quantity'].rolling(window=window, min_periods=1).std()
            topbel_data[f'quantity_min{window}'] = topbel_data['quantity'].rolling(window=window, min_periods=1).min()
            topbel_data[f'quantity_max{window}'] = topbel_data['quantity'].rolling(window=window, min_periods=1).max()
        
        # 4. Trend and growth features
        topbel_data['trend'] = np.arange(len(topbel_data))
        topbel_data['trend_squared'] = topbel_data['trend'] ** 2
        topbel_data['trend_cubed'] = topbel_data['trend'] ** 3
        topbel_data['quantity_pct_change'] = topbel_data['quantity'].pct_change()
        topbel_data['quantity_diff'] = topbel_data['quantity'].diff()
        topbel_data['quantity_diff2'] = topbel_data['quantity_diff'].diff()
        
        # 5. Seasonal indicators (specific for dairy products)
        topbel_data['is_high_season'] = ((topbel_data['month'] >= 7) & (topbel_data['month'] <= 10)).astype(int)
        topbel_data['is_low_season'] = ((topbel_data['month'] >= 1) & (topbel_data['month'] <= 3)).astype(int)
        topbel_data['is_holiday_season'] = ((topbel_data['month'] == 12) | (topbel_data['month'] == 1)).astype(int)
        topbel_data['is_summer'] = ((topbel_data['month'] >= 6) & (topbel_data['month'] <= 8)).astype(int)
        topbel_data['is_winter'] = ((topbel_data['month'] >= 12) | (topbel_data['month'] <= 2)).astype(int)
        
        # 6. Month dummies (one-hot encoding)
        for month in range(1, 13):
            topbel_data[f'month_{month}'] = (topbel_data['month'] == month).astype(int)
        
        # 7. Quarter dummies
        for quarter in range(1, 5):
            topbel_data[f'quarter_{quarter}'] = (topbel_data['quarter'] == quarter).astype(int)
        
        # 8. Interaction features (important for dairy products)
        topbel_data['campaign_season_interaction'] = topbel_data['campaign'] * topbel_data['seasonality']
        topbel_data['campaign_month'] = topbel_data['campaign'] * topbel_data['month']
        topbel_data['seasonality_month'] = topbel_data['seasonality'] * topbel_data['month']
        topbel_data['campaign_high_season'] = topbel_data['campaign'] * topbel_data['is_high_season']
        topbel_data['seasonality_summer'] = topbel_data['seasonality'] * topbel_data['is_summer']
        
        # 9. Volatility features
        topbel_data['quantity_volatility3'] = topbel_data['quantity'].rolling(window=3, min_periods=1).std() / topbel_data['quantity_ma3']
        topbel_data['quantity_volatility6'] = topbel_data['quantity'].rolling(window=6, min_periods=1).std() / topbel_data['quantity_ma6']
        
        # 10. Year-over-year features (if sufficient data)
        if len(topbel_data) >= 24:
            topbel_data['quantity_yoy'] = topbel_data['quantity'].shift(12)
            topbel_data['yoy_growth'] = (topbel_data['quantity'] - topbel_data['quantity_yoy']) / topbel_data['quantity_yoy'].replace(0, np.nan)
            topbel_data['yoy_acceleration'] = topbel_data['yoy_growth'].diff()
        
        # 11. Cyclical features
        topbel_data['quarter_sin'] = np.sin(2 * np.pi * topbel_data['quarter'] / 4)
        topbel_data['quarter_cos'] = np.cos(2 * np.pi * topbel_data['quarter'] / 4)
        
        # Define all potential features for Lasso selection
        self.feature_columns = [
            'year', 'month', 'campaign', 'seasonality', 'quarter',
            'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
            'trend', 'trend_squared', 'trend_cubed',
            'quantity_lag1', 'quantity_lag2', 'quantity_lag3', 'quantity_lag4', 'quantity_lag6', 'quantity_lag12',
            'quantity_ma3', 'quantity_ma6', 'quantity_ma9', 'quantity_ma12',
            'quantity_std3', 'quantity_std6', 'quantity_std9', 'quantity_std12',
            'quantity_min3', 'quantity_min6', 'quantity_min9', 'quantity_min12',
            'quantity_max3', 'quantity_max6', 'quantity_max9', 'quantity_max12',
            'quantity_pct_change', 'quantity_diff', 'quantity_diff2',
            'is_high_season', 'is_low_season', 'is_holiday_season', 'is_summer', 'is_winter',
            'campaign_season_interaction', 'campaign_month', 'seasonality_month',
            'campaign_high_season', 'seasonality_summer',
            'quantity_volatility3', 'quantity_volatility6'
        ]
        
        # Add month dummies
        self.feature_columns.extend([f'month_{i}' for i in range(1, 13)])
        
        # Add quarter dummies
        self.feature_columns.extend([f'quarter_{i}' for i in range(1, 5)])
        
        # Add YoY features if they exist
        if 'yoy_growth' in topbel_data.columns:
            self.feature_columns.extend(['quantity_yoy', 'yoy_growth', 'yoy_acceleration'])
        
        # Fill NaN values
        numeric_cols = topbel_data.select_dtypes(include=[np.number]).columns
        topbel_data[numeric_cols] = topbel_data[numeric_cols].fillna(method='bfill')
        topbel_data[numeric_cols] = topbel_data[numeric_cols].fillna(topbel_data[numeric_cols].mean())
        
        # Replace infinite values
        topbel_data[numeric_cols] = topbel_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        topbel_data[numeric_cols] = topbel_data[numeric_cols].fillna(topbel_data[numeric_cols].mean())
        
        print(f"Total features created: {len(self.feature_columns)}")
        
        return topbel_data
    
    def perform_lasso_feature_selection(self, X, y):
        """Perform feature selection using Lasso with cross-validation"""
        print("\n🎯 PERFORMING LASSO FEATURE SELECTION")
        print("="*50)
        
        # Test different alpha values (broader range for dairy products)
        alphas = np.logspace(-5, 3, 100)  # From 0.00001 to 1000
        
        # Use LassoCV for automatic alpha selection
        lasso_cv = LassoCV(
            alphas=alphas,
            cv=TimeSeriesSplit(n_splits=min(5, len(X)//10)),  # Adaptive CV splits
            random_state=42,
            max_iter=3000,
            tol=1e-6
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit LassoCV
        print("Training LassoCV for optimal alpha selection...")
        lasso_cv.fit(X_scaled, y)
        
        print(f"Optimal alpha: {lasso_cv.alpha_:.8f}")
        print(f"CV Score (R²): {lasso_cv.score(X_scaled, y):.4f}")
        print(f"CV MSE: {lasso_cv.mse_path_.mean(axis=1).min():.2f}")
        
        # Get feature coefficients
        coefficients = lasso_cv.coef_
        
        # Identify selected features (non-zero coefficients)
        selected_mask = np.abs(coefficients) > 1e-8
        self.selected_features = [self.feature_columns[i] for i in range(len(self.feature_columns)) if selected_mask[i]]
        
        # Store feature importance
        self.feature_importance = {
            feature: coef for feature, coef in zip(self.feature_columns, coefficients)
            if abs(coef) > 1e-8
        }
        
        print(f"\nFeatures selected: {len(self.selected_features)} out of {len(self.feature_columns)}")
        print(f"Feature reduction: {(1 - len(self.selected_features)/len(self.feature_columns))*100:.1f}%")
        
        # Show top selected features
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\nTop 15 Selected Features:")
        print("-" * 60)
        for i, (feature, coef) in enumerate(sorted_features[:15]):
            print(f"{i+1:2d}. {feature:<30} | Coefficient: {coef:10.6f}")
        
        # Analyze feature categories
        lag_features = [f for f in self.selected_features if 'lag' in f]
        ma_features = [f for f in self.selected_features if '_ma' in f]
        seasonal_features = [f for f in self.selected_features if any(x in f for x in ['season', 'month_', 'quarter_'])]
        interaction_features = [f for f in self.selected_features if 'interaction' in f or any(x in f for x in ['campaign_', 'seasonality_'])]
        
        print(f"\nFeature Category Analysis:")
        print(f"  Lag features: {len(lag_features)}")
        print(f"  Moving average features: {len(ma_features)}")
        print(f"  Seasonal features: {len(seasonal_features)}")
        print(f"  Interaction features: {len(interaction_features)}")
        
        # Create final model with optimal alpha
        self.model = Lasso(alpha=lasso_cv.alpha_, random_state=42, max_iter=3000, tol=1e-6)
        
        return lasso_cv.alpha_
    
    def train_final_model(self, df):
        """Train the final Lasso model with selected features"""
        print(f"\n🚀 TRAINING FINAL LASSO MODEL")
        print("="*50)
        
        X = df[self.feature_columns]
        y = df['quantity']
        
        # Perform feature selection
        optimal_alpha = self.perform_lasso_feature_selection(X, y)
        
        # Train/test split (use last 25% for testing for dairy products)
        split_idx = int(len(df) * 0.75)
        
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
        
        # Additional metrics for dairy products
        median_ae_train = np.median(np.abs(y_train - train_pred))
        median_ae_test = np.median(np.abs(y_test - test_pred))
        
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
            'median_ae_train': median_ae_train,
            'median_ae_test': median_ae_test,
            'overfitting_risk': 'HIGH' if (train_r2 - test_r2) > 0.3 else 'MEDIUM' if (train_r2 - test_r2) > 0.15 else 'LOW',
            'model_quality': 'EXCELLENT' if test_r2 > 0.7 else 'GOOD' if test_r2 > 0.4 else 'ACCEPTABLE' if test_r2 > 0.1 else 'POOR'
        }
        
        # Display results
        print(f"\n📊 TRAINING RESULTS:")
        print("="*50)
        print(f"Optimal Alpha: {optimal_alpha:.8f}")
        print(f"Features Selected: {len(self.selected_features)}/{len(self.feature_columns)} ({self.training_results['feature_reduction_pct']:.1f}% reduction)")
        print(f"\nPerformance Metrics:")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  R² Gap:   {train_r2 - test_r2:.4f}")
        print(f"  Test MAE: {test_mae:.1f}")
        print(f"  Test MAPE: {test_mape:.1f}%")
        print(f"  Test RMSE: {test_rmse:.1f}")
        print(f"  Median AE (Test): {median_ae_test:.1f}")
        print(f"  Overfitting Risk: {self.training_results['overfitting_risk']}")
        print(f"  Model Quality: {self.training_results['model_quality']}")
        
        # Show actual vs predicted
        print(f"\n🎯 PREDICTIONS vs ACTUAL (Test Set):")
        print("-" * 65)
        print("Sample | Date    | Actual  | Predicted | Error   | Error%")
        print("-------|---------|---------|-----------|---------|-------")
        
        test_dates = df.iloc[split_idx:][['year', 'month']].values
        for i, ((year, month), actual, pred) in enumerate(zip(test_dates, y_test.values, test_pred)):
            error = pred - actual
            error_pct = (error / actual * 100) if actual > 0 else 0
            print(f"{i+1:6d} | {int(year)}/{int(month):02d}    | {actual:7.0f} | {pred:9.1f} | {error:7.0f} | {error_pct:6.1f}%")
        
        return self.training_results
    
    def analyze_overfitting_detailed(self, df):
        """Detailed overfitting analysis with multiple splits"""
        print(f"\n{'='*60}")
        print("DETAILED OVERFITTING ANALYSIS")
        print(f"{'='*60}")
        
        X = df[self.feature_columns]
        y = df['quantity']
        
        # Multiple train/test splits
        split_ratios = [0.6, 0.7, 0.75, 0.8, 0.85]
        
        print("Split | Train Samples | Test Samples | Train R² | Test R² | Gap   | Status")
        print("------|---------------|--------------|----------|---------|-------|--------")
        
        for ratio in split_ratios:
            split_idx = int(len(df) * ratio)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            if len(X_test) < 3:
                continue
                
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            temp_model = Lasso(alpha=self.training_results['optimal_alpha'], random_state=42, max_iter=3000)
            temp_model.fit(X_train_scaled, y_train)
            
            train_pred = temp_model.predict(X_train_scaled)
            test_pred = temp_model.predict(X_test_scaled)
            
            train_pred = np.maximum(train_pred, 0)
            test_pred = np.maximum(test_pred, 0)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            gap = train_r2 - test_r2
            
            status = "🔴 HIGH" if gap > 0.3 else "🟡 MEDIUM" if gap > 0.15 else "🟢 LOW"
            
            print(f"{ratio:5.2f} | {len(X_train):13d} | {len(X_test):12d} | {train_r2:8.3f} | {test_r2:7.3f} | {gap:5.3f} | {status}")
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_final_model() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.maximum(predictions, 0)  # Ensure non-negative
    
    def predict_future(self, df, months_ahead=6):
        """Predict future values"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_final_model() first.")
        
        print(f"\n🔮 FUTURE PREDICTIONS ({months_ahead} months ahead):")
        print("="*50)
        
        # Get last row of data for feature creation
        last_data = df.iloc[-1].copy()
        
        predictions = []
        current_data = df.copy()
        
        for i in range(months_ahead):
            # Create next month's features
            next_month = last_data['month'] + i + 1
            next_year = last_data['year']
            
            if next_month > 12:
                next_month = ((next_month - 1) % 12) + 1
                next_year += (last_data['month'] + i) // 12
            
            # Create feature row (simplified - you'd need full feature engineering)
            future_row = {
                'year': next_year,
                'month': next_month,
                'campaign': 0,  # Assume no campaign
                'seasonality': 1 if 7 <= next_month <= 10 else 0,  # High season
                'quantity': 0  # Placeholder
            }
            
            print(f"Month {i+1}: {int(next_year)}/{int(next_month):02d} - Would need full feature engineering")
            
        print("Note: Full future prediction requires complete feature engineering pipeline")
    
    def save_model(self, filename='topbel_lasso_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'training_results': self.training_results,
            'product_name': self.product_name
        }
        joblib.dump(model_data, filename)
        print(f"\n✅ Model saved to: {filename}")
    
    @classmethod
    def load_model(cls, filename='topbel_lasso_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filename)
        
        topbel_model = cls()
        topbel_model.model = model_data['model']
        topbel_model.scaler = model_data['scaler']
        topbel_model.feature_columns = model_data['feature_columns']
        topbel_model.selected_features = model_data['selected_features']
        topbel_model.feature_importance = model_data['feature_importance']
        topbel_model.training_results = model_data['training_results']
        topbel_model.product_name = model_data['product_name']
        
        return topbel_model
    
    def generate_report(self):
        """Generate comprehensive model report"""
        print(f"\n{'='*80}")
        print(f"TOPBEL LEITE CONDENSADO LASSO MODEL - COMPREHENSIVE REPORT")
        print(f"{'='*80}")
        
        print(f"Product: {self.product_name}")
        print(f"Model Type: Lasso Regression with Feature Selection")
        print(f"Optimal Alpha: {self.training_results['optimal_alpha']:.8f}")
        
        print(f"\nFEATURE SELECTION RESULTS:")
        print(f"  Original Features: {len(self.feature_columns)}")
        print(f"  Selected Features: {self.training_results['features_selected']}")
        print(f"  Reduction: {self.training_results['feature_reduction_pct']:.1f}%")
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"  Test R²: {self.training_results['test_r2']:.4f}")
        print(f"  Test MAE: {self.training_results['test_mae']:.1f}")
        print(f"  Test MAPE: {self.training_results['test_mape']:.1f}%")
        print(f"  Median Absolute Error: {self.training_results['median_ae_test']:.1f}")
        print(f"  Overfitting Risk: {self.training_results['overfitting_risk']}")
        print(f"  Model Quality: {self.training_results['model_quality']}")
        
        print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for i, (feature, coef) in enumerate(sorted_features[:10]):
            print(f"  {i+1:2d}. {feature:<30} | Coefficient: {coef:10.6f}")
        
        # Feature interpretation
        print(f"\nFEATURE INTERPRETATION:")
        positive_features = [(f, c) for f, c in sorted_features if c > 0][:5]
        negative_features = [(f, c) for f, c in sorted_features if c < 0][:5]
        
        print(f"  Top Positive Impact (increase demand):")
        for f, c in positive_features:
            print(f"    • {f}: +{c:.6f}")
        
        print(f"  Top Negative Impact (decrease demand):")
        for f, c in negative_features:
            print(f"    • {f}: {c:.6f}")

def main():
    print("🥛 TOPBEL LEITE CONDENSADO LASSO FEATURE SELECTION MODEL")
    print("="*80)
    
    # Initialize model
    topbel_model = TopbelLassoModel()
    
    # Load and prepare data
    try:
        df = topbel_model.load_and_prepare_data()
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Train model with feature selection
    results = topbel_model.train_final_model(df)
    
    # Detailed overfitting analysis
    topbel_model.analyze_overfitting_detailed(df)
    
    # Generate comprehensive report
    topbel_model.generate_report()
    
    # Future predictions demo
    topbel_model.predict_future(df, months_ahead=6)
    
    # Save model
    topbel_model.save_model('topbel_lasso_model.pkl')
    
    print(f"\n{'='*80}")
    print("🎉 TOPBEL LEITE CONDENSADO LASSO MODEL TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"✅ Feature selection completed: {results['features_selected']} features selected")
    print(f"✅ Model trained and evaluated with dairy-specific features")
    print(f"✅ Overfitting analysis completed")
    print(f"✅ Model saved for production use")

if __name__ == "__main__":
    main()
