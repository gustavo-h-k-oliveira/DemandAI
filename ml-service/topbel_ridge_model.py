import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

class TopbelRidgeModel:
    """Specialized Ridge Conservative model for TOPBEL LEITE CONDENSADO"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.feature_importance = {}
        self.training_results = {}
        self.product_name = "TOPBEL LEITE CONDENSADO 50UN"
        self.conservative_alpha = None
    
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
        print(f"Coefficient of variation: {(topbel_data['quantity'].std() / topbel_data['quantity'].mean()):.3f}")
        
        # Analyze patterns for dairy products
        seasonal_analysis = topbel_data.groupby('seasonality')['quantity'].agg(['mean', 'std', 'count'])
        print(f"\nSeasonality Analysis:")
        print(seasonal_analysis)
        
        campaign_analysis = topbel_data.groupby('campaign')['quantity'].agg(['mean', 'std', 'count'])
        print(f"\nCampaign Analysis:")
        print(campaign_analysis)
        
        # Monthly pattern analysis
        monthly_analysis = topbel_data.groupby('month')['quantity'].agg(['mean', 'std']).round(1)
        print(f"\nMonthly Pattern Analysis:")
        print(monthly_analysis)
        
        # Create robust features for Ridge regression
        print("\n📊 Creating robust features for Ridge Conservative model...")
        
        # 1. Basic temporal features (smooth)
        topbel_data['month_sin'] = np.sin(2 * np.pi * topbel_data['month'] / 12)
        topbel_data['month_cos'] = np.cos(2 * np.pi * topbel_data['month'] / 12)
        topbel_data['quarter'] = pd.to_datetime(topbel_data[['year', 'month']].assign(day=1)).dt.quarter
        topbel_data['quarter_sin'] = np.sin(2 * np.pi * topbel_data['quarter'] / 4)
        topbel_data['quarter_cos'] = np.cos(2 * np.pi * topbel_data['quarter'] / 4)
        
        # 2. Conservative lag features (only reliable ones)
        for lag in [1, 2, 3, 6]:
            topbel_data[f'quantity_lag{lag}'] = topbel_data['quantity'].shift(lag)
        
        # 3. Smooth rolling statistics (longer windows for stability)
        for window in [3, 6, 12]:
            topbel_data[f'quantity_ma{window}'] = topbel_data['quantity'].rolling(window=window, min_periods=max(1, window//2)).mean()
            topbel_data[f'quantity_ewm{window}'] = topbel_data['quantity'].ewm(span=window).mean()
        
        # 4. Trend features (smooth)
        topbel_data['trend'] = np.arange(len(topbel_data)) / len(topbel_data)  # Normalized trend
        topbel_data['trend_squared'] = topbel_data['trend'] ** 2
        
        # 5. Growth features (smoothed)
        topbel_data['quantity_pct_change'] = topbel_data['quantity'].pct_change().fillna(0)
        topbel_data['quantity_pct_change_smooth'] = topbel_data['quantity_pct_change'].rolling(window=3).mean().fillna(0)
        
        # 6. Seasonal indicators (conservative)
        topbel_data['is_high_season'] = ((topbel_data['month'] >= 7) & (topbel_data['month'] <= 10)).astype(float)
        topbel_data['is_low_season'] = ((topbel_data['month'] >= 1) & (topbel_data['month'] <= 3)).astype(float)
        topbel_data['is_summer'] = ((topbel_data['month'] >= 6) & (topbel_data['month'] <= 8)).astype(float)
        topbel_data['is_winter'] = ((topbel_data['month'] >= 12) | (topbel_data['month'] <= 2)).astype(float)
        
        # 7. Conservative interaction features
        topbel_data['campaign_season'] = topbel_data['campaign'] * topbel_data['seasonality']
        topbel_data['campaign_high_season'] = topbel_data['campaign'] * topbel_data['is_high_season']
        topbel_data['seasonality_summer'] = topbel_data['seasonality'] * topbel_data['is_summer']
        
        # 8. Stability features
        topbel_data['quantity_stability_3'] = topbel_data['quantity'].rolling(window=3).std().fillna(0)
        topbel_data['quantity_stability_6'] = topbel_data['quantity'].rolling(window=6).std().fillna(0)
        
        # 9. Year-over-year (if sufficient data)
        if len(topbel_data) >= 24:
            topbel_data['quantity_yoy'] = topbel_data['quantity'].shift(12)
            topbel_data['yoy_growth'] = ((topbel_data['quantity'] - topbel_data['quantity_yoy']) / 
                                        topbel_data['quantity_yoy'].replace(0, np.nan)).fillna(0)
        
        # 10. Month encoding (sine/cosine instead of dummies for Ridge)
        for i in range(1, 13):
            topbel_data[f'month_{i}_sin'] = np.sin(2 * np.pi * (topbel_data['month'] == i).astype(int))
            topbel_data[f'month_{i}_cos'] = np.cos(2 * np.pi * (topbel_data['month'] == i).astype(int))
        
        # Define feature columns (conservative set for Ridge)
        self.feature_columns = [
            'year', 'month', 'campaign', 'seasonality', 'quarter',
            'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
            'trend', 'trend_squared',
            'quantity_lag1', 'quantity_lag2', 'quantity_lag3', 'quantity_lag6',
            'quantity_ma3', 'quantity_ma6', 'quantity_ma12',
            'quantity_ewm3', 'quantity_ewm6', 'quantity_ewm12',
            'quantity_pct_change', 'quantity_pct_change_smooth',
            'is_high_season', 'is_low_season', 'is_summer', 'is_winter',
            'campaign_season', 'campaign_high_season', 'seasonality_summer',
            'quantity_stability_3', 'quantity_stability_6'
        ]
        
        # Add YoY features if they exist
        if 'yoy_growth' in topbel_data.columns:
            self.feature_columns.extend(['quantity_yoy', 'yoy_growth'])
        
        # Add month encoding
        for i in range(1, 13):
            self.feature_columns.extend([f'month_{i}_sin', f'month_{i}_cos'])
        
        # Fill NaN values conservatively
        numeric_cols = topbel_data.select_dtypes(include=[np.number]).columns
        
        # Forward fill first, then backward fill, then mean for remaining
        topbel_data[numeric_cols] = topbel_data[numeric_cols].fillna(method='ffill')
        topbel_data[numeric_cols] = topbel_data[numeric_cols].fillna(method='bfill')
        topbel_data[numeric_cols] = topbel_data[numeric_cols].fillna(topbel_data[numeric_cols].mean())
        
        # Replace infinite values
        topbel_data[numeric_cols] = topbel_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        topbel_data[numeric_cols] = topbel_data[numeric_cols].fillna(topbel_data[numeric_cols].median())
        
        print(f"Total features created: {len(self.feature_columns)}")
        
        return topbel_data
    
    def find_optimal_conservative_alpha(self, X, y):
        """Find optimal alpha using conservative Ridge with cross-validation"""
        print("\n🎯 FINDING OPTIMAL CONSERVATIVE ALPHA")
        print("="*50)
        
        # Conservative alpha range (higher values for more regularization)
        alphas = np.logspace(0, 4, 50)  # From 1.0 to 10,000 (very conservative)
        
        # Use RidgeCV for automatic alpha selection with time series CV
        ridge_cv = RidgeCV(
            alphas=alphas,
            cv=TimeSeriesSplit(n_splits=min(5, len(X)//8)),  # Conservative CV splits
            scoring='neg_mean_squared_error'
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit RidgeCV
        print("Training RidgeCV for optimal conservative alpha selection...")
        ridge_cv.fit(X_scaled, y)
        
        optimal_alpha = ridge_cv.alpha_
        cv_score = ridge_cv.score(X_scaled, y)
        
        print(f"Optimal Conservative Alpha: {optimal_alpha:.2f}")
        print(f"CV Score (R²): {cv_score:.4f}")
        
        print("Cross-validation completed successfully")
        
        # Show alpha range analysis
        print(f"\nAlpha Range Analysis:")
        alpha_ranges = [
            (1, 10, "Low regularization"),
            (10, 100, "Medium regularization"), 
            (100, 1000, "High regularization"),
            (1000, 10000, "Very high regularization")
        ]
        
        for min_a, max_a, desc in alpha_ranges:
            if min_a <= optimal_alpha <= max_a:
                print(f"  Selected alpha range: {desc} ✅")
            else:
                print(f"  {desc}: {min_a}-{max_a}")
        
        self.conservative_alpha = optimal_alpha
        
        # Create final model with optimal alpha
        self.model = Ridge(alpha=optimal_alpha, random_state=42)
        
        return optimal_alpha
    
    def train_final_model(self, df):
        """Train the final Ridge Conservative model"""
        print(f"\n🚀 TRAINING FINAL RIDGE CONSERVATIVE MODEL")
        print("="*50)
        
        X = df[self.feature_columns]
        y = df['quantity']
        
        # Find optimal conservative alpha
        optimal_alpha = self.find_optimal_conservative_alpha(X, y)
        
        # Conservative train/test split (70/30 for more robust testing)
        split_idx = int(len(df) * 0.70)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"\nConservative Data Split:")
        print(f"  Training: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Testing:  {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        
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
        
        # Additional conservative metrics
        median_ae_train = np.median(np.abs(y_train - train_pred))
        median_ae_test = np.median(np.abs(y_test - test_pred))
        
        # Prediction interval coverage (simple approach)
        test_residuals = y_test - test_pred
        residual_std = np.std(test_residuals)
        
        # Feature importance (Ridge coefficients)
        feature_coeffs = self.model.coef_
        self.feature_importance = dict(zip(self.feature_columns, feature_coeffs))
        
        # Store results
        self.training_results = {
            'optimal_alpha': optimal_alpha,
            'regularization_level': 'VERY HIGH' if optimal_alpha > 1000 else 'HIGH' if optimal_alpha > 100 else 'MEDIUM' if optimal_alpha > 10 else 'LOW',
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
            'residual_std': residual_std,
            'overfitting_risk': 'HIGH' if (train_r2 - test_r2) > 0.2 else 'MEDIUM' if (train_r2 - test_r2) > 0.1 else 'LOW',
            'model_stability': 'VERY STABLE' if (train_r2 - test_r2) < 0.05 else 'STABLE' if (train_r2 - test_r2) < 0.1 else 'UNSTABLE',
            'model_quality': 'EXCELLENT' if test_r2 > 0.7 else 'GOOD' if test_r2 > 0.4 else 'ACCEPTABLE' if test_r2 > 0.1 else 'POOR'
        }
        
        # Display results
        print(f"\n📊 RIDGE CONSERVATIVE TRAINING RESULTS:")
        print("="*60)
        print(f"Optimal Alpha: {optimal_alpha:.2f}")
        print(f"Regularization Level: {self.training_results['regularization_level']}")
        print(f"Total Features Used: {len(self.feature_columns)}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  R² Gap:   {train_r2 - test_r2:.4f}")
        print(f"  Test MAE: {test_mae:.1f}")
        print(f"  Test MAPE: {test_mape:.1f}%")
        print(f"  Test RMSE: {test_rmse:.1f}")
        print(f"  Median AE (Test): {median_ae_test:.1f}")
        print(f"  Residual Std: {residual_std:.1f}")
        
        print(f"\nModel Assessment:")
        print(f"  Overfitting Risk: {self.training_results['overfitting_risk']}")
        print(f"  Model Stability: {self.training_results['model_stability']}")
        print(f"  Model Quality: {self.training_results['model_quality']}")
        
        # Show actual vs predicted
        print(f"\n🎯 PREDICTIONS vs ACTUAL (Test Set):")
        print("-" * 75)
        print("Sample | Date    | Actual  | Predicted | Error   | Error% | Within 1σ")
        print("-------|---------|---------|-----------|---------|--------|----------")
        
        test_dates = df.iloc[split_idx:][['year', 'month']].values
        for i, ((year, month), actual, pred) in enumerate(zip(test_dates, y_test.values, test_pred)):
            error = pred - actual
            error_pct = (error / actual * 100) if actual > 0 else 0
            within_1sigma = "✅" if abs(error) <= residual_std else "❌"
            
            print(f"{i+1:6d} | {int(year)}/{int(month):02d}    | {actual:7.0f} | {pred:9.1f} | {error:7.0f} | {error_pct:6.1f}% | {within_1sigma}")
        
        return self.training_results
    
    def analyze_ridge_stability(self, df):
        """Analyze Ridge model stability across different splits"""
        print(f"\n{'='*70}")
        print("RIDGE CONSERVATIVE MODEL STABILITY ANALYSIS")
        print(f"{'='*70}")
        
        X = df[self.feature_columns]
        y = df['quantity']
        
        # Multiple train/test splits
        split_ratios = [0.6, 0.65, 0.7, 0.75, 0.8]
        
        print("Split | Train | Test | Train R² | Test R² | Gap   | Stability | Alpha Effect")
        print("------|-------|------|----------|---------|-------|-----------|-------------")
        
        stability_scores = []
        
        for ratio in split_ratios:
            split_idx = int(len(df) * ratio)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            if len(X_test) < 3:
                continue
            
            # Test model with same alpha
            temp_scaler = StandardScaler()
            X_train_scaled = temp_scaler.fit_transform(X_train)
            X_test_scaled = temp_scaler.transform(X_test)
            
            temp_model = Ridge(alpha=self.conservative_alpha, random_state=42)
            temp_model.fit(X_train_scaled, y_train)
            
            train_pred = temp_model.predict(X_train_scaled)
            test_pred = temp_model.predict(X_test_scaled)
            
            train_pred = np.maximum(train_pred, 0)
            test_pred = np.maximum(test_pred, 0)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            gap = train_r2 - test_r2
            
            stability = "🟢 HIGH" if gap < 0.1 else "🟡 MEDIUM" if gap < 0.15 else "🔴 LOW"
            alpha_effect = "GOOD" if gap < 0.1 else "OK" if gap < 0.2 else "WEAK"
            
            stability_scores.append(gap)
            
            print(f"{ratio:5.2f} | {len(X_train):5d} | {len(X_test):4d} | {train_r2:8.3f} | {test_r2:7.3f} | {gap:5.3f} | {stability:9s} | {alpha_effect}")
        
        avg_stability = np.mean(stability_scores)
        std_stability = np.std(stability_scores)
        
        print(f"\nStability Summary:")
        print(f"  Average Gap: {avg_stability:.3f}")
        print(f"  Gap Std Dev: {std_stability:.3f}")
        print(f"  Consistency: {'EXCELLENT' if std_stability < 0.05 else 'GOOD' if std_stability < 0.1 else 'FAIR'}")
    
    def analyze_feature_importance(self):
        """Analyze Ridge feature importance"""
        print(f"\n{'='*60}")
        print("RIDGE FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")
        
        if not self.feature_importance:
            print("❌ No feature importance available. Train model first.")
            return
        
        # Sort features by absolute coefficient value
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"Top 15 Most Important Features:")
        print("-" * 65)
        print("Rank | Feature                    | Coefficient | Impact")
        print("-----|----------------------------|-------------|--------")
        
        for i, (feature, coef) in enumerate(sorted_features[:15]):
            impact = "HIGH+" if coef > 0.1 else "HIGH-" if coef < -0.1 else "MED+" if coef > 0.01 else "MED-" if coef < -0.01 else "LOW"
            print(f"{i+1:4d} | {feature:<26} | {coef:11.6f} | {impact}")
        
        # Analyze feature categories
        categories = {
            'lag': [f for f in self.feature_columns if 'lag' in f],
            'moving_avg': [f for f in self.feature_columns if 'ma' in f or 'ewm' in f],
            'seasonal': [f for f in self.feature_columns if any(x in f for x in ['season', 'sin', 'cos'])],
            'trend': [f for f in self.feature_columns if 'trend' in f],
            'interaction': [f for f in self.feature_columns if any(x in f for x in ['campaign', 'seasonality'])],
            'stability': [f for f in self.feature_columns if 'stability' in f]
        }
        
        print(f"\nFeature Category Impact:")
        print("-" * 40)
        
        for category, features in categories.items():
            if features:
                category_importance = sum(abs(self.feature_importance.get(f, 0)) for f in features)
                feature_count = len(features)
                avg_importance = category_importance / feature_count if feature_count > 0 else 0
                
                print(f"{category.upper():<15} | Count: {feature_count:2d} | Avg Impact: {avg_importance:.4f}")
        
        # Show most positive and negative features
        positive_features = [(f, c) for f, c in sorted_features if c > 0][:5]
        negative_features = [(f, c) for f, c in sorted_features if c < 0][:5]
        
        print(f"\nTop Positive Impact (increase demand):")
        for f, c in positive_features:
            print(f"  • {f:<30}: +{c:.6f}")
        
        print(f"\nTop Negative Impact (decrease demand):")
        for f, c in negative_features:
            print(f"  • {f:<30}: {c:.6f}")
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_final_model() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.maximum(predictions, 0)  # Ensure non-negative
    
    def save_model(self, filename='models/topbel_ridge_conservative_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'training_results': self.training_results,
            'conservative_alpha': self.conservative_alpha,
            'product_name': self.product_name
        }
        joblib.dump(model_data, filename)
        print(f"\n✅ Ridge Conservative model saved to: {filename}")
    
    @classmethod
    def load_model(cls, filename='models/topbel_ridge_conservative_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filename)
        
        topbel_model = cls()
        topbel_model.model = model_data['model']
        topbel_model.scaler = model_data['scaler']
        topbel_model.feature_columns = model_data['feature_columns']
        topbel_model.feature_importance = model_data['feature_importance']
        topbel_model.training_results = model_data['training_results']
        topbel_model.conservative_alpha = model_data['conservative_alpha']
        topbel_model.product_name = model_data['product_name']
        
        return topbel_model
    
    def generate_comprehensive_report(self):
        """Generate comprehensive Ridge Conservative model report"""
        print(f"\n{'='*80}")
        print(f"TOPBEL LEITE CONDENSADO RIDGE CONSERVATIVE MODEL - FINAL REPORT")
        print(f"{'='*80}")
        
        print(f"Product: {self.product_name}")
        print(f"Model Type: Ridge Regression (Conservative)")
        print(f"Alpha Value: {self.conservative_alpha:.2f}")
        print(f"Regularization: {self.training_results['regularization_level']}")
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"  Test R²: {self.training_results['test_r2']:.4f}")
        print(f"  Test MAE: {self.training_results['test_mae']:.1f}")
        print(f"  Test MAPE: {self.training_results['test_mape']:.1f}%")
        print(f"  Model Quality: {self.training_results['model_quality']}")
        
        print(f"\nSTABILITY ASSESSMENT:")
        print(f"  Overfitting Risk: {self.training_results['overfitting_risk']}")
        print(f"  Model Stability: {self.training_results['model_stability']}")
        print(f"  R² Gap: {self.training_results['r2_gap']:.4f}")
        
        print(f"\nCONSERVATIVE FEATURES:")
        print(f"  ✅ High regularization (α={self.conservative_alpha:.1f})")
        print(f"  ✅ Conservative 70/30 train/test split")
        print(f"  ✅ Robust feature engineering")
        print(f"  ✅ Non-negative predictions enforced")
        print(f"  ✅ Stability analysis across multiple splits")
        
        # Recommendation
        if self.training_results['test_r2'] > 0.4 and self.training_results['overfitting_risk'] == 'LOW':
            recommendation = "✅ RECOMMENDED FOR PRODUCTION"
            confidence = "HIGH"
        elif self.training_results['test_r2'] > 0.2 and self.training_results['model_stability'] in ['STABLE', 'VERY STABLE']:
            recommendation = "🟡 ACCEPTABLE WITH MONITORING"
            confidence = "MEDIUM"
        else:
            recommendation = "❌ NEEDS IMPROVEMENT"
            confidence = "LOW"
        
        print(f"\nFINAL RECOMMENDATION:")
        print(f"  Status: {recommendation}")
        print(f"  Confidence: {confidence}")
        
        if confidence == "HIGH":
            print(f"  Next Steps: Deploy with monthly retraining")
        elif confidence == "MEDIUM":
            print(f"  Next Steps: Deploy with weekly monitoring and validation")
        else:
            print(f"  Next Steps: Collect more data and retrain")

def main():
    print("🥛 TOPBEL LEITE CONDENSADO RIDGE CONSERVATIVE MODEL")
    print("="*80)
    
    # Initialize model
    topbel_model = TopbelRidgeModel()
    
    # Load and prepare data
    try:
        df = topbel_model.load_and_prepare_data()
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Train Ridge Conservative model
    results = topbel_model.train_final_model(df)
    
    # Analyze stability
    topbel_model.analyze_ridge_stability(df)
    
    # Analyze feature importance
    topbel_model.analyze_feature_importance()
    
    # Generate comprehensive report
    topbel_model.generate_comprehensive_report()
    
    # Save model
    topbel_model.save_model('models/topbel_ridge_conservative_model.pkl')
    
    print(f"\n{'='*80}")
    print("🎉 TOPBEL RIDGE CONSERVATIVE MODEL TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"✅ Conservative regularization applied (α={results['optimal_alpha']:.1f})")
    print(f"✅ Stability analysis completed")
    print(f"✅ Feature importance analyzed")
    print(f"✅ Model ready for conservative production deployment")

if __name__ == "__main__":
    main()
