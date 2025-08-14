import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
import os
import gdown
warnings.filterwarnings('ignore')

from pathlib import Path

class LoanExplainer:
    def __init__(self,
                 model_path: str = None,
                 cal_model_path: str = None,
                 threshold_path: str = None):
        base_dir = Path(__file__).parent.parent  # repo root
        self.model = joblib.load(model_path or base_dir / "models/model_xgb.joblib")
        self.cal_model = joblib.load(cal_model_path or base_dir / "models/model_xgb_calibrated.joblib")
        self.feature_names = None
        self.threshold = joblib.load(threshold_path or base_dir / "models/model_threshold.joblib")
        self.explainer = None
        self.shap_values_cache = {}
        
    def _get_explainer(self):
        """Lazy load SHAP explainer"""
        if self.explainer is None:
            print("Initializing SHAP explainer...")
            self.explainer = shap.Explainer(self.model)
        return self.explainer
    
    def predict_loan(self, X: pd.DataFrame) -> Dict:
        """Predict loan default probability and decision"""
        # Get raw prediction
        raw_proba = self.model.predict_proba(X)[0, 1]
        
        # Get calibrated prediction
        cal_proba = self.cal_model.predict_proba(X)[0, 1]
        
        # Make decision
        decision = "Approved" if cal_proba <= self.threshold else "Declined"
        confidence = abs(cal_proba - self.threshold)
        
        return {
            'raw_probability': raw_proba,
            'calibrated_probability': cal_proba,
            'decision': decision,
            'confidence': confidence,
            'threshold': self.threshold
        }
    
    def explain_single_loan(self, X: pd.DataFrame, return_fig: bool = True):
        """Explain a single loan application"""
        explainer = self._get_explainer()
        
        # Calculate SHAP values
        shap_values = explainer(X)
        
        # Get prediction info
        pred_info = self.predict_loan(X)
        
        if return_fig:
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.plots.waterfall(shap_values[0], show=False)
            
            # Add title with prediction info
            decision_color = 'green' if pred_info['decision'] == 'Approved' else 'red'
            plt.suptitle(
                f"Loan Decision: {pred_info['decision']} | "
                f"Default Probability: {pred_info['calibrated_probability']:.3f} | "
                f"Threshold: {pred_info['threshold']:.3f}",
                fontsize=14, color=decision_color, fontweight='bold'
            )
            plt.tight_layout()
            return fig, pred_info, shap_values
        
        return pred_info, shap_values
    
    def get_feature_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Get global feature importance using SHAP"""
        explainer = self._get_explainer()
        shap_values = explainer(X.sample(min(1000, len(X))))  # Sample for efficiency
        if self.feature_names is None:
            self.feature_names = list(X.columns)

        # Calculate mean absolute SHAP values
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values.values).mean(0)
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def create_summary_plot(self, X: pd.DataFrame, max_display: int = 20):
        """Create SHAP summary plot"""
        explainer = self._get_explainer()
        sample_size = min(1000, len(X))
        X_sample = X.sample(sample_size, random_state=42)
        shap_values = explainer(X_sample)
        if self.feature_names is None:
            self.feature_names = list(X.columns)

        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, 
                         feature_names=self.feature_names, 
                         max_display=max_display, show=False)
        plt.tight_layout()
        return fig
    
    def create_dependence_plot(self, X: pd.DataFrame, feature: str, interaction_feature: str = None):
        """Create SHAP dependence plot using modern API"""
        try:
            explainer = self._get_explainer()
            sample_size = min(1000, len(X))
            X_sample = X.sample(sample_size, random_state=42).reset_index(drop=True)
            
            shap_values = explainer(X_sample)
            available_features = list(X_sample.columns)
            
            if feature not in available_features:
                raise ValueError(f"Feature '{feature}' not found")
            
            feature_idx = available_features.index(feature)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if interaction_feature and interaction_feature in available_features:
                interaction_idx = available_features.index(interaction_feature)
                shap.plots.scatter(
                    shap_values[:, feature_idx], 
                    color=shap_values[:, interaction_idx],
                    show=False,
                    ax=ax
                )
                title = f"SHAP Dependence: {feature} (colored by {interaction_feature})"
            else:
                # For auto-coloring, pass the entire shap_values object
                shap.plots.scatter(
                    shap_values[:, feature_idx],
                    color=shap_values,
                    show=False,
                    ax=ax
                )
                title = f"SHAP Dependence: {feature} (auto-colored)"
            
            ax.set_xlabel(feature)
            ax.set_ylabel(f"SHAP value for {feature}")
            plt.title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error: {str(e)}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', 
                    transform=ax.transAxes)
            return fig




    
    def get_risk_distribution(self, X: pd.DataFrame) -> Dict:
        """Get risk score distribution statistics"""
        probabilities = self.cal_model.predict_proba(X)[:, 1]
        
        return {
            'mean_risk': probabilities.mean(),
            'median_risk': np.median(probabilities),
            'std_risk': probabilities.std(),
            'percentiles': {
                '25th': np.percentile(probabilities, 25),
                '75th': np.percentile(probabilities, 75),
                '95th': np.percentile(probabilities, 95),
                '99th': np.percentile(probabilities, 99)
            },
            'approval_rate': (probabilities <= self.threshold).mean(),
            'high_risk_rate': (probabilities > 0.5).mean()
        }
    
    def simulate_threshold_impact(self, X: pd.DataFrame, threshold_range: Tuple[float, float] = (0.1, 0.5)):
        """Simulate impact of different thresholds"""
        probabilities = self.cal_model.predict_proba(X)[:, 1]
        thresholds = np.linspace(threshold_range[0], threshold_range[1], 50)
        
        results = []
        for threshold in thresholds:
            approved = probabilities <= threshold
            approval_rate = approved.mean()
            avg_risk_approved = probabilities[approved].mean() if approved.sum() > 0 else 0
            
            results.append({
                'threshold': threshold,
                'approval_rate': approval_rate,
                'avg_risk_approved': avg_risk_approved,
                'expected_bad_rate': avg_risk_approved
            })
        
        return pd.DataFrame(results)
    
    def batch_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict for multiple loans"""
        raw_probas = self.model.predict_proba(X)[:, 1]
        cal_probas = self.cal_model.predict_proba(X)[:, 1]
        decisions = np.where(cal_probas <= self.threshold, 'Approved', 'Declined')
        
        results_df = pd.DataFrame({
            'raw_probability': raw_probas,
            'calibrated_probability': cal_probas,
            'decision': decisions,
            'risk_category': pd.cut(cal_probas, 
                                  bins=[0, 0.1, 0.25, 0.5, 1.0], 
                                  labels=['Low', 'Medium', 'High', 'Very High'])
        })
        
        return results_df


# Convenience functions for backward compatibility
def load_test_data(test_path: str = "data/processed/model_df.parquet") -> Tuple[pd.DataFrame]:
    """Load test data, downloading from Google Drive if not present"""
    
    # Google Drive file ID
    file_id = "1xypiRV2aUU2jXEUG2k8yqsBwToyVFCqP"
    
    if not os.path.exists(test_path):
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        print(f"Downloading test data from Google Drive to {test_path} ...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", test_path, quiet=False)

    df = pd.read_parquet(test_path)
    X = df.drop(columns=['loan_status']) if "loan_status" in df.columns else df
    y = df["loan_status"] if "loan_status" in df.columns else None
    return X, y

def create_explainer():
    """Create and return explainer instance"""
    return LoanExplainer()

if __name__ == "__main__":
    # Example usage
    explainer = create_explainer()
    
    # Load some test data
    X_test, y_test = load_test_data()
    
    # Example: explain a single loan
    sample_loan = X_test.iloc[[0]]  # First loan
    fig, pred_info, shap_vals = explainer.explain_single_loan(sample_loan)
    plt.show()
    
    print(f"Prediction: {pred_info}")
    
    # Example: get feature importance
    importance_df = explainer.get_feature_importance(X_test.head(500))
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))