import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import seaborn as sns

def make_total_price_scorer(A_values):
    
    def total_price_error_loss_scorer(estimator, X_features, y_true_ppm):
        y_pred_ppm = estimator.predict(X_features)
    
        y_true_total = y_true_ppm.values * A_values
        y_pred_total = y_pred_ppm * A_values
        
        mse_total = np.mean((y_true_total - y_pred_total) ** 2)
    
        return -mse_total
    
    return total_price_error_loss_scorer

def analyze_feature_importance(model, X, y, feature_names=None, method='auto', n_repeats=10, random_state=42, scoring=None):
    
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    model_name = type(model).__name__
    
    if method == 'auto':
        if hasattr(model, 'feature_importances_'):
            method = 'builtin'
        elif hasattr(model, 'coef_'):
            method = 'builtin'
        else:
            method = 'permutation'
    
    if method == 'builtin':
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_type = 'Gini/Entropy Importance'
            
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) > 1:
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                importances = np.abs(model.coef_)
            importance_type = 'Coefficient Magnitude'
            
        else:
            raise ValueError(f"{model_name} has no built-in feature importance atrributes.")
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Method': importance_type
        })
        
    elif method == 'permutation':
        perm_importance = permutation_importance(
            model, X_array, y, scoring=scoring,
            n_repeats=n_repeats, 
            random_state=random_state,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std,
            'Method': 'Permutation Importance'
        })
    
    else:
        raise ValueError(f"Method {method} is not available")
    
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    total_importance = importance_df['Importance'].sum()
    if total_importance > 0:
        importance_df['Percentage'] = (importance_df['Importance'] / total_importance * 100).round(2)
    else:
        importance_df['Percentage'] = 0
    
    return importance_df


def plot_feature_importance(importance_df, top_n=20, figsize=(15, 12), title=None):
    plot_df = importance_df.head(top_n).copy()
    plot_df = plot_df.sort_values('Importance', ascending=True)  
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'Std' in plot_df.columns:
        ax.barh(plot_df['Feature'], plot_df['Importance'], xerr=plot_df['Std'], 
                capsize=5, alpha=0.7, color='steelblue')
    else:
        ax.barh(plot_df['Feature'], plot_df['Importance'], alpha=0.7, color='steelblue')
    
    ax.set_xlabel('Importance', fontsize=50)
    ax.set_ylabel('Feature', fontsize=50)
    
    if title:
        ax.set_title(title, fontsize=22, fontweight='bold')
    else:
        method = plot_df['Method'].iloc[0] if 'Method' in plot_df.columns else 'Feature Importance'
        ax.set_title(f'Top {top_n} Features - {method}', fontsize=22, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    return fig, ax

