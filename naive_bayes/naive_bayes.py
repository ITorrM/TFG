import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap  
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import joblib

def train_eval_model():
    df = pd.read_csv("../csv/dataset_features.csv").fillna(0)
    X = df.drop(columns=['label'])
    y = df['label']
    
    cols_to_bypass = ['protocol', 'mac_pending', 'rpl_code']
    cols_to_bypass = [c for c in cols_to_bypass if c in X.columns]
    cols_to_discretize = [c for c in X.columns if c not in cols_to_bypass]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('bins', KBinsDiscretizer(n_bins=110, encode='ordinal', strategy='uniform'), cols_to_discretize),
            ('bypass', 'passthrough', cols_to_bypass)
        ]
    )

    class_names = ['0: Normal', '1: UDP Flood', '2: Frame Pend', '3: DIS Flood', '4: Inc. Rank']
    num_classes = len(class_names)

    nb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('nb', CategoricalNB(
            min_categories=110,
            fit_prior=True
        ))
    ])

    nb_pipeline.fit(X_train, y_train)

    y_pred = nb_pipeline.predict(X_test)
    y_score = nb_pipeline.predict_proba(X_test)

    print("\n" + "="*60)
    print("Resultados de la evaluacion")
    print("="*60)
    print(classification_report(y_test, y_pred, digits=4))

    if not os.path.exists('resultados'): os.makedirs('resultados')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.savefig('resultados/matriz_confusion_nb.png', dpi=300)
    plt.close()

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    plt.figure(figsize=(10, 8))
    colors = ['navy', 'darkorange', 'cornflowerblue', 'red', 'green']
    for i, color in zip(range(num_classes), colors):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, color=color, lw=2, label=f'{class_names[i]} (AP = {ap:.4f})')
        
    plt.xlabel('Sensibilidad (Recall)', fontsize=12)
    plt.ylabel('Precisión (Precision)', fontsize=12)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('resultados/curvas_pr_nb.png', dpi=300)
    plt.close()

    nb_model = nb_pipeline.named_steps['nb']
    preprocessor_model = nb_pipeline.named_steps['preprocessor']
    
    X_train_transformed = preprocessor_model.transform(X_train)
    X_test_transformed = preprocessor_model.transform(X_test)
    feature_names_out = cols_to_discretize + cols_to_bypass
    
    background = shap.kmeans(X_train_transformed, 50)
    explainer = shap.KernelExplainer(nb_model.predict_proba, background)
    
    X_test_sample = shap.sample(X_test_transformed, 200)
    shap_values = explainer.shap_values(X_test_sample, silent=True)
    
    if isinstance(shap_values, list):
        shap_attacks = [shap_values[1], shap_values[2], shap_values[3], shap_values[4]]
    else:
        shap_attacks = [shap_values[:, :, 1], shap_values[:, :, 2], shap_values[:, :, 3], shap_values[:, :, 4]]
        
    attack_class_names = ['1: UDP Flood', '2: Frame Pend', '3: DIS Flood', '4: Inc. Rank']
    
    mean_abs_shaps = [np.abs(sa).mean(axis=0) for sa in shap_attacks]
    sum_mean_abs = np.sum(mean_abs_shaps, axis=0)
    max_bar_length = np.max(sum_mean_abs) if np.max(sum_mean_abs) > 0 else 1.0 
    
    shap_attacks_norm = [sa / max_bar_length for sa in shap_attacks]
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_attacks_norm, 
        X_test_sample, 
        feature_names=feature_names_out, 
        class_names=attack_class_names, 
        plot_type="bar",
        show=False
    )
    
    plt.xlabel("")
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig('resultados/shap_resumen_ataques_nb.png', dpi=300, bbox_inches='tight')
    plt.close() 

if __name__ == "__main__":
    train_eval_model()
