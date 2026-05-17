import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import joblib
import warnings

warnings.filterwarnings('ignore')

def train_eval_model():
    df = pd.read_csv("../csv/dataset_features.csv").fillna(0)
    X = df.drop(columns=['label'])
    y = df['label']
    feature_names_out = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class_weights = {0: 1.0, 1: 8.0, 2: 1.0, 3: 5.0, 4: 3.0}
    class_names = ['0: Normal', '1: UDP Flood', '2: Frame Pend', '3: DIS Flood', '4: Inc. Rank']
    num_classes = len(class_names)
    plot_colors = ['navy', 'darkorange', 'green', 'red', 'purple']

    sample_weights_train = np.array([class_weights[label] for label in y_train])

    xgb_model = XGBClassifier(
        n_estimators=300,        
        learning_rate=0.1,       
        max_depth=21,            
        num_class=num_classes,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)

    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)

    print("\n" + "="*60)
    print("Resultados de evaluacion")
    print("="*60)
    print(classification_report(y_test, y_pred, digits=4))

    if not os.path.exists('resultados'): os.makedirs('resultados')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Etiqueta real', fontsize=12)
    plt.xlabel('Etiqueta estimada', fontsize=12)
    plt.tight_layout()
    plt.savefig('resultados/matriz_confusion_xgb.png', dpi=300)
    plt.close()

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])

    plt.figure(figsize=(10, 8))
    for i, color in zip(range(num_classes), plot_colors):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{class_names[i]} (AP = {ap:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Sensibilidad (Recall)', fontsize=12)
    plt.ylabel('Precisión (Precision)', fontsize=12)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('resultados/curvas_pr_xgb.png', dpi=300)
    plt.close()

    model_path = 'resultados/xgboost.joblib'
    joblib.dump(xgb_model, model_path)

if __name__ == "__main__":
    train_eval_model()
