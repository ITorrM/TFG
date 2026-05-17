import os
import pandas as pd
import numpy as np
import joblib  
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def clean_window(df, col_time='time_window'):
    interval = 600 if df[col_time].max() < 100000 else 600000
    mask_repair = (df[col_time] % interval == 0) & (df[col_time] > 0)
    return df[~mask_repair].copy()

def train_eval_model():
    os.makedirs("resultados", exist_ok=True)
    
    df = pd.read_csv("../csv/dataset_flujos.csv")

    df = clean_window(df)
    col_mac = 'mac_src' if 'mac_src' in df.columns else 'src_mac'

    cols_to_drop = ['time_window', col_mac, 'label']
    X = df.drop(columns=cols_to_drop)
    
    model = IsolationForest(
        n_estimators=300,      
        contamination=0.25,    
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)

    if not os.path.exists('resultados'): os.makedirs('resultados')

    joblib.dump(model, "resultados/isolation_forest.joblib")

    df['anomaly_score'] = model.decision_function(X) 
    
    UMBRAL_NIVEL_1 = 0.0

    estudios = [
        ('7',  0, 'Benigna',  'Ataque 1'), 
        ('12', 1, 'Atacante', 'Ataque 1'),
        ('25', 2, 'Atacante', 'Ataque 2'),
        ('34', 0, 'Benigna',  'Ataque 3'),
        ('38', 3, 'Atacante', 'Ataque 3'),
        ('44', 0, 'Benigna',  'Ataque 4'),
        ('52', 4, 'Atacante', 'Ataque 4')
    ]

    for node_id, label_val, rol, escenario in estudios:
        df_plot = df[df[col_mac].astype(str) == str(node_id)].sort_values(by='time_window')
        if df_plot.empty: continue
        
        df_null = df_plot[df_plot['protocol'] == 0]
        df_udp = df_plot[df_plot['protocol'] == 1]
        df_rpl = df_plot[df_plot['protocol'] == 2]

        score_min, score_max = df_plot['anomaly_score'].min(), df_plot['anomaly_score'].max()
        limite_y_score = (min(score_min, UMBRAL_NIVEL_1) - 0.05, max(score_max, UMBRAL_NIVEL_1) + 0.05)

        fig1, ax1 = plt.subplots(figsize=(16, 6))
        
        if not df_null.empty: ax1.plot(df_null['time_window'], df_null['pkt_count'], color='purple', label='Volumen NULL', linewidth=1.5, alpha=0.5)
        if not df_udp.empty:  ax1.plot(df_udp['time_window'], df_udp['pkt_count'], color='blue', label='Volumen UDP', linewidth=1.5, alpha=0.5)
        if not df_rpl.empty:  ax1.plot(df_rpl['time_window'], df_rpl['pkt_count'], color='green', label='Volumen RPL', linewidth=1.5, alpha=0.5)

        ax1.set_ylabel('Volumen (Paquetes)', fontweight='bold')
        ax1.set_xlabel('Tiempo (ms)', fontweight='bold')
        ax1.grid(True, alpha=0.2)

        ax1_score = ax1.twinx()
        if not df_udp.empty:  ax1_score.scatter(df_udp['time_window'], df_udp['anomaly_score'], color='green', marker='x', s=45, label='Score UDP', zorder=5)
        if not df_null.empty: ax1_score.scatter(df_null['time_window'], df_null['anomaly_score'], color='red', marker='x', s=45, label='Score NULL', zorder=5)
        if not df_rpl.empty:  ax1_score.scatter(df_rpl['time_window'], df_rpl['anomaly_score'], color='orange', marker='x', s=45, label='Score RPL', zorder=5)

        ax1_score.axhline(UMBRAL_NIVEL_1, color='black', linewidth=1.5, label=f'Umbral {UMBRAL_NIVEL_1}')
        ax1_score.set_ylabel('Anomaly Score', color='red', fontweight='bold')
        ax1_score.tick_params(axis='y', labelcolor='red')
        ax1_score.set_ylim(limite_y_score)
        
        ax1.legend(loc='upper left', frameon=True, fontsize=9)
        ax1_score.legend(loc='upper right', frameon=True, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"resultados/volumen_mota_{node_id}_{rol.lower()}.png", dpi=200)
        plt.close(fig1)

if __name__ == "__main__":
    train_eval_model()
