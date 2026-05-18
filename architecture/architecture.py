import pandas as pd
import numpy as np
import joblib

def extract_if_features(chunk_df):
    # Eliminamos los ACK
    df = chunk_df[chunk_df['is_ack'] == 0].copy()
    if df.empty: return pd.DataFrame()
    
    df['mac_src'] = df['mac_src'].fillna(0).astype(str)
    
    if 'protocol' not in df.columns:
        conditions = [df['is_udp'] == 1, df['is_rpl'] == 1]
        df['protocol'] = np.select(conditions, [1, 2], default=0)

    df = df.sort_values(by=['mac_src', 'protocol', 'timestamp_ms'])
    df['iat_ms'] = df.groupby(['mac_src', 'protocol'])['timestamp_ms'].diff()

    aggregations = {
        'timestamp_ms': ['count'],
        'frame_len': ['sum', 'mean'],
        'iat_ms': ['mean', 'std']
    }

    flow_df = df.groupby(['mac_src', 'protocol']).agg(aggregations).reset_index()
    flow_df.columns = ['src_mac', 'protocol', 'pkt_count', 'flow_bytes', 'byte_mean', 'iat_mean', 'iat_std']
    flow_df.fillna(0.0, inplace=True)

    flow_df['pkt_rate'] = flow_df['pkt_count'] / 20.0
    flow_df['byte_rate'] = flow_df['flow_bytes'] / 20.0

    return flow_df

def extract_xgb_features(chunk_df):
    window_20s = "20s"
    window_40s = "40s"
    
    df = chunk_df.copy()
    df = df[(df['ip_src'] != 1) & (df['ip_src'] != '1')]
    if df.empty: return pd.DataFrame()

    if 'protocol' not in df.columns:
        conditions = [df['is_udp'] == 1, df['is_rpl'] == 1]
        df['protocol'] = np.select(conditions, [1, 2], default=0)

    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms']).fillna(0)
    df['rpl_rank'] = pd.to_numeric(df['rpl_rank']).fillna(0)
    
    if 'rpl_code' in df.columns and 'is_rpl' in df.columns:
        df['rpl_code'] = pd.to_numeric(df['rpl_code'])
        mask = (df['is_rpl'] == 1) & (df['rpl_code'].notna())
        df.loc[mask, 'rpl_code'] += 1
    else:
        df['rpl_code'] = 0.0

    df['node_src'] = np.where((df['ip_src'] != -1) & (df['ip_src'] != '-1') & (df['ip_src'].notna()), df['ip_src'], df['mac_src'])
    df['node_dst'] = np.where((df['ip_dst'] != -1) & (df['ip_dst'] != '-1') & (df['ip_dst'].notna()), df['ip_dst'], df['mac_dst'])
    
    df['node_dst'] = np.where(df['protocol'] == 1, df['node_dst'], 'IGNORED_DST')

    df['node_src'] = df['node_src'].astype(str)
    df['node_dst'] = df['node_dst'].astype(str)

    df = df.sort_values(by=['node_src', 'protocol', 'node_dst', 'timestamp_ms']).reset_index(drop=True)
    
    df['delta_rank'] = df.groupby(['node_src', 'protocol'])['rpl_rank'].diff().fillna(0)
    df['iat_ms'] = df.groupby(['node_src', 'protocol', 'node_dst'])['timestamp_ms'].diff().fillna(0)
    df['iat_jitter'] = df.groupby(['node_src', 'protocol', 'node_dst'])['iat_ms'].diff().abs().fillna(0)

    df.index = pd.to_timedelta(df['timestamp_ms'], unit='ms')
    
    grouped = df.groupby(['node_src', 'protocol', 'node_dst'])

    df['pkt_count'] = grouped['frame_len'].rolling(window_20s).count().fillna(0).values.astype(int)
    
    df['bytes_sum_20s'] = grouped['frame_len'].rolling(window_20s).sum().fillna(0).values
    
    df['window_mean'] = grouped['bytes_sum_20s'].rolling(window_20s).mean().fillna(0).values
    df['window_std'] = grouped['bytes_sum_20s'].rolling(window_20s).std().fillna(0).values
    
    df['iat_window_mean'] = grouped['iat_ms'].rolling(window_20s).mean().fillna(0).values
    df['iat_window_std']  = grouped['iat_ms'].rolling(window_20s).std().fillna(0).values
    df['iat_window_max']  = grouped['iat_ms'].rolling(window_20s).max().fillna(0).values
    df['iat_window_min']  = grouped['iat_ms'].rolling(window_20s).min().fillna(0).values
    
    df['jitter_window_mean'] = grouped['iat_jitter'].rolling(window_20s).mean().fillna(0).values

    count_40s = grouped['frame_len'].rolling(window_40s).count().fillna(0).values
    prev_20s = count_40s - df['pkt_count']
    df['pkt_count_delta'] = df['pkt_count'] - prev_20s

    df['rank_trend'] = (
        grouped['rpl_rank'].rolling(window_20s).max().values - grouped['rpl_rank'].rolling(window_20s).min().values
    )

    df['burst_intensity'] = df['bytes_sum_20s'] / 20.0
    df['iat_z_score'] = (df['iat_ms'] - df['iat_window_mean']) / (df['iat_window_std'] + 1e-6)
    df['iat_cv'] = df['iat_window_std'] / (df['iat_window_mean'] + 1e-6)
    df['iat_range'] = df['iat_window_max'] - df['iat_window_min']

    df = df.reset_index(drop=True)
    df = df.sort_values(by='timestamp_ms').reset_index(drop=True)
    
    latest_states = df.groupby(['node_src', 'protocol', 'node_dst']).tail(1).copy()
    latest_states = latest_states.fillna(0.0)
    
    # En caso de no extraer bien el mac_pending, se inserta para garantizar la existencia de la columna
    if 'mac_pending' not in latest_states.columns:
        latest_states['mac_pending'] = 0.0
        
    return latest_states

def architecture_simulation():
    df = pd.read_csv("../csv/validacion.csv")
    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms']).fillna(0)
        
    df = df.sort_values("timestamp_ms")
    max_ts = df["timestamp_ms"].max()
    
    # Se cargan los modelo y se declaran las clases
    if_model = joblib.load("modelos/isolation_forest.joblib")
    xgb_model = joblib.load("modelos/xgboost.joblib")
    class_names = {0: 'Normal', 1: 'UDP Flood', 2: 'Frame Pend', 3: 'DIS Flood', 4: 'Inc. Rank'}
    
    # Direcciones a no analizar
    blacklist = ['0', '-1', 'nan', 'fe80::', '::', 'ff02::1a', 'UNKNOWN', 'None']

    total_paquetes_dataset = len(df)
    total_inspecciones_profundas = 0

    for current_time in range(20000, int(max_ts) + 20000, 20000):
        
        # Se crean abos bloques (chunks) de datos para cada modelo
        chunk_20s = df[(df['timestamp_ms'] >= current_time - 20000) & (df['timestamp_ms'] < current_time)]
        chunk_40s = df[(df['timestamp_ms'] >= current_time - 40000) & (df['timestamp_ms'] < current_time)]

        if chunk_20s.empty: continue
            
        print("="*140)
        print(f"Ventana: ({current_time/1000}s) | Bloque procesándose en XGBoost")
        
        # LLamada de extraccion de caracteristicas para el iForest, igual que en el script flow_generation.py
        if_feats = extract_if_features(chunk_20s)
        if if_feats.empty: continue
            
        X_if = if_feats[['protocol', 'pkt_count', 'pkt_rate', 'byte_mean', 'byte_rate', 'iat_mean', 'iat_std']]
        scores = if_model.decision_function(X_if)
        
        if_feats['score'] = scores
        # Se puede filtrar por la puntuacion anomala, en este caso se respeta el umbral de cero usado en el capitulo 6
        if_feats['is_anomaly'] = scores < 0.0
        
        # Operacion set() para sacar los flujos anomalos de capa 2
        alertas = set()
        
        for _, row in if_feats.iterrows():
            mac = str(row['src_mac']).strip()
            proto_val = int(row['protocol'])
            proto_str = {0: 'NULL', 1: 'UDP', 2: 'RPL'}.get(proto_val, 'UNK')
            score = row['score']
            
            # En caso de anomalia imprime como alerta, si no da el OK
            if row['is_anomaly']:
                print(f"  [ALERTA] MAC: {mac:>25} | Proto: {proto_str:<4} | Score: {score:.4f}")
                alertas.add((mac, proto_val))
            else:
                print(f"  [OK]     MAC: {mac:>25} | Proto: {proto_str:<4} | Score: {score:.4f}")
                
        if alertas:
            # Se pasa a la extraccion de caracteristicas del chunk de 40 segundos, al ser esto una prueba de concepto ya viene calculado
            xgb_feats = extract_xgb_features(chunk_40s)
            
            if not xgb_feats.empty:
                for _, row in xgb_feats.iterrows():
                    mac_original = str(row['mac_src']).strip()
                    proto_original = int(row['protocol'])
                    
                    if (mac_original, proto_original) not in alertas:
                        continue
                        
                    node_src = str(row['node_src']).strip()
                    if node_src in blacklist:
                        continue
                    
                    # Solo evalua aquellos que den la alerta, se crea el vector de entrada limpio
                    vector_xgb = [
                        row['mac_pending'], row['rpl_code'], row['protocol'], row['delta_rank'],
                        row['iat_ms'], row['iat_jitter'], row['window_mean'], row['window_std'],
                        row['jitter_window_mean'], float(row['pkt_count_delta']), row['rank_trend'],
                        row['burst_intensity'], row['iat_z_score'], row['iat_cv'], row['iat_range']
                    ]
                    
                    # Al poder haber NaNs al trabajar con DataFrames, pasamos el vector a float y rellenamos con ceros.
                    clean_vector = [0.0 if pd.isna(v) else float(v) for v in vector_xgb]
                    
                    # Evaluamos la instancia, donde tomamos probs siendo el primer elemento del vector de salida
                    probs = xgb_model.predict_proba([clean_vector])[0]
                    clase_predicha = np.argmax(probs)
                    confianza = probs[clase_predicha] * 100.0
                    
                    # Para hacer conteo de pasos por capa 2
                    total_inspecciones_profundas += 1
                                                                 
                    if clase_predicha > 0:
                        tipo = class_names.get(clase_predicha)
                        print(f"      Inspeccion de cabeceras -> Origen Real: {node_src:<39} | ATAQUE ({tipo}) [Confianza: {confianza:.1f}%]")
                    else:
                        print(f"      Inspeccion de cabeceras -> Origen Real: {node_src:<39} | Falsa Alarma [Confianza: {confianza:.1f}%]")

    print('\n')
    print(f"Total de paquetes procesados: {total_paquetes_dataset:,}")
    print(f"Total de flujos sospechosos evaluados en Capa 2: {total_inspecciones_profundas:,}")
    
    if total_paquetes_dataset > 0:
        porcentaje_inspeccion = (total_inspecciones_profundas / total_paquetes_dataset) * 100
        ahorro = 100.0 - porcentaje_inspeccion
        print(f"\nCarga de trabajo en Capa 2: {porcentaje_inspeccion:.2f}% del volumen total.")
        print(f"Eficiencia del filtro: El Isolation Forest evito la inspeccion de cabeceras en el {ahorro:.2f}% de las interacciones.")
    print("="*140 + "\n")

if __name__ == "__main__":
    architecture_simulation()
