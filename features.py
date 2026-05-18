import pandas as pd
import numpy as np

# Se puede modificar para jugar con las ventanas, en un inicio eran de 30, pero por la justificacion 
# realizada en el capitulo 4, se pasa a 20 segundos
output_csv = "csv/dataset_features.csv"
window_20s = "20s"
window_40s = "40s"

def features():
    df = pd.read_csv("csv/trafico.csv")
    # Al ser el servidor de tipo Echo, es conveniente la eliminacion de este en el escenario UDP Flood,
    # puesto que al replicar los ataques y ser considerado beningno puede confundir al modelo
    df = df[df['ip_src'] != 1]
    
    # Implementamos la caracteristica protocol
    if 'is_udp' in df.columns and 'is_rpl' in df.columns:
        condiciones = [df['is_udp'] == 1, df['is_rpl'] == 1]
        df['protocol'] = np.select(condiciones, [1, 2], default=0)
    # Se rellena con ceros puesto que los NaN pueden fastidiar el calculo y afectar a los resultados
    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'], errors='coerce').fillna(0)
    df['rpl_rank'] = pd.to_numeric(df['rpl_rank'], errors='coerce').fillna(0)
    
    # Al rellenar con ceros, UDP y tramas MAC coincidiran con el código de mensaje DIS. Esto hace que se mezcle 
    # el contexto de los distintos protocolos, por lo que se suma +1 a los códigos de los mensajes RPL
    if 'rpl_code' in df.columns and 'is_rpl' in df.columns:
        df['rpl_code'] = pd.to_numeric(df['rpl_code'], errors='coerce')
        mask = (df['is_rpl'] == 1) & (df['rpl_code'].notna())
        df.loc[mask, 'rpl_code'] += 1
    
    # Para la generacion de tuplas, definimos un identificador de nodo basado en lo descrito en el capitulo 4
    df['node_src'] = np.where((df['ip_src'] != -1) & (df['ip_src'].notna()), df['ip_src'], df['mac_src'])
    df['node_dst'] = np.where((df['ip_dst'] != -1) & (df['ip_dst'].notna()), df['ip_dst'], df['mac_dst'])
    
    # En UDP la tupla considera el destino de los datos, luego se filtra
    df['node_dst'] = np.where(df['protocol'] == 1, df['node_dst'], 'IGNORED_DST')
    
    # Se realiza una reordenacion bajo las tuplas definidas, esto se realiza por intentos de creacion donde daba
    # valores anomalos que no eran posibles
    df = df.sort_values(by=['flow_src', 'protocol', 'flow_dst', 'timestamp_ms']).reset_index(drop=True)
    
    # Implementacion de la característica delta_rank mediante diff()
    df['delta_rank'] = df.groupby(['flow_src', 'protocol'])['rpl_rank'].diff().fillna(0)
    # Implementacion de la caracteristica iat_ms
    df['iat_ms'] = df.groupby(['flow_src', 'protocol', 'flow_dst'])['timestamp_ms'].diff().fillna(0)
    # Implementacion del jitter a partir de iat_ms, se toma el valor absoluto
    df['iat_jitter'] = df.groupby(['flow_src', 'protocol', 'flow_dst'])['iat_ms'].diff().abs().fillna(0)
    
    # Para realizar los calculos por ventanas mediante la funcion rolling() de Pandas, es necesario hacer una conversion
    # de los tiempos. Pasan a ser el indice por el que se guiará la función rolling(). Por problemas de reordenacion vistos,
    # se toma en milisegundos aunque en teoria tambien se puede hacer una conversion a segundos
    df.index = pd.to_timedelta(df['timestamp_ms'], unit='ms')
    
    # Agrupamos por tuplas
    grouped = df.groupby(['flow_src', 'protocol', 'flow_dst'])
    
    # Sacamos la cuenta de paquetes para el calculo posterior del delta. Durante las primeras evaluaciones del modelo, 
    # se vio un sobreajuste en torno a esta característica, posiblemente por las condiciones del escenario de Cooja, por 
    # lo que se decidió extraer del dataset para un mejor aprendizaje
    df['pkt_count'] = grouped['frame_len'].rolling(window_20s).count().fillna(0).values.astype(int)
    
    # Implementacion de la caracteristica bytes_sum_20s, que representa la suma de bytes en la ventana actual
    df['bytes_sum_20s'] = grouped['frame_len'].rolling(window_20s).sum().fillna(0).values
    
    # Implementacion de la media y desviacion estandar del volumen capturado en ventana
    df['window_mean'] = grouped['bytes_sum_20s'].rolling(window_20s).mean().fillna(0).values
    df['window_std'] = grouped['bytes_sum_20s'].rolling(window_20s).std().fillna(0).values
    
    # Implementacion de la media y desviacion estandar del IAT capturado en ventana
    df['iat_window_mean'] = grouped['iat_ms'].rolling(window_20s).mean().fillna(0).values
    df['iat_window_std']  = grouped['iat_ms'].rolling(window_20s).std().fillna(0).values
    df['iat_window_max']  = grouped['iat_ms'].rolling(window_20s).max().fillna(0).values
    df['iat_window_min']  = grouped['iat_ms'].rolling(window_20s).min().fillna(0).values
    
    # Implementacion del valor medio del jitter en ventana
    df['jitter_window_mean'] = grouped['iat_jitter'].rolling(window_20s).mean().fillna(0).values
    
    # Implementacion del delta en comparacion a la ventana anterior
    count_40s = grouped['frame_len'].rolling(window_40s).count().fillna(0).values
    prev_20s = count_40s - df['pkt_count']
    df['pkt_count_delta'] = df['pkt_count'] - prev_20s
    
    # Implementacion de la tendencia de rango en base a los maximos y minimos observados en ventana
    df['rank_trend'] = (
        grouped['rpl_rank'].rolling(window_20s).max().values - 
        grouped['rpl_rank'].rolling(window_20s).min().values
    )
    # Implementacion de la intensidad de rafaga de volumen
    df['burst_intensity'] = df['bytes_sum_20s'] / 20.0

    # Implementacion del Z Score para cada paquete en base a las caracteristicas de ventana sacadas
    df['iat_z_score'] = (df['iat_ms'] - df['iat_window_mean']) / (df['iat_window_std'] + 1e-6)
    
    #Implementacion del coeficiente de variacion de los tiempos entre llegadas en ventana
    df['iat_cv'] = df['iat_window_std'] / (df['iat_window_mean'] + 1e-6)

    # Implementacion de la característica de rango de IAT, mediante los valores maximo y minimo
    # observados en ventana
    df['iat_range'] = df['iat_window_max'] - df['iat_window_min']
    
    # Destruimos el indice creado para ventanas y reorganizamos en base a los valores de tiempo originales
    df = df.reset_index(drop=True)
    df = df.sort_values(by='timestamp_ms').reset_index(drop=True)
    
    # Finalmente, nos libramos de las columnas que no nos hacen falta para el dataset
    if 'bytes_sum_20s' in df.columns:
        df = df.drop(columns=['bytes_sum_20s'])

    drop_columns = [
        'mac_src', 'mac_dst', 'ip_src', 'ip_dst', 
        'flow_src', 'flow_dst', 
        'mac_pan_id', 'rpl_instance_id',
        'udp_src_port', 'udp_dst_port',
        'rpl_dodagid', 'mac_seq_number', 'rpl_dao_sequence', 'rpl_dtsn',
        'payload_hex', 'frame_number', 'timestamp_ms', 'is_udp', 'is_ack',
        'is_rpl', 'mac_security', 'mac_fcf', 'mac_ack_req', 'rpl_rank',
        'mac_pan_id_comp', 'mac_dst_mode', 'mac_src_mode', 'mac_version',
        'payload_size', 'udp_len', 'frame_len', 'rpl_version', 'rpl_mop', 'rpl_flags',
        'rpl_type', 'mac_frame_type', 'mac_src_str', 'mac_dst_str', 'ip_src_str',
        'ip_dst_str', 'rpl_dodagid_str',
        'iat_window_mean', 'iat_window_std', 'iat_window_max', 'iat_window_min',
        'pkt_count'
    ]

    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    # Por seguridad, hacemos un último relleno de ceros para evitar NaNs
    df = df.fillna(0)
    
    print(f"Guardado {output_csv}")
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    features()
