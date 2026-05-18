import pandas as pd
import numpy as np

def flow_generation(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Se eliminan los ACK mediante un copy
    df = df[df['is_ack'] == 0].copy()
    
    # Manejamos las ventanas por segundos, luego hacemos la conversión
    df['time_window'] = (df['timestamp_ms'] // 20000) * 20
    
    # Filtramos las condiciones para la asignacion de la caracteristica protocol
    condiciones = [df['is_udp'] == 1, df['is_rpl'] == 1]
    df['protocol'] = np.select(condiciones, [1, 2], default=0)
    
    # Con las columnas creadas, ordenamos el tráfico en las ventanas
    df = df.sort_values(by=['time_window', 'mac_src', 'protocol', 'timestamp_ms'])
    
    # La característica iat_mean requiere conocer los tiempos de cada paquete, luego sacamos
    # la diferencia mediante diff()
    df['iat_ms'] = df.groupby(['time_window', 'mac_src', 'protocol'])['timestamp_ms'].diff()
    
    # Para facilitar el cálculo, hacemos una agregación de las operaciones existentes
    operaciones = {
        'frame_len': ['count', 'sum', 'mean', 'std'],  
        'iat_ms': ['mean', 'std'], 
        'label': ['max']                               
    }
    # Con agg(), podemos realizar las operaciones sin requerir de mas lineas, puesto que la tupla
    # mac_src, mac_dst, protocol no cambia por otras como en la fase de estudio de modelos
    df_flujos = df.groupby(['time_window', 'mac_src', 'protocol']).agg(operaciones).reset_index()

    df_flujos.columns = [
        'time_window', 'mac_src', 'protocol',
        'pkt_count', 'flow_bytes', 'pkt_len_mean', 'pkt_len_std',
        'iat_mean', 'iat_std', 'label'
    ]

    # Rellenamos el df original con inplace=True
    df_flujos.fillna(0, inplace=True)
    
    # A través de las operaciones realizadas previamente, podemos calcular el coeficiente
    # de variacion
    df_flujos['pkt_len_cv'] = np.where(
        df_flujos['pkt_len_mean'] > 0, 
        df_flujos['pkt_len_std'] / df_flujos['pkt_len_mean'], 
        0 
    )
    
    # Lo mismo con la tasa de bytes
    df_flujos['byte_rate'] = df_flujos['flow_bytes'] / 20.0

    cols_finales = [
        'time_window','mac_src', 'protocol', 
        'pkt_count', 'flow_bytes', 'byte_rate', 
        'pkt_len_mean', 'pkt_len_std', 'pkt_len_cv', 
        'iat_mean', 'iat_std', 'label'
    ]
    
    # Registramos las columnas finales definidas
    df_flujos = df_flujos[cols_finales]

    cols_enteras = ['time_window', 'protocol', 'pkt_count', 'flow_bytes', 'label']
    for col in cols_enteras:
        df_flujos[col] = df_flujos[col].astype(int)
        
    print(f"Dataset basado en flujos generado en: {output_csv}")
    df_flujos.to_csv(output_csv, index=False)


if __name__ == "__main__":
    flow_generation('csv/trafico.csv', 'csv/dataset_flujos.csv')
