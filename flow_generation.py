import pandas as pd
import numpy as np

def flow_generation(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    df = df[df['is_ack'] == 0].copy()
    
    df['time_window'] = (df['timestamp_ms'] // 20000) * 20

    condiciones = [df['is_udp'] == 1, df['is_rpl'] == 1]
    df['protocol'] = np.select(condiciones, [1, 2], default=0)

    df = df.sort_values(by=['time_window', 'mac_src', 'protocol', 'timestamp_ms'])
    
    df['iat_ms'] = df.groupby(['time_window', 'mac_src', 'protocol'])['timestamp_ms'].diff()

    operaciones = {
        'frame_len': ['count', 'sum', 'mean', 'std'],  
        'iat_ms': ['mean', 'std'], 
        'label': ['max']                               
    }

    df_flujos = df.groupby(['time_window', 'mac_src', 'protocol']).agg(operaciones).reset_index()

    df_flujos.columns = [
        'time_window', 'mac_src', 'protocol',
        'pkt_count', 'flow_bytes', 'pkt_len_mean', 'pkt_len_std',
        'iat_mean', 'iat_std', 'label'
    ]

    df_flujos.fillna(0, inplace=True)

    df_flujos['pkt_len_cv'] = np.where(
        df_flujos['pkt_len_mean'] > 0, 
        df_flujos['pkt_len_std'] / df_flujos['pkt_len_mean'], 
        0 
    )
    
    df_flujos['byte_rate'] = df_flujos['flow_bytes'] / 20.0

    cols_finales = [
        'time_window','mac_src', 'protocol', 
        'pkt_count', 'flow_bytes', 'byte_rate', 
        'pkt_len_mean', 'pkt_len_std', 'pkt_len_cv', 
        'iat_mean', 'iat_std', 'label'
    ]
    
    df_flujos = df_flujos[cols_finales]

    cols_enteras = ['time_window', 'protocol', 'pkt_count', 'flow_bytes', 'label']
    for col in cols_enteras:
        df_flujos[col] = df_flujos[col].astype(int)
        
    print(f"Dataset basado en flujos generado en: {output_csv}")
    df_flujos.to_csv(output_csv, index=False)


if __name__ == "__main__":
    flow_generation('csv/trafico.csv', 'csv/dataset_flujos.csv')
