import pandas as pd
import numpy as np

output_csv = "csv/dataset_features.csv"
window_20s = "20s"
window_40s = "40s"

def generate_edge_features():
    df = pd.read_csv("csv/trafico.csv")
    df = df[df['ip_src'] != 1]

    if 'protocol' not in df.columns:
        if 'is_udp' in df.columns and 'is_rpl' in df.columns:
            condiciones = [df['is_udp'] == 1, df['is_rpl'] == 1]
            df['protocol'] = np.select(condiciones, [1, 2], default=0)
        else:
            print("No se encuentra la columna 'protocol'")

    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'], errors='coerce').fillna(0)
    df['rpl_rank'] = pd.to_numeric(df['rpl_rank'], errors='coerce').fillna(0)
    
    if 'rpl_code' in df.columns and 'is_rpl' in df.columns:
        df['rpl_code'] = pd.to_numeric(df['rpl_code'], errors='coerce')
        mask = (df['is_rpl'] == 1) & (df['rpl_code'].notna())
        df.loc[mask, 'rpl_code'] += 1
    
    df['flow_src'] = np.where((df['ip_src'] != -1) & (df['ip_src'].notna()), df['ip_src'], df['mac_src'])
    df['flow_dst'] = np.where((df['ip_dst'] != -1) & (df['ip_dst'].notna()), df['ip_dst'], df['mac_dst'])
    
    df['flow_dst'] = np.where(df['protocol'] == 1, df['flow_dst'], 'IGNORED_DST')

    df = df.sort_values(by=['flow_src', 'protocol', 'flow_dst', 'timestamp_ms']).reset_index(drop=True)
    
    df['delta_rank'] = df.groupby(['flow_src', 'protocol'])['rpl_rank'].diff().fillna(0)
    df['iat_ms'] = df.groupby(['flow_src', 'protocol', 'flow_dst'])['timestamp_ms'].diff().fillna(0)
    df['iat_jitter'] = df.groupby(['flow_src', 'protocol', 'flow_dst'])['iat_ms'].diff().abs().fillna(0)

    df.index = pd.to_timedelta(df['timestamp_ms'], unit='ms')
    
    grouped = df.groupby(['flow_src', 'protocol', 'flow_dst'])

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
        grouped['rpl_rank'].rolling(window_20s).max().values - 
        grouped['rpl_rank'].rolling(window_20s).min().values
    )

    df['burst_intensity'] = df['bytes_sum_20s'] / 20.0
    df['iat_z_score'] = (df['iat_ms'] - df['iat_window_mean']) / (df['iat_window_std'] + 1e-6)
    
    df['iat_cv'] = df['iat_window_std'] / (df['iat_window_mean'] + 1e-6)
    df['iat_range'] = df['iat_window_max'] - df['iat_window_min']

    df = df.reset_index(drop=True)
    df = df.sort_values(by='timestamp_ms').reset_index(drop=True)
    
    if 'bytes_sum_20s' in df.columns:
        df = df.drop(columns=['bytes_sum_20s'])

    if 'is_ack' in df.columns:
        df = df[df['is_ack'] == 0].copy()

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
    df = df.fillna(0)

    if 'label' in df.columns:
        cols = [c for c in df.columns if c != 'label'] + ['label']
        df = df[cols]

    print(f"--> Guardando {output_csv} con {len(df.columns)} columnas")
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    generate_edge_features()
