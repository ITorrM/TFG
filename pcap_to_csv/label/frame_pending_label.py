import pandas as pd

INPUT_CSV = "csv/frame_pending_mapped.csv"
OUTPUT_CSV = "csv/frame_pending_labeled.csv"
ATTACKER_IDS = [24, 25, 26]
ATTACK_LABEL = 2

def label_frame_pending_attack():
    print(f"--> Cargando {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"No se encuentra el archivo {INPUT_CSV}")
        return
    
    # Se filtran las MAC atacantes
    df['mac_src'] = pd.to_numeric(df['mac_src'], errors='coerce')
    
    # Se filtra el trafico benigno, la comprobacion se realiza por los NaNs
    for col in ['mac_pending', 'is_rpl', 'is_udp']:
        df[col] = df[col].fillna(0).astype(int)

    # Se etiqueta todo como benigno
    df['label'] = 0
    
    # Enmascaramiento para las tramas MAC atacantes
    mask = (
        (df['mac_src'].isin(ATTACKER_IDS)) & 
        (df['mac_pending'] == 1) &
        (df['is_rpl'] == 0) &
        (df['is_udp'] == 0)
    )
    
    # Se etiqueta con 2 por orden de implementacion
    df.loc[mask, 'label'] = ATTACK_LABEL
    
    df_final = df.sort_values(by='frame_number')
    df_final.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    label_frame_pending_attack()