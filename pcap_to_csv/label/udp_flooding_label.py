import pandas as pd

INPUT_CSV = "csv/udp_flooding_mapped.csv" 
OUTPUT_CSV = "csv/udp_flooding_labeled.csv" 
ATTACKER_IDS = [11, 12, 13] 
LABEL_ATAQUE = 1

def udp_flooding_label():
    print(f"Cargando {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"No se encuentra el archivo {INPUT_CSV}")
        return
    
    # Se verifica que no haya NaNs
    df['ip_src'] = pd.to_numeric(df['ip_src'])
    df['is_udp'] = df['is_udp'].fillna(0).astype(int)
    
    # Se etiqueta todo el tráfico como benigno
    df['label'] = 0
    
    # Se prepara la máscara para realizar el etiquetado, será todo el tráfico UDP
    # asociado a la IP atacante
    mask = (
        (df['ip_src'].isin(ATTACKER_IDS)) & 
        (df['is_udp'] == 1)
    )
    
    # Asignamos la etiqueta 1 para el ataque
    df.loc[mask, 'label'] = LABEL_ATAQUE
    
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    udp_flooding_label()