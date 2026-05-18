import pandas as pd

INPUT_CSV = "csv/dis_flooding_attack_mapped.csv" 
OUTPUT_CSV = "csv/dis_flooding_labeled.csv"
ATTACKER_IDS = [37, 38, 39]
LABEL_ATAQUE = 3   

def dis_flooding_label():
    print(f"--> Cargando {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"No se encuentra el archivo {INPUT_CSV}")
        return
    
    # Se comienza filtrando las filas que sean mensajes RPL para los atacantes
    df['ip_src'] = pd.to_numeric(df['ip_src'])
    df['is_rpl'] = df['is_rpl'].fillna(0).astype(int)
    
    # Se identifican los mensajes DIS
    df['is_dis_packet'] = (
        (df['is_rpl'] == 1) & 
        (df['rpl_type'] == 155) & 
        (df['rpl_code'] == 0)
    ).astype(int)

    # Se etiqueta todo como trafico benigno en un inicio
    df['label'] = 0
    
    # La mascara que se construye mediante la agregacion de ambos filtros
    mask = (
        (df['ip_src'].isin(ATTACKER_IDS)) & 
        (df['is_dis_packet'] == 1)
    )
    
    # Se asigna la etiqueta 3 por orden de implementacion
    df.loc[mask, 'label'] = LABEL_ATAQUE
    
    # Se borra la columna auxiliar
    df_final = df.drop(columns=['is_dis_packet'])
    
    df_final.to_csv(OUTPUT_CSV, index=False)
        
if __name__ == "__main__":
    dis_flooding_label()