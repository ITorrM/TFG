import pandas as pd
import os

input_file = "csv/increase_rank_mapped.csv"
output_file = "csv/increase_rank_labeled.csv"

# Se toman las motas afectadas en el escenario como atacantes tambien
affected_motes = [41, 42, 44, 46, 47, 49, 50, 51, 52]
attack_label = 4

def increase_rank_attack_label():
    print(f"--> Cargando {input_file}")
    
    if not os.path.exists(input_file):
        print(f"No se encuentra el archivo {input_file}")
        return

    # Usamos low_memory=False po warnings saltantes al manipular las columnas
    df = pd.read_csv(input_file, low_memory=False)
    
    df['ip_src'] = pd.to_numeric(df['ip_src'])
    
    # Se requiere unicamente los mensajes DIO, se rellena los NaN para evitar fallos
    required_columns = ['is_rpl', 'rpl_code']
    for col in required_columns:
        df[col] = pd.to_numeric(df[col]).fillna(0)

    df['label'] = 0
    
    # Se etiquetan solo los mensajes DIO asociados a las IP atacantes
    attack_condition = (
        (df['ip_src'].isin(affected_motes)) & 
        (df['is_rpl'] == 1) & 
        (df['rpl_code'] == 1)
    )
    
    # Se asigna la etiqueta 4 al ser el último ataque implementado
    df.loc[attack_condition, 'label'] = attack_label
    
    # Por problemas de descolocacion, se vuelve a ordenar por el número de trama
    if 'frame_number' in df.columns:
        df = df.sort_values(by='frame_number')
        
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    increase_rank_attack_label()