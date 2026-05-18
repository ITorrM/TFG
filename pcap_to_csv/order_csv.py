import pandas as pd
import argparse
import os

def order(input_files, output_file):
    dataframes = []
    for current_file in input_files:
        if os.path.exists(current_file):
            print(f"Cargando {current_file}")
            df = pd.read_csv(current_file)
            dataframes.append(df)
        else:
            print(f"No se encontro {current_file}")
            
    if not dataframes:
        print("Error al cargar")
        return

    # La salida sigue en formato DataFrames de Pandas
    final_df = pd.concat(dataframes, ignore_index=True)
    
    # Realizamos una conversion numerica del tiempo para ordenacion, se ignoran 
    # los indices producidos por estos
    if 'timestamp_ms' in final_df.columns:
        final_df['timestamp_ms'] = pd.to_numeric(final_df['timestamp_ms'])
        final_df = final_df.sort_values(by='timestamp_ms', ignore_index=True)
    # Al estar odenaros por tiempo, se borra la columna de numero de trama
    if 'frame_number' in final_df.columns:
        final_df = final_df.drop(columns=['frame_number'])
    # Se genera una nueva columna de numero de tramas, no es necesario, pero de cara a analizar la
    # salida es recomentadble
    final_df.insert(0, 'frame_number', range(len(final_df)))
    
    print(f"Guardando en {output_file}...")
    final_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("Lista de archivos CSV a combinar (separados por espacio).")
    parser.add_argument("-o", "--output", default="csv/dataset.csv", 
                        help="Ruta del archivo CSV de salida.")
    
    args = parser.parse_args()
    order(args.files, args.output)