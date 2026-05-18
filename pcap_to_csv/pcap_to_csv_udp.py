import pandas as pd
import numpy as np
from scapy.all import *
import struct
import argparse
import json
import sys

# --- CONFIGURACIÓN ---
conf.dot15d4_protocol = "sixlowpan"

# Mapa de sustitución de direcciones
MAC_TO_IP = {
    "00:12:74:01:00:01:01:01": "fe80::212:7401:1:101",
    "00:12:74:02:00:02:02:02": "fe80::212:7402:2:202",
    "00:12:74:03:00:03:03:03": "fe80::212:7403:3:303",
    "00:12:74:04:00:04:04:04": "fe80::212:7404:4:404",
    "00:12:74:05:00:05:05:05": "fe80::212:7405:5:505",
    "00:12:74:06:00:06:06:06": "fe80::212:7406:6:606",
    "00:12:74:07:00:07:07:07": "fe80::212:7407:7:707",
    "00:12:74:08:00:08:08:08": "fe80::212:7408:8:808",
    "00:12:74:09:00:09:09:09": "fe80::212:7409:9:909",
    "00:12:74:0a:00:0a:0a:0a": "fe80::212:740a:a:a0a",
    "00:12:74:0b:00:0b:0b:0b": "fe80::212:740b:b:b0b",
    "00:12:74:0c:00:0c:0c:0c": "fe80::212:740c:c:c0c",
    "00:12:74:0d:00:0d:0d:0d": "fe80::212:740d:d:d0d",
    "00:00:00:00:00:00:ff:ff": "ff02::1a", 
    "ff:ff": "ff02::1a",                   
    "ffff": "ff02::1a"                     
}

# Función para representar las direcciones extraidas como enteros en formato
# de direcciones EUI-64. Se representa en formato hexadecimal, y se concatenan en 
# bloques de 1 byte por el separador ":". Se toma como referencia el siguiente
# ejemplo: https://stackoverflow.com/questions/11006702/elegant-format-for-a-mac-address-in-python-3-2
def format_mac(mac_int):
    if mac_int is None: return None
    try:
        h = "{:016x}".format(mac_int)
        return ":".join(h[i:i+2] for i in range(0, len(h), 2))
    except: return str(mac_int)

# Función para representar las direcciones extraidas como enteros en formato
# de direcciones EUI-64. Desempaqueta la entrada en una cadena de bytes con formato
# b'\xfe\x80\x00\ ... \x0d\xb0\x00 por ejemplo, y se concatenan en bucle por el separador ":".
# De referencia, se toma: https://docs.python.org/3/library/struct.html
def format_ipv6_from_bytes(b):
    try:
        parts = struct.unpack(">8H", b)
        return ":".join(f"{p:x}" for p in parts)
    except: return None

# Función para extraer el Frame Control Field (FCF) de capa MAC en crudo. Se extraen los 
# dos primeros bytes, y se devuelve como un entero de 16 bits. Se requiere formato Little
# Endian, puesto que su representación en Scapy y Wireshark es así.
def get_raw_fcf(dot15):
    if hasattr(dot15, "fcf") and dot15.fcf is not None:
        return int(dot15.fcf)
    raw = bytes(dot15)[:2]
    if len(raw) == 2:
        return raw[0] + (raw[1] << 8)
    return None

# Función auxiliar para extraer los distintos campos del FCF en diversos campos por medio 
# de desplazamiento de bits y enmascaramiento. En caso de no encontrar, se deja la entrada
# correspondiente como un NaN de Pandas
def extract_mac_flags(fcf_val):
    if fcf_val is None:
        return {
            "mac_frame_type": np.nan, "mac_security": np.nan, "mac_pending": np.nan,
            "mac_ack_req": np.nan, "mac_pan_id_comp": np.nan,
            "mac_dst_mode": np.nan, "mac_version": np.nan, "mac_src_mode": np.nan
        }
    return {
        "mac_frame_type":      fcf_val & 0b111,
        "mac_security":       (fcf_val >> 3) & 1,
        "mac_pending":        (fcf_val >> 4) & 1,
        "mac_ack_req":        (fcf_val >> 5) & 1,
        "mac_pan_id_comp":    (fcf_val >> 6) & 1,
        "mac_dst_mode":       (fcf_val >> 10) & 0b11,
        "mac_version":        (fcf_val >> 12) & 0b11,
        "mac_src_mode":       (fcf_val >> 14) & 0b11
    }

# Función para extraer los bytes asociados a los distintos mensajes RPL encapusalados
# en ICMPv6 y crear diversos campos según su significado. Los campos de tipo de mensaje 
# ICMP y tipo de mensaje RPL son comunes en todos los mensajes, siendo el resto de campos
# variables en función del tipo de mensaje. No se extraen todos en su totalidad, sino los
# más relevantes tras una primera evaluación de los PCAPs en Wireshark
def parse_rpl_raw(pkt):
    if not pkt.haslayer(ICMPv6RPL): return {}
    rpl_layer = pkt[ICMPv6RPL]
    raw_bytes = bytes(rpl_layer) + bytes(rpl_layer.payload)
    if len(raw_bytes) < 4: return {}

    rpl_data = {
        "icmp_type": raw_bytes[0],
        "rpl_code": raw_bytes[1]
    }
    code = raw_bytes[1]
    
    # Extracción de mensajes DIO
    if code == 1 and len(raw_bytes) >= 8:
        rpl_data["rpl_instance_id"] = raw_bytes[4]
        rpl_data["rpl_version"]     = raw_bytes[5]
        rpl_data["rpl_rank"]        = (raw_bytes[6] << 8) + raw_bytes[7]
        if len(raw_bytes) >= 11:
            rpl_data["rpl_mop"]   = raw_bytes[8]
            rpl_data["rpl_dtsn"]  = raw_bytes[9]
            rpl_data["rpl_flags"] = raw_bytes[10]
        if len(raw_bytes) >= 28:
            rpl_data["rpl_dodagid"] = format_ipv6_from_bytes(raw_bytes[12:28])
            
    # Extracción de mensajes DIS
    elif code == 0 and len(raw_bytes) >= 5:
        rpl_data["rpl_flags"] = raw_bytes[4]
            
    # Extracción de mensajes DAO
    elif code == 2 and len(raw_bytes) >= 6:
        rpl_data["rpl_instance_id"] = raw_bytes[4]
        rpl_data["rpl_flags"] = raw_bytes[5]
        if len(raw_bytes) >= 8: rpl_data["rpl_dao_sequence"] = raw_bytes[7]
        if len(raw_bytes) >= 24: rpl_data["rpl_dodagid"] = format_ipv6_from_bytes(raw_bytes[8:24])
        
    return rpl_data

# Función principal del programa, la cual se encarga de abrir el paquete y extraer los 
# campos más relevantes en columnas. Debido a la extensión de la misma, se va comentando
# a lo largo de su implementación
def process_pcap(pcap_file, output_csv, address_map_file=None):
    # Apertura del paquete mediante el objeto PcapReader. En caso de no encontrar el archivo,
    # se lanza una excepción y se sale del programa.
    print(f"Leyendo PCAP: {pcap_file}")
    try: 
        packets = PcapReader(pcap_file)
    except FileNotFoundError: 
        print(f"Archivo {pcap_file} no encontrado.")
        sys.exit(1)

    # En el caso que se use un mapa de direcciones en formato JSON, se realiza una apertura del
    # mismo en modo lectura. En caso de no encontrar el declarado o de tener problemas de formato,
    # se captura el error y se lanza una excepción con el código de error.
    address_map = None
    if address_map_file:
        try:
            with open(address_map_file, 'r') as f:
                address_map = json.load(f)
            print(f"Mapa de direcciones cargado: {address_map_file}")
        except Exception as e:
            print(f"Error de lectura: {e}")
            sys.exit(1)

    # Declaramos las filas y hacemos un contador para ver como va extrayendo, cada 1000 paquetes
    # imprime
    rows = []
    start_time = None
    
    for pkt in packets:
        count += 1
        if count % 1000 == 0: print(f"   Procesando {count}...")

        # Se sacan unas columnas para facilitar el trabajo de etiquetado, y que contextualizan si
        # el paquete es ACK, UDP o RPL
        is_udp_pkt = pkt.haslayer(UDP)
        is_rpl_pkt = pkt.haslayer(ICMPv6RPL)
        
        is_ack_pkt = False
        fcf_int = None
        
        mac_src_str = None
        mac_dst_str = None

        is_dot15 = pkt.haslayer(Dot15d4)

        if is_dot15:
            dot15 = pkt[Dot15d4]
            # Llamada a la función de extracción pura de FCF
            fcf_int = get_raw_fcf(dot15)
            # Se saca las direcciones por medio de la función de formateo, la comprobacion
            # es necesaria, puesto que los ACK no contienen las direcciones según lo visualizado
            # en Wireshark
            if hasattr(dot15, "src_addr"): mac_src_str = format_mac(dot15.src_addr)
            if hasattr(dot15, "dest_addr"): mac_dst_str = format_mac(dot15.dest_addr)

        if not is_dot15:
            continue
        # La manera más exacta es trabajar con los Timestamp, sin embargo, la precisión de estos 
        # puede ser dificil de manejar dependiendo de la escala temporal. Algo mas sencillo es trabajar
        # directamente con la columnta time y pasarla a milisegundos 
        current_time = float(pkt.time)
        if start_time is None: start_time = current_time
        time_ms = int((current_time - start_time) * 1000)

        # Se rellena con los metadatos del paquete
        row = {
            "frame_number": count,
            "timestamp_ms": time_ms,
            "frame_len": len(pkt),
            "is_udp": 1 if is_udp_pkt else 0,
            "is_rpl": 1 if is_rpl_pkt else 0,
            "is_ack": 1 if is_ack_pkt else 0
        }

        # Se estructura la capa MAC
        row["mac_src"] = mac_src_str
        row["mac_dst"] = mac_dst_str
        row["mac_fcf"] = fcf_int if fcf_int is not None else np.nan
        
        if pkt.haslayer(Dot15d4):
            row["mac_seq_number"] = pkt[Dot15d4].seqnum
            
            # Se busca el PAN ID de los paquetes, puede variar según si es ascendente o
            # descendente los mensajes. En los casos de ACK, se deja como NaN
            if hasattr(dot15, "dest_panid") and dot15.dest_panid is not None:
                row["mac_pan_id"] = dot15.dest_panid
            elif hasattr(dot15, "src_panid") and dot15.src_panid is not None:
                row["mac_pan_id"] = dot15.src_panid
            else:
                row["mac_pan_id"] = np.nan
            # Llamada a la funcion de extraccion de flags
            row.update(extract_mac_flags(fcf_int))

        # Construccion de la capa de red
        ip_src_final = None
        ip_dst_final = None

        if pkt.haslayer(IPv6):
            ip_src_raw = pkt[IPv6].src
            ip_dst_raw = pkt[IPv6].dst
            
            # Se trata de recuperar las direcciones en formato IPv6, al tener comprimidas varias
            # como sucede con el tráfico UDP se filtra por 'fe::80' o su versión a descomprimir '::'
            if not ip_src_raw or ip_src_raw == "fe80::" or ip_src_raw == "::":
                ip_src_final = MAC_TO_IP.get(mac_src_str, ip_src_raw)
            else:
                ip_src_final = ip_src_raw
                
            # Lo mismo para el destino
            if not ip_dst_raw or ip_dst_raw == "fe80::" or ip_dst_raw == "::":
                ip_dst_final = MAC_TO_IP.get(mac_dst_str, ip_dst_raw)
            else:
                ip_dst_final = ip_dst_raw

        # Comrpobacion para RPL broadcast, mensajes DIS
        if is_rpl_pkt and not ip_dst_final:
            rpl_layer = pkt[ICMPv6RPL]
            if hasattr(rpl_layer, 'code') and rpl_layer.code in [0, 1]:
                # DIO (1) y DIS (0) casi siempre van a Multicast
                ip_dst_final = "ff02::1a"

        row["ip_src"] = ip_src_final
        row["ip_dst"] = ip_dst_final

        # En el caso de UDP es más sencillo, puesto que no difiere la lógica de la pilo TCP/IP tradicional
        if is_udp_pkt:
            udp = pkt[UDP]
            row["udp_src_port"] = int(udp.sport)
            row["udp_dst_port"] = int(udp.dport)
            row["udp_len"] = int(udp.len)
            payload = bytes(udp.payload)
            row["payload_size"] = len(payload)
            row["payload_hex"] = payload.hex()

        # Se llama a la función de parseo puro y se actualiza los campos
        if is_rpl_pkt:
            rpl_fields = parse_rpl_raw(pkt)
            row.update(rpl_fields)
        # Se agrega finalmente a la entrada del CSV
        rows.append(row)

    print(f"--> Generando CSV con {len(rows)} filas")
    # Se toma como NaNs los nulos, aunque requerirá de un relleno para varios modelos
    df = pd.DataFrame(rows)

    if not df.empty:
        if address_map:
            cols_to_map = ['mac_src', 'mac_dst', 'ip_src', 'ip_dst', 'rpl_dodagid']
            
            for col in cols_to_map:
                if col in df.columns:
                    # Se guarda la direccion en texto para realizar el mapeo
                    df[col + "_str"] = df[col]
                    # Mapeamos las direcciones, los nulos se rellenan con cero
                    df[col] = df[col].map(address_map).fillna(0).astype(int)
        else:
            print("--> No se proporcionó mapa. Las direcciones se mantienen en su formato original.")

        # Se ordena de forma similar a como se visualiza en Wireshark
        cols = [
            "frame_number", "timestamp_ms", "frame_len", 
            "is_udp", "is_rpl", "is_ack",
            "mac_src", "mac_dst", "mac_seq_number", "mac_pan_id", "mac_fcf",
            "mac_frame_type", "mac_security", "mac_pending", 
            "mac_ack_req", "mac_pan_id_comp", 
            "mac_dst_mode", "mac_src_mode", "mac_version",
            "ip_src", "ip_dst",
            "udp_src_port", "udp_dst_port", "udp_len", "payload_size", "payload_hex",
            "icmp_type", "rpl_code", 
            "rpl_rank", "rpl_version", "rpl_instance_id", 
            "rpl_mop", "rpl_dtsn", "rpl_flags",
            "rpl_dao_sequence", "rpl_dodagid"
        ]
        
        # Se anexan las columnas restantes usadas
        remaining = [c for c in df.columns if c not in cols]
        df = df[cols + remaining]
        
        df.to_csv(output_csv, index=False)
        print(f"Dataset guardado en: {output_csv}")
        
    else:
        print("No se encontraron paquetes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pcap", help="Ruta al archivo PCAP original de Cooja")
    parser.add_argument("output_csv", help="Ruta donde se guarda el CSV resultante")
    parser.add_argument("--map", dest="address_map", help="Ruta al archivo JSON con el mapa de direcciones (es opcional)", default=None)
    
    args = parser.parse_args()
    
    process_pcap(args.input_pcap, args.output_csv, args.address_map)