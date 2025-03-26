# Importamos el módulo socket para realizar las conexiones de red
import socket

# Definimos la función get_open_ports
def get_open_ports(target, port_range, verbose=False):
    open_ports = []  # Lista que almacenará los puertos abiertos

    # Verificamos si el target es una URL o IP válida
    try:
        # Intentamos obtener la dirección IP a partir del nombre de dominio (si es URL)
        target_ip = socket.gethostbyname(target)
    except socket.gaierror:
        # Si ocurre un error, devolvemos el mensaje adecuado
        if '.' in target:
            return "Error: Invalid IP address"
        else:
            return "Error: Invalid hostname"
    
    # Iteramos sobre el rango de puertos
    for port in range(port_range[0], port_range[1] + 1):
        # Usamos socket para intentar conectar con el puerto
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)  # Timeout de 1 segundo para cada intento de conexión
        result = sock.connect_ex((target_ip, port))  # Intentamos conectar
        
        if result == 0:
            # Si el puerto está abierto, lo añadimos a la lista
            open_ports.append(port)
        
        sock.close()  # Cerramos el socket para liberar recursos
    
    # Si estamos en modo verbose, devolvemos el formato detallado
    if verbose:
        if open_ports:
            service_info = "\n".join([f"{port}   {get_service_name(port)}" for port in open_ports])
            return f"Open ports for {target} ({target_ip})\nPORT     SERVICE\n{service_info}"
        else:
            return f"Open ports for {target} ({target_ip})\nNo open ports found"
    
    # Si no estamos en modo verbose, simplemente devolvemos los puertos abiertos
    return open_ports

# Función que retorna el nombre del servicio para un puerto dado
def get_service_name(port):
    # Diccionario con puertos comunes y sus servicios
    common_ports = {
        22: "ssh",
        80: "http",
        443: "https",
        21: "ftp",
        25: "smtp",
        110: "pop3",
        23: "telnet",
    }
    return common_ports.get(port, "Unknown service")
