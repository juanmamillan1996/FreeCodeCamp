import hashlib

# Función para hacer el hash de una contraseña utilizando SHA-1
def hash_password(password, salt="", use_salts=False):
    # Si se usan sales, agregamos las sales a la contraseña
    if use_salts:
        password = salt + password + salt  # Prepend y append de la sal
    # Creamos el hash con SHA-1
    password_hash = hashlib.sha1(password.encode('utf-8')).hexdigest()
    return password_hash

# Función principal para verificar el hash
def crack_password(hashed_password, use_salts=False):
    # Leemos las contraseñas del archivo top-10000-passwords.txt
    try:
        with open('top-10000-passwords.txt', 'r') as f:
            passwords = f.readlines()
    except FileNotFoundError:
        return "ERROR: top-10000-passwords.txt not found"

    # Leemos las sales del archivo known-salts.txt si use_salts es True
    salts = []
    if use_salts:
        try:
            with open('known-salts.txt', 'r') as f:
                salts = f.readlines()
        except FileNotFoundError:
            return "ERROR: known-salts.txt not found"

    # Iteramos sobre las contraseñas
    for password in passwords:
        password = password.strip()  # Eliminamos saltos de línea
        
        if use_salts:
            # Probamos con cada sal para la contraseña
            for salt in salts:
                salt = salt.strip()  # Eliminamos saltos de línea de la sal
                hashed_attempt = hash_password(password, salt, use_salts)
                if hashed_attempt == hashed_password:
                    return password
        else:
            # Si no se usan sales, solo probamos la contraseña tal cual
            hashed_attempt = hash_password(password)
            if hashed_attempt == hashed_password:
                return password

    # Si no se encuentra ninguna coincidencia
    return "PASSWORD NOT IN DATABASE"
