import random

def player(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)
    
    # Estrategia inicial: Si no hay historial, empezar con Piedra ('R')
    if not opponent_history:
        return 'R'
    
    # Contar cuántas veces el oponente ha jugado cada movimiento
    counts = {
        'R': opponent_history.count('R'),
        'P': opponent_history.count('P'),
        'S': opponent_history.count('S')
    }
    
    # Predecir el próximo movimiento del oponente (el que más ha usado)
    predicted_move = max(counts, key=counts.get)
    
    # Jugar la mejor respuesta contra el movimiento predicho
    counter_moves = {'R': 'P', 'P': 'S', 'S': 'R'}
    return counter_moves[predicted_move]
