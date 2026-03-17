import logging
import os
import yaml

def setup_logger():
    # Carrega config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    log_file = config['logging']['file_name']
    
    # Cria a pasta de logs 
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    return logging

# Inicializa para uso global
logger = setup_logger()