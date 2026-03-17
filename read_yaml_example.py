import cv2
import yaml
import time

# 1. CARREGAR AS CONFIGURAÇÕES DO YAML
def load_config(path="config.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Extracao dos dados
width = config['video_specs']['resolution']['width']
height = config['video_specs']['resolution']['height']
fps_target = config['video_specs']['fps']
model_name = config['ai_model']['architecture']

print(f"--- Iniciando {config['project_info']['name']} v{config['project_info']['version']} ---")
print(f"Modelo: {model_name} | Local: {config['project_info']['location']}")

# 2. CONFIGURAR A CAPTURA DE VÍDEO

cap = cv2.VideoCapture(0) # (Aqui você pode usar '0' para webcam ou o caminho de um arquivo de vídeo)

# Aplicando a resolução do YAML
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 3. LOOP PRINCIPAL DE PROCESSAMENTO
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Simulação da IA: Aqui entrará o seu modelo Mask R-CNN ou YOLO
    # Por enquanto, vamos apenas exibir informações na tela
    # Tera que ser alterado no futuro para mostrar as detecções reais
    cv2.putText(frame, f"Monitorando: {config['processing']['breed_focus']}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Altura: {config['camera_setup']['height_meters']}m", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Exibir o vídeo em tempo real
    cv2.imshow("BioWeight_PR - Monitoramento Dorsal", frame)

    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()