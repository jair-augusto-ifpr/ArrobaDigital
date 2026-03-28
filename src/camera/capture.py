import cv2
import threading

class Capture:
    def __init__(self, src=0, name="Camera"):
        """
        Inicializa o fluxo de vídeo.
        :param src: 0 para webcam, "http://ip:porta/video" para celular/Câmera IP, ou "video.mp4" para arquivo.
        :param name: Nome da thread (útil para debug se você tiver múltiplas câmeras).
        """
        self.src = src
        self.name = name
        self.cap = cv2.VideoCapture(self.src)
        
        if not self.cap.isOpened():
            raise ValueError(f"[ERRO CRÍTICO] Não foi possível abrir a fonte de vídeo: {self.src}")
            
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        """Inicia a thread para ler frames em segundo plano."""
        t = threading.Thread(target=self.update, name=self.name, daemon=True)
        t.start()
        return self

    def update(self):
        """Loop contínuo que roda em segundo plano atualizando o frame."""
        while True:
            if self.stopped:
                self.cap.release()
                return


            ret, frame = self.cap.read()

            if not ret:
                self.stopped = True
                self.ret = False
                return
            
            self.ret = ret
            self.frame = frame

    def read(self):
        """Retorna o status e o frame mais recente (idêntico ao cv2.VideoCapture padrão)."""
        return self.ret, self.frame

    def stop(self):
        """Sinaliza para a thread parar de ler e liberar a câmera."""
        self.stopped = True