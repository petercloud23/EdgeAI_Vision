# ==============================================================================
# CLIENTE DE TESTE: LEITURA DE VÍDEO E INFERÊNCIA VIA ENDPOINTS HTTP DA API
# ==============================================================================
import cv2
import requests
import numpy as np

# 1. CONFIGURAÇÃO DOS ENDPOINTS DA API LOCAL
# URL base gerada pelo main.py / Docker
URL_BASE = "http://127.0.0.1:8000"
ENDPOINT_JSON = f"{URL_BASE}/predict"
ENDPOINT_IMAGEM = f"{URL_BASE}/predict/image"

# Defina qual endpoint deseja testar alterando a variável abaixo:
# True: Exibe a imagem anotada processada e desenhada pelo próprio servidor
# False: Consome o JSON de dados e você mesmo pode tratar as coordenadas
TESTAR_RETORNO_DE_IMAGEM = True

# 2. FONTE DE VÍDEO LOCAL
caminho_do_video = "video_teste.mp4"
cap = cv2.VideoCapture(caminho_do_video)

if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o arquivo de vídeo '{caminho_do_video}'.")
    exit()

print("Conectado à API. Iniciando transmissão de frames...")
print("Pressione 'q' na janela de exibição para encerrar.")

# 3. LOOP PRINCIPAL DE TRANSMISSÃO
while cap.isOpened():
    sucesso, frame = cap.read()
    
    if not sucesso:
        print("Fim do vídeo ou falha ao ler o quadro.")
        break

    # Codifica o frame atual do OpenCV (matriz numpy) para formato JPEG em memória
    sucesso_codificacao, buffer = cv2.imencode(".jpg", frame)
    if not sucesso_codificacao:
        continue

    # Transforma o buffer em um arquivo binário pronto para envio multipart/form-data
    arquivos = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}

    try:
        if TESTAR_RETORNO_DE_IMAGEM:
            # Opção A: Consome o endpoint que retorna a imagem PNG com as caixas prontas
            resposta = requests.post(ENDPOINT_IMAGEM, files=arquivos)
            
            if resposta.status_code == 200:
                # Reconverte o binário da imagem PNG recebida de volta para matriz OpenCV
                nparr = np.frombuffer(resposta.content, np.uint8)
                frame_anotado = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Exibe a imagem anotada gerada pelo servidor
                cv2.imshow("EdgeAI_Vision - Stream de Video via API", frame_anotado)
        else:
            # Opção B: Consome o endpoint JSON e printa as coordenadas no terminal
            resposta = requests.post(ENDPOINT_JSON, files=arquivos)
            if resposta.status_code == 200:
                dados = resposta.json()
                print(f"Latência: {dados['inference_time_ms']}ms | Detecções: {dados['detections']}")
                
                # Exibe o frame original limpo enquanto o terminal mostra os dados
                cv2.imshow("EdgeAI_Vision - Stream de Video via API", frame)

    except requests.exceptions.ConnectionError:
        print("Erro: Não foi possível conectar à API. Certifique-se de que o main.py ou o container Docker está rodando.")
        break

    # Tecla de atalho para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Execução interrompida pelo usuário.")
        break

# 4. FINALIZAÇÃO E LIMPEZA
cap.release()
cv2.destroyAllWindows()
print("Recursos finalizados com sucesso.")
