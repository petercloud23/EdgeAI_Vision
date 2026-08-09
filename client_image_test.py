# ==============================================================================
# CLIENTE DE TESTE: LEITURA DE IMAGEM E INFERÊNCIA VIA ENDPOINT HTTP DA API
# ==============================================================================

import cv2
import requests
import numpy as np
import time

# ==============================================================================
# 1. CONFIGURAÇÃO DO ENDPOINT DA API
# ==============================================================================

# IMPORTANTE: manter o protocolo http://
URL_BASE = "http://127.0.0.1:8000"

ENDPOINT_IMAGEM = f"{URL_BASE}/predict/image"

# ==============================================================================
# 2. IMAGEM LOCAL PARA TESTE
# ==============================================================================

caminho_da_imagem = "imagem_teste.jpg"

# Também pode ser:
# caminho_da_imagem = "imagem_teste.png"


# ==============================================================================
# 3. LEITURA DA IMAGEM
# ==============================================================================

imagem = cv2.imread(caminho_da_imagem)

if imagem is None:
    print(f"Erro: Não foi possível abrir a imagem '{caminho_da_imagem}'.")
    exit()

print("Imagem carregada com sucesso.")
print("Enviando imagem para a API...")


# ==============================================================================
# 4. CODIFICAÇÃO DA IMAGEM
# ==============================================================================

sucesso_codificacao, buffer = cv2.imencode(".jpg", imagem)

if not sucesso_codificacao:
    print("Erro: Não foi possível converter a imagem para JPEG.")
    exit()


# ==============================================================================
# 5. PREPARAÇÃO DO ARQUIVO PARA ENVIO
# ==============================================================================

arquivos = {
    "file": (
        "imagem_teste.jpg",
        buffer.tobytes(),
        "image/jpeg"
    )
}


# ==============================================================================
# 6. ENVIO PARA A API
# ==============================================================================

try:

    inicio = time.perf_counter()

    resposta = requests.post(
        ENDPOINT_IMAGEM,
        files=arquivos,
        timeout=60
    )

    fim = time.perf_counter()

    tempo_total = (fim - inicio) * 1000


# ==============================================================================
# 7. PROCESSAMENTO DA RESPOSTA
# ==============================================================================

    if resposta.status_code == 200:

        print("Inferência realizada com sucesso!")
        print(f"Tempo total da requisição: {tempo_total:.2f} ms")

        # Converte a resposta da API novamente para imagem OpenCV
        nparr = np.frombuffer(
            resposta.content,
            np.uint8
        )

        imagem_anotada = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if imagem_anotada is None:
            print("Erro: A API retornou um conteúdo que não pôde ser convertido em imagem.")
            exit()

        # ----------------------------------------------------------------------
        # Exibe o resultado
        # ----------------------------------------------------------------------

        cv2.imshow(
            "EdgeAI_Vision - Resultado da Inferencia",
            imagem_anotada
        )

        print()
        print("Imagem processada pela IA.")
        print("Pressione qualquer tecla na janela da imagem para encerrar.")

        cv2.waitKey(0)

    else:

        print("Erro na API.")
        print(f"Status HTTP: {resposta.status_code}")
        print(f"Resposta: {resposta.text}")


# ==============================================================================
# 8. TRATAMENTO DE ERROS DE CONEXÃO
# ==============================================================================

except requests.exceptions.ConnectionError:

    print()
    print("ERRO: Não foi possível conectar à API.")
    print("Verifique se o container Docker está em execução.")

except requests.exceptions.Timeout:

    print()
    print("ERRO: A API demorou muito para responder.")

except requests.exceptions.RequestException as erro:

    print()
    print(f"ERRO na comunicação com a API: {erro}")


# ==============================================================================
# 9. FINALIZAÇÃO
# ==============================================================================

cv2.destroyAllWindows()

print()
print("Teste finalizado.")