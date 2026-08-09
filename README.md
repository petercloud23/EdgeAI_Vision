# EdgeAI_Vision - Detecção Autônoma de EPIs na Borda Industrial (Raspberry Pi 5)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNXRuntime-CPU_ARM64-orange.svg)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-Multi--Arch_ARM64-blue.svg)](https://www.docker.com/)

O **EdgeAI_Vision** é um sistema completo e autônomo de Visão Computacional para Borda (*Edge AI*), projetado para detecção e verificação de Equipamentos de Proteção Individual (EPIs) em tempo real em ambientes industriais, operando de forma 100% autônoma em hardware **Raspberry Pi 5** (CPU ARM64).

---

## 1. Descrição Geral e Arquitetura da Solução

Em ambientes industriais, a verificação em tempo real do uso de EPIs (como **capacetes**, **coletes**, **luvas**, **óculos de proteção** e presença de **pessoas**) é fundamental para a prevenção de acidentes e conformidade com normas de segurança do trabalho (NR-6).

O **EdgeAI_Vision** resolve esse desafio processando o fluxo de imagens diretamente na borda (*Edge Computing*), eliminando a latência e os custos de tráfego de rede para a nuvem e garantindo o funcionamento contínuo mesmo sem conexão com a internet.

### Arquitetura do Sistema

```
[ Câmera / Imagem ]
        │
        ▼
[ API FastAPI (Uvicorn HTTP Server) ] ── (host 0.0.0.0:8000)
        │
        ▼
[ Pipeline EPIDetector (inference.py) ]
  ├── Preprocessamento (Resize 640x640, RGB, Letterbox, Normalização)
  ├── ONNX Runtime Inference (Execução em CPU ARM64 Cortex-A76)
  └── Pós-processamento (NMS - Non-Maximum Suppression & Mapeamento BBox)
        │
        ├──► POST /predict        ──► Payload JSON (latência em ms, detecções)
        └──► POST /predict/image  ──► Imagem PNG Anotada com caixas e rótulos
```

---

## 2. Justificativa do Modelo & Otimização para CPU

A escolha do modelo de IA deve equilibrar a precisão média ($mAP$) necessária para identificar objetos pequenos (como óculos ou luvas) e o orçamento computacional da CPU de borda.

| Modelo / Arquitetura | Formato | Resolução | FPS estimado (CPU Raspberry Pi 5) | Memória RAM | $mAP_{50}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n (Recomendado)** | **ONNX (INT8 / FP16)** | **$640 \times 640$** | **$\approx 18 - 25\text{ FPS}$** | **$\approx 250\text{ MB}$** | **$\approx 37.3\%$** |
| YOLOv11n | ONNX (INT8) | $640 \times 640$ | $\approx 15 - 22\text{ FPS}$ | $\approx 280\text{ MB}$ | $\approx 39.0\%$ |
| MobileNetV3-SSD | ONNX (FP32) | $320 \times 320$ | $\approx 35 - 45\text{ FPS}$ | $\approx 180\text{ MB}$ | $\approx 28.5\%$ |

### Por que o YOLOv8n em formato ONNX INT8?
1. **Compromisso Precisão vs. Latência:** O YOLOv8n oferece detecção multi-classe precisa para objetos de tamanhos variados mantendo o tempo de inferência entre $15\text{ ms}$ e $30\text{ ms}$ por quadro na CPU ARM Cortex-A76.
2. **Quantização INT8:** A quantização reduz o tamanho dos pesos de $12\text{ MB}$ para $\approx 3.5\text{ MB}$, reduzindo drasticamente a largura de banda de memória utilizada e permitindo até $2\times$ de ganho de velocidade em relação ao modelo PyTorch original (.pt).
3. **Sem Dependência de GPU:** A inferência é otimizada via **ONNX Runtime CPU Execution Provider**, dispensando placas aceleradoras dedicadas.

---

## 3. Instruções de Implantação Autônoma (Produção 24/7)

A solução foi desenvolvida para operar **24/7 de forma desacoplada**, sem necessidade de manter sessões de terminal, conexões SSH ou IDEs ativas.

### Subindo a Aplicação via Docker Compose

1. Clone o repositório no Raspberry Pi 5:
   ```bash
   git clone https://github.com/seu-usuario/EdgeAI_Vision.git
   cd EdgeAI_Vision
   ```

2. Compile e inicie o container em segundo plano (*Detached Mode*):
   ```bash
   docker compose up -d --build
   ```

3. Verifique o status do container:
   ```bash
   docker compose ps
   ```

O container continuará executando continuamente no daemon do Docker (`dockerd`), com a política `restart: always` ativada. Reinicializações físicas do Raspberry Pi OS farão o serviço subir automaticamente no boot da placa.

---

### (Opcional) Configuração de Serviço Autônomo no `systemd`

Caso deseje gerenciar a API diretamente no sistema operacional nativo sem Docker:

1. Crie o arquivo `/etc/systemd/system/edgeai-vision.service`:
   ```ini
   [Unit]
   Description=EdgeAI Vision EPI Detection API Service
   After=network.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/EdgeAI_Vision
   ExecStart=/usr/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

2. Ative e inicie o serviço:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable edgeai-vision
   sudo systemctl start edgeai-vision
   ```

---

## 4. Exemplos Práticos de Uso da API (Endpoints REST)

A API aceita requisições diretas de qualquer dispositivo conectado à rede local industrial.

### Endpoint 1: `POST /predict` (Gera resposta estruturada em JSON)

**Comando cURL:**
```bash
curl -X POST -F "file=@imagem_teste.jpg" http://<IP_DO_RASPBERRY_PI>:8000/predict
```

**Payload JSON de Resposta (200 OK):**
```json
{
  "success": true,
  "inference_time_ms": 17.02,
  "detections": [
    {
      "label": "pessoa",
      "confidence": 0.92,
      "bbox": [80, 60, 640, 540]
    },
    {
      "label": "capacete",
      "confidence": 0.88,
      "bbox": [280, 72, 520, 180]
    },
    {
      "label": "colete",
      "confidence": 0.85,
      "bbox": [200, 192, 600, 420]
    }
  ]
}
```

---

### Endpoint 2: `POST /predict/image` (Gera Imagem PNG Anotada)

**Comando cURL:**
```bash
curl -X POST -F "file=@imagem_teste.jpg" http://<IP_DO_RASPBERRY_PI>:8000/predict/image -o resultado_anotado.png
```

**Retorno:** Arquivo de imagem `image/png` contendo as *bounding boxes* coloridas e rótulos de identificação desenhados com fundo preenchido para alta legibilidade.

---

### Endpoints de Diagnóstico

- **`GET /health`**: `{"status": "healthy", "use_mock": false}`
- **`GET /`**: `{"service": "EdgeAI_Vision API", "status": "online", "classes": [...]}`

---

## 5. Análise de Viabilidade na Raspberry Pi 5 & Operação na Borda

### Arquitetura de Hardware e Suporte ARM64
A **Raspberry Pi 5** é equipada com o processador quad-core **Broadcom BCM2712 (ARM Cortex-A76 a 2.4 GHz)**. Todas as imagens Docker (`python:3.11-slim`) e bibliotecas (`onnxruntime`, `opencv-python-headless`) foram compiladas e validadas nativamente para a arquitetura `linux/arm64`.

### Estratégia na Ausência de GPU Dedicada
A ausência de NPU ou GPU dedicada é superada utilizando aceleração por software:
- **VETORIZAÇÃO NEON SIMD:** O ONNX Runtime utiliza instruções vetoriais NEON nativas dos núcleos ARM Cortex-A76 para multiplicação rápida de matrizes em INT8.
- **Multithreading Nativo:** O servidor Uvicorn gerencia as requisições de I/O enquanto o ONNX Runtime distribui o cálculo dos tensores entre os 4 núcleos físicos da CPU.

### Consumo de Recursos e Gerenciamento Térmico
- **Consumo de Memória RAM:** $\approx 200\text{ MB} - 350\text{ MB}$ (extremamente leve em relação aos 4 GB ou 8 GB disponíveis na placa).
- **Taxa de Processamento Esperada:** $\approx 18\text{ a } 25\text{ FPS}$ em resolução $640 \times 640$, suficiente para análise de vídeo em tempo real ($15\text{ FPS}$).
- **Mitigação de Aquecimento (*Thermal Throttling*):** Recomenda-se o uso do **Raspberry Pi Active Cooler** (cooler oficial com dissipador de alumínio). Com o cooler ativo, a temperatura da CPU estabiliza em $\approx 55^\circ\text{C}$ sob carga total contínua de inferência, evitando a queda de frequência da CPU.

### Desacoplamento Total do Ecossistema Antigravity
A plataforma **Google Antigravity** desempenha exclusivamente o papel de **ambiente de desenvolvimento e engenharia assistida**. Após a compilação e criação dos artefatos (Dockerfile, código Python e modelos ONNX), o runtime em produção roda **100% autônomo** no sistema operacional do dispositivo de borda. Encerrar IDEs, deslogar de terminais ou desligar computadores de desenvolvimento **não impacta** a disponibilidade do serviço industrial.
