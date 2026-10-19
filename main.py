import io
import logging
from typing import Dict, Any, List
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

from inference import EPIDetector

# Configuração de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeAI_API")

# Inicialização da API
app = FastAPI(
    title="EdgeAI_Vision - API de Detecção de EPIs",
    description="API REST autônoma para detecção de Equipamentos de Proteção Individual em tempo real.",
    version="1.0.0",
)

# Instância do Detector de EPIs
detector = EPIDetector()

# Paleta de Cores BGR para Anotação por Classe
CLASS_COLORS = {
    "pessoa": (255, 128, 0),      # Azul claro
    "capacete": (0, 255, 0),      # Verde brilhante
    "colete": (0, 215, 255),      # Amarelo/Dourado
    "luva": (255, 255, 0),        # Ciano
    "oculos": (255, 0, 255),      # Magenta
}
DEFAULT_COLOR = (0, 165, 255)      # Laranja para classes desconhecidas


@app.get("/")
def read_root():
    """Endpoint de status do serviço."""
    return {
        "service": "EdgeAI_Vision API",
        "status": "online",
        "mock_mode": detector.use_mock,
        "classes": detector.CLASSES,
    }


@app.get("/health")
def health_check():
    """Endpoint de verificação de integridade (Health Check)."""
    return {"status": "healthy", "use_mock": detector.use_mock}


@app.post("/predict")
async def predict_json(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Endpoint HTTP POST /predict
    - Entrada: Arquivo de imagem enviado via multipart/form-data.
    - Saída: Payload JSON com status, tempo de inferência (ms) e lista de detecções.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    # Validação do tipo de conteúdo
    if file.content_type and not file.content_type.startswith("image/"):
        logger.warning(f"Tipo de arquivo rejeitado: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo inválido '{file.content_type}'. Envie uma imagem válida (JPEG, PNG, etc.).",
        )

    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="O arquivo enviado está vazio.")

        result = detector.predict(contents)

        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"Erro ao processar imagem: {result.get('error', 'Falha desconhecida')}",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no processamento do endpoint /predict: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Erro interno do servidor ao processar a imagem: {str(e)}"
        )


@app.post("/predict/image")
async def predict_annotated_image(file: UploadFile = File(...)):
    """
    Endpoint HTTP POST /predict/image
    - Entrada: Arquivo de imagem enviado via multipart/form-data.
    - Saída: Imagem PNG anotada com bounding boxes e rótulos das detecções de EPIs.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo inválido '{file.content_type}'. Envie uma imagem válida.",
        )

    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="O arquivo enviado está vazio.")

        # Decodifica imagem para matriz NumPy (OpenCV BGR)
        nparr = np.frombuffer(contents, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            raise HTTPException(
                status_code=400,
                detail="Não foi possível decodificar o arquivo de imagem enviado.",
            )

        # Inferência
        result = detector.predict(img_np)

        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"Erro na inferência da imagem: {result.get('error', 'Falha desconhecida')}",
            )

        annotated_img = img_np.copy()
        detections = result.get("detections", [])

        # Desenha Bounding Boxes e Rótulos na Imagem Original
        for det in detections:
            label = det["label"]
            confidence = det["confidence"]
            bbox = det["bbox"]  # [x1, y1, x2, y2]

            x1, y1, x2, y2 = bbox
            color = CLASS_COLORS.get(label.lower(), DEFAULT_COLOR)

            # Desenha retângulo principal
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness=3)

            # Texto do Rótulo
            text = f"{label.upper()} {int(confidence * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            # Tamanho da caixa de texto
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Posição da caixa de texto
            label_y1 = max(0, y1 - text_h - 10)
            label_y2 = y1
            label_x2 = x1 + text_w + 10

            # Desenha fundo preenchido para o texto (garante legibilidade)
            cv2.rectangle(
                annotated_img,
                (x1, label_y1),
                (label_x2, label_y2),
                color,
                thickness=-1,
            )

            # Desenha o texto do rótulo
            cv2.putText(
                annotated_img,
                text,
                (x1 + 5, label_y2 - 5),
                font,
                font_scale,
                (0, 0, 0),  # Texto em preto sobre fundo colorido
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        # Codifica imagem anotada para formato PNG
        success, encoded_img = cv2.imencode(".png", annotated_img)
        if not success:
            raise HTTPException(
                status_code=500, detail="Erro ao codificar a imagem anotada para PNG."
            )

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no processamento do endpoint /predict/image: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Erro interno do servidor ao gerar imagem anotada: {str(e)}"
        )


if __name__ == "__main__":
    logger.info("Iniciando servidor FastAPI em 0.0.0.0:8000...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
