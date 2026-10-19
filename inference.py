import os
import time
import logging
from typing import Union, List, Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

# Configuração de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EPIDetector")


class EPIDetector:
    """
    Detector de Equipamentos de Proteção Individual (EPI) otimizado para ONNX Runtime.
    Mapeia detecções de: pessoa, capacete, colete, luva, oculos.
    """

    CLASSES = ["pessoa", "capacete", "colete", "luva", "oculos"]

    def __init__(
        self,
        model_path: str = "models/epi_yolov8n.onnx",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
    ):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size
        self.session = None
        self.use_mock = False

        self._load_model()

    def _load_model(self) -> None:
        """Carrega o modelo ONNX em CPU local. Ativa modo simulado (mock) caso o arquivo não exista."""
        if os.path.exists(self.model_path):
            try:
                # Prioriza provedor de execução CPU local
                providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                
                # Obtém nomes de entrada e saída do modelo
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [output.name for output in self.session.get_outputs()]
                logger.info(f"Modelo ONNX carregado com sucesso de '{self.model_path}'.")
                return
            except Exception as e:
                logger.warning(f"Falha ao carregar o modelo ONNX de '{self.model_path}': {e}. Ativando modo simulado.")
        else:
            logger.warning(
                f"Arquivo de modelo não encontrado em '{self.model_path}'. "
                f"Inicializando em MOCK MODE (Detecção Simulada) para testes."
            )

        self.use_mock = True

    def preprocess(self, image_np: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Pré-processamento da imagem:
        - Redimensionamento mantendo a proporção (letterbox) para 640x640.
        - Conversão BGR -> RGB.
        - Normalização dos pixels [0.0, 1.0].
        - Transposição para o formato (1, 3, 640, 640).
        """
        orig_h, orig_w = image_np.shape[:2]
        target_w, target_h = self.input_size

        # Fator de escala
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Redimensiona imagem
        resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Preenchimento (letterbox) para 640x640
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        padded_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded_img[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # BGR para RGB
        rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)

        # Normalização [0, 1] e formato (1, 3, H, W)
        blob = rgb_img.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # (1, 3, H, W)

        return blob, scale, (pad_x, pad_y)

    def postprocess(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """
        Pós-processamento dos tensores do YOLOv8:
        - Leitura das bounding boxes e probabilidades por classe.
        - Aplicação de NMS (Non-Maximum Suppression).
        - Mapeamento das coordenadas de volta para a resolução original da imagem.
        """
        orig_h, orig_w = orig_shape
        pad_x, pad_y = pad

        # Saída padrão YOLOv8: shape (1, 4 + C, 8400)
        preds = outputs[0]
        if preds.ndim == 3:
            preds = preds[0]  # Shape (4 + C, 8400)

        # Se transposto, ajusta para (8400, 4 + C)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        num_classes = len(self.CLASSES)
        boxes = []
        confidences = []
        class_ids = []

        # Itera sobre os candidatos
        for pred in preds:
            box = pred[:4]
            scores = pred[4 : 4 + num_classes]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence >= self.conf_thres:
                cx, cy, w, h = box
                
                # Converte de letterbox 640x640 para imagem original
                x1 = (cx - w / 2 - pad_x) / scale
                y1 = (cy - h / 2 - pad_y) / scale
                x2 = (cx + w / 2 - pad_x) / scale
                y2 = (cy + h / 2 - pad_y) / scale

                # Normalização e limites dentro da imagem original
                x1 = max(0, min(orig_w, x1))
                y1 = max(0, min(orig_h, y1))
                x2 = max(0, min(orig_w, x2))
                y2 = max(0, min(orig_h, y2))

                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])  # Formato [x, y, w, h] para cv2.NMSBoxes
                confidences.append(confidence)
                class_ids.append(class_id)

        # Aplica NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.iou_thres)

        detections = []
        if len(indices) > 0:
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            for idx in indices:
                x, y, w, h = boxes[idx]
                label_idx = class_ids[idx]
                label_name = (
                    self.CLASSES[label_idx]
                    if label_idx < len(self.CLASSES)
                    else f"class_{label_idx}"
                )

                detections.append(
                    {
                        "label": label_name,
                        "confidence": float(round(confidences[idx], 4)),
                        "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    }
                )

        return detections

    def predict_mock(self, orig_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Gera detecções simuladas para testes quando o modelo ONNX não está presente."""
        orig_h, orig_w = orig_shape
        mock_detections = [
            {
                "label": "pessoa",
                "confidence": 0.92,
                "bbox": [
                    int(orig_w * 0.1),
                    int(orig_h * 0.1),
                    int(orig_w * 0.8),
                    int(orig_h * 0.9),
                ],
            },
            {
                "label": "capacete",
                "confidence": 0.88,
                "bbox": [
                    int(orig_w * 0.35),
                    int(orig_h * 0.12),
                    int(orig_w * 0.65),
                    int(orig_h * 0.30),
                ],
            },
            {
                "label": "colete",
                "confidence": 0.85,
                "bbox": [
                    int(orig_w * 0.25),
                    int(orig_h * 0.32),
                    int(orig_w * 0.75),
                    int(orig_h * 0.70),
                ],
            },
        ]
        return mock_detections

    def predict(
        self, image_input: Union[str, bytes, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Executa o pipeline completo de inferência local.
        Retorna dicionário contendo 'success', 'inference_time_ms' e lista de 'detections'.
        """
        start_time = time.perf_counter()

        try:
            # Converte entrada para np.ndarray BGR (formato do OpenCV)
            if isinstance(image_input, (str, os.PathLike)):
                image_np = cv2.imread(str(image_input))
                if image_np is None:
                    raise ValueError(f"Não foi possível carregar a imagem do caminho: {image_input}")
            elif isinstance(image_input, bytes):
                nparr = np.frombuffer(image_input, np.uint8)
                image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image_np is None:
                    raise ValueError("Buffer de imagem corrompido ou formato inválido.")
            elif isinstance(image_input, Image.Image):
                image_np = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                image_np = image_input
            else:
                raise TypeError(f"Tipo de entrada não suportado: {type(image_input)}")

            orig_shape = image_np.shape[:2]

            # Execução Mock se o modelo não foi carregado
            if self.use_mock:
                # Simula pequena latência de CPU (~15ms)
                time.sleep(0.015)
                detections = self.predict_mock(orig_shape)
            else:
                # 1. Pré-processamento
                input_tensor, scale, pad = self.preprocess(image_np)

                # 2. Inferência ONNX em CPU
                outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

                # 3. Pós-processamento e NMS
                detections = self.postprocess(outputs, orig_shape, scale, pad)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return {
                "success": True,
                "inference_time_ms": round(elapsed_ms, 2),
                "detections": detections,
            }

        except Exception as e:
            logger.error(f"Erro durante a inferência: {e}", exc_info=True)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": False,
                "error": str(e),
                "inference_time_ms": round(elapsed_ms, 2),
                "detections": [],
            }


if __name__ == "__main__":
    # Teste rápido do EPIDetector
    print("Testando EPIDetector localmente...")
    detector = EPIDetector()
    
    # Cria imagem sintética de teste (640x480)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_img, "EPI Test Image", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    result = detector.predict(test_img)
    print("\nResultado da Inferência:")
    print(f"Sucesso: {result['success']}")
    print(f"Tempo de Inferência: {result['inference_time_ms']} ms")
    print(f"Detecções ({len(result['detections'])}):")
    for det in result["detections"]:
        print(f" - {det['label']} (confiança: {det['confidence']}): BBox {det['bbox']}")
