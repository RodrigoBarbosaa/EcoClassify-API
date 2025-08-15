from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow import keras
import base64
import io
from PIL import Image
import logging
from typing import List
import uvicorn

# lidar com app e rotas

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da aplicação
app = FastAPI(
    title="Image Classification API",
    description="API para classificação de imagens usando modelo Keras",
    version="1.0.0"
)

# Modelo de dados para entrada
class ImageRequest(BaseModel):
    image_data: str  # Base64 encoded image
    image_format: str = "PNG"  # Formato da imagem (PNG, JPEG, etc.)

# Modelo de dados para saída
class PredictionResponse(BaseModel):
    predictions: List[float]
    predicted_class: int
    confidence: float

# Variável global para o modelo
model = None
INPUT_SHAPE = (28, 28, 1)  # Exemplo: MNIST shape
NUM_CLASSES = 10

def load_model(model_path: str = "model.keras"):
    global model
    try:
        model = keras.models.load_model(model_path)
        logger.info(f"Modelo carregado com sucesso: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return False

def preprocess_image(image_data: str) -> np.ndarray:
    try:
        # Decodifica base64
        image_bytes = base64.b64decode(image_data)
        
        # Converte para PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Converte para escala de cinza se necessário
        if image.mode != 'L':
            image = image.convert('L')
        
        # Redimensiona para o tamanho esperado pelo modelo
        image = image.resize((INPUT_SHAPE[0], INPUT_SHAPE[1]))
        
        # Converte para array numpy
        image_array = np.array(image)
        
        # Normaliza valores para 0-1
        image_array = image_array.astype(np.float32) / 255.0
        
        # Adiciona dimensão do batch e canal se necessário
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Erro no preprocessamento da imagem: {e}")
        raise HTTPException(status_code=400, detail=f"Erro no processamento da imagem: {str(e)}")

# carrega o modelo
@app.on_event("startup")
async def startup_event():
    success = load_model()
    if not success:
        logger.warning("Modelo não foi carregado. Tentando criar modelo de exemplo...")
        create_example_model()

# chamar para ligar o render
@app.get("/")
async def root():
    return {
        "message": "Image Classification API",
        "status": "running",
        "model_loaded": model is not None
    }

# endpoint pro frontend
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    if not model:
        create_example_model() #TODO: remover essa linha em prod
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        # Preprocessa a imagem
        processed_image = preprocess_image(request.image_data)
        
        # Faz a predição
        predictions = model.predict(processed_image)
        predictions_list = predictions[0].tolist()
        
        # Encontra a classe predita e confiança
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        logger.info(f"Predição realizada: classe={predicted_class}, confiança={confidence:.4f}")
        
        return PredictionResponse(
            predictions=predictions_list,
            predicted_class=predicted_class,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


# Modelo para teste
def create_example_model():
    global model
    
    try:
        # Modelo simples para classificação MNIST-like
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Salva o modelo
        model.save("model.keras")
        logger.info("Modelo de exemplo criado e salvo como 'model.keras'")
        
    except Exception as e:
        logger.error(f"Erro ao criar modelo de exemplo: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)