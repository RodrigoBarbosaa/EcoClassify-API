from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
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
from fastapi.middleware.cors import CORSMiddleware

#from fastapi.templating import Jinja2Templates

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da aplicação
app = FastAPI(
    title="Garbage Classification API",
    description="API para classificar imagens de lixo usando um modelo Keras pré-treinado",
    version="1.0.0"
)

origins = [
    "http://localhost:8000",
    "https://eco-scan-app.lovable.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], # Especificar ou usar ["*"]
    allow_headers=["*"],
)

# Modelo de dados para entrada
class ImageRequest(BaseModel):
    image_data: str  # Imagem codificada em Base64

# Modelo de dados para saída
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_predictions: dict  # Para incluir todas as probabilidades

# Variável global para o modelo e outras configurações
model = None
IMG_SIZE = (224, 224)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def load_model(model_path: str = "waste_classifier_mobilenetv2.keras"):
    global model
    try:
        model = keras.models.load_model(model_path)
        logger.info(f"Modelo carregado com sucesso: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {e}")
        return False

def preprocess_image(image_data: str) -> np.ndarray:
   
    try:
        # Decodifica a string base64 para bytes
        image_bytes = base64.b64decode(image_data)
        
        # Converte para um objeto PIL Image e garante o formato RGB
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Redimensiona para o tamanho esperado pelo modelo
        image = image.resize(IMG_SIZE)
        
        # Converte para um array numpy. Os valores dos pixels estarão em [0, 255]
        image_array = np.array(image)
        
        # Adiciona a dimensão do batch para que o shape seja (1, 224, 224, 3)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
        
    except Exception as e:
        logger.error(f"Erro no pré-processamento da imagem: {e}")
        raise HTTPException(status_code=400, detail=f"Erro no processamento da imagem: {str(e)}")

# Rota de inicialização
@app.on_event("startup")
async def startup_event():
    load_model()

# Crie uma instância de Jinja2Templates
#templates = Jinja2Templates(directory="templates")

# Rota principal para a página de teste
@app.get("/")
async def root():
    return {"message": "API de Classificação de Lixo está no ar. Use o endpoint /predict para enviar imagens."}
    #return templates.TemplateResponse("index.html", {"request": request})

# Endpoint para o frontend
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    
    try:
        # Pré-processa a imagem
        processed_image = preprocess_image(request.image_data)
        
        # Faz a predição
        predictions = model.predict(processed_image)
        predictions_list = predictions[0].tolist()
        
        # Encontra a classe predita e sua confiança
        predicted_class_index = int(np.argmax(predictions[0]))
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(predictions[0]))
        
        # Mapeia todas as probabilidades para nomes de classes
        all_predictions_dict = {
            class_name: prob for class_name, prob in zip(class_names, predictions_list)
        }
        
        logger.info(f"Predição realizada: classe={predicted_class_name}, confiança={confidence:.4f}")
        
        return PredictionResponse(
            predicted_class=predicted_class_name,
            confidence=confidence,
            all_predictions=all_predictions_dict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)