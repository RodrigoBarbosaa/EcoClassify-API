# API de Classificação de Imagens com FastAPI e Keras para o EcoClassify

Esta API permite classificar imagens usando um modelo de deep learning criado com Keras/TensorFlow. A API recebe imagens codificadas em base64 via JSON e retorna as predições do modelo.

## 🚀 Características

- **FastAPI**: Framework moderno e rápido para APIs
- **Keras/TensorFlow**: Suporte completo para modelos de deep learning
- **Processamento de Imagens**: Preprocessamento automático de imagens

## 📋 Requisitos

- Python 3.9+
- TensorFlow 2.15+
- FastAPI
- Pillow (processamento de imagens)

## 🛠 Instalação

### Método 1: Instalação Local

1. **Clone o repositório**:
```bash
git clone <seu-repositorio>
cd image-classifier-api
```

2. **Crie um ambiente virtual**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

4. **Execute a API**:
```bash
python main.py
# ou
uvicorn main:app --reload
```

## 📝 Uso da API

### Endpoints Disponíveis

#### 1. Health Check
```
GET /
```
Retorna o status da API e se o modelo está carregado.

#### 2. Informações do Modelo
```
GET /model/info
```
Retorna informações detalhadas sobre o modelo carregado.

#### 3. Classificação de Imagem
```
POST /predict
```

### 3. Teste Manual com curl
```bash
# Health check
curl -X GET "http://localhost:8000/"

# Informações do modelo
curl -X GET "http://localhost:8000/model/info"

# Predição 
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"image_data": "IMAGE_BASE64", "image_format": "PNG"}'
```

## 📊 Formato das Imagens

A API espera imagens em formato base64. As imagens são automaticamente:
- Convertidas para escala de cinza
- Redimensionadas para o formato esperado pelo modelo
- Normalizadas (valores 0-1)
