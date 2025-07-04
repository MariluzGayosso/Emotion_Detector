# Usa una imagen base con Python 3.13
FROM python:3.13-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia requirements.txt primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el código del proyecto al contenedor
COPY . .

# Expone el puerto 8501 para Streamlit
EXPOSE 8501

# Comando para ejecutar la app de Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
