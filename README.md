# Proyecto Sumo-RL-Vehiculares

## Descripción

Este proyecto implementa un agente de control de semáforos basado en Aprendizaje por Refuerzo Profundo (Deep Reinforcement Learning) entrenado en un entorno de simulación de tráfico.

---

## Entrenamiento

El proceso de entrenamiento consta de dos pasos principales:

1. **Construcción y ejecución del contenedor Docker**

   El entrenamiento se realiza ejecutando el `Dockerfile` incluido en el repositorio, que prepara el entorno con todas las dependencias necesarias.

   Para construir y ejecutar el contenedor:

   ```bash
   docker build -t sumo-rl-entrenamiento .
   docker run --rm -it sumo-rl-entrenamiento
2. **Ejecución del script de entrenamiento**

Dentro del contenedor o entorno preparado, se debe ejecutar el script principal. Este script utiliza una arquitectura de red neuronal convolucional (CNN) para procesar el estado del tráfico y entrenar al agente.[1]

```bash
python EntrenamientoCNN.py
