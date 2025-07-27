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
