import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np  
import glob  
from pathlib import Path  
  
def moving_average(interval, window_size):  
    """Función para suavizar datos con media móvil"""  
    if window_size == 1:  
        return interval  
    window = np.ones(int(window_size)) / float(window_size)  
    return np.convolve(interval, window, "same")  
  
def analyze_ray_results():  
    """Analizar resultados de Ray Tune"""  
    try:  
        # Método alternativo: leer directamente los archivos JSON  
        import json  
        import os  
          
        results_dir = "/workspace/ray_results/PPO_Grid4x4_CNN"  
          
        # Buscar archivos de resultados  
        result_files = []  
        for root, dirs, files in os.walk(results_dir):  
            for file in files:  
                if file == "result.json":  
                    result_files.append(os.path.join(root, file))  
          
        if not result_files:  
            print("No se encontraron archivos de resultados de Ray")  
            return  
          
        # Leer y combinar datos  
        all_data = []  
        for file_path in result_files:  
            with open(file_path, 'r') as f:  
                for line in f:  
                    try:  
                        data = json.loads(line)  
                        all_data.append(data)  
                    except json.JSONDecodeError:  
                        continue  
          
        if not all_data:  
            print("No se pudieron cargar datos de Ray")  
            return  
          
        # Convertir a DataFrame  
        df = pd.DataFrame(all_data)  
          
        # Crear gráficos  
        plt.figure(figsize=(15, 10))  
          
        # Subplot 1: Recompensa por episodio  
        plt.subplot(2, 3, 1)  
        if 'episode_reward_mean' in df.columns:  
            plt.plot(df['training_iteration'], df['episode_reward_mean'])  
            plt.title('Recompensa Media por Episodio')  
            plt.xlabel('Iteración')  
            plt.ylabel('Recompensa Media')  
          
        # Subplot 2: Longitud de episodio  
        plt.subplot(2, 3, 2)  
        if 'episode_len_mean' in df.columns:  
            plt.plot(df['training_iteration'], df['episode_len_mean'])  
            plt.title('Longitud Media de Episodio')  
            plt.xlabel('Iteración')  
            plt.ylabel('Pasos')  
          
        # Subplot 3: Timesteps totales  
        plt.subplot(2, 3, 3)  
        if 'timesteps_total' in df.columns:  
            plt.plot(df['training_iteration'], df['timesteps_total'])  
            plt.title('Timesteps Totales')  
            plt.xlabel('Iteración')  
            plt.ylabel('Timesteps')  
          
        # Subplot 4: Learning rate  
        plt.subplot(2, 3, 4)  
        if 'info' in df.columns and 'learner' in df.iloc[0].get('info', {}):  
            lr_data = [row.get('info', {}).get('learner', {}).get('default_policy', {}).get('cur_lr', None)   
                      for _, row in df.iterrows()]  
            lr_data = [x for x in lr_data if x is not None]  
            if lr_data:  
                plt.plot(range(len(lr_data)), lr_data)  
                plt.title('Learning Rate')  
                plt.xlabel('Iteración')  
                plt.ylabel('LR')  
          
        # Subplot 5: Policy loss  
        plt.subplot(2, 3, 5)  
        if 'info' in df.columns:  
            policy_loss = [row.get('info', {}).get('learner', {}).get('default_policy', {}).get('learner_stats', {}).get('policy_loss', None)   
                          for _, row in df.iterrows()]  
            policy_loss = [x for x in policy_loss if x is not None]  
            if policy_loss:  
                plt.plot(range(len(policy_loss)), policy_loss)  
                plt.title('Policy Loss')  
                plt.xlabel('Iteración')  
                plt.ylabel('Loss')  
          
        plt.tight_layout()  
        plt.savefig('ray_training_metrics.png', dpi=300, bbox_inches='tight')  
        plt.close()  
        print("Gráfico de métricas de Ray guardado como 'ray_training_metrics.png'")  
          
    except Exception as e:  
        print(f"Error al analizar resultados de Ray: {e}")  
  
def analyze_sumo_csv():  
    """Analizar CSVs de SUMO-RL"""  
    # Buscar archivos CSV de SUMO  
    csv_files = glob.glob("outputs/grid4x4/ppo_advanced*.csv")  
      
    if not csv_files:  
        print("No se encontraron archivos CSV de SUMO")  
        return  
      
    print(f"Encontrados {len(csv_files)} archivos CSV de SUMO")  
      
    # Combinar todos los CSVs  
    main_df = pd.DataFrame()  
    for file in csv_files:  
        try:  
            df = pd.read_csv(file)  
            if main_df.empty:  
                main_df = df  
            else:  
                main_df = pd.concat([main_df, df], ignore_index=True)  
        except Exception as e:  
            print(f"Error leyendo {file}: {e}")  
      
    if main_df.empty:  
        print("No se pudieron cargar datos de SUMO")  
        return  
      
    # Crear gráficos de métricas de tráfico  
    plt.figure(figsize=(15, 10))  
      
    # Subplot 1: Tiempo total de espera  
    plt.subplot(2, 3, 1)  
    if 'system_total_waiting_time' in main_df.columns:  
        plt.plot(main_df['step'], main_df['system_total_waiting_time'])  
        plt.title('Tiempo Total de Espera')  
        plt.xlabel('Paso de Simulación')  
        plt.ylabel('Tiempo de Espera (s)')  
      
    # Subplot 2: Velocidad media del sistema  
    plt.subplot(2, 3, 2)  
    if 'system_mean_speed' in main_df.columns:  
        plt.plot(main_df['step'], main_df['system_mean_speed'])  
        plt.title('Velocidad Media del Sistema')  
        plt.xlabel('Paso de Simulación')  
        plt.ylabel('Velocidad (m/s)')  
      
    # Subplot 3: Vehículos detenidos  
    plt.subplot(2, 3, 3)  
    if 'system_total_stopped' in main_df.columns:  
        plt.plot(main_df['step'], main_df['system_total_stopped'])  
        plt.title('Vehículos Detenidos')  
        plt.xlabel('Paso de Simulación')  
        plt.ylabel('Número de Vehículos')  
      
    # Subplot 4: Vehículos en movimiento  
    plt.subplot(2, 3, 4)  
    if 'system_total_running' in main_df.columns:  
        plt.plot(main_df['step'], main_df['system_total_running'])  
        plt.title('Vehículos en Movimiento')  
        plt.xlabel('Paso de Simulación')  
        plt.ylabel('Número de Vehículos')  
      
    # Subplot 5: Tiempo medio de espera  
    plt.subplot(2, 3, 5)  
    if 'system_mean_waiting_time' in main_df.columns:  
        plt.plot(main_df['step'], main_df['system_mean_waiting_time'])  
        plt.title('Tiempo Medio de Espera')  
        plt.xlabel('Paso de Simulación')  
        plt.ylabel('Tiempo de Espera (s)')  
      
    plt.tight_layout()  
    plt.savefig('sumo_traffic_metrics.png', dpi=300, bbox_inches='tight')  
    plt.close()  
    print("Gráfico de métricas de SUMO guardado como 'sumo_traffic_metrics.png'")  
  
if __name__ == "__main__":  
    print("Analizando métricas de entrenamiento...")  
    analyze_ray_results()  
    analyze_sumo_csv()  
    print("Análisis completado!")