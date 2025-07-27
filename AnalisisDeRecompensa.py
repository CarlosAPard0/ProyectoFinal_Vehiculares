import json  
import pandas as pd  
import matplotlib.pyplot as plt  
from pathlib import Path  
import glob  
  
def extract_training_metrics(results_dir):  
    """Extrae métricas de entrenamiento de los archivos de Ray RLlib"""  
    results_path = Path(results_dir)  
    all_data = []  
      
    # Buscar todos los directorios de trials  
    for trial_dir in results_path.glob("PPO_*"):  
        result_file = trial_dir / "result.json"  
          
        if result_file.exists():  
            print(f"Procesando: {result_file}")  
              
            with open(result_file, 'r') as f:  
                for line in f:  
                    try:  
                        result = json.loads(line.strip())  
                          
                        # Extraer métricas clave  
                        data_point = {  
                            "trial_id": trial_dir.name,  
                            "training_iteration": result.get("training_iteration", 0),  
                            "episode_reward_mean": result.get("episode_reward_mean", 0),  
                            "episode_reward_max": result.get("episode_reward_max", 0),  
                            "episode_reward_min": result.get("episode_reward_min", 0),  
                            "timesteps_total": result.get("timesteps_total", 0),  
                            "episodes_total": result.get("episodes_total", 0),  
                            "time_total_s": result.get("time_total_s", 0)  
                        }  
                        all_data.append(data_point)  
                          
                    except json.JSONDecodeError:  
                        continue  
      
    return pd.DataFrame(all_data)  
  
def plot_reward_evolution(df, save_path=None):  
    """Grafica la evolución de la recompensa durante el entrenamiento"""  
    plt.figure(figsize=(12, 8))  
      
    # Si hay múltiples trials, graficar cada uno  
    if 'trial_id' in df.columns:  
        for trial_id in df['trial_id'].unique():  
            trial_data = df[df['trial_id'] == trial_id]  
            plt.plot(trial_data['training_iteration'],   
                    trial_data['episode_reward_mean'],   
                    label=f'Trial: {trial_id}', alpha=0.7)  
    else:  
        plt.plot(df['training_iteration'], df['episode_reward_mean'])  
      
    plt.xlabel('Iteraciones de Entrenamiento')  
    plt.ylabel('Recompensa Media por Episodio')  
    plt.title('Evolución de la Recompensa Durante el Entrenamiento PPO')  
    plt.grid(True, alpha=0.3)  
    plt.legend()  
      
    if save_path:  
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  
      
    plt.show()  
  
# Usar el script  
if __name__ == "__main__":  
    # Ruta a tus resultados  
    results_directory = "/workspace/ray_results/PPO_Grid4x4_CNN"  
      
    # Extraer datos  
    df = extract_training_metrics(results_directory)  
      
    if not df.empty:  
        print(f"Datos extraídos: {len(df)} puntos de datos")  
        print(f"Rango de iteraciones: {df['training_iteration'].min()} - {df['training_iteration'].max()}")  
        print(f"Rango de recompensas: {df['episode_reward_mean'].min():.3f} - {df['episode_reward_mean'].max():.3f}")  
          
        # Graficar evolución  
        plot_reward_evolution(df, save_path="reward_evolution.png")  
          
        # Guardar datos en CSV para análisis posterior  
        df.to_csv("training_metrics.csv", index=False)  
        print("Métricas guardadas en training_metrics.csv")  
          
    else:  
        print("No se encontraron datos de entrenamiento en el directorio especificado")