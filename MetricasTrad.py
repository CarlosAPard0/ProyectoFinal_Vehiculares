import os      
import sys      
import numpy as np      
import pandas as pd      
import matplotlib.pyplot as plt    
from pathlib import Path    
    
# Ensure SUMO_HOME is set      
if "SUMO_HOME" in os.environ:      
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")    
    sys.path.append(tools)      
else:      
    sys.exit("Please declare the environment variable 'SUMO_HOME'")      
      
import sumo_rl      
      
# --- Configuration ---      
NUM_EPISODES = 10      
USE_GUI = False      
#OUTPUT_CSV_PATH = "outputs/traditional_signals_results"      
OUTPUT_CSV_PATH = "outputs/evaluation_results_trad"
      
def create_traditional_env():      
    """Crea el entorno SUMO-RL con semáforos tradicionales (tiempo fijo)"""      
    os.makedirs(OUTPUT_CSV_PATH, exist_ok=True)      
          
    env = sumo_rl.SumoEnvironment(  # Usar SumoEnvironment directamente, no parallel_env  
        net_file="grid3x3.net.xml",      
        route_file="rutas_nuevas.rou.xml",      
        out_csv_name=os.path.join(OUTPUT_CSV_PATH, "traditional_evaluation"),  
        use_gui=USE_GUI,      
        num_seconds=360,      
        begin_time=0,      
        time_to_teleport=300,    
        add_system_info=True,  # Habilitar métricas del sistema    
        add_per_agent_info=True,  # Habilitar métricas por agente  
        fixed_ts=True,  # CLAVE: Usar semáforos de tiempo fijo  
        single_agent=False  # Mantener multi-agente para consistencia con PPO  
    )      
    return env      
  
def collect_step_metrics(info):    
    """Extrae métricas del paso actual del info del entorno"""    
    if isinstance(info, dict) and 'system_mean_waiting_time' in info:  
        return {    
            'step': info.get('step', 0),    
            'system_mean_waiting_time': info.get('system_mean_waiting_time', 0),    
            'system_mean_speed': info.get('system_mean_speed', 0),    
            'system_total_stopped': info.get('system_total_stopped', 0),    
            'system_total_running': info.get('system_total_running', 0),    
            'system_total_arrived': info.get('system_total_arrived', 0),    
            'system_total_departed': info.get('system_total_departed', 0),    
            'system_total_waiting_time': info.get('system_total_waiting_time', 0),    
        }  
    return {}  
  
def run_traditional_evaluation():      
    """Ejecuta la evaluación con semáforos tradicionales"""      
    print(f"Starting traditional traffic signals evaluation for {NUM_EPISODES} episodes...")      
      
    env = create_traditional_env()      
      
    episode_rewards = []    
    all_episode_metrics = []    
      
    for episode in range(NUM_EPISODES):      
        print(f"--- Episode {episode + 1}/{NUM_EPISODES} ---")      
              
        # Reset del entorno  
        observations = env.reset()  
              
        done = {"__all__": False}  
        total_episode_reward = 0.0    
        episode_metrics = []    
        step_count = 0    
              
        while not done["__all__"]:      
            # Con fixed_ts=True, no necesitamos proporcionar acciones  
            # El entorno seguirá automáticamente los tiempos fijos definidos  
            observations, rewards, done, info = env.step({})  
                  
            # Recopilar métricas del paso  
            step_metrics = collect_step_metrics(info)  
            if step_metrics:  
                step_metrics['episode'] = episode + 1  
                step_metrics['step_in_episode'] = step_count  
                step_metrics['total_reward'] = sum(rewards.values()) if isinstance(rewards, dict) else 0  
                episode_metrics.append(step_metrics)  
                
            # Acumular recompensas (aunque no son relevantes para semáforos fijos)  
            if isinstance(rewards, dict):  
                total_episode_reward += sum(rewards.values())  
            step_count += 1  
      
        episode_rewards.append(total_episode_reward)    
        all_episode_metrics.extend(episode_metrics)    
        print(f"Episode {episode + 1} finished. Total Reward: {total_episode_reward:.2f}")    
            
        # Guardar CSV para este episodio  
        env.save_csv(os.path.join(OUTPUT_CSV_PATH, "traditional_evaluation"), episode + 1)    
      
    env.close()    
        
    # Guardar métricas agregadas    
    save_aggregated_metrics(all_episode_metrics, episode_rewards)    
        
    return episode_rewards, all_episode_metrics  
  
def save_aggregated_metrics(all_metrics, episode_rewards):    
    """Guarda métricas agregadas y genera visualizaciones"""    
        
    # Crear DataFrame con todas las métricas    
    df_metrics = pd.DataFrame(all_metrics)    
        
    # Guardar CSV agregado    
    metrics_file = os.path.join(OUTPUT_CSV_PATH, "traditional_evaluation_aggregated.csv")    
    df_metrics.to_csv(metrics_file, index=False)    
    print(f"✓ Métricas agregadas guardadas en: {metrics_file}")    
        
    # Guardar resumen por episodio    
    episode_summary = []    
    for episode in range(1, NUM_EPISODES + 1):    
        episode_data = df_metrics[df_metrics['episode'] == episode]    
        if not episode_data.empty:    
            summary = {    
                'episode': episode,    
                'total_reward': episode_rewards[episode - 1],    
                'mean_waiting_time': episode_data['system_mean_waiting_time'].mean(),    
                'mean_speed': episode_data['system_mean_speed'].mean(),    
                'total_stopped_avg': episode_data['system_total_stopped'].mean(),    
                'total_arrived': episode_data['system_total_arrived'].iloc[-1],    
                'total_departed': episode_data['system_total_departed'].iloc[-1],    
            }    
            episode_summary.append(summary)    
        
    df_summary = pd.DataFrame(episode_summary)    
    summary_file = os.path.join(OUTPUT_CSV_PATH, "traditional_evaluation_episode_summary.csv")    
    df_summary.to_csv(summary_file, index=False)    
    print(f"✓ Resumen por episodio guardado en: {summary_file}")    
        
    # Generar visualizaciones    
    generate_performance_plots(df_metrics, df_summary)    
  
def generate_performance_plots(df_metrics, df_summary):    
    """Genera gráficas PNG del desempeño"""    
        
    plt.style.use('default')    
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))    
    fig.suptitle('Traditional Traffic Signals Performance Evaluation', fontsize=16, fontweight='bold')    
        
    # Gráfica 1: Tiempo de espera promedio por episodio    
    axes[0, 0].plot(df_summary['episode'], df_summary['mean_waiting_time'],     
                    'r-o', linewidth=2, markersize=6)    
    axes[0, 0].set_xlabel('Episode')    
    axes[0, 0].set_ylabel('Mean Waiting Time (s)')    
    axes[0, 0].set_title('Average Waiting Time per Episode')    
    axes[0, 0].grid(True, alpha=0.3)    
        
    # Gráfica 2: Velocidad promedio por episodio    
    axes[0, 1].plot(df_summary['episode'], df_summary['mean_speed'],     
                    'g-o', linewidth=2, markersize=6)    
    axes[0, 1].set_xlabel('Episode')    
    axes[0, 1].set_ylabel('Mean Speed (m/s)')    
    axes[0, 1].set_title('Average Speed per Episode')    
    axes[0, 1].grid(True, alpha=0.3)    
        
    # Gráfica 3: Vehículos detenidos promedio por episodio    
    axes[0, 2].plot(df_summary['episode'], df_summary['total_stopped_avg'],     
                    'orange', linewidth=2, markersize=6)    
    axes[0, 2].set_xlabel('Episode')    
    axes[0, 2].set_ylabel('Average Stopped Vehicles')    
    axes[0, 2].set_title('Average Stopped Vehicles per Episode')    
    axes[0, 2].grid(True, alpha=0.3)    
        
    # Gráfica 4: Evolución temporal del tiempo de espera    
    df_avg_by_step = df_metrics.groupby('step_in_episode').agg({    
        'system_mean_waiting_time': 'mean',    
        'system_mean_speed': 'mean',    
        'system_total_stopped': 'mean'    
    }).reset_index()    
        
    axes[1, 0].plot(df_avg_by_step['step_in_episode'], df_avg_by_step['system_mean_waiting_time'],     
                    'r-', linewidth=2)    
    axes[1, 0].set_xlabel('Step in Episode')    
    axes[1, 0].set_ylabel('Mean Waiting Time (s)')    
    axes[1, 0].set_title('Waiting Time Evolution (Avg across episodes)')    
    axes[1, 0].grid(True, alpha=0.3)    
        
    # Gráfica 5: Evolución temporal de la velocidad    
    axes[1, 1].plot(df_avg_by_step['step_in_episode'], df_avg_by_step['system_mean_speed'],     
                    'g-', linewidth=2)    
    axes[1, 1].set_xlabel('Step in Episode')    
    axes[1, 1].set_ylabel('Mean Speed (m/s)')    
    axes[1, 1].set_title('Speed Evolution (Avg across episodes)')    
    axes[1, 1].grid(True, alpha=0.3)    
        
    # Gráfica 6: Vehículos detenidos    
    axes[1, 2].plot(df_avg_by_step['step_in_episode'], df_avg_by_step['system_total_stopped'],     
                    'orange', linewidth=2)    
    axes[1, 2].set_xlabel('Step in Episode')    
    axes[1, 2].set_ylabel('Total Stopped Vehicles')    
    axes[1, 2].set_title('Stopped Vehicles Evolution (Avg across episodes)')    
    axes[1, 2].grid(True, alpha=0.3)    
        
    plt.tight_layout()    
        
    # Guardar la imagen    
    plot_file = os.path.join(OUTPUT_CSV_PATH, "traditional_signals_performance.png")    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')    
    print(f"✓ Gráficas de desempeño guardadas en: {plot_file}")    
        
    # Mostrar estadísticas finales    
    print("\n=== ESTADÍSTICAS DE DESEMPEÑO - SEMÁFOROS TRADICIONALES ===")    
    print(f"Tiempo de espera promedio: {df_summary['mean_waiting_time'].mean():.2f} ± {df_summary['mean_waiting_time'].std():.2f} s")    
    print(f"Velocidad promedio: {df_summary['mean_speed'].mean():.2f} ± {df_summary['mean_speed'].std():.2f} m/s")    
    print(f"Vehículos detenidos promedio: {df_summary['total_stopped_avg'].mean():.1f} ± {df_summary['total_stopped_avg'].std():.1f}")    
        
    plt.show()    
  
if __name__ == "__main__":      
    all_rewards, all_metrics = run_traditional_evaluation()      
          
    print("\n--- Traditional Signals Evaluation Summary ---")      
    mean_reward = np.mean(all_rewards)      
    std_reward = np.std(all_rewards)      
    print(f"Number of episodes: {NUM_EPISODES}")      
    print(f"Mean episodic reward: {mean_reward:.2f}")      
    print(f"Standard deviation of reward: {std_reward:.2f}")      
          
    print("Traditional signals evaluation complete.")