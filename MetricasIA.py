import os    
import sys    
import numpy as np    
import pandas as pd    
import ray    
from ray.rllib.algorithms.algorithm import Algorithm    
from ray.tune.registry import register_env    
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv    
import supersuit as ss  
import matplotlib.pyplot as plt  
from pathlib import Path  
  
# Ensure SUMO_HOME is set    
if "SUMO_HOME" in os.environ:    
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")  
    sys.path.append(tools)    
else:    
    sys.exit("Please declare the environment variable 'SUMO_HOME'")    
    
import sumo_rl    
    
# --- Evaluation Configuration ---    
#CHECKPOINT_PATH = "/workspace/ray_results/PPO_Grid4x4_Advanced/PPO_grid4x4_ddb0f_00000_0_2025-07-26_04-13-21/checkpoint_003121"    
CHECKPOINT_PATH = "/workspace/ray_results/PPO_Grid4x4_CNN/PPO_grid3x3_69bf3_00000_0_2025-07-26_21-42-22/checkpoint_004999"
NUM_EPISODES = 10    
USE_GUI = False    
OUTPUT_CSV_PATH = "outputs/evaluation_results"    
    
def recreate_env_from_config():    
    """    
    Recreates the SUMO-RL environment with the exact same configuration    
    and wrappers used during training.    
    """    
    def advanced_reward_function(traffic_signal):    
        waiting_time_reward = traffic_signal._diff_waiting_time_reward()    
        speed_reward = traffic_signal._average_speed_reward() * 0.1    
        queue_penalty = traffic_signal._queue_reward() * 0.05    
        pressure_reward = traffic_signal._pressure_reward() * 0.02    
        phase_stability_bonus = 0.1 if traffic_signal.time_since_last_phase_change > traffic_signal.min_green else -0.05    
        return waiting_time_reward + speed_reward + queue_penalty + pressure_reward + phase_stability_bonus    
    
    os.makedirs(OUTPUT_CSV_PATH, exist_ok=True)    
        
    env = sumo_rl.parallel_env(    
        net_file="grid3x3.net.xml",    
        route_file="rutas_nuevas.rou.xml",    
        out_csv_name=os.path.join(OUTPUT_CSV_PATH, "ppo_evaluation"),  # Cambiar nombre para identificar  
        use_gui=USE_GUI,    
        reward_fn=advanced_reward_function,    
        num_seconds=360,    
        begin_time=0,    
        time_to_teleport=300,  
        add_system_info=True,  # Habilitar métricas del sistema  
        add_per_agent_info=True,  # Habilitar métricas por agente  
    )    
    env = ss.pad_observations_v0(env)    
    env = ss.frame_stack_v1(env, 3)    
    return env    
  
register_env("grid3x3", lambda _: ParallelPettingZooEnv(recreate_env_from_config()))  
  
def collect_step_metrics(infos):  
    """Extrae métricas del paso actual"""  
    metrics = {}  
      
    # Buscar métricas del sistema en cualquier agente (todas tienen la misma info del sistema)  
    for agent_id, info in infos.items():  
        if 'system_mean_waiting_time' in info:  
            metrics.update({  
                'step': info.get('step', 0),  
                'system_mean_waiting_time': info.get('system_mean_waiting_time', 0),  
                'system_mean_speed': info.get('system_mean_speed', 0),  
                'system_total_stopped': info.get('system_total_stopped', 0),  
                'system_total_running': info.get('system_total_running', 0),  
                'system_total_arrived': info.get('system_total_arrived', 0),  
                'system_total_departed': info.get('system_total_departed', 0),  
                'system_total_waiting_time': info.get('system_total_waiting_time', 0),  
            })  
            break  
      
    return metrics  
  
def run_evaluation():    
    """    
    Loads the algorithm from a checkpoint and runs the evaluation loop.    
    """    
    print(f"Initializing Ray...")    
    ray.init(ignore_reinit_error=True)    
    
    print(f"Restoring algorithm from checkpoint: {CHECKPOINT_PATH}")    
    algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)    
        
    policy = algo.get_policy("default_policy")    
    env = recreate_env_from_config()    
    
    episode_rewards = []  
    all_episode_metrics = []  # Para almacenar métricas de todos los episodios  
    print(f"\nStarting evaluation for {NUM_EPISODES} episodes...")    
    
    for episode in range(NUM_EPISODES):    
        print(f"--- Episode {episode + 1}/{NUM_EPISODES} ---")    
            
        # Handle tuple return from env.reset() with SuperSuit wrappers    
        reset_result = env.reset()    
        if isinstance(reset_result, tuple):    
            obs_dict, infos = reset_result    
        else:    
            obs_dict = reset_result    
            
        # Initialize LSTM states for all agents    
        #initial_state = policy.get_initial_state()    
        #states = {agent_id: initial_state for agent_id in obs_dict.keys()}    
            
        done = False    
        total_episode_reward = 0.0  
        episode_metrics = []  # Métricas de este episodio  
        step_count = 0  
        while not done:      
            actions = {}      
            for agent_id, agent_obs in obs_dict.items():      
                # Para modelos CNN sin LSTM, solo necesitas la acción  
                action = algo.compute_single_action(      
                    observation=agent_obs,      
                    policy_id="default_policy",      
                )      
                actions[agent_id] = action      
    
            # Handle 5 return values from env.step()    
            obs_dict, rewards, terminations, truncations, infos = env.step(actions)    
                
            # Recopilar métricas del paso  
            step_metrics = collect_step_metrics(infos)  
            if step_metrics:  
                step_metrics['episode'] = episode + 1  
                step_metrics['step_in_episode'] = step_count  
                step_metrics['total_reward'] = sum(rewards.values())  
                episode_metrics.append(step_metrics)  
              
            # Check if episode is done    
            done = all(terminations.values()) or all(truncations.values())    
                
            total_episode_reward += sum(rewards.values())  
            step_count += 1  
    
        episode_rewards.append(total_episode_reward)  
        all_episode_metrics.extend(episode_metrics)  
        print(f"Episode {episode + 1} finished. Total Reward: {total_episode_reward:.2f}")  
          
        # Guardar CSV para este episodio usando el método nativo de SUMO-RL  
        env.unwrapped.env.save_csv(os.path.join(OUTPUT_CSV_PATH, "ppo_evaluation"), episode + 1)  
    
    env.close()  
      
    # Guardar métricas agregadas  
    save_aggregated_metrics(all_episode_metrics, episode_rewards)  
      
    #return episode_rewards, all_episode_metrics  
    return episode_rewards, all_episode_metrics  # En lugar de solo episode_rewards

  
def save_aggregated_metrics(all_metrics, episode_rewards):  
    """Guarda métricas agregadas y genera visualizaciones"""  
      
    # Crear DataFrame con todas las métricas  
    df_metrics = pd.DataFrame(all_metrics)  
      
    # Guardar CSV agregado  
    metrics_file = os.path.join(OUTPUT_CSV_PATH, "ppo_evaluation_aggregated.csv")  
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
                'total_arrived': episode_data['system_total_arrived'].iloc[-1],  # Valor final  
                'total_departed': episode_data['system_total_departed'].iloc[-1],  
            }  
            episode_summary.append(summary)  
      
    df_summary = pd.DataFrame(episode_summary)  
    summary_file = os.path.join(OUTPUT_CSV_PATH, "ppo_evaluation_episode_summary.csv")  
    df_summary.to_csv(summary_file, index=False)  
    print(f"✓ Resumen por episodio guardado en: {summary_file}")  
      
    # Generar visualizaciones  
    generate_performance_plots(df_metrics, df_summary)  
  
def generate_performance_plots(df_metrics, df_summary):  
    """Genera gráficas PNG del desempeño"""  
      
    # Configurar estilo  
    plt.style.use('default')  
      
    # Crear figura con múltiples subplots  
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  
    fig.suptitle('PPO Model Performance Evaluation', fontsize=16, fontweight='bold')  
      
    # Gráfica 1: Tiempo de espera promedio por episodio  
    axes[0, 0].plot(df_summary['episode'], df_summary['mean_waiting_time'],   
                    'b-o', linewidth=2, markersize=6)  
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
      
    # Gráfica 3: Recompensa total por episodio  
    axes[0, 2].plot(df_summary['episode'], df_summary['total_reward'],   
                    'r-o', linewidth=2, markersize=6)  
    axes[0, 2].set_xlabel('Episode')  
    axes[0, 2].set_ylabel('Total Reward')  
    axes[0, 2].set_title('Total Reward per Episode')  
    axes[0, 2].grid(True, alpha=0.3)  
      
    # Gráfica 4: Evolución temporal del tiempo de espera (promedio de todos los episodios)  
    df_avg_by_step = df_metrics.groupby('step_in_episode').agg({  
        'system_mean_waiting_time': 'mean',  
        'system_mean_speed': 'mean',  
        'system_total_stopped': 'mean'  
    }).reset_index()  
      
    axes[1, 0].plot(df_avg_by_step['step_in_episode'], df_avg_by_step['system_mean_waiting_time'],   
                    'b-', linewidth=2)  
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
    plot_file = os.path.join(OUTPUT_CSV_PATH, "ppo_evaluation_performance.png")  
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')  
    print(f"✓ Gráficas de desempeño guardadas en: {plot_file}")  
      
    # Mostrar estadísticas finales  
    print("\n=== ESTADÍSTICAS DE DESEMPEÑO ===")  
    print(f"Tiempo de espera promedio: {df_summary['mean_waiting_time'].mean():.2f} ± {df_summary['mean_waiting_time'].std():.2f} s")  
    print(f"Velocidad promedio: {df_summary['mean_speed'].mean():.2f} ± {df_summary['mean_speed'].std():.2f} m/s")  
    print(f"Recompensa promedio: {df_summary['total_reward'].mean():.2f} ± {df_summary['total_reward'].std():.2f}")  
    print(f"Vehículos detenidos promedio: {df_summary['total_stopped_avg'].mean():.1f} ± {df_summary['total_stopped_avg'].std():.1f}")  
      
    plt.show()  

if __name__ == "__main__":    
    all_rewards, all_metrics = run_evaluation()    
        
    print("\n--- Evaluation Summary ---")    
    mean_reward = np.mean(all_rewards)    
    std_reward = np.std(all_rewards)    
    print(f"Number of episodes: {NUM_EPISODES}")    
    print(f"Mean episodic reward: {mean_reward:.2f}")    
    print(f"Standard deviation of reward: {std_reward:.2f}")    
        
    ray.shutdown()    
    print("Evaluation complete.")