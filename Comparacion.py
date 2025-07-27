import os  
import sys  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from pathlib import Path  
import ray  
from ray.rllib.algorithms.algorithm import Algorithm  
from ray.tune.registry import register_env  
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv  
import supersuit as ss  
  
# Ensure SUMO_HOME is set  
if "SUMO_HOME" in os.environ:  
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")  
    sys.path.append(tools)  
else:  
    sys.exit("Please declare the environment variable 'SUMO_HOME'")  
  
import sumo_rl  
  
# --- Configuration ---  
USE_GUI = False  
CHECKPOINT_PATH = "/workspace/ray_results/PPO_Grid4x4_CNN/PPO_grid3x3_69bf3_00000_0_2025-07-26_21-42-22/checkpoint_004999"  
COMPARISON_DIR = "comparacion"  
SIMULATION_TIME = 550  # 1 hora de simulación  
DELTA_TIME = 5  # Métricas cada 5 segundos  
  
def create_comparison_directory():  
    """Crea el directorio de comparación"""  
    os.makedirs(COMPARISON_DIR, exist_ok=True)  
    return COMPARISON_DIR  
  
def create_traditional_env():  
    """Crea el entorno SUMO-RL con semáforos tradicionales"""  
    env = sumo_rl.SumoEnvironment(  
        net_file="grid3x3.net.xml",  
        route_file="rutas_nuevas1.rou.xml",  
        use_gui=USE_GUI,  
        num_seconds=SIMULATION_TIME,  
        delta_time=DELTA_TIME,  
        begin_time=0,  
        time_to_teleport=300,  
        add_system_info=True,  
        add_per_agent_info=True,  
        fixed_ts=True,  # Semáforos de tiempo fijo  
        single_agent=False  
    )  
    return env  
  
def create_ai_env():  
    """Crea el entorno SUMO-RL para el modelo de IA"""  
    def advanced_reward_function(traffic_signal):  
        waiting_time_reward = traffic_signal._diff_waiting_time_reward()  
        speed_reward = traffic_signal._average_speed_reward() * 0.1  
        queue_penalty = traffic_signal._queue_reward() * 0.05  
        pressure_reward = traffic_signal._pressure_reward() * 0.02  
        phase_stability_bonus = 0.1 if traffic_signal.time_since_last_phase_change > traffic_signal.min_green else -0.05  
        return waiting_time_reward + speed_reward + queue_penalty + pressure_reward + phase_stability_bonus  
      
    env = sumo_rl.parallel_env(  
        net_file="grid3x3.net.xml",  
        route_file="rutas_nuevas.rou.xml",  
        use_gui=USE_GUI,  
        reward_fn=advanced_reward_function,  
        num_seconds=SIMULATION_TIME,  
        delta_time=DELTA_TIME,  
        begin_time=0,  
        time_to_teleport=300,  
        add_system_info=True,  
        add_per_agent_info=True,  
    )  
    env = ss.pad_observations_v0(env)  
    env = ss.frame_stack_v1(env, 3)  
    return env  
  
def collect_step_metrics(info_or_infos, simulation_time):  
    """Extrae métricas del paso actual con tiempo de simulación"""  
    metrics = {}  
      
    if isinstance(info_or_infos, dict):  
        if 'system_mean_waiting_time' in info_or_infos:  
            # Caso tradicional: info directo  
            info = info_or_infos  
        else:  
            # Caso IA: buscar en cualquier agente  
            info = None  
            for agent_id, agent_info in info_or_infos.items():  
                if 'system_mean_waiting_time' in agent_info:  
                    info = agent_info  
                    break  
          
        if info:  
            metrics = {  
                'simulation_time': simulation_time,  
                'system_mean_waiting_time': info.get('system_mean_waiting_time', 0),  
                'system_mean_speed': info.get('system_mean_speed', 0),  
                'system_total_stopped': info.get('system_total_stopped', 0),  
                'system_total_running': info.get('system_total_running', 0),  
                'system_total_arrived': info.get('system_total_arrived', 0),  
                'system_total_departed': info.get('system_total_departed', 0),  
                'system_total_waiting_time': info.get('system_total_waiting_time', 0),  
            }  
      
    return metrics  
  
def run_traditional_evaluation():  
    """Ejecuta una simulación completa con semáforos tradicionales"""  
    print("Evaluando semáforos tradicionales...")  
      
    env = create_traditional_env()  
    all_metrics = []  
      
    observations = env.reset()  
    done = {"__all__": False}  
      
    while not done["__all__"]:  
        observations, rewards, done, info = env.step({})  
          
        # Obtener tiempo actual de simulación  
        current_time = env.sim_step  
          
        step_metrics = collect_step_metrics(info, current_time)  
        if step_metrics:  
            step_metrics['model_type'] = 'Tradicional'  
            all_metrics.append(step_metrics)  
          
        # Mostrar progreso cada 300 segundos (5 minutos)  
        if int(current_time) % 300 == 0 and current_time > 0:  
            print(f"  Progreso tradicional: {current_time:.0f}s / {SIMULATION_TIME}s")  
      
    env.close()  
    print(f"✓ Simulación tradicional completada: {len(all_metrics)} puntos de datos")  
    return all_metrics  
  
def run_ai_evaluation():  
    """Ejecuta una simulación completa con el modelo de IA"""  
    print("Evaluando modelo de IA...")  
      
    ray.init(ignore_reinit_error=True)  
      
    register_env("grid3x3", lambda _: ParallelPettingZooEnv(create_ai_env()))  
    algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)  
    env = create_ai_env()  
      
    all_metrics = []  
      
    reset_result = env.reset()  
    if isinstance(reset_result, tuple):  
        obs_dict, infos = reset_result  
    else:  
        obs_dict = reset_result  
      
    done = False  
      
    while not done:  
        actions = {}  
        for agent_id, agent_obs in obs_dict.items():  
            action = algo.compute_single_action(  
                observation=agent_obs,  
                policy_id="default_policy",  
            )  
            actions[agent_id] = action  
          
        obs_dict, rewards, terminations, truncations, infos = env.step(actions)  
          
        # Obtener tiempo actual de simulación  
        current_time = env.unwrapped.env.sim_step  
          
        step_metrics = collect_step_metrics(infos, current_time)  
        if step_metrics:  
            step_metrics['model_type'] = 'IA (PPO)'  
            all_metrics.append(step_metrics)  
          
        # Mostrar progreso cada 300 segundos (5 minutos)  
        if int(current_time) % 300 == 0 and current_time > 0:  
            print(f"  Progreso IA: {current_time:.0f}s / {SIMULATION_TIME}s")  
          
        done = all(terminations.values()) or all(truncations.values())  
      
    env.close()  
    ray.shutdown()  
    print(f"✓ Simulación IA completada: {len(all_metrics)} puntos de datos")  
    return all_metrics  
  
def generate_temporal_metric_plots(df_combined, output_dir):  
    """Genera PNGs individuales para cada métrica a lo largo del tiempo"""  
      
    # Preparar datos por modelo  
    df_traditional = df_combined[df_combined['model_type'] == 'Tradicional']  
    df_ai = df_combined[df_combined['model_type'] == 'IA (PPO)']  
      
    # Convertir tiempo de simulación a minutos para mejor visualización  
    df_traditional['time_minutes'] = df_traditional['simulation_time'] / 60  
    df_ai['time_minutes'] = df_ai['simulation_time'] / 60  
      
    # 1. Tiempo de Espera Promedio a lo largo del tiempo  
    plt.figure(figsize=(12, 6))  
    plt.plot(df_traditional['time_minutes'], df_traditional['system_mean_waiting_time'],   
             'r-', linewidth=2, label='Semáforos Tradicionales', alpha=0.8)  
    plt.plot(df_ai['time_minutes'], df_ai['system_mean_waiting_time'],   
             'b-', linewidth=2, label='Modelo IA (PPO)', alpha=0.8)  
    plt.xlabel('Tiempo de Simulación (minutos)')  
    plt.ylabel('Tiempo de Espera Promedio (s)')  
    plt.title('Evolución del Tiempo de Espera Promedio')  
    plt.legend()  
    plt.grid(True, alpha=0.3)  
    plt.tight_layout()  
    plt.savefig(os.path.join(output_dir, '01_tiempo_espera_temporal.pdf'), dpi=300, bbox_inches='tight')  
    plt.close()  
      
    # 2. Vehículos Detenidos a lo largo del tiempo  
    plt.figure(figsize=(12, 6))  
    plt.plot(df_traditional['time_minutes'], df_traditional['system_total_stopped'],   
             'r-', linewidth=2, label='Semáforos Tradicionales', alpha=0.8)  
    plt.plot(df_ai['time_minutes'], df_ai['system_total_stopped'],   
             'b-', linewidth=2, label='Modelo IA (PPO)', alpha=0.8)  
    plt.xlabel('Tiempo de Simulación (minutos)')  
    plt.ylabel('Vehículos Detenidos')  
    plt.title('Evolución de Vehículos Detenidos (Longitud de Cola)')  
    plt.legend()  
    plt.grid(True, alpha=0.3)  
    plt.tight_layout()  
    plt.savefig(os.path.join(output_dir, '02_vehiculos_detenidos_temporal.pdf'), dpi=300, bbox_inches='tight')  
    plt.close()  
      
    # 3. Velocidad Promedio a lo largo del tiempo  
    plt.figure(figsize=(12, 6))  
    plt.plot(df_traditional['time_minutes'], df_traditional['system_mean_speed'],   
             'r-', linewidth=2, label='Semáforos Tradicionales', alpha=0.8)  
    plt.plot(df_ai['time_minutes'], df_ai['system_mean_speed'],   
             'b-', linewidth=2, label='Modelo IA (PPO)', alpha=0.8)  
    plt.xlabel('Tiempo de Simulación (minutos)')  
    plt.ylabel('Velocidad Promedio (m/s)')  
    plt.title('Evolución de la Velocidad Promedio')  
    plt.legend()  
    plt.grid(True, alpha=0.3)  
    plt.tight_layout()  
    plt.savefig(os.path.join(output_dir, '03_velocidad_temporal.pdf'), dpi=300, bbox_inches='tight')  
    plt.close()  
      
    # 4. Throughput acumulativo a lo largo del tiempo  
    plt.figure(figsize=(12, 6))  
    plt.plot(df_traditional['time_minutes'], df_traditional['system_total_arrived'],   
             'r-', linewidth=2, label='Semáforos Tradicionales', alpha=0.8)  
    plt.plot(df_ai['time_minutes'], df_ai['system_total_arrived'],   
             'b-', linewidth=2, label='Modelo IA (PPO)', alpha=0.8)  
    plt.xlabel('Tiempo de Simulación (minutos)')  
    plt.ylabel('Vehículos Procesados (Acumulativo)')  
    plt.title('Evolución del Throughput del Sistema')  
    plt.legend()  
    plt.grid(True, alpha=0.3)  
    plt.tight_layout()  
    plt.savefig(os.path.join(output_dir, '04_throughput_temporal.pdf'), dpi=300, bbox_inches='tight')  
    plt.close()  
      
    # 5. Comparación general en subplots  
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  
    fig.suptitle('Comparación Temporal: IA vs Semáforos Tradicionales', fontsize=16, fontweight='bold')  
      
    # Tiempo de espera  
    axes[0, 0].plot(df_traditional['time_minutes'], df_traditional['system_mean_waiting_time'],   
                    'r-', linewidth=2, label='Tradicional', alpha=0.8)  
    axes[0, 0].plot(df_ai['time_minutes'], df_ai['system_mean_waiting_time'],   
                    'b-', linewidth=2, label='IA (PPO)', alpha=0.8)  
    axes[0, 0].set_title('Tiempo de Espera Promedio')  
    axes[0, 0].set_ylabel('Tiempo (s)')  
    axes[0, 0].legend()  
    axes[0, 0].grid(True, alpha=0.3)  
      
    # Vehículos detenidos  
    axes[0, 1].plot(df_traditional['time_minutes'], df_traditional['system_total_stopped'],   
                    'r-', linewidth=2, label='Tradicional', alpha=0.8)  
    axes[0, 1].plot(df_ai['time_minutes'], df_ai['system_total_stopped'],   
                    'b-', linewidth=2, label='IA (PPO)', alpha=0.8)  
    axes[0, 1].set_title('Vehículos Detenidos')  
    axes[0, 1].set_ylabel('Cantidad')  
    axes[0, 1].legend()  
    axes[0, 1].grid(True, alpha=0.3)  
      
    # Velocidad  
    axes[1, 0].plot(df_traditional['time_minutes'], df_traditional['system_mean_speed'],   
                    'r-', linewidth=2, label='Tradicional', alpha=0.8)  
    axes[1, 0].plot(df_ai['time_minutes'], df_ai['system_mean_speed'],   
                    'b-', linewidth=2, label='IA (PPO)', alpha=0.8)  
    axes[1, 0].set_title('Velocidad Promedio')  
    axes[1, 0].set_xlabel('Tiempo (minutos)')  
    axes[1, 0].set_ylabel('Velocidad (m/s)')  
    axes[1, 0].legend()  
    axes[1, 0].grid(True, alpha=0.3)  
      
    # Throughput  
    axes[1, 1].plot(df_traditional['time_minutes'], df_traditional['system_total_arrived'],   
                    'r-', linewidth=2, label='Tradicional', alpha=0.8)  
    axes[1, 1].plot(df_ai['time_minutes'], df_ai['system_total_arrived'],   
                    'b-', linewidth=2, label='IA (PPO)', alpha=0.8)  
    axes[1, 1].set_title('Throughput Acumulativo')  
    axes[1, 1].set_xlabel('Tiempo (minutos)')  
    axes[1, 1].set_ylabel('Vehículos Procesados')  
    axes[1, 1].legend()  
    axes[1, 1].grid(True, alpha=0.3)  
      
    plt.tight_layout()  
    plt.savefig(os.path.join(output_dir, '05_comparacion_temporal_general.pdf'), dpi=300, bbox_inches='tight')  
    plt.close()  
      
    print(f"✓ Gráficas temporales guardadas en: {output_dir}")  
  
def main():  
    """Función principal que ejecuta la comparación temporal completa"""  
    print("=== INICIANDO COMPARACIÓN TEMPORAL: IA vs SEMÁFOROS TRADICIONALES ===")  
      
    # Crear directorio de comparación  
    output_dir = create_comparison_directory()  
      
    # Ejecutar evaluaciones  
    print("\n1. Ejecutando simulación con semáforos tradicionales...")  
    traditional_metrics = run_traditional_evaluation()  
      
    print("\n2. Ejecutando simulación con modelo IA...")  
    ai_metrics = run_ai_evaluation()  
      
    # Combinar datos  
    print("\n3. Procesando y generando comparaciones temporales...")  
    all_metrics = traditional_metrics + ai_metrics  
    df_combined = pd.DataFrame(all_metrics)  
      
    # Guardar datos combinados  
    combined_file = os.path.join(output_dir, 'metricas_temporales_combinadas.csv')  
    df_combined.to_csv(combined_file, index=False)  
    print(f"✓ Datos temporales guardados en: {combined_file}")  
      
    # Generar gráficas temporales  
    generate_temporal_metric_plots(df_combined, output_dir)  
      
    # Mostrar estadísticas finales  
    print("\n=== RESUMEN DE COMPARACIÓN TEMPORAL ===")  
    traditional_data = df_combined[df_combined['model_type'] == 'Tradicional']  
    ai_data = df_combined[df_combined['model_type'] == 'IA (PPO)']  
      
    print(f"Tiempo de espera promedio:")  
    print(f"  - Tradicional: {traditional_data['system_mean_waiting_time'].mean():.2f}s")  
    print(f"  - IA (PPO): {ai_data['system_mean_waiting_time'].mean():.2f}s")  
      
    print(f"Velocidad promedio:")  
    print(f"  - Tradicional: {traditional_data['system_mean_speed'].mean():.2f} m/s")  
    print(f"  - IA (PPO): {ai_data['system_mean_speed'].mean():.2f} m/s")  
      
    print(f"Vehículos detenidos promedio:")  
    print(f"  - Tradicional: {traditional_data['system_total_stopped'].mean():.1f}")  
    print(f"  - IA (PPO): {ai_data['system_total_stopped'].mean():.1f}")  
      
    print(f"Throughput final:")  
    print(f"  - Tradicional: {traditional_data['system_total_arrived'].iloc[-1]:.0f} vehículos")  
    print(f"  - IA (PPO): {ai_data['system_total_arrived'].iloc[-1]:.0f} vehículos")  
      
    print(f"\n✓ Comparación temporal completa. Revisa la carpeta '{output_dir}' para ver todos los PNGs generados.")  
  
if __name__ == "__main__":  
    main()
    
    