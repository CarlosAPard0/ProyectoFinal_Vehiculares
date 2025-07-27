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
from scipy import stats  
  
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
COMPARISON_DIR = "comparacion_varios_episodios"  
SIMULATION_TIME = 560  
DELTA_TIME = 5  
NUM_SCENARIOS = 10  # 10 archivos de rutas diferentes  
  


# Configuración de matplotlib (con fuentes más grandes)
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 22,          # Aumentado de 12
    'axes.titlesize': 24,     # Aumentado de 14
    'axes.labelsize': 22,     # Aumentado de 12
    'xtick.labelsize': 20,    # Aumentado de 10
    'ytick.labelsize': 20,    # Aumentado de 10
    'legend.fontsize': 21,    # Aumentado de 11
    'lines.linewidth': 2,   # Un poco más grueso para la nueva escala
    'grid.alpha': 0.3,
    'axes.grid': True,
    'figure.autolayout': True
})
  
COLORS = {  
    'traditional': '#d62728',  
    'ai': '#1f77b4',  
}  
  
def moving_average(interval, window_size=5):  
    """Aplica promedio móvil para suavizar las líneas"""  
    if window_size == 1:  
        return interval  
    window = np.ones(int(window_size)) / float(window_size)  
    return np.convolve(interval, window, "same")  
  
def create_comparison_directory():  
    """Crea el directorio de comparación"""  
    os.makedirs(COMPARISON_DIR, exist_ok=True)  
    return COMPARISON_DIR  
  
def create_traditional_env(route_file):  
    """Crea el entorno SUMO-RL con semáforos tradicionales"""  
    env = sumo_rl.SumoEnvironment(  
        net_file="grid3x3.net.xml",  
        route_file=route_file,  
        use_gui=USE_GUI,  
        num_seconds=SIMULATION_TIME,  
        delta_time=DELTA_TIME,  
        begin_time=0,  
        time_to_teleport=300,  
        add_system_info=True,  
        add_per_agent_info=True,  
        fixed_ts=True,  
        single_agent=False  
    )  
    return env  
  
def create_ai_env(route_file):  
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
        route_file=route_file,  
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
            info = info_or_infos  
        else:  
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
  
def run_traditional_evaluation_all_scenarios():  
    """Ejecuta evaluación tradicional para todos los escenarios"""  
    print("Evaluando semáforos tradicionales en todos los escenarios...")  
      
    all_scenarios_metrics = []  
      
    for scenario in range(1, NUM_SCENARIOS + 1):  
        route_file = f"rutas_nuevas{scenario}.rou.xml"  
        print(f"  Escenario {scenario}: {route_file}")  
          
        env = create_traditional_env(route_file)  
        scenario_metrics = []  
          
        observations = env.reset()  
        done = {"__all__": False}  
          
        while not done["__all__"]:  
            observations, rewards, done, info = env.step({})  
              
            current_time = env.sim_step  
            step_metrics = collect_step_metrics(info, current_time)  
            if step_metrics:  
                step_metrics['model_type'] = 'Tradicional'  
                step_metrics['scenario'] = scenario  
                scenario_metrics.append(step_metrics)  
          
        env.close()  
        all_scenarios_metrics.extend(scenario_metrics)  
        print(f"    ✓ Escenario {scenario} completado: {len(scenario_metrics)} puntos")  
      
    return all_scenarios_metrics  
  
def run_ai_evaluation_all_scenarios():  
    """Ejecuta evaluación de IA para todos los escenarios"""  
    print("Evaluando modelo de IA en todos los escenarios...")  
      
    ray.init(ignore_reinit_error=True)  
      
    # CORREGIDO: Aplicar los mismos wrappers que en create_ai_env()  
    def create_registered_env(_):  
        def advanced_reward_function(traffic_signal):  
            waiting_time_reward = traffic_signal._diff_waiting_time_reward()  
            speed_reward = traffic_signal._average_speed_reward() * 0.1  
            queue_penalty = traffic_signal._queue_reward() * 0.05  
            pressure_reward = traffic_signal._pressure_reward() * 0.02  
            phase_stability_bonus = 0.1 if traffic_signal.time_since_last_phase_change > traffic_signal.min_green else -0.05  
            return waiting_time_reward + speed_reward + queue_penalty + pressure_reward + phase_stability_bonus  
          
        env = sumo_rl.parallel_env(  
            net_file="grid3x3.net.xml",  
            route_file="rutas_nuevas1.rou.xml",  # Usar el primer archivo como default  
            use_gui=USE_GUI,  
            reward_fn=advanced_reward_function,  
            num_seconds=SIMULATION_TIME,  
            delta_time=DELTA_TIME,  
            begin_time=0,  
            time_to_teleport=300,  
            add_system_info=True,  
            add_per_agent_info=True,  
        )  
        # IMPORTANTE: Aplicar los mismos wrappers  
        env = ss.pad_observations_v0(env)  
        env = ss.frame_stack_v1(env, 3)  
        return ParallelPettingZooEnv(env)  
      
    register_env("grid3x3", create_registered_env)  
      
    # Ahora cargar el algoritmo  
    algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)  
            
    all_scenarios_metrics = []  
      
    for scenario in range(1, NUM_SCENARIOS + 1):  
        route_file = f"rutas_nuevas{scenario}.rou.xml"  
        print(f"  Escenario {scenario}: {route_file}")  
          
        # Crear entorno específico para este escenario (sin registrar)  
        env = create_ai_env(route_file)  
        scenario_metrics = []  
          
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
              
            current_time = env.unwrapped.env.sim_step  
            step_metrics = collect_step_metrics(infos, current_time)  
            if step_metrics:  
                step_metrics['model_type'] = 'IA (PPO)'  
                step_metrics['scenario'] = scenario  
                scenario_metrics.append(step_metrics)  
              
            done = all(terminations.values()) or all(truncations.values())  
          
        env.close()  
        all_scenarios_metrics.extend(scenario_metrics)  
        print(f"    ✓ Escenario {scenario} completado: {len(scenario_metrics)} puntos")  
      
    ray.shutdown()  
    return all_scenarios_metrics  
def calculate_confidence_intervals(data, confidence=0.95):  
    """Calcula intervalos de confianza del 95%"""  
    n = len(data)  
    mean = np.mean(data)  
    std_err = stats.sem(data)  
    h = std_err * stats.t.ppf((1 + confidence) / 2., n-1)  
    return mean, mean - h, mean + h  
  
def generate_scenario_based_plots_with_ci(df_combined, output_dir):  
    """Genera gráficas basadas en promedios por escenario con intervalos de confianza del 95%"""  
      
    # Calcular promedios por escenario (esto es lo que quieres)  
    scenario_summaries = df_combined.groupby(['scenario', 'model_type']).agg({  
        'system_mean_waiting_time': 'mean',  
        'system_mean_speed': 'mean',   
        'system_total_stopped': 'mean',  
        'system_total_arrived': 'last'  # Usar el último valor (throughput final)  
    }).reset_index()  
      
    # Separar por tipo de modelo  
    trad_scenarios = scenario_summaries[scenario_summaries['model_type'] == 'Tradicional']  
    ai_scenarios = scenario_summaries[scenario_summaries['model_type'] == 'IA (PPO)']  
      
    # Función para calcular intervalos de confianza del 95%  
    def calculate_ci_95(data):  
        n = len(data)  
        mean = np.mean(data)  
        std_err = stats.sem(data)  
        ci = std_err * stats.t.ppf(0.975, n-1)  
        return mean, ci  
      
    # Calcular estadísticas para cada métrica  
    metrics = {  
        'waiting_time': {  
            'trad_values': trad_scenarios['system_mean_waiting_time'].values,  
            'ai_values': ai_scenarios['system_mean_waiting_time'].values,  
            'ylabel': 'Tiempo de Espera Promedio (s)',  
            'title': 'Tiempo de Espera Promedio por Escenario'  
        },  
        'speed': {  
            'trad_values': trad_scenarios['system_mean_speed'].values,  
            'ai_values': ai_scenarios['system_mean_speed'].values,  
            'ylabel': 'Velocidad Promedio (m/s)',  
            'title': 'Velocidad Promedio por Escenario'  
        },  
        'stopped': {  
            'trad_values': trad_scenarios['system_total_stopped'].values,  
            'ai_values': ai_scenarios['system_total_stopped'].values,  
            'ylabel': 'Vehículos Detenidos Promedio',  
            'title': 'Vehículos Detenidos Promedio por Escenario'  
        },  
        'throughput': {  
            'trad_values': trad_scenarios['system_total_arrived'].values,  
            'ai_values': ai_scenarios['system_total_arrived'].values,  
            'ylabel': 'Throughput Final (vehículos)',  
            'title': 'Throughput Final por Escenario'  
        }  
    }  
      
    # Generar gráficas individuales para cada métrica  
    for i, (metric_name, metric_data) in enumerate(metrics.items(), 1):  
        plt.figure(figsize=(12, 8))  
          
        # Calcular estadísticas  
        trad_mean, trad_ci = calculate_ci_95(metric_data['trad_values'])  
        ai_mean, ai_ci = calculate_ci_95(metric_data['ai_values'])  
          
        # Crear gráfica de barras con intervalos de confianza  
        x_pos = [0, 1]  
        means = [trad_mean, ai_mean]  
        cis = [trad_ci, ai_ci]  
        colors = [COLORS['traditional'], COLORS['ai']]  
        labels = ['Semáforos Tradicionales', 'Modelo IA (PPO)']  
          
        bars = plt.bar(x_pos, means, color=colors, alpha=0.7, width=0.6)  
        plt.errorbar(x_pos, means, yerr=cis, fmt='none', color='black',   
                    capsize=10, capthick=2, linewidth=2)  
          
        # Añadir valores en las barras  
        for j, (bar, mean, ci) in enumerate(zip(bars, means, cis)):  
            height = bar.get_height()  
            plt.text(bar.get_x() + bar.get_width()/2., height + ci + height*0.02,  
                    f'{mean:.2f} ± {ci:.2f}',  
                    ha='center', va='bottom', fontweight='bold', fontsize=19)  
          
        plt.ylabel(metric_data['ylabel'], fontweight='bold')  
        plt.title(f"{metric_data['title']}\n(Promedio de {NUM_SCENARIOS} escenarios con IC 95%)",   
                 fontsize=14, fontweight='bold', pad=20)  
        plt.xticks(x_pos, labels)  
        plt.grid(True, alpha=0.3, axis='y')  
          
        # Añadir puntos individuales para mostrar la variabilidad  
        for j, values in enumerate([metric_data['trad_values'], metric_data['ai_values']]):  
            x_scatter = [x_pos[j]] * len(values)  
            plt.scatter(x_scatter, values, color='black', alpha=0.6, s=30, zorder=3)  
          
        plt.tight_layout()  
        plt.savefig(os.path.join(output_dir, f'0{i}_{metric_name}_por_escenario.pdf'),   
                   dpi=300, bbox_inches='tight', facecolor='white')  
        plt.close()  
      
    # Gráfica de comparación general (4 subplots)  
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  
    fig.suptitle('Comparación Estadística por Escenarios: IA vs Semáforos Tradicionales\n' +  
                f'(Promedio de {NUM_SCENARIOS} escenarios con IC 95%)',   
                fontsize=16, fontweight='bold')  
      
    subplot_data = [  
        ('waiting_time', axes[0, 0]),  
        ('speed', axes[0, 1]),   
        ('stopped', axes[1, 0]),  
        ('throughput', axes[1, 1])  
    ]  
      
    for metric_name, ax in subplot_data:  
        metric_data = metrics[metric_name]  
          
        # Calcular estadísticas  
        trad_mean, trad_ci = calculate_ci_95(metric_data['trad_values'])  
        ai_mean, ai_ci = calculate_ci_95(metric_data['ai_values'])  
          
        # Crear gráfica de barras  
        x_pos = [0, 1]  
        means = [trad_mean, ai_mean]  
        cis = [trad_ci, ai_ci]  
        colors = [COLORS['traditional'], COLORS['ai']]  
        labels = ['Tradicional', 'IA (PPO)']  
          
        bars = ax.bar(x_pos, means, color=colors, alpha=0.7, width=0.6)  
        ax.errorbar(x_pos, means, yerr=cis, fmt='none', color='black',   
                   capsize=8, capthick=1.5, linewidth=1.5)  
          
        # Añadir puntos individuales  
        for j, values in enumerate([metric_data['trad_values'], metric_data['ai_values']]):  
            x_scatter = [x_pos[j]] * len(values)  
            ax.scatter(x_scatter, values, color='black', alpha=0.5, s=20, zorder=3)  
          
        ax.set_ylabel(metric_data['ylabel'])  
        ax.set_title(metric_data['title'].replace(' por Escenario', ''))  
        ax.set_xticks(x_pos)  
        ax.set_xticklabels(labels)  
        ax.grid(True, alpha=0.3, axis='y')  
      
    plt.tight_layout()  
    plt.savefig(os.path.join(output_dir, '05_comparacion_por_escenarios_general.png'),   
               dpi=300, bbox_inches='tight', facecolor='white')  
    plt.close()  
      
    print(f"✓ Gráficas basadas en escenarios guardadas en: {output_dir}")  
      
    # Retornar los datos de resumen para usar en main()  
    return scenario_summaries  
def main():  
    """Función principal que ejecuta la comparación estadística completa"""  
    print("=== INICIANDO COMPARACIÓN ESTADÍSTICA: IA vs SEMÁFOROS TRADICIONALES ===")  
    print(f"Ejecutando {NUM_SCENARIOS} escenarios diferentes para análisis robusto")  
      
    # Crear directorio de comparación  
    output_dir = create_comparison_directory()  
      
    # Ejecutar evaluaciones para todos los escenarios  
    print("\n1. Ejecutando simulaciones con semáforos tradicionales...")  
    traditional_metrics = run_traditional_evaluation_all_scenarios()  
      
    print("\n2. Ejecutando simulaciones con modelo IA...")  
    ai_metrics = run_ai_evaluation_all_scenarios()  
      
    # Combinar datos de todos los escenarios  
    print("\n3. Procesando y generando comparaciones estadísticas...")  
    all_metrics = traditional_metrics + ai_metrics  
    df_combined = pd.DataFrame(all_metrics)  
      
    # Guardar datos combinados  
    combined_file = os.path.join(output_dir, 'metricas_estadisticas_combinadas.csv')  
    df_combined.to_csv(combined_file, index=False)  
    print(f"✓ Datos de {NUM_SCENARIOS} escenarios guardados en: {combined_file}")  
      
    # CAMBIADO: Generar gráficas basadas en escenarios  
    scenario_summaries = generate_scenario_based_plots_with_ci(df_combined, output_dir)  
      
    # Mostrar estadísticas finales con intervalos de confianza  
    print("\n=== RESUMEN ESTADÍSTICO DE COMPARACIÓN ===")  
      
    # Usar los datos ya calculados  
    trad_scenarios = scenario_summaries[scenario_summaries['model_type'] == 'Tradicional']  
    ai_scenarios = scenario_summaries[scenario_summaries['model_type'] == 'IA (PPO)']  
      
    # Función para calcular intervalos de confianza del 95%  
    def calculate_ci_95(data):  
        mean = np.mean(data)  
        std_err = stats.sem(data)  
        ci = std_err * stats.t.ppf(0.975, len(data)-1)  
        return mean, ci  
      
    # Mostrar resultados  
    metrics_info = [  
        ('Tiempo de espera promedio', 'system_mean_waiting_time', 's'),  
        ('Velocidad promedio', 'system_mean_speed', 'm/s'),  
        ('Vehículos detenidos promedio', 'system_total_stopped', ''),  
        ('Throughput final', 'system_total_arrived', 'vehículos')  
    ]  
      
    for name, column, unit in metrics_info:  
        trad_mean, trad_ci = calculate_ci_95(trad_scenarios[column])  
        ai_mean, ai_ci = calculate_ci_95(ai_scenarios[column])  
          
        print(f"{name} (IC 95%):")  
        print(f"  - Tradicional: {trad_mean:.2f} ± {trad_ci:.2f}{unit}")  
        print(f"  - IA (PPO): {ai_mean:.2f} ± {ai_ci:.2f}{unit}")  
      
    # Pruebas de significancia estadística  
    from scipy.stats import ttest_ind  
      
    print(f"\n=== PRUEBAS DE SIGNIFICANCIA ESTADÍSTICA ===")  
      
    for name, column, _ in metrics_info:  
        t_stat, p_val = ttest_ind(trad_scenarios[column], ai_scenarios[column])  
        print(f"{name} - t-test: t={t_stat:.3f}, p={p_val:.4f}")  
        if p_val < 0.05:  
            print("  ✓ Diferencia estadísticamente significativa (p < 0.05)")  
        else:  
            print("  ✗ No hay diferencia estadísticamente significativa (p ≥ 0.05)")  
      
    print(f"\n✓ Comparación estadística completa con {NUM_SCENARIOS} escenarios.")  
    print(f"✓ Revisa la carpeta '{output_dir}' para ver todos los PNGs con intervalos de confianza.")  
  
if __name__ == "__main__":  
    main()