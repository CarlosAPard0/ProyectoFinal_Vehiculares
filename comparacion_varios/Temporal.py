import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy import stats  
  
def plot_speed_evolution_with_ci(csv_file, save_path=None):  
    """  
    Crea una gráfica de evolución temporal de velocidad promedio con intervalos de confianza  
    """  
    # Cargar datos  
    df = pd.read_csv(csv_file)  
      
    # Configurar matplotlib con el mismo estilo que tu código  
    plt.rcParams.update({  
        'figure.figsize': (14, 8),  
        'font.size': 16,  
        'axes.titlesize': 18,  
        'axes.labelsize': 16,  
        'xtick.labelsize': 14,  
        'ytick.labelsize': 14,  
        'legend.fontsize': 15,  
        'lines.linewidth': 2,  
        'grid.alpha': 0.3,  
        'axes.grid': True,  
        'figure.autolayout': True  
    })  
      
    # Colores consistentes con tu código  
    COLORS = {  
        'Tradicional': '#d62728',  
        'IA (PPO)': '#1f77b4'  
    }  
      
    plt.figure(figsize=(14, 8))  
      
    # Procesar cada tipo de modelo  
    for model_type in df['model_type'].unique():  
        model_data = df[df['model_type'] == model_type]  
          
        # Agrupar por tiempo de simulación y calcular estadísticas  
        time_stats = []  
        for time_point in sorted(model_data['simulation_time'].unique()):  
            time_data = model_data[model_data['simulation_time'] == time_point]  
            speeds = time_data['system_mean_speed'].values  
              
            if len(speeds) > 1:  
                # Calcular media e intervalo de confianza del 95%  
                mean_speed = np.mean(speeds)  
                std_err = stats.sem(speeds)  
                ci = std_err * stats.t.ppf(0.975, len(speeds)-1)  
                  
                time_stats.append({  
                    'time': time_point,  
                    'mean': mean_speed,  
                    'ci_lower': mean_speed - ci,  
                    'ci_upper': mean_speed + ci,  
                    'std_err': std_err  
                })  
            else:  
                # Si solo hay un punto, no hay intervalo de confianza  
                time_stats.append({  
                    'time': time_point,  
                    'mean': speeds[0] if len(speeds) > 0 else 0,  
                    'ci_lower': speeds[0] if len(speeds) > 0 else 0,  
                    'ci_upper': speeds[0] if len(speeds) > 0 else 0,  
                    'std_err': 0  
                })  
          
        # Convertir a arrays para plotting  
        times = [stat['time'] for stat in time_stats]  
        means = [stat['mean'] for stat in time_stats]  
        ci_lower = [stat['ci_lower'] for stat in time_stats]  
        ci_upper = [stat['ci_upper'] for stat in time_stats]  
          
        # Plotear línea principal  
        color = COLORS.get(model_type, '#333333')  
        plt.plot(times, means, color=color, label=model_type, linewidth=2.5)  
          
        # Plotear intervalo de confianza  
        plt.fill_between(times, ci_lower, ci_upper,   
                        color=color, alpha=0.2,   
                        label=f'IC 95% - {model_type}')  
      
    # Configurar gráfica  
    plt.xlabel('Tiempo de Simulación (s)', fontweight='bold')  
    plt.ylabel('Velocidad Promedio (m/s)', fontweight='bold')  
    plt.title('Evolución Temporal de la Velocidad Promedio\ncon Intervalos de Confianza del 95%',   
              fontweight='bold', pad=20)  
    plt.legend(loc='best')  
    plt.grid(True, alpha=0.3)  
      
    # Mejorar el formato del eje X  
    plt.xticks(np.arange(0, max(df['simulation_time']) + 50, 50))  
      
    if save_path:  
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')  
        print(f"Gráfica guardada en: {save_path}")  
      
    plt.show()  
  
def apply_moving_average_to_evolution(csv_file, window_size=5, save_path=None):  
    """  
    Versión con promedio móvil para suavizar las líneas  
    """  
    # Función de promedio móvil (similar a tu código)  
    def moving_average(interval, window_size=5):  
        if window_size == 1:  
            return interval  
        window = np.ones(int(window_size)) / float(window_size)  
        return np.convolve(interval, window, "same")  
      
    df = pd.read_csv(csv_file)  
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
    plt.figure(figsize=(14, 8))  
      
    COLORS = {  
        'Tradicional': '#d62728',  
        'IA (PPO)': '#1f77b4'  
    }  
      
    for model_type in df['model_type'].unique():  
        model_data = df[df['model_type'] == model_type]  
          
        # Agrupar por tiempo y calcular estadísticas  
        time_stats = []  
        for time_point in sorted(model_data['simulation_time'].unique()):  
            time_data = model_data[model_data['simulation_time'] == time_point]  
            speeds = time_data['system_mean_speed'].values  
              
            if len(speeds) > 1:  
                mean_speed = np.mean(speeds)  
                std_err = stats.sem(speeds)  
                ci = std_err * stats.t.ppf(0.975, len(speeds)-1)  
            else:  
                mean_speed = speeds[0] if len(speeds) > 0 else 0  
                ci = 0  
              
            time_stats.append({  
                'time': time_point,  
                'mean': mean_speed,  
                'ci': ci  
            })  
          
        times = [stat['time'] for stat in time_stats]  
        means = [stat['mean'] for stat in time_stats]  
        cis = [stat['ci'] for stat in time_stats]  
          
        # Aplicar promedio móvil  
        smoothed_means = moving_average(means, window_size)  
        smoothed_cis = moving_average(cis, window_size)  
          
        color = COLORS.get(model_type, '#333333')  
        plt.plot(times, smoothed_means, color=color, label=f'{model_type}', linewidth=2.5)  
          
        # Intervalo de confianza suavizado  
        ci_lower = np.array(smoothed_means) - np.array(smoothed_cis)  
        ci_upper = np.array(smoothed_means) + np.array(smoothed_cis)  
        plt.fill_between(times, ci_lower, ci_upper,   
                        color=color, alpha=0.2)  
      
    plt.xlabel('Tiempo de Simulación (s)', fontweight='bold')  
    plt.ylabel('Velocidad Promedio (m/s)', fontweight='bold')  
    plt.title(f'Evolución Temporal de la Velocidad Promedio\n' +  
              'con Intervalos de Confianza del 95%', fontweight='bold', pad=20)  
    plt.legend(loc='best')  
    plt.grid(True, alpha=0.3)  
    plt.xticks(np.arange(0, max(df['simulation_time']) + 50, 50))  
      
    if save_path:  
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')  
        print(f"Gráfica suavizada guardada en: {save_path}")  
      
    plt.show()  
  
# Uso del script  
if __name__ == "__main__":  
    csv_file = "metricas_estadisticas_combinadas.csv"  # Tu archivo CSV  
    
    # Gráfica normal  
    plot_speed_evolution_with_ci(csv_file, "velocidad_temporal_con_ic.png")  
      
    # Gráfica suavizada  
    apply_moving_average_to_evolution(csv_file, window_size=1, save_path="03_velocidad_con_ic.pdf")
