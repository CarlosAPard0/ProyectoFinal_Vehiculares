import glob  
import matplotlib.pyplot as plt  
import pandas as pd  
import numpy as np  
import os  
  
def moving_average(interval, window_size):  
    if window_size == 1:  
        return interval  
    window = np.ones(int(window_size)) / float(window_size)  
    return np.convolve(interval, window, "same")  
  
def plot_df(df, color, xaxis, yaxis, ma=1, label=""):  
    df_copy = df.copy()  
    df_copy[yaxis] = pd.to_numeric(df_copy[yaxis], errors="coerce")  
    df_copy[xaxis] = pd.to_numeric(df_copy[xaxis], errors="coerce")  
    df_copy = df_copy.dropna(subset=[xaxis, yaxis])  
      
    if df_copy.empty:  
        print(f"No hay datos válidos para graficar {yaxis} vs {xaxis}")  
        return  
      
    grouped = df_copy.groupby(xaxis)[yaxis]  
    mean = grouped.mean()  
    std = grouped.std()  
      
    if ma > 1:  
        mean = pd.Series(moving_average(mean.values, ma), index=mean.index)  
        std = pd.Series(moving_average(std.values, ma), index=std.index)  
      
    x = mean.index.values  
    plt.plot(x, mean.values, label=label, color=color)  
    plt.fill_between(x, mean.values + std.values, mean.values - std.values, alpha=0.25, color=color)  
  
# Crear carpeta PPO  
os.makedirs("PPO", exist_ok=True)  
  
# Procesar archivos CSV  
csv_files = glob.glob("outputs/grid4x4/ppo_advanced*")  
  
if csv_files:  
    # Combinar todos los archivos CSV  
    main_df = pd.DataFrame()  
    for f in csv_files:  
        df = pd.read_csv(f)  
        if main_df.empty:  
            main_df = df  
        else:  
            main_df = pd.concat((main_df, df), ignore_index=True)  
      
    print(f"Columnas disponibles: {list(main_df.columns)}")  
      
    # Definir métricas importantes para graficar  
    metrics_to_plot = [  
        ('system_total_waiting_time', 'Total Waiting Time (s)', 'blue'),  
        ('system_mean_waiting_time', 'Mean Waiting Time (s)', 'red'),  
        ('system_mean_speed', 'Mean Speed (m/s)', 'green'),  
        ('system_total_stopped', 'Total Stopped Vehicles', 'orange'),  
        ('system_total_running', 'Total Running Vehicles', 'purple'),  
        ('system_total_arrived', 'Total Arrived Vehicles', 'brown'),  
    ]  
      
    # Generar gráfica para cada métrica  
    for metric, title, color in metrics_to_plot:  
        if metric in main_df.columns:  
            plt.figure(figsize=(10, 6))  
            plot_df(main_df, color=color, xaxis='step', yaxis=metric, ma=100, label='PPO')  
            plt.title(f'{title} - PPO Training Results')  
            plt.ylabel(title)  
            plt.xlabel('Time step (seconds)')  
            plt.grid(True, alpha=0.3)  
            plt.legend()  
            plt.savefig(f'PPO/{metric}_results.pdf', bbox_inches="tight")  
            plt.close()  
            print(f"Gráfica generada: PPO/{metric}_results.pdf")  
        else:  
            print(f"Métrica {metric} no encontrada en los datos")  
      
    # Gráfica combinada de métricas principales  
    plt.figure(figsize=(15, 10))  
      
    # Subplot para tiempo de espera  
    plt.subplot(2, 2, 1)  
    if 'system_total_waiting_time' in main_df.columns:  
        plot_df(main_df, color='blue', xaxis='step', yaxis='system_total_waiting_time', ma=100, label='PPO')  
        plt.title('Total Waiting Time')  
        plt.ylabel('Time (s)')  
        plt.grid(True, alpha=0.3)  
      
    # Subplot para velocidad promedio  
    plt.subplot(2, 2, 2)  
    if 'system_mean_speed' in main_df.columns:  
        plot_df(main_df, color='green', xaxis='step', yaxis='system_mean_speed', ma=100, label='PPO')  
        plt.title('Mean Speed')  
        plt.ylabel('Speed (m/s)')  
        plt.grid(True, alpha=0.3)  
      
    # Subplot para vehículos detenidos  
    plt.subplot(2, 2, 3)  
    if 'system_total_stopped' in main_df.columns:  
        plot_df(main_df, color='orange', xaxis='step', yaxis='system_total_stopped', ma=100, label='PPO')  
        plt.title('Total Stopped Vehicles')  
        plt.ylabel('Number of vehicles')  
        plt.grid(True, alpha=0.3)  
      
    # Subplot para vehículos en movimiento  
    plt.subplot(2, 2, 4)  
    if 'system_total_running' in main_df.columns:  
        plot_df(main_df, color='purple', xaxis='step', yaxis='system_total_running', ma=100, label='PPO')  
        plt.title('Total Running Vehicles')  
        plt.ylabel('Number of vehicles')  
        plt.grid(True, alpha=0.3)  
      
    plt.tight_layout()  
    plt.savefig('PPO/combined_metrics.pdf', bbox_inches="tight")  
    plt.show()  
    print("Gráfica combinada generada: PPO/combined_metrics.pdf")  
  
else:  
    print("No se encontraron archivos CSV en outputs/grid4x4/ppo_advanced*")