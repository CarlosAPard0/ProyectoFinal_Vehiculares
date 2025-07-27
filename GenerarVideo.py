import os    
import sys    
import torch    
import torch.nn as nn    
import pickle    
import numpy as np    
import supersuit as ss    
import subprocess    
import shutil    
import time    
from tqdm import trange    
from pyvirtualdisplay.smartdisplay import SmartDisplay    
from PIL import Image    
import threading    
from queue import Queue    
    
if "SUMO_HOME" in os.environ:    
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")    
    sys.path.append(tools)    
else:    
    sys.exit("Please declare the environment variable 'SUMO_HOME'")    
    
import sumo_rl    
  
class PolicyWithCNN(nn.Module):      
    def __init__(self, obs_space, action_space, model_config):      
        super().__init__()  
          
        # Tu configuración específica del entrenamiento  
        conv_filters = model_config.get('conv_filters', [  
            [8, [2, 1], 1],   # 8 filtros, kernel 2x1, stride 1  
            [16, [2, 1], 1],  # 16 filtros para mayor capacidad  
            [32, [1, 1], 1],  # Capa final de convolución  
        ])  
          
        # Para frame stacking de 3 frames, la entrada tiene shape (3, 19)  
        # donde 19 es el tamaño de observación base por frame  
        input_channels = 3  # frame stacking de 3 frames  
        obs_per_frame = obs_space[0] // 3  # 57 // 3 = 19  
          
        # Crear capas convolucionales 1D (no 2D)  
        conv_layers = []  
        in_channels = input_channels  
        current_length = obs_per_frame  
          
        for filters, kernel_size, stride in conv_filters:  
            # Usar Conv1d para datos 1D con frame stacking  
            conv_layers.append(nn.Conv1d(in_channels, filters, kernel_size[0], stride))  
              
            # Activación según configuración  
            activation = model_config.get('conv_activation', 'relu')  
            if activation == 'swish':  
                conv_layers.append(nn.SiLU())  
            else:  
                conv_layers.append(nn.ReLU())  
                  
            in_channels = filters  
            # Calcular nueva longitud después de convolución  
            current_length = (current_length - kernel_size[0]) // stride + 1  
              
        self.conv_layers = nn.Sequential(*conv_layers)  
          
        # Capas densas después de convoluciones  
        fcnet_hiddens = model_config.get('fcnet_hiddens', [256, 128])  
        conv_output_size = in_channels * current_length  
          
        fc_layers = []  
        input_size = conv_output_size  
          
        for hidden_size in fcnet_hiddens:  
            fc_layers.append(nn.Linear(input_size, hidden_size))  
              
            # Activación según configuración  
            activation = model_config.get('fcnet_activation', 'relu')  
            if activation == 'swish':  
                fc_layers.append(nn.SiLU())  
            else:  
                fc_layers.append(nn.ReLU())  
                  
            input_size = hidden_size  
              
        self.fc_layers = nn.Sequential(*fc_layers)  
          
        # Cabezas de salida  
        self.action_head = nn.Linear(input_size, action_space)  
        self.value_head = nn.Linear(input_size, 1)  
          
    def forward(self, obs, return_value=False):  
        # Reshape para convoluciones 1D: (batch, channels, length)  
        # obs shape: (batch, 57) -> (batch, 3, 19)  
        batch_size = obs.shape[0]  
        obs_per_frame = obs.shape[1] // 3  
        x = obs.view(batch_size, 3, obs_per_frame)  
          
        # Procesar a través de capas convolucionales  
        x = self.conv_layers(x)  
          
        # Flatten para capas densas  
        x = x.view(batch_size, -1)  
          
        # Procesar a través de capas densas  
        x = self.fc_layers(x)  
          
        # Generar salidas  
        action_logits = self.action_head(x)  
        if return_value:  
            value = self.value_head(x)  
            return action_logits, value  
        return action_logits
def load_extracted_model():      
    with open("model_info.pkl", "rb") as f:      
        model_info = pickle.load(f)      
          
    obs_space = model_info['observation_space'].shape  
    action_space = model_info['action_space'].n      
    model_config = model_info['model_config']      
          
    print(f"Cargando modelo CNN con obs_space={obs_space}, action_space={action_space}")      
          
    model = PolicyWithCNN(obs_space, action_space, model_config)      
          
    if os.path.exists("model_weights_pytorch.pth"):      
        try:      
            state_dict = torch.load("model_weights_pytorch.pth", map_location='cpu')      
            model.load_state_dict(state_dict, strict=False)      
            print("Pesos PyTorch CNN cargados exitosamente")      
        except Exception as e:      
            print(f"No se pudieron cargar pesos PyTorch: {e}")      
          
    return model    
def make_env():      
    def advanced_reward_function(traffic_signal):      
        waiting_time_reward = traffic_signal._diff_waiting_time_reward()      
        speed_reward = traffic_signal._average_speed_reward() * 0.1      
        queue_penalty = traffic_signal._queue_reward() * 0.05      
        pressure_reward = traffic_signal._pressure_reward() * 0.02      
        phase_stability_bonus = 0.1 if traffic_signal.time_since_last_phase_change > traffic_signal.min_green else -0.05      
        return waiting_time_reward + speed_reward + queue_penalty + pressure_reward + phase_stability_bonus      
      
    env = sumo_rl.parallel_env(      
        net_file="grid3x3.net.xml",      
        route_file="rutas_nuevas1.rou.xml",      
        out_csv_name="outputs/grid3x3/high_freq_video",  # Cambiar nombre  
        use_gui=True,      
        virtual_display=(1980, 1080),      
        render_mode="rgb_array",      
        reward_fn=advanced_reward_function,      
        num_seconds=560,  # Coincidir con entrenamiento  
        delta_time=5,      
        begin_time=0,      
        time_to_teleport=300,  # Coincidir con entrenamiento  
    )      
    env = ss.pad_observations_v0(env)      
    env = ss.frame_stack_v1(env, 3)      
    return env  

  
def capture_high_frequency_video():  
    """Captura video con alta frecuencia temporal independiente de las acciones del agente"""  
      
    model = load_extracted_model()  
    model.eval()  
      
    env = make_env()  
      
    # Preparar directorio para frames  
    if os.path.exists("temp"):  
        shutil.rmtree("temp")  
    os.mkdir("temp")  
      
    reset_result = env.reset()  
    if isinstance(reset_result, tuple):  
        observations, info = reset_result  
    else:  
        observations = reset_result  
      
    #model.reset_lstm_states()  
      
    # Configuración de captura de alta frecuencia  
    simulation_duration = 560  # segundos de simulación  
    capture_interval = 0.25    # capturar cada 0.25 segundos  
    agent_decision_interval = 5  # agentes deciden cada 5 segundos  
      
    total_frames_expected = int(simulation_duration / capture_interval)  
    frames_per_agent_decision = int(agent_decision_interval / capture_interval)  # 20 frames por decisión  
      
    print(f"Capturando video de alta frecuencia:")  
    print(f"  - Duración: {simulation_duration}s")  
    print(f"  - Intervalo de captura: {capture_interval}s")  
    print(f"  - Frames esperados: {total_frames_expected}")  
    print(f"  - Frames por decisión de agente: {frames_per_agent_decision}")  
      
    frames_captured = 0  
    current_actions = {}  
    last_agent_decision_time = 0  
      
    # Obtener acciones iniciales  
    for agent_id in observations.keys():  
        obs_tensor = torch.FloatTensor(observations[agent_id]).unsqueeze(0)  
        with torch.no_grad():  
            action_logits = model(obs_tensor)  
            probs = torch.softmax(action_logits, dim=1)  
            action = torch.multinomial(probs, 1).item()  
        current_actions[agent_id] = action  
      
    # Aplicar acciones iniciales  
    env.step(current_actions)  
      
    # Loop principal de captura de alta frecuencia  
    for frame_idx in trange(total_frames_expected):  
        current_sim_time = frame_idx * capture_interval  
          
        # Verificar si es tiempo de que los agentes tomen nuevas decisiones  
        if current_sim_time - last_agent_decision_time >= agent_decision_interval:  
            try:  
                # Obtener nuevas observaciones y acciones  
                step_result = env.step(current_actions)  
                  
                if len(step_result) == 5:  
                    observations, rewards, terminations, truncations, infos = step_result  
                    done = all(terminations.values()) or all(truncations.values())  
                elif len(step_result) == 4:  
                    observations, rewards, dones, infos = step_result  
                    done = dones.get("__all__", False)  
                else:  
                    done = True  
                  
                if done:  
                    print(f"Simulación terminada en tiempo {current_sim_time:.2f}s")  
                    break  
                  
                # Obtener nuevas acciones del modelo  
                for agent_id in observations.keys():  
                    obs_tensor = torch.FloatTensor(observations[agent_id]).unsqueeze(0)  
                    with torch.no_grad():  
                        action_logits = model(obs_tensor)  
                        probs = torch.softmax(action_logits, dim=1)  
                        action = torch.multinomial(probs, 1).item()  
                    current_actions[agent_id] = action  
                  
                last_agent_decision_time = current_sim_time  
                  
            except Exception as e:  
                print(f"Error en decisión de agente en tiempo {current_sim_time:.2f}s: {e}")  
                break  
          
        # CAPTURAR FRAME DEL ESCENARIO ACTUAL  
        try:  
            img_array = env.render()  
            if img_array is not None:  
                img = Image.fromarray(img_array)  
                img.save(f"temp/img{frames_captured:06d}.jpg")  
                frames_captured += 1  
            else:  
                print(f"Warning: render() devolvió None en frame {frame_idx}")  
        except Exception as e:  
            print(f"Error capturando frame {frame_idx}: {e}")  
            continue  
          
        # Progreso cada 100 frames  
        if (frame_idx + 1) % 100 == 0:  
            elapsed_sim_time = (frame_idx + 1) * capture_interval  
            print(f"  Tiempo simulado: {elapsed_sim_time:.1f}s, Frames: {frames_captured}")  
          
        # Pequeña pausa para simular el paso del tiempo  
        time.sleep(0.01)  # Pausa mínima para no sobrecargar  
      
    print(f"Captura completada. Total de frames: {frames_captured}")  
      
    # Cerrar ambiente  
    env.close()  
      
    # CREAR VIDEO DE ALTA FRECUENCIA  
    if frames_captured > 10:  
        print(f"Creando video de alta frecuencia con {frames_captured} frames...")  
          
        try:  
            # Calcular FPS para reproducción en tiempo real o acelerada  
            # Para tiempo real: fps = 1/capture_interval = 4 FPS  
            # Para reproducción más fluida: usar 20-30 FPS  
            target_fps = 15  # Reproducción acelerada y fluida  
              
            result = subprocess.run([  
                "ffmpeg", "-y",   
                "-framerate", str(target_fps),  
                "-i", "temp/img%06d.jpg",  # 6 dígitos para muchos frames  
                "-c:v", "libx264",  
                "-preset", "fast",  
                "-pix_fmt", "yuv420p",  
                "-crf", "18",  # Alta calidad  
                "high_frequency_traffic_video.mp4"  
            ], capture_output=True, text=True, timeout=300)  
              
            if result.returncode == 0:  
                print(f"✅ VIDEO DE ALTA FRECUENCIA GENERADO: high_frequency_traffic_video.mp4")  
                print(f"   Framerate de reproducción: {target_fps} FPS")  
                print(f"   Frames totales: {frames_captured}")  
                print(f"   Resolución temporal: {1/capture_interval:.1f} capturas por segundo de simulación")  
                print(f"   Duración del video: {frames_captured/target_fps:.1f} segundos")  
                  
                if os.path.exists("high_frequency_traffic_video.mp4"):  
                    size = os.path.getsize("high_frequency_traffic_video.mp4")  
                    print(f"   Tamaño: {size/1024/1024:.1f} MB")  
            else:  
                print(f"❌ Error en ffmpeg: {result.stderr}")  
                  
        except Exception as e:  
            print(f"❌ Error creando video: {e}")  
    else:  
        print("❌ No hay suficientes frames para video")  
      
    # Limpiar archivos temporales  
    if os.path.exists("temp"):  
        shutil.rmtree("temp")  
      
    return frames_captured

if __name__ == "__main__":    
    if not os.path.exists("model_info.pkl"):    
        print("Error: No se encontró model_info.pkl")    
        sys.exit(1)    
        
    print("Creando display virtual...")    
    display = SmartDisplay(visible=0, size=(1920, 1080))    
    display.start()    
    print("Display virtual iniciado.")    
        
    try:    
        print("Iniciando captura de video...")    
            
        frames = capture_high_frequency_video()  # CORRECCIÓN AQUÍ  
            
        print("🎬 ¡Video completado!")    
            
    except Exception as e:    
        print(f"❌ Error durante la captura: {e}")    
        import traceback    
        traceback.print_exc()    
        
    finally:    
        print("Deteniendo display virtual...")    
        display.stop()    
        print("Display virtual detenido.")
