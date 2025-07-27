import os  
import sys  
import ray  
import torch  
import pickle  
from ray.rllib.algorithms.ppo import PPO  
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv  
from ray.tune.registry import register_env  
import supersuit as ss  
  
if "SUMO_HOME" in os.environ:  
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")  
    sys.path.append(tools)  
else:  
    sys.exit("Please declare the environment variable 'SUMO_HOME'")  
  
import sumo_rl  
  
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
        route_file="rutas_nuevas.rou.xml",  
        out_csv_name="outputs/grid4x4/ppo_extraction",  
        use_gui=False,  
        reward_fn=advanced_reward_function,  
        num_seconds=360,  
        begin_time=0,  
        time_to_teleport=300,  
    )  
    env = ss.pad_observations_v0(env)  
    env = ss.frame_stack_v1(env, 3)  
    return env  
  
def extract_model_weights(checkpoint_path):  
    """Extrae los pesos del modelo desde un checkpoint de Ray RLlib"""  
      
    # Inicializar Ray en modo local  
    ray.init(local_mode=True, ignore_reinit_error=True)  
      
    try:  
        # Registrar el ambiente (necesario para cargar el checkpoint)  
        env_name = "grid3x3"
        register_env(  
            env_name,  
            lambda _: ParallelPettingZooEnv(make_env())  
        )  
          
        print("Cargando checkpoint...")  
        # Cargar el algoritmo desde el checkpoint  
        algo = PPO.from_checkpoint(checkpoint_path)  
          
        # Obtener la política por defecto  
        policy = algo.get_policy("default_policy")  
          
        # Extraer los pesos del modelo  
        model_weights = policy.get_weights()  
          
        # Guardar los pesos en diferentes formatos  
          
        # 1. Formato pickle (completo)  
        with open("model_weights_complete.pkl", "wb") as f:  
            pickle.dump(model_weights, f)  
        print("Pesos completos guardados en: model_weights_complete.pkl")  
          
        # 2. Solo los pesos de la red neuronal (PyTorch)  
        if hasattr(policy.model, 'state_dict'):  
            torch_weights = policy.model.state_dict()  
            torch.save(torch_weights, "model_weights_pytorch.pth")  
            print("Pesos PyTorch guardados en: model_weights_pytorch.pth")  
          
        # 3. Información del modelo  
        model_info = {  
            'observation_space': policy.observation_space,  
            'action_space': policy.action_space,  
            'model_config': policy.config.get('model', {}),  
        }  
          
        with open("model_info.pkl", "wb") as f:  
            pickle.dump(model_info, f)  
        print("Información del modelo guardada en: model_info.pkl")  
          
        # Mostrar información del modelo  
        print("\n=== INFORMACIÓN DEL MODELO ===")  
        print(f"Espacio de observación: {policy.observation_space}")  
        print(f"Espacio de acción: {policy.action_space}")  
        print(f"Configuración del modelo: {policy.config.get('model', {})}")  
          
        return model_weights, model_info  
          
    finally:  
        ray.shutdown()  
  
if __name__ == "__main__":  
    #checkpoint_path = "/workspace/ray_results/PPO_Grid4x4_Advanced/PPO_grid4x4_ddb0f_00000_0_2025-07-26_04-13-21/checkpoint_003121/"  
    checkpoint_path = "/workspace/ray_results/PPO_Grid4x4_CNN/PPO_grid3x3_69bf3_00000_0_2025-07-26_21-42-22/checkpoint_004999"

      
    if not os.path.exists(checkpoint_path):  
        print(f"Error: No se encontró el checkpoint en {checkpoint_path}")  
        sys.exit(1)  
      
    print("Extrayendo pesos del modelo...")  
    weights, info = extract_model_weights(checkpoint_path)  
    print("Extracción completada!")