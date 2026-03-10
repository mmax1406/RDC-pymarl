from envs.mpe.kalman_GRU_estimator import *
import torch
import collections
import numpy as np
import time

def init_kalmans(env, currentObs=None, Q_noise=0.05, R_noise=0.02):
    """
    Initializes one GRU-Kalman Filter per agent.
    Each filter estimates the agent's entire observation vector.
    """
    kalmans = {}
    # PettingZoo MPE uses strings for agent keys
    for agent_id in range(len(currentObs)):
        # Get s_dim from the observation space of this specific agent
        s_dim = env.observation_space[agent_id].shape[0]
        
        # Determine environment name for weight loading
        env_name = env.spec.id
        
        estimator = GRUEstimator(
            model_path=f"src/envs/mpe/weights/policy_aware_gru_{env_name}.pth",
            s_dim=s_dim, 
            h_dim=128
        )

        ic = None
        if currentObs is not None:
            val = currentObs[agent_id]
            ic = val.detach().cpu().numpy().flatten() if torch.is_tensor(val) else np.array(val).flatten()

        kf = GRUKalmanFilter(
            estimator, 
            Q=np.eye(s_dim) * Q_noise,
            R=np.eye(s_dim) * R_noise,
            ic=ic
        )
        kalmans[agent_id] = kf
    return kalmans

def init_Buffers(env, max_delay, obs=None):
    """Initializes observation deques for each agent."""
    obsBuffers = {agent: collections.deque(maxlen=max_delay + 1) for agent in env.possible_agents}
    
    for agent_id in env.possible_agents:
        obs_shape = env.observation_spaces[agent_id].shape
        # Fill buffer with zeros or initial observation
        initial_val = np.zeros(obs_shape) if obs is None else obs[agent_id].detach().cpu().numpy()
        for _ in range(max_delay + 1):
            obsBuffers[agent_id].append(initial_val.copy())
            
    return obsBuffers

def reset_buffers_kalman(env, obsBuffer, kalmans, max_delay, currentObs=None):
    """Full reset for a new MPE episode."""
    for agent_id in env.possible_agents:
        obsBuffer[agent_id].clear()
        
        val = currentObs[agent_id]
        val_np = val.detach().cpu().numpy().flatten() if torch.is_tensor(val) else np.array(val).flatten()

        for _ in range(max_delay + 1):
            obsBuffer[agent_id].append(val_np.copy())

        kalmans[agent_id].reset(ic=val_np.copy())
    
    return obsBuffer, kalmans

def update_Buffers(env, obsBuffers, currentObs):
    """
    currentObs: List of Tensors from env.step()
    currentActions: List or Tensor of actions
    """
    for i in range(len(env.agents)):
        obsBuffers[i].append(currentObs[i][-1].detach().cpu().numpy())
        
    return obsBuffers

def get_delays(env, delay=0, max_delay=13, use_dynamic_delay=False):
    """
    Calculates delays. 
    Note: For MPE, we usually apply the same delay to the whole observation.
    """
    delays = {agent: delay for agent in env.possible_agents}
    
    if use_dynamic_delay:
        for agent_id in env.possible_agents:
            # Distance-based dynamic delay logic
            agent_obj = next(a for a in env.unwrapped.world.agents if a.name == agent_id)
            dist = np.linalg.norm(agent_obj.state.p_pos)
            d = int(np.clip((dist // 5) + np.random.randint(-1, 2), 0, max_delay))
            delays[agent_id] = d
                
    return delays

def fix_static_landmarks(env_info, agent_obs, clean_history_buffer):
    """
    Reconstructs perfectly accurate relative landmark positions.
    Assumes agent_obs[:4] has already been 'injected' with current clean data.
    """
    fixed_obs = agent_obs.copy()
    
    # 1. Get current absolute position (Already injected into indices 2:4)
    p_self_now = agent_obs[2:4]
    
    # 2. Get absolute position at step 0 (Start of buffer)
    p_self_start = clean_history_buffer[0][2:4]

    if env_info["map_type"] == "tag":
        # Tag: 2 Obstacles (Indices 4:8)
        # Step A: Find absolute landmark positions [L_rel_0 + P_self_0]
        l_abs = clean_history_buffer[0][4:8] + np.tile(p_self_start, 2)
        # Step B: Project to current relative frame [L_abs - P_self_now]
        fixed_obs[4:8] = l_abs - np.tile(p_self_now, 2)

    elif env_info["map_type"] == "spread":
        # Spread: 3 Landmarks (Indices 4:10)
        # Note: spread has 3 landmarks, so we need 6 indices (4,5, 6,7, 8,9)
        l_abs = clean_history_buffer[0][4:10] + np.tile(p_self_start, 3)
        fixed_obs[4:10] = l_abs - np.tile(p_self_now, 3)

    return fixed_obs

def transform_to_absolute(s_raw, env_name):
    s_abs = s_raw.copy()
    p_self = s_raw[:, :, :, 2:4]
    # Using your specific indices for simple_tag
    indices = [4, 6, 8, 12, 14] if 'tag' in env_name else [4, 6, 8, 10, 14]
    for idx in indices:
        s_abs[:, :, :, idx:idx+2] += p_self
    return s_abs

def get_observation_KF(env, obsBuffers, d, kalmans, t_now):
    """
    Core logic: Predicts the current observation from a delayed history.
    """
    clean_obs, delayed_obs, kalman_fixed_obs = {}, {}, {}

    for agent_id in range(env['n_agents']):
        # 1. Get Ground Truth (Latest in buffer)
        clean_obs[agent_id] = obsBuffers[agent_id][-1].copy()
        kf = kalmans[agent_id]
        
        # 2. Get Delayed Measurement (z)
        h_idx = max(0, len(obsBuffers[agent_id]) - 1 - d)
        z_delayed = obsBuffers[agent_id][h_idx].flatten()
        delayed_obs[agent_id] = z_delayed.copy()

        if t_now>d:
            # Update the 'grounded' state with the measurement
            kf.predict()
            kf.update(z_delayed)
            # Predict the Rollout (Fast-Forward to the present)
            fixed_val, _ = kf.predict_future(d)
        else:
            # Update the necessary steps
            d = int(min(d,t_now))
            # Predict the Rollout (Fast-Forward to the present)
            fixed_val, _ = kf.predict_future(d)
        k_fixed = np.array(fixed_val).flatten()

        # 5. Injection Logic: Keep self-info clean (Real-time)
        k_fixed[:4] = clean_obs[agent_id][:4]
        kalman_fixed_obs[agent_id] = k_fixed

        # Test if fixint the landmarks position does something
        kalman_fixed_obs[agent_id] = fix_static_landmarks(env, kalman_fixed_obs[agent_id], obsBuffers[agent_id])

    return clean_obs, delayed_obs, kalman_fixed_obs