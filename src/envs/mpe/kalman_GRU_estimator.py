import torch
import torch.nn as nn
import numpy as np

class GRUEstimator(nn.Module):
    def __init__(self, model_path, s_dim, h_dim=512, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Match your training architecture (4 layers, 512 hidden)
        self.gru = nn.GRU(s_dim, h_dim, num_layers=3, batch_first=True)
        self.ln = nn.LayerNorm(h_dim)
        self.fc = nn.Linear(h_dim, s_dim)

        # Buffers for normalization
        self.register_buffer('mu_s', torch.zeros(s_dim))
        self.register_buffer('std_s', torch.ones(s_dim))
        self.register_buffer('mu_d', torch.zeros(s_dim))
        self.register_buffer('std_d', torch.ones(s_dim))
        
        # Load the checkpoint
        ckpt = torch.load(model_path, map_location=self.device)
        # Handle both full-save and state-dict-only formats
        self.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        
        if 'mu_s' in ckpt:
            self.mu_s.copy_(ckpt['mu_s'])
            self.std_s.copy_(ckpt['std_s'])
            self.mu_d.copy_(ckpt['mu_d'])
            self.std_d.copy_(ckpt['std_d'])
            
        self.to(self.device)
        self.eval()

    def forward(self, s_norm, h=None):
        out, h_new = self.gru(s_norm, h)
        return self.fc(out), h_new

    def predict_step(self, x_curr, h_in):
        """
        Processes a single step on the GPU.
        Input: x_curr (Tensor or Numpy), h_in (Tensor or None)
        """
        with torch.no_grad():
            # Ensure input is a GPU Tensor
            if isinstance(x_curr, np.ndarray):
                x_tensor = torch.from_numpy(x_curr).float().to(self.device)
            else:
                x_tensor = x_curr.to(self.device).float()
            
            if h_in is not None:
                h_in = h_in.to(self.device)
            
            # Normalize -> Predict -> Denormalize
            s_norm = (x_tensor - self.mu_s) / self.std_s
            delta_norm_seq, h_out = self.forward(s_norm.view(1, 1, -1), h_in)
            delta_phys = (delta_norm_seq.view(-1) * self.std_d) + self.mu_d
            
            x_next = x_tensor + delta_phys
            return x_next, h_out

class GRUKalmanFilter:
    def __init__(self, env_info, Q = 0.05, R = 0.02, ic=None):
        self.env_name = env_info['map_type']
        self.s_dim = env_info['obs_shape']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_path = f"src/envs/mpe/weights/model_simple_{self.env_name}_v3_v3_GRU.pth"
        self.est = GRUEstimator(model_path, self.s_dim, h_dim=512, device=self.device)
        
        # Move noise covariances to GPU
        self.Q = torch.from_numpy(Q).float().to(self.device)
        self.R = torch.from_numpy(R).float().to(self.device)
        
        # Use the reset method for initial setup
        self.reset(ic)

    def transform_to_absolute(self, s_raw):
        """Converts relative observations to absolute world coordinates."""
        s_abs = s_raw.copy()
        p_self = s_raw[..., 2:4] 
        indices = [4, 6, 8, 12, 14] if 'tag' in self.env_name else [4, 6, 8, 10, 14]
        for idx in indices:
            if idx + 1 < s_abs.shape[-1]:
                s_abs[..., idx:idx+2] += p_self
        return s_abs
    
    def transform_to_relative(self, s_abs):
        """Converts relative observations to absolute world coordinates."""
        s_raw = s_abs.clone()
        p_self = s_abs[..., 2:4] 
        indices = [4, 6, 8, 12, 14] if 'tag' in self.env_name else [4, 6, 8, 10, 14]
        for idx in indices:
            if idx + 1 < s_raw.shape[-1]:
                s_raw[..., idx:idx+2] -= p_self
        return s_raw 

    def reset(self, ic=None):
        """
        Grounds the filter. Moves the initial condition to GPU 
        and wipes the hidden state.
        """
        if ic is not None:
            ic = self.transform_to_absolute(ic)
            if isinstance(ic, np.ndarray):
                self.x = torch.from_numpy(ic).float().to(self.device)
            else:
                self.x = ic.to(self.device).float()
        else:
            self.x = None
            
        self.h = None
        # P must also live on the GPU
        self.P = torch.eye(self.Q.shape[0], device=self.device) * 0.1

    def predict(self):
        if self.x is None: return
        # Logic stays on GPU
        self.x, self.h = self.est.predict_step(self.x, self.h)
        self.P = self.P + self.Q

    def update(self, z_measurement):
        if self.x is None:
            self.reset(z_measurement)
            return

        # Ensure measurement is a GPU tensor
        z_measurement = self.transform_to_absolute(z_measurement)
        if isinstance(z_measurement, np.ndarray):
            z_meas = torch.from_numpy(z_measurement).float().to(self.device).flatten()
        else:
            z_meas = z_measurement.to(self.device).float().flatten()

        # Kalman Math on GPU
        dim = self.x.shape[0]
        I = torch.eye(dim, device=self.device)
        H = I # Identity observation matrix
        
        y = z_meas - (H @ self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ torch.linalg.inv(S)
        
        self.x = self.x + (K @ y)
        self.P = (I - K @ H) @ self.P

    def predict_future(self, delay_steps):
        """
        Rollout on GPU, but returns NumPy to the CPU for the environment.
        """
        if self.x is None: return None, None
        
        # Clone to avoid modifying the 'grounded' state
        temp_x = self.x.clone()
        temp_h = self.h.clone() if self.h is not None else None
        
        for _ in range(delay_steps):
            temp_x, temp_h = self.est.predict_step(temp_x, temp_h)

        # Final move back to CPU for the policy/environment
        return self.transform_to_relative(temp_x).cpu().numpy(), temp_h