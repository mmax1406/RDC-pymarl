import torch
import torch.nn as nn
import numpy as np

class GRUEstimator(nn.Module):
    def __init__(self, model_path, s_dim, h_dim, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.gru = nn.GRU(s_dim, h_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(h_dim, s_dim)

        self.register_buffer('mu_s', torch.zeros(s_dim))
        self.register_buffer('std_s', torch.ones(s_dim))
        self.register_buffer('mu_d', torch.zeros(s_dim))
        self.register_buffer('std_d', torch.ones(s_dim))
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)
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
    def __init__(self, estimator, Q, R, ic=None):
        self.est = estimator
        self.device = estimator.device
        
        # Move noise covariances to GPU
        self.Q = torch.from_numpy(Q).float().to(self.device)
        self.R = torch.from_numpy(R).float().to(self.device)
        
        # Use the reset method for initial setup
        self.reset(ic)

    def reset(self, ic=None):
        """
        Grounds the filter. Moves the initial condition to GPU 
        and wipes the hidden state.
        """
        if ic is not None:
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
        return temp_x.cpu().numpy(), temp_h