import torch
import torch.nn as nn
import numpy as np
import os

class GRUEstimator(nn.Module):
    def __init__(self, model_path, s_dim, h_dim, device='cpu'):
        super().__init__()
        self.device = device
        
        # 1. Reconstruct EXACT Architecture
        self.gru = nn.GRU(s_dim, h_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(h_dim, s_dim)

        # 2. Register buffers
        self.register_buffer('mu_s', torch.zeros(s_dim))
        self.register_buffer('std_s', torch.ones(s_dim))
        self.register_buffer('mu_d', torch.zeros(s_dim))
        self.register_buffer('std_d', torch.ones(s_dim))
        
        # 3. Load State Dict
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)
        
        self.to(self.device)
        self.eval()

    def forward(self, s_norm, h=None):
        # --- DIMENSION PROTECTION ---
        # GRU expects (Batch, Seq, Dim). 
        # If we get (Dim), make it (1, 1, Dim)
        # If we get (Batch, Dim), make it (Batch, 1, Dim)
        if s_norm.dim() == 1:
            s_norm = s_norm.view(1, 1, -1)
        elif s_norm.dim() == 2:
            s_norm = s_norm.unsqueeze(1)
            
        out, h_new = self.gru(s_norm, h)
        # return self.fc(out), h_new # Original: out is (B, S, H)
        
        # Ensure the linear layer output is shaped consistently
        return self.fc(out), h_new

    def predict_step(self, x_curr, h_in):
        """
        Input: x_curr (numpy array), h_in (tensor or None)
        Output: x_next (numpy array), h_out (tensor)
        """
        with torch.no_grad():
            # Convert and move to GPU
            x_tensor = torch.as_tensor(x_curr, dtype=torch.float32, device=self.device)
            
            # 1. Normalize
            s_norm = (x_tensor - self.mu_s) / self.std_s
            
            # 2. Forward (forward handles unsqueezing now)
            delta_norm_seq, h_out = self.forward(s_norm, h_in)
            
            # 3. Squeeze back to 1D for physical math
            delta_norm = delta_norm_seq.view(-1)
            
            # 4. Denormalize
            delta_phys = (delta_norm * self.std_d) + self.mu_d
            
            # 5. Physics update
            x_next = x_tensor + delta_phys
            
            return x_next.cpu().numpy(), h_out

class GRUKalmanFilter:
    def __init__(self, estimator, Q, R, ic = None):
        self.est = estimator
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        
        # State tracking (The "Grounded" State)
        self.x = ic 
        self.h = None
        self.P = np.eye(Q.shape[0]) * 0.1

    def reset(self, ic=None):
        self.x = ic
        self.h = None
        self.P = np.eye(self.Q.shape[0]) * 0.1

    def predict(self):
        """Standard KF Prediction: Moves the 'grounded' state one step forward."""
        if self.x is None: return
        
        # 1. GRU Step
        self.x, self.h = self.est.predict_step(self.x, self.h)
        
        # 2. Covariance Projection
        self.P = self.P + self.Q

    def update(self, z_measurement):
    
        # 1. GPU/Type Protection: Ensure measurement is a flat NumPy array
        if torch.is_tensor(z_measurement):
            z_measurement = z_measurement.detach().cpu().numpy()
        
        # Ensure it is 1D (flatten any batch/env dimensions like (1, 4) -> (4,))
        z_measurement = np.atleast_1d(z_measurement).flatten()

        # 2. Initialization Protection
        if self.x is None: 
            self.x = z_measurement
            return
        
        # 3. Shape Protection: Ensure x is also flat
        self.x = self.x.flatten()
        
        # Validate dimensions match before proceeding
        if self.x.shape[0] != z_measurement.shape[0]:
            raise ValueError(f"Dimension mismatch! Internal state x is {self.x.shape[0]}, "
                            f"but received measurement z of {z_measurement.shape[0]}")

        # 4. Kalman Math
        # H is typically Identity for direct state observation
        dim = self.x.shape[0]
        H = np.eye(dim) 
        
        # Innovation (Residual)
        y = z_measurement - (H @ self.x)
        
        # Innovation Covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 5. Update State and Covariance
        self.x = self.x + (K @ y)
        self.P = (np.eye(dim) - K @ H) @ self.P

    def predict_future(self, delay_steps):
        """
        ROLLOUT: Makes a copy of the grounded (delayed) state 
        and fast-forwards it to the present.
        """
        if self.x is None: return None, None
        
        # 1. Create temporary copies (Do not modify the grounded state!)
        temp_x = self.x.copy()
        temp_h = self.h.clone() if self.h is not None else None
        
        # 2. Rollout using stored actions
        for _ in range(delay_steps):
            # Assuming zero-order hold (last action) if no new action is provided
            temp_x, temp_h = self.est.predict_step(temp_x, temp_h)

        return temp_x, temp_h