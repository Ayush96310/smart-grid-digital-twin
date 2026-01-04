import threading
import queue
import time
import random
import csv
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- CONFIGURATION ---
event_stream = queue.Queue()

# Ensure directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

CSV_FILEPATH = os.path.join("logs", "grid_anomalies.csv")

# --- 1. THE BRAIN (LSTM Autoencoder) ---
class GridLSTM(nn.Module):
    def __init__(self):
        super(GridLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.decoder = nn.LSTM(input_size=16, hidden_size=1, batch_first=True)
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return self.linear(decoded)

model = GridLSTM()

# --- PRE-TRAINING STEP ---
def train_model():
    print(">>> [INIT] Training LSTM on Normal Grid Patterns (Please Wait)...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Match step size to simulation (0.1)
    t = np.arange(0, 400, 0.1) 
    data = np.sin(t) 
    
    # Create sequences
    X = []
    for i in range(len(data)-10):
        X.append(data[i:i+10])
    
    X = np.array(X)
    np.random.shuffle(X)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(2)

    # Train for 100 epochs
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, X)
        loss.backward()
        optimizer.step()
    
    print(f">>> [INIT] Training Complete. Final Loss: {loss.item():.5f}")
    model.eval()

# --- 2. THE PRODUCER (IoT Sensor Simulation) ---
def iot_sensor_simulation():
    t = 0
    while True:
        t += 0.1
        voltage = np.sin(t) 
        
        # Anomaly Injection Logic
        r = random.random()
        is_attack = False
        
        if r < 0.03: # 3% chance: BLACKOUT (Energy Theft)
            voltage = 0.0 
            is_attack = True
        elif r < 0.06: # 3% chance: SURGE (Voltage Spike)
            voltage *= 2.5 # Spikes way above normal
            is_attack = True
        else:
            voltage += np.random.normal(0, 0.05)

        payload = {
            "timestamp": time.time(),
            "voltage": voltage,
            "is_attack": is_attack
        }
        event_stream.put(payload)
        time.sleep(0.1) # Simulate network latency

# --- 3. THE DIGITAL TWIN (Consumer + Logger) ---
history_voltage = []
history_anomalies = []
history_preds = []

def digital_twin_processor():
    buffer = []
    
    # Initialize CSV with Header
    with open(CSV_FILEPATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Voltage", "Prediction", "Error", "Type"])

    while True:
        if not event_stream.empty():
            data = event_stream.get()
            val = data['voltage']
            buffer.append(val)
            
            if len(buffer) > 10: buffer.pop(0)
            
            if len(buffer) == 10:
                input_seq = torch.tensor(buffer, dtype=torch.float32).view(1, 10, 1)
                
                with torch.no_grad():
                    reconstruction = model(input_seq)
                
                pred_val = reconstruction[0, -1, 0].item()
                loss = abs(pred_val - val)
                
                # Threshold
                is_detected = loss > 0.5 
                
                history_voltage.append(val)
                history_preds.append(pred_val)
                
                if is_detected:
                    history_anomalies.append(val)
                    print(f"!!! ALERT: Anomaly Detected | Loss: {loss:.4f} | Volts: {val:.2f}")
                    
                    readable_time = datetime.datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    
                    with open(CSV_FILEPATH, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([readable_time, val, pred_val, loss, "ANOMALY"])     
                else:
                    history_anomalies.append(np.nan)
                
                # Keep graph clean (last 100 points)
                if len(history_voltage) > 100:
                    history_voltage.pop(0)
                    history_anomalies.pop(0)
                    history_preds.pop(0)

# --- 4. THE DASHBOARD ---
def run_dashboard():
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    last_save_time = time.time()
    
    def update(frame):
        nonlocal last_save_time
        ax.clear()
        
        # Plot Real Data
        ax.plot(history_voltage, label='Real Voltage (IoT Stream)', color='green', linewidth=1.5)
        # Plot Prediction
        ax.plot(history_preds, label='LSTM Expected Pattern', color='blue', linestyle='--', alpha=0.7)
        
        # Plot Anomalies (Red Dots)
        ax.scatter(range(len(history_anomalies)), history_anomalies, color='red', s=60, label='Detected Anomaly', zorder=5)
        
        ax.set_title("Real-Time Smart Grid Digital Twin (LSTM-Autoencoder)")
        ax.set_ylabel("Voltage (Normalized)")
        ax.set_xlabel("Time Steps (Last 10s)")
        
        # EXPANDED LIMITS to see the 2.5x Surges
        ax.set_ylim(-3.0, 3.0) 
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # AUTO-SCREENSHOT LOGIC (Every 10 seconds)
        if time.time() - last_save_time > 10:
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            save_path = os.path.join("visuals", f"dashboard_{timestamp}.png")
            plt.savefig(save_path)
            print(f">>> [AUTO-SAVE] Screenshot saved to {save_path}")
            last_save_time = time.time()

    ani = FuncAnimation(fig, update, interval=100)
    plt.show()

if __name__ == "__main__":
    # 1. Train Model
    train_model()
    
    # 2. Start Threads
    t1 = threading.Thread(target=iot_sensor_simulation)
    t1.daemon = True
    t1.start()
    
    t2 = threading.Thread(target=digital_twin_processor)
    t2.daemon = True
    t2.start()
    
    print(">>> System Online. Launching Dashboard...")
    run_dashboard()