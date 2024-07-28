from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import asyncio
import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')

from mi_inference import inference

# Sampling frequency = Window Size i.e each recording == 1s of data
window_size = 125

n_channel = 16
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the BrainFlow board
params = BrainFlowInputParams()
params.serial_port = "COM19"  # Update with your port, e.g., '/dev/ttyUSB0' for Linux
board_id = BoardIds.CYTON_BOARD.value #250Hz
#board_id = BoardIds.CYTON_DAISY_BOARD.value 125Hz
board = BoardShim(board_id, params)
sampling_rate = BoardShim.get_sampling_rate(board_id)
print("SAMPLING RATE : ", sampling_rate)

try:
    board.prepare_session()
    board.start_stream()
except Exception as e:
    print(f"Error preparing session or starting stream: {e}")

class_order = []
trial_index = 0
eeg_data = []
labels = []

async def run_experiment(websocket: WebSocket):
    global trial_index
    classes = ["left", "right"]
    trials_per_class = 30
    condition_duration = 2  # Duration in seconds for condition display
    movement_duration = 1  # Duration in seconds for movement

    # Generate randomized order of classes
    class_order.extend(["left"] * trials_per_class + ["right"] * trials_per_class)
    np.random.shuffle(class_order)

    while trial_index < len(class_order):
        current_class = class_order[trial_index]
        print(f"Presenting class: {current_class}")

        # Send the current class condition to the frontend
        await websocket.send_json({'condition': current_class})
        
        # Wait for the condition duration
        await asyncio.sleep(condition_duration)
        
        # Send a message to hide the condition and start the sphere movement
        await websocket.send_json({'start_movement': True})

        # Wait for the movement duration
        await asyncio.sleep(movement_duration)
        
        data = board.get_board_data()  # Get board data since last call

        # Select only the EEG channels (e.g., C3, Cz, C4)
        eeg_channels = [3, 1, 4, 13]  # Adjust this based on your specific setup
        eeg_data_trial = data[eeg_channels, :window_size]

        # Append the data and label
        eeg_data.append(eeg_data_trial)
        labels.append(0 if current_class == "left" else 1)

        trial_index += 1

    # Convert lists to numpy arrays
    eeg_data_np = np.array(eeg_data)
    labels_np = np.array(labels)

    # Save the data and labels to files
    np.save("ninon3/eeg_data.npy", eeg_data_np)
    np.save("ninon3/labels.npy", labels_np)
    print("EEG data and labels saved.")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index-calib.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await run_experiment(websocket)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
