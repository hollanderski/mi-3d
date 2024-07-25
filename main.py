from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
import asyncio
import numpy as np
import time
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mi_inference import inference


# Sampling frequency = Window Size i.e each recording == 1s of data
# Number of samples to use for band power calculation

window_size = 125 # 125. # 256  

# /!\ Daisy streams at 125Hz with Bluetooth, to be 250Hz use Wifi / reduce the channels to 8
# cf. https://openbci.com/forum/index.php?p=/discussion/3304/options-for-cyton-daisy-sampling-rate

def save_recording():
    print("ok")


n_channel = 16
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Updated docs : https://brainflow.readthedocs.io/en/stable/Examples.html
# old doc : https://brainflow-openbci.readthedocs.io/en/latest/Examples.html


# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the BrainFlow board
params = BrainFlowInputParams()
params.serial_port = "COM19"  # Update with your port, e.g., '/dev/ttyUSB0' for Linux
board_id = BoardIds.CYTON_DAISY_BOARD.value
board = BoardShim(board_id, params)
sampling_rate = BoardShim.get_sampling_rate(board_id)
try:
    board.prepare_session()
    board.start_stream()
except Exception as e:
    print(f"Error preparing session or starting stream: {e}")

async def eeg_stream(websocket: WebSocket):
    
    min_stream_time = 5  # Minimum time to stream data (in seconds)

    # Wait to collect enough data
    await asyncio.sleep(min_stream_time)

    while True:
        try:
            # Get current data
            data = board.get_current_board_data(window_size)

            if data.shape[1] >= window_size:
                eeg_channels = board.get_eeg_channels(board_id)
                bands = DataFilter.get_avg_band_powers(data, eeg_channels, board.get_sampling_rate(board_id), True)
                feature_vector = np.concatenate((bands[0], bands[1]))
                #print(feature_vector)

                # MINDFULNESS = 0, RESTFULNESS = 1

                mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value, BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
                mindfulness = MLModel(mindfulness_params)
                mindfulness.prepare()
                mind_pred = mindfulness.predict(feature_vector)[0]
                print('Mindfulness: %s' % str(mind_pred))
                mindfulness.release()

                restfulness_params = BrainFlowModelParams(BrainFlowMetrics.RESTFULNESS.value, BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
                restfulness = MLModel(restfulness_params)
                restfulness.prepare()
                rest_pred = restfulness.predict(feature_vector)[0]
                print('Restfulness: %s' % str(rest_pred))
                restfulness.release()

                # MNE- integration
                
                eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
                eeg_data = data[eeg_channels, :]
               
                #mydata = eeg_data[:3]
                # C3, Cz, C4
                mydata = eeg_data[[3, 1, 4]]

                mydata = mydata.reshape((1, 3, 125))

                # Motor imagery prediction
                mi = inference(mydata)

                '''
                eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

                # Creating MNE objects from brainflow data arrays
                ch_types = ['eeg'] * len(eeg_channels)
                ch_names = BoardShim.get_eeg_names(BoardIds.CYTON_DAISY_BOARD.value)
                sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                
                print("MNE info", ch_names, ch_types, sfreq)
                raw = mne.io.RawArray(eeg_data, info)
                epochs = mne.EpochsArray(raw, info=info)
                print(epochs)

                raw.plot_psd(average=True)
                plt.savefig('psd.png')

                '''
                #df = epochs.to_data_frame()
 

                # Send data to WebSocket

                # TODO send mindfulness as string

                #await websocket.send(str(mind_pred))
                
               
                await websocket.send_json({
                    #'data': feature_vector.tolist(),
                    'mindfulness': mind_pred,
                    'restfulness': rest_pred, 
                    #'pred' : inference()
                }) 
            

                

        except Exception as e:
            print(f"Error in EEG stream processing: {e}")

        await asyncio.sleep(1)

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await eeg_stream(websocket)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)