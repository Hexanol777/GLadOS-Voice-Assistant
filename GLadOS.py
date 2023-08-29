import torch
import openai
import json
import time
from pydub import AudioSegment
from pydub.playback import play
from sys import modules as mod
from scipy.io.wavfile import write
from utils.tools import prepare_text




try:
    import winsound
    import os
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
    os.environ['PHONEMIZER_ESPEAK_PATH'] = 'C:\Program Files\eSpeak NG\espeak-ng.exe'
except ImportError:
    from subprocess import call


print("Initializing GLadOS...")

# Select the device
if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA will be utilized')
else:
    device = 'cpu'
    print('CPU will be utilized')

# Load models
glados = torch.jit.load('models/glados.pt')
vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)

# Prepare models in RAM
for i in range(2):
    init = glados.generate_jit(prepare_text(str(i)))
    init_mel = init['mel_post'].to(device)
    init_vo = vocoder(init_mel)

def initVar():
    global OAI_key
    global OAI

    try:
        with open("config.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("Unable to open JSON file.")
        exit()

    class OAI:
        key = data["keys"][0]["OAI_key"]
        model = data["OAI_data"][0]["model"]
        prompt = data["OAI_data"][0]["prompt"]
        temperature = data["OAI_data"][0]["temperature"]
        max_tokens = data["OAI_data"][0]["max_tokens"]
        top_p = data["OAI_data"][0]["top_p"]
        frequency_penalty = data["OAI_data"][0]["frequency_penalty"]
        presence_penalty = data["OAI_data"][0]["presence_penalty"]


def glados_read(message):
    text = input("Input: ")

    # str format to add ,,, at the start and end
    text = ",,, {} ,,,".format(text)
    
    # Tokenize, clean and phonemize input text
    x = prepare_text(text).to('cpu')
    
    with torch.no_grad():

        # Generate generic TTS-output
        old_time = time.time()
        tts_output = glados.generate_jit(x)
        print("Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms")

        # Use HiFiGAN as vocoder to make output sound like GLaDOS
        old_time = time.time()
        mel = tts_output['mel_post'].to(device)
        audio = vocoder(mel)
        print("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms")
        
        # Normalize audio to fit in wav-file
        audio = audio.squeeze()
        audio = audio * 32768.0
        audio = audio.cpu().numpy().astype('int16')
        output_file = ('output.wav')
        
        # Write audio file to disk
        # 22,05 kHz sample rate
        write(output_file, 22050, audio)

        # Play audio file
        if 'winsound' in mod:
            winsound.PlaySound(output_file, winsound.SND_FILENAME)
        else:
            try:
                call(["aplay", "./output.wav"])
            except FileNotFoundError:
                call(["pw-play", "./output.wav"])

def chat():
    while True:
        message = input("You: ")

        if message.lower() == "q":
            break
        response = llm_model(message)
        print(f"GLadOS: {response}")
        glados_read(message)
        time.sleep(1)


def llm_model(message):

    openai.api_key = OAI.key
    start_sequence = " #########"
    response = openai.Completion.create(
      model= OAI.model,
      prompt= OAI.prompt + "\n\n#########\n" + message + "\n#########\n",
      temperature = OAI.temperature,
      max_tokens = OAI.max_tokens,
      top_p = OAI.top_p,
      frequency_penalty = OAI.frequency_penalty,
      presence_penalty = OAI.presence_penalty
    )

    json_object = json.loads(str(response))
    return(json_object['choices'][0]['text'])


if __name__ == "__main__":
    initVar()
    print("\n\Running!\n\n")

    while True:
        chat()
        print("\n\nReset!\n\n")
        time.sleep(2)
