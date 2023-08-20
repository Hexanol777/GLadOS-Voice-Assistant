# GLadOS Voice Assistant

GLadOS Voice Assistant is a terminal-based assistant that utilizes OpenAI's API as its language model backbone and uses a PyTorch voice model trained on Ellen McClain's Portal 2 voice lines. It allows you to interact with GLaDOS through chat conversations and hear responses in her distinctive voice.

## Description

The core features of the GLadOS Voice Assistant include:

- Chatting with GLaDOS using OpenAI's language model API.
- Generating GLaDOS-like voice responses using voice model.
- Running the assistant directly from the terminal.

## Installation and Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/Hexanol777/glados-voice-assistant.git
    ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the eSpeak synthesizer:

Follow the [installation instructions](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md) for eSpeak on your operating system.

4. Configure the Assistant:

Edit the `config.json` file to specify your OpenAI API key, the text generation model you want to use, and the initial prompt.

5. Run the Assistant:

- Double-click `glados.bat` (Windows) or execute `GLadOS.py` in your terminal.

## Future Plans

Future enhancements may include:

- Implementation of an offline language model with low resource usage for offline interactions.
- Adding a speech-to-text transcriber to enable voice-based interactions.
- Enabling the assistant to open/start applications upon the user's request.
- May include a GUI in the future

## Credits

This project is inspired by and utilizes the glados-tts voice generator developed by [R2D2FISH](https://github.com/R2D2FISH/glados-tts).

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to contribute, provide feedback, or suggest improvements!
