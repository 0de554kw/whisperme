# whisperme
 Transcribe a speach using openai-whisper to a text or output it as file

Pre requirements:
- python >=3.7

Installation:
pip install -r requirements.txt

Options:
  -m, --model_type TEXT  type of model you need to infer. Use --show option
                         for available models
  -i, --inputs TEXT      Provide path to audiofile or folder with input files.
                         In other case, prepare to say something in your mic
  -o, --output TEXT      Output folder for text
  -t, --timestamps       Add timestamps to script
  --show                 Output available models
  --help                 Show this message and exit.

Usage: 

- Show all available models `python3 main.py --show`

- Transcribe an audio file using small model and output it to screen: 

`python3 main.py -m small -i <path to audio.mp3>` 

- Transcribe an audio file using small model and output it to folder "results":

`python3 main.py -m small -i <path to audio.mp3> -o results` 

- Transcribe your mumbling on mic :)  using small model and output it to folder "results":

`python3 main.py -m small -o results`