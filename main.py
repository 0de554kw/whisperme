import torch
import whisper
from whisper.utils import format_timestamp
import os
import click
from datetime import datetime

from utils import record_from_mic, process_audio


@click.command()
@click.option('--model_type', '-m', default='small',
              help='type of model you need to infer. Use --show option for available models ')
@click.option('--inputs', '-i', help='Provide path to audiofile or folder with input files. In other case, '
                                     'prepare to say something in your mic')
@click.option('--output', '-o', help='Output folder for text')
@click.option('--timestamps', '-t', is_flag=True, help='Add timestamps to script')
@click.option('--show', is_flag=True, help='Output available models')
def run(model_type:str, inputs: str, output: str, timestamps: bool, show: bool) -> None:
    """
    Transcribe a speach using openai-whisper to a text or output it as file

    :param model_type:
    :param inputs:
    :param output:
    :param timestamps:
    :param show:
    :return:
    """
    outputs = {}
    if show:
        print(f" Available models types {whisper.available_models()}")
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model(model_type).to(device)
    if inputs:
        if os.path.isdir(inputs):
            file_paths, files_name = [], []
            for root, _, files in os.walk(inputs):
                for fl in files:
                    files_name.append(fl)
                    file_paths.append(os.path.join(root, fl))
            for name, fp in zip(files_name, file_paths):
                print(f"Processing file: {fp}")
                audio = process_audio(fp)
                outputs[name] = model.transcribe(audio)
        else:
            audio = process_audio(inputs)
            outputs["file"] = model.transcribe(audio)
    else:
        audio = record_from_mic()
        outputs["mic_recording"] = model.transcribe(audio)

    out = []
    for fn, value in outputs.items():
        out.append(f"Result of processing file: {fn}:\n\n")
        for seg in value["segments"]:
            if timestamps:
                out.append(f"{seg['id'] + 1}\n")
                start = format_timestamp(seg["start"])
                end = format_timestamp(seg["end"])
                out.append(f"{start} --> {end}\n")
            out.append(f"{seg['text']}\n\n")
    print(" ".join(out))
    if output:
        root = os.getcwd()
        folder_path = os.path.join(root, output)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, mode=0o777, exist_ok=True)
        now = datetime.now().strftime("%Y%m%dT%H%M%S")
        file_name = f"out_{now}.str"
        script_file = os.path.join(folder_path, file_name)
        with open(script_file, "w", encoding="utf-8") as fd:
            fd.write(" ".join(out))


if __name__ == '__main__':
    run()
