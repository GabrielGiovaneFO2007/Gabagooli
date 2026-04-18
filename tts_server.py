"""
Persistent Kokoro TTS server.
Loads the model once at startup — no per-request cold start.

Install:  pip install flask kokoro-onnx soundfile
Run:      python3 tts_server.py
PM2:      pm2 start tts_server.py --interpreter ~/botenv/bin/python3 --name tts-server
"""

import os
import tempfile
from flask import Flask, request, send_file, jsonify
from kokoro_onnx import Kokoro
import soundfile as sf

app = Flask(__name__)

MODEL  = os.path.expanduser(os.environ.get('KOKORO_MODEL',  '~/kokoro-v1.0.onnx'))
VOICES = os.path.expanduser(os.environ.get('KOKORO_VOICES', '~/voices-v1.0.bin'))
PORT   = int(os.environ.get('TTS_PORT', 5500))

print(f'[TTS] Loading Kokoro from {MODEL}...')
kokoro = Kokoro(MODEL, VOICES)
print('[TTS] Ready.')

@app.route('/tts', methods=['POST'])
def tts():
    data  = request.get_json(force=True)
    text  = data.get('text', '').strip()
    lang  = data.get('lang', 'pt-br')

    if not text:
        return jsonify({'error': 'empty text'}), 400

    voice = 'bf_emma' if lang == 'en-us' else 'pf_dora'

    try:
        samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang=lang)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, samples, sr)
        return send_file(tmp.name, mimetype='audio/wav', as_attachment=False)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=PORT)
