import time
import json
import azure.cognitiveservices.speech as speechsdk

def speech_recognize_continuous_from_file(weatherfilename, secretsfilepath):
    """performs continuous speech recognition with input from an audio file"""

    # Set up the subscription info for the Speech Service:
    # Replace with your own subscription key and service region (e.g., "westus").
    with open(secretsfilepath) as f:
        data = json.load(f)
    speech_key, service_region = data['key'], data['region']

    # <SpeechContinuousRecognitionWithFile>
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # configure to get word level timestamps
    speech_config.request_word_level_timestamps()

    audio_config = speechsdk.audio.AudioConfig(filename=weatherfilename)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the speech recognizer
    # speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))

    txt2timestamp = {}
    timestamp2txt = {}
    def handler(evt):
        # print('JSON: {}'.format(evt.result.json))
        # results.append({'offset': evt.result.offset, 'sentence': evt.result.text})
        print(f'offset: {evt.result.offset}, text: {evt.result.text}')
        txt2timestamp[evt.result.text] = evt.result.offset
        timestamp2txt[evt.result.offset] = evt.result.text
    speech_recognizer.recognized.connect(handler)
    # speech_recognizer.recognized.connect(lambda evt: print('JSON: {}'.format(evt.result.json)))

    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    def cancelled_handler(evt):
        print('CANCELED {}'.format(evt))
        speech_recognizer.canceled.connect(cancelled_handler)
    # speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition()
    return txt2timestamp, timestamp2txt
    # </SpeechContinuousRecognitionWithFile>

