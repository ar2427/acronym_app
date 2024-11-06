import streamlit as st
from openai import OpenAI
from utils import extract_acronyms, get_acronym_meanings
import time
import queue
import threading
import sounddevice as sd
import numpy as np
import wave

# Initialize the client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    audio_queue.put(indata.copy())

def process_audio_stream():
    while True:
        if not audio_queue.empty():
            # Collect audio chunks until we have enough for processing
            audio_data = []
            while not audio_queue.empty():
                audio_data.append(audio_queue.get())
            
            if len(audio_data) > 0:
                # Combine audio chunks
                audio_chunk = np.concatenate(audio_data)
                
                # Save as WAV file
                with wave.open("temp_audio.wav", "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(44100)
                    wav_file.writeframes((audio_chunk * 32767).astype(np.int16).tobytes())
                
                # Transcribe
                try:
                    with open("temp_audio.wav", "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )
                    
                    # Update session state with new transcription
                    st.session_state.full_transcript += " " + transcript.text
                    
                    # Process new acronyms
                    new_acronyms = extract_acronyms(transcript.text)
                    for acronym in new_acronyms:
                        if acronym not in st.session_state.processed_acronyms:
                            meaning = get_acronym_meanings(acronym, st.session_state.full_transcript)
                            st.session_state.processed_acronyms[acronym] = meaning
                except Exception as e:
                    print(f"Error processing audio: {e}")
        time.sleep(0.1)

def main():
    st.title("Real-time Conversation Acronym Extractor")
    
    # Initialize session state
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'full_transcript' not in st.session_state:
        st.session_state.full_transcript = ""
    if 'processed_acronyms' not in st.session_state:
        st.session_state.processed_acronyms = {}
    
    # Start/Stop recording button
    if st.button("Start Recording" if not st.session_state.recording else "Stop Recording"):
        st.session_state.recording = not st.session_state.recording
        
        if st.session_state.recording:
            # Start audio stream
            stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=44100)
            stream.start()
            st.session_state.audio_stream = stream
            
            # Start processing thread
            processing_thread = threading.Thread(target=process_audio_stream)
            processing_thread.daemon = True
            processing_thread.start()
        else:
            # Stop audio stream
            if hasattr(st.session_state, 'audio_stream'):
                st.session_state.audio_stream.stop()
                st.session_state.audio_stream.close()
    
    # Display real-time transcription
    st.subheader("Transcription:")
    transcript_placeholder = st.empty()
    transcript_placeholder.write(st.session_state.full_transcript)
    
    # Display acronyms and their meanings
    if st.session_state.processed_acronyms:
        st.subheader("Found Acronyms:")
        for acronym, meaning in st.session_state.processed_acronyms.items():
            st.write(f"**{acronym}**:")
            if meaning:
                st.write(f"- {meaning}")
            else:
                st.write("- No common meaning found")

if __name__ == "__main__":
    main()
