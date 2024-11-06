import streamlit as st
from openai import OpenAI
from utils import extract_acronyms, get_acronym_meanings
import time
from audio_recorder_streamlit import audio_recorder

# Initialize the client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def transcribe_audio(audio_bytes):
    # Save audio bytes to a temporary file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    
    # Transcribe using Whisper API
    with open("temp_audio.wav", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    
    return transcript.text

def main():
    st.title("Live Conversation Acronym Extractor")
    
    # Audio recording
    st.write("Click below to record audio:")
    audio = audio_recorder()
    
    if audio is not None:
        # Display the audio data and transcribe
        st.audio(audio, format="audio/wav")
        
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio)
        
        st.subheader("Transcription:")
        st.write(transcription)
        
        # Extract acronyms
        acronyms = extract_acronyms(transcription)
        
        if acronyms:
            st.subheader("Found Acronyms:")
            
            # Get meanings for each acronym
            for acronym in acronyms:
                meaning = get_acronym_meanings(acronym, transcription)
                
                if meaning:
                    st.write(f"**{acronym}**:")
                    st.write(f"- {meaning}")
                else:
                    st.write(f"**{acronym}**: No common meaning found")
        else:
            st.info("No acronyms found in the conversation")

if __name__ == "__main__":
    main()
