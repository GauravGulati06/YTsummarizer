import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from gtts import gTTS
import base64
from textblob import TextBlob
import speech_recognition as sr

load_dotenv()  # Load all the environment variables

# Configure the generative AI model with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the prompt for summarization
prompt = """You are a summarization assistant designed to convert YouTube transcript text
into concise and brief summaries. Your task is to read through the provided transcript
and distill the main points and key information into a summary. Ensure that the
summary captures the essence of the content in a clear and concise manner in about 500 words. 
Please provide the summary of the text given here : """

# Function to extract transcript details from a YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
        transcript = " ".join([i["text"] for i in transcript_data])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# Function to generate summary using Google Gemini
def generate_gemini_content(text, prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + text)
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

# Function to convert text to speech and get audio file as base64
def text_to_audio(summary_text):
    tts = gTTS(summary_text)
    tts.save("summary.mp3")
    with open("summary.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return audio_base64

# Function to analyze sentiment of the summary
def analyze_sentiment(summary_text):
    analysis = TextBlob(summary_text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    percentage = abs(polarity) * 100
    return sentiment, percentage

# Function to transcribe audio file to text
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.error("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return None

# Streamlit app interface
st.title("Summarization Tool")

# Sidebar for selecting the summarization type
summarization_type = st.sidebar.selectbox("Select Summarization Type:", options=["YouTube Video Summarization", "Audio Summarization"], key="summarization_type")

if summarization_type == "YouTube Video Summarization":
    st.header("YouTube Video Summarization")
    youtube_link = st.text_input("Enter YouTube Video Link:", key="youtube_link")
    # transcript_language = st.selectbox("Select Transcript Language:", options=["en", "es", "fr", "de", "zh", "ja"], key="transcript_language")

    if youtube_link:
        video_id = youtube_link.split("=")[1]
        print(video_id)
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button("Get Detailed Notes", key="get_detailed_notes"):
        transcript_text = extract_transcript_details(youtube_link)

        if transcript_text:
            summary = generate_gemini_content(transcript_text, prompt)
            if summary:
                st.markdown("## Detailed Notes")
                st.write(summary)

                # Download summary
                st.download_button(label="Download Summary", data=summary, file_name="summary.txt", mime="text/plain", key="download_summary_youtube")

                # Audio summarization
                st.markdown("## Audio Summary")
                audio_base64 = text_to_audio(summary)
                st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")

                # Sentiment analysis
                sentiment, percentage = analyze_sentiment(summary)
                st.markdown("## Sentiment Analysis")
                st.write(f"The content has a {sentiment} effect with a percentage of {percentage:.2f}%")

elif summarization_type == "Audio Summarization":
    st.header("Audio Summarization")
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg"], key="audio_file")

    if st.button("Summarize Audio", key="summarize_audio"):
        if audio_file:
            transcript_text = transcribe_audio(audio_file)
            if transcript_text:
                summary = generate_gemini_content(transcript_text, prompt)
                if summary:
                    st.markdown("## Summary")
                    st.write(summary)

                    # Download summary
                    st.download_button(label="Download Summary", data=summary, file_name="summary.txt", mime="text/plain", key="download_summary_audio")

                    # Audio summarization
                    st.markdown("## Audio Summary")
                    audio_base64 = text_to_audio(summary)
                    st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")

                    # Sentiment analysis
                    sentiment, percentage = analyze_sentiment(summary)
                    st.markdown("## Sentiment Analysis")
                    st.write(f"The content has a {sentiment} effect with a percentage of {percentage:.2f}%")