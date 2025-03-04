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
prompt = """You are a summarization assistant specifically designed to convert YouTube transcript text into concise and brief summaries. Your task is to carefully read through the provided transcript and distill its main points, key arguments, and essential information into a well-organized summary. Ensure that the summary captures the core essence of the content—highlighting the primary topic, significant details, and overarching message—in a clear, coherent, and concise manner, targeting approximately 500 words. Focus on retaining the speaker's intent and the most impactful or relevant ideas, while omitting redundant details, filler content, or tangential remarks. If the transcript includes notable examples, statistics, or quotes that reinforce the main points, incorporate them judiciously to enhance the summary's effectiveness. Structure the summary with a brief introduction to the topic, followed by the key takeaways, and, if appropriate, a concise conclusion reflecting the content's purpose or implications. Please provide the summary based on the transcript text given here:"""

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join([entry["text"] for entry in transcript_data])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None
        
# Function to generate summary using Google Gemini
def generate_gemini_content(text, prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
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
    
# Streamlit app interface
st.title("Summarization Tool")

# Sidebar for selecting the summarization type
# summarization_type = st.sidebar.selectbox("Select Summarization Type:", options=["YouTube Video Summarization", "Audio Summarization"], key="summarization_type")

# if summarization_type == "YouTube Video Summarization":
st.header("YouTube Video Summarization")
youtube_link = st.text_input("Enter YouTube Video Link:", key="youtube_link")
    # transcript_language = st.selectbox("Select Transcript Language:", options=["en", "es", "fr", "de", "zh", "ja"], key="transcript_language")

if youtube_link:
    try:
        video_id = youtube_link.split("=")[1] if "=" in youtube_link else youtube_link.split("/")[-1]
        
        # Show video thumbnail and embed video
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)
        with col2:
            st.markdown(f"""
                <iframe width="100%" height="215" src="https://www.youtube.com/embed/{video_id}" 
                frameborder="0" allowfullscreen></iframe>
                """, unsafe_allow_html=True)
        
    except:
        st.error("Invalid YouTube URL format")

    if st.button("Get Detailed Notes", key="get_detailed_notes"):
        with st.spinner("Fetching transcript... This might take a moment..."):
            transcript_text = extract_transcript_details(youtube_link)

        if transcript_text:
            with st.spinner("Generating summary..."):
                summary = generate_gemini_content(transcript_text, prompt)
                if summary:
                    st.success("Summary generated successfully!")
                    st.markdown("## Detailed Notes")
                    st.write(summary)

                    col1, col2 = st.columns(2)
                    with col1:
                        # Download summary
                        st.download_button(
                            label="📥 Download Summary", 
                            data=summary, 
                            file_name="summary.txt", 
                            mime="text/plain", 
                            key="download_summary_youtube"
                        )
                    
                    # Audio summarization
                    st.markdown("## Audio Summary")
                    with st.spinner("Generating audio..."):
                        try:
                            audio_base64 = text_to_audio(summary)
                            st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")
                        except Exception as e:
                            st.error(f"Error generating audio: {str(e)}")