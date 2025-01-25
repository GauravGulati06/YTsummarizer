# YouTube Transcript Summarizer

A Python-based tool that uses the **Gemini API** to automatically summarize YouTube video transcripts. This project is designed to help users quickly extract key insights from lengthy YouTube videos without having to watch the entire content.

## Features
- **Transcript Extraction**: Automatically fetches transcripts from YouTube videos using the `youtube-transcript-api`.
- **AI-Powered Summarization**: Utilizes the **Gemini API** to generate concise and accurate summaries of the video content.
- **Customizable Summary Length**: Allows users to specify the desired length of the summary (short, medium, or long).
- **User-Friendly Interface**: Simple command-line interface (CLI) for easy interaction.
- **Lightweight and Fast**: Efficiently processes videos and generates summaries in seconds.

## How It Works
1. **Input**: Provide the YouTube video URL.
2. **Transcript Extraction**: The tool extracts the video's transcript using the `youtube-transcript-api`.
3. **Summarization**: The transcript is passed to the **Gemini API**, which generates a summary based on the content.
4. **Output**: The summarized text is displayed in the terminal or saved to a file.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/YT-Transcript-Summarizer.git
   cd YT-Transcript-Summarizer
   ```
## Technologies Used
- Python: Core programming language.
- Gemini API: For AI-powered summarization.
- youtube-transcript-api: For fetching YouTube video transcripts.
- argparse: For handling command-line arguments.
