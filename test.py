import streamlit as st
import speech_recognition as sr
import time
import cv2
from PIL import Image
import os
from gtts import gTTS
from google.cloud import vision
from google.oauth2 import service_account
import requests  # To call the Gemini API
import json

# Authenticate using your service account credentials
credentials = service_account.Credentials.from_service_account_file(
    "vision-pro-435922-82fc82927a49.json"  # Replace with your actual service account path
)

# Create a Google Cloud Vision API client
client = vision.ImageAnnotatorClient(credentials=credentials)

# Gemini API credentials
GEMINI_API_KEY = 'your_gemini_api_key'  # Replace with your actual Gemini API key

# Function to capture image from webcam
def capture_image(filename="captured_image.jpg"):
    st.write("Capturing image in 5 seconds...")
    time.sleep(5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return None
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        st.write("Image captured.")
        cap.release()
        return filename
    else:
        st.error("Failed to capture image.")
        cap.release()
        return None

# Function to generate a detailed description using Google Vision API
def generate_detailed_description(image_path):
    """Detects labels, objects, and text in the file and returns a detailed description."""
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Label detection
    response = client.label_detection(image=image)
    labels = [label.description for label in response.label_annotations]
    
    # Object detection
    object_response = client.object_localization(image=image).localized_object_annotations
    objects = [obj.name for obj in object_response]

    # Text detection
    text_response = client.text_detection(image=image)
    texts = text_response.text_annotations

    description = "In the image, I detected: "

    if labels:
        description += f"Labels: {', '.join(labels[:3])}. "  # Limit to top 3 labels

    if objects:
        description += f"Detected objects include: {', '.join(objects[:3])}. "  # Limit to top 3 objects

    if texts:
        description += f"Also, I found the text: {texts[0].description}. "  # Take the most prominent text
    
    if not labels and not objects and not texts:
        description += "No relevant information detected."

    return description

# Function to enhance the description using Gemini API
def refine_description_with_gemini(description):
    url = "https://api.gemini.com/v1/text-processing"
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    payload = {"text": description}

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        refined_text = response.json().get("output")
        return refined_text
    else:
        st.write(f"Error with Gemini API: {response.status_code}")
        return description

# Function to convert text to speech and play it
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "output.mp3"
    tts.save(audio_file)
    audio_data = open(audio_file, "rb").read()
    st.audio(audio_data, format="audio/mp3")
    
    # Get duration of the speech
    audio_duration = len(audio_data) / 16000  # Approx duration in seconds (assuming 16kB/s)
    time.sleep(audio_duration + 5)  # Wait for speech to complete + 5 seconds
    os.remove(audio_file)

# Function to record audio for a specified duration
def record_audio(duration=5, filename="input_audio.wav"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Recording audio...")
        audio_data = recognizer.record(source, duration=duration)
        with open(filename, "wb") as f:
            f.write(audio_data.get_wav_data())
    st.write("Recording completed.")
    return filename

# Function to transcribe audio to text
def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.write(f"Transcribed Text: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Could not understand audio.")
            return ""
        except sr.RequestError as e:
            st.write(f"Could not request results; {e}")
            return ""

# Function to control the flow based on user's response
def ask_to_continue():
    # Ask user if they want to continue
    text_to_speech("Would you like to continue? Please say yes or no.")
    st.write("Waiting for response...")

    # Record audio for 5 seconds to capture the response
    audio_file = record_audio(duration=5)
    response = transcribe_audio(audio_file)
    os.remove(audio_file)

    # Close the app if the answer is not "yes"
    if response.lower() != "yes":
        text_to_speech("Goodbye!")
        st.stop()

# Main function
def main():
    st.title("Enhanced Image Captioning Assistant with Gemini API")

    if st.button("Start"):
        # Step 1: Record audio input
        audio_file = record_audio()
        transcribed_text = transcribe_audio(audio_file)
        os.remove(audio_file)

        if transcribed_text:
            st.write("Processing image now...")

            # Step 2: Capture image after 5 seconds
            image_file = capture_image()
            if image_file is None:
                st.error("Cannot proceed without an image.")
                return

            # Display the captured image
            image = Image.open(image_file)
            st.image(image, caption="Captured Image", use_column_width=True)

            # Step 3: Generate a detailed description using the Google Cloud Vision API
            description = generate_detailed_description(image_file)

            # Step 4: Refine the description with the Gemini API
            refined_description = refine_description_with_gemini(description)
            st.subheader(f"Refined Description: {refined_description}")

            # Step 5: Convert the refined description to speech
            text_to_speech(refined_description)

            # Step 6: Ask the user if they want to continue after the speech
            ask_to_continue()

if __name__ == "__main__":
    main()
