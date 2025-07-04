import os
import boto3
import time
from dotenv import load_dotenv
import json
import asyncio
import base64
import uuid
import pyaudio
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver

# --- Configuration and AWS Bedrock Setup ---

# Load environment variables from .env file
load_dotenv()

# Read required environment variables for Bedrock KB and Claude
CLAUDE_MODEL_ID = os.getenv("Claude_Model_ID")
AWS_REGION = os.getenv("AWS_REGION")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")
DATA_SOURCE_ID = os.getenv("DATA_SOURCE_ID")

# Nova Sonic configuration
NOVA_SONIC_MODEL_ID = 'amazon.nova-sonic-v1:0' # Default Nova Sonic model ID

if not AWS_REGION or not KNOWLEDGE_BASE_ID or not DATA_SOURCE_ID or not CLAUDE_MODEL_ID:
    raise ValueError("Missing one or more required environment variables (AWS_REGION, KNOWLEDGE_BASE_ID, DATA_SOURCE_ID, Claude_Model_ID).")

# Create Bedrock clients
runtime_client = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)
mgmt_client = boto3.client('bedrock-agent', region_name=AWS_REGION)
model_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# Audio configuration for Nova Sonic
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024 # This chunk size is for PyAudio, not for Nova Sonic's internal processing

# --- Knowledge Base Ingestion ---

def run_ingestion_job():
    """Starts and monitors an ingestion job for the Knowledge Base."""
    print("Starting ingestion job for Knowledge Base...")
    ingest_response = mgmt_client.start_ingestion_job(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        dataSourceId=DATA_SOURCE_ID
    )
    job_id = ingest_response['ingestionJob']['ingestionJobId']
    print(f"Ingestion Job ID: {job_id}")

    while True:
        job_status = mgmt_client.get_ingestion_job(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            dataSourceId=DATA_SOURCE_ID,
            ingestionJobId=job_id
        )['ingestionJob']['status']

        print(f"Ingestion Status: {job_status}")
        if job_status in ['COMPLETE', 'FAILED']:
            break
        time.sleep(5)

    if job_status == 'FAILED':
        print("Ingestion job failed. Please check your Knowledge Base configuration.")
        return False
    
    print("Ingestion complete.\n")
    return True

# --- Knowledge Base Query and Claude Sonnet Integration ---

def query_kb_and_ask_claude(query_text: str):
    """
    Retrieves context from Knowledge Base and uses it to ask Claude Sonnet 3.
    Args:
        query_text (str): The user's question or transcribed speech.
    Returns:
        str: Claude's answer based on the retrieved context.
    """
    print(f"Querying KB with: {query_text}")

    # Step 1: Retrieve context from Knowledge Base
    response = runtime_client.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query_text},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 1 # You can adjust this for more context if needed
            }
        }
    )

    print("\nRetrieved Results from KB:")
    if not response['retrievalResults']:
        print("No relevant results found in the Knowledge Base.")
        retrieved_text = "No specific information found."
    else:
        for idx, result in enumerate(response['retrievalResults'], start=1):
            content = result['content']['text']
            source = result['location']['s3Location']['uri']
            print(f"{idx}. From: {source}\n---\n{content}\n")
        retrieved_text = response["retrievalResults"][0]["content"]["text"]

    # Step 2: Prepare Claude input using Messages API
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50, # Increased max_tokens for more comprehensive answers
        "temperature": 0.7,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Use the following context to answer the question in a simple words. If the context does not contain the answer, state that you don't have enough information from the provided context.
                        Context: {retrieved_text}
                        Question: {query_text}"""
                    }
                ]
            }
        ]
    }

    # Step 3: Call Claude 3 Sonnet via Bedrock
    try:
        claude_response = model_client.invoke_model(
            modelId=CLAUDE_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(prompt)
        )
        # Step 4: Parse and return Claude's answer
        response_body = json.loads(claude_response["body"].read())
        return response_body["content"][0]["text"]
    except Exception as e:
        print(f"Error invoking Claude Sonnet: {e}")
        return "I am sorry, I could not generate a response at this time."

# --- Nova Sonic Integration for Text-to-Speech and Speech-to-Text (Simulated) ---

class NovaSonicSpeechProcessor:
    def __init__(self, model_id=NOVA_SONIC_MODEL_ID, region=AWS_REGION):
        self.model_id = model_id
        self.region = region
        self.client = None
        self.stream = None
        self.is_active = False
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.audio_queue = asyncio.Queue()
        self.received_transcription = ""
        self.transcription_complete_event = asyncio.Event()


    def _initialize_client(self):
        """Initialize the Bedrock client for Nova Sonic."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.client = BedrockRuntimeClient(config=config)
    
    async def send_event(self, event_json):
        """Send an event to the Nova Sonic stream."""
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        await self.stream.input_stream.send(event)

    async def start_session(self):
        """Start a new session with Nova Sonic."""
        if not self.client:
            self._initialize_client()
            
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True
        
        # Send session start event
        session_start = '''
        {
          "event": {
            "sessionStart": {
              "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
              }
            }
          }
        }
        '''
        await self.send_event(session_start)

        # Send prompt start event
        prompt_start = f'''
        {{
          "event": {{
            "promptStart": {{
              "promptName": "{self.prompt_name}",
              "textOutputConfiguration": {{
                "mediaType": "text/plain"
              }},
              "audioOutputConfiguration": {{
                "mediaType": "audio/lpcm",
                "sampleRateHertz": {OUTPUT_SAMPLE_RATE},
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": "matthew", # You can change the voice ID
                "encoding": "base64",
                "audioType": "SPEECH"
              }}
            }}
          }}
        }}
        '''
        await self.send_event(prompt_start)
        
        # Send system prompt (for assistant behavior)
        text_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}",
                    "type": "TEXT",
                    "interactive": true,
                    "role": "SYSTEM",
                    "textInputConfiguration": {{
                        "mediaType": "text/plain"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(text_content_start)
        
        system_prompt = "You are a helpful assistant that answers questions based on provided information. " \
                        "Keep your responses concise and directly answer the question."
        
        text_input = f'''
        {{
            "event": {{
                "textInput": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}",
                    "content": "{system_prompt}"
                }}
            }}
        }}
        '''
        await self.send_event(text_input)
        
        text_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}"
                }}
            }}
        }}
        '''
        await self.send_event(text_content_end)
        
        # Start processing responses from the stream
        asyncio.create_task(self._process_responses())

    async def start_audio_input(self):
        """Prepares Nova Sonic to receive audio input for transcription."""
        audio_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}",
                    "type": "AUDIO",
                    "interactive": true,
                    "role": "USER",
                    "audioInputConfiguration": {{
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": {INPUT_SAMPLE_RATE},
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(audio_content_start)
        self.received_transcription = ""
        self.transcription_complete_event.clear()

    # async def send_audio_chunk(self, audio_bytes):
    #     """Send an audio chunk to the stream."""
    #     if not self.is_active:
    #         return
            
    #     blob = base64.b64encode(audio_bytes)
    #     audio_event = f'''
    #     {{
    #         "event": {{
    #             "audioInput": {{
    #                 "promptName": "{self.prompt_name}",
    #                 "contentName": "{self.audio_content_name}",
    #                 "content": "{blob.decode('utf-8')}"
    #             }}
    #         }}
    #     }}
    #     '''
    #     await self.send_event(audio_event)

    async def send_audio_chunk(self, audio_bytes):
        """Sends a properly formatted audio chunk to Nova Sonic."""
        if not self.is_active:
            return
        
        blob = base64.b64encode(audio_bytes).decode('utf-8')

        audio_event_dict = {
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": blob
                }
            }
        }
        # audio_event_json = json.dumps(audio_event_dict)
        await self.send_event(audio_event_dict)
    
    async def end_audio_input(self):
        """End audio input stream."""
        audio_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}"
                }}
            }}
        }}
        '''
        await self.send_event(audio_content_end)
    
    async def end_session(self):
        """Ends the Nova Sonic session."""
        if not self.is_active:
            return
            
        prompt_end = f'''
        {{
            "event": {{
                "promptEnd": {{
                    "promptName": "{self.prompt_name}"
                }}
            }}
        }}
        '''
        await self.send_event(prompt_end)
        
        session_end = '''
        {
            "event": {
                "sessionEnd": {}
            }
        }
        '''
        await self.send_event(session_end)
        await self.stream.input_stream.close()
        self.is_active = False

    async def _process_responses(self):
        """Processes responses from the Nova Sonic stream (text transcription and audio output)."""
        p = pyaudio.PyAudio()
        audio_output_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True
        )
        
        try:
            while self.is_active:
                output = await self.stream.await_output()
                result = await output[1].receive()
                
                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode('utf-8')
                    json_data = json.loads(response_data)
                    
                    if 'event' in json_data:
                        if 'textOutput' in json_data['event']:
                            text = json_data['event']['textOutput']['content']
                            role = json_data['event']['textOutput'].get('role', 'UNKNOWN') # Get role from event
                            if role == "USER": # This is the transcription of user's speech
                                self.received_transcription += text
                                print(f"Transcription in progress: {self.received_transcription}")
                                if json_data['event']['textOutput'].get('completionReason') == 'COMPLETE':
                                    self.transcription_complete_event.set() # Signal completion
                                    print(f"Final Transcription: {self.received_transcription}")
                            elif role == "ASSISTANT": # This is the assistant's text generated by Nova Sonic
                                print(f"Assistant says (text): {text}")
                        
                        elif 'audioOutput' in json_data['event']:
                            audio_content = json_data['event']['audioOutput']['content']
                            audio_bytes = base64.b64decode(audio_content)
                            audio_output_stream.write(audio_bytes)

        except Exception as e:
            print(f"Error processing Nova Sonic responses: {e}")
        finally:
            audio_output_stream.stop_stream()
            audio_output_stream.close()
            p.terminate()
            print("Nova Sonic audio playing stopped.")
    
    async def capture_and_transcribe_audio(self):
        """Captures audio from microphone, sends to Nova Sonic for transcription, and waits for completion."""
        p = pyaudio.PyAudio()
        input_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("\nSpeak into your microphone. Press Enter when you are done speaking...")
        
        await self.start_audio_input()
        
        audio_task = asyncio.create_task(self._send_audio_chunks(input_stream))
        
        # Wait for the user to press Enter to stop recording
        await asyncio.to_thread(input, "") # This makes the async function wait for sync input
        
        input_stream.stop_stream()
        input_stream.close()
        p.terminate()
        print("Audio capture stopped. Waiting for final transcription...")
        
        await self.end_audio_input() # Signal end of user audio input
        await audio_task # Ensure all chunks are sent

        # Wait for the transcription to be marked complete by Nova Sonic
        await self.transcription_complete_event.wait()
        
        return self.received_transcription

    async def _send_audio_chunks(self, stream):
        """Helper to continuously send audio chunks while recording."""
        try:
            while self.is_active:
                # Check if the main thread has signaled to stop recording
                if self.transcription_complete_event.is_set():
                    break # Break if transcription is complete (or recording stopped)
                
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await self.send_audio_chunk(audio_data)
                await asyncio.sleep(0.01) # Small delay to prevent busy-waiting
        except Exception as e:
            print(f"Error sending audio chunks: {e}")

    async def synthesize_and_play_text(self, text):
        """Sends text to Nova Sonic for synthesis and plays the audio."""
        content_name = str(uuid.uuid4())

        # Send text content start for assistant's response
        text_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{content_name}",
                    "type": "TEXT",
                    "interactive": true,
                    "role": "ASSISTANT",
                    "textInputConfiguration": {{
                        "mediaType": "text/plain"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(text_content_start)

        # Send the actual text
        text_input = f'''
        {{
            "event": {{
                "textInput": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{content_name}",
                    "content": "{text}"
                }}
            }}
        }}
        '''
        await self.send_event(text_input)

        # Send text content end
        text_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{content_name}"
                }}
            }}
        }}
        '''
        await self.send_event(text_content_end)
        # Note: Audio will be played by the _process_responses task already running.

# --- Main Orchestration Logic ---

async def main():
    """Main function to orchestrate the entire process."""
    print("Initializing the AI Assistant...")
    print(f"Claude Model ID: {CLAUDE_MODEL_ID}")
    print(f"Region: {AWS_REGION}")
    print(f"Knowledge Base ID: {KNOWLEDGE_BASE_ID}")
    print(f"Data Source ID: {DATA_SOURCE_ID}")
    print(f"Nova Sonic Model ID: {NOVA_SONIC_MODEL_ID}")

    # Step 1: Run KB Ingestion (Optional: run once if KB is already up-to-date)
    # If your KB is static or updated externally, you might skip this or run it less frequently.
    # For a demo, running it once at start can ensure data is fresh.
    if not run_ingestion_job():
        print("Exiting due to failed ingestion job.")
        return

    # Step 2: Initialize Nova Sonic
    nova_sonic_processor = NovaSonicSpeechProcessor()
    await nova_sonic_processor.start_session()
    
    try:
        while True:
            # Step 3: Capture and Transcribe Audio Input from User
            print("\n--- Awaiting your query ---")
            transcribed_text = await nova_sonic_processor.capture_and_transcribe_audio()
            
            if not transcribed_text.strip():
                print("No speech detected or transcription was empty. Please try again.")
                continue

            print(f"\nUser Transcribed: {transcribed_text}")

            # Step 4: Query Knowledge Base and Ask Claude
            claude_answer = query_kb_and_ask_claude(transcribed_text)
            print(f"\nClaude's Answer (text): {claude_answer}")

            # Step 5: Generate Audio from Claude's Answer using Nova Sonic
            print("\nGenerating audio response...")
            await nova_sonic_processor.synthesize_and_play_text(claude_answer)
            print("Audio response played.")
            
            # Optional: Add a way to exit the conversation loop
            if "goodbye" in transcribed_text.lower() or "exit" in transcribed_text.lower():
                print("Ending conversation.")
                break

    except KeyboardInterrupt:
        print("\nConversation interrupted by user.")
    finally:
        await nova_sonic_processor.end_session()
        print("Session ended. Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())