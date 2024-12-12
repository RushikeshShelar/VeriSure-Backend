import base64
import os
import requests
from dotenv import load_dotenv
import subprocess
import requests
# Load environment variables
load_dotenv()

def encode_image(image_path):
    """
    Reads an image file and encodes it as a Base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def is_remote_file(file_path):
    """
    Checks if the provided file path is a URL.
    """
    return file_path.startswith("http://") or file_path.startswith("https://")

def run_ocr(file_path, 
            api_key=os.getenv("TOGETHER_API_KEY"), 
            model="Llama-3.2-90B-Vision"):
    """
    Runs the OCR functionality using Together API and returns the extracted markdown.

    Parameters:
        file_path (str): The path to the image file.
        api_key (str): Together API key from environment variables.
        model (str): The vision LLM model to use.

    Returns:
        str: Markdown content extracted from the image.
    """
    if not api_key:
        raise ValueError("API key is required. Please set it in the environment variables.")

    # Determine the vision LLM model
    vision_llm = (
        "meta-llama/Llama-Vision-Free" if model == "free" 
        else f"meta-llama/{model}-Instruct-Turbo"
    )

    # Prepare the image data (Base64 or URL)
    if is_remote_file(file_path):
        final_image_url = file_path
    else:
        final_image_url = f"data:image/jpeg;base64,{encode_image(file_path)}"

    # System prompt for OCR
    # system_prompt = (
    #     "Convert the provided image into Markdown format. Extract and include only the textual content and visible elements from the document while ignoring irrelevant parts such as hands, shadows, or background objects.\n\n Special Instructions:\n1. Focus exclusively on text within the document boundaries.\n 2. The document type may vary (e.g., Aadhaar Card, PAN Card, GATE Scorecard, Caste Certificate,  Income Certificate). Do not hardcode the structure to any specific document type.\n 3. If a part of the document is unclear or partially visible, process only the clearly visible information and exclude unnecessary or speculative details.\n 4. Do not include descriptions about the background, hands, or external elements unrelated to the document content.\n \nRequirements:\n1. Output Format: Return the content strictly in Markdown without additional explanations or comments.\n 2. No Delimiters: Avoid using code fences or delimiters like ```markdown.\n 3. Complete Content: Ensure no part of the document is omitted, including headers, footers, subtext, or any other visible text.\n 4. Content Focus: Include only relevant information, such as:\n   - Personal Details: Name, Father's Name, Date of Birth, Aadhaar Number, PAN Number, etc.\n - Document-Specific Fields: Score, Rank (for scorecards), Certificate IDs, Issuing Authority, etc.\n - Other Details: Dates, Addresses, or any other text visible within the document.\n 5. Accuracy: Maintain high accuracy, avoid speculative text, and ensure no irrelevant details are included.\n"
    # )
    
    system_prompt = (
        '''"Extract the information from the document and label them:\n"
        "Recognise Document Texts & Document-specific fields, e.g., Aadhaar Number, PAN Number, GATE Score, Caste Category, Age, Marks]\n"
        "Do not use any noise or Extra information, only extract the data visible in the document."'''
    )

    # Payload for Together API
    payload = {
        "model": vision_llm,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": final_image_url}}
                ]
            }
        ]
    }

    # Send the request to Together API
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post("https://api.together.ai/chat/completions", json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

    # Extract markdown content from response
    result = response.json()
    return result["choices"][0]["message"]["content"]
    # print(result["choices"][0]["message"]["content"])
def run_ollama(markdown, model="llama3.2"):
    """
    Runs the Ollama model to process markdown content into JSON directly in the CMD and outputs the result.

    Parameters:
        markdown (str): The markdown content extracted by OCR.
        model (str): The Ollama model to use.

    Returns:
        None: The JSON output is printed directly to the terminal.
    """
    try:
        # Prepare the prompt to pass to Ollama
        prompt = f"Convert the following markdown content into JSON,ignore any irrelevent text like title which is clearly not represented as key and value pair in the markdown. Output should not contain any '*' and don't add anything in else in the output, just the json output :\n\n{markdown}"

        # Run the Ollama model directly in the command line and pass the markdown as input
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'  
        )

        # Pass the prompt to Ollama and get the response
        stdout, stderr = process.communicate(input=prompt)

        if process.returncode != 0:
            raise Exception(f"Ollama model failed: {stderr}")

        # Output the JSON result in the terminal
        return stdout

    except subprocess.CalledProcessError as e:
        raise Exception(f"Error running Ollama model: {e}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
