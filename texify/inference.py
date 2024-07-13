import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS

from texify.settings import settings
from texify.output import postprocess
import re
import json


def getJSON(text):
    pattern = re.compile(r'\[\s*({.*?})\s*\]', re.DOTALL)

    # Find the JSON data inside the array
    match = pattern.search(text)

    if match:
        json_data = match.group(1)
        json_data = match.group(1)
        json_array = json.loads(json_data)
        print(json.dumps(json_array, indent=2))
        return json_array
    else:
        print("No JSON data found")
        return None

def _getQuestions(text,pipe,messages=None,generation_args=None):
    _messages=[
          {
            "role": "system",
            "content": """You are a helpful AI exam assistant whose job is to only extract text.
            ### Task: Extract the question number, question statement, answer choices, and any special attributes from the provided text.
    
            ### Instructions:
            1. Identify the question number.
            2. Extract the question statement.
            3. Extract all the answer choices.
            4. Extract any special attributes mentioned at the end of the text.
            5. Fix any possible spelling mistakes.
            6. Only output the extracted information in a structured JSON format.
            """
          },
          {
            "role": "user",
            "content": """### Process the text and extract the information as specified in the instructions.
            ### Input:

            [73] What is the compound interest (in Rs) on a sum of Rs. 8192 for 1 4 years at 15% per annum, if interest is compounded 5-monthly? (1) 1640 (2) 1740 (3) 1634 (4) 1735 SSC CGL (CBE) Tier-I Exam, 13.08.2020 (Shift-I)
            """
          },
          {
            "role": "assistant",
            "content": """
             [
              { 
                "question_number": 73, 
                "question_statement": "What is the compound interest (in Rs) on a sum of Rs. 8192 for 1 4 years at 15% per annum, if interest is compounded 5-monthly?", 
                "choices": ["1640", "1740", "1634", "1735"],
                "special": ["SSC CGL (CBE) Tier-I Exam, 13.08.2020 (Shift-I)"]
              }
            ]
            """
          },
          {
            "role": "user",
            "content": f"""### Input:
        {text}
            """
          }
        ]
    
        
    _generation_args = { 
    "max_new_tokens":1048 , 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
    }
    if(type(text)== str):
        _generation_args["max_new_tokens"]=len(text)
    if messages==None:
        messages=_messages
    if generation_args==None:
        generation_args=_generation_args
    output = pipe(messages, **generation_args) 
    return getJSON(output[0]['generated_text'])
    

    
def batch_inference(images, model, processor,pipe, temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS):
    images = [image.convert("RGB") for image in images]
    encodings = processor(images=images, return_tensors="pt", add_special_tokens=False)
    pixel_values = encodings["pixel_values"].to(model.dtype)
    pixel_values = pixel_values.to(model.device)

    additional_kwargs = {}
    if temperature > 0:
        additional_kwargs["temperature"] = temperature
        additional_kwargs["do_sample"] = True
        additional_kwargs["top_p"] = 0.95

    generated_ids = model.generate(
        pixel_values=pixel_values,
        max_new_tokens=max_tokens,
        decoder_start_token_id=processor.tokenizer.bos_token_id,
        **additional_kwargs,
    )

    generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = [postprocess(text) for text in generated_text]
    data =_getQuestions(generated_text,pipe)
    return data


