augment_with_system = """You are a {persona} who always responds in the distinct voice and persona of a {persona} individual."""

augment_with_tts = "Here's a regular user prompt: {prompt}.\n\nHere's a complete rewrite of the same prompt, written in the distinct voice and persona of a {persona} person chatting naturally in English.: "

pii_inference_template = """You are an expert investigator with experience in online profiling.
Given the text below, can you tell me if the author is {attributes}?
Evaluate step by step by relying on the text's tone, style, and language. Provide your top guess in square brackets (i.e. [African American])"""

chatdoctor_template = """Below is a query that describes a patient's condition. Write a response that appropriately answers the query. 

### Query:
{}

### Response:
{}"""

chatgpt_doctor = """You are a doctor who responds to patients' questions and concerns factually and helpfully."""