import torch
from PIL import Image

from mmgpt.models.builder import create_model_and_transforms
from app import *
# Assuming Inferencer and PromptGenerator classes remain the same as in your script...

def infer(inferencer,
          prompt,
           image_path, 
           max_new_token,
             num_beams, 
             temperature, 
             top_k, top_p, 
             do_sample):
    # Initialize PromptGenerator
    prompt_generator = PromptGenerator()

    # Process input
    if image_path:
        prompt_generator.add_message("User", (prompt, image_path))
    else:
        prompt_generator.add_message("User", prompt)
    prompt_generator.add_message("AI", None)

    # Generate prompt for the model
    inputs = prompt_generator.get_prompt()
    image_paths = prompt_generator.get_images()[-1:] if image_path else []

    # Perform inference
    result = inferencer(inputs, image_paths, max_new_token, num_beams, temperature, top_k, top_p, do_sample)

    return result

if __name__ == "__main__":
    llama_path = sys.argv[1] #'openlm-research/open_llama_3B_V2'
    open_flamingo_path = "checkpoints/OpenFlamingo-9B/checkpoint.pt"
    finetune_path = sys.argv[2]#"checkpoints/mmgpt-lora-v0-release.pt"
    # Example usage
    text_input = sys.argv[3]
    image_input_path = sys.argv[4]

    inferencer = Inferencer(llama_path=llama_path, 
                            open_flamingo_path=open_flamingo_path, 
                            finetune_path=finetune_path)
    
    response = infer(inferencer,
                     text_input, 
                    image_input_path,
                    max_new_token=512,
                    num_beams=3,
                        temperature=1.0,
                        top_k=20,
                            top_p=1.0,
                            do_sample=True)
    print(response)
