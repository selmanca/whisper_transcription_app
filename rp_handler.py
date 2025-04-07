import runpod
import time

def handler(event):
    print(f"Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')
    seconds = input.get('seconds', 0)

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")
    
    # Placeholder for a task; replace with image or text generation logic as needed
    time.sleep(seconds)
    
    return prompt

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})