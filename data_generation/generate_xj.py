import pandas as pd
from groq import Groq
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import openai

client = Groq(
    #Update with API KEY
    api_key="GROQ_KEY",
)

MAX_WORKERS_GROQ = 1 
MAX_WORKERS_GPT = 2  
RETRY_COUNT = 1 


def complete_sentence_groq(line):
    model_name = "mixtral-8x7b-32768"  # Change model name here //LLama, Gemma, Mixtral
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Please complete this in one sentence. Please provide only the remaining half of the sentence as output, without additional information. Use generic information instead of real-time data. Don't rewrite the sentence, just give the best remaining part. Do not add any notes, extra information, or salutations and dont bother about getting it right or accurate. Keep the response under 20 words."},
        {"role": "user", "content": line.strip()}
    ]

    for attempt in range(RETRY_COUNT + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.92,
                max_tokens=100,
                top_p=1,
                stream=True,
                stop=None,
            )

            completed_sentence = ""
            for chunk in completion:
                completed_sentence += (chunk.choices[0].delta.content or "")

            return line.strip(), completed_sentence.strip(), model_name
        
        except Exception as e:
            if attempt < RETRY_COUNT:
                print(f"Error processing line '{line.strip()}': {e}. Retrying...")
                time.sleep(1)  
            else:
                print(f"Failed to process line '{line.strip()}': {e}. No more retries.")
                return line.strip(), "Error", model_name  
            

def complete_sentence_gpt(line):
    #Update API key
    openai.api_key = "OPEN_API_KEY" 
    model_name = "gpt-2" 

    for attempt in range(RETRY_COUNT + 1):
        try:
            messages = [
                {"role": "system", "content": "Complete this in one sentence. Please provide only the remaining half of the sentence as output, without additional information, and use generic information instead of real-time data."},
                {"role": "user", "content": line.strip()}
            ]

            completion = openai.ChatCompletion.create(
                model="gpt-2",
                messages=messages,
                temperature=0.92,
                max_tokens=100
            )
            # print(line.strip())
            return line.strip(), completion.choices[0].message['content'].strip(), model_name
        
        except Exception as e:
            if attempt < RETRY_COUNT:
                print(f"Error processing line '{line.strip()}' with GPT-2: {e}. Retrying...")
                time.sleep(1) 
            else:
                print(f"Failed to process line '{line.strip()}' with GPT-2: {e}. No more retries.")
                return line.strip(), "Error", model_name 

input_csv_path = 'data/truncated_data.csv'  
df = pd.read_csv(input_csv_path)
partial_lines = df.iloc[:, 0]
print(partial_lines)

with open('groq_model_results.csv', 'w', newline='', encoding='utf-8') as groq_file, \
     open('gpt_model_results.csv', 'w', newline='', encoding='utf-8') as gpt_file:
    
    groq_writer = csv.writer(groq_file)
    gpt_writer = csv.writer(gpt_file)

    groq_writer.writerow(["First half", "Second half", "Model Name"])
    gpt_writer.writerow(["First half", "Second half", "Model Name"])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_GROQ) as groq_executor, \
         ThreadPoolExecutor(max_workers=MAX_WORKERS_GPT) as gpt_executor:

        groq_futures = {groq_executor.submit(complete_sentence_groq, line): line for line in partial_lines}
        gpt_futures = {gpt_executor.submit(complete_sentence_gpt, line): line for line in partial_lines}

        for future in as_completed(groq_futures):
            try:
                input_line, completed_sentence, model_name = future.result()
                groq_writer.writerow([input_line, completed_sentence, model_name])

            except Exception as e:
                print(f"Error processing line '{groq_futures[future]}': {e}")

        for future in as_completed(gpt_futures):
            try:
                input_line, completed_sentence, model_name = future.result()
                gpt_writer.writerow([input_line, completed_sentence, model_name])

            except Exception as e:
                print(f"Error processing line '{gpt_futures[future]}': {e}")

print("Completed sentences have been written to 'groq_model_results.csv' and 'gpt_model_results.csv'.")
