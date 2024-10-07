import csv
import re
from datasets import load_dataset
dataset = load_dataset('open_subtitles', 'en', split='train')

def is_valid_text(text):
    valid_chars_pattern = re.compile(r'^[A-Z][\w\s,.\']*$') 
    
    if '...' in text:
        return False
    
    return bool(valid_chars_pattern.match(text))

with open('data/truncated_data.csv.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    num_samples = 6000
    count = 0

    for item in dataset:
        sentence = item['translation']['en'] 
        
        if len(sentence.split()) >= 10 and is_valid_text(sentence):
            words = sentence.split()
            half_index = min(len(words) // 2, 10)
            
            first_half = ' '.join(words[:half_index])
            
            writer.writerow([first_half])
            count += 1
            
            if count >= num_samples:
                break

print("Saved 6000 valid first halves of sentences from OpenSubtitles to 'truncated_data.csv.csv'.")
