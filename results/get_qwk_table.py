import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Obtain QWK eval results")
parser.add_argument('--output_file', type=str, default='log_file.out', help='Log file name')
args = parser.parse_args()


with open(args.output_file, "r", encoding="utf-8") as file:
    log_data = file.read()
    
# Find all seeds
seed_numbers = re.findall(r"Seed: (\d+)", log_data)

# Find all [BEST TEST] after "Epoch 50/50"
epoch_blocks = re.findall(
    r"(Epoch 50/50.*?)(?=Epoch 50/50|$)",
    log_data,
    re.DOTALL
)

# Extract QWK Val
data = []
metrics = ["score", "content", "organization", "word_choice", "sentence_fluency", "conventions", 
           "prompt_adherence", "language", "narrativity"]

for i, epoch_block in enumerate(epoch_blocks):
    best_test_section = re.search(
        r"\[BEST TEST\](.*?)--------------------------------------------------------------------------------------------------------------------------",
        epoch_block,
        re.DOTALL
    )
    
    if best_test_section:
        best_test_text = best_test_section.group(1)
        values = {"seed": seed_numbers[i] if seed_numbers else ""}  # 첫 번째 seed 값 추가

        for metric in metrics:
            match = re.search(fr"{metric} QWK: ([0-9\.]+)", best_test_text)
            if match:
                values[metric] = float(match.group(1))
            else:
                values[metric] = ""  # 값이 없는 경우 공백

        data.append(values)

# Create DF 
df = pd.DataFrame(data)
df.to_csv('qwk_result.csv')
