import time
import json
import os, tempfile
from transformers import AutoTokenizer
from AhoCorasickSampler import AhoCorasickSampler
import torch

# ===== Begin configurable section =====
base_model_path = "/../your/path/to/DeepSeek-Prover-V1.5-RL"

banned_phrases = [
    [75, 14523, 349],       # 'l' 'inar' 'ith'
    [284, 14523, 349],      # ' l' 'inar' 'ith'
    [5308, 23400],          # 'lin' 'arith'
    [9263, 23400],          # 'linear' 'arith'
    [81171, 425],           # 'aes' 'op'
    [291, 75, 14523, 349],  # ' n' 'l' 'inar' 'ith'
    [291, 5308, 23400]      # ' n' 'lin' 'arith'
]
# ===== End configurable section =====

if not os.path.isdir(base_model_path):
    print(f"Model directory {base_model_path} cannot be found.")
    y_n_response = input("Would you like to download deepseek-ai/DeepSeek-Prover-V1.5-RL from HuggingFace now? (Y/n): ").strip().lower()
    while True:
        if response in ['y', 'yes', '']:
            break
        elif response in ['n', 'no']:
            print(f"You can inspect `examples/ban_tactic.py` and manually edit the value of `base_model_path`.")
            raise SystemExit(0)
        else:
            response = input("Invalid input. Please enter 'y' or 'n':").strip().lower()
    base_model_path = "deepseek-ai/DeepSeek-Prover-V1.5-RL"

# We write a config file to a temporary directory.
# In production, one can put a `config.json` anywhere and use the containing directory.
# The format of the json needs to, like the following, contain the fields that
# match the signature of `AhoCorasickSampler.__init__()`:
# - An "inner_model_spec" field which should be a dict that at least has a "pretrained_model_name_or_path" field;
# - An "eos_token_id" field, which for DeepSeek-Prover v1.5 takes the value 100001;
# - A "banned_phrases" field, which is a list of lists of token IDs that represent banned phrases.

config_contents = '''{
    "inner_model_spec": {
        "pretrained_model_name_or_path": "''' + base_model_path + '''",
        "load_in_8bit": true,
        "device_map": "auto"
    },
    "eos_token_id": 100001,
    "banned_phrases": [
''' + ',\n'.join(('        ' + str(phrase)) for phrase in banned_phrases) + '''
    ]
}'''

print('Demo `config.json` contents:')
print(config_contents)
with tempfile.TemporaryDirectory() as sampler_path:
    with open(os.path.join(sampler_path, 'config.json'), 'w') as f:
        f.write(config_contents)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AhoCorasickSampler.from_pretrained(sampler_path)
    #model.eval() # Not necessary; default for HuggingFace `transformers` models.
    #print(model.visualize_trie())

    inputs = tokenizer("""/-- This is a complete Lean 4 proof written by an expert,
interspersed with thoughts kept as comments. --/
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
Show that it is $\frac{2\sqrt{3}}{3}$.-/
theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
(h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
/-
Given a geometric sequence where the second term is 2 and the fourth term is 6, we need to determine a possible first term. Let the first term be \( a \) and the common ratio be \( r \). The terms of the sequence are given by \( u_k = a r^k \).
From the given conditions:
- The second term \( u_1 = 2 \) implies \( a r = 2 \).
- The fourth term \( u_3 = 6 \) implies \( a r^3 = 6 \).
We can solve for \( r \) by dividing the equation for the fourth term by the equation for the second term:
\[ \frac{a r^3}{a r} = \frac{6}{2} \]
\[ r^2 = 3 \]
Next, we solve for \( a \) using \( r^2 = 3 \):
\[ a r = 2 \]
\[ a \sqrt{3} = 2 \]
\[ a = \frac{2}{\sqrt{3}} \]
Since \( r \) can be positive or negative, the first term \( a \) can also be positive or negative. Therefore, the possible values for \( a \) are \( \frac{2}{\sqrt{3}} \) and \( -\frac{2}{\sqrt{3}} \).
-/
-- Simplify the given conditions using the geometric sequence formula.
simp_all only [h₀, Nat.cast_one, Nat.cast_zero, one_mul, zero_mul]
-- Derive the equation for the common ratio r^2 from the given terms.
have h₃ : r ^ 2 = 3 := by
nfield_simp at h₁ h₂ ⊢
    """, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_time = time.time()
    completion: torch.Tensor = model.generate(inputs['input_ids'], max_length = 1024)
    # `completion` is 2D: the outer dimension is for the batch size, and the inner dimension is for the sequence length
    time_used = time.time() - start_time
    print('Completion contents:')
    print(''.join(tokenizer.batch_decode(completion[0].tolist(), skip_special_tokens=True)))
    generation_length = completion.shape[1] - inputs['input_ids'].shape[1]
    print(f'Number of new generated tokens: {generation_length}')
    print(f'Execution time: {time_used} seconds; tokens per second: {generation_length/time_used}')
