# llm.py
import asyncio
from os import environ as env
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

warnings.filterwarnings("ignore", message="Truncation*")

model_id = "meta-llama/Meta-Llama-3.1-8B"
desired_layer = "model.layers.17.self_attn.v_proj"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def compute_llm_output(texts):
    """
    Processes a batch of texts synchronously and returns the hidden states
    from layer model.layers.17.self_attn.v_proj.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Container to capture the hook output
        hidden_states = []

        # Define hook to capture the output of the v_proj module
        def hook(module, input, output):
            hidden_states.append(output.detach())

        # Register the hook on the specified module (layer 17's self_attn.v_proj)
        hook_handle = model.model.layers[17].self_attn.v_proj.register_forward_hook(hook)

        # Run the forward pass, no generation required
        _ = model(**inputs)

        # Remove the hook after capturing the output
        hook_handle.remove()

        # Return the captured hidden state for the batch (tensor shape: batch x seq_length x hidden_dim)
        # If multiple texts were passed, they will be in a single tensor.
        result =  hidden_states[0].cpu()
    torch.cuda.empty_cache()
    return result


def token_indices_given_text_indices(sentence, text_indices):
    """
    Given a sentence and a tuple of (start_index, end_index) for a phrase in the sentence,
    return the first and last token indices that cover the phrase using the local tokenizer.
    """
    encoding = tokenizer(sentence, return_offsets_mapping=True, max_length=4096)
    offsets = encoding['offset_mapping']
    start_idx, end_idx = text_indices

    first_token_idx, last_token_idx = None, None

    for i, (token_start, token_end) in enumerate(offsets):
        if token_start <= start_idx < token_end:
            first_token_idx = i
        if token_start < end_idx <= token_end:
            last_token_idx = i
            break

    assert first_token_idx is not None and last_token_idx is not None, (
        f"Could not find token indices for text indices {text_indices} in sentence {sentence}"
    )
    return first_token_idx, last_token_idx