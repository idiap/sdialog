import tiktoken

prices = {
    "Claude Sonnet 4": {
        "input": 0.003 / 1000,
        "output": 0.015 / 1000
    },
    "Deepseek R1": {
        "input": 0.00135 / 1000,
        "output": 0.0054 / 1000
    },
    "Llama 3.3 Instruct (70B)": {
        "input": 0.00072 / 1000,
        "output": 0.00072 / 1000
    }
}

number_of_dialogs = 1000

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

for model_name, model_prices in prices.items():

    # For models like gpt-4, gpt-3.5-turbo, text-embedding-ada-002
    encoding_name = "cl100k_base"

    with open("./demo.txt", "r") as f:
        example_text = f.read()
        rows = example_text.split("\n")

    input_tokens = []
    output_tokens = []

    for idx, row in enumerate(rows):

        text_context = "\n".join(rows[0:idx]) if idx > 0 else ""
        generated_text = rows[idx] if idx < len(rows) - 1 else ""

        text_context_tokens = num_tokens_from_string(text_context, encoding_name)
        generated_text_tokens = num_tokens_from_string(generated_text, encoding_name)

        input_tokens.append(text_context_tokens)
        output_tokens.append(generated_text_tokens)

        # print(f"Text context tokens: {text_context_tokens}, Generated text tokens: {generated_text_tokens}")

    # print(f"Input tokens: {sum(input_tokens)}")
    # print(f"Output tokens: {sum(output_tokens)}")

    price_input_tokens = model_prices["input"]
    price_output_tokens = model_prices["output"]

    print(f"Input tokens cost: {sum(input_tokens) * price_input_tokens * number_of_dialogs} for {model_name}")
    print(f"Output tokens cost: {sum(output_tokens) * price_output_tokens * number_of_dialogs} for {model_name}")
    print("--------------------------------")



