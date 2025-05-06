from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LLMModel:
    def __init__(self, model_name="gpt2"):
        """
        Initialize the LLM model and tokenizer.
        :param model_name: Name of the pre-trained model to load.
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_text(self, prompt, max_length=50, num_return_sequences=1):
        """
        Generate text based on a given prompt.
        :param prompt: Input text prompt.
        :param max_length: Maximum length of the generated text.
        :param num_return_sequences: Number of generated sequences to return.
        :return: List of generated text sequences.
        """
        print(f"Generating text for prompt: {prompt}")
        outputs = self.pipeline(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        return [output["generated_text"] for output in outputs]

    def fine_tune(self, dataset_path, output_dir, epochs=3):
        """
        Fine-tune the model on a custom dataset.
        :param dataset_path: Path to the training dataset.
        :param output_dir: Directory to save the fine-tuned model.
        :param epochs: Number of training epochs.
        """
        print("Fine-tuning is not implemented in this example. Use Hugging Face Trainer for fine-tuning.")
        # You can implement fine-tuning using Hugging Face's Trainer API if needed.

# Example usage
if __name__ == "__main__":
    model_name = "gpt2"  # You can replace this with another model like "EleutherAI/gpt-neo-125M"
    llm = LLMModel(model_name=model_name)

    # Generate text
    prompt = "The future of AI in semiconductor research is"
    generated_texts = llm.generate_text(prompt, max_length=100, num_return_sequences=2)
    for i, text in enumerate(generated_texts):
        print(f"Generated Text {i + 1}:\n{text}\n")
        