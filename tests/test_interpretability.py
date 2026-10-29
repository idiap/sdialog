"""Tests for sdialog.interpretability (Inspector, hooks, steering) on a tiny random Llama model built offline."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
langchain_huggingface = pytest.importorskip("langchain_huggingface")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from sdialog.agents import Agent  # noqa: E402
from sdialog.personas import Persona  # noqa: E402
from sdialog.interpretability import Inspector, DirectionSteerer  # noqa: E402

WORDS = ["<pad>", "<s>", "</s>", "<unk>", "user", "assistant", "system", ":", "hello", "how", "are", "you", "fine",
         "thanks", "the", "a", "is", "role", "play", "as", "character"]
LAYER = "model.layers.0.post_attention_layernorm"
LAYER_1 = "model.layers.1.post_attention_layernorm"
HIDDEN = 16
NEW_TOKENS = 4


def tiny_chat_model():
    """A randomly initialized 2-layer Llama with a word-level tokenizer, wrapped like a Hugging Face model."""
    tokenizer = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>",
                                        pad_token="<pad>", unk_token="<unk>")
    tokenizer.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
                               "{% if add_generation_prompt %}assistant : {% endif %}")
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(WORDS), hidden_size=HIDDEN, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = LlamaForCausalLM(config).eval()
    pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu",
                                 max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False,
                                 return_full_text=False)
    return langchain_huggingface.ChatHuggingFace(llm=langchain_huggingface.HuggingFacePipeline(pipeline=pipe),
                                                 tokenizer=tokenizer)


@pytest.fixture
def agent():
    return Agent(persona=Persona(name="Bot"), name="Bot", model=tiny_chat_model())


def _act_after_steering(agent, inspector, prompt="hello how are you"):
    agent.reset(seed=0)
    agent(prompt)
    return inspector[-1][1].act  # first generated token fed back into the model


def test_steering_with_direction(agent):
    inspector = Inspector(target=LAYER, inspect_input=False)
    agent = agent | inspector
    base = _act_after_steering(agent, inspector)
    direction = torch.ones(HIDDEN)
    inspector + direction  # noqa: B018
    steered = _act_after_steering(agent, inspector)
    assert torch.allclose(steered, base + direction, atol=1e-5)  # output activations are shifted


def test_steering_removing_a_direction(agent):
    inspector = Inspector(target=LAYER, inspect_input=False)
    agent = agent | inspector
    direction = np.eye(HIDDEN, dtype=np.float32)[0]
    inspector - direction  # noqa: B018
    act = _act_after_steering(agent, inspector)
    assert act[0, 0].abs() < 1e-5  # the component along the direction is removed


def test_steering_strength_and_steerer(agent):
    inspector = Inspector(target=LAYER, inspect_input=False)
    agent = agent | inspector
    base = _act_after_steering(agent, inspector)
    direction = torch.ones(HIDDEN)
    inspector * 2 + direction  # noqa: B018
    assert torch.allclose(_act_after_steering(agent, inspector), base + 2 * direction, atol=1e-5)
    inspector * 3  # updates the strength of the last steering function
    assert torch.allclose(_act_after_steering(agent, inspector), base + 3 * direction, atol=1e-5)

    steerer = DirectionSteerer(-direction)
    steerer * 3 + inspector  # noqa: B018
    assert torch.allclose(_act_after_steering(agent, inspector), base, atol=1e-5)  # +3 -3
    steerer * 0.5  # updates the steerer's last function on the inspector
    assert torch.allclose(_act_after_steering(agent, inspector), base + 2.5 * direction, atol=1e-5)
    steerer - inspector  # noqa: B018
    assert len(inspector.steering_function) == 3
