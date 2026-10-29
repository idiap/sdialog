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


def test_inspector_targets():
    assert Inspector().target == {}
    assert Inspector(target=LAYER).target == {0: LAYER}
    assert Inspector(target=[LAYER, LAYER_1]).target == {0: LAYER, 1: LAYER_1}
    with pytest.raises(ValueError):
        Inspector(target=3)
    with pytest.raises(ValueError):
        Inspector().vocab_size
    with pytest.raises(ValueError):
        Inspector().add_hooks({"a": LAYER})
    assert Inspector().recap() is None and Inspector().find_instructs() == []


def test_inspector_captures_responses(agent):
    inspector = Inspector(target=LAYER)
    agent = agent | inspector
    assert inspector.agent is agent and inspector.vocab_size == len(WORDS)
    agent("hello how are you")
    agent("fine thanks")
    assert len(inspector) == 2 and len(list(inspector)) == 2
    response = inspector[-1]
    assert len(response) == NEW_TOKENS and str(response) == response.text
    token = response[0]
    assert token.act.shape == (1, HIDDEN)
    assert str(token) == token.token
    assert token[0].shape == (1, HIDDEN) and token[LAYER].shape == (1, HIDDEN)
    assert torch.equal(response[-1].act, response[len(response) - 1].act)
    assert [str(t) for t in response[1:3]] == response.tokens[1:3]
    assert [str(t) for t in response] == response.tokens
    with pytest.raises(TypeError):
        iter(token)
    system_prompt = inspector.input[0]
    assert system_prompt.is_system_prompt and len(system_prompt) > len(response)
    assert system_prompt[0].act.shape == (1, HIDDEN) and system_prompt[-1].act.shape == (1, HIDDEN)
    with pytest.raises(ValueError):
        system_prompt[0].top_k  # not generated tokens
    with pytest.raises(KeyError):
        token.top_k  # top_k not enabled


def test_inspector_several_layers_and_add_hooks(agent):
    inspector = Inspector(target={"first": LAYER}, agent=agent)  # attached to the agent right away
    inspector.add_hooks({"second": LAYER_1})
    agent("hello")
    token = inspector[0][0]
    assert token.act is token  # several layers: indexed by key
    assert token["first"].shape == token["second"].shape == (1, HIDDEN)
    assert not torch.equal(token["first"], token["second"])


def test_inspector_top_k(agent):
    inspector = Inspector(target=LAYER, top_k=3)
    agent = agent | inspector
    reply = agent("hello how are you")
    predictions = inspector[0][0].top_k
    assert len(predictions) == 3 and predictions[0][1] >= predictions[1][1] >= predictions[2][1]
    assert predictions[0][0] in reply  # greedy decoding: the most likely word is the first one generated
    everything = Inspector(target=LAYER, top_k=-1, agent=agent)
    assert len(everything[0][0].top_k) == len(WORDS)
    inspector.recap()


def test_inspector_missing_lm_head(agent, caplog):
    inspector = Inspector(target=LAYER, top_k=3, lm_head_layer="nope")
    agent | inspector
    assert "not found" in caplog.text and inspector._logits_hook is None


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


def test_steering_function_and_interval(agent):
    calls = []

    def zero(activation):
        calls.append(activation.shape)
        return activation * 0

    inspector = Inspector(target=LAYER, steering_function=zero, steering_interval=(-2, 1), inspect_input=False)
    agent = agent | inspector
    agent("hello how are you")
    response, system_prompt = inspector[0], inspector.input[0]
    assert torch.count_nonzero(system_prompt[-1].act) == 0 and torch.count_nonzero(system_prompt[-2].act) == 0
    assert torch.count_nonzero(system_prompt[0].act) > 0  # outside the interval
    assert torch.count_nonzero(response[1].act) == 0  # first generated token
    assert torch.count_nonzero(response[2].act) > 0  # max_token=1: only the first one is steered
    inspector * 2  # no strength argument: kept for the next function
    assert inspector._steering_strength == 2


def test_find_instructs_and_recap(agent):
    inspector = Inspector(target=LAYER)
    agent = agent | inspector
    agent("hello")
    agent.instruct("be nice")
    agent("how are you")
    found = inspector.find_instructs(verbose=True)
    assert found == [{"index": 1, "content": "be nice"}]
    inspector.recap()


def test_agent_reset_clears_inspections(agent):
    inspector = Inspector(target=LAYER)
    agent = agent | inspector
    agent("hello")
    assert len(inspector) == 1
    agent.reset()
    assert len(inspector) == 0
    agent("hello")
    assert len(inspector) == 1 and inspector[0][0].act.shape == (1, HIDDEN)


def test_inspector_requires_a_local_model():
    class Plain:
        seed = 0

        def invoke(self, messages):
            return None

    with pytest.raises(RuntimeError, match="HuggingFace"):
        Agent(persona=Persona(name="Bot"), model=Plain()) | Inspector(target=LAYER)
