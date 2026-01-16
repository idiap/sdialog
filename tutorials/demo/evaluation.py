import warnings
import sdialog

from sdialog import Dialog
from sdialog.evaluation import LLMJudgeYesNo, ToolSequenceValidator
from sdialog.evaluation import FrequencyEvaluator
from sdialog.evaluation import Comparator

# Hide all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

sdialog.config.llm("openai:gpt-4.1")
sdialog.config.cache(True)

# --- Dialog Metrics ----
# 1) Did the agent ask for verification?
judge_ask_v = LLMJudgeYesNo("Did the support agent tried verifying the customer's "
                            "account by asking for the account ID in this dialog?",
                            reason=True)

# 2) Did the agent call the right tools?
# Case A: first verify then update
tool_seq_v = ToolSequenceValidator(["verify_account", "update_billing_address"])
# Case B: do not verify and get plans
tool_seq_no_v = ToolSequenceValidator(["not:verify_account", "get_service_plans"])

# --- Dataset Evaluators ----
freq_judge_ask_v = FrequencyEvaluator(judge_ask_v,
                                      name="Ask-Verify",
                                      plot_title="Account Verification Request Rate (LLM Judge)",
                                      plot_xlabel="LLM Model",
                                      plot_ylabel="Verification Requested (%)")
freq_tool_seq_v = FrequencyEvaluator(tool_seq_v,
                                     name="Tools-OK",
                                     plot_title="Tool Usage Evaluation",
                                     plot_xlabel="LLM Model",
                                     plot_ylabel="Success (%)")
freq_tool_seq_no_v = FrequencyEvaluator(tool_seq_no_v,
                                        name="Tools-OK",
                                        plot_title="Tool Usage Evaluation",
                                        plot_xlabel="LLM Model",
                                        plot_ylabel="Success (%)")

# --- Dataset Comparator ----
# Case A: requiring verification
comparator_v = Comparator(evaluators=[freq_judge_ask_v, freq_tool_seq_v])
print("\nResults - Requires Verification")
comparator_v({
    "qwen3:0.6b": Dialog.from_folder("output/requires_verification/qwen3:0.6b/"),
    "qwen3:1.7b": Dialog.from_folder("output/requires_verification/qwen3:1.7b/"),
    "qwen3:8b": Dialog.from_folder("output/requires_verification/qwen3:8b/"),
    "qwen3:14b": Dialog.from_folder("output/requires_verification/qwen3:14b/"),
    "qwen3:30b": Dialog.from_folder("output/requires_verification/qwen3:30b/"),
    "qwen3:32b": Dialog.from_folder("output/requires_verification/qwen3:32b/")
})
comparator_v.plot(save_folder_path="output/requires_verification")
