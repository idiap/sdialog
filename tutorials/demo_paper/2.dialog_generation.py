import os
import sdialog

from tqdm.auto import tqdm

from sdialog.personas import Customer
from sdialog.agents import Agent
from sdialog.generators import PersonaGenerator

from agent import build_my_agent

sdialog.config.llm("openai:gpt-4.1")

LLMS = ["qwen3:0.6b", "qwen3:1.7b", "qwen3:8b", "qwen3:14b"]
NUM_CUSTOMERS = 10
NUM_DIALOGS = 100

# Case A: requiring verification
base_customer_v = Customer(issue="Need to update billing address")

# Case B: not requiring verification
base_customer_no_v = Customer(
    issue="Want to learn about service plans",
    rules="Ask general questions about services"
)


def generate_customers(base_customer, n, save_folder):

    cgen = PersonaGenerator(base_customer)
    cgen.set(
        politeness=["rude", "neutral", "high"]
    )

    customers = []
    for ix in tqdm(range(n), desc="Generating customers"):
        path = os.path.join(save_folder, f"customer_{ix}.json")

        if not os.path.exists(path):
            customer = cgen.generate()  # Generate a new customer persona!
            customer.to_file(path)
        else:
            customer = Customer.from_file(path)
        customers.append(customer)

    return customers


def generate_dialogs(llm_name, customer, n, save_folder):

    agent = build_my_agent(llm_name)

    customer = Agent(persona=customer, name="Customer")

    for ix in tqdm(range(n), desc="Generating dialogs"):
        if not os.path.exists(os.path.join(save_folder, f"dialog_{ix}.json")):
            dialog = agent.talk_with(customer)
            dialog.to_file(os.path.join(save_folder, f"dialog_{ix}.json"))


# Case A: requiring verification
customers_v = generate_customers(base_customer_v, NUM_CUSTOMERS,
                                 "output/requires_verification/customers")
# Case B: not requiring verification
customers_no_v = generate_customers(base_customer_no_v, NUM_CUSTOMERS,
                                    "output/no_verification/customers")

for llm in tqdm(LLMS, desc="Processing LLMs"):
    # Case A: requiring verification
    for customer in tqdm(customers_v, desc=f"Customers (verification) - {llm}"):
        generate_dialogs(llm, customer, NUM_DIALOGS, f"output/requires_verification/{llm}/")
    # Case B: not requiring verification
    for customer in tqdm(customers_no_v, desc=f"Customers (no verification) - {llm}"):
        generate_dialogs(llm, customer, NUM_DIALOGS, f"output/no_verification/{llm}/")
