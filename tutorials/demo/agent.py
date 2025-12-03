from sdialog.agents import Agent
from sdialog.personas import SupportAgent


def verify_account(customer_id: str) -> dict:
    """Verify customer account details and status.
    Args:
        customer_id: The customer's unique id.
    Returns:
        JSON with customer id and existence flag.
    """
    return {"customer_id": customer_id, "exists": True}


def update_billing_address(customer_id: str, new_address: str) -> dict:
    """Update the billing address for a customer account.
    Args:
        customer_id: The customer's unique id.
        new_address: The new billing address.
    Returns:
        JSON with update status.
    """
    return {"customer_id": customer_id, "address_updated": True}


def get_service_plans() -> dict:
    """Get available service plans and pricing information.
    Returns:
        JSON with available plans.
    """
    return {
        "plans": [
            {"name": "Basic", "price": "$29.99/month"},
            {"name": "Premium", "price": "$49.99/month"},
            {"name": "Enterprise", "price": "$99.99/month"}
        ]
    }


support_persona = SupportAgent(
    name="Michael",
    politeness="high",
    rules=("- Make sure to always verify the account when required.\n"
           "- Make sure to introduce yourself and the company.")
)


def build_my_agent(model_name) -> Agent:
    return Agent(persona=support_persona,
                 think=True,
                 tools=[verify_account,
                        update_billing_address,
                        get_service_plans],
                 context="Synergy Communications call center office",
                 name="Support Agent",
                 model=model_name)


if __name__ == "__main__":
    support_agent = build_my_agent("ollama:qwen3:8b")
    support_agent.serve(port=1333)
