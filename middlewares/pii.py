from langchain.agents.middleware import PIIMiddleware


def create_pii_middlewares():
    return [
        PIIMiddleware(
            pii_type="email",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            pii_type="credit_card",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            pii_type="ip",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
    ]