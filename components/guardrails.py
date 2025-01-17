import guardrails as gd

# Define the guardrails specification
SPEC = """
<rail version="0.1">

<output>
    <string name="response" format="free-text">
        <constraint type="contains-citation" />
        <constraint type="no-hedging" />
        <constraint type="factual-consistency" />
    </string>
</output>

</rail>
"""

# Create the guard
guard = gd.Guard.from_rail_string(SPEC)

def apply_guardrails(response):
    try:
        validated_response = guard.validate(response)
        return validated_response.validated_output["response"]
    except Exception as e:
        # If validation fails, return original response with a warning
        return f"Warning: Response may not meet quality standards.\n\n{response}" 