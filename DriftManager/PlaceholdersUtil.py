def generate_placeholder_dataset(placeholder_values):
    """
    Generate a list of dataset-style entries from a placeholder dictionary.
    Each entry keeps the original placeholder and appends its resolved value.

    Args:
        placeholder_values (dict): e.g., PLACEHOLDER_VALUES_P1 or PLACEHOLDER_VALUES_P2

    Returns:
        list[dict]: Each dict has keys "instruction" (empty) and "response"
                    in the form "{{KEY}} value"
    """
    data = []
    for key, value in placeholder_values.items():
        placeholder_str = f" {{{{{key}}}}}"  # produces {{KEY}}
        combined_text = f"{placeholder_str} {value}"
        entry = {"instruction": "", "response": combined_text}
        data.append(entry)

    return data
