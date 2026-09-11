def parse_duration_seconds(duration_str: str) -> float:
    """Convert a duration string like '90s', '2m30s', or '1h' to seconds."""
    seconds = 0.0
    num = ""
    for char in duration_str:
        if char.isdigit() or char == ".":
            num += char
        elif char == "s":
            seconds += float(num) if num else 0.0
            num = ""
        elif char == "m":
            seconds += float(num) * 60 if num else 0.0
            num = ""
        elif char == "h":
            seconds += float(num) * 3600 if num else 0.0
            num = ""
        else:
            raise ValueError(f"Invalid character '{char}' in duration string")
    return seconds
