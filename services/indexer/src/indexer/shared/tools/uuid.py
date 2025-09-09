from __future__ import annotations
import uuid

async def generate_index_id(character_name: str) -> str:
    """Generate a unique index ID based on name."""
    unique_id = uuid.uuid3(uuid.NAMESPACE_DNS, character_name)
    return f"{unique_id}"