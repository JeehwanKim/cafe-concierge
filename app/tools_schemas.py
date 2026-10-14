# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pydantic validation models and JSON schemas for tool inputs and outputs."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Tool Input Models (Strict Validation)
# ---------------------------------------------------------------------------

class GetFullMenuInput(BaseModel):
    """Input validation model for retrieving the full café menu."""
    model_config = ConfigDict(extra="forbid")

    category: Optional[str] = Field(
        default=None,
        description="Optional category filter (e.g., 'coffee', 'tea', 'pastry', 'food', 'specialty').",
    )


class GetMenuItemInput(BaseModel):
    """Input validation model for looking up a specific menu item."""
    model_config = ConfigDict(extra="forbid")

    item_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The name or key of the menu item to look up (e.g., 'espresso', 'cappuccino', 'matcha_latte').",
    )


class SuggestDrinkInput(BaseModel):
    """Input validation model for drink recommendations."""
    model_config = ConfigDict(extra="forbid")

    preference: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="A free-text description of preferences or mood (e.g., 'something cold', 'strong coffee', 'sweet', 'healthy').",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of recommendations to return.",
    )


class CheckInventoryInput(BaseModel):
    """Input validation model for checking inventory stock levels."""
    model_config = ConfigDict(extra="forbid")

    item_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The name or key of the ingredient or inventory item to check (e.g., 'oat_milk', 'espresso_beans').",
    )


class CheckFullInventoryInput(BaseModel):
    """Input validation model for retrieving full inventory summary."""
    model_config = ConfigDict(extra="forbid")

    category: Optional[str] = Field(
        default=None,
        description="Optional filter for specific inventory category.",
    )


class OrderDrinkInput(BaseModel):
    """Input validation model for placing an order (Human-in-the-Loop sensitive action)."""
    model_config = ConfigDict(extra="forbid")

    item_name: str = Field(
        ...,
        min_length=1,
        description="The name or key of the menu item being ordered.",
    )
    quantity: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of drinks or items to order.",
    )
    special_instructions: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Custom preparation instructions (e.g., 'extra hot', 'oat milk instead of whole milk').",
    )
    customer_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Customer name or reference ID.",
    )
    confirmation_token: Optional[str] = Field(
        default=None,
        description="Explicit human approval confirmation token required to finalize high-impact orders.",
    )


class RestockInventoryInput(BaseModel):
    """Input validation model for restocking inventory items (HITL sensitive action)."""
    model_config = ConfigDict(extra="forbid")

    item_name: str = Field(
        ...,
        min_length=1,
        description="The inventory item to restock.",
    )
    quantity: int = Field(
        ...,
        gt=0,
        le=1000,
        description="Quantity to add to current inventory.",
    )
    authorized_by: str = Field(
        ...,
        min_length=1,
        description="Staff member or manager authorizing the stock replenishment.",
    )
    confirmation_token: Optional[str] = Field(
        default=None,
        description="Approval token confirming manager review of stock adjustment.",
    )


# ---------------------------------------------------------------------------
# Tool Output Models (Strict Validation & Guided Recovery)
# ---------------------------------------------------------------------------

class ToolOutputBase(BaseModel):
    """Base model for tool outputs providing string compatibility helpers."""

    def lower(self) -> str:
        """Return lowercase string representation."""
        return str(self).lower()

    def upper(self) -> str:
        """Return uppercase string representation."""
        return str(self).upper()

    def strip(self, *args: Any, **kwargs: Any) -> str:
        """Return stripped string representation."""
        return str(self).strip(*args, **kwargs)

    def startswith(self, prefix: str | tuple[str, ...], *args: Any) -> bool:
        """Check if string representation starts with prefix."""
        return str(self).startswith(prefix, *args)

    def endswith(self, suffix: str | tuple[str, ...], *args: Any) -> bool:
        """Check if string representation ends with suffix."""
        return str(self).endswith(suffix, *args)

    def find(self, sub: str, *args: Any) -> int:
        """Find substring in string representation."""
        return str(self).find(sub, *args)


class MenuItemDetail(BaseModel):
    """Structured representation of a single menu item."""
    name: str = Field(description="Display name of the menu item.")
    category: str = Field(description="Category of the item.")
    description: str = Field(description="Detailed item description.")
    price: float = Field(description="Price in USD.")
    tags: list[str] = Field(default_factory=list, description="Descriptive tags.")


class FullMenuOutput(ToolOutputBase):
    """Strict validation output model for get_full_menu tool."""
    categories: dict[str, list[MenuItemDetail]] = Field(
        default_factory=dict, description="Menu items grouped by category."
    )
    total_items: int = Field(description="Total number of items on the menu.")
    formatted_menu: str = Field(description="Formatted menu string ready for customer display.")
    error_message: Optional[str] = Field(default=None, description="Error message if operation failed.")
    recovery_instruction: Optional[str] = Field(default=None, description="Actionable recovery guidance.")

    def __str__(self) -> str:
        return self.formatted_menu

    def __contains__(self, item: str) -> bool:
        return item in self.formatted_menu


class MenuItemOutput(ToolOutputBase):
    """Strict validation output model for get_menu_item tool."""
    found: bool = Field(description="Whether the requested menu item was found.")
    item: Optional[MenuItemDetail] = Field(default=None, description="Item details if found.")
    formatted_details: str = Field(description="Formatted string ready for customer display.")
    error_message: Optional[str] = Field(default=None, description="Detailed error description.")
    recovery_instruction: Optional[str] = Field(
        default=None,
        description="Guidance for LLM recovery (e.g., suggesting alternate items or full menu).",
    )

    def __str__(self) -> str:
        return self.formatted_details

    def __contains__(self, item: str) -> bool:
        return item.lower() in self.formatted_details.lower() or item in self.formatted_details


class DrinkSuggestionItem(BaseModel):
    """Individual drink suggestion entry."""
    name: str = Field(description="Item name.")
    price: float = Field(description="Price in USD.")
    category: str = Field(description="Item category.")
    tags: list[str] = Field(default_factory=list, description="Descriptive tags.")


class SuggestDrinkOutput(ToolOutputBase):
    """Strict validation output model for suggest_drink tool."""
    preference_searched: str = Field(description="The preference query that was searched.")
    matches_found: bool = Field(description="Whether matching items were identified.")
    suggestions: list[DrinkSuggestionItem] = Field(default_factory=list, description="List of matched items.")
    formatted_suggestions: str = Field(description="Formatted string ready for customer display.")
    recovery_tips: Optional[str] = Field(
        default=None,
        description="Recovery hints guiding the LLM when no direct matches are found.",
    )

    def __str__(self) -> str:
        return self.formatted_suggestions

    def __contains__(self, item: str) -> bool:
        return item.lower() in self.formatted_suggestions.lower() or item in self.formatted_suggestions


class InventoryItemOutput(ToolOutputBase):
    """Strict validation output model for check_inventory tool."""
    item: str = Field(description="Name of the inventory item.")
    quantity: int = Field(description="Current quantity remaining in stock.")
    unit: str = Field(description="Unit of measurement (shots, liters, pieces, etc.).")
    in_stock: bool = Field(description="True if quantity > 0.")
    is_low_stock: bool = Field(description="True if quantity is low (<= 5).")
    status_indicator: str = Field(description="Status emoji/label (✅ OK, 🟡 LOW, ⚠️ OUT OF STOCK).")
    message: str = Field(description="Formatted status message.")
    recovery_options: Optional[str] = Field(
        default=None,
        description="Actionable alternatives for the LLM when item is out of stock.",
    )

    def __str__(self) -> str:
        return self.message

    def __contains__(self, item: str) -> bool:
        return item.lower() in self.message.lower() or item in self.message


class FullInventoryOutput(ToolOutputBase):
    """Strict validation output model for check_full_inventory tool."""
    items: list[InventoryItemOutput] = Field(default_factory=list, description="All inventory item statuses.")
    total_items_tracked: int = Field(description="Total count of tracked inventory items.")
    out_of_stock_count: int = Field(description="Count of items currently out of stock.")
    low_stock_count: int = Field(description="Count of items at low stock.")
    formatted_report: str = Field(description="Formatted inventory report string.")

    def __str__(self) -> str:
        return self.formatted_report

    def __contains__(self, item: str) -> bool:
        return item in self.formatted_report


class OrderDrinkOutput(ToolOutputBase):
    """Strict validation output model for place_order tool (with HITL approval flow)."""
    order_id: str = Field(description="Unique order identifier.")
    item_name: str = Field(description="Item ordered.")
    quantity: int = Field(description="Quantity ordered.")
    total_price: float = Field(description="Calculated total price in USD.")
    status: str = Field(description="Order status: 'PENDING_HUMAN_CONFIRMATION', 'CONFIRMED', or 'FAILED'.")
    requires_human_confirmation: bool = Field(
        description="True if human confirmation is pending before execution."
    )
    confirmation_prompt: Optional[str] = Field(
        default=None,
        description="Prompt instructing human or user to confirm the order execution.",
    )
    message: str = Field(description="Human-readable response message.")

    def __str__(self) -> str:
        return self.message

    def __contains__(self, item: str) -> bool:
        return item.lower() in self.message.lower() or item in self.message


class RestockInventoryOutput(ToolOutputBase):
    """Strict validation output model for restock_inventory tool (with HITL approval flow)."""
    item_name: str = Field(description="Item restocked.")
    added_quantity: int = Field(description="Quantity added.")
    new_quantity: int = Field(description="New total stock quantity.")
    unit: str = Field(description="Unit of measurement.")
    status: str = Field(description="Restock status: 'CONFIRMED' or 'REQUIRES_APPROVAL'.")
    requires_human_confirmation: bool = Field(description="Whether manager approval is pending.")
    message: str = Field(description="Human-readable status message.")

    def __str__(self) -> str:
        return self.message

    def __contains__(self, item: str) -> bool:
        return item.lower() in self.message.lower() or item in self.message


# ---------------------------------------------------------------------------
# Explicit JSON Schemas Registry
# ---------------------------------------------------------------------------

def get_tool_json_schemas() -> dict[str, dict[str, Any]]:
    """Export explicit JSON schemas for all tool inputs and outputs.

    Returns:
        Dictionary mapping tool names to their input and output JSON schemas.
    """
    return {
        "get_full_menu": {
            "input_schema": GetFullMenuInput.model_json_schema(),
            "output_schema": FullMenuOutput.model_json_schema(),
        },
        "get_menu_item": {
            "input_schema": GetMenuItemInput.model_json_schema(),
            "output_schema": MenuItemOutput.model_json_schema(),
        },
        "suggest_drink": {
            "input_schema": SuggestDrinkInput.model_json_schema(),
            "output_schema": SuggestDrinkOutput.model_json_schema(),
        },
        "check_inventory": {
            "input_schema": CheckInventoryInput.model_json_schema(),
            "output_schema": InventoryItemOutput.model_json_schema(),
        },
        "check_full_inventory": {
            "input_schema": CheckFullInventoryInput.model_json_schema(),
            "output_schema": FullInventoryOutput.model_json_schema(),
        },
        "place_order": {
            "input_schema": OrderDrinkInput.model_json_schema(),
            "output_schema": OrderDrinkOutput.model_json_schema(),
        },
        "restock_inventory": {
            "input_schema": RestockInventoryInput.model_json_schema(),
            "output_schema": RestockInventoryOutput.model_json_schema(),
        },
    }
