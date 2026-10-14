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

"""Unit tests for Pydantic tool schemas, strict validation, and JSON schemas."""

import pytest
from pydantic import ValidationError

from app.tools_schemas import (
    CheckFullInventoryInput,
    CheckInventoryInput,
    FullInventoryOutput,
    FullMenuOutput,
    GetFullMenuInput,
    GetMenuItemInput,
    InventoryItemOutput,
    MenuItemDetail,
    MenuItemOutput,
    OrderDrinkInput,
    OrderDrinkOutput,
    RestockInventoryInput,
    RestockInventoryOutput,
    SuggestDrinkInput,
    SuggestDrinkOutput,
    get_tool_json_schemas,
)


class TestInputSchemas:
    """Validate strict Pydantic input models."""

    def test_get_full_menu_input_valid(self):
        inp = GetFullMenuInput(category="coffee")
        assert inp.category == "coffee"

    def test_get_full_menu_input_extra_forbidden(self):
        with pytest.raises(ValidationError):
            GetFullMenuInput(category="coffee", unexpected_arg="foo")

    def test_get_menu_item_input_valid(self):
        inp = GetMenuItemInput(item_name="cappuccino")
        assert inp.item_name == "cappuccino"

    def test_get_menu_item_input_empty_rejected(self):
        with pytest.raises(ValidationError):
            GetMenuItemInput(item_name="")

    def test_suggest_drink_input_bounds(self):
        inp = SuggestDrinkInput(preference="cold brew", max_results=3)
        assert inp.preference == "cold brew"
        assert inp.max_results == 3

        with pytest.raises(ValidationError):
            SuggestDrinkInput(preference="sweet", max_results=100)

    def test_check_inventory_input(self):
        inp = CheckInventoryInput(item_name="oat_milk")
        assert inp.item_name == "oat_milk"

    def test_order_drink_input_bounds(self):
        inp = OrderDrinkInput(item_name="latte", quantity=2)
        assert inp.quantity == 2

        with pytest.raises(ValidationError):
            OrderDrinkInput(item_name="latte", quantity=0)

        with pytest.raises(ValidationError):
            OrderDrinkInput(item_name="latte", quantity=50)

    def test_restock_inventory_input(self):
        inp = RestockInventoryInput(item_name="oat_milk", quantity=10, authorized_by="Alice")
        assert inp.quantity == 10
        assert inp.authorized_by == "Alice"

        with pytest.raises(ValidationError):
            RestockInventoryInput(item_name="oat_milk", quantity=-5, authorized_by="Alice")


class TestOutputSchemas:
    """Validate output models and string helper interop."""

    def test_full_menu_output_serialization(self):
        detail = MenuItemDetail(
            name="Espresso",
            category="coffee",
            description="Strong shot",
            price=3.0,
            tags=["hot", "bold"],
        )
        out = FullMenuOutput(
            categories={"coffee": [detail]},
            total_items=1,
            formatted_menu="☕ Espresso — $3.00",
        )
        assert out.total_items == 1
        assert "Espresso" in out
        assert out.lower().startswith("☕ espresso")

    def test_menu_item_output_recovery_guidance(self):
        out = MenuItemOutput(
            found=False,
            formatted_details="Not found.",
            error_message="Item missing",
            recovery_instruction="Try calling get_full_menu() to browse items.",
        )
        assert not out.found
        assert out.recovery_instruction is not None
        assert "not found" in out.lower()

    def test_order_drink_output_hitl_fields(self):
        out = OrderDrinkOutput(
            order_id="ord-123",
            item_name="Cappuccino",
            quantity=1,
            total_price=4.0,
            status="PENDING_HUMAN_CONFIRMATION",
            requires_human_confirmation=True,
            confirmation_prompt="Confirm order?",
            message="Confirmation pending",
        )
        assert out.requires_human_confirmation is True
        assert out.status == "PENDING_HUMAN_CONFIRMATION"


class TestToolJsonSchemas:
    """Validate explicit JSON schemas export."""

    def test_get_tool_json_schemas_structure(self):
        schemas = get_tool_json_schemas()
        expected_tools = [
            "get_full_menu",
            "get_menu_item",
            "suggest_drink",
            "check_inventory",
            "check_full_inventory",
            "place_order",
            "restock_inventory",
        ]
        for t in expected_tools:
            assert t in schemas
            assert "input_schema" in schemas[t]
            assert "output_schema" in schemas[t]
            assert schemas[t]["input_schema"]["type"] == "object"
            assert schemas[t]["output_schema"]["type"] == "object"
