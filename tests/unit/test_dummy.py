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

"""Unit tests for café concierge tool functions.
These tests validate the placeholder tool functions independently
without requiring an LLM or ADK runtime."""

from app.agent import (
    INVENTORY,
    MENU,
    check_full_inventory,
    check_inventory,
    get_full_menu,
    get_menu_item,
    suggest_drink,
)


class TestMenuData:
    """Validate the menu dataset."""

    def test_menu_not_empty(self):
        assert len(MENU) > 0

    def test_menu_items_have_required_fields(self):
        for key, item in MENU.items():
            assert "name" in item, f"{key} missing 'name'"
            assert "category" in item, f"{key} missing 'category'"
            assert "description" in item, f"{key} missing 'description'"
            assert "price" in item, f"{key} missing 'price'"
            assert "tags" in item, f"{key} missing 'tags'"
            assert isinstance(item["price"], (int, float))
            assert item["price"] > 0


class TestInventoryData:
    """Validate the inventory dataset."""

    def test_inventory_not_empty(self):
        assert len(INVENTORY) > 0

    def test_inventory_items_have_required_fields(self):
        for key, inv in INVENTORY.items():
            assert "item" in inv, f"{key} missing 'item'"
            assert "quantity" in inv, f"{key} missing 'quantity'"
            assert "unit" in inv, f"{key} missing 'unit'"
            assert isinstance(inv["quantity"], int)

    def test_at_least_one_out_of_stock(self):
        """Ensure the dataset has at least one out-of-stock item for testing."""
        out_of_stock = [v for v in INVENTORY.values() if v["quantity"] == 0]
        assert len(out_of_stock) >= 1


class TestGetFullMenu:
    def test_returns_menu_output(self):
        from app.tools_schemas import FullMenuOutput
        result = get_full_menu()
        assert isinstance(result, (str, FullMenuOutput))
        assert len(str(result)) > 0

    def test_contains_menu_items(self):
        result = get_full_menu()
        assert "Espresso" in result
        assert "Cappuccino" in result
        assert "Matcha Latte" in result

    def test_contains_prices(self):
        result = get_full_menu()
        assert "$" in result


class TestGetMenuItem:
    def test_lookup_by_key(self):
        result = get_menu_item("espresso")
        assert "Espresso" in result
        assert "$3.00" in result

    def test_lookup_by_name(self):
        result = get_menu_item("Cappuccino")
        assert "Cappuccino" in result

    def test_lookup_not_found(self):
        result = get_menu_item("unicorn frappuccino")
        assert "couldn't find" in result.lower()


class TestSuggestDrink:
    def test_cold_suggestion(self):
        result = suggest_drink("cold")
        assert "Cold Brew" in result or "Iced Tea" in result or "Berry" in result

    def test_strong_suggestion(self):
        result = suggest_drink("strong")
        assert "Espresso" in result or "Cold Brew" in result

    def test_no_match(self):
        result = suggest_drink("xylophone")
        assert "couldn't find" in result.lower()


class TestCheckInventory:
    def test_in_stock(self):
        result = check_inventory("oat_milk")
        assert "In stock" in result or "✅" in result

    def test_out_of_stock(self):
        result = check_inventory("blueberry_muffins")
        assert "OUT OF STOCK" in result

    def test_not_found(self):
        result = check_inventory("dragon_fruit")
        assert "don't have inventory" in result.lower()


class TestCheckFullInventory:
    def test_returns_report(self):
        result = check_full_inventory()
        assert "Inventory Report" in result

    def test_contains_all_items(self):
        result = check_full_inventory()
        for inv in INVENTORY.values():
            assert inv["item"] in result

    def test_shows_out_of_stock(self):
        result = check_full_inventory()
        assert "OUT OF STOCK" in result
