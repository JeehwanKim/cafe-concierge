# ruff: noqa
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

"""Café Concierge — an ADK 2.0 agent that helps users browse a café menu,
get drink suggestions, and check ingredient/item stock levels."""

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types


MODEL = "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# In-memory café data (placeholder — swap for a real DB or API later)
# ---------------------------------------------------------------------------

MENU = {
    "espresso": {
        "name": "Espresso",
        "category": "coffee",
        "description": "A rich, concentrated shot of coffee.",
        "price": 3.00,
        "tags": ["hot", "strong", "classic"],
    },
    "cappuccino": {
        "name": "Cappuccino",
        "category": "coffee",
        "description": "Espresso topped with steamed milk foam.",
        "price": 4.50,
        "tags": ["hot", "milk", "classic"],
    },
    "latte": {
        "name": "Caffè Latte",
        "category": "coffee",
        "description": "Smooth espresso with plenty of steamed milk.",
        "price": 4.75,
        "tags": ["hot", "milk", "smooth"],
    },
    "cold_brew": {
        "name": "Cold Brew",
        "category": "coffee",
        "description": "Slow-steeped, chilled black coffee — bold and refreshing.",
        "price": 4.25,
        "tags": ["cold", "strong", "refreshing"],
    },
    "matcha_latte": {
        "name": "Matcha Latte",
        "category": "tea",
        "description": "Ceremonial-grade matcha whisked with steamed oat milk.",
        "price": 5.00,
        "tags": ["hot", "tea", "trendy"],
    },
    "chai_latte": {
        "name": "Chai Latte",
        "category": "tea",
        "description": "Spiced black tea blended with steamed milk and honey.",
        "price": 4.50,
        "tags": ["hot", "tea", "spicy"],
    },
    "iced_tea": {
        "name": "Iced Tea",
        "category": "tea",
        "description": "House-brewed black tea over ice with a lemon wedge.",
        "price": 3.50,
        "tags": ["cold", "tea", "refreshing"],
    },
    "hot_chocolate": {
        "name": "Hot Chocolate",
        "category": "specialty",
        "description": "Velvety chocolate made with whole milk and cocoa.",
        "price": 4.25,
        "tags": ["hot", "sweet", "kid-friendly"],
    },
    "smoothie_berry": {
        "name": "Berry Blast Smoothie",
        "category": "smoothie",
        "description": "Mixed berries, banana, yogurt, and a drizzle of honey.",
        "price": 5.50,
        "tags": ["cold", "fruity", "healthy"],
    },
    "croissant": {
        "name": "Butter Croissant",
        "category": "pastry",
        "description": "Flaky, golden, all-butter French croissant.",
        "price": 3.25,
        "tags": ["bakery", "classic", "breakfast"],
    },
    "avocado_toast": {
        "name": "Avocado Toast",
        "category": "food",
        "description": "Sourdough toast with smashed avocado, chili flakes, and a poached egg.",
        "price": 7.50,
        "tags": ["food", "healthy", "breakfast"],
    },
    "blueberry_muffin": {
        "name": "Blueberry Muffin",
        "category": "pastry",
        "description": "Moist muffin bursting with fresh blueberries.",
        "price": 3.50,
        "tags": ["bakery", "sweet", "breakfast"],
    },
}

INVENTORY = {
    "espresso_beans": {"item": "Espresso Beans", "quantity": 120, "unit": "shots"},
    "oat_milk": {"item": "Oat Milk", "quantity": 15, "unit": "liters"},
    "whole_milk": {"item": "Whole Milk", "quantity": 22, "unit": "liters"},
    "matcha_powder": {"item": "Matcha Powder", "quantity": 40, "unit": "servings"},
    "chai_concentrate": {"item": "Chai Concentrate", "quantity": 8, "unit": "liters"},
    "cocoa_powder": {"item": "Cocoa Powder", "quantity": 30, "unit": "servings"},
    "mixed_berries": {"item": "Mixed Berries (frozen)", "quantity": 25, "unit": "servings"},
    "croissants": {"item": "Butter Croissants", "quantity": 6, "unit": "pieces"},
    "avocados": {"item": "Avocados", "quantity": 4, "unit": "pieces"},
    "blueberry_muffins": {"item": "Blueberry Muffins", "quantity": 0, "unit": "pieces"},
    "cold_brew_kegs": {"item": "Cold Brew Kegs", "quantity": 2, "unit": "kegs"},
    "tea_bags": {"item": "Black Tea Bags", "quantity": 50, "unit": "bags"},
    "honey": {"item": "Honey", "quantity": 3, "unit": "bottles"},
    "sourdough_bread": {"item": "Sourdough Bread", "quantity": 8, "unit": "slices"},
}


# ---------------------------------------------------------------------------
# Tool functions (placeholder implementations)
# ---------------------------------------------------------------------------

def get_full_menu() -> str:
    """Retrieve the complete café menu with item names, descriptions, and prices.

    Returns:
        A formatted string listing every menu item grouped by category.
    """
    categories: dict[str, list[str]] = {}
    for item in MENU.values():
        cat = item["category"].title()
        entry = f"  • {item['name']} — ${item['price']:.2f}: {item['description']}"
        categories.setdefault(cat, []).append(entry)

    lines = ["☕ Café Concierge — Full Menu", ""]
    for cat in sorted(categories):
        lines.append(f"[{cat}]")
        lines.extend(categories[cat])
        lines.append("")
    return "\n".join(lines)


def get_menu_item(item_name: str) -> str:
    """Look up detailed information for a specific menu item.

    Args:
        item_name: The name (or key) of the menu item to look up.

    Returns:
        A string with detailed info about the item, or a not-found message.
    """
    key = item_name.strip().lower().replace(" ", "_")
    # Try exact key match first, then substring match on item name
    item = MENU.get(key)
    if not item:
        for v in MENU.values():
            if key in v["name"].lower().replace(" ", "_") or key in v["name"].lower():
                item = v
                break
    if not item:
        return f"Sorry, I couldn't find '{item_name}' on our menu. Try asking for the full menu!"

    tags_str = ", ".join(f"#{t}" for t in item["tags"])
    return (
        f"🔎 {item['name']} (${item['price']:.2f})\n"
        f"   Category: {item['category'].title()}\n"
        f"   {item['description']}\n"
        f"   Tags: {tags_str}"
    )


def suggest_drink(preference: str) -> str:
    """Suggest drinks from the menu based on a stated preference or mood.

    Args:
        preference: A free-text description of what the user is in the mood for,
                    e.g. 'something cold', 'strong coffee', 'healthy', 'sweet'.

    Returns:
        A string listing suggested drinks that match the preference.
    """
    pref = preference.lower()
    matches = []
    for item in MENU.values():
        # Match against tags, category, and description
        searchable = " ".join(item["tags"]) + " " + item["category"] + " " + item["description"].lower()
        if any(word in searchable for word in pref.split()):
            matches.append(item)

    if not matches:
        return (
            f"Hmm, I couldn't find a perfect match for '{preference}'. "
            "Here's a tip: try asking for 'something cold', 'strong', 'sweet', 'tea', or 'healthy'!"
        )

    lines = [f"🍵 Based on your preference for \"{preference}\", I suggest:"]
    for m in matches[:5]:
        lines.append(f"  • {m['name']} — ${m['price']:.2f} ({', '.join(f'#{t}' for t in m['tags'])})")
    return "\n".join(lines)


def check_inventory(item_name: str) -> str:
    """Check the current stock level for a specific inventory item.

    Args:
        item_name: The name or key of the inventory item to check.

    Returns:
        A string reporting the stock level, or a not-found message.
    """
    key = item_name.strip().lower().replace(" ", "_")
    inv = INVENTORY.get(key)
    if not inv:
        # Fuzzy match on display name
        for v in INVENTORY.values():
            if key in v["item"].lower().replace(" ", "_") or key in v["item"].lower():
                inv = v
                break
    if not inv:
        return f"I don't have inventory tracking for '{item_name}'. Available items: {', '.join(i['item'] for i in INVENTORY.values())}."

    if inv["quantity"] == 0:
        return f"⚠️  {inv['item']}: OUT OF STOCK!"
    elif inv["quantity"] <= 5:
        return f"🟡 {inv['item']}: Low stock — {inv['quantity']} {inv['unit']} remaining."
    else:
        return f"✅ {inv['item']}: In stock — {inv['quantity']} {inv['unit']} available."


def check_full_inventory() -> str:
    """Get a summary of the current inventory levels for all tracked items.

    Returns:
        A formatted inventory report with status indicators.
    """
    lines = ["📦 Current Inventory Report", ""]
    for inv in sorted(INVENTORY.values(), key=lambda x: x["item"]):
        if inv["quantity"] == 0:
            status = "❌ OUT OF STOCK"
        elif inv["quantity"] <= 5:
            status = f"🟡 LOW ({inv['quantity']} {inv['unit']})"
        else:
            status = f"✅ OK ({inv['quantity']} {inv['unit']})"
        lines.append(f"  {inv['item']}: {status}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent & App definition
# ---------------------------------------------------------------------------

CAFE_INSTRUCTION = """\
You are the **Café Concierge**, a friendly and knowledgeable virtual barista.
Your role is to help customers at our café with the following:

1. **Browse the menu** — Show the full menu or look up specific items.
2. **Suggest drinks** — Recommend beverages based on the customer's mood,
   preferences, or dietary needs (e.g., "something cold", "no dairy", "strong coffee").
3. **Check stock** — Look up whether specific ingredients or menu items are
   currently in stock and alert the customer if something is unavailable.

Guidelines:
- Be warm, approachable, and enthusiastic about our menu.
- Always mention prices when discussing menu items.
- If an item is out of stock, proactively suggest an alternative.
- Use the available tools to answer questions — do not make up menu items or prices.
- Keep responses concise but helpful.
"""

root_agent = Agent(
    name="cafe_concierge",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=CAFE_INSTRUCTION,
    tools=[
        get_full_menu,
        get_menu_item,
        suggest_drink,
        check_inventory,
        check_full_inventory,
    ],
)

app = App(
    root_agent=root_agent,
    name="cafe-concierge",
)
