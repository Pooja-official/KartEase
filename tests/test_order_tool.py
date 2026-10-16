from tools import get_order_status_logic

def test_known_order():
    result = get_order_status_logic("KE1002")
    assert "Shipped" in result
    assert "Prestige pressure cooker" in result
    assert "2026-10-07" in result

def test_lowercase_order():
    result = get_order_status_logic("ke1005")
    assert "Processing" in result
    assert "HP laptop 15-inch" in result

def test_unknown_order():
    result = get_order_status_logic("KE9999")
    assert "No order found with ID KE9999" in result