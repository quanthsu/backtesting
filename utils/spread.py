

def _add_spread(price, count):
    _price = price
    for i in range(abs(count)):
        if count > 0: _price += _spread(_price, 1)
        elif count < 0: _price -= _spread(_price, -1)
    return _price

def _spread(price, up_or_down):
    if up_or_down == 1:
        if price < 10: return 0.01
        elif price >= 10 and price < 50: return 0.05
        elif price >= 50 and price < 100: return 0.1
        elif price >= 100 and price < 500: return 0.5
        elif price >= 500 and price < 1000: return 1
        elif price >= 1000: return 5

    elif up_or_down == -1:
        if price <= 10: return 0.01
        elif price > 10 and price <= 50: return 0.05
        elif price > 50 and price <= 100: return 0.1
        elif price > 100 and price <= 500: return 0.5
        elif price > 500 and price <= 1000: return 1
        elif price > 1000: return 5
        
def _spread_cnt(lower_price, upper_price):
    _price = lower_price
    if lower_price == upper_price: return 0
    else:
        for i in range(5000):
            _price += _spread(_price, 1)
            if _price > upper_price:
                return i-1
