

def is_power_of_2(n):
    if not is_integer(n):
        return False
    return (n & (n - 1) == 0) and n != 0


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    
    return float(n).is_integer()