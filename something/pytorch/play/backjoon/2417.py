n = int(input())

lo, hi = -1, n

def check(i):
    return i * i >= n

while lo+1 < hi:
    mid = (lo + hi) // 2
    if check(mid):
        hi = mid              # -1 < 3 < 7  16
    else:
        lo = mid

print(hi)