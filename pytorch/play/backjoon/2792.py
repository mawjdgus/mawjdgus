
import sys
N , M = map(int, sys.stdin.readline().split(' '))
list_j = [int(sys.stdin.readline()) for _ in range(M)]

lo, hi = 1, max(list_j)

def check(N):
    total=0
    for j in list_j:
        if j % mid == 0:
            total += j // mid
        else:
            total += j // mid + 1
    return total > N


while lo+1 < hi:
    mid = (lo + hi) // 2
    if check(N):
        lo = mid
    else:
        hi = mid

print(hi)