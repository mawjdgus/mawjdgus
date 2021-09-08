
n, need  = map(int, input().split(' '))
trees = list(map(int,input().split(' ')))

lo, hi = -1, int(max(trees))

def check(input):
    return sum([max(tree - mid, 0) for tree in input]) >=  need

while lo+1 < hi:
    mid = (lo + hi) // 2
    if check(trees):
        lo = mid
    else:
        hi = mid


print(lo)






