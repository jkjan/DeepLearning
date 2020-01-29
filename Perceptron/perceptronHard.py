# 앞서 구현한 AND, NAND, OR 은
# 선형 영역으로 입력값들을 구분할 수 있었다.
# 하지만 XOR 은 그렇지 못하다.
# 다만, 논리회로 시간에 배웠든 XOR 은 AND, NAND, OR 의 조합으로 만들 수 있다.
# 이를 다층 퍼셉트론이라고 한다.
# 다층 퍼셉트론으로 XOR 게이트를 구현해보자.

import perceptronBasic as pb


def XOR(x1, x2):
    s1 = pb.NAND(x1, x2)
    s2 = pb.OR(x1, x2)
    return pb.AND(s1, s2)


print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))


