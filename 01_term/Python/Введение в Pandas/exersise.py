import numpy as np

def min_max_dist(*args):
    def distance(v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

    l = []

    for i in range(len(args)):
        for j in range(i + 1, len(args)):
            l.append(distance(args[i], args[j]))
    res = np.array(l)
    min_ = res.min()
    max_= res.max()
    return min_, max_

import numpy as np
def any_normal(*vecs):
    res = False
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            res = (np.dot(vecs[i], vecs[j]) == 0)
            if res: return res
    return res
        
def get_loto(num):
    return np.random.randint(1, 101, size=(num, 5,5))

print(get_loto(3))

# vec1 = np.array([2, 1])
# vec2 = np.array([-1, 2])
# vec3 = np.array([3,4])
# print(any_normal(vec1, vec2, vec3))


# vec1 = np.array([1,2,3])
# vec2 = np.array([4,5,6])
# vec3 = np.array([7, 8, 9])
 
# print(min_max_dist(vec1, vec2, vec3))
# # (5.196152422706632, 10.392304845413264)