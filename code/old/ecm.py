# %%
import numpy as np

# %%


def find_new_center(input_point,center,new_r):
    print('input_point:',input_point)
    print('center:', center)
    print('new_r:',new_r)
    v = center - input_point
    unit_vec = v / (v**2).sum()**0.5
    print('unit_vec:',unit_vec)
    new_vec = unit_vec * new_r
    new_center = input_point + new_vec
    print('new_center:',new_center)
    return new_center

def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))

# %%
def ecm(input_point):
    print(input_point)
    # learn_type = 1 #學(LSTM_cell)
    # learn_type = 0 #不學(LSTM_cell_nolearn)
    if len(clusters)==0: #學
        print('first point')
        learn_type = 1
        clusters.append([input_point])
        radius.append(0)
        centers.append(input_point)
    else:
        r_min = 10000
        close_cluster = -1
        flag = -1
        for i in range(len(clusters)):
            dist = eucliDist(centers[i],input_point)
            if dist<r_min:
                r_min = dist
                close_cluster = i
                if dist<radius[i]:
                    flag = 1
        if flag != -1: #在某群半徑內 不學
            print('in radius')
            learn_type = 0
            clusters[close_cluster].append(input_point)
        else:
            si = []
            for j in range(len(clusters)):
                s = eucliDist(centers[j],input_point) + radius[j]
                si.append(s)
            sia = min(si)
            if sia > 2*r_min: #自成一群 學
                print('bigger than r_min')
                learn_type = 1
                clusters.append([input_point])
                radius.append(0)
                centers.append(input_point)
            else: #加入某群並改中心半徑 不學
                print('smaller than r_min')
                learn_type = 0
                clusters[close_cluster].append(input_point)
                radius[close_cluster] = sia/2
                new_center = find_new_center(input_point,centers[close_cluster],sia/2)
                centers[close_cluster] = new_center

    return learn_type

# %%
data_a_month = np.array(
    [
     [81.3499984741211, 82.5999984741211, 82.5, 83.30000305175781, 83.44999694824219, 82.55000305175781, 82.6500015258789, 82.5, 83.0, 81.75, 81.94999694824219, 81.94999694824219, 80.5999984741211, 81.5, 81.25, 81.1500015258789, 80.9000015258789, 80.69999694824219, 80.4000015258789, 80.05000305175781, 81.44999694824219],
     [102400.0, 120700.0, 138000.0, 126300.0, 72400.0, 68600.0, 110700.0, 89500.0, 83700.0, 163200.0, 64700.0, 113300.0, 53800.0, 135800.0, 167600.0, 96400.0, 151700.0, 132000.0, 149200.0, 65600.0, 54000.0],
     [0.11746395710891079, 0.11912344199547836, 0.127298691892572, 0.12659927136235996, 0.12854404661582505, 0.12787718082248525, 0.13470584969830549, 0.1293602252558059, 0.12419032853411432, 0.12008584478857405, 0.13160393287345312, 0.12959973188200063, 0.12901223172892728, 0.13294167419811648, 0.13845024378667828, 0.13783148796494069, 0.13758782819292179, 0.13718159604907482, 0.1362273733863873, 0.1360776012977379, 0.12281450118727147]
    ]
)
data = []
for i in range(len(data_a_month[0])):
    data.append([data_a_month[0][i],data_a_month[1][i],data_a_month[2][i]])
data = np.array(data)
data
# %%
# ECM初始化
clusters = []
radius = []
centers = []
for i in range(len(data)):
    print(i)
    ltype = ecm(data[i])
    print(ltype)
    print()
# %%
