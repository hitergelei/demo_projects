
def hcp():
    idx_list_hcp = []
    Ec_list_hcp = []
    idx_min_hcp = 0  # hcp的晶格常数
    Ec_min_hcp = 0  # hcp的结合能

    idx_list_fcc = []
    Ec_list_fcc = []
    idx_min_fcc = 0  # fcc的晶格常数
    Ec_min_fcc = 0  # fcc的结合能

    idx_list_bcc = []
    Ec_list_bcc = []
    idx_min_bcc = 0  # bcc的晶格常数
    Ec_min_bcc = 0  # bcc的结合能

    deta_hcp_bcc = 0  # hcp->bcc的结构能量差
    deta_hcp_fcc = 0  # hcp->fcc的结构能量差



    print('----------The <{hcp.sh}> calculation is finished!, 提取Ec_out_hcp.dat的数据')
    for line in open('Ec_out_hcp.dat', 'r').readlines():
        # print(line)
        if line.startswith("@@@@"):
            idx_hcp = eval(line.split(":")[1].strip().split(" ")[0])
            Ec_hcp = eval(line.split(":")[1].strip().split(" ")[1])
            # if Ec_hcp < 0:   # 只装入<0的数字和索引
            idx_list_hcp.append(idx_hcp)
            Ec_list_hcp.append(Ec_hcp)

    idx_min_hcp = idx_list_hcp[Ec_list_hcp.index(min(Ec_list_hcp))]  # 晶格常数
    Ec_min_hcp = min(Ec_list_hcp)  # 最小的值为结合能

    return idx_min_hcp, Ec_min_hcp

if __name__ == '__main__':
    hcp()
