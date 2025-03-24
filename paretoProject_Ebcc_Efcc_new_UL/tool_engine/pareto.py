import os,copy,sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt



def dominates(p1, p2):
    """
    original code: Eugene J. Ragasa, UF
    """
    for x,y in zip(p1,p2):
        if y < x:
            return False
    return True

def pareto_bruteforce(pts, indices = None):
    """
    original code: Eugene J. Ragasa, UF
    """
    if indices is None:
        indices = list(range(len(pts)))
    result = []
    for i in indices:
        for j in indices:
            if i == j: continue
            if dominates(pts[j], pts[i]):
                break
        else:
            result.append(i)
    return result

def pareto_merge(lo, hi, i, dim):
    """
    original code: Dmitriy Morozon, LBL
    """
    if len(lo) == 0 or len(hi) == 0:
        return lo + hi

    survivors = set()
    for j in range(dim):
        if i == j: continue
        m = min(p[j] for p in lo)
        survivors.update(k for k in range(len(hi)) if hi[k][j] < m)

    return lo + [hi[k] for k in survivors]

def pareto(pts, indices = None, i = 0):
    """
    original code: Dmitriy Morozov, LBL
    """
    if indices is None:
        indices = list(range(len(pts)))
    l = len(indices)
    if l <= 1:
        return indices

    if l < 1000:
        return pareto_bruteforce(pts, indices)

    indices.sort(key = lambda x: pts[x][i])     # lazy: should use partition instead

    dim = len(pts[0])
    optimalLo = pareto(pts, indices[:l//2], (i + 1) % dim)
    optimalHi = pareto(pts, indices[l//2:], (i + 1) % dim)

    return pareto_bruteforce(pts, optimalLo + optimalHi)     # lazy: FIXME
    #return pareto_merge(optimalLo, optimalHi, i, dim)


class SimulationResults(object):
    """this class processes the simulation results

    Args:
      n_simulations (int): number of simulations read from the output file
      qoi_type (str): supported qoi types are
          'abserr' - absolute error
    """
    def __init__(self):
        """default constructor"""


        # filenames
        self.fname_log_file = "pyposmat.log"
        self.fname_pareto = None
        self.fname_cull= None

        # initialize variables [ATTRIBUTES]
        self._names = None # names of the
        self._types = None
        self._qoi_names = []  # array of qoi names
        self._err_names = []

        self._results = None                 # numpy array of all simulation data
        self._pareto = None                  # numpy array of the pareto set
        self._cull = None                    # numpy array of the culled pareto set




    @property
    def err_names(self):
        return self._err_names

    # @property
    # def results(self):
    #     """numpy.array: numpy array of results"""
    #     return self._results

    @property
    def pareto(self):
        """numpy.array: numpy array of pareto results"""
        return self._pareto
    #
    # @property
    # def culled(self):
    #     """numpy.array: numpy array of pareto results"""
    #     return self._culled

    def __read_file(self, fname, file_type):  # fname = '20170201_results_10k.out'; file_type = 'results'

        # read file into memory
        f_in = open(fname, 'r')
        lines_in = f_in.readlines()  # 读取'20170201_results_10k.out'文件所有的内容（从1~10002行）
        f_in.close()

        if file_type == 'results':  # 第一次采样的数据：results数据集
            # read header lines（读取头文件：一共有32列）
            self._names = [n.strip() for n in lines_in[0].strip().split(',')]  # 获取'20170201_results_10k.out'的第1行内容（名字）
            self._types = [t.strip() for t in lines_in[1].strip().split(',')]  # 获取'20170201_results_10k.out'的第2行内容（类型）
        elif file_type == 'pareto':
            # check to see if pareto header line is the same as the pareto line（检查pareto标头行是否与pareto行相同）
            if self._names == [n.strip() for n in lines_in[0].strip().split(',')]:
                pass
            else:
                if self._names is None:
                    errmsg = "The results file must be read before the pareto file"
                    raise RuntimeError(errmsg)
                else:
                    errmsg = "The pareto names header does not match results file"
                    raise RuntimeError(errmsg)

            # check to see if pareto types header line is the same as the pareto line
            if self._types == [t.strip() for t in lines_in[1].strip().split(',')]:
                pass
            else:
                if self._types is None:
                    errmsg = "The results file must be read before the pareto file"
                    raise RuntimeError(errmsg)
                else:
                    errmsg = "the pareto types header does not match results file"
                    raise RuntimeError(errmsg)

        results = []
        for i in range(2,len(lines_in)):  # len(lines_in) = 10002; 从'20170201_results_10k.out'文件的第3行开始读取
            result =  [v.strip() for v in lines_in[i].strip().split(',')]   # 最开始遍历是获得第3行的数据，以后遍历都是按行读取数据
            for j,t in enumerate(self._types):  # 对'20170201_results_10k.out'文件的第i行内容(types)进行遍历,并且把该行的字符串格式转换为int和float格式
                if t == "sim_id":
                    result[j] = int(float(result[j]))  #  存的是sim_id列的值：第一次值为0
                else:
                    # everything else is a float
                    result[j] = float(result[j])
            results.append(result)   #  reuslt由原来的[]，变成[[x2_0,x2_1,...,x2_31], [x3_0,x3_1,...,x3_31],...,[x1001_0,x10001_1,...,x10001_31]]

        # convert into numpy file（转换成numpy格式）
        if file_type == 'results':
            self._param_names = [self._names[i] for i,v in enumerate(self._types) if v == 'param']
            self._qoi_names = [self._names[i] for i,v in enumerate(self._types) if v == 'qoi']
            self._err_names = [self._names[i] for i,v in enumerate(self._types) if v == 'err']
            self._results = np.array(results)   # 转成numpy格式
        elif file_type == 'pareto':
            self._pareto = np.array(results)
        elif file_type == 'culled':
            self._culled = np.array(results)



    def read_simulation_results(self,
                                fname_sims,
                                fname_pareto=None,
                                fname_cull=None):
        """
        read simulations results from a file into a memory.

        Args:
            fname_sims (str): the filename containing the simulation results from
                LAMMPS simulations - 包含来自LAMMPS模拟的模拟结果的文件名   via chj 第1次迭代，fname_results_out = 'results_000.out'；
            fname_pareto (str): the filename containing the pareto set results - 包含pareto集合结果的文件名
            fname_cull (str): the filename contain culled pareto set results - 包含精选的帕累托集合结果的文件名
        """

        self.fname_sims = fname_sims  # fname_sims = '20170201_results_10k.out'
        self.__read_file(fname_sims, 'results')

        # remove rows that have NaN as result
        rows_to_remove = []
        for i in range(1, self._results.shape[0]):
            if np.isnan(self._results[i, :]).any():
                rows_to_remove.append(i)  # 存在NaN值的行号，先存起来
        self._results = np.delete(self._results, rows_to_remove, axis=0)  # 剔除self._results中存在NaN值的数据

        if fname_pareto is not None:
            self.fname_pareto = fname_pareto
            self.__read_file(fname_pareto, 'pareto')

        if fname_cull is not None:
            self.fname_cull = fname_cull
            self.__read_file(fname_cull, 'culled')


    def _create_dataset_for_pareto_analysis(self, err_names=None):  # err_names = ['MgO_NaCl.a0.err','MgO_NaCl.c11.err','MgO_NaCl.c12.err','MgO_NaCl.c44.err','MgO_NaCl.B.err','MgO_NaCl.G.err','MgO_NaCl.fr_a.err','MgO_NaCl.fr_c.err','MgO_NaCl.sch.err','MgO_NaCl.001s.err']

        """iCreates a dataset for pareto analysis

        This method creates a dataset necessary for pareto analysis   # 我创建了一个用于pareto分析的数据集。此方法创建一个pareto分析所需的数据集
        Arguments:
        err_names (list of str): - contains the identifiers for the error   # err_names（str的列表）：-包含错误的标识符
        """

        print("creating dataset for pareto analysis")

        if err_names is None:
            err_names = self._err_names

        # get indices of error names   # 获取错误（误差ε）名称的索引
        err_idx = [self._names.index(n) for n in err_names]  # 该索引对应的是20170201_results_10k.out文件的第22-31列（即.err的列索引）

        # select the sim_id column and err_names columns
        results_err = self._results[:, [0] + err_idx]  # 选择sim_id所在列（0列）和.err所在列（22-31列）的所有数据 --- 共11列
        results_abs_err = np.abs(results_err)  # 将误差变为绝对值误差,   results_abs_err.shape = (9999, 11)  11列 = sim_id所在列（0列） + .err所在列（22-31列）的10列

        # make dataset
        n_row, n_col = results_abs_err.shape  # n_row为9999行, n_col为11列
        self._pareto_dataset = []
        for i_row in range(n_row):
            self._pareto_dataset.append(Datapoint(i_row))
            for i_col in range(n_col):
                number = results_abs_err[i_row, i_col]  # 对元素进行遍历
                self._pareto_dataset[i_row].addNumber(-number)  # 例如当i_row = 0，i_col=0，self._pareto_dataset = [-0.0]: -1；当i_row = 0，i_col=1，self._pareto_dataset = [-0.0, -0.0693195346]: -1；i_row = 0，i_col=2，self._pareto_dataset =[-0.0, -0.0693195346, -78.1500664]: -1
                # 例如当i_row = 1，i_col=0，self._pareto_dataset = [[-0.0, -0.0693195346, -78.1500664, -64.4549177, -13.8200639, -69.016634, -6.85257432, -1.65067758, -2.8160998, -2.13976071, -0.0238602339]: -1, [-1.0]: -1]


    def calculate_pareto_set(self):

        self._create_dataset_for_pareto_analysis(err_names=self._err_names)  # self._err_names = ['MgO_NaCl.a0.err','MgO_NaCl.c11.err','MgO_NaCl.c12.err','MgO_NaCl.c44.err','MgO_NaCl.B.err','MgO_NaCl.G.err','MgO_NaCl.fr_a.err','MgO_NaCl.fr_c.err','MgO_NaCl.sch.err','MgO_NaCl.001s.err']

        bruteforce_algo(self._pareto_dataset)  # 使用暴力算法

        # mark pareto set
        pareto_set = []
        pareto_set_ids = []
        for s in self._pareto_dataset:
            if s.paretoStatus == 1:
                pareto_set_ids.append(s.id)  # s.id是等价于sim_id，是一个索引
                pareto_set.append(s.vec)  # s.vec是一个sim_id + err的11*1的列向量

        # pareto_set = -np.array(pareto_set)
        self._pareto = self._results[pareto_set_ids, :]  # 例如pareto_set_ids = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,20,22,23,25,27,30,31,33,37,39,51,54,55,56,...],得到的是对应的sim_id + 参数列 + qoi列 + err列（共32列）


    #--------------------------------------------------------------------------
    # methods for calculating the culled pareto set
    #--------------------------------------------------------------------------
    def calculate_culled_set(self, cull_type="percentile", pct=80.):
        """
        Arguments:
        cull_type - supports the different culling of the pareto set by
            different mechanisms.  The current mechanisms are 'percentile'
            and 'pct_error'
        pct - is a float variable.

        Returns:
            Nothing

        Raises:
            RuntimeError: If any key in qoierr_threshold is not contained
                in the attribute error_names, it will check to see if
                the key value is contained in qoi_names and attempt to
                change the key value.  If this is not successful, then
                a RuntimeError will be returned.
        """

        if cull_type == "percentile":
            self._calculate_culled_set_by_percentile(pct)
        else:
            raise RuntimeError("unknown cull_type")


    def _calculate_culled_set_by_percentile(self, pct_kept=80.):
        """
        Arguments:
        pct_kept (float, 10.0) - number between 1 and 100 indicating the
        pct of simulations within the Pareto set which should be kept

        Returns:

        a numpy array with observations indexed in rows, and parameters
        and quantities of interst indexed in columns.  The column index is the
        same as the array in "self.all_names".
        """

        # TODO:
        # A Newton-Ralphson method to get more accurate performance requirements
        # to prevent over culling of the Pareto set.

        if not (0 <= pct_kept <= 100.):
            errmsg = "pct_kept must be between 1 and 100, the value {} was passed."
            errmsg = errmsg.format(pct_kept)
            raise ValueError(errmsg)
        else:
            self.pct_kept = pct_kept

        err_keys = self._err_names
        self._perf_req = {}
        for err_key in err_keys:
            self._perf_req[err_key] = 0.
        n_sims, n_qoi = self._pareto.shape

        # intialize variables
        pctl_threshold = 100  # searching for 100% within the Pareto set
        # to 0% in the pareto set
        is_culled = False  # intialize

        while not is_culled:
            rows_to_delete = []
            pctl_threshold -= 0.1
            # calculate percentile cutoffs
            for err_key in self._perf_req.keys():
                if pctl_threshold < 0:
                    errmsg = "While searching for the pctl_threshold, the \
                              percentile error dropped below zero resulting \
                              in an error."
                    raise ValueError(errmsg)
                else:
                    qoi_data = self.get_data(err_key, 'pareto', 'abserror')
                    cutoff = np.percentile(qoi_data, pctl_threshold)
                    self._perf_req[err_key] = cutoff

            # cull the pareto set by the performance requirements
            for err_key in self._perf_req.keys():
                pareto = np.copy(self.pareto)
                for idx in range(n_sims):
                    ps = pareto[idx, :]

                    # determine if row needs to be deleted
                    is_delete_row = False
                    for qoi_name in self._perf_req.keys():
                        qoi_idx = self._names.index(qoi_name)
                        if ps[qoi_idx] > self._perf_req[qoi_name]:
                            is_delete_row = True

                            # add row for deletion if requirements met.
                    if is_delete_row:
                        rows_to_delete.append(idx)

            # check to see if the pareto set has been sufficiently culled
            n_culled = len(rows_to_delete)
            pct_culled = float(n_culled) / float(n_sims)
            if pct_kept / 100. > 1 - pct_culled:
                is_culled = True

        self._culled = np.delete(self._pareto,
                                 rows_to_delete,
                                 axis=0)
        return self._culled.copy()

    def get_data(self, name, ds_type, err_type='abserr'):
        """
        Arguments:

        name (str) - string of parameter or quantity of interest
        ds_type (str) - string of which dataset we are taking the data from.
          The ds_types which are supported are: all, pareto, pareto_culled

        Returns:

        a numpy array of the data asked for
        """
        idx = self._names.index(name)

        # get data by dataset
        data = None  # initialize
        if ds_type == 'results':
            data = self._results[:, idx]
        elif ds_type == 'pareto':
            data = self._pareto[:, idx]
        elif ds_type == 'culled':
            data = self._culled[:, idx]

        if self._types[idx] == 'err':
            # transform errors if necessary
            if err_type == 'err':
                # no transformation required
                data = self._results[:, idx]
            elif err_type == 'abserr':              # hjchen: 这个可以不要？因为xxx_00x.out文件中的第2行没有.abserr开头的
                # transform for absolute errors
                data = np.abs(self._results[:, idx])
        else:
            # tranformation not necessary
            data = self._results[:, idx]

        return copy.deepcopy(data)

    def write_pareto_set(self, fname_out='pareto.out'):
        """Write the pareto set to file.

        This function prints the calculated pareto set to file.

        Args:
            fname_out(str) - the filename (default: pareto.out)
        """

        # create header
        str_names = ", ".join(self._names) + "\n"
        str_types = ", ".join(self._types) + "\n"

        # create body
        str_body = ""
        for sim_result in self._pareto:
            str_body += ", ".join([str(num) for num in sim_result]) + "\n"

        # write results
        f = open(fname_out, 'w')
        f.write(str_names)
        f.write(str_types)
        f.write(str_body)
        f.close()

    def write_culled_set(self, fname_out='culled.out'):
        # create header
        str_names = ", ".join(self._names) + "\n"
        str_types = ", ".join(self._types) + "\n"

        # create body
        str_body = ""
        for sim_result in self._culled:
            str_body += ", ".join([str(num) for num in sim_result]) + "\n"

        # write results
        f = open(fname_out, 'w')
        f.write(str_names)
        f.write(str_types)
        f.write(str_body)
        f.close()


#--------------------------------------------------------------------------
class Datapoint:
    """Defines a point in K-dimensional space"""  # 在K维空间中定义一个点

    def __init__(self, id):
        self.id = id  # datapoint id (0,..N-1)
        self.vec = []  # the K-dim vector   # 例如err列的[-0.0, -0.0693195346, -78.1500664, -64.4549177, -13.8200639, -69.016634, -6.85257432, -1.65067758, -2.8160998, -2.13976071, -0.0238602339]
        self.paretoStatus = -1  # -1=dont know, 1=pareto, 0=not pareto
        self.dominatedCount = 0  # number of datapoints that dominate this point
        self.dominatingSet = []  # set of vectors this one is dominating

    def addNumber(self, num):
        """Adds a number to one dimension of this datapoint"""
        self.vec.append(num)

    def addToDominatingSet(self, id2):
        """Add id of of dominating point"""
        self.dominatingSet.append(id2)

    def dominates(self, other):  # 例如：当n=0,m=0时，other = self.vec = [-0.0, -0.0693195346, -78.1500664, -64.4549177, -13.8200639, -69.016634, -6.85257432, -1.65067758, -2.8160998, -2.13976071, -0.0238602339]
        """Returns true if self[k]>=other[k] for all k and self[k]>other[k] for at least one k"""
        assert isinstance(other, Datapoint)
        gte = 0  # count of self[k]>=other[k]
        gt = 0  # count of self[k]>other[k]
        for k in range(len(self.vec)):
            if self.vec[k] >= other.vec[k]:
                gte += 1
                if self.vec[k] > other.vec[k]:
                    gt += 1

        return (gte == len(self.vec) and (gt > 0))  # 只有向量长度都相同，且一个向量里的元素self.vec[k]比另一个向量对应的元素other.vec[k]都大，才能说前者（self.vec）被后者（other.vec）支配

    def __repr__(self):
        return self.vec.__repr__() + ": " + str(self.paretoStatus)



def bruteforce_algo(dataset):  # dataset = self._pareto_dataset = [[a0,a1,...,a10]: -1, [b0,b1,...,b10]: -1, ... , [x0,x1,...,x10]: -1]  有9999个样本（每个样本包含11列和一个-1）
    num_pareto = 0

    # pairwise comparisons
    for n in range(len(dataset)):  # len(dataset) = 9999
        if np.mod(n, 100) == 0:
            print("\t n={}".format(n))
        for m in range(len(dataset)):
            if dataset[m].dominates(dataset[n]):  # dataset[m].dominates(dataset[n])返回的一个True or False， 表示如果前一个k维的点（这里是11维）
                dataset[n].dominatedCount += 1
                dataset[m].addToDominatingSet(n)

    # find first pareto front
    for n in range(len(dataset)):
        if dataset[n].dominatedCount == 0:
            dataset[n].paretoStatus = 1
            num_pareto += 1
        else:
            dataset[n].paretoStatus = 0


