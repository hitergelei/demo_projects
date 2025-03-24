import time
from datetime import datetime
from tool_engine import pareto
from tool_engine import pyposmat



start_time = time.time()

# --- simulation parameters ---
n_simulations = 100  # number of simulation per iteration loop  每个迭代循环中的模拟次数（参数采样的次数？）
n_iterations = 10    # number of iteration loops   迭代循环次数

# supported cull_types are: percentile, pct_error
cull_type = 'percentile'
cull_param = 50.


class IterativeSampler(pyposmat.PyPosmatEngine):
    def __init__(self, n_iterations,
                       n_simulations,
                       cull_type,
                       cull_param):

        super().__init__()       # 等同于pyposmat.PyPosmatEngine.__init__(self)
        self._n_iterations = n_iterations
        self._n_simulations = n_simulations
        self._fname_results_format = "results_{:03d}.out"
        self._fname_pareto_format = "pareto_{:03d}.out"
        self._fname_culled_format = "culled_{:03d}.out"
        self._cull_type = cull_type
        self._cull_param = cull_param



    def run(self, n_iterations=None):
        if n_iterations is not None:   # 如果迭代次数不为None
            self._n_iterations = n_iterations

        for i_iter in range(0, self._n_iterations):   # # number of iteration loops   迭代循环次数：例如 迭代10次
            self._log('starting iteration loop {}'.format(i_iter))
            fname_results_out = self._fname_results_format.format(i_iter)  # results_000.out
            fname_pareto_out  = self._fname_pareto_format.format(i_iter)   # pareto_000.out
            fname_culled_out  = self._fname_culled_format.format(i_iter)   # culled_000.out

            # generations
            if i_iter == 0:
                # use uniform sampling the first time around
                # To Do： 完成均匀分布采样的代码
                self.sample_parameter_space(n_simulations = self._n_simulations,
                                            fname_results = fname_results_out,  #  fname_results_out为lammps计算结果？ = results_000.out
                                            sampler_type = 'uniform')
            else:
                # use the culled results from the previous iteration
                fname_results_in = self._fname_culled_format.format(i_iter-1)  #  例如i_iter=1时：fname_results_in =  culled_000.out

                self.sample_parameter_space(n_simulations = self._n_simulations,
                                            fname_results = fname_results_out,  # 例如i_iter=1时：fname_results_out =  results_001.out
                                            sampler_type = 'kde',
                                            fname_results_in = fname_results_in) # 例如i_iter=1时：fname_results_in =  culled_000.out


            # TO DO : 2020-12-11
            sim_results = pareto.SimulationResults()
            sim_results.read_simulation_results(fname_sims=fname_results_out)  #  对result_00x.out进行NaN剔除？
            sim_results.calculate_pareto_set()  #  对result_00x.out进行pareto最优
            sim_results.calculate_culled_set(self._cull_type,
                                             self._cull_param)

            sim_results.write_pareto_set(fname_pareto_out)   # 写入pareto_00x.out文件
            sim_results.write_culled_set(fname_culled_out)   # 写入culled_00x.out文件


if __name__ == '__main__':

    mcsampler = IterativeSampler(n_iterations=n_iterations,
                                 n_simulations=n_simulations,
                                 cull_type=cull_type,
                                 cull_param=cull_param)
    mcsampler.run()
    print("{} second".format(time.time() - start_time))





    # from input.potential_param_config import parameter_distribution
    # from input.DFT_qois import qoi_name_list
    # from input.DFT_qois import qoi_value_list
    # _param_names = list(parameter_distribution.keys())
    # # self._param_dict = {}
    # _param_info = parameter_distribution
    #
    # _qoi_names = qoi_name_list
    # _error_names = ["{}.err".format(q) for q in _qoi_names]  # self._error_names共有10个.err
    # _names = _param_names + _qoi_names + _error_names  # self._names为.out文件的第一行数据，但注意self._names为31个（即，除掉sim_id）
    # _types = len(_param_names) * ['param'] \
    #               + len(_qoi_names) * ['qoi'] \
    #               + len(_error_names) * ['err']  # self._types为.out文件的第二行数据（共有31个）
    # print("_names = ", _names)
    # print("_types = ", _types)
