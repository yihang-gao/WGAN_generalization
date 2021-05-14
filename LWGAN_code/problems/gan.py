from utils import problem, registry
from models import BaseGAN
from utils.loaddata import get_data


@registry.register_problem('GAN')
class GANProblem(problem.Problem):
    def __init__(self, hparam):
        super().__init__(hparam)

    def load_data(self):
        self.train_data, self.test_data = get_data(self.hparam['dataset_name'],
                                                   self.hparam['data_shape'],
                                                   self.hparam['buffer_size'],
                                                   self.hparam['batch_size'])

    def train_model(self):
        gan = BaseGAN(z_shape=self.hparam['z_shape'],
                      out_dim=self.hparam['data_shape'],
                      epochs=self.hparam['epochs'],
                      batchsize=self.hparam['batch_size'],
                      test_num=self.hparam['test_num'],
                      g_depth=self.hparam['g_depth'],
                      g_width=self.hparam['g_width'],
                      d_depth=self.hparam['d_depth'],
                      d_width=self.hparam['d_width'],
                      lrg=self.hparam['lrg'],
                      lrd=self.hparam['lrd'],
                      beta_1=self.hparam['beta_1'],
                      beta_2=self.hparam['beta_2'],
                      bjorck_beta=self.hparam['bjorck_beta'],
                      bjorck_iter=self.hparam['bjorck_iter'],
                      bjorck_order=self.hparam['bjorck_order'],
                      group_size=self.hparam['group_size'],
                      num_critic=self.hparam['num_critic']
                      )
        gan.train(self.train_data, self.test_data)

    def test_model(self):
        pass
