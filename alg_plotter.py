# from alg_GLOBALS import *
# import logging

from matplotlib import pyplot as plt
from termcolor import colored

"""
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
"""


def get_weighted_coverage_list(coverage_list):
    weighted_coverage_list = []
    item = coverage_list[0]
    alpha = 0.9
    for next_item in coverage_list:
        item = item * alpha + (1 - alpha) * next_item
        weighted_coverage_list.append(item)
    return weighted_coverage_list


class ALGPlotter:
    """
    This object is responsible for plotting, logging and neptune updating.
    """

    def __init__(self, plot_life=True, plot_neptune=False, name='nameless', tags=None, plot_per=1):

        self.plot_life = plot_life
        self.plot_neptune = plot_neptune
        self.name = name
        self.tags = [] if tags is None else tags
        self.plot_per = plot_per

        self.run = {}
        self.neptune_initiated = False

        if self.plot_life:
            self.fig = plt.figure(figsize=plt.figaspect(.5))
            self.fig.suptitle(f'{name}')
            self.fig.tight_layout()
            # ax_1 = fig.add_subplot(1, 1, 1, projection='3d')
            self.ax_1 = self.fig.add_subplot(1, 2, 1)
            self.ax_2 = self.fig.add_subplot(1, 2, 2)
            # self.ax_3 = self.fig.add_subplot(1, 5, 3)
            # self.ax_4 = self.fig.add_subplot(1, 5, 4)
            # self.ax_5 = self.fig.add_subplot(1, 5, 5)

            self.mean_list = []
            self.std_list = []
            self.loss_list_actor = []
            self.loss_list_critic = []
            self.list_state_mean_1 = []
            self.list_state_std_1 = []
            self.list_state_mean_2 = []
            self.list_state_std_2 = []

        print(colored(f'~[INFO]: "ALGPlotter instance created."', 'green'))

    def plot(self, i, env, scores,
             # actor_mean, actor_std, loss, loss_critic, actor_output_tensor, observations_tensor, state_stat_mean, state_stat_std
             ):
        if self.plot_life:
            # PLOT
            # mean_list.append(actor_output_tensor.mean().detach().squeeze().item())
            # self.mean_list.append(actor_mean.mean().detach().squeeze().item())
            # self.std_list.append(actor_std.mean().detach().squeeze().item())
            # self.loss_list_actor.append(loss.item())
            # self.loss_list_critic.append(loss_critic.item())
            # self.list_state_mean_1.append(state_stat_mean[0])
            # self.list_state_mean_2.append(state_stat_mean[1])
            # self.list_state_std_1.append(state_stat_std[0])
            # self.list_state_std_2.append(state_stat_std[1])
            # print(i, self.plot_per, i % self.plot_per)

            if i % self.plot_per == 0:
                # AX 1
                self.ax_1.cla()
                # PLOT POSITIONS
                for pos in env.positions:
                    self.ax_1.plot(pos.x, pos.y, color=pos.color, marker=pos.marker, markersize=10)

                # PLOT AGENTS
                for agent in env.agents:
                    self.ax_1.plot(agent.x, agent.y, color=agent.color, marker=agent.marker, markersize=14)

                    # PLOT DOMAINS
                    for pos in agent.domain:
                        self.ax_1.plot(pos.x, pos.y, color=agent.color, marker=agent.marker, markersize=14, alpha=0.2)

                    # PLOT PATHS
                    if len(agent.path) > 0:
                        pos_list_x, pos_list_y = zip(*agent.path)
                        self.ax_1.plot(pos_list_x, pos_list_y, color=agent.color, marker='X', markersize=7, linewidth=4, alpha=0.5)

                # self.ax_1.legend()
                self.ax_1.set_title('Env')

                # AX 2
                self.ax_2.cla()
                self.ax_2.plot(scores, label='scores')
                avg_scores = get_weighted_coverage_list(scores)
                self.ax_2.plot(avg_scores, label='avg scores')
                self.ax_2.set_title('Scores')
                self.ax_2.legend()

                # self.fig.pause(0.05)
                plt.pause(0.05)
                # self.fig.show()

    def plot_close(self):
        if self.plot_life:
            plt.close()

    # def neptune_init(self, params=None):
    #
    #     if params is None:
    #         params = {}
    #
    #     if self.plot_neptune:
    #         self.run = neptune.init(project='1919ars/MA-implementations',
    #                                 tags=self.tags,
    #                                 name=f'{self.name}')
    #
    #         self.run['parameters'] = params
    #         self.neptune_initiated = True

    def neptune_plot(self, update_dict: dict):
        if self.plot_neptune:

            if not self.neptune_initiated:
                raise RuntimeError('~[ERROR]: Initiate NEPTUNE!')

            for k, v in update_dict.items():
                self.run[k].log(v)

    def neptune_close(self):
        if self.plot_neptune and self.neptune_initiated:
            self.run.stop()


# AX 1
# self.ax_1.cla()
# input_values_np = observations_tensor.squeeze().numpy()
# x = input_values_np[:, 0]
# y = input_values_np[:, 1]
#
# actor_output_tensor_np = actor_output_tensor.detach().squeeze().numpy()
# self.ax_1.scatter(x, y, actor_output_tensor_np[:, 0], marker='.', alpha=0.09, label='action 1')
# self.ax_1.scatter(x, y, actor_output_tensor_np[:, 1], marker='x', alpha=0.09, label='action 2')
# # critic_output_tensor_np = critic_output_tensor.detach().squeeze().numpy()
# # ax_1.scatter(x, y, critic_output_tensor_np, marker='.', alpha=0.1, label='critic values')
# self.ax_1.set_title('Outputs of NN')
# self.ax_1.legend()
#
# # AX 2
# self.ax_2.cla()
# self.ax_2.plot(self.mean_list, label='mean')
# self.ax_2.plot(self.std_list, label='std')
# self.ax_2.set_title('Mean & STD')
# self.ax_2.legend()
#
# # AX 3
# self.ax_3.cla()
# self.ax_3.plot(self.loss_list_actor, label='actor')
# self.ax_3.plot(self.loss_list_critic, label='critic')
# self.ax_3.set_title('Loss')
# self.ax_3.legend()
#
# # AX 4
# self.ax_4.cla()
# self.ax_4.plot(scores, label='scores')
# self.ax_4.plot(avg_scores, label='avg scores')
# self.ax_4.set_title('Scores')
# self.ax_4.legend()
#
# # AX 5
# self.ax_5.cla()
# self.ax_5.plot(self.list_state_mean_1, label='m1', marker='.', color='b')
# self.ax_5.plot(self.list_state_std_1, label='s1', linestyle='--', color='b')
# self.ax_5.plot(self.list_state_mean_2, label='m2', marker='.', color='g')
# self.ax_5.plot(self.list_state_std_2, label='s2', linestyle='--', color='g')
# self.ax_5.set_title('State stat')
# self.ax_5.legend()