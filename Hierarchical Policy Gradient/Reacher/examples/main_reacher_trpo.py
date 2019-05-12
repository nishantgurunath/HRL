import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent import Agent


parser = argparse.ArgumentParser(description='PyTorch A2C example')
parser.add_argument('--env-name', default="Reacher-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=5000, metavar='N',
                    help='minimal batch size per A2C update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-model-interval', type=int, default=100, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env = gym.make(args.env_name)
state,_ = env.reset()
state_dim = state.shape[0]
subgoal_dim = 3
 
is_disc_action = len(env.action_space.shape) == 0
#running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)
running_state = None
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_mgr = DiscretePolicy(state_dim, 5)
        policy_wrk = Policy(state_dim + subgoal_dim, env.action_space.shape[0], log_std=args.log_std)
    value_mgr = Value(state_dim)
    value_wrk = Value(state_dim + subgoal_dim)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
policy_mgr.to(device)
policy_wrk.to(device)
value_mgr.to(device)
value_wrk.to(device)

optim_policy_m = torch.optim.Adam(policy_mgr.parameters(), lr=0.01)
optim_policy_w = torch.optim.Adam(policy_wrk.parameters(), lr=0.01)
optim_value_m = torch.optim.Adam(value_mgr.parameters(), lr=0.01)
optim_value_w = torch.optim.Adam(value_wrk.parameters(), lr=0.01)

"""create agent"""
agent = Agent(env, policy_mgr, policy_wrk, device, running_state=running_state, render=args.render, num_threads=args.num_threads)


def update_params(batch_mgr, batch_wrk):
    states_mgr = torch.from_numpy(np.stack(batch_mgr.state)).to(dtype).to(device)
    directions = torch.from_numpy(np.stack(batch_mgr.action)).to(dtype).to(device)
    rewards_mgr = torch.from_numpy(np.stack(batch_mgr.reward)).to(dtype).to(device)
    masks_mgr = torch.from_numpy(np.stack(batch_mgr.mask)).to(dtype).to(device)

    states_wrk = torch.from_numpy(np.stack(batch_wrk.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch_wrk.action)).to(dtype).to(device)
    rewards_wrk = torch.from_numpy(np.stack(batch_wrk.reward)).to(dtype).to(device)
    masks_wrk = torch.from_numpy(np.stack(batch_wrk.mask)).to(dtype).to(device)

    with torch.no_grad():
        values_mgr = value_mgr(states_mgr)
        values_wrk = value_wrk(states_wrk)

    """get advantage estimation from the trajectories"""
    advantages_mgr, returns_mgr = estimate_advantages(rewards_mgr, masks_mgr, values_mgr, args.gamma, args.tau, device)
    advantages_wrk, returns_wrk = estimate_advantages(rewards_wrk, masks_wrk, values_wrk, args.gamma, args.tau, device)

    #print (torch.sum(torch.isnan(advantages_mgr)*1.0), torch.sum(torch.isnan(returns_mgr)*1.0))
    #print (torch.sum(torch.isnan(advantages_wrk)*1.0), torch.sum(torch.isnan(returns_wrk)*1.0))

    """perform TRPO update"""
    policy_loss_m = 0
    trpo_step(policy_mgr, value_mgr, states_mgr, directions, returns_mgr, advantages_mgr, args.max_kl, args.damping, args.l2_reg)

    trpo_step(policy_wrk, value_wrk, states_wrk, actions, returns_wrk, advantages_wrk, args.max_kl, args.damping, args.l2_reg)



def main_loop():
    avg_reward = 0
    #avg_loss_m = 0
    #avg_loss_w = 0
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch_mgr, batch_wrk, log = agent.collect_samples(args.min_batch_size)
        t0 = time.time()
        update_params(batch_mgr, batch_wrk)
        t1 = time.time()

        avg_reward += log['avg_reward']/args.log_interval
        #avg_loss_m += loss_m/args.log_interval
        #avg_loss_w += loss_w/args.log_interval

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}\tM_avg {:.2f}\tW_avg {:.2f}\tM_steps {:.2f}\tW_steps {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward'], log['mgr_reward'], log['wrk_reward'], log['mgr_steps'], log['num_steps']))
            avg_reward = 0 
            #avg_loss_m = 0
            #avg_loss_w = 0

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_mgr, policy_wrk, value_mgr, value_wrk)
            pickle.dump((policy_mgr, policy_wrk, value_mgr, value_wrk, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_trpo.p{}'.format(args.env_name,i_iter+1)), 'wb'))
            to_device(device,  policy_mgr, policy_wrk, value_mgr, value_wrk)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
