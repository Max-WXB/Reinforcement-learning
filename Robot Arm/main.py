from env import ArmEnv
from rl import DDPG

# 全局变量
MAX_EPISODES = 500
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound




# set RL method
rl = DDPG(a_dim, s_dim, a_bound)



# start trianing
def train():
	for i in range(MAX_EPISODES):
		s = env.reset()							# 初始化回合设置
		ep_r = 0.
		for j in range(MAX_EP_STEPS):
			env.render()						# 环境渲染
			a = rl.choose_action(s)				# RL 选择动作
			s_, r, done = env.step(a)			# 在环境中施加动作

			# DDPG强化学习需要村存放记忆库
			rl.store_transition(s, a, r, s_)

			ep_r += r
			if rl.memory_full:
				rl.learn()						# 记忆库满了，开始学习

			s = s_ 								# 变为下一回合
			if done or j == MAX_EP_STEPS-1:
				print('Ep: %i | %s | ep_r: %.1f | steps: %i' % (i, '---' if not done else 'done', ep_r, j))
				break
	rl.save()



def eval():
	rl.restore()
	env.render()
	env.viewer.set_vsync(True)
	while True:
		s = env.reset()
		for _ in range(200):
			env.render()
			a = rl.choose_action(s)
			s, r, done = env.step(a)
			if done:
				break


if ON_TRAIN:
	train()
else:
	eval()


