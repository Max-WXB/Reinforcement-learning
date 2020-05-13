import pyglet
import numpy as np 

class ArmEnv(object):
	viewer = None
	dt = 0.1								# 转动的速度和dt有关
	action_bound = [-1, 1]					# 转动的角度范围	
	goal = {'x':100., 'y':100., 'l':40}		# 蓝色goal的x，y坐标和长度l
	state_dim = 2							# 两个观测值
	action_dim = 2							# 两个动作

	def __init__(self):
		# （2，2）的表
		self.arm_info = np.zeros(
			2, dtype=[('l', np.float32), ('r', np.float32)])
		self.arm_info['l'] = 100
		self.arm_info['r'] = np.pi/6



	def step(self, action):
		done = False
		r = 0.

		# 计算单位时间 dt 内旋转的角度，将角度限制在360度内
		action = np. clip(action, *self.action_bound)
		self.arm_info['r'] += action * self.dt
		self.arm_info['r'] %= np.pi * 2    # normalize

		s = self.arm_info['r']

		# 如果手指接触到蓝色的 goal, 我们判定结束回合 (done)
		# 所以需要计算 finger 的坐标
		(a1l, a2l) = self.arm_info['l']  # radius, arm length
		(a1r, a2r) = self.arm_info['r']  # radian, angle
		a1xy = np.array([200., 200.])    # a1 start (x0, y0)
		a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
		finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

		# 根据 finger 和 goal 的坐标得出 done and reward
		if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2:
			if self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
				done = True
				r = 1.      # finger 在 goal 以内
		return s, r, done



	def reset(self):
		self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
		return self.arm_info['r']



	def render(self):
		if self.viewer is None:
			self.viewer = Viewer(self.arm_info, self.goal)
		self.viewer.render()



	def sample_action(self):
		return np.random.rand(2) - 0.5




class Viewer(pyglet.window.Window):
	bar_thc = 5						# 手臂的厚度

	def __init__(self, arm_info, goal):
		# 画出手臂
		# 创建窗口的继承
		# vsync 如果是 True， 按屏幕刷频率书刷新，反之不按那个频率
		super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
		# 窗口背景颜色
		pyglet.gl.glClearColor(1, 1, 1, 1)
		# 添加 arm 信息
		self.arm_info = arm_info
		# 添加窗口中心点， 手臂的根
		self.center_coord = np.array([200, 200])
		# 将手臂的作图信息放入这个batch
		self.batch = pyglet.graphics.Batch()		# display whole batch at once

		# 添加蓝点
		self.point = self.batch.add(
			4, pyglet.gl.GL_QUADS, None,    # 4 corners
			('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,
					 goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
					 goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
					 goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2,
					 ]),
			('c3B', (86, 109, 249) * 4))		# color RGB颜色值

		# 添加一条手臂
		self.arm1 = self.batch.add(
			4, pyglet.gl.GL_QUADS, None,
			('v2f', [250, 250,                # location
					 250, 300,
					 260, 300,
					 260, 250]),
			('c3B', (249, 86, 86) * 4,))

		# 添加一条手臂
		self.arm2 = self.batch.add(
			4, pyglet.gl.GL_QUADS, None,
			('v2f', [100, 150,                # location
					 100, 160,
					 200, 160,
					 200, 150]),
			('c3B', (249, 86, 86) * 4,))



	def render(self):
		self._update_arm()
		self.switch_to()
		self.dispatch_events()
		self.dispatch_event('on_draw')
		self.flip()
	



	def on_draw(self):
		# 刷新手臂等位置
		self.clear()						# 清屏
		self.batch.draw()					# 画上 batch 里面的内容




	def _update_arm(self):
		# 更新手臂的位置信息
		(a1l, a2l) = self.arm_info['l']     # radius, arm length
		(a1r, a2r) = self.arm_info['r']     # radian, angle
		a1xy = self.center_coord            # a1 start (x0, y0)
		a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
		a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

		a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
		xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
		xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
		xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
		xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

		xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
		xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
		xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
		xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

		self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
		self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))



if __name__ == '__main__':
	env = ArmEnv()
	while True:
		env.render()
		env.step(env.sample_action())