import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'
from PIL import Image
import numpy as np
import math
import os


def cdvig(input):
	return np.array(([1, 0, 0, input[0]],[0, 1, 0, input[1]],[0, 0, 1, input[2]],[0, 0, 0, 1]))

def zoom(input):
	return np.array(([input[0], 0, 0, 0], [0, input[1], 0, 0], [0, 0, input[2], 0],[0, 0, 0, 1]))

def rx(r) :
	return np.array(([1, 0, 0, 0],[0, np.cos(r), -np.sin(r), 0],[0, np.sin(r), np.cos(r), 0],[0, 0, 0, 1]))

def ry(r):
	return np.array(([np.cos(r), 0, np.sin(r), 0],[0, 1, 0, 0], [-np.sin(r), 0, np.cos(r), 0],[0, 0, 0, 1]))

def rz(r):
	return np.array(([np.cos(r), -np.sin(r), 0, 0],[np.sin(r), np.cos(r), 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]))

def rxyz(a):
	r = a * math.pi / 180
	return rx(r[0]) @ ry(r[1]) @ rz(r[2])


# локальные -> мировые
def CalcMo2W(lock_v: np.array, a: np.array, input: np.array) -> np.array:
	return cdvig(lock_v) @ rxyz(a) @ zoom(input)


# нормировка вектора
def normal(v):
	v_normal = np.linalg.norm(v)
	return v_normal if v_normal == 0 else v / v_normal

# скалярное произведение
def scalar_vector(vec1, vec2):
	return np.sum(vec1 * vec2)

# задать значение пикселя на изображении
def DROW(img, x, y, color):
	if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
		img[y, x] = color
	return img


# алгоритм изображение линии
def ALG(img,x0, y0, x1, y1, color):
    a = False
    if x0>x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    
    if y1-y0 > x1-x0:
        x0, y0, x1, y1 = y0, x0, y1, x1
        a = True
    if abs(y1-y0) > abs(x1-x0):
        x0, y0, x1, y1 = y1, x1, y0, x0
        a = True
    delta_y = abs(y1 - y0)
    delta_x = abs(x1 - x0)
    error = 0.0
    y = y0
    for x in range(x0, x1+1):
        if a: DROW(img, y, x, color)
        else: DROW(img, x, y, color)
        error = error + delta_y
        if 2*error > delta_x:
            y += (1 if y1 > y0 else -1)
            error -= delta_x
    return img
# поворот камеры
def RCamera(loc_v, look_v):
	gv = normal(loc_v - look_v)
	bv = normal(np.cross(np.array([0, 1, 0]), gv))
	av = normal(np.cross(gv, bv))
	RC = np.eye(4) 

	RC[0, :3], RC[1, :3], RC[2, :3] = bv, av, gv

	return RC

# от мировых к камере
def CalcMw2c(lock_v, look_v):
	return RCamera(lock_v, look_v) @ cdvig(-lock_v)

# переход к преспективным координатам
def CalcPers(top, bottom, right, left, fear, near):
	return np.array(([2 * near / (right - left), 0, (right + left) / (right - left), 0], [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
					 [0, 0, -(fear + near) / (fear - near), - (2 * fear * near / (fear - near))], [0, 0, -1, 0]))

# переход к ортографическим координатам
def CalcOrth(top, bottom, right: int, left: int, fear: int, near: int) -> np.array:
	return np.array(([2 / (right - left), 0, 0, -(right + left) / (right - left)],
					 [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
					 [0, 0, -2 / (fear - near), -(fear + near) / (fear - near)],
					 [0, 0, 0, 1]))


# переход к окну вывода
def CalcMviewport(p: np.array, s: np.array) -> np.array:
	return cdvig([p[0] + (s[0] - 1) / 2, p[1] + (s[1] - 1) / 2, 0]) @ zoom([(s[0] - 1) / 2, (s[1] - 1) / 2, 1])

# класс рендеринга
class grafika:
	def __init__(self, model, texture, image_size):
		self.image_size = image_size

		with open(model, 'r') as file:
			data = [x.split() for x in file.read().splitlines()]

		# парсинг параметров модели
		v_list, vt_list, f_list = list(), list(), list()
		for d in data:
			if len(d) == 0:
				continue

			type_data = d[0]

			if type_data == 'v':
				v_list.append(d[1:])

			elif type_data == 'vt':
				vt_list.append(d[1:])

			elif type_data == 'f':
				f_list.append([d[1].split('/'), 
                               d[2].split('/'),
                                d[3].split('/')])
		self.v3 = np.array(v_list).astype(np.float32)
		self.vt = np.array(vt_list).astype(np.float32)
		self.f = np.array(f_list).astype(np.int32) - 1
		self.texture = np.asarray(Image.open(texture)).astype(np.int32)
		self.f_num = self.f.shape[0]

	def init(self, loc_v_model, rot_v_model,scl_v_model,locCam, lookCam, projection_select, top, bottom,right,left,fear,near, view_p, view_s):

		
		self.v4 = np.ones((self.v3.shape[0], 4)) # переход к проективным
		self.v4[:, :3] = self.v3
		
		self.locCam = locCam
		self.vecCam = normal(lookCam - locCam)

		self.v4 = (CalcMw2c(locCam, lookCam) @ CalcMo2W(loc_v_model, rot_v_model, scl_v_model) @ self.v4.T).T

		self.normals = np.zeros((np.size(self.f[:, 0, 0]), 3))
		self.viewVec = np.zeros((np.size(self.f[:, 0, 0]), 3)) 
		for i in range(self.f_num):
			x1, y1, z1 = self.get_point(i, 0)
			x2, y2, z2 = self.get_point(i, 1)
			x3, y3, z3 = self.get_point(i, 2)
			mean_f = np.mean(np.array([[x1, y1, z1],[x2, y2, z2],[x3, y3, z3]]), axis=0)
			self.normals[i] = normal(np.cross([x2 - x1, y2 - y1, z2 - z1], [x3 - x1, y3 - y1, z3 - z1]))
			self.viewVec[i] = normal(locCam - mean_f)
		
		# перспективная или ортографическая
		if projection_select:
			self.v4 = (CalcOrth(top, bottom, right, left, fear, near) @ self.v4.T).T
		else:
			self.v4 = (CalcPers(top, bottom, right, left, fear, near) @ self.v4.T).T
			self.v4[:, [0, 1]] /= self.v4[:, 2].reshape(-1, 1)
			self.v4[:, 3] = 1
		self.v4 = (CalcMviewport(view_p, view_s) @ self.v4.T).T
		self.v4 = np.int_(np.round(self.v4))

	def get_point(self, idx: int, num: int) -> [int, int, int]:
		return self.v4[self.f[idx, num, 0], 0], self.v4[self.f[idx, num, 0], 1], self.v4[self.f[idx, num, 0], 2]

	# проволочная модель
	def wire(self):
		image = np.zeros((self.image_size[0], self.image_size[1], 3)).astype(np.uint8)
		for i in range(self.f_num):
			x1, y1, z1 = self.get_point(i, 0)
			x2, y2, z2 = self.get_point(i, 1)
			x3, y3, z3 = self.get_point(i, 2)
			color_wire = [255, 255, 255]
			image = ALG(image, x1, y1, x2, y2, color_wire)
			image = ALG(image, x2, y2, x3, y3, color_wire)
			image = ALG(image, x3, y3, x1, y1, color_wire)
		return image

	def gray(self) -> np.array:
		image = np.zeros((self.image_size[0], self.image_size[1], 3)).astype(np.uint8)
		z_buffer = np.ones(self.image_size)
		for i in range(self.f_num):
			scl = scalar_vector(self.normals[i], self.vecCam)
			if scl < 0:
				continue

			x1, y1, z1 = self.get_point(i, 0)
			x2, y2, z2 = self.get_point(i, 1)
			x3, y3, z3 = self.get_point(i, 2)

			if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
				continue

			xmin, xmax = np.min([x1, x2, x3]), np.max([x1, x2, x3])
			ymin, ymax = np.min([y1, y2, y3]), np.max([y1, y2, y3])
			T_inv = np.linalg.inv(np.array([[x1, x2, x3],[y1, y2, y3],[1, 1, 1]]))
			for x_point in range(xmin, xmax + 1):
				for y_point in range(ymin, ymax + 1):
					if not (0 <= y_point < self.image_size[0] 
                    and 0 <= x_point < self.image_size[1]):
						continue
					X = np.array(([x_point, y_point, 1]))
					coefs_bara = T_inv @ X
					if not np.all(coefs_bara >= 0):
						continue

					# z-buffer
					z = scalar_vector(np.array([z1, z2, z3]), coefs_bara)
					if z < z_buffer[y_point, x_point]:
						image = DROW(image, x_point, y_point, 255 * np.abs(scl))
						z_buffer[y_point, x_point] = z
					else:
						continue
		return image.astype(np.uint8)

	def render_texture_model(self) -> np.array:
		image = np.zeros((self.image_size[0], self.image_size[1], 3)).astype(np.uint8)
		z_buffer = np.ones(self.image_size)
		for i in range(self.f_num):
			scl = scalar_vector(self.normals[i], self.vecCam)
			if scl < 0:
				continue

			vtx0, vty0, vtz0 = self.vt[self.f[i, 0, 1], 0], self.vt[self.f[i, 0, 1], 1], self.vt[self.f[i, 0, 1], 2]
			vtx1, vty1, vtz1 = self.vt[self.f[i, 1, 1], 0], self.vt[self.f[i, 1, 1], 1], self.vt[self.f[i, 1, 1], 2]
			vtx2, vty2, vtz2 = self.vt[self.f[i, 2, 1], 0], self.vt[self.f[i, 2, 1], 1], self.vt[self.f[i, 2, 1], 2]
			x1, y1, z1 = self.get_point(i, 0)
			x2, y2, z2 = self.get_point(i, 1)
			x3, y3, z3 = self.get_point(i, 2)

			if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
				continue
			xmin = np.min([x1, x2, x3])
			xmax = np.max([x1, x2, x3])
			ymin = np.min([y1, y2, y3])
			ymax = np.max([y1, y2, y3])
			T_inv = np.linalg.inv(np.array([[x1, x2, x3],
											[y1, y2, y3],
											[1, 1, 1]]))
			for x_point in range(xmin, xmax + 1):
				for y_point in range(ymin, ymax + 1):
					if not (0 <= x_point < self.image_size[1] and 0 <= y_point < self.image_size[0]):
						continue
					X = np.array(([x_point, y_point, 1]))
					coefs_bara = T_inv @ X
					if not np.all(coefs_bara >= 0):
						continue
						
					# z-buffer
					z = scalar_vector(np.array([z1, z2, z3]), coefs_bara)
					if z < z_buffer[y_point, x_point]:

						t_coord = [scalar_vector(coefs_bara, np.array([vtx0, vtx1, vtx2])), scalar_vector(coefs_bara, np.array([vty0, vty1, vty2]))]

						image = DROW(image, x_point, y_point, self.texture[int((1 - t_coord[1]) * (self.image_size[1] - 1)), int(t_coord[0] * (self.image_size[0] - 1))])
						z_buffer[y_point, x_point] = z
					else:
						continue
		return image.astype(np.uint8)

	def render_anim(self, count_frame, loc_v_model, rot_v_model, scl_v_model, locCam, lookCam, projection_select, top, bottom, right, left, fear, near, view_p: np.array, # левая верхная точка viewport
		view_s): # размер окна


		frames = []
		fig = plt.figure()

		defualt_pos = np.copy(locCam)
		defualt_pos = np.append(defualt_pos, 0)
		T_affin = cdvig(locCam)
		T_affin_inv = np.linalg.inv(T_affin)
		for i in range(count_frame + 1):
			fi = 2 * np.pi * (i / count_frame)
			new_pos = T_affin @ ry(fi) @ T_affin_inv @ defualt_pos
			new_pos = new_pos[:-1]

			
			self.init(loc_v_model, rot_v_model, scl_v_model, 
					new_pos, lookCam,
					projection_select,
					top, bottom, right, left, fear, near,
					view_p, view_s)

			
			image = self.wire()
			frames.append([plt.imshow(image)])

			print('Кадр {}'.format(i))

		ani = animation.ArtistAnimation(fig, frames, interval=30, blit=True, repeat_delay=0)
		writer = PillowWriter(fps=24)
		ani.save("rotate_camera.gif", writer=writer)
		print('The end')


def main():
	model = 'source/african_head.obj'
	texture = 'source/african_head_diffuse.tga'
	image_size = [1024, 1024]

	loc_v_model = np.array([0, 0, 0])
	rot_v_model = np.array([0, 180, 0])
	scl_v_model = np.array([17, 17, 17])

	locCam = np.array([0, 0, 175])
	lookCam = np.array([0, 0, 0])

	projection_select = False
	top = 25
	bottom = -25
	right = 25
	left = -25
	fear = -300
	near = 500

	# параметры окна вывода
	view_p = (0, 0)
	view_s = (1024, 1024)


	model_head = grafika(model=model, texture=texture, image_size=image_size)
	model_head.init(loc_v_model, rot_v_model, scl_v_model, locCam, lookCam, projection_select, top, bottom, right, left, fear, near, view_p, view_s)

	plt.figure('Проволочная модель')
	image = model_head.wire()
	plt.imshow(image)
	plt.show()

	plt.figure('Модель с серыми гранями')
	image = model_head.gray()
	plt.imshow(image)
	plt.show()

	plt.figure('Модель с текстурами')
	image = model_head.render_texture_model()
	plt.imshow(image)
	plt.show()

	count_frame = 30
	model_head.render_anim(count_frame, loc_v_model, rot_v_model, scl_v_model, locCam, lookCam, projection_select, top, bottom, right, left, fear, near, view_p, view_s) 

if __name__ == '__main__':
	main()