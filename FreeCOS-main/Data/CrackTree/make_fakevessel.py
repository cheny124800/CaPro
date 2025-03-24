#from turtle import *
import os
import turtle
import numpy as np
import random
import math
from PIL import Image
#tempNumpyArray=np.load("DRIVE_instensity.npy")
#score = tempNumpyArray.tolist()

def draw_background(a_turtle):
    """ Draw a background rectangle. """
    ts = a_turtle.getscreen()
    canvas = ts.getcanvas()
    height = ts.getcanvas()._canvas.winfo_height()
    width = ts.getcanvas()._canvas.winfo_width()

    turtleheading = a_turtle.heading()
    turtlespeed = a_turtle.speed()
    penposn = a_turtle.position()
    penstate = a_turtle.pen()

    a_turtle.penup()
    a_turtle.speed(0)  # fastest
    a_turtle.goto(-width/2-2, -height/2+3)
    # print("turtle.Screen().bgcolor()",turtle.Screen().bgcolor())
    a_turtle.fillcolor(255,255,255)
    a_turtle.begin_fill()
    a_turtle.setheading(0)
    a_turtle.forward(width)
    a_turtle.setheading(90)
    a_turtle.forward(height)
    a_turtle.setheading(180)
    a_turtle.forward(width)
    a_turtle.setheading(270)
    a_turtle.forward(height)
    a_turtle.end_fill()

    a_turtle.penup()
    a_turtle.setposition(*penposn)
    a_turtle.pen(penstate)
    a_turtle.setheading(turtleheading)
    a_turtle.speed(turtlespeed)

def makecolor(turtle,new_length,width,x,y,theta):
    #Tile one pixel or two pixel
    turtle_screen.colormode(255)
    # system.draw(turtle)
    #Need more set of different colorset#Get random colorset
    random_R = np.random.randint(0,255)
    random_G = np.random.randint(0, 255)
    random_B = np.random.randint(0, 255)
    R_insity = random_R
    G_insity = random_G
    B_insity = random_B

    # while R_insity<180:
    #     idx_random_2 = np.random.randint(0, len(score) - 1)
    #     R_insity = score[ridx_random_2][0]
    Ar1 = np.random.uniform(-3,3)
    Ag1 = np.random.uniform(-3,3)
    Ab1 = np.random.uniform(-3,3)
    turtle.pu()
    turtle.goto((x, y))
    turtle.setheading(theta)
    width = width
    if width<2:
        width = 2
    turtle.pensize(width)
    #init_width too thin single guassian<=2
    new_instenisty_r = int(np.clip(R_insity + Ar1,0,255))
    new_instenisty_g = int(np.clip(G_insity + Ag1,0,255))
    new_instenisty_b = int(np.clip(B_insity + Ab1,0,255))
    turtle.pencolor(new_instenisty_r, new_instenisty_g, new_instenisty_b)
    #init_width Too thick with bin_guassian>2
    #new_instenisty = 255 - guassian(Ar1, width / 2, 0, width / 2)
    turtle.pd()
    turtle.forward(new_length)
    turtle.pu()
    turtle.goto((x, y))
    turtle.setheading(theta)
    turtle.pd()
    # turtle.pensize(width/2)
    # new_instenisty_r = np.maximum(int(255 - guassian(Ar1, width/2, 0, width/2)),0)
    # new_instenisty_g = np.maximum(int(255 - guassian(Ag1, width / 2, 0, width / 2)),0)
    # new_instenisty_b = np.maximum(int(255 - guassian(Ab1, width / 2, 0, width / 2)),0)
    turtle.pencolor(new_instenisty_r, new_instenisty_g, new_instenisty_b)
    #turtle.pensize(10)
    turtle.forward(new_length)

class LSystem_vessel():
    def __init__(self, axiom, rules, rules_2, rules_3, theta=0, width=5, dtheta_1=40, dtheta_2=30, start=(-350,0), length=80,iteration=3,width_1=0.79, width_2 = 0.5):
        self.sentence = axiom
        self.rules = rules
        self.relus_2 = rules_2
        self.rules_3 = rules_3
        self.iteration = iteration
        self.width = width
        self.theta = theta
        self.dtheta_1 = dtheta_1
        self.dtheta_2 = dtheta_2
        self.length = length
        self.positions = []
        self.start = start
        self.lamda_1 =  width_1
        self.lamda_2 = width_2

        self.x, self.y = start
        self.lines = []  # 新增属性，用于存储线段坐标

    def __str__(self):
        return self.sentence

    def generate(self):
        self.x, self.y = self.start
        for iter in range(self.iteration):
            newStr = ""
            for char in self.sentence:
                mapped = char
                try:
                    p = random.random()
                    #mapped = self.rules[char]
                    if p>0.4:
                        mapped = self.rules[char]
                    elif 0.8>p>=0.4:
                        mapped = self.rules_3[char]
                    else:
                        mapped = self.relus_2[char]
                except:
                    pass
                newStr += mapped
            self.sentence = newStr

    def draw(self, turtle):
        #print("sentence",self.sentence)


        turtle.pu()
        turtle.hideturtle()
        turtle.speed(0)
        turtle.goto((self.x, self.y))
        turtle.setheading(self.theta)
        flag = False
        for char in self.sentence:
            turtle.pd()
            #LR_p = random.random()
            if char == 'F' or char == 'G':
                # p = random.random()
                # print("p",p)
                # if p>0.8:
                #width = np.random.randint(low=2, high=8)
                turtle.pu()
                turtle.setheading(self.theta)
                if flag == True:
                    new_length = self.length*self.lamda_1
                else:
                    new_length = self.length*self.lamda_2
                turtle.pensize(self.width)
                #print("x,y", turtle.position())
                x, y = turtle.position()
                theta = self.theta
                width = self.width
                #makecolor(turtle, new_length)
                makecolor(turtle, new_length, width, x, y, theta)
                #turtle.forward(new_length)
                self.x, self.y = turtle.position()
                #("x,y",turtle.position())
                #print("Head",turtle.heading())
                # 将线段的起点、终点、角度、宽度存储到 lines 中
                x_end, y_end = turtle.position()
                self.lines.append({
                    "x_start": x, "y_start": y,
                    "x_end": x_end, "y_end": y_end,
                    "angle": theta, "width": width
                })
            elif char == '+':
                #self.theta += self.dthetad
                dtheta = np.random.randint(low=10, high=40)
                self.theta += dtheta
                self.width = self.width * self.lamda_1
                turtle.right(self.theta)
            elif char == '-':
                #self.theta -= self.dtheta
                self.width = self.width * self.lamda_2
                dtheta = np.random.randint(low=10, high=40)
                self.theta -= dtheta
                turtle.left(self.theta)
            elif char == '[':
                #self.positions.append({'x' : self.x, 'y' : self.y, 'theta' : self.theta,'width':self.width,"new_width_0":self.width*0.79})
                self.positions.append({'x': self.x, 'y': self.y, 'theta': self.theta, 'width': self.width, 'length': self.length,"new_width_0": self.width * 0.79})
                flag = True
            elif char == ']':
                turtle.pu()
                position = self.positions.pop()
                #self.x, self.y, self.theta, self.width = position['x'], position['y'], position['theta'], position['width']
                self.x, self.y, self.theta, self.width = position['x'], position['y'], position['theta'], position['width']
                flag = False
                turtle.goto((self.x, self.y))
                turtle.setheading(self.theta)


def save_txt(txt_data_folder, i, x1_top, y1_top, x2_top, y2_top, x1_bottom, y1_bottom, x2_bottom, y2_bottom):
    """
    计算矩形框相关参数并保存到txt文件。
    """

    # 检查坐标是否超出范围
    def is_out_of_range(x, y):
        return not (-450 <= x <= 450 and -450 <= y <= 450)

    # 如果有任何一个坐标超出范围，直接返回
    if any(is_out_of_range(x, y) for x in [x1_top, x1_bottom, x2_top, x2_bottom] for y in
           [y1_top, y1_bottom, y2_top, y2_bottom]):
        print("坐标超出范围，退出函数。")
        return

    print((f"原始数据：{x1_top},{y1_top},{x2_top},{y2_top},{x2_bottom},{y2_bottom},{x1_bottom},{y1_bottom}"))
    # 将坐标转换为 512x512 坐标系
    def convert_to_512x512(x, y):
        # 转换x坐标
        x_new = (x + 389.5) * 512.0 / 779.0
        # 转换y坐标并反转
        y_new = (389.5 - y) * 512.0 / 779.0
        return x_new, y_new

    # 转换所有输入的顶点坐标
    x1_top, y1_top = convert_to_512x512(x1_top, y1_top)
    x1_bottom, y1_bottom = convert_to_512x512(x1_bottom, y1_bottom)
    x2_top, y2_top = convert_to_512x512(x2_top, y2_top)
    x2_bottom, y2_bottom = convert_to_512x512(x2_bottom, y2_bottom)
    # 保存为txt文件
    if not os.path.exists(txt_data_folder):
        os.makedirs(txt_data_folder)

    file_name = os.path.join(txt_data_folder, f"{i}.txt")

    # 将数据写入txt文件
    with open(file_name, 'a') as f:
        f.write(f"{x1_top},{y1_top},{x2_top},{y2_top},{x2_bottom},{y2_bottom},{x1_bottom},{y1_bottom},crack\n")

    # print(f"Saved data to {file_name}")


def draw_tilted_rectangle(x1, y1, x2, y2, width):
    """
    根据线段的起点和终点坐标绘制与线段角度一致的外接矩形
    """
    # 计算单位向量
    length_factor = 1
    # 计算线段的中点
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    dx = x2 - x1
    dy = y2 - y1
    # print(f"width = {width}")
    if 0 < width <= 1.2:
        length_factor = 8
    elif 1.2 < width < 2.2:
        length_factor = 4.8
    elif 2.2 <= width < 3.95:
        length_factor = 2.5
    else:
        length_factor = 1.5
    # length_factor = 1
    length = math.sqrt(dx ** 2 + dy ** 2)

    dx = dx * length_factor
    dy = dy * length_factor
    unit_dx = dx / length
    unit_dy = dy / length


    # 计算垂直方向的单位向量
    perp_dx = -unit_dy
    perp_dy = unit_dx

    # 根据宽度和padding计算矩形的四个顶点
    # half_width = width * np.random.uniform(2.7, 3)
    half_width = width * np.random.uniform(2.7, 3)

    x1_top = x1 + perp_dx * half_width
    y1_top = y1 + perp_dy * half_width
    x1_bottom = x1 - perp_dx * half_width
    y1_bottom = y1 - perp_dy * half_width

    x2_top = x2 + perp_dx * half_width
    y2_top = y2 + perp_dy * half_width
    x2_bottom = x2 - perp_dx * half_width
    y2_bottom = y2 - perp_dy * half_width

    # 在绘制矩形之前调用save_txt保存数据
    save_txt('txt_data', i, x1_top, y1_top, x2_top, y2_top, x1_bottom, y1_bottom, x2_bottom, y2_bottom)

    '''''''''''
    # 绘制矩形框
    turtle.pencolor(0, 255, 0)  # 设置画笔颜色为绿色
    turtle.pu()

    turtle.goto(x1_top, y1_top)  # 移动到矩形的第一个点
    turtle.pd()
    turtle.goto(x2_top, y2_top)  # 绘制第一条边
    turtle.goto(x2_bottom, y2_bottom)  # 绘制第二条边
    turtle.goto(x1_bottom, y1_bottom)  # 绘制第三条边
    turtle.goto(x1_top, y1_top)  # 绘制第四条边（闭合矩形）
    turtle.pu()
    turtle.pencolor(0, 0, 0)  # 设置画笔颜色为黑色
    '''''''''''

# rules = {"F":"F-F[+F-F][-F+F]"}
# rules_2 = {"F":"F+F[+F[+F]-F]-F+F[+F-F[+F]-F]"}
# rules_3 = {"F":"F[+F]+F[+F]+F[+F]"}
# rules_4 = {"F":"F-F-F[+F-F][-F-F]F+F"}
#rules_2 = {"F":"FF-F[-F+F+F]+[+F-F-F]"}
#rules_3 = {"F":"F"}


#crack kind need random start postion
rules = {"F":"F-F+F[+F-F+F-F][-F+F]"}
# OCT_rules_2
rules_2 = {"F":"F[-F+F+F][+F-F-F+F]"}
rules_3 = {"F":"F[+F]+F[+F]+F[+F]"}
rules_4 = {"F":"F-F-F[+F-F][-F-F]F+F"}
#net kind(0,-350,0) (180,0,350)
#rules = {"F":"F-F+F[+F-F]"}
#rules_2 = {"F":"F+F+F-F[-F+F]"}

# rules = {"F":"F"}
# CRACK_rules_2
# rules_2 = {"F":"F"}
path = "[+F-F][-F]"
print(path)

#parameter Range
#init_theta -20-20
#width5-30
#dtheta20-40
#stare=(-350~-150,-100~100)
#init_theta 160-200 stare=(350~150,-100~100)
#length = 80~110
#iteration = 3~4
#width_1=0.79, width_2 = 0.5
Num_image = 1000
# Start_theta = (-20,20)
# Start_theta_2 = (160,200)
# Start_position_x = (-350, -150)
# Start_position_y = (-100, -100)
# Start_position_x2 = (150, 350)
# Ratio_LW = (0.5,1)
#
# Dtheta = (20,50)
# Width = (5,30)
# Length_range = (80, 110)

Start_theta = (-80,20)
Start_theta_2 = (170,210)
#Start_theta = (20,30)
#Start_theta_2 = (25,35)
Start_position_x = (-350, -150)
Start_position_y = (-100, -100)
Start_position_x2 = (150, 350)
Ratio_LW = (0.7,1)

Dtheta = (20,120)
#Width = (8,14) #XCAD/DRIVE/STARE
#Width = (2,12)
#Width = (20,30) #STARE
#Width = (4, 6) #Crack
#Width = (2, 4) #Crack
Width = (2, 8) #DRIVE_new
#Length_range = (90, 150) #XCAD STARE DRIVE
Length_range = (120, 200) #XCAD STARE DRIVE


for i  in range(0, Num_image):
    p2 = random.random()
    if p2>0.5:
        path = "[+F-F][-F]"
    else:
        path = "F"
    p_vessel = random.random()
    if p_vessel>0.5:
        r1= rules
        r2= rules_2
        r3 = rules_3
    else:
        r1 = rules_4
        r2 = rules_3
        r3 = rules
    print("image_num", i)
    dtheta_1 = np.random.uniform(Dtheta[0],Dtheta[1])
    dtheta_2 = np.random.uniform(Dtheta[0],Dtheta[1])
    init_width = np.random.randint(Width[0],Width[1]+1)
    init_length = np.random.randint(Length_range[0],Length_range[1]+1)
    iteration = np.random.randint(1,3)
    #iteration = 1

    Ratio_lw_1 = np.random.uniform(Ratio_LW[0],Ratio_LW[1])
    Ratio_lw_2 = np.random.uniform(Ratio_LW[0], Ratio_LW[1])

    p = random.random()
    if p>0.5:
        init_theta = np.random.uniform(Start_theta[0], Start_theta[1])
        x_position = np.random.randint(Start_position_x[0],Start_position_x[1]+1)
    else:
        init_theta = np.random.uniform(Start_theta_2[0], Start_theta_2[1])
        x_position = np.random.randint(Start_position_x2[0],Start_position_x2[1]+1)
    y_position = np.random.randint(Start_position_y[0],Start_position_y[1]+1)

    system = LSystem_vessel(path, r1, r2, r3, theta=init_theta, width=init_width, dtheta_1=dtheta_1, dtheta_2=dtheta_2, start=(x_position, y_position),length=init_length, iteration=iteration, width_1=Ratio_lw_1, width_2=Ratio_lw_2)
    system.generate()
    turtle.speed(0)
    turtle.delay(0)
    turtle.tracer(False)
    turtle_screen = turtle.Screen()  # create graphics window
    turtle_screen.colormode(255)
    #ratip=1.36
    turtle.setup(800,800)
    turtle_screen.screensize(800, 800)
    turtle.bgcolor(0,0,0)
    draw_background(turtle)
    system.draw(turtle)

    # print(f"img{i}:")
    for line in system.lines:
        x1 = line["x_start"]
        y1 = line["y_start"]
        x2 = line["x_end"]
        y2 = line["y_end"]
        angle = line["angle"]
        width = line["width"]

        draw_tilted_rectangle(x1, y1, x2, y2, width)

    file_name = "./fake_rgbvessel_thin/"+str(i)+'.png'
    file_origin = "./fake_rgbvessel_thin/ " + str(i) + 'origin' +'.png'
    # 确保目标文件夹存在
    output_dir = './fake_rgbvessel_thin'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tsimg = turtle.getscreen()
    tsimg.getcanvas().postscript(file="work_vessel.eps")
    im = Image.open("work_vessel.eps")
    # im.save(file_origin)
    out = im.resize((512,512))
    #out = im
    im_array = np.array(out)
    print("im_array",im_array.shape)
    out.save(file_name)
    turtle.reset()