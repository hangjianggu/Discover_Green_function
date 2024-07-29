import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['text.usetex'] = False

def trapezoidal(x):
    """Trapezoidal weights for trapezoidal rule integration."""
    diff = np.diff(x, axis=0)
    weights = np.zeros(x.shape)
    weights[1:-1] = diff[1:] + diff[:-1]
    weights[0] = diff[0]
    weights[-1] = diff[-1]
    weights = weights / 2
    return weights

def mul(x, y):
    return np.multiply(x, y)

def add(x, y):
    return np.add(x, y)

def sub(x, y):
    return np.subtract(x, y)

def exp(x):
    return np.exp(x)

def large(x1,x2):
    # result = 1 if np.all(x1 > x2) else 0
    return (x1 > x2).astype(int)

def g(x1,x2):
    # result = 1 if np.all(x1 > x2) else 0
    return (x1 > x2).astype(int)

def less(x1,x2):
    # result = 1 if np.all(x1 <= x2) else 0
    return (x1 <= x2).astype(int)

def sin(x):
    return np.sin(x)

def sinh(x):
    return np.sinh(x)

def tanh(x):
    return np.tanh(x)

def n2(x):
    return x**2

def div(x,y):
    return np.divide(x,y)

pde = 'jump_green_0.5'
data = scio.loadmat(f'./dso/task/pde/data/{pde}.mat')

try:
    ExactGreen = data.get('ExactGreen')[0]
except:
    ExactGreen = ''
F = data.get('F')
U = data.get('U')
u_hom = data.get('U_hom')
xg = data.get('X')
yg = data.get('Y')
weight_y = trapezoidal(yg)
XG, YG = np.meshgrid(xg, yg)
x = XG.flatten()
y = YG.flatten()
FW = np.multiply(weight_y, F)

def G_expression(x, y):
    return eval(ExactGreen + '+ 0*x + 0*y')

G_exact = G_expression(XG, YG)
u_exact = np.matmul(G_exact.T, FW)+u_hom

def G_learn_expression(u1, u2):
    ## laplace
    # res = (-0.999986 * u1*u2 + 0.999995 * u1)*(u1 <= u2) + (1.000014 * u2 + -1.000016 * u1*u2)*(u1 > u2)

    ## advection
    # res = 1.7*(x*y-np.exp(y-y**2)+x)*(x<=y)-0.7*y*(x>y)
    # res = (4.307806 * x**2 * y - 3.258455 * x)*(x <= y) - 0.049034 * np.exp(np.exp(y))*(x > y)
    # res = -0.535 * (x <= y) - 0.585 * (x>y) * np.exp(y*(-1.127+0.87+np.exp(0.00072*(-6943*x+4384)*y+1.593)))
    # res = 3.68*y*(95+(0.045+x)*(9.3+x))
    # const = [-2.8874821119103835, -2.8291618346368117, -1.1176884929826825, -0.21874524575128418]
    # res = mul(sub(mul(u2,mul(exp(sub(mul(const[0],u1),const[1])),add(mul(u2,u2),const[2]))),const[3]),u1)
    # const = [1.091080217831772, -14.937762252484369, -12.983470311581772, 0.029481484458349617]
    # res = mul(sub(u2, mul(less(u2, u2), const[0])), add(mul(mul(sub(add(mul(u1, const[1]), u1), const[2]), u2), u1), const[3]))

    # const = [-0.5229970655704044, -5.041931432501887, 2.6566584054730207, 127.78838884586365, 127.78838886884009, -257.43169561626354]
    # res = \
    # 1.000000 * const[0] * (u1<=u2) + \
    # 1.000000 * mul(exp(const[1]), add(const[2], mul(add(mul(add(add(exp(u1), const[3]), const[4]), u1), const[5]), u1))) * (u1>u2)

    # const = [2.8902985574229194, 0.18310273063653598]
    # res = mul(sub(exp(u2),const[0]),add(u1,const[1])) * (u1<=u2) + \
    #       mul(sub(exp(u1), const[0]), add(u2, const[1])) * (u1>u2)

    ### helmholtz  K=2
    # res = (-0.000826 * y*np.sin(y))*(x <= y) + (-0.031250 * np.sin(2*x**2 - y - x))*(x > y)
    # const = [1.791040641914593,1.700459701290741,-0.09058616486299743,-0.015436367274511097]
    # res = mul(sin(sub(const[0],mul(const[1],sub(u2,mul(u1,const[2]))))),sub(const[3],u1)) * (u1 <= u2) + \
    #       mul(sin(sub(const[0],mul(const[1],sub(u1,mul(u2,const[2]))))),sub(const[3],u2)) * (u1 > u2)

    ### helmholtz K=3
    # const = [-0.0013344566026447634, 3.0561483385762935, 9.27353954215488, -10.417509494749734]
    # res = mul(mul(sin(mul(sub(const[0],u1),const[1])),add(u2,add(const[2],mul(const[3],u2)))),u2) * (u1 <= u2) + \
    #       mul(mul(sin(mul(sub(const[0],u2),const[1])),add(u1,add(const[2],mul(const[3],u1)))),u1) * (u1 > u2)

    # res = np.sin(3.0561483385762935 * (-0.0013344566026447634 - u1)) * (-9.417509494749734 * u2 + 9.27353954215488) * u2 * (u1 <= u2) + \
    #       np.sin(3.0561483385762935 * (-0.0013344566026447634 - u2)) * (-9.417509494749734 * u1 + 9.27353954215488) * u1 * (u1 > u2)

    # res = mul(mul(sin(mul(-u1, const[1])), add(const[2], mul(const[3]+1, u2))), u2) * (u1 <= u2) + \
    #       mul(mul(sin(mul(-u2, const[1])), add(const[2], mul(const[3]+1, u1))), u1) * (u1 > u2)

    # res = mul(mul(sin(mul(-u1, const[1])), add(const[2], mul(-const[2], u2))), u2) * (u1 <= u2) + \
    #       mul(mul(sin(mul(-u2, const[1])), add(const[2], mul(const[3]+1, u1))), u1) * (u1 > u2)

    ### boundary_layer
    # res = (1.102007 * np.sin(np.exp(np.exp(x-(np.exp(np.exp(y)) + y - 2*x)*x))))*(x <= y) + (0.014330 * x)*(x > y)

    ### nonlinear_biharmonic
    # res = (-0.004611 * np.exp(np.cos(x)) + -0.013700 * x * y + 0.083663 * x)*(x <= y) + (-0.008909 * y*np.cos(x) + 0.054248 * y + -0.007598 * x * y)*(x > y)

    ### airy
    # res = (-0.295335 * x + 0.396217 * np.sin(y) * np.sin(x))*(x <= y) + (0.387928 * np.exp(np.sin(y)) + -0.258604 * y + -0.443329 * np.exp(y-x) + -0.223681 * x)*(x > y)

    ### jump_green 0.7
    # res = -169.190538 * np.sin((np.log(np.tanh(np.sqrt(np.exp((y+np.tanh(np.tanh(y)))**2))))*y)**2)*(x <= y)
    # const = [0.2711100106535888, -0.9303066909230954]
    # res = mul(add(u1,const[0]),add(const[1],u2)) * (u1<=u2) + \
    #       mul(add(u2, const[0]), add(const[1], u1)) * (u1 > u2)

    # const = [1.0644423617498222, 2.2441425635191434, 4.429130142758898, 0.9355576164870617]
    # res = mul(u1, sub(add(u2, const[0]), add(u2, add(exp(sub(const[1], mul(const[2], u2))), const[3])))) * (u1 <= u2) + \
    #       mul(u2, sub(add(u1, const[0]), add(u1, add(exp(sub(const[1], mul(const[2], u1))), const[3])))) * (u1 > u2)

    # res = u1 * (const[0] - const[3] - exp(const[1] - const[2] * u2)) * (u1 <= u2) + \
    #       u2 * (const[0] - const[3] - exp(const[1] - const[2] * u1)) * (u1 > u2)

    ### jump_green 0.5
    # const = [0.39795509674907387, 0.9189769424713818, 1.0223874496455032, 0.518720647067885]
    # res = mul(add(sub(u1, mul(const[0], add(const[1], exp(u1)))), const[2]), sub(u2, add(const[3], u1))) * (u1 <= u2) + \
    #       mul(add(sub(u2, mul(const[0], add(const[1], exp(u2)))), const[2]), sub(u1, add(const[3], u2))) * (u1 > u2)

    const = [0.37136178274432213, 1.018840378922846, 9.15962676713713]
    res = mul(sub(u1, add(u2, const[0])), exp(mul(sub(mul(const[1], u1), u2), const[2]))) * (u1 <= u2) + \
          mul(sub(u2, add(u1, const[0])), exp(mul(sub(mul(const[1], u2), u1), const[2]))) * (u1 > u2)

    ### negative helmholtz
    # const = [4.2424622921408055, 1.889099608667547, 1.8893087305703886, -8.015867557385091, 8.272347209887615e-05]
    # res = mul(sinh(mul(const[0], sub(const[1], mul(const[2], u2)))), mul(sinh(mul(u1, const[3])), const[4])) * (u1<=u2) + \
    #       mul(sinh(mul(const[0], sub(const[1], mul(const[2], u1)))), mul(sinh(mul(u2, const[3])), const[4])) * (u1>u2)

    # res = 8.272347209887615e-05 * sinh(-8.015867557385091 * u1) * sinh(4.2424622921408055 * (1.889099608667547 - 1.8893087305703886* u2)) * (u1<=u2) + \
    #       8.272347209887615e-05 * sinh(-8.015867557385091 * u2) * sinh(4.2424622921408055 * (1.889099608667547 - 1.8893087305703886 * u1)) * (u1 > u2)

    ### schrodinger
    # const = [0.24716516131134278, 0.10354027732672516]
    # res = mul(const[0], add(const[1], sinh(tanh(exp(u1)))))* (u1<=u2)+ \
    #       mul(const[0], add(const[1], sinh(tanh(exp(u2))))) * (u1 > u2)

    # res = exp(sub(sub(u1,sub(u2,u1)),u2)) * (u1<=u2)+ \
    #       exp(sub(sub(u2,sub(u1,u2)),u1)) * (u1>u2)

    # const = [-4.973574031258685, -2.0139009247063697, 1.02048599548458, 1.0204859954849765]
    # res =  exp(sub(const[0],mul(add(const[1],u2),add(u1,add(const[2],const[3])))))* (u1<=u2)+ \
    #        exp(sub(const[0], mul(add(const[1], u1), add(u2, add(const[2], const[3]))))) * (u1 > u2)

    # const = [0.7407965809490832, 0.5588719481957463]
    # res = mul(div(add(exp(u1), add(const[0], u1)), exp(exp(u2))), const[1]) * (u1<=u2) + \
    #       mul(div(add(exp(u2), add(const[0], u2)), exp(exp(u1))), const[1]) * (u1 > u2)

    # const = [-0.5944178295456755, 0.9721449281688718, -0.14783553259117513, -0.1478355325911842]
    # res = mul(sin(const[0]),mul(sub(u2,const[1]),add(const[2],add(const[3],exp(u1))))) * (u1<=u2) + \
    #       mul(sin(const[0]), mul(sub(u1,const[1]), add(const[2], add(const[3], exp(u2))))) * (u1>u2)

    ### periodic_helmholtz K=3
    # res = sin(abs(n2(sub(u1,tanh(u1))))) * (u1<=u2)+ \
    #       sin(abs(n2(sub(u2,tanh(u2))))) * (u1>u2)

    # const = [-0.3627211587143596, 0.0617374333326827, 0.06173743700346199, 12.685243670020096, 0.02580931629111419]
    # res = 1.000000 * mul(add(add(const[0], sin(mul(add(u2, add(sub(const[1], u1), const[2])), const[3]))), u2), const[4])

    # const = [0.1467851881443401, 2.0801645816079164, -0.500978354449627, 0.17173227322613638]
    # res = 1.000000 * mul(sub(add(u1,const[0]),mul(mul(const[1],u2),sin(sin(add(const[2],u1))))),const[3])

    # const = [-1.5974403607765615, -4.902393804099493, -4.7147280030466225]
    # res = 1.000000 * sin(sub(add(sub(mul(const[0], sin(sub(u2, add(u1, const[1])))), const[2]), u2), u1)) * (u1<=u2)+ \
    #       1.000000 * sin(sub(add(sub(mul(const[0], sin(sub(u1, add(u2, const[1])))), const[2]), u1), u2)) * (u1>u2)

    # const = [0.16708520067155852, 0.8992963754793366, 0.023598608783566143, 2.783785216430562, 0.39944917902714805, 3.000000462879595]
    # res = 1.000000 * mul(const[0],sin(mul(add(u2, mul(const[1], mul(mul(sub(const[2], u1), const[3]), const[4]))), const[5]))) * (u1<=u2)+ \
    #       1.000000 * mul(const[0],sin(mul(add(u1, mul(const[1], mul(mul(sub(const[2], u2), const[3]), const[4]))), const[5]))) * (u1>u2)


    ### periodic_helmholtz K=1 Reward: 0.9899931406037226 MSE:6.6391842052876244e-12 Epoch 6
    # const = [1.0429180312468367, 2.070785823711078]
    # res = mul(const[0], sin(add(const[1], sub(u1, u2)))) * (u1<=u2) + \
    #       mul(const[0], sin(add(const[1], sub(u2, u1)))) * (u1 > u2)

    ### periodic_helmholtz K=2 Reward: 0.9899860787934304 MSE:2.6219599420768265e-11 Epoch 14
    # const = [0.29709875225770926, 1.095132024722128, 0.9048658063990868, 1.09513202472194, 1.9999993638354403]
    # res = mul(const[0], sin(mul(sub(const[1], sub(const[2], sub(u1, sub(u2, const[3])))), const[4]))) * (u1<=u2) + \
    #       mul(const[0], sin(mul(sub(const[1], sub(const[2], sub(u2, sub(u1, const[3])))), const[4]))) * (u1 > u2)

    ### periodic_helmholtz K=4 Reward: 0.9899618639936627 MSE:1.5890444011593202e-10 Epoch 6
    # const = [0.2364376370882087, 1.292382557607218, 0.7737658954668436, 0.2902489278637268, 4.000016227358466, 0.13746810040992372]
    # res = mul(sin(mul(add(u2, sub(mul(sub(const[0], mul(const[1], u1)), const[2]), const[3])), const[4])), const[5]) * (u1 <= u2) + \
    #       mul(sin(mul(add(u1, sub(mul(sub(const[0], mul(const[1], u2)), const[2]), const[3])), const[4])), const[5]) * (u1 > u2)

    ### periodic_helmholtz K=5 Reward: 0.9899693476698538 MSE:8.625615703644965e-11 Epoch 6
    # const = [0.5609408377703974, 1.1657670172905357, 0.8578050693056026, 0.6670194559035478, 5.00000863205386, 0.167091591512985]
    # res = mul(sin(mul(add(u2, sub(mul(sub(const[0], mul(const[1], u1)), const[2]), const[3])), const[4])), const[5]) * (u1 <= u2) + \
    #       mul(sin(mul(add(u1, sub(mul(sub(const[0], mul(const[1], u2)), const[2]), const[3])), const[4])), const[5]) * (u1 > u2)

    # ### periodic_helmholtz K=6 Reward: 0.9899914300174356 MSE:7.342213958635686e-12 Epoch 1
    # const = [-0.5905138154870413, 5.999998725967086, 0.23820059734882124]
    # res = mul(const[0], sin(mul(const[1], add(const[2], sub(u1, u2))))) * (u1 <= u2) + \
    #       mul(const[0], sin(mul(const[1], add(const[2], sub(u2, u1))))) * (u1 > u2)

    ### periodic_helmholtz K=7 Reward: 0.989977662404203 MSE:6.397318980669839e-11 Epoch 1
    # const = [0.20362590344221626, 7.0000022082913524, 0.2756005475488402]
    # res = mul(const[0], sin(mul(const[1], add(const[2], sub(u1, u2))))) * (u1 <= u2) + \
    #       mul(const[0], sin(mul(const[1], add(const[2], sub(u2, u1))))) * (u1 > u2)

    ### periodic_helmholtz K=8 Reward: 0.9899454898189566 MSE:2.6440759375613776e-10 Epoch 1
    # const = [0.08258458166619653, 8.000005139928755, 0.3036512031776696]
    # res = mul(const[0], sin(mul(const[1], add(const[2], sub(u1, u2))))) * (u1 <= u2) + \
    #       mul(const[0], sin(mul(const[1], add(const[2], sub(u2, u1))))) * (u1 > u2)

    ### periodic_helmholtz K=9 Reward: 0.9899182546539744 MSE:7.514796001695088e-10 Epoch 1
    # const = [0.05683233041232587, 8.999985946575542, 0.3254676068572878]
    # res = mul(const[0], sin(mul(const[1], add(const[2], sub(u1, u2))))) * (u1 <= u2) + \
    #       mul(const[0], sin(mul(const[1], add(const[2], sub(u2, u1))))) * (u1 > u2)

    ### periodic_helmholtz K=10 Reward: 0.9899032677749764 MSE:4.4223741458918036e-10 Epoch 21
    # const = [0.05214185057675406, 3.1960025005700546e-06, 9.99998255385074, 9.000001620344719, 3.4292116119360196]
    # res = mul(const[0], add(const[1], sin(add(sub(mul(const[2], u1), mul(const[3], u2)), sub(const[4], u2)))))* (u1 <= u2) + \
    #       mul(const[0], add(const[1], sin(add(sub(mul(const[2], u2), mul(const[3], u1)), sub(const[4], u1))))) * (u1 > u2)


    ### cubic_helmholtz
    # const = [-0.13531419742603037, 1.8450915331433742, 0.8379293319647101, 0.7423786987879455]
    # res = 1.000000 * mul(const[0], add(sin(mul(add(u2, sub(const[1], u1)), const[2])), const[3]))

    ### vis
    # res = 1.000000 * mul(sub(sub(u2, sigmoid(sub(u1, u1))), sigmoid(u1)), sigmoid(u1))

    # advection_diffusion_jump
    # const = [0.5073699891533507, 1.4926299971973698, 1.3053149209363766]
    # res = mul(add(const[0], sub(u2, const[1])), add(const[2], u1))* (u1 <= u2) + \
    #       mul(add(const[0], sub(u1, const[1])), add(const[2], u2)) * (u1 > u2)

    return res

def G_learn_pysr(x0, x1):
    # res = np.sin(x * (np.exp(1.32) - y)) * (np.sin(((y * np.exp(1.16)) - np.sin(-0.54)) - x) * 0.2)
    res = 0.12115009 + ((((x1 - x0) - x0) + x1) * -0.056956377) * (((x1 - x0) - x0) + x1)

    res = (-0.4486167 - g(sin(g(sin(g(sin(g(x0, 0.1743755)), x0)), x1)), sin(x1 - -1.9800668)))
    return res

G_learn = G_learn_expression(XG, YG)
u_learn = np.matmul(G_learn.T, FW)+u_hom

error = np.mean((U - u_learn) ** 2)
print('error: ', error)
reward = (1-0.01)/(1 + np.sqrt(np.mean((U - u_learn)**2)/np.var(U)))
print('reward: ', reward)


# G_learn_pysr = G_learn_pysr(XG, YG)
# u_learn_pysr = np.matmul(G_learn_pysr.T, FW)+u_hom

# fig, axs = plt.subplots(1, 3, figsize=(9, 3))
fig = plt.figure(figsize=(9, 3))
gs = gridspec.GridSpec(1, 3)
gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.6, hspace=0.6)


ax = plt.subplot(gs[0])
# h = ax.imshow(G_exact, interpolation='lanczos', cmap='jet',
#               extent=[np.min(x), np.max(x),
#                       np.min(y), np.max(y)],
#               origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(h, cax=cax)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$', rotation=0, labelpad=12)
ax.set_title('(a) Exact Green\'s function', fontsize=10)
# ax.set_aspect('equal')

# Plot the learned Green's function
ax = plt.subplot(gs[1])
h = ax.imshow(G_learn, interpolation='lanczos', cmap='jet',
              extent=[np.min(x), np.max(x),
                      np.min(y), np.max(y)],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$', rotation=0, labelpad=12)
ax.set_title('(b) Learned Green\'s function by DISCOVER', fontsize=10)
# ax.set_aspect('equal')

# ax = axs[1, 0]
# h = ax.imshow(G_learn_pysr, interpolation='lanczos', cmap='jet',
#               extent=[np.min(x), np.max(x),
#                       np.min(y), np.max(y)],
#               origin='lower', aspect='auto')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(h, cax=cax)
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$', rotation=0, labelpad=12)
# ax.set_title('(c) Learned Green\'s function by PYSR', fontsize=10)


ax = plt.subplot(gs[2])
ax.plot(xg, U[:, 15], label='Exact u')
ax.plot(xg, u_learn[:, 15], linestyle=':', label='u calculated by DISCOVER')
# ax.plot(xg, u_learn_pysr[:, 15], linestyle=':', label='u calculated by PySR')
divider = make_axes_locatable(ax)
ax.set_xlim(np.min(xg), np.max(xg))
ax.set_ylim(np.min(U[:, 15]), np.max(U[:, 15]))
ax.set_xlabel('$x$')
ax.set_title('(c) Solution comparisons', fontsize=10)
ax.legend(loc='upper center')
# ax.set_aspect(1)
# for ax in axs.flat:
#     ax.axis('off')
# plt.tight_layout()
# for ax in axs.flat:
#     ax.set_aspect('equal')


# plt.subplots_adjust(wspace=1.5)
plt.savefig(f"result/{pde}.pdf")
plt.show()

