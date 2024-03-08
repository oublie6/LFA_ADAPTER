import matplotlib.pyplot as plt
import mplcursors

# 创建示例数据
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 使用mplcursors库
cursor = mplcursors.cursor(ax, hover=True)

@cursor.connect("add")
def on_add(sel):
    x_idx = sel.target[0]
    y_idx = sel.target[1]
    
    def update_text(sel):
        sel.annotation.set_text(f'({x_idx+1}, {y_idx+1})')
    
    def on_move(event):
        x_idx, y_idx = int(event.xdata), int(event.ydata)
        x[x_idx] = event.xdata
        y[y_idx] = event.ydata
        update_text(sel)
        sel.annotation.get_bbox_patch().set_alpha(0.8)
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    update_text(sel)

plt.show()