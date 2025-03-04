import threading
import random
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

def validate_coord(x: float) -> float:
    if 0 <= x < 1:
        return x
    raise ValueError("Coordinate must be in [0, 1).")

class Node:
    """
    Represents a node (leaf or internal) of the quadtree covering the unit square.
    """
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float, 
                 level: int = 0, parent = None):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.level = level
        self.parent = parent
        self.children = None    
        self.load = 0.0        
    
    def is_leaf(self) -> bool:
        return self.children is None
    
    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max
    
    def subdivide(self) -> None:
        """Subdivide this node into four quadrants and distribute load evenly."""
        mid_x = (self.x_min + self.x_max) / 2.0
        mid_y = (self.y_min + self.y_max) / 2.0
        self.children = {
            0: Node(self.x_min, self.y_min, mid_x, mid_y, self.level + 1, self),
            1: Node(mid_x, self.y_min, self.x_max, mid_y, self.level + 1, self),
            2: Node(self.x_min, mid_y, mid_x, self.y_max, self.level + 1, self),
            3: Node(mid_x, mid_y, self.x_max, self.y_max, self.level + 1, self)
        }
        distributed_load = self.load / 4.0
        for child in self.children.values():
            child.load = distributed_load
        self.load = 0.0
        print(f"Subdividing node at level {self.level}: [{self.x_min}, {self.y_min}) -> "
              f"[{self.x_max}, {self.y_max})")
    
    def can_merge(self, merge_threshold: int) -> bool:
        """Return True if all children are leaves and their total load is below merge_threshold."""
        if self.is_leaf():
            return False
        total = sum(child.load for child in self.children.values())
        if any(not child.is_leaf() for child in self.children.values()):
            return False
        return total < merge_threshold
    
    def merge(self) -> None:
        """Merge children back into this node, aggregating their load."""
        total_load = sum(child.load for child in self.children.values())
        print(f"Merging node at level {self.level}: [{self.x_min}, {self.y_min}) -> "
              f"[{self.x_max}, {self.y_max}) (child load = {total_load:.2f})")
        self.children = None
        self.load = total_load

class FractalGrid:
    """
    A quadtree over the unit square with lazy subdivision, merging, and load decay.
    All operations are synchronized using a global reentrant lock.
    """
    def __init__(self, x_min=0, y_min=0, x_max=1, y_max=1, max_level=6, 
                 split_threshold=5, merge_threshold=2, decay_factor=0.9, decay_interval=1.0):
        self.root = Node(x_min, y_min, x_max, y_max)
        self.max_level = max_level
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        self.lock = threading.RLock()
    
    def _find_leaf(self, node: Node, x: float, y: float) -> Node:
        with self.lock:
            current = node
            while not current.is_leaf():
                found = False
                for child in current.children.values():
                    if child.contains(x, y):
                        current = child
                        found = True
                        break
                if not found:
                    break
            return current
    
    def get(self, x: float, y: float):
        x = validate_coord(x)
        y = validate_coord(y)
        with self.lock:
            leaf = self._find_leaf(self.root, x, y)
            leaf.load += 1.0
            return leaf.load
    
    def set(self, x: float, y: float, _unused=None) -> None:
        x = validate_coord(x)
        y = validate_coord(y)
        with self.lock:
            leaf = self._find_leaf(self.root, x, y)
            leaf.load += 1.0
            self._rebalance(leaf)
    
    def _rebalance(self, node: Node) -> None:
        if (node.load > self.split_threshold and node.level < self.max_level 
                and node.is_leaf()):
            node.subdivide()
    
    def periodic_rebalance(self) -> None:
        with self.lock:
            self._merge_repeatedly(self.root)
    
    def _merge_repeatedly(self, node: Node) -> None:
        if not node.is_leaf():
            for child in node.children.values():
                self._merge_repeatedly(child)
            if node.can_merge(self.merge_threshold):
                node.merge()
    
    def decay_load(self, decay_factor: float) -> None:
        """Decay the load on all leaves. Loads below 1e-6 are set to 0."""
        with self.lock:
            def decay(n: Node):
                if n.is_leaf():
                    n.load *= decay_factor
                    if n.load < 1e-6:
                        n.load = 0.0
                else:
                    for c in n.children.values():
                        decay(c)
            decay(self.root)
    
    def get_leaves(self):
        with self.lock:
            def gather(n: Node):
                if n.is_leaf():
                    return [n]
                result = []
                for c in n.children.values():
                    result.extend(gather(c))
                return result
            return gather(self.root)

class LoadDecayThread(threading.Thread):
    """
    Periodically decays the load of all leaves.
    """
    def __init__(self, grid: FractalGrid):
        super().__init__()
        self.grid = grid
        self.running = True

    def run(self):
        while self.running:
            time.sleep(self.grid.decay_interval)
            self.grid.decay_load(self.grid.decay_factor)
            print(f"Decayed load by factor {self.grid.decay_factor}")

    def stop(self):
        self.running = False

class GlobalAdaptiveController(threading.Thread):
    def __init__(self, grid: FractalGrid, interval: float = 3.0):
        super().__init__()
        self.grid = grid
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            time.sleep(self.interval)
            with self.grid.lock:
                print(f"Adaptive Controller: split_threshold = {self.grid.split_threshold}, "
                      f"merge_threshold = {self.grid.merge_threshold}")
            self.grid.periodic_rebalance()

    def stop(self):
        self.running = False

def worker_loop(thread_id: int, grid: FractalGrid, pause_event: threading.Event) -> None:
    while True:
        if pause_event.is_set():
            time.sleep(0.1)
            continue
        x = random.uniform(0.45, 0.55)
        y = random.uniform(0.45, 0.55)
        grid.set(x, y)
        time.sleep(0.005)

def update_anim(frame, grid: FractalGrid, ax, max_level: int):
    with grid.lock:
        leaves = grid.get_leaves()
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    for node in leaves:
        w = node.x_max - node.x_min
        h = node.y_max - node.y_min
        color = plt.cm.viridis(node.level / (max_level + 1))
        rect = Rectangle((node.x_min, node.y_min), w, h, fill=False,
                         edgecolor=color, linewidth=1)
        ax.add_patch(rect)
        ax.text(node.x_min + w/2, node.y_min + h/2,
                f"L{node.level}\n{node.load:.2f}",
                ha='center', va='center', fontsize=6, color=color)
    ax.set_title("Quadtree Visualization")
    return ax.patches

if __name__ == "__main__":
    grid = FractalGrid(split_threshold=5, merge_threshold=2, decay_factor=0.9, decay_interval=1.0)
    controller = GlobalAdaptiveController(grid, interval=3.0)
    controller.start()
    
    decay_thread = LoadDecayThread(grid)
    decay_thread.start()
    
    pause_event = threading.Event()
    pause_event.clear()
    
    for tid in range(4):
        t = threading.Thread(target=worker_loop, args=(tid, grid, pause_event), daemon=True)
        t.start()
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("Quadtree Visualization")
    plt.subplots_adjust(left=0.1, bottom=0.45)
    
    ax_split = plt.axes([0.1, 0.35, 0.8, 0.03])
    ax_merge = plt.axes([0.1, 0.3, 0.8, 0.03])
    slider_split = Slider(ax_split, 'Split Thresh', 1, 20, valinit=grid.split_threshold, valstep=1)
    slider_merge = Slider(ax_merge, 'Merge Thresh', 1, 20, valinit=grid.merge_threshold, valstep=1)
    
    ax_decay = plt.axes([0.1, 0.25, 0.8, 0.03])
    ax_interval = plt.axes([0.1, 0.2, 0.8, 0.03])
    slider_decay = Slider(ax_decay, 'Decay Factor', 0.5, 1.0, valinit=grid.decay_factor, valstep=0.01)
    slider_interval = Slider(ax_interval, 'Decay Interval', 0.5, 5.0, valinit=grid.decay_interval, valstep=0.1)
    
    def slider_split_cb(val):
        with grid.lock:
            grid.split_threshold = int(val)
        print("Updated split_threshold to", grid.split_threshold)
        grid.periodic_rebalance()
    
    def slider_merge_cb(val):
        with grid.lock:
            grid.merge_threshold = int(val)
        print("Updated merge_threshold to", grid.merge_threshold)
        grid.periodic_rebalance()
    
    def slider_decay_cb(val):
        with grid.lock:
            grid.decay_factor = float(val)
        print("Updated decay_factor to", grid.decay_factor)
    
    def slider_interval_cb(val):
        with grid.lock:
            grid.decay_interval = float(val)
        print("Updated decay_interval to", grid.decay_interval)
    
    slider_split.on_changed(slider_split_cb)
    slider_merge.on_changed(slider_merge_cb)
    slider_decay.on_changed(slider_decay_cb)
    slider_interval.on_changed(slider_interval_cb)
    
    ax_pause = plt.axes([0.1, 0.1, 0.2, 0.05])
    button_pause = Button(ax_pause, "Pause Workers", hovercolor='0.975')
    
    def toggle_pause(_):
        if pause_event.is_set():
            print("Resuming workers.")
            pause_event.clear()
            button_pause.label.set_text("Pause Workers")
        else:
            print("Pausing workers.")
            pause_event.set()
            button_pause.label.set_text("Resume Workers")
    button_pause.on_clicked(toggle_pause)
    
    ani = animation.FuncAnimation(
        fig,
        update_anim,
        fargs=(grid, ax, grid.max_level),
        interval=500,
        cache_frame_data=False
    )
    
    plt.show(block=True)
    
    controller.stop()
    controller.join()
    decay_thread.stop()
    decay_thread.join()
