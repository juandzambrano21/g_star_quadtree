A Python implementation of a dynamically evolving quadtree, featuring:

- **Lazy Subdivision & Merging:**  
  The quadtree covers the unit square and subdivides nodes on demand when the load (i.e. update count) exceeds a specified split threshold. Conversely, nodes merge when their children’s combined load falls below a merge threshold. When subdividing, the parent node's load is evenly distributed among the children to prevent rapid re‑splitting in hot regions.

- **Load Decay:**  
  The system continuously decays the load over time (using a configurable decay factor and interval). This prevents obsolete load from causing unnecessary subdivisions.

- **Thread-Safe Operations:**  
  All operations on the quadtree are protected by a global reentrant lock (RLock) so that updates, splits, merges, and visualization happen in a consistent, thread‑safe manner.

- **Adaptive Controller:**  
  A background thread periodically prints current hyperparameters and triggers periodic merging of nodes.

- **Interactive Hyperparameter Tuning:**  
  The visualization window (built using Matplotlib) includes interactive sliders that let you adjust the following in real time:
  - **Split Threshold:** The load at which a node subdivides.
  - **Merge Threshold:** The maximum total load of children that allows merging.
  - **Decay Factor:** The multiplier applied to load on each decay pass.
  - **Decay Interval:** The time interval (in seconds) between load decay passes.

  There is also a pause/resume button to temporarily halt worker updates, so you can better observe the merging and decay behavior.

- **Live Visualization:**  
  Matplotlib's `FuncAnimation` updates the display in real time, drawing each leaf node as a rectangle annotated with its level and current (decayed) load.
