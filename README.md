# 💧 2D Fluid Simulation (CPU + SDL3)

This project is a real-time interactive **2D fluid dynamics simulation** written in **C++** using **SDL3** for rendering. It simulates fluid flow using grid-based methods, pressure projection, and advection — similar in principle to "stable fluids" and visual CFD tools.

## 🖥 Features

- 🌀 Smoke density advection
- 🌊 Velocity field integration and visualization
- 🧱 Interactive square obstacle (movable with mouse)
- 🧭 Toggleable streamlines and pressure visualization
- ⌨️ Keyboard and mouse input for full control
- 🔧 Optimized for CPU performance with real-time interactivity

## 🎮 Controls

### Keyboard
| Key       | Action                          |
|----------|----------------------------------|
| `P`      | Pause/unpause the simulation     |
| `M`      | Step forward one frame (when paused) |
| `C`      | Clear and reset the current scene |
| `0`      | Load **Wind Tunnel** scene       |
| `1`      | Load **Paint** scene             |
| `S`      | Toggle **streamlines**           |
| `R`      | Toggle **pressure view**         |

### Mouse
- **Left Click + Drag**: Place or move an **obstacle**
- **Right Click + Drag**: Inject **smoke** and **velocity** into the fluid

## 🧪 Scenes

- **Scene 0**: *Wind Tunnel*  
  Simulates airflow past an obstacle. Preloaded with boundary conditions and inlet velocity.

- **Scene 1**: *Paint*  
  A sandbox mode for drawing with smoke and interacting freely.

## 🛠 Requirements

- C++ compiler with C++17 support
- SDL3 (Simple DirectMedia Layer 3)

### SDL3 Installation
You can download and install SDL3 from [libsdl.org](https://www.libsdl.org/index.php) or use your package manager.
