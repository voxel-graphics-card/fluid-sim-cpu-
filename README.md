# 💧 2D Fluid Simulation (CPU + SDL3)

This project is a real-time interactive **2D fluid dynamics simulation** written in **C++** using **SDL3** for rendering. It simulates fluid flow using grid-based methods, pressure projection, and advection — similar in principle to "stable fluids" and visual CFD tools.

# based on :

https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf

https://www.youtube.com/watch?v=alhpH6ECFvQ


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

### Some kinda cool pictures:

   #fluid4.cpp

![Screenshot 2025-05-25 165954](https://github.com/user-attachments/assets/e61c0c17-e748-4dad-899d-114ff99c9078)


![Screenshot 2025-05-25 170038](https://github.com/user-attachments/assets/9e7b602f-63a9-4b4c-b1e8-4a894d2df754)


![Screenshot 2025-05-25 170111](https://github.com/user-attachments/assets/eb03a2d7-0845-4e4f-8607-5d0bc334550c)

  #fluid.cpp

![Screenshot 2025-05-23 165315](https://github.com/user-attachments/assets/74c3b8e3-662d-4227-ac49-1ccb8b188bb2)

![fluidsim](https://github.com/user-attachments/assets/bfc80958-7419-4953-8f45-83aaad6f3012)

![64andN+4](https://github.com/user-attachments/assets/a3ed1c41-c156-4d71-b60f-f1599c606c2c)

