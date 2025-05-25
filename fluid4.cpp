#include <SDL3/SDL.h>     // SDL3 header
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::iota
#include <algorithm> // For std::min, std::max

// --- Global Constants and Scene Setup ---
int SCREEN_WIDTH = 600;
int SCREEN_HEIGHT = 600;

const double DT_DEFAULT = 1.0 / 120.0;
const int NUM_ITERS_DEFAULT = 20; // Reduced iterations for smoother performance on lower-end PCs
const double OVER_RELAXATION_DEFAULT = 1.9;

double ADD_DENSITY_AMOUNT = 20.0;
double ADD_VELOCITY_AMOUNT = 50.0;

// Smoothing for mouse input to make velocity application smoother
double SMOOTHED_MOUSE_VEL_X = 0.0;
double SMOOTHED_MOUSE_VEL_Y = 0.0;
const double MOUSE_VELOCITY_SMOOTHING_FACTOR = 0.2; // 0.0 (no smoothing) to 1.0 (full smoothing, but less responsive)

// Field types for sampleField function
enum FieldType {
    U_FIELD = 0,
    V_FIELD = 1,
    S_FIELD = 2,
    M_FIELD = 3 // Added for mass/smoke field
};

// Clamp function to keep values within bounds
template <typename T>
T clamp(T val, T min_val, T max_val) {
    return std::max(min_val, std::min(val, max_val));
}

// Scene configuration
struct Scene {
    double gravity;
    double dt;
    int numIters;
    long long frameNr;
    double overRelaxation;
    double obstacleX;
    double obstacleY;
    double obstacleRadius; // Represents half the side length of the square obstacle
    bool paused;
    int sceneNr;
    bool showStreamlines;
    bool showPressure;
    bool showSmoke;
};

Scene scene;

// --- Fluid Class Definition (similar to JS) ---
class Fluid {
public:
    double density;
    int numX, numY; // numX/numY include boundaries (+2 cells)
    int numCells;
    double h; // Cell size

    std::vector<double> u, v; // Velocity components
    std::vector<double> newU, newV;
    std::vector<double> p; // Pressure
    std::vector<double> s; // Solid/fluid mask (0.0 for solid, 1.0 for fluid)
    std::vector<double> m; // Smoke/mass
    std::vector<double> newM;

    Fluid(double density_val, int numX_val, int numY_val, double h_val) :
        density(density_val),
        numX(numX_val + 2), // Add 2 for boundary cells
        numY(numY_val + 2),
        numCells((numX_val + 2) * (numY_val + 2)), // Corrected calculation of numCells
        h(h_val),
        u(numCells), v(numCells),
        newU(numCells), newV(numCells),
        p(numCells), s(numCells),
        m(numCells), newM(numCells)
    {
        init_vectors();
    }

    // Helper to get 1D index from 2D coordinates (i*n + j)
    int IX(int i, int j) const {
        i = clamp(i, 0, numX - 1);
        j = clamp(j, 0, numY - 1);
        return i * numY + j;
    }

    void init_vectors() {
        std::fill(u.begin(), u.end(), 0.0);
        std::fill(v.begin(), v.end(), 0.0);
        std::fill(newU.begin(), newU.end(), 0.0);
        std::fill(newV.begin(), newV.end(), 0.0);
        std::fill(p.begin(), p.end(), 0.0);
        std::fill(s.begin(), s.end(), 1.0); // Default to fluid
        std::fill(m.begin(), m.end(), 0.0); // Default to no smoke
        std::fill(newM.begin(), newM.end(), 0.0);
    }

    void integrate(double dt, double gravity) {
        for (int i = 1; i < numX; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (s[IX(i, j)] != 0 && s[IX(i, j - 1)] != 0) {
                    v[IX(i, j)] += gravity * dt;
                }
            }
        }
    }

    void solveIncompressibility(int numIters, double dt) {
        const double cp = density * h / dt;

        for (int iter = 0; iter < numIters; iter++) {
            for (int i = 1; i < numX - 1; i++) {
                for (int j = 1; j < numY - 1; j++) {
                    if (s[IX(i, j)] == 0) continue; // Skip solid cells

                    double sx0 = s[IX(i - 1, j)];
                    double sx1 = s[IX(i + 1, j)];
                    double sy0 = s[IX(i, j - 1)];
                    double sy1 = s[IX(i, j + 1)];
                    double sum_s = sx0 + sx1 + sy0 + sy1;
                    if (sum_s == 0) continue;

                    double div = u[IX(i + 1, j)] - u[IX(i, j)] +
                                 v[IX(i, j + 1)] - v[IX(i, j)];

                    double p_val = -div / sum_s;
                    p_val *= scene.overRelaxation; // Apply over-relaxation
                    p[IX(i, j)] += cp * p_val;

                    u[IX(i, j)] -= sx0 * p_val;
                    u[IX(i + 1, j)] += sx1 * p_val;
                    v[IX(i, j)] -= sy0 * p_val;
                    v[IX(i, j + 1)] += sy1 * p_val;
                }
            }
        }
    }

    void extrapolate() {
        for (int i = 0; i < numX; i++) {
            u[IX(i, 0)] = u[IX(i, 1)];
            u[IX(i, numY - 1)] = u[IX(i, numY - 2)];
        }
        for (int j = 0; j < numY; j++) {
            v[IX(0, j)] = v[IX(1, j)];
            v[IX(numX - 1, j)] = v[IX(numX - 2, j)];
        }
    }

    double sampleField(double x, double y, FieldType fieldType) const {
        const double h1 = 1.0 / h;
        const double h2 = 0.5 * h;

        x = clamp(x, h, (numX - 2) * h); // Clamp x to active fluid domain
        y = clamp(y, h, (numY - 2) * h); // Clamp y to active fluid domain

        double dx = 0.0;
        double dy = 0.0;
        const std::vector<double>* f;

        switch (fieldType) {
            case U_FIELD: f = &u; dy = h2; break;
            case V_FIELD: f = &v; dx = h2; break;
            case S_FIELD: f = &s; dx = h2; dy = h2; break;
            case M_FIELD: f = &m; dx = h2; dy = h2; break;
            default: return 0.0;
        }

        int x0 = clamp(static_cast<int>((x - dx) * h1), 0, numX - 1);
        double tx = ((x - dx) - x0 * h) * h1;
        int x1 = clamp(x0 + 1, 0, numX - 1);

        int y0 = clamp(static_cast<int>((y - dy) * h1), 0, numY - 1);
        double ty = ((y - dy) - y0 * h) * h1;
        int y1 = clamp(y0 + 1, 0, numY - 1);

        double sx = 1.0 - tx;
        double sy = 1.0 - ty;

        double val = sx * sy * (*f)[IX(x0, y0)] +
                     tx * sy * (*f)[IX(x1, y0)] +
                     tx * ty * (*f)[IX(x1, y1)] +
                     sx * ty * (*f)[IX(x0, y1)];
        return val;
    }

    double avgU(int i, int j) const {
        return (u[IX(i, j - 1)] + u[IX(i, j)] +
                u[IX(i + 1, j - 1)] + u[IX(i + 1, j)]) * 0.25;
    }

    double avgV(int i, int j) const {
        return (v[IX(i - 1, j)] + v[IX(i, j)] +
                v[IX(i - 1, j + 1)] + v[IX(i, j + 1)]) * 0.25;
    }

    void advectVel(double dt) {
        newU = u;
        newV = v;

        const double h_val = h;
        const double h2 = 0.5 * h_val;

        for (int i = 1; i < numX; i++) {
            for (int j = 1; j < numY; j++) {
                // u component
                if (s[IX(i, j)] != 0 && s[IX(i - 1, j)] != 0 && j < numY - 1) {
                    double x = i * h_val;
                    double y = j * h_val + h2;
                    double u_val_orig = u[IX(i, j)];
                    double v_val_avg = avgV(i, j);
                    x = x - dt * u_val_orig;
                    y = y - dt * v_val_avg;
                    newU[IX(i, j)] = sampleField(x, y, U_FIELD);
                }
                // v component
                if (s[IX(i, j)] != 0 && s[IX(i, j - 1)] != 0 && i < numX - 1) {
                    double x = i * h_val + h2;
                    double y = j * h_val;
                    double u_val_avg = avgU(i, j);
                    double v_val_orig = v[IX(i, j)];
                    x = x - dt * u_val_avg;
                    y = y - dt * v_val_orig;
                    newV[IX(i, j)] = sampleField(x, y, V_FIELD);
                }
            }
        }
        u = newU;
        v = newV;
    }

    void advectSmoke(double dt) {
        newM = m;
        const double h_val = h;
        const double h2 = 0.5 * h_val;

        for (int i = 1; i < numX - 1; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (s[IX(i, j)] != 0) {
                    const double x = i * h_val + h2;
                    const double y = j * h_val + h2;
                    const double u_val = sampleField(x, y, U_FIELD);
                    const double v_val = sampleField(x, y, V_FIELD);

                    const double prevX = x - dt * u_val;
                    const double prevY = y - dt * v_val;

                    newM[IX(i, j)] = sampleField(prevX, prevY, M_FIELD);
                }
            }
        }
        m = newM;
    }

    void simulate(double dt, double gravity, int numIters) {
        integrate(dt, gravity);
        std::fill(p.begin(), p.end(), 0.0); // Clear pressure field
        solveIncompressibility(numIters, dt);
        extrapolate();
        advectVel(dt);
        advectSmoke(dt);
    }

    void clear() {
        init_vectors();
    }

    void add_density(double x, double y, double amount) {
        int i = static_cast<int>(x / h);
        int j = static_cast<int>(y / h);

        i = clamp(i, 0, numX - 3);
        j = clamp(j, 0, numY - 3);

        m[IX(i + 1, j + 1)] = clamp(m[IX(i + 1, j + 1)] + amount, 0.0, 1.0);
    }

    void add_velocity(double x, double y, double amountX, double amountY) {
        int i = static_cast<int>(x / h);
        int j = static_cast<int>(y / h);

        i = clamp(i, 0, numX - 3);
        j = clamp(j, 0, numY - 3);

        u[IX(i + 1, j + 1)] += amountX;
        v[IX(i + 1, j + 1)] += amountY;
    }
};

// Global Fluid pointer
Fluid* fluid = nullptr;
double cScale = 1.0; // Global cScale declaration and initialization

// Function to set up a specific scene
void setObstacle(double x, double y, bool reset); // Forward declaration

void setupScene(int sceneNr) {
    // Reset all scene parameters to a default state first
    scene.sceneNr = sceneNr;
    scene.gravity = 0.0;
    scene.dt = DT_DEFAULT;
    scene.numIters = NUM_ITERS_DEFAULT;
    scene.overRelaxation = OVER_RELAXATION_DEFAULT;
    scene.obstacleX = 0.0;
    scene.obstacleY = 0.0;
    scene.obstacleRadius = 0.15; // Default for scenes with obstacle
    scene.paused = false;
    scene.showStreamlines = false;
    scene.showPressure = false;
    scene.showSmoke = true; // Default to showing smoke

    int res = 64; // Optimized resolution for smoother performance on lower-end PCs

    // Determine resolution and other basic parameters based on sceneNr
    if (sceneNr == 0) { // Wind Tunnel
        res = 64; // Set resolution for Wind Tunnel
        scene.dt = 1.0 / 60.0;
        scene.showSmoke = true;
        scene.obstacleRadius = 0.15;
    } else if (sceneNr == 1) { // Paint
        res = 64; // Set resolution for Paint
        scene.overRelaxation = 1.0;
        scene.showSmoke = true;
        scene.obstacleRadius = 0.1;
    }

    // Create new fluid object with calculated resolution
    // Calculate active fluid dimensions based on screen size and desired resolution
    const double simHeight = 1.1; // Reference height for simulation domain
    const double simWidth = (static_cast<double>(SCREEN_WIDTH) / SCREEN_HEIGHT) * simHeight;
    const double h_val = simHeight / res; // Cell size

    const int numX_fluid_active = static_cast<int>(simWidth / h_val);
    const int numY_fluid_active = static_cast<int>(simHeight / h_val);

    if (fluid) delete fluid;
    fluid = new Fluid(1000.0, numX_fluid_active, numY_fluid_active, h_val);

    // Calculate rendering scale (pixels per simulation unit)
    cScale = std::min(SCREEN_WIDTH / ((fluid->numX - 2) * fluid->h), SCREEN_HEIGHT / ((fluid->numY - 2) * fluid->h));

    // Initialize all cells to fluid (s=1.0)
    std::fill(fluid->s.begin(), fluid->s.end(), 1.0);
    fluid->clear(); // Clear fluid state (velocities, smoke, pressure)

    // Apply scene-specific boundaries and initial conditions
    if (sceneNr == 0) { // Wind Tunnel
        // Set top and bottom boundaries to solid
        for (int i = 0; i < fluid->numX; i++) {
            fluid->s[fluid->IX(i, 0)] = 0.0; // Bottom solid
            fluid->s[fluid->IX(i, fluid->numY - 1)] = 0.0; // Top solid
        }
        // Initial velocity at the inlet (left boundary)
        const double inVel = 2.0;
        for (int j = 1; j < fluid->numY - 1; j++) { // Exclude corners
            fluid->u[fluid->IX(1, j)] = inVel; // Set u velocity at the first active column
        }

        // Set obstacle for Wind Tunnel
        scene.obstacleX = 0.4 * (numX_fluid_active * h_val);
        scene.obstacleY = 0.5 * (numY_fluid_active * h_val);
        setObstacle(scene.obstacleX, scene.obstacleY, true); // Use setObstacle to place it
    } else if (sceneNr == 1) { // Paint
        // No fixed boundaries, all fluid (s=1.0)
        // Obstacle is handled by mouse interaction
    }

    scene.frameNr = 0; // Reset frame number for new scene
    scene.paused = false; // Start simulation for new scene
}

void setObstacle(double x, double y, bool reset) {
    if (!fluid) return;

    double vx = 0.0;
    double vy = 0.0;

    if (!reset) {
        vx = (x - scene.obstacleX) / scene.dt;
        vy = (y - scene.obstacleY) / scene.dt;
    }

    scene.obstacleX = x;
    scene.obstacleY = y;
    const double half_side = scene.obstacleRadius;

    // IMPORTANT: Always clear the entire fluid domain (set s=1.0) before applying boundaries or obstacles
    fluid->s.assign(fluid->numCells, 1.0);

    // Re-apply fixed boundaries based on sceneNr
    if (scene.sceneNr == 0) { // Wind Tunnel
        for (int i = 0; i < fluid->numX; i++) {
            fluid->s[fluid->IX(i, 0)] = 0.0;
            fluid->s[fluid->IX(i, fluid->numY - 1)] = 0.0;
        }
    }
    // For Paint scene (sceneNr == 1), fluid->s remains all 1.0 (fluid) as intended.


    for (int i = 1; i < fluid->numX - 1; i++) {
        for (int j = 1; j < fluid->numY - 1; j++) {
            const double cell_center_x_sim = (i - 1 + 0.5) * fluid->h;
            const double cell_center_y_sim = (j - 1 + 0.5) * fluid->h;

            if (cell_center_x_sim >= x - half_side && cell_center_x_sim <= x + half_side &&
                cell_center_y_sim >= y - half_side && cell_center_y_sim <= y + half_side) {
                fluid->s[fluid->IX(i, j)] = 0.0; // Mark as solid
                if (scene.sceneNr == 1) { // Paint scene
                    fluid->m[fluid->IX(i, j)] = 0.5 + 0.5 * std::sin(0.1 * scene.frameNr);
                } else {
                    fluid->m[fluid->IX(i, j)] = 1.0; // Add smoke/mass
                }
                fluid->u[fluid->IX(i, j)] = vx;
                fluid->u[fluid->IX(i + 1, j)] = vx;
                fluid->v[fluid->IX(i, j)] = vy;
                fluid->v[fluid->IX(i, j + 1)] = vy;
            }
        }
    }
}

// --- Main Application Loop ---
int main(int argc, char* args[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("2D Fluid Simulation (CPU Rendering)", SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_RESIZABLE);
    if (!window) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL); // Reverted to no flags
    if (!renderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Initial scene setup (e.g., Wind Tunnel)
    setupScene(0); // Default to Wind Tunnel

    bool quit = false;
    SDL_Event e;
    Uint64 lastTime = SDL_GetPerformanceCounter(); // Initialize lastTime here

    int prevMouseX = -1, prevMouseY = -1;
    bool mouseDownLeft = false;
    bool mouseDownRight = false;

    while (!quit) {
        double current_time = SDL_GetPerformanceCounter();
        double dt_actual = (current_time - lastTime) / (double)SDL_GetPerformanceFrequency(); // Use lastTime
        lastTime = current_time;

        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_EVENT_QUIT) {
                quit = true;
            } else if (e.type == SDL_EVENT_KEY_DOWN) {
                switch (e.key.key) {
                    case SDLK_P: scene.paused = !scene.paused; break;
                    case SDLK_M: if (scene.paused) { fluid->simulate(scene.dt, scene.gravity, scene.numIters); scene.frameNr++; } break;
                    case SDLK_C: fluid->clear(); setupScene(scene.sceneNr); break; // Re-setup current scene
                    case SDLK_0: setupScene(0); break; // Wind Tunnel
                    case SDLK_1: setupScene(1); break; // Paint
                    case SDLK_S: // Toggle Streamlines
                        scene.showStreamlines = !scene.showStreamlines;
                        std::cout << "Streamlines: " << (scene.showStreamlines ? "ON" : "OFF") << std::endl;
                        break;
                    case SDLK_R: // Toggle Pressure
                        if (scene.showPressure) {
                            scene.showPressure = false;
                            scene.showSmoke = true; // Revert to smoke if pressure is turned off
                        } else {
                            scene.showPressure = true;
                            scene.showSmoke = false;
                        }
                        scene.showStreamlines = false;
                        std::cout << "Pressure: " << (scene.showPressure ? "ON" : "OFF") << std::endl;
                        break;
                    case SDLK_K: // Toggle Smoke
                        if (scene.showSmoke) {
                            scene.showSmoke = false;
                            scene.showPressure = true; // Revert to pressure if smoke is turned off
                        } else {
                            scene.showSmoke = true;
                            scene.showPressure = false;
                        }
                        scene.showStreamlines = false;
                        std::cout << "Smoke: " << (scene.showSmoke ? "ON" : "OFF") << std::endl;
                        break;
                }
            } else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                // Convert screen coordinates to simulation coordinates (0 to 1 range relative to active fluid domain)
                double mouseX_sim = (double)e.button.x / SCREEN_WIDTH * ((fluid->numX - 2) * fluid->h);
                double mouseY_sim = (double)(SCREEN_HEIGHT - e.button.y) / SCREEN_HEIGHT * ((fluid->numY - 2) * fluid->h);

                if (e.button.button == SDL_BUTTON_LEFT) {
                    mouseDownLeft = true;
                    setObstacle(mouseX_sim, mouseY_sim, true); // Set obstacle at click location
                } else if (e.button.button == SDL_BUTTON_RIGHT) {
                    mouseDownRight = true;
                    prevMouseX = e.button.x;
                    prevMouseY = e.button.y;
                    SMOOTHED_MOUSE_VEL_X = 0.0; // Reset smoothed velocities
                    SMOOTHED_MOUSE_VEL_Y = 0.0;
                }
            } else if (e.type == SDL_EVENT_MOUSE_BUTTON_UP) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    mouseDownLeft = false;
                } else if (e.button.button == SDL_BUTTON_RIGHT) {
                    mouseDownRight = false;
                    prevMouseX = -1;
                    prevMouseY = -1;
                }
            } else if (e.type == SDL_EVENT_MOUSE_MOTION) {
                double currentMouseX_sim = (double)e.motion.x / SCREEN_WIDTH * ((fluid->numX - 2) * fluid->h);
                double currentMouseY_sim = (double)(SCREEN_HEIGHT - e.motion.y) / SCREEN_HEIGHT * ((fluid->numY - 2) * fluid->h);

                if (mouseDownLeft) {
                    setObstacle(currentMouseX_sim, currentMouseY_sim, false); // Move obstacle
                } else if (mouseDownRight) {
                    fluid->add_density(currentMouseX_sim, currentMouseY_sim, ADD_DENSITY_AMOUNT * scene.dt);

                    double current_delta_x = 0;
                    double current_delta_y = 0;
                    if (prevMouseX != -1 && prevMouseY != -1) {
                        current_delta_x = (double)(e.motion.x - prevMouseX);
                        current_delta_y = -(double)(e.motion.y - prevMouseY); // Invert Y-delta for simulation
                    }

                    SMOOTHED_MOUSE_VEL_X = SMOOTHED_MOUSE_VEL_X * (1.0 - MOUSE_VELOCITY_SMOOTHING_FACTOR) + current_delta_x * MOUSE_VELOCITY_SMOOTHING_FACTOR;
                    SMOOTHED_MOUSE_VEL_Y = SMOOTHED_MOUSE_VEL_Y * (1.0 - MOUSE_VELOCITY_SMOOTHING_FACTOR) + current_delta_y * MOUSE_VELOCITY_SMOOTHING_FACTOR;
                    
                    fluid->add_velocity(currentMouseX_sim, currentMouseY_sim, 
                                        SMOOTHED_MOUSE_VEL_X * ADD_VELOCITY_AMOUNT * scene.dt, 
                                        SMOOTHED_MOUSE_VEL_Y * ADD_VELOCITY_AMOUNT * scene.dt);
                }
                prevMouseX = e.motion.x;
                prevMouseY = e.motion.y;
            } else if (e.type == SDL_EVENT_WINDOW_RESIZED) {
                SDL_GetWindowSize(window, &SCREEN_WIDTH, &SCREEN_HEIGHT);
                // Re-calculate scene parameters based on new aspect ratio
                setupScene(scene.sceneNr); 
            }
        }

        // Update fluid simulation
        if (!scene.paused) {
            fluid->simulate(scene.dt, scene.gravity, scene.numIters);
            scene.frameNr++;
        }

        // --- CPU Rendering ---
        SDL_SetRenderDrawColor(renderer, 51, 76, 76, 255); // Dark teal background
        SDL_RenderClear(renderer);

        // Calculate cell dimensions for rendering
        double cell_render_width = SCREEN_WIDTH / static_cast<double>(fluid->numX - 2);
        double cell_render_height = SCREEN_HEIGHT / static_cast<double>(fluid->numY - 2);

        for (int i = 1; i < fluid->numX - 1; ++i) {
            for (int j = 1; j < fluid->numY - 1; ++j) {
                int screen_x = static_cast<int>((i - 1) * cell_render_width);
                int screen_y = static_cast<int>((fluid->numY - 2 - (j - 1) - 1) * cell_render_height); // Invert Y for rendering

                SDL_FRect cell_rect = {
                    static_cast<float>(screen_x),
                    static_cast<float>(screen_y),
                    static_cast<float>(cell_render_width),
                    static_cast<float>(cell_render_height)
                };

                // Render obstacle (always render if it's solid)
                if (fluid->s[fluid->IX(i, j)] < 0.5) { // Assuming s=0 for solid
                    SDL_SetRenderDrawColor(renderer, 25, 25, 25, 255); // Dark gray for obstacle
                    SDL_RenderFillRect(renderer, &cell_rect);
                } else { // Render fluid (smoke or pressure)
                    if (scene.showPressure) {
                        double pressure_val = fluid->p[fluid->IX(i, j)];
                        double max_p = 1e-6;
                        for (int x_idx = 1; x_idx < fluid->numX - 1; ++x_idx) {
                            for (int y_idx = 1; y_idx < fluid->numY - 1; ++y_idx) {
                                max_p = std::max(max_p, std::abs(fluid->p[fluid->IX(x_idx, y_idx)]));
                            }
                        }
                        float normalized_p = static_cast<float>(pressure_val / max_p);
                        Uint8 r = 0, g = 0, b = 0;
                        if (normalized_p > 0) { // Positive pressure -> Red
                            r = static_cast<Uint8>(clamp(normalized_p, 0.0f, 1.0f) * 255);
                        } else { // Negative pressure -> Blue
                            b = static_cast<Uint8>(clamp(std::abs(normalized_p), 0.0f, 1.0f) * 255);
                        }
                        SDL_SetRenderDrawColor(renderer, r, g, b, 255);
                        SDL_RenderFillRect(renderer, &cell_rect);
                    } else if (scene.showSmoke) {
                        double smoke_density = fluid->m[fluid->IX(i, j)];
                        Uint8 color_val;
                        if (scene.sceneNr == 1) { // Paint scene scientific color map
                            float val = clamp(static_cast<float>(smoke_density), 0.0f, 0.9999f);
                            float m_sci = 0.25f;
                            int num_sci = static_cast<int>(floor(val / m_sci));
                            float s_sci = (val - static_cast<float>(num_sci) * m_sci) / m_sci;
                            
                            Uint8 r = 0, g = 0, b = 0;
                            if (num_sci == 0) { r = 0; g = static_cast<Uint8>(s_sci * 255); b = 255; } // Blue to Cyan
                            else if (num_sci == 1) { r = 0; g = 255; b = static_cast<Uint8>((1.0f - s_sci) * 255); } // Cyan to Green
                            else if (num_sci == 2) { r = static_cast<Uint8>(s_sci * 255); g = 255; b = 0; } // Green to Yellow
                            else if (num_sci == 3) { r = 255; g = static_cast<Uint8>((1.0f - s_sci) * 255); b = 0; } // Yellow to Red
                            else { r = 255; g = 255; b = 255; } // White
                            SDL_SetRenderDrawColor(renderer, r, g, b, 255);
                        } else { // Grayscale for Wind Tunnel
                            color_val = static_cast<Uint8>(smoke_density * 255);
                            SDL_SetRenderDrawColor(renderer, color_val, color_val, color_val, 255);
                        }
                        SDL_RenderFillRect(renderer, &cell_rect);
                    }
                }
                
                // Draw grid lines
                SDL_SetRenderDrawColor(renderer, 70, 70, 70, 255); // Dark gray for grid lines
                SDL_RenderRect(renderer, &cell_rect); // Changed from SDL_RenderDrawRectF to SDL_RenderRect
            }
        }

        // Render Streamlines (lines)
        if (scene.showStreamlines) {
            SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255); // Green color for streamlines

            int num_streamlines_x = 30; // Number of starting points along X
            int num_streamlines_y = 30; // Number of starting points along Y
            double start_x_spacing = (fluid->numX - 2) * fluid->h / (num_streamlines_x + 1);
            double start_y_spacing = (fluid->numY - 2) * fluid->h / (num_streamlines_y + 1);

            double line_step = fluid->h * 0.5; // Integration step for streamlines
            int max_segments = 100; // Max segments per streamline

            for (int i = 0; i < num_streamlines_x; ++i) {
                for (int j = 0; j < num_streamlines_y; ++j) {
                    double currentX_sim = (i + 0.5) * start_x_spacing;
                    double currentY_sim = (j + 0.5) * start_y_spacing;

                    // Start a new streamline
                    for (int k = 0; k < max_segments; ++k) {
                        // Sample velocity at current point
                        double u_val = fluid->sampleField(currentX_sim, currentY_sim, U_FIELD);
                        double v_val = fluid->sampleField(currentX_sim, currentY_sim, V_FIELD);

                        // If velocity is very small or outside fluid, stop tracing this streamline
                        if (fluid->sampleField(currentX_sim, currentY_sim, S_FIELD) < 0.5 || (std::abs(u_val) < 1e-6 && std::abs(v_val) < 1e-6)) {
                            break; // Stop tracing
                        }

                        float startX_screen = static_cast<float>(currentX_sim * cScale);
                        float startY_screen = static_cast<float>(SCREEN_HEIGHT - (currentY_sim * cScale));

                        // Move along the streamline
                        currentX_sim += u_val * line_step;
                        currentY_sim += v_val * line_step;

                        float endX_screen = static_cast<float>(currentX_sim * cScale);
                        float endY_screen = static_cast<float>(SCREEN_HEIGHT - (currentY_sim * cScale));
                        
                        SDL_RenderLine(renderer, startX_screen, startY_screen, endX_screen, endY_screen);
                    }
                }
            }
        }

        SDL_RenderPresent(renderer);
    }

    // Cleanup
    delete fluid; // Ensure fluid object is deleted
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
