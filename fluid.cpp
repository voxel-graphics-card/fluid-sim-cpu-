#include <GL/glew.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

#include <cmath>     // For std::fmod, std::sqrt, std::abs
#include <vector>    // For std::vector
#include <iostream>  // For std::cerr, std::cout
#include <algorithm> // For std::min, std::max, std::fill, std::sort, std::lower_bound
#include <cstdlib>   // For srand, rand
#include <ctime>     // For time

// --- Global Constants ---
const int N = 64; // The size of the calculation grid (from Processing's 'n')
const int GRID_SIZE = N + 2; // Extra grid space for boundary (from Processing's 'gridSize')

// Macro to convert 2D to 1D index, considering boundaries for a (N+2)x(N+2) grid
#define IX(i, j) ((i) + (GRID_SIZE) * (j))

// --- Custom Clamp Function ---
template<typename T>
T custom_clamp_fluid(T val, T low, T high) {
    return std::max(low, std::min(val, high));
}

// --- ColorRGB Struct ---
struct ColorRGB {
    float r;
    float g;
    float b;
};

// --- HSV to RGB Conversion Functions ---
void hsv2rgb(float h, float s, float v_val, float& r, float& g, float& b) {
    h = std::fmod(h, 360.0f); // Ensure h is within [0, 360)
    if (h < 0) h += 360.0f;

    float c = v_val * s;
    float x = c * (1 - std::abs(std::fmod(h / 60.0f, 2) - 1));
    float m = v_val - c;

    if (0 <= h && h < 60) {
        r = c, g = x, b = 0;
    } else if (60 <= h && h < 120) {
        r = x, g = c, b = 0;
    } else if (120 <= h && h < 180) {
        r = 0, g = c, b = x;
    } else if (180 <= h && h < 240) {
        r = 0, g = x, b = c;
    } else if (240 <= h && h < 300) {
        r = x, g = 0, b = c;
    } else { // 300 <= h && h < 360
        r = c, g = 0, b = x;
    }

    r += m;
    g += m;
    b += m;

    // Clamp to 0-1 range
    r = custom_clamp_fluid(r, 0.0f, 1.0f);
    g = custom_clamp_fluid(g, 0.0f, 1.0f);
    b = custom_clamp_fluid(b, 0.0f, 1.0f);
}

ColorRGB hsv2rgb(float h, float s, float v_val) {
    ColorRGB color;
    hsv2rgb(h, s, v_val, color.r, color.g, color.b);
    return color;
}

// --- Gradient Class (not directly used for fluid rendering, but kept) ---
class Gradient {
public:
    enum DefaultGradientType {
        DEFAULT_BLACK_TO_WHITE = 0,
        DEFAULT_RANDOM = 1,
        DEFAULT_SPECTRUM = 2,
        DEFAULT_INFRARED = 3,
        DEFAULT_BLACKBODY = 4,
        DEFAULT_NEON = 5,
        DEFAULT_WINTER = 6,
        DEFAULT_SUMMER = 7
    };

private:
    struct GradientNode {
        float location;
        ColorRGB col;
        GradientNode(float l, ColorRGB c) : location(l), col(c) {}
    };

    std::vector<GradientNode> nodes;

public:
    Gradient() {
        makeDefaultGradient(DEFAULT_BLACK_TO_WHITE);
    }

    void addNode(float location, ColorRGB col) {
        nodes.push_back(GradientNode(location, col));
        std::sort(nodes.begin(), nodes.end(), [](const GradientNode& a, const GradientNode& b) {
            return a.location < b.location;
        });
    }

    void clear() {
        nodes.clear();
    }

    void makeDefaultGradient(DefaultGradientType defaultGradient) {
        clear();

        switch (defaultGradient) {
            case DEFAULT_BLACK_TO_WHITE:
                addNode(0, {0, 0, 0});
                addNode(1, {1, 1, 1});
                break;
            case DEFAULT_RANDOM:
                makeRandomGradient(4);
                break;
            case DEFAULT_SPECTRUM:
                addNode(0, {1, 0, 0});
                addNode(0.25f, {1, 1, 0});
                addNode(0.5f, {0, 1, 0});
                addNode(0.75f, {0, 1, 1});
                addNode(1, {0, 0, 1});
                break;
            case DEFAULT_INFRARED:
                addNode(0, {0, 0, 0});
                addNode(1.0f / 6, {0, 0, 0.5f});
                addNode(2.0f / 6, {0.5f, 0, 0.5f});
                addNode(3.0f / 6, {1, 0, 0});
                addNode(4.0f / 6, {1, 0.5f, 0});
                addNode(5.0f / 6, {1, 1, 0});
                addNode(1, {1, 1, 1});
                break;
            case DEFAULT_BLACKBODY:
                addNode(0, {0, 0, 0});
                addNode(1.0f / 5, {0, 0.25f, 1});
                addNode(2.0f / 5, {0, 0.75f, 1});
                addNode(3.0f / 5, {1, 0.25f, 0});
                addNode(4.0f / 5, {1, 0.75f, 0});
                addNode(1, {1, 1, 1});
                break;
            case DEFAULT_NEON:
                addNode(0, {0, 0, 0});
                addNode(0.25f, {0.2f, 0.2f, 1});
                addNode(0.5f, {0, 0.6f, 1});
                addNode(0.75f, {0.9f, 0, 0.5f});
                addNode(1, {1, 0, 1});
                break;
            case DEFAULT_WINTER:
                addNode(0, {0.3f, 0.5f, 1});
                addNode(0.5f, {0.9f, 0.9f, 0.9f});
                addNode(1, {0.6f, 0.6f, 0.6f});
                break;
            case DEFAULT_SUMMER:
                addNode(0, {0.2f, 0.3f, 1});
                addNode(0.25f, {1, 0, 0.5f});
                addNode(0.5f, {1, 0.5f, 0.2f});
                addNode(0.75f, {0.8f, 0.3f, 0});
                addNode(1, {1, 0.8f, 0});
                break;
        }
    }

    void makeRandomGradient(int numColours) {
        clear();
        for (int n_node = 0; n_node < numColours; n_node++) {
            float location;
            if (n_node == 0) {
                location = 0.0f;
            }
            else if (n_node == numColours - 1) {
                location = 1.0f;
            }
            else {
                float locationMin = static_cast<float>(n_node) / numColours;
                float locationMax = static_cast<float>(n_node + 1) / numColours;
                location = locationMin + static_cast<float>(rand()) / RAND_MAX * (locationMax - locationMin);
            }

            float r = static_cast<float>(static_cast<int>(static_cast<float>(rand()) / RAND_MAX * 2.5f)) * 0.5f;
            float g = static_cast<float>(static_cast<int>(static_cast<float>(rand()) / RAND_MAX * 2.5f)) * 0.5f;
            float b = static_cast<float>(static_cast<int>(static_cast<float>(rand()) / RAND_MAX * 2.5f)) * 0.5f;

            addNode(location, {r, g, b});
        }
    }

    ColorRGB getColour(float location) const {
        location = custom_clamp_fluid(location, 0.0f, 1.0f);

        if (nodes.empty()) return {0, 0, 0};
        if (nodes.size() == 1) return nodes[0].col;

        auto it = std::lower_bound(nodes.begin(), nodes.end(), location, [](const GradientNode& node, float loc) {
            return node.location < loc;
        });

        if (it == nodes.begin()) {
            return nodes.front().col;
        }
        if (it == nodes.end()) {
            return nodes.back().col;
        }

        const GradientNode& node1 = *(it - 1);
        const GradientNode& node2 = *it;

        float bandScale = node2.location - node1.location;
        if (bandScale == 0.0f) return node1.col;

        float bandDelta = (location - node1.location) / bandScale;

        float r = bandDelta * (node2.col.r - node1.col.r) + node1.col.r;
        float g = bandDelta * (node2.col.g - node1.col.g) + node1.col.g;
        float b = bandDelta * (node2.col.b - node1.col.b) + node1.col.b;
        return {r, g, b};
    }

    std::vector<ColorRGB> makeArrayOfColours(int numColours) const {
        std::vector<ColorRGB> cols(numColours);
        if (numColours == 0) return cols;

        for (int i = 0; i < numColours; i++) {
            float location = static_cast<float>(i) / (numColours - 1);
            cols[i] = getColour(location);
        }
        return cols;
    }
};

// --- Fluid Simulation Core Functions (free functions) ---
void set_bnd(int b, std::vector<float>& x) {
    for (int i = 1; i <= N; i++) {
        x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
    }
    for (int j = 1; j <= N; j++) {
        x[IX(0, j)] = b == 1 ? -x[IX(1, j)] : x[IX(1, j)];
        x[IX(N + 1, j)] = b == 1 ? -x[IX(N, j)] : x[IX(N, j)];
    }
    x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N + 1)] = 0.5f * (x[IX(1, N + 1)] + x[IX(0, N)]);
    x[IX(N + 1, 0)] = 0.5f * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    x[IX(N + 1, N + 1)] = 0.5f * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

void lin_solve(int b, std::vector<float>& x, const std::vector<float>& x0, float a, float c) {
    for (int k = 0; k < 20; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= N; i++) {
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                                    x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
            }
        }
        set_bnd(b, x);
    }
}

void diffuse(int b, std::vector<float>& x, std::vector<float>& x0, float diff, float dt) {
    float a = dt * diff * N * N;
    lin_solve(b, x, x0, a, 1 + 4 * a);
}

void advect(int b, std::vector<float>& d, std::vector<float>& d0,
            const std::vector<float>& u, const std::vector<float>& v, float dt) {
    float i0, i1, j0, j1;
    float s0, s1, t0, t1;
    float dt0 = dt * N;

    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            float x = i - dt0 * u[IX(i, j)];
            float y = j - dt0 * v[IX(i, j)];

            x = custom_clamp_fluid(x, 0.5f, (float)N + 0.5f);
            int i_int = static_cast<int>(x);
            i0 = static_cast<float>(i_int);
            i1 = i0 + 1;

            y = custom_clamp_fluid(y, 0.5f, (float)N + 0.5f);
            int j_int = static_cast<int>(y);
            j0 = static_cast<float>(j_int);
            j1 = j0 + 1;

            s1 = x - i0;
            s0 = 1.0f - s1;
            t1 = y - j0;
            t0 = 1.0f - t1;

            int i_idx0 = static_cast<int>(i0);
            int i_idx1 = static_cast<int>(i1);
            int j_idx0 = static_cast<int>(j0);
            int j_idx1 = static_cast<int>(j1);

            d[IX(i, j)] = s0 * (t0 * d0[IX(i_idx0, j_idx0)] + t1 * d0[IX(i_idx0, j_idx1)]) +
                          s1 * (t0 * d0[IX(i_idx1, j_idx0)] + t1 * d0[IX(i_idx1, j_idx1)]);
        }
    }
    set_bnd(b, d);
}

void project(std::vector<float>& u, std::vector<float>& v,
            std::vector<float>& p, std::vector<float>& div) {
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                                    v[IX(i, j + 1)] - v[IX(i, j - 1)]) / N;
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(0, div);
    set_bnd(0, p);

    lin_solve(0, p, div, 1, 4);

    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {
            u[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * N;
            v[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * N;
        }
    }
    set_bnd(1, u);
    set_bnd(2, v);
}

// --- FluidSim Class ---
class FluidSim {
public:
    std::vector<float> densR;
    std::vector<float> densG;
    std::vector<float> densB;

    std::vector<float> densRPrev;
    std::vector<float> densGPrev;
    std::vector<float> densBPrev;

    std::vector<float> u_vel;
    std::vector<float> v_vel;
    std::vector<float> uPrev_vel;
    std::vector<float> vPrev_vel;

    std::vector<float> p_poisson;
    std::vector<float> div_vel;

    float viscosity = 0.0001f;
    float dt = 0.2f;
    float diffusion = 0.0001f;
    float fadeSpeed = 0.003f;

    FluidSim() {
        densR.resize(GRID_SIZE * GRID_SIZE, 0.0f);
        densG.resize(GRID_SIZE * GRID_SIZE, 0.0f);
        densB.resize(GRID_SIZE * GRID_SIZE, 0.0f);
        densRPrev.resize(GRID_SIZE * GRID_SIZE, 0.0f);
        densGPrev.resize(GRID_SIZE * GRID_SIZE, 0.0f);
        densBPrev.resize(GRID_SIZE * GRID_SIZE, 0.0f);

        u_vel.resize(GRID_SIZE * GRID_SIZE, 0.0f);
        v_vel.resize(GRID_SIZE * GRID_SIZE, 0.0f);
        uPrev_vel.resize(GRID_SIZE * GRID_SIZE, 0.0f);
        vPrev_vel.resize(GRID_SIZE * GRID_SIZE, 0.0f);
        p_poisson.resize(GRID_SIZE * GRID_SIZE, 0.0f);
        div_vel.resize(GRID_SIZE * GRID_SIZE, 0.0f);

        reset();
    }

    void initField(std::vector<float>& f) {
        std::fill(f.begin(), f.end(), 0.0f);
    }

    void initVelocity() {
        initField(u_vel);
        initField(v_vel);
        initField(uPrev_vel);
        initField(vPrev_vel);
    }

    void initDensity() {
        initField(densR);
        initField(densG);
        initField(densB);
        initField(densRPrev);
        initField(densGPrev);
        initField(densBPrev);
    }

    void reset() {
        initVelocity();
        initDensity();
    }

    void addDensitySource(std::vector<float>& dens_channel, const std::vector<float>& dens_prev_channel, float dt_val) {
        for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            dens_channel[i] += dt_val * dens_prev_channel[i];
        }
    }

    void addVelocitySource(std::vector<float>& vel_channel, const std::vector<float>& vel_prev_channel, float dt_val) {
        for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            vel_channel[i] += dt_val * vel_prev_channel[i];
        }
    }

    void setForceArea(int x_grid, int y_grid, float r_src, float g_src, float b_src, float vx_src, float vy_src) {
        // Clamp grid coordinates to valid range (1 to N)
        x_grid = custom_clamp_fluid(x_grid, 1, N);
        y_grid = custom_clamp_fluid(y_grid, 1, N);

        densRPrev[IX(x_grid, y_grid)] += r_src;
        densGPrev[IX(x_grid, y_grid)] += g_src;
        densBPrev[IX(x_grid, y_grid)] += b_src;

        uPrev_vel[IX(x_grid, y_grid)] += vx_src;
        vPrev_vel[IX(x_grid, y_grid)] += vy_src;
    }

    void fadeDensity() {
        float decayFactor = 1.0f - fadeSpeed;
        for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            densR[i] *= decayFactor;
            densG[i] *= decayFactor;
            densB[i] *= decayFactor;

            if (densR[i] < 0.0f) densR[i] = 0.0f;
            if (densG[i] < 0.0f) densG[i] = 0.0f;
            if (densB[i] < 0.0f) densB[i] = 0.0f;
        }
    }

    void step() {
        addVelocitySource(u_vel, uPrev_vel, dt);
        addVelocitySource(v_vel, vPrev_vel, dt);

        std::swap(u_vel, uPrev_vel);
        std::swap(v_vel, vPrev_vel);

        diffuse(1, u_vel, uPrev_vel, viscosity, dt);
        diffuse(2, v_vel, vPrev_vel, viscosity, dt);

        project(u_vel, v_vel, p_poisson, div_vel);

        std::swap(u_vel, uPrev_vel);
        std::swap(v_vel, vPrev_vel);

        advect(1, u_vel, uPrev_vel, uPrev_vel, vPrev_vel, dt);
        advect(2, v_vel, vPrev_vel, uPrev_vel, vPrev_vel, dt);

        project(u_vel, v_vel, p_poisson, div_vel);

        addDensitySource(densR, densRPrev, dt);
        addDensitySource(densG, densGPrev, dt);
        addDensitySource(densB, densBPrev, dt);

        std::swap(densR, densRPrev);
        std::swap(densG, densGPrev);
        std::swap(densB, densBPrev);

        diffuse(0, densR, densRPrev, diffusion, dt);
        diffuse(0, densG, densGPrev, diffusion, dt);
        diffuse(0, densB, densBPrev, diffusion, dt);

        advect(0, densR, densRPrev, u_vel, v_vel, dt);
        advect(0, densG, densGPrev, u_vel, v_vel, dt);
        advect(0, densB, densBPrev, u_vel, v_vel, dt);

        fadeDensity();

        initField(uPrev_vel);
        initField(vPrev_vel);
        initField(densRPrev);
        initField(densGPrev);
        initField(densBPrev);
    }
};

// --- Renderer Class ---
class Renderer {
private:
    GLuint shaderProgram;
    GLuint textureID;
    GLuint vaoID;
    GLuint vboID;
    GLuint eboID;

    bool checkGLError(const std::string& msg) {
        GLenum err;
        bool hasError = false;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL ERROR (" << msg << "): " << err << std::endl;
            hasError = true;
        }
        return hasError;
    }

    GLuint compileShader(GLenum type, const char* source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, NULL);
        glCompileShader(shader);

        int success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cerr << "Shader compilation error (Type: " << type << "): " << infoLog << std::endl;
        }
        checkGLError("compileShader");
        return shader;
    }

    GLuint createShaderProgram(const char* vertexShaderSource, const char* fragmentShaderSource) {
        GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
        GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

        GLuint program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);

        int success;
        char infoLog[512];
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(program, 512, NULL, infoLog);
            std::cerr << "Shader program linking error: " << infoLog << std::endl;
        }

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        checkGLError("createShaderProgram");
        return program;
    }

public:
    Renderer() : shaderProgram(0), textureID(0), vaoID(0), vboID(0), eboID(0) {}

    ~Renderer() {
        shutdown();
    }

    bool init(int width, int height) {
        glewExperimental = GL_TRUE;
        GLenum glewError = glewInit();
        if (glewError != GLEW_OK) {
            std::cerr << "Error initializing GLEW! " << glewGetErrorString(glewError) << std::endl;
            return false;
        }
        checkGLError("glewInit");

        glViewport(0, 0, width, height);
        checkGLError("glViewport");
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        checkGLError("glClearColor");

        const char* vertexShaderSource = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            void main() {
                gl_Position = vec4(aPos, 1.0);
                TexCoord = aTexCoord;
            }
        )";

        const char* fragmentShaderSource = R"(
            #version 330 core
            out vec4 FragColor;
            in vec2 TexCoord;
            uniform sampler2D fluidTexture;
            void main() {
                FragColor = texture(fluidTexture, TexCoord);
            }
        )";

        shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
        if (!shaderProgram) {
            return false;
        }

        float vertices[] = {
             1.0f,  1.0f, 0.0f,  1.0f, 1.0f,
             1.0f, -1.0f, 0.0f,  1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,  0.0f, 1.0f
        };
        unsigned int indices[] = {
            0, 1, 3,
            1, 2, 3
        };

        glGenVertexArrays(1, &vaoID);
        checkGLError("glGenVertexArrays");
        glGenBuffers(1, &vboID);
        checkGLError("glGenBuffers VBO");
        glGenBuffers(1, &eboID);
        checkGLError("glGenBuffers EBO");

        glBindVertexArray(vaoID);
        checkGLError("glBindVertexArray");

        glBindBuffer(GL_ARRAY_BUFFER, vboID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        checkGLError("glBufferData VBO");

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboID);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        checkGLError("glBufferData EBO");

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        checkGLError("glVertexAttribPointer 0");
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        checkGLError("glVertexAttribPointer 1");

        glBindVertexArray(0);
        checkGLError("glBindVertexArray unbind");

        glGenTextures(1, &textureID);
        checkGLError("glGenTextures");
        glBindTexture(GL_TEXTURE_2D, textureID);
        checkGLError("glBindTexture");
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        checkGLError("glTexParameteri");
        // Initialize texture with NULL data, will be updated with glTexSubImage2D
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, N, N, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        checkGLError("glTexImage2D");

        glBindTexture(GL_TEXTURE_2D, 0);
        checkGLError("glBindTexture unbind");

        return true;
    }

    void render(const FluidSim& fluid) {
        glClear(GL_COLOR_BUFFER_BIT);
        checkGLError("glClear");

        std::vector<GLubyte> textureData(N * N * 3);
        const float display_max_density = 0.1f;

        static int debug_frame_count = 0;
        debug_frame_count++;
        if (debug_frame_count % 60 == 0) {
            int sample_idx = IX(N / 2, N / 2);
            float sample_r = fluid.densR[sample_idx];
            float sample_g = fluid.densG[sample_idx];
            float sample_b = fluid.densB[sample_idx];
            //std::cout << "DEBUG: Sample Density (R,G,B) at (" << N/2 << "," << N/2 << "): ("
            //          << sample_r << ", " << sample_g << ", " << sample_b << ")" << std::endl;
        }

        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                float r_val = fluid.densR[IX(i + 1, j + 1)];
                float g_val = fluid.densG[IX(i + 1, j + 1)];
                float b_val = fluid.densB[IX(i + 1, j + 1)];

                int idx = (j * N + i) * 3;
                textureData[idx + 0] = static_cast<GLubyte>(custom_clamp_fluid(r_val / display_max_density * 255.0f, 0.0f, 255.0f));
                textureData[idx + 1] = static_cast<GLubyte>(custom_clamp_fluid(g_val / display_max_density * 255.0f, 0.0f, 255.0f));
                textureData[idx + 2] = static_cast<GLubyte>(custom_clamp_fluid(b_val / display_max_density * 255.0f, 0.0f, 255.0f));
            }
        }

        glActiveTexture(GL_TEXTURE0);
        checkGLError("glActiveTexture");
        glBindTexture(GL_TEXTURE_2D, textureID);
        checkGLError("glBindTexture render");
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, N, N, GL_RGB, GL_UNSIGNED_BYTE, textureData.data());
        checkGLError("glTexSubImage2D");

        glUseProgram(shaderProgram);
        checkGLError("glUseProgram");
        glUniform1i(glGetUniformLocation(shaderProgram, "fluidTexture"), 0);
        checkGLError("glUniform1i");

        glBindVertexArray(vaoID);
        checkGLError("glBindVertexArray render");
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        checkGLError("glDrawElements");

        glBindVertexArray(0);
        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        checkGLError("Cleanup render");
    }

    void shutdown() {
        // Explicitly unbind objects before deleting
        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        if (shaderProgram) {
            glDeleteProgram(shaderProgram);
            checkGLError("glDeleteProgram");
            shaderProgram = 0;
        }
        if (textureID) {
            glDeleteTextures(1, &textureID);
            checkGLError("glDeleteTextures");
            textureID = 0;
        }
        if (vboID) {
            glDeleteBuffers(1, &vboID);
            checkGLError("glDeleteBuffers VBO");
            vboID = 0;
        }
        if (eboID) {
            glDeleteBuffers(1, &eboID);
            checkGLError("glDeleteBuffers EBO");
            eboID = 0;
        }
        if (vaoID) {
            glDeleteVertexArrays(1, &vaoID);
            checkGLError("glDeleteVertexArrays");
            vaoID = 0;
        }
    }
};

// --- Main Program ---
int main(int argc, char* args[]) {
    srand(static_cast<unsigned int>(time(0)));

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    const int SCREEN_WIDTH = 800;
    const int SCREEN_HEIGHT = 700;

    // Correctly calculate pixel size for X and Y to map to N grid cells
    // This assumes the N x N fluid grid stretches to fill the window
    const float GRID_PIXEL_SIZE_X = static_cast<float>(SCREEN_WIDTH) / N;
    const float GRID_PIXEL_SIZE_Y = static_cast<float>(SCREEN_HEIGHT) / N;

    SDL_Window* window = SDL_CreateWindow("RGB Fluid Simulation", SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_OPENGL);
    if (window == NULL) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    if (glContext == NULL) {
        std::cerr << "OpenGL context could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    Renderer renderer;
    if (!renderer.init(SCREEN_WIDTH, SCREEN_HEIGHT)) {
        std::cerr << "Renderer initialization failed!" << std::endl;
        SDL_GL_DestroyContext(glContext);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    FluidSim fluid;

    bool quit = false;
    SDL_Event e;

    int prevMouseX = 0;
    int prevMouseY = 0;
    bool mousePressed = false;
    float currentHue = 0.0f;

    Uint64 lastFrameTime = SDL_GetPerformanceCounter();

    while (!quit) {
        Uint64 currentFrameTime = SDL_GetPerformanceCounter();
        float deltaTime = (float)(currentFrameTime - lastFrameTime) / SDL_GetPerformanceFrequency();
        lastFrameTime = currentFrameTime;

        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_EVENT_QUIT) {
                quit = true;
            } else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    mousePressed = true;
                    prevMouseX = e.button.x;
                    prevMouseY = e.button.y;
                }
            } else if (e.type == SDL_EVENT_MOUSE_BUTTON_UP) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    mousePressed = false;
                }
            } else if (e.type == SDL_EVENT_MOUSE_MOTION) {
                if (mousePressed) {
                    int mouseX = e.motion.x;
                    int mouseY = e.motion.y;

                    // vx: positive for rightward mouse movement (matches fluid u_vel)
                    float vx = (float)(mouseX - prevMouseX);
                    // vy: Invert to match fluid's Y-axis (positive v_vel is upward on screen)
                    // If mouse moves down (mouseY increases), prevMouseY - mouseY is negative, pushing fluid down.
                    float vy = (float)(prevMouseY - mouseY);

                    // Map mouse coordinates (0 to SCREEN_WIDTH/HEIGHT-1) to fluid grid coordinates (0 to N-1)
                    // Invert Y-axis for mouse input to match OpenGL's bottom-up Y-axis
                    int fluidGridX_0_N_minus_1 = static_cast<int>(mouseX / GRID_PIXEL_SIZE_X);
                    int fluidGridY_0_N_minus_1 = static_cast<int>((SCREEN_HEIGHT - 1 - mouseY) / GRID_PIXEL_SIZE_Y);

                    // Convert to 1-indexed fluid grid coordinates (1 to N)
                    int fluidGridX = fluidGridX_0_N_minus_1 + 1;
                    int fluidGridY = fluidGridY_0_N_minus_1 + 1;

                    // Clamp to valid fluid grid indices (1 to N) to handle any floating point inaccuracies near boundaries
                    fluidGridX = custom_clamp_fluid(fluidGridX, 1, N);
                    fluidGridY = custom_clamp_fluid(fluidGridY, 1, N);

                    currentHue += 0.5f;
                    if (currentHue >= 360.0f) {
                        currentHue = 0.0f;
                    }
                    ColorRGB color = hsv2rgb(currentHue, 1.0f, 1.0f);

                    float density_add_strength = 50.0f;
                    // Note: velocity_strength is implicitly applied by vx, vy from mouse motion
                    // If you want more pronounced velocity, you could multiply vx, vy by a factor.

                    fluid.setForceArea(fluidGridX, fluidGridY,
                                       color.r * density_add_strength,
                                       color.g * density_add_strength,
                                       color.b * density_add_strength,
                                       vx, vy);
                    prevMouseX = mouseX;
                    prevMouseY = mouseY;
                }
            } else if (e.type == SDL_EVENT_KEY_DOWN) {
                switch (e.key.scancode) {
                    case SDL_SCANCODE_R: fluid.reset(); std::cout << "Fluid simulation reset." << std::endl; break;
                    case SDL_SCANCODE_V: fluid.viscosity += 0.0001f; std::cout << "Viscosity: " << fluid.viscosity << std::endl; break;
                    case SDL_SCANCODE_C: fluid.viscosity -= 0.0001f; if(fluid.viscosity < 0) fluid.viscosity = 0; std::cout << "Viscosity: " << fluid.viscosity << std::endl; break;
                    case SDL_SCANCODE_F: fluid.fadeSpeed += 0.001f; if(fluid.fadeSpeed > 1) fluid.fadeSpeed = 1; std::cout << "Fade Speed: " << fluid.fadeSpeed << std::endl; break;
                    case SDL_SCANCODE_G: fluid.fadeSpeed -= 0.001f; if(fluid.fadeSpeed < 0) fluid.fadeSpeed = 0; std::cout << "Fade Speed: " << fluid.fadeSpeed << std::endl; break;
                    case SDL_SCANCODE_D: fluid.diffusion += 0.0001f; std::cout << "Diffusion: " << fluid.diffusion << std::endl; break;
                    case SDL_SCANCODE_S: fluid.diffusion -= 0.0001f; if(fluid.diffusion < 0) fluid.diffusion = 0; std::cout << "Diffusion: " << fluid.diffusion << std::endl; break;
                    case SDL_SCANCODE_T: fluid.dt += 0.01f; std::cout << "Delta Time: " << fluid.dt << std::endl; break;
                    case SDL_SCANCODE_Y: fluid.dt -= 0.01f; if(fluid.dt < 0.001f) fluid.dt = 0.001f; std::cout << "Delta Time: " << fluid.dt << std::endl; break;
                }
            }
        }

        fluid.step();
        renderer.render(fluid);
        SDL_GL_SwapWindow(window);
    }

    // Call shutdown while GL context is still valid
    renderer.shutdown();
    SDL_GL_DestroyContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}