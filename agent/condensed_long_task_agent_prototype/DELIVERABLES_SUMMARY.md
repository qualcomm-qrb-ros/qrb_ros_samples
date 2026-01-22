# Tiled Matrix Multiplication - Educational Implementation

## Deliverables

### 1. tiled_matmul.c (189 lines)
Complete C implementation featuring:
- 16x16 matrix multiplication using 4x4 tiles
- Educational comments explaining tiling concept
- Random matrix initialization for testing
- Result verification against naive implementation
- Performance timing comparison
- Clear variable naming for student comprehension

### 2. tiled_matmul_animation.html (567 lines)
Interactive HTML animation featuring:
- Side-by-side code and visualization panels
- Step-by-step execution with line highlighting
- Visual matrix representation with color-coded highlighting
- Real-time status displays:
  - Current tile position (row, col)
  - Element position within tile
  - Tile iteration counter
  - Operation description
  - Accumulator value
- Interactive controls:
  - Start: Begin continuous animation
  - Step Forward: Advance one step at a time
  - Pause: Stop continuous animation
  - Reset: Return to initial state
  - Speed Control: Adjust animation speed (100-2000ms)
- Professional styling with responsive design

## Usage

### Compile and Run C Code:
```bash
gcc -o tiled_matmul tiled_matmul.c -lm
./tiled_matmul
```

### View Animation:
Open `tiled_matmul_animation.html` in any modern web browser.

## Educational Features

1. **Tiling Concept**: Code demonstrates cache-friendly memory access patterns
2. **Visual Learning**: Animation shows exactly how tiles are processed
3. **Step-by-Step**: Students can pause and examine each operation
4. **Clear Comments**: Every section explained for beginners
5. **Verification**: Includes correctness checking against naive method

## File Statistics
- Total lines: 756
- C code: 189 lines
- HTML/CSS/JS: 567 lines
- Both files fully commented and production-ready
