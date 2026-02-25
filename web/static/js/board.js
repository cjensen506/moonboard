/**
 * MoonBoard canvas overlay for rendering and interacting with hold positions.
 *
 * The MoonBoard image (665x1023 native) has an 18-row x 11-column grid.
 * Row 1 is at the BOTTOM, row 18 at the TOP. Columns A-K go left to right.
 */

// Grid boundaries in native image pixels (665x1023).
// These define the outer edges of the grid; cell centres are offset by half a step.
// Calibrated against bolt positions in rows 1, 17, and 18.
const GRID = {
  left: 69,
  right: 634,
  top: 63,
  bottom: 985,
  cols: 11,
  rows: 18,
};

const COL_STEP = (GRID.right - GRID.left) / GRID.cols;
const ROW_STEP = (GRID.bottom - GRID.top) / GRID.rows;

// Hold type colours
const HOLD_COLORS = {
  start: "#4caf50",
  middle: "#2196f3",
  end: "#f44336",
};

/**
 * Convert grid position to pixel centre on the native image.
 * @param {number} col 0-indexed (0=A, 10=K)
 * @param {number} row 0-indexed (0=row1 bottom, 17=row18 top)
 */
export function gridToPixel(col, row) {
  return {
    x: GRID.left + COL_STEP * (col + 0.5),
    y: GRID.bottom - ROW_STEP * (row + 0.5),
  };
}

/**
 * Convert native-image pixel to grid position.
 * Returns null if outside grid.
 */
export function pixelToGrid(px, py) {
  const col = Math.floor((px - GRID.left) / COL_STEP);
  const row = Math.floor((GRID.bottom - py) / ROW_STEP);
  if (col < 0 || col > 10 || row < 0 || row > 17) return null;
  return {
    col,
    row,
    letter: String.fromCharCode(65 + col),
    number: row + 1,
    label: String.fromCharCode(65 + col) + (row + 1),
  };
}

/**
 * Parse a hold description like "E6" into {col, row} (0-indexed).
 */
export function parseHold(desc) {
  const letter = desc.match(/[A-Ka-k]/)[0].toUpperCase();
  const num = parseInt(desc.match(/\d+/)[0], 10);
  return { col: letter.charCodeAt(0) - 65, row: num - 1 };
}

/**
 * MoonBoard renderer manages a canvas overlay on an image element.
 */
export class MoonBoard {
  constructor(imgEl, canvasEl) {
    this.img = imgEl;
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext("2d");
    this._resizeObserver = new ResizeObserver(() => this._syncSize());

    // Wait for image to load before syncing
    if (imgEl.complete) {
      this._syncSize();
    } else {
      imgEl.addEventListener("load", () => this._syncSize(), { once: true });
    }
    this._resizeObserver.observe(imgEl);
  }

  _syncSize() {
    // Match canvas to the image's content area, accounting for border offset.
    // clientWidth/Height = content + padding (no border).
    const w = this.img.clientWidth;
    const h = this.img.clientHeight;
    const borderLeft = parseInt(getComputedStyle(this.img).borderLeftWidth) || 0;
    const borderTop = parseInt(getComputedStyle(this.img).borderTopWidth) || 0;
    this.canvas.width = w;
    this.canvas.height = h;
    this.canvas.style.width = w + "px";
    this.canvas.style.height = h + "px";
    this.canvas.style.left = borderLeft + "px";
    this.canvas.style.top = borderTop + "px";
    this.scaleX = w / (this.img.naturalWidth || 665);
    this.scaleY = h / (this.img.naturalHeight || 1023);
  }

  /** Clear the canvas */
  clear() {
    this._syncSize();
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Draw holds on the canvas.
   * @param {Array} moves - [{description: "E6", isStart: bool, isEnd: bool}, ...]
   */
  drawHolds(moves) {
    this.clear();
    const ctx = this.ctx;
    const radius = 14 * this.scaleX;

    for (const move of moves) {
      const pos = parseHold(move.description);
      const px = gridToPixel(pos.col, pos.row);
      const x = px.x * this.scaleX;
      const y = px.y * this.scaleY;

      let type = "middle";
      if (move.isStart) type = "start";
      else if (move.isEnd) type = "end";

      const color = HOLD_COLORS[type];

      // Outer ring
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.stroke();

      // Semi-transparent fill
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color + "40";
      ctx.fill();
    }
  }

  /**
   * Enable interactive clicking. Calls callback(gridPos) on click.
   * gridPos = {col, row, letter, number, label}
   */
  enableInteractive(callback) {
    this.canvas.addEventListener("click", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      // Convert to native image pixels
      const natX = clickX / this.scaleX;
      const natY = clickY / this.scaleY;

      const grid = pixelToGrid(natX, natY);
      if (grid) callback(grid);
    });
  }

  /**
   * Draw a single highlight circle (e.g. for hover).
   */
  drawHighlight(col, row, color = "#ffffff60") {
    const px = gridToPixel(col, row);
    const x = px.x * this.scaleX;
    const y = px.y * this.scaleY;
    const radius = 14 * this.scaleX;

    this.ctx.beginPath();
    this.ctx.arc(x, y, radius, 0, Math.PI * 2);
    this.ctx.fillStyle = color;
    this.ctx.fill();
  }
}
