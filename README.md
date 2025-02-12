# Orderbook Animation Generator

This tool creates animated visualizations of cryptocurrency order books and price movements using data from the Coin Metrics API. The animation combines a candlestick chart with a real-time order book visualization.

## Features

- Dual-panel visualization showing:
  - Candlestick chart (10-minute window)
  - Order book depth chart
- Smooth transitions between frames
- Configurable time ranges and markets
- Support for any market available through Coin Metrics API
- Automatic y-axis scaling with moving average smoothing

## Prerequisites

- Python 3.x
- FFmpeg (for video generation)
- Coin Metrics API key

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install pandas numpy matplotlib seaborn requests
```

## Usage

```bash
python orderbook_animation.py --market kraken-eth-usd-spot --start 2025-02-02T00:00:00Z --end 2025-02-04T00:00:00Z
```

### Command Line Arguments

- `--market`: Market identifier (default: kraken-eth-usd-spot)
- `--start`: Start time in ISO format (default: 2025-02-02T00:00:00Z)
- `--end`: End time in ISO format (default: 2025-02-04T00:00:00Z)
- `--test`: Generate test frames instead of animation
- `--frames`: Number of frames for test mode (default: 30)
- `--output`: Output file name (default: orderbook_animation.mp4 or test_frames/)

## Output

The script generates either:
- An MP4 video file showing the animated order book and price movements
- A series of test frames in a specified directory (when using --test)

## Development

To debug the visualization:
1. Use the `--test` flag to generate individual frames
2. Frames will be saved in the `test_frames/` directory
3. Inspect frames to verify visualization correctness

