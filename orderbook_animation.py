import pandas as pd
from coinmetrics.api_client import CoinMetricsClient
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import imageio
import tqdm
import argparse

class OrderBookAnimator:
    def __init__(self, api_key=None):
        """Initialize the OrderBook animator with optional API key"""
        self.api_key = api_key or os.getenv('CM_API_KEY')
        if not self.api_key:
            raise ValueError("API key not found. Please set CM_API_KEY environment variable")
        self.client = CoinMetricsClient(self.api_key)
        
    def fetch_available_markets(self):
        """Fetch and return available markets"""
        order_books = self.client.catalog_market_orderbooks_v2().to_dataframe()
        return order_books
    
    def fetch_market_data(self, market, start_time, end_time):
        """
        Fetch orderbook data for a specific market and time range
        
        Args:
            market (str): Market identifier (e.g., 'kraken-eth-usd-spot')
            start_time (str): Start time in ISO format
            end_time (str): End time in ISO format
            
        Returns:
            list: List of orderbook snapshots, each containing bids and asks
        """
        try:
            # Get order book snapshots using parallel processing
            # We use parallel processing with 10-minute increments to handle large data efficiently
            orderbook_data = self.client.get_market_orderbooks(
                markets=market,
                granularity='1m',
                start_time=start_time,
                end_time=end_time,
                page_size=10000
            ).parallel(time_increment=relativedelta(minutes=10)).to_list()
            
            return orderbook_data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

    def preprocess_orderbook_data(self, raw_orderbook_data):
        """
        Transform raw orderbook data into a pandas DataFrame with normalized structure
        
        Args:
            raw_orderbook_data (list): List of orderbook snapshots from the API
            
        Returns:
            pd.DataFrame: Processed orderbook data with columns:
                - time: timestamp of the snapshot
                - price: price level
                - size: order size at this price
                - side: 'bid' or 'ask'
                - market: market identifier
                - coin_metrics_id: unique identifier
                - database_time: time of database entry
        """
        processed_orders = []
        
        # Iterate through each orderbook snapshot
        for snapshot in raw_orderbook_data:
            # Process bids
            for bid in snapshot['bids']:
                # Copy over metadata fields to each bid
                for field in ['market', 'time', 'coin_metrics_id', 'database_time']:
                    bid[field] = snapshot[field]
                bid['side'] = 'bid'
                processed_orders.append(bid)
            
            # Process asks
            for ask in snapshot['asks']:
                # Copy over metadata fields to each ask
                for field in ['market', 'time', 'coin_metrics_id', 'database_time']:
                    ask[field] = snapshot[field]
                ask['side'] = 'ask'
                processed_orders.append(ask)
        
        # Convert to DataFrame and set time as index, preserving UTC timezone
        orderbook_df = pd.DataFrame(processed_orders)
        orderbook_df['time'] = pd.to_datetime(orderbook_df['time'], utc=True)  # Force UTC
        orderbook_df = orderbook_df.set_index('time')
        
        # Convert price and size to float
        orderbook_df['price'] = orderbook_df['price'].astype(float)
        orderbook_df['size'] = orderbook_df['size'].astype(float)
        
        return orderbook_df

    def plot_orderbook_snapshot(self, snapshot_df, output_file=None):
        """
        Create a histogram plot for a single orderbook snapshot with fixed ranges
        """
        # Convert price and size to float type
        snapshot_df = snapshot_df.copy()
        snapshot_df['price'] = snapshot_df['price'].astype(float)
        snapshot_df['size'] = snapshot_df['size'].astype(float)
        
        # Fixed ranges
        min_price, max_price = 3000, 3200
        max_size = 500
        
        # Filter data based on price range
        snapshot_df = snapshot_df[
            (snapshot_df['price'] >= min_price) & 
            (snapshot_df['price'] <= max_price)
        ]
        
        # Create price bins with $1 width
        price_step = 1.0
        num_bins = int((max_price - min_price) / price_step) + 1
        price_bins = np.linspace(min_price, max_price, num_bins)

        # Create histograms for bids and asks
        bid_mask = snapshot_df['side'] == 'bid'
        ask_mask = snapshot_df['side'] == 'ask'
        
        bid_histogram = snapshot_df[bid_mask].groupby(
            pd.cut(snapshot_df[bid_mask]['price'], bins=price_bins)
        )['size'].sum()
        
        ask_histogram = snapshot_df[ask_mask].groupby(
            pd.cut(snapshot_df[ask_mask]['price'], bins=price_bins)
        )['size'].sum()

        # Calculate bin midpoints
        bid_midpoints = [(interval.left + interval.right) / 2 for interval in bid_histogram.index]
        ask_midpoints = [(interval.left + interval.right) / 2 for interval in ask_histogram.index]

        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot asks and bids
        plt.barh(ask_midpoints, ask_histogram.values, height=0.9, label='Asks', color='red', alpha=0.5)
        plt.barh(bid_midpoints, bid_histogram.values, height=0.9, label='Bids', color='green', alpha=0.5)

        # Set fixed axis limits
        plt.ylim(min_price, max_price)
        plt.xlim(0, max_size)

        plt.xlabel('Size')
        plt.ylabel('Price')
        plt.title('Order Book Snapshot')
        plt.legend()

        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
            
    def preprocess_candle_data(self, raw_candle_data):
        """Transform raw candle data into a pandas DataFrame"""
        candles = pd.DataFrame(raw_candle_data)
        candles['time'] = pd.to_datetime(candles['time'], utc=True)  # Force UTC
        candles = candles.set_index('time')
        
        # Convert price columns to float
        price_columns = ['price_open', 'price_close', 'price_high', 'price_low']
        for col in price_columns:
            candles[col] = candles[col].astype(float)
        
        return candles

    def calculate_y_limits(self, orderbook_df):
        """Calculate y-axis limits for each timestamp in the orderbook data."""
        # Ensure the index is a datetime index
        orderbook_df.index = pd.to_datetime(orderbook_df.index)

        # Group by timestamp and calculate min and max prices
        y_limits_df = orderbook_df.groupby(orderbook_df.index).agg(
            y_min=('price', 'min'),
            y_max=('price', 'max')
        ).reset_index()

        # Calculate centered moving averages over a 20-minute window (10 minutes before and after)
        window_size = 20
        y_limits_df['y_min_ma'] = y_limits_df['y_min'].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
        y_limits_df['y_max_ma'] = y_limits_df['y_max'].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()

        # Add padding as a percentage of the price range
        padding_percentage = 0.25  # 5% padding
        price_range = y_limits_df['y_max_ma'] - y_limits_df['y_min_ma']
        padding = price_range * padding_percentage
        
        y_limits_df['y_min_ma'] = y_limits_df['y_min_ma'] - padding
        y_limits_df['y_max_ma'] = y_limits_df['y_max_ma'] + padding

        return y_limits_df.set_index('time')

    def create_animation(self, orderbook_df, candle_df, output_file='orderbook.mp4', frame_interval_ms=100):
        """Create an animated visualization with sliding window candlestick chart and orderbook"""
        if orderbook_df is None or orderbook_df.empty:
            print("No market data available for animation")
            return
        
        # Calculate y-axis limits
        y_limits_df = self.calculate_y_limits(orderbook_df)

        # Create figure
        fig, (ax_price, ax_book) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[1, 2])
        
        # Create progress bar
        pbar = tqdm.tqdm(total=len(y_limits_df), desc="Generating frames")
        
        def animate(frame):
            # Pass the calculated moving average limits to the plot frame
            self._plot_frame(
                fig=fig,
                ax_price=ax_price,
                ax_book=ax_book,
                timestamp=y_limits_df.index[frame],  # Access the timestamp from the index directly
                candle_df=candle_df,
                orderbook_df=orderbook_df,
                y_min=y_limits_df['y_min_ma'].iloc[frame],  # Pass moving average min
                y_max=y_limits_df['y_max_ma'].iloc[frame]   # Pass moving average max
            )
            pbar.update(1)
        
        # Create and save animation
        print(f"Creating animation with {len(y_limits_df)} frames...")
        anim = animation.FuncAnimation(
            fig, 
            animate,
            frames=len(y_limits_df),
            interval=frame_interval_ms,
            repeat=False
        )
        
        writer = animation.FFMpegWriter(fps=1000/frame_interval_ms)
        anim.save(output_file, writer=writer)
        pbar.close()
        plt.close()

    def fetch_market_candles(self, market, start_time, end_time):
        """
        Fetch market candle data from CoinMetrics API using parallel method
        """
        try:
            print(f"Fetching candle data for {market}")
            
            candle_data = self.client.get_market_candles(
                markets=[market],
                start_time=start_time,
                end_time=end_time,
                frequency='1m'
            ).parallel(time_increment=relativedelta(minutes=10)).to_list()
            
            if candle_data and len(candle_data) > 0:
                print(f"Successfully fetched {len(candle_data)} candles")
                return candle_data
            else:
                print("No candle data returned from API")
                return None
            
        except Exception as e:
            print(f"Error fetching candle data: {str(e)}")
            return None

    def create_test_frames(self, orderbook_df, candle_df, output_dir='test_frames', num_frames=30):
        """Create test frames for visual inspection and layout tuning"""
        os.makedirs(output_dir, exist_ok=True)

        # Calculate y-axis limits
        y_limits_df = self.calculate_y_limits(orderbook_df)

        # Get timestamps for the first num_frames
        timestamps = list(y_limits_df.index)[0:num_frames]

        for frame_index, timestamp in enumerate(timestamps):
            # Create a new figure for each frame
            fig, (ax_price, ax_book) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[1, 2])
            
            # Plot the frame data
            self.plot_frame_data(fig, ax_price, ax_book, timestamp, candle_df, orderbook_df,
                                 y_min=y_limits_df['y_min'].iloc[frame_index],
                                 y_max=y_limits_df['y_max'].iloc[frame_index])

            # Save the frame as an image file
            frame_filename = os.path.join(output_dir, f'frame_{frame_index:03d}.png')
            plt.savefig(frame_filename)
            plt.close(fig)  # Close the figure to free memory

    def _plot_frame(self, fig, ax_price, ax_book, timestamp, candle_df, orderbook_df, y_min, y_max, max_size=500):
        """Helper method to plot a single frame of the visualization"""
        # Clear both plots
        ax_price.clear()
        ax_book.clear()
        
        # Use common plotting method
        self.plot_frame_data(fig, ax_price, ax_book, timestamp, candle_df, orderbook_df, y_min, y_max, max_size)

    def plot_frame_data(self, fig, ax_price, ax_book, timestamp, candle_df, orderbook_df, y_min, y_max, max_size=500):
        """Common method to plot frame data for both test frames and animation"""
        # Set fixed y-axis limits for both plots
        y_range = [y_min, y_max]
        ax_price.set_ylim(y_range)
        ax_book.set_ylim(y_range)

        # Get last 10 candles up to current timestamp
        mask = candle_df.index <= timestamp
        current_candles = candle_df[mask].tail(10)
        
        # Plot candles
        x_positions = range(len(current_candles))
        for x_pos, (idx, candle) in enumerate(current_candles.iterrows()):
            color = 'green' if candle['price_close'] >= candle['price_open'] else 'red'
            body_bottom = min(candle['price_open'], candle['price_close'])
            body_height = abs(candle['price_close'] - candle['price_open'])
            
            ax_price.bar(x_pos, body_height, bottom=body_bottom, color=color, 
                        alpha=0.5, width=0.8)
            ax_price.vlines(x_pos, candle['price_low'], candle['price_high'], color=color)
        
        # Set candlestick chart properties
        ax_price.set_xlim(-0.5, 9.5)
        x_labels = [pd.Timestamp(idx).strftime('%H:%M') for idx in current_candles.index]
        ax_price.set_xticks(x_positions)
        ax_price.set_xticklabels(x_labels, rotation=45)
        ax_price.grid(True, alpha=0.3)
        ax_price.set_title('Price Chart (10-minute window)')
        
        # Plot orderbook
        snapshot = orderbook_df.loc[timestamp]
        
        # Create price bins with fixed width
        price_step = 1.0
        num_bins = int((y_max - y_min) / price_step) + 1
        price_bins = np.linspace(y_min, y_max, num_bins)
        
        # Create histograms
        bid_mask = snapshot['side'] == 'bid'
        ask_mask = snapshot['side'] == 'ask'
        
        bid_histogram = snapshot[bid_mask].groupby(
            pd.cut(snapshot[bid_mask]['price'].astype(float), bins=price_bins),
            observed=True
        )['size'].sum()
        
        ask_histogram = snapshot[ask_mask].groupby(
            pd.cut(snapshot[ask_mask]['price'].astype(float), bins=price_bins),
            observed=True
        )['size'].sum()
        
        # Plot orderbook data
        bid_midpoints = [(interval.left + interval.right) / 2 for interval in bid_histogram.index]
        ask_midpoints = [(interval.left + interval.right) / 2 for interval in ask_histogram.index]
        
        ax_book.barh(ask_midpoints, ask_histogram.values, height=0.9, label='Asks', color='red', alpha=0.5)
        ax_book.barh(bid_midpoints, bid_histogram.values, height=0.9, label='Bids', color='green', alpha=0.5)
        
        # Set orderbook properties and ensure y-axis limits are maintained
        ax_book.set_xlim(0, max_size)
        ax_book.set_ylim(y_range)  # Set again to ensure it's maintained
        ax_book.grid(True, alpha=0.3)
        
        timestamp_str = pd.Timestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        ax_book.set_xlabel('Size')
        ax_book.set_ylabel('Price')
        ax_book.set_title(f'Order Book - {timestamp_str}')
        ax_book.legend()
        
        # Add market name between the plots
        market_name = orderbook_df['market'].iloc[0].upper()
        fig.suptitle(market_name, y=0.5, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        # Adjust layout to make room for the market name
        plt.subplots_adjust(hspace=0.3)

def main():
    parser = argparse.ArgumentParser(description='Create orderbook and candlestick animations')
    parser.add_argument('--market', default='kraken-eth-usd-spot',
                      help='Market identifier (default: kraken-eth-usd-spot)')
    parser.add_argument('--start', default='2025-02-02T00:00:00Z',
                      help='Start time in ISO format (default: 2025-02-02T00:00:00Z)')
    parser.add_argument('--end', default='2025-02-04T00:00:00Z',
                      help='End time in ISO format (default: 2025-02-04T00:00:00Z)')
    parser.add_argument('--test', action='store_true',
                      help='Generate test frames instead of animation')
    parser.add_argument('--frames', type=int, default=30,
                      help='Number of frames for test mode (default: 30)')
    parser.add_argument('--output', default=None,
                      help='Output file name (default: orderbook_animation.mp4 or test_frames/)')
    
    args = parser.parse_args()
    
    # Initialize animator
    animator = OrderBookAnimator()
    
    print(f"Fetching data for {args.market} from {args.start} to {args.end}")
    
    # Fetch both orderbook and candle data
    raw_market_data = animator.fetch_market_data(args.market, args.start, args.end)
    raw_candle_data = animator.fetch_market_candles(args.market, args.start, args.end)
    
    if raw_market_data is not None and raw_candle_data is not None:
        print("\nProcessing orderbook data...")
        processed_orderbook = animator.preprocess_orderbook_data(raw_market_data)
        
        print("\nProcessing candle data...")
        processed_candles = animator.preprocess_candle_data(raw_candle_data)
        
        if args.test:
            output_dir = args.output or 'test_frames'
            print("\nGenerating test frames...")
            animator.create_test_frames(processed_orderbook, processed_candles, 
                                     output_dir=output_dir, num_frames=args.frames)
        else:
            output_file = args.output or 'orderbook_animation.mp4'
            print("\nCreating animation...")
            animator.create_animation(processed_orderbook, processed_candles, output_file)

if __name__ == "__main__":
    main() 