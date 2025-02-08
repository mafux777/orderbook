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

    def get_data_bounds(self, df, column, lower_percentile=10, upper_percentile=90):
        """
        Calculate bounds for a column based on percentiles
        
        Args:
            df: DataFrame containing the data
            column: Column name to calculate bounds for
            lower_percentile: Lower percentile cutoff (default: 10)
            upper_percentile: Upper percentile cutoff (default: 90)
        
        Returns:
            tuple: (min_value, max_value)
        """
        min_value = df[column].quantile(lower_percentile/100)
        max_value = df[column].quantile(upper_percentile/100)
        return min_value, max_value

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

    def create_animation(self, orderbook_df, candle_df, output_file='orderbook.mp4', frame_interval_ms=100):
        """
        Create an animated visualization with growing candlestick chart and orderbook
        """
        if orderbook_df is None or orderbook_df.empty:
            print("No market data available for animation")
            return
        
        # Fixed ranges for orderbook
        min_price, max_price = 2600, 3200
        max_size = 500  # Fixed at 500 as requested
        
        # Create price bins with $1 width
        price_step = 1.0
        num_bins = int((max_price - min_price) / price_step) + 1
        price_bins = np.linspace(min_price, max_price, num_bins)
        
        # Get unique timestamps and ensure consistent timezone handling
        timestamps = pd.to_datetime(orderbook_df.index.unique())  # Keep UTC
        timestamps = sorted(timestamps)
        
        # Create figure with two subplots
        fig, (ax_price, ax_book) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[1, 2])
        
        def animate(frame):
            current_timestamp = timestamps[frame]
            
            # Clear both plots
            ax_price.clear()
            ax_book.clear()
            
            # Plot candlestick data (top subplot)
            # Get only candles up to current timestamp (both are now UTC)
            mask = candle_df.index <= current_timestamp
            current_candles = candle_df[mask]
            
            # Plot each candle
            for idx, candle in current_candles.iterrows():
                # Determine candle color (green for up, red for down)
                color = 'green' if candle['price_close'] >= candle['price_open'] else 'red'
                
                # Plot candle body
                body_bottom = min(candle['price_open'], candle['price_close'])
                body_height = abs(candle['price_close'] - candle['price_open'])
                ax_price.bar(idx, body_height, bottom=body_bottom, color=color, alpha=0.5, width=0.8)
                
                # Plot high/low wicks
                ax_price.vlines(idx, candle['price_low'], candle['price_high'], color=color)
            
            # Set price chart properties
            ax_price.set_ylim(min_price, max_price)
            ax_price.set_title('ETH/USD Price')
            
            # Plot orderbook data (bottom subplot)
            snapshot = orderbook_df.loc[current_timestamp]
            snapshot = snapshot[
                (snapshot['price'] >= min_price) & 
                (snapshot['price'] <= max_price)
            ]
            
            # Create histograms
            bid_mask = snapshot['side'] == 'bid'
            ask_mask = snapshot['side'] == 'ask'
            
            bid_histogram = snapshot[bid_mask].groupby(
                pd.cut(snapshot[bid_mask]['price'].astype(float), bins=price_bins)
            )['size'].sum()
            
            ask_histogram = snapshot[ask_mask].groupby(
                pd.cut(snapshot[ask_mask]['price'].astype(float), bins=price_bins)
            )['size'].sum()
            
            # Calculate midpoints
            bid_midpoints = [(interval.left + interval.right) / 2 for interval in bid_histogram.index]
            ask_midpoints = [(interval.left + interval.right) / 2 for interval in ask_histogram.index]
            
            # Plot orderbook
            ax_book.barh(ask_midpoints, ask_histogram.values, height=0.9, label='Asks', color='red', alpha=0.5)
            ax_book.barh(bid_midpoints, bid_histogram.values, height=0.9, label='Bids', color='green', alpha=0.5)
            
            # Set fixed axis limits
            ax_book.set_ylim(min_price, max_price)
            ax_book.set_xlim(0, max_size)
            
            # Labels and title
            timestamp_str = pd.Timestamp(current_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            ax_book.set_xlabel('Size')
            ax_book.set_ylabel('Price')
            ax_book.set_title(f'Order Book - {timestamp_str}')
            ax_book.legend()
            
            # Adjust layout
            plt.tight_layout()
        
        # Create and save animation
        print(f"Saving animation with {len(timestamps)} frames...")
        anim = animation.FuncAnimation(
            fig, 
            animate,
            frames=len(timestamps),
            interval=frame_interval_ms,
            repeat=False
        )
        
        writer = animation.FFMpegWriter(fps=1000/frame_interval_ms)
        anim.save(output_file, writer=writer)
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
        
        # Get timestamps for the first num_frames minutes
        timestamps = pd.to_datetime(orderbook_df.index.unique())
        timestamps = sorted(timestamps)[:num_frames]
        
        # Calculate price range from all candles in our timeframe
        window_start = timestamps[0]
        window_end = timestamps[-1]
        window_mask = (candle_df.index >= window_start) & (candle_df.index <= window_end)
        window_candles = candle_df[window_mask]
        
        # Calculate price range
        price_min = min(
            window_candles['price_low'].min(),
            window_candles['price_open'].min(),
            window_candles['price_close'].min()
        )
        price_max = max(
            window_candles['price_high'].max(),
            window_candles['price_open'].max(),
            window_candles['price_close'].max()
        )
        
        # Add 1% padding to price range
        price_padding = (price_max - price_min) * 0.01
        price_min -= price_padding
        price_max += price_padding
        
        # Create price bins with $1 width for orderbook histogram
        price_step = 1.0
        num_bins = int((price_max - price_min) / price_step) + 1
        price_bins = np.linspace(price_min, price_max, num_bins)
        
        # Fixed max size for orderbook
        max_size = 500
        
        print(f"Generating {len(timestamps)} frames...")
        print(f"Price range: {price_min:.2f} - {price_max:.2f}")
        
        # Create progress bar
        pbar = tqdm.tqdm(enumerate(timestamps), total=len(timestamps), desc="Generating frames")
        
        for i, timestamp in pbar:
            fig, (ax_price, ax_book) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[1, 2])
            
            # Get last 10 candles up to current timestamp
            mask = candle_df.index <= timestamp
            current_candles = candle_df[mask].tail(10)  # Get only last 10 candles
            
            # Calculate x-positions for candles (0 to 9 for the last 10 candles)
            x_positions = range(len(current_candles))
            
            # Plot each candle with fixed width
            for x_pos, (idx, candle) in enumerate(current_candles.iterrows()):
                color = 'green' if candle['price_close'] >= candle['price_open'] else 'red'
                body_bottom = min(candle['price_open'], candle['price_close'])
                body_height = abs(candle['price_close'] - candle['price_open'])
                
                # Plot candle body with fixed width (e.g., 0.8)
                ax_price.bar(x_pos, body_height, bottom=body_bottom, color=color, 
                            alpha=0.5, width=0.8)
                # Plot wicks
                ax_price.vlines(x_pos, candle['price_low'], candle['price_high'], color=color)
            
            # Set fixed x-axis range (0-10 for 10 candles)
            ax_price.set_xlim(-0.5, 9.5)  # Add 0.5 padding on each side
            
            # Format x-axis to show times for the candles
            x_labels = [pd.Timestamp(idx).strftime('%H:%M') for idx in current_candles.index]
            ax_price.set_xticks(x_positions)
            ax_price.set_xticklabels(x_labels, rotation=45)
            
            # Plot orderbook data (bottom subplot)
            snapshot = orderbook_df.loc[timestamp]
            snapshot = snapshot[
                (snapshot['price'] >= price_min) & 
                (snapshot['price'] <= price_max)
            ]
            
            # Create histograms with observed=True to fix warnings
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
            
            # Calculate midpoints
            bid_midpoints = [(interval.left + interval.right) / 2 for interval in bid_histogram.index]
            ask_midpoints = [(interval.left + interval.right) / 2 for interval in ask_histogram.index]
            
            # Plot orderbook
            ax_book.barh(ask_midpoints, ask_histogram.values, height=0.9, label='Asks', color='red', alpha=0.5)
            ax_book.barh(bid_midpoints, bid_histogram.values, height=0.9, label='Bids', color='green', alpha=0.5)
            
            # Set fixed axis limits and grid
            ax_book.set_ylim(price_min, price_max)
            ax_book.set_xlim(0, max_size)
            ax_book.grid(True, alpha=0.3)
            
            # Labels and title
            timestamp_str = pd.Timestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            ax_book.set_xlabel('Size')
            ax_book.set_ylabel('Price')
            ax_book.set_title(f'Order Book - {timestamp_str}')
            ax_book.legend()
            
            # Adjust spacing between subplots
            plt.tight_layout()
            
            # Save frame
            frame_file = os.path.join(output_dir, f'frame_{i:03d}.png')
            plt.savefig(frame_file, dpi=100, bbox_inches='tight')
            plt.close()

def main():
    # Initialize animator
    animator = OrderBookAnimator()
    
    # Use the same market and time range as before
    market = 'kraken-eth-usd-spot'
    start_time = '2025-02-02T00:00:00Z'
    end_time = '2025-02-02T00:30:00Z'  # 30 minutes of data
    
    print(f"Fetching data for {market} from {start_time} to {end_time}")
    
    # Fetch both orderbook and candle data
    raw_market_data = animator.fetch_market_data(market, start_time, end_time)
    raw_candle_data = animator.fetch_market_candles(market, start_time, end_time)
    
    if raw_market_data is not None and raw_candle_data is not None:
        print("\nProcessing orderbook data...")
        processed_orderbook = animator.preprocess_orderbook_data(raw_market_data)
        
        print("\nProcessing candle data...")
        processed_candles = animator.preprocess_candle_data(raw_candle_data)
        
        print("\nGenerating test frames...")
        animator.create_test_frames(processed_orderbook, processed_candles, num_frames=30)  # 30 frames

if __name__ == "__main__":
    main() 