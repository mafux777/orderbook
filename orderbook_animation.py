import pandas as pd
from coinmetrics.api_client import CoinMetricsClient
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import imageio

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
        
        # Convert to DataFrame and set time as index
        orderbook_df = pd.DataFrame(processed_orders).set_index('time')
        
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
            
    def create_animation(self, orderbook_df, output_file='orderbook.gif', frame_duration_ms=100):
        """Create an animated visualization of the orderbook over time"""
        if orderbook_df is None or orderbook_df.empty:
            print("No market data available for animation")
            return
        
        # Get unique timestamps (one per minute)
        timestamps = sorted(orderbook_df.index.unique())
        print(f"Generating {len(timestamps)} unique frames...")
        
        # Create a list to store our frames
        frames = []
        
        # Set up the figure with a specific DPI
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        # Generate each unique frame
        for i, timestamp in enumerate(timestamps):
            snapshot = orderbook_df.loc[timestamp]
            ax.clear()
            
            # Use the same plotting logic as plot_orderbook_snapshot
            min_price, max_price = 3000, 3200
            max_size = 500
            
            # Filter data based on price range
            snapshot_df = snapshot[
                (snapshot['price'] >= min_price) & 
                (snapshot['price'] <= max_price)
            ]
            
            # Create price bins with $1 width
            price_step = 1.0
            num_bins = int((max_price - min_price) / price_step) + 1
            price_bins = np.linspace(min_price, max_price, num_bins)
            
            # Create histograms (with observed=True to fix warning)
            bid_mask = snapshot_df['side'] == 'bid'
            ask_mask = snapshot_df['side'] == 'ask'
            
            bid_histogram = snapshot_df[bid_mask].groupby(
                pd.cut(snapshot_df[bid_mask]['price'], bins=price_bins),
                observed=True
            )['size'].sum()
            
            ask_histogram = snapshot_df[ask_mask].groupby(
                pd.cut(snapshot_df[ask_mask]['price'], bins=price_bins),
                observed=True
            )['size'].sum()
            
            # Calculate midpoints
            bid_midpoints = [(interval.left + interval.right) / 2 for interval in bid_histogram.index]
            ask_midpoints = [(interval.left + interval.right) / 2 for interval in ask_histogram.index]
            
            # Plot the data
            ax.barh(ask_midpoints, ask_histogram.values, height=0.9, label='Asks', color='red', alpha=0.5)
            ax.barh(bid_midpoints, bid_histogram.values, height=0.9, label='Bids', color='green', alpha=0.5)
            
            # Set fixed axis limits
            ax.set_ylim(min_price, max_price)
            ax.set_xlim(0, max_size)
            
            timestamp_str = pd.Timestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            ax.set_xlabel('Size')
            ax.set_ylabel('Price')
            ax.set_title(f'Order Book - {timestamp_str}')
            ax.legend()
            
            # Instead of buffer manipulation, save to a temporary file and read it back
            temp_file = f'temp_frame_{i}.png'
            fig.savefig(temp_file)
            img = imageio.imread(temp_file)
            frames.append(img)
            os.remove(temp_file)  # Clean up temporary file
            
            if i % 100 == 0:  # Progress update every 100 frames
                print(f"Generated frame {i}/{len(timestamps)}")
        
        plt.close()
        
        # Save as GIF
        print(f"Saving animation with {len(frames)} frames...")
        imageio.mimsave(output_file, frames, duration=frame_duration_ms/1000.0)

def main():
    # Initialize animator
    animator = OrderBookAnimator()
    
    # Use the same market and time range as the notebook
    market = 'kraken-eth-usd-spot'
    start_time = '2025-02-02T00:00:00Z'
    end_time = '2025-02-03T00:00:00Z'
    
    print(f"Fetching data for {market} from {start_time} to {end_time}")
    raw_market_data = animator.fetch_market_data(market, start_time, end_time)
    
    if raw_market_data is not None:
        print(f"Fetched {len(raw_market_data)} orderbook snapshots")
        if len(raw_market_data) > 0:
            print("\nProcessing orderbook data...")
            processed_data = animator.preprocess_orderbook_data(raw_market_data)
            print(f"Processed data shape: {processed_data.shape}")
            
            
            
            print("\nCreating animation...")
            animator.create_animation(processed_data, 'orderbook_animation.gif')
            print("Animation saved to orderbook_animation.gif")

if __name__ == "__main__":
    main() 