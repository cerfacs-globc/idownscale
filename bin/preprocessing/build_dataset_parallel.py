import sys
import os
import argparse
sys.path.append('.')

from bin.preprocessing.build_dataset import DatasetBuilder
from iriscc.settings import DATES

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Smart Parallel Production Wrapper")
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--i_start', type=int, default=0)
    parser.add_argument('--i_end', type=int, default=None)
    args = parser.parse_args()

    builder = DatasetBuilder(args.exp)
    
    i_end = args.i_end if args.i_end is not None else len(DATES)
    print(f"--- Parallel Production Task: {args.exp} ---")
    print(f"--- Index Range: {args.i_start} to {i_end} ---")

    for i in range(args.i_start, i_end):
        if i >= len(DATES):
            break
        date = DATES[i]
        date_str = date.date().strftime('%Y%m%d')
        output_file = os.path.join(builder.dataset, f"sample_{date_str}.npz")
        
        # Skip-Existing Protocol for seamless production resume
        if os.path.exists(output_file):
            continue
            
        builder.process_date(date)
