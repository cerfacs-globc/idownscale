import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def main():
    parser = argparse.ArgumentParser(description="Generate PDF evaluation report")
    parser.add_argument('--exp', type=str, default='exp5', help='Experiment name')
    parser.add_argument('--test-name', type=str, default='unet_all', help='Test name')
    parser.add_argument('--simu', type=str, default='gcm', help='Simulation (gcm/rcm)')
    parser.add_argument('--ssp', type=str, default='ssp585', help='SSP scenario')
    args = parser.parse_args()

    graph_dir = Path('graph') / 'metrics' / args.exp
    metric_dir = Path('metrics') / args.exp
    output_pdf = graph_dir / f'evaluation_report_{args.exp}_{args.test_name}_{args.ssp}.pdf'

    print(f"Generating report: {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        # Page 1: Title and Summary Table
        fig, ax = plt.subplots(figsize=(11.69, 8.27)) # A4 Landscape
        ax.axis('off')
        ax.text(0.5, 0.9, f"Downscaling Evaluation Report: {args.exp}", 
                fontsize=24, ha='center', weight='bold')
        ax.text(0.5, 0.82, f"Target: {args.test_name} | Simu: {args.simu} | SSP: {args.ssp}", 
                fontsize=16, ha='center')
        
        # Load VALUE metrics
        metrics_path = metric_dir / f'value_metrics_{args.exp}_{args.test_name}.csv'
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            # Transpose for vertical table if single row
            if len(df) == 1:
                df_table = df.T.reset_index()
                df_table.columns = ['Metric', 'Value']
            else:
                df_table = df

            table = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                            loc='center', cellLoc='center', colWidths=[0.3, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax.text(0.5, 0.7, "VALUE Framework Statistics (Historical 2000-2014)", 
                    fontsize=14, ha='center', weight='bold')
        else:
            ax.text(0.5, 0.5, "VALUE metrics not found. Run Phase 6.2 first.", 
                    fontsize=14, ha='center', color='red')

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Spatial Trends
        spatial_img = graph_dir / f'{args.exp}_spatial_futur_trend_{args.ssp}_{args.simu}.png'
        if spatial_img.exists():
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            img = mpimg.imread(spatial_img)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Future Spatial Trends ({args.ssp})", fontsize=16, weight='bold')
            pdf.savefig(fig)
            plt.close(fig)

        # Page 3: Variability
        var_img = graph_dir / f'{args.exp}_variability_futur_trend_{args.ssp}_{args.simu}.png'
        if var_img.exists():
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            img = mpimg.imread(var_img)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Temporal Variability Analysis ({args.ssp})", fontsize=16, weight='bold')
            pdf.savefig(fig)
            plt.close(fig)

        # Page 4: Histograms
        hist_img = graph_dir / f'{args.exp}_hist_futur_trend_{args.ssp}_unet_{args.simu}.png'
        if hist_img.exists():
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            img = mpimg.imread(hist_img)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Distribution Comparison (UNet vs {args.simu})", fontsize=16, weight='bold')
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Report successfully generated at: {output_pdf}")

if __name__ == '__main__':
    main()
