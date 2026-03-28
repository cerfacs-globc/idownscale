import argparse
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add current directory to path for iriscc imports
sys.path.append('.')
from iriscc.settings import OUTPUT_DIR, GRAPHS_DIR

def main():
    parser = argparse.ArgumentParser(description="Generate PDF comparison report (BC vs AI)")
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    parser.add_argument('--test-name', type=str, default='unet_all', help='Test name')
    parser.add_argument('--ssp', type=str, default='ssp585', help='SSP scenario')
    args = parser.parse_args()

    report_dir = OUTPUT_DIR / args.exp / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = report_dir / f'comparison_report_{args.exp}_{args.ssp}.pdf'
    
    graph_dir = GRAPHS_DIR / 'metrics' / args.exp

    print(f"Generating Comparison Report: {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        # Page 1: Title
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        ax.text(0.5, 0.7, f"Bias Correction vs AI Downscaling Comparison", 
                fontsize=28, ha='center', weight='bold', color='navy')
        ax.text(0.5, 0.6, f"Experiment: {args.exp} | Scenario: {args.ssp}", 
                fontsize=18, ha='center')
        ax.text(0.5, 0.4, "Evolution Analysis:\nRaw GCM \u2192 Ibicus BC \u2192 AI Matrix", 
                fontsize=16, ha='center', style='italic')
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: PDF Evolution (Histograms)
        pdf_img = graph_dir / f'{args.exp}_pdf_evolution_{args.ssp}.png'
        if pdf_img.exists():
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            img = mpimg.imread(pdf_img)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("Probability Density Function (PDF) Evolution", fontsize=18, weight='bold')
            pdf.savefig(fig)
            plt.close(fig)

        # Page 3: Spatial Trends (Final Result)
        spatial_img = graph_dir / f'{args.exp}_spatial_futur_trend_{args.ssp}_gcm.png'
        if spatial_img.exists():
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            img = mpimg.imread(spatial_img)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("Future Spatial Trends Analysis", fontsize=18, weight='bold')
            pdf.savefig(fig)
            plt.close(fig)

        # Page 4: Temporal Variability
        var_img = graph_dir / f'{args.exp}_variability_futur_trend_{args.ssp}_gcm.png'
        if var_img.exists():
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            img = mpimg.imread(var_img)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("Temporal Variability Evolution", fontsize=18, weight='bold')
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Comparison report successfully generated at: {output_pdf}")

if __name__ == '__main__':
    main()
