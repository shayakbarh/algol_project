from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.visualization import ZScaleInterval
from pathlib import Path


# Function to list FITS files in a directory with specific prefix and suffix
def list_fits_files(folder_path, prefix="algol", suffix="_dsfc.fits"):

    folder_path = Path(folder_path)
    
    # Return a list of all matching FITS files using pathlib
    return list(folder_path.glob(f"{prefix}*{suffix}"))


# Function to plot FITS files and save them to a PDF 
def plot_fits_to_pdf(fits_files, output_pdf_path):
    
    output_pdf_path = Path(output_pdf_path)
    
    zscale = ZScaleInterval()

    with PdfPages(output_pdf_path) as pdf:
        for file_path in fits_files:
            with fits.open(file_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header

                # Extract metadata directly from the header
                timestamp = header.get("DATE-OBS", "Unknown Time")
                exposure = header.get("EXPOSURE", "Unknown Exposure")

            # Determine display limits using ZScale
            vmin, vmax = zscale.get_limits(data)

            # Plot the image with dynamic color scaling
            plt.imshow(data, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
            plt.colorbar(label="Pixel Value")
            plt.title(f"Timestamp: {timestamp}\nExposure: {exposure} sec")

            # Save the plot to the PDF
            pdf.savefig()
            plt.close()

    print(f"PDF saved at: {output_pdf_path}")


# Main function to generate a PDF from FITS files in a specified folder
def generate_pdf_from_fits(input_folder, output_pdf="frame_images.pdf"):

    input_folder = Path(input_folder)
    
    # List FITS files
    fits_files = list_fits_files(input_folder)

    if not fits_files:
        print("No matching FITS files found.")
        return

    # Generate PDF output path
    output_pdf_path = output_pdf
    print(f"Generating PDF: {output_pdf_path}")
    
    # Plot FITS files and save to PDF
    plot_fits_to_pdf(fits_files, output_pdf_path)


# execute the function with the desired input folder path
input_folder_path = r"D:\AA-2\algol_observation\Data\calibration_data"
generate_pdf_from_fits(input_folder_path)