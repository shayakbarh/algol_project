from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from scipy.spatial import KDTree
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# Function to get sorted FITS files from the directory
def get_sorted_files(input_folder):
    return sorted([f for f in input_folder.glob("algol*_dsfc.fits")], key=lambda x: x.name.split("_")[2])


# Function to display the image with ZScale normalization
def display_image(image_data, title):
    vmin, vmax = ZScaleInterval().get_limits(image_data)
    plt.imshow(image_data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Pixel Value")
    plt.title(title)



# Function to detect stars in the image using DAOStarFinder
def detect_stars(data):
    sources = DAOStarFinder(fwhm=5.0, threshold=1.5 * np.std(data))(data)
    return np.array([sources["xcentroid"], sources["ycentroid"]]).T if sources else np.empty((0, 2))


# Function to find a specific star near a given position (x, y) within a bounding box
def find_star(data, x, y, box=20):
    # Crop the image around the star position
    y0, y1, x0, x1 = map(int, [max(0, y-box), min(data.shape[0], y+box), max(0, x-box), min(data.shape[1], x+box)])
    cropped = np.nan_to_num(data[y0:y1, x0:x1])

    # Detect stars in the cropped image
    stars = detect_stars(cropped)

    if stars.size == 0: return None
    stars += [x0, y0] # Adjust positions to original image coordinates

    return tuple(stars[KDTree(stars).query([x, y])[1]])


# Function to track the position of Algol in a series of FITS files
def track_algol(files, get_guess):
    algol_positions, timestamps = [], []

    for f in tqdm(files, desc="Tracking Star"):
        with fits.open(f) as hdul:
            data, hdr = hdul[0].data, hdul[0].header
            guess = get_guess(hdr) # Initial guess for Algol's position

            algol_positions.append(find_star(data, *guess))
            timestamps.append(hdr["DATE-OBS"]) # Record the timestamp

    return algol_positions, timestamps


# Function to select reference stars from the first FITS file
def select_reference_stars(first_file, num):

    with fits.open(first_file) as hdul:
        data = hdul[0].data

    plt.figure(figsize=(8,6))
    display_image(data, "Select Reference Stars")
    stars = plt.ginput(num, timeout=0)  # user input for star positions
    plt.close()

    return [find_star(data, *p) for p in stars]


# Function to track the positions of reference stars across all FITS files
def track_references(files, init_refs):
    ref_tracks = [[p] for p in init_refs]

    # Loop through each file and track the reference stars
    for f in tqdm(files[1:], desc="Tracking Refs"):
        with fits.open(f) as hdul:
            data = hdul[0].data

        for i, track in enumerate(ref_tracks):
            last = track[-1]

            if last is None: track.append(None); continue
            stars = detect_stars(data)

            if len(stars) == 0: track.append(None); continue
            track.append(tuple(stars[KDTree(stars).query(last)[1]]))

    return ref_tracks


# Function to save the positions of Algol and reference stars in a PDF
def save_positions_pdf(files, algol_pos, ref_pos, times, output):

    # define colors and markers for reference stars
    ref_markers = ['o', 's', '+', 'x', 'd']
    ref_colors = ['red', 'black', 'purple', 'green']

    with PdfPages(output) as pdf:
        for i, f in tqdm(enumerate(files), desc="Saving Positions"):
            with fits.open(f) as hdul:
                data, hdr = hdul[0].data, hdul[0].header

            plt.figure(figsize=(8, 6))
            display_image(data, f"Frame {i+1}: {times[i]}\nExposure: {hdr['EXPOSURE']}s")

            if algol_pos[i]: plt.scatter(*algol_pos[i], c='red', marker='*', s=100, label='Algol')

            for j, pos_list in enumerate(ref_pos):
                if i < len(pos_list) and pos_list[i]:
                    marker = ref_markers[j % len(ref_markers)]  # Cycle through markers
                    color = ref_colors[j % len(ref_colors)] # Cycle through colors
                    plt.scatter(*pos_list[i], c=color, marker=marker, s=50, alpha=0.5, label=f"Ref {j+1}")

                    #plt.scatter(*pos_list[i], c='red', marker='o', s=50, alpha=0.5, label=f"Ref {j+1}")

            plt.legend()
            pdf.savefig()
            plt.close()

    print(f"Saved to {output}")


# Function to measure the flux of Algol and reference stars 
def measure_flux(files, algol_pos, ref_pos, times, r=5, adu=48.0):

    def calc_flux(data, pos, r):
        ap, ann = CircularAperture(pos, r=r), CircularAnnulus(pos, r_in=2*r, r_out=3*r)
        phot, bkg = aperture_photometry(data, ap), aperture_photometry(data, ann)
        bkg_mean = bkg["aperture_sum"] / ann.area
        bkg_total = bkg_mean * ap.area
        return (phot["aperture_sum"][0] - bkg_total) * adu, (phot["aperture_sum"][0] + bkg_total) * adu

    aflux, rflux = {}, [{} for _ in ref_pos]

    for i, f in tqdm(enumerate(files), desc="Measuring Flux"):
        with fits.open(f) as hdul:
            data, exptime = hdul[0].data, hdul[0].header["EXPOSURE"]

        t = times[i]

        af, av = calc_flux(data, algol_pos[i], r)
        aflux[t] = (af/exptime, av/(exptime**2))

        for j, rp in enumerate(ref_pos):
            rf, rv = calc_flux(data, rp[i], r)
            rflux[j][t] = (rf/exptime, rv/(exptime**2))

    return aflux, rflux



# Function to calculate the relative flux of Algol to reference stars
def relative_flux(aflux, rflux):
    rel = {}

    for j, ref in enumerate(rflux):
        rel[f"ref_{j+1}"] = {}

        for t in aflux:
            if t in ref:
                fa, va = aflux[t] # Algol flux and variance
                fr, vr = ref[t]   # Reference flux and variance 

                rf = fa / fr    # Relative flux
                rv = rf**2 * (va/fa**2 + vr/fr**2) # Variance propagation

                rel[f"ref_{j+1}"][t] = (rf, np.sqrt(rv)) # Store relative flux and its error

    return rel



# Function to save the light curves of Algol in a PDF
def save_lightcurves(rel_flux, times, output="algol_lightcurves.pdf"):

    with PdfPages(output) as pdf:
        for ref, flux in tqdm(rel_flux.items(), desc="Plotting Lightcurves"):
            time_labels = [t.split("T")[1][:-7] for t in times]
            flux_values = np.array([flux[t][0] for t in times]).flatten()
            flux_errors = np.array([flux[t][1] for t in times]).flatten()

            plt.figure(figsize=(12, 6))
            plt.errorbar(time_labels, flux_values, yerr=flux_errors, marker='o', linestyle ='-', ecolor='lightcoral', capsize=3, label =f"Algol / {ref}")

            plt.xlabel("Observation Time (UTC)")
            plt.ylabel(f"Relative Flux")
            plt.title(f"Algol Light Curve")
            plt.grid(True, alpha=0.5)

            step = max(1, len(times) // 20)  # Show every 20th label 
            plt.xticks(time_labels[::step], rotation=45)
            plt.ylim(0, 260)
            plt.legend()
            pdf.savefig()
            plt.close()
    print(f"Saved light curves to {output}")



# Main function to execute the photometry process
def main(folder_path):
    # Get sorted FITS files from the input folder
    folder = Path(folder_path)
    files = get_sorted_files(folder)

    # track Algol
    algol_pos, times = track_algol(files, lambda hdr: (hdr['PHDLOCKX'], hdr['PHDLOCKY']))

    # Select reference stars
    print("Select reference stars in the first frame.")
    n = int(input("Number of reference stars: "))
    init_refs = select_reference_stars(files[0], n)

    # Track reference stars
    ref_pos = track_references(files, init_refs)

    # Save positions of Algol and reference stars to a PDF
    save_positions_pdf(files, algol_pos, ref_pos, times, "tracked_stars.pdf")

    # Measure flux of Algol and reference stars
    aflux, rflux = measure_flux(files, algol_pos, ref_pos, times)

    # Calculate relative flux of Algol to reference stars
    rel_flux = relative_flux(aflux, rflux)

    # Save light curves of Algol in a PDF
    save_lightcurves(rel_flux, times)


# Execute the photometry process
input_folder = r"D:\AA-2\algol_observation\Data\calibration_data"
main(input_folder)
