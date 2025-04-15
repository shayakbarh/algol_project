import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from pathlib import Path

def organize_files(input_folder):
    flat_file = None
    dark_files = []
    science_files = {"algol": [], "capella": []}

    for file in Path(input_folder).glob("*.fits"):
        parts = file.stem.lower().split("_")
        if len(parts) < 2:
            continue

        file_type, exposure_str = parts[0], parts[1]
        try:
            exposure = float(exposure_str)
        except ValueError:
            continue

        if file_type == "flat":
            flat_file = str(file)
        elif file_type == "dark":
            dark_files.append((str(file), exposure))
        elif file_type in science_files:
            science_files[file_type].append((str(file), exposure))

    print(f"Files organized: {len(science_files['algol'])} Algol, "
          f"{len(science_files['capella'])} Capella, "
          f"{len(dark_files)} Dark, and 1 Flat frame.")

    return flat_file, dark_files, science_files


def process_flat(flat_file, output_folder):
    if flat_file is None:
        print("WARNING: Flat frame not found. Skipping flat frame processing.")
        return None

    flat_path = Path(flat_file)
    output_path = Path(output_folder) / "processed_flat.fits"

    with fits.open(flat_path) as hdul:
        flat_data = hdul[0].data.astype(np.float64)
        header = hdul[0].header

    smoothed = gaussian_filter(flat_data, sigma=50)
    processed_flat = flat_data / smoothed

    fits.writeto(output_path, processed_flat, header, overwrite=True)
    print(f"Processed flat frame saved at {output_path}")

    return processed_flat



def subtract_dark(science_files, dark_files, output_folder):
    output_folder = Path(output_folder)
    #output_folder.mkdir(parents=True, exist_ok=True)

    # Load all dark frames into a dictionary indexed by exposure time
    dark_by_exposure = {
        exp: fits.getdata(path).astype(np.float64)  # 'exp' is the exposure time (in seconds)
        for path, exp in dark_files
    }

    dark_subtracted = {}

    for category, files in science_files.items():
        for science_path, exp in files:
            science_path = Path(science_path)
            with fits.open(science_path) as hdu:
                science_data = hdu[0].data.astype(np.float64)
                header = hdu[0].header

            dark_data = dark_by_exposure.get(exp)
            if dark_data is not None:
                corrected_data= science_data - dark_data
            else:
                print(f"WARNING: No matching dark frame for exposure {exp}s. Using raw data.")
                corrected_data = science_data

            #out_path = output_folder / science_path.with_suffix("").name.replace(".fits", "_darksub.fits")
            out_path = output_folder / (science_path.stem + "_ds.fits")  # ds means dark subtracted 
            fits.writeto(out_path, corrected_data, header, overwrite=True)
            print(f"Dark-subtracted frame saved at {out_path}")

            dark_subtracted[str(science_path)] = (corrected_data, header)

    return dark_subtracted



def apply_flat_correction(dark_subtracted_data, flat_frame, output_folder):
    if flat_frame is None:
        print("WARNING: Flat frame not processed. Skipping flat correction.")
        return

    output_folder = Path(output_folder)
    #output_folder.mkdir(parents=True, exist_ok=True)

    for sci_path, (dark_subtracted, sci_header) in dark_subtracted_data.items():
        corrected_data = dark_subtracted / flat_frame
        out_name = Path(sci_path).with_suffix("").name + "_dsfc.fits"  # dsfc means dark subtracted flat corrected
        out_path = output_folder / out_name
        fits.writeto(out_path, corrected_data, sci_header, overwrite=True)
        print(f"Flat-corrected frame saved at {out_path}")


def run_pipeline(input_folder, output_folder):
    print("Starting calibration pipeline...")

    # Step 1: Organise files
    flat_file, dark_files, science_files = organize_files(input_folder)

    # Step 2: Process flat frame
    flat_frame = process_flat(flat_file, output_folder)

    # Step 3: Subtract dark frames
    dark_subtracted_data = subtract_dark(science_files, dark_files, output_folder)

    # Step 4: Apply flat correction
    apply_flat_correction(dark_subtracted_data, flat_frame, output_folder)

    print("Calibration pipeline completed.")



# Define input and output directories using pathlib
input_folder = r"D:\AA-2\algol_observation\Data\reduction_data"
output_folder = r"D:\AA-2\algol_observation\Data\calibration_data"

# Create output directory if it doesn't exist
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Run the pipeline
run_pipeline(input_folder, output_folder)