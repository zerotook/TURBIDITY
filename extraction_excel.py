import georaster, numpy as np, os, pdb
import pandas as pd

csv_path = '../pixel_data.csv'

def get_surrounding_pixels(image, lon, lat):
	'''

	the function calculates values on pixels located nearby a specified coordinate

	'''
	# create a raster with image

	raster = georaster.SingleBandRaster(image,load_data=False, latlon=True)
	
	# get surrounding pixels; output of (mean, arr[pixel_values])
	

	mean, arr = raster.value_at_coords(lon,lat, latlon=True, window=5, return_window=True)
	#mean, arr = raster.value_at_coords(lon,lat, latlon=True)
	coord = raster.value_at_coords(lon, lat, latlon=True, window=5)
	arr = arr.flatten()
	arr = arr[np.logical_not(np.isnan(arr))]
	if len(arr) == 0:
		return
	

	# calculate standard deviation

	stdev = np.std(arr)
	maximum = max(arr)

	# Return a dictionary of results
	return {'FILENAME': image, 'coordinates': f'{lat},{lon}', 'MEAN': mean, 'STDEV': stdev, '5X5 window pixel values': arr, 'maximum': maximum}

def _get_images():
    images = os.listdir()
    for i in range(len(images)):
        images[i] = f'./{images[i]}'
    return images

if __name__ == '__main__':
	os.chdir('./images')
	coords = [(-118.56346,34.03882), (-118.52146,34.02446), (-118.52636,34.02875),(-118.428279,33.91333),(-118.449425,33.951278),(-118.679,34.032),(-118.678,34.033),(-118.5097567,34.01979456),(-118.58106,34.03691),(-118.68128,34.02964),(-118.819056,34.01101746),(-118.49595,34.00557),(-118.47364,33.97914),(-118.562863,34.03959)]
	images = _get_images()
	
	results = []
	
	for img in images:
		for lon, lat in coords:
			try:
				result = get_surrounding_pixels(img, lon, lat)
				if result:
					results.append(result)
			except:
				print(f"ERROR: {img} could not locate pixels at {lon}, {lat} :) ")
	
	# Save results to Excel
	df = pd.DataFrame(results)
	df.to_excel('pixel_data.xlsx', index=False)
