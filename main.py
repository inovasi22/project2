from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Simpan file yang di-upload
    file = request.files['image']
    filename = file.filename
    file.save(filename)

    # Baca gambar
    img = cv2.imread(filename)
    # Set the target color in BGR format
    target_color = np.array([35, 62, 136])  # corresponds to ##883e23 in BGR format
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask of the pixels with the target color
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(mask)  # invert the mask

    # Apply the mask to the original image to isolate the region with the target color
    region = cv2.bitwise_and(img, img, mask=mask)

    # Convert the region to grayscale and threshold it to create a binary mask of the region
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, region_mask = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY)

    # Calculate the area of the region with the target color
    #area = cv2.countNonZero(region_mask) * (1.0 ** 2)  # assumes pixel size of 1.0
    area = cv2.countNonZero(region_mask) 
    
    # Return the result
    #are_gambar_semua
    areagambar = img.shape[0] * img.shape[1] * (1.0 ** 2) 

    #hitung semua area
    allarea= (area+areagambar)

    #luas uang
    luasuang=(allarea/areagambar)
    #return f"The area of the region with color {target_color} is {area} square units."
    return render_template('result.html', filename=filename, area=area, areagambar=areagambar )
   
if __name__ == '__main__':
    app.run(debug=True)
